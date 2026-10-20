"""Production SVI calibration of option chains, slice by slice, with quote
hygiene, maturity-aware bands, out-of-sample baselines and real
no-arbitrage diagnostics.

Pipeline per (trade_date, expiry) group:

1. :func:`derivx.vol.quotes.clean_option_slice`: drop expired / same-day
   rows, IV placeholders, zero bids, crossed and stale quotes, ITM options
   and duplicates; infer the forward from put-call parity (carry fallback).
2. Keep strikes within a standardised moneyness band ``|k| <= std_band *
   atm_vol * sqrt(tau)`` (same width in standard deviations at every
   maturity; ``k = ln(K/F)``), and optionally a forward-moneyness band
   ``moneyness_band`` on ``K/F``.
3. Require ``min_points`` quotes (more than the 5 SVI parameters) with at
   least ``min_per_side`` on each side of the forward.
4. Fit raw SVI in total variance (:func:`derivx.vol.svi.fit_svi_strike_slice`,
   equal weights in vol, soft-L1 loss at 50 vol bp); with ``enforce_arbitrage`` the
   butterfly and Lee conditions are penalised when the free fit violates
   them.
5. Diagnostics: Gatheral butterfly density g(k) on a dense grid, Lee wing
   slopes, calendar check against the previous expiry of the same trade date
   (total variance non-decreasing at every k), Jacobian condition number at
   the solution, parameters at bounds, errors in vol bp, and a k-fold
   out-of-sample comparison with a cubic-spline baseline.

Units: vols and RMSEs are decimal vols (``*1e4`` = vol bp); ``tau`` in
ACT/365F years; SVI parameters are total-variance parameters for ``tau``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from derivx.vol.quotes import CleanSlice, apply_std_band, clean_option_slice
from derivx.vol.svi import SviFit, check_calendar, fit_svi_strike_slice, svi_iv

SVI_N_PARAMS = 5


@dataclass
class CalibratedSlice:
    """One calibrated expiry. ``params`` are raw-SVI total-variance
    parameters for ``tau``; ``rmse`` is the in-sample IV RMSE (decimal vol);
    ``diagnostics`` is a flat, JSON-friendly dict (see module docs)."""

    trade_date: pd.Timestamp
    expiry: pd.Timestamp
    forward: float
    tau: float
    params: Dict[str, float]
    rmse: float
    n_points: int
    strikes: np.ndarray
    observed_ivs: np.ndarray
    diagnostics: Dict[str, object] = field(default_factory=dict)
    benchmark: Optional[Dict[str, object]] = None
    fit: Optional[SviFit] = field(default=None, repr=False)
    clean: Optional[CleanSlice] = field(default=None, repr=False)

    def iv(self, K):
        """Fitted Black vol at strike(s) K."""

        return svi_iv(K, self.params, F=self.forward, T=self.tau)


@dataclass
class SkippedSlice:
    trade_date: pd.Timestamp
    expiry: pd.Timestamp
    reason: str
    n_points: int


@dataclass
class ChainCalibration:
    """Everything :func:`calibrate_chain` produced: fitted ``slices``,
    ``skipped`` slices with the reason, and the ``calendar`` table (one row
    per adjacent expiry pair of each trade date)."""

    slices: List[CalibratedSlice]
    skipped: List[SkippedSlice]
    calendar: pd.DataFrame

    def table(self) -> pd.DataFrame:
        rows = []
        for s in self.slices:
            d = s.diagnostics
            rows.append(
                {
                    "trade_date": s.trade_date.date(),
                    "expiry": s.expiry.date(),
                    "tau": s.tau,
                    "forward": s.forward,
                    "points": s.n_points,
                    "rmse_bp": d.get("rmse_bp"),
                    "cv_rmse_bp": (s.benchmark or {}).get("rmse_svi_cv", np.nan) * 1e4,
                    "spline_cv_rmse_bp": (s.benchmark or {}).get("rmse_spline_cv", np.nan) * 1e4,
                    "butterfly_ok": d.get("butterfly_ok"),
                    "min_g": d.get("butterfly_min_g"),
                    "lee_ok": d.get("lee_ok"),
                    "calendar_ok": d.get("calendar_ok"),
                    "cond_J": d.get("condition_number"),
                    "at_bound": d.get("params_at_bound"),
                }
            )
        return pd.DataFrame(rows)


def _groups(chain: pd.DataFrame, trade_date) -> List[tuple]:
    df = chain.copy()
    df["expiry"] = pd.to_datetime(df["expiry"])
    if trade_date is not None:
        df["trade_date"] = pd.Timestamp(trade_date)
    elif "trade_date" not in df.columns:
        raise ValueError(
            "chain needs a 'trade_date' column (the valuation date tau is measured from), or pass trade_date="
        )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ("trade_date", "expiry"):
        if df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_convert(None)
        df[col] = df[col].dt.normalize()
    return list(df.groupby(["trade_date", "expiry"], sort=True))


def _band_filter(clean: CleanSlice, std_band: Optional[float], moneyness_band: Optional[Sequence[float]]):
    if std_band is not None:
        clean = apply_std_band(clean, std_band)
    if moneyness_band is not None:
        lo, hi = float(moneyness_band[0]), float(moneyness_band[1])
        if not 0 < lo < hi:
            raise ValueError("moneyness_band must be (low, high) with 0 < low < high (K/F)")
        m = clean.quotes["strike"].to_numpy(float) / clean.forward
        inside = (m >= lo) & (m <= hi)
        out = clean.quotes[~inside].assign(reason="outside_band")
        clean.quotes = clean.quotes[inside].reset_index(drop=True)
        clean.removed = pd.concat([clean.removed, out], ignore_index=True) if not out.empty else clean.removed
    return clean


def spread_weights(quotes: pd.DataFrame, floor: float = 1e-4) -> np.ndarray:
    """Weights 1 / (iv_ask - iv_bid) per quote (floored at ``floor`` vol),
    from a slice cleaned with ``recompute_iv=True``; a quote without a bid
    vol gets the smallest weight of the slice."""

    if "iv_bid" not in quotes.columns or "iv_ask" not in quotes.columns:
        raise ValueError("weights='spread' needs bid/ask implied vols: pass hygiene={'recompute_iv': True}")
    width = (quotes["iv_ask"] - quotes["iv_bid"]).to_numpy(float)
    w = 1.0 / np.maximum(width, floor)
    finite = np.isfinite(w)
    if not finite.any():
        raise ValueError("no quote has both a bid and an ask implied vol")
    return np.where(finite, w, w[finite].min())


def calibrate_chain(
    chain: pd.DataFrame,
    *,
    enforce_arbitrage: bool = True,
    std_band: Optional[float] = 3.0,
    moneyness_band: Optional[Sequence[float]] = None,
    min_points: int = 8,
    min_per_side: int = 2,
    trade_date=None,
    rate: float = 0.0,
    div_yield: float = 0.0,
    min_days: float = 1.0,
    weights: Optional[str] = None,
    loss: str = "soft_l1",
    f_scale: float = 0.005,
    cv_folds: int = 5,
    hygiene: Optional[Dict[str, object]] = None,
) -> ChainCalibration:
    """Calibrate SVI to every (trade_date, expiry) slice of ``chain``.

    ``rate`` / ``div_yield`` (continuously compounded) set the discount
    factor used in put-call parity (and for vols recomputed from mids) and
    the carry-forward fallback. ``weights``: None (equal), 'vega', or
    'spread' (1 / bid-ask width in vol; needs ``hygiene={'recompute_iv':
    True}``).
    ``min_points`` must exceed the 5 SVI parameters. ``hygiene`` passes
    extra keyword arguments to :func:`clean_option_slice` (e.g.
    ``{"max_stale_days": None}``). ``cv_folds=0`` skips the out-of-sample
    benchmark."""

    if min_points <= SVI_N_PARAMS:
        raise ValueError(
            f"min_points={min_points} would let 5-parameter SVI interpolate the quotes exactly: use >= 6 (default 8)"
        )
    if weights not in (None, "vega", "spread"):
        raise ValueError(f"weights must be None, 'vega' or 'spread', got {weights!r}")
    if weights == "spread" and not (hygiene or {}).get("recompute_iv"):
        raise ValueError("weights='spread' needs bid/ask implied vols: pass hygiene={'recompute_iv': True}")
    if chain is None or chain.empty:
        return ChainCalibration([], [], pd.DataFrame())
    if "expiry" not in chain.columns:
        raise ValueError("Chain must contain 'expiry' column")
    slices: List[CalibratedSlice] = []
    skipped: List[SkippedSlice] = []
    for (td, exp), group in _groups(chain, trade_date):
        try:
            clean = clean_option_slice(
                group, trade_date=td, rate=rate, div_yield=div_yield, min_days=min_days, **(hygiene or {})
            )
        except ValueError as exc:
            skipped.append(SkippedSlice(td, exp, str(exc), len(group)))
            continue
        if clean.tau * 365.0 < min_days:
            skipped.append(SkippedSlice(td, exp, f"expired or same-day (tau = {clean.tau:.4f}y)", len(group)))
            continue
        if clean.quotes.empty:
            skipped.append(SkippedSlice(td, exp, f"no quotes left after hygiene: {clean.counts}", len(group)))
            continue
        clean = _band_filter(clean, std_band, moneyness_band)
        q = clean.quotes
        n_left, n_right = int((q["k"] < 0).sum()), int((q["k"] > 0).sum())
        if len(q) < min_points or n_left < min_per_side or n_right < min_per_side:
            skipped.append(
                SkippedSlice(
                    td,
                    exp,
                    f"{len(q)} quotes ({n_left} below / {n_right} above the forward) after hygiene and band; "
                    f"need {min_points} with {min_per_side} per side",
                    len(q),
                )
            )
            continue
        strikes = q["strike"].to_numpy(float)
        vols = q["implied_vol"].to_numpy(float)
        try:
            fit = fit_svi_strike_slice(
                strikes,
                vols,
                F=clean.forward,
                T=clean.tau,
                weights=spread_weights(q) if weights == "spread" else weights,
                loss=loss,
                f_scale=f_scale,
                enforce_no_arbitrage=enforce_arbitrage,
            )
        except ValueError as exc:
            skipped.append(SkippedSlice(td, exp, f"fit failed: {exc}", len(q)))
            continue
        diagnostics: Dict[str, object] = {
            "tau": clean.tau,
            "forward": clean.forward,
            "forward_source": clean.forward_source,
            "spot": clean.spot,
            "n_input": clean.n_input,
            "removed": clean.counts,
            "rmse_bp": fit.rmse_bp,
            "max_abs_error_bp": fit.max_abs_error_bp,
            "condition_number": fit.jacobian_condition,
            "params_at_bound": [k for k, v in fit.at_bound.items() if v],
            "min_variance": fit.min_variance,
            "n_downweighted": fit.n_downweighted,
            "enforce_arbitrage": bool(enforce_arbitrage),
            "arbitrage_penalised": fit.penalised,
            **fit.butterfly.as_dict(),
            **fit.lee,
            "calendar_ok": None,
            "k_min": float(q["k"].min()),
            "k_max": float(q["k"].max()),
            "moneyness_min": float(strikes.min() / clean.forward),
            "moneyness_max": float(strikes.max() / clean.forward),
            "optimizer": {"success": fit.success, "nfev": fit.nfev, "starts": fit.n_starts},
        }
        result = CalibratedSlice(
            trade_date=td,
            expiry=exp,
            forward=clean.forward,
            tau=clean.tau,
            params=fit.params,
            rmse=fit.rmse,
            n_points=fit.n_points,
            strikes=fit.strikes,
            observed_ivs=fit.market_vols,
            diagnostics=diagnostics,
            fit=fit,
            clean=clean,
        )
        if cv_folds:
            result.benchmark = benchmark_vs_baseline(result, folds=cv_folds)
        slices.append(result)

    calendar_rows = []
    by_date: Dict[pd.Timestamp, List[CalibratedSlice]] = {}
    for s in slices:
        by_date.setdefault(s.trade_date, []).append(s)
    for td, group in by_date.items():
        group.sort(key=lambda s: s.tau)
        for prev, nxt in zip(group, group[1:]):
            lo = min(prev.diagnostics["k_min"], nxt.diagnostics["k_min"])
            hi = max(prev.diagnostics["k_max"], nxt.diagnostics["k_max"])
            check = check_calendar([(prev.tau, prev.params), (nxt.tau, nxt.params)], np.linspace(lo, hi, 401))[0]
            nxt.diagnostics.update(check.as_dict())
            calendar_rows.append({"trade_date": td, "expiry_1": prev.expiry, "expiry_2": nxt.expiry, **check.as_dict()})
    return ChainCalibration(slices, skipped, pd.DataFrame(calendar_rows))


def calibrate_with_constraints(
    chain: pd.DataFrame,
    enforce_arbitrage: bool = True,
    moneyness_band: Optional[Sequence[float]] = None,
    min_points: int = 8,
    **kwargs,
) -> List[CalibratedSlice]:
    """:func:`calibrate_chain` returning only the fitted slices (skipped
    slices and their reasons are in ``calibrate_chain(...).skipped``).
    ``moneyness_band`` is on forward moneyness K/F."""

    return calibrate_chain(
        chain, enforce_arbitrage=enforce_arbitrage, moneyness_band=moneyness_band, min_points=min_points, **kwargs
    ).slices


def _folds(n: int, folds: int) -> List[np.ndarray]:
    """Interior indices (1..n-2) split into ``folds`` interleaved folds."""

    interior = np.arange(1, n - 1)
    folds = max(1, min(folds, interior.size))
    return [interior[i::folds] for i in range(folds)]


def kfold_errors(k: np.ndarray, vols: np.ndarray, predict, folds: int = 5) -> np.ndarray:
    """Out-of-sample errors: for each fold of interior points, ``predict(
    train_mask, test_mask)`` returns vols at the test points from a model
    fitted on the training points. End points are never held out (no
    extrapolation is scored). Returns predicted minus observed per interior
    point (NaN where a fold's fit failed)."""

    n = k.size
    err = np.full(n, np.nan)
    for test in _folds(n, folds):
        train = np.ones(n, dtype=bool)
        train[test] = False
        mask = np.zeros(n, dtype=bool)
        mask[test] = True
        try:
            err[test] = np.asarray(predict(train, mask), dtype=float) - vols[test]
        except ValueError:
            continue
    return err[1:-1]


def spline_predict(k: np.ndarray, vols: np.ndarray):
    """Baseline: natural cubic spline of implied vol in k = ln(K/F)."""

    def predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        if train.sum() < 3:
            raise ValueError("spline baseline needs 3 training points")
        return CubicSpline(k[train], vols[train], bc_type="natural")(k[test])

    return predict


def benchmark_vs_baseline(
    calibrated_slice: CalibratedSlice, chain_slice: Optional[pd.DataFrame] = None, folds: int = 5
) -> Dict[str, object]:
    """Out-of-sample comparison of SVI with a cubic-spline baseline.

    Both models are refitted ``folds`` times with an interleaved fold of
    interior strikes held out and scored on it (``k``-fold CV, vol units).
    A spline scored on its own knots has zero error by construction, so
    only held-out errors are compared. ``relative_improvement = 1 -
    rmse_svi_cv / rmse_spline_cv`` (> 0: SVI predicts unseen strikes
    better). ``chain_slice`` is accepted for backward compatibility and
    ignored: the slice's own cleaned quotes are used."""

    fit = calibrated_slice.fit
    strikes = np.asarray(calibrated_slice.strikes, dtype=float)
    vols = np.asarray(calibrated_slice.observed_ivs, dtype=float)
    F, T = calibrated_slice.forward, calibrated_slice.tau
    k = np.log(strikes / F)
    if strikes.size < SVI_N_PARAMS + 3:
        raise ValueError("benchmark needs at least 8 quotes (6 to refit SVI plus held-out points)")

    def svi_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        refit = fit_svi_strike_slice(
            strikes[train],
            vols[train],
            F=F,
            T=T,
            weights=None if fit is None else fit.weights[train],
            loss="soft_l1" if fit is None else fit.loss,
            f_scale=0.005 if fit is None or fit.f_scale is None else fit.f_scale,
            enforce_no_arbitrage=False if fit is None else fit.enforce_no_arbitrage,
            x0=calibrated_slice.params,
        )
        return refit.vol(strikes[test])

    e_svi = kfold_errors(k, vols, svi_predict, folds)
    e_spl = kfold_errors(k, vols, spline_predict(k, vols), folds)
    both = np.isfinite(e_svi) & np.isfinite(e_spl)
    rmse_svi = float(np.sqrt(np.mean(e_svi[both] ** 2))) if both.any() else float("nan")
    rmse_spl = float(np.sqrt(np.mean(e_spl[both] ** 2))) if both.any() else float("nan")
    improvement = 1.0 - rmse_svi / rmse_spl if np.isfinite(rmse_spl) and rmse_spl > 0 else float("nan")
    return {
        "rmse_svi_in_sample": float(calibrated_slice.rmse),
        "rmse_svi_cv": rmse_svi,
        "rmse_spline_cv": rmse_spl,
        "relative_improvement": float(improvement),
        "n_scored": int(both.sum()),
        "folds": int(min(folds, max(strikes.size - 2, 1))),
        "method": "k-fold CV on interior strikes (end points never held out)",
    }
