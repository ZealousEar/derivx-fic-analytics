"""Smile-model comparison on one cleaned expiry slice.

Models (:data:`MODELS`):

* ``svi`` - raw SVI in total variance (5 parameters), equal weights in vol.
* ``sabr`` - Hagan lognormal SABR with fixed ``beta`` (default 1, the usual
  choice for equity index and FX) and alpha, rho, nu fitted with start point
  and bounds scaled by the forward (3 parameters).
* ``spline`` - natural cubic spline of implied vol in k = ln(K/F). It
  interpolates the quotes, so its in-sample error is zero by construction;
  it is a baseline for the out-of-sample score only. (This replaces the old
  'local_vol' label, which was this same spline and not a local-volatility
  model; Dupire local vol lives in :mod:`derivx.vol.ssvi`.)

Every model is scored on the same quotes: the slice is cleaned with
:func:`derivx.vol.quotes.clean_option_slice` (expired/same-day expiries
rejected, OTM quotes only, one quote per strike, put-call-parity forward,
vols recomputed from bid/ask mids when the slice has them, discounted with
``discount`` - the trade date's Treasury curve or put-call parity, see
:mod:`derivx.vol.discounting` - or the flat ``rate``)
and restricted to ``|k| <= std_band * atm_vol * sqrt(tau)``. Scores are
decimal vols (``*_bp`` columns: vol bp): ``rmse`` in sample and
``rmse_cv`` from k-fold cross-validation on interior strikes, which is the
comparable number across models with different numbers of parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from derivx.calibration.production import CV_STARTS, kfold_errors, spline_predict
from derivx.vol.discounting import Discounting
from derivx.vol.quotes import CleanSlice, apply_std_band, clean_option_slice, day_count
from derivx.vol.sabr import calibrate_sabr_smile, hagan_sabr_iv
from derivx.vol.svi import fit_svi_strike_slice

MODELS: Dict[str, str] = {
    "svi": "raw SVI in total variance, soft-L1 loss at 50 vol bp (5 params)",
    "sabr": "Hagan lognormal SABR, beta fixed, scale-aware alpha (3 params)",
    "spline": "natural cubic spline of IV in ln(K/F): interpolates quotes, score rmse_cv only",
}
N_PARAMS = {"svi": 5, "sabr": 3, "spline": 0}


@dataclass
class ModelComparison:
    """Fits and scores of several smile models on one cleaned slice."""

    table: pd.DataFrame
    clean: CleanSlice
    fits: Dict[str, Any] = field(default_factory=dict)

    @property
    def quotes(self) -> pd.DataFrame:
        return self.clean.quotes

    @property
    def forward(self) -> float:
        return self.clean.forward

    @property
    def tau(self) -> float:
        return self.clean.tau

    def predict(self, model: str, K) -> np.ndarray:
        """Model vols at strikes ``K`` (spline: within the quoted range)."""

        if model not in self.fits:
            raise ValueError(f"model {model!r} was not fitted (fitted: {sorted(self.fits)})")
        K = np.asarray(K, dtype=float)
        fit = self.fits[model]
        if model == "spline":
            return fit(np.log(K / self.forward))
        if model == "sabr":
            return np.asarray(hagan_sabr_iv(self.forward, K, self.tau, **fit), dtype=float)
        return np.asarray(fit.vol(K), dtype=float)


def _check_models(models: Iterable[str]) -> List[str]:
    out = list(models)
    for m in out:
        if m == "local_vol":
            raise ValueError(
                "'local_vol' was a cubic spline of implied vol, not a local-volatility model: use 'spline' "
                "(and derivx.vol.ssvi for Dupire local vol)"
            )
        if m not in MODELS:
            raise ValueError(f"unknown model {m!r}; choose from {sorted(MODELS)}")
    if not out:
        raise ValueError("select at least one model")
    return out


def prepare_slice(
    chain: pd.DataFrame,
    *,
    trade_date=None,
    rate: float = 0.0,
    div_yield: float = 0.0,
    min_days: float = 2.0,
    std_band: Optional[float] = 3.0,
    min_points: int = 8,
    hygiene: Optional[dict] = None,
    discount: Optional[Discounting] = None,
) -> CleanSlice:
    """Clean one expiry slice for model comparison (see module docs);
    raises ``ValueError`` (with what was removed) if it is expired, closer
    than ``min_days``, or has fewer than ``min_points`` usable quotes."""

    if chain is None or chain.empty:
        raise ValueError("chain must contain strike/implied_vol data")
    hyg = dict(hygiene or {})
    if "discount_factor" not in hyg:
        hyg["discount"] = discount
    clean = clean_option_slice(chain, trade_date=trade_date, rate=rate, div_yield=div_yield, min_days=min_days, **hyg)
    if clean.days < min_days:
        raise ValueError(
            f"expiry {clean.expiry.date()} is {clean.days} day(s) after the trade date "
            f"{clean.trade_date.date()}: expired or too short to score (min_days={min_days:g})"
        )
    if std_band is not None:
        clean = apply_std_band(clean, std_band)
    if len(clean.quotes) < min_points:
        raise ValueError(
            f"{len(clean.quotes)} usable quotes after hygiene (removed: {clean.counts}); need {min_points}"
        )
    return clean


def fit_models(
    chain: pd.DataFrame,
    models: Iterable[str] = ("svi", "sabr", "spline"),
    *,
    sabr_beta: float = 1.0,
    folds: int = 5,
    **prepare_kwargs,
) -> ModelComparison:
    """Clean one slice, fit each model and score it in and out of sample.
    A model whose fit fails gets NaN scores and the error in ``status``."""

    models = _check_models(models)
    clean = prepare_slice(chain, **prepare_kwargs)
    q = clean.quotes
    K = q["strike"].to_numpy(float)
    vols = q["implied_vol"].to_numpy(float)
    k = q["k"].to_numpy(float)
    F, T = clean.forward, clean.tau
    rows = []
    fits: Dict[str, Any] = {}
    fit: Any  # SviFit, SabrCalibration or CubicSpline, by model
    for name in models:
        row = {"model": name, "description": MODELS[name], "n_params": N_PARAMS[name], "n_points": int(K.size)}
        try:
            if name == "svi":
                fit = fit_svi_strike_slice(K, vols, F=F, T=T)
                fitted = fit.vol(K)

                def predict(train, test):  # starts from the training quotes only (no full-sample warm start)
                    refit = fit_svi_strike_slice(K[train], vols[train], F=F, T=T, n_starts=CV_STARTS)
                    return refit.vol(K[test])

            elif name == "sabr":
                fit = calibrate_sabr_smile(K, vols, T=T, F=F, beta=sabr_beta)
                fitted = fit.model_vols

                def predict(train, test):
                    refit = calibrate_sabr_smile(K[train], vols[train], T=T, F=F, beta=sabr_beta, n_starts=3)
                    return hagan_sabr_iv(F, K[test], T, **refit)

            else:
                fit = CubicSpline(k, vols, bc_type="natural")
                fitted = fit(k)
                predict = spline_predict(k, vols)
            err = fitted - vols
            cv = kfold_errors(k, vols, predict, folds)
            cv = cv[np.isfinite(cv)]
            row.update(
                rmse=float(np.sqrt(np.mean(err**2))),
                rmse_cv=float(np.sqrt(np.mean(cv**2))) if cv.size else float("nan"),
                max_abs_error_bp=float(np.max(np.abs(err)) * 1e4),
                status="ok",
            )
            if name == "sabr":
                row["status"] = "ok" + (
                    " (param at bound: " + ", ".join(p for p, v in fit.at_bound.items() if v) + ")"
                    if fit.any_at_bound
                    else ""
                )
            fits[name] = fit
        except ValueError as exc:
            row.update(rmse=float("nan"), rmse_cv=float("nan"), max_abs_error_bp=float("nan"), status=f"failed: {exc}")
        row["rmse_bp"] = row["rmse"] * 1e4
        row["rmse_cv_bp"] = row["rmse_cv"] * 1e4
        rows.append(row)
    cols = [
        "model",
        "rmse",
        "rmse_bp",
        "rmse_cv",
        "rmse_cv_bp",
        "max_abs_error_bp",
        "n_params",
        "n_points",
        "status",
        "description",
    ]
    table = pd.DataFrame(rows)[cols]
    table.attrs.update(
        trade_date=clean.trade_date.date().isoformat(),
        expiry=clean.expiry.date().isoformat(),
        tau=T,
        forward=F,
        forward_source=clean.forward_source,
        discount_factor=clean.discount_factor,
        rate=clean.rate,
        discount_source=clean.discount_source,
        removed=clean.counts,
    )
    return ModelComparison(table=table, clean=clean, fits=fits)


def compare_models(chain: pd.DataFrame, models: Iterable[str] = ("svi", "sabr", "spline"), **kwargs) -> pd.DataFrame:
    """Score smile models on one expiry slice; the table of
    :func:`fit_models` (slice metadata in ``table.attrs``)."""

    return fit_models(chain, models, **kwargs).table


def select_expiry(chain: pd.DataFrame, target_days: float = 30.0, min_days: float = 2.0, trade_date=None):
    """The expiry closest to ``target_days`` among those at least
    ``min_days`` after the trade date (a fixed-tenor slice instead of the
    first, often expired, expiry). Raises ``ValueError`` if none qualifies."""

    if trade_date is None:
        if "trade_date" not in chain.columns:
            raise ValueError("pass trade_date or include a 'trade_date' column")
        dates = pd.to_datetime(chain["trade_date"]).dt.normalize().unique()
        if len(dates) != 1:
            raise ValueError("chain has several trade dates: pass trade_date or filter first")
        trade_date = dates[0]
    expiries = pd.to_datetime(chain["expiry"]).dt.normalize().unique()
    days = np.array([day_count(trade_date, e) for e in expiries], dtype=float)
    valid = days >= min_days
    if not valid.any():
        raise ValueError(f"no expiry at least {min_days:g} days after the trade date")
    i = int(np.argmin(np.where(valid, np.abs(days - target_days), np.inf)))
    return pd.Timestamp(expiries[i])


def compare_models_by_expiry(
    chain: pd.DataFrame,
    models: Iterable[str] = ("svi", "sabr", "spline"),
    *,
    rate: float = 0.0,
    min_days: float = 2.0,
    max_days: Optional[float] = None,
    sabr_beta: float = 1.0,
    folds: int = 5,
    std_band: Optional[float] = 3.0,
    min_points: int = 8,
    discount: Optional[Discounting] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """:func:`fit_models` on every (trade_date, expiry) slice of ``chain``
    with ``min_days <= days to expiry <= max_days``. Forwards come from
    :func:`derivx.calibration.production.clean_chain` (put-call parity, or
    interpolated from the other expiries' parity forwards when parity
    fails), and so do the discount factors (``discount``, else the flat
    ``rate``), so they match the calibration tab. Returns ``(scores,
    skipped)``: one row per (slice, model) with the table columns of
    :func:`fit_models` plus ``trade_date``, ``expiry``, ``days``, ``tau``,
    ``forward``, ``forward_method``; and one row per skipped slice with the
    reason. No slice is dropped silently and no score is clipped."""

    from derivx.calibration.production import clean_chain

    models = _check_models(models)
    cleaned, bad = clean_chain(chain, rate=rate, min_days=min_days, discount=discount)
    skipped = [{"trade_date": s.trade_date, "expiry": s.expiry, "reason": s.reason} for s in bad]
    frames = []
    for td, exp, group, clean in cleaned:
        base = {"trade_date": td, "expiry": exp}
        if clean.days < min_days:
            skipped.append({**base, "reason": f"{clean.days} day(s) to expiry < min_days={min_days:g}"})
            continue
        if max_days is not None and clean.days > max_days:
            skipped.append({**base, "reason": f"{clean.days} days to expiry > max_days={max_days:g}"})
            continue
        try:
            comp = fit_models(
                group,
                models,
                sabr_beta=sabr_beta,
                folds=folds,
                trade_date=td,
                rate=rate,
                min_days=min_days,
                std_band=std_band,
                min_points=min_points,
                hygiene={
                    "forward": clean.forward,
                    "forward_source": clean.forward_source,
                    "discount_factor": clean.discount_factor,
                    "discount_source": clean.discount_source,
                },
            )
        except ValueError as exc:
            skipped.append({**base, "reason": str(exc)})
            continue
        frames.append(
            comp.table.assign(
                trade_date=td,
                expiry=exp,
                days=clean.days,
                tau=clean.tau,
                forward=clean.forward,
                forward_method=clean.forward_method,
                rate_pct=100.0 * clean.rate,
                discount_method=clean.discount_method,
            )
        )
    scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not scores.empty:
        lead = ["trade_date", "expiry", "days", "tau", "forward", "forward_method", "rate_pct", "discount_method"]
        scores = scores[lead + [c for c in scores.columns if c not in lead]]
    return scores, pd.DataFrame(skipped, columns=["trade_date", "expiry", "reason"])


def summarise_comparison(scores: pd.DataFrame, metric: str = "rmse_cv_bp") -> pd.DataFrame:
    """Per model: slices scored, failures, median / mean / max of ``metric``
    and how often the model has the lowest ``metric`` among the models
    scored on the same slice (ties count for each). Every scored slice is
    included: nothing is clipped or filtered."""

    if scores is None or scores.empty:
        return pd.DataFrame()
    key = ["trade_date", "expiry"]
    ok = scores[np.isfinite(scores[metric])]
    best = ok.groupby(key)[metric].transform("min")
    ok = ok.assign(best=np.isclose(ok[metric], best, rtol=0.0, atol=1e-12))
    rows = []
    for model, g in scores.groupby("model", sort=False):
        good = ok[ok["model"] == model]
        rows.append(
            {
                "model": model,
                "slices": int(len(g)),
                "failed": int((~np.isfinite(g[metric])).sum()),
                f"median_{metric}": float(good[metric].median()) if len(good) else np.nan,
                f"mean_{metric}": float(good[metric].mean()) if len(good) else np.nan,
                f"max_{metric}": float(good[metric].max()) if len(good) else np.nan,
                "best_on": int(good["best"].sum()),
            }
        )
    return pd.DataFrame(rows)


def pairwise_wins(scores: pd.DataFrame, a: str, b: str, metric: str = "rmse_cv_bp") -> Dict[str, Any]:
    """On slices where both ``a`` and ``b`` were scored: how many times
    ``a``'s ``metric`` is lower, higher or equal, and the median of
    ``metric(a) - metric(b)`` (negative: ``a`` is better)."""

    if scores is None or scores.empty:
        return {"slices": 0, "a_better": 0, "b_better": 0, "ties": 0, "median_diff": float("nan")}
    key = ["trade_date", "expiry"]
    pa = scores[scores["model"] == a].set_index(key)[metric]
    pb = scores[scores["model"] == b].set_index(key)[metric]
    both = pd.concat([pa.rename("a"), pb.rename("b")], axis=1, join="inner").dropna()
    diff = both["a"] - both["b"]
    return {
        "slices": int(len(both)),
        "a_better": int((diff < 0).sum()),
        "b_better": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "median_diff": float(diff.median()) if len(diff) else float("nan"),
    }


MODEL_LABELS = {"svi": "SVI", "sabr": "SABR", "spline": "IV spline"}


def describe_pairwise(
    a: str,
    b: str,
    res: Dict[str, Any],
    labels: Optional[Dict[str, str]] = None,
    items: str = "expiries",
    unit: str = "vol bp",
) -> str:
    """One sentence on :func:`pairwise_wins` ``res``, worded by its sign
    (never 'A outperforms B' when A is worse). ``labels`` name the models
    (default :data:`MODEL_LABELS`); ``items`` are what was scored (expiries,
    dates, points) and ``unit`` the RMSE unit."""

    names = MODEL_LABELS if labels is None else labels
    la, lb = names.get(a, a), names.get(b, b)
    n = int(res.get("slices", 0))
    if not n:
        return f"{la} vs {lb}: nothing was scored by both."
    wins_a, wins_b, d = int(res["a_better"]), int(res["b_better"]), float(res["median_diff"])
    if wins_a > wins_b:
        lead = f"{la} has the lower out-of-sample RMSE on {wins_a} of {n} {items}"
    elif wins_b > wins_a:
        lead = f"{lb} has the lower out-of-sample RMSE on {wins_b} of {n} {items}"
    else:
        lead = f"{la} and {lb} each have the lower out-of-sample RMSE on {wins_a} of {n} {items}"
    who = la if d < 0 else lb if d > 0 else "neither"
    return f"{lead} (median {la} - {lb} = {d:+.1f} {unit}; {who} better at the median)."
