"""Walk-forward validation of smile forecasts on a daily history of option
chain snapshots (one snapshot per trade date).

Setup
-----
For every origin date ``t`` the target is the snapshot exactly ``horizon``
trading days later (exchange calendar, :mod:`derivx.data.calendar`); an
origin without one is skipped and reported, so every score has the stated
horizon. Each (trade date, expiry) slice is cleaned with
:func:`derivx.calibration.production.clean_chain` (expired and same-day
expiries dropped, OTM quotes only, one quote per strike, put-call-parity
forwards, vols re-inverted from bid/ask mids at ``rate``). Origin quotes are
kept within ``|k| <= std_band * atm_vol * sqrt(tau)`` (k = ln(K/F)).

Calibration window: SVI (total variance, fitted with T = the origin's tau)
is fitted to the origin slice's quotes pooled over the last ``window``
snapshots up to and including ``t`` (each day's quotes at its own k =
ln(K/F_day)); ``window = 1`` uses the origin day only. Pooling assumes the
smile is stable in k over the window.

Forecasts of the target day's quote at the same expiry, strike and option
type (quotes are matched on (expiry, strike, option_type): a call is never
scored against a put):

* ``svi_sticky_strike`` - the origin SVI at the strike, on the origin
  forward: uses origin information only.
* ``persistence`` - the origin quote itself: no change (benchmark of
  ``svi_sticky_strike``).
* ``svi_sticky_moneyness`` - the origin SVI at the same k on the *target*
  day's forward: conditional on the realised forward move.
* ``sticky_moneyness`` - the origin quotes, linear in k, read at the target
  k: also conditional on the realised forward (benchmark of
  ``svi_sticky_moneyness``).

Scores (vols in decimals, ``*_bp`` = vol basis points): errors are forecast
minus target vol. ``noise_bp`` is the RMS half bid/ask width in vol of the
target quotes: a forecast error at that level is within the quoted market.
``skill = 1 - RMSE / RMSE(benchmark)`` (> 0: better than the benchmark).
``win_rate`` is the share of informative quotes (actual move larger than
the target's half spread) on which the forecast error is smaller than the
benchmark's, with a Wilson 95% interval. ``dm_t`` is a Diebold-Mariano
statistic on the daily mean squared-error differential (clustered by
target day; negative favours the forecast; needs 3+ days).

Regimes: a constant-maturity 30-day ATM vol per snapshot (ATM vol of each
slice at k = 0, total variance linear in T between the two expiries that
bracket 30 days; no extrapolation), bucketed low / mid / high by its
percentile (terciles) within the loaded history. The buckets use the whole
sample's distribution (labels only, never forecasts) and need
``min_regime_days`` origin days to be reported as sufficient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from derivx.calibration.production import clean_chain
from derivx.data.calendar import add_trading_days, trading_days_between
from derivx.vol.quotes import CleanSlice, apply_std_band, atm_vol
from derivx.vol.svi import fit_svi_strike_slice, svi_iv

PREDICTORS: Dict[str, str] = {
    "svi_sticky_strike": "origin SVI at the same strike on the origin forward (origin information only)",
    "persistence": "origin quote at the same expiry, strike and type: no change",
    "svi_sticky_moneyness": "origin SVI at the same k = ln(K/F) on the target forward (conditional on the forward)",
    "sticky_moneyness": "origin quotes, linear in k, at the target k (conditional on the forward)",
}
BENCHMARK: Dict[str, Optional[str]] = {
    "svi_sticky_strike": "persistence",
    "persistence": None,
    "svi_sticky_moneyness": "sticky_moneyness",
    "sticky_moneyness": "persistence",
}


@dataclass(frozen=True)
class WalkForwardSettings:
    """``horizon``: trading days from origin to target; ``window``:
    snapshots pooled in the origin calibration; ``min_days`` /
    ``max_days``: calendar days to expiry an origin slice must be within
    (``max_days`` None: no limit); ``rate``: continuously compounded
    discount rate for vols re-inverted from mids; ``warm_start``: start each
    SVI fit from the previous origin's parameters for the same expiry (one
    start instead of several; about twice as fast, the same fits to 1e-4
    vol bp on the synthetic history)."""

    horizon: int = 1
    window: int = 1
    std_band: Optional[float] = 3.0
    rate: float = 0.0
    min_days: float = 2.0
    max_days: Optional[float] = None
    min_points: int = 8
    enforce_arbitrage: bool = True
    warm_start: bool = True
    regime_tenor_days: float = 30.0
    min_regime_days: int = 3

    def __post_init__(self) -> None:
        if int(self.horizon) < 1:
            raise ValueError("horizon must be >= 1 trading day")
        if int(self.window) < 1:
            raise ValueError("window must be >= 1 snapshot")
        if self.min_points <= 5:
            raise ValueError("min_points must exceed the 5 SVI parameters (use >= 6)")
        if self.std_band is not None and self.std_band <= 0:
            raise ValueError("std_band must be > 0 or None")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class WalkForwardResult:
    """Everything a run produced (see the module docs). ``slices``: one row
    per (origin, expiry); ``quotes``: one row per scored quote; ``daily``:
    per target day; ``summary``: per predictor, pooled over every scored
    quote; ``regimes``: per snapshot; ``by_regime``: per origin bucket;
    ``skipped``: origins or slices not scored, with the reason."""

    settings: WalkForwardSettings
    slices: pd.DataFrame
    quotes: pd.DataFrame
    daily: pd.DataFrame
    summary: pd.DataFrame
    regimes: pd.DataFrame
    by_regime: pd.DataFrame
    skipped: pd.DataFrame
    dates: List[pd.Timestamp] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.quotes.empty


def wilson_interval(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (NaN when n = 0)."""

    if n <= 0:
        return float("nan"), float("nan")
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def diebold_mariano(daily_diff: Iterable[float]) -> float:
    """t-statistic of the mean of a daily loss differential (NaN for fewer
    than 3 days or zero dispersion)."""

    d = np.asarray([x for x in daily_diff if np.isfinite(x)], dtype=float)
    if d.size < 3:
        return float("nan")
    sd = float(d.std(ddof=1))
    return float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else float("nan")


def constant_maturity_atm_vol(slices: Iterable[CleanSlice], tenor_days: float = 30.0) -> Tuple[float, str]:
    """ATM vol at ``tenor_days`` (ACT/365F) from one snapshot's cleaned
    slices: each slice's vol at k = 0 (linear in k between the quotes either
    side), then total variance ``vol^2 T`` linear in T between the two
    expiries bracketing the tenor. Returns ``(vol, method)``; NaN with the
    reason when no pair of expiries brackets the tenor."""

    pts = []
    for c in slices:
        q = c.quotes
        if c.tau <= 0 or q.empty or not ((q["k"] <= 0).any() and (q["k"] >= 0).any()):
            continue
        pts.append((c.tau, atm_vol(q), c.days))
    target = tenor_days / 365.0
    if not pts:
        return float("nan"), "no slice has quotes on both sides of the forward"
    pts.sort()
    taus = np.array([p[0] for p in pts])
    if target < taus[0] or target > taus[-1]:
        return float("nan"), f"no expiries bracket {tenor_days:g} days (have {pts[0][2]}-{pts[-1][2]} days)"
    w = np.array([v * v * t for t, v, _ in pts])
    j = int(np.clip(np.searchsorted(taus, target), 1, taus.size - 1))
    var = float(np.interp(target, taus, w))
    return float(np.sqrt(var / target)), f"total variance linear in T between {pts[j - 1][2]}d and {pts[j][2]}d"


def regime_buckets(values: pd.Series) -> pd.DataFrame:
    """Percentile (within ``values``) and tercile bucket of each value;
    'n/a' for missing values or with fewer than 3 finite values."""

    v = pd.to_numeric(values, errors="coerce")
    pct = v.rank(pct=True, method="average")
    bucket = np.where(pct <= 1 / 3, "low", np.where(pct <= 2 / 3, "mid", "high"))
    bucket = np.where(v.notna(), bucket, "n/a")
    if int(v.notna().sum()) < 3:
        bucket = np.full(len(v), "n/a", dtype=object)
    return pd.DataFrame({"percentile": pct, "bucket": bucket}, index=values.index)


def _history(chains: Mapping[object, pd.DataFrame]) -> Dict[pd.Timestamp, pd.DataFrame]:
    days: Dict[pd.Timestamp, pd.DataFrame] = {}
    for key, df in chains.items():
        if df is None or df.empty:
            continue
        if "trade_date" in df.columns:
            tds = pd.to_datetime(df["trade_date"]).dt.normalize().unique()
            if len(tds) != 1:
                raise ValueError(f"snapshot {key} holds {len(tds)} trade dates: pass one snapshot per trade date")
            td = pd.Timestamp(tds[0])
            if key is not None and pd.Timestamp(key).normalize() != td:
                raise ValueError(f"snapshot keyed {pd.Timestamp(key).date()} has trade_date {td.date()}")
        else:
            td = pd.Timestamp(key).normalize()
            df = df.assign(trade_date=td)
        if td in days:
            raise ValueError(f"two snapshots for trade date {td.date()}")
        days[td] = df
    return dict(sorted(days.items()))


def _rmse(err: np.ndarray) -> float:
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err**2))) if err.size else float("nan")


def _skill(rmse: float, benchmark: float) -> float:
    """1 - rmse / benchmark; NaN when the benchmark error is 0 or missing
    (no skill can be measured against a perfect benchmark)."""

    if not np.isfinite(benchmark) or benchmark <= 0 or not np.isfinite(rmse):
        return float("nan")
    return float(1.0 - rmse / benchmark)


def run_walk_forward(
    chains: Mapping[object, pd.DataFrame], settings: Optional[WalkForwardSettings] = None, **kwargs
) -> WalkForwardResult:
    """Walk-forward test over ``chains`` (``{trade_date: snapshot}``; see
    the module docs). Settings come from ``settings`` or keyword arguments
    of :class:`WalkForwardSettings`."""

    cfg = settings or WalkForwardSettings(**kwargs)
    history = _history(chains)
    dates = list(history)
    cleaned: Dict[pd.Timestamp, Dict[pd.Timestamp, Tuple[CleanSlice, CleanSlice]]] = {}
    skipped: List[Dict[str, object]] = []
    regime_rows = []
    for td, df in history.items():
        slices, bad = clean_chain(df, rate=cfg.rate, min_days=1.0)
        for s in bad:
            skipped.append({"origin": td, "expiry": s.expiry, "stage": "hygiene", "reason": s.reason})
        day: Dict[pd.Timestamp, Tuple[CleanSlice, CleanSlice]] = {}
        for _, exp, _, clean in slices:
            if clean.days < 1 or clean.quotes.empty:
                continue
            banded = apply_std_band(clean, cfg.std_band) if cfg.std_band is not None else clean
            day[pd.Timestamp(exp)] = (clean, banded)
        cleaned[td] = day
        vol, how = constant_maturity_atm_vol([c for c, _ in day.values()], cfg.regime_tenor_days)
        regime_rows.append({"trade_date": td, "atm_vol_cm": vol, "method": how})
    regimes = pd.DataFrame(regime_rows, columns=["trade_date", "atm_vol_cm", "method"])
    if not regimes.empty:
        regimes = pd.concat([regimes, regime_buckets(regimes["atm_vol_cm"])], axis=1)
    bucket_of = dict(zip(regimes.get("trade_date", []), regimes.get("bucket", [])))

    date_set = set(dates)
    warm: Dict[pd.Timestamp, Dict[str, float]] = {}  # last fitted params per expiry (warm starts)
    quote_frames: List[pd.DataFrame] = []
    slice_rows: List[Dict[str, object]] = []
    for i, t in enumerate(dates):
        target = add_trading_days(t, int(cfg.horizon))
        if target not in date_set:
            nxt = dates[i + 1] if i + 1 < len(dates) else None
            gap = (
                f"; the next snapshot, {nxt.date()}, is {trading_days_between(t, nxt)} trading day(s) later"
                if nxt
                else ""
            )
            skipped.append(
                {
                    "origin": t,
                    "expiry": pd.NaT,
                    "stage": "horizon",
                    "reason": f"no snapshot {cfg.horizon} trading day(s) later ({target.date()}){gap}",
                }
            )
            continue
        window_days = dates[max(0, i - int(cfg.window) + 1) : i + 1]
        if len(window_days) < int(cfg.window):
            skipped.append(
                {
                    "origin": t,
                    "expiry": pd.NaT,
                    "stage": "window",
                    "reason": f"needs {cfg.window} snapshots up to the origin, has {len(window_days)}",
                }
            )
            continue
        for exp, (full_t, band_t) in sorted(cleaned[t].items()):
            base = {"origin": t, "expiry": exp}
            if full_t.days < cfg.min_days:
                skipped.append(
                    {**base, "stage": "slice", "reason": f"{full_t.days} day(s) to expiry < {cfg.min_days:g}"}
                )
                continue
            if cfg.max_days is not None and full_t.days > cfg.max_days:
                continue  # outside the requested maturity range (not a failure)
            tgt = cleaned[target].get(exp)
            if tgt is None:
                skipped.append({**base, "stage": "slice", "reason": f"expiry not in the {target.date()} snapshot"})
                continue
            q = band_t.quotes.sort_values("k")
            n_left, n_right = int((q["k"] < 0).sum()), int((q["k"] > 0).sum())
            if len(q) < cfg.min_points or n_left < 2 or n_right < 2:
                skipped.append(
                    {
                        **base,
                        "stage": "slice",
                        "reason": f"{len(q)} origin quotes ({n_left} below / {n_right} above F); need "
                        f"{cfg.min_points} with 2 per side",
                    }
                )
                continue
            pool_k, pool_v = [], []
            for d in window_days:
                pair = cleaned[d].get(exp)
                if pair is not None and not pair[1].quotes.empty:
                    pool_k.append(pair[1].quotes["k"].to_numpy(float))
                    pool_v.append(pair[1].quotes["implied_vol"].to_numpy(float))
            k_pool, v_pool = np.concatenate(pool_k), np.concatenate(pool_v)
            F_t, T_t = full_t.forward, full_t.tau
            try:
                fit = fit_svi_strike_slice(
                    F_t * np.exp(k_pool),
                    v_pool,
                    F=F_t,
                    T=T_t,
                    enforce_no_arbitrage=cfg.enforce_arbitrage,
                    x0=warm.get(exp) if cfg.warm_start else None,
                )
            except ValueError as exc:
                skipped.append({**base, "stage": "fit", "reason": f"SVI fit failed: {exc}"})
                continue
            warm[exp] = fit.params
            tq = tgt[0].quotes
            cols = ["strike", "option_type", "k", "implied_vol"] + [c for c in ("iv_bid", "iv_ask") if c in tq]
            m = q[["strike", "option_type", "k", "implied_vol"]].merge(
                tq[cols], on=["strike", "option_type"], how="inner", suffixes=("_origin", "_target")
            )
            if m.empty:
                skipped.append({**base, "stage": "match", "reason": "no target quote at an origin (strike, type)"})
                continue
            K = m["strike"].to_numpy(float)
            iv_o, iv_t = m["implied_vol_origin"].to_numpy(float), m["implied_vol_target"].to_numpy(float)
            F_T = tgt[0].forward
            k_target = np.log(K / F_T)
            preds = {
                "svi_sticky_strike": np.asarray(svi_iv(K, fit.params, F=F_t, T=T_t), dtype=float),
                "persistence": iv_o,
                "svi_sticky_moneyness": np.asarray(svi_iv(K, fit.params, F=F_T, T=T_t), dtype=float),
                "sticky_moneyness": np.interp(k_target, q["k"].to_numpy(float), q["implied_vol"].to_numpy(float)),
            }
            if "iv_bid" in m and "iv_ask" in m:
                half = 0.5 * (m["iv_ask"].to_numpy(float) - m["iv_bid"].to_numpy(float))
            else:
                half = np.full(K.size, np.nan)
            frame = pd.DataFrame(
                {
                    "origin": t,
                    "target": target,
                    "horizon_td": int(cfg.horizon),
                    "expiry": exp,
                    "days_to_expiry": int(full_t.days),
                    "strike": K,
                    "option_type": m["option_type"].to_numpy(),
                    "k_origin": m["k_origin"].to_numpy(float),
                    "k_target": k_target,
                    "iv_origin": iv_o,
                    "iv_target": iv_t,
                    "half_spread": half,
                }
            )
            for name, pred in preds.items():
                frame[f"pred_{name}"] = pred
                frame[f"err_{name}"] = pred - iv_t
            quote_frames.append(frame)
            row = {
                "origin": t,
                "target": target,
                "horizon_td": int(cfg.horizon),
                "expiry": exp,
                "days_to_expiry": int(full_t.days),
                "n_points": int(len(frame)),
                "window_snapshots": len(pool_k),
                "n_calibration": int(k_pool.size),
                "svi_fit_rmse_bp": fit.rmse_bp,
                "forward_origin": F_t,
                "forward_target": F_T,
                "noise_bp": _rmse(half) * 1e4,
                "regime": bucket_of.get(t, "n/a"),
            }
            for name in PREDICTORS:
                row[f"rmse_bp_{name}"] = _rmse(frame[f"err_{name}"]) * 1e4
            row["skill_svi_vs_persistence"] = _skill(row["rmse_bp_svi_sticky_strike"], row["rmse_bp_persistence"])
            slice_rows.append(row)

    quotes = pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame()
    slices_df = pd.DataFrame(slice_rows)
    skipped_df = pd.DataFrame(skipped, columns=["origin", "expiry", "stage", "reason"])
    daily = _daily(quotes, bucket_of)
    summary = _summary(quotes)
    by_regime = _by_regime(quotes, bucket_of, cfg.min_regime_days)
    return WalkForwardResult(
        settings=cfg,
        slices=slices_df,
        quotes=quotes,
        daily=daily,
        summary=summary,
        regimes=regimes,
        by_regime=by_regime,
        skipped=skipped_df,
        dates=dates,
    )


def _daily(quotes: pd.DataFrame, bucket_of: Mapping[pd.Timestamp, str]) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()
    rows = []
    for (origin, target), g in quotes.groupby(["origin", "target"], sort=True):
        row = {
            "origin": origin,
            "target": target,
            "n_quotes": int(len(g)),
            "n_expiries": int(g["expiry"].nunique()),
            "noise_bp": _rmse(g["half_spread"]) * 1e4,
            "regime": bucket_of.get(origin, "n/a"),
        }
        for name in PREDICTORS:
            row[f"rmse_bp_{name}"] = _rmse(g[f"err_{name}"]) * 1e4
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()
    informative = np.abs(quotes["iv_target"] - quotes["iv_origin"]) > quotes["half_spread"].fillna(0.0)
    rows = []
    for name, description in PREDICTORS.items():
        err = quotes[f"err_{name}"].to_numpy(float)
        bench = BENCHMARK[name]
        row: Dict[str, object] = {
            "predictor": name,
            "benchmark": bench or "",
            "n_quotes": int(np.isfinite(err).sum()),
            "n_days": int(quotes["target"].nunique()),
            "rmse_bp": _rmse(err) * 1e4,
            "mae_bp": float(np.nanmean(np.abs(err)) * 1e4),
            "noise_bp": _rmse(quotes["half_spread"]) * 1e4,
        }
        if bench:
            e_b = quotes[f"err_{bench}"].to_numpy(float)
            row["benchmark_rmse_bp"] = _rmse(e_b) * 1e4
            row["skill"] = _skill(row["rmse_bp"], row["benchmark_rmse_bp"])
            sq = pd.DataFrame({"day": quotes["target"], "d": err**2 - e_b**2})
            row["dm_t"] = diebold_mariano(sq.groupby("day")["d"].mean())
            mask = informative.to_numpy() & np.isfinite(err) & np.isfinite(e_b)
            wins = int((np.abs(err[mask]) < np.abs(e_b[mask])).sum())
            n = int(mask.sum())
            lo, hi = wilson_interval(wins, n)
            row.update(win_rate=wins / n if n else np.nan, win_n=n, win_lo=lo, win_hi=hi)
        else:
            row.update(benchmark_rmse_bp=np.nan, skill=np.nan, dm_t=np.nan, win_rate=np.nan, win_n=0)
            row.update(win_lo=np.nan, win_hi=np.nan)
        row["description"] = description
        rows.append(row)
    return pd.DataFrame(rows)


def _by_regime(quotes: pd.DataFrame, bucket_of: Mapping[pd.Timestamp, str], min_days: int) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()
    q = quotes.assign(regime=quotes["origin"].map(lambda d: bucket_of.get(d, "n/a")))
    rows = []
    for bucket in ("low", "mid", "high", "n/a"):
        g = q[q["regime"] == bucket]
        if g.empty:
            continue
        n_days = int(g["origin"].nunique())
        row = {
            "regime": bucket,
            "origin_days": n_days,
            "n_quotes": int(len(g)),
            "sufficient": bool(bucket != "n/a" and n_days >= min_days),
        }
        for name in PREDICTORS:
            row[f"rmse_bp_{name}"] = _rmse(g[f"err_{name}"]) * 1e4
        row["skill_svi_vs_persistence"] = _skill(row["rmse_bp_svi_sticky_strike"], row["rmse_bp_persistence"])
        rows.append(row)
    return pd.DataFrame(rows)


def walk_forward_test(
    chains: Mapping[object, pd.DataFrame], window: int = 1, horizon: int = 1, **kwargs
) -> pd.DataFrame:
    """Per-slice table of :func:`run_walk_forward` (kept for scripts)."""

    return run_walk_forward(chains, window=window, horizon=horizon, **kwargs).slices
