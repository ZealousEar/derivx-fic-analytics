"""Walk-forward test of yield-curve forecasts on a dated curve history
(e.g. the US Treasury par curve, the Bank of England OIS curve or the ECB
AAA curve from :class:`derivx.data.rates_history.HistoryStore`).

Setup: ``levels`` holds one curve per observation date (decimals by
tenor). For every origin ``t`` with ``window`` observations up to and
including it, the target is the observation ``horizon`` rows later; every
step from the window's first date to the target must span at most
``max_gap_days`` calendar days (a daily history: a monthly stretch is
skipped and reported, never scored as a daily forecast). Models are
estimated on the window only (no look-ahead).

Forecasts of the target curve, per tenor:

* ``random_walk`` - the origin curve: no change (the benchmark of every
  other forecast);
* ``ns_ar1``      - Diebold-Li (2006): Nelson-Siegel factors (decay fixed
  at 1.368y, :data:`derivx.rates.curve_models.DL_TAU`) by least squares on
  each window date, an AR(1) per factor on the window, iterated ``horizon``
  steps; the forecast adds the forecast factor change to the origin curve,
  so the origin's fitting residual (persistent for par curves) carries;
* ``ns_curve``    - the same factor forecast read off the Nelson-Siegel
  curve alone (the paper's forecast; the origin residual is dropped);
* ``pca_ar1``     - principal components of the window's levels (first
  ``n_factors``), an AR(1) per score, the forecast score change applied to
  the origin curve.

Scores (bp): errors are forecast minus actual. ``skill = 1 - RMSE /
RMSE(random walk)`` (> 0 beats it); ``days won`` counts target dates on
which the forecast's mean squared error across tenors is below the random
walk's, with a Wilson 95% interval; ``dm_t`` is the Diebold-Mariano
statistic on that daily differential (overlapping horizons handled as in
:func:`derivx.validation.backtest.diebold_mariano`; negative favours the
forecast). Random walks are hard to beat at short horizons: expect skill
near zero or negative unless the window holds a trend.

Regimes: the trailing ``vol_window``-observation standard deviation of the
daily change of the curve's average level (bp per observation) at each
origin, bucketed low / mid / high by terciles of this history (labels only,
from the whole sample), with ``min_regime_days`` origins to be sufficient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from derivx.rates.conventions import parse_tenor
from derivx.rates.curve_models import DL_TAU, ns_loadings
from derivx.validation.backtest import diebold_mariano, regime_buckets, wilson_interval

BP = 1e4
PREDICTORS: Dict[str, str] = {
    "random_walk": "origin curve: no change (benchmark)",
    "ns_ar1": "Diebold-Li NS factors, AR(1) on the window, change added to the origin curve",
    "ns_curve": "Diebold-Li NS factors, AR(1), read off the NS curve (origin residual dropped)",
    "pca_ar1": "PCA of the window's levels, AR(1) per score, change added to the origin curve",
}
BENCHMARK = "random_walk"


@dataclass(frozen=True)
class CurveForecastSettings:
    """``horizon``: observations ahead (business days for a daily history);
    ``window``: observations the models are estimated on (>= 30);
    ``max_gap_days``: calendar days one observation step may span;
    ``n_factors``: PCA factors; ``vol_window`` and ``min_regime_days`` set
    the regime labels."""

    horizon: int = 1
    window: int = 250
    max_gap_days: int = 5
    n_factors: int = 3
    vol_window: int = 20
    min_regime_days: int = 5

    def __post_init__(self) -> None:
        if int(self.horizon) < 1:
            raise ValueError("horizon must be >= 1 observation")
        if int(self.window) < 30:
            raise ValueError("window must be >= 30 observations (an AR(1) per factor needs a sample)")
        if not 1 <= int(self.n_factors) <= 5:
            raise ValueError("n_factors must be 1-5")
        if int(self.max_gap_days) < 1 or int(self.vol_window) < 5:
            raise ValueError("max_gap_days must be >= 1 and vol_window >= 5")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class CurveForecastResult:
    """``errors``: one row per (origin, tenor) with each predictor's error
    in bp; ``daily``: per target date; ``summary``: per predictor;
    ``by_tenor``: RMSE per tenor and predictor; ``regimes``: per origin;
    ``by_regime``; ``skipped``: origins not scored, with the reason."""

    settings: CurveForecastSettings
    tenors: Tuple[str, ...]
    errors: pd.DataFrame
    daily: pd.DataFrame
    summary: pd.DataFrame
    by_tenor: pd.DataFrame
    regimes: pd.DataFrame
    by_regime: pd.DataFrame
    skipped: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.errors.empty


def _ar1_forecast(x: np.ndarray, h: int) -> float:
    """h-step forecast of the last value of ``x`` from an AR(1) fitted by
    least squares on ``x`` (x_{s+1} = c + phi x_s); with a unit root or a
    degenerate sample the forecast is the last value."""

    a, b = x[:-1], x[1:]
    va = float(np.var(a))
    if va <= 0:
        return float(x[-1])
    phi = float(np.cov(a, b, bias=True)[0, 1] / va)
    c = float(b.mean() - phi * a.mean())
    if not np.isfinite(phi) or abs(phi) >= 1.0:
        return float(x[-1])
    y = float(x[-1])
    for _ in range(h):
        y = c + phi * y
    return y


def _rmse(e: np.ndarray) -> float:
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e * e))) if e.size else float("nan")


def run_curve_walk_forward(levels: pd.DataFrame, settings: CurveForecastSettings = CurveForecastSettings()):
    """Walk-forward over ``levels`` (index: dates, oldest first after
    sorting; columns: tenors; decimals; rows with a missing tenor are
    dropped). See the module docs."""

    cfg = settings
    if levels is None or levels.empty:
        raise ValueError("no curve history to test")
    tenors = sorted((str(c) for c in levels.columns), key=lambda t: parse_tenor(t).years)
    if len(tenors) < 4:
        raise ValueError(f"pick at least 4 tenors for the factor models (got {len(tenors)})")
    frame = levels[tenors].sort_index().dropna(how="any").astype(float)
    dates = [pd.Timestamp(d) for d in frame.index]
    Y = frame.to_numpy(float)
    n, m = Y.shape
    h, w = int(cfg.horizon), int(cfg.window)
    if n < w + h:
        raise ValueError(f"{n} complete curves: need window + horizon = {w + h}")
    years = np.array([parse_tenor(t).years for t in tenors])
    X = ns_loadings(years, DL_TAU)
    betas = np.linalg.lstsq(X, Y.T, rcond=None)[0].T  # (n, 3) NS factors per date
    gaps = np.array([(dates[i + 1] - dates[i]).days for i in range(n - 1)])
    bad_step = np.r_[0, np.cumsum(gaps > cfg.max_gap_days)]  # bad steps before each row
    level = Y.mean(axis=1) * BP
    dlevel = np.r_[np.nan, np.diff(level)]
    dlevel[np.r_[False, gaps > cfg.max_gap_days]] = np.nan
    rows, skipped, regime_rows = [], [], []
    k = int(cfg.n_factors)
    for i in range(w - 1, n - h):
        origin, target = dates[i], dates[i + h]
        lo = i - w + 1
        if bad_step[i + h] - bad_step[lo] > 0:
            skipped.append(
                {
                    "origin": origin,
                    "target": target,
                    "reason": f"a step between {dates[lo].date()} and "
                    f"{target.date()} spans more than {cfg.max_gap_days} calendar days (not a daily window)",
                }
            )
            continue
        win = Y[lo : i + 1]
        y0, actual = Y[i], Y[i + h]
        b_hat = np.array([_ar1_forecast(betas[lo : i + 1, j], h) for j in range(3)])
        ns_change = X @ (b_hat - betas[i])
        mu = win.mean(axis=0)
        _, _, vt = np.linalg.svd(win - mu, full_matrices=False)
        L = vt[:k].T
        scores = (win - mu) @ L
        s_hat = np.array([_ar1_forecast(scores[:, j], h) for j in range(k)])
        preds = {
            "random_walk": y0,
            "ns_ar1": y0 + ns_change,
            "ns_curve": X @ b_hat,
            "pca_ar1": y0 + L @ (s_hat - scores[-1]),
        }
        vol_seg = dlevel[max(lo, i - int(cfg.vol_window) + 1) : i + 1]
        vol_seg = vol_seg[np.isfinite(vol_seg)]
        regime_rows.append(
            {"origin": origin, "realised_vol_bp": float(np.std(vol_seg, ddof=1)) if vol_seg.size >= 5 else np.nan}
        )
        for j, tenor in enumerate(tenors):
            row = {
                "origin": origin,
                "target": target,
                "tenor": tenor,
                "years": years[j],
                "actual": actual[j],
                "origin_level": y0[j],
                "move_bp": (actual[j] - y0[j]) * BP,
            }
            for name, p in preds.items():
                row[f"err_bp_{name}"] = (p[j] - actual[j]) * BP
            rows.append(row)
    errors = pd.DataFrame(rows)
    regimes = pd.DataFrame(regime_rows, columns=["origin", "realised_vol_bp"])
    if not regimes.empty:
        regimes = pd.concat([regimes, regime_buckets(regimes["realised_vol_bp"])], axis=1)
    bucket_of = dict(zip(regimes.get("origin", []), regimes.get("bucket", [])))
    skipped_df = pd.DataFrame(skipped, columns=["origin", "target", "reason"])
    if errors.empty:
        return CurveForecastResult(
            cfg,
            tuple(tenors),
            errors,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            regimes,
            pd.DataFrame(),
            skipped_df,
        )
    errors["regime"] = errors["origin"].map(lambda d: bucket_of.get(d, "n/a"))
    daily = (
        errors.groupby(["origin", "target"], sort=True)
        .agg(regime=("regime", "first"), **{f"rmse_bp_{p}": (f"err_bp_{p}", _rmse) for p in PREDICTORS})
        .reset_index()
    )
    summary = _summary(errors, h)
    by_tenor = pd.DataFrame(
        [
            {"tenor": t, **{f"rmse_bp_{p}": _rmse(g[f"err_bp_{p}"]) for p in PREDICTORS}}
            for t, g in errors.groupby("tenor", sort=False)
        ]
    )
    by_regime = []
    for bucket in ("low", "mid", "high", "n/a"):
        g = errors[errors["regime"] == bucket]
        if g.empty:
            continue
        n_days = int(g["origin"].nunique())
        row = {
            "regime": bucket,
            "origin_days": n_days,
            "sufficient": bool(bucket != "n/a" and n_days >= cfg.min_regime_days),
        }
        for p in PREDICTORS:
            row[f"rmse_bp_{p}"] = _rmse(g[f"err_bp_{p}"])
        by_regime.append(row)
    return CurveForecastResult(
        cfg, tuple(tenors), errors, daily, summary, by_tenor, regimes, pd.DataFrame(by_regime), skipped_df
    )


def _summary(errors: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    sq = errors.assign(**{f"sq_{p}": errors[f"err_bp_{p}"] ** 2 for p in PREDICTORS})
    daily = sq.groupby("target", sort=True)[[f"sq_{p}" for p in PREDICTORS]].mean()
    bench = _rmse(errors[f"err_bp_{BENCHMARK}"])
    for name, description in PREDICTORS.items():
        rmse = _rmse(errors[f"err_bp_{name}"])
        row: Dict[str, object] = {
            "predictor": name,
            "rmse_bp": rmse,
            "mae_bp": float(np.nanmean(np.abs(errors[f"err_bp_{name}"]))),
            "n_points": int(errors[f"err_bp_{name}"].notna().sum()),
            "n_days": int(daily.shape[0]),
        }
        if name == BENCHMARK:
            row.update(
                benchmark="",
                benchmark_rmse_bp=np.nan,
                skill=np.nan,
                dm_t=np.nan,
                day_win_rate=np.nan,
                day_win_lo=np.nan,
                day_win_hi=np.nan,
            )
        else:
            diff = daily[f"sq_{name}"] - daily[f"sq_{BENCHMARK}"]
            wins, n_days = int((diff < 0).sum()), int(diff.size)
            lo, hi = wilson_interval(wins, n_days)
            row.update(
                benchmark=BENCHMARK,
                benchmark_rmse_bp=bench,
                skill=float(1.0 - rmse / bench) if bench > 0 else np.nan,
                dm_t=diebold_mariano(diff.to_numpy(float), horizon=horizon),
                day_win_rate=wins / n_days if n_days else np.nan,
                day_win_lo=lo,
                day_win_hi=hi,
            )
        row["description"] = description
        rows.append(row)
    return pd.DataFrame(rows)
