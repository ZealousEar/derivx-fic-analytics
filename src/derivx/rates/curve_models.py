"""Cross-sectional yield-curve models fitted to one date's rates by
maturity, and their in-sample and out-of-sample fit.

Models (:data:`CURVE_MODELS`), in maturity ``t`` (years) with decay time
``tau`` (years)::

    NS(t)       = b0 + b1 L1(t / tau) + b2 L2(t / tau)
    L1(x)       = (1 - e^-x) / x,      L2(x) = L1(x) - e^-x
    Svensson(t) = NS(t) + b3 L2(t / tau2)

* ``ns``       - Nelson-Siegel (1987) with ``tau`` fitted (a log grid over
  0.1-10 years refined by a bounded scalar search), betas by least squares;
* ``ns_dl``    - Nelson-Siegel with the Diebold-Li (2006) decay fixed at
  lambda = 0.0609 per month (``tau = 1 / (12 x 0.0609) = 1.368`` years), the
  factor model the curve forecasts use (:mod:`derivx.validation.curve_forecast`);
* ``svensson`` - Svensson (1994), the two decays on a log grid over 0.1-10
  years refined by a bounded quasi-Newton search (reported with tau < tau2);
* ``spline``   - natural cubic spline in maturity through the quotes (it
  interpolates: its in-sample error is zero by construction, so only the
  out-of-sample score compares).

The models are fitted to the rates as published (par yields for the
Treasury curve, a common smoothing approximation; Diebold and Li fit zero
yields). Scores are in bp: ``rmse_bp`` in sample and ``rmse_cv_bp`` by
leave-one-out over the interior tenors (the shortest and longest tenors are
never held out, so nothing is extrapolated), the comparable number across
models with different numbers of parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, minimize_scalar

from derivx.rates.conventions import parse_tenor

BP = 1e4
DL_LAMBDA_PER_MONTH = 0.0609
DL_TAU = 1.0 / (12.0 * DL_LAMBDA_PER_MONTH)  # 1.368 years
TAU_BOUNDS = (0.1, 10.0)
CURVE_MODELS: Dict[str, str] = {
    "ns": "Nelson-Siegel, decay fitted (4 params)",
    "ns_dl": "Nelson-Siegel, Diebold-Li decay fixed at 1.368y (3 params)",
    "svensson": "Svensson, two decays fitted (6 params)",
    "spline": "natural cubic spline in maturity: interpolates, scored out of sample only",
}
N_PARAMS = {"ns": 4, "ns_dl": 3, "svensson": 6, "spline": 0}


def _l1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-8
    safe = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x / 2.0, (1.0 - np.exp(-safe)) / safe)


def ns_loadings(t: Sequence[float], tau: float) -> np.ndarray:
    """(n, 3) Nelson-Siegel loadings (level, slope, curvature) at
    maturities ``t`` (years) for decay time ``tau`` (years)."""

    if not (np.isfinite(tau) and tau > 0):
        raise ValueError(f"the Nelson-Siegel decay time must be > 0 years, got {tau}")
    x = np.asarray(t, dtype=float) / tau
    l1 = _l1(x)
    return np.column_stack([np.ones_like(x), l1, l1 - np.exp(-x)])


def svensson_loadings(t: Sequence[float], tau: float, tau2: float) -> np.ndarray:
    """(n, 4) Svensson loadings: the NS loadings plus a second curvature."""

    if not (np.isfinite(tau2) and tau2 > 0):
        raise ValueError(f"the second Svensson decay time must be > 0 years, got {tau2}")
    x2 = np.asarray(t, dtype=float) / tau2
    return np.column_stack([ns_loadings(t, tau), _l1(x2) - np.exp(-x2)])


@dataclass
class CurveFit:
    """One fitted curve: ``params`` (betas in the rates' units, decays in
    years), the maturities and rates it was fitted to, and ``predict``."""

    model: str
    params: Dict[str, float]
    t: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    _predict: Callable[[np.ndarray], np.ndarray] = field(repr=False, default=None)

    def predict(self, t) -> np.ndarray:
        return np.asarray(self._predict(np.asarray(t, dtype=float)), dtype=float)

    @property
    def rmse_bp(self) -> float:
        e = self.predict(self.t) - self.y
        return float(np.sqrt(np.mean(e * e)) * BP)


def _check(t: Sequence[float], y: Sequence[float], n_min: int) -> Tuple[np.ndarray, np.ndarray]:
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("maturities and rates must be equal-length 1-D sequences")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))) or np.any(t <= 0):
        raise ValueError("maturities must be > 0 years and rates finite")
    if np.any(np.diff(np.sort(t)) <= 0):
        raise ValueError("each maturity once")
    if t.size < n_min:
        raise ValueError(f"{t.size} maturities for a model that needs at least {n_min}")
    order = np.argsort(t)
    return t[order], y[order]


def _ols(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = X @ beta - y
    return beta, float(r @ r)


def _best_tau(t: np.ndarray, y: np.ndarray, loadings: Callable[[float], np.ndarray]) -> float:
    grid = np.exp(np.linspace(np.log(TAU_BOUNDS[0]), np.log(TAU_BOUNDS[1]), 24))
    sse = [_ols(loadings(g), y)[1] for g in grid]
    i = int(np.argmin(sse))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    if hi <= lo:
        return float(grid[i])
    res = minimize_scalar(lambda g: _ols(loadings(g), y)[1], bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    return float(res.x if res.fun <= sse[i] else grid[i])


def fit_ns(t: Sequence[float], y: Sequence[float], tau: Optional[float] = None) -> CurveFit:
    """Nelson-Siegel fit; ``tau`` fixed when given (``DL_TAU`` for
    Diebold-Li), else fitted. Needs 4 maturities (5 with ``tau`` fitted)."""

    t, y = _check(t, y, 4 if tau is not None else 5)
    tau = float(tau) if tau is not None else _best_tau(t, y, lambda g: ns_loadings(t, g))
    beta, _ = _ols(ns_loadings(t, tau), y)
    params = {"b0": float(beta[0]), "b1": float(beta[1]), "b2": float(beta[2]), "tau": tau}
    return CurveFit("ns", params, t, y, lambda s: ns_loadings(s, tau) @ beta)


def fit_svensson(t: Sequence[float], y: Sequence[float]) -> CurveFit:
    """Svensson fit: betas by least squares for any pair of decays, the two
    decays by a log grid over 0.1-10 years then a bounded quasi-Newton
    search in log space from the best grid pair (the two curvature terms
    are the same function, so the pair is reported with ``tau < tau2``).
    Needs 7 maturities."""

    t, y = _check(t, y, 7)
    lo, hi = np.log(TAU_BOUNDS[0]), np.log(TAU_BOUNDS[1])
    grid = np.linspace(lo, hi, 10)

    def sse(v: np.ndarray) -> float:  # in bp^2, so the optimizer's tolerances bite
        return _ols(svensson_loadings(t, float(np.exp(v[0])), float(np.exp(v[1]))), y)[1] * BP * BP

    pairs = [(a, b) for i, a in enumerate(grid) for b in grid[i + 1 :]]
    x0 = np.array(min(pairs, key=lambda v: sse(np.array(v))))
    res = minimize(
        sse, x0, method="L-BFGS-B", bounds=[(lo, hi), (lo, hi)], options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500}
    )
    v = res.x if res.fun <= sse(x0) else x0
    tau, tau2 = sorted(float(np.exp(x)) for x in v)
    beta, _ = _ols(svensson_loadings(t, tau, tau2), y)
    params = {
        "b0": float(beta[0]),
        "b1": float(beta[1]),
        "b2": float(beta[2]),
        "b3": float(beta[3]),
        "tau": tau,
        "tau2": tau2,
    }
    return CurveFit("svensson", params, t, y, lambda s: svensson_loadings(s, tau, tau2) @ beta)


def fit_spline(t: Sequence[float], y: Sequence[float]) -> CurveFit:
    t, y = _check(t, y, 3)
    spline = CubicSpline(t, y, bc_type="natural")
    return CurveFit("spline", {}, t, y, spline)


def fit_curve_model(model: str, t: Sequence[float], y: Sequence[float]) -> CurveFit:
    """Fit one of :data:`CURVE_MODELS`."""

    if model == "ns":
        return fit_ns(t, y)
    if model == "ns_dl":
        fit = fit_ns(t, y, tau=DL_TAU)
        fit.model = "ns_dl"
        return fit
    if model == "svensson":
        return fit_svensson(t, y)
    if model == "spline":
        return fit_spline(t, y)
    raise ValueError(f"unknown curve model {model!r}; choose from {sorted(CURVE_MODELS)}")


def loo_errors(model: str, t: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Leave-one-out errors (model minus quote, decimals) at the interior
    maturities: each one predicted from a fit to the others. NaN where the
    remaining maturities are too few for the model."""

    t, y = _check(t, y, 3)
    out = np.full(t.size - 2, np.nan)
    for j, i in enumerate(range(1, t.size - 1)):
        keep = np.ones(t.size, dtype=bool)
        keep[i] = False
        try:
            out[j] = float(fit_curve_model(model, t[keep], y[keep]).predict([t[i]])[0]) - y[i]
        except ValueError:
            continue
    return out


def compare_curve_models(
    levels: pd.DataFrame, models: Sequence[str] = tuple(CURVE_MODELS), tenors: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit ``models`` to the curve on every date of ``levels`` (index:
    dates; columns: tenors; decimals). Returns ``(scores, skipped)``: one
    row per (date, model) with ``rmse_bp``, ``rmse_cv_bp``, ``max_abs_bp``,
    ``n_params``, ``n_tenors`` and the parameters; and dates or models that
    could not be fitted, with the reason."""

    bad = [m for m in models if m not in CURVE_MODELS]
    if bad or not models:
        raise ValueError(f"models must be a non-empty subset of {sorted(CURVE_MODELS)}, got {list(models)}")
    cols = list(tenors) if tenors else list(levels.columns)
    years = np.array([parse_tenor(str(c)).years for c in cols])
    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    for day, row in levels[cols].iterrows():
        y = row.to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 4:
            skipped.append({"date": day, "model": "", "reason": f"{int(ok.sum())} tenors quoted; need at least 4"})
            continue
        for model in models:
            try:
                fit = fit_curve_model(model, years[ok], y[ok])
                cv = loo_errors(model, years[ok], y[ok])
            except ValueError as exc:
                skipped.append({"date": day, "model": model, "reason": str(exc)})
                continue
            e = fit.predict(fit.t) - fit.y
            cv = cv[np.isfinite(cv)]
            rows.append(
                {
                    "trade_date": day,
                    "expiry": "curve",
                    "model": model,
                    "rmse_bp": float(np.sqrt(np.mean(e * e)) * BP),
                    "rmse_cv_bp": float(np.sqrt(np.mean(cv * cv)) * BP) if cv.size else float("nan"),
                    "max_abs_bp": float(np.max(np.abs(e)) * BP),
                    "n_params": N_PARAMS[model],
                    "n_tenors": int(ok.sum()),
                    **{f"param_{k}": v for k, v in fit.params.items()},
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(skipped, columns=["date", "model", "reason"])
