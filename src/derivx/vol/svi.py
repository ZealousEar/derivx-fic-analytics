"""Raw SVI smiles in total implied variance: evaluation, static-arbitrage
checks and calibration.

Conventions
-----------
* ``k = ln(K / F)`` is forward log-moneyness (F the forward to the expiry).
* ``w(k) = sigma_BS(k)^2 T`` is total implied variance, T in years; the
  implied vol is ``sqrt(w / T)``. Raw SVI (Gatheral 2004):

      w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))

  with ``b >= 0``, ``|rho| < 1``, ``sigma > 0`` and ``a + b sigma
  sqrt(1 - rho^2) >= 0`` (the minimum of w; ``a`` itself may be negative).
  Parameters are total-variance parameters of one expiry: they scale with T
  and are comparable with other raw-SVI systems (e.g. QuantLib's
  ``SviSmileSection``, which takes ``[a, b, sigma, rho, m]``).
* Butterfly arbitrage: Gatheral's density factor
  ``g(k) = (1 - k w'/(2w))^2 - w'^2/4 (1/w + 1/4) + w''/2`` must be >= 0
  (the risk-neutral density is ``n(d2) g(k) / sqrt(w)`` per unit k).
* Roger Lee's moment bounds: the wing slopes of w, ``b (1 - rho)`` (left)
  and ``b (1 + rho)`` (right), must be <= 2.
* Calendar arbitrage: w(k, T) must be non-decreasing in T at every k
  (slices compared at the same forward log-moneyness).
* Enforcing no arbitrage (``fit_svi_strike_slice(enforce_no_arbitrage=
  True)``): when the free fit fails the butterfly or Lee check, it is
  replaced by a constrained fit (SLSQP with ``g(k) >= 1e-4`` on the check
  grid and both Lee slopes <= 2, analytic gradients), started from the two
  best quasi-explicit start points that already satisfy those constraints.
  Only if neither passes the final checks does it start from an SSVI slice
  ``w = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))`` with
  ``theta phi (1 + |rho|) < 4`` and ``theta phi^2 (1 + |rho|) <= 4``, which
  is free of butterfly arbitrage by construction (Gatheral-Jacquier 2014,
  Theorem 4.2), and from the free fit. The best start that passes the final
  checks is kept; if the constrained fit fails, the SSVI slice itself is
  returned (``arbitrage_repair`` says which).
* Start points: quasi-explicit fits (for fixed (m, sigma), w is linear in
  (a, b rho, b)) on a grid around the quotes and on a grid over the whole
  parameter box, ranked by the fit objective itself. The box grid lets the
  solver start near the far-out optimum of short-dated V-shaped smiles
  (sigma ~ 1, m at a bound) instead of crawling there along a flat valley.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares, minimize

SVI_PARAMS = ("a", "b", "rho", "m", "sigma")
# SciPy's notice when an SLSQP step lands a rounding error past a bound: SciPy
# clips the point back into the bounds before evaluating it (SciPy 1.11 does
# so on the walk-forward's synthetic history), and the result is clipped again
# below, so it is the one warning the constrained refit silences.
_SLSQP_CLIP_NOTICE = "Values in x were outside bounds during a minimize step"
LEE_MAX_SLOPE = 2.0
_RHO_MAX = 0.999
_G_MARGIN = 1e-4  # constrained fits keep g >= 1e-4 on their constraint grid
_LEE_MARGIN = 1e-6
# Constraint-satisfying quasi-explicit starts the constrained refit tries (best objective first): half the free
# starts, between 1 and REFIT_STARTS (2 for the default 4 free starts, 1 for the 2 of a cross-validation refit).
REFIT_STARTS = 2
# With enforcement, the remaining free starts are skipped once the first free start's objective is this far
# (relative) below the constrained fit's: the free optimum then lies outside the arbitrage-free set.
CERTIFICATE_GAP = 0.01
LOSSES = ("linear", "soft_l1", "huber", "cauchy", "arctan")


def _get(params: Mapping, name: str) -> float:
    if name == "sigma" and "sigma" not in params and "s" in params:
        return float(params["s"])  # legacy key
    try:
        return float(params[name])
    except KeyError as exc:
        raise ValueError(f"SVI params need keys {SVI_PARAMS}; missing {name!r}") from exc


def svi_tuple(params: Mapping) -> Tuple[float, float, float, float, float]:
    """``(a, b, rho, m, sigma)`` from a mapping (``'s'`` accepted for
    sigma), validated: raises ``ValueError`` on b < 0, |rho| >= 1,
    sigma <= 0 or a negative minimum total variance."""

    a, b, rho, m, sigma = (_get(params, n) for n in SVI_PARAMS)
    if not all(np.isfinite([a, b, rho, m, sigma])):
        raise ValueError(f"SVI params must be finite, got {dict(zip(SVI_PARAMS, (a, b, rho, m, sigma)))}")
    if b < 0:
        raise ValueError(f"SVI b must be >= 0, got {b}")
    if abs(rho) >= 1:
        raise ValueError(f"SVI rho must satisfy |rho| < 1, got {rho}")
    if sigma <= 0:
        raise ValueError(f"SVI sigma must be > 0, got {sigma}")
    w_min = a + b * sigma * np.sqrt(1.0 - rho * rho)
    if w_min < -1e-14:
        raise ValueError(f"SVI params give a negative minimum total variance a + b sigma sqrt(1 - rho^2) = {w_min:.3g}")
    return a, b, rho, m, sigma


def _raw_w(k, a, b, rho, m, sigma):
    u = np.asarray(k, dtype=float) - m
    return a + b * (rho * u + np.sqrt(u * u + sigma * sigma))


def svi_total_variance(k, params: Mapping):
    """Total implied variance w(k) at forward log-moneyness ``k``."""

    w = _raw_w(k, *svi_tuple(params))
    return float(w) if np.ndim(w) == 0 else w


def svi_derivatives(k, params: Mapping) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(w, dw/dk, d2w/dk2)`` at ``k``."""

    a, b, rho, m, sigma = svi_tuple(params)
    u = np.asarray(k, dtype=float) - m
    R = np.sqrt(u * u + sigma * sigma)
    return a + b * (rho * u + R), b * (rho + u / R), b * sigma * sigma / R**3


def _resolve_T(params: Mapping, T: Optional[float]) -> float:
    if T is None:
        T = getattr(params, "T", None)
        if T is None and isinstance(params, Mapping) and "T" in params:
            T = params["T"]
    if T is None:
        raise ValueError(
            "T (years to expiry) is required: SVI params are total-variance params, so the vol is sqrt(w/T)"
        )
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"T must be > 0 years, got {T}")
    return T


def svi_iv(K, params: Mapping, F: float, T: Optional[float] = None):
    """Black implied vol ``sqrt(w(ln(K/F)) / T)`` at strike(s) ``K``.

    ``F`` (forward) is required; ``T`` defaults to ``params.T`` when the
    params carry it (an :class:`SviFit`). Raises ``ValueError`` on K <= 0,
    F <= 0, T <= 0 or invalid params (never clamps a negative variance)."""

    if F is None:
        raise ValueError("F (the forward) is required: SVI is a function of ln(K/F)")
    F = float(F)
    K = np.asarray(K, dtype=float)
    if not np.isfinite(F) or F <= 0 or np.any(~np.isfinite(K)) or np.any(K <= 0):
        raise ValueError("svi_iv needs F > 0 and finite strikes K > 0")
    T = _resolve_T(params, T)
    w = _raw_w(np.log(K / F), *svi_tuple(params))
    vol = np.sqrt(np.maximum(w, 0.0) / T)
    return float(vol) if np.ndim(vol) == 0 else vol


# ---------------------------------------------------------------------------
# Static-arbitrage checks
# ---------------------------------------------------------------------------


def butterfly_density(k, params: Mapping) -> np.ndarray:
    """Gatheral's g(k); the slice is free of butterfly arbitrage iff g >= 0."""

    w, w1, w2 = svi_derivatives(k, params)
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2
    return g


def default_k_grid(params: Mapping, k_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Dense check grid: +-3 in k around m, a fine grid of +-10 sigma
    around m (where the curvature sits), and ``k_range`` if given."""

    _, _, _, m, sigma = svi_tuple(params)
    parts = [np.linspace(m - 3.0, m + 3.0, 1201), m + sigma * np.linspace(-10.0, 10.0, 401)]
    if k_range is not None:
        parts.append(np.linspace(k_range[0], k_range[1], 401))
    return np.unique(np.concatenate(parts))


def _negative_ranges(k: np.ndarray, bad: np.ndarray) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    start = None
    for i, flag in enumerate(bad):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(bad) - 1):
            end = i if flag else i - 1
            out.append((float(k[start]), float(k[end])))
            start = None
    return out


@dataclass(frozen=True)
class ButterflyCheck:
    """Result of :func:`check_butterfly` (g is dimensionless)."""

    ok: bool
    min_g: float
    k_at_min: float
    negative_ranges: List[Tuple[float, float]]
    k_min: float
    k_max: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "butterfly_ok": self.ok,
            "butterfly_min_g": self.min_g,
            "butterfly_k_at_min": self.k_at_min,
            "butterfly_negative_k": [list(r) for r in self.negative_ranges],
        }


def check_butterfly(params: Mapping, k_grid: Optional[Sequence[float]] = None, tol: float = 1e-10) -> ButterflyCheck:
    """Butterfly (density) check of one slice on ``k_grid`` (default
    :func:`default_k_grid`): fails where g(k) < -tol and says where."""

    k = np.sort(np.asarray(default_k_grid(params) if k_grid is None else k_grid, dtype=float))
    g = butterfly_density(k, params)
    i = int(np.nanargmin(g))
    bad = g < -tol
    return ButterflyCheck(
        ok=not bool(bad.any()),
        min_g=float(g[i]),
        k_at_min=float(k[i]),
        negative_ranges=_negative_ranges(k, bad),
        k_min=float(k[0]),
        k_max=float(k[-1]),
    )


def lee_wing_check(params: Mapping) -> Dict[str, object]:
    """Roger Lee's moment bounds on the asymptotic slopes of w:
    ``b (1 - rho) <= 2`` (left) and ``b (1 + rho) <= 2`` (right)."""

    _, b, rho, _, _ = svi_tuple(params)
    left, right = b * (1.0 - rho), b * (1.0 + rho)
    return {
        "lee_left_slope": float(left),
        "lee_right_slope": float(right),
        "lee_ok": bool(left <= LEE_MAX_SLOPE + 1e-12 and right <= LEE_MAX_SLOPE + 1e-12),
    }


@dataclass(frozen=True)
class CalendarCheck:
    """Calendar check between two expiries T1 < T2 (total variance units)."""

    T1: float
    T2: float
    ok: bool
    min_dw: float
    k_at_min: float
    negative_ranges: List[Tuple[float, float]]

    def as_dict(self) -> Dict[str, object]:
        return {
            "calendar_ok": self.ok,
            "calendar_min_dw": self.min_dw,
            "calendar_k_at_min": self.k_at_min,
            "calendar_negative_k": [list(r) for r in self.negative_ranges],
        }


def check_calendar(
    slices: Sequence[Tuple[float, Mapping]],
    k_grid: Optional[Sequence[float]] = None,
    tol: float = 1e-12,
) -> List[CalendarCheck]:
    """Calendar check across expiries: for each adjacent pair (sorted by
    T) total variance must not decrease at any ``k`` of ``k_grid``
    (default ``linspace(-1.5, 1.5, 601)``). ``slices`` are ``(T, params)``."""

    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 601)
    k = np.sort(np.asarray(k_grid, dtype=float))
    ordered = sorted(((float(T), p) for T, p in slices), key=lambda s: s[0])
    out: List[CalendarCheck] = []
    for (T1, p1), (T2, p2) in zip(ordered, ordered[1:]):
        dw = _raw_w(k, *svi_tuple(p2)) - _raw_w(k, *svi_tuple(p1))
        i = int(np.argmin(dw))
        bad = dw < -tol
        out.append(CalendarCheck(T1, T2, not bool(bad.any()), float(dw[i]), float(k[i]), _negative_ranges(k, bad)))
    return out


def basic_no_arb_checks(params_over_maturities: Iterable[Mapping]) -> Dict[str, float]:
    """Static-arbitrage counts over a set of total-variance SVI slices.

    Returns counts of slices that violate each condition: ``viol_b`` (b <
    0), ``viol_rho`` (|rho| >= 1), ``viol_min_variance`` (a + b sigma
    sqrt(1-rho^2) < 0), ``viol_butterfly`` (g(k) < 0 somewhere),
    ``viol_lee`` (a wing slope > 2), and ``viol_calendar`` (adjacent pairs,
    by ``'T'``, where w decreases; only slices that carry ``T`` count), plus
    ``min_g`` over all valid slices."""

    counts = dict.fromkeys(
        ("viol_b", "viol_rho", "viol_min_variance", "viol_butterfly", "viol_lee", "viol_calendar"), 0.0
    )
    min_g = np.inf
    timed: List[Tuple[float, Mapping]] = []
    for p in params_over_maturities:
        a, b, rho, m, sigma = (_get(p, n) for n in SVI_PARAMS)
        if b < 0:
            counts["viol_b"] += 1
        if abs(rho) >= 1:
            counts["viol_rho"] += 1
        if b < 0 or abs(rho) >= 1 or sigma <= 0:
            continue
        if a + b * sigma * np.sqrt(1.0 - rho * rho) < 0:
            counts["viol_min_variance"] += 1
            continue
        bf = check_butterfly(p)
        min_g = min(min_g, bf.min_g)
        counts["viol_butterfly"] += 0 if bf.ok else 1
        counts["viol_lee"] += 0 if lee_wing_check(p)["lee_ok"] else 1
        T = getattr(p, "T", None) if not (isinstance(p, Mapping) and "T" in p) else p["T"]
        if T is not None:
            timed.append((float(T), p))
    counts["viol_calendar"] = float(sum(0 if c.ok else 1 for c in check_calendar(timed)))
    counts["min_g"] = float(min_g) if np.isfinite(min_g) else float("nan")
    return counts


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SviFit(Mapping):
    """A calibrated raw-SVI slice (total-variance parameters).

    Read-only mapping over ``a, b, rho, m, sigma``; also carries ``T`` and
    ``F`` so ``svi_iv(K, fit, F)`` works. Errors are in implied-vol units
    (decimal vol): ``rmse_bp = rmse * 1e4`` vol bp. Residuals are model
    minus market."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float
    F: float
    rmse: float
    max_abs_error: float
    n_points: int
    n_dropped: int
    at_bound: Dict[str, bool]
    success: bool
    message: str
    nfev: int
    n_starts: int
    jacobian_condition: float
    param_se: Dict[str, float]
    butterfly: ButterflyCheck
    lee: Dict[str, object]
    enforce_no_arbitrage: bool
    loss: str
    f_scale: Optional[float]
    n_downweighted: int
    strikes: np.ndarray = field(repr=False)
    market_vols: np.ndarray = field(repr=False)
    model_vols: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    penalised: bool = False  # True when enforce_no_arbitrage replaced an arbitrageable free fit
    arbitrage_repair: str = "none"  # 'none', 'constrained', 'ssvi_slice' or 'failed' (see module docs)

    def __getitem__(self, key: str) -> float:
        if key not in SVI_PARAMS:
            raise KeyError(key)
        return float(getattr(self, key))

    def __iter__(self) -> Iterator[str]:
        return iter(SVI_PARAMS)

    def __len__(self) -> int:
        return len(SVI_PARAMS)

    @property
    def params(self) -> Dict[str, float]:
        return {n: float(getattr(self, n)) for n in SVI_PARAMS}

    @property
    def min_variance(self) -> float:
        return float(self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2))

    @property
    def residuals(self) -> np.ndarray:
        return self.model_vols - self.market_vols

    @property
    def rmse_bp(self) -> float:
        return self.rmse * 1e4

    @property
    def rmse_vol_points(self) -> float:
        """RMSE in vol points (percent of vol): 0.25 means 25 vol bp."""

        return self.rmse * 100.0

    @property
    def max_abs_error_bp(self) -> float:
        return self.max_abs_error * 1e4

    @property
    def arbitrage_free(self) -> bool:
        return bool(self.butterfly.ok and self.lee["lee_ok"])

    def total_variance(self, k):
        return svi_total_variance(k, self)

    def vol(self, K):
        return svi_iv(K, self, self.F, self.T)

    def summary(self) -> Dict[str, object]:
        return {
            **self.params,
            "T": self.T,
            "rmse_bp": self.rmse_bp,
            "max_abs_error_bp": self.max_abs_error_bp,
            "n_points": self.n_points,
            "butterfly_min_g": self.butterfly.min_g,
            "butterfly_ok": self.butterfly.ok,
            **self.lee,
            "arbitrage_repair": self.arbitrage_repair,
            "at_bound": ", ".join(k for k, v in self.at_bound.items() if v),
            "success": self.success,
        }


_X_NAMES = ("min_variance", "b", "rho", "m", "sigma")


def _x_to_raw(x: np.ndarray) -> Tuple[float, float, float, float, float]:
    wmin, b, rho, m, sigma = x
    return wmin - b * sigma * np.sqrt(1.0 - rho * rho), b, rho, m, sigma


def _raw_to_x(a, b, rho, m, sigma) -> np.ndarray:
    return np.array([a + b * sigma * np.sqrt(1.0 - rho * rho), b, rho, m, sigma])


def _linear_fits(k, w_obs, omega, ms, sigmas, lb, ub) -> np.ndarray:
    """Quasi-explicit (Zeliade) start points, one row of x = (min_variance,
    b, rho, m, sigma) per (m, sigma) pair: for fixed (m, sigma), w is linear
    in (a, b rho, b), fitted by weighted least squares in total variance
    (weights ``omega``, the first-order map from variance to vol errors).
    All pairs are solved at once (normal equations with a 1e-12 relative
    ridge for the near-collinear columns of tiny sigma); rows are clipped
    into the box."""

    u = k[None, :] - ms[:, None]
    X = np.stack([np.ones_like(u), u, np.sqrt(u * u + sigmas[:, None] ** 2)], axis=2) * omega[None, :, None]
    XtX = np.einsum("snj,snk->sjk", X, X)
    Xty = np.einsum("snj,n->sj", X, w_obs * omega)
    ridge = 1e-12 * np.trace(XtX, axis1=1, axis2=2)[:, None, None] * np.eye(3)[None]
    a, c_rho, b = np.linalg.solve(XtX + ridge, Xty[..., None])[..., 0].T
    b = np.maximum(b, 1e-6)
    rho = np.clip(c_rho / b, -0.95, 0.95)
    x = np.column_stack([np.maximum(a + b * sigmas * np.sqrt(1.0 - rho * rho), 0.0), b, rho, ms, sigmas])
    return np.clip(x, lb + 1e-12, ub - 1e-12)


def _objective_rows(X, k, vols, T, wts, loss, f_scale) -> np.ndarray:
    """The fit objective ``0.5 f^2 sum rho((r / f)^2)`` (``0.5 sum r^2`` for
    loss='linear'; r the weighted vol residuals, model minus market) of each
    row of X (x parametrisation), vectorised over the rows."""

    wmin, b, rho, m, sigma = (X[:, i, None] for i in range(5))
    u = k[None, :] - m
    w = wmin - b * sigma * np.sqrt(1.0 - rho * rho) + b * (rho * u + np.sqrt(u * u + sigma * sigma))
    r = wts * (np.sqrt(np.maximum(w, 0.0) / T) - vols)
    f2 = 1.0 if loss == "linear" else float(f_scale) ** 2
    cost = 0.5 * f2 * np.sum(_loss_rho(r * r / f2, loss)[0], axis=1)
    return np.where(np.all(np.isfinite(r), axis=1), cost, np.inf)


def _seed_pool(k, vols, T, wts, w_obs, omega, lb, ub, loss, f_scale) -> Tuple[np.ndarray, np.ndarray]:
    """Candidate starts and their objective, best first: quasi-explicit fits
    on a grid around the quotes (m over the quoted k range and 0, sigma from
    5% to 100% of the quoted span) and on a grid over the whole box (m over
    its bounds, sigma log-spaced from 1e-4 to its upper bound). The box grid
    matters on short-dated V-shaped smiles, whose free optimum lies far out
    (sigma ~ 1, m at a bound: SVI degenerating to a parabola over the
    quotes); from a start near the quotes a local solver crawls there along
    a flat valley in 1,000+ steps."""

    kmin, kmax = float(k.min()), float(k.max())
    span = max(kmax - kmin, 1e-3)
    near_m = np.unique(np.r_[np.linspace(kmin, kmax, 5), 0.0])
    near_s = span * np.array([0.05, 0.15, 0.4, 1.0]) + 1e-3
    box_m = np.linspace(lb[3], ub[3], 13)
    box_s = np.geomspace(max(lb[4], 1e-4), ub[4], 12)
    ms = np.r_[np.repeat(near_m, near_s.size), np.repeat(box_m, box_s.size)]
    sigmas = np.r_[np.tile(near_s, near_m.size), np.tile(box_s, box_m.size)]
    X = _linear_fits(k, w_obs, omega, ms, sigmas, lb, ub)
    cost = _objective_rows(X, k, vols, T, wts, loss, f_scale)
    order = np.argsort(cost, kind="stable")
    return X[order], cost[order]


def _distinct(X: np.ndarray, lb: np.ndarray, ub: np.ndarray, count: int) -> List[np.ndarray]:
    """The first ``count`` rows of X that differ from every earlier pick by
    more than 5% of the box width in some coordinate."""

    out: List[np.ndarray] = []
    for x in X:
        if all(np.max(np.abs(x - y) / (ub - lb)) > 0.05 for y in out):
            out.append(x)
        if len(out) >= count:
            break
    return out


def _robust_residuals(resid, jac, loss: str, f_scale: float):
    """Residuals ``r~ = r sqrt(rho(z) / z)`` (z = (r / f)^2) and their
    Jacobian, for which plain least squares minimises exactly the robust
    objective ``0.5 f^2 sum rho(z)`` that scipy's ``loss`` minimises
    (``r~^2 = f^2 rho(z)``). Implemented for 'linear' (r~ = r) and 'soft_l1'
    (rho(z) / z = 2 / (sqrt(1 + z) + 1), smooth at 0)."""

    if loss == "linear":
        return resid, jac
    if loss != "soft_l1":
        raise ValueError(f"robust residuals are implemented for 'linear' and 'soft_l1', not {loss!r}")
    f2 = float(f_scale) ** 2

    def scaled(r):
        q = np.sqrt(1.0 + r * r / f2)
        s = np.sqrt(2.0 / (q + 1.0))
        return q, s

    def fun(x):
        r = resid(x)
        return r * scaled(r)[1]

    def grad(x):
        r = resid(x)
        q, s = scaled(r)
        # d(r s)/dr = s + r ds/dr with ds/dr = -s^3 r / (4 q f^2)
        return jac(x) * (s - s**3 * r * r / (4.0 * q * f2))[:, None]

    return fun, grad


def _local_fit(resid, jac, xs, lb, ub, loss: str, f_scale: float):
    """One local free fit from ``xs`` (box ``[lb, ub]``); returns scipy's
    result of the final TRF run, with ``nfev`` counting every evaluation.

    For 'linear' and 'soft_l1' a MINPACK Levenberg-Marquardt run on the
    equivalent plain least-squares residuals (:func:`_robust_residuals`)
    goes first: residuals are evaluated at the point clipped into the box,
    and an exterior penalty ``1e3 (x - clip(x)) / (ub - lb)`` pulls the
    iterate back. Its steps cost a fraction of TRF's in Python overhead.
    scipy's bounded TRF (the reference solver, the same robust loss) then
    polishes from the LM point clipped into the box, so the result is a
    TRF stationary point of the exact bounded problem; it converges in 1-10
    steps from there unless the optimum lies on a bound along a flat valley,
    where TRF takes over the remaining work. On SPY slices this is never
    worse than TRF from ``xs`` (worst +4e-9 relative objective) and 2x
    faster overall. Other losses run TRF alone."""

    nfev = 0
    start = xs
    if loss in ("linear", "soft_l1"):
        fun, grad = _robust_residuals(resid, jac, loss, f_scale)
        width = ub - lb

        def penalised(x):
            xc = np.clip(x, lb, ub)
            return np.r_[fun(xc), 1e3 * (x - xc) / width]

        def penalised_jac(x):
            xc = np.clip(x, lb, ub)
            inside = (x >= lb) & (x <= ub)
            return np.vstack([grad(xc) * inside[None, :], np.diag(np.where(inside, 0.0, 1e3 / width))])

        try:
            with np.errstate(all="ignore"):
                lm = least_squares(penalised, xs, jac=penalised_jac, method="lm", x_scale="jac", max_nfev=1000)
            if np.all(np.isfinite(lm.x)):
                start, nfev = np.clip(lm.x, lb + 1e-12, ub - 1e-12), int(lm.nfev)
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - defensive: TRF from xs instead
            start = xs
    res = least_squares(
        resid, x0=start, jac=jac, bounds=(lb, ub), loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=3000
    )
    res.nfev = int(res.nfev) + nfev
    return res


def fit_svi_strike_slice(
    Ks: Iterable[float],
    ivs: Iterable[float],
    F: float,
    T: float,
    *,
    weights: Union[None, str, Sequence[float]] = None,
    loss: str = "soft_l1",
    f_scale: Optional[float] = None,
    enforce_no_arbitrage: bool = False,
    n_starts: int = 4,
    x0: Optional[Mapping] = None,
    check_grid: Optional[Sequence[float]] = None,
) -> SviFit:
    """Fit raw SVI (total variance) to one expiry's smile in implied-vol
    space.

    Parameters
    ----------
    Ks, ivs : strikes and Black implied vols (per-annum decimals). Non-finite
        or non-positive quotes are dropped (``n_dropped``).
    F, T : forward and years to expiry (both required; ``k = ln(K/F)``).
    weights : None (equal weights in vol, the default), 'vega' (Black vega:
        approximates a price-space fit but lets noisy far wings drift) or an
        array (e.g. 1 / bid-ask spread in vol); scaled to mean 1.
    loss, f_scale : scipy loss on the weighted vol residuals; ``f_scale``
        (default 0.005 = 50 vol bp) is the residual beyond which a robust
        loss down-weights a quote (``n_downweighted`` reports how many).
        ``loss='linear'`` is plain least squares.
    enforce_no_arbitrage : when the free fit has butterfly or Lee arbitrage,
        replace it with a constrained fit (see module docs: g(k) >= 1e-4 on
        a dense grid around m and the quotes, or on ``check_grid`` when
        given, and Lee slopes <= 2), or with an arbitrage-free SSVI slice
        if that fails; ``penalised`` / ``arbitrage_repair`` say what
        happened and the ``butterfly`` / ``lee`` checks report the result.
    n_starts : start points (quasi-explicit linear fits over (m, sigma) on a
        grid near the quotes and one over the whole box, ranked by the fit
        objective, distinct); the search stops early once two starts reach
        the same optimum, or, with ``enforce_no_arbitrage``, once the first
        start is more than ``CERTIFICATE_GAP`` (1%) cheaper than the
        constrained fit (the free optimum is then arbitrageable and is
        replaced anyway). ``n_starts`` of the result counts the runs.
        ``x0`` (params) replaces them with one warm start. Each start runs
        MINPACK Levenberg-Marquardt, then scipy's bounded TRF from its point
        (see :func:`_local_fit`).

    The constraint ``a + b sigma sqrt(1-rho^2) >= 0`` is imposed exactly by
    fitting the minimum variance instead of ``a`` (``a`` may be negative).
    Bounds: ``0 <= b <= 4``, ``|rho| <= 0.999``, ``m`` within one quoted
    k-span of the quotes, ``1e-4 <= sigma <= max(2, 4 span)``; ``at_bound``
    flags parameters that end within 0.1% of a bound (``min_variance`` at
    0 means the variance floor binds). Raises ``ValueError`` with fewer
    than 6 valid quotes (5 parameters + 1).
    """

    Ks = np.asarray(list(Ks) if not isinstance(Ks, np.ndarray) else Ks, dtype=float).ravel()
    ivs = np.asarray(list(ivs) if not isinstance(ivs, np.ndarray) else ivs, dtype=float).ravel()
    if Ks.shape != ivs.shape:
        raise ValueError(f"Ks and ivs must have the same length ({Ks.size} vs {ivs.size})")
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    if F is None or not np.isfinite(F) or F <= 0:
        raise ValueError("F (the forward, > 0) is required: SVI is fitted in k = ln(K/F)")
    if T is None or not np.isfinite(T) or T <= 0:
        raise ValueError("T (years to expiry, > 0) is required: SVI is fitted in total variance iv^2 T")
    F, T = float(F), float(T)
    keep = np.isfinite(Ks) & np.isfinite(ivs) & (Ks > 0) & (ivs > 0)
    w_user = None
    if weights is not None and not isinstance(weights, str):
        w_user = np.asarray(weights, dtype=float).ravel()
        if w_user.shape != Ks.shape:
            raise ValueError("weights must have one entry per quote")
        keep &= np.isfinite(w_user) & (w_user > 0)
    n_dropped = int((~keep).sum())
    K, vols = Ks[keep], ivs[keep]
    if K.size < len(SVI_PARAMS) + 1:
        raise ValueError(f"{K.size} valid quotes for 5 SVI parameters: need at least 6")
    order = np.argsort(K)
    K, vols = K[order], vols[order]
    k = np.log(K / F)
    w_obs = vols * vols * T
    if weights is None:
        wts = np.ones(K.size)
    elif isinstance(weights, str):
        if weights != "vega":
            raise ValueError("weights must be None, 'vega' or an array")
        from derivx.vol.implied import vega_weights

        wts = vega_weights(F, K, T, vols)
    else:
        wts = w_user[keep][order]
    wts = wts / wts.mean()

    kmin, kmax = float(k.min()), float(k.max())
    span = max(kmax - kmin, 1e-3)
    lb = np.array([0.0, 0.0, -_RHO_MAX, kmin - span, 1e-4])
    ub = np.array([1.5 * float(w_obs.max()) + 1e-12, 4.0, _RHO_MAX, kmax + span, max(2.0, 4.0 * span)])
    if f_scale is None:
        f_scale = 0.005
    fixed_grid = None if check_grid is None else np.asarray(check_grid, dtype=float)
    big = 10.0 * float(vols.max())

    def resid(x: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = _x_to_raw(x)
        w = _raw_w(k, a, b, rho, m, sigma)
        out = wts * (np.sqrt(np.maximum(w, 0.0) / T) - vols)
        if not np.all(np.isfinite(out)):
            out = np.full(K.size, big)
        return out

    def jac(x: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of the weighted vol residuals in (min_variance,
        b, rho, m, sigma), with a = min_variance - b sigma sqrt(1 - rho^2)."""

        wmin, b, rho, m, sigma = x
        c = np.sqrt(1.0 - rho * rho)
        u = k - m
        R = np.sqrt(u * u + sigma * sigma)
        w = wmin - b * sigma * c + b * (rho * u + R)
        dvol_dw = wts / (2.0 * np.sqrt(np.maximum(w, 1e-300) * T))
        dw = np.column_stack(
            [
                np.ones_like(k),
                rho * u + R - sigma * c,
                b * u + b * sigma * rho / c,
                -b * (rho + u / R),
                b * (sigma / R - c),
            ]
        )
        return dw * dvol_dw[:, None]

    omega = wts / (2.0 * np.maximum(vols, 1e-4) * T)
    pool = None
    if x0 is not None:
        starts = [np.clip(_raw_to_x(*svi_tuple(x0)), lb + 1e-12, ub - 1e-12)]
    else:
        pool = _seed_pool(k, vols, T, wts, w_obs, omega, lb, ub, loss, f_scale)
        starts = _distinct(pool[0], lb, ub, max(1, int(n_starts)))
    refit_starts = min(REFIT_STARTS, max(1, int(n_starts) // 2))

    def free_ok(x: np.ndarray) -> bool:
        p = dict(zip(SVI_PARAMS, (float(v) for v in _x_to_raw(x))))
        return bool(check_butterfly(p, default_k_grid(p, (kmin, kmax))).ok and lee_wing_check(p)["lee_ok"])

    def refit(x_free: np.ndarray):
        nonlocal pool
        if pool is None:
            pool = _seed_pool(k, vols, T, wts, w_obs, omega, lb, ub, loss, f_scale)
        return _arbitrage_free_refit(
            x_free, k, vols, T, wts, loss, f_scale, lb, ub, (kmin, kmax), resid, jac, fixed_grid, pool[0], refit_starts
        )

    best, runs, repaired, repaired_from = None, 0, None, None
    for xs in starts:
        res = _local_fit(resid, jac, xs, lb, ub, loss, f_scale)
        runs += 1
        agree = best is not None and abs(res.cost - best.cost) <= 1e-6 * max(res.cost, best.cost, 1e-300)
        if best is None or res.cost < best.cost:
            best = res
        if agree:  # two starts reached the same optimum: the remaining (worse-seeded) starts are not run
            break
        if enforce_no_arbitrage and runs == 1 and len(starts) > 1 and not free_ok(best.x):
            # The free optimum costs at most this start's objective. When that is clearly (CERTIFICATE_GAP)
            # below the constrained fit's, the free optimum cannot be arbitrage-free (it would be a feasible
            # point cheaper than the constrained optimum), so the constrained fit is the answer and the other
            # free starts would be wasted (they crawl along flat valleys on short-dated smiles).
            repaired, repaired_from = refit(best.x), best.x
            c_repaired = float(_objective_rows(repaired[0][None, :], k, vols, T, wts, loss, f_scale)[0])
            if repaired[2] and best.cost < (1.0 - CERTIFICATE_GAP) * c_repaired:
                break
    x_fit, success, message, nfev = best.x, bool(best.success), str(best.message), int(best.nfev)
    penalised, repair = False, "none"
    if enforce_no_arbitrage and not free_ok(x_fit):
        if repaired is None or repaired_from is not x_fit:
            repaired = refit(x_fit)
        x_fit, repair, ok, extra_nfev, message = repaired
        penalised, success, nfev = True, ok, nfev + extra_nfev
    a, b, rho, m, sigma = _x_to_raw(x_fit)
    params = {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sigma": float(sigma)}
    model = np.sqrt(np.maximum(_raw_w(k, a, b, rho, m, sigma), 0.0) / T)
    err = model - vols
    n = K.size
    J = jac(np.asarray(x_fit, dtype=float))  # weighted residual Jacobian (vol units) at the solution
    # A parameter is at a bound when it sits within 0.1% of the box width of it; for b's lower bound the
    # tolerance is 0.1% of its natural size (max total variance per unit of quoted k), so a short-dated
    # slice's small but real b (e.g. 0.002 at 1 day) is not flagged against a box of width 4.
    b_scale = float(w_obs.max()) / span
    at_bound = {}
    for i, name in enumerate(_X_NAMES):
        width = ub[i] - lb[i]
        tol_lo = 1e-3 * (min(width, b_scale) if name == "b" else width)
        at_bound[name] = bool(x_fit[i] - lb[i] <= tol_lo or ub[i] - x_fit[i] <= 1e-3 * width)
    try:
        sv = np.linalg.svd(J, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        cond = float("nan")
    s2 = float(np.sum((wts * err) ** 2) / (n - len(SVI_PARAMS)))
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        se = {name: float(np.sqrt(max(cov[i, i], 0.0))) for i, name in enumerate(_X_NAMES)}
    except np.linalg.LinAlgError:
        se = dict.fromkeys(_X_NAMES, float("inf"))
    grid = default_k_grid(params, (kmin, kmax)) if fixed_grid is None else fixed_grid
    return SviFit(
        **params,
        T=T,
        F=F,
        rmse=float(np.sqrt(np.mean(err**2))),
        max_abs_error=float(np.max(np.abs(err))),
        n_points=int(n),
        n_dropped=n_dropped,
        at_bound=at_bound,
        success=bool(success),
        message=message,
        nfev=int(nfev),
        n_starts=runs,
        jacobian_condition=cond,
        param_se=se,
        butterfly=check_butterfly(params, grid),
        lee=lee_wing_check(params),
        enforce_no_arbitrage=bool(enforce_no_arbitrage),
        loss=loss,
        f_scale=None if loss == "linear" else float(f_scale),
        n_downweighted=0 if loss == "linear" else int(np.sum(np.abs(wts * err) > f_scale)),
        strikes=K,
        market_vols=vols,
        model_vols=model,
        weights=wts,
        penalised=penalised,
        arbitrage_repair=repair,
    )


# ---------------------------------------------------------------------------
# Arbitrage-free refit
# ---------------------------------------------------------------------------


def _loss_rho(z: np.ndarray, loss: str) -> Tuple[np.ndarray, np.ndarray]:
    """scipy.optimize.least_squares' ``rho(z)`` and ``rho'(z)`` for ``z =
    (r / f_scale)^2``."""

    if loss == "linear":
        return z, np.ones_like(z)
    if loss == "soft_l1":
        t = np.sqrt(1.0 + z)
        return 2.0 * (t - 1.0), 1.0 / t
    if loss == "huber":
        small = z <= 1.0
        root = np.sqrt(np.maximum(z, 1.0))
        return np.where(small, z, 2.0 * root - 1.0), np.where(small, 1.0, 1.0 / root)
    if loss == "cauchy":
        return np.log1p(z), 1.0 / (1.0 + z)
    if loss == "arctan":
        return np.arctan(z), 1.0 / (1.0 + z * z)
    raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")


def _g_raw(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Gatheral's g(k) without parameter validation (NaN where w <= 0)."""

    u = k - m
    R = np.sqrt(u * u + sigma * sigma)
    w = a + b * (rho * u + R)
    w1 = b * (rho + u / R)
    w2 = b * sigma * sigma / R**3
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2
    return np.where(w > 0, g, np.nan)


def _g_and_grad(k: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gatheral's g(k) and its analytic gradient ``dg/dx`` (shape (k.size,
    5)) in the fit parametrisation x = (min_variance, b, rho, m, sigma),
    with a = min_variance - b sigma sqrt(1 - rho^2). Where w(k) <= 0, g is
    NaN and its gradient row 0 (the constraint treats those points as
    violated). With u = k - m and R = sqrt(u^2 + sigma^2): w = a + b (rho u
    + R), w' = b (rho + u / R), w'' = b sigma^2 / R^3, and
    g = A^2 - w'^2 / 4 (1 / w + 1 / 4) + w'' / 2 with A = 1 - k w' / (2 w)."""

    wmin, b, rho, m, sigma = (float(v) for v in x)
    c = np.sqrt(1.0 - rho * rho)
    a = wmin - b * sigma * c
    u = k - m
    R2 = u * u + sigma * sigma
    R = np.sqrt(R2)
    R3 = R2 * R
    s2 = sigma * sigma
    w = a + b * (rho * u + R)
    w1 = b * (rho + u / R)
    w2 = b * s2 / R3
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_w = 1.0 / w
        A = 1.0 - 0.5 * k * w1 * inv_w
        g = A * A - 0.25 * w1 * w1 * (inv_w + 0.25) + 0.5 * w2
        g_w = (A * k * w1 + 0.25 * w1 * w1) * inv_w * inv_w  # dg/dw
        g_w1 = -A * k * inv_w - 0.5 * w1 * (inv_w + 0.25)  # dg/dw'
    # dg/d(raw) for raw = (a, b, rho, m, sigma); dg/dw'' = 1/2
    d_a = g_w
    d_b = g_w * (rho * u + R) + g_w1 * (rho + u / R) + 0.5 * s2 / R3
    d_rho = g_w * b * u + g_w1 * b
    d_m = -g_w * w1 - g_w1 * w2 + 1.5 * b * s2 * u / (R3 * R2)
    d_sigma = g_w * b * sigma / R - g_w1 * b * u * sigma / R3 + 0.5 * b * sigma * (2.0 * R2 - 3.0 * s2) / (R3 * R2)
    grad = np.column_stack([d_a, d_b - sigma * c * d_a, d_rho + b * sigma * rho / c * d_a, d_m, d_sigma - b * c * d_a])
    ok = w > 0
    return np.where(ok, g, np.nan), np.where(ok[:, None], grad, 0.0)


def _ssvi_slice_x(theta: float, rho: float, phi: float) -> np.ndarray:
    """(min_variance, b, rho, m, sigma) of the SSVI slice (theta, rho, phi)."""

    c = np.sqrt(1.0 - rho * rho)
    a, b, m, sigma = 0.5 * theta * (1.0 - rho * rho), 0.5 * theta * phi, -rho / phi, c / phi
    return _raw_to_x(a, b, rho, m, sigma)


def _fit_ssvi_slice(k, vols, T, wts, loss, f_scale) -> Tuple[np.ndarray, int]:
    """Best SSVI slice in the Gatheral-Jacquier butterfly-free region,
    parameterised by (theta, rho, u) with ``phi = u * min(4 / c, sqrt(4 / c))``,
    ``c = theta (1 + |rho|)`` and ``0 < u < 1``."""

    w_obs = vols * vols * T
    order = np.argsort(k)
    theta0 = float(max(np.interp(0.0, k[order], w_obs[order]), 1e-12))

    def phi_of(theta, rho, u):
        c = theta * (1.0 + abs(rho))
        return u * min(4.0 / c, np.sqrt(4.0 / c))

    def resid(y):
        theta, rho, u = y
        phi = phi_of(theta, rho, u)
        w = 0.5 * theta * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1.0 - rho * rho))
        return wts * (np.sqrt(np.maximum(w, 0.0) / T) - vols)

    lb = np.array([1e-12, -_RHO_MAX, 1e-4])
    ub = np.array([4.0 * float(w_obs.max()) + 1e-12, _RHO_MAX, 1.0 - 1e-6])
    best, nfev = None, 0
    for rho0, u0 in ((-0.6, 0.5), (-0.2, 0.2), (0.2, 0.5)):
        y0 = np.clip(np.array([theta0, rho0, u0]), lb + 1e-15, ub - 1e-15)
        res = least_squares(resid, y0, bounds=(lb, ub), loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=500)
        nfev += int(res.nfev)
        if best is None or res.cost < best.cost:
            best = res
    theta, rho, u = best.x
    return _ssvi_slice_x(theta, rho, phi_of(theta, rho, u)), nfev


def _passes_checks(x: np.ndarray, k_range: Tuple[float, float], grid: Optional[np.ndarray]) -> Tuple[bool, float]:
    p = dict(zip(SVI_PARAMS, (float(v) for v in _x_to_raw(x))))
    try:
        bf = check_butterfly(p, default_k_grid(p, k_range) if grid is None else grid)
        lee = lee_wing_check(p)["lee_ok"]
    except ValueError:
        return False, -np.inf
    return bool(bf.ok and lee), float(bf.min_g)


def _constraint_grid(xs: Sequence[np.ndarray], k_range: Tuple[float, float]) -> np.ndarray:
    """Moderate grid for the SLSQP constraints (the dense check grid is added
    only where a solution dips below zero between these points): 51 points
    over the check range widened by 1.5 on each side, 17 over the check
    range, and per slice in ``xs`` 21 points over m +- 3 and 13 over
    m +- 6 sigma; at most 136 points.

    It is kept well under ~200 constraint rows: from about 210 rows SciPy's
    SLSQP hands its subproblems to multi-threaded BLAS (OpenBLAS), whose
    thread start-up then costs more than the solve on this 5-parameter
    problem (2-10x slower SLSQP steps on a busy 4-core machine, measured
    with SciPy 1.17; the 396-point grid used before made a whole-sample
    calibration 41 s instead of 18 s with default BLAS threads). Fits
    differ from the 396-point grid's by at most 0.0005 vol bp in sample and
    0.0032 vol bp out of sample on the committed SPY slices."""

    lo, hi = k_range
    parts = [np.linspace(lo - 1.5, hi + 1.5, 51), np.linspace(lo, hi, 17)]
    for x in xs:
        _, _, _, m, sigma = _x_to_raw(x)
        parts += [np.linspace(m - 3.0, m + 3.0, 21), m + sigma * np.linspace(-6.0, 6.0, 13)]
    return np.unique(np.concatenate(parts))


def _feasible_seeds(pool: np.ndarray, grid: np.ndarray, lb: np.ndarray, ub: np.ndarray, count: int) -> List[np.ndarray]:
    """The best-objective distinct rows of the seed pool (sorted best first)
    that already satisfy the refit's constraints on ``grid``: g >= 1e-4
    (w > 0) and both Lee slopes <= 2 - 1e-6. g is evaluated for every row at
    once."""

    wmin, b, rho, m, sigma = (pool[:, i, None] for i in range(5))
    a = wmin - b * sigma * np.sqrt(1.0 - rho * rho)
    u = grid[None, :] - m
    R = np.sqrt(u * u + sigma * sigma)
    w = a + b * (rho * u + R)
    w1 = b * (rho + u / R)
    w2 = b * sigma * sigma / R**3
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - grid * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2
    g = np.where(w > 0, g, -np.inf)
    lee = LEE_MAX_SLOPE - _LEE_MARGIN
    bb, rr = pool[:, 1], pool[:, 2]
    feasible = (np.min(g, axis=1) >= _G_MARGIN) & (bb * (1.0 + rr) <= lee) & (bb * (1.0 - rr) <= lee)
    return _distinct(pool[feasible], lb, ub, count)


def _constrained_slsqp(x_start, x_free, cost_x, grid, lo, ub, k_range, fixed_grid) -> Tuple[np.ndarray, bool, int]:
    """SLSQP from ``x_start`` on the robust objective ``cost_x`` (value and
    gradient) with g(k) >= 1e-4 on ``grid`` and both Lee slopes <= 2 - 1e-6,
    analytic constraint gradients (:func:`_g_and_grad`), variables scaled to
    O(1) and the objective normalised by its value at the starts. When the
    solution dips below zero between grid points, the points of the dense
    check grid around the dip are added and the fit resumed (up to three
    rounds; never with a caller's fixed ``grid``). Returns ``(x,
    passes_checks, nfev)``."""

    scale = np.maximum(np.maximum(np.abs(x_free), np.abs(x_start)), np.array([1e-8, 1e-8, 0.1, 1e-3, 1e-4]))
    norm = max(cost_x(x_free)[0], cost_x(x_start)[0], 1e-300)  # SLSQP needs an O(1) objective
    y_lo, y_hi = lo / scale, ub / scale
    lee = LEE_MAX_SLOPE - _LEE_MARGIN

    def cost(y):
        c, grad = cost_x(y * scale)
        return c / norm, grad * scale / norm

    def constraints(kg):
        def fun(yy):
            x = yy * scale
            g = np.nan_to_num(_g_raw(kg, *_x_to_raw(x)), nan=-1.0) - _G_MARGIN
            return np.r_[g, lee - x[1] * (1.0 + x[2]), lee - x[1] * (1.0 - x[2])]

        def grad(yy):
            x = yy * scale
            _, dg = _g_and_grad(kg, x)
            b, rho = x[1], x[2]
            lee_rows = np.array([[0.0, -(1.0 + rho), -b, 0.0, 0.0], [0.0, -(1.0 - rho), b, 0.0, 0.0]])
            return np.vstack([dg, lee_rows]) * scale

        return [{"type": "ineq", "fun": fun, "jac": grad}]

    y = np.clip(x_start, lo, ub) / scale
    ok, nfev = False, 0
    for _ in range(3):  # add the dense check grid where a solution dipped between constraint points
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_SLSQP_CLIP_NOTICE, category=RuntimeWarning)
            res = minimize(
                cost,
                y,
                jac=True,
                method="SLSQP",
                bounds=list(zip(y_lo, y_hi)),
                constraints=constraints(grid),
                options={"maxiter": 500, "ftol": 1e-10},
            )
        nfev += int(res.nfev)
        y = np.clip(res.x, y_lo, y_hi)
        ok, _ = _passes_checks(y * scale, k_range, fixed_grid)
        if ok or fixed_grid is not None:
            break
        grid = np.unique(np.r_[grid, _dip_points(y * scale, k_range)])
    return y * scale, ok, nfev


def _dip_points(x: np.ndarray, k_range: Tuple[float, float]) -> np.ndarray:
    """Points of the dense check grid (:func:`default_k_grid`) where g of
    the slice x is below twice the constraint margin (or undefined), with
    their neighbours on that grid: the constraints added where a solution
    dipped between constraint points. (Adding the whole dense grid, about
    2,000 points, made each later SLSQP step several times dearer.)"""

    raw = _x_to_raw(x)
    dense = default_k_grid(dict(zip(SVI_PARAMS, (float(v) for v in raw))), k_range)
    low = ~(np.nan_to_num(_g_raw(dense, *raw), nan=-1.0) >= 2.0 * _G_MARGIN)
    near = low | np.r_[low[1:], False] | np.r_[False, low[:-1]]
    return dense[near]


def _arbitrage_free_refit(
    x_free, k, vols, T, wts, loss, f_scale, lb, ub, k_range, resid, jac, fixed_grid, pool, n_seeds=REFIT_STARTS
):
    """Replace an arbitrageable fit; returns ``(x, repair, ok, nfev, message)``
    with ``repair`` in {'constrained', 'ssvi_slice', 'failed'} (see module
    docs). The objective is the same robust loss on the weighted vol
    residuals as the free fit. SLSQP starts from the ``n_seeds`` best
    distinct quasi-explicit starts of ``pool`` (the free fit's seed pool)
    that already satisfy the constraints; these sit close to the constrained
    optimum (within a few % of its objective on real slices), where the SSVI
    slice used before was 5-20x off and SLSQP needed 250-550 steps from it.
    Only when none of them gives a fit that passes the final butterfly and
    Lee checks does the refit fall back to SLSQP from the SSVI slice, then
    from the free fit, and to the SSVI slice itself. The lowest-cost passing
    candidate is kept."""

    f2 = 1.0 if loss == "linear" else float(f_scale) ** 2

    def cost_x(x):
        r = resid(x)
        rho_z, drho = _loss_rho((r * r) / f2, loss)
        return 0.5 * f2 * float(np.sum(rho_z)), (drho * r) @ jac(x)

    lo = lb.copy()
    lo[0] = max(lo[0], 1e-9 * float(np.max(vols * vols * T)))  # keep w > 0 so g is defined
    candidates = []
    nfev = 0
    screen = _constraint_grid([x_free], k_range) if fixed_grid is None else fixed_grid
    for x_start in _feasible_seeds(pool, screen, lb, ub, n_seeds):
        grid = _constraint_grid([x_free, x_start], k_range) if fixed_grid is None else fixed_grid
        x, ok, n = _constrained_slsqp(x_start, x_free, cost_x, grid, lo, ub, k_range, fixed_grid)
        nfev += n
        if ok:
            candidates.append((cost_x(x)[0], x, "constrained"))
    if not candidates:  # fallback: from the SSVI slice (butterfly-free by construction), then the free fit
        x_ssvi, n = _fit_ssvi_slice(k, vols, T, wts, loss, f_scale)
        nfev += n
        if _passes_checks(x_ssvi, k_range, fixed_grid)[0]:
            candidates.append((cost_x(x_ssvi)[0], x_ssvi, "ssvi_slice"))
        grid = _constraint_grid([x_free, x_ssvi], k_range) if fixed_grid is None else fixed_grid
        for x_start in (x_ssvi, x_free):
            x, ok, n = _constrained_slsqp(x_start, x_free, cost_x, grid, lo, ub, k_range, fixed_grid)
            nfev += n
            if ok:
                candidates.append((cost_x(x)[0], x, "constrained"))
                break  # the start from the (feasible) SSVI slice worked: the free start rarely beats it
    if candidates:
        _, x_best, how = min(candidates, key=lambda c: c[0])
        message = {
            "constrained": "constrained SLSQP fit (g(k) >= 1e-4, Lee slopes <= 2): butterfly and Lee checks pass",
            "ssvi_slice": "arbitrage-free SSVI slice (the constrained fits were not better)",
        }[how]
        return x_best, how, True, nfev, message
    return x_free, "failed", False, nfev, "no arbitrage-free fit found: the free fit is returned and flagged"
