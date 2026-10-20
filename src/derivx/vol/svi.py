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
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

SVI_PARAMS = ("a", "b", "rho", "m", "sigma")
LEE_MAX_SLOPE = 2.0
_RHO_MAX = 0.999
_PENALTY = 10.0  # weight (vol units per unit of g) of the arbitrage penalties
_G_MARGIN = 1e-3  # penalised fits aim for g >= 1e-3 on the penalty grid
_PENALISED_MAX_NFEV = 300  # the penalty is not smooth: cap the refit; the final check reports what remains


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
    penalised: bool = False  # True when the arbitrage penalties were applied (enforce and the free fit failed)

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
            "at_bound": ", ".join(k for k, v in self.at_bound.items() if v),
            "success": self.success,
        }


_X_NAMES = ("min_variance", "b", "rho", "m", "sigma")


def _x_to_raw(x: np.ndarray) -> Tuple[float, float, float, float, float]:
    wmin, b, rho, m, sigma = x
    return wmin - b * sigma * np.sqrt(1.0 - rho * rho), b, rho, m, sigma


def _raw_to_x(a, b, rho, m, sigma) -> np.ndarray:
    return np.array([a + b * sigma * np.sqrt(1.0 - rho * rho), b, rho, m, sigma])


def _seeds(k, w_obs, omega, lb, ub, count) -> List[np.ndarray]:
    """Start points from the quasi-explicit (Zeliade) linear fit: for fixed
    (m, sigma), w is linear in (a, b rho, b)."""

    kmin, kmax = float(k.min()), float(k.max())
    span = max(kmax - kmin, 1e-3)
    cands = []
    for m in np.unique(np.r_[np.linspace(kmin, kmax, 5), 0.0]):
        for sig in span * np.array([0.05, 0.15, 0.4, 1.0]) + 1e-3:
            u = k - m
            X = np.column_stack([np.ones_like(k), u, np.sqrt(u * u + sig * sig)])
            coef, *_ = np.linalg.lstsq(X * omega[:, None], w_obs * omega, rcond=None)
            a, c_rho, b = coef
            b = max(b, 1e-6)
            rho = float(np.clip(c_rho / b, -0.95, 0.95))
            x = _raw_to_x(a, b, rho, m, sig)
            x[0] = max(x[0], 0.0)
            x = np.clip(x, lb + 1e-12, ub - 1e-12)
            pred = _raw_w(k, *_x_to_raw(x))
            cost = float(np.sum((omega * (pred - w_obs)) ** 2))
            cands.append((cost, x))
    cands.sort(key=lambda c: c[0])
    out: List[np.ndarray] = []
    for _, x in cands:
        if all(np.max(np.abs(x - y) / (ub - lb)) > 0.05 for y in out):
            out.append(x)
        if len(out) >= count:
            break
    return out


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
        refit with penalties for g(k) < 1e-3 on the checked k range
        (``check_grid``, default that of :func:`default_k_grid`) and for Lee
        slopes above 2; the result's ``butterfly`` / ``lee`` checks report
        what remains (penalties do not guarantee exactness).
    n_starts : start points (from a quasi-explicit linear fit over (m,
        sigma)); ``x0`` (params) replaces them with one warm start.

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
    quoted = np.linspace(kmin, kmax, 41)

    def penalty_grid(m: float, sigma: float) -> np.ndarray:
        """The range :func:`default_k_grid` checks (fixed length)."""

        if fixed_grid is not None:
            return fixed_grid
        return np.concatenate([m + np.linspace(-3.0, 3.0, 241), m + sigma * np.linspace(-10.0, 10.0, 81), quoted])

    big = 10.0 * float(vols.max())

    def resid(x: np.ndarray, penalise: bool = False) -> np.ndarray:
        a, b, rho, m, sigma = _x_to_raw(x)
        w = _raw_w(k, a, b, rho, m, sigma)
        out = wts * (np.sqrt(np.maximum(w, 0.0) / T) - vols)
        if not np.all(np.isfinite(out)):
            out = np.full(K.size, big)
        if not penalise:
            return out
        p = {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}
        g = butterfly_density(penalty_grid(m, sigma), p)
        pen_g = np.where(np.isfinite(g), np.maximum(_G_MARGIN - g, 0.0), 1.0)
        pen_lee = max(0.0, b * (1.0 + abs(rho)) - (LEE_MAX_SLOPE - _G_MARGIN))
        return np.concatenate([out, _PENALTY * pen_g, [_PENALTY * pen_lee]])

    def jac(x: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of the unpenalised residuals in (min_variance,
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

    if x0 is not None:
        starts = [np.clip(_raw_to_x(*svi_tuple(x0)), lb + 1e-12, ub - 1e-12)]
    else:
        omega = wts / (2.0 * np.maximum(vols, 1e-4) * T)
        starts = _seeds(k, w_obs, omega, lb, ub, max(1, int(n_starts)))
    best = None
    for xs in starts:
        res = least_squares(
            resid, x0=xs, jac=jac, bounds=(lb, ub), loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=3000
        )
        if best is None or res.cost < best.cost:
            best = res
    penalised = False
    if enforce_no_arbitrage:
        p_free = dict(zip(SVI_PARAMS, (float(v) for v in _x_to_raw(best.x))))
        free_ok = check_butterfly(p_free, default_k_grid(p_free, (kmin, kmax))).ok and lee_wing_check(p_free)["lee_ok"]
        if not free_ok:  # refit from the unconstrained optimum with the arbitrage penalties
            best = least_squares(
                lambda x: resid(x, True),
                x0=best.x,
                bounds=(lb, ub),
                loss=loss,
                f_scale=f_scale,
                x_scale="jac",
                max_nfev=_PENALISED_MAX_NFEV,
            )
            penalised = True
    a, b, rho, m, sigma = _x_to_raw(best.x)
    params = {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sigma": float(sigma)}
    model = np.sqrt(np.maximum(_raw_w(k, a, b, rho, m, sigma), 0.0) / T)
    err = model - vols
    n = K.size
    J = best.jac[:n]
    at_bound = {}
    for i, name in enumerate(_X_NAMES):
        width = ub[i] - lb[i]
        at_bound[name] = bool(best.x[i] - lb[i] <= 1e-3 * width or ub[i] - best.x[i] <= 1e-3 * width)
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
    grid = default_k_grid(params, (kmin, kmax))
    return SviFit(
        **params,
        T=T,
        F=F,
        rmse=float(np.sqrt(np.mean(err**2))),
        max_abs_error=float(np.max(np.abs(err))),
        n_points=int(n),
        n_dropped=n_dropped,
        at_bound=at_bound,
        success=bool(best.success),
        message=str(best.message),
        nfev=int(best.nfev),
        n_starts=len(starts),
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
    )
