"""SABR smiles: Hagan et al. (2002) lognormal and normal expansions,
shifted variants, ATM alpha and calibration.

Model: ``dF = alpha_t (F + s)^beta dW``, ``d alpha_t = nu alpha_t dZ``,
``dW dZ = rho dt``, with shift ``s >= 0`` (``s = 0`` is plain SABR).

Conventions
-----------
* ``F`` forward, ``K`` strike, in price units (a decimal rate for rates);
  ``T`` years to expiry; ``shift`` in the same units as F (0.02 = 2%).
* ``hagan_sabr_iv`` returns the Black (lognormal) vol of the *shifted*
  forward, a per-annum decimal. ``hagan_sabr_normal_vol`` returns the
  Bachelier vol in price units per sqrt(year) (``* 1e4`` = bp for rates).
* ``alpha`` has units of ``vol * (F + s)^(1 - beta)``: ``alpha ~ atm_vol *
  F^(1-beta)`` (lognormal) or ``atm_normal_vol / F^beta``. It is *not*
  comparable across beta or across price levels; calibration therefore
  scales its start point and bounds by ``(F + s)^(1-beta)``.

Formula (lognormal, Hagan 2002 eq. 2.17a), with ``f = F + s``, ``k = K + s``,
``l = ln(f/k)``, ``A = (fk)^((1-beta)/2)``::

    sigma_B = alpha / (A [1 + (1-b)^2 l^2/24 + (1-b)^4 l^4/1920])   # chi denominator
              * z / x(z)
              * [1 + T ((1-b)^2 alpha^2 / (24 A^2) + rho b nu alpha / (4 A) + (2 - 3 rho^2) nu^2 / 24)]
    z = nu / alpha * A * l,   x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho))

``z/x(z)`` uses its series ``1 - rho z/2 + (2 - 3 rho^2) z^2/12`` near z = 0
and a cancellation-free form elsewhere, so the smile is smooth through
K = F, and beta = 1 (``A = 1``, ``z = nu/alpha ln(f/k)``) and nu = 0 need no
special case. The normal-vol expansion (Hagan 2002 eq. B.69a) is::

    sigma_N = alpha (fk)^(b/2) [1 + l^2/24 + l^4/1920] / [1 + (1-b)^2 l^2/24 + (1-b)^4 l^4/1920]
              * z / x(z)
              * [1 + T (-b (2-b) alpha^2 / (24 A^2) + rho b nu alpha / (4 A) + (2 - 3 rho^2) nu^2 / 24)]

Both match QuantLib's ``sabrVolatility`` / ``shiftedSabrVolatility``
(lognormal and ``VolatilityType.Normal``). The expansion is asymptotic in
``nu^2 T``: when the time correction makes a vol non-positive (long T,
strongly negative rho, large nu) the functions raise rather than return
the negative number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ("alpha", "beta", "rho", "nu", "shift")
VOL_TYPES = ("lognormal", "normal")
RHO_MAX = 0.9999  # calibration bound on |rho|


def _validate_params(alpha, beta, rho, nu, T) -> None:
    alpha, beta, rho, nu, T = (np.asarray(x, dtype=float) for x in (alpha, beta, rho, nu, T))
    for name, x in (("alpha", alpha), ("beta", beta), ("rho", rho), ("nu", nu), ("T", T)):
        if not np.all(np.isfinite(x)):
            raise ValueError(f"SABR {name} must be finite, got {x}")
    if np.any(alpha <= 0):
        raise ValueError(f"SABR alpha must be > 0, got {alpha}")
    if np.any((beta < 0) | (beta > 1)):
        raise ValueError(f"SABR beta must be in [0, 1], got {beta}")
    if np.any(np.abs(rho) >= 1):
        raise ValueError(f"SABR rho must satisfy |rho| < 1, got {rho}")
    if np.any(nu < 0):
        raise ValueError(f"SABR nu (vol of vol) must be >= 0, got {nu}")
    if np.any(T < 0):
        raise ValueError(f"T (years to expiry) must be >= 0, got {T}")


def _shifted(F, K, shift) -> Tuple[np.ndarray, np.ndarray]:
    F, K, shift = (np.asarray(x, dtype=float) for x in (F, K, shift))
    if not (np.all(np.isfinite(F)) and np.all(np.isfinite(K)) and np.all(np.isfinite(shift))):
        raise ValueError("F, K and shift must be finite")
    f, k = F + shift, K + shift
    if np.any(f <= 0):
        raise ValueError(
            f"SABR needs F + shift > 0 (got F + shift = {np.min(f):.6g}): raise the shift for negative forwards"
        )
    if np.any(k <= 0):
        raise ValueError(
            f"SABR needs K + shift > 0 (got K + shift = {np.min(k):.6g}): drop strikes at or below -shift "
            "or raise the shift"
        )
    return f, k


def z_over_x(z, rho) -> np.ndarray:
    """``z / x(z)`` of Hagan's formula, smooth at z = 0 (series for
    |z| < 1e-6) and free of cancellation for large |z| of either sign."""

    z = np.asarray(z, dtype=float)
    rho = np.broadcast_to(np.asarray(rho, dtype=float), z.shape)
    out = np.empty(z.shape)
    small = np.abs(z) < 1e-6
    zs, rs = z[small], rho[small]
    out[small] = 1.0 - 0.5 * rs * zs + (2.0 - 3.0 * rs * rs) * zs * zs / 12.0
    zl, rl = z[~small], rho[~small]
    sq = np.sqrt(1.0 - 2.0 * rl * zl + zl * zl)
    core = (zl * zl - 2.0 * rl * zl) / (sq + 1.0)  # sqrt(B) - 1 without cancellation
    with np.errstate(divide="ignore", invalid="ignore"):
        x_pos = np.log1p((core + zl) / (1.0 - rl))
        x_neg = -np.log1p((core - zl) / (1.0 + rl))
    x = np.where(zl >= 0, x_pos, x_neg)
    out[~small] = zl / x
    return out


def _raw_vols(F, K, T, alpha, beta, rho, nu, shift, vol_type: str) -> np.ndarray:
    """Hagan vols without input checks (inputs already shifted-valid)."""

    f = np.asarray(F, dtype=float) + shift
    k = np.asarray(K, dtype=float) + shift
    omb = 1.0 - beta
    lf, lk = np.log(f), np.log(k)
    lg = lf - lk
    A = np.exp(0.5 * omb * (lf + lk))  # (fk)^((1-b)/2)
    z = (nu / alpha) * A * lg
    zx = z_over_x(z, rho)
    c = omb * omb * lg * lg
    chi = 1.0 + c / 24.0 + c * c / 1920.0
    ccorr = rho * beta * nu * alpha / (4.0 * A) + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    if vol_type == "lognormal":
        corr = 1.0 + T * (omb * omb * alpha * alpha / (24.0 * A * A) + ccorr)
        return alpha / (A * chi) * zx * corr
    l2 = lg * lg
    num = 1.0 + l2 / 24.0 + l2 * l2 / 1920.0
    corr = 1.0 + T * (-beta * (2.0 - beta) * alpha * alpha / (24.0 * A * A) + ccorr)
    return alpha * np.exp(0.5 * beta * (lf + lk)) * num / chi * zx * corr


def _check_vol_type(vol_type: str) -> str:
    if vol_type not in VOL_TYPES:
        raise ValueError(f"vol_type must be one of {VOL_TYPES}, got {vol_type!r}")
    return vol_type


def sabr_vol(F, K, T, alpha, beta, rho, nu, shift=0.0, vol_type: str = "lognormal"):
    """Hagan SABR vol of ``vol_type`` ('lognormal' Black vol of F + shift,
    or 'normal' Bachelier vol). Vectorised over every argument; returns a
    float for scalar inputs. Raises ``ValueError`` on invalid inputs or when
    the expansion breaks down (vol <= 0)."""

    _check_vol_type(vol_type)
    _validate_params(alpha, beta, rho, nu, T)
    _shifted(F, K, shift)
    with np.errstate(over="ignore", invalid="ignore"):
        vol = _raw_vols(F, K, T, alpha, beta, rho, nu, shift, vol_type)
    if np.any(~np.isfinite(vol)):
        raise ValueError("SABR vol is not finite for these inputs (overflow in the expansion)")
    if np.any(vol <= 0):
        K_arr = np.broadcast_to(np.asarray(K, dtype=float), np.shape(vol))
        raise ValueError(
            f"Hagan's SABR expansion breaks down here (vol <= 0 at K = {K_arr[vol <= 0][:3]}): "
            "nu^2 T is too large for the asymptotic formula; shorten T or reduce nu"
        )
    return float(vol) if np.ndim(vol) == 0 else vol


def hagan_sabr_iv(F, K, T, alpha, beta, rho, nu, shift=0.0):
    """Hagan (2002) SABR Black vol (per-annum decimal) of the shifted
    forward ``F + shift`` at strike(s) ``K``. See module docs."""

    return sabr_vol(F, K, T, alpha, beta, rho, nu, shift, "lognormal")


def hagan_sabr_normal_vol(F, K, T, alpha, beta, rho, nu, shift=0.0):
    """Hagan (2002) SABR normal (Bachelier) vol in price units per
    sqrt(year) (``* 1e4`` = bp for rates). Any beta in [0, 1]; beta = 0 is
    the normal SABR model. See module docs."""

    return sabr_vol(F, K, T, alpha, beta, rho, nu, shift, "normal")


def sabr_alpha_from_atm(F, T, atm_vol, beta, rho, nu, shift=0.0, vol_type: str = "lognormal") -> float:
    """Alpha that reproduces ``atm_vol`` at K = F exactly: the smallest
    positive root of Hagan's ATM cubic (West 2005).

    Lognormal: ``atm_vol f^(1-b) = alpha + T [(1-b)^2 alpha^3 / (24 f^(2-2b))
    + rho b nu alpha^2 / (4 f^(1-b)) + (2 - 3 rho^2) nu^2 alpha / 24]``;
    normal: the same with ``-b(2-b)`` for ``(1-b)^2`` and an ``f^b`` factor."""

    _check_vol_type(vol_type)
    _validate_params(1.0, beta, rho, nu, T)
    f = float(F) + float(shift)
    if f <= 0:
        raise ValueError("SABR needs F + shift > 0")
    if not np.isfinite(atm_vol) or atm_vol <= 0:
        raise ValueError(f"atm_vol must be a positive number, got {atm_vol}")
    omb = 1.0 - beta
    p = f**omb
    c1 = 1.0 + T * (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    c2 = T * rho * beta * nu / (4.0 * p)
    if vol_type == "lognormal":
        coeffs = [T * omb * omb / (24.0 * p * p), c2, c1, -atm_vol * p]
    else:
        fb = f**beta
        coeffs = [-T * beta * (2.0 - beta) * fb / (24.0 * p * p), c2 * fb, c1 * fb, -atm_vol]
    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) <= 1e-10 * np.maximum(1.0, np.abs(roots.real))].real
    positive = np.sort(real[real > 0])
    if positive.size == 0:
        raise ValueError(
            "no positive alpha reproduces this ATM vol with these rho, nu and T (the ATM cubic has no "
            "positive root): Hagan's expansion is outside its range"
        )
    return float(positive[0])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SabrCalibration(Mapping):
    """A calibrated SABR smile.

    Behaves as a read-only mapping of the model parameters (alpha, beta,
    rho, nu, shift), so ``hagan_sabr_iv(F, K, T, **fit)`` works. Errors are
    in the units of the fitted vols (``vol_type``): ``rmse`` is a decimal
    vol, ``rmse_bp = rmse * 1e4`` (vol bp for lognormal, bp of normal vol
    for normal); residuals are model minus market."""

    alpha: float
    beta: float
    rho: float
    nu: float
    shift: float
    F: float
    T: float
    vol_type: str
    rmse: float
    max_abs_error: float
    n_points: int
    n_dropped: int
    free: Tuple[str, ...]
    fixed: Tuple[str, ...]
    at_bound: Dict[str, bool]
    success: bool
    message: str
    nfev: int
    n_starts: int
    jacobian_condition: float
    strikes: np.ndarray = field(repr=False)
    market_vols: np.ndarray = field(repr=False)
    model_vols: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    alpha_from_atm: bool = False  # alpha solved from the ATM vol, not fitted
    loss: str = "linear"
    f_scale: Optional[float] = None

    # Mapping protocol over the model parameters
    def __getitem__(self, key: str) -> float:
        if key not in PARAM_NAMES:
            raise KeyError(key)
        return float(getattr(self, key))

    def __iter__(self) -> Iterator[str]:
        return iter(PARAM_NAMES)

    def __len__(self) -> int:
        return len(PARAM_NAMES)

    @property
    def params(self) -> Dict[str, float]:
        """Model parameters as a plain dict."""

        return {name: float(getattr(self, name)) for name in PARAM_NAMES}

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
    def any_at_bound(self) -> bool:
        return any(self.at_bound.values())

    def vol(self, K, T: Optional[float] = None):
        """Model vol (same ``vol_type``) at strikes ``K``."""

        return sabr_vol(
            self.F, K, self.T if T is None else T, self.alpha, self.beta, self.rho, self.nu, self.shift, self.vol_type
        )

    def summary(self) -> Dict[str, object]:
        """Flat dict for tables: params, errors in bp, flags."""

        return {
            **self.params,
            "vol_type": self.vol_type,
            "rmse_bp": self.rmse_bp,
            "max_abs_error_bp": self.max_abs_error_bp,
            "n_points": self.n_points,
            "fixed": ", ".join(self.fixed),
            "at_bound": ", ".join(k for k, v in self.at_bound.items() if v),
            "success": self.success,
        }


def atm_vol_from_quotes(f: float, k: np.ndarray, vols: np.ndarray) -> float:
    """Quoted vol at the (shifted) forward ``f``: a quadratic in ln(k/f)
    through the three quotes nearest the money when f is inside the quoted
    range, else the nearest quote. ``k`` are shifted strikes."""

    x = np.log(np.asarray(k, dtype=float) / f)
    vols = np.asarray(vols, dtype=float)
    if x.size >= 3 and x.min() < 0.0 < x.max():
        near = np.argsort(np.abs(x))[:3]
        if np.unique(x[near]).size == 3:
            coef = np.polyfit(x[near], vols[near], 2)
            value = float(np.polyval(coef, 0.0))
            lo, hi = vols[near].min(), vols[near].max()
            if lo > 0 and 0.5 * lo <= value <= 2.0 * hi:
                return value
    order = np.argsort(x)
    return float(np.interp(0.0, x[order], vols[order]))


def calibrate_sabr_smile(
    Ks: Sequence[float],
    ivs: Sequence[float],
    T: float,
    F: float,
    beta: float = 0.5,
    shift: float = 0.0,
    *,
    vol_type: str = "lognormal",
    weights: Union[None, str, Sequence[float]] = None,
    fix_beta: bool = True,
    alpha_from_atm: bool = False,
    atm_vol: Optional[float] = None,
    loss: str = "linear",
    f_scale: Optional[float] = None,
    n_starts: Optional[int] = None,
) -> SabrCalibration:
    """Least-squares SABR fit to one expiry's smile.

    Parameters
    ----------
    Ks, ivs : strikes and market vols of ``vol_type`` (Black vols of the
        shifted forward, or normal vols in price units). Non-finite quotes
        are dropped and counted in ``n_dropped``.
    T, F : years to expiry and forward; ``shift`` as in the model.
    beta : fixed beta (the usual choice: 1 for FX/equity, 0-0.5 for rates),
        or the start value when ``fix_beta=False``.
    weights : None (equal), 'vega' (Black or Bachelier vega, down-weights
        far wings), or an array (e.g. 1 / bid-ask spread in vol). Scaled to
        mean 1, so residuals and ``f_scale`` stay in vol units.
    alpha_from_atm : solve alpha from the ATM vol (``atm_vol``, or the quotes
        interpolated at K = F) with :func:`sabr_alpha_from_atm`, so the fit
        reprices ATM exactly and only rho, nu (and beta) are free.
    loss, f_scale : scipy loss ('linear', 'soft_l1', 'huber', 'cauchy');
        ``f_scale`` is the residual size (vol units) beyond which a robust
        loss down-weights quotes. Defaults: 25 vol bp (0.0025) lognormal,
        1 bp (1e-4) normal.
    n_starts : number of (rho, nu) start points (default 9, or 4 with
        ``alpha_from_atm``); the best fit is returned.

    Start point and bounds scale with the forward: alpha is searched in
    ``[1/50, 50] x atm_vol (F+s)^(1-beta)`` (lognormal) or
    ``atm_vol / (F+s)^beta`` (normal), so the fit does not depend on the
    price level of the underlying; ``|rho| <= 0.9999``; ``0 <= nu <=
    max(5, 4/sqrt(T))``. ``at_bound`` flags any parameter that ends within
    0.1% of its bound. Raises ``ValueError`` when fewer valid quotes than
    free parameters + 1 remain.
    """

    _check_vol_type(vol_type)
    Ks = np.asarray(Ks, dtype=float).ravel()
    ivs = np.asarray(ivs, dtype=float).ravel()
    if Ks.shape != ivs.shape:
        raise ValueError(f"Ks and ivs must have the same length ({Ks.size} vs {ivs.size})")
    T = float(T)
    F = float(F)
    shift = float(shift)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"T must be > 0 years to calibrate a smile, got {T}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    f = F + shift
    if not np.isfinite(f) or f <= 0:
        raise ValueError("SABR needs F + shift > 0: raise the shift for negative forwards")
    keep = np.isfinite(Ks) & np.isfinite(ivs) & (ivs > 0) & (Ks + shift > 0)
    w_user = None
    if weights is not None and not isinstance(weights, str):
        w_user = np.asarray(weights, dtype=float).ravel()
        if w_user.shape != Ks.shape:
            raise ValueError("weights must have one entry per quote")
        keep &= np.isfinite(w_user) & (w_user > 0)
    n_dropped = int((~keep).sum())
    K, vols = Ks[keep], ivs[keep]
    free = (["rho", "nu"] if alpha_from_atm else ["alpha", "rho", "nu"]) + ([] if fix_beta else ["beta"])
    if K.size <= len(free):
        raise ValueError(f"{K.size} valid quotes for {len(free)} free SABR parameters: need at least {len(free) + 1}")

    if weights is None:
        w = np.ones(K.size)
    elif isinstance(weights, str):
        if weights != "vega":
            raise ValueError("weights must be None, 'vega' or an array")
        from derivx.vol.implied import bachelier_vega, black_vega

        if vol_type == "lognormal":
            w = np.asarray(black_vega(F, K, T, vols, shift=shift), dtype=float)
        else:
            w = np.asarray(bachelier_vega(F, K, T, vols), dtype=float)
        w = np.maximum(w, 1e-6 * w.max())
    else:
        w = w_user[keep]
    w = w / w.mean()

    if atm_vol is not None and (not np.isfinite(atm_vol) or atm_vol <= 0):
        raise ValueError("atm_vol must be a positive number")
    atm = float(atm_vol) if atm_vol is not None else atm_vol_from_quotes(f, K + shift, vols)

    def alpha_scale(b: float) -> float:
        return atm * f ** (1.0 - b) if vol_type == "lognormal" else atm / f**b

    nu_max = max(5.0, 4.0 / np.sqrt(T))
    names = list(free)
    lb, ub = [], []
    for name in names:
        if name == "alpha":
            lb.append(np.log(1.0 / 50.0))
            ub.append(np.log(50.0))
        elif name == "rho":
            lb.append(-RHO_MAX)
            ub.append(RHO_MAX)
        elif name == "nu":
            lb.append(0.0)
            ub.append(nu_max)
        else:  # beta
            lb.append(0.0)
            ub.append(1.0)
    lb, ub = np.array(lb), np.array(ub)
    big = 10.0 * max(float(np.max(vols)), 1e-12)

    def unpack(x: np.ndarray) -> Tuple[float, float, float, float]:
        vals = dict(zip(names, x))
        b = float(vals.get("beta", beta))
        r, n = float(vals["rho"]), float(vals["nu"])
        if alpha_from_atm:
            a = sabr_alpha_from_atm(F, T, atm, b, r, n, shift, vol_type)
        else:
            a = float(np.exp(vals["alpha"]) * alpha_scale(b))
        return a, b, r, n

    def model(a: float, b: float, r: float, n: float) -> np.ndarray:
        with np.errstate(all="ignore"):
            return _raw_vols(F, K, T, a, b, r, n, shift, vol_type)

    def resid(x: np.ndarray) -> np.ndarray:
        try:
            a, b, r, n = unpack(x)
        except ValueError:
            return np.full(K.size, big)
        m = model(a, b, r, n)
        out = w * (m - vols)
        return np.where(np.isfinite(m) & (m > 0), out, big)

    if f_scale is None:
        f_scale = 0.0025 if vol_type == "lognormal" else 1e-4
    if n_starts is None:
        n_starts = 4 if alpha_from_atm else 9
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    lo_nu, mid_nu, hi_nu = np.array([0.3, 0.8, 1.6]) / np.sqrt(np.clip(T, 1.0 / 52.0, 10.0))
    seeds = [
        (0.0, mid_nu),
        (-0.5, lo_nu),
        (0.5, lo_nu),
        (-0.5, hi_nu),
        (0.5, hi_nu),
        (0.0, lo_nu),
        (0.0, hi_nu),
        (-0.5, mid_nu),
        (0.5, mid_nu),
    ][:n_starts]

    best = None
    for r0, n0 in seeds:
        x0 = []
        for name in names:
            x0.append({"alpha": 0.0, "rho": r0, "nu": min(n0, 0.9 * nu_max), "beta": beta}[name])
        x0 = np.clip(np.array(x0), lb + 1e-9, ub - 1e-9)
        res = least_squares(resid, x0=x0, bounds=(lb, ub), loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=2000)
        if best is None or res.cost < best.cost:
            best = res
    a, b, r, n = unpack(best.x)
    m = model(a, b, r, n)
    err = m - vols
    finite = np.isfinite(err)
    rmse = float(np.sqrt(np.mean(err[finite] ** 2))) if finite.all() else float("nan")

    at_bound: Dict[str, bool] = {}
    for i, name in enumerate(names):
        span = ub[i] - lb[i]
        at_bound[name] = bool(best.x[i] - lb[i] <= 1e-3 * span or ub[i] - best.x[i] <= 1e-3 * span)
    try:
        sv = np.linalg.svd(best.jac, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        cond = float("nan")
    fixed = tuple(p for p in ("beta", "shift") if p not in free)
    return SabrCalibration(
        alpha=a,
        beta=b,
        rho=r,
        nu=n,
        shift=shift,
        F=F,
        T=T,
        vol_type=vol_type,
        rmse=rmse,
        max_abs_error=float(np.max(np.abs(err))) if finite.all() else float("nan"),
        n_points=int(K.size),
        n_dropped=n_dropped,
        free=tuple(free),
        fixed=fixed,
        at_bound=at_bound,
        success=bool(best.success) and finite.all() and bool(np.all(m > 0)),
        message=str(best.message),
        nfev=int(best.nfev),
        n_starts=len(seeds),
        jacobian_condition=cond,
        strikes=K,
        market_vols=vols,
        model_vols=m,
        weights=w,
        alpha_from_atm=alpha_from_atm,
        loss=loss,
        f_scale=f_scale if loss != "linear" else None,
    )


# ---------------------------------------------------------------------------
# Cube of smiles keyed by (expiry, tenor)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SabrCube:
    """SABR smiles on a full (expiry x tenor) grid, e.g. a swaption cube.

    ``smiles`` maps ``(expiry_years, tenor_years)`` to a
    :class:`SabrCalibration`; all must share ``beta``, ``shift`` and
    ``vol_type``. Between grid points the ATM vol (of ``vol_type``), rho and
    nu are interpolated bilinearly in (expiry, tenor), flat outside the grid,
    and alpha is re-solved from the interpolated ATM vol at the target
    forward (:func:`sabr_alpha_from_atm`), so the cube reprices its own ATM
    points and scales with the forward."""

    smiles: Dict[Tuple[float, float], SabrCalibration]

    def __post_init__(self) -> None:
        if not self.smiles:
            raise ValueError("SabrCube needs at least one smile")
        expiries = sorted({float(e) for e, _ in self.smiles})
        tenors = sorted({float(t) for _, t in self.smiles})
        missing = [(e, t) for e in expiries for t in tenors if (e, t) not in self.smiles]
        if missing:
            raise ValueError(f"SabrCube needs a full expiry x tenor grid; missing {missing[:5]}")
        first = next(iter(self.smiles.values()))
        for key, fit in self.smiles.items():
            if (fit.beta, fit.shift, fit.vol_type) != (first.beta, first.shift, first.vol_type):
                raise ValueError(f"smile {key} has a different beta, shift or vol_type from the rest of the cube")

    @property
    def expiries(self) -> List[float]:
        return sorted({float(e) for e, _ in self.smiles})

    @property
    def tenors(self) -> List[float]:
        return sorted({float(t) for _, t in self.smiles})

    def _grid(self, attr: str) -> np.ndarray:
        out = np.empty((len(self.expiries), len(self.tenors)))
        for i, e in enumerate(self.expiries):
            for j, t in enumerate(self.tenors):
                fit = self.smiles[(e, t)]
                if attr == "atm":
                    out[i, j] = fit.vol(fit.F)
                else:
                    out[i, j] = getattr(fit, attr)
        return out

    @staticmethod
    def _bilinear(xs, ys, grid, x, y) -> float:
        xs, ys = np.asarray(xs), np.asarray(ys)
        x, y = float(np.clip(x, xs[0], xs[-1])), float(np.clip(y, ys[0], ys[-1]))
        rows = np.array([np.interp(y, ys, grid[i]) for i in range(xs.size)])
        return float(np.interp(x, xs, rows))

    def params_at(self, expiry: float, tenor: float, F: float) -> Dict[str, float]:
        """Interpolated SABR parameters for (expiry, tenor) at forward F."""

        first = next(iter(self.smiles.values()))
        atm = self._bilinear(self.expiries, self.tenors, self._grid("atm"), expiry, tenor)
        rho = self._bilinear(self.expiries, self.tenors, self._grid("rho"), expiry, tenor)
        nu = self._bilinear(self.expiries, self.tenors, self._grid("nu"), expiry, tenor)
        alpha = sabr_alpha_from_atm(F, expiry, atm, first.beta, rho, nu, first.shift, first.vol_type)
        return {"alpha": alpha, "beta": first.beta, "rho": rho, "nu": nu, "shift": first.shift}

    def vol(self, expiry: float, tenor: float, F: float, K):
        """Cube vol (of the cube's ``vol_type``) at strike(s) K."""

        first = next(iter(self.smiles.values()))
        p = self.params_at(expiry, tenor, F)
        return sabr_vol(F, K, expiry, p["alpha"], p["beta"], p["rho"], p["nu"], p["shift"], first.vol_type)
