"""SSVI: an arbitrage-free implied-variance surface (Gatheral & Jacquier
2014) with Dupire local volatility.

Surface, in total implied variance ``w = sigma_BS^2 T`` and forward
log-moneyness ``k = ln(K / F(T))``::

    w(k, T) = theta(T) / 2 * (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)),   phi = phi(theta(T))
    phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))                       (power law)

``theta(T)`` is the ATM total variance, given at the fitted expiries and
linear in T in between (from theta(0) = 0; flat ATM vol beyond the last
expiry). The fit imposes the sufficient no-static-arbitrage conditions of
Gatheral-Jacquier (Theorem 4.1, Remark 4.4): ``theta`` non-decreasing in T
(by fitting non-negative increments), ``0 < gamma <= 1/2`` and
``eta (1 + |rho|) <= 2`` (by fitting ``eta = u 2 / (1 + |rho|)`` with
``0 < u <= 1``). Each slice is a raw SVI with ``a = theta (1 - rho^2) / 2``,
``b = theta phi / 2``, ``m = -rho / phi``, ``sigma = sqrt(1 - rho^2) / phi``.

Dupire local vol from the surface (Gatheral 2006, eq. 1.10):
``sigma_loc^2(k, T) = (dw/dT at fixed k) / g(k, T)`` with g the butterfly
density factor of :func:`derivx.vol.svi.butterfly_density`. Because theta
is piecewise linear in T, local vol jumps at the fitted expiries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from derivx.vol.svi import butterfly_density, check_butterfly, check_calendar


@dataclass(frozen=True)
class SsviSurface:
    """Power-law SSVI surface. ``expiries`` (years, increasing) and
    ``theta`` (ATM total variance, non-decreasing) define theta(T)."""

    expiries: np.ndarray
    theta: np.ndarray
    rho: float
    eta: float
    gamma: float

    def __post_init__(self) -> None:
        T = np.asarray(self.expiries, dtype=float)
        th = np.asarray(self.theta, dtype=float)
        if T.ndim != 1 or T.size == 0 or T.shape != th.shape:
            raise ValueError("expiries and theta must be 1-D arrays of the same length")
        if np.any(T <= 0) or np.any(np.diff(T) <= 0):
            raise ValueError("expiries must be positive and strictly increasing")
        if np.any(th <= 0):
            raise ValueError("theta (ATM total variance) must be > 0")
        if np.any(np.diff(th) < -1e-14):
            raise ValueError("theta must be non-decreasing in T (otherwise there is calendar arbitrage at the money)")
        if abs(self.rho) >= 1 or self.eta <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("SSVI needs |rho| < 1, eta > 0 and 0 < gamma <= 1")
        object.__setattr__(self, "expiries", T)
        object.__setattr__(self, "theta", th)

    # -- term structure ---------------------------------------------------
    def theta_at(self, T) -> np.ndarray:
        """ATM total variance at T (linear in T through (0, 0) and the
        expiries; flat ATM vol, i.e. theta proportional to T, beyond)."""

        T = np.asarray(T, dtype=float)
        taus, th = np.r_[0.0, self.expiries], np.r_[0.0, self.theta]
        inside = np.interp(T, taus, th)
        beyond = self.theta[-1] * T / self.expiries[-1]
        return np.where(T > self.expiries[-1], beyond, inside)

    def dtheta_dT(self, T) -> np.ndarray:
        """Right derivative of theta(T)."""

        T = np.asarray(T, dtype=float)
        taus, th = np.r_[0.0, self.expiries], np.r_[0.0, self.theta]
        slopes = np.diff(th) / np.diff(taus)
        j = np.clip(np.searchsorted(taus, T, side="right") - 1, 0, slopes.size - 1)
        return np.where(T >= self.expiries[-1], self.theta[-1] / self.expiries[-1], slopes[j])

    def phi(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return self.eta / (theta**self.gamma * (1.0 + theta) ** (1.0 - self.gamma))

    # -- surface ----------------------------------------------------------
    def _w_theta(self, k, theta) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        p = self.phi(theta)
        r = self.rho
        return 0.5 * theta * (1.0 + r * p * k + np.sqrt((p * k + r) ** 2 + 1.0 - r * r))

    def total_variance(self, k, T) -> np.ndarray:
        return self._w_theta(k, self.theta_at(T))

    def implied_vol(self, k, T) -> np.ndarray:
        """Black vol at forward log-moneyness k and T (> 0)."""

        T = np.asarray(T, dtype=float)
        if np.any(T <= 0):
            raise ValueError("T must be > 0")
        return np.sqrt(self.total_variance(k, T) / T)

    def slice_svi(self, T: float) -> Dict[str, float]:
        """Raw-SVI parameters of the slice at T (total variance)."""

        th = float(self.theta_at(T))
        p = float(self.phi(th))
        r = self.rho
        return {"a": 0.5 * th * (1 - r * r), "b": 0.5 * th * p, "rho": r, "m": -r / p, "sigma": np.sqrt(1 - r * r) / p}

    def dw_dT(self, k, T) -> np.ndarray:
        """dw/dT at fixed k: dw/dtheta * dtheta/dT (analytic in theta)."""

        k = np.asarray(k, dtype=float)
        th = self.theta_at(T)
        p = self.phi(th)
        dlogphi = -self.gamma / th + (self.gamma - 1.0) / (1.0 + th)
        dp = p * dlogphi
        r = self.rho
        S = np.sqrt((p * k + r) ** 2 + 1.0 - r * r)
        dw_dth = self._w_theta(k, th) / th + 0.5 * th * (r * k * dp + (p * k + r) * k * dp / S)
        return dw_dth * self.dtheta_dT(T)

    def local_vol(self, k, T) -> np.ndarray:
        """Dupire local vol at forward log-moneyness k and T (per-annum
        decimal). NaN where the density factor g <= 0 (not expected inside
        the arbitrage-free parameter set)."""

        k = np.asarray(k, dtype=float)
        g = butterfly_density(k, self.slice_svi(float(T)))
        num = self.dw_dT(k, T)
        with np.errstate(divide="ignore", invalid="ignore"):
            lv2 = np.where(g > 0, num / g, np.nan)
        return np.sqrt(np.where(lv2 >= 0, lv2, np.nan))

    # -- checks -----------------------------------------------------------
    def arbitrage_report(self, k_grid: Optional[Sequence[float]] = None) -> Dict[str, object]:
        """Parameter conditions plus numerical butterfly checks per expiry
        and calendar checks between expiries on ``k_grid``."""

        k = np.linspace(-2.0, 2.0, 801) if k_grid is None else np.asarray(k_grid, dtype=float)
        slices = [(float(T), self.slice_svi(float(T))) for T in self.expiries]
        bf = [check_butterfly(p, k) for _, p in slices]
        cal = check_calendar(slices, k)
        return {
            "eta_condition": float(self.eta * (1.0 + abs(self.rho))),
            "eta_condition_ok": bool(self.eta * (1.0 + abs(self.rho)) <= 2.0 + 1e-12),
            "gamma_ok": bool(0.0 < self.gamma <= 0.5),
            "theta_monotone": bool(np.all(np.diff(self.theta) >= -1e-14)),
            "butterfly_ok": all(b.ok for b in bf),
            "butterfly_min_g": float(min(b.min_g for b in bf)),
            "calendar_ok": all(c.ok for c in cal),
            "calendar_min_dw": float(min((c.min_dw for c in cal), default=np.nan)),
        }


@dataclass
class SsviFit:
    """Result of :func:`fit_ssvi`. RMSEs are decimal vols (x 1e4 = vol bp)."""

    surface: SsviSurface
    rmse: float
    rmse_by_expiry: List[float]
    n_points: int
    success: bool
    message: str
    at_bound: Dict[str, bool] = field(default_factory=dict)

    @property
    def rmse_bp(self) -> float:
        return self.rmse * 1e4


def fit_ssvi(
    slices: Sequence[Tuple[float, Sequence[float], Sequence[float]]],
    weights: Optional[Sequence[Sequence[float]]] = None,
    loss: str = "soft_l1",
    f_scale: float = 0.005,
) -> SsviFit:
    """Fit a power-law SSVI surface to several expiries at once.

    ``slices`` are ``(T, k, implied_vols)`` with ``k = ln(K/F(T))``;
    ``weights`` optionally one array per slice. Residuals are in implied vol
    (decimal), with a soft-L1 loss at ``f_scale`` (50 vol bp). The fitted
    surface satisfies the no-arbitrage conditions of the module docs by
    construction. Needs at least 2 quotes per expiry and more quotes than
    parameters (n_expiries + 3)."""

    data = sorted(((float(T), np.asarray(k, float), np.asarray(v, float)) for T, k, v in slices), key=lambda s: s[0])
    if not data:
        raise ValueError("fit_ssvi needs at least one expiry")
    Ts = np.array([d[0] for d in data])
    if np.any(Ts <= 0) or np.any(np.diff(Ts) <= 0):
        raise ValueError("expiries must be positive and distinct")
    ks = [d[1] for d in data]
    for T, k, v in data:
        finite = np.all(np.isfinite(k)) and np.all(np.isfinite(v)) and np.all(v > 0)
        if k.shape != v.shape or k.size < 2 or not finite:
            raise ValueError(f"expiry T={T:.4f}: need >= 2 finite (k, vol > 0) quotes of equal length")
    n_total = sum(k.size for k in ks)
    n_par = len(data) + 3
    if n_total <= n_par:
        raise ValueError(f"{n_total} quotes for {n_par} SSVI parameters")
    if weights is None:
        wts = [np.ones(k.size) for k in ks]
    else:
        wts = [np.asarray(w, float) for w in weights]
        if len(wts) != len(data) or any(w.shape != k.shape for w, k in zip(wts, ks)):
            raise ValueError("weights must give one array per expiry, one weight per quote")
    allw = np.concatenate(wts)
    wts = [w / allw.mean() for w in wts]
    theta0 = np.array([np.interp(0.0, k[np.argsort(k)], v[np.argsort(k)]) ** 2 * T for (T, k, v) in data])
    theta0 = np.maximum.accumulate(np.maximum(theta0, 1e-8))
    inc0 = np.diff(np.r_[0.0, theta0])
    wmax = max(float(np.max(v**2 * T)) for (T, _, v) in data)
    n = len(data)
    lb = np.r_[1e-10, np.zeros(n - 1), -0.999, 1e-4, 0.01]
    ub = np.r_[np.full(n, 4.0 * wmax + 1e-8), 0.999, 1.0, 0.5]

    def build(x: np.ndarray) -> SsviSurface:
        theta = np.cumsum(x[:n])
        rho, u, gamma = x[n], x[n + 1], x[n + 2]
        return SsviSurface(Ts, theta, rho, u * 2.0 / (1.0 + abs(rho)), gamma)

    def resid(x: np.ndarray) -> np.ndarray:
        try:
            s = build(x)
        except ValueError:
            return np.full(n_total, 1.0)
        out = [w * (np.sqrt(s.total_variance(k, T) / T) - v) for (T, k, v), w in zip(data, wts)]
        r = np.concatenate(out)
        return np.where(np.isfinite(r), r, 1.0)

    best = None
    for rho0 in (-0.6, -0.2, 0.3):
        for u0 in (0.3, 0.8):
            x0 = np.clip(np.r_[inc0, rho0, u0, 0.4], lb + 1e-12, ub - 1e-12)
            res = least_squares(resid, x0, bounds=(lb, ub), loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=4000)
            if best is None or res.cost < best.cost:
                best = res
    surface = build(best.x)
    per = []
    sq = []
    for T, k, v in data:
        e = np.sqrt(surface.total_variance(k, T) / T) - v
        per.append(float(np.sqrt(np.mean(e**2))))
        sq.append(e**2)
    names = ["rho", "u", "gamma"]
    at_bound = {}
    for i, name in enumerate(names):
        j = n + i
        width = ub[j] - lb[j]
        at_bound[name] = bool(best.x[j] - lb[j] <= 1e-3 * width or ub[j] - best.x[j] <= 1e-3 * width)
    return SsviFit(
        surface=surface,
        rmse=float(np.sqrt(np.mean(np.concatenate(sq)))),
        rmse_by_expiry=per,
        n_points=int(n_total),
        success=bool(best.success),
        message=str(best.message),
        at_bound=at_bound,
    )
