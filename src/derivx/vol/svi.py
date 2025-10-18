from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy.optimize import least_squares


def _svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, s: float) -> np.ndarray:
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def fit_svi_strike_slice(Ks: Iterable[float], ivs: Iterable[float], F: float | None = None) -> Dict[str, float]:
    """Fit SVI params to a single-maturity slice.

    Parameters
    ----------
    Ks : iterable
        Strikes
    ivs : iterable
        Black implied vols (per annum)
    F : float, optional
        Forward; if None, uses mean(Ks)
    """

    Ks = np.asarray(Ks, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    if F is None:
        F = float(np.mean(Ks))
    k = np.log(Ks / F)
    w_obs = (ivs**2)  # total variance at T=1 for slice fit baseline

    # Initial guess and bounds
    x0 = np.array([1e-4, 0.1, 0.0, 0.0, 0.1])  # a, b, rho, m, s
    lb = np.array([0.0, 1e-6, -0.999, -5.0, 1e-6])
    ub = np.array([2.0, 5.0, 0.999, 5.0, 2.0])

    def resid(x):
        a, b, rho, m, s = x
        return _svi_total_variance(k, a, b, rho, m, s) - w_obs

    res = least_squares(resid, x0=x0, bounds=(lb, ub), loss="huber")
    a, b, rho, m, s = res.x
    return {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "s": float(s)}


def svi_iv(K: float, params: Dict[str, float], F: float | None = None) -> float:
    """Return per‑annum IV from SVI params at unit maturity baseline."""

    if F is None:
        F = K
    k = np.log(K / F)
    w = _svi_total_variance(k, params["a"], params["b"], params["rho"], params["m"], params["s"])  # noqa: E501
    return float(np.sqrt(max(w, 1e-12)))


def basic_no_arb_checks(params_over_maturities: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Basic SVI diagnostics: enforce simple parameter sanity.

    Returns simple metrics that can be displayed in UI.
    """

    viol_b = 0
    viol_rho = 0
    for p in params_over_maturities:
        if p["b"] <= 0:
            viol_b += 1
        if abs(p["rho"]) >= 1:
            viol_rho += 1
    return {"viol_b": float(viol_b), "viol_rho": float(viol_rho)}


