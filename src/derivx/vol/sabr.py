from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import least_squares


def hagan_sabr_iv(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
) -> float:
    """Hagan 2002 SABR lognormal vol (with optional shift)."""

    Fs = F + shift
    Ks = K + shift
    if T <= 0 or alpha <= 0:
        return 0.0
    if np.isclose(Fs, Ks):
        term = (1 - beta) ** 2 / 24 * (alpha**2) / (Fs ** (2 - 2 * beta))
        term += 0.25 * rho * beta * nu * alpha / (Fs ** (1 - beta))
        term += (2 - 3 * rho**2) / 24 * (nu**2)
        return float(alpha / (Fs ** (1 - beta)) * (1 + term * T))

    z = (nu / alpha) * (Fs ** (1 - beta) - Ks ** (1 - beta)) / (1 - beta)
    xz = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    denom = (Fs * Ks) ** ((1 - beta) / 2)
    pre = alpha / denom
    chi = 1 + (((1 - beta) ** 2) / 24) * (np.log(Fs / Ks)) ** 2 + (((1 - beta) ** 4) / 1920) * (np.log(Fs / Ks)) ** 4  # noqa: E501
    vol = pre * (z / xz) * (1 + (((1 - beta) ** 2) / 24) * (alpha**2) / (denom**2) + (rho * beta * nu * alpha) / (4 * denom) + ((2 - 3 * rho**2) / 24) * (nu**2))  # noqa: E501
    return float(vol)


def calibrate_sabr_smile(
    Ks: np.ndarray,
    ivs: np.ndarray,
    T: float,
    F: float,
    beta: float = 0.5,
    shift: float = 0.0,
) -> Dict[str, float]:
    """Calibrate SABR by least squares to a single maturity smile."""

    Ks = np.asarray(Ks, dtype=float)
    ivs = np.asarray(ivs, dtype=float)

    x0 = np.array([0.2, 0.0, 0.5])  # alpha, rho, nu
    lb = np.array([1e-6, -0.999, 1e-6])
    ub = np.array([5.0, 0.999, 5.0])

    def resid(x):
        a, r, n = x
        model = np.array([hagan_sabr_iv(F, k, T, a, beta, r, n, shift) for k in Ks])
        return model - ivs

    res = least_squares(resid, x0=x0, bounds=(lb, ub), loss="huber")
    a, r, n = res.x
    return {"alpha": float(a), "beta": float(beta), "rho": float(r), "nu": float(n), "shift": float(shift)}  # noqa: E501


