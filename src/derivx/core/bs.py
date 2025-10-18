from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm

from .common import EPS, as_array, clamp_positive, ensure_broadcastable


def bs_price_greeks(
    S, K, T, r, sigma, call: bool = True, q: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Black–Scholes price and Greeks (vectorized) with dividend/FX carry q.

    Parameters
    ----------
    S, K, T, r, sigma : array-like
        Spot, strike, time to maturity (years), risk-free rate, volatility.
        Can be broadcastable arrays.
    call : bool, optional
        True for call, False for put.
    q : float, optional
        Dividend yield / foreign rate.

    Returns
    -------
    price, delta, gamma, vega, theta, rho : np.ndarray
        Arrays broadcast to common shape.
    """

    S, K, T, r, sigma, q = ensure_broadcastable(S, K, T, r, sigma, q)

    S = clamp_positive(S)
    K = clamp_positive(K)
    T = clamp_positive(T)
    sigma = clamp_positive(sigma)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)

    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if call:
        price = S * disc_q * Nd1 - K * disc_r * Nd2
        delta = disc_q * Nd1
        theta = (
            -(S * disc_q * nd1 * sigma) / (2 * sqrtT)
            - r * K * disc_r * Nd2
            + q * S * disc_q * Nd1
        )
        rho = K * T * disc_r * Nd2
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        delta = disc_q * (Nd1 - 1.0)
        theta = (
            -(S * disc_q * nd1 * sigma) / (2 * sqrtT)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)

    gamma = (disc_q * nd1) / (S * sigma * sqrtT)
    vega = S * disc_q * nd1 * sqrtT

    return price, delta, gamma, vega, theta, rho


def put_call_parity_residual(
    call_price: np.ndarray,
    put_price: np.ndarray,
    S,
    K,
    T,
    r,
    q: float = 0.0,
) -> np.ndarray:
    """Return residual of put–call parity: C - P - (S e^{-qT} - K e^{-rT})."""

    S, K, T, r, q = ensure_broadcastable(S, K, T, r, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    return as_array(call_price) - as_array(put_price) - (S * disc_q - K * disc_r)


