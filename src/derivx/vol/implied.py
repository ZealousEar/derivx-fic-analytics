from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import brentq

from ..core.bs import bs_price_greeks
from ..core.common import as_array, clamp_positive


def _price_minus_mkt(
    sigma: float, price: float, S: float, K: float, T: float, r: float, call: bool, q: float
) -> float:
    p, *_ = bs_price_greeks(S, K, T, r, sigma, call=call, q=q)
    return float(p - price)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    call: bool = True,
    q: float = 0.0,
    bracket: Tuple[float, float] = (1e-6, 5.0),
    tol: float = 1e-10,
    maxiter: int = 100,
) -> float:
    """Implied volatility via brentq with safe brackets; returns np.nan on fail."""

    S = float(S); K = float(K); T = max(float(T), 1e-12); r = float(r)
    price = float(price)
    a, b = bracket
    try:
        return brentq(
            _price_minus_mkt, a, b,
            args=(price, S, K, T, r, call, q), xtol=tol, maxiter=maxiter
        )
    except Exception:
        return float("nan")


def batch_implied_vols(
    prices, S, Ks, T, r, call: bool = True, q: float = 0.0
) -> np.ndarray:
    """Vectorized implied vols for a chain; invalid quotes -> np.nan."""

    Ks = as_array(Ks)
    out = np.empty_like(Ks, dtype=float)
    for i, K in enumerate(Ks):
        out[i] = implied_vol(prices[i], S, K, T, r, call=call, q=q)
    return out


