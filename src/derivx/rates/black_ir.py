from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy.stats import norm

from .term_structure import ZeroCurve


def price_caplet_black(F: float, K: float, T: float, sigma: float, df: float) -> float:
    """Black caplet price (per unit notional)."""

    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(df * (F * norm.cdf(d1) - K * norm.cdf(d2)))


def price_cap_floor(
    F_curve: ZeroCurve,
    K: float,
    tenors: Iterable[float],
    vol_curve: ZeroCurve,
) -> Dict[str, object]:
    """Price a cap (sum of caplets) under Black; return PV, DV01, PV01, buckets."""

    ts = np.asarray(list(tenors), dtype=float)
    ts = ts[ts > 0]
    dfs = F_curve.df(ts)
    fwds = np.array([F_curve.fwd_rate(t - 1e-6, t) for t in ts])
    sigmas = vol_curve.df(ts)  # using df to hold sigma values along ts

    caplets = np.array([
        price_caplet_black(fwds[i], K, ts[i], sigmas[i], dfs[i]) for i in range(len(ts))
    ])
    pv = float(np.sum(caplets))

    # DV01/PV01: approximate using 1bp parallel bump on rates (df adjustment)
    bump = 1e-4
    dfs_bumped = dfs * np.exp(-bump * ts)
    caplets_bumped = np.array([
        price_caplet_black(fwds[i], K, ts[i], sigmas[i], dfs_bumped[i])
        for i in range(len(ts))
    ])
    pv_bumped = float(np.sum(caplets_bumped))
    dv01 = (pv_bumped - pv) / bump
    pv01 = dv01  # same definition under parallel bump for price per 1bp

    buckets = {float(t): float(c) for t, c in zip(ts, caplets)}
    return {"PV": pv, "DV01": dv01, "PV01": pv01, "buckets": buckets}


