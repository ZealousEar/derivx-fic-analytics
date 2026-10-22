"""Legacy Black-76 caplet helpers. New code uses :mod:`derivx.rates.capfloor`
(schedules, accruals, Black / shifted Black / Bachelier, risk)."""

from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable

import numpy as np

from .term_structure import ZeroCurve


def price_caplet_black(F: float, K: float, T: float, sigma: float, df: float) -> float:
    """Black-76 caplet per unit notional and unit accrual fraction::

        df * (F N(d1) - K N(d2)),   d1,2 = (ln(F/K) +- sigma^2 T / 2) / (sigma sqrt(T))

    ``F`` (forward) and ``K`` (strike) are decimal rates > 0, ``T`` years
    from today to the fixing, ``sigma`` the lognormal (Black) vol as a
    decimal (0.20 = 20%), ``df`` the discount factor to the payment date.
    Multiply by notional x accrual fraction for a premium in currency. At
    ``T = 0`` or ``sigma = 0`` the value is the discounted intrinsic
    ``df * max(F - K, 0)``. Raises ``ValueError`` for ``F <= 0`` or ``K <= 0``
    (use the Bachelier or shifted Black models of :mod:`derivx.rates.capfloor`),
    ``T < 0``, ``sigma < 0`` or ``df <= 0``.
    """

    from .capfloor import black_price

    if not (math.isfinite(df) and df > 0.0):
        raise ValueError(f"df must be a positive discount factor, got {df!r}")
    if not math.isfinite(T) or T < 0.0:
        raise ValueError(f"T must be >= 0 years to the fixing, got {T!r}")
    return float(df * black_price(float(F), float(K), float(T), float(sigma), True))


def price_cap_floor(
    F_curve: ZeroCurve,
    K: float,
    tenors: Iterable[float],
    vol_curve: ZeroCurve,
) -> Dict[str, object]:
    """Deprecated approximation kept for old notebooks; use
    :class:`derivx.rates.capfloor.CapFloor`.

    It prices one caplet per entry of ``tenors`` on the instantaneous forward
    with no accrual fraction, and its DV01 freezes forwards and is per unit of
    rate (not per bp), so its numbers do not match a real cap.
    """

    warnings.warn(
        "price_cap_floor is a deprecated approximation; use derivx.rates.capfloor.CapFloor",
        DeprecationWarning,
        stacklevel=2,
    )

    ts = np.asarray(list(tenors), dtype=float)
    ts = ts[ts > 0]
    dfs = F_curve.df(ts)
    fwds = np.array([F_curve.fwd_rate(t - 1e-6, t) for t in ts])
    sigmas = vol_curve.df(ts)  # using df to hold sigma values along ts

    caplets = np.array([price_caplet_black(fwds[i], K, ts[i], sigmas[i], dfs[i]) for i in range(len(ts))])
    pv = float(np.sum(caplets))

    # DV01/PV01: approximate using 1bp parallel bump on rates (df adjustment)
    bump = 1e-4
    dfs_bumped = dfs * np.exp(-bump * ts)
    caplets_bumped = np.array([price_caplet_black(fwds[i], K, ts[i], sigmas[i], dfs_bumped[i]) for i in range(len(ts))])
    pv_bumped = float(np.sum(caplets_bumped))
    dv01 = (pv_bumped - pv) / bump
    pv01 = dv01  # same definition under parallel bump for price per 1bp

    buckets = {float(t): float(c) for t, c in zip(ts, caplets)}
    return {"PV": pv, "DV01": dv01, "PV01": pv01, "buckets": buckets}
