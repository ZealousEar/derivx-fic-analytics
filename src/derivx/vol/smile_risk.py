"""Smile shape and smile-consistent risk.

* :func:`svi_atm_skew_curvature`: ATM vol, skew ``d sigma / dk`` and
  curvature ``d2 sigma / dk2`` at k = ln(K/F) = 0 of an SVI slice, and
  :func:`smile_term_structure` for a set of calibrated slices.
* :func:`smile_deltas`: Black-76 forward delta under two smile dynamics:
  sticky strike (the vol at a fixed strike does not move when F moves) and
  sticky moneyness (the vol at fixed k = ln(K/F) does not move; to first
  order this is also 'sticky delta'):
  ``delta_sticky_moneyness = delta_sticky_strike - vega * (d sigma/dk) / F``.
* :func:`sabr_deltas`: SABR deltas. Hagan's ``delta = delta_BS + vega
  d sigma/dF`` (alpha held fixed) and Bartlett's (2006) correction, which
  moves alpha with F through the correlation: ``+ vega (d sigma/d alpha)
  rho nu / (F + s)^beta``.

Units: ``vega`` per 1.00 of vol, deltas per unit of forward notional
(discounted by ``df``); skew per unit of log-moneyness (a skew of -0.1 is
-1 vol point per 10% of log-moneyness), curvature per unit k^2.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from derivx.vol.sabr import hagan_sabr_iv
from derivx.vol.svi import svi_derivatives


def svi_atm_skew_curvature(params: Mapping, T: float) -> Dict[str, float]:
    """ATM vol, skew and curvature in k of an SVI total-variance slice."""

    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be > 0 years")
    w, w1, w2 = (float(x) for x in svi_derivatives(0.0, params))
    if w <= 0:
        raise ValueError("ATM total variance is zero: skew is undefined")
    vol = np.sqrt(w / T)
    skew = w1 / (2.0 * np.sqrt(w * T))
    curvature = w2 / (2.0 * np.sqrt(w * T)) - w1 * w1 / (4.0 * w * np.sqrt(w * T))
    return {"atm_vol": float(vol), "skew": float(skew), "curvature": float(curvature)}


def smile_term_structure(slices: Iterable) -> pd.DataFrame:
    """ATM vol, skew and curvature by expiry for calibrated slices (objects
    with ``expiry``, ``tau`` and SVI ``params``, e.g. ``CalibratedSlice``).
    ``skew_x_sqrtT`` (= skew sqrt(T)) is the scale-free skew: roughly
    constant across expiries for a stable smile."""

    rows = []
    for s in slices:
        d = svi_atm_skew_curvature(s.params, s.tau)
        rows.append({"expiry": s.expiry, "tau": s.tau, **d, "skew_x_sqrtT": d["skew"] * np.sqrt(s.tau)})
    return pd.DataFrame(rows).sort_values("tau").reset_index(drop=True) if rows else pd.DataFrame(rows)


def _black(F, K, T, vol, df, is_call):
    F, K, T, vol = (np.asarray(x, dtype=float) for x in (F, K, T, vol))
    sd = vol * np.sqrt(T)
    d1 = np.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    call_delta = df * norm.cdf(d1)
    delta = call_delta if is_call else call_delta - df
    vega = df * F * norm.pdf(d1) * np.sqrt(T)
    price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if not is_call:
        price = price - df * (F - K)
    return price, delta, vega


def smile_deltas(F, K, T, vol, dvol_dk, df: float = 1.0, is_call: bool = True) -> Dict[str, np.ndarray]:
    """Sticky-strike and sticky-moneyness forward deltas (see module docs).
    ``dvol_dk`` is the smile slope d sigma / d ln(K/F) at K."""

    F, K, T, vol, dvol_dk = (np.asarray(x, dtype=float) for x in (F, K, T, vol, dvol_dk))
    if np.any(F <= 0) or np.any(K <= 0) or np.any(T <= 0) or np.any(vol <= 0):
        raise ValueError("deltas need F > 0, K > 0, T > 0 and vol > 0")
    _, delta, vega = _black(F, K, T, vol, df, is_call)
    return {"sticky_strike": delta, "sticky_moneyness": delta - vega * dvol_dk / F, "vega": vega}


def sabr_deltas(
    F: float,
    K,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
    df: float = 1.0,
    is_call: bool = True,
) -> Dict[str, np.ndarray]:
    """Black (shifted) delta at the SABR vol, Hagan's SABR delta and
    Bartlett's delta, by central differences of the Hagan vol. Lognormal
    SABR (Black vol of F + shift)."""

    K = np.asarray(K, dtype=float)
    f = F + shift
    vol = np.asarray(hagan_sabr_iv(F, K, T, alpha, beta, rho, nu, shift), dtype=float)
    _, delta, vega = _black(f, K + shift, T, vol, df, is_call)
    hF = 1e-5 * f
    ha = 1e-5 * alpha
    dvol_dF = (
        np.asarray(hagan_sabr_iv(F + hF, K, T, alpha, beta, rho, nu, shift))
        - np.asarray(hagan_sabr_iv(F - hF, K, T, alpha, beta, rho, nu, shift))
    ) / (2 * hF)
    dvol_da = (
        np.asarray(hagan_sabr_iv(F, K, T, alpha + ha, beta, rho, nu, shift))
        - np.asarray(hagan_sabr_iv(F, K, T, alpha - ha, beta, rho, nu, shift))
    ) / (2 * ha)
    hagan = delta + vega * dvol_dF
    bartlett = hagan + vega * dvol_da * rho * nu / f**beta
    return {"black": delta, "hagan": hagan, "bartlett": bartlett, "vol": vol, "vega": vega}
