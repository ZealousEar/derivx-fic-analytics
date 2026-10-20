"""QuantLib reference values for the vol library (tests/test_sabr.py,
tests/test_implied.py, tests/test_svi.py, tests/test_term_structure.py).

* ``sabr``: ``ql.sabrVolatility`` / ``ql.shiftedSabrVolatility`` in
  lognormal (``ShiftedLognormal``) and ``Normal`` vol, on strike grids from
  0.3F to 3F including strikes 1e-6 away from the forward, for equity, FX
  and rates levels, beta in {0, 0.5, 0.7, 1}, T from 1 week to 10 years,
  and negative forwards with a shift.
* ``black_implied`` / ``bachelier_implied``: prices from ``ql.blackFormula``
  / ``ql.bachelierBlackFormula`` and QuantLib's inversions
  (``blackFormulaImpliedStdDev`` / ``bachelierBlackFormulaImpliedVol``),
  calls and puts, ITM and OTM, discounted, shifted, vols up to 300%.
* ``svi``: ``ql.SviSmileSection`` vols and the sign of its numerical density
  for Axel Vogt's arbitrageable slice (Gatheral & Jacquier 2014, Example
  3.1) and an arbitrage-free slice.
* ``greeks``: Black-76 vanna, volga and charm by central finite differences
  of ``ql.blackFormula`` (independent of derivx's analytic formulas).
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import QuantLib as ql

SABR_CASES = [
    # F, T, alpha, beta, nu, rho, shift
    (0.03, 5.0, 0.05, 0.5, 0.4, -0.3, 0.0),
    (0.03, 10.0, 0.008, 0.0, 0.3, 0.2, 0.0),
    (0.03, 1.0, 0.0085, 0.0, 0.5, -0.1, 0.0),
    (-0.002, 2.0, 0.01, 0.5, 0.5, 0.1, 0.02),
    (0.001, 3.0, 0.005, 0.3, 0.35, 0.25, 0.03),
    (100.0, 0.25, 2.0, 0.5, 1.0, -0.6, 0.0),
    (100.0, 1.0, 0.2, 1.0, 0.6, -0.3, 0.0),
    (1.08, 7.0 / 365.0, 0.1, 1.0, 2.0, -0.2, 0.0),
    (150.0, 0.5, 0.15 * 150.0**0.3, 0.7, 0.8, 0.1, 0.0),
    (5800.0, 1.0, 0.16, 1.0, 0.8, -0.7, 0.0),
]
STRIKE_RATIOS = [0.3, 0.5, 0.8, 0.95, 0.99, 1.0 - 1e-6, 1.0, 1.0 + 1e-6, 1.01, 1.05, 1.2, 1.5, 2.0, 3.0]


def _sabr() -> List[Dict[str, object]]:
    out = []
    for F, T, alpha, beta, nu, rho, shift in SABR_CASES:
        strikes = [(F + shift) * r - shift for r in STRIKE_RATIOS]
        for vol_type, qvt in (("lognormal", ql.ShiftedLognormal), ("normal", ql.Normal)):
            if shift:
                vols = [ql.shiftedSabrVolatility(k, F, T, alpha, beta, nu, rho, shift, qvt) for k in strikes]
            else:
                vols = [ql.sabrVolatility(k, F, T, alpha, beta, nu, rho, qvt) for k in strikes]
            out.append(
                {
                    "F": F,
                    "T": T,
                    "alpha": alpha,
                    "beta": beta,
                    "nu": nu,
                    "rho": rho,
                    "shift": shift,
                    "vol_type": vol_type,
                    "strikes": strikes,
                    "vols": vols,
                }
            )
    return out


BLACK_CASES = [
    # F, K, T, vol, df, shift
    (100.0, 120.0, 1.0, 0.25, 0.97, 0.0),
    (100.0, 80.0, 0.5, 0.2, 0.99, 0.0),
    (100.0, 100.0, 2.0, 0.6, 0.95, 0.0),
    (100.0, 150.0, 0.1, 3.0, 1.0, 0.0),
    (1.08, 1.2, 0.25, 0.1, 0.99, 0.0),
    (0.03, 0.02, 5.0, 0.3, 0.85, 0.0),
    (-0.002, 0.001, 2.0, 0.15, 0.98, 0.02),
    (5800.0, 4500.0, 1.0, 0.3, 0.96, 0.0),
]


def _black() -> List[Dict[str, object]]:
    out = []
    for F, K, T, vol, df, shift in BLACK_CASES:
        for is_call in (True, False):
            t = ql.Option.Call if is_call else ql.Option.Put
            std = vol * math.sqrt(T)
            price = ql.blackFormula(t, K, F, std, df, shift)
            ql_std = ql.blackFormulaImpliedStdDev(t, K, F, price, df, shift, ql.nullDouble(), 1e-14, 1000)
            out.append(
                {
                    "F": F,
                    "K": K,
                    "T": T,
                    "vol": vol,
                    "df": df,
                    "shift": shift,
                    "is_call": is_call,
                    "price": price,
                    "ql_vol": ql_std / math.sqrt(T),
                }
            )
    return out


BACHELIER_CASES = [
    # F, K, T, normal vol, df
    (0.03, 0.04, 1.0, 0.008, 0.97),
    (0.03, 0.03, 5.0, 0.0085, 0.85),
    (-0.002, 0.001, 2.0, 0.006, 0.99),
    (-0.005, -0.01, 0.5, 0.004, 1.0),
    (100.0, 90.0, 0.5, 20.0, 0.98),
]


def _bachelier() -> List[Dict[str, object]]:
    out = []
    for F, K, T, vol, df in BACHELIER_CASES:
        for is_call in (True, False):
            t = ql.Option.Call if is_call else ql.Option.Put
            price = ql.bachelierBlackFormula(t, K, F, vol * math.sqrt(T), df)
            ql_vol = ql.bachelierBlackFormulaImpliedVol(t, K, F, T, price, df)
            out.append(
                {"F": F, "K": K, "T": T, "vol": vol, "df": df, "is_call": is_call, "price": price, "ql_vol": ql_vol}
            )
    return out


SVI_CASES = {
    # Gatheral & Jacquier (2014), Example 3.1 (Axel Vogt): butterfly arbitrage
    "vogt": {"a": -0.0410, "b": 0.1331, "rho": 0.3060, "m": 0.3586, "sigma": 0.4153},
    "clean": {"a": 0.04, "b": 0.4, "rho": -0.4, "m": 0.0, "sigma": 0.2},
}


def _svi() -> List[Dict[str, object]]:
    out = []
    F, T = 1.0, 1.0
    ks = list(np.linspace(-1.5, 1.5, 31))
    for name, p in SVI_CASES.items():
        sec = ql.SviSmileSection(T, F, [p["a"], p["b"], p["sigma"], p["rho"], p["m"]])
        strikes = [F * math.exp(k) for k in ks]
        vols = [sec.volatility(k) for k in strikes]
        density = [sec.density(k, 1.0, 1e-4) for k in strikes]
        out.append({"name": name, "params": p, "F": F, "T": T, "strikes": strikes, "vols": vols, "density": density})
    return out


GREEK_CASES = [
    # F, K, T, vol, df
    (100.0, 80.0, 0.5, 0.2, 0.98),
    (100.0, 100.0, 0.5, 0.2, 0.98),
    (100.0, 120.0, 0.5, 0.2, 0.98),
    (1.08, 1.12, 0.25, 0.09, 0.995),
    (5800.0, 5200.0, 1.0, 0.22, 0.96),
]


def _greeks() -> List[Dict[str, object]]:
    """Central differences of QuantLib Black prices. The discount factor
    is ``exp(-r T)`` with r fixed, so charm moves df with T."""

    out = []
    for F, K, T, vol, df in GREEK_CASES:
        r = -math.log(df) / T

        def price(f=F, s=vol, t=T):
            return ql.blackFormula(ql.Option.Call, K, f, s * math.sqrt(t), math.exp(-r * t))

        hF, hs, ht = 1e-4 * F, 1e-4, 1e-5
        delta = (price(f=F + hF) - price(f=F - hF)) / (2 * hF)
        vega = (price(s=vol + hs) - price(s=vol - hs)) / (2 * hs)
        volga = (price(s=vol + hs) - 2 * price() + price(s=vol - hs)) / hs**2
        vanna = (
            price(f=F + hF, s=vol + hs)
            - price(f=F + hF, s=vol - hs)
            - price(f=F - hF, s=vol + hs)
            + price(f=F - hF, s=vol - hs)
        ) / (4 * hF * hs)

        def delta_at(t):
            return (price(f=F + hF, t=t) - price(f=F - hF, t=t)) / (2 * hF)

        charm = -(delta_at(T + ht) - delta_at(T - ht)) / (2 * ht)  # calendar time passing: T falls
        out.append(
            {
                "F": F,
                "K": K,
                "T": T,
                "vol": vol,
                "df": df,
                "price": price(),
                "delta": delta,
                "vega": vega,
                "volga": volga,
                "vanna": vanna,
                "charm": charm,
            }
        )
    return out


def generate() -> Dict[str, object]:
    return {
        "sabr": _sabr(),
        "black_implied": _black(),
        "bachelier_implied": _bachelier(),
        "svi": _svi(),
        "greeks": _greeks(),
    }
