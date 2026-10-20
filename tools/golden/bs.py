"""QuantLib reference values for the Black-Scholes-Merton core
(tests/test_bs.py).

* ``cases``: European calls and puts priced by ``ql.AnalyticEuropeanEngine``
  on a ``BlackScholesMertonProcess`` with flat continuously compounded
  ``r`` and ``q`` curves and a flat vol (ACT/365F, so ``T = days / 365``),
  from 1 day to 30 years, spot / strike from 0.6 to 1.5, vols from 5% to
  150%, negative rates and ``q`` above ``r``, at equity, index and FX levels.
  First-order Greeks are QuantLib's analytic ``delta``, ``gamma``, ``vega``,
  ``theta`` (per year), ``rho`` and ``dividendRho`` (per 1.00 of vol and
  rate). ``npv_1d`` is the NPV after moving the evaluation date one day
  (curves and vol anchored to it), i.e. the one-day theta by repricing
  (omitted for 1-day options, which QuantLib treats as expired after the
  move).
* Second-order Greeks are central finite differences of QuantLib's own
  analytic values from ``ql.BlackCalculator`` in continuous time:
  ``vanna = d delta / d sigma``, ``volga = d vega / d sigma`` and
  ``charm = -d delta / dT`` (calendar time passing), independent of
  derivx's formulas. (``BlackCalculator.vanna`` agrees with the difference;
  ``BlackCalculator.volga`` in QuantLib 1.44 does not: it is the difference
  divided by ``sqrt(T)``, so it is not used.)
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Tuple

import QuantLib as ql

from golden.common import qd, set_evaluation_date

EVAL = dt.date(2026, 10, 20)
DAYS = [1, 7, 30, 182, 365, 1825, 3650, 10950]
STRIKE_RATIOS = [0.6, 0.9, 1.0, 1.1, 1.5]
# (r, q, sigma): cycled over the (strike, expiry) grid, two per cell
MARKETS: List[Tuple[float, float, float]] = [
    (0.05, 0.02, 0.20),
    (0.03, 0.08, 0.35),
    (-0.01, 0.0, 0.05),
    (0.045, 0.015, 0.60),
    (0.02, 0.04, 1.50),
    (0.0, 0.03, 0.12),
]
LEVELS = [
    # spot, strike ratios, days, (r, q, sigma) list: FX and index levels
    (1.08, [0.95, 1.0, 1.04], [7, 91, 365], [(0.04, 0.025, 0.08), (0.01, 0.045, 0.12)]),
    (5800.0, [0.8, 1.0, 1.1], [30, 365, 1825], [(0.042, 0.013, 0.18), (0.042, 0.013, 0.30)]),
]


def _flat(rate: float) -> ql.YieldTermStructureHandle:
    return ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), rate, ql.Actual365Fixed(), ql.Continuous))


def _option(S: float, K: float, days: int, r: float, q: float, vol: float, call: bool) -> ql.VanillaOption:
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        _flat(q),
        _flat(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.NullCalendar(), vol, ql.Actual365Fixed())),
    )
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, K)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(qd(EVAL + dt.timedelta(days=days))))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


def _calc(S: float, K: float, T: float, r: float, q: float, vol: float, call: bool) -> ql.BlackCalculator:
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, K)
    return ql.BlackCalculator(payoff, S * math.exp((r - q) * T), vol * math.sqrt(T), math.exp(-r * T))


def _second_order(S: float, K: float, T: float, r: float, q: float, vol: float, call: bool) -> Dict[str, float]:
    hv, ht = 1e-5, min(1e-6, T / 10.0)
    up, dn = _calc(S, K, T, r, q, vol + hv, call), _calc(S, K, T, r, q, vol - hv, call)
    later, earlier = _calc(S, K, T - ht, r, q, vol, call), _calc(S, K, T + ht, r, q, vol, call)
    return {
        "vanna": (up.delta(S) - dn.delta(S)) / (2 * hv),
        "volga": (up.vega(T) - dn.vega(T)) / (2 * hv),
        "charm": (later.delta(S) - earlier.delta(S)) / (2 * ht),
    }


def _case(S: float, K: float, days: int, r: float, q: float, vol: float, call: bool) -> Dict[str, object]:
    set_evaluation_date(EVAL)
    option = _option(S, K, days, r, q, vol, call)
    T = days / 365.0
    out: Dict[str, object] = {
        "S": S,
        "K": K,
        "days": days,
        "T": T,
        "r": r,
        "q": q,
        "sigma": vol,
        "call": call,
        "price": option.NPV(),
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega(),
        "theta": option.theta(),
        "rho": option.rho(),
        "phi": option.dividendRho(),
        **_second_order(S, K, T, r, q, vol, call),
    }
    if days > 1:
        npv = option.NPV()
        set_evaluation_date(EVAL + dt.timedelta(days=1))
        out["theta_1d"] = option.NPV() - npv
        set_evaluation_date(EVAL)
    return out


def _cases() -> List[Dict[str, object]]:
    out = []
    i = 0
    for ratio in STRIKE_RATIOS:
        for days in DAYS:
            for call in (True, False):
                for _ in range(2):
                    r, q, vol = MARKETS[i % len(MARKETS)]
                    i += 1
                    out.append(_case(100.0, 100.0 * ratio, days, r, q, vol, call))
    for spot, ratios, days_list, markets in LEVELS:
        for ratio in ratios:
            for days in days_list:
                for r, q, vol in markets:
                    for call in (True, False):
                        out.append(_case(spot, spot * ratio, days, r, q, vol, call))
    return out


def generate() -> Dict[str, object]:
    try:
        return {"evaluation_date": EVAL.isoformat(), "cases": _cases()}
    finally:
        set_evaluation_date(EVAL)
