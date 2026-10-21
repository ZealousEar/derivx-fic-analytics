"""QuantLib references for the forward swap rates the realised-vs-implied
analytic reads off a curve history (tests/test_realised_implied.py).

Each case is one curve on 2026-01-15 with 30/360 (bond basis) times and a
null calendar, so every tenor is an exact number of months/12 (the year
fractions :mod:`derivx.validation.realised_implied` uses):

* ``par_semi``: deposits at 1M and 3M plus par bonds with semi-annual
  coupons at every 6M to 10Y (``ql.FixedRateBondHelper`` priced at 100),
  bootstrapped by ``ql.PiecewiseLogLinearDiscount``: the par-yield
  bootstrap of a Treasury curve.
* ``par_annual``: deposits at 3M and 6M plus annual par bonds 1Y..10Y:
  an OIS-style swap curve (zero-coupon below 1Y).
* ``zero``: continuously compounded spot rates, ``ql.ZeroCurve`` linear in
  the zero rate, flat before the first maturity (the reference date
  carries the first rate).

Par quotes sit on every coupon date, so QuantLib's bootstrap and derivx's
(which interpolates par rates between quoted maturities) solve the same
equations. Per case: discount factors at a set of times, and forward swap
rates ``E`` into ``T``: from QuantLib's discount factors on QuantLib's own
backward schedule (``ql.Schedule``, short first period), checked against
``ql.VanillaSwap.fairRate`` with a matching Ibor leg.

The curves are synthetic test inputs (smooth shapes), not market data.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Tuple

import QuantLib as ql

from .common import set_evaluation_date

ASOF = (2026, 1, 15)
DC = ql.Thirty360(ql.Thirty360.BondBasis)
CAL = ql.NullCalendar()


def _ref() -> ql.Date:
    return ql.Date(ASOF[2], ASOF[1], ASOF[0])


def _months(years: float) -> int:
    m = round(years * 12)
    assert abs(m / 12 - years) < 1e-12, years
    return int(m)


def _label(months: int) -> str:
    return f"{months // 12}Y" if months % 12 == 0 else f"{months}M"


def _par_rate(t: float) -> float:
    return 0.036 + 0.009 * (1 - math.exp(-t / 2.5)) - 0.0004 * t


def _zero_rate(t: float) -> float:
    return 0.031 + 0.012 * (1 - math.exp(-t / 3.0)) + 0.002 * math.sin(t / 4.0)


def _par_curve(money: List[int], coupon_months: int, last_months: int):
    ref = _ref()
    helpers, tenors, rates = [], [], []
    for m in money:
        r = _par_rate(m / 12)
        helpers.append(ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(r)), ql.Period(m, ql.Months), 0, CAL,
                                            ql.Unadjusted, False, DC))  # fmt: skip
        tenors.append(_label(m))
        rates.append(r)
    for m in range(coupon_months, last_months + 1, coupon_months):
        r = _par_rate(m / 12)
        schedule = ql.Schedule(ref, ref + ql.Period(m, ql.Months), ql.Period(coupon_months, ql.Months), CAL,
                               ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)  # fmt: skip
        helpers.append(ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(100.0)), 0, 100.0, schedule, [r], DC,
                                              ql.Unadjusted))  # fmt: skip
        tenors.append(_label(m))
        rates.append(r)
    curve = ql.PiecewiseLogLinearDiscount(ref, helpers, DC)
    return curve, tenors, rates


def _zero_curve(months: List[int]):
    ref = _ref()
    rates = [_zero_rate(m / 12) for m in months]
    dates = [ref] + [ref + ql.Period(m, ql.Months) for m in months]
    curve = ql.ZeroCurve(dates, [rates[0]] + rates, DC, CAL, ql.Linear(), ql.Continuous, ql.Annual)
    return curve, [_label(m) for m in months], rates


def _forward(curve, expiry: float, tenor: float, frequency: int) -> Tuple[float, float]:
    """(forward swap rate from QuantLib discount factors on QuantLib's
    schedule, VanillaSwap.fairRate with a matching Ibor leg)."""

    ref = _ref()
    start, end = ref + ql.Period(_months(expiry), ql.Months), ref + ql.Period(_months(expiry + tenor), ql.Months)
    period = ql.Period(12 // frequency, ql.Months)
    if tenor * frequency < 1 - 1e-12:
        period = ql.Period(_months(tenor), ql.Months)
    schedule = ql.Schedule(start, end, period, CAL, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    dates = list(schedule)
    annuity = sum(DC.yearFraction(a, b) * curve.discount(b) for a, b in zip(dates[:-1], dates[1:]))
    rate = (curve.discount(start) - curve.discount(end)) / annuity
    handle = ql.YieldTermStructureHandle(curve)
    index = ql.IborIndex("TEST", period, 0, ql.USDCurrency(), CAL, ql.Unadjusted, False, DC, handle)
    swap = ql.VanillaSwap(ql.VanillaSwap.Payer, 1.0, schedule, rate, DC, schedule, index, 0.0, DC)
    swap.setPricingEngine(ql.DiscountingSwapEngine(handle))
    return rate, swap.fairRate()


def _case(name: str, kind: str, frequency: int, curve, tenors, rates, times, points) -> Dict[str, object]:
    ref = _ref()
    dfs = [curve.discount(ref + ql.Period(_months(t), ql.Months)) for t in times]
    forwards = []
    for expiry, tenor in points:
        rate, fair = _forward(curve, expiry, tenor, frequency)
        assert abs(rate - fair) < 1e-12, (name, expiry, tenor, rate, fair)
        forwards.append({"expiry": expiry, "tenor": tenor, "rate": rate})
    return {"name": name, "kind": kind, "frequency": frequency, "tenors": tenors, "rates": rates,
            "times": list(times), "dfs": dfs, "forwards": forwards}  # fmt: skip


def generate() -> Dict[str, object]:
    set_evaluation_date(dt.date(*ASOF))
    cases = []
    curve, tenors, rates = _par_curve([1, 3], 6, 120)
    cases.append(
        _case(
            "par_semi",
            "par",
            2,
            curve,
            tenors,
            rates,
            [1 / 12, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.75, 7.0, 10.0],
            [(0.25, 0.25), (0.25, 2.0), (1.0, 5.0), (0.5, 1.25), (2.0, 3.0), (1.0, 9.0), (5.0, 5.0)],
        )
    )
    curve, tenors, rates = _par_curve([3, 6], 12, 120)
    cases.append(
        _case(
            "par_annual",
            "par",
            1,
            curve,
            tenors,
            rates,
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.25, 10.0],
            [(0.25, 0.5), (0.25, 1.0), (1.0, 5.0), (0.5, 2.5), (2.0, 8.0), (0.75, 1.5)],
        )
    )
    curve, tenors, rates = _zero_curve([3, 6, 12, 24, 60, 120, 360])
    cases.append(
        _case(
            "zero",
            "zero",
            1,
            curve,
            tenors,
            rates,
            [1 / 12, 0.25, 0.75, 1.5, 3.0, 7.0, 20.0, 30.0],
            [(1 / 12, 0.25), (0.25, 10.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0), (0.5, 1.5)],
        )
    )
    return {"asof": dt.date(*ASOF).isoformat(), "cases": cases}
