"""QuantLib bootstraps of the same quotes derivx bootstraps
(tests/test_curves.py): OIS curves in three currencies, a single-curve IRS
curve and the Treasury bill/par-bond curve."""

from __future__ import annotations

import datetime as dt

import QuantLib as ql

from derivx.rates.curves import CurveQuote, bill_price, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, sofr_example_rows, ust_par_rows
from derivx.rates.quotes import clean_quotes

from .common import CALENDARS, qd, set_evaluation_date

CURVE_CLASS = {"loglinear": ql.PiecewiseLogLinearDiscount, "linear_zero": ql.PiecewiseLinearZero}
OIS_SETUPS = {
    "USD SOFR": (ql.Sofr, CALENDARS["USD"], 2, 2, 0.0),
    "EUR ESTR": (ql.Estr, CALENDARS["TARGET"], 2, 1, -0.015),
    "GBP SONIA": (ql.Sonia, CALENDARS["LONDON"], 0, 0, 0.003),
}
IRS_QUOTES = [
    ("DEPO", "3M", 0.0410),
    ("IRS", "1Y", 0.0398),
    ("IRS", "2Y", 0.0385),
    ("IRS", "5Y", 0.0380),
    ("IRS", "10Y", 0.0392),
    ("IRS", "30Y", 0.0401),
]


def _sample_dates(asof: dt.date) -> list:
    return [asof + dt.timedelta(days=d) for d in list(range(1, 400, 9)) + list(range(400, 365 * 32, 61))]


def _zeros(curve, asof: dt.date) -> list:
    return [
        [d.isoformat(), curve.zeroRate(qd(d), ql.Actual365Fixed(), ql.Continuous).rate()] for d in _sample_dates(asof)
    ]


def _ois_case(conv: str, interp: str, quotes: list) -> dict:
    index_cls, cal, spot, lag, _ = OIS_SETUPS[conv]
    handle = ql.RelinkableYieldTermStructureHandle()
    index = index_cls(handle)
    helpers = []
    for q in quotes:
        quote = ql.QuoteHandle(ql.SimpleQuote(q.rate))
        if q.kind == "DEPO":
            helpers.append(
                ql.DepositRateHelper(
                    quote, ql.Period(q.tenor), spot, cal, ql.ModifiedFollowing, False, index.dayCounter()
                )
            )
        else:
            helpers.append(
                ql.OISRateHelper(
                    spot,
                    ql.Period(q.tenor),
                    quote,
                    index,
                    ql.YieldTermStructureHandle(),
                    False,
                    lag,
                    ql.Following,
                    ql.Annual,
                    cal,
                    ql.Period(0, ql.Days),
                    0.0,
                    ql.Pillar.LastRelevantDate,
                    ql.Date(),
                    ql.RateAveraging.Compound,
                    False,
                )
            )
    curve = CURVE_CLASS[interp](qd(EXAMPLE_ASOF), helpers, ql.Actual365Fixed())
    curve.enableExtrapolation()
    return {
        "conventions": conv,
        "interpolation": interp,
        "quotes": [[q.kind, q.tenor, q.rate] for q in quotes],
        "zeros": _zeros(curve, EXAMPLE_ASOF),
    }


def _irs_case() -> dict:
    cal = CALENDARS["USD"]
    handle = ql.RelinkableYieldTermStructureHandle()
    index = ql.IborIndex(
        "TERM3M", ql.Period(3, ql.Months), 2, ql.USDCurrency(), cal, ql.ModifiedFollowing, False, ql.Actual360(), handle
    )
    helpers = []
    for kind, tenor, rate in IRS_QUOTES:
        quote = ql.QuoteHandle(ql.SimpleQuote(rate))
        if kind == "DEPO":
            helpers.append(
                ql.DepositRateHelper(quote, ql.Period(tenor), 2, cal, ql.ModifiedFollowing, False, ql.Actual360())
            )
        else:
            helpers.append(
                ql.SwapRateHelper(
                    quote,
                    ql.Period(tenor),
                    cal,
                    ql.Semiannual,
                    ql.ModifiedFollowing,
                    ql.Thirty360(ql.Thirty360.BondBasis),
                    index,
                )
            )
    curve = ql.PiecewiseLogLinearDiscount(qd(EXAMPLE_ASOF), helpers, ql.Actual365Fixed())
    curve.enableExtrapolation()
    return {
        "conventions": "USD SOFR",
        "interpolation": "loglinear",
        "quotes": [list(q) for q in IRS_QUOTES],
        "zeros": _zeros(curve, EXAMPLE_ASOF),
    }


def _ust_case() -> dict:
    day, rows = ust_par_rows()
    quotes, _, _ = clean_quotes(rows)
    ours = bootstrap(day, quotes, "USD UST")
    cal = CALENDARS["USGOV"]
    helpers = []
    for inst in ours.instruments:  # QuantLib bond helpers on the same dates
        if inst.kind == "BILL":
            price = 100.0 * bill_price(inst.quote.rate, (inst.end - inst.start).days)
            bond = ql.ZeroCouponBond(1, cal, 100.0, qd(inst.end), ql.Following, 100.0, qd(day))
            helpers.append(ql.BondHelper(ql.QuoteHandle(ql.SimpleQuote(price)), bond))
        else:
            sched = ql.Schedule(
                qd(inst.start),
                qd(inst.end),
                ql.Period(6, ql.Months),
                cal,
                ql.Unadjusted,
                ql.Unadjusted,
                ql.DateGeneration.Backward,
                False,
            )
            helpers.append(
                ql.FixedRateBondHelper(
                    ql.QuoteHandle(ql.SimpleQuote(100.0)),
                    1,
                    100.0,
                    sched,
                    [inst.quote.rate],
                    ql.ActualActual(ql.ActualActual.Bond, sched),
                    ql.Following,
                )
            )
    curve = ql.PiecewiseLogLinearDiscount(qd(day), helpers, ql.Actual365Fixed())
    curve.enableExtrapolation()
    return {
        "conventions": "USD UST",
        "interpolation": "loglinear",
        "asof": day.isoformat(),
        "quotes": [[q.kind, q.tenor, q.rate] for q in quotes],
        "zeros": _zeros(curve, day),
    }


def generate() -> dict:
    set_evaluation_date(EXAMPLE_ASOF)
    sofr, _, _ = clean_quotes(sofr_example_rows())
    cases = []
    for conv, (_, _, _, _, offset) in OIS_SETUPS.items():
        quotes = [CurveQuote(q.kind, q.tenor, q.rate + offset) for q in sofr]
        for interp in CURVE_CLASS:
            cases.append(_ois_case(conv, interp, quotes))
    cases.append(_irs_case())
    set_evaluation_date(EXAMPLE_ASOF)
    cases.append(_ust_case())
    return {"asof": EXAMPLE_ASOF.isoformat(), "cases": cases}
