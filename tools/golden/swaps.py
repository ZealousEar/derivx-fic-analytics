"""QuantLib OvernightIndexedSwap / VanillaSwap prices of derivx swaps
(tests/test_swaps.py).

``cases`` and ``seasoned``: each QuantLib swap is built from the derivx
schedule dates on an exact copy of the example SOFR curve, so any difference
is in the pricing, not the dates. Seasoned cases value swaps traded on
EXAMPLE_ASOF at a later date, with past fixings taken from the original
curve. The ``make_ois`` cases check derivx's OIS conventions against
``ql.MakeOIS`` defaults.

``independent``: swaps described only by a conventions preset and their
terms (:func:`reference`). QuantLib generates the dates itself from the
preset (spot lag, calendar, MF, backward schedule; a swap no longer than one
period plus 31 days is one period, the curve instruments' rule), with generic
overnight / term indices on the preset's calendar and day counts, so a
wrongly mapped convention in derivx shows up. The curve is a daily-node
QuantLib copy of the derivx discount factors, exact on every date, so any
interpolation can be checked. Cases cover EUR ESTR with negative rates, GBP
SONIA, linear-zero and monotone-convex curves, odd and weekend dates, tenors
beyond the last quote, and seasoned swaps valued on weekends and holidays.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import QuantLib as ql

from derivx.rates.conventions import Calendar
from derivx.rates.curves import CONVENTIONS, Curve, CurveQuote, Market, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, example_curve, sofr_example_rows
from derivx.rates.swaps import VanillaSwap

from .common import BDC, CALENDARS, DAY_COUNTS, pyd, qd, ql_curve, set_evaluation_date

NOTIONAL = 10_000_000.0
TENORS = ("1Y", "18M", "5Y", "10Y", "30Y")
SEASONED_DATES = (dt.date(2027, 2, 17), dt.date(2028, 1, 18))


def case_swap(case: Dict[str, object]) -> VanillaSwap:
    """The derivx swap a fixture case describes (mirrored in the tests)."""

    return VanillaSwap.from_conventions(
        "USD SOFR",
        case["tenor"],
        case["fixed_rate"],
        notional=case["notional"],
        receive_fixed=case["receive_fixed"],
        kind=case["kind"],
        forward_start=case["forward_start"],
        spread=case["spread"],
        trade_date=dt.date.fromisoformat(case["trade_date"]),
    )


def _schedule(dates: List[dt.date]) -> ql.Schedule:
    return ql.Schedule(ql.DateVector([qd(d) for d in dates]))


def ql_swap(swap: VanillaSwap, asof: dt.date, handle: ql.YieldTermStructureHandle) -> ql.Swap:
    """QuantLib copy of ``swap`` on ``handle`` (projection and discounting),
    from the derivx schedule dates. Fixings before ``asof`` must already be
    in the index history (see :func:`add_fixings`)."""

    cal = CALENDARS[swap.calendar]
    kind = ql.Swap.Receiver if swap.receive_fixed else ql.Swap.Payer
    fixed_dates, float_dates = swap.schedule_dates(asof)
    if swap.float_index == "OIS":
        if fixed_dates != float_dates or swap.fixed_day_count != "ACT/360" or swap.float_day_count != "ACT/360":
            raise ValueError("the QuantLib OIS copy needs one schedule and ACT/360 legs (SOFR)")
        out = ql.OvernightIndexedSwap(
            kind, swap.notional, _schedule(fixed_dates), swap.fixed_rate, DAY_COUNTS[swap.fixed_day_count],
            ql.Sofr(handle), swap.spread, swap.payment_lag, ql.Following, cal,
        )
    else:
        if swap.payment_lag:
            raise ValueError("ql.VanillaSwap has no payment lag")
        out = ql.VanillaSwap(
            kind, swap.notional, _schedule(fixed_dates), swap.fixed_rate, DAY_COUNTS[swap.fixed_day_count],
            _schedule(float_dates), term_index(swap, handle), swap.spread, DAY_COUNTS[swap.float_day_count],
        )
    out.setPricingEngine(ql.DiscountingSwapEngine(handle))
    return out


def term_index(swap: VanillaSwap, handle: ql.YieldTermStructureHandle) -> ql.IborIndex:
    return ql.IborIndex(
        "TERM", ql.Period(swap.float_months, ql.Months), swap.fixing_lag, ql.USDCurrency(), CALENDARS[swap.calendar],
        ql.ModifiedFollowing, False, DAY_COUNTS[swap.float_day_count], handle,
    )


def add_fixings(swap: VanillaSwap, asof: dt.date, fixings: Curve, handle: ql.YieldTermStructureHandle) -> None:
    """Past fixings implied by the ``fixings`` curve: daily SOFR fixings
    whose compounded growth is the curve's discount-factor ratio, or the
    curve's forward for each TERM period that fixed before ``asof``."""

    cal = Calendar(swap.calendar)
    if swap.float_index == "OIS":
        index = ql.Sofr(handle)
        day = swap.effective_date(asof)
        while day < asof:
            nxt = cal.add_business_days(day, 1)
            rate = (fixings.df(day) / fixings.df(nxt) - 1.0) * 360.0 / (nxt - day).days
            index.addFixing(qd(day), rate)
            day = nxt
        return
    index = term_index(swap, handle)
    for p in swap.periods(asof):
        if p.leg == "float" and p.fixing < asof:
            index.addFixing(qd(p.fixing), fixings.forward_rate(p.start, p.end, swap.float_day_count))


def _legs(swap: VanillaSwap, out: ql.Swap) -> tuple:
    if swap.float_index == "OIS":
        return out.fixedLeg(), out.overnightLeg(), out.overnightLegNPV()
    return out.fixedLeg(), out.floatingLeg(), out.floatingLegNPV()


def ql_values(swap: VanillaSwap, asof: dt.date, handle: ql.YieldTermStructureHandle) -> Dict[str, object]:
    out = ql_swap(swap, asof, handle)
    fixed_leg, float_leg, float_npv = _legs(swap, out)
    live = lambda leg: [[pyd(cf.date()).isoformat(), cf.amount()] for cf in leg if pyd(cf.date()) > asof]  # noqa: E731
    return {
        "npv": out.NPV(),
        "fair_rate": out.fairRate(),
        "fixed_leg_bps": out.fixedLegBPS(),
        "float_leg_npv": float_npv,
        "fixed_flows": live(fixed_leg),
        "float_flows": live(float_leg),
    }


def _case(
    kind: str, tenor: str, forward_start: Optional[str], receive: bool, rate: float, spread: float,
    valuation: dt.date = EXAMPLE_ASOF,
) -> Dict[str, object]:
    case = {
        "kind": kind, "tenor": tenor, "forward_start": forward_start, "receive_fixed": receive, "notional": NOTIONAL,
        "fixed_rate": rate, "spread": spread, "trade_date": EXAMPLE_ASOF.isoformat(),
        "valuation_date": valuation.isoformat(),
    }
    swap = case_swap(case)
    curve = example_curve(asof=valuation)
    handle = ql_curve(curve)
    set_evaluation_date(valuation)
    ql.IndexManager.instance().clearHistories()
    if valuation > EXAMPLE_ASOF:
        add_fixings(swap, valuation, example_curve(), handle)
    fixed_dates, float_dates = swap.schedule_dates(valuation)
    case.update(
        fixed_dates=[d.isoformat() for d in fixed_dates],
        float_dates=[d.isoformat() for d in float_dates],
        **ql_values(swap, valuation, handle),
    )
    ql.IndexManager.instance().clearHistories()
    return case


def _make_ois_case(tenor: str, rate: float, payment_lag: Optional[int]) -> Dict[str, object]:
    """``ql.MakeOIS`` with its defaults (and optionally a payment lag)."""

    set_evaluation_date(EXAMPLE_ASOF)
    handle = ql_curve(example_curve())
    extra = {} if payment_lag is None else {"paymentLag": payment_lag}
    out = ql.MakeOIS(ql.Period(tenor), ql.Sofr(handle), rate, nominal=NOTIONAL, receiveFixed=True,
                     pricingEngine=ql.DiscountingSwapEngine(handle), **extra)
    schedule = [pyd(d).isoformat() for d in out.fixedSchedule().dates()]
    pays = [pyd(cf.date()).isoformat() for cf in out.fixedLeg()]
    return {
        "tenor": tenor, "fixed_rate": rate, "payment_lag": payment_lag, "notional": NOTIONAL,
        "schedule": schedule, "fixed_pay_dates": pays,
        "npv": out.NPV(), "fair_rate": out.fairRate(), "fixed_leg_bps": out.fixedLegBPS(),
    }


# --------------------------------------------------------------------------
# Independent cases: QuantLib dates and indices from the conventions preset
# --------------------------------------------------------------------------


def spec(conv: str, kind: str, tenor: Optional[str], rate: float, **terms: object) -> Dict[str, object]:
    """A swap described by a conventions preset and its terms (JSON-ready).

    ``shift`` moves every example quote, ``valuation_shift`` moves them again
    on the valuation date; ``overrides`` replaces leg fields on both sides.
    """

    out = {
        "conv": conv, "kind": kind, "tenor": tenor, "fixed_rate": rate, "interp": "loglinear", "shift": 0.0,
        "trade_date": EXAMPLE_ASOF.isoformat(), "valuation_date": None, "valuation_shift": 0.0,
        "forward_start": None, "start": None, "maturity": None, "receive_fixed": True, "spread": 0.0,
        "notional": NOTIONAL, "overrides": {},
    }
    out.update(terms)
    out["valuation_date"] = out["valuation_date"] or out["trade_date"]
    return out


def spec_market(case: Dict[str, object]) -> Market:
    """The derivx market of a spec (mirrored in the tests): example quotes
    shifted and bootstrapped with the preset on the trade date, re-dated
    (and shifted again) on the valuation date with the trade-date curve as
    fixings."""

    def curve(asof: dt.date, shift: float) -> Curve:
        quotes = [CurveQuote(r["kind"], r["tenor"], r["rate"] + shift) for r in sofr_example_rows()]
        return bootstrap(asof, quotes, case["conv"], case["interp"])

    trade, valuation = (dt.date.fromisoformat(case[k]) for k in ("trade_date", "valuation_date"))
    fixings = curve(trade, case["shift"])
    if valuation == trade:
        return Market(curve=fixings)
    return Market(curve=curve(valuation, case["shift"] + case["valuation_shift"]), fixings=fixings)


def spec_swap(case: Dict[str, object]) -> VanillaSwap:
    """The derivx swap of a spec (mirrored in the tests)."""

    fields = dict(case["overrides"])
    fields.update({k: dt.date.fromisoformat(case[k]) for k in ("start", "maturity") if case[k]})
    return VanillaSwap.from_conventions(
        case["conv"], case["tenor"], case["fixed_rate"], notional=case["notional"],
        receive_fixed=case["receive_fixed"], kind=case["kind"], forward_start=case["forward_start"],
        spread=case["spread"], trade_date=dt.date.fromisoformat(case["trade_date"]), **fields,
    )


def _legs_from_preset(case: Dict[str, object]) -> Dict[str, object]:
    conv = CONVENTIONS[case["conv"]]
    if case["kind"] == "OIS":
        legs = dict(fixed_months=conv.ois_fixed_months, fixed_day_count=conv.ois_day_count,
                    float_months=conv.ois_fixed_months, float_day_count=conv.ois_day_count,
                    payment_lag=conv.payment_lag, fixing_days=0)
    else:  # term index fixed spot-lag days before the period (2 USD/EUR, 0 GBP)
        legs = dict(fixed_months=conv.irs_fixed_months, fixed_day_count=conv.irs_fixed_day_count,
                    float_months=conv.irs_float_months, float_day_count=conv.irs_float_day_count,
                    payment_lag=0, fixing_days=conv.spot_lag)
    legs.update(case["overrides"])
    return legs


def ql_dates(case: Dict[str, object]) -> tuple:
    """QuantLib's own (fixed, float) schedules for a spec."""

    conv, legs = CONVENTIONS[case["conv"]], _legs_from_preset(case)
    cal, bdc = CALENDARS[conv.calendar], BDC[conv.bdc]
    if case["start"]:
        start = qd(dt.date.fromisoformat(case["start"]))
    else:
        spot = cal.advance(cal.adjust(qd(dt.date.fromisoformat(case["trade_date"]))), conv.spot_lag, ql.Days)
        start = cal.advance(spot, ql.Period(case["forward_start"]), bdc) if case["forward_start"] else spot
    end = qd(dt.date.fromisoformat(case["maturity"])) if case["maturity"] else start + ql.Period(case["tenor"])
    out = []
    for months in (legs["fixed_months"], legs["float_months"]):
        if end - start <= months * 31:
            out.append(ql.Schedule(ql.DateVector([cal.adjust(start, bdc), cal.adjust(end, bdc)])))
        else:
            out.append(ql.Schedule(start, end, ql.Period(months, ql.Months), cal, bdc, bdc,
                                   ql.DateGeneration.Backward, False))
    return out[0], out[1]


def _daily_curve(curve: Curve, last: dt.date) -> ql.YieldTermStructureHandle:
    """QuantLib curve with a node on every date up to ``last``: exact derivx
    discount factors on any date, whatever the interpolation."""

    days = [curve.asof + dt.timedelta(days=k) for k in range((last - curve.asof).days + 1)]
    dfs = [1.0] + [float(x) for x in curve.df(days[1:])]
    ts = ql.DiscountCurve([qd(d) for d in days], dfs, ql.Actual365Fixed())
    return ql.YieldTermStructureHandle(ts)


def reference(case: Dict[str, object]) -> Dict[str, object]:
    """QuantLib NPV, fair rate, fixed-leg BPS, float-leg NPV, schedule dates
    and live cash flows of a spec, from QuantLib's own dates and indices."""

    conv, legs = CONVENTIONS[case["conv"]], _legs_from_preset(case)
    cal = CALENDARS[conv.calendar]
    market = spec_market(case)
    valuation = dt.date.fromisoformat(case["valuation_date"])
    fixed_schedule, float_schedule = ql_dates(case)
    last = pyd(cal.advance(float_schedule.dates()[-1], legs["payment_lag"] + 5, ql.Days))
    handle = _daily_curve(market.curve, last)
    set_evaluation_date(valuation)
    ql.IndexManager.instance().clearHistories()
    fixed_dc, float_dc = DAY_COUNTS[legs["fixed_day_count"]], DAY_COUNTS[legs["float_day_count"]]
    side = ql.Swap.Receiver if case["receive_fixed"] else ql.Swap.Payer
    n_fixed, n_float = len(fixed_schedule.dates()) - 1, len(float_schedule.dates()) - 1
    fixings = market.fixing_curve
    # Generic indices on the preset's calendar and day counts (the currency
    # does not enter the price).
    if case["kind"] == "OIS":
        index = ql.OvernightIndex(f"{conv.name} ON", 0, ql.USDCurrency(), cal, float_dc, handle)
        day = float_schedule.dates()[0]
        while day < qd(valuation):
            nxt = cal.advance(day, 1, ql.Days)
            growth = fixings.df(pyd(day)) / fixings.df(pyd(nxt))
            index.addFixing(day, (growth - 1.0) / float_dc.yearFraction(day, nxt))
            day = nxt
        swap = ql.OvernightIndexedSwap(
            side, [case["notional"]] * n_fixed, fixed_schedule, case["fixed_rate"], fixed_dc,
            [case["notional"]] * n_float, float_schedule, index, case["spread"], legs["payment_lag"], ql.Following, cal,
        )
        float_leg = swap.overnightLeg()
    else:
        index = ql.IborIndex(f"{conv.name} TERM", ql.Period(legs["float_months"], ql.Months), legs["fixing_days"],
                             ql.USDCurrency(), cal, ql.ModifiedFollowing, False, float_dc, handle)
        dates = float_schedule.dates()
        for s, e in zip(dates[:-1], dates[1:]):
            fixing = index.fixingDate(s)
            if fixing < qd(valuation):
                rate = (fixings.df(pyd(s)) / fixings.df(pyd(e)) - 1.0) / float_dc.yearFraction(s, e)
                index.addFixing(fixing, rate)
        swap = ql.VanillaSwap(side, case["notional"], fixed_schedule, case["fixed_rate"], fixed_dc, float_schedule,
                              index, case["spread"], float_dc)
        float_leg = swap.floatingLeg()
    swap.setPricingEngine(ql.DiscountingSwapEngine(handle))
    live = lambda leg: [[pyd(cf.date()).isoformat(), cf.amount()] for cf in leg if pyd(cf.date()) > valuation]  # noqa: E731
    out = {
        "fixed_dates": [pyd(d).isoformat() for d in fixed_schedule.dates()],
        "float_dates": [pyd(d).isoformat() for d in float_schedule.dates()],
        "npv": swap.NPV(),
        "fair_rate": swap.fairRate(),
        "fixed_leg_bps": swap.fixedLegBPS(),
        "float_leg_npv": swap.overnightLegNPV() if case["kind"] == "OIS" else swap.floatingLegNPV(),
        "fixed_flows": live(swap.fixedLeg()),
        "float_flows": live(float_leg),
        "curve_dfs": [float(x) for x in market.curve.dfs],
    }
    ql.IndexManager.instance().clearHistories()
    set_evaluation_date(EXAMPLE_ASOF)
    return out


def independent_specs() -> List[Dict[str, object]]:
    eur = dict(shift=-0.045)  # every EUR quote below zero
    return [
        # USD SOFR: other interpolations, month-end spot, weekend start, leap day, long tenors, stubs.
        spec("USD SOFR", "OIS", "10Y", 0.0371, interp="monotone_convex", trade_date="2027-02-24"),
        spec("USD SOFR", "IRS", "7Y", 0.0365, interp="linear_zero", forward_start="18M", receive_fixed=False,
             spread=0.0011),
        spec("USD SOFR", "OIS", None, 0.0362, start="2027-01-30", maturity="2032-02-29"),
        spec("USD SOFR", "OIS", "50Y", 0.0368),
        spec("USD SOFR", "OIS", "40Y", 0.0371, interp="monotone_convex", forward_start="10Y", receive_fixed=False),
        spec("USD SOFR", "IRS", "13M", 0.0372, spread=-0.0007),
        spec("USD SOFR", "OIS", "2W", 0.0391),
        spec("USD SOFR", "OIS", "6Y", 0.0358, interp="linear_zero",
             overrides={"fixed_months": 3, "fixed_day_count": "30E/360"}),
        spec("USD SOFR", "OIS", "5Y", 0.0372, interp="linear_zero", valuation_date="2027-03-06",
             valuation_shift=0.0025, spread=0.001),
        spec("USD SOFR", "IRS", "5Y", 0.0372, interp="monotone_convex", valuation_date="2027-11-25",
             valuation_shift=-0.003),
        spec("USD SOFR", "OIS", "3Y", 0.0351, forward_start="2Y", valuation_date="2027-07-03", valuation_shift=0.001,
             receive_fixed=False),
        # EUR ESTR with negative rates.
        spec("EUR ESTR", "OIS", "5Y", -0.0080, **eur),
        spec("EUR ESTR", "IRS", "10Y", -0.0065, interp="monotone_convex", forward_start="2Y", receive_fixed=False,
             **eur),
        spec("EUR ESTR", "OIS", "4Y", -0.0090, interp="linear_zero", valuation_date="2027-03-29",
             valuation_shift=0.002, spread=-0.0005, **eur),
        spec("EUR ESTR", "IRS", None, -0.0070, start="2027-05-31", maturity="2034-05-31", **eur),
        # GBP SONIA: spot lag 0, ACT/365F, term index fixed on the period start.
        spec("GBP SONIA", "OIS", "5Y", 0.0361, interp="linear_zero"),
        spec("GBP SONIA", "IRS", "5Y", 0.0368),
        spec("GBP SONIA", "IRS", "20Y", 0.0384, interp="monotone_convex", forward_start="5Y", receive_fixed=False),
        spec("GBP SONIA", "IRS", "5Y", 0.0368, valuation_date="2027-04-16", valuation_shift=0.004, spread=0.0009),
        spec("GBP SONIA", "IRS", "3Y", 0.0371, valuation_date="2027-08-31", valuation_shift=-0.002),
        spec("GBP SONIA", "OIS", "3Y", 0.0371, interp="monotone_convex", valuation_date="2027-12-27",
             valuation_shift=0.0015),
    ]


def generate() -> dict:
    cases = []
    i = 0
    for kind in ("OIS", "IRS"):
        for tenor in TENORS:
            for forward_start in (None, "1Y"):
                for receive in (True, False):
                    rate = 0.0340 + 0.0007 * (i % 6)
                    spread = (0.0, 0.0015, -0.0008)[i % 3]
                    cases.append(_case(kind, tenor, forward_start, receive, rate, spread))
                    i += 1
    seasoned = []
    for kind in ("OIS", "IRS"):
        for valuation in SEASONED_DATES:
            seasoned.append(_case(kind, "5Y", None, True, 0.0372, 0.0010, valuation))
            seasoned.append(_case(kind, "2Y", "6M", False, 0.0351, 0.0, valuation))
    make_ois = [_make_ois_case(t, 0.0365, lag) for t in ("1Y", "5Y", "10Y") for lag in (None, 2)]
    set_evaluation_date(EXAMPLE_ASOF)
    return {
        "asof": EXAMPLE_ASOF.isoformat(),
        "curve_dfs": [float(x) for x in example_curve().dfs],
        "cases": cases,
        "seasoned": seasoned,
        "make_ois": make_ois,
        "independent": [{"spec": case, **reference(case)} for case in independent_specs()],
    }
