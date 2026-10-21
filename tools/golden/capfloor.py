"""QuantLib caps and floors for tests/test_capfloor.py.

Two oracles:

* :func:`ql_value` builds QuantLib's ``Cap``/``Floor`` from an ``IborLeg``
  on the schedule derivx generates, with a custom ``IborIndex`` (the cap's
  fixing lag, calendar, convention and day count) projecting and
  discounting on an exact copy of the curve, priced with
  ``BlackCapFloorEngine`` (plain and displaced) or
  ``BachelierCapFloorEngine``. A vol term structure is priced caplet by
  caplet, each at its own constant vol interpolated here (linear in
  ACT/365F expiry, flat outside). Strikes are set from QuantLib's own ATM
  rate.
* :func:`ql_independent` lets QuantLib generate the schedule itself
  (``MakeCapFloor`` for spot and forward starts, ``ql.Schedule`` from an
  explicit start) on any derivx curve, copied at every date the leg uses,
  with past fixings from QuantLib's own flat curve. It checks dates as
  well as prices, on awkward valuation dates, all three currency presets,
  every interpolation and negative rates.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Mapping, Optional, Tuple

import QuantLib as ql

from derivx.rates.capfloor import CapFloor
from derivx.rates.conventions import parse_tenor
from derivx.rates.curves import CONVENTIONS, CurveQuote, Market, bootstrap, flat_curve
from derivx.rates.examples import EXAMPLE_ASOF, example_market, sofr_example_rows

from .common import BDC, CALENDARS, DAY_COUNTS, pyd, qd, ql_curve, set_evaluation_date

MODELS = {"bachelier": (0.0095, 0.0), "black": (0.27, 0.0), "shifted_black": (0.18, 0.02)}
VOL_TERMS = {
    "bachelier": {0.5: 0.0080, 1.0: 0.0095, 2.0: 0.0105, 5.0: 0.0100, 10.0: 0.0088},
    "black": {0.5: 0.24, 1.0: 0.27, 2.0: 0.29, 5.0: 0.28, 10.0: 0.25},
    "shifted_black": {0.5: 0.15, 1.0: 0.17, 2.0: 0.18, 5.0: 0.18, 10.0: 0.16},
}
STRUCTURES = [("2Y", 3), ("5Y", 3), ("10Y", 6), ("5Y", 6)]
MONEYNESS = {"ITM": -0.01, "ATM": 0.0, "OTM": 0.01}  # strike offset from ATM for a cap; floors mirror it
FORWARD_STARTS = [("1Y", "5Y", 3), ("2Y", "3Y", 6)]
SEASONED_START = dt.date(2025, 7, 15)
NOTIONAL = 10_000_000.0

# Independent-schedule cases: (valuation date, curve preset, interpolation, shift added to every example quote,
# model, kind, tenor, forward start or explicit start, index months, strike offset from ATM).
SATURDAY, MONTH_END, LEAP, YEAR_END = (
    dt.date(2026, 10, 24),
    dt.date(2027, 1, 29),
    dt.date(2028, 2, 28),
    dt.date(2026, 12, 29),
)
FOLLOWING_IS_NOT_MF = dt.date(2026, 10, 28)  # USD spot + 1Y = Sat 2027-10-30: Following -> Nov 1, MF -> Oct 29
INDEPENDENT = [
    (EXAMPLE_ASOF, "USD SOFR", "loglinear", 0.0, "bachelier", "cap", "5Y", None, 3, 0.0),
    (SATURDAY, "USD SOFR", "linear_zero", 0.0, "black", "floor", "7Y", "1Y", 3, -0.005),
    (MONTH_END, "USD SOFR", "monotone_convex", 0.0, "shifted_black", "cap", "4Y", "2Y", 6, 0.005),
    (LEAP, "USD SOFR", "loglinear", 0.0, "bachelier", "floor", "2Y", "1M", 1, 0.0),
    (YEAR_END, "USD SOFR", "monotone_convex", 0.0, "black", "cap", "3Y", None, 12, 0.0),
    (FOLLOWING_IS_NOT_MF, "USD SOFR", "loglinear", 0.0, "bachelier", "cap", "5Y", "1Y", 3, 0.0),
    (FOLLOWING_IS_NOT_MF, "USD SOFR", "linear_zero", 0.0, "shifted_black", "floor", "10Y", "1Y", 6, 0.0),
    (EXAMPLE_ASOF, "USD SOFR", "linear_zero", 0.0, "bachelier", "cap", "3Y", dt.date(2026, 3, 31), 3, 0.0),
    (MONTH_END, "EUR ESTR", "loglinear", -0.045, "bachelier", "floor", "5Y", None, 6, -0.003),
    (SATURDAY, "EUR ESTR", "monotone_convex", -0.045, "shifted_black", "cap", "10Y", "3M", 3, 0.002),
    (LEAP, "EUR ESTR", "linear_zero", -0.045, "bachelier", "cap", "20Y", "5Y", 12, 0.0),
    (YEAR_END, "EUR ESTR", "loglinear", 0.0, "black", "floor", "5Y", "6M", 6, 0.0),
    (
        EXAMPLE_ASOF,
        "EUR ESTR",
        "monotone_convex",
        -0.045,
        "shifted_black",
        "floor",
        "4Y",
        dt.date(2025, 11, 28),
        6,
        0.0,
    ),
    (EXAMPLE_ASOF, "GBP SONIA", "loglinear", 0.0, "black", "cap", "5Y", None, 3, 0.0),
    (SATURDAY, "GBP SONIA", "linear_zero", 0.0, "bachelier", "floor", "3Y", "2Y", 3, 0.004),
    (MONTH_END, "GBP SONIA", "monotone_convex", 0.0, "shifted_black", "cap", "13M", None, 1, 0.0),
    (LEAP, "GBP SONIA", "loglinear", 0.0, "bachelier", "cap", "30Y", "10Y", 12, 0.0),
    (YEAR_END, "GBP SONIA", "linear_zero", 0.0, "black", "floor", "2Y", dt.date(2026, 5, 29), 3, 0.0),
]
FIXINGS = (dt.date(2024, 1, 2), 0.045)  # flat continuously compounded curve for fixings before the valuation date


def _flat_linear(points: Mapping[float, float], t: float) -> float:
    xs = sorted(points)
    if t <= xs[0]:
        return points[xs[0]]
    for a, b in zip(xs[:-1], xs[1:]):
        if t <= b:
            return points[a] + (points[b] - points[a]) * (t - a) / (b - a)
    return points[xs[-1]]


def _engine(curve: ql.YieldTermStructureHandle, model: str, vol: float, shift: float, asof: dt.date, cal):
    vol_type = ql.Normal if model == "bachelier" else ql.ShiftedLognormal
    surface = ql.ConstantOptionletVolatility(
        qd(asof), cal, ql.Following, vol, ql.Actual365Fixed(), vol_type, shift if model == "shifted_black" else 0.0
    )
    handle = ql.OptionletVolatilityStructureHandle(surface)
    if model == "bachelier":
        return ql.BachelierCapFloorEngine(curve, handle)
    return ql.BlackCapFloorEngine(curve, handle)


def _index(cf: CapFloor, handle: ql.YieldTermStructureHandle) -> ql.IborIndex:
    return ql.IborIndex(
        "DERIVX-TERM",
        ql.Period(cf.index_months, ql.Months),
        cf.fixing_lag,
        ql.USDCurrency(),
        CALENDARS[cf.calendar],
        BDC[cf.bdc],
        False,
        DAY_COUNTS[cf.day_count],
        handle,
    )


def ql_value(cf: CapFloor, market: Market) -> Dict[str, object]:
    """QuantLib NPV, ATM rate and per-caplet values of ``cf`` in ``market``
    (whose curve must be a bootstrapped log-linear one)."""

    asof = market.asof
    set_evaluation_date(asof)
    ql.IndexManager.instance().clearHistories()
    curve = ql_curve(market.curve)
    cal, dc = CALENDARS[cf.calendar], DAY_COUNTS[cf.day_count]
    index = _index(cf, curve)
    periods = cf.periods(asof)
    fixings = []
    for p in periods:  # past fixings of periods not yet paid come from the market's fixings curve
        if p.fix < asof <= p.pay:
            rate = market.projected_rate(p.start, p.end, cf.day_count, p.fix)
            index.addFixing(qd(p.fix), rate)
            fixings.append([p.fix.isoformat(), rate])
    schedule = ql.Schedule(ql.DateVector([qd(periods[0].start)] + [qd(p.end) for p in periods]))
    leg = ql.IborLeg([cf.notional], schedule, index, dc, BDC[cf.bdc], [cf.fixing_lag])
    make = ql.Cap if cf.is_cap else ql.Floor
    full = make(leg, [cf.strike])
    live = [i for i, c in enumerate(leg) if c.date() > qd(asof)]

    if isinstance(cf.vol, Mapping):
        rows, npv = [], 0.0
        for i in live:
            coupon = ql.as_floating_rate_coupon(leg[i])
            fix = coupon.fixingDate()
            t = ql.Actual365Fixed().yearFraction(qd(asof), fix) if fix > qd(asof) else 0.0
            single = make(ql.Leg([leg[i]]), [cf.strike])
            single.setPricingEngine(_engine(curve, cf.model, _flat_linear(cf.vol, t), cf.shift, asof, cal))
            npv += single.NPV()
            rows.append(_caplet_row(single, 0, coupon))
    else:
        full.setPricingEngine(_engine(curve, cf.model, float(cf.vol), cf.shift, asof, cal))
        npv = full.NPV()
        rows = [_caplet_row(full, i, ql.as_floating_rate_coupon(leg[i])) for i in live]
    out = {"npv": npv, "atm": full.atmRate(curve.currentLink()), "caplets": rows, "fixings": fixings}
    if not isinstance(cf.vol, Mapping):
        out["implied"] = _implied(full, curve, cf, asof, cal)
    return out


def _ql_cap(cf: CapFloor, index: ql.IborIndex, asof: dt.date) -> Tuple[ql.CapFloor, bool]:
    """QuantLib's own cap on ``cf``'s terms and whether the forward start had
    to be given explicitly: MakeVanillaSwap adjusts a forward start with
    Following, derivx (like its VanillaSwap) with the cap's ``bdc``."""

    cal, bdc = CALENDARS[cf.calendar], BDC[cf.bdc]
    tenor = ql.Period(parse_tenor(cf.tenor).months, ql.Months)
    if cf.start is not None:
        effective = qd(cf.start)
        schedule = ql.Schedule(
            effective,
            effective + tenor,
            ql.Period(cf.index_months, ql.Months),
            cal,
            bdc,
            bdc,
            ql.DateGeneration.Backward,
            False,
        )
        leg = list(ql.IborLeg([cf.notional], schedule, index, DAY_COUNTS[cf.day_count], bdc, [cf.fixing_lag]))
        make = ql.Cap if cf.is_cap else ql.Floor
        return make(ql.Leg(leg[1:] if cf.exclude_first else leg), [cf.strike]), False
    if cf.exclude_first is not None:
        raise ValueError("MakeCapFloor decides the first caplet itself: leave exclude_first as None")
    kind = ql.CapFloor.Cap if cf.is_cap else ql.CapFloor.Floor
    forward = ql.Period(parse_tenor(cf.forward_start).months, ql.Months) if cf.forward_start else ql.Period(0, ql.Days)
    # Not chained: each with* call returns a reference into this object, which must stay alive.
    make = ql.QuantLib._MakeCapFloor(kind, tenor, index, cf.strike, forward)
    make.withNominal(cf.notional)
    explicit = False
    if cf.forward_start:
        spot = cal.advance(cal.adjust(qd(asof)), cf.fixing_lag, ql.Days)
        own = cal.adjust(spot + forward, bdc)
        if cal.adjust(spot + forward, ql.Following) != own:
            make.withEffectiveDate(own, False)
            explicit = True
    return make.makeCapFloor(), explicit


def ql_independent(cf: CapFloor, market: Market, fixings: Optional[Tuple[dt.date, float]] = None) -> Dict[str, object]:
    """QuantLib dates and values of ``cf`` on QuantLib's own schedule (see
    :func:`_ql_cap`) on any derivx curve. ``fixings`` = (date, rate) is the
    flat continuously compounded ACT/365F curve that supplies fixings before
    the valuation date. Rows: [fixing, accrual start, accrual end, payment,
    vega per unit vol] for each unpaid caplet."""

    asof = market.asof
    set_evaluation_date(asof)
    ql.IndexManager.instance().clearHistories()
    handle = ql.RelinkableYieldTermStructureHandle(ql.FlatForward(qd(asof), 0.03, ql.Actual365Fixed()))
    index = _index(cf, handle)
    inst, explicit_start = _ql_cap(cf, index, asof)
    leg = [ql.as_floating_rate_coupon(c) for c in inst.floatingLeg()]
    today = qd(asof)
    nodes = set()
    for c in leg:
        if c.date() > today:
            used = (c.accrualStartDate(), c.accrualEndDate(), c.date(), index.valueDate(c.fixingDate()))
            nodes.update(pyd(d) for d in used)
    nodes = sorted(d for d in nodes if d > asof)
    copy = ql.DiscountCurve(
        [today] + [qd(d) for d in nodes], [1.0] + [float(market.curve.df(d)) for d in nodes], ql.Actual365Fixed()
    )
    handle.linkTo(copy)
    dc = DAY_COUNTS[cf.day_count]
    past = ql.FlatForward(qd(fixings[0]), fixings[1], ql.Actual365Fixed(), ql.Continuous) if fixings else None
    for c in leg:
        if c.fixingDate() < today and c.date() > today:
            if past is None:
                raise ValueError("a caplet fixed before the valuation date: pass fixings")
            rate = past.forwardRate(c.accrualStartDate(), c.accrualEndDate(), dc, ql.Simple).rate()
            index.addFixing(c.fixingDate(), rate)
    inst.setPricingEngine(_engine(handle, cf.model, float(cf.vol), cf.shift, asof, CALENDARS[cf.calendar]))
    vega = inst.optionletsVega()
    rows = [
        [_iso(c.fixingDate()), _iso(c.accrualStartDate()), _iso(c.accrualEndDate()), _iso(c.date()), vega[i]]
        for i, c in enumerate(leg)
        if c.date() > today
    ]
    return {"npv": inst.NPV(), "atm": inst.atmRate(copy), "caplets": rows, "explicit_start": explicit_start}


def _caplet_row(inst, i: int, coupon) -> List[object]:
    """[fixing, pay, price, vega per unit vol, forward, discount factor]."""

    return [
        _iso(coupon.fixingDate()),
        _iso(coupon.date()),
        inst.optionletsPrice()[i],
        inst.optionletsVega()[i],
        inst.optionletsAtmForward()[i],
        inst.optionletsDiscountFactor()[i],
    ]


def _implied(inst, curve, cf: CapFloor, asof: dt.date, cal) -> Dict[str, float]:
    """QuantLib's flat implied vol for the price at 1.15x the case's vol."""

    inst.setPricingEngine(_engine(curve, cf.model, 1.15 * float(cf.vol), cf.shift, asof, cal))
    price = inst.NPV()
    vol_type = ql.Normal if cf.model == "bachelier" else ql.ShiftedLognormal
    shift = cf.shift if cf.model == "shifted_black" else 0.0
    vol = inst.impliedVolatility(price, curve, float(cf.vol), 1e-15, 1000, 1e-7, 4.0, vol_type, shift)
    return {"price": price, "vol": vol}


def _iso(day: ql.Date) -> str:
    return pyd(day).isoformat()


def _spec(cf: CapFloor) -> Dict[str, object]:
    vol = [[t, v] for t, v in sorted(cf.vol.items())] if isinstance(cf.vol, Mapping) else cf.vol
    return {
        "notional": cf.notional,
        "kind": cf.kind,
        "strike": cf.strike,
        "tenor": cf.tenor,
        "vol": vol,
        "model": cf.model,
        "shift": cf.shift,
        "start": cf.start.isoformat() if cf.start else None,
        "forward_start": cf.forward_start,
        "index_months": cf.index_months,
        "day_count": cf.day_count,
        "calendar": cf.calendar,
        "fixing_lag": cf.fixing_lag,
        "exclude_first": cf.exclude_first,
    }


def _case(name: str, cf: CapFloor, market: Market) -> Dict[str, object]:
    return {"name": name, "spec": _spec(cf), **ql_value(cf, market)}


def _atm(market: Market, tenor: str, months: int, forward_start=None) -> float:
    probe = CapFloor(NOTIONAL, "cap", 0.0, tenor, 0.01, forward_start=forward_start, index_months=months)
    return float(ql_value(probe, market)["atm"])


def independent_market(asof: dt.date, preset: str, interpolation: str, quote_shift: float, fixings=None) -> Market:
    """The example quotes (moved by ``quote_shift``) bootstrapped with a
    preset's conventions; mirrored in tests/test_capfloor.py."""

    quotes = [CurveQuote(r["kind"], r["tenor"], r["rate"] + quote_shift) for r in sofr_example_rows()]
    market = Market(bootstrap(asof, quotes, preset, interpolation))
    return market.replace(fixings=flat_curve(*fixings)) if fixings else market


def _independent_cases() -> List[Dict[str, object]]:
    cases = []
    for asof, preset, interpolation, quote_shift, model, kind, tenor, begin, months, offset in INDEPENDENT:
        conv = CONVENTIONS[preset]
        vol, shift = MODELS[model]
        start = begin if isinstance(begin, dt.date) else None
        fixings = FIXINGS if start is not None else None
        market = independent_market(asof, preset, interpolation, quote_shift, fixings)
        terms = dict(
            forward_start=None if start else begin,
            start=start,
            index_months=months,
            day_count=conv.irs_float_day_count,
            calendar=conv.calendar,
            fixing_lag=conv.spot_lag,
            exclude_first=True if start else None,
        )
        atm = CapFloor(NOTIONAL, kind, 0.0, tenor, 0.01, **terms).atm_strike(market)
        cf = CapFloor(NOTIONAL, kind, atm + offset, tenor, vol, model, shift, **terms)
        cases.append(
            {
                "name": f"{preset} {interpolation} {asof} {cf.label} {model}",
                "asof": asof.isoformat(),
                "preset": preset,
                "interpolation": interpolation,
                "quote_shift": quote_shift,
                "fixings": [FIXINGS[0].isoformat(), FIXINGS[1]] if fixings else None,
                "spec": _spec(cf),
                **ql_independent(cf, market, fixings),
            }
        )
    return cases


def generate() -> dict:
    market = example_market()
    cases = []
    for model, (vol, shift) in MODELS.items():
        for tenor, months in STRUCTURES:
            atm = _atm(market, tenor, months)
            for kind in ("cap", "floor"):
                for label, offset in MONEYNESS.items():
                    strike = atm + (offset if kind == "cap" else -offset)
                    cf = CapFloor(NOTIONAL, kind, strike, tenor, vol, model, shift, index_months=months)
                    cases.append(_case(f"{model} {tenor} {months}M {kind} {label}", cf, market))
        for fwd, tenor, months in FORWARD_STARTS:
            atm = _atm(market, tenor, months, fwd)
            for kind in ("cap", "floor"):
                cf = CapFloor(NOTIONAL, kind, atm, tenor, vol, model, shift, forward_start=fwd, index_months=months)
                cases.append(_case(f"{model} {fwd}x{tenor} {months}M {kind} ATM", cf, market))
        atm = _atm(market, "10Y", 3)
        for kind in ("cap", "floor"):
            cf = CapFloor(NOTIONAL, kind, atm, "10Y", VOL_TERMS[model], model, shift)
            cases.append(_case(f"{model} 10Y 3M {kind} ATM vol term structure", cf, market))
            strike = 0.03 if kind == "cap" else 0.04  # in the money on the fixed period
            cf = CapFloor(NOTIONAL, kind, strike, "3Y", vol, model, shift, start=SEASONED_START)
            cases.append(_case(f"{model} seasoned 3Y 3M {kind} from {SEASONED_START}", cf, market))
        if model != "black":
            for kind in ("cap", "floor"):
                cf = CapFloor(NOTIONAL, kind, -0.005, "5Y", vol, model, shift)
                cases.append(_case(f"{model} 5Y 3M {kind} negative strike", cf, market))
    independent = _independent_cases()
    set_evaluation_date(EXAMPLE_ASOF)
    return {"asof": EXAMPLE_ASOF.isoformat(), "cases": cases, "independent": independent}
