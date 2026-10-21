"""QuantLib references for the multi-curve stack (tests/test_multicurve.py).

* ``curves``: dual-curve bootstraps. QuantLib builds each projection curve
  itself (``PiecewiseLogLinearDiscount`` / ``PiecewiseLinearZero``) from
  ``DepositRateHelper``, ``FraRateHelper``, ``FuturesRateHelper`` (convexity
  from ``HullWhite.convexityBias``), ``SwapRateHelper(...,
  discountingCurve=ois)`` and ``IborIborBasisSwapRateHelper`` with
  ``ql.Euribor6M`` / ``ql.Euribor3M``, discounting on an exact copy of the
  derivx €STR curve (the OIS bootstrap itself is checked by curves.json),
  and the 3M curve on QuantLib's own 6M curve. The SOFR case bootstraps the
  OIS curve with ``SofrFutureRateHelper`` (SR1 averaged, SR3 compounded,
  two of them inside their reference period) and ``OISRateHelper``.
* ``swaps`` / ``basis`` / ``caps`` / ``fras`` / ``futures``: instruments
  priced by QuantLib on exact copies of the derivx curves (``VanillaSwap``
  from ``MakeVanillaSwap`` against ``ql.Euribor6M``/``3M`` with OIS
  discounting, a ``Swap`` of two ``IborLeg`` for 3s6s, ``Cap``/``Floor`` on
  EURIBOR with ``BlackCapFloorEngine``/``BachelierCapFloorEngine`` on the
  OIS curve, ``ForwardRateAgreement``, ``OvernightIndexFuture``), including
  seasoned trades with fixings.
* ``convexity``: ``HullWhite.convexityBias`` on a grid.

Quotes are synthetic test inputs (the illustrative SOFR set shifted for
€STR, EURIBOR quotes from an assumed basis), as are the past fixings; none
of it is market data.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional

import QuantLib as ql

from derivx.rates.capfloor import CapFloor
from derivx.rates.curves import Curve, CurveQuote, Market, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, sofr_example_rows
from derivx.rates.indices import HullWhite, get_index, imm_date, parse_contract
from derivx.rates.multicurve import ProjectionInstrument, bootstrap_projection, illustrative_quotes
from derivx.rates.stir import ForwardRateAgreement
from derivx.rates.swaps import BasisSwap, VanillaSwap

from .common import pyd, qd, ql_curve, set_evaluation_date

HW = HullWhite(0.03, 0.0085)
NOTIONAL = 10_000_000.0
PROFILES = {
    "EURIBOR 6M": ((0.0, 24.0), (2.0, 20.0), (10.0, 14.0), (30.0, 10.0)),
    "EURIBOR 3M": ((0.0, 14.0), (2.0, 12.0), (10.0, 8.0), (30.0, 6.0)),
}
NEGATIVE = dt.date(2027, 2, 26)  # Friday, last TARGET business day of February: end-of-month dates
SETS = {  # name: (valuation date, shift of the example quotes for the €STR curve, basis profile scale)
    "base": (EXAMPLE_ASOF, -0.012, 1.0),
    "negative": (NEGATIVE, -0.045, 0.5),
    "seasoned": (dt.date(2027, 7, 30), -0.010, 1.1),
}
SIX_M = (("DEPO", "6M"), ("FRA", "1X7"), ("FRA", "2X8"), ("FRA", "3X9"), ("FRA", "4X10"), ("FRA", "5X11"),
         ("FRA", "6X12"), ("FRA", "9X15"), ("FRA", "12X18"), ("IRS", "2Y"), ("IRS", "3Y"), ("IRS", "5Y"),
         ("IRS", "7Y"), ("IRS", "10Y"), ("IRS", "15Y"), ("IRS", "20Y"), ("IRS", "30Y"))  # fmt: skip
THREE_M_FRA = (("DEPO", "3M"), ("FRA", "1X4"), ("FRA", "2X5"), ("FRA", "3X6"), ("FRA", "6X9"), ("FRA", "9X12"),
               ("IRS", "2Y"), ("IRS", "3Y"), ("IRS", "5Y"), ("IRS", "10Y"))  # fmt: skip
SOFR_FIXING_START = dt.date(2026, 9, 16)  # first day of the in-progress SR3 SEP26 period


# --------------------------------------------------------------------------
# Curve sets (mirrored in tests/test_multicurve.py)
# --------------------------------------------------------------------------


def eur_ois(asof: dt.date, shift: float) -> Curve:
    quotes = [CurveQuote(r["kind"], r["tenor"], r["rate"] + shift) for r in sofr_example_rows()]
    return bootstrap(asof, quotes, "EUR ESTR")


def _scaled(profile, scale: float):
    return tuple((y, b * scale) for y, b in profile)


def quote_set(name: str) -> Dict[str, List[List[object]]]:
    """Quotes of a curve set: €STR (OIS), EURIBOR 6M and EURIBOR 3M (DEPO,
    futures, 3s6s basis) generated from an assumed basis."""

    asof, shift, scale = SETS[name]
    ois = eur_ois(asof, shift)
    six = illustrative_quotes(ois, "EURIBOR 6M", profile=_scaled(PROFILES["EURIBOR 6M"], scale))
    keep6 = {t for _, t in SIX_M}
    six = [q for q in six if q.tenor in keep6]
    c6 = bootstrap_projection(asof, six, "EURIBOR 6M", ois)
    three = illustrative_quotes(ois, "EURIBOR 3M", HW, basis_curve=c6, profile=_scaled(PROFILES["EURIBOR 3M"], scale))
    synthetic_3m = illustrative_quotes(ois, "EURIBOR 3M", profile=_scaled(PROFILES["EURIBOR 3M"], scale))
    depo3 = next(q for q in synthetic_3m if q.kind == "DEPO")
    fra_irs = []
    c3 = bootstrap_projection(asof, three, "EURIBOR 3M", ois, basis=c6, convexity=HW)
    for kind, tenor in THREE_M_FRA:
        if kind == "DEPO":
            fra_irs.append(depo3)
            continue
        inst = ProjectionInstrument(CurveQuote(kind, tenor, 0.0), asof, get_index("EURIBOR 3M"), ois)
        fra_irs.append(CurveQuote(kind, tenor, round(inst.implied(c3), 8)))
    rows = lambda qs: [[q.kind, q.tenor, q.rate] for q in qs]  # noqa: E731
    return {"ois_shift": shift, "6M": rows(six), "3M": rows(three), "3M_fra_irs": rows(fra_irs)}


def build(asof: dt.date, quotes: Dict[str, object], interp6: str = "loglinear", interp3: str = "loglinear") -> Market:
    """derivx curve set from stored quotes."""

    ois = eur_ois(asof, quotes["ois_shift"])
    q6 = [CurveQuote(*q) for q in quotes["6M"]]
    q3 = [CurveQuote(*q) for q in quotes["3M"]]
    c6 = bootstrap_projection(asof, q6, "EURIBOR 6M", ois, interp6)
    c3 = bootstrap_projection(asof, q3, "EURIBOR 3M", ois, interp3, basis=c6, convexity=HW)
    return Market(curve=ois, projections={"EURIBOR 6M": c6, "EURIBOR 3M": c3})


# --------------------------------------------------------------------------
# QuantLib helpers
# --------------------------------------------------------------------------


def _daily(curve: Curve, last: dt.date) -> ql.YieldTermStructureHandle:
    days = [curve.asof + dt.timedelta(days=k) for k in range((last - curve.asof).days + 1)]
    dfs = [1.0] + [float(x) for x in curve.df(days[1:])]
    ts = ql.DiscountCurve([qd(d) for d in days], dfs, ql.Actual365Fixed())
    ts.enableExtrapolation()
    return ql.YieldTermStructureHandle(ts)


def copy(curve: Curve) -> ql.YieldTermStructureHandle:
    if curve.interpolation == "loglinear" and curve.pillar_dates is not None:
        return ql_curve(curve)
    return _daily(curve, curve.asof + dt.timedelta(days=365 * 62))


def _q(value: float) -> ql.QuoteHandle:
    return ql.QuoteHandle(ql.SimpleQuote(value))


def _sample(asof: dt.date) -> List[dt.date]:
    return [asof + dt.timedelta(days=d) for d in list(range(1, 800, 11)) + list(range(800, 365 * 31, 47))]


def _zeros(curve, asof: dt.date) -> List[List[object]]:
    return [[d.isoformat(), curve.zeroRate(qd(d), ql.Actual365Fixed(), ql.Continuous).rate()] for d in _sample(asof)]


def _futures_convexity(asof: dt.date, contract: str, price: float) -> tuple:
    year, month = parse_contract(contract, asof)
    start = ql.Date.nthWeekday(3, ql.Wednesday, month, year)
    end = ql.TARGET().advance(start, ql.Period(3, ql.Months), ql.ModifiedFollowing)
    dc = ql.Actual365Fixed()
    t, T = dc.yearFraction(qd(asof), start), dc.yearFraction(qd(asof), end)
    return start, ql.HullWhite.convexityBias(price, t, T, HW.sigma, HW.mean_reversion)


def _projection_helpers(kind_rows, index_name: str, ois, asof: dt.date, h6=None) -> list:
    index_cls = {"EURIBOR 6M": ql.Euribor6M, "EURIBOR 3M": ql.Euribor3M}[index_name]
    months = get_index(index_name).months
    helpers = []
    for kind, tenor, rate in kind_rows:
        if kind == "DEPO":
            helpers.append(
                ql.DepositRateHelper(
                    _q(rate), ql.Period(tenor), 2, ql.TARGET(), ql.ModifiedFollowing, True, ql.Actual360()
                )  # fmt: skip
            )
        elif kind == "FRA":
            start = int(tenor.upper().split("X")[0])
            helpers.append(ql.FraRateHelper(_q(rate), start, index_cls()))
        elif kind == "FUT":
            price = 100.0 * (1.0 - rate)
            start, ca = _futures_convexity(asof, tenor, price)
            helpers.append(ql.FuturesRateHelper(_q(price), start, index_cls(), _q(ca)))
        elif kind == "IRS":
            helpers.append(
                ql.SwapRateHelper(
                    _q(rate),
                    ql.Period(tenor),
                    ql.TARGET(),
                    ql.Annual,
                    ql.ModifiedFollowing,
                    ql.Thirty360(ql.Thirty360.BondBasis),
                    index_cls(),
                    ql.QuoteHandle(),
                    ql.Period(0, ql.Days),
                    ois,
                )  # fmt: skip
            )
        elif kind == "BASIS":
            helpers.append(
                ql.IborIborBasisSwapRateHelper(
                    _q(rate),
                    ql.Period(tenor),
                    2,
                    ql.TARGET(),
                    ql.ModifiedFollowing,
                    False,
                    index_cls(),
                    ql.Euribor6M(h6),
                    ois,
                    True,
                )  # fmt: skip
            )
        else:
            raise ValueError(kind)
    assert months
    return helpers


CURVE_CLASS = {"loglinear": ql.PiecewiseLogLinearDiscount, "linear_zero": ql.PiecewiseLinearZero}


def _ql_projection(asof, rows, index_name, ois, interp, h6=None):
    curve = CURVE_CLASS[interp](qd(asof), _projection_helpers(rows, index_name, ois, asof, h6), ql.Actual365Fixed())
    curve.enableExtrapolation()
    return curve


def _curve_cases() -> List[Dict[str, object]]:
    out = []
    for name in ("base", "negative"):
        asof = SETS[name][0]
        set_evaluation_date(asof)
        quotes = quote_set(name)
        ois_curve = eur_ois(asof, quotes["ois_shift"])
        ois = ql_curve(ois_curve)
        for interp in ("loglinear", "linear_zero") if name == "base" else ("loglinear",):
            c6 = _ql_projection(asof, quotes["6M"], "EURIBOR 6M", ois, interp)
            h6 = ql.YieldTermStructureHandle(c6)
            c3 = _ql_projection(asof, quotes["3M"], "EURIBOR 3M", ois, interp, h6)
            c3b = _ql_projection(asof, quotes["3M_fra_irs"], "EURIBOR 3M", ois, interp)
            out.append(
                {
                    "set": name,
                    "asof": asof.isoformat(),
                    "interpolation": interp,
                    "quotes": quotes,
                    "pillars_6m": [pyd(d).isoformat() for d in c6.dates()[1:]],
                    "pillars_3m": [pyd(d).isoformat() for d in c3.dates()[1:]],
                    "zeros_6m": _zeros(c6, asof),
                    "zeros_3m": _zeros(c3, asof),
                    "zeros_3m_fra_irs": _zeros(c3b, asof),
                    "convexity_3m": [
                        [q[1], _futures_convexity(asof, q[1], 100.0 * (1.0 - q[2]))[1]]
                        for q in quotes["3M"]
                        if q[0] == "FUT"
                    ],  # fmt: skip
                }
            )
    set_evaluation_date(EXAMPLE_ASOF)
    return out


# --------------------------------------------------------------------------
# SOFR futures in the OIS curve
# --------------------------------------------------------------------------

# QuantLib bootstraps these itself: every contract's period ends on a business day, where QuantLib's pillar
# and derivx's coincide (an SR1 month ending on a weekend or holiday is solved at the next business day by
# derivx, so that it reprices; REPRICED checks those with QuantLib's own futures pricer).
SOFR_FUTURES = (("SR3", "SEP26", 96.1200), ("SR1", "NOV26", 96.1650), ("SR3", "DEC26", 96.3150),
                ("SR1", "FEB27", 96.3800), ("SR3", "MAR27", 96.4400), ("SR3", "JUN27", 96.5300),
                ("SR3", "SEP27", 96.5900))  # fmt: skip
REPRICED = (("SR1", "OCT26", 96.0950), ("SR3", "SEP26", 96.1200), ("SR1", "NOV26", 96.1650),
            ("SR1", "DEC26", 96.2300), ("SR3", "DEC26", 96.3150), ("SR1", "JAN27", 96.3000),
            ("SR3", "MAR27", 96.4400), ("SR3", "JUN27", 96.5300), ("SR3", "SEP27", 96.5900))  # fmt: skip
SOFR_OIS = (("DEPO", "1W"), ("OIS", "2Y"), ("OIS", "3Y"), ("OIS", "5Y"), ("OIS", "7Y"), ("OIS", "10Y"),
            ("OIS", "15Y"), ("OIS", "20Y"), ("OIS", "30Y"))  # fmt: skip


def sofr_fixings() -> Dict[dt.date, float]:
    """Synthetic SOFR fixings (test inputs, not market data) for the
    business days from SOFR_FIXING_START to the day before EXAMPLE_ASOF."""

    from derivx.rates.conventions import Calendar

    cal, out, day, k = Calendar("USD"), {}, SOFR_FIXING_START, 0
    while day < EXAMPLE_ASOF:
        if cal.is_business_day(day):
            out[day] = round(0.0392 + 0.0004 * math.sin(k / 3.0), 6)
            k += 1
        day += dt.timedelta(days=1)
    return out


def sofr_quotes(futures=SOFR_FUTURES) -> List[CurveQuote]:
    rates = {r["tenor"]: r["rate"] for r in sofr_example_rows()}
    out = [CurveQuote(kind, tenor, rates[tenor]) for kind, tenor in SOFR_OIS]
    out += [CurveQuote(kind, contract, (100.0 - price) / 100.0) for kind, contract, price in futures]
    return out


def _sofr_case() -> Dict[str, object]:
    asof = EXAMPLE_ASOF
    set_evaluation_date(asof)
    ql.IndexManager.instance().clearHistories()
    fixings = sofr_fixings()
    quotes = sofr_quotes()
    ours = bootstrap(asof, quotes, "USD SOFR", convexity=HW, fixings=fixings)
    convexity = {(i.kind, i.quote.tenor): i.convexity_adjustment for i in ours.instruments}
    index = ql.Sofr()
    for day, rate in fixings.items():
        index.addFixing(qd(day), rate)
    cal = ql.UnitedStates(ql.UnitedStates.SOFR)
    helpers = []
    for q in quotes:
        if q.kind == "DEPO":
            helpers.append(ql.DepositRateHelper(_q(q.rate), ql.Period(q.tenor), 2, cal, ql.ModifiedFollowing, False,
                                                ql.Actual360()))  # fmt: skip
        elif q.kind == "OIS":
            helpers.append(
                ql.OISRateHelper(
                    2,
                    ql.Period(q.tenor),
                    _q(q.rate),
                    ql.Sofr(),
                    ql.YieldTermStructureHandle(),
                    False,
                    2,
                    ql.Following,
                    ql.Annual,
                    cal,
                    ql.Period(0, ql.Days),
                    0.0,
                    ql.Pillar.LastRelevantDate,
                    ql.Date(),
                    ql.RateAveraging.Compound,
                    False,
                )  # fmt: skip
            )
        else:
            year, month = parse_contract(q.tenor, asof)
            freq = ql.Monthly if q.kind == "SR1" else ql.Quarterly
            helpers.append(ql.SofrFutureRateHelper(100.0 * (1.0 - q.rate), month, year, freq,
                                                   convexity[(q.kind, q.tenor)]))  # fmt: skip
    curve = ql.PiecewiseLogLinearDiscount(qd(asof), helpers, ql.Actual365Fixed())
    curve.enableExtrapolation()
    out = {
        "asof": asof.isoformat(),
        "quotes": [[q.kind, q.tenor, q.rate] for q in quotes],
        "fixings": [[d.isoformat(), r] for d, r in fixings.items()],
        "pillars": [pyd(d).isoformat() for d in curve.dates()[1:]],
        "zeros": _zeros(curve, asof),
        "convexity": [[k, t, v] for (k, t), v in convexity.items() if k in ("SR1", "SR3")],
    }
    # A derivx curve with SR1 months ending on weekends/holidays and part-fixed contracts: QuantLib's
    # OvernightIndexFuture on an exact copy prices every contract at its quote.
    repriced_quotes = sofr_quotes(REPRICED)
    repriced = bootstrap(asof, repriced_quotes, "USD SOFR", convexity=HW, fixings=fixings)
    handle = ql_curve(repriced)
    prices = []
    for inst in repriced.instruments:
        if inst.kind not in ("SR1", "SR3"):
            continue
        averaging = ql.RateAveraging.Compound if inst.kind == "SR3" else ql.RateAveraging.Simple
        future = ql.OvernightIndexFuture(ql.Sofr(handle), qd(inst.start), qd(inst.end), _q(inst.convexity_adjustment),
                                         averaging)  # fmt: skip
        prices.append([inst.kind, inst.quote.tenor, future.NPV(), inst.convexity_adjustment])
    out["repriced_quotes"] = [[q.kind, q.tenor, q.rate] for q in repriced_quotes]
    out["prices"] = prices
    ql.IndexManager.instance().clearHistories()
    return out


# --------------------------------------------------------------------------
# Instruments on exact curve copies
# --------------------------------------------------------------------------


def _market(set_name: str, interp6: str = "loglinear") -> Market:
    asof = SETS[set_name][0]
    return build(asof, quote_set(set_name), interp6)


def _index(name: str, handle) -> ql.IborIndex:
    return {"EURIBOR 6M": ql.Euribor6M, "EURIBOR 3M": ql.Euribor3M}[name](handle)


def _past_fixings(index_name: str, dates: List[dt.date], market_then: Market) -> Dict[dt.date, float]:
    """Synthetic past fixings: each period's forward on the trade-date curve."""

    ix = get_index(index_name)
    out = {}
    for fixing in dates:
        value = ix.value_date(fixing)
        out[fixing] = market_then.projection(index_name).forward_rate(value, ix.maturity_date(value), ix.day_count)
    return out


SWAP_SPECS = (
    # (set, index, tenor, forward start, fixed rate, receive fixed, spread, interp6)
    ("base", "EURIBOR 6M", "10Y", None, 0.0275, True, 0.0, "loglinear"),
    ("base", "EURIBOR 6M", "5Y", "2Y", 0.0262, False, 0.0010, "loglinear"),
    ("base", "EURIBOR 6M", "30Y", None, 0.0280, True, -0.0005, "linear_zero"),
    ("base", "EURIBOR 3M", "7Y", None, 0.0268, False, 0.0, "loglinear"),
    ("base", "EURIBOR 3M", "2Y", "1Y", 0.0250, True, 0.0, "loglinear"),
    ("negative", "EURIBOR 6M", "10Y", None, -0.0020, True, 0.0, "loglinear"),
    ("negative", "EURIBOR 3M", "5Y", "6M", -0.0030, False, 0.0, "loglinear"),
)


def _swap_case(spec, trade_date: Optional[dt.date] = None) -> Dict[str, object]:
    set_name, index_name, tenor, fwd, rate, receive, spread, interp6 = spec
    market = _market(set_name, interp6)
    asof = market.asof
    trade = trade_date or asof
    swap = VanillaSwap.from_index(index_name, tenor, rate, notional=NOTIONAL, receive_fixed=receive,
                                  forward_start=fwd, spread=spread, trade_date=trade)  # fmt: skip
    fixings = {}
    if trade_date is not None:  # seasoned: past fixings from the trade-date curves
        then = build(trade_date, quote_set("base"), interp6)
        past = [p.fixing for p in swap.periods(asof) if p.leg == "float" and p.fixing < asof]
        fixings = _past_fixings(index_name, past, then)
        market = market.replace(index_fixings={index_name: fixings})
    set_evaluation_date(asof)
    ql.IndexManager.instance().clearHistories()
    ois = copy(market.curve)
    proj = copy(market.projection(index_name))
    index = _index(index_name, proj)
    for day, value in fixings.items():
        index.addFixing(qd(day), value)
    fixed_dates, float_dates = swap.schedule_dates(asof)
    cal = ql.TARGET()
    fixed = ql.Schedule(qd(fixed_dates[0]), qd(swap.termination_date(asof)), ql.Period(1, ql.Years), cal,
                        ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Backward, False)  # fmt: skip
    flt = ql.Schedule(qd(float_dates[0]), qd(swap.termination_date(asof)), index.tenor(), cal, ql.ModifiedFollowing,
                      ql.ModifiedFollowing, ql.DateGeneration.Backward, False)  # fmt: skip
    side = ql.Swap.Receiver if receive else ql.Swap.Payer
    out = ql.VanillaSwap(side, NOTIONAL, fixed, rate, ql.Thirty360(ql.Thirty360.BondBasis), flt, index, spread,
                         index.dayCounter())  # fmt: skip
    out.setPricingEngine(ql.DiscountingSwapEngine(ois))
    result = {
        "spec": list(spec),
        "trade_date": trade.isoformat(),
        "valuation_date": asof.isoformat(),
        "fixings": [[d.isoformat(), r] for d, r in fixings.items()],
        "fixed_dates": [pyd(d).isoformat() for d in fixed.dates()],
        "float_dates": [pyd(d).isoformat() for d in flt.dates()],
        "npv": out.NPV(),
        "fair_rate": out.fairRate(),
        "fixed_leg_bps": out.fixedLegBPS(),
        "float_leg_npv": out.floatingLegNPV(),
    }
    ql.IndexManager.instance().clearHistories()
    return result


BASIS_SPECS = (
    # (set, tenor, forward start, spread, receive spread leg)
    ("base", "10Y", None, 0.0008, True),
    ("base", "5Y", "1Y", 0.0005, False),
    ("base", "30Y", None, 0.0006, False),
    ("negative", "7Y", None, 0.0004, True),
)


def _basis_case(spec) -> Dict[str, object]:
    set_name, tenor, fwd, spread, receive = spec
    market = _market(set_name)
    asof = market.asof
    swap = BasisSwap(NOTIONAL, spread, tenor, forward_start=fwd, receive_spread_leg=receive, trade_date=asof)
    set_evaluation_date(asof)
    ql.IndexManager.instance().clearHistories()
    ois = copy(market.curve)
    i3 = ql.Euribor3M(copy(market.projection("EURIBOR 3M")))
    i6 = ql.Euribor6M(copy(market.projection("EURIBOR 6M")))
    start, end = qd(swap.effective_date(asof)), qd(swap.termination_date(asof))
    legs = []
    for index, spreads in ((i3, [spread]), (i6, [0.0])):
        sched = ql.MakeSchedule(effectiveDate=start, terminationDate=end, tenor=index.tenor(), calendar=ql.TARGET(),
                                convention=ql.ModifiedFollowing, terminalDateConvention=ql.ModifiedFollowing,
                                rule=ql.DateGeneration.Backward, endOfMonth=False)  # fmt: skip
        leg = ql.IborLeg([NOTIONAL], sched, index, index.dayCounter(), ql.ModifiedFollowing, [2], [1.0], spreads)
        legs.append((leg, [pyd(d).isoformat() for d in sched.dates()]))
    inst = ql.Swap([legs[0][0], legs[1][0]], [not receive, receive])
    inst.setPricingEngine(ql.DiscountingSwapEngine(ois))
    bps0 = inst.legBPS(0)
    fair = spread - inst.NPV() / (bps0 / 1e-4)
    return {
        "spec": list(spec),
        "valuation_date": asof.isoformat(),
        "spread_dates": legs[0][1],
        "other_dates": legs[1][1],
        "npv": inst.NPV(),
        "fair_spread": fair,
        "spread_leg_bps": abs(bps0),
    }


CAP_SPECS = (
    # (set, index, kind, tenor, forward start, strike offset from ATM, model, vol)
    ("base", "EURIBOR 6M", "cap", "5Y", None, 0.0, "bachelier", 0.0090),
    ("base", "EURIBOR 6M", "floor", "10Y", "2Y", -0.005, "black", 0.30),
    ("base", "EURIBOR 3M", "cap", "2Y", None, 0.004, "bachelier", 0.0085),
    ("base", "EURIBOR 3M", "floor", "3Y", "1Y", 0.0, "black", 0.28),
    ("negative", "EURIBOR 6M", "cap", "7Y", None, 0.002, "bachelier", 0.0070),
    ("negative", "EURIBOR 3M", "floor", "4Y", None, -0.001, "bachelier", 0.0060),
)


def _cap_case(spec) -> Dict[str, object]:
    set_name, index_name, kind, tenor, fwd, offset, model, vol = spec
    market = _market(set_name)
    asof = market.asof
    probe = CapFloor.from_index(index_name, NOTIONAL, kind, 0.0, tenor, 0.01, "bachelier", forward_start=fwd)
    set_evaluation_date(asof)
    ql.IndexManager.instance().clearHistories()
    ois = copy(market.curve)
    index = _index(index_name, copy(market.projection(index_name)))
    periods = probe.periods(asof)
    schedule = ql.Schedule(ql.DateVector([qd(periods[0].start)] + [qd(p.end) for p in periods]))
    leg = ql.IborLeg([NOTIONAL], schedule, index, index.dayCounter(), ql.ModifiedFollowing, [2])
    make = ql.Cap if kind == "cap" else ql.Floor
    atm = make(leg, [0.0]).atmRate(ois.currentLink())
    strike = atm + offset
    inst = make(leg, [strike])
    vol_type = ql.Normal if model == "bachelier" else ql.ShiftedLognormal
    surface = ql.ConstantOptionletVolatility(
        qd(asof), ql.TARGET(), ql.Following, vol, ql.Actual365Fixed(), vol_type, 0.0
    )
    handle = ql.OptionletVolatilityStructureHandle(surface)
    engine = ql.BachelierCapFloorEngine(ois, handle) if model == "bachelier" else ql.BlackCapFloorEngine(ois, handle)
    inst.setPricingEngine(engine)
    return {"spec": list(spec), "valuation_date": asof.isoformat(), "atm": atm, "strike": strike, "npv": inst.NPV(),
            "dates": [pyd(d).isoformat() for d in schedule.dates()]}  # fmt: skip


FRA_SPECS = (
    # (set, index, months to start, fixed rate, pay fixed)
    ("base", "EURIBOR 6M", 6, 0.0250, True),
    ("base", "EURIBOR 6M", 1, 0.0260, False),
    ("base", "EURIBOR 3M", 9, 0.0240, True),
    ("negative", "EURIBOR 6M", 12, -0.0030, False),
)


def _fra_case(spec) -> Dict[str, object]:
    set_name, index_name, months, rate, pay_fixed = spec
    market = _market(set_name)
    asof = market.asof
    fra = ForwardRateAgreement(index_name, rate, months_to_start=months, notional=NOTIONAL, pay_fixed=pay_fixed)
    set_evaluation_date(asof)
    ql.IndexManager.instance().clearHistories()
    ois = copy(market.curve)
    index = _index(index_name, copy(market.projection(index_name)))
    value = fra.value_date(asof)
    side = ql.Position.Long if pay_fixed else ql.Position.Short
    inst = ql.ForwardRateAgreement(index, qd(value), side, rate, NOTIONAL, ois)
    return {"spec": list(spec), "valuation_date": asof.isoformat(), "value_date": value.isoformat(),
            "npv": inst.NPV(), "forward": inst.forwardRate().rate(), "amount": inst.amount()}  # fmt: skip


def _euribor_futures() -> List[List[object]]:
    market = _market("base")
    asof = market.asof
    set_evaluation_date(asof)
    proj = copy(market.projection("EURIBOR 3M"))
    out = []
    for contract in ("DEC26", "MAR27", "SEP27", "JUN28"):
        start = imm_date(*parse_contract(contract, asof))
        end = ql.TARGET().advance(qd(start), ql.Period(3, ql.Months), ql.ModifiedFollowing)
        fwd = proj.forwardRate(qd(start), end, ql.Actual360(), ql.Simple).rate()
        out.append([contract, fwd])
    return out


def _convexity_grid() -> List[List[float]]:
    rows = []
    for price in (95.0, 97.5, 101.0):
        for t, length in ((0.1, 0.25), (1.0, 0.25), (4.0, 0.5), (9.0, 0.25)):
            for sigma, a in ((0.0085, 0.03), (0.012, 0.0), (0.006, 0.25)):
                rows.append(
                    [price, t, t + length, sigma, a, ql.HullWhite.convexityBias(price, t, t + length, sigma, a)]
                )
    return rows


def generate() -> dict:
    data = {
        "hw": [HW.mean_reversion, HW.sigma],
        "sets": {name: {"asof": asof.isoformat(), "quotes": quote_set(name)} for name, (asof, _, _) in SETS.items()},
        "curves": _curve_cases(),
        "sofr": _sofr_case(),
        "swaps": [_swap_case(s) for s in SWAP_SPECS],
        "seasoned_swaps": [
            _swap_case(("seasoned", "EURIBOR 6M", "5Y", None, 0.0262, True, 0.0, "loglinear"), EXAMPLE_ASOF),
            _swap_case(("seasoned", "EURIBOR 3M", "3Y", None, 0.0255, False, 0.0005, "loglinear"), EXAMPLE_ASOF),
        ],
        "basis": [_basis_case(s) for s in BASIS_SPECS],
        "caps": [_cap_case(s) for s in CAP_SPECS],
        "fras": [_fra_case(s) for s in FRA_SPECS],
        "euribor_futures": _euribor_futures(),
        "convexity": _convexity_grid(),
    }
    set_evaluation_date(EXAMPLE_ASOF)
    ql.IndexManager.instance().clearHistories()
    return data
