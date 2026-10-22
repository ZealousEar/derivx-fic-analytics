"""QuantLib FixedRateBond analytics of derivx bonds (tests/test_bonds.py).

Each QuantLib bond is built as a Treasury: an unadjusted Schedule backward
from maturity (end-of-month rule for a month-end maturity, ``firstDate`` for
a long first coupon), ActualActual(Bond, schedule) or Thirty360(BondBasis),
one settlement day on the UnitedStates GovernmentBond calendar. Two copies:

* Following payments: derivx ``convention="true"`` yields, curve prices,
  z-spreads on the Treasury curve and par-par asset swaps on the SOFR curve;
* Unadjusted payments: derivx ``convention="street"`` yields (cash flows on
  the scheduled coupon dates).

QuantLib cannot apply a yield to the Following copy of a bond maturing on a
weekend (its schedule-based ACT/ACT stops at maturity), so such a case has
street yields only. Yields and spreads are solved to 1e-15.

Spreads on other swap conventions: the example swap quotes bootstrapped as
SOFR, ESTR and SONIA curves give the OIS par rate to a bond's maturity (an
OvernightIndexedSwap on the derivx spot-start schedule) and par-par asset
swaps with a floating leg on the curve's OIS day count. Par yields come from
the Treasury curve of a day whose spot is a month end (Apr 30), so par bonds
pay on month ends.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import QuantLib as ql

from derivx.rates.bonds import FixedRateBond
from derivx.rates.curves import Curve, bootstrap, swap_schedule
from derivx.rates.examples import EXAMPLE_ASOF, example_curve, sofr_example_rows, ust_curve
from derivx.rates.quotes import clean_quotes

from .common import CALENDARS, DAY_COUNTS, pyd, qd, ql_curve, set_evaluation_date

YIELDS = (0.0475, 0.02, 0.085)
PRICES = (99.25, 92.0, 104.5)
ACCURACY = 1e-15
SETTLE = "2026-10-20"
CASES: List[Dict[str, object]] = [
    {"name": "10Y note, regular, weekend coupons", "bond": {"coupon": 0.04375, "maturity": "2035-11-15"}},
    {"name": "10Y note, dated", "bond": {"coupon": 0.04375, "maturity": "2035-11-15", "dated": "2025-11-15"}},
    {
        "name": "10Y note settling the Monday after a Sunday coupon",
        "bond": {"coupon": 0.04375, "maturity": "2035-11-15"},
        "settle": "2026-11-16",
    },
    {"name": "2Y EOM note (Aug 31 / Feb 28, leap Feb 29)", "bond": {"coupon": 0.0375, "maturity": "2028-08-31"}},
    {"name": "5Y EOM note maturing Feb 28", "bond": {"coupon": 0.04, "maturity": "2031-02-28", "dated": "2026-02-28"}},
    {"name": "short first coupon", "bond": {"coupon": 0.045, "maturity": "2031-11-14", "dated": "2026-10-01"}},
    {
        "name": "long first coupon, settling in its first quasi-period",
        "bond": {"coupon": 0.05, "maturity": "2036-05-15", "dated": "2026-07-01", "first_coupon": "2027-05-15"},
    },
    {
        "name": "long first coupon, settling in its second quasi-period",
        "bond": {"coupon": 0.05, "maturity": "2036-05-15", "dated": "2026-07-01", "first_coupon": "2027-05-15"},
        "settle": "2026-12-01",
    },
    {"name": "30Y bond", "bond": {"coupon": 0.04625, "maturity": "2056-02-15"}},
    {"name": "quarterly ACT/ACT", "bond": {"coupon": 0.06, "maturity": "2030-12-16", "frequency": 4}},
    {
        "name": "30/360 semiannual",
        "bond": {"coupon": 0.0525, "maturity": "2033-06-15", "dated": "2023-06-15", "day_count": "30/360"},
    },
    {
        "name": "30/360 annual, short first coupon",
        "bond": {
            "coupon": 0.061,
            "maturity": "2034-03-15",
            "dated": "2026-09-01",
            "day_count": "30/360",
            "frequency": 1,
        },
    },
    {"name": "Sunday maturity", "bond": {"coupon": 0.0425, "maturity": "2037-11-15"}, "street_only": True},
]
TRADE_DATES = ("2026-10-19", "2026-11-10", "2026-11-25", "2026-12-24", "2027-07-02")
SWAP_CONVENTIONS = ("USD SOFR", "EUR ESTR", "GBP SONIA")
SPREAD_CASES = (0, 3, 6)  # the 10Y note, the 2Y EOM note and the long first coupon
PAR_DATE = "2026-04-29"
PAR_MATURITIES = ("2028-04-30", "2029-04-30", "2031-04-30", "2033-01-15", "2036-04-30", "2041-02-28", "2056-04-30")
FREQUENCY = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly}


def case_bond(spec: Dict[str, object]) -> FixedRateBond:
    """The derivx bond a fixture case describes (mirrored in the tests)."""

    dates = {k: dt.date.fromisoformat(v) for k, v in spec.items() if k in ("maturity", "dated", "first_coupon")}
    return FixedRateBond(**{**spec, **dates})


def ql_bond(bond: FixedRateBond, settle: dt.date, payment: int):
    """(QuantLib bond, its day counter). A bond without a dated date is
    issued on the regular coupon date before settlement."""

    cal = CALENDARS[bond.calendar]
    issue = bond.dated or bond.coupon_dates(settle)[0]
    first = qd(bond.first_coupon) if bond.first_coupon else ql.Date()
    schedule = ql.Schedule(
        qd(issue),
        qd(bond.maturity),
        ql.Period(bond.months, ql.Months),
        cal,
        ql.Unadjusted,
        ql.Unadjusted,
        ql.DateGeneration.Backward,
        bond.end_of_month,
        first,
    )
    if bond.day_count == "ACT/ACT ICMA":
        dc = ql.ActualActual(ql.ActualActual.Bond, schedule)
    else:
        dc = DAY_COUNTS[bond.day_count]
    out = ql.FixedRateBond(bond.settle_lag, bond.face, schedule, [bond.coupon], dc, payment)
    return out, dc


def _yield_block(bond: ql.FixedRateBond, dc: ql.DayCounter, freq: int, settle: ql.Date) -> Dict[str, list]:
    accrued = ql.BondFunctions.accruedAmount(bond, settle)
    out: Dict[str, list] = {k: [] for k in ("clean", "dirty", "macaulay", "modified", "convexity", "bps", "yield")}
    for y in YIELDS:
        rate = ql.InterestRate(y, dc, ql.Compounded, freq)
        clean = ql.BondFunctions.cleanPrice(bond, rate, settle)
        out["clean"].append(clean)
        out["dirty"].append(clean + accrued)
        out["macaulay"].append(ql.BondFunctions.duration(bond, rate, ql.Duration.Macaulay, settle))
        out["modified"].append(ql.BondFunctions.duration(bond, rate, ql.Duration.Modified, settle))
        out["convexity"].append(ql.BondFunctions.convexity(bond, rate, settle))
        out["bps"].append(ql.BondFunctions.bps(bond, rate, settle))
    for price in PRICES:
        bp = ql.BondPrice(price, ql.BondPrice.Clean)
        out["yield"].append(ql.BondFunctions.bondYield(bond, bp, dc, ql.Compounded, freq, settle, ACCURACY, 1000, 0.05))
    return out


def swap_curve(conventions: str) -> Curve:
    """The example swap quotes bootstrapped on ``conventions`` (mirrored in
    the tests)."""

    quotes, _, _ = clean_quotes(sofr_example_rows())
    return bootstrap(EXAMPLE_ASOF, quotes, conventions)


def _asset_swap_spread(
    bond: ql.FixedRateBond,
    ours: FixedRateBond,
    settle: dt.date,
    price: float,
    handle: ql.YieldTermStructureHandle,
    day_count: str = "ACT/360",
) -> float:
    """QuantLib par-par AssetSwap with a floating leg from settlement through
    the bond's payment dates on ``day_count``, on an index projected off
    ``handle``."""

    cal = CALENDARS[ours.calendar]
    dc = DAY_COUNTS[day_count]
    index = ql.IborIndex("FLOAT", ql.Period(6, ql.Months), 0, ql.USDCurrency(), cal, ql.Following, False, dc, handle)
    dates = ql.DateVector([qd(d) for d in [settle] + [c.pay for c in ours.coupons(settle)]])
    swap = ql.AssetSwap(True, bond, price, index, 0.0, ql.Schedule(dates), dc, True)
    swap.setPricingEngine(ql.DiscountingSwapEngine(handle))
    return swap.fairSpread()


def _ois_rate(curve: Curve, handle: ql.YieldTermStructureHandle, maturity: dt.date) -> float:
    """Fair rate of a QuantLib OvernightIndexedSwap on the derivx spot-start
    schedule to ``maturity`` (the curve's OIS conventions)."""

    conv = curve.conventions
    cal = conv.cal
    spot = cal.add_business_days(cal.adjust(curve.asof, "F"), conv.spot_lag)
    dates = swap_schedule(spot, maturity, conv.ois_fixed_months, cal, conv.bdc)
    dc = DAY_COUNTS[conv.ois_day_count]
    ql_cal = CALENDARS[conv.calendar]
    index = ql.OvernightIndex(f"{conv.name} ON", 0, ql.USDCurrency(), ql_cal, dc, handle)
    schedule = ql.Schedule(ql.DateVector([qd(d) for d in dates]))
    swap = ql.OvernightIndexedSwap(
        ql.Swap.Payer, 1.0, schedule, 0.0, dc, index, 0.0, conv.payment_lag, ql.Following, ql_cal
    )
    swap.setPricingEngine(ql.DiscountingSwapEngine(handle))
    return swap.fairRate()


def _swap_spreads() -> List[Dict[str, object]]:
    settle = dt.date.fromisoformat(SETTLE)
    out = []
    for name in SWAP_CONVENTIONS:
        curve = swap_curve(name)
        handle = ql_curve(curve)
        for i in SPREAD_CASES:
            bond = case_bond(CASES[i]["bond"])
            following, _ = ql_bond(bond, settle, ql.Following)
            out.append(
                {
                    "conventions": name,
                    "case": i,
                    "ois_rate": _ois_rate(curve, handle, bond.maturity),
                    "asw": [
                        _asset_swap_spread(following, bond, settle, p, handle, curve.conventions.ois_day_count)
                        for p in PRICES
                    ],
                }
            )
    return out


def _par_yields() -> Dict[str, object]:
    """Coupons that price QuantLib bonds dated at spot at par on the Treasury
    curve of PAR_DATE (prices are linear in the coupon)."""

    asof = dt.date.fromisoformat(PAR_DATE)
    set_evaluation_date(asof)
    handle = ql_curve(ust_curve(asof))
    cal = CALENDARS["USGOV"]
    settle = pyd(cal.advance(qd(asof), 1, ql.Days))
    rates = []
    for text in PAR_MATURITIES:
        maturity = dt.date.fromisoformat(text)
        eom = FixedRateBond(0.0, maturity).end_of_month
        schedule = ql.Schedule(
            qd(settle),
            qd(maturity),
            ql.Period(6, ql.Months),
            cal,
            ql.Unadjusted,
            ql.Unadjusted,
            ql.DateGeneration.Backward,
            eom,
        )
        dc = ql.ActualActual(ql.ActualActual.Bond, schedule)
        p0, p1 = (
            ql.BondFunctions.cleanPrice(
                ql.FixedRateBond(1, 100.0, schedule, [c], dc, ql.Following), handle.currentLink(), qd(settle)
            )
            for c in (0.0, 0.01)
        )
        rates.append(0.01 * (100.0 - p0) / (p1 - p0))
    set_evaluation_date(EXAMPLE_ASOF)
    return {"asof": PAR_DATE, "settle": settle.isoformat(), "maturities": list(PAR_MATURITIES), "rates": rates}


def _schedule_dates(bond: ql.FixedRateBond) -> List[ql.Date]:
    """Unadjusted accrual dates of a QuantLib bond's coupons."""

    coupons = [c for c in map(ql.as_coupon, bond.cashflows()) if c is not None]
    return [coupons[0].accrualStartDate()] + [c.accrualEndDate() for c in coupons]


def _case(case: Dict[str, object], ust: ql.YieldTermStructureHandle, sofr: ql.YieldTermStructureHandle) -> dict:
    bond = case_bond(case["bond"])
    settle = dt.date.fromisoformat(str(case.get("settle", SETTLE)))
    s = qd(settle)
    freq = FREQUENCY[bond.frequency]
    following, dc_f = ql_bond(bond, settle, ql.Following)
    street, dc_s = ql_bond(bond, settle, ql.Unadjusted)
    true: Optional[Dict[str, list]] = None if case.get("street_only") else _yield_block(following, dc_f, freq, s)
    z_spreads, asw = [], []
    for price in PRICES:
        bp = ql.BondPrice(price, ql.BondPrice.Clean)
        z_spreads.append(
            ql.BondFunctions.zSpread(
                following, bp, ust.currentLink(), ql.Actual365Fixed(), ql.Continuous, ql.Annual, s, ACCURACY, 1000, 0.0
            )
        )
        asw.append(_asset_swap_spread(following, bond, settle, price, sofr))
    return {
        **case,
        "settle": settle.isoformat(),
        "schedule": [pyd(d).isoformat() for d in _schedule_dates(following)],
        "cashflows": [[pyd(cf.date()).isoformat(), cf.amount()] for cf in following.cashflows() if cf.date() > s],
        "accrued": ql.BondFunctions.accruedAmount(following, s),
        "true": true,
        "street": _yield_block(street, dc_s, freq, s),
        "curve_clean": ql.BondFunctions.cleanPrice(following, ust.currentLink(), s),
        "z_spread": z_spreads,
        "asw": asw,
    }


def generate() -> dict:
    set_evaluation_date(EXAMPLE_ASOF)
    ust = ql_curve(ust_curve())
    sofr = ql_curve(example_curve())
    first = case_bond(CASES[0]["bond"])
    reference, _ = ql_bond(first, dt.date.fromisoformat(SETTLE), ql.Following)
    settlements = [[d, pyd(reference.settlementDate(qd(dt.date.fromisoformat(d)))).isoformat()] for d in TRADE_DATES]
    return {
        "asof": EXAMPLE_ASOF.isoformat(),
        "yields": list(YIELDS),
        "prices": list(PRICES),
        "settlements": settlements,
        "cases": [_case(c, ust, sofr) for c in CASES],
        "swap_spreads": _swap_spreads(),
        "par_yields": _par_yields(),
    }
