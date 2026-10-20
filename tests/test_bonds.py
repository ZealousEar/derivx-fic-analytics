"""Bond calculator: QuantLib golden values, schedules and day counts, yield
and spread properties, positions and the Streamlit tab."""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from derivx.rates.bonds import (
    FixedRateBond,
    benchmark_bond,
    bill_bond_equivalent_yield,
    bill_discount_yield,
    bill_money_market_yield,
    bill_price_from_bey,
    bill_price_from_discount,
    bond_position,
    format_32nds,
    ois_par_rate,
    par_yield,
    parse_32nds,
)
from derivx.rates.conventions import Calendar, add_months, make_schedule
from derivx.rates.curves import Market, bootstrap, flat_curve
from derivx.rates.examples import EXAMPLE_ASOF, example_curve, example_market, sofr_example_rows, ust_curve
from derivx.rates.quotes import clean_quotes
from derivx.risk.sensitivities import key_rate_pv01, parallel_pv01

ROOT = Path(__file__).resolve().parents[1]
D = dt.date.fromisoformat
SPOT = D("2026-10-20")
NOTIONAL = 10_000_000.0


@lru_cache(maxsize=None)
def _market() -> Market:
    return example_market()


def case_bond(spec) -> FixedRateBond:
    """Mirror of tools/golden/bonds.case_bond."""

    dates = {k: D(v) for k, v in spec.items() if k in ("maturity", "dated", "first_coupon")}
    return FixedRateBond(**{**spec, **dates})


@lru_cache(maxsize=None)
def swap_curve(conventions: str):
    """Mirror of tools/golden/bonds.swap_curve."""

    quotes, _, _ = clean_quotes(sofr_example_rows())
    return bootstrap(EXAMPLE_ASOF, quotes, conventions)


# --------------------------------------------------------------------------
# QuantLib reference values
# --------------------------------------------------------------------------


def test_bond_analytics_match_quantlib(golden):
    data = golden("bonds")
    market = _market()
    yields, prices = data["yields"], data["prices"]
    for case in data["cases"]:
        bond, settle, name = case_bond(case["bond"]), D(case["settle"]), case["name"]
        assert [d.isoformat() for d in bond.coupon_dates(settle)] == case["schedule"], name
        flows = bond.cashflows(settle)
        ours = [[p.isoformat(), c] for p, c in zip(flows["pay"], flows["coupon"])] + [
            [flows["pay"].iloc[-1].isoformat(), bond.face]
        ]
        assert [f[0] for f in ours] == [f[0] for f in case["cashflows"]], name
        assert np.allclose([f[1] for f in ours], [f[1] for f in case["cashflows"]], rtol=0, atol=1e-12), name
        assert bond.accrued(settle) == pytest.approx(case["accrued"], abs=1e-12), name

        for convention in ("street", "true"):
            ref = case[convention]
            if ref is None:
                assert convention == "true" and case["street_only"]
                continue
            for i, y in enumerate(yields):
                clean = bond.price_from_yield(y, settle, convention=convention)
                dirty = bond.price_from_yield(y, settle, clean=False, convention=convention)
                assert clean == pytest.approx(ref["clean"][i], abs=1e-9), (name, convention, y)
                assert dirty == pytest.approx(ref["dirty"][i], abs=1e-9), (name, convention, y)
                assert bond.macaulay_duration(y, settle, convention) == pytest.approx(ref["macaulay"][i], rel=1e-10)
                assert bond.modified_duration(y, settle, convention) == pytest.approx(ref["modified"][i], rel=1e-10)
                assert bond.convexity(y, settle, convention) == pytest.approx(ref["convexity"][i], rel=1e-10)
                assert bond.bps(y, settle, convention) == pytest.approx(ref["bps"][i], abs=1e-12)
                # QuantLib's DV01: -modified duration x dirty price x 1bp.
                ql_dv01 = -ref["modified"][i] * ref["dirty"][i] * 1e-4
                assert bond.dv01(y, settle, convention=convention) == pytest.approx(ql_dv01, rel=1e-10)
            for i, price in enumerate(prices):
                ytm = bond.yield_from_price(price, settle, convention=convention)
                assert ytm == pytest.approx(ref["yield"][i], abs=1e-10), (name, convention, price)

        govt = market.govt
        assert bond.price_from_curve(govt, settle) == pytest.approx(case["curve_clean"], abs=1e-9), name
        for i, price in enumerate(prices):
            assert bond.z_spread(price, govt, settle) == pytest.approx(case["z_spread"][i], abs=1e-10), name
            assert bond.asset_swap_spread(price, market, settle) == pytest.approx(case["asw"][i], abs=1e-10), name

    reference = case_bond(data["cases"][0]["bond"])
    for trade, settle in data["settlements"]:
        assert reference.settlement_date(D(trade)) == D(settle)


def test_spreads_on_other_swap_conventions_and_par_yields_match_quantlib(golden):
    data = golden("bonds")
    settle, govt = D(data["cases"][0]["settle"]), _market().govt
    for row in data["swap_spreads"]:
        curve = swap_curve(row["conventions"])
        bond, market = case_bond(data["cases"][row["case"]]["bond"]), Market(curve=curve, govt=govt)
        name = (row["conventions"], bond.label)
        assert ois_par_rate(curve, bond.maturity) == pytest.approx(row["ois_rate"], abs=1e-14), name
        for price, ref in zip(data["prices"], row["asw"]):
            assert bond.asset_swap_spread(price, market, settle) == pytest.approx(ref, abs=1e-13), name

    # Spot Apr 30: the curve's par bonds and the benchmarks pay on month ends.
    par = data["par_yields"]
    curve, spot = ust_curve(D(par["asof"])), D(par["settle"])
    for maturity, ref in zip(par["maturities"], par["rates"]):
        assert par_yield(curve, D(maturity), spot) == pytest.approx(ref, abs=1e-13), maturity
    note = benchmark_bond(curve, 2, spot)
    assert note.coupon_dates()[1:3] == [D("2026-10-31"), D("2027-04-30")]
    assert note.coupon == pytest.approx(par["rates"][0], abs=1e-13)


def _live_quantlib():
    ql = pytest.importorskip("QuantLib")
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    from golden import bonds as gb
    from golden.common import qd, set_evaluation_date

    return ql, gb, qd, set_evaluation_date


def _random_bond(rng: random.Random) -> tuple:
    """(bond, settle) with random maturity, coupon, conventions and an odd
    first coupon one time in three."""

    frequency = rng.choice((1, 2, 2, 4))
    day_count = rng.choice(("ACT/ACT ICMA", "ACT/ACT ICMA", "30/360"))
    maturity = EXAMPLE_ASOF + dt.timedelta(days=rng.randint(200, 30 * 365))
    if rng.random() < 0.2:  # month-end maturities exercise the end-of-month rule
        maturity = add_months(maturity, 0, eom=True)
    # Business-day settlement: on a weekend after an unadjusted coupon date
    # QuantLib's unadjusted-payment copy would already have paid that coupon.
    settle = Calendar("USGOV").adjust(
        EXAMPLE_ASOF + dt.timedelta(days=rng.randint(0, min(700, (maturity - EXAMPLE_ASOF).days - 40)))
    )
    dated = first = None
    later = [d for d in FixedRateBond(0.0, maturity, frequency=frequency).coupon_dates(settle) if d > settle]
    # An odd first coupon needs regular ones after it: QuantLib's ACT/ACT
    # takes a lone odd period as an odd *last* period (reference period
    # starting at the dated date), where ICMA uses the period ending at it.
    if rng.random() < 0.35 and len(later) > 2:
        dated = settle - dt.timedelta(days=rng.randint(0, 500))
        regular = [d for d in FixedRateBond(0.0, maturity, frequency=frequency).coupon_dates(dated) if d > dated]
        # For a long first coupon QuantLib steps its quasi-coupon dates back
        # from the first coupon, losing a clipped 29th/30th (see
        # test_odd_first_coupons); derivx keeps the bond's coupon day.
        clipped = maturity.day > 28 and not FixedRateBond(0.0, maturity).end_of_month
        if rng.random() < 0.5 and regular[1] < later[1] and not clipped:
            first = regular[1]  # long first coupon
    coupon = round(rng.uniform(0.0, 0.09), 5)
    return FixedRateBond(coupon, maturity, dated, first, frequency=frequency, day_count=day_count), settle


def test_random_bonds_match_live_quantlib():
    ql, gb, qd, set_evaluation_date = _live_quantlib()
    set_evaluation_date(EXAMPLE_ASOF)
    rng = random.Random(20261019)
    freqs = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly}
    checked = 0
    for _ in range(400):
        bond, settle = _random_bond(rng)
        s = qd(settle)
        y, price = rng.uniform(-0.005, 0.12), rng.uniform(80.0, 120.0)
        conventions = [("street", ql.Unadjusted)]
        if bond.cal.is_business_day(bond.maturity):  # QuantLib's true yield stops at maturity
            conventions.append(("true", ql.Following))
        for convention, payment in conventions:
            qb, dc = gb.ql_bond(bond, settle, payment)
            assert bond.accrued(settle) == pytest.approx(ql.BondFunctions.accruedAmount(qb, s), abs=1e-11), bond
            rate = ql.InterestRate(y, dc, ql.Compounded, freqs[bond.frequency])
            ref = ql.BondFunctions.cleanPrice(qb, rate, s)
            assert bond.price_from_yield(y, settle, convention=convention) == pytest.approx(ref, abs=1e-9), bond
            ref_mod = ql.BondFunctions.duration(qb, rate, ql.Duration.Modified, s)
            assert bond.modified_duration(y, settle, convention) == pytest.approx(ref_mod, rel=1e-9), bond
            ref_cx = ql.BondFunctions.convexity(qb, rate, s)
            assert bond.convexity(y, settle, convention) == pytest.approx(ref_cx, rel=1e-9), bond
            ref_y = ql.BondFunctions.bondYield(
                qb, ql.BondPrice(price, ql.BondPrice.Clean), dc, ql.Compounded, freqs[bond.frequency], s, 1e-15, 1000,
                0.05,
            )
            assert bond.yield_from_price(price, settle, convention=convention) == pytest.approx(ref_y, abs=1e-10)
            checked += 1
    assert checked > 600


def test_other_conventions_match_live_quantlib():
    """Every day count, monthly coupons, TARGET/LONDON calendars, MF and P
    payments, settlement lags 0-2 and settlement up to decades forward."""

    ql, gb, qd, set_evaluation_date = _live_quantlib()
    from golden.common import BDC, pyd

    set_evaluation_date(EXAMPLE_ASOF)
    rng = random.Random(7)
    freqs = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly, 12: ql.Monthly}
    day_counts = ("ACT/ACT ICMA", "30/360", "30E/360", "ACT/360", "ACT/365F", "ACT/ACT ISDA")
    checked = 0
    while checked < 150:
        frequency, calendar = rng.choice((1, 2, 4, 12)), rng.choice(("USGOV", "TARGET", "LONDON"))
        maturity = EXAMPLE_ASOF + dt.timedelta(days=rng.randint(400, 50 * 365))
        if rng.random() < 0.2:
            maturity = add_months(maturity, 0, eom=True)
        elif maturity.day > 28:  # clipped-day long first coupons differ by design (test_odd_first_coupons)
            maturity = maturity.replace(day=28)
        settle = EXAMPLE_ASOF + dt.timedelta(days=rng.randint(0, (maturity - EXAMPLE_ASOF).days - 200))
        settle = Calendar(calendar).adjust(settle)
        dated = first = None
        if rng.random() < 0.4:
            dated = settle - dt.timedelta(days=rng.randint(0, 5 * 365))
            regular = [d for d in FixedRateBond(0.0, maturity, frequency=frequency).coupon_dates(dated) if d > dated]
            first = regular[1] if len(regular) > 2 and rng.random() < 0.5 else None
        bond = FixedRateBond(
            round(rng.uniform(0.0, 0.12), 5), maturity, dated, first, frequency=frequency,
            day_count=rng.choice(day_counts), calendar=calendar, settle_lag=rng.randint(0, 2),
            payment_bdc=rng.choice(("F", "MF", "P")),
        )
        if not bond.cal.is_business_day(maturity) or (dated is not None and len(bond.coupon_dates()) < 3):
            continue  # QuantLib's true yield stops at maturity; lone odd periods differ by design
        s = qd(settle)
        qb, dc = gb.ql_bond(bond, settle, BDC[bond.payment_bdc])
        cashflows = {}
        for cf in qb.cashflows():
            if cf.date() > s:
                cashflows[pyd(cf.date())] = cashflows.get(pyd(cf.date()), 0.0) + cf.amount()
        flows = bond.cashflows(settle)
        assert dict(zip(flows["pay"], flows["amount"])) == pytest.approx(cashflows, abs=1e-11), bond
        assert bond.accrued(settle) == pytest.approx(ql.BondFunctions.accruedAmount(qb, s), abs=1e-11), bond
        y, price = rng.uniform(-0.03, 0.15), rng.uniform(60.0, 140.0)
        rate = ql.InterestRate(y, dc, ql.Compounded, freqs[frequency])
        assert bond.price_from_yield(y, settle, convention="true") == pytest.approx(
            ql.BondFunctions.cleanPrice(qb, rate, s), abs=1e-9
        ), bond
        assert bond.modified_duration(y, settle, "true") == pytest.approx(
            ql.BondFunctions.duration(qb, rate, ql.Duration.Modified, s), rel=1e-11
        ), bond
        assert bond.convexity(y, settle, "true") == pytest.approx(ql.BondFunctions.convexity(qb, rate, s), rel=1e-11)
        ref_y = ql.BondFunctions.bondYield(
            qb, ql.BondPrice(price, ql.BondPrice.Clean), dc, ql.Compounded, freqs[frequency], s, 1e-15, 1000, 0.05
        )
        assert bond.yield_from_price(price, settle, convention="true") == pytest.approx(ref_y, abs=1e-12), bond
        assert bond.settlement_date(settle) == pyd(qb.settlementDate(s)), bond
        checked += 1


# --------------------------------------------------------------------------
# Schedules, day counts and accrued interest
# --------------------------------------------------------------------------


def test_end_of_month_schedules():
    aug = FixedRateBond(0.0375, D("2028-08-31"))
    assert aug.end_of_month
    assert aug.coupon_dates(SPOT) == [D(d) for d in ("2026-02-28", "2026-08-31", "2027-02-28", "2027-08-31",
                                                     "2028-02-29", "2028-08-31")]
    assert aug.coupons(SPOT)[0].pay == D("2027-03-01")  # Sunday Feb 28 pays Monday
    feb = FixedRateBond(0.04, D("2031-02-28"), dated=D("2026-02-28"))
    assert feb.coupon_dates()[:4] == [D("2026-02-28"), D("2026-08-31"), D("2027-02-28"), D("2027-08-31")]
    assert feb.coupon_dates() == make_schedule(D("2026-02-28"), D("2031-02-28"), 6, "USGOV", "U", "U", eom=True)
    not_eom = FixedRateBond(0.04, D("2031-02-28"), dated=D("2026-02-28"), eom=False)
    assert not_eom.coupon_dates()[1] == D("2026-08-28")
    with pytest.raises(ValueError, match="end-of-month"):
        FixedRateBond(0.04, D("2031-02-27"), eom=True)


def test_odd_first_coupons():
    short = FixedRateBond(0.045, D("2031-11-14"), dated=D("2026-10-01"))
    first = short.coupons(SPOT)[0]
    assert (first.start, first.end) == (D("2026-10-01"), D("2026-11-14"))
    assert first.accrual == pytest.approx(44 / (2 * 184))  # 44 days of the May 14 - Nov 14 quasi-period
    assert short.coupon_dates() == make_schedule(D("2026-10-01"), D("2031-11-14"), 6, "USGOV", "U", "U")

    long = FixedRateBond(0.05, D("2036-05-15"), dated=D("2026-07-01"), first_coupon=D("2027-05-15"))
    first = long.coupons(SPOT)[0]
    assert (first.start, first.end) == (D("2026-07-01"), D("2027-05-15"))
    assert first.accrual == pytest.approx(137 / (2 * 184) + 0.5)
    assert long.accrued(D("2026-12-01")) == pytest.approx(5.0 * (137 / (2 * 184) + 16 / (2 * 181)))
    assert long.year_fraction(D("2026-07-01"), D("2027-05-15")) == pytest.approx(first.accrual)

    # Quasi-coupon dates keep the bond's coupon day: an Aug 30 bond's period
    # before its Feb 28 coupon starts on Aug 30 (QuantLib: Aug 28).
    clipped = FixedRateBond(0.04, D("2031-08-30"), dated=D("2026-04-15"), first_coupon=D("2027-02-28"))
    assert clipped.coupons(SPOT)[0].accrual == pytest.approx(137 / (2 * 183) + 0.5)
    short_clipped = FixedRateBond(0.04, D("2028-08-30"), dated=D("2027-12-01"))  # first coupon Feb 29, 2028
    assert short_clipped.coupons(SPOT)[0].accrual == pytest.approx(90 / (2 * 183))  # Aug 30 - Feb 29 (QuantLib: Aug 29)
    # A lone odd period is measured against the regular period ending at
    # maturity (QuantLib: the one starting at the dated date).
    lone = FixedRateBond(0.04, D("2027-10-25"), dated=D("2027-05-17"))
    assert lone.coupons(SPOT)[0].accrual == pytest.approx(161 / (2 * 183))

    with pytest.raises(ValueError, match="regular coupon date"):
        FixedRateBond(0.05, D("2036-05-15"), dated=D("2026-07-01"), first_coupon=D("2027-05-14"))
    with pytest.raises(ValueError, match="needs a dated date"):
        FixedRateBond(0.05, D("2036-05-15"), first_coupon=D("2027-05-15"))


def test_regular_coupons_settlement_and_accrued():
    bond = FixedRateBond(0.04375, D("2035-11-15"))
    coupons = bond.coupons(SPOT)
    assert all(c.amount == pytest.approx(100 * 0.04375 / 2, abs=1e-14) for c in coupons)
    assert coupons[1].end == D("2027-05-15") and coupons[1].pay == D("2027-05-17")  # Saturday
    assert bond.settlement_date(D("2026-11-10")) == D("2026-11-12")  # Veterans Day
    assert bond.accrued(SPOT) == pytest.approx(100 * 0.04375 / 2 * 158 / 184)
    # The Sunday Nov 15 coupon is paid on settlement Monday: the seller keeps
    # it, and the next coupon has accrued one day.
    monday = D("2026-11-16")
    assert bond.coupons(monday)[0].start == D("2026-11-15")
    assert bond.accrued(monday) == pytest.approx(100 * 0.04375 / 2 / 181)
    assert bond.cash(D("2026-11-15"), monday, NOTIONAL) == pytest.approx(NOTIONAL * 0.04375 / 2)
    flows = bond.cashflows(SPOT)
    assert flows["principal"].sum() == 100.0 and flows["principal"].iloc[-1] == 100.0
    assert flows["amount"].sum() == pytest.approx(100.0 + 100 * 0.04375 / 2 * 19)


def test_when_issued_settlement_before_the_dated_date():
    bond = FixedRateBond(0.045, D("2031-11-14"), dated=D("2026-11-01"))
    assert bond.accrued(SPOT) == 0.0
    y = 0.045
    assert bond.price_from_yield(y, SPOT) == bond.price_from_yield(y, SPOT, clean=False)
    assert bond.yield_from_price(bond.price_from_yield(y, SPOT), SPOT) == pytest.approx(y, abs=1e-13)


def test_invalid_bonds_are_rejected_clearly():
    for kwargs, message in [
        (dict(frequency=3), "frequency"),
        (dict(day_count="ACT/999"), "day count"),
        (dict(calendar="MARS"), "calendar"),
        (dict(payment_bdc="X"), "payment_bdc"),
        (dict(face=0.0), "face"),
        (dict(dated=D("2036-01-01")), "dated date"),
    ]:
        with pytest.raises(ValueError, match=message):
            FixedRateBond(0.04, D("2035-11-15"), **kwargs)
    with pytest.raises(ValueError, match="coupon"):
        FixedRateBond(-0.01, D("2035-11-15"))
    bond = FixedRateBond(0.04, D("2027-05-17"))
    with pytest.raises(ValueError, match="no cash flows"):
        bond.price_from_yield(0.04, D("2027-05-17"))
    with pytest.raises(ValueError, match="positive"):
        bond.yield_from_price(-50.0, SPOT)
    with pytest.raises(ValueError, match="yield must be above"):
        bond.price_from_yield(-2.5, SPOT)
    with pytest.raises(ValueError, match="convention"):
        bond.price_from_yield(0.04, SPOT, convention="bloomberg")
    with pytest.raises(ValueError, match="needs `since`"):
        bond.coupon_dates()


# --------------------------------------------------------------------------
# Yield, duration and risk properties
# --------------------------------------------------------------------------

PROPERTY_BONDS = [
    FixedRateBond(0.04375, D("2035-11-15")),
    FixedRateBond(0.0375, D("2028-08-31")),
    FixedRateBond(0.05, D("2036-05-15"), dated=D("2026-07-01"), first_coupon=D("2027-05-15")),
    FixedRateBond(0.0525, D("2033-06-15"), day_count="30/360"),
    FixedRateBond(0.061, D("2034-03-15"), dated=D("2026-09-01"), day_count="30/360", frequency=1),
    FixedRateBond(0.06, D("2030-12-16"), frequency=4),
    FixedRateBond(0.03, D("2029-01-15"), frequency=12),
    FixedRateBond(0.0, D("2031-08-15")),
]


@pytest.mark.parametrize("bond", PROPERTY_BONDS, ids=lambda b: f"{b.label} {b.day_count} f{b.frequency}")
def test_price_yield_round_trip(bond):
    for settle in (SPOT, D("2027-03-09"), D("2028-02-29")):
        for convention in ("street", "true"):
            for y in (-0.01, 0.0, 0.0001, 0.045, 0.15):
                for clean in (True, False):
                    price = bond.price_from_yield(y, settle, clean, convention)
                    assert bond.yield_from_price(price, settle, clean, convention) == pytest.approx(y, abs=1e-12)


@pytest.mark.parametrize("frequency", [1, 2, 4, 12])
@pytest.mark.parametrize("day_count", ["ACT/ACT ICMA", "30/360"])
def test_coupon_equal_to_yield_prices_at_par_on_a_coupon_date(frequency, day_count):
    # 30/360 month-end coupons are unequal (Aug 31 - Feb 28 is 178 days, Feb
    # 28 - Aug 31 is 183), so those bonds are not at par.
    month_ends = (D("2031-08-31"), D("2029-02-28")) if day_count == "ACT/ACT ICMA" else ()
    for maturity in (D("2036-05-15"), *month_ends):
        bond = FixedRateBond(0.0525, maturity, frequency=frequency, day_count=day_count)
        settle = next(d for d in bond.coupon_dates(SPOT) if d > SPOT and bond.cal.is_business_day(d))
        assert bond.accrued(settle) == 0.0
        assert bond.price_from_yield(0.0525, settle) == pytest.approx(100.0, abs=1e-12)
    if day_count == "30/360" and frequency == 2:
        coupons = FixedRateBond(0.0525, D("2031-08-31"), day_count=day_count).coupons(SPOT)
        assert [round(c.accrual * 360) for c in coupons[:3]] == [178, 183, 179]  # to Feb 28, Aug 31, Feb 29


def test_risk_numbers_match_finite_differences():
    h, h2 = 1e-6, 1e-4
    for bond in PROPERTY_BONDS:
        for convention in ("street", "true"):
            y = 0.047
            dirty = lambda v: bond.price_from_yield(v, SPOT, clean=False, convention=convention)  # noqa: E731
            slope = (dirty(y + h) - dirty(y - h)) / (2 * h)
            curvature = (dirty(y + h2) - 2 * dirty(y) + dirty(y - h2)) / h2**2
            assert bond.dv01(y, SPOT, convention=convention) == pytest.approx(slope * 1e-4, rel=1e-7)
            assert bond.dv01(y, SPOT, convention=convention) < 0
            assert bond.dv01(y, SPOT, NOTIONAL, convention) == pytest.approx(slope * 1e-4 * NOTIONAL / 100, rel=1e-7)
            assert bond.convexity(y, SPOT, convention) == pytest.approx(curvature / dirty(y), rel=1e-6)
            mac, mod = bond.macaulay_duration(y, SPOT, convention), bond.modified_duration(y, SPOT, convention)
            assert mac == pytest.approx(mod * (1 + y / bond.frequency), rel=1e-14)
            assert bond.dv01(y, SPOT, convention=convention) * 1e4 * bond.yield_value_32nd(y, SPOT, convention) == (
                pytest.approx(-1 / 32, rel=1e-12)
            )
    bond = PROPERTY_BONDS[0]
    richer = FixedRateBond(bond.coupon + 1e-4, bond.maturity)  # BPS: +1bp of coupon at the same yield
    gain = richer.price_from_yield(0.047, SPOT, clean=False) - bond.price_from_yield(0.047, SPOT, clean=False)
    assert bond.bps(0.047, SPOT) == pytest.approx(gain, rel=1e-10)
    zero = FixedRateBond(0.0, D("2031-08-15"))
    assert zero.macaulay_duration(0.04, SPOT) == pytest.approx(zero.yield_times(SPOT)[-1], rel=1e-14)


def test_street_and_true_yields_differ_only_by_payment_dates():
    bond = FixedRateBond(0.04375, D("2035-11-15"))  # several coupons fall on weekends
    street, true = bond.yield_times(SPOT, "street"), bond.yield_times(SPOT, "true")
    assert np.all(true >= street) and np.any(true > street)
    assert np.allclose(np.diff(street) * 2, 1.0, atol=1e-14)  # whole periods after the first
    assert street[0] * 2 == pytest.approx(26 / 184)
    on_business_days = FixedRateBond(0.04, D("2027-04-20"), dated=D("2026-10-20"))
    assert np.array_equal(on_business_days.yield_times(SPOT, "street"), on_business_days.yield_times(SPOT, "true"))


# --------------------------------------------------------------------------
# Curves and spreads
# --------------------------------------------------------------------------


@pytest.mark.parametrize("interpolation", ["loglinear", "linear_zero", "monotone_convex"])
@pytest.mark.parametrize("day", [EXAMPLE_ASOF, D("2026-04-29"), D("2026-06-29")], ids=str)
def test_curves_own_par_bonds_have_zero_z_spread(interpolation, day):
    # Spot on Apr 30 / Jun 30: month-end par bonds pay Oct 31 / Dec 31.
    curve = ust_curve(day, interpolation=interpolation)
    bonds = [inst for inst in curve.instruments if inst.kind == "BOND"]
    assert len(bonds) == 7
    for inst in bonds:
        bond = FixedRateBond(inst.quote.rate, inst.end, dated=inst.start)
        assert bond.price_from_curve(curve, inst.start) == pytest.approx(100.0, abs=1e-10)
        assert abs(bond.z_spread(100.0, curve, inst.start)) < 1e-12
        assert par_yield(curve, inst.end, inst.start) == pytest.approx(inst.quote.rate, abs=1e-13)
        years = int(inst.tenor.years)
        assert benchmark_bond(curve, years, inst.start).coupon == pytest.approx(inst.quote.rate, abs=1e-13)


def test_ois_par_rate_reprices_the_swap_curve_quotes():
    curve = example_curve()
    checked = 0
    for inst in curve.instruments:
        if inst.kind == "OIS" and inst.tenor.unit in ("M", "Y"):
            assert ois_par_rate(curve, add_months(inst.start, inst.tenor.months)) == pytest.approx(
                inst.quote.rate, abs=1e-13
            )
            checked += 1
    assert checked >= 10
    with pytest.raises(ValueError, match="spot"):
        ois_par_rate(curve, EXAMPLE_ASOF)


def test_spreads_on_the_example_market():
    market = _market()
    govt = market.govt
    bond = benchmark_bond(govt, 10, SPOT)
    assert bond.price_from_curve(govt, SPOT) == pytest.approx(100.0, abs=1e-10)
    assert abs(bond.z_spread(100.0, govt, SPOT)) < 1e-12
    assert abs(bond.g_spread(100.0, market, SPOT)) < 1e-12  # par bond: street yield = coupon = par yield
    assert bond.i_spread(100.0, market, SPOT) == pytest.approx(bond.coupon - ois_par_rate(market.curve, bond.maturity))

    swap_price = bond.price_from_curve(market.curve, SPOT)
    assert abs(bond.asset_swap_spread(swap_price, market, SPOT)) < 1e-13
    asw = [bond.asset_swap_spread(swap_price - k, market, SPOT) for k in range(4)]
    assert np.allclose(np.diff(asw), asw[1], rtol=1e-12)  # linear in price: 1 point / (face x annuity)
    assert asw[1] > 0  # cheaper than the swap curve: positive spread

    cheap = [bond.z_spread(p, govt, SPOT) for p in (101.0, 100.0, 99.0)]
    assert cheap[0] < cheap[1] < cheap[2]
    g = bond.g_spread(99.0, market, SPOT)
    assert g == pytest.approx(bond.yield_from_price(99.0, SPOT) - par_yield(govt, bond.maturity, SPOT))

    with pytest.raises(ValueError, match="before the curve date"):
        bond.z_spread(99.0, govt, EXAMPLE_ASOF - dt.timedelta(days=1))
    no_govt = Market(curve=market.curve)
    with pytest.raises(ValueError, match="Treasury"):
        bond.pv(no_govt)
    with pytest.raises(ValueError, match="Treasury"):
        bond.g_spread(99.0, no_govt, SPOT)


def test_z_spread_shifts_continuous_zero_rates():
    curve = flat_curve(EXAMPLE_ASOF, 0.04)
    bond = FixedRateBond(0.05, D("2031-11-14"), dated=D("2026-10-01"))
    price = bond.price_from_curve(curve, SPOT, z_spread=0.0125)
    assert price == pytest.approx(bond.price_from_curve(flat_curve(EXAMPLE_ASOF, 0.0525), SPOT), abs=1e-12)
    assert bond.z_spread(price, curve, SPOT) == pytest.approx(0.0125, abs=1e-13)


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------


def test_bond_position_value_cash_and_key_rate_risk():
    market = _market()
    govt = market.govt
    bond = benchmark_bond(govt, 10, SPOT)
    z = bond.z_spread(99.0, govt, SPOT)
    position = bond_position(bond, NOTIONAL, z)
    assert position.kind == "bond" and position.name.startswith("LONG")
    dirty = 99.0 + bond.accrued(SPOT)
    t = govt.time(SPOT)
    expected = NOTIONAL / 100 * dirty * govt.df(SPOT) * np.exp(-z * t)  # nothing pays between asof and spot
    assert position.pv(market) == pytest.approx(expected, rel=1e-12)

    rows = key_rate_pv01(position.pv, market, which="govt")
    pv01 = parallel_pv01(position.pv, market, which="govt")
    assert pv01 < 0
    assert sum(r["pv01"] for r in rows) == pytest.approx(pv01, rel=1e-3)
    biggest = max(rows, key=lambda r: abs(r["pv01"]))
    assert biggest["tenor"] == "10Y"
    assert pv01 == pytest.approx(bond.dv01(bond.yield_from_price(99.0, SPOT), SPOT, NOTIONAL), rel=0.03)

    short = bond_position(bond, -NOTIONAL, z)
    assert short.name.startswith("SHORT")
    assert short.pv(market) == pytest.approx(-position.pv(market))

    year = EXAMPLE_ASOF + dt.timedelta(days=366)
    assert position.cash(market, EXAMPLE_ASOF, year) == pytest.approx(NOTIONAL * bond.coupon)
    assert position.cash(market, EXAMPLE_ASOF, bond.maturity) == pytest.approx(NOTIONAL * (1 + bond.coupon * 10))
    assert bond.pv(market.replace(curve=market.curve), z, NOTIONAL) == pytest.approx(position.pv(market))
    matured = FixedRateBond(0.04, D("2026-04-15"))
    assert matured.pv(market) == 0.0


# --------------------------------------------------------------------------
# Price formats and T-bills
# --------------------------------------------------------------------------


def test_parse_32nds():
    cases = {
        "99-16": 99.5,
        "99-16+": 99 + 16.5 / 32,
        "99-162": 99 + 16.25 / 32,
        "99-166": 99 + 16.75 / 32,
        " 101-047 ": 101 + 4.875 / 32,
        "100-00": 100.0,
        "99-1": 99 + 1 / 32,
        "99-31+": 99 + 31.5 / 32,
        "99.515625": 99.515625,
        "100": 100.0,
    }
    for text, value in cases.items():
        assert parse_32nds(text) == value, text
    for bad in ("99-32", "99-168", "99-", "abc", "", "-1", "0", "99-16++", "nan"):
        with pytest.raises(ValueError):
            parse_32nds(bad)


def test_format_32nds_round_trips():
    assert format_32nds(99 + 16.5 / 32) == "99-16+"
    assert format_32nds(99 + 16.25 / 32) == "99-162"
    assert format_32nds(99.5) == "99-16"
    assert format_32nds(99.999999) == "100-00"
    assert format_32nds(0.0) == "0-00"
    for k in range(95 * 256, 105 * 256):
        assert parse_32nds(format_32nds(k / 256)) == k / 256
    with pytest.raises(ValueError):
        format_32nds(float("nan"))


def test_bill_conventions():
    price = bill_price_from_discount(0.05, 91)
    assert price == pytest.approx(100 * (1 - 0.05 * 91 / 360))
    assert bill_discount_yield(price, 91) == pytest.approx(0.05)
    assert bill_bond_equivalent_yield(price, 91) == pytest.approx(365 * 0.05 / (360 - 0.05 * 91))
    assert bill_money_market_yield(price, 91) == pytest.approx(360 * 0.05 / (360 - 0.05 * 91))
    for days in (28, 91, 182, 183, 273, 364):
        for bey in (-0.002, 0.01, 0.0425, 0.09):
            assert bill_bond_equivalent_yield(bill_price_from_bey(bey, days), days) == pytest.approx(bey, abs=1e-13)
    # Beyond half a year the semiannual BEY sits below the simple ACT/365 rate.
    p = bill_price_from_discount(0.05, 364)
    assert bill_bond_equivalent_yield(p, 364) < (100 / p - 1) * 365 / 364
    with pytest.raises(ValueError, match="positive"):
        bill_discount_yield(99.0, 0)
    for to_yield in (bill_discount_yield, bill_bond_equivalent_yield, bill_money_market_yield):
        with pytest.raises(ValueError, match="price must be positive"):
            to_yield(0.0, 91)


# --------------------------------------------------------------------------
# Streamlit tab
# --------------------------------------------------------------------------

APP = """
import streamlit as st
from derivx.rates.curves import Market
from derivx.rates.examples import example_market

from derivx.ui.rates_bonds import render

market = example_market()
if st.session_state.get("bond_test_no_govt"):
    market = Market(curve=market.curve)
st.session_state["bond_test_positions"] = [p.name for p in render(market)]
"""


def _app(**state) -> AppTest:
    at = AppTest.from_string(APP, default_timeout=120)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_bond_tab_default_inputs():
    at = _app()
    assert not at.error
    assert at.session_state["bond_test_positions"] == ["LONG UST 10Y 5.150% 20-OCT-2036"]
    assert _metric(at, "CLEAN (32NDS)") == "100-00"
    assert _metric(at, "YIELD (%, STREET)") == "5.1500"
    assert float(_metric(at, "DV01 ($, +1BP)").replace(",", "")) < 0
    spreads = at.dataframe[0].value
    assert abs(spreads["SPREAD (BP)"].iloc[0]) < 1e-8 and abs(spreads["SPREAD (BP)"].iloc[1]) < 1e-8
    assert len(at.dataframe) == 2 and len(at.dataframe[1].value) == 20
    charts = [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]
    assert len(charts) == 2
    for fig in charts:
        layout = fig["layout"]
        axes = [name for name in layout if name.startswith(("xaxis", "yaxis"))]
        assert set(axes) == set(layout["meta"]["zoom"]["axes"]) == {"xaxis", "yaxis"}
        for name in axes:
            lo, hi = layout[name]["minallowed"], layout[name]["maxallowed"]
            assert math.isfinite(lo) and math.isfinite(hi) and lo < hi and layout[name]["range"] == [lo, hi]
    widget_keys = [w.key for kind in ("radio", "selectbox", "number_input", "text_input", "date_input") for w in
                   getattr(at, kind)]
    assert widget_keys and all(key.startswith("bond_") for key in widget_keys)
    for tenor in (2, 5, 30):
        at.selectbox(key="bond_tenor").set_value(tenor).run()
        assert not at.exception and not at.error
        assert at.session_state["bond_test_positions"][0].startswith(f"LONG UST {tenor}Y")


def test_bond_tab_price_and_yield_inputs():
    at = _app()
    at.text_input(key="bond_price_text").set_value("99-16+").run()
    assert _metric(at, "CLEAN PRICE (PER 100)") == "99.515625" and _metric(at, "CLEAN (32NDS)") == "99-16+"
    at.date_input(key="bond_settle").set_value(D("2026-11-20")).run()  # the quote survives a new settlement
    assert not at.exception and _metric(at, "CLEAN PRICE (PER 100)") == "99.515625"
    for bad in ("99-32", "abc", "5000"):
        at.text_input(key="bond_price_text").set_value(bad).run()
        assert not at.exception and at.error
        assert at.session_state["bond_test_positions"] == []
    at.radio(key="bond_input_mode").set_value("Yield").run()
    at.number_input(key="bond_yield_pct").set_value(5.5).run()
    assert not at.exception and not at.error
    assert float(_metric(at, "CLEAN PRICE (PER 100)")) < 100.0
    at.radio(key="bond_side").set_value("Short").run()
    assert at.session_state["bond_test_positions"][0].startswith("SHORT")
    assert float(_metric(at, "DV01 ($, +1BP)").replace(",", "")) > 0


def test_bond_tab_custom_bonds_and_edge_inputs():
    at = _app()
    at.radio(key="bond_source").set_value("Custom").run()
    assert not at.exception and not at.error
    assert at.session_state["bond_test_positions"] == ["LONG 4.250% 15-OCT-2035 BOND"]
    at.checkbox(key="bond_odd").check().run()
    at.radio(key="bond_first_kind").set_value("Long").run()
    at.selectbox(key="bond_day_count").set_value("30/360").run()
    at.selectbox(key="bond_frequency").set_value(1).run()
    assert not at.exception and not at.error
    assert len(at.session_state["bond_test_positions"]) == 1

    at.date_input(key="bond_dated").set_value(D("2036-01-01")).run()
    assert not at.exception and at.error and at.session_state["bond_test_positions"] == []
    at.checkbox(key="bond_odd").uncheck().run()
    at.date_input(key="bond_maturity").set_value(D("2026-10-21")).run()
    assert not at.exception and not at.error  # a one-day bond still prices
    at.date_input(key="bond_settle").set_value(D("2026-10-24")).run()
    assert not at.exception and at.error  # settlement after maturity
    at.date_input(key="bond_maturity").set_value(D("2035-10-15")).run()
    assert not at.exception and at.warning  # Saturday settlement
    at.number_input(key="bond_coupon_pct").set_value(0.0).run()
    assert not at.exception and not at.error


def test_bond_tab_keeps_the_quote_through_errors_and_mode_switches():
    at = _app()
    at.text_input(key="bond_price_text").set_value("99-16+").run()
    at.date_input(key="bond_settle").set_value(D("2040-01-01")).run()  # after maturity: the quote is not drawn
    assert at.error and at.session_state["bond_test_positions"] == []
    assert _metric(at, "PRICE (PER 100)") == f"{100 * (1 - 0.04 * 91 / 360):.6f}"  # the T-bill calculator still shows
    at.date_input(key="bond_settle").set_value(SPOT).run()
    assert not at.error and _metric(at, "CLEAN (32NDS)") == "99-16+"

    at.radio(key="bond_input_mode").set_value("Yield").run()  # the price shows as its yield
    y = float(_metric(at, "YIELD (%, STREET)"))
    assert at.number_input(key="bond_yield_pct").value == pytest.approx(y, abs=1e-4)
    assert _metric(at, "CLEAN PRICE (PER 100)") == "99.515625"
    at.radio(key="bond_input_mode").set_value("Price").run()  # untouched: the typed text comes back
    assert at.text_input(key="bond_price_text").value == "99-16+"
    at.radio(key="bond_input_mode").set_value("Yield").run()
    at.number_input(key="bond_yield_pct").set_value(5.5).run()
    at.radio(key="bond_input_mode").set_value("Price").run()
    assert not at.error and _metric(at, "YIELD (%, STREET)") == "5.5000"


def test_bond_tab_coupon_on_february_29():
    at = _app()
    at.radio(key="bond_source").set_value("Custom").run()
    at.date_input(key="bond_maturity").set_value(D("2028-08-29")).run()  # pays Feb 29, 2028
    assert not at.exception and not at.error
    assert any("Feb 29 / Aug 29" in c.value for c in at.caption)


def test_bond_tab_without_treasury_curve_and_bill_calculator():
    at = _app(bond_test_no_govt=True)
    assert at.error and at.session_state["bond_test_positions"] == []
    assert _metric(at, "PRICE (PER 100)") == f"{100 * (1 - 0.04 * 91 / 360):.6f}"  # the T-bill calculator still works
    at = _app()
    at.radio(key="bond_bill_mode").set_value("Price").run()
    at.number_input(key="bond_bill_price").set_value(97.5).run()
    assert not at.exception
    assert _metric(at, "PRICE (PER 100)") == "97.500000"
    at.radio(key="bond_bill_mode").set_value("Discount yield").run()
    at.number_input(key="bond_bill_days").set_value(364).run()
    assert not at.exception and float(_metric(at, "BOND-EQUIVALENT (%)")) > 4.0
