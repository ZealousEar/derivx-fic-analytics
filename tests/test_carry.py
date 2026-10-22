"""Carry and roll-down: a hand-computed bond example, coupons inside the
horizon, par Treasuries on the static curve, and swaps (forward-starting
par minus spot par) against discount-factor formulas and full revaluation
on the scenario engine's rolled market."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from derivx.rates import carry
from derivx.rates.bonds import FixedRateBond, benchmark_bond
from derivx.rates.examples import example_market
from derivx.rates.swaps import VanillaSwap
from derivx.risk.scenarios import carry_market

D = dt.date.fromisoformat


@pytest.fixture(scope="module")
def market():
    return example_market()


def test_hand_computed_bond_carry_and_roll():
    """4% semiannual 10Y note bought at 100 on a coupon date (2026-05-15),
    3.5% repo, to 2026-08-15: 92 of the period's 184 days."""

    bond = FixedRateBond(0.04, D("2036-05-15"))
    s0, s1, repo = D("2026-05-15"), D("2026-08-15"), 0.035
    result = carry.bond_carry(bond, s0, s1, 100.0, repo, roll_yield=0.0395)
    assert result.ytm == pytest.approx(0.04, abs=1e-13) and result.days == 92
    assert result.income == pytest.approx(1.0, abs=1e-12)  # 100 x 4% / 2 x 92/184
    assert result.financing == pytest.approx(100.0 * repo * 92 / 360, abs=1e-12)
    dirty1 = 100.0 * math.sqrt(1.02)  # half a period at the unchanged yield: par grows by (1 + y/2)^(1/2)
    assert result.pull == pytest.approx(dirty1 - 1.0 - 100.0, abs=1e-10)
    assert result.carry == pytest.approx(dirty1 - 100.0 - 100.0 * repo * 92 / 360, abs=1e-10)
    flows = [2.0] * 19 + [102.0]
    periods = np.arange(20) + 0.5
    rolled = sum(cf / (1.0 + 0.0395 / 2) ** n for cf, n in zip(flows, periods))
    assert result.roll == pytest.approx(rolled - dirty1, abs=1e-9)
    dv01 = sum(cf * n / 2 / 1.02 ** (n + 1) for cf, n in zip(flows, periods)) * 1e-4
    assert result.dv01 == pytest.approx(dv01, rel=1e-10)
    assert result.carry_bp == pytest.approx(result.carry / dv01)
    assert result.roll_bp == pytest.approx(5.0, rel=0.01)  # a 5bp yield fall, priced exactly
    assert result.total_bp == pytest.approx(result.carry_bp + result.roll_bp)
    no_roll = carry.bond_carry(bond, s0, s1, 100.0, repo)
    assert no_roll.roll == 0.0 and no_roll.carry == pytest.approx(result.carry)


def test_coupon_paid_inside_the_horizon():
    bond = FixedRateBond(0.05, D("2031-05-15"))
    s0, s1 = D("2026-04-15"), D("2026-07-15")
    result = carry.bond_carry(bond, s0, s1, 101.0, 0.04)
    assert result.income == pytest.approx(bond.accrued(s1) - bond.accrued(s0) + 2.5, abs=1e-12)
    assert result.dirty == pytest.approx(101.0 + bond.accrued(s0))
    assert result.financing == pytest.approx(result.dirty * 0.04 * 91 / 360)
    # A premium bond pulls to par: its clean price falls at an unchanged yield.
    assert result.pull < 0
    y0 = bond.yield_from_price(101.0, s0)
    assert result.pull == pytest.approx(bond.price_from_yield(y0, s1) - 101.0)


def test_bond_carry_rejects_bad_inputs():
    bond = FixedRateBond(0.04, D("2036-05-15"))
    with pytest.raises(ValueError, match="after the settlement"):
        carry.bond_carry(bond, D("2026-05-15"), D("2026-05-15"), 100.0, 0.03)
    with pytest.raises(ValueError, match="decimal"):
        carry.bond_carry(bond, D("2026-05-15"), D("2026-08-15"), 100.0, 3.5)
    with pytest.raises(ValueError, match="positive"):
        carry.bond_carry(bond, D("2026-05-15"), D("2026-08-15"), -1.0, 0.03)
    with pytest.raises(ValueError, match="matures"):
        carry.bond_carry(FixedRateBond(0.04, D("2026-07-15")), D("2026-05-15"), D("2026-08-15"), 100.0, 0.03)


def _static_dirty(curve, bond, settle, horizon_settle) -> float:
    """Each cash flow left at the horizon discounted on today's curve at the
    date as many days after today's settlement as it is after the horizon
    settlement (the curve held static in time to maturity)."""

    flows = bond.cashflows(horizon_settle)
    shifted = [settle + (pay - horizon_settle) for pay in flows["pay"]]
    return float(sum(a * curve.df(d) for a, d in zip(flows["amount"], shifted)) / curve.df(settle))


def test_treasury_par_bonds_on_the_static_curve(market):
    repo = 0.0388
    rows = carry.treasury_carry(market, "3M", repo)
    assert [r.label.split()[0] for r in rows] and len(rows) == len(carry.BOND_TENORS)
    settle = market.govt.conventions.cal.add_business_days(market.asof, 1)
    for tenor, r in zip(carry.BOND_TENORS, rows):
        quote = next(q.rate for q in market.govt.quotes if q.tenor == tenor)
        assert r.clean == pytest.approx(100.0, abs=1e-9) and r.ytm == pytest.approx(quote, abs=1e-10)
        # carry + roll is the static-curve P&L of the financed position: the horizon dirty price on today's
        # curve held static in time to maturity + coupons - today's dirty price - financing.
        bond = benchmark_bond(market.govt, int(tenor[:-1]), settle)
        static = _static_dirty(market.govt, bond, settle, r.horizon_settle)
        assert carry.static_dirty_price(bond, market.govt, settle, r.horizon_settle) == pytest.approx(static, abs=1e-11)
        pnl = static + bond.cash(settle, r.horizon_settle) - r.dirty - r.financing
        assert r.carry + r.roll == pytest.approx(pnl, abs=1e-9)
        assert r.ytm_roll == pytest.approx(bond.yield_from_price(static, r.horizon_settle, clean=False), abs=1e-12)
        # Constant-yield carry of a par bond ~ 100 x (yield x days/365 - repo x days/360).
        assert r.carry == pytest.approx(100.0 * (r.ytm * r.days / 365 - repo * r.days / 360), rel=0.05)
    # Steep front end (1M bill 4.11%, 2Y 4.77%): the 2Y rolls down the most per bp of DV01.
    assert rows[0].roll_bp == max(r.roll_bp for r in rows)
    table = carry.bond_table(rows, carry.BOND_TENORS, dv01=25_000.0)
    assert table["total_ccy"].to_numpy() == pytest.approx(25_000.0 * table["total_bp"].to_numpy())
    longer = carry.treasury_carry(market, "6M", repo)
    assert all(b.carry > a.carry for a, b in zip(rows, longer))  # more accrual over six months


def test_bond_roll_down_follows_the_slope_of_the_par_curve(market):
    """Review finding: rolling down a curve rebuilt on the horizon date (its
    pillars moved by business-day adjustments) gave the 3Y a -0.46bp 1M roll
    while par yields rise 8bp from 2Y to 3Y. On the static curve every bond
    on the rising 2Y-20Y stretch rolls down positively, the 30Y (below the
    20Y: an inverted 20s30s) rolls up, and roll-down grows about linearly
    with the horizon."""

    par = {q.tenor: q.rate for q in market.govt.quotes}
    assert all(par[a] < par[b] for a, b in zip(("2Y", "3Y", "5Y", "7Y", "10Y"), ("3Y", "5Y", "7Y", "10Y", "20Y")))
    assert par["30Y"] < par["20Y"]
    rolls = {h: {t: r.roll_bp for t, r in zip(carry.BOND_TENORS, carry.treasury_carry(market, h, 0.0388))}
             for h in carry.HORIZONS}  # fmt: skip
    for h, by_tenor in rolls.items():
        assert all(by_tenor[t] > 0 for t in ("2Y", "3Y", "5Y", "7Y", "10Y", "20Y")), (h, by_tenor)
        assert by_tenor["30Y"] < 0, (h, by_tenor)
    for tenor in carry.BOND_TENORS:
        assert 2.5 < rolls["3M"][tenor] / rolls["1M"][tenor] < 4.0, tenor
        assert 1.6 < rolls["6M"][tenor] / rolls["3M"][tenor] < 2.6, tenor


def test_static_dirty_price_on_a_flat_curve():
    """On a flat zero curve (4% continuously compounded) the static price at
    the horizon discounts each remaining flow at 4% for its time from the
    horizon settlement, whatever the time from today."""

    from derivx.rates.curves import Curve

    asof = D("2026-05-14")
    times = np.linspace(0.0, 40.0, 81)
    flat = Curve(asof, times, np.exp(-0.04 * times), "loglinear")
    bond = FixedRateBond(0.05, D("2036-05-15"))
    s0, s1 = D("2026-05-15"), D("2026-08-17")
    static = carry.static_dirty_price(bond, flat, s0, s1)
    flows = bond.cashflows(s1)
    direct = sum(a * math.exp(-0.04 * (p - s1).days / 365.0) for a, p in zip(flows["amount"], flows["pay"]))
    assert static == pytest.approx(direct, rel=1e-12)
    with pytest.raises(ValueError, match="horizon settlement"):
        carry.static_dirty_price(bond, flat, s1, s0)


def test_treasury_carry_needs_the_treasury_curve(market):
    with pytest.raises(ValueError, match="Treasury curve"):
        carry.treasury_carry(market.replace(govt=None), "3M", 0.04)
    with pytest.raises(ValueError, match="horizon must be one of"):
        carry.treasury_carry(market, "2M", 0.04)


def _par(market, swap: VanillaSwap) -> float:
    """Par rate from discount factors: sum (DF_s/DF_e - 1) DF_pay over the
    float periods / sum tau DF_pay over the fixed periods."""

    df = market.curve.df
    periods = swap.periods(market.asof)
    floating = sum((df(p.start) / df(p.end) - 1.0) * df(p.pay) for p in periods if p.leg == "float")
    annuity = sum(p.accrual * df(p.pay) for p in periods if p.leg == "fixed")
    return floating / annuity


def test_swap_carry_against_discount_factors(market):
    conv = market.curve.conventions
    for tenor in ("2Y", "10Y", "30Y"):
        r = carry.swap_carry(market, tenor, "3M")
        quote = next(q.rate for q in market.curve.quotes if q.tenor == tenor)
        assert r.spot_par == pytest.approx(quote, abs=1e-12)  # the curve reprices its own quote
        spot = VanillaSwap.from_conventions(conv, tenor, 0.0, 1.0, trade_date=market.asof)
        fwd = VanillaSwap.from_conventions(conv, tenor, 0.0, 1.0, forward_start="3M", trade_date=market.asof)
        months = int(tenor[:-1]) * 12
        longer = VanillaSwap.from_conventions(conv, f"{months + 3}M", 0.0, 1.0, trade_date=market.asof)
        assert r.spot_par == pytest.approx(_par(market, spot), abs=1e-13)
        assert r.forward_par == pytest.approx(_par(market, fwd), abs=1e-13)
        assert r.long_par == pytest.approx(_par(market, longer), abs=1e-13)
        assert r.total_bp == pytest.approx((r.forward_par - r.spot_par) * 1e4)
        assert r.roll_bp == pytest.approx((r.long_par - r.spot_par) * 1e4)
        assert r.carry_bp + r.roll_bp == pytest.approx(r.total_bp)
    # Inverted front end (3.55% 2Y against 3.89% 3M OIS): receiving 2Y loses carry.
    two = carry.swap_carry(market, "2Y", "3M")
    assert two.short_par > two.spot_par and two.carry_bp < 0


@pytest.mark.parametrize("horizon", ["3M", "6M"])
def test_swap_carry_is_fixed_against_floating_over_the_horizon(market, horizon):
    """Review finding: carry was the forward drop minus the slide of the
    spot T swap to T - h, which is not the fixed-vs-floating accrual it was
    described as (6M 10Y: -0.13bp against -0.44bp). The forward drop splits
    exactly into carry = (S(0,T+h) - S(0,h)) x A(0,h) / A(h,T+h) and roll =
    S(0,T+h) - S(0,T)."""

    conv = market.curve.conventions
    h = carry.HORIZONS[horizon]
    short = VanillaSwap.from_conventions(conv, f"{h}M", 0.0, 1.0, trade_date=market.asof)
    for tenor in carry.SWAP_TENORS:
        r = carry.swap_carry(market, tenor, horizon)
        fwd = VanillaSwap.from_conventions(conv, tenor, 0.0, 1.0, forward_start=f"{h}M", trade_date=market.asof)
        accrual = (r.long_par - r.short_par) * short.annuity(market) / fwd.annuity(market) * 1e4
        assert r.carry_bp == pytest.approx(accrual, abs=1e-9), tenor
        assert r.short_par == pytest.approx(_par(market, short), abs=1e-13)
    # Front end inverted (3M 3.89% over 2Y 3.55%): the 2Y receiver pays carry and the 2Y-to-2Y6M slide is small.
    two = carry.swap_carry(market, "2Y", "6M")
    assert two.carry_bp == pytest.approx(-7.70, abs=0.01) and two.roll_bp == pytest.approx(-2.43, abs=0.01)


@pytest.mark.parametrize("horizon", list(carry.HORIZONS))
def test_forward_starting_receiver_earns_the_total_on_the_static_curve(market, horizon):
    """At the horizon, on today's quotes rebuilt on that date (the scenario
    engine's carry market), the forward-starting receiver struck at F(h,T)
    is the spot T swap: worth (F - S(0,T)) x its annuity."""

    conv = market.curve.conventions
    at = carry.horizon_date(market.asof, horizon, conv.cal)
    rolled = carry_market(market, (at - market.asof).days)
    for tenor in ("5Y", "10Y"):
        r = carry.swap_carry(market, tenor, horizon)
        start = f"{carry.HORIZONS[horizon]}M"
        fwd = VanillaSwap.from_conventions(conv, tenor, r.forward_par, 1e7, forward_start=start, trade_date=market.asof)
        assert fwd.pv(market) == pytest.approx(0.0, abs=1e-6)  # struck at par today
        assert fwd.pv(rolled) / fwd.annuity(rolled) == pytest.approx(r.total_bp, abs=1e-9)


def test_swap_carry_table_and_errors(market):
    rows = [carry.swap_carry(market, t, "1M") for t in carry.SWAP_TENORS]
    table = carry.swap_table(rows, dv01=1_000.0)
    assert list(table["tenor"]) == list(carry.SWAP_TENORS)
    assert table["total_ccy"].to_numpy() == pytest.approx(1_000.0 * table["total_bp"].to_numpy())
    ten = table.set_index("tenor").loc["10Y"]
    spot = VanillaSwap.from_conventions(market.curve.conventions, "10Y", 0.0, 1e6, trade_date=market.asof)
    assert ten["dv01_per_mm"] == pytest.approx(spot.annuity(market), rel=1e-12)
    with pytest.raises(ValueError, match="longer than"):
        carry.swap_carry(market, "3M", "3M")
    with pytest.raises(ValueError, match="horizon"):
        carry.swap_carry(market, "5Y", "12M")


def test_bond_carry_components_match_live_quantlib():
    """Coupon accrual, pull to par and DV01 of :func:`carry.bond_carry`
    against QuantLib's accrued interest, clean price at an unchanged yield
    and cash flows, for bonds whose horizon crosses a coupon date."""

    ql = pytest.importorskip("QuantLib")
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from golden import bonds as gb
    from golden.common import qd, set_evaluation_date

    s0 = D("2026-10-20")
    set_evaluation_date(D("2026-10-19"))
    cases = [
        (FixedRateBond(0.025, D("2031-02-15")), D("2027-04-20"), 97.25),
        (FixedRateBond(0.06, D("2045-11-15")), D("2027-01-20"), 108.5),
        (FixedRateBond(0.0425, D("2033-03-01"), day_count="30/360"), D("2027-04-20"), 99.0),
    ]
    for bond, s1, clean in cases:
        r = carry.bond_carry(bond, s0, s1, clean, 0.0388)
        qb, dc = gb.ql_bond(bond, s0, ql.Unadjusted)
        rate = ql.InterestRate(r.ytm, dc, ql.Compounded, ql.Semiannual)
        y_ref = ql.BondFunctions.bondYield(
            qb, ql.BondPrice(clean, ql.BondPrice.Clean), dc, ql.Compounded, ql.Semiannual, qd(s0), 1e-15, 1000, 0.05
        )
        assert r.ytm == pytest.approx(y_ref, abs=1e-10)
        paid = sum(cf.amount() for cf in qb.cashflows() if qd(s0) < cf.date() <= qd(s1))
        assert paid > 0  # the horizon crosses a coupon date
        income = ql.BondFunctions.accruedAmount(qb, qd(s1)) - ql.BondFunctions.accruedAmount(qb, qd(s0)) + paid
        assert r.income == pytest.approx(income, abs=1e-10)
        assert r.pull == pytest.approx(ql.BondFunctions.cleanPrice(qb, rate, qd(s1)) - clean, abs=1e-9)
        dirty0 = clean + ql.BondFunctions.accruedAmount(qb, qd(s0))
        assert r.financing == pytest.approx(dirty0 * 0.0388 * ql.Actual360().yearFraction(qd(s0), qd(s1)), abs=1e-12)
        # DV01 = dP/dy x 1bp = modified duration x dirty price x 1bp at the horizon settlement and unchanged yield.
        dirty1 = ql.BondFunctions.cleanPrice(qb, rate, qd(s1)) + ql.BondFunctions.accruedAmount(qb, qd(s1))
        modified = ql.BondFunctions.duration(qb, rate, ql.Duration.Modified, qd(s1))
        assert r.dv01 == pytest.approx(modified * dirty1 * 1e-4, rel=1e-9)
