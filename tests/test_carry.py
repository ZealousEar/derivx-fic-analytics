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


def test_treasury_par_bonds_on_the_static_curve(market):
    repo = 0.0388
    rows = carry.treasury_carry(market, "3M", repo)
    assert [r.label.split()[0] for r in rows] and len(rows) == len(carry.BOND_TENORS)
    rolled = market.govt.rebuilt(asof=carry.horizon_date(market.asof, "3M", market.govt.conventions.cal))
    settle = market.govt.conventions.cal.add_business_days(market.asof, 1)
    for tenor, r in zip(carry.BOND_TENORS, rows):
        quote = next(q.rate for q in market.govt.quotes if q.tenor == tenor)
        assert r.clean == pytest.approx(100.0, abs=1e-9) and r.ytm == pytest.approx(quote, abs=1e-10)
        # carry + roll is the static-curve P&L of the financed position: the horizon dirty price off today's
        # quotes rebuilt on the horizon date + coupons - today's dirty price - financing.
        bond = benchmark_bond(market.govt, int(tenor[:-1]), settle)
        static = bond.price_from_curve(rolled, r.horizon_settle, clean=False)
        pnl = static + bond.cash(settle, r.horizon_settle) - r.dirty - r.financing
        assert r.carry + r.roll == pytest.approx(pnl, abs=1e-9)
        # Constant-yield carry of a par bond ~ 100 x (yield x days/365 - repo x days/360).
        assert r.carry == pytest.approx(100.0 * (r.ytm * r.days / 365 - repo * r.days / 360), rel=0.05)
    # Steep front end (1M bill 4.11%, 2Y 4.77%): the 2Y rolls down the most per bp of DV01.
    assert rows[0].roll_bp == max(r.roll_bp for r in rows)
    table = carry.bond_table(rows, carry.BOND_TENORS, dv01=25_000.0)
    assert table["total_ccy"].to_numpy() == pytest.approx(25_000.0 * table["total_bp"].to_numpy())
    longer = carry.treasury_carry(market, "6M", repo)
    assert all(b.carry > a.carry for a, b in zip(rows, longer))  # more accrual over six months


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
        rolled = VanillaSwap.from_conventions(conv, f"{months - 3}M", 0.0, 1.0, trade_date=market.asof)
        assert r.spot_par == pytest.approx(_par(market, spot), abs=1e-13)
        assert r.forward_par == pytest.approx(_par(market, fwd), abs=1e-13)
        assert r.rolled_par == pytest.approx(_par(market, rolled), abs=1e-13)
        assert r.total_bp == pytest.approx((r.forward_par - r.spot_par) * 1e4)
        assert r.roll_bp == pytest.approx((r.spot_par - r.rolled_par) * 1e4)
        assert r.carry_bp + r.roll_bp == pytest.approx(r.total_bp)
    # Inverted front end (3.55% 2Y against 3.89% 3M OIS): receiving 2Y loses carry.
    two = carry.swap_carry(market, "2Y", "3M")
    assert two.short_par > two.spot_par and two.carry_bp < 0


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
