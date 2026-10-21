"""Scenario engine: shock profiles, historical episodes, shocked and rolled
markets, the P&L explain (identities, explain quality, horizon Greeks),
library positions, QuantLib cross-checks and the Streamlit tab."""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from derivx.rates.conventions import Calendar, add_months, make_schedule, parse_tenor, year_fraction
from derivx.rates.curves import CONVENTIONS, CurveQuote, Instrument, Market, bootstrap, flat_curve
from derivx.rates.examples import EXAMPLE_ASOF, example_market
from derivx.risk.scenarios import (
    EXPLAIN_COLUMNS,
    TOTAL,
    Shock,
    butterfly,
    carry_market,
    historical,
    historical_episodes,
    historical_table,
    horizon_date,
    key_rates,
    parallel,
    quote_risk,
    run_scenario,
    scenario_grid,
    shocked_market,
    standard_library,
    twist,
)
from derivx.risk.sensitivities import Position, bumped_market, key_rate_pv01, parallel_pv01

ROOT = Path(__file__).resolve().parents[1]
EPISODES = {
    "2013 taper tantrum": ("2013-05-02", "2013-09-05"),
    "2020 dash for cash": ("2020-03-09", "2020-03-18"),
    "2022 hiking cycle": ("2022-01-03", "2022-10-21"),
    "Sept 2022 gilt crisis week (UST)": ("2022-09-21", "2022-09-28"),
    "2023 SVB": ("2023-03-08", "2023-03-15"),
}


@pytest.fixture(scope="module")
def market():
    return example_market()


def _quote_position(market, index, which="curve", notional=1e6):
    """Receive the curve's own quote ``index`` (a par instrument re-dated to
    each market's valuation date)."""

    quote = getattr(market, which).quotes[index]

    def pv(m):
        curve = getattr(m, which)
        return notional * Instrument(quote, m.asof, curve.conventions).npv(curve)

    return Position(f"{which} {quote.tenor} {quote.kind}", pv, kind="bond" if which == "govt" else "swap")


def _pinned_ois(market, tenor="10Y", rate=None, notional=1e6, receive=True):
    """OIS swap with dates fixed on ``market.asof`` (the curve's OIS
    conventions), valued on any later market; at ``rate=None`` it is the par
    swap of that tenor."""

    conv = market.curve.conventions
    cal = conv.cal
    spot = cal.add_business_days(cal.adjust(market.asof, "F"), conv.spot_lag)
    dates = make_schedule(
        spot, add_months(spot, parse_tenor(tenor).months), conv.ois_fixed_months, cal, conv.bdc, conv.bdc
    )
    periods = [
        (s, e, cal.add_business_days(e, conv.payment_lag), year_fraction(s, e, conv.ois_day_count))
        for s, e in zip(dates[:-1], dates[1:])
    ]
    if rate is None:
        dfs = [market.curve.df(p) for _, _, p, _ in periods]
        floating = sum((market.compounded_growth(s, e) - 1.0) * d for (s, e, _, _), d in zip(periods, dfs))
        rate = floating / sum(tau * d for (_, _, _, tau), d in zip(periods, dfs))
    sign = 1.0 if receive else -1.0

    def flows(m):
        return [(pay, sign * notional * (rate * tau - (m.compounded_growth(s, e) - 1.0))) for s, e, pay, tau in periods]

    def pv(m):
        return sum(amount * m.curve.df(pay) for pay, amount in flows(m) if pay > m.asof)

    def cash(m, start, end):
        return sum(amount for pay, amount in flows(m) if start < pay <= end)

    side = "REC" if receive else "PAY"
    return Position(f"{side} {tenor} pinned", pv, kind="swap", cash=cash, details={"rate": rate, "dates": periods})


def _sloped_market(slope):
    """SOFR-convention curve rising (slope > 0) or falling with tenor."""

    tenors = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
    quotes = [CurveQuote("DEPO", "1W", 0.035)]
    quotes += [CurveQuote("OIS", t, 0.035 + slope * (1.0 - math.exp(-parse_tenor(t).years / 5.0))) for t in tenors]
    return Market(curve=bootstrap(EXAMPLE_ASOF, quotes, "USD SOFR"))


def _row(rows, name):
    return next(r for r in rows if r["position"] == name)


# --------------------------------------------------------------------------
# Shock profiles
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pivot", [0.5, 2.0, 5.0, 7.3, 10.0, 25.0])
@pytest.mark.parametrize("ends", [(-25.0, 25.0), (40.0, -13.0), (0.0, 7.0)])
def test_twist_is_exactly_zero_at_the_pivot(market, pivot, ends):
    shock = twist(pivot, *ends)
    for curve in (market.curve, market.govt):
        years = np.array([inst.tenor.years for inst in curve.instruments])
        assert shock.bp_at([pivot], (years.min(), years.max()))[0] == 0.0
        bumps = shock.quote_bumps(curve)
        assert bumps[np.argmin(years)] == ends[0] and bumps[np.argmax(years)] == ends[1]
        assert np.all(bumps[years == pivot] == 0.0)
        # Linear on each side of the pivot, so short and long sides keep their signs.
        assert np.all(np.sign(bumps[years < pivot]) * np.sign(ends[0]) >= 0)
        assert np.all(np.sign(bumps[years > pivot]) * np.sign(ends[1]) >= 0)


def test_twist_with_anchors_and_steepener_on_the_curve(market):
    steepener = twist(5.0, -12.5, 12.5, short_years=2.0, long_years=10.0)
    np.testing.assert_array_equal(
        steepener.bp_at([0.1, 2.0, 3.5, 5.0, 7.5, 10.0, 30.0]), [-12.5, -12.5, -6.25, 0.0, 6.25, 12.5, 12.5]
    )
    bumps = dict(zip(market.curve.labels, twist(5.0, -25.0, 25.0).quote_bumps(market.curve)))
    assert bumps["5Y OIS"] == 0.0 and bumps["10Y OIS"] - bumps["2Y OIS"] > 0  # 2s10s steepens
    assert bumps["30Y OIS"] == 25.0 and bumps["10Y OIS"] == pytest.approx(25.0 * 5 / 25)
    with pytest.raises(ValueError, match="pivot"):
        twist(5.0, -10, 10, short_years=6.0)
    with pytest.raises(ValueError, match="pivot"):
        twist(0.0, -10, 10)


def test_butterfly_and_key_rate_profiles(market):
    fly = butterfly(-5.0, 20.0, 5.0, wings=(2.0, 10.0))
    np.testing.assert_allclose(fly.bp_at([0.5, 2.0, 3.5, 5.0, 7.5, 10.0, 30.0]), [-5, -5, 7.5, 20, 7.5, -5, -5])
    with pytest.raises(ValueError, match="between the wings"):
        butterfly(0.0, 10.0, 40.0)
    # Equal key-rate moves add up to a parallel move (Ho key-rate tents).
    keys = {t: 10.0 for t in ("1Y", "2Y", "5Y", "10Y", "30Y")}
    np.testing.assert_array_equal(key_rates(keys).quote_bumps(market.curve), parallel(10.0).quote_bumps(market.curve))
    one = key_rates({"5Y": 0.0, "7Y": 15.0, "10Y": 0.0})
    moved = {label: bp for label, bp in zip(market.curve.labels, one.quote_bumps(market.curve)) if bp}
    assert moved == {"7Y OIS": 15.0}
    assert key_rates({"18M": 4.0, 3: 6.0}).profile == ((1.5, 4.0), (3.0, 6.0))
    with pytest.raises(ValueError, match="same tenor"):
        key_rates({"1Y": 1.0, "12M": 2.0})


def test_shock_validation(market):
    for days in (-1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="horizon"):
            Shock("bad", horizon_days=days)
    with pytest.raises(ValueError, match="finite"):
        Shock("bad", profile=((1.0, float("nan")),))
    for field_name in ("normal_vol_bp", "lognormal_vol_pts", "long_bp"):
        with pytest.raises(ValueError, match=field_name):
            Shock("bad", **{field_name: float("inf")})
    with pytest.raises(ValueError, match="normal_vol_bp"):
        Shock("bad", normal_vol_bp=None)
    with pytest.raises(ValueError, match="horizon"):
        horizon_date(market, -3)
    with pytest.raises(ValueError, match="positive"):
        quote_risk([_quote_position(market, 9)], market, bp=0.0)
    assert not Shock("nothing").moves_rates and not parallel(5.0, curve=False, govt=False).moves_rates
    assert parallel(5.0).moves_rates and twist(5.0, 0.0, 1.0).moves_rates
    custom = parallel(10.0, name="mine", govt=False, horizon_days=7)
    assert (custom.name, custom.govt, custom.horizon_days) == ("mine", False, 7)


# --------------------------------------------------------------------------
# Historical episodes
# --------------------------------------------------------------------------


def test_historical_csv_covers_every_episode(market):
    table = historical_table()
    assert list(table.columns) == ["episode", "start", "end", "tenor_years", "change_bp", "source"]
    assert historical_episodes() == list(EPISODES)
    tenors = [1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    for name, (start, end) in EPISODES.items():
        rows = table[table["episode"] == name]
        assert list(rows["tenor_years"]) == tenors  # exact, so the 1M change lands on the 1M quotes
        one_month = dict(zip(market.govt.labels, historical(name).quote_bumps(market.govt)))["1M BILL"]
        assert one_month == rows["change_bp"].iloc[0]
        assert rows["source"].str.startswith("FRED DGS").all()
        used_start, used_end = dt.date.fromisoformat(rows["start"].iloc[0]), dt.date.fromisoformat(rows["end"].iloc[0])
        assert abs((used_start - dt.date.fromisoformat(start)).days) <= 4
        assert abs((used_end - dt.date.fromisoformat(end)).days) <= 4
        assert rows["change_bp"].notna().all() and rows["change_bp"].abs().max() < 600
        shock = historical(name)
        assert shock.profile == tuple(zip(rows["tenor_years"], rows["change_bp"]))
        assert start[:4] in shock.description
    hikes = table[table["episode"] == "2022 hiking cycle"].set_index("tenor_years")["change_bp"]
    assert hikes.loc[2.0] == 371.0 and hikes.min() > 200  # 2Y 0.78% -> 4.49%
    assert historical("2023 SVB", scale=-0.5).profile[4][1] == pytest.approx(56.0)
    with pytest.raises(ValueError, match="unknown episode"):
        historical("1994 bond massacre")


def test_standard_library():
    library = standard_library()
    names = [s.name for s in library]
    assert len(set(names)) == len(names)
    assert names[:4] == ["Parallel +25bp", "Parallel -25bp", "Parallel +100bp", "Parallel -100bp"]
    assert set(EPISODES) <= set(names)
    steepener = next(s for s in library if "steepener" in s.name)
    assert steepener.bp_at([10.0])[0] - steepener.bp_at([2.0])[0] == 25.0
    assert all(s.curve and s.govt and s.horizon_days == 0 for s in library)


# --------------------------------------------------------------------------
# Shocked curves
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shock",
    [
        twist(5.0, -40.0, 60.0),
        historical("2020 dash for cash"),
        key_rates({"2Y": 0, "5Y": 30, "10Y": 0}),
        parallel(-150.0, horizon_days=45),
    ],
)
def test_shocked_curves_reprice_their_shocked_quotes(market, shock):
    moved = shocked_market(market, shock)
    assert moved.asof == horizon_date(market, shock.horizon_days)
    assert moved.fixings is market.curve
    for which in ("curve", "govt"):
        base, curve = getattr(market, which), getattr(moved, which)
        bumps = shock.quote_bumps(base)
        np.testing.assert_allclose(
            [q.rate for q in curve.quotes], [q.rate + b * 1e-4 for q, b in zip(base.quotes, bumps)], rtol=0, atol=1e-15
        )
        assert max(abs(r["error_bp"]) for r in curve.repricing()) < 1e-8
    today = shocked_market(market, shock, at_horizon=False)
    assert today.asof == market.asof


def test_flags_choose_which_curves_move(market):
    swap, bond = _quote_position(market, 11), _quote_position(market, 11, "govt")
    only_govt = run_scenario([swap, bond], market, parallel(10.0, curve=False))
    only_swap = run_scenario([swap, bond], market, parallel(10.0, govt=False))
    assert _row(only_govt, swap.name)["pnl"] == 0.0 and _row(only_govt, bond.name)["pnl"] < 0
    assert _row(only_swap, bond.name)["pnl"] == 0.0 and _row(only_swap, swap.name)["pnl"] < 0
    moved = shocked_market(market, parallel(10.0, govt=False))
    assert moved.govt is market.govt


def test_horizon_date_rolls_to_a_business_day(market):
    assert horizon_date(market, 0) == market.asof
    assert horizon_date(market, 30) == dt.date(2026, 11, 18)
    assert horizon_date(market, 26) == dt.date(2026, 11, 16)  # Saturday 14 Nov -> Monday
    with pytest.raises(ValueError, match="fixings"):
        carry_market(market.replace(fixings=market.curve), 30)


# --------------------------------------------------------------------------
# P&L explain
# --------------------------------------------------------------------------


def test_one_bp_parallel_matches_parallel_pv01(market):
    pos = _quote_position(market, 11)
    pv01 = parallel_pv01(pos.pv, market)
    up = run_scenario([pos], market, parallel(1.0, govt=False))[0]
    down = run_scenario([pos], market, parallel(-1.0, govt=False))[0]
    assert (up["pnl"] - down["pnl"]) / 2.0 == pytest.approx(pv01, rel=1e-12)
    assert up["pnl"] == pytest.approx(pv01, rel=1e-3)
    assert up["delta"] == pytest.approx(sum(r["pv01"] for r in key_rate_pv01(pos.pv, market)), rel=1e-12)
    assert up["delta"] == pytest.approx(pv01, rel=1e-5)
    assert abs(up["residual"]) < 1e-6 * abs(up["pnl"]) and up["base_pv"] == pytest.approx(0.0, abs=1e-6)


def test_gamma_is_half_the_diagonal_key_rate_gammas(market):
    pos = _quote_position(market, 9)
    shock = twist(5.0, -30.0, 20.0, govt=False)
    row = run_scenario([pos], market, shock)[0]
    bumps, base = shock.quote_bumps(market.curve), pos.pv(market)

    def second_difference(i):
        up, down = (pos.pv(bumped_market(market, "curve", {i: bp})) for bp in (1.0, -1.0))
        return up + down - 2.0 * base

    diag = sum(0.5 * b * b * second_difference(i) for i, b in enumerate(bumps) if b)
    assert row["gamma"] == pytest.approx(diag, rel=1e-12)
    risk = quote_risk([pos], market, "curve")
    assert risk.computed.all() and risk.labels == tuple(market.curve.labels)
    assert run_scenario([pos], market, shock, risk={"curve": risk})[0] == row


@pytest.mark.parametrize("bp", [100.0, -100.0])
def test_delta_gamma_explain_a_100bp_move_on_linear_instruments(market, bp):
    positions = [
        _quote_position(market, 11),
        _quote_position(market, 9),
        _quote_position(market, 3),
        _quote_position(market, 11, "govt"),
        _pinned_ois(market, "7Y", rate=0.03),
    ]
    for row in run_scenario(positions, market, parallel(bp)):
        explained = row["delta"] + row["gamma"] + row["cross_gamma"]
        assert abs(row["pnl"] - explained) < 0.01 * abs(row["pnl"]), row
        assert row["vol"] == 0.0 and row["carry"] == 0.0
        assert row["residual"] == pytest.approx(row["pnl"] - explained, abs=1e-9)
    # Diagonal gammas alone miss the cross-gammas between quotes (the 10Y
    # annuity moves with every shorter quote): more than 1% of the 10Y move.
    ten = run_scenario(positions[:1], market, parallel(bp))[0]
    assert abs(ten["pnl"] - ten["delta"] - ten["gamma"]) > 0.02 * abs(ten["pnl"])


@pytest.mark.parametrize(
    "shock",
    [
        parallel(2.0),
        twist(5.0, -3.0, 4.0),
        butterfly(-1.0, 2.0, 7.0),
        key_rates({"1Y": 0.0, "3Y": -2.5, "10Y": 1.0}),
        historical("2023 SVB", 0.02),
    ],
)
def test_residual_is_small_for_small_shocks(market, shock):
    positions = [_quote_position(market, i) for i in (5, 9, 11, 14)] + [_quote_position(market, 9, "govt")]
    rows = run_scenario(positions, market, shock)
    scale = max(abs(r["pnl"]) for r in rows)
    assert scale > 1.0
    for row in rows:
        assert abs(row["residual"]) < 1e-4 * scale + 1e-6, row
        explained = sum(row[c] for c in EXPLAIN_COLUMNS)
        assert explained == pytest.approx(row["pnl"], rel=1e-12, abs=1e-9)


def test_vol_shocks_are_explained_by_vega(market):
    vol = Position(
        "vol", lambda m: 1e4 * (m.vol_shift_normal / 1e-4) + 3e3 * (m.vol_shift_lognormal / 0.01), vol_model="normal"
    )
    black = Position("black", vol.pv, vol_model="lognormal")
    rows = run_scenario([vol, black], market, Shock("vols", normal_vol_bp=5.0, lognormal_vol_pts=-2.0))
    assert _row(rows, "vol")["vol"] == pytest.approx(5e4, rel=1e-10)
    assert _row(rows, "vol")["pnl"] == pytest.approx(5e4 - 6e3, rel=1e-10)  # moves both shifts
    assert _row(rows, "black")["vol"] == pytest.approx(-6e3, rel=1e-10)
    assert all(r["delta"] == r["gamma"] == r["cross_gamma"] == r["carry"] == 0.0 for r in rows)


def test_carry_of_a_receiver_over_30_days_has_the_expected_sign():
    for slope, sign in ((0.02, 1.0), (-0.02, -1.0)):
        m = _sloped_market(slope)
        rec, pay = _pinned_ois(m, "10Y"), _pinned_ois(m, "10Y", receive=False)
        rec.name, pay.name = "rec", "pay"
        assert Instrument(m.curve.quotes[9], m.asof, m.curve.conventions).npv(m.curve) * 1e6 == pytest.approx(
            rec.pv(m), abs=1e-6
        )
        rows = run_scenario([rec, pay], m, Shock("carry", horizon_days=30))
        r, p = _row(rows, "rec"), _row(rows, "pay")
        # Rising curve: receive more fixed than overnight pays and roll down the curve.
        assert np.sign(r["carry"]) == sign and abs(r["carry"]) > 100
        assert p["carry"] == pytest.approx(-r["carry"], rel=1e-12)
        assert r["pnl"] == r["carry"] and r["residual"] == 0.0 and r["delta"] == 0.0


def test_explain_is_signed_to_the_holder_and_scales_with_notional(market):
    rec = _pinned_ois(market, "7Y", rate=0.036)
    pay = _pinned_ois(market, "7Y", rate=0.036, notional=-2e6)  # a receiver of -2mm is a payer of 2mm
    shock = twist(5.0, -30.0, 20.0, horizon_days=45)
    r, p = run_scenario([rec, pay], market, shock)
    for column in ("base_pv", "shocked_pv", "cash", "pnl") + EXPLAIN_COLUMNS:
        assert p[column] == pytest.approx(-2.0 * r[column], rel=1e-9, abs=1e-9), column
    assert r["delta"] < 0  # a receiver loses when the (long end of the) curve rises


def test_carry_counts_cash_paid_in_the_window(market):
    swap = _pinned_ois(market, "2Y", rate=0.04)
    first_pay = swap.details["dates"][0][2]
    days = (first_pay - market.asof).days + 3
    row = run_scenario([swap], market, Shock("roll", horizon_days=days))[0]
    rolled = carry_market(market, days)
    paid = swap.cash(rolled, market.asof, rolled.asof)
    assert paid > 0 and row["cash"] == pytest.approx(paid, rel=1e-14)
    assert row["carry"] == pytest.approx(swap.pv(rolled) + paid - swap.pv(market), rel=1e-12)


@pytest.mark.parametrize("days", [30, 365, 730])
def test_shock_at_the_horizon_is_explained_by_the_rolled_market(market, days):
    swap = _pinned_ois(market, "10Y", rate=0.0365)
    shock = twist(5.0, -20.0, 30.0, horizon_days=days, normal_vol_bp=10.0)
    row = run_scenario([swap], market, shock)[0]
    horizon, rolled = shocked_market(market, shock), carry_market(market, days)
    assert row["shocked_pv"] == pytest.approx(swap.pv(horizon), rel=1e-12)
    assert row["pnl"] == pytest.approx(row["shocked_pv"] + row["cash"] - row["base_pv"], rel=1e-12)
    move = row["pnl"] - row["carry"]
    assert move == pytest.approx(swap.pv(horizon) - swap.pv(rolled), rel=1e-12)
    # The shock hits the rolled market, so its Greeks explain it: delta is the
    # rolled key-rate ladder times the quote moves, and the residual is tiny.
    ladder = [r["pv01"] for r in key_rate_pv01(swap.pv, rolled)]
    assert row["delta"] == pytest.approx(float(np.dot(ladder, shock.quote_bumps(rolled.curve))), rel=1e-12)
    assert row["vol"] == 0.0 and abs(row["residual"]) < 1e-5 * abs(move)
    # Today's ladder would miss the risk rolling down the curve by more than 2%.
    instant = run_scenario([swap], market, replace(shock, horizon_days=0))[0]
    assert abs(instant["delta"] - row["delta"]) > 0.02 * abs(move)


def test_trades_that_mature_before_the_horizon_have_no_shock_risk(market):
    swap = _pinned_ois(market, "1Y", rate=0.04)
    row = run_scenario([swap], market, parallel(50.0, horizon_days=730))[0]
    paid = swap.cash(carry_market(market, 730), market.asof, horizon_date(market, 730))
    assert row["shocked_pv"] == 0.0 and row["cash"] == pytest.approx(paid, rel=1e-14) and paid > 0
    assert row["delta"] == row["gamma"] == row["cross_gamma"] == 0.0
    assert row["pnl"] == pytest.approx(row["carry"], rel=1e-14) and row["residual"] == pytest.approx(0.0, abs=1e-9)


def test_precomputed_risk_is_reused_only_on_the_date_the_shock_hits(market):
    pos = _quote_position(market, 9)
    shock = parallel(5.0, govt=False, horizon_days=30)
    fresh = run_scenario([pos], market, shock)
    today = quote_risk([pos], market, "curve")
    rolled = quote_risk([pos], carry_market(market, 30), "curve")
    assert today.asof == market.asof and rolled.asof == horizon_date(market, 30)
    assert run_scenario([pos], market, shock, risk={"curve": today}) == fresh
    assert run_scenario([pos], market, shock, risk={"curve": rolled}) == fresh


def test_vols_are_only_bumped_when_the_shock_moves_them(market):
    def pv(m):
        if m.vol_shift_normal:
            raise AssertionError("bumped a vol the shock does not move")
        return 1e6 * m.curve.df(5.0)

    row = run_scenario([Position("zero", pv, vol_model="normal")], market, parallel(10.0, lognormal_vol_pts=3.0))[0]
    assert row["vol"] == 0.0 and row["delta"] < 0


def test_run_scenario_edge_cases(market):
    assert run_scenario([], market, parallel(10.0)) == []
    flat = Market(curve=flat_curve(EXAMPLE_ASOF, 0.03))
    pos = Position("zero", lambda m: 1e6 * m.curve.df(5.0))
    with pytest.raises(ValueError, match="no quotes"):
        run_scenario([pos], flat, parallel(10.0))
    rows = run_scenario([pos], flat, Shock("vol only", normal_vol_bp=5.0))
    assert rows[0]["pnl"] == 0.0
    with pytest.raises(ValueError, match="no finite value .* for broken"):
        run_scenario([pos, Position("broken", lambda m: float("nan"))], market, parallel(10.0))


def test_scenario_grid(market):
    positions = [_quote_position(market, 11), _quote_position(market, 11, "govt"), _quote_position(market, 11)]
    shocks = standard_library()
    grid = scenario_grid(positions, market, shocks)
    assert list(grid.columns) == [positions[0].name, positions[1].name, f"{positions[0].name} (2)", TOTAL]
    assert list(grid.index) == [s.name for s in shocks]
    np.testing.assert_allclose(grid[TOTAL], grid.drop(columns=TOTAL).sum(axis=1), rtol=1e-14)
    explain = run_scenario(positions[:2], market, shocks[2])
    assert grid.iloc[2, 0] == pytest.approx(explain[0]["pnl"], rel=1e-12)
    assert grid.iloc[2, 1] == pytest.approx(explain[1]["pnl"], rel=1e-12)
    assert (grid.loc["Parallel +100bp", TOTAL] < 0) and (grid.loc["2023 SVB", TOTAL] > 0)
    empty = scenario_grid([], market, shocks[:2])
    assert list(empty.columns) == [TOTAL] and (empty[TOTAL] == 0).all()


# --------------------------------------------------------------------------
# Library positions (when those modules are present)
# --------------------------------------------------------------------------


def test_vanilla_swap_positions(market):
    swaps = pytest.importorskip("derivx.rates.swaps")
    swap = swaps.VanillaSwap.from_conventions("USD SOFR", "10Y", 0.0365, trade_date=market.asof)
    position = swaps.swap_position(swap)
    mine = _pinned_ois(market, "10Y", rate=0.0365)
    assert position.pv(market) == pytest.approx(mine.pv(market), rel=1e-10)
    shock = parallel(100.0, horizon_days=30)
    theirs, ours = run_scenario([position, mine], market, shock)
    for key in ("pnl", "delta", "carry", "cash"):
        assert theirs[key] == pytest.approx(ours[key], rel=1e-9, abs=1e-6)
    instant = run_scenario([position], market, parallel(100.0))[0]
    assert abs(instant["residual"]) < 0.01 * abs(instant["pnl"])


def test_bond_positions(market):
    bonds = pytest.importorskip("derivx.rates.bonds")
    settle = market.govt.instruments[0].start
    par = bonds.bond_position(bonds.benchmark_bond(market.govt, 10, settle), 1e6)
    # A Nov-15 coupon is paid on Monday 16 Nov 2026, inside a 30-day window.
    seasoned = bonds.bond_position(bonds.FixedRateBond(0.04, dt.date(2031, 11, 15)), 1e6, name="4% Nov-31")
    swap = _quote_position(market, 11)
    rows = run_scenario([par, seasoned, swap], market, parallel(100.0, curve=False))
    assert _row(rows, swap.name)["pnl"] == 0.0
    for row in rows[:2]:
        assert row["delta"] < 0 and row["gamma"] > 0  # long bonds lose when yields rise, and are convex
        assert abs(row["pnl"] - row["delta"] - row["gamma"] - row["cross_gamma"]) < 0.01 * abs(row["pnl"])
    assert _row(rows, par.name)["delta"] == pytest.approx(parallel_pv01(par.pv, market, "govt") * 100, rel=1e-5)
    row = run_scenario([seasoned], market, Shock("carry", horizon_days=30))[0]
    rolled = carry_market(market, 30)
    assert row["cash"] == pytest.approx(20_000.0, rel=1e-12)  # 4% / 2 on $1mm face
    assert row["carry"] == pytest.approx(seasoned.pv(rolled) + 20_000.0 - seasoned.pv(market), rel=1e-12)


def test_cap_positions_explain_vol_and_rates(market):
    capfloor = pytest.importorskip("derivx.rates.capfloor")
    normal = capfloor.CapFloor(1e7, "cap", 0.04, "5Y", vol=0.0095, model="bachelier")
    black = capfloor.CapFloor(1e7, "floor", 0.035, "5Y", vol=0.25, model="black")
    positions = [capfloor.capfloor_position(cf, market.asof) for cf in (normal, black)]
    rows = run_scenario(positions, market, Shock("vols", normal_vol_bp=5.0, lognormal_vol_pts=2.0))
    for row, cf in zip(rows, (normal, black)):
        step = 5.0 if cf.model == "bachelier" else 2.0
        assert row["vol"] == pytest.approx(cf.vega(market) * step, rel=1e-4)  # central difference vs analytic
        assert abs(row["residual"]) < 0.01 * abs(row["pnl"])
    rates = run_scenario(positions, market, parallel(25.0, govt=False))
    assert rates[0]["delta"] > 0 and rates[1]["delta"] < 0  # long cap gains, long floor loses as rates rise
    for row in rates:
        assert abs(row["residual"]) < 0.01 * abs(row["pnl"])


# --------------------------------------------------------------------------
# QuantLib (dev only)
# --------------------------------------------------------------------------


def test_twisted_curve_and_pnl_match_quantlib(market):
    ql = pytest.importorskip("QuantLib")
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    from golden.common import CALENDARS, qd, set_evaluation_date

    set_evaluation_date(market.asof)
    cal = CALENDARS["USD"]

    def ql_curve(quotes):
        handle = ql.RelinkableYieldTermStructureHandle()
        index = ql.Sofr(handle)
        helpers = []
        for q in quotes:
            quote = ql.QuoteHandle(ql.SimpleQuote(q.rate))
            if q.kind == "DEPO":
                helpers.append(
                    ql.DepositRateHelper(quote, ql.Period(q.tenor), 2, cal, ql.ModifiedFollowing, False, ql.Actual360())
                )
            else:
                helpers.append(
                    ql.OISRateHelper(
                        2,
                        ql.Period(q.tenor),
                        quote,
                        index,
                        ql.YieldTermStructureHandle(),
                        False,
                        2,
                        ql.Following,
                        ql.Annual,
                        cal,
                    )
                )
        curve = ql.PiecewiseLogLinearDiscount(qd(market.asof), helpers, ql.Actual365Fixed())
        curve.enableExtrapolation()
        return ql.YieldTermStructureHandle(curve)

    def ql_swap_npv(handle, rate, periods):
        schedule = ql.Schedule(ql.DateVector([qd(periods[0][0])] + [qd(e) for _, e, _, _ in periods]))
        swap = ql.OvernightIndexedSwap(
            ql.Swap.Receiver, 1e6, schedule, rate, ql.Actual360(), ql.Sofr(handle), 0.0, 2, ql.Following, cal
        )
        swap.setPricingEngine(ql.DiscountingSwapEngine(handle))
        return swap.NPV()

    shock = twist(5.0, -35.0, 45.0, govt=False)
    moved = shocked_market(market, shock).curve
    base_ql, moved_ql = ql_curve(market.curve.quotes), ql_curve(moved.quotes)
    days = [market.asof + dt.timedelta(days=d) for d in range(3, 365 * 31, 47)]
    ours = np.array([moved.df(d) for d in days])
    theirs = np.array([moved_ql.discount(qd(d)) for d in days])
    np.testing.assert_allclose(ours, theirs, rtol=1e-12)
    swap = _pinned_ois(market, "10Y", rate=0.0365)
    row = run_scenario([swap], market, shock)[0]
    rate, periods = swap.details["rate"], swap.details["dates"]
    expected = ql_swap_npv(moved_ql, rate, periods) - ql_swap_npv(base_ql, rate, periods)
    assert row["pnl"] == pytest.approx(expected, rel=1e-9)


PRESET_CASES = [
    ("EUR ESTR", "loglinear", dt.date(2027, 2, 26), -0.0055),  # negative short rates, last business day of Feb
    ("GBP SONIA", "linear_zero", dt.date(2028, 2, 29), 0.047),  # spot lag 0 on a leap day, ACT/365F
    ("USD SOFR", "linear_zero", dt.date(2026, 12, 24), 0.039),  # Christmas Eve
]


@pytest.mark.parametrize("preset, interp, asof, level", PRESET_CASES)
def test_horizon_scenarios_match_quantlib_on_other_presets(preset, interp, asof, level):
    """QuantLib bootstraps today's and the shocked quotes itself (on the
    horizon date), takes past fixings from today's curve and prices the
    swaps from their derivx dates; P&L, horizon PV and cash must agree."""

    ql = pytest.importorskip("QuantLib")
    swaps = pytest.importorskip("derivx.rates.swaps")
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    from golden.common import CALENDARS, DAY_COUNTS, pyd, qd

    conv = CONVENTIONS[preset]
    cal, dc = CALENDARS[conv.calendar], DAY_COUNTS[conv.ois_day_count]
    make_index = {"USD SOFR": ql.Sofr, "EUR ESTR": ql.Estr, "GBP SONIA": ql.Sonia}[preset]

    def ql_curve(day, quotes):
        ql.Settings.instance().evaluationDate = qd(day)
        index = make_index(ql.YieldTermStructureHandle())
        helpers = []
        for q in quotes:
            quote = ql.QuoteHandle(ql.SimpleQuote(q.rate))
            if q.kind == "DEPO":
                helpers.append(
                    ql.DepositRateHelper(
                        quote,
                        ql.Period(q.tenor),
                        conv.spot_lag,
                        cal,
                        ql.ModifiedFollowing,
                        False,
                        DAY_COUNTS[conv.depo_day_count],
                    )
                )
            else:  # derivx OIS quotes never roll to month ends (QuantLib's default does from a month-end spot)
                helpers.append(
                    ql.OISRateHelper(
                        conv.spot_lag,
                        ql.Period(q.tenor),
                        quote,
                        index,
                        ql.YieldTermStructureHandle(),
                        False,
                        conv.payment_lag,
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
        build = ql.PiecewiseLogLinearDiscount if interp == "loglinear" else ql.PiecewiseLinearZero
        curve = build(qd(day), helpers, ql.Actual365Fixed(), ql.IterativeBootstrap(1e-15))
        curve.enableExtrapolation()
        return ql.YieldTermStructureHandle(curve)

    def ql_swap(handle, trade):
        dates = ql.Schedule(ql.DateVector([qd(d) for d in trade.schedule_dates(asof)[0]]))
        side = ql.Swap.Receiver if trade.receive_fixed else ql.Swap.Payer
        out = ql.OvernightIndexedSwap(
            side,
            trade.notional,
            dates,
            trade.fixed_rate,
            dc,
            make_index(handle),
            0.0,
            conv.payment_lag,
            ql.Following,
            cal,
        )
        out.setPricingEngine(ql.DiscountingSwapEngine(handle))
        return out

    tenors = ["1M", "3M", "6M", "1Y", "18M", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y", "40Y"]
    quotes = [CurveQuote("DEPO", "1W", level)] + [
        CurveQuote("OIS", t, level + 0.009 * (1.0 - math.exp(-parse_tenor(t).years / 6.0))) for t in tenors
    ]
    market = Market(curve=bootstrap(asof, quotes, preset, interp))
    trades = [
        swaps.VanillaSwap.from_conventions(
            preset,
            tenor,
            level + 0.006,
            notional=1e7,
            receive_fixed=receive,
            kind="OIS",
            forward_start=forward,
            trade_date=asof,
        )
        for tenor, forward, receive in (("18M", None, True), ("5Y", "1Y", False), ("30Y", None, True))
    ]
    positions = [swaps.swap_position(t) for t in trades]
    stub_end = trades[0].schedule_dates(asof)[0][1]  # the 18M swap's 6M front stub, paid payment_lag days later
    shocks = [
        twist(5.0, -40.0, 25.0, horizon_days=(stub_end - asof).days + 1),
        parallel(-60.0, horizon_days=400),
        historical("2023 SVB"),
    ]
    paid = []
    for shock in shocks:
        rows = run_scenario(positions, market, shock)
        paid.append(rows[0]["cash"])
        horizon = horizon_date(market, shock.horizon_days)
        today = ql_curve(asof, market.curve.quotes)
        npv0 = [ql_swap(today, t).NPV() for t in trades]
        fixings, day = [], trades[0].schedule_dates(asof)[0][0]
        while day < horizon:
            nxt = Calendar(conv.calendar).add_business_days(day, 1)
            growth = today.discount(qd(day)) / today.discount(qd(nxt))
            fixings.append((qd(day), (growth - 1.0) / dc.yearFraction(qd(day), qd(nxt))))
            day = nxt
        bumps = shock.quote_bumps(market.curve) * 1e-4
        moved = ql_curve(horizon, [CurveQuote(q.kind, q.tenor, q.rate + b) for q, b in zip(market.curve.quotes, bumps)])
        ql.IndexManager.instance().clearHistories()
        index = make_index(moved)
        for fixing_date, rate in fixings:
            index.addFixing(fixing_date, rate)
        for row, trade, npv in zip(rows, trades, npv0):
            swap = ql_swap(moved, trade)
            sign = 1.0 if trade.receive_fixed else -1.0
            cash = sum(
                s * cf.amount()
                for leg, s in ((swap.fixedLeg(), sign), (swap.overnightLeg(), -sign))
                for cf in leg
                if asof < pyd(cf.date()) <= horizon
            )
            assert row["shocked_pv"] == pytest.approx(swap.NPV(), rel=1e-12, abs=1e-6), (shock.name, trade.label)
            assert row["cash"] == pytest.approx(cash, rel=1e-12, abs=1e-6), (shock.name, trade.label)
            assert row["pnl"] == pytest.approx(swap.NPV() + cash - npv, rel=1e-10, abs=1e-6), (shock.name, trade.label)
        ql.IndexManager.instance().clearHistories()
    assert paid[1] != 0.0  # the 400-day window holds the stub and first annual payments


# --------------------------------------------------------------------------
# Streamlit tab
# --------------------------------------------------------------------------

APP = """
import datetime as dt
import streamlit as st
from derivx.rates.curves import Instrument, Market, flat_curve
from derivx.rates.examples import example_market
from derivx.risk.sensitivities import Position
from derivx.ui.rates_scenarios import render

if st.session_state.get("scn_test_flat"):
    market = Market(curve=flat_curve(dt.date(2026, 10, 19), 0.03))
elif st.session_state.get("scn_test_no_govt"):
    market = example_market().replace(govt=None)
elif st.session_state.get("scn_test_flat_govt"):
    market = example_market().replace(govt=flat_curve(dt.date(2026, 10, 19), 0.04))
else:
    market = example_market()


def fragile(m):
    if m.curve.quotes[11].rate < 0.03:
        raise ValueError("the 10Y quote is below 3%")
    return 0.0


positions = []
if not market.curve.quotes:
    positions.append(Position("ZERO 5Y", lambda m: 1e6 * m.curve.df(5.0)))
elif not st.session_state.get("scn_test_empty"):
    q = market.curve.quotes[11]
    positions.append(Position("REC 10Y", lambda m: 1e6 * Instrument(q, m.asof, m.curve.conventions).npv(m.curve),
                              kind="swap"))
    if market.govt is not None and market.govt.quotes:
        b = market.govt.quotes[11]
        positions.append(Position("UST 10Y", lambda m: 1e6 * Instrument(b, m.asof, m.govt.conventions).npv(m.govt),
                                  kind="bond"))
    positions.append(Position("VOL", lambda m: 1e3 * m.vol_shift_normal / 1e-4, kind="cap", vol_model="normal"))
    if st.session_state.get("scn_test_fragile"):
        positions.append(Position("FRAGILE", fragile))
    if st.session_state.get("scn_test_broken"):
        positions.append(Position("BROKEN", lambda m: float("nan")))
    if st.session_state.get("scn_test_lowvol"):
        positions.append(Position("LOW VOL", lambda m: 1e3 * max(0.005 + m.vol_shift_normal, 0.0) / 1e-4,
                                  kind="cap", vol_model="normal", details={"vol": {1.0: 0.008, 5.0: 0.005}}))
render(market, positions)
"""


def _app(**state) -> AppTest:
    at = AppTest.from_string(APP, default_timeout=120)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_scenario_tab_default_and_every_shock_type():
    at = _app()
    assert not at.error
    assert [m.label for m in at.metric] == ["TOTAL P&L ($)", "EXPLAINED ($)", "RESIDUAL ($)", "RESIDUAL (% OF P&L)"]
    assert float(_metric(at, "TOTAL P&L ($)").replace(",", "")) < 0  # +25bp parallel on receivers
    assert len(at.dataframe) == 2
    for kind in ("Twist", "Butterfly", "Key rates", "Historical"):
        at.selectbox(key="scn_type").set_value(kind).run()
        assert not at.exception and not at.error, kind
    assert at.caption and any("FRED" in c.value for c in at.caption)
    at.selectbox(key="scn_hist").set_value("2023 SVB").run()
    at.number_input(key="scn_hist_scale").set_value(-2.0).run()
    assert not at.exception and not at.error
    at.selectbox(key="scn_type").set_value("Key rates").run()
    assert at.warning  # every key-rate move is zero
    at.number_input(key="scn_kr_10Y").set_value(500.0).run()
    at.number_input(key="scn_kr_2Y").set_value(-500.0).run()
    at.radio(key="scn_chart_view").set_value("ZERO RATES").run()
    at.radio(key="scn_chart_curve").set_value("govt").run()
    assert not at.exception and not at.error and not at.warning


def test_scenario_tab_edge_inputs():
    at = _app()
    at.number_input(key="scn_par_bp").set_value(-500.0).run()
    at.number_input(key="scn_horizon").set_value(730).run()
    at.number_input(key="scn_vol_normal").set_value(200.0).run()
    at.number_input(key="scn_vol_lognormal").set_value(-50.0).run()
    assert not at.exception and not at.error
    at.checkbox(key="scn_move_curve").uncheck().run()
    at.checkbox(key="scn_move_govt").uncheck().run()
    assert not at.exception and not at.error
    at.selectbox(key="scn_type").set_value("Twist").run()
    at.number_input(key="scn_tw_short_years").set_value(8.0).run()  # short end after the pivot
    assert not at.exception and at.error
    at.number_input(key="scn_tw_short_years").set_value(2.0).run()
    at.number_input(key="scn_tw_long_years").set_value(10.0).run()
    assert not at.exception and not at.error
    at.selectbox(key="scn_type").set_value("Butterfly").run()
    at.number_input(key="scn_fly_body_years").set_value(15.0).run()  # body beyond the 10Y wing
    assert not at.exception and at.error


def test_scenario_tab_without_positions_or_quotes():
    empty = _app(scn_test_empty=True)
    assert not empty.error and empty.info and not empty.metric
    flat = _app(scn_test_flat=True)
    assert flat.error and not flat.metric
    no_govt = _app(scn_test_no_govt=True)
    assert not no_govt.error and len(no_govt.dataframe) == 2
    grid = no_govt.dataframe[1].value
    assert list(grid.columns) == ["SCENARIO", "REC 10Y", "VOL", TOTAL] and len(grid) == len(standard_library()) + 1
    assert isinstance(grid, pd.DataFrame)


def _chart_limits(at) -> dict:
    """{chart key: {axis: zoom limits}}, checking every axis of every chart
    starts at and is confined to its limits."""

    out = {}
    for chart in at.get("plotly_chart"):
        layout = json.loads(chart.proto.spec)["layout"]
        zoom = layout["meta"]["zoom"]["axes"]
        axes = {name for name in layout if name.startswith(("xaxis", "yaxis"))}
        assert axes == set(zoom), (chart.proto.id, axes)
        for name in axes:
            axis, lim = layout[name], zoom[name]
            assert axis["range"] == [lim["min"], lim["max"]] and axis["autorange"] is False
            assert (axis["minallowed"], axis["maxallowed"]) == (lim["min"], lim["max"]) and lim["min_span"] > 0
        key = next(k for k in ("scn_curve_chart", "scn_waterfall") if chart.proto.id.endswith(k))
        out[key] = zoom
    return out


def test_scenario_tab_charts_bound_every_axis():
    at = _app()
    assert set(_chart_limits(at)) == {"scn_curve_chart", "scn_waterfall"}
    at.selectbox(key="scn_type").set_value("Key rates").run()  # nothing moves: flat shock, zero P&L
    limits = _chart_limits(at)
    shock_axis, pnl_axis = limits["scn_curve_chart"]["yaxis2"], limits["scn_waterfall"]["yaxis"]
    assert shock_axis["max"] - shock_axis["min"] >= 2.0 and pnl_axis["max"] - pnl_axis["min"] >= 2.0
    at.radio(key="scn_chart_view").set_value("ZERO RATES").run()
    at.number_input(key="scn_horizon").set_value(365).run()
    assert set(_chart_limits(at)) == {"scn_curve_chart", "scn_waterfall"} and not at.exception


def test_scenario_tab_reports_failed_scenarios_and_floored_vols():
    at = _app(scn_test_fragile=True)
    assert not at.error and any("Parallel -100bp: the 10Y quote is below 3%" in w.value for w in at.warning)
    grid = at.dataframe[1].value
    assert "Parallel -100bp" not in set(grid["SCENARIO"]) and len(grid) == len(standard_library())
    at.number_input(key="scn_par_bp").set_value(-150.0).run()  # the explain fails; the library grid still shows
    assert not at.exception and at.error and not at.metric and len(at.dataframe) == 1
    broken = _app(scn_test_broken=True)
    assert [e.value for e in broken.error] == ["Cannot value the portfolio today: no finite value for BROKEN"]
    low = _app(scn_test_lowvol=True)
    assert not low.warning
    low.number_input(key="scn_vol_normal").set_value(-60.0).run()
    assert not low.exception and any("LOW VOL" in w.value and "below zero" in w.value for w in low.warning)


def test_scenario_tab_with_a_quoteless_treasury_curve():
    at = _app(scn_test_flat_govt=True)
    assert not at.error and not at.warning and len(at.dataframe) == 2
    assert at.checkbox(key="scn_move_govt").disabled and len(at.radio(key="scn_chart_curve").options) == 1
    at.number_input(key="scn_horizon").set_value(30).run()  # a curve without quotes cannot be rolled forward
    assert not at.exception and any("govt" in e.value for e in at.error)
