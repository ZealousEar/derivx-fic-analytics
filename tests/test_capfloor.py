import datetime as dt
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from derivx.rates.capfloor import (
    CapFloor,
    bachelier_delta,
    bachelier_price,
    bachelier_vega,
    black_delta,
    black_price,
    black_vega,
    cap_floor_parity,
    capfloor_position,
    implied_black_vol,
    implied_normal_vol,
    model_price,
    model_vega,
    shifted_black_delta,
    shifted_black_price,
    shifted_black_vega,
)
from derivx.rates.curves import CONVENTIONS, CurveQuote, Market, bootstrap, flat_curve
from derivx.rates.examples import EXAMPLE_ASOF, example_market, sofr_example_rows
from derivx.rates.swaps import VanillaSwap
from derivx.risk.sensitivities import key_rate_pv01, parallel_pv01

D = dt.date.fromisoformat
ROOT = Path(__file__).resolve().parents[1]
UNIT = {"bachelier": 1e-4, "black": 0.01, "shifted_black": 0.01}
MODELS = [("bachelier", 0.0095, 0.0), ("black", 0.27, 0.0), ("shifted_black", 0.18, 0.02)]
SEASONED = dict(start=dt.date(2025, 7, 15), tenor="3Y")


@pytest.fixture(scope="module")
def market():
    return example_market()


def _from_spec(spec: dict) -> CapFloor:
    vol = dict(spec["vol"]) if isinstance(spec["vol"], list) else spec["vol"]
    return CapFloor(**{**spec, "vol": vol, "start": D(spec["start"]) if spec["start"] else None})


def _case_market(asof: dt.date, preset: str, interpolation: str, quote_shift: float, fixings=None) -> Market:
    """tools/golden/capfloor.independent_market: the example quotes moved
    by ``quote_shift`` on a preset's conventions, past fixings from a flat
    curve ``fixings = (date, rate)``."""

    quotes = [CurveQuote(r["kind"], r["tenor"], r["rate"] + quote_shift) for r in sofr_example_rows()]
    market = Market(bootstrap(asof, quotes, preset, interpolation))
    return market.replace(fixings=flat_curve(*fixings)) if fixings else market


# --------------------------------------------------------------------------
# QuantLib reference values
# --------------------------------------------------------------------------


def test_caps_and_floors_match_quantlib(golden, market):
    """Caps/floors, ITM/ATM/OTM, 3M/6M, 2Y/5Y/10Y, forward start, vol term
    structure, seasoned and negative strikes against QuantLib's Black,
    displaced Black and Bachelier cap/floor engines on the same schedule."""

    data = golden("capfloor")
    assert D(data["asof"]) == market.asof
    assert len(data["cases"]) == 100
    for case in data["cases"]:
        cf = _from_spec(case["spec"])
        name = case["name"]
        table = cf.caplets(market)
        rows = case["caplets"]
        assert [d.isoformat() for d in table["fix"]] == [r[0] for r in rows], name
        assert [d.isoformat() for d in table["pay"]] == [r[1] for r in rows], name
        ql_price = np.array([r[2] for r in rows])
        # atol 1e-9 (on 1e7 notional) covers deep OTM caplets worth ~1e-5: QuantLib's normal CDF has
        # relative errors up to ~1e-7 for z in (-7, -5), where it switches to an asymptotic series.
        assert cf.pv(market) == pytest.approx(case["npv"], rel=1e-10, abs=1e-9), name
        np.testing.assert_allclose(table["pv"], ql_price, rtol=1e-10, atol=1e-9, err_msg=name)
        np.testing.assert_allclose(table["vega"] / UNIT[cf.model], [r[3] for r in rows], rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(table["forward"], [r[4] for r in rows], rtol=0, atol=1e-15, err_msg=name)
        np.testing.assert_allclose(table["df"], [r[5] for r in rows], rtol=1e-15, err_msg=name)
        assert cf.atm_strike(market) == pytest.approx(case["atm"], abs=1e-15), name
        for fix, rate in case["fixings"]:  # the fixed period priced off the fixing QuantLib was given
            row = table[table["fix"] == D(fix)].iloc[0]
            assert row["fixed"] and row["forward"] == pytest.approx(rate, abs=1e-16)
        if "implied" in case:
            vol = cf.implied_flat_vol(case["implied"]["price"], market)
            assert vol == pytest.approx(case["implied"]["vol"], rel=1e-10), name
            assert vol == pytest.approx(1.15 * cf.vol, rel=1e-10), name


def _assert_matches_independent(cf: CapFloor, market: Market, ref: dict, name: str) -> None:
    """Dates, NPV, ATM rate and caplet vegas against QuantLib's own schedule
    (tools/golden/capfloor.ql_independent)."""

    table = cf.caplets(market)
    dates = [[d.isoformat() for d in row] for row in zip(table["fix"], table["start"], table["end"], table["pay"])]
    assert dates == [r[:4] for r in ref["caplets"]], name
    assert cf.pv(market) == pytest.approx(ref["npv"], rel=1e-10, abs=1e-9), name
    assert cf.atm_strike(market) == pytest.approx(ref["atm"], abs=1e-15), name
    vegas = [r[4] for r in ref["caplets"]]
    np.testing.assert_allclose(table["vega"] / UNIT[cf.model], vegas, rtol=1e-10, atol=1e-8, err_msg=name)


def test_dates_and_values_match_quantlibs_own_schedule(golden):
    """QuantLib generates these schedules itself (MakeCapFloor, or
    ql.Schedule from an explicit start) on awkward valuation dates (a
    Saturday, month and year ends, a leap-year February) for USD SOFR, EUR
    ESTR (negative rates) and GBP SONIA conventions on every curve
    interpolation, including seasoned caps with past fixings."""

    cases = golden("capfloor")["independent"]
    assert {c["preset"] for c in cases} == set(CONVENTIONS) - {"USD UST"}
    assert {c["interpolation"] for c in cases} == {"loglinear", "linear_zero", "monotone_convex"}
    assert any(c["quote_shift"] < 0 for c in cases)
    # forward starts that QuantLib's Following would move into the next month are given to it explicitly
    assert sum(c["explicit_start"] for c in cases) >= 2
    seasoned = 0
    for case in cases:
        fixings = (D(case["fixings"][0]), case["fixings"][1]) if case["fixings"] else None
        market = _case_market(D(case["asof"]), case["preset"], case["interpolation"], case["quote_shift"], fixings)
        cf = _from_spec(case["spec"])
        _assert_matches_independent(cf, market, case, case["name"])
        seasoned += bool(cf.caplets(market)["fixed"].any())
    assert seasoned >= 3


def test_random_caps_match_live_quantlib():
    """Random conventions, interpolations, valuation dates (any calendar day),
    structures, strikes, models and seasoning against QuantLib's own
    schedule (skipped when QuantLib is not installed)."""

    pytest.importorskip("QuantLib")
    sys.path.insert(0, str(ROOT / "tools"))
    from golden.capfloor import ql_independent

    rng = random.Random(7)
    for i in range(40):
        preset = ("USD SOFR", "EUR ESTR", "GBP SONIA")[i % 3]
        conv = CONVENTIONS[preset]
        negative = preset == "EUR ESTR" and i % 2 == 0
        asof = EXAMPLE_ASOF + dt.timedelta(days=rng.randrange(-900, 900))
        model, vol, shift = MODELS[i % 3] if not (negative and i % 3 == 1) else MODELS[2]
        vol *= rng.uniform(0.3, 2.0)
        months = rng.choice([1, 3, 6, 12])
        tenor = rng.choice(["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "27M"])
        fwd = rng.choice([None, None, "1M", "3M", "1Y", "2Y", "10Y"])
        low = 0.01 if model == "black" else (-0.015 if negative else -0.01)  # above -shift for shifted Black
        terms = dict(
            index_months=months, day_count=conv.irs_float_day_count, calendar=conv.calendar,
            fixing_lag=conv.spot_lag,
        )
        fixings = None
        if i % 4 == 0:
            start = asof - dt.timedelta(days=rng.randrange(20, 700))
            fixings = (start - dt.timedelta(days=10), 0.045)
            terms.update(start=start, exclude_first=bool(rng.randrange(2)))
        else:
            terms.update(forward_start=fwd)
        market = _case_market(asof, preset, rng.choice(["loglinear", "linear_zero", "monotone_convex"]),
                              -0.045 if negative else 0.0, fixings)
        cf = CapFloor(3e6, rng.choice(["cap", "floor"]), rng.uniform(low, 0.07), tenor, vol, model, shift, **terms)
        if not cf.caplets(market).shape[0]:
            continue
        _assert_matches_independent(cf, market, ql_independent(cf, market, fixings), f"case {i}: {preset} {asof} {cf}")


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

GRID = [
    (F, K, T, s) for F in (0.01, 0.035, 0.06) for K in (0.005, 0.03, 0.05) for T in (0.1, 1.0, 7.5) for s in (0.1, 0.4)
]


def test_put_call_parity_for_each_kernel():
    for F, K, T, s in GRID:
        assert black_price(F, K, T, s) - black_price(F, K, T, s, False) == pytest.approx(F - K, abs=1e-16)
        f, k = F - 0.02, K - 0.02
        shifted = shifted_black_price(f, k, T, s, 0.02) - shifted_black_price(f, k, T, s, 0.02, False)
        assert shifted == pytest.approx(F - K, abs=1e-16)
        sn = s * 0.03
        assert bachelier_price(F - 0.03, K - 0.03, T, sn) - bachelier_price(F - 0.03, K - 0.03, T, sn, False) == (
            pytest.approx(F - K, abs=1e-16)
        )
        for delta in (black_delta, lambda *a: shifted_black_delta(*a[:4], 0.01, *a[4:])):
            assert delta(F, K, T, s) - delta(F, K, T, s, False) == pytest.approx(1.0, abs=1e-15)


def test_kernel_greeks_match_finite_differences():
    h = 1e-6
    for F, K, T, s in GRID:
        sn = s * 0.03
        cases = [
            (lambda f, v: black_price(f, K, T, v), black_vega(F, K, T, s), black_delta(F, K, T, s), s),
            (
                lambda f, v: shifted_black_price(f, K, T, v, 0.015),
                shifted_black_vega(F, K, T, s, 0.015),
                shifted_black_delta(F, K, T, s, 0.015),
                s,
            ),
            (lambda f, v: bachelier_price(f, K, T, v), bachelier_vega(F, K, T, sn), bachelier_delta(F, K, T, sn), sn),
        ]
        for price, vega, delta, vol in cases:
            assert (price(F, vol + h) - price(F, vol - h)) / (2 * h) == pytest.approx(vega, rel=1e-6, abs=1e-8)
            assert (price(F + h, vol) - price(F - h, vol)) / (2 * h) == pytest.approx(delta, rel=1e-6, abs=1e-8)


def test_implied_vols_round_trip():
    # Only where there is time value to invert: deep in the money it is lost in rounding.
    for F, K, T, s in GRID:
        for call in (True, False):
            intrinsic = max(F - K if call else K - F, 0.0)
            price = black_price(F, K, T, s, call)
            if price - intrinsic > 1e-8:
                assert implied_black_vol(price, F, K, T, call) == pytest.approx(s, rel=1e-10)
            shifted = shifted_black_price(F - 0.02, K - 0.02, T, s, 0.02, call)
            if shifted - intrinsic > 1e-8:
                vol = implied_black_vol(shifted, F - 0.02, K - 0.02, T, call, shift=0.02)
                assert vol == pytest.approx(s, rel=1e-10)
            sn = s * 0.03
            normal = bachelier_price(F - 0.03, K - 0.03, T, sn, call)
            if normal - intrinsic > 1e-8:
                assert implied_normal_vol(normal, F - 0.03, K - 0.03, T, call) == pytest.approx(sn, rel=1e-10)
    assert implied_normal_vol(0.0, 0.03, 0.04, 1.0) == 0.0  # zero time value is zero vol
    assert implied_black_vol(0.01, 0.04, 0.03, 1.0) == 0.0


def test_implied_vols_reject_prices_outside_arbitrage_bounds():
    with pytest.raises(ValueError, match="below its zero-vol"):
        implied_black_vol(0.009, 0.04, 0.03, 1.0)
    with pytest.raises(ValueError, match="infinite-vol"):
        implied_black_vol(0.04, 0.04, 0.03, 1.0)  # a call is worth less than the forward
    with pytest.raises(ValueError, match="infinite-vol"):
        implied_black_vol(0.05, 0.04, 0.03, 1.0, is_call=False)  # a put less than the strike
    with pytest.raises(ValueError, match="below its zero-vol"):
        implied_normal_vol(-1e-6, 0.03, 0.04, 1.0)
    with pytest.raises(ValueError, match="expired"):
        implied_normal_vol(0.001, 0.03, 0.03, 0.0)
    with pytest.raises(ValueError, match="finite"):
        implied_normal_vol(float("nan"), 0.03, 0.03, 1.0)


def test_black_rejects_non_positive_rates_and_points_to_other_models():
    for F, K in [(-0.001, 0.02), (0.02, 0.0), (0.0, 0.01)]:
        with pytest.raises(ValueError, match="normal .*shifted"):
            black_price(F, K, 1.0, 0.2)
    with pytest.raises(ValueError, match="increase the shift"):
        shifted_black_price(-0.03, 0.01, 1.0, 0.2, 0.02)
    assert shifted_black_price(-0.005, -0.004, 1.0, 0.2, 0.02) > 0.0
    with pytest.raises(ValueError, match="volatility"):
        bachelier_price(0.01, 0.01, 1.0, -0.001)
    assert math.isfinite(bachelier_price(-0.01, -0.02, 2.0, 0.008))


def test_kernels_reject_non_finite_inputs_and_unknown_models():
    nan = float("nan")
    for kernel in (black_price, bachelier_price, black_vega, bachelier_vega, black_delta, bachelier_delta):
        for args in ((nan, 0.03, 1.0), (0.03, nan, 1.0), (0.03, 0.03, nan), (0.03, math.inf, 1.0)):
            with pytest.raises(ValueError, match="finite"):
                kernel(*args, 0.2)
    with pytest.raises(ValueError, match="shift must be a finite"):
        shifted_black_price(0.01, 0.01, 1.0, 0.2, nan)
    with pytest.raises(ValueError, match="finite"):
        implied_black_vol(0.01, 0.03, 0.03, nan)
    for dispatch in (model_price, model_vega):
        with pytest.raises(ValueError, match="model must be one of"):
            dispatch("sabr", 0.03, 0.03, 1.0, 0.2)


# --------------------------------------------------------------------------
# Cap / floor properties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model,vol,shift", MODELS)
def test_cap_minus_floor_is_payer_swap(market, model, vol, shift):
    for extra in ({}, {"forward_start": "2Y"}, SEASONED, {"index_months": 6, "tenor": "10Y"}):
        terms = {"tenor": "5Y", **extra}
        cf = CapFloor(1e7, "cap", 0.0335, terms.pop("tenor"), vol, model, shift, **terms)
        parity = cap_floor_parity(cf, market)
        assert parity["residual"] == pytest.approx(0.0, abs=1e-8 * 1e7), extra
        assert parity["cap"] > 0 and parity["floor"] > 0
    atm = replace(cf, strike=cf.atm_strike(market))
    assert atm.pv(market) == pytest.approx(replace(atm, kind="floor").pv(market), rel=1e-12)


def test_black_and_bachelier_agree_when_vols_are_converted_by_price(market):
    black = CapFloor(1e7, "cap", 0.036, "10Y", 0.27, "black")
    table = black.caplets(market)
    normal_vols = {
        t: implied_normal_vol(pv / (1e7 * tau * df), f, 0.036, t)
        for t, pv, tau, df, f in zip(table["expiry"], table["pv"], table["tau"], table["df"], table["forward"])
    }
    normal = replace(black, model="bachelier", vol=normal_vols)
    assert normal.pv(market) == pytest.approx(black.pv(market), rel=1e-12)
    np.testing.assert_allclose(normal.caplets(market)["pv"], table["pv"], rtol=1e-11)


@pytest.mark.parametrize("model,vol,shift", MODELS)
def test_vega_matches_a_market_vol_shift(market, model, vol, shift):
    cf = CapFloor(1e7, "floor", 0.034, "7Y", {1.0: vol, 5.0: 0.9 * vol}, model, shift, forward_start="1Y")
    field = "vol_shift_normal" if model == "bachelier" else "vol_shift_lognormal"
    h = 1e-3 * UNIT[model]
    up = cf.pv(market.replace(**{field: h}))
    down = cf.pv(market.replace(**{field: -h}))
    assert cf.vega(market) == pytest.approx((up - down) / (2 * h) * UNIT[model], rel=1e-7)
    assert cf.vega(market) == pytest.approx(cf.caplets(market)["vega"].sum(), rel=1e-14)
    other = "vol_shift_lognormal" if model == "bachelier" else "vol_shift_normal"
    assert cf.pv(market.replace(**{other: 0.5})) == cf.pv(market)  # the other model's shift does nothing


def test_seasoned_cap_values_the_fixed_caplet_at_intrinsic(market):
    cf = CapFloor(1e7, "cap", 0.03, SEASONED["tenor"], 0.0095, start=SEASONED["start"])
    table = cf.caplets(market)
    fixed = table[table["fixed"]]
    assert len(fixed) == 1 and fixed["fix"].iloc[0] < market.asof < fixed["pay"].iloc[0]
    row = fixed.iloc[0]
    assert row["pv"] == pytest.approx(1e7 * row["tau"] * row["df"] * (row["forward"] - 0.03), rel=1e-14)
    assert row["vega"] == 0.0 and row["expiry"] == 0.0
    # a different vol leaves it alone; a different fixings curve moves only it
    assert replace(cf, vol=0.02).caplets(market)["pv"].iloc[0] == row["pv"]
    fixings = flat_curve(dt.date(2025, 1, 2), 0.05)
    moved = cf.caplets(market.replace(fixings=fixings))
    assert moved["forward"].iloc[0] == pytest.approx(fixings.forward_rate(row["start"], row["end"]), rel=1e-14)
    np.testing.assert_array_equal(moved["pv"].iloc[1:], table["pv"].iloc[1:])
    # periods already paid are gone; a fully expired cap is worth nothing
    assert table["pay"].min() > market.asof
    expired = replace(cf, start=dt.date(2020, 1, 15), tenor="2Y")
    assert expired.pv(market) == 0.0 and expired.caplets(market).empty
    with pytest.raises(ValueError, match="fixed"):
        cf.implied_flat_vol(row["pv"], market.replace(curve=market.curve.rebuilt(asof=dt.date(2028, 4, 20))))


def test_fixing_today_pays_intrinsic(market):
    # the first period fixes on the valuation date when the schedule starts at spot
    cf = CapFloor(1e7, "floor", 0.05, "1Y", 0.01, exclude_first=False)
    first = cf.caplets(market).iloc[0]
    assert first["fix"] == market.asof and first["fixed"]
    assert first["pv"] == pytest.approx(1e7 * first["tau"] * first["df"] * (0.05 - first["forward"]), rel=1e-14)


def test_schedule_conventions_and_first_period_rule(market):
    spot = CapFloor(1e6, "cap", 0.03, "5Y", 0.01)
    periods = spot.periods(market.asof)
    assert len(periods) == 19  # 20 quarters, the first (fixing at trade) excluded
    full = replace(spot, exclude_first=False).periods(market.asof)
    assert len(full) == 20 and full[0].fix == market.asof and full[0].start == D("2026-10-21")
    assert full[1:] == periods
    forward = CapFloor(1e6, "cap", 0.03, "5Y", 0.01, forward_start="1Y")
    assert len(forward.periods(market.asof)) == 20  # a forward start keeps its first caplet
    by_start = {p.start: p for p in periods}
    assert D("2028-10-21") not in by_start and by_start[D("2028-10-23")].end == D("2029-01-22")  # Saturday, MF
    assert by_start[D("2030-01-22")].fix == D("2030-01-17")  # two SOFR business days back over MLK Day
    assert all(p.pay == p.end and p.tau == (p.end - p.start).days / 360.0 for p in periods)
    six = CapFloor(1e6, "cap", 0.03, "10Y", 0.01, index_months=6)
    assert len(six.periods(market.asof)) == 19
    stub = CapFloor(1e6, "cap", 0.03, "13M", 0.01, exclude_first=False).periods(market.asof)
    assert (stub[0].end - stub[0].start).days < 40  # short front stub, regular periods to maturity
    lower = CapFloor(1e6, "cap", 0.03, "5Y", 0.01, bdc="mf", day_count="act/360")
    assert (lower.bdc, lower.day_count) == ("MF", "ACT/360") and lower.periods(market.asof) == periods


def test_forward_start_rolls_from_its_adjusted_start_like_a_swap(market):
    # spot 2026-10-21 + 2Y is Saturday 2028-10-21: the cap starts on Monday the 23rd and rolls on the 23rd
    cf = CapFloor(1e6, "cap", 0.03, "3Y", 0.01, forward_start="2Y", exclude_first=False)
    dates = cf.schedule(market.asof)
    assert (dates[0], dates[1], dates[-1]) == (D("2028-10-23"), D("2029-01-23"), D("2031-10-23"))
    swap = VanillaSwap.from_conventions("USD SOFR", "3Y", 0.03, kind="IRS", forward_start="2Y", trade_date=market.asof)
    assert swap.schedule_dates(market.asof)[1] == dates
    assert cf.pinned(market.asof).start == D("2028-10-23")


def test_pinned_trade_keeps_its_dates_on_a_later_valuation_date(market):
    for cf in (CapFloor(1e7, "cap", 0.035, "5Y", 0.01), CapFloor(1e7, "cap", 0.035, "3Y", 0.01, forward_start="6M")):
        pinned = cf.pinned(market.asof)
        assert pinned.periods(market.asof) == cf.periods(market.asof)
        assert pinned.pv(market) == cf.pv(market)
        later = market.asof + dt.timedelta(days=45)
        assert pinned.periods(later) == cf.periods(market.asof)
        assert cf.periods(later) != cf.periods(market.asof)


def test_vol_term_structure_is_linear_in_expiry_and_flat_outside():
    cf = CapFloor(1e6, "cap", 0.03, "5Y", {1.0: 0.01, 3.0: 0.012})
    np.testing.assert_allclose(cf.vol_at([0.0, 0.5, 1.0, 2.0, 3.0, 9.0]), [0.01, 0.01, 0.01, 0.011, 0.012, 0.012])
    assert CapFloor(1e6, "cap", 0.03, "5Y", 0.0123).vol_at(4.2) == 0.0123


def test_premium_annuity_and_implied_flat_vol(market):
    cf = CapFloor(2.5e7, "cap", 0.037, "10Y", {0.5: 0.008, 5.0: 0.0105, 10.0: 0.009})
    pv = cf.pv(market)
    assert cf.premium_bp_running(market) == pytest.approx(pv / cf.annuity(market) * 1e4, rel=1e-14)
    table = cf.caplets(market)
    assert cf.annuity(market) == pytest.approx(2.5e7 * (table["tau"] * table["df"]).sum(), rel=1e-14)
    flat = cf.implied_flat_vol(pv, market)
    assert replace(cf, vol=flat).pv(market) == pytest.approx(pv, rel=1e-12)
    assert 0.008 < flat < 0.0105
    short = replace(cf, notional=-2.5e7)
    assert short.pv(market) == pytest.approx(-pv, rel=1e-14) and short.implied_flat_vol(-pv, market) == flat
    with pytest.raises(ValueError, match="price 1000 is above its zero-vol .*value -"):  # in currency, sold
        short.implied_flat_vol(1000.0, market)
    black = CapFloor(1e7, "floor", 0.033, "5Y", 0.25, "black")
    assert black.implied_flat_vol(black.pv(market), market) == pytest.approx(0.25, rel=1e-11)
    with pytest.raises(ValueError, match="infinite-vol"):
        black.implied_flat_vol(1e7, market)
    with pytest.raises(ValueError, match="zero-vol"):
        cf.implied_flat_vol(-1.0, market)


def test_position_risk_signs_and_cash(market):
    cap = capfloor_position(CapFloor(1e7, "cap", 0.036, "5Y", 0.01), asof=market.asof)
    floor = capfloor_position(CapFloor(1e7, "floor", 0.036, "5Y", 0.25, "black"), asof=market.asof)
    assert (cap.kind, cap.vol_model, floor.vol_model) == ("cap", "normal", "lognormal")
    assert cap.name == "5Y CAP 3.600% 3M" and cap.details["type"] == "cap"
    assert parallel_pv01(cap.pv, market) > 0 > parallel_pv01(floor.pv, market)
    rows = key_rate_pv01(cap.pv, market)
    assert sum(r["pv01"] for r in rows) == pytest.approx(parallel_pv01(cap.pv, market), rel=0.02)
    assert rows[0]["pv01"] == pytest.approx(0.0, abs=1.0)  # 1W deposit barely matters to a 5Y cap
    trade = cap.details["trade"]
    first = trade.caplets(market).iloc[0]
    expected = 1e7 * first["tau"] * max(first["forward"] - 0.036, 0.0)
    assert cap.cash(market, market.asof, first["pay"]) == pytest.approx(expected, abs=1e-9)
    assert cap.cash(market, first["pay"], first["pay"]) == 0.0


def test_negative_rate_curve_needs_a_normal_or_shifted_model():
    quotes = [CurveQuote("OIS", t, r) for t, r in [("1M", -0.006), ("1Y", -0.005), ("5Y", -0.002), ("10Y", 0.001)]]
    market = Market(bootstrap(EXAMPLE_ASOF, quotes, "EUR ESTR"))
    with pytest.raises(ValueError, match="caplet fixing .*normal"):
        CapFloor(1e7, "cap", 0.001, "5Y", 0.3, "black", calendar="TARGET").pv(market)
    assert CapFloor(1e7, "floor", -0.004, "5Y", 0.006, calendar="TARGET").pv(market) > 0
    assert CapFloor(1e7, "cap", -0.004, "5Y", 0.15, "shifted_black", 0.02, calendar="TARGET").pv(market) > 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"kind": "collar"}, "kind"),
        ({"model": "sabr"}, "model"),
        ({"notional": 0.0}, "notional"),
        ({"tenor": "3W"}, "months or years"),
        ({"tenor": "5X"}, "tenor"),
        ({"index_months": 4}, "index_months"),
        ({"forward_start": "1Y", "start": dt.date(2027, 1, 4)}, "either"),
        ({"vol": -0.01}, "non-negative"),
        ({"vol": {}}, "empty"),
        ({"vol": {1.0: 0.01, -1.0: 0.01}}, "expiries"),
        ({"model": "black", "strike": -0.001}, "positive strike"),
        ({"model": "shifted_black", "strike": -0.03, "shift": 0.02}, "strike \\+ shift"),
        ({"day_count": "ACT/ACT ICMA"}, "bond day count"),
        ({"calendar": "MARS"}, "calendar"),
        ({"bdc": "XX"}, "bdc"),
        ({"fixing_lag": 1.5}, "fixing_lag"),
        ({"fixing_lag": -1}, "fixing_lag"),
        ({"fixing_lag": None}, "fixing_lag"),
        ({"model": "shifted_black", "shift": float("nan")}, "shift"),
        ({"strike": float("nan")}, "strike"),
    ],
)
def test_bad_inputs_raise_clear_errors(kwargs, match):
    base = dict(notional=1e6, kind="cap", strike=0.03, tenor="5Y", vol=0.01)
    with pytest.raises(ValueError, match=match):
        CapFloor(**{**base, **kwargs})


def test_a_cap_with_no_caplet_left_after_the_first_is_excluded(market):
    with pytest.raises(ValueError, match="no caplet"):
        CapFloor(1e6, "cap", 0.03, "3M", 0.01).pv(market)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

APP = """
import streamlit as st
from derivx.rates.examples import example_market
from derivx.ui.rates_capfloor import render

st.session_state["cap_test_positions"] = render(example_market())
"""


def _keys(node) -> list:
    children = getattr(node, "children", None)
    if not isinstance(children, dict):
        return []
    keys = []
    for child in children.values():
        if getattr(child, "key", None) is not None:
            keys.append(child.key)
        keys += _keys(child)
    return keys


def _assert_tab_contract(at: AppTest) -> None:
    """No traceback, every widget and chart keyed ``cap_``, and every axis of
    every chart bounded (declarative limits plus the zoom guard's meta)."""

    assert not at.exception, [e.value for e in at.exception]
    keys = _keys(at.main)
    assert keys and all(k.startswith("cap_") for k in keys), keys
    for chart in at.get("plotly_chart"):
        layout = json.loads(chart.proto.spec)["layout"]
        limits = layout["meta"]["zoom"]["axes"]
        axes = [name for name in layout if name.startswith(("xaxis", "yaxis"))]
        assert axes and set(axes) <= set(limits), (axes, list(limits))
        for name in axes:
            assert layout[name]["minallowed"] == limits[name]["min"] == layout[name]["range"][0], name
            assert layout[name]["maxallowed"] == limits[name]["max"] == layout[name]["range"][1], name


def _app() -> AppTest:
    at = AppTest.from_string(APP, default_timeout=120)
    at.run()
    _assert_tab_contract(at)
    return at


def test_tab_renders_and_returns_the_position():
    at = _app()
    positions = at.session_state["cap_test_positions"]
    assert len(positions) == 1 and positions[0].kind == "cap"
    assert [m.label for m in at.metric][:2] == ["PV", "PREMIUM (BP RUNNING)"]
    assert not at.error
    pv = float(at.metric[0].value.replace(",", ""))
    assert pv == pytest.approx(positions[0].pv(example_market()), abs=0.01)
    assert len(at.get("plotly_chart")) == 2 and len(at.get("download_button")) == 2
    assert not at.warning


WIDGETS = {
    "cap_kind": "radio",
    "cap_index": "radio",
    "cap_strike_mode": "radio",
    "cap_vol_mode": "radio",
    "cap_tenor": "selectbox",
    "cap_forward_start": "selectbox",
    "cap_model": "selectbox",
    "cap_atm_offset_bp": "number_input",
    "cap_strike_pct": "number_input",
    "cap_flat_vol_bachelier": "number_input",
}


def _set(at: AppTest, key: str, value) -> None:
    getattr(at, WIDGETS[key])(key=key).set_value(value).run()
    _assert_tab_contract(at)


@pytest.mark.parametrize(
    "changes",
    [
        {"cap_kind": "Floor", "cap_index": "6M", "cap_tenor": "10Y", "cap_forward_start": "2Y"},
        {"cap_model": "Black (%)", "cap_atm_offset_bp": 100.0},
        {"cap_model": "Shifted Black (%)", "cap_vol_mode": "Term structure"},
        {"cap_vol_mode": "Term structure", "cap_tenor": "30Y", "cap_forward_start": "10Y"},
        {"cap_strike_mode": "Absolute %", "cap_strike_pct": -1.0, "cap_flat_vol_bachelier": 0.0, "cap_tenor": "1Y"},
    ],
)
def test_tab_handles_edge_inputs(changes):
    at = _app()
    for key, value in changes.items():
        _set(at, key, value)
    assert not at.error, [e.value for e in at.error]
    assert len(at.session_state["cap_test_positions"]) == 1
    beyond_curve = any("after the curve's last quote" in w.value for w in at.warning)
    assert beyond_curve == (changes.get("cap_forward_start") == "10Y")  # 10Yx30Y pays past the 30Y quote


def test_tab_reports_black_with_a_negative_strike_as_an_error():
    at = _app()
    _set(at, "cap_model", "Black (%)")
    _set(at, "cap_atm_offset_bp", -500.0)
    assert at.error and "positive strike" in at.error[0].value
    assert at.session_state["cap_test_positions"] == []
