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
    lyashenko_mercurio_time,
    model_price,
    model_vega,
    shifted_black_delta,
    shifted_black_price,
    shifted_black_vega,
)
from derivx.rates.conventions import Calendar
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
            index_months=months,
            day_count=conv.irs_float_day_count,
            calendar=conv.calendar,
            fixing_lag=conv.spot_lag,
        )
        fixings = None
        if i % 4 == 0:
            start = asof - dt.timedelta(days=rng.randrange(20, 700))
            fixings = (start - dt.timedelta(days=10), 0.045)
            terms.update(start=start, exclude_first=bool(rng.randrange(2)))
        else:
            terms.update(forward_start=fwd)
        market = _case_market(
            asof,
            preset,
            rng.choice(["loglinear", "linear_zero", "monotone_convex"]),
            -0.045 if negative else 0.0,
            fixings,
        )
        cf = CapFloor(3e6, rng.choice(["cap", "floor"]), rng.uniform(low, 0.07), tenor, vol, model, shift, **terms)
        if not cf.caplets(market).shape[0]:
            continue
        _assert_matches_independent(cf, market, ql_independent(cf, market, fixings), f"case {i}: {preset} {asof} {cf}")


def _assert_matches_compounded(cf: CapFloor, market: Market, ref: dict, name: str) -> None:
    """Dates, values, forwards, DFs, effective vols and vegas against
    QuantLib's compounded overnight caplets (tools/golden/capfloor.ql_compounded)."""

    table = cf.caplets(market)
    dates = [[d.isoformat() for d in r] for r in zip(table["first_fix"], table["fix"], table["start"], table["end"],
                                                     table["pay"])]  # fmt: skip
    assert dates == [r[:5] for r in ref["caplets"]], name
    # atol 1e-8 on 1e7 notional: QuantLib's normal CDF tail (see test_caps_and_floors_match_quantlib)
    assert cf.pv(market) == pytest.approx(ref["npv"], rel=1e-10, abs=1e-8), name
    np.testing.assert_allclose(table["pv"], [r[5] for r in ref["caplets"]], rtol=1e-10, atol=1e-8, err_msg=name)
    np.testing.assert_allclose(table["forward"], [r[6] for r in ref["caplets"]], rtol=0, atol=1e-14, err_msg=name)
    np.testing.assert_allclose(table["df"], [r[7] for r in ref["caplets"]], rtol=1e-14, err_msg=name)
    live = [(ours, r[8]) for ours, r, fixed in zip(table["effective_vol"], ref["caplets"], table["fixed"]) if not fixed]
    assert all(theirs is not None for _, theirs in live), name
    np.testing.assert_allclose([o for o, _ in live], [t for _, t in live], rtol=1e-12, err_msg=name)
    # QuantLib's vega is a central difference of its prices: agreement to its truncation error
    np.testing.assert_allclose(table["vega"] / UNIT[cf.model], [r[9] for r in ref["caplets"]], rtol=1e-7, atol=1e-3,
                               err_msg=name)  # fmt: skip


def test_compounded_caps_floors_and_collars_match_quantlib(golden):
    """Caps, floors and collars on SOFR, €STR and SONIA compounded in
    arrears against QuantLib's OvernightLeg with naked caps/floors and the
    BlackCompoundingOvernightIndexedCouponPricer: Lyashenko-Mercurio and
    effective vol inputs, all three models, spot and forward starts, sold
    trades, negative rates, a valuation date inside a period (past daily
    fixings) and a period that has fixed but not yet paid."""

    cases = golden("capfloor")["compounded"]
    specs = [c["spec"] for c in cases]
    assert {c["preset"] for c in cases} == {"USD SOFR", "EUR ESTR", "GBP SONIA"}
    assert {s["kind"] for s in specs} == {"cap", "floor", "collar"}
    assert {s["model"] for s in specs} == {"bachelier", "black", "shifted_black"}
    assert {s["effective_vol"] for s in specs} == {True, False} and any(s["notional"] < 0 for s in specs)
    assert {s["payment_lag"] for s in specs} == {0, 1, 2}
    in_period = ended_unpaid = 0
    for case in cases:
        fixings = (D(case["fixings"][0]), case["fixings"][1]) if case["fixings"] else None
        market = _case_market(D(case["asof"]), case["preset"], case["interpolation"], case["quote_shift"], fixings)
        cf = _from_spec(case["spec"])
        assert cf.compounded, case["name"]
        _assert_matches_compounded(cf, market, case, case["name"])
        table = cf.caplets(market)
        in_period += bool(((table["start"] < market.asof) & ~table["fixed"]).any())
        ended_unpaid += bool(table["fixed"].any())
    assert in_period >= 2 and ended_unpaid >= 1


def test_random_compounded_caps_match_live_quantlib():
    """Random presets, interpolations, valuation dates, structures, kinds
    (collars too), models, vol inputs and seasoning against QuantLib's
    compounded overnight caplets (skipped without QuantLib)."""

    pytest.importorskip("QuantLib")
    sys.path.insert(0, str(ROOT / "tools"))
    from golden.capfloor import ql_compounded

    rng = random.Random(11)
    for i in range(30):
        preset = ("USD SOFR", "EUR ESTR", "GBP SONIA")[i % 3]
        negative = preset == "EUR ESTR" and i % 2 == 0
        asof = EXAMPLE_ASOF + dt.timedelta(days=rng.randrange(-700, 700))
        model, vol, shift = MODELS[i % 3] if not (negative and i % 3 == 1) else MODELS[2]
        vol *= rng.uniform(0.3, 2.0)
        fixings, terms = None, dict(index_months=rng.choice([1, 3, 6, 12]))
        if i % 4 == 0:
            terms["start"] = asof - dt.timedelta(days=rng.randrange(5, 400))
            fixings = (terms["start"] - dt.timedelta(days=10), 0.045)
        else:
            terms["forward_start"] = rng.choice([None, None, "3M", "1Y", "2Y"])
        market = _case_market(asof, preset, rng.choice(["loglinear", "linear_zero", "monotone_convex"]),
                              -0.045 if negative else 0.0, fixings)  # fmt: skip
        kind = rng.choice(["cap", "floor", "collar"])
        low = 0.012 if model == "black" else (-0.015 if negative else -0.01)
        strike = rng.uniform(low + 0.01, 0.07)
        extra = dict(floor_strike=strike - rng.uniform(0.001, 0.01)) if kind == "collar" else {}
        if model == "black" and extra and extra["floor_strike"] <= 0:
            extra["floor_strike"] = strike / 2
        cf = CapFloor.from_overnight(preset, rng.choice([1, -1]) * 4e6, kind, strike,
                                     rng.choice(["1Y", "2Y", "5Y", "10Y", "27M"]), vol, model, shift,
                                     effective_vol=bool(rng.randrange(2)), **terms, **extra)  # fmt: skip
        if not cf.caplets(market).shape[0]:
            continue
        _assert_matches_compounded(cf, market, ql_compounded(cf, market), f"case {i}: {preset} {asof} {cf}")


def test_lyashenko_mercurio_time():
    # before the period: t_s + (t_e - t_s) / 3; the vol decays linearly over the accrual period
    np.testing.assert_allclose(lyashenko_mercurio_time([1.0, 0.0], [1.25, 0.25]), [1.0 + 0.25 / 3, 0.25 / 3])
    inside = lyashenko_mercurio_time(-0.1, 0.15)  # 0.1 of a 0.25 period gone: (0.15^3) / (3 * 0.25^2)
    assert inside == pytest.approx(0.15**3 / (3 * 0.25**2), rel=1e-15)
    assert lyashenko_mercurio_time(-0.3, 0.0) == 0.0 and lyashenko_mercurio_time(-0.3, -0.05) == 0.0
    assert lyashenko_mercurio_time(0.5, 0.5) == 0.5  # one-day period: all variance before it
    days = np.linspace(-0.3, 0.75, 211)  # valuation moving through the period [0.5, 0.75] seen from 0
    times = lyashenko_mercurio_time(0.5 - days, 0.75 - days)
    assert np.all(np.diff(times) < 0) and times[-1] == 0.0  # falls continuously to zero at the period end
    at_start = lyashenko_mercurio_time(np.array([1e-12, -1e-12]), 0.25)  # continuous as the period starts
    assert at_start[0] == pytest.approx(at_start[1], abs=1e-11)
    with pytest.raises(ValueError, match="last fixing"):
        lyashenko_mercurio_time(1.0, 0.5)


def test_compounded_caplet_is_black_on_the_lyashenko_mercurio_time(market):
    """Each live caplet is the model price on the forward with variance
    vol^2 x T_LM (or vol^2 x t_e with an effective vol), so the compounded
    caplet is worth less than a caplet with the vol running to the last
    fixing, and its effective vol is vol x sqrt(T_LM / t_e)."""

    for model, vol, shift in MODELS:
        cf = CapFloor.from_overnight("USD SOFR", 1e7, "cap", 0.036, "5Y", vol, model, shift)
        table = cf.caplets(market)
        t_s = np.array([(d - market.asof).days / 365 for d in table["first_fix"]])
        t_e = np.array([(d - market.asof).days / 365 for d in table["fix"]])
        np.testing.assert_allclose(table["variance_time"], t_s + (t_e - t_s) / 3, rtol=1e-15)
        np.testing.assert_allclose(table["expiry"], t_e, rtol=1e-15)
        for row in table.itertuples():
            unit = model_price(model, row.forward, 0.036, row.variance_time, vol, shift)
            assert row.pv == pytest.approx(1e7 * row.tau * row.df * unit, rel=1e-13)
        np.testing.assert_allclose(table["effective_vol"], vol * np.sqrt(table["variance_time"] / t_e), rtol=1e-14)
        effective = replace(cf, effective_vol=True)
        assert effective.pv(market) > cf.pv(market)
        np.testing.assert_allclose(effective.caplets(market)["variance_time"], t_e, rtol=1e-15)
        # the same caplet with the vol set to the effective vol, as an effective-vol input, has the same value
        same = replace(effective, vol=dict(zip(t_e, table["effective_vol"])))
        np.testing.assert_allclose(same.caplets(market)["pv"], table["pv"], rtol=1e-12)


def test_compounded_rate_and_schedule_follow_the_ois_conventions(market):
    """A compounded caplet pays the OIS floating coupon's rate: its ATM strike
    on the OIS schedule is the OIS par rate. Daily fixings run from the
    period start to the business day before its end; payment is the
    preset's OIS payment lag later; the first period is kept (it is not
    known at trade)."""

    cf = CapFloor.from_overnight("USD SOFR", 1e7, "cap", 0.035, "5Y", 0.0095, index_months=12)
    swap = VanillaSwap.from_conventions(market.curve.conventions, "5Y", 0.0, kind="OIS", trade_date=market.asof)
    assert cf.atm_strike(market) == pytest.approx(swap.par_rate(market), abs=1e-14)
    quarterly = CapFloor.from_overnight("USD SOFR", 1e7, "cap", 0.035, "5Y", 0.0095)
    periods = quarterly.periods(market.asof)
    cal = Calendar("USD")
    assert len(periods) == 20 and periods[0].start == quarterly.effective_date(market.asof)
    for p in periods:
        assert p.first_fix == p.start and p.fix == cal.add_business_days(p.end, -1) and p.fix < p.end
        assert p.pay == cal.add_business_days(p.end, 2)
        assert p.tau == pytest.approx((p.end - p.start).days / 360, rel=1e-15)
    sonia = CapFloor.from_overnight("GBP SONIA", 1e7, "floor", 0.035, "2Y", 0.0095)
    assert (sonia.day_count, sonia.calendar, sonia.payment_lag, sonia.fixing_lag) == ("ACT/365F", "LONDON", 0, 0)
    for p in sonia.periods(market.asof):
        assert p.pay == p.end
    # forward rates compound on the discount curve
    row = quarterly.caplets(market).iloc[3]
    growth = market.curve.df(row["start"]) / market.curve.df(row["end"])
    assert row["forward"] == pytest.approx((growth - 1) / row["tau"], rel=1e-14)


def test_compounded_caplet_inside_its_period_uses_past_fixings(market):
    """Seasoned compounded cap: the running period mixes fixings (the
    fixings curve) and forecasts (today's curve); its variance shrinks with
    the part still to fix; one that has fixed but not paid is intrinsic."""

    start = market.asof - dt.timedelta(days=40)
    fixings = flat_curve(start - dt.timedelta(days=10), 0.05)
    seasoned = market.replace(fixings=fixings)
    cf = CapFloor.from_overnight("USD SOFR", 1e7, "cap", 0.034, "1Y", 0.0095, start=start)
    table = cf.caplets(seasoned)
    first = table.iloc[0]
    assert first["start"] < market.asof < first["fix"] and not first["fixed"]
    growth = fixings.df(first["start"]) / fixings.df(market.asof) / market.curve.df(first["end"])
    assert first["forward"] == pytest.approx((growth - 1) / first["tau"], rel=1e-14)
    t_s = (first["first_fix"] - market.asof).days / 365
    t_e = (first["fix"] - market.asof).days / 365
    assert first["variance_time"] == pytest.approx(t_e**3 / (3 * (t_e - t_s) ** 2), rel=1e-14)
    # moving the fixings moves only the running period
    other = cf.caplets(market.replace(fixings=flat_curve(start - dt.timedelta(days=10), 0.02)))
    assert other["forward"].iloc[0] < first["forward"]
    np.testing.assert_array_equal(other["pv"].iloc[1:], table["pv"].iloc[1:])
    # all fixings in, payment pending (SOFR pays two business days after the period end): intrinsic
    ended = CapFloor.from_overnight("USD SOFR", 1e7, "cap", 0.03, "2Y", 0.0095, start=dt.date(2026, 7, 16))
    row = ended.caplets(seasoned).iloc[0]
    assert row["end"] <= market.asof < row["pay"] and row["fixed"] and row["vega"] == 0.0
    assert row["pv"] == pytest.approx(1e7 * row["tau"] * row["df"] * max(row["forward"] - 0.03, 0.0), rel=1e-14)


@pytest.mark.parametrize("model,vol,shift", MODELS)
@pytest.mark.parametrize("rate_type", ["term", "compounded"])
def test_collar_is_long_cap_short_floor(market, model, vol, shift, rate_type):
    terms = dict(rate_type="compounded", payment_lag=2) if rate_type == "compounded" else {}
    collar = CapFloor(1e7, "collar", 0.040, "5Y", vol, model, shift, floor_strike=0.030, **terms)
    cap = CapFloor(1e7, "cap", 0.040, "5Y", vol, model, shift, **terms)
    floor = CapFloor(1e7, "floor", 0.030, "5Y", vol, model, shift, **terms)
    assert collar.pv(market) == pytest.approx(cap.pv(market) - floor.pv(market), rel=1e-13)
    assert collar.vega(market) == pytest.approx(cap.vega(market) - floor.vega(market), rel=1e-12)
    assert collar.decay(market) == pytest.approx(cap.decay(market) - floor.decay(market), rel=1e-11, abs=1e-9)
    table = collar.caplets(market)
    np.testing.assert_allclose(table["cap_pv"], cap.caplets(market)["pv"], rtol=1e-14)
    np.testing.assert_allclose(table["floor_pv"], -floor.caplets(market)["pv"], rtol=1e-14)
    np.testing.assert_allclose(table["pv"], table["cap_pv"] + table["floor_pv"], rtol=1e-14)
    assert (table["floor_strike"] == 0.030).all() and "floor_pv" not in cap.caplets(market)
    lender = replace(collar, notional=-1e7)  # long the floor, short the cap
    assert lender.pv(market) == pytest.approx(-collar.pv(market), rel=1e-14)
    field = "vol_shift_normal" if model == "bachelier" else "vol_shift_lognormal"
    h = 1e-3 * UNIT[model]
    bumped = (collar.pv(market.replace(**{field: h})) - collar.pv(market.replace(**{field: -h}))) / (2 * h)
    assert collar.vega(market) == pytest.approx(bumped * UNIT[model], rel=1e-6)
    end = max(p.pay for p in collar.periods(market.asof))
    assert collar.cash(market, market.asof, end) == pytest.approx(
        cap.cash(market, market.asof, end) - floor.cash(market, market.asof, end), abs=1e-6
    )
    with pytest.raises(ValueError, match="imply the cap's and the floor's vols separately"):
        collar.implied_flat_vol(collar.pv(market), market)
    assert "COLLAR 4.000%/3.000%" in collar.label


def test_collar_legs_read_their_own_strikes_on_a_smile(market):
    """On stripped caplet vols by strike each leg of a collar reads its own
    strike's vol."""

    from derivx.rates.capvol import strip_surface

    strikes = np.array([0.025, 0.035, 0.045])
    flat = np.array([[0.0110, 0.0095, 0.0090], [0.0112, 0.0098, 0.0093], [0.0108, 0.0096, 0.0092]])
    terms = dict(rate_type="compounded", payment_lag=2)
    surface, _ = strip_surface(market, ["2Y", "5Y", "10Y"], strikes, flat, terms=terms)
    collar = CapFloor(1e7, "collar", 0.045, "5Y", surface, floor_strike=0.025, **terms)
    table = collar.caplets(market)
    np.testing.assert_allclose(table["vol"], surface.caplet_vols(table["expiry"].to_numpy(), 0.045), rtol=1e-15)
    np.testing.assert_allclose(table["floor_vol"], surface.caplet_vols(table["expiry"].to_numpy(), 0.025), rtol=1e-15)
    assert not np.allclose(table["vol"], table["floor_vol"])
    cap = CapFloor(1e7, "cap", 0.045, "5Y", surface, **terms).pv(market)
    floor = CapFloor(1e7, "floor", 0.025, "5Y", surface, **terms).pv(market)
    assert collar.pv(market) == pytest.approx(cap - floor, rel=1e-13)


@pytest.mark.parametrize("model,vol,shift", MODELS)
def test_compounded_cap_greeks_parity_and_implied_vol(market, model, vol, shift):
    cf = CapFloor.from_overnight("USD SOFR", 1e7, "floor", 0.034, "7Y", {1.0: vol, 5.0: 0.9 * vol}, model, shift,
                                 forward_start="1Y")  # fmt: skip
    field = "vol_shift_normal" if model == "bachelier" else "vol_shift_lognormal"
    h = 1e-3 * UNIT[model]
    up, down = cf.pv(market.replace(**{field: h})), cf.pv(market.replace(**{field: -h}))
    assert cf.vega(market) == pytest.approx((up - down) / (2 * h) * UNIT[model], rel=1e-7)
    parity = cap_floor_parity(cf, market)
    assert parity["residual"] == pytest.approx(0.0, abs=1e-8 * 1e7)
    flat = replace(cf, vol=vol)
    assert flat.implied_flat_vol(flat.pv(market), market) == pytest.approx(vol, rel=1e-10)
    # time decay: the PV with every option time a business day shorter (Lyashenko-Mercurio times recomputed)
    days = (Calendar("USD").add_business_days(market.asof, 1) - market.asof).days
    table = cf.caplets(market)
    t_s = np.array([(d - market.asof).days / 365 for d in table["first_fix"]]) - days / 365
    t_e = table["expiry"].to_numpy() - days / 365
    later = sum(1e7 * r.tau * r.df * model_price(model, r.forward, 0.034, t, r.vol, shift, False)
                for r, t in zip(table.itertuples(), lyashenko_mercurio_time(t_s, t_e)))  # fmt: skip
    assert cf.decay(market) == pytest.approx(later - cf.pv(market), rel=1e-9)
    assert cf.decay(market) < 0


def test_compounded_caps_strip_and_reprice(market):
    """Flat vols of compounded caps strip into caplet vols on their
    Lyashenko-Mercurio times, and every quoted cap reprices."""

    from derivx.rates.capvol import strip_caplet_vols

    terms = dict(rate_type="compounded", payment_lag=2, fixing_lag=2)
    mats, flat = ["1Y", "2Y", "5Y", "10Y"], [0.0090, 0.0100, 0.0098, 0.0088]
    result = strip_caplet_vols(market, mats, flat, None, "constant", terms=terms)
    assert result.max_error < 1e-12
    for m, v in zip(mats, flat):
        probe = CapFloor(1.0, "cap", 0.0, m, v, **terms)
        atm = probe.atm_strike(market)
        quoted = CapFloor(1.0, "cap", atm, m, v, **terms).pv(market)
        assert CapFloor(1.0, "cap", atm, m, result.curve, **terms).pv(market) == pytest.approx(quoted, abs=1e-14)


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
        ({"kind": "straddle"}, "kind"),
        ({"kind": "collar"}, "collar needs floor_strike"),
        ({"kind": "collar", "floor_strike": 0.03}, "must be below its cap strike"),
        ({"floor_strike": 0.02}, "floor_strike is for collars"),
        ({"kind": "collar", "floor_strike": -0.001, "model": "black"}, "positive strike"),
        ({"rate_type": "averaged"}, "rate_type"),
        ({"rate_type": "compounded", "projection": "EURIBOR 3M"}, "overnight rate of the discount curve"),
        ({"effective_vol": True}, "effective_vol is for compounded caps"),
        ({"rate_type": "compounded", "effective_vol": "yes"}, "effective_vol must be True or False"),
        ({"payment_lag": -1}, "payment_lag"),
        ({"payment_lag": 0.5}, "payment_lag"),
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
# Theta and gamma (RR-9)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model,vol,shift", MODELS)
def test_decay_is_the_time_value_lost_to_the_next_business_day(market, model, vol, shift):
    cf = CapFloor(1e7, "cap", 0.0365, "5Y", vol, model, shift, forward_start="3M")
    table = cf.caplets(market)
    days = 1  # Monday 2026-10-19 -> Tuesday
    expected = 0.0
    for row in table.itertuples():
        scale = 1e7 * row.tau * row.df
        later = max(row.expiry - days / 365.0, 0.0)
        expected += scale * (model_price(model, row.forward, cf.strike, later, row.vol, shift)
                             - model_price(model, row.forward, cf.strike, row.expiry, row.vol, shift))  # fmt: skip
    assert cf.decay(market) == pytest.approx(expected, rel=1e-12)
    assert cf.decay(market) < 0 and replace(cf, notional=-1e7).decay(market) == pytest.approx(-expected, rel=1e-12)
    friday = Market(bootstrap(D("2026-10-23"), [CurveQuote(r["kind"], r["tenor"], r["rate"])
                                                for r in sofr_example_rows()], "USD SOFR"))  # fmt: skip
    assert cf.decay(friday) == pytest.approx(cf.decay(friday, days=3), rel=1e-15)  # Friday -> Monday
    assert cf.decay(market, days=0) == 0.0
    with pytest.raises(ValueError, match="0 or more"):
        cf.decay(market, days=-1)


def test_a_caplet_fixing_tomorrow_decays_to_its_intrinsic_value(market):
    probe = CapFloor(1e7, "cap", 0.03, "6M", 0.01, start=dt.date(2026, 10, 22), exclude_first=False)
    cf = replace(probe, strike=float(probe.caplets(market)["forward"].iloc[0]))  # first caplet at the money
    table = cf.caplets(market)
    first = table.iloc[0]
    assert first["fix"] == dt.date(2026, 10, 20) and first["expiry"] == pytest.approx(1 / 365)
    lost = -first["pv"]  # at the money: intrinsic value zero, so its whole value goes
    assert lost < 0
    rest = cf.decay(market) - lost
    assert rest < 0 and abs(rest) < abs(lost)  # the second caplet, 3M longer, loses less in one day


def test_full_theta_and_curve_gamma_of_caps_and_floors(market):
    from derivx.risk.scenarios import carry_market, theta
    from derivx.risk.sensitivities import rate_pv01_gamma

    cap = CapFloor(1e7, "cap", 0.036, "5Y", 0.0095)
    floor = replace(cap, kind="floor")
    positions = [capfloor_position(cap, market.asof), capfloor_position(floor, market.asof)]
    full = theta(positions, market)
    rolled = carry_market(market, 1)
    for value, position in zip(full, positions):
        assert value == pytest.approx(position.pv(rolled) + position.cash(rolled, market.asof, rolled.asof)
                                      - position.pv(market), rel=1e-12)  # fmt: skip
    assert full[0] < 0 and full[1] < 0  # long options lose time value
    (pv01_c, gamma_c), (pv01_f, gamma_f) = (rate_pv01_gamma(p.pv, market) for p in positions)
    assert pv01_c > 0 > pv01_f and gamma_c > 0 and gamma_f > 0
    swap_pv01, swap_gamma = rate_pv01_gamma(cap.swap_pv, market)  # cap - floor = payer swap on the strip
    assert pv01_c - pv01_f == pytest.approx(swap_pv01, rel=1e-9)
    assert gamma_c - gamma_f == pytest.approx(swap_gamma, abs=0.5)


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
    assert [m.label for m in at.metric][:2] == ["PV ($)", "PREMIUM (BP RUNNING)"]  # amounts name their currency
    assert not at.error
    pv = float(at.metric[0].value.replace(",", ""))
    assert pv == pytest.approx(positions[0].pv(example_market()), abs=0.01)
    assert len(at.get("plotly_chart")) == 3  # strip, vega, ladder
    downloads = [b.proto.id.rsplit("-", 1)[-1] for b in at.get("download_button")]
    assert downloads == ["cap_caplets_csv", "cap_ladder_csv", "cap_export_zip", "cap_export_manifest"]
    assert not at.warning
    labels = [m.label for m in at.metric]
    assert {"GAMMA ($/BP²)", "THETA ($, 1 BUS. DAY)", "VEGA ($/1BP)", "PV01 ($/BP)"} <= set(labels)
    theta = float(next(m.value for m in at.metric if m.label == "THETA ($, 1 BUS. DAY)").replace(",", ""))
    assert theta < 0  # a long cap loses time value


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
    "cap_strip_layout": "radio",
    "cap_strip_method": "radio",
    "cap_smile_expiry": "selectbox",
    "cap_reset": "selectbox",
    "cap_rfr_vol": "radio",
    "cap_floor_strike_pct": "number_input",
    "cap_floor_offset_bp": "number_input",
    "cap_collar_offset_bp": "number_input",
}


def _set(at: AppTest, key: str, value) -> None:
    getattr(at, WIDGETS[key])(key=key).set_value(value).run()
    _assert_tab_contract(at)


@pytest.mark.parametrize(
    "changes",
    [
        {"cap_kind": "Floor", "cap_index": "6M term", "cap_tenor": "10Y", "cap_forward_start": "2Y"},
        {"cap_kind": "Collar", "cap_reset": "1M", "cap_rfr_vol": "Effective to last fixing", "cap_tenor": "2Y"},
        {
            "cap_kind": "Collar",
            "cap_strike_mode": "Absolute %",
            "cap_strike_pct": 4.5,
            "cap_floor_strike_pct": 3.0,
            "cap_model": "Black (%)",
        },
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


@pytest.mark.parametrize(
    "changes",
    [
        {"cap_vol_mode": "Stripped from cap quotes"},
        {"cap_vol_mode": "Stripped from cap quotes", "cap_strip_method": "Linear", "cap_tenor": "7Y"},
        {"cap_index": "3M term", "cap_vol_mode": "Stripped from cap quotes", "cap_strip_layout": "By strike",
         "cap_smile_expiry": "10Y (9.75Y)"},
        {"cap_vol_mode": "Stripped from cap quotes", "cap_strip_layout": "By strike",
         "cap_smile_expiry": "10Y (10.01Y)"},
        {"cap_kind": "Collar", "cap_vol_mode": "Stripped from cap quotes", "cap_strip_layout": "By strike"},
        {"cap_vol_mode": "Stripped from cap quotes", "cap_strip_layout": "By strike", "cap_strike_mode": "Absolute %",
         "cap_strike_pct": 4.25, "cap_kind": "Floor", "cap_forward_start": "2Y"},
    ],
)  # fmt: skip
def test_tab_strips_caplet_vols_from_cap_quotes(changes):
    at = _app()
    for key, value in changes.items():
        _set(at, key, value)
    assert not at.error, [e.value for e in at.error]
    charts = [c.proto.id for c in at.get("plotly_chart")]
    assert any(i.endswith("cap_vol_strip_chart") for i in charts)
    assert any(i.endswith("cap_smile_chart") for i in charts) == (changes.get("cap_strip_layout") == "By strike")
    repricing = next(d.value for d in at.dataframe if "ERROR (BP UPFRONT)" in d.value.columns)
    assert repricing["ERROR (BP UPFRONT)"].abs().max() < 1e-9
    position = at.session_state["cap_test_positions"][0]
    assert position.details["trade"].stripped_vol
    strip = json.loads(next(c for c in at.get("plotly_chart") if c.proto.id.endswith("cap_vol_strip_chart")).proto.spec)
    badge = next(a["text"] for a in strip["layout"]["annotations"] if a["text"].startswith("SOURCE:"))
    assert "ILLUSTRATIVE" in badge


def test_stripped_vols_need_the_normal_model_and_show_quote_vegas():
    at = _app()
    _set(at, "cap_vol_mode", "Stripped from cap quotes")
    at.checkbox(key="cap_quote_vega").check().run()
    _assert_tab_contract(at)
    vegas = next(d.value for d in at.dataframe if "QUOTED CAP" in d.value.columns)
    total = float(next(m.value for m in at.metric if m.label == "VEGA ($/1BP)").replace(",", ""))
    five = vegas.loc[vegas["QUOTED CAP"] == "5Y", "VEGA (PER +1BP FLAT VOL)"].iloc[0]
    assert five == pytest.approx(total, rel=0.01)  # the 5Y ATM cap is the 5Y quote: its vega sits on that quote
    _set(at, "cap_model", "Black (%)")
    assert at.error and "pick the Normal (bp) model" in at.error[0].value
    assert at.session_state["cap_test_positions"] == []


@pytest.mark.parametrize("index", ["SOFR compounded", "3M term"])
def test_a_forward_start_trade_strips_the_spot_start_cap_quotes(index):
    """Cap vols are quoted on spot-start caps: picking a forward start for
    the trade used to strip the quotes as forward-start caps too (the 1Y
    quote's caplets fixing 2.25Y-2.75Y ahead), so the trade was priced on
    wrong caplet vols. The quotes are caps on the trade's rate: SOFR
    compounded in arrears (every caplet, the last fixing a business day
    before the 1Y end) or the 3M term rate (first caplet excluded, the last
    fixing at 0.75Y)."""

    from derivx.rates.capvol import strip_caplet_vols
    from derivx.rates.vol_examples import illustrative_cap_atm

    at = _app()
    _set(at, "cap_index", index)
    _set(at, "cap_vol_mode", "Stripped from cap quotes")
    spot_curve = at.session_state["cap_test_positions"][0].details["trade"].vol
    _set(at, "cap_forward_start", "2Y")
    assert not at.error, [e.value for e in at.error]
    position = at.session_state["cap_test_positions"][0]
    trade = position.details["trade"]
    assert position.details["forward_start"] == "2Y" and trade.vol == spot_curve
    assert (trade.start - EXAMPLE_ASOF).days > 700
    repricing = next(d.value for d in at.dataframe if "ERROR (BP UPFRONT)" in d.value.columns)
    first = repricing.iloc[0]
    if index == "3M term":
        assert first["LAST CAPLET EXPIRY (Y)"] < 1.0 and first["CAPLETS"] == 3  # the 1Y quote: a spot-start 1Y cap
    else:
        assert 1.0 < first["LAST CAPLET EXPIRY (Y)"] < 1.01 and first["CAPLETS"] == 4
    quotes = illustrative_cap_atm("USD")
    terms = {"index_months": 3, "day_count": "ACT/360", "calendar": "USD", "fixing_lag": 2}
    if index == "SOFR compounded":
        terms.update(rate_type="compounded", payment_lag=2, effective_vol=False)
    expected = strip_caplet_vols(example_market(), list(quotes["maturity"]), quotes["vol_bp"].to_numpy() * 1e-4,
                                 terms=terms).curve  # fmt: skip
    assert trade.vol.nodes == expected.nodes
    np.testing.assert_allclose(trade.vol.vols, expected.vols, rtol=1e-13)
    at.checkbox(key="cap_quote_vega").check().run()
    _assert_tab_contract(at)
    vegas = next(d.value for d in at.dataframe if "QUOTED CAP" in d.value.columns)
    by_quote = vegas.set_index("QUOTED CAP")["VEGA (PER +1BP FLAT VOL)"]
    # The 2Y x 5Y cap's caplets fix by 7Y (its last one within days of the 7Y quote's): quotes from 12Y on
    # cannot move it, and +1bp on every quote moves every caplet vol by about 1bp.
    assert by_quote["7Y"] > 0.0 and by_quote["12Y":].abs().max() < 1e-6
    total = float(next(m.value for m in at.metric if m.label == "VEGA ($/1BP)").replace(",", ""))
    assert by_quote.sum() == pytest.approx(total, rel=0.02)


# The patch is undone after every run: AppTest runs scripts in the test process, so a module attribute left
# patched leaks into every later test (final review: under numpy 2 the quote was written as np.float64(...),
# a NameError here that then broke the BOOK and multi-curve UI tests that ran after this file).
EDGE_APP = """
import streamlit as st
import derivx.ui.rates_capfloor as cap
from derivx.rates.examples import example_market

_base = cap.illustrative_cap_atm


def _edge(currency):
    table = _base(currency)
    table.loc[table.index[-1], "vol_bp"] = {vol_bp:.4f}
    return table


cap.illustrative_cap_atm = _edge
try:
    st.session_state["cap_test_positions"] = cap.render(example_market())
finally:
    cap.illustrative_cap_atm = _base
"""


def test_quote_vegas_at_the_stripping_limit_do_not_break_the_tab():
    """Quotes the strip can just meet (a 30Y flat vol 0.4bp above the lowest
    non-negative caplet vols allow): the -1bp bump of that quote cannot be
    stripped, which used to raise a traceback below the metrics."""

    from derivx.rates.capvol import strip_caplet_vols
    from derivx.rates.vol_examples import illustrative_cap_atm

    quotes = illustrative_cap_atm("USD")
    mats, flat = list(quotes["maturity"]), quotes["vol_bp"].to_numpy() * 1e-4
    terms = {"index_months": 3, "day_count": "ACT/360", "calendar": "USD", "fixing_lag": 2}

    def strips(vol: float) -> bool:
        try:
            strip_caplet_vols(example_market(), mats, np.r_[flat[:-1], vol], terms=terms)
        except ValueError:
            return False
        return True

    lo, hi = 1e-4, flat[-1]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if strips(mid) else (mid, hi)
    at = AppTest.from_string(EDGE_APP.format(vol_bp=float((hi + 0.4e-4) * 1e4)), default_timeout=120)
    at.run()
    _set(at, "cap_vol_mode", "Stripped from cap quotes")
    _set(at, "cap_tenor", "30Y")
    at.checkbox(key="cap_quote_vega").check().run()
    _assert_tab_contract(at)
    assert not at.error, [e.value for e in at.error]
    vegas = next(d.value for d in at.dataframe if "QUOTED CAP" in d.value.columns)
    assert vegas.set_index("QUOTED CAP").loc["30Y", "DIFFERENCE"] == "up only"
    assert any("One-sided at" in c.value and "30Y" in c.value for c in at.caption)
    import derivx.ui.rates_capfloor as cap_ui

    assert cap_ui.illustrative_cap_atm is illustrative_cap_atm  # the script undid its patch: nothing leaks


def test_tab_reports_black_with_a_negative_strike_as_an_error():
    at = _app()
    _set(at, "cap_model", "Black (%)")
    _set(at, "cap_atm_offset_bp", -500.0)
    assert at.error and "positive strike" in at.error[0].value
    assert at.session_state["cap_test_positions"] == []


def test_tab_prices_sofr_compounded_caps_by_default_and_flags_term_caps():
    """USD caps trade on SOFR compounded in arrears: that is the tab's
    default, with the Lyashenko-Mercurio option time and effective vol per
    caplet. A 3M/6M term cap on the SOFR curve is still available, marked as
    the approximation it is."""

    at = _app()
    trade = at.session_state["cap_test_positions"][0].details["trade"]
    assert trade.compounded and not trade.effective_vol and (trade.payment_lag, trade.index_months) == (2, 3)
    caplets = next(d.value for d in at.dataframe if "OPTION TIME (Y)" in d.value.columns)
    assert {"FIRST FIXING", "LAST FIXING", "EFF VOL (bp)"} <= set(caplets.columns)
    assert (caplets["OPTION TIME (Y)"] < caplets["EXPIRY (Y)"]).all()
    assert (caplets["EFF VOL (bp)"] < caplets["VOL (bp)"]).all()
    schedule = next(c.value for c in at.caption if c.value.startswith("Schedule:"))
    assert "SOFR compounded in arrears" in schedule and "Lyashenko-Mercurio" in schedule
    assert not at.warning
    _set(at, "cap_rfr_vol", "Effective to last fixing")
    effective = at.session_state["cap_test_positions"][0].details["trade"]
    assert effective.effective_vol and effective.pv(example_market()) > trade.pv(example_market())
    _set(at, "cap_index", "3M term")
    term = at.session_state["cap_test_positions"][0].details["trade"]
    assert not term.compounded and term.projection is None
    assert any("Term-rate approximation" in w.value and "SOFR" in w.value for w in at.warning)


def test_tab_prices_a_collar_as_long_cap_short_floor():
    at = _app()
    _set(at, "cap_kind", "Collar")
    assert not at.error, [e.value for e in at.error]
    collar = at.session_state["cap_test_positions"][0].details["trade"]
    market = example_market()
    atm = replace(collar, kind="cap", floor_strike=None, strike=0.0).atm_strike(market)
    assert collar.strike == pytest.approx(atm + 0.005, abs=1e-15)  # ATM +/- 50bp by default
    assert collar.floor_strike == pytest.approx(atm - 0.005, abs=1e-15)
    cap = replace(collar, kind="cap", floor_strike=None).pv(market)
    floor = replace(collar, kind="floor", strike=collar.floor_strike, floor_strike=None).pv(market)
    pv = float(at.metric[0].value.replace(",", ""))
    assert pv == pytest.approx(cap - floor, abs=0.01)
    caplets = next(d.value for d in at.dataframe if "FLOOR STRIKE (%)" in d.value.columns)
    np.testing.assert_allclose(caplets["PV ($)"], caplets["CAP PV ($)"] + caplets["FLOOR PV ($)"], rtol=1e-12)
    assert (caplets["FLOOR PV ($)"] <= 0).all()
    strip = json.loads(next(c for c in at.get("plotly_chart") if c.proto.id.endswith("cap_strip_chart")).proto.spec)
    assert {"CAP STRIKE", "FLOOR STRIKE", "FORWARD"} <= {t["name"] for t in strip["data"]}
    _set(at, "cap_floor_offset_bp", 75.0)  # floor above the cap: refused with the reason
    assert at.error and "must be below its cap strike" in at.error[0].value


def test_stripped_vols_carry_the_caplets_they_were_stripped_on(market):
    """Flat cap quotes imply different caplet vols on term caplets, on
    compounded caplets with a Lyashenko-Mercurio vol and with an effective
    vol: stripped vols record which, and a cap of another basis is refused
    instead of being priced on vols that mean something else."""

    from derivx.rates.capvol import CapletVolSurface, strip_caplet_vols

    mats, flat = ["1Y", "2Y", "5Y"], [0.0090, 0.0100, 0.0098]
    bases = {}
    for name, terms in {
        "term": {},
        "lyashenko_mercurio": dict(rate_type="compounded", payment_lag=2),
        "effective": dict(rate_type="compounded", payment_lag=2, effective_vol=True),
    }.items():
        curve = strip_caplet_vols(market, mats, flat, None, "constant", terms=terms).curve
        assert curve.vol_basis == name
        bases[name] = (curve, terms)
    assert bases["term"][0].vols != bases["lyashenko_mercurio"][0].vols != bases["effective"][0].vols
    for name, (curve, terms) in bases.items():
        CapFloor(1e7, "cap", 0.035, "5Y", curve, **terms)  # its own basis prices
        for other, (_, other_terms) in bases.items():
            if other != name:
                with pytest.raises(ValueError, match="stripped on"):
                    CapFloor(1e7, "cap", 0.035, "5Y", curve, **other_terms)
    lm, effective = bases["lyashenko_mercurio"][0], bases["effective"][0]
    with pytest.raises(ValueError, match="vol basis"):
        CapletVolSurface((replace(lm, strike=0.03), replace(effective, strike=0.04)))
