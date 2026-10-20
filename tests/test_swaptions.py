"""European swaptions: QuantLib golden values (Black, shifted Black and
Bachelier on single- and dual-curve markets), Greeks, implied vols, expiry
handling and the scenario engine."""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from derivx.rates.capfloor import VOL_UNIT, model_price
from derivx.rates.curves import CurveQuote, Market, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, example_market, sofr_example_rows
from derivx.rates.indices import HullWhite
from derivx.rates.multicurve import bootstrap_projection
from derivx.rates.swaptions import Swaption, atm_straddle_approx_bp, forward_and_annuity, swaption_position
from derivx.risk.scenarios import carry_market, parallel, run_scenario, theta, vegas
from derivx.risk.sensitivities import rate_pv01_gamma

D = dt.date.fromisoformat
NOTIONAL = 10_000_000.0


def _eur_ois(asof: dt.date, shift: float):
    quotes = [CurveQuote(r["kind"], r["tenor"], r["rate"] + shift) for r in sofr_example_rows()]
    return bootstrap(asof, quotes, "EUR ESTR")


def _build(asof: dt.date, quotes: dict) -> Market:
    """Mirror of tools/golden/multicurve.py:build."""

    ois = _eur_ois(asof, quotes["ois_shift"])
    c6 = bootstrap_projection(asof, [CurveQuote(*q) for q in quotes["6M"]], "EURIBOR 6M", ois)
    c3 = bootstrap_projection(asof, [CurveQuote(*q) for q in quotes["3M"]], "EURIBOR 3M", ois, basis=c6,
                              convexity=HullWhite(0.03, 0.0085))  # fmt: skip
    return Market(curve=ois, projections={"EURIBOR 6M": c6, "EURIBOR 3M": c3})


@lru_cache(maxsize=None)
def _markets() -> dict:
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).parent / "golden" / "swaptions.json").read_text())["data"]
    sets = data["sets"]
    return {
        "usd": example_market(),
        "eur": _build(D(sets["base"]["asof"]), sets["base"]["quotes"]),
        "eur_negative": _build(D(sets["negative"]["asof"]), sets["negative"]["quotes"]),
    }


@pytest.fixture(scope="module")
def usd() -> Market:
    return _markets()["usd"]


def _swaption(case: dict, notional: float = NOTIONAL) -> Swaption:
    return Swaption(notional, case["expiry"], case["tenor"], case["strike"], case["kind"], case["vol"], case["model"],
                    case["shift"], index=case["index"])  # fmt: skip


# --------------------------------------------------------------------------
# QuantLib reference values
# --------------------------------------------------------------------------


def test_swaptions_match_quantlib(golden):
    """Payer and receiver swaptions, ITM/ATM/OTM, 1M to 10Y expiries on 1Y to
    30Y SOFR OIS and EURIBOR swaps (dual curve, negative rates), against
    QuantLib's Black, shifted Black and Bachelier swaption engines: dates,
    NPV, annuity, forward, delta, vega, implied vols and time decay."""

    data = golden("swaptions")
    assert len(data["cases"]) == 234
    for case in data["cases"]:
        market = _markets()[case["market"]]
        sw = _swaption(case)
        name = f"{case['market']} {sw.label} {case['model']}"
        inputs = sw.inputs(market)
        swap = sw.underlying(market)
        assert inputs.expiry_date == D(case["expiry_date"]), name
        assert inputs.start == D(case["start"]), name
        assert inputs.maturity == D(case["maturity"]), name
        fixed_pay = [p.pay for p in swap.periods(market.asof) if p.leg == "fixed"]
        assert [d.isoformat() for d in fixed_pay] == case["fixed_pay"], name
        assert inputs.forward == pytest.approx(case["forward"], abs=1e-15), name
        assert inputs.annuity * NOTIONAL == pytest.approx(case["annuity"], rel=1e-13), name
        g = sw.greeks(market)
        assert g["pv"] == pytest.approx(case["npv"], rel=1e-11, abs=1e-7), name
        assert sw.pv(market) == pytest.approx(case["npv"], rel=1e-11, abs=1e-7), name
        assert g["delta_fwd"] / 1e-4 == pytest.approx(case["delta"], rel=1e-10, abs=1e-6), name
        assert g["vega"] / VOL_UNIT[case["model"]] == pytest.approx(case["vega"], rel=1e-10, abs=1e-6), name
        assert g["decay"] == pytest.approx(case["decay"], rel=1e-9, abs=1e-7), name
        assert g["decay_days"] == case["decay_days"]
        # Implied vols: they reprice the premium to rounding and agree with QuantLib's solver (which works on
        # the full in-the-money price, so it loses digits where the time value is small: ~1e-9 relative).
        implied = sw.implied_vol(case["implied"]["price"], market)
        assert implied == pytest.approx(case["implied"]["vol"], rel=1e-7), name
        assert replace(sw, vol=implied).pv(market) == pytest.approx(case["implied"]["price"], rel=1e-12), name
        other = case["implied_other"]
        if other is not None:
            vol = sw.implied_vol(case["npv"], market, model=other["model"], shift=other["shift"])
            assert vol == pytest.approx(other["vol"], rel=1e-7), name
            again = replace(sw, model=other["model"], shift=other["shift"], vol=vol)
            assert again.pv(market) == pytest.approx(case["npv"], rel=1e-12), name


def test_straddle_is_payer_plus_receiver_and_sold_is_the_negative(golden):
    data = golden("swaptions")
    by_key = {}
    for case in data["cases"]:
        key = (case["market"], case["expiry"], case["tenor"], case["strike"], case["model"])
        by_key.setdefault(key, {})[case["kind"]] = case
    checked = 0
    for (name, *_), pair in by_key.items():
        if set(pair) != {"payer", "receiver"}:
            continue
        market = _markets()[name]
        straddle = _swaption({**pair["payer"], "kind": "straddle"})
        g = straddle.greeks(market)
        assert g["pv"] == pytest.approx(pair["payer"]["npv"] + pair["receiver"]["npv"], rel=1e-11)
        assert g["delta_fwd"] / 1e-4 == pytest.approx(pair["payer"]["delta"] + pair["receiver"]["delta"], abs=1e-4)
        assert g["vega"] / VOL_UNIT[straddle.model] == pytest.approx(2 * pair["payer"]["vega"], rel=1e-10)
        sold = _swaption(pair["payer"], notional=-NOTIONAL)
        gs = sold.greeks(market)
        gp = _swaption(pair["payer"]).greeks(market)
        for key in ("pv", "premium_bp", "delta_fwd", "delta_notional", "gamma_fwd", "vega", "vega_normal_bp",
                    "decay"):  # fmt: skip
            assert gs[key] == pytest.approx(-gp[key], rel=1e-14, abs=1e-12), key
        checked += 1
    assert checked >= 100


@pytest.mark.parametrize("market_name", ["usd", "eur", "eur_negative"])
def test_payer_minus_receiver_is_the_forward_swap_priced_by_the_swap_engine(market_name):
    """Parity on R2's swap engine: payer - receiver = value of paying the
    strike on the underlying forward swap (annuity x (forward - strike))."""

    market = _markets()[market_name]
    index = None if market_name == "usd" else "EURIBOR 6M"
    for expiry, tenor, offset in (("1Y", "5Y", 0.0), ("3M", "10Y", 40.0), ("5Y", "20Y", -60.0)):
        atm = Swaption(NOTIONAL, expiry, tenor, 0.0, index=index).atm_strike(market)
        for model, vol, shift in (("bachelier", 0.008, 0.0), ("shifted_black", 0.2, 0.04)):
            K = atm + offset * 1e-4
            payer = Swaption(NOTIONAL, expiry, tenor, K, "payer", vol, model, shift, index=index)
            receiver = replace(payer, kind="receiver")
            swap = payer.underlying(market)  # the payer swap at the strike
            assert not swap.receive_fixed and swap.fixed_rate == K
            parity = payer.pv(market) - receiver.pv(market)
            assert parity == pytest.approx(swap.pv(market), rel=1e-10, abs=1e-6)
            assert forward_and_annuity(swap, market)[0] == pytest.approx(swap.par_rate(market), abs=1e-15)


# --------------------------------------------------------------------------
# Greeks
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,vol,shift,kind",
    [("bachelier", 0.0095, 0.0, "payer"), ("black", 0.27, 0.0, "receiver"), ("shifted_black", 0.18, 0.02, "straddle")],
)
def test_analytic_greeks_match_finite_differences(usd, model, vol, shift, kind):
    sw = Swaption(-NOTIONAL, "2Y", "10Y", 0.03, kind, vol, model, shift)
    sw = replace(sw, strike=sw.atm_strike(usd) + 0.0035)
    inputs = sw.inputs(usd)
    g = sw.greeks(usd)
    h = 1e-6  # in rate units
    up = sw.value(replace(inputs, forward=inputs.forward + h), vol)
    down = sw.value(replace(inputs, forward=inputs.forward - h), vol)
    mid = sw.value(inputs, vol)
    assert g["delta_fwd"] == pytest.approx((up - down) / (2 * h) * 1e-4, rel=1e-6)
    assert g["gamma_fwd"] == pytest.approx((up + down - 2 * mid) / h**2 * 1e-8, rel=1e-4)
    dv = 1e-7
    vega = (sw.value(inputs, vol + dv) - sw.value(inputs, vol - dv)) / (2 * dv) * VOL_UNIT[model]
    assert g["vega"] == pytest.approx(vega, rel=1e-6)
    assert g["delta_notional"] == pytest.approx(g["delta_fwd"] / (inputs.annuity * 1e-4), rel=1e-12)
    # Equivalent vols: the same price in the other convention, and its vega there.
    eq = sw.equivalent_vols(inputs, vol)
    bach = Swaption(-NOTIONAL, "2Y", "10Y", sw.strike, kind, eq["normal"], "bachelier")
    assert bach.pv(usd) == pytest.approx(g["pv"], rel=1e-9)
    assert bach.greeks(usd)["vega"] == pytest.approx(g["vega_normal_bp"], rel=1e-9)
    lognormal = Swaption(-NOTIONAL, "2Y", "10Y", sw.strike, kind, eq["lognormal"], "shifted_black", shift)
    assert lognormal.pv(usd) == pytest.approx(g["pv"], rel=1e-9)
    assert lognormal.greeks(usd)["vega"] == pytest.approx(g["vega_lognormal_pt"], rel=1e-9)


def test_atm_straddle_matches_the_bachelier_closed_form(usd):
    sw = Swaption(NOTIONAL, "1Y", "10Y", 0.0, "straddle", 0.0090)
    sw = replace(sw, strike=sw.atm_strike(usd))
    inputs = sw.inputs(usd)
    unit_bp = sw.pv(usd) / (NOTIONAL * inputs.annuity) * 1e4
    assert unit_bp == pytest.approx(atm_straddle_approx_bp(0.0090, inputs.expiry_years), rel=1e-12)
    assert sw.greeks(usd)["delta_fwd"] == pytest.approx(0.0, abs=1e-6)  # an ATM normal straddle is delta neutral


def test_decay_is_negative_for_a_long_option_and_full_theta_adds_carry(usd):
    sw = Swaption(NOTIONAL, "6M", "5Y", 0.0, "receiver", 0.0100)
    sw = replace(sw, strike=sw.atm_strike(usd))
    g = sw.greeks(usd)
    assert g["decay"] < 0
    position = swaption_position(sw, usd)
    total = theta([position], usd)[0]
    rolled = carry_market(usd, 1)
    # Full theta = value on the rolled market - today; decay holds the forward and annuity of today.
    assert total == pytest.approx(position.pv(rolled) - position.pv(usd), rel=1e-12)
    assert rolled.asof == D("2026-10-20")
    assert abs(total - g["decay"]) < abs(g["decay"])  # the same order: carry/roll-down is the difference


def test_curve_pv01_and_gamma_signs(usd):
    payer = Swaption(NOTIONAL, "1Y", "10Y", 0.0, "payer", 0.0095)
    payer = replace(payer, strike=payer.atm_strike(usd))
    receiver = replace(payer, kind="receiver")
    pv01_p, gamma_p = rate_pv01_gamma(payer.pv, usd)
    pv01_r, gamma_r = rate_pv01_gamma(receiver.pv, usd)
    assert pv01_p > 0 > pv01_r and gamma_p > 0 and gamma_r > 0
    g = payer.greeks(usd)
    assert pv01_p == pytest.approx(g["delta_fwd"], rel=0.25)  # forward delta plus the annuity's own move
    # payer - receiver is the forward payer swap: its PV01 is the difference.
    swap = payer.underlying(usd)
    assert pv01_p - pv01_r == pytest.approx(rate_pv01_gamma(swap.pv, usd)[0], rel=1e-9)


# --------------------------------------------------------------------------
# Implied vols, expiry and validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["payer", "receiver", "straddle"])
@pytest.mark.parametrize(
    "model,vol,shift", [("bachelier", 0.0085, 0.0), ("black", 0.3, 0.0), ("shifted_black", 0.2, 0.03)]
)
def test_implied_vol_round_trips(usd, kind, model, vol, shift):
    for offset in (-150.0, 0.0, 80.0):
        sw = Swaption(NOTIONAL, "3Y", "5Y", 0.03, kind, vol, model, shift)
        sw = replace(sw, strike=sw.atm_strike(usd) + offset * 1e-4)
        assert sw.implied_vol(sw.pv(usd), usd) == pytest.approx(vol, rel=1e-9)
        sold = replace(sw, notional=-NOTIONAL)
        assert sold.implied_vol(sold.pv(usd), usd) == pytest.approx(vol, rel=1e-9)


def test_implied_vol_rejects_premiums_outside_the_bounds(usd):
    sw = Swaption(NOTIONAL, "1Y", "5Y", 0.0, "payer", 0.009)
    sw = replace(sw, strike=sw.atm_strike(usd) - 0.01)  # 100bp in the money
    intrinsic = NOTIONAL * sw.inputs(usd).annuity * 0.01
    with pytest.raises(ValueError, match="below the zero-vol"):
        sw.implied_vol(0.9 * intrinsic, usd)
    with pytest.raises(ValueError, match="infinite-vol"):
        replace(sw, model="black", vol=0.2).implied_vol(1e9, usd)
    with pytest.raises(ValueError, match="intrinsic"):
        replace(sw, kind="straddle").implied_vol(0.5 * intrinsic, usd)
    with pytest.raises(ValueError, match="finite"):
        sw.implied_vol(float("nan"), usd)


def test_expiry_day_is_intrinsic_and_after_expiry_raises(usd):
    sw = Swaption(NOTIONAL, "1Y", "5Y", 0.0, "payer", 0.009)
    sw = replace(sw, strike=sw.atm_strike(usd) - 0.002).pinned(usd)
    assert sw.expiry is None and sw.expiry_date == D("2027-10-19")
    on_expiry = carry_market(usd, (sw.expiry_date - usd.asof).days)
    assert on_expiry.asof == sw.expiry_date
    inputs = sw.inputs(on_expiry)
    assert inputs.expiry_years == 0.0
    assert sw.pv(on_expiry) == pytest.approx(NOTIONAL * inputs.annuity * (inputs.forward - sw.strike), rel=1e-12)
    assert sw.greeks(on_expiry)["vega"] == 0.0
    later = carry_market(usd, (sw.expiry_date - usd.asof).days + 1)
    with pytest.raises(ValueError, match="expired"):
        sw.pv(later)
    with pytest.raises(ValueError, match="no time value"):
        sw.implied_vol(sw.pv(on_expiry), on_expiry)


def test_pinned_swaption_keeps_its_dates_on_a_rolled_market(usd):
    sw = Swaption(NOTIONAL, "1Y", "10Y", 0.035, "payer", 0.009)
    pinned = sw.pinned(usd)
    rolled = carry_market(usd, 30)
    assert pinned.inputs(rolled).expiry_date == sw.inputs(usd).expiry_date
    assert sw.inputs(rolled).expiry_date > sw.inputs(usd).expiry_date  # an unpinned one re-dates
    assert pinned.inputs(rolled).expiry_years == pytest.approx(sw.inputs(usd).expiry_years - 30 / 365, abs=1e-12)


def test_market_vol_shifts_move_the_matching_model(usd):
    normal = Swaption(NOTIONAL, "1Y", "5Y", 0.036, "payer", 0.009)
    black = Swaption(NOTIONAL, "1Y", "5Y", 0.036, "payer", 0.25, "black")
    shifted = usd.replace(vol_shift_normal=0.0005, vol_shift_lognormal=0.02)
    assert normal.pv(shifted) == pytest.approx(replace(normal, vol=0.0095).pv(usd), rel=1e-14)
    assert black.pv(shifted) == pytest.approx(replace(black, vol=0.27).pv(usd), rel=1e-14)
    floored = usd.replace(vol_shift_normal=-0.02)
    assert normal.pv(floored) == pytest.approx(replace(normal, vol=0.0).pv(usd), rel=1e-14)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(kind="call"), "kind"),
        (dict(model="sabr"), "model"),
        (dict(notional=0.0), "notional"),
        (dict(strike=float("nan")), "strike"),
        (dict(vol=-0.01), "vol"),
        (dict(model="black", strike=-0.001, vol=0.2), "positive strike"),
        (dict(model="shifted_black", strike=-0.03, shift=0.02, vol=0.2), r"strike \+ shift"),
        (dict(expiry=None), "expiry"),
        (dict(expiry="2D"), "weeks, months or years"),
        (dict(tenor="10D"), "months or years"),
        (dict(index="LIBOR 3M"), "LIBOR"),
        (dict(index="EURIBOR 6M", conventions="EUR ESTR"), "not both"),
        (dict(conventions="JPY TONA"), "conventions"),
    ],
)
def test_bad_inputs_raise_clear_errors(kwargs, match):
    base = dict(notional=NOTIONAL, expiry="1Y", tenor="5Y", strike=0.03, kind="payer", vol=0.009)
    with pytest.raises(ValueError, match=match):
        Swaption(**{**base, **kwargs})


def test_black_on_a_negative_forward_points_to_other_models():
    market = _markets()["eur_negative"]
    sw = Swaption(NOTIONAL, "1Y", "5Y", 0.001, "payer", 0.2, "black", index="EURIBOR 6M")
    assert sw.inputs(market).forward < 0
    with pytest.raises(ValueError, match="normal .Bachelier. or shifted Black"):
        sw.pv(market)


def test_a_projection_index_is_required_on_the_market(usd):
    sw = Swaption(NOTIONAL, "1Y", "5Y", 0.03, index="EURIBOR 6M")
    with pytest.raises(ValueError, match="no EURIBOR 6M projection curve"):
        sw.pv(usd)


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


def test_swaption_positions_flow_through_the_scenario_engine(usd):
    payer = Swaption(NOTIONAL, "1Y", "10Y", 0.0, "payer", 0.0095)
    payer = replace(payer, strike=payer.atm_strike(usd))
    black = Swaption(-NOTIONAL, "2Y", "5Y", 0.03, "receiver", 0.26, "black")
    black = replace(black, strike=black.atm_strike(usd))
    positions = [swaption_position(payer, usd), swaption_position(black, usd)]
    assert [p.vol_model for p in positions] == ["normal", "lognormal"]
    assert positions[0].kind == "swaption" and positions[0].details["vol"] == payer.vol
    v = vegas(positions, usd)
    assert v[0] == pytest.approx(payer.greeks(usd)["vega"], rel=1e-6)  # central difference of +/-1bp
    assert v[1] == pytest.approx(black.greeks(usd)["vega"], rel=1e-4)  # +/-1 vol point: vomma shows at 1e-5
    shock = replace(parallel(10.0), normal_vol_bp=5.0, lognormal_vol_pts=-1.0, horizon_days=7)
    rows = run_scenario(positions, usd, shock)
    for row, p in zip(rows, positions):
        assert row["kind"] == "swaption"
        assert abs(row["residual"]) < 0.02 * abs(row["pnl"]) + 1.0, row
        assert row["vol"] != 0.0 and row["carry"] != 0.0


def test_random_swaptions_match_live_quantlib():
    """Random expiries, tenors, strikes, sides and models against QuantLib
    built on the fly (skipped without QuantLib)."""

    ql = pytest.importorskip("QuantLib")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from golden.common import ql_curve, set_evaluation_date
    from golden.swaptions import _engine, _usd_swaption, _vol_handle

    market = example_market()
    set_evaluation_date(EXAMPLE_ASOF)
    disc = ql_curve(market.curve)
    rng = random.Random(20261020)
    for _ in range(25):
        expiry = rng.choice(["2W", "1M", "2M", "9M", "18M", "4Y", "7Y", "15Y"])
        tenor = rng.choice(["1Y", "3Y", "7Y", "12Y", "25Y"])
        kind = rng.choice(["payer", "receiver"])
        model, vol, shift = rng.choice([("bachelier", 0.011, 0.0), ("black", 0.31, 0.0), ("shifted_black", 0.2, 0.01)])
        atm = Swaption(1.0, expiry, tenor, 0.0).atm_strike(market)
        strike = round(atm + rng.uniform(-0.012, 0.012), 6)
        sw = Swaption(NOTIONAL, expiry, tenor, strike, kind, vol, model, shift)
        q = _usd_swaption(EXAMPLE_ASOF, expiry, tenor, strike, kind == "payer", disc)
        q.setPricingEngine(_engine(disc, _vol_handle(EXAMPLE_ASOF, ql.UnitedStates(ql.UnitedStates.SOFR), model, vol,
                                                     shift), model))  # fmt: skip
        assert sw.pv(market) == pytest.approx(q.NPV(), rel=1e-10, abs=1e-6), (expiry, tenor, kind, model)
        assert sw.inputs(market).forward == pytest.approx(q.atmForward(), abs=1e-15)
    set_evaluation_date(EXAMPLE_ASOF)


def test_unit_value_uses_the_cap_kernels():
    sw = Swaption(1.0, "1Y", "1Y", 0.03, "straddle", 0.2, "shifted_black", 0.01)
    from derivx.rates.swaptions import SwaptionInputs

    inputs = SwaptionInputs(EXAMPLE_ASOF, EXAMPLE_ASOF, EXAMPLE_ASOF, 1.5, 0.031, 0.95)
    expected = model_price("shifted_black", 0.031, 0.03, 1.5, 0.2, 0.01, True) + model_price(
        "shifted_black", 0.031, 0.03, 1.5, 0.2, 0.01, False
    )
    assert sw.unit_value(inputs, 0.2) == pytest.approx(expected, rel=1e-15)
    assert sw.value(inputs, 0.2) == pytest.approx(0.95 * expected, rel=1e-15)
    assert math.isfinite(np.float64(sw.unit_value(inputs, 0.0)))
