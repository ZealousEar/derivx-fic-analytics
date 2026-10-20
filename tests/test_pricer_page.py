"""EQUITY / FX pricer page (AppTest): presets, input validation, expiry,
the PUT surface, determinism, widget state across mode switches, the
implied-vol solver, the smile sources and the hero card."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _vol_chains import synth_chain
from streamlit.testing.v1 import AppTest

from derivx.core.bs import bs_greeks, bs_price

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"
INPUTS = ("spot", "strike", "T", "r", "q", "vol")


def _app(mode: str = "[ EQUITY ]") -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if mode != "[ EQUITY ]":
        at.sidebar.radio[0].set_value(mode).run()
    assert not at.exception
    return at


def _inputs(at: AppTest, prefix: str = "eq") -> dict:
    return {name: at.number_input(key=f"{prefix}_{name}").value for name in INPUTS}


def _hero(at: AppTest) -> str:
    return next(m.value for m in at.markdown if "dx-hero-value" in m.value and "<style>" not in m.value)


def _premium(at: AppTest) -> float:
    text = re.search(r"dx-hero-value'>([^<]+)<", _hero(at)).group(1)
    return float(text.replace(",", ""))


def _arr(value) -> np.ndarray:
    """Plotly JSON array: a list or a typed array {'dtype', 'bdata', 'shape'}."""

    if isinstance(value, dict) and "bdata" in value:
        out = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        shape = value.get("shape")
        return out.reshape([int(n) for n in str(shape).split(",")]) if shape else out
    return np.asarray(value, dtype=float)


def _frame(at: AppTest, column: str) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if column in d.value.columns)


def _charts(at: AppTest) -> dict:
    out = {}
    for chart in at.get("plotly_chart"):
        spec = json.loads(chart.proto.spec)
        out[spec["layout"]["meta"]["chart"]] = spec
    return out


@pytest.mark.parametrize("prefix, mode", [("eq", "[ EQUITY ]"), ("fx", "[ FX ]")])
def test_presets_set_the_inputs(prefix, mode):
    """UIE-03 / PP-08: choosing a preset writes the inputs and reprices."""

    from derivx.ui.pricer_page import MODES

    spec = next(s for s in MODES.values() if s.prefix == prefix)
    at = _app(mode)
    for preset in ("HIGH VOL", "DEEP ITM", "FAR OTM", "DEFAULT"):
        at.radio(key=f"{prefix}_preset").set_value(preset).run()
        assert not at.exception
        values = _inputs(at, prefix)
        assert values == pytest.approx(dict(spec.presets[preset]))
        v = values
        expected = float(bs_price(v["spot"], v["strike"], v["T"], v["r"], v["vol"], q=v["q"]))
        assert _premium(at) == pytest.approx(expected, abs=10.0**-spec.price_decimals)
    if prefix == "eq":
        at.radio(key="eq_preset").set_value("HIGH VOL").run()
        assert _premium(at) == pytest.approx(36.2613, abs=1e-4)  # the audit's fresh-session value


@pytest.mark.parametrize("mode, prefix", [("[ EQUITY ]", "eq"), ("[ FX ]", "fx")])
@pytest.mark.parametrize(
    "field, value, message",
    [
        ("spot", 0.0, "SPOT.S must be greater than 0"),
        ("spot", -100.0, "SPOT.S must be greater than 0"),  # below min_value: Streamlit resets it to 0
        ("strike", 0.0, "STRIKE.K must be greater than 0"),
        ("strike", -50.0, "STRIKE.K must be greater than 0"),
    ],
)
def test_bad_inputs_show_a_friendly_error_not_a_traceback(mode, prefix, field, value, message):
    """PP-05 / UIE-01: S <= 0 or K <= 0 used to raise in the smile fit."""

    at = _app(mode)
    at.number_input(key=f"{prefix}_{field}").set_value(value).run()
    assert not at.exception
    assert any(message in e.value for e in at.error)
    assert not at.get("plotly_chart")  # nothing is priced from invalid inputs


def test_expiry_shows_a_warning_and_the_limit_values():
    """PP-06 / ui-engineering-V4: T = 0 gives intrinsic value, a step delta
    and zero gamma / theta instead of gamma 19,947 and theta -10,930."""

    at = _app()
    at.number_input(key="eq_strike").set_value(90.0).run()
    at.number_input(key="eq_T").set_value(0.0).run()
    assert not at.exception
    assert any("EXPIRED (T = 0)" in w.value for w in at.warning)
    assert _premium(at) == pytest.approx(10.0)  # intrinsic, S = 100, K = 90
    cards = " ".join(m.value for m in at.markdown if "dx-greek-value" in m.value and "<style>" not in m.value)
    values = dict(re.findall(r"dx-greek-name'>([^<]+)</div><div class='dx-greek-value'>([^<]+)<", cards))
    assert values["DELTA Δ"] == "1.000" and values["GAMMA Γ"] == "0" and values["THETA Θ"] == "0"
    assert values["VEGA ν"] == "0"
    charts = _charts(at)
    assert set(charts) == {"scenario", "surface"}  # no smile at T = 0 (zero total variance)
    assert any("needs T > 0" in i.value for i in at.info)
    # ATM at expiry: delta is 1/2 (the documented midpoint), still no NaN
    at.number_input(key="eq_strike").set_value(100.0).run()
    assert _premium(at) == 0.0 and not at.exception


def test_put_surface_prices_puts():
    """PP-02 / ui-engineering-V2: the surface used to price calls for a PUT."""

    at = _app()
    at.number_input(key="eq_strike").set_value(120.0).run()
    at.number_input(key="eq_T").set_value(2.0).run()
    call_z = _arr(_charts(at)["surface"]["data"][0]["z"])
    at.selectbox(key="eq_type").set_value("PUT").run()
    surf = _charts(at)["surface"]
    x, y, z = (_arr(surf["data"][0][k]) for k in ("x", "y", "z"))
    X, Y = np.meshgrid(x, y)
    np.testing.assert_allclose(z, bs_price(X, 120.0, 2.0, 0.05, Y, call=False), rtol=1e-12, atol=1e-12)
    assert not np.allclose(z, call_z)
    marker = surf["data"][1]
    assert marker["z"][0] == pytest.approx(_premium(at), abs=1e-4) == pytest.approx(16.5087, abs=1e-4)
    assert "PUT VALUE" in surf["layout"]["scene"]["zaxis"]["title"]["text"]


def test_scenario_grid_is_the_position_pnl():
    """PP-15: spot x vol full-revaluation P&L for the signed position."""

    at = _app()
    at.number_input(key="eq_position").set_value(-200.0).run()
    at.number_input(key="eq_scn_days").set_value(7.0).run()
    chart = _charts(at)["scenario"]["data"][0]
    z = _arr(chart["z"])
    x, y = chart["x"], chart["y"]
    assert x[0] == "-10%" and x[-1] == "+10%" and y[0] == "-5" and y[-1] == "+5"
    centre = z[y.index("+0"), x.index("+0%")]
    decay = float(bs_price(100, 100, 1 - 7 / 365, 0.05, 0.2)) - float(bs_price(100, 100, 1, 0.05, 0.2))
    assert centre == pytest.approx(-200 * decay)  # a short option earns a week of decay
    corner = z[y.index("+5"), x.index("+10%")]
    assert corner == pytest.approx(-200 * (float(bs_price(110, 100, 1 - 7 / 365, 0.05, 0.25)) - 10.450583572185565))
    table = _frame(at, "+10.0%")
    assert table.shape == (7, 9)


def test_surface_domain_scales_with_vol_and_time_not_a_fixed_box():
    """PP-09: the spot axis covers spot and strike +- n sigma sqrt(T)."""

    at = _app("[ FX ]")
    x = _arr(_charts(at)["surface"]["data"][0]["x"])
    assert 0.9 < x[0] < 1.08 < x[-1] < 1.3  # FX level, not 1e-6..11.08
    at.radio(key="fx_surf_axes").set_value("SPOT × TIME TO EXPIRY").run()
    surf = _charts(at)["surface"]
    y = _arr(surf["data"][0]["y"])
    assert y[0] == 0.0 and y[-1] == pytest.approx(0.25)
    assert "TIME TO EXPIRY" in surf["layout"]["scene"]["yaxis"]["title"]["text"]
    assert surf["layout"]["scene"]["aspectmode"] == "manual"  # still bound_scene-bounded


def _figures(at: AppTest) -> list:
    return [c.proto.spec for c in at.get("plotly_chart")]


@pytest.mark.parametrize("mode", ["[ EQUITY ]", "[ FX ]"])
def test_reruns_are_deterministic(mode):
    """PP-01 / UIE-05 / VM-14 / G14: identical inputs give identical figures
    (no random 'MARKET' smile)."""

    at = _app(mode)
    first = _figures(at)
    at.run()
    assert _figures(at) == first
    smile = _charts(at)["smile"]
    assert [t["name"] for t in smile["data"]][0].endswith("(illustrative)")
    assert all("MARKET" not in t["name"] for t in smile["data"])


def test_mode_round_trip_keeps_values_and_modes_do_not_share_keys():
    """UIE-04: EQUITY -> RATES -> EQUITY keeps the inputs; FX has its own."""

    at = _app()
    at.number_input(key="eq_vol").set_value(0.33).run()
    at.number_input(key="eq_spot").set_value(123.0).run()
    at.selectbox(key="eq_type").set_value("PUT").run()
    at.slider(key="eq_surf_res").set_value(31).run()
    premium = _premium(at)
    at.sidebar.radio[0].set_value("[ FX ]").run()
    assert _inputs(at, "fx")["spot"] == pytest.approx(1.08)  # not the equity market
    at.number_input(key="fx_vol").set_value(0.123).run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    at.sidebar.radio[0].set_value("[ EQUITY ]").run()
    assert not at.exception
    assert _inputs(at, "eq")["vol"] == pytest.approx(0.33) and _inputs(at, "eq")["spot"] == pytest.approx(123.0)
    assert at.selectbox(key="eq_type").value == "PUT" and at.slider(key="eq_surf_res").value == 31
    assert _premium(at) == pytest.approx(premium)
    at.sidebar.radio[0].set_value("[ FX ]").run()
    assert _inputs(at, "fx")["vol"] == pytest.approx(0.123)


def test_implied_vol_solver_round_trips_and_explains_bound_violations():
    """PP-15: premium -> implied vol, with the no-arbitrage bounds."""

    at = _app()
    premium = float(bs_price(100.0, 100.0, 1.0, 0.05, 0.27))
    at.number_input(key="eq_iv_premium").set_value(premium).run()
    assert any("IMPLIED VOL 27.0000%" in m.value for m in at.markdown)
    at.button(key="eq_use_iv").click().run()
    assert _inputs(at)["vol"] == pytest.approx(0.27, abs=1e-10)
    assert _premium(at) == pytest.approx(premium, abs=1e-4)
    at.number_input(key="eq_iv_premium").set_value(1.0).run()  # below max(S e^-qT - K e^-rT, 0) = 4.877
    assert any("below the lower bound" in e.value for e in at.error)
    at.number_input(key="eq_iv_premium").set_value(100.0).run()  # = S e^-qT, the infinite-vol value
    assert any("at or above the upper bound" in e.value for e in at.error)
    assert not at.exception


def test_greeks_panel_has_units_and_distinct_labels():
    """PP-13 / UIE-21: units on every Greek, 'RHO (RATE.R)' is not the SVI
    rho, small values are not printed as 0.0000, second-order Greeks."""

    at = _app()
    cards = " ".join(m.value for m in at.markdown if "dx-greek-value" in m.value and "<style>" not in m.value)
    units = ("per 1.00 of spot", "per +1% spot", "per +1 vol pt", "per calendar day", "per +1bp of r", "1-day repriced")
    for label in units:
        assert label in cards
    for name in ("VANNA", "VOLGA", "CHARM", "RHO (RATE.R)", "RHO (DIV.Q)"):
        assert name in cards
    params = _frame(at, "parameter")
    assert set(params["parameter"]) == {"SVI a", "SVI b", "SVI rho", "SVI m", "SVI sigma"}
    # deep OTM short-dated: tiny Greeks keep significant figures
    at.number_input(key="eq_strike").set_value(130.0).run()
    at.number_input(key="eq_T").set_value(0.05).run()
    cards = " ".join(m.value for m in at.markdown if "dx-greek-value" in m.value and "<style>" not in m.value)
    delta = re.search(r"DELTA Δ</div><div class='dx-greek-value'>([^<]+)<", cards).group(1)
    assert "e-" in delta and float(delta) == pytest.approx(float(bs_greeks(100, 130, 0.05, 0.05, 0.2).delta), rel=1e-3)
    assert "PUT-CALL PARITY" not in " ".join(m.value for m in at.markdown)


def test_model_check_compares_analytic_greeks_with_finite_differences():
    """PP-11 / UIE-21: the tautological parity card is replaced."""

    at = _app()
    table = _frame(at, "greek")
    assert list(table["greek"]) == ["delta", "gamma", "vega", "theta", "rho", "phi", "vanna", "volga", "charm"]
    assert bool(table["ok"].all())
    assert any("Analytic Greeks vs central differences of the pricer [OK]" in m.value for m in at.markdown)


def test_hero_shows_context_and_no_dollar_sign_outside_usd():
    """PP-17: spot, forward, moneyness and tenor; '$' never, currency code
    only when the context gives one."""

    at = _app()
    hero = _hero(at)
    for text in ("SPOT", "FORWARD", "105.13", "K / F", "0.9512", "1.0000Y · 365D", " USD"):
        assert text in hero
    assert "$" not in hero
    at.selectbox(key="eq_ccy").set_value("OTHER").run()
    assert "USD" not in _hero(at) and "price units per one share" in _hero(at)
    at.sidebar.radio[0].set_value("[ FX ]").run()
    hero = _hero(at)
    assert "$" not in hero and "USD" not in hero and "quote-currency units per 1 unit of base currency" in hero
    at.number_input(key="fx_spot").set_value(1.0835).run()
    assert "1.08350" in _hero(at)  # FX precision: the priced spot is what is shown


def _loaded_chain() -> pd.DataFrame:
    return synth_chain(days=(30, 91), spot=580.0)


def test_market_smile_from_a_loaded_chain_is_labelled_and_fitted():
    """VM-05 / PP-10 / VM-14: a loaded chain gives a MARKET slice fitted
    with SVI on its forward and T; the ATM marker sits at the forward."""

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["market_chain"] = _loaded_chain()
    at.session_state["market_chain_meta"] = {"ticker": "SYNTH", "source": "test fixture"}
    at.run()
    assert not at.exception
    assert _charts(at)["smile"]["data"][0]["name"].endswith("(illustrative)")  # the default stays illustrative
    at.radio(key="eq_smile_src").set_value("MARKET").run()
    assert not at.exception
    smile = _charts(at)["smile"]
    names = [t["name"] for t in smile["data"]]
    assert names[0].startswith("MARKET SYNTH") and "SVI fit" in names and "ATM-F" in names
    atm = next(t for t in smile["data"] if t["name"] == "ATM-F")
    F = 580.0 * np.exp((0.04 - 0.012) * 30 / 365)
    assert _arr(atm["x"])[0] == pytest.approx(F, rel=2e-3)  # parity forward, not spot
    caption = " ".join(c.value for c in at.caption)
    assert "SVI fit RMSE" in caption and "trade 2026-10-19" in at.selectbox(key="eq_smile_expiry").value
    assert any("MARKET</span>SYNTH" in m.value for m in at.markdown)
    at.run()
    assert _charts(at)["smile"] == smile  # cached and deterministic


def test_fx_page_never_offers_an_equity_chain_as_its_market_smile():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["market_chain"] = _loaded_chain()
    at.run()
    at.sidebar.radio[0].set_value("[ FX ]").run()
    assert not at.exception
    assert not [r for r in at.radio if r.key == "fx_smile_src"]
    assert _charts(at)["smile"]["data"][0]["name"] == "SABR (illustrative)"
