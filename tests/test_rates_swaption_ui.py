"""SWAPTION tab: premium and Greeks on the SABR cube, labelled illustrative
quotes, pasted grids, bounded charts, and swaptions in the scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from derivx.rates.examples import example_market

ROOT = Path(__file__).resolve().parents[1]
FULL_APP = ROOT / "src" / "derivx" / "ui" / "streamlit_app.py"
APP = """
import streamlit as st
from derivx.rates.examples import example_market
from derivx.ui.rates_swaption import render

st.session_state["swn_test_positions"] = render(example_market())
"""
PASTED = """expiry,tenor,-50,ATM,+50
1Y,5Y,104.0,100.0,102.0
1Y,10Y,99.0,95.0,97.5
2Y,5Y,101.0,97.0,99.0
2Y,10Y,96.0,92.0,94.5
"""


def _metric(at: AppTest, label: str) -> float:
    value = next(m.value for m in at.metric if m.label == label)
    return float(value.replace(",", ""))


def _charts(at: AppTest) -> dict:
    return {c.proto.id.rsplit("-", 1)[-1]: json.loads(c.proto.spec) for c in at.get("plotly_chart")}


def _in_use(at: AppTest) -> str:
    return next(c.value for c in at.caption if c.value.startswith("Vol quotes in use:"))


def _assert_contract(at: AppTest) -> None:
    """No traceback; every widget keyed ``swn_``; every chart axis bounded."""

    assert not at.exception, [e.value for e in at.exception]
    for chart in at.get("plotly_chart"):
        layout = json.loads(chart.proto.spec)["layout"]
        limits = layout["meta"]["zoom"]["axes"]
        axes = [name for name in layout if name.startswith(("xaxis", "yaxis"))]
        assert axes and set(axes) <= set(limits), (chart.proto.id, axes)
        for name in axes:
            assert layout[name]["minallowed"] == limits[name]["min"] == layout[name]["range"][0]
            assert layout[name]["maxallowed"] == limits[name]["max"] == layout[name]["range"][1]
    for kind in ("radio", "selectbox", "number_input", "checkbox", "text_area", "button"):
        for widget in getattr(at, kind):
            assert widget.key is None or widget.key.startswith(("swn_", "derivx_")), widget.key


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_string(APP, default_timeout=300)
    at.run()
    _assert_contract(at)
    return at


def test_tab_prices_on_the_illustrative_cube_and_says_so(app):
    assert not app.error, [e.value for e in app.error]
    position = app.session_state["swn_test_positions"][0]
    assert position.kind == "swaption" and position.vol_model == "normal"
    pv = _metric(app, "PV ($)")
    assert pv == pytest.approx(position.pv(example_market()), abs=0.01)
    assert _metric(app, "PREMIUM (BP)") == pytest.approx(pv / 1e7 * 1e4, abs=1e-3)
    trade = position.details["trade"]
    assert trade.expiry_date is not None and trade.strike == pytest.approx(trade.atm_strike(example_market()))
    for label in ("DELTA ($/BP FWD)", "PV01 ($/BP CURVE)", "GAMMA ($/BP²)", "VEGA ($/BP NORMAL)", "VEGA ($/VOL PT)",
                  "THETA ($, 1 BUS. DAY)", "FWD SWAP RATE (%)"):  # fmt: skip
        assert label in [m.label for m in app.metric]
    assert _metric(app, "THETA ($, 1 BUS. DAY)") < 0 < _metric(app, "VEGA ($/BP NORMAL)")
    charts = _charts(app)
    assert {"swn_smile_chart", "swn_atm_heatmap", "swn_vega_heatmap"} <= set(charts)
    for key in ("swn_smile_chart", "swn_atm_heatmap", "swn_vega_heatmap"):
        badge = next(a["text"] for a in charts[key]["layout"]["annotations"] if a["text"].startswith("SOURCE:"))
        assert "ILLUSTRATIVE" in badge
    smile = charts["swn_smile_chart"]
    assert {t["name"] for t in smile["data"]} == {"SABR (CUBE)", "QUOTES", "TRADE STRIKE"}
    assert smile["layout"]["yaxis"]["minallowed"] >= 0.0
    params = next(d.value for d in app.dataframe if "RMSE (BP)" in d.value.columns)
    assert params["STATUS"].iloc[0] == "fitted" and params["BETA"].iloc[0] == 0.0 and params["RMSE (BP)"].iloc[0] < 0.1
    residuals = next(d.value for d in app.dataframe if "RESIDUAL (BP)" in d.value.columns)
    assert len(residuals) == 9 and residuals["RESIDUAL (BP)"].abs().max() < 0.1
    assert _in_use(app).startswith("Vol quotes in use: ILLUSTRATIVE quotes (not market data): USD swaption")


def test_every_model_gives_the_cube_premium(app):
    pv = _metric(app, "PV ($)")
    for label, unit in (("Shifted Black (%)", "%"), ("Black (%)", "%"), ("Normal (bp)", "BP")):
        app.selectbox(key="swn_model").set_value(label).run()
        _assert_contract(app)
        assert not app.error, [e.value for e in app.error]
        assert _metric(app, "PV ($)") == pytest.approx(pv, abs=0.02)
        assert any(m.label == f"VOL ({unit})" for m in app.metric)
    assert app.session_state["swn_test_positions"][0].vol_model == "normal"


def test_off_grid_points_strikes_sides_and_flat_vols(app):
    app.selectbox(key="swn_expiry").set_value("4Y").run()
    app.selectbox(key="swn_tenor").set_value("7Y").run()
    _assert_contract(app)
    assert any("between quoted points" in c.value for c in app.caption)
    assert not any("RESIDUAL (BP)" in d.value.columns for d in app.dataframe)
    app.radio(key="swn_kind").set_value("Straddle").run()
    app.radio(key="swn_side").set_value("Short").run()
    app.radio(key="swn_strike_mode").set_value("Absolute %").run()
    app.number_input(key="swn_strike_pct").set_value(4.25).run()
    _assert_contract(app)
    assert not app.error
    trade = app.session_state["swn_test_positions"][0].details["trade"]
    assert trade.kind == "straddle" and trade.notional < 0 and trade.strike == pytest.approx(0.0425)
    assert _metric(app, "PV ($)") < 0 < _metric(app, "THETA ($, 1 BUS. DAY)")
    app.radio(key="swn_vol_source").set_value("Flat vol").run()
    _assert_contract(app)
    assert "swn_vega_heatmap" not in _charts(app)
    assert app.session_state["swn_test_positions"][0].details["vol"] == pytest.approx(95e-4)


def test_implied_vol_from_the_premium_shown(app):
    vol = _metric(app, "VOL (BP)")
    assert _metric(app, "IMPLIED NORMAL VOL (BP)") == pytest.approx(vol, abs=0.01)
    app.number_input(key="swn_implied_premium").set_value(0.0).run()
    _assert_contract(app)
    assert any("below the zero-vol" in c.value or "no bachelier vol" in c.value for c in app.caption)


def test_pasted_quotes_replace_the_illustrative_grid_until_reset(app):
    app.text_area(key="swn_grid_paste").input(PASTED).run()
    app.button(key="swn_grid_paste_use").click().run()
    _assert_contract(app)
    assert not app.error, [e.value for e in app.error]
    assert _in_use(app) == "Vol quotes in use: uploaded: pasted quotes."
    heat = _charts(app)["swn_atm_heatmap"]
    assert heat["data"][0]["x"] == ["5Y", "10Y"] and heat["data"][0]["y"] == ["2Y", "1Y"]  # shortest on top
    badge = next(a["text"] for a in heat["layout"]["annotations"] if a["text"].startswith("SOURCE:"))
    assert "pasted quotes" in badge and "ILLUSTRATIVE" not in badge
    assert _metric(app, "VOL (BP)") == pytest.approx(100.0, abs=1e-6)  # 1Y x 5Y ATM quote
    app.text_area(key="swn_grid_paste").input("expiry,tenor,+50\n1Y,5Y,99\n").run()
    app.button(key="swn_grid_paste_use").click().run()
    _assert_contract(app)
    assert any("no ATM quote" in e.value for e in app.error)
    assert _in_use(app) == "Vol quotes in use: uploaded: pasted quotes."  # the good grid is kept
    app.button(key="swn_grid_clear").click().run()
    _assert_contract(app)
    assert "ILLUSTRATIVE" in _in_use(app)


def test_swaptions_flow_into_the_scenarios_of_the_full_app():
    at = AppTest.from_file(str(FULL_APP), default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    at.number_input(key="scn_vol_normal").set_value(10.0).run()
    assert not at.exception and not at.error
    explain = next(d.value for d in at.dataframe if "KIND" in d.value.columns and "VOL ($)" in d.value.columns)
    row = explain[explain["KIND"] == "swaption"].iloc[0]
    assert row["VOL ($)"] > 0 and abs(row["RESIDUAL ($)"]) < 0.02 * abs(row["TOTAL P&L ($)"]) + 1.0
    grid = next(d.value for d in at.dataframe if "SCENARIO" in d.value.columns)
    assert any(c.startswith("1Yx5Y PAYER") or "PAYER" in c for c in grid.columns)


def test_euribor_swaptions_on_the_multi_curve_market():
    at = AppTest.from_file(str(FULL_APP), default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    at.selectbox(key="rates_conv").set_value("EUR ESTR").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert at.radio(key="swn_index_EUR").value == "EURIBOR 6M"
    assert any("projected on EURIBOR 6M" in c.value for c in at.caption)
    assert any(m.label == "PV (€)" for m in at.metric)
    at.radio(key="swn_index_EUR").set_value("OIS").run()
    assert not at.exception and not at.error
    assert any("EUR ESTR OIS swap" in c.value for c in at.caption)
