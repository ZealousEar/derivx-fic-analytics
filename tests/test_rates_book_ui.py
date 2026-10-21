"""BOOK tab: the blotter prices into risk, charts are zoom-bounded and
badged, bad rows say why, edits survive the editor not being drawn, every
section exports with a manifest, and SCENARIOS and RESEARCH run on the book
by default with the tab trades one click away."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from derivx.rates import portfolio as pf
from derivx.rates.indices import HullWhite

ROOT = Path(__file__).resolve().parents[1]
FULL_APP = ROOT / "src" / "derivx" / "ui" / "streamlit_app.py"
APP = """
import streamlit as st
from derivx.rates.examples import example_market
from derivx.ui import rates_book

market = example_market()
if not st.session_state.get("hide"):
    positions = rates_book.render(market)
    st.session_state["n_positions"] = len(positions)
"""
CHARTS = ("book_krd_chart", "book_vega_heatmap", "book_scenario_chart", "book_hedge_chart")
DOWNLOADS = ("book_blotter_csv", "book_krd_csv", "book_vega_csv", "book_scenario_csv", "book_hedge_csv",
             "book_export_zip", "book_export_manifest")  # fmt: skip


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _app(**state) -> AppTest:
    at = AppTest.from_string(APP, default_timeout=600)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    _ok(at)
    return at


def _charts(at: AppTest) -> dict:
    return {c.proto.id.rsplit("-", 1)[-1]: json.loads(c.proto.spec) for c in at.get("plotly_chart")}


def _downloads(at: AppTest) -> dict:
    return {b.proto.id.rsplit("-", 1)[-1]: b for b in at.get("download_button")}


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


@pytest.fixture(scope="module")
def book() -> AppTest:
    return _app()


def test_example_book_renders_with_bounded_badged_charts(book):
    assert not book.error, [e.value for e in book.error]
    charts = _charts(book)
    assert set(CHARTS) <= set(charts)
    for key in CHARTS:
        layout = charts[key]["layout"]
        for name, lim in layout["meta"]["zoom"]["axes"].items():
            assert layout[name]["minallowed"] == lim["min"] and layout[name]["maxallowed"] == lim["max"], (key, name)
    for key in ("book_krd_chart", "book_vega_heatmap", "book_scenario_chart"):
        badge = next(a["text"] for a in charts[key]["layout"]["annotations"] if a["text"].startswith("SOURCE:"))
        assert "AS OF 2026-10-19" in badge.replace("<br>", " ")
    assert "ILLUSTRATIVE" in next(a["text"] for a in charts["book_vega_heatmap"]["layout"]["annotations"])
    krd = charts["book_krd_chart"]
    bars = [t for t in krd["data"] if t["type"] == "bar"]
    assert len(bars) >= 5 and krd["layout"]["barmode"] == "relative"  # stacked by trade
    assert [t["name"] for t in krd["data"] if t["type"] == "scatter"] == ["BOOK"]
    assert set(DOWNLOADS) <= set(_downloads(book))


def test_book_numbers_are_the_portfolio_module(book):
    from derivx.rates.examples import example_market

    summary = next(d.value for d in book.dataframe if "DESCRIPTION" in d.value.columns)
    assert list(summary["ID"])[:10] == [f"US{i}" for i in range(1, 11)] and summary["ID"].iloc[-1] == "TOTAL"
    market = example_market()
    trades, _ = pf.parse_blotter(pf.example_blotter("USD"))
    swaps = [t for t in trades if t.type in ("swap", "bond", "future")]  # no vols involved
    risk = pf.price_book(swaps, market, pf.VolSet(hull_white=HullWhite(0.03, 0.0085)))
    for trade, pv in zip(risk.trades, risk.pv):
        assert summary.loc[summary["ID"] == trade.id, "PV"].iloc[0] == pytest.approx(pv, rel=1e-9)
    assert _metric(book, "BOOK PV ($)").replace(",", "") == f"{round(summary['PV'].iloc[-1]):d}"
    vol = dict(zip(summary["ID"], summary["VOL"]))
    assert vol["US7"] == "stripped" and vol["US9"] == "cube" and vol["US8"] == "flat"
    assert _metric(book, "HIST VAR 1D ($)") != "n/a"
    assert any(m.label.startswith("RESIDUAL RISK") for m in book.metric)


def test_book_export_zip_holds_the_tables_and_the_manifest(monkeypatch):
    from derivx.ui import rates_exports

    seen = {}
    original = rates_exports.export_row

    def spy(kind, market, tables, inputs, model, key, store=None):
        record = original(kind, market, tables, inputs, model, key, store)
        seen[kind] = (tables, record, rates_exports.bundle_zip(record, tables))
        return record

    monkeypatch.setattr(rates_exports, "export_row", spy)
    _app()
    tables, record, payload = seen["book"]
    assert {
        "blotter",
        "book_summary",
        "key_rates_by_trade",
        "vega_buckets",
        "cashflows",
        "scenario_grid",
        "var_table",
        "hedges",
        "curve_nodes",
        "curve_grid",
        "risk_report",
        "sabr_ois",
        "caplets_3m",
    } <= set(tables)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
    assert names[0] == "manifest.json" and "blotter.csv" in names and "risk_report.csv" in names
    assert manifest["kind"] == "book" and len(manifest["inputs"]["blotter"]["trades"]) == 10
    assert manifest["data"]["store"]["rows_sha256"] and manifest["data"]["valuation_date"] == "2026-10-19"
    assert manifest["model"]["vols"]["cube:OIS"].startswith("ILLUSTRATIVE")
    assert record.seed is None


def test_bad_rows_say_why_and_the_rest_is_priced():
    table = pf.example_blotter("USD").astype(object)
    table.loc[0, "direction"] = "sideways"
    table.loc[1, "tenor"] = None
    state = {"book_state": {"table": table, "source": "test book", "example": None, "gen": 7}}
    at = _app(**state)
    errors = [e.value for e in at.error]
    assert any("Rows left out of the book" in e and "row 1 (US1): direction 'sideways'" in e for e in errors)
    assert any("row 2 (US2): give a swap a tenor (5Y) or a maturity date, not neither" in e for e in errors)
    summary = next(d.value for d in at.dataframe if "DESCRIPTION" in d.value.columns)
    assert "US1" not in set(summary["ID"]) and "US3" in set(summary["ID"])


def test_clear_and_reload_the_example():
    at = _app()
    at.button(key="book_clear").click().run()
    _ok(at)
    assert any("The book is empty" in i.value for i in at.info)
    assert at.session_state["n_positions"] == 0
    at.button(key="book_load_example").click().run()
    _ok(at)
    assert at.session_state["n_positions"] == 10


def test_edits_survive_the_editor_not_being_drawn():
    at = _app()
    edited = pf.example_blotter("USD").astype(object).iloc[:3].copy()
    at.session_state["book_current"] = edited  # as if the user had deleted every row but three
    at.session_state["hide"] = True
    at.run()  # another page: the editor is not drawn and Streamlit drops its state
    at.session_state["hide"] = False
    at.run()
    _ok(at)
    assert at.session_state["n_positions"] == 3
    assert any(c.value.startswith("Blotter: EXAMPLE USD") and "(edited by you)" in c.value for c in at.caption)


def test_hedges_join_the_positions_when_asked(book):
    before = book.session_state["n_positions"]
    book.checkbox(key="book_add_hedges").check().run()
    _ok(book)
    assert book.session_state["n_positions"] > before
    book.checkbox(key="book_add_hedges").uncheck().run()
    book.checkbox(key="book_hedge_penalty").uncheck().run()
    _ok(book)
    assert not any("worst first-order loss" in c.value for c in book.caption)


def test_scenarios_and_research_run_on_the_book_by_default():
    at = AppTest.from_file(str(FULL_APP), default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    _ok(at)
    assert not at.error, [e.value for e in at.error]
    assert at.sidebar.radio(key="rates_risk_scope").value == "BOOK"
    assert any(c.value.startswith("RISK ON: BOOK (10 positions)") for c in at.caption)
    grid = next(d.value for d in at.dataframe if "SCENARIO" in d.value.columns and "US1" in d.value.columns)
    assert {"PC1 level +2 sd", "2022 hiking cycle", "Parallel +25bp"} <= set(grid["SCENARIO"])
    var_book = _metric(at, "HIST VAR 1D ($)")
    var_research = _metric(at, "HIST VAR 99% ($)")  # RESEARCH > VAR on the same positions, same risk
    assert var_book == var_research
    at.sidebar.radio(key="rates_risk_scope").set_value("TAB TRADES").run()
    _ok(at)
    assert any(c.value.startswith("RISK ON: TAB TRADES (4 positions)") for c in at.caption)
    at.sidebar.radio(key="rates_risk_scope").set_value("BOOK + TAB TRADES").run()
    _ok(at)
    assert any(c.value.startswith("RISK ON: BOOK + TAB TRADES (14 positions)") for c in at.caption)
    exports = {b.proto.id.rsplit("-", 1)[-1] for b in at.get("download_button")}
    for tab in ("rates_curve", "swap", "swn", "cap", "bond", "book", "scn", "rs_var"):
        assert {f"{tab}_export_zip", f"{tab}_export_manifest"} <= exports, tab
