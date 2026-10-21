"""BOOK tab: the blotter prices into risk, charts are zoom-bounded and
badged, bad rows say why, edits survive the editor not being drawn, every
section exports with a manifest, and SCENARIOS and RESEARCH run on the book
by default with the tab trades one click away."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from derivx.rates import portfolio as pf
from derivx.rates.examples import EXAMPLE_ASOF
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
        "caplets_ois_3m",
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


# --------------------------------------------------------------------------
# Review findings (regressions)
# --------------------------------------------------------------------------

UPLOAD_APP = """
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from derivx.rates.examples import example_market
from derivx.ui import rates_book


class _File:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


_real = DeltaGenerator.file_uploader


def _fake(self, label, *args, **kwargs):  # AppTest cannot drive a file uploader: the test sets the file
    if label != "UPLOAD BLOTTER CSV":
        return _real(self, label, *args, **kwargs)
    upload = st.session_state.get("fake_upload")
    return None if upload is None else _File(*upload)


DeltaGenerator.file_uploader = _fake
try:
    st.session_state["n_positions"] = len(rates_book.render(example_market()))
finally:
    DeltaGenerator.file_uploader = _real
"""
GOOD = b"id,type,currency,direction,notional,tenor,index\nU1,IRS,usd,rec,10mm,5Y,OIS\nU2,swap,USD,Pay,5mm,2Y,OIS\n"


def _upload_app() -> AppTest:
    at = AppTest.from_string(UPLOAD_APP, default_timeout=600)
    at.run()
    _ok(at)
    return at


def _upload(at: AppTest, name, payload) -> None:
    at.session_state["fake_upload"] = None if name is None else (name, payload)
    at.run()
    _ok(at)


def test_unreadable_uploads_say_why_and_keep_the_book():
    at = _upload_app()
    for name, payload, text in [
        ("ids.csv", b"id\nX1\nX2\n", "missing columns: type, currency, direction, notional"),  # was a traceback
        ("other.csv", b"foo,bar\n1,2\n", "no blotter columns in the header (foo, bar)"),  # was a traceback
        ("empty.csv", b"", "the file is empty"),  # was a traceback (csv.Error)
    ]:
        _upload(at, name, payload)
        errors = [e.value for e in at.error]
        assert any(e.startswith(f"{name} was not loaded: ") and text in e for e in errors), errors
        assert at.session_state["n_positions"] == 10  # the example book stays
        at.run()  # the reason stays up while that file is in the uploader
        assert any(e.startswith(f"{name} was not loaded") for e in (x.value for x in at.error))
    _upload(at, None, None)
    assert not any("was not loaded" in e.value for e in at.error)


def test_the_same_file_uploads_again_after_the_book_was_replaced():
    at = _upload_app()
    _upload(at, "mine.csv", GOOD)
    assert at.session_state["n_positions"] == 2
    table = at.session_state["book_state"]["table"]
    assert list(table["type"]) == ["swap", "swap"] and list(table["direction"]) == ["receive", "pay"]
    assert list(table["currency"]) == ["USD", "USD"]  # the editor's select cells show these
    at.button(key="book_clear").click().run()
    _ok(at)
    assert at.session_state["n_positions"] == 0
    _upload(at, None, None)  # the user removes the file from the uploader ...
    _upload(at, "mine.csv", GOOD)  # ... and drops the same file again: it used to be ignored
    assert at.session_state["n_positions"] == 2


def test_hedge_cards_fit_their_values():
    """Metric cards hold about 20 characters at 1280px: the hedge cards used
    to show '8,995 of 36,...' and 'SQUARED RISK REMOV...'."""

    at = _app()
    hedge = [m for m in at.metric if any(w in m.label for w in ("RESIDUAL", "UNHEDGED", "BID/ASK"))]
    assert [m.label for m in hedge] == ["RESIDUAL RISK ($/BP)", "UNHEDGED RISK ($/BP)", "BID/ASK COST ($)"]
    for metric in at.metric:
        assert len(metric.label) <= 20 and len(str(metric.value)) <= 14, (metric.label, metric.value)
    assert any("of the squared risk is removed" in c.value for c in at.caption)


MIXED_APP = """
import streamlit as st
from derivx.data.rates_history import bundled_store
from derivx.rates import curve_io
from derivx.rates.examples import EXAMPLE_ASOF
from derivx.ui import rates_book

market, _, _ = curve_io.default_market(bundled_store(), EXAMPLE_ASOF, "EUR ESTR")
st.session_state["n_positions"] = len(rates_book.render(market))
"""


def _mixed_book() -> pd.DataFrame:
    table = pf.example_blotter("EUR").astype(object)
    bond = {c: None for c in pf.COLUMNS} | dict(id="UST", type="bond", currency="USD", direction="long",
                                                notional="10mm", maturity="2035-08-15", rate=4.25)  # fmt: skip
    return pd.concat([table, pd.DataFrame([bond])], ignore_index=True).astype(object)


def _mixed_app(**state) -> AppTest:
    at = AppTest.from_string(MIXED_APP, default_timeout=600)
    at.session_state["book_state"] = {"table": _mixed_book(), "source": "test book", "example": None, "gen": 3}
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    _ok(at)
    return at


def _frame(at: AppTest, column: str) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if column in d.value.columns)


def _row(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    return frame[frame[column] == label].iloc[0]


def test_a_euro_book_shows_euro_cards_only():
    """The euro market carries the US Treasury curve, but a book of euro
    trades holds no dollars: no dollar hedge cards (they read 0 before). Its
    totals are in euros, and in dollars only as the converted (IN $) total
    of the default reporting currency."""

    at = AppTest.from_string(MIXED_APP, default_timeout=600)
    table = pf.example_blotter("EUR").astype(object)
    at.session_state["book_state"] = {"table": table, "source": "test book", "example": None, "gen": 3}
    at.run()
    _ok(at)
    labels = [m.label for m in at.metric]
    assert [x for x in labels if x.startswith(("RESIDUAL", "UNHEDGED", "BID/ASK"))] == [
        "RESIDUAL RISK (€/BP)", "UNHEDGED RISK (€/BP)", "BID/ASK COST (€)"
    ]  # fmt: skip
    assert "BOOK PV (€)" in labels and "BOOK PV (IN $)" in labels
    assert not [x for x in labels if "$" in x and "IN $" not in x]  # no unconverted dollar amount
    summary = _frame(at, "DESCRIPTION")
    assert set(summary[summary["ID"] != "ALL IN USD"]["CCY"]) == {"EUR"} and "TOTAL" in set(summary["ID"])
    assert any("% in EUR of the squared risk" in c.value for c in at.caption)
    assert not any("% in USD" in c.value for c in at.caption)  # was "100.00% in USD" of a zero dollar ladder


def test_a_book_in_two_currencies_is_totalled_per_currency_and_converted():
    """Review finding (R2, A1): totals added EUR trade P&L to USD bond P&L
    without FX conversion. A EUR book with a US Treasury now shows a total
    per currency and the whole book in USD at the ECB reference rate of the
    valuation date, in the summary, the cards, the scenario grid (with FX
    rows), the VaR, the hedges (per currency) and the export; every card
    still fits."""

    from derivx.risk.reporting import default_reporting

    at = _mixed_app()
    assert at.session_state["n_positions"] == 8
    eurusd = default_reporting(EXAMPLE_ASOF).fx.quote("EURUSD")
    texts = [w.value for w in at.warning] + [c.value for c in at.caption] + [m.label for m in at.metric]
    assert not any("without FX conversion" in t or "MIXED" in t or "mixed" in t for t in texts)

    summary = _frame(at, "DESCRIPTION")
    assert {"TOTAL EUR", "TOTAL USD", "ALL IN USD"} <= set(summary["ID"])
    eur, usd, book = (_row(summary, "ID", k) for k in ("TOTAL EUR", "TOTAL USD", "ALL IN USD"))
    assert (eur["CCY"], usd["CCY"], book["CCY"]) == ("EUR", "USD", "USD")
    for column in ("PV", "PV01", "GAMMA", "VEGA (NORMAL BP)"):
        assert book[column] == pytest.approx(usd[column] + eur[column] * eurusd, rel=1e-12)
    trades = summary[~summary["ID"].str.startswith(("TOTAL", "ALL IN"))]
    assert eur["PV"] == pytest.approx(trades[trades["CCY"] == "EUR"]["PV"].sum(), rel=1e-12)
    assert "EURUSD" in book["DESCRIPTION"] and str(EXAMPLE_ASOF) in book["DESCRIPTION"]
    assert _metric(at, "BOOK PV (IN $)") == f"{round(book['PV']) + 0.0:,.0f}"
    assert {"BOOK PV (€)", "BOOK PV ($)", "HIST VAR 1D (IN $)"} <= {m.label for m in at.metric}

    grid = _frame(at, "SCENARIO")
    assert {"EU1 (EUR)", "UST (USD)", "TOTAL EUR", "TOTAL USD", "ALL IN USD"} <= set(grid.columns)
    rates = grid[~grid["SCENARIO"].str.startswith("FX ")]
    np.testing.assert_allclose(rates["ALL IN USD"], rates["TOTAL USD"] + rates["TOTAL EUR"] * eurusd, rtol=1e-10)
    up = _row(grid, "SCENARIO", "FX EURUSD +10%")
    assert up["TOTAL EUR"] == 0.0 and up["TOTAL USD"] == 0.0  # FX moves no local value
    assert up["ALL IN USD"] == pytest.approx(eur["PV"] * eurusd * 0.10, rel=1e-10)  # the euro book's translation
    assert _row(grid, "SCENARIO", "FX EURUSD -10%")["ALL IN USD"] == pytest.approx(-up["ALL IN USD"], rel=1e-10)

    var = _frame(at, "HIST VAR")
    assert list(var["ID"])[-3:] == ["TOTAL USD", "TOTAL EUR", "ALL IN USD"]
    assert _metric(at, "HIST VAR 1D (IN $)") == f"{round(_row(var, 'ID', 'ALL IN USD')['HIST VAR']) + 0.0:,.0f}"

    labels = [m.label for m in at.metric]
    assert {"RESIDUAL RISK (€/BP)", "UNHEDGED RISK (€/BP)", "RESIDUAL RISK ($/BP)", "UNHEDGED RISK ($/BP)"} <= set(
        labels
    )
    hedges = _frame(at, "HEDGE")
    assert set(hedges[hedges["CURVE"] == "TREASURY"]["CCY"]) == {"USD"}
    assert set(hedges[hedges["CURVE"] != "TREASURY"]["CCY"]) == {"EUR"}
    for metric in at.metric:
        assert len(metric.label) <= 20 and len(str(metric.value)) <= 14, (metric.label, metric.value)


def test_mixed_book_reports_in_euros_and_at_rates_typed_in():
    """REPORTING CURRENCY EUR converts the Treasury into euros; ENTER RATES
    converts at the typed EURUSD and says so."""

    from derivx.risk.reporting import default_reporting

    eurusd = default_reporting(EXAMPLE_ASOF).fx.quote("EURUSD")
    at = _mixed_app(rates_report_ccy="EUR")
    summary = _frame(at, "DESCRIPTION")
    eur, usd, book = (_row(summary, "ID", k) for k in ("TOTAL EUR", "TOTAL USD", "ALL IN EUR"))
    assert book["CCY"] == "EUR" and book["PV"] == pytest.approx(eur["PV"] + usd["PV"] / eurusd, rel=1e-12)
    assert "BOOK PV (IN €)" in [m.label for m in at.metric]
    grid = _frame(at, "SCENARIO")
    assert "ALL IN EUR" in grid.columns
    # EURUSD +10%: the dollar bond is worth 1/1.1 - 1 = -9.09% in euros
    up = _row(grid, "SCENARIO", "FX EURUSD +10%")
    assert up["ALL IN EUR"] == pytest.approx(usd["PV"] / eurusd * (1.0 / 1.1 - 1.0), rel=1e-10)

    typed = _mixed_app(rates_report_ccy="EUR", rates_fx_source="ENTER RATES", rates_fx_eurusd=1.25,
                       rates_fx_gbpusd=1.40)  # fmt: skip
    summary = _frame(typed, "DESCRIPTION")
    eur, usd, book = (_row(summary, "ID", k) for k in ("TOTAL EUR", "TOTAL USD", "ALL IN EUR"))
    assert book["PV"] == pytest.approx(eur["PV"] + usd["PV"] / 1.25, rel=1e-12)
    assert "EURUSD 1.25000" in book["DESCRIPTION"] and "entered by you" in book["DESCRIPTION"]


def test_mixed_book_export_states_the_conversion(monkeypatch):
    from derivx.ui import rates_exports

    seen = {}
    original = rates_exports.export_row

    def spy(kind, market, tables, inputs, model, key, store=None):
        seen[kind] = (tables, inputs)
        return original(kind, market, tables, inputs, model, key, store)

    monkeypatch.setattr(rates_exports, "export_row", spy)
    _mixed_app()
    tables, inputs = seen["book"]
    fx = inputs["reporting"]
    assert fx["reporting_currency"] == "USD" and fx["converted"] and fx["position_currencies"] == ["USD", "EUR"]
    assert fx["fx_asof"] == EXAMPLE_ASOF and fx["fx"]["EURUSD"] > 0 and "ECB" in fx["fx_source"]
    report = tables["risk_report"]
    pv = report[report["section"] == "pv"].set_index("id")
    assert pv.loc["ALL IN USD", "value"] == pytest.approx(
        pv.loc["TOTAL USD", "value"] + pv.loc["TOTAL EUR", "value"] * fx["fx"]["EURUSD"], rel=1e-12
    )
    assert {"TOTAL USD", "TOTAL EUR", "ALL IN USD"} <= set(tables["scenario_grid"].columns)
    assert set(tables["hedges"]["CCY"]) <= {"EUR", "USD"}


def test_reruns_skip_work_nobody_asked_for(monkeypatch):
    """A3: the export ZIP and manifest are written when their button is
    clicked, not on every rerun, and the scenario library (episodes and PCA
    factors read from the history) is built once per store and date."""

    from derivx.ui import rates_exports, rates_scenarios

    calls = {"zip": 0, "library": 0}
    zipper, library = rates_exports.bundle_zip, rates_scenarios.standard_library

    def counted_zip(*args, **kwargs):
        calls["zip"] += 1
        return zipper(*args, **kwargs)

    def counted_library(*args, **kwargs):
        calls["library"] += 1
        return library(*args, **kwargs)

    monkeypatch.setattr(rates_exports, "bundle_zip", counted_zip)
    monkeypatch.setattr(rates_scenarios, "standard_library", counted_library)
    at = _app()
    at.number_input(key="book_hedge_move").set_value(6.0).run()
    _ok(at)
    assert calls == {"zip": 0, "library": 1}
    export = _downloads(at)["book_export_zip"]
    assert export.proto.label == "DOWNLOAD TABLES + MANIFEST (ZIP)"
