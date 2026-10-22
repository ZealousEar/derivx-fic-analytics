"""RESEARCH tab: RV, carry and roll-down, PCA and VaR views render without
tracebacks, every chart is zoom-bounded and badged, every table has a CSV
export, inputs that cannot work say why, and the full app wires the tab
and the PCA factor shock into the Rates page."""

from __future__ import annotations

import base64
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
FULL_APP = ROOT / "src" / "derivx" / "ui" / "streamlit_app.py"
APP = """
import streamlit as st
from derivx.rates.examples import example_market
from derivx.rates.bonds import benchmark_bond, bond_position
from derivx.rates.swaps import VanillaSwap, swap_position
from derivx.ui import rates_research

market = example_market()
swap = VanillaSwap.from_conventions(market.curve.conventions, "10Y", 0.0373, 1e7, trade_date=market.asof)
settle = market.govt.conventions.cal.add_business_days(market.asof, 1)
positions = [swap_position(swap), bond_position(benchmark_bond(market.govt, 10, settle), 1e7)]
if st.session_state.get("no_positions"):
    positions = []
rates_research.render(market, positions)
"""
CHARTS = ("rs_rv_chart", "rs_carry_chart", "rs_pca_loadings", "rs_pca_explained", "rs_pca_scores", "rs_pca_heatmap",
          "rs_var_hist")  # fmt: skip
DOWNLOADS = ("rs_rv_csv", "rs_carry_csv", "rs_pca_csv", "rs_pca_explained_csv", "rs_pca_rolling_csv",
             "rs_pca_risk_csv", "rs_var_csv", "rs_var_pnl_csv", "rs_var_report_csv")  # fmt: skip


PCA_START = "rs_pca_start_UST_PAR_2014-12-31_2026-10-19"  # keyed on the daily history's first and last dates
DATED_APP = """
import datetime as dt
import streamlit as st
from derivx.rates.curves import Market
from derivx.rates.examples import example_curve, ust_curve
from derivx.rates.swaps import VanillaSwap, swap_position
from derivx.ui import rates_research

asof = st.session_state.get("asof", dt.date(2026, 10, 19))
market = Market(curve=example_curve(asof=asof), govt=ust_curve(asof) if asof.year == 2026 else None)
swap = VanillaSwap.from_conventions(market.curve.conventions, "10Y", 0.0373, 1e7, trade_date=market.asof)
rates_research.render(market, [swap_position(swap)])
"""


def _pca_caption(at: AppTest) -> str:
    return next(c.value for c in at.caption if c.value.startswith("PCA of daily changes"))


def _app(**state) -> AppTest:
    at = AppTest.from_string(APP, default_timeout=300)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    _ok(at)
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _charts(at: AppTest) -> dict:
    return {c.proto.id.rsplit("-", 1)[-1]: json.loads(c.proto.spec) for c in at.get("plotly_chart")}


def _values(trace_values) -> list:
    """Trace data as a list (Plotly sends numpy arrays as base64 typed arrays)."""

    if isinstance(trace_values, dict) and "bdata" in trace_values:
        return np.frombuffer(base64.b64decode(trace_values["bdata"]), dtype=trace_values["dtype"]).tolist()
    return list(trace_values)


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _number(text: str) -> float:
    return float(text.replace(",", "").rstrip("%"))


def _downloads(at: AppTest) -> set:
    return {d.proto.id.rsplit("-", 1)[-1] for d in at.get("download_button")}


def test_every_view_renders_bounded_badged_and_exportable():
    at = _app()
    assert not at.error, [e.value for e in at.error]
    charts = _charts(at)
    assert set(charts) == set(CHARTS)
    for key, spec in charts.items():
        layout = spec["layout"]
        for name, lim in layout["meta"]["zoom"]["axes"].items():
            assert layout[name]["minallowed"] == lim["min"] and layout[name]["maxallowed"] == lim["max"], key
    badge = next(a["text"] for a in charts["rs_rv_chart"]["layout"]["annotations"] if a["text"].startswith("SOURCE"))
    assert "US Treasury par yield curve (US Treasury, bundled snapshot)" in badge.replace("<br>", " ")
    assert set(DOWNLOADS) <= _downloads(at)
    # RV: the 2s10s spread on 2026-10-19 (4.77% 2Y, 5.15% 10Y).
    assert _metric(at, "LATEST (BP)") == "38.0"
    # PCA: the audit's level/slope/curvature split on the daily Treasury history.
    assert _number(_metric(at, "PC1 LEVEL")) == pytest.approx(77.9, abs=0.2)
    assert _number(_metric(at, "PC2 SLOPE")) == pytest.approx(12.9, abs=0.2)
    # VaR: historical and delta-normal numbers for the book, full revaluation on request.
    assert _number(_metric(at, "HIST VAR 99% ($)")) > 0 and _metric(at, "FULL-REVAL VAR ($)") == "not run"
    assert _number(_metric(at, "HIST ES 99% ($)")) >= _number(_metric(at, "HIST VAR 99% ($)"))
    sources = next(d.value for d in at.dataframe if "FLAG" in d.value.columns)
    assert set(sources["FLAG"]) == {"REAL: the curve's own market", "PROXY: another market's moves"}
    assert any("swap spreads held" in note for note in sources["NOTE"])


def test_rv_structures_weights_and_bad_inputs():
    at = _app()
    at.selectbox(key="rs_rv_structure").set_value("2s5s10s").run()
    at.radio(key="rs_rv_weights").set_value("PCA").run()
    _ok(at)
    assert not at.error
    caption = next(c.value for c in at.caption if c.value.startswith("2s5s10s (PCA)"))
    assert "5Y +2.000" in caption
    ranking = next(d.value for d in at.dataframe if "STRUCTURE" in d.value.columns)
    assert "2s5s10s" in set(ranking["STRUCTURE"]) and "CARRY / VOL" in ranking.columns
    at.radio(key="rs_rv_sort").set_value("CARRY / VOL").run()
    _ok(at)
    at.selectbox(key="rs_rv_structure").set_value("CUSTOM").run()
    at.text_input(key="rs_rv_custom").input("2Y:-1, 4Y:1").run()
    _ok(at)
    assert any("not quoted in this history" in e.value for e in at.error)
    at.text_input(key="rs_rv_custom").input("2Y:-1 5Y:2").run()
    _ok(at)
    assert any("separate legs with commas" in e.value for e in at.error)
    at.text_input(key="rs_rv_custom").input("5Y:-1, 30Y:1").run()
    _ok(at)
    assert not at.error and _metric(at, "LATEST (BP)") == "54.0"


def test_rv_on_a_history_that_is_daily_only_lately_says_so():
    at = _app()
    at.selectbox(key="rs_rv_dataset").set_value("EUR_AAA_ECB").run()
    _ok(at)
    assert not at.error
    assert any("holds only" in w.value for w in at.warning)  # the 1Y window covers about two months
    ranking = next(d.value for d in at.dataframe if "STRUCTURE" in d.value.columns)
    assert ranking["CARRY+ROLL LONG, 3M (BP)"].isna().all() if "CARRY+ROLL LONG, 3M (BP)" in ranking else True
    assert any("US Treasury history only" in c.value for c in at.caption)


def test_carry_views_match_their_tables():
    at = _app()
    chart = _charts(at)["rs_carry_chart"]
    table = next(d.value for d in at.dataframe if "CARRY (BP)" in d.value.columns and "BOND" in d.value.columns)
    assert _values(chart["data"][0]["y"]) == pytest.approx(list(table["CARRY (BP)"]))
    assert _values(chart["data"][2]["y"]) == pytest.approx(list(table["CARRY+ROLL (BP)"]))
    assert any("SOFR fixing for 2026-10-16" in c.value for c in at.caption)
    at.radio(key="rs_carry_kind").set_value("SWAPS (RECEIVE FIXED)").run()
    _ok(at)
    swaps = next(d.value for d in at.dataframe if "SPOT PAR (%)" in d.value.columns)
    assert list(swaps["TENOR"]) == ["2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
    assert swaps.set_index("TENOR").loc["10Y", "SPOT PAR (%)"] == pytest.approx(3.73)
    at.radio(key="rs_carry_horizon").set_value("6M").run()
    _ok(at)
    assert any(c.value.startswith("Receive fixed") and "(6M)" in c.value for c in at.caption)


def test_pca_inputs_and_factor_risk():
    at = _app()
    risk = next(d.value for d in at.dataframe if "POSITION" in d.value.columns and "PC1 +1 SD ($)" in d.value.columns)
    assert list(risk["POSITION"])[-1] == "TOTAL" and risk["PC1 +1 SD ($)"].iloc[0] < 0  # a receiver loses
    at.radio(key="rs_pca_heat_factor").set_value("PC2").run()
    _ok(at)
    at.multiselect(key="rs_pca_tenors_UST_PAR").set_value(["2Y", "10Y"]).run()
    _ok(at)
    assert any("at least three tenors" in w.value for w in at.warning)
    at.multiselect(key="rs_pca_tenors_UST_PAR").set_value(["2Y", "5Y", "10Y", "30Y"]).run()
    at.date_input(key=PCA_START).set_value(dt.date(2026, 6, 1)).run()  # ~100 changes < 252
    _ok(at)
    assert not at.error
    assert any("longer than the" in w.value for w in at.warning)
    at.date_input(key=PCA_START).set_value(dt.date(2026, 10, 1)).run()  # too few for a PCA
    _ok(at)
    assert any("Cannot run the PCA" in e.value for e in at.error)


def test_var_full_revaluation_and_lookback():
    at = _app()
    hist = _number(_metric(at, "HIST VAR 99% ($)"))
    at.button(key="rs_var_full").click().run()
    _ok(at)
    full = _number(_metric(at, "FULL-REVAL VAR ($)"))
    assert full == pytest.approx(hist, rel=0.02)
    table = next(d.value for d in at.dataframe if "FULL-REVAL VAR ($)" in d.value.columns)
    assert list(table["POSITION"])[-1] == "TOTAL"
    at.radio(key="rs_var_lookback").set_value(250).run()
    at.radio(key="rs_var_conf").set_value(0.975).run()
    _ok(at)
    assert _metric(at, "FULL-REVAL VAR ($)") == "not run"  # a new window needs a new run
    assert any(m.label == "HIST VAR 97.5% ($)" for m in at.metric)


def test_swap_spreads_need_real_swap_history():
    at = _app()
    assert any(c.value.startswith("No USD swap history with daily data") for c in at.caption)
    import pandas as pd

    from derivx.data import rates_history as rh

    ust = rh.bundled_store().complete("UST_PAR")
    days = [d for d in ust.index if d >= dt.date(2025, 6, 2)]
    wide = pd.DataFrame({"date": [d.isoformat() for d in days], "2Y": 100 * ust.loc[days, "2Y"].to_numpy() + 0.3,
                         "10Y": 100 * ust.loc[days, "10Y"].to_numpy() - 0.2})  # fmt: skip
    frame, _, _ = rh.parse_history_upload(wide, "USD", "percent", "swaps.csv")
    at = _app(rates_user_history={"USD": {"name": "swaps.csv", "sha": "abc", "frame": frame}})
    table = next(d.value for d in at.dataframe if "SWAP HISTORY" in d.value.columns)
    assert list(table["TENOR"]) == ["2Y", "10Y"]
    assert list(table["LATEST (BP)"]) == pytest.approx([30.0, -20.0])
    assert "rs_rv_spreads_csv" in _downloads(at)


def test_no_positions():
    at = _app(no_positions=True)
    assert not at.error
    assert any("No positions" in i.value for i in at.info)


def test_research_tab_and_pca_shock_in_the_full_app():
    at = AppTest.from_file(str(FULL_APP), default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    _ok(at)
    assert not at.error, [e.value for e in at.error]
    assert {"rs_rv_chart", "rs_var_hist", "rs_pca_heatmap"} <= set(_charts(at))
    at.radio(key="rs_carry_kind").set_value("SWAPS (RECEIVE FIXED)").run()
    _ok(at)
    assert any("ILLUSTRATIVE quotes" in w.value for w in at.warning)  # USD swap quotes are not market data
    at.selectbox(key="scn_type").set_value("PCA factor").run()
    _ok(at)
    assert not at.error, [e.value for e in at.error]
    assert any(c.value.startswith("PC1 level +2 sd:") for c in at.caption)
    sources = [d.value for d in at.dataframe if "FLAG" in d.value.columns and "DATA" in d.value.columns]
    assert sources and "US Treasury par yield curve" in set(sources[0]["DATA"])
    at.selectbox(key="scn_pca_factor").set_value(2).run()
    at.number_input(key="scn_pca_days").set_value(10).run()
    _ok(at)
    assert any(c.value.startswith("PC2 slope +2 sd 10d:") for c in at.caption)


def test_pca_dates_follow_the_valuation_date_and_the_tenors():
    """Review finding: the FROM/TO dates were kept by key alone, so a window
    picked for one valuation date or tenor set stuck: after 2026-06-30 the
    PCA still ended there on 2026-10-19, and after picking the 2026-only 6W
    bill the start stayed in 2026 once the usual tenors were back."""

    at = AppTest.from_string(DATED_APP, default_timeout=300)
    at.session_state["asof"] = dt.date(2026, 6, 30)
    at.run()
    _ok(at)
    assert "2015-01-02 to 2026-06-30" in _pca_caption(at)
    at.session_state["asof"] = dt.date(2026, 10, 19)
    at.run()
    _ok(at)
    assert "2015-01-02 to 2026-10-19 (2950 changes" in _pca_caption(at)
    at.multiselect(key="rs_pca_tenors_UST_PAR").set_value(["6W", "2M", "4M"]).run()
    _ok(at)
    assert "2026-01-05 to 2026-10-19" in _pca_caption(at)  # the 6W bill is quoted from 2026 only
    at.multiselect(key="rs_pca_tenors_UST_PAR").set_value(["3M", "1Y", "2Y", "5Y", "10Y", "30Y"]).run()
    _ok(at)
    assert "2015-01-02 to 2026-10-19 (2950 changes" in _pca_caption(at)


def test_var_follows_the_store_content_and_says_when_the_window_is_short():
    """Review finding: the VaR scenarios were cached on the store's label and
    row count, so a store with revised values (same rows) kept the old VaR.
    A valuation date early in the daily history gets fewer days than asked
    for, and the tab says so."""

    from derivx.data import rates_history as rh

    base = rh.bundled_store()
    at = AppTest.from_string(DATED_APP, default_timeout=300)
    at.session_state["rates_store"] = base
    at.run()
    _ok(at)
    before = _number(_metric(at, "HIST VAR 99% ($)"))
    revised = base.frame.copy()
    recent = (revised["dataset"] == "UST_PAR") & (revised["asof"] >= dt.date(2025, 1, 1))
    revised.loc[recent, "value_decimal"] *= 1.2  # same rows, other values
    other = rh.build_store([revised], base.label)
    assert len(other.frame) == len(base.frame)
    at.session_state["rates_store"] = other
    at.run()
    _ok(at)
    assert _number(_metric(at, "HIST VAR 99% ($)")) != pytest.approx(before, rel=1e-3)
    early = AppTest.from_string(DATED_APP, default_timeout=300)
    early.session_state["asof"] = dt.date(2015, 6, 30)
    early.run()
    _ok(early)
    early.radio(key="rs_var_lookback").set_value(250).run()
    early.radio(key="rs_var_conf").set_value(0.95).run()
    _ok(early)
    short = "holds only 125 daily changes (from 2014-12-31), not the 250 asked for"
    assert any(short in w.value for w in early.warning)
    assert _metric(early, "SCENARIO DAYS") == "125"


BUCKET_APP = """
import streamlit as st
from derivx.rates.capfloor import CapFloor, capfloor_position
from derivx.rates.examples import example_market
from derivx.rates.swaptions import Swaption, swaption_position
from derivx.ui import rates_book, rates_research

market = example_market()
book = rates_book.render(market)
swaption = swaption_position(Swaption(1e7, "2Y", "10Y", 0.037, "payer", 0.0095), market, "TAB SWAPTION")
cap = capfloor_position(CapFloor(1e7, "cap", 0.04, "5Y", 0.0090, calendar="USD"), asof=market.asof, name="TAB CAP")
rates_research.render(market, book + [swaption, cap])
"""


def test_risk_report_has_vega_buckets(monkeypatch):
    """Review finding (R4, A2): the RESEARCH risk report had one vega per
    position. It now carries each option's vega by bucket, as the BOOK tab
    does: book swaptions priced on the SABR cube by the cube's ATM points
    they move, other swaptions by option expiry x swap tenor (allocated),
    caps by caplet expiry; buckets add up to the position's normal vega,
    with a TOTAL per currency and bucket."""

    from derivx.rates import portfolio as pf
    from derivx.rates.examples import example_market
    from derivx.ui import rates_exports

    seen = {}
    original = rates_exports.export_row

    def spy(kind, market, tables, inputs, model, key, store=None):
        seen[kind] = tables
        return original(kind, market, tables, inputs, model, key, store)

    monkeypatch.setattr(rates_exports, "export_row", spy)
    at = AppTest.from_string(BUCKET_APP, default_timeout=600)
    at.run()
    _ok(at)
    report = seen["value at risk"]["risk_report"]
    buckets = report[report["section"] == "vega_bucket"]
    assert set(buckets["currency"]) == {"USD"} and (report["currency"] != "").all()
    market = example_market()
    from derivx.rates.capfloor import CapFloor
    from derivx.rates.swaptions import Swaption

    swaption = Swaption(1e7, "2Y", "10Y", 0.037, "payer", 0.0095).pinned(market)
    flat = buckets[buckets["position"] == "TAB SWAPTION"]
    assert set(flat["unit"]) == {"P&L per +1bp normal vol (allocated)"}
    assert set(flat["item"]) == {"2Y x 10Y"}  # on the grid: all of its vega in one bucket
    assert flat["value"].sum() == pytest.approx(swaption.greeks(market)["vega_normal_bp"], rel=1e-9)
    cap = buckets[buckets["position"] == "TAB CAP"]
    cf = CapFloor(1e7, "cap", 0.04, "5Y", 0.0090, calendar="USD").pinned(market.asof)
    expected = pf.cap_normal_vegas(cf, market)
    assert cap["value"].sum() == pytest.approx(expected["vega"].sum(), rel=1e-9)
    assert all(item.endswith("x CAP 3M") for item in cap["item"]) and len(cap) > 3  # by caplet expiry
    cube = buckets[buckets["position"] == "US9"]
    assert set(cube["unit"]) == {"P&L per +1bp normal vol (cube)"} and set(cube["item"]) == {"1Y x 10Y"}
    vega = report[(report["section"] == "vega") & (report["position"] == "US9")]["value"].iloc[0]
    assert cube["value"].sum() == pytest.approx(vega, rel=0.02)  # ATM-point bumps: second-order terms aside
    totals = buckets[buckets["position"] == "TOTAL"].set_index("item")
    trades = buckets[buckets["position"] != "TOTAL"]
    for item, value in trades.groupby("item")["value"].sum().items():
        assert totals.loc[item, "value"] == pytest.approx(value, rel=1e-12, abs=1e-9)
    assert {"US1", "US6"} - set(buckets["position"]) == {"US1", "US6"}  # swaps and bonds have no vega
    assert any("vega by bucket" in c.value for c in at.caption)
