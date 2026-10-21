"""The multi-curve Rates page: EURIBOR projection-curve editors with
illustrative quotes, forward and basis charts, EURIBOR swaps and caps with
per-curve key-rate ladders and hedges, and per-curve scenarios."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from derivx.rates.curves import CurveQuote, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, sofr_example_rows
from derivx.rates.multicurve import bootstrap_projection, illustrative_quotes
from derivx.ui import rates_curve

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "derivx" / "ui" / "streamlit_app.py"


def _assert_bounded(at: AppTest) -> None:
    for chart in at.get("plotly_chart"):
        layout = json.loads(chart.proto.spec)["layout"]
        limits = layout["meta"]["zoom"]["axes"]
        axes = [name for name in layout if name.startswith(("xaxis", "yaxis"))]
        assert axes and set(axes) <= set(limits), (chart.proto.id, axes)
        for name in axes:
            assert layout[name]["minallowed"] == limits[name]["min"], (chart.proto.id, name)
            assert layout[name]["maxallowed"] == limits[name]["max"], (chart.proto.id, name)


def _chart(at: AppTest, key: str) -> dict:
    return json.loads(next(c for c in at.get("plotly_chart") if c.proto.id.endswith(key)).proto.spec)


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


@pytest.fixture(scope="module")
def eur() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    at.selectbox(key="rates_conv").set_value("EUR ESTR").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_curve_tab_builds_euribor_projection_curves_from_labelled_quotes(eur):
    assert not eur.error, [e.value for e in eur.error]
    tables = {d.key: d.value for d in eur.dataframe if d.key and d.key.startswith("rates_proj_")}
    assert len(tables) == 2
    six = next(v for k, v in tables.items() if "EURIBOR_6M" in k)
    three = next(v for k, v in tables.items() if "EURIBOR_3M" in k)
    assert list(six["instrument"].unique()) == ["DEPO", "FRA", "IRS"]
    assert {"DEPO", "FUT", "BASIS"} == set(three["instrument"])
    for table in (six, three):
        assert table["source"].str.startswith("ILLUSTRATIVE").all()  # never shown as market data
    futures = three[three["instrument"] == "FUT"]
    assert futures["rate"].between(90.0, 101.0).all()  # quoted as prices
    errors = [m.value for m in eur.metric if m.label == "MAX REPRICING ERROR"]
    assert len(errors) == 3 and all(float(v.split()[0]) < 1e-6 for v in errors)
    forwards, basis = _chart(eur, "rates_mc_forwards"), _chart(eur, "rates_mc_basis")
    names = [t["name"] for t in forwards["data"]]
    assert {"OIS 3M FWD", "OIS 6M FWD", "EURIBOR 3M FWD", "EURIBOR 6M FWD"} <= set(names)
    assert any("BASIS QUOTES" in t["name"] for t in basis["data"])
    badge = next(a["text"] for a in forwards["layout"]["annotations"] if a["text"].startswith("SOURCE:"))
    assert "ILLUSTRATIVE" in badge
    _assert_bounded(eur)


def test_euribor_swap_has_per_curve_ladders_and_a_multi_curve_hedge(eur):
    eur.radio(key="swap_kind").set_value("EURIBOR 6M").run()
    assert not eur.exception and not eur.error, [e.value for e in eur.error]
    assert "EURIBOR 6M" in eur.session_state["swap_kind"]
    ladder_charts = [c.proto.id for c in eur.get("plotly_chart") if "swap_ladder_chart" in c.proto.id]
    assert any(i.endswith("swap_ladder_chart_euribor_6m") for i in ladder_charts)
    assert float(_metric(eur, "LADDER RISK REMOVED (%)")) > 99.0
    hedge = next(d.value for d in eur.dataframe if "CURVE" in d.value.columns and "BENCHMARK" in d.value.columns)
    used = hedge[hedge["NOTIONAL (€)"] > 0]
    assert set(used["CURVE"]) == {"EURIBOR 6M (PROJECTION)"}  # a par swap is its own curve's 5Y quote
    pv01 = float(_metric(eur, "PV01 (€, +1BP)").replace(",", ""))
    assert pv01 < 0  # receiver
    # Values are in euros, and the hedge's PV01 is its P&L for +1bp on every rate quote, as the swap's.
    swap_labels = [m.label for m in eur.metric if m.label.startswith(("PV ", "PV01", "ANNUITY", "SWAP ", "HEDGED "))]
    assert len(swap_labels) >= 5 and not any("$" in label for label in swap_labels), swap_labels
    swap_pv01 = float(_metric(eur, "SWAP PV01 (€)").replace(",", ""))
    assert swap_pv01 == pytest.approx(pv01, rel=1e-6, abs=0.01)
    _assert_bounded(eur)


def test_euribor_cap_and_basis_scenarios(eur):
    eur.radio(key="cap_index").set_value("EURIBOR 3M").run()
    assert not eur.exception and not eur.error, [e.value for e in eur.error]
    caps = [c.proto.id for c in eur.get("plotly_chart") if "cap_ladder_chart" in c.proto.id]
    assert len(caps) == 3  # €STR, EURIBOR 6M (the 3M curve rides on it) and EURIBOR 3M
    eur.selectbox(key="scn_type").set_value("Tenor basis").run()
    assert not eur.exception and not eur.error, [e.value for e in eur.error]
    by_curve = next(d.value for d in eur.dataframe if "POSITION" in d.value.columns
                    and any("EURIBOR 3M" in c for c in d.value.columns))  # fmt: skip
    assert len(by_curve) >= 2
    eur.selectbox(key="scn_type").set_value("IBOR-OIS basis").run()
    assert not eur.exception and not eur.error
    grid = next(d.value for d in eur.dataframe if "SCENARIO" in d.value.columns)
    assert any("IBOR-OIS basis" in s for s in grid["SCENARIO"]) and any("Tenor basis" in s for s in grid["SCENARIO"])
    eur.radio(key="scn_chart_curve").set_value("EURIBOR 6M").run()
    assert not eur.exception
    _assert_bounded(eur)


def test_single_curve_without_projection_curves(eur):
    eur.checkbox(key="rates_mc_on").uncheck().run()
    assert not eur.exception and not eur.error
    assert eur.radio(key="swap_kind").options == ["OIS", "IRS"]
    assert not any(c.proto.id.endswith("rates_mc_forwards") for c in eur.get("plotly_chart"))
    eur.checkbox(key="rates_mc_on").check().run()
    assert not eur.exception


def test_sofr_futures_build_into_the_cached_curve():
    """The CURVE tab's cached builder takes SR1/SR3 quotes with Hull-White
    convexity and the store's SOFR fixings for part-fixed contracts."""

    from derivx.data.rates_history import bundled_store

    rates = {r["tenor"]: r["rate"] for r in sofr_example_rows()}
    quotes = (("DEPO", "1W", rates["1W"]), ("SR1", "OCT26", 0.0390), ("SR3", "DEC26", 0.0368),
              ("SR3", "MAR27", 0.0355), ("OIS", "2Y", rates["2Y"]), ("OIS", "10Y", rates["10Y"]))  # fmt: skip
    fixings = rates_curve._ois_fixings(bundled_store(), "USD", EXAMPLE_ASOF)
    assert fixings and fixings[-1][0] < EXAMPLE_ASOF and all(dt.date(2026, 6, 1) < d for d, _ in fixings)
    curve = rates_curve._build(EXAMPLE_ASOF, quotes, "USD SOFR", "loglinear", (0.03, 0.0085), fixings)
    rows = curve.repricing()
    assert max(abs(r["error_bp"]) for r in rows) < 1e-8
    assert next(r for r in rows if r["tenor"] == "MAR27")["convexity_bp"] > 0.0
    table = rates_curve._repricing_table(curve)
    assert {"Price", "Convexity (bp)"} <= set(table.columns)


def test_illustrative_prefill_is_consistent_with_the_ois_curve():
    quotes = [CurveQuote(r["kind"], r["tenor"], r["rate"] - 0.012) for r in sofr_example_rows()]
    ois = bootstrap(EXAMPLE_ASOF, quotes, "EUR ESTR")
    six = bootstrap_projection(EXAMPLE_ASOF, illustrative_quotes(ois, "EURIBOR 6M"), "EURIBOR 6M", ois)
    three = illustrative_quotes(ois, "EURIBOR 3M", basis_curve=six)
    assert {q.kind for q in three} == {"DEPO", "FUT", "BASIS"}
    assert all(0.0 < q.rate < 0.002 for q in three if q.kind == "BASIS")  # 3s6s spread positive, a few bp


def _eur_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    at.selectbox(key="rates_conv").set_value("EUR ESTR").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _editors(at: AppTest) -> dict:
    return {d.key: d.value for d in at.dataframe if d.key and d.key.startswith("rates_proj_")}


def _futures_table(at: AppTest):
    tables = (d.value for d in at.dataframe if "Convexity (bp)" in d.value.columns)
    return next(t for t in tables if "FUT" in set(t["Instrument"]))


def test_interpolation_and_convexity_inputs_keep_the_euribor_tables():
    """The EURIBOR editors (and so the user's edits) survive a change of
    interpolation or of the Hull-White inputs: the illustrative quotes are
    inputs, not functions of those settings. The convexity inputs move the
    3M curve under fixed futures prices (they used to re-price the
    illustrative futures, so the 3M forwards never moved)."""

    at = _eur_app()
    before = _editors(at)
    futures = _futures_table(at)
    at.number_input(key="rates_hw_sigma").set_value(150.0).run()
    assert not at.exception and not at.error
    assert _editors(at).keys() == before.keys()
    moved = _futures_table(at)
    fut = futures["Instrument"] == "FUT"
    assert list(moved["Price"][fut]) == list(futures["Price"][fut])  # the quotes stay
    assert (moved["Convexity (bp)"][fut] > futures["Convexity (bp)"][fut]).all()  # more vol, more convexity
    at.selectbox(key="rates_interp").set_value("linear_zero").run()
    assert not at.exception and not at.error
    assert _editors(at).keys() == before.keys()
    for key, table in _editors(at).items():
        assert table.equals(before[key])


def test_eur_labels_are_in_euros_and_ladder_sums_do_not_mix_basis_with_rates():
    """EUR trades are valued in euros; the cap caption compares the rate-quote
    buckets with the PV01 (basis spreads held) and reports the 3s6s spread
    buckets on their own; a carry horizon past a futures fixing revalues a
    EURIBOR 3M book; a book with a dollar bond is totalled per currency and
    converted to the reporting currency (review finding R2 / A1: the totals
    used to add euros to dollars)."""

    at = _eur_app()
    at.radio(key="swap_kind").set_value("EURIBOR 3M").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert {"PV (€)", "PV01 (€, +1BP)", "ANNUITY (€ PER BP)"} <= set(labels)
    hedge = next(d.value for d in at.dataframe if "CURVE" in d.value.columns and "BENCHMARK" in d.value.columns)
    assert {"PV01 (€, +1BP)", "BASIS PV01 (€, +1BP)", "PAR SPREAD (BP)"} <= set(hedge.columns)
    swap_caption = next(c.value for c in at.caption if "rate-quote buckets sum to" in c.value)
    assert "basis-spread buckets" in swap_caption
    ladder = json.loads(next(c for c in at.get("plotly_chart") if c.proto.id.endswith("swap_ladder_chart")).proto.spec)
    assert ladder["layout"]["yaxis"]["title"]["text"] == "PV01 (€ PER +1BP)"

    at.radio(key="cap_index").set_value("EURIBOR 3M").run()
    assert not at.exception and not at.error
    cap_caption = next(c.value for c in at.caption if "rate-quote buckets sum to" in c.value and "PV01 of" in c.value)
    text = cap_caption.split("The rate-quote buckets sum to ")[1]
    rate_sum = float(text.split(" against a PV01 of ")[0].replace(",", ""))
    pv01 = float(text.split(" against a PV01 of ")[1].split(";")[0].rstrip(".").replace(",", ""))
    assert rate_sum == pytest.approx(pv01, rel=1e-4, abs=0.05)
    assert "basis-spread buckets" in cap_caption

    at.sidebar.radio(key="rates_risk_scope").set_value("TAB TRADES").run()  # the tabs' swap, cap and bond
    at.number_input(key="scn_horizon").set_value(90).run()  # past the DEC26 futures fixing of the 3M curve
    assert not at.exception and not at.error, [e.value for e in at.error]
    # EUR swaps and caps with a USD bond: a total per currency, and the book in USD at the ECB rate
    from derivx.risk.reporting import default_reporting

    eurusd = default_reporting(at.session_state["rates_asof"]).fx.quote("EURUSD")
    explain = next(d.value for d in at.dataframe if "TOTAL P&L" in d.value.columns)
    rows = explain.set_index("POSITION")
    assert {"TOTAL EUR", "TOTAL USD", "ALL IN USD"} <= set(rows.index)
    assert set(rows.loc[[i for i in rows.index if not str(i).startswith(("TOTAL", "ALL IN"))], "CCY"]) == {"EUR", "USD"}
    for column in ("TOTAL P&L", "DELTA", "CARRY", "RESIDUAL"):
        expected = rows.loc["TOTAL USD", column] + rows.loc["TOTAL EUR", column] * eurusd
        assert rows.loc["ALL IN USD", column] == pytest.approx(expected, rel=1e-10, abs=1e-6)
    assert float(_metric(at, "TOTAL P&L (IN $)").replace(",", "")) == pytest.approx(
        rows.loc["ALL IN USD", "TOTAL P&L"], abs=0.006
    )
    assert not any("without FX conversion" in w.value for w in at.warning)
    _assert_bounded(at)

    # EURUSD +10% on SCENARIOS: no local amount moves; the converted total gains the euro positions'
    # translation, shown as the explain's FX column (ALL IN row only) and an FX bar of the waterfall.
    at.number_input(key="scn_fx_eurusd").set_value(10.0).run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    shocked = next(d.value for d in at.dataframe if "TOTAL P&L" in d.value.columns).set_index("POSITION")
    fx = next(c for c in shocked.columns if str(c).startswith("FX"))
    for ccy in ("TOTAL EUR", "TOTAL USD"):
        assert shocked.loc[ccy, "TOTAL P&L"] == pytest.approx(rows.loc[ccy, "TOTAL P&L"], rel=1e-12, abs=1e-9)
        assert math.isnan(shocked.loc[ccy, fx])
    translation = shocked.loc["ALL IN USD", fx]
    assert abs(translation) > 1.0
    assert shocked.loc["ALL IN USD", "TOTAL P&L"] == pytest.approx(
        rows.loc["TOTAL USD", "TOTAL P&L"] + rows.loc["TOTAL EUR", "TOTAL P&L"] * eurusd + translation, rel=1e-10
    )
    assert float(_metric(at, "TOTAL P&L (IN $)").replace(",", "")) == pytest.approx(
        shocked.loc["ALL IN USD", "TOTAL P&L"], abs=0.006
    )
    assert "FX" in _chart(at, "scn_waterfall")["data"][0]["x"]
    assert any("FX = the FX shock's translation" in c.value for c in at.caption)


def test_estr_compounded_cap_on_the_eur_curve_set(eur):
    """On the EUR curve set the CAP-FLOOR tab also prices caps on €STR
    compounded in arrears: forwards and discounting on the €STR curve (one
    key-rate ladder), paid one business day after each period, as €STR OIS."""

    eur.radio(key="cap_index").set_value("€STR compounded").run()
    assert not eur.exception and not eur.error, [e.value for e in eur.error]
    caps = [c.proto.id for c in eur.get("plotly_chart") if "cap_ladder_chart" in c.proto.id]
    assert len(caps) == 1  # the €STR curve only: no EURIBOR projection
    schedule = next(c.value for c in eur.caption if c.value.startswith("Schedule:"))
    assert "€STR compounded in arrears" in schedule and "paid 1 business day after" in schedule
    assert not any("Term-rate approximation" in w.value for w in eur.warning)
    caplets = next(d.value for d in eur.dataframe if "OPTION TIME (Y)" in d.value.columns)
    assert (caplets["FIRST FIXING"] < caplets["LAST FIXING"]).all()
    assert (caplets["OPTION TIME (Y)"] < caplets["EXPIRY (Y)"]).all()  # Lyashenko-Mercurio: vol decays in-period
    eur.radio(key="cap_index").set_value("EURIBOR 6M").run()
    assert not eur.exception


def test_reporting_currency_sidebar_converts_every_tab_and_survives_a_mode_switch():
    """The sidebar REPORTING CURRENCY (A1) drives the converted totals of
    the SCENARIOS explain on EUR swaps and caps with a dollar bond, states
    the ECB rate it used, and keeps its value across EQUITY and back."""

    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    at.selectbox(key="rates_conv").set_value("EUR ESTR").run()
    at.sidebar.radio(key="rates_risk_scope").set_value("TAB TRADES").run()
    assert not at.exception, [e.value for e in at.exception]
    report = at.sidebar.selectbox(key="rates_report_ccy")
    assert report.value == "USD" and list(report.options) == ["USD", "EUR", "GBP"]
    caption = next(c.value for c in at.sidebar.caption if c.value.startswith("EURUSD"))
    assert "ECB euro reference rates" in caption and str(at.session_state["rates_asof"]) in caption
    report.set_value("EUR").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    explain = next(d.value for d in at.dataframe if "TOTAL P&L" in d.value.columns)
    assert "ALL IN EUR" in set(explain["POSITION"]) and "ALL IN USD" not in set(explain["POSITION"])
    assert any(m.label == "TOTAL P&L (IN €)" for m in at.metric)
    at.sidebar.radio[0].set_value("[ EQUITY ]").run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.selectbox(key="rates_report_ccy").value == "EUR"
    assert any(m.label == "TOTAL P&L (IN €)" for m in at.metric)
