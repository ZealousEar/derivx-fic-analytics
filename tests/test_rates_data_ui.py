"""Rates page market data in the app: DATA SOURCE modes, quote prefills
with per-row provenance, source/as-of badges on curve charts, uploads that
survive leaving the page, the live fallback and historical scenarios with
per-curve source flags."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from derivx.data import rates_history as rh
from derivx.data import rates_sources as rs
from derivx.data.cache import DiskCache
from derivx.rates.examples import EXAMPLE_ASOF
from derivx.ui import rates_data

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "derivx" / "ui" / "streamlit_app.py"
FIXTURES = ROOT / "tests" / "fixtures" / "rates"
FLAT = b"instrument,tenor,rate\nOIS,1Y,3\nOIS,5Y,3\nOIS,10Y,3\n"


def _rates() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    assert not at.exception, at.exception
    return at


def _badge(at: AppTest, key: str) -> str:
    chart = next(c for c in at.get("plotly_chart") if c.proto.id.endswith(key))
    texts = [a["text"] for a in json.loads(chart.proto.spec)["layout"].get("annotations", [])]
    return next(t for t in texts if t.startswith("SOURCE:")).replace("<br>", " ")  # badges wrap long sources


def _quotes(at: AppTest) -> pd.DataFrame:
    return next(d for d in at.dataframe if d.key and d.key.startswith("rates_quotes_")).value


def _uploader(at: AppTest, label: str):
    """An upload widget by label (its key changes each time it is cleared)."""

    return next(f for f in at.file_uploader if f.label.startswith(label))


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _usd_default(at: AppTest) -> None:
    assert at.radio(key="rates_source").value == "BUNDLED SNAPSHOT"
    asof = at.selectbox(key="rates_asof")
    store = rh.bundled_store()
    assert asof.value == EXAMPLE_ASOF and len(asof.options) == len(store.dates("UST_PAR"))
    quotes = _quotes(at)
    assert list(quotes.columns) == ["instrument", "tenor", "rate", "source"]
    assert quotes.iloc[0].tolist() == ["DEPO", "ON", 3.88, "SOFR (NY Fed) fixing for 2026-10-16"]
    assert set(quotes["source"].iloc[1:]) == {rates_data.ILLUSTRATIVE}
    assert any("ILLUSTRATIVE quotes (not market data)" in c.value for c in at.caption)
    swap = _badge(at, "rates_curve_chart")
    assert swap == (
        "SOURCE: USD SOFR | ILLUSTRATIVE quotes (not market data) + SOFR (NY Fed) fixing for 2026-10-16 | "
        "AS OF 2026-10-19"
    )
    assert _badge(at, "rates_treasury_chart") == (
        "SOURCE: US Treasury par yields (US Treasury, bundled snapshot) | AS OF 2026-10-19"
    )
    # The scenario tab's curve chart carries the same badge for the curve it shows.
    assert _badge(at, "scn_curve_chart") == swap
    assert not at.error


def _gbp_from_the_bank_of_england(at: AppTest) -> None:
    at.selectbox(key="rates_conv").set_value("GBP SONIA").run()
    assert not at.exception and not at.error
    quotes = _quotes(at)
    boe = rh.bundled_store().ois_par("GBP_OIS_BOE", EXAMPLE_ASOF)
    assert quotes.iloc[0]["tenor"] == "ON" and quotes.iloc[0]["source"].startswith("SONIA (Bank of England via FRED)")
    by_tenor = quotes.set_index("tenor")
    assert by_tenor.loc["10Y", "rate"] == pytest.approx(100 * boe["10Y"], abs=1e-6)
    assert by_tenor.loc["10Y", "source"] == "Bank of England UK OIS (SONIA) curve 2026-10-19, par OIS derived"
    assert any(c.value.startswith("Real market data") for c in at.caption)
    assert "Bank of England UK OIS (SONIA) curve 2026-10-19" in _badge(at, "rates_curve_chart")
    zeros = rh.bundled_store().curve("GBP_OIS_BOE", EXAMPLE_ASOF, "zero_cc")
    assert float(_metric(at, "10Y ZERO").rstrip("%")) == pytest.approx(100 * zeros["10Y"], abs=2e-3)


def _eur_proxy_and_old_dates(at: AppTest) -> None:
    at.selectbox(key="rates_conv").set_value("EUR ESTR").run()
    quotes = _quotes(at)
    assert quotes.iloc[0]["source"].startswith("ESTR (ECB) fixing for")
    assert all(s.startswith("PROXY: ECB euro area AAA government curve") for s in quotes["source"].iloc[1:])
    assert any("labelled PROXY" in c.value for c in at.caption)
    # Month ends back to 2008 come from the bundled sample: a 2013 EUR proxy from that month end's ECB curve.
    at.selectbox(key="rates_asof").set_value(dt.date(2013, 5, 31)).run()
    assert not at.exception and not at.error
    assert _quotes(at).iloc[0]["source"].startswith("PROXY: ECB euro area AAA government curve 2013-05-31")
    assert _badge(at, "rates_treasury_chart").endswith("AS OF 2013-05-31")
    assert "FRED, bundled snapshot" in _badge(at, "rates_treasury_chart")
    # Near-zero 2013 sterling rates: the prefill is in percent, so nothing is misread as decimals.
    at.selectbox(key="rates_conv").set_value("GBP SONIA").run()
    assert not at.exception and not at.error
    assert _quotes(at).iloc[0]["rate"] < 1.0 and at.radio(key="rates_units").value == "percent"
    # Before the BoE OIS archive starts (2009) the page says so and falls back to the illustrative set.
    at.selectbox(key="rates_asof").set_value(dt.date(2008, 1, 31)).run()
    assert not at.exception and not at.error
    expected = "No spot (zero) rates of the Bank of England UK OIS (SONIA) curve on or before 2008-01-31"
    assert any(expected in w.value for w in at.warning)
    assert set(_quotes(at)["source"]) == {rates_data.ILLUSTRATIVE}


def _uploads_survive_and_clear(at: AppTest) -> None:
    at.selectbox(key="rates_conv").set_value("GBP SONIA").run()
    at.radio(key="rates_source").set_value("UPLOAD").run()
    assert any("no quotes file yet" in c.value for c in at.caption)
    at.file_uploader(key="rates_upload_0").set_value(("flat3.csv", FLAT, "text/csv")).run()
    assert not at.exception and not at.error
    assert any("Using uploaded file flat3.csv (3 quotes)" in c.value for c in at.sidebar.caption)
    flat = _metric(at, "10Y ZERO")
    assert set(_quotes(at)["source"]) == {"uploaded: flat3.csv"}
    assert "uploaded file flat3.csv" in _badge(at, "rates_curve_chart")
    # Another page and back: the sidebar choices and the parsed upload are kept.
    at.sidebar.radio[0].set_value("[ EQUITY ]").run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    assert not at.exception and not at.error  # every tab's inputs come back too (the bond price included)
    assert at.radio(key="rates_source").value == "UPLOAD" and at.selectbox(key="rates_conv").value == "GBP SONIA"
    assert any("Using uploaded file flat3.csv" in c.value for c in at.sidebar.caption)
    assert _metric(at, "10Y ZERO") == flat
    at.button(key="rates_upload_clear").click().run()
    assert not any("Using uploaded file" in c.value for c in at.sidebar.caption)
    assert _quotes(at).iloc[1]["source"].startswith("Bank of England")
    assert at.file_uploader(key="rates_upload_1").value is None  # the widget comes back empty
    at.run()
    assert not any("Using uploaded file" in c.value for c in at.sidebar.caption)


def _bad_uploads_are_reported(at: AppTest) -> None:
    at.radio(key="rates_source").set_value("UPLOAD").run()
    _uploader(at, "SWAP QUOTES").set_value(("bad.csv", b"tenor,level\n1Y,3\n", "text/csv")).run()
    assert not at.exception and any("bad.csv: missing columns: instrument, rate" in e.value for e in at.error)
    _uploader(at, "TREASURY PAR CURVE").set_value(("t.csv", b"a,b\n1,2\n", "text/csv")).run()
    assert any("t.csv" in e.value and "Treasury" in e.value for e in at.error)
    # The last good Treasury upload stays in use, and the page says which one.
    assert any("Using uploaded Treasury curve treasury.csv" in c.value for c in at.sidebar.caption)


def _uploaded_treasury_wins(at: AppTest) -> None:
    at.radio(key="rates_source").set_value("UPLOAD").run()
    payload = (FIXTURES / "treasury_2026.csv").read_bytes()
    _uploader(at, "TREASURY PAR CURVE").set_value(("treasury.csv", payload, "text/csv")).run()
    assert not at.exception and not at.error
    assert "upload: treasury.csv" in _badge(at, "rates_treasury_chart")


def test_live_mode_falls_back_with_a_notice(monkeypatch):
    failed = rh.LiveRefresh(ok={"SOFR (NY Fed)": False}, notices=["SOFR (NY Fed): unavailable (timed out)"])
    monkeypatch.setattr(rh, "refresh_live", lambda *a, **kw: failed)
    rates_data._refresh.clear()
    at = _rates()
    at.radio(key="rates_source").set_value("LIVE (CACHED)").run()
    assert not at.exception and not at.error
    warnings = [w.value for w in at.warning]
    assert any("No live source answered" in w for w in warnings) and any("timed out" in w for w in warnings)
    assert "bundled snapshot" in _badge(at, "rates_treasury_chart")
    rates_data._refresh.clear()


def test_live_mode_prices_from_cached_downloads(monkeypatch):
    cache = DiskCache(Path(os.environ["DERIVX_CACHE_DIR"]))
    frame = rs.parse_fred_csv((FIXTURES / "fred_ust.csv").read_text())
    cache.store(cache.key("FRED", {"ids": "test"}), frame, b"x", {"source": "FRED", "url": "fixture"})
    ok = rh.LiveRefresh(ok={"US Treasury par curve (FRED H.15)": True})
    monkeypatch.setattr(rh, "refresh_live", lambda *a, **kw: ok)
    rates_data._refresh.clear()
    at = _rates()
    at.radio(key="rates_source").set_value("LIVE (CACHED)").run()
    at.selectbox(key="rates_asof").set_value(dt.date(2026, 10, 16)).run()
    assert not at.exception and not at.error
    assert "FRED, downloaded" in _badge(at, "rates_treasury_chart")
    rates_data._refresh.clear()


def _historical_sources(at: AppTest) -> None:
    at.selectbox(key="scn_type").set_value("Historical").run()
    at.selectbox(key="scn_hist").set_value("2013 taper tantrum").run()
    assert not at.exception and not at.error
    table = next(d.value for d in at.dataframe if list(d.value.columns)[:3] == ["CURVE", "CURRENCY", "DATA"])
    rows = table.set_index("CURVE")
    assert rows.loc["SWAP", "DATA"] == "ICE USD swap rates (vs 3M LIBOR)" and rows.loc["SWAP", "FLAG"].startswith(
        "REAL"
    )
    assert rows.loc["TREASURY", "DATA"] == "US Treasury par yield curve"
    at.selectbox(key="scn_hist").set_value("2023 SVB").run()
    table = next(d.value for d in at.dataframe if list(d.value.columns)[:3] == ["CURVE", "CURRENCY", "DATA"])
    assert table.set_index("CURVE").loc["SWAP", "FLAG"].startswith("PROXY")
    grid_caption = next(c.value for c in at.caption if c.value.startswith("Full-revaluation P&L"))
    assert "2013 taper tantrum (2013-05-02 to 2013-09-05): ICE USD swap rates" in grid_caption
    # Any window of the store.
    at.selectbox(key="scn_type").set_value("History window").run()
    assert not at.exception and not at.error
    at.selectbox(key="scn_win_start").set_value(dt.date(2022, 1, 31)).run()
    at.selectbox(key="scn_win_end").set_value(dt.date(2022, 10, 31)).run()
    assert not at.exception and not at.error
    assert any("History 2022-01-31 to 2022-10-31" in c.value for c in at.caption)
    at.selectbox(key="scn_win_start").set_value(dt.date(2026, 10, 19)).run()  # end before start
    assert any("must be after" in e.value for e in at.error)


def _uploaded_swap_history(at: AppTest) -> None:
    history = b"date,2Y,10Y\n2023-03-08,4.95,3.85\n2023-03-15,4.10,3.45\n"
    at.file_uploader(key="scn_hist_upload_0").set_value(("usd_swaps.csv", history, "text/csv")).run()
    assert not at.exception and not at.error
    assert any("Using usd_swaps.csv: 2 dates from 2023-03-08 to 2023-03-15, 2 tenors" in c.value for c in at.caption)
    at.selectbox(key="scn_type").set_value("Historical").run()
    at.selectbox(key="scn_hist").set_value("2023 SVB").run()
    table = next(d.value for d in at.dataframe if list(d.value.columns)[:3] == ["CURVE", "CURRENCY", "DATA"])
    swap = table.set_index("CURVE").loc["SWAP"]
    assert swap["FLAG"] == "YOUR UPLOAD" and swap["DATA"] == "Your uploaded USD swap-curve history"
    at.button(key="scn_hist_clear").click().run()
    table = next(d.value for d in at.dataframe if list(d.value.columns)[:3] == ["CURVE", "CURRENCY", "DATA"])
    assert table.set_index("CURVE").loc["SWAP", "FLAG"].startswith("PROXY")


def test_quote_prefills_and_badges_by_currency():
    at = _rates()
    _usd_default(at)
    _gbp_from_the_bank_of_england(at)
    _eur_proxy_and_old_dates(at)


def test_uploads():
    at = _rates()
    _uploads_survive_and_clear(at)
    _uploaded_treasury_wins(at)
    _bad_uploads_are_reported(at)


def test_historical_scenarios():
    at = _rates()
    _uploaded_swap_history(at)
    _historical_sources(at)
