"""ECB reference rates: parser, crosses, the bundled snapshot and the
network fetch (monkeypatched: tests never use the network)."""

from __future__ import annotations

import datetime as dt
import http.client
import io
import urllib.error
from pathlib import Path

import pytest

from derivx.fx import market_data
from derivx.fx.market_data import FxDataError, bundled_ecb_rates, fetch_ecb_rates, parse_ecb_csv

FIXTURE = Path(__file__).parent / "fixtures" / "ecb_exr_2026-10-20.csv"


def test_parse_the_recorded_response_and_compute_crosses():
    rates = parse_ecb_csv(FIXTURE.read_text())
    assert rates.asof == dt.date(2026, 10, 20) and rates.origin == "live" and rates.fetched_at
    assert rates.per_eur["USD"] == 1.1215 and rates.per_eur["JPY"] == 177.34
    assert rates.cross("EURUSD") == pytest.approx(1.1215)
    assert rates.cross("GBPUSD") == pytest.approx(1.1215 / 0.84028)
    assert rates.cross("USDJPY") == pytest.approx(177.34 / 1.1215)
    assert rates.cross("EURCHF") == pytest.approx(0.9237)
    assert rates.cross("NZDUSD") == pytest.approx(1.1215 / 1.9748)
    assert set(rates.history) == {dt.date(2026, 10, 16), dt.date(2026, 10, 19), dt.date(2026, 10, 20)}
    assert "not dealable" in rates.description and "2026-10-20" in rates.description


def test_latest_complete_date_is_used_when_a_currency_is_missing():
    text = FIXTURE.read_text().replace("EXR.D.JPY.EUR.SP00.A,D,JPY,EUR,SP00,A,2026-10-20,177.34", "")
    rates = parse_ecb_csv(text)
    assert rates.asof == dt.date(2026, 10, 19) and rates.per_eur["JPY"] == 177.63


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "has no rates"),
        ("a,b\n1,2\n", "has no rates"),
        ("KEY,CURRENCY,CURRENCY_DENOM,TIME_PERIOD,OBS_VALUE\nx,USD,EUR,2026-13-01,1.1\n", "bad date or value"),
        ("KEY,CURRENCY,CURRENCY_DENOM,TIME_PERIOD,OBS_VALUE\nx,USD,EUR,2026-10-20,-1\n", "not positive"),
        ("KEY,CURRENCY,CURRENCY_DENOM,TIME_PERIOD,OBS_VALUE\nx,USD,EUR,2026-10-20,1.1\n", "no date"),
    ],
)
def test_malformed_responses_raise_a_clear_error(text, message):
    with pytest.raises(FxDataError, match=message):
        parse_ecb_csv(text)


def test_bundled_snapshot_is_the_offline_fallback():
    rates = bundled_ecb_rates()
    assert rates.origin == "bundled" and rates.asof == dt.date(2026, 10, 20)
    assert "bundled snapshot" in rates.description
    assert rates.cross("USDCAD") == pytest.approx(1.5758 / 1.1215)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_parses_the_portal_response_with_a_timeout(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["timeout"] = request.full_url, timeout
        return _Response(FIXTURE.read_bytes())

    monkeypatch.setattr(market_data.urllib.request, "urlopen", fake_urlopen)
    rates = fetch_ecb_rates(timeout=2.5)
    assert rates.cross("EURUSD") == pytest.approx(1.1215) and rates.origin == "live"
    assert seen["timeout"] == 2.5
    assert seen["url"].startswith("https://data-api.ecb.europa.eu/service/data/EXR/D.USD+GBP+JPY")
    assert "format=csvdata" in seen["url"]


@pytest.mark.parametrize(
    "error, message",
    [
        (urllib.error.URLError("timed out"), "not reachable: timed out"),
        (urllib.error.HTTPError("u", 403, "Forbidden", {}, None), "HTTP 403"),
        (TimeoutError("read timeout"), "not reachable"),
    ],
)
def test_fetch_failures_raise_fx_data_error_with_the_reason(monkeypatch, error, message):
    def failing(request, timeout):
        raise error

    monkeypatch.setattr(market_data.urllib.request, "urlopen", failing)
    with pytest.raises(FxDataError, match=message):
        fetch_ecb_rates()


class _CutOff(_Response):
    def read(self, *args):
        raise http.client.IncompleteRead(b"KEY,FREQ", 4096)


@pytest.mark.parametrize(
    "response, message",
    [(_CutOff(b""), "cut off \\(IncompleteRead\\)"), (_Response(b"\xff\xfe\x00K"), "not UTF-8")],
)
def test_truncated_or_binary_responses_raise_fx_data_error(monkeypatch, response, message):
    """Review: a connection dropped mid-response (http.client.IncompleteRead)
    or a non-text body escaped as a raw exception from the FETCH button's
    callback (a traceback on the page) instead of the offline fallback."""

    monkeypatch.setattr(market_data.urllib.request, "urlopen", lambda request, timeout: response)
    with pytest.raises(FxDataError, match=message):
        fetch_ecb_rates()


def test_unknown_currency_in_a_cross_is_reported():
    rates = parse_ecb_csv(FIXTURE.read_text(), currencies=("USD",))
    partial = market_data.EcbRates(rates.asof, {"USD": 1.1215}, "live")
    with pytest.raises(FxDataError, match="no JPY rate"):
        partial.cross("USDJPY")
