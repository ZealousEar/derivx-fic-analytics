"""Rates market-data sources: parsers against recorded responses, explicit
unit conversion, the xlsx reader, HTTP retries, the cache and the offline
fallback. No test touches the network (see conftest.no_network)."""

from __future__ import annotations

import datetime as dt
import io
import json
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivx.data import rates_sources as rs
from derivx.data.cache import DiskCache
from derivx.data.xlsx import column_index, excel_date, read_workbook

FIXTURES = Path(__file__).parent / "fixtures" / "rates"
D = dt.date.fromisoformat


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Server:
    """Stands in for the network: maps URL substrings to payloads or
    exceptions, and records every request."""

    def __init__(self, routes):
        self.routes = routes
        self.urls = []

    def __call__(self, request, timeout):
        url = request.full_url
        self.urls.append((url, timeout))
        for needle, answer in self.routes.items():
            if needle in url:
                answer = answer.pop(0) if isinstance(answer, list) else answer
                if isinstance(answer, Exception):
                    raise answer
                return FakeResponse(answer)
        raise urllib.error.URLError(f"no route for {url}")


@pytest.fixture
def serve(monkeypatch):
    def install(routes):
        server = Server(routes)
        monkeypatch.setattr(rs, "_urlopen", server)
        monkeypatch.setattr(rs.time, "sleep", lambda s: None)
        return server

    return install


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.org", code, "error", {}, None)


# --------------------------------------------------------------------------
# Units and the long format
# --------------------------------------------------------------------------


def test_unit_conversion_is_explicit():
    np.testing.assert_allclose(rs.to_decimal([4.25, -0.5], "percent"), [0.0425, -0.005])
    np.testing.assert_allclose(rs.to_decimal([425.0], "bp"), [0.0425])
    np.testing.assert_allclose(rs.to_decimal([0.0425], "decimal"), [0.0425])
    np.testing.assert_allclose(rs.to_decimal([1.26344146], "index"), [1.26344146])  # a level, not a rate
    assert np.isnan(rs.to_decimal([4.25], "unknown")).all()  # never guessed from the magnitude
    with pytest.raises(ValueError, match="unknown unit 'pct'"):
        rs.to_decimal([1.0], "pct")


def test_tenor_years():
    assert rs.tenor_years("ON") == pytest.approx(1 / 365)
    assert rs.tenor_years("1M") == 1 / 12
    assert rs.tenor_years("18M") == 1.5
    assert rs.tenor_years("30D") == pytest.approx(30 / 365)
    assert np.isnan(rs.tenor_years(None))


def test_long_frame_drops_missing_values_and_keeps_columns():
    frame = rs.long_frame(
        [D("2026-01-02")] * 3, "X", ["A", "B", "C"], ["1Y", "2Y", None], ["4.5", None, "1.2"],
        ["percent", "percent", "index"], ["par_yield", "par_yield", "index_level"], ["D", "D", "D"],
    )  # fmt: skip
    assert tuple(frame.columns) == rs.LONG_COLUMNS
    assert list(frame["series"]) == ["A", "C"]
    assert list(frame["value_decimal"]) == [0.045, 1.2]


# --------------------------------------------------------------------------
# Parsers on recorded responses
# --------------------------------------------------------------------------


def test_parse_fred_treasury_series():
    frame = rs.parse_fred_csv(fixture("fred_ust.csv").decode())
    assert set(frame["series"]) == set(rs.UST_FRED_IDS)
    assert set(frame["dataset"]) == {"UST_PAR"} and set(frame["measure"]) == {"par_yield"}
    assert sorted(set(frame["asof"])) == [D("2026-10-13"), D("2026-10-14"), D("2026-10-15"), D("2026-10-16")]
    ten = frame[(frame["series"] == "DGS10") & (frame["asof"] == D("2026-10-16"))].iloc[0]
    assert (ten["tenor"], ten["tenor_years"], ten["value_original"], ten["unit_original"]) == (
        "10Y",
        10.0,
        5.09,
        "percent",
    )
    assert ten["value_decimal"] == pytest.approx(0.0509, abs=1e-15)
    one = frame[(frame["series"] == "DGS1MO") & (frame["asof"] == D("2026-10-16"))].iloc[0]
    assert (one["tenor"], one["tenor_years"]) == ("1M", 1 / 12)


def test_parse_fred_ice_swaps_and_fixings():
    swaps = rs.parse_fred_csv(fixture("fred_dswp.csv").decode())
    assert set(swaps["dataset"]) == {"USD_SWAP_ICE"} and set(swaps["measure"]) == {"par_swap"}
    assert dict(zip(swaps["series"], swaps["tenor"]))["DSWP30"] == "30Y"
    fixings = rs.parse_fred_csv(fixture("fred_fixings.csv").decode())
    sofr = fixings[fixings["series"] == "SOFR"]
    assert max(sofr["asof"]) == D("2026-10-16")  # the 2026-10-19 cell is blank: dropped, not zero
    index = fixings[fixings["series"] == "SOFRINDEX"].iloc[-1]
    assert index["unit_original"] == "index" and index["value_decimal"] == pytest.approx(1.26344146)
    sonia = fixings[fixings["series"] == "IUDSOIA"].iloc[-1]
    assert (sonia["tenor"], sonia["measure"], sonia["value_decimal"]) == (
        "ON",
        "overnight_fixing",
        pytest.approx(0.037344),
    )


def test_parse_fred_handles_dots_unknown_series_and_bad_payloads():
    text = "observation_date,DGS2,MYSERIES\n2026-01-02,.,7\n2026-01-05,4.25,8\n"
    frame = rs.parse_fred_csv(text)
    assert len(frame[frame["series"] == "DGS2"]) == 1
    mine = frame[frame["series"] == "MYSERIES"]
    assert set(mine["unit_original"]) == {"unknown"} and mine["value_decimal"].isna().all()
    overridden = rs.parse_fred_csv(text, units={"MYSERIES": "bp"})
    assert list(overridden[overridden["series"] == "MYSERIES"]["value_decimal"]) == [0.0007, 0.0008]
    with pytest.raises(ValueError, match="not a FRED CSV"):
        rs.parse_fred_csv("<html><body>Error</body></html>")


def test_parse_fred_api_json():
    frame = rs.parse_fred_json(fixture("fred_api_dgs10.json").decode(), "DGS10")
    assert list(frame["asof"]) == [D("2026-10-13"), D("2026-10-14"), D("2026-10-15"), D("2026-10-16")]  # "." dropped
    assert frame["value_decimal"].iloc[-1] == pytest.approx(0.0509)
    vintages = {
        "observations": [
            {"date": "2026-01-02", "realtime_start": "2026-01-03", "value": "4.10"},
            {"date": "2026-01-02", "realtime_start": "2026-02-01", "value": "4.12"},  # revised
        ]
    }
    assert rs.parse_fred_json(json.dumps(vintages), "DGS10")["value_original"].tolist() == [4.12]
    with pytest.raises(ValueError, match="not a FRED API"):
        rs.parse_fred_json('{"error_code": 400}', "DGS10")


def test_parse_treasury_csv_matches_the_published_layout():
    frame = rs.parse_treasury_csv(fixture("treasury_2026.csv").decode("utf-8-sig"))
    assert frame["asof"].nunique() == 5 and max(frame["asof"]) == D("2026-10-19")
    day = frame[frame["asof"] == D("2026-10-19")].set_index("tenor")
    assert list(day.index) == ["1M", "6W", "2M", "3M", "4M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    assert day.loc["10Y", "value_decimal"] == pytest.approx(0.0515) and day.loc["6W", "series"] == "1.5 Month"
    bom = rs.parse_treasury_csv(("﻿" + fixture("treasury_2026.csv").decode()).encode().decode("utf-8-sig"))
    assert len(bom) == len(frame)
    with pytest.raises(ValueError, match="Treasury"):
        rs.parse_treasury_csv("a,b\n1,2\n")


def test_parse_nyfed():
    sofr = rs.parse_nyfed_json(fixture("nyfed_sofr_last3.json").decode())
    assert list(sofr["series"].unique()) == ["SOFR"] and len(sofr) == 3
    assert sofr.iloc[0][["asof", "value_decimal", "tenor"]].tolist() == [D("2026-10-16"), pytest.approx(0.0388), "ON"]
    effr = rs.parse_nyfed_json(fixture("nyfed_effr_search.json").decode())
    assert set(effr["series"]) == {"EFFR"} and min(effr["asof"]) == D("2026-10-13")
    averages = rs.parse_nyfed_json(fixture("nyfed_sofrai_last2.json").decode())
    latest = averages[averages["asof"] == max(averages["asof"])].set_index("series")
    assert (
        latest.loc["SOFR30DAYAVG", "tenor"] == "30D" and latest.loc["SOFR30DAYAVG", "measure"] == "compounded_average"
    )
    assert latest.loc["SOFRINDEX", "unit_original"] == "index"
    with pytest.raises(ValueError, match="refRates"):
        rs.parse_nyfed_json("{}")


def test_parse_ecb():
    estr = rs.parse_ecb_csv(fixture("ecb_estr.csv").decode())  # full-detail layout with extra columns
    assert set(estr["series"]) == {"ESTR"} and set(estr["dataset"]) == {"FIXINGS"} and len(estr) == 3
    curve = rs.parse_ecb_csv(fixture("ecb_yc_par.csv").decode())
    assert set(curve["dataset"]) == {"EUR_AAA_ECB"}
    by = curve[curve["asof"] == D("2026-10-19")].set_index("tenor")
    assert by.loc["10Y", "measure"] == "par_yield_cc" and by.loc["10Y", "value_decimal"] == pytest.approx(
        0.034368164102
    )
    assert by.loc["126M", "measure"] == "zero_cc" and by.loc["126M", "tenor_years"] == 10.5  # SR_10Y6M
    with pytest.raises(ValueError, match="missing"):
        rs.parse_ecb_csv("KEY,VALUE\nx,1\n")
    with pytest.raises(ValueError, match="unsupported ECB"):
        rs.parse_ecb_csv("KEY,TIME_PERIOD,OBS_VALUE\nICP.M.U2.N.000000.4.ANR,2026-01-02,1.1\n")
    with pytest.raises(ValueError, match="unsupported ECB exchange-rate series"):  # monthly averages, not daily
        rs.parse_ecb_csv("KEY,TIME_PERIOD,OBS_VALUE\nEXR.M.USD.EUR.SP00.A,2026-01-02,1.1\n")


def test_parse_boe_latest_archive():
    frame = rs.parse_boe_ois_zip(fixture("boe_latest.zip"))
    assert set(frame["dataset"]) == {"GBP_OIS_BOE"} and set(frame["measure"]) == {"zero_cc"}  # GLC skipped
    assert sorted(set(frame["asof"])) == [D("2026-10-01"), D("2026-10-02"), D("2026-10-05")]
    day = frame[frame["asof"] == D("2026-10-01")].set_index("tenor")
    assert len(day) == 100 and day["tenor_years"].min() == 1 / 12 and day["tenor_years"].max() == 25.0
    assert day.loc["1M", "value_original"] == pytest.approx(3.7438932259111803)  # first cell of the spot short end
    assert day.loc["25Y", "series"] == "BoE OIS spot 25Y"
    # 2Y is on both sheets; the short-end value is kept and the two agree to 1e-8 percent.
    assert day.loc["2Y", "value_original"] == 4.537110666774925
    assert day.loc["2Y", "value_original"] == pytest.approx(4.537110675557911, abs=1e-7)


def test_parse_boe_older_layout_and_bad_payloads():
    frame = rs.parse_boe_ois_workbook(fixture("boe_ois_2009_2015.xlsx"))
    assert frame["tenor_years"].max() == 5.0  # the 2009-2015 workbook stops at five years
    assert min(frame["asof"]) >= D("2009-01-01")
    with pytest.raises(ValueError, match="zip"):
        rs.parse_boe_ois_zip(b"not a zip")
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("GLC Nominal daily data current month.xlsx", b"")
    with pytest.raises(ValueError, match="no OIS workbook"):
        rs.parse_boe_ois_zip(empty.getvalue())


def test_xlsx_reader():
    assert excel_date(46296.0) == D("2026-10-01")
    assert excel_date(0.0, date1904=True) == D("1904-01-01")
    assert column_index("A") == 0 and column_index("Z") == 25 and column_index("AA") == 26
    book = read_workbook(
        zipfile.ZipFile(io.BytesIO(fixture("boe_latest.zip"))).read("OIS daily data current month.xlsx")
    )
    assert "4. spot curve" in book.sheets and book.sheet("info")[0][0] == "Bank of England UK OIS curve data"
    with pytest.raises(ValueError, match="no sheet"):
        book.sheet("missing")
    with pytest.raises(ValueError, match="not an xlsx"):
        read_workbook(b"plain text")
    with pytest.raises(ValueError, match="not an Excel date"):
        excel_date("2026-01-01")


# --------------------------------------------------------------------------
# HTTP, cache and offline fallback
# --------------------------------------------------------------------------


def test_http_get_retries_with_backoff_then_gives_up(serve, monkeypatch):
    sleeps = []
    server = serve({"example.org": [urllib.error.URLError("reset"), http_error(503), b"ok"]})
    monkeypatch.setattr(rs.time, "sleep", sleeps.append)
    assert rs.http_get("https://example.org/a", timeout=3, retries=2, backoff=0.5) == b"ok"
    assert sleeps == [0.5, 1.0] and [t for _, t in server.urls] == [3, 3, 3]
    serve({"example.org": [urllib.error.URLError("down")] * 3})
    with pytest.raises(rs.SourceError, match="unreachable after 3 attempts"):
        rs.http_get("https://example.org/a", retries=2)


def test_http_get_retries_a_transfer_cut_short_then_gives_up_with_a_source_error(serve):
    import http.client

    cut = http.client.IncompleteRead(b"x" * 10, 90)  # 10 bytes of a 100-byte body, then the connection dropped
    server = serve({"example.org": [cut, b"ok"]})
    assert rs.http_get("https://example.org/a", retries=1) == b"ok" and len(server.urls) == 2
    serve({"example.org": [http.client.IncompleteRead(b"", 5), http.client.RemoteDisconnected("closed")]})
    with pytest.raises(rs.SourceError, match=r"unreachable after 2 attempts \(closed\)"):
        rs.http_get("https://example.org/a", retries=1)
    serve({"example.org": [http.client.IncompleteRead(b"ab", 3)]})
    with pytest.raises(rs.SourceError, match=r"IncompleteRead\(2 bytes read, 3 more expected\)"):
        rs.http_get("https://example.org/a", retries=0)


def test_http_get_does_not_retry_refusals(serve):
    server = serve({"example.org": [http_error(403), b"never"]})
    with pytest.raises(rs.SourceBlocked, match="403"):
        rs.http_get("https://example.org/a", retries=3)
    assert len(server.urls) == 1
    serve({"example.org": [http_error(404)]})
    with pytest.raises(rs.SourceError, match="HTTP 404"):
        rs.http_get("https://example.org/a", retries=3)


def test_fred_fetch_builds_the_request_and_caches(serve, tmp_path):
    server = serve({"fredgraph.csv": [fixture("fred_ust.csv")]})
    cache = DiskCache(tmp_path)
    got = rs.fred_series(rs.UST_FRED_IDS, D("2026-10-13"), D("2026-10-16"), offline=False, cache=cache)
    url, timeout = server.urls[0]
    assert url.startswith(rs.FRED_CSV_URL) and "id=DGS1MO,DGS3MO" in url and timeout == rs.DEFAULT_TIMEOUT
    assert "cosd=" + ",".join(["2026-10-13"] * 11) in url and "coed=" + ",".join(["2026-10-16"] * 11) in url
    assert not got.from_cache and len(got.frame) == 44 and len(got.sha256) == 64
    entry = cache.entries()[0]
    assert entry.meta["rows"] == 44 and entry.meta["first"] == "2026-10-13" and entry.meta["sha256"] == got.sha256
    # Fresh enough: no request. Offline: the cache only.
    again = rs.fred_series(rs.UST_FRED_IDS, D("2026-10-13"), D("2026-10-16"), offline=False, max_age=60, cache=cache)
    offline = rs.fred_series(rs.UST_FRED_IDS, D("2026-10-13"), D("2026-10-16"), offline=True, cache=cache)
    assert len(server.urls) == 1 and again.from_cache and offline.from_cache
    pd.testing.assert_frame_equal(again.frame, got.frame)
    assert offline.notices and "offline" in offline.notices[0]


def test_failed_fetch_falls_back_to_the_stale_cache(serve, tmp_path):
    cache = DiskCache(tmp_path)
    serve(
        {"newyorkfed": [fixture("nyfed_sofr_last3.json"), urllib.error.URLError("down"), urllib.error.URLError("down")]}
    )
    first = rs.nyfed_rate("sofr", last_n=3, offline=False, cache=cache)
    second = rs.nyfed_rate("sofr", last_n=3, offline=False, retries=1, cache=cache)
    assert second.from_cache and second.stale and "fetch failed" in second.notices[0]
    pd.testing.assert_frame_equal(first.frame, second.frame)
    serve({"newyorkfed": b"<html>maintenance</html>"})
    third = rs.nyfed_rate("sofr", last_n=3, offline=False, cache=cache)  # 200 but not JSON
    assert third.stale and "refRates" in third.notices[0]


def test_failures_without_a_cache_raise_clear_errors(serve, tmp_path):
    cache = DiskCache(tmp_path)
    with pytest.raises(rs.SourceError, match="offline: no cached"):
        rs.ecb_estr(last_n=3, offline=True, cache=cache)
    serve({"data-api.ecb": b"<html>maintenance</html>"})
    with pytest.raises(rs.SourceError, match="unexpected response"):
        rs.ecb_estr(last_n=3, offline=False, cache=cache)
    assert rs.is_offline()  # the test session itself runs offline


def test_fred_api_key_is_used_but_never_stored(serve, tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "secret-key-123")
    server = serve({"api.stlouisfed.org": [fixture("fred_api_dgs10.json")]})
    cache = DiskCache(tmp_path)
    got = rs.fred_series(["DGS10"], D("2026-10-12"), vintage=D("2026-10-20"), offline=False, cache=cache)
    assert "api_key=secret-key-123" in server.urls[0][0] and "realtime_start=2026-10-20" in server.urls[0][0]
    assert "secret" not in got.url and "REDACTED" in got.url
    stored = (tmp_path / f"{cache.entries()[0].key}.json").read_text()
    assert "secret" not in stored
    monkeypatch.delenv("FRED_API_KEY")
    with pytest.raises(ValueError, match="FRED_API_KEY"):
        rs.fred_series(["DGS10"], vintage=D("2026-10-20"), offline=False, cache=cache)


@pytest.mark.parametrize("raw", ['"abc123"', "'abc123'", "  abc123\n", ' "abc123" '])
def test_fred_api_key_ignores_quotes_and_whitespace(monkeypatch, raw):
    monkeypatch.setenv("FRED_API_KEY", raw)
    assert rs.fred_api_key() == "abc123"
    monkeypatch.setenv("FRED_API_KEY", '""')
    assert rs.fred_api_key() == ""  # an empty quoted value means no key


def test_missing_fred_series_are_reported(serve, tmp_path):
    serve({"fredgraph.csv": [fixture("fred_ust.csv")]})
    got = rs.fred_series(["DGS10", "NOPE"], D("2026-10-13"), offline=False, cache=DiskCache(tmp_path))
    assert any("NOPE" in n for n in got.notices)


def test_source_request_shapes(serve, tmp_path):
    cache = DiskCache(tmp_path)
    server = serve({
        "daily-treasury-rates.csv/2026": fixture("treasury_2026.csv"),
        "effr/search.json": fixture("nyfed_effr_search.json"),
        "data/YC/": fixture("ecb_yc_par.csv"),
        "latest-yield-curve-data.zip": fixture("boe_latest.zip"),
    })  # fmt: skip
    treasury = rs.treasury_par_curve(2026, offline=False, cache=cache)
    assert server.urls[0][1] == rs.TREASURY_TIMEOUT and treasury.frame["asof"].nunique() == 5
    rs.nyfed_rate("effr", start=D("2026-10-13"), end=D("2026-10-16"), offline=False, cache=cache)
    assert "startDate=2026-10-13&endDate=2026-10-16" in server.urls[1][0]
    rs.ecb_yield_curve(["3M", "30M", "10Y"], "AAA", "spot", last_n=2, offline=False, cache=cache)
    assert "G_N_A.SV_C_YM.SR_3M+SR_2Y6M+SR_10Y" in server.urls[2][0]
    boe = rs.boe_ois_curve("latest", start=D("2026-10-02"), offline=False, cache=cache)
    assert min(boe.frame["asof"]) == D("2026-10-02")
    with pytest.raises(ValueError, match="year"):
        rs.treasury_par_curve(1980, cache=cache)
    with pytest.raises(ValueError, match="either last_n or start"):
        rs.nyfed_rate("sofr", cache=cache)
    with pytest.raises(ValueError, match="unknown NY Fed rate"):
        rs.nyfed_rate("libor", last_n=1, cache=cache)
    with pytest.raises(ValueError, match="curve must be"):
        rs.ecb_yield_curve(curve="AA", cache=cache)
    with pytest.raises(ValueError, match="archive must be"):
        rs.boe_ois_curve("weekly", cache=cache)


def test_cache_store_and_unwritable_directory(tmp_path):
    frame = rs.parse_nyfed_json(fixture("nyfed_sofr_last3.json").decode())
    cache = DiskCache(tmp_path / "cache")
    key = cache.key("NY Fed", {"rate": "sofr", "last_n": 3})
    assert key == cache.key("NY Fed", {"last_n": 3, "rate": "sofr"})  # order-independent
    entry = cache.store(key, frame, b"payload", {"source": "NY Fed", "url": "u"})
    loaded = cache.load(key)
    assert loaded.meta["sha256"] == entry.meta["sha256"] and loaded.age_seconds() < 60
    pd.testing.assert_frame_equal(loaded.frame, frame)
    assert not list((tmp_path / "cache").glob(".tmp-*"))  # temporary files renamed or removed
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    readonly = DiskCache(blocked / "cache")
    assert readonly.store(key, frame, b"payload", {"source": "NY Fed"}) is None
    assert "not writable" in readonly.write_error
    assert readonly.entries() == [] and readonly.load(key) is None


# --------------------------------------------------------------------------
# Review regressions
# --------------------------------------------------------------------------


def _boe_zip(edit) -> bytes:
    """The recorded BoE archive with its OIS workbook's sheets passed
    through ``edit(name, xml) -> xml``."""

    outer = zipfile.ZipFile(io.BytesIO(fixture("boe_latest.zip")))
    book = zipfile.ZipFile(io.BytesIO(outer.read("OIS daily data current month.xlsx")))
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as out:
        for name in book.namelist():
            data = book.read(name)
            out.writestr(name, edit(name, data) if name.startswith("xl/worksheets/sheet") else data)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("OIS daily data current month.xlsx", inner.getvalue())
    return archive.getvalue()


def test_boe_curves_dated_on_weekends_or_holidays_move_to_their_business_day():
    """The month-end archive labels the last business day's curve with the
    calendar month end (Saturday 2009-01-31 is Friday 2009-01-30's curve;
    Sunday 2013-03-31 after Good Friday is Thursday 2013-03-28's)."""

    assert rs.boe_curve_date(D("2009-01-31")) == D("2009-01-30")
    assert rs.boe_curve_date(D("2013-03-31")) == D("2013-03-28")
    assert rs.boe_curve_date(D("2026-08-31")) == D("2026-08-28")  # summer bank holiday Monday
    assert rs.boe_curve_date(D("2026-10-19")) == D("2026-10-19")
    # The 2026-10-05 curve relabelled as Saturday 2026-10-10 parses as Friday 2026-10-09.
    saturday = _boe_zip(lambda name, xml: xml.replace(b'r="A8" s="5"><v>46300</v>', b'r="A8" s="5"><v>46305</v>'))
    frame = rs.parse_boe_ois_zip(saturday)
    assert sorted(set(frame["asof"])) == [D("2026-10-01"), D("2026-10-02"), D("2026-10-09")]
    assert all(day.weekday() < 5 for day in frame["asof"])


def test_corrupt_payloads_become_source_errors_or_stale_cache(serve, tmp_path):
    """A truncated workbook (broken XML) or a corrupt archive member used to
    escape as ParseError/zlib.error and crash the LIVE page."""

    truncated = _boe_zip(lambda name, xml: xml[: len(xml) // 2])
    with pytest.raises(ValueError, match="corrupt or truncated"):
        rs.parse_boe_ois_zip(truncated)
    member = zipfile.ZipFile(io.BytesIO(fixture("boe_latest.zip"))).getinfo("OIS daily data current month.xlsx")
    broken = bytearray(fixture("boe_latest.zip"))
    start = member.header_offset + 30 + len(member.filename.encode()) + len(member.extra) + 200
    broken[start : start + 64] = bytes(64)  # damage the compressed bytes of the OIS workbook
    with pytest.raises(ValueError, match="corrupt or truncated"):
        rs.parse_boe_ois_zip(bytes(broken))
    cache = DiskCache(tmp_path)
    serve({"latest-yield-curve-data.zip": [truncated]})
    with pytest.raises(rs.SourceError, match="BoE: unexpected response"):
        rs.boe_ois_curve("latest", offline=False, cache=cache)
    serve({"latest-yield-curve-data.zip": [fixture("boe_latest.zip"), truncated]})
    good = rs.boe_ois_curve("latest", offline=False, cache=cache)
    stale = rs.boe_ois_curve("latest", offline=False, cache=cache)
    assert stale.from_cache and stale.stale and "corrupt or truncated" in stale.notices[0]
    pd.testing.assert_frame_equal(stale.frame, good.frame)


def test_any_parser_exception_is_a_source_error(serve, tmp_path, monkeypatch):
    def explode(text):
        raise TypeError("unexpected layout")

    monkeypatch.setattr(rs, "parse_nyfed_json", explode)
    serve({"newyorkfed": [b"{}"]})
    with pytest.raises(rs.SourceError, match=r"NY Fed: unexpected response \(TypeError: unexpected layout\)"):
        rs.nyfed_rate("sofr", last_n=3, offline=False, cache=DiskCache(tmp_path))
