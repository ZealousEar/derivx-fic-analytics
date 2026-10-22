"""History store: the committed sample and its manifest, priorities between
snapshot, downloads and uploads, curve moves between dates, user history
uploads, the live refresh and its background Treasury download."""

from __future__ import annotations

import datetime as dt
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivx.data import rates_history as rh
from derivx.data import rates_sources as rs
from derivx.data.cache import DiskCache
from derivx.rates.curves import bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, ust_curve
from derivx.rates.market_quotes import treasury_quotes

FIXTURES = Path(__file__).parent / "fixtures" / "rates"
D = dt.date.fromisoformat


@pytest.fixture(scope="module")
def store():
    return rh.bundled_store()


def test_manifest_describes_every_committed_file():
    manifest = rh.load_manifest()
    files = {entry["file"]: entry for entry in manifest["files"]}
    assert set(files) == {p.name for p in rh.SAMPLE_DIR.glob("*.csv")}
    for name, entry in files.items():
        path = rh.SAMPLE_DIR / name
        assert rh.file_sha256(path) == entry["sha256"], f"{name} changed without rebuilding the manifest"
        header = pd.read_csv(path, nrows=0).columns
        assert header[0] == "date" and set(header[1:]) == set(entry["series"]), name
        unit = "fx" if entry["dataset"] == rh.FX_DATASET else "percent"  # FX: units of a currency per 1 EUR
        assert entry["unit"] == unit and entry["attribution"] and entry["urls"] and entry["fetched_at"]
        assert entry["dataset"] in rh.DATASETS
    total = sum((rh.SAMPLE_DIR / n).stat().st_size for n in files)
    # small enough to commit: the daily 2015-2025 Treasury curves are ~190 kB of it, the daily ECB FX rates
    # since 2015 (seven currencies, as published) ~190 kB
    assert total < 600_000 and (rh.SAMPLE_DIR / "fx_ecb.csv").stat().st_size < 200_000


def test_bundled_store_coverage(store):
    assert store.datasets() == ["EUR_AAA_ECB", "FIXINGS", "FX_ECB", "GBP_OIS_BOE", "USD_SWAP_ICE", "UST_PAR"]
    ust = store.dates("UST_PAR")
    snapshot = set(rs.parse_treasury_csv(rh.TREASURY_SNAPSHOT.read_text())["asof"])
    assert snapshot <= set(ust) and ust[-1] == max(snapshot) == EXAMPLE_ASOF
    assert ust[0] <= D("2008-01-31") and len(ust) > 400  # month ends back to 2008 plus every 2026 business day
    daily = [d for d in ust if d >= D("2015-01-02")]
    assert len(daily) > 2900 and max((b - a).days for a, b in zip(daily[:-1], daily[1:])) <= 4  # every business day
    assert store.coverage("GBP_OIS_BOE")[0] <= D("2009-01-31") and store.coverage("GBP_OIS_BOE")[1] >= D("2026-10-16")
    assert store.curve("GBP_OIS_BOE", store.coverage("GBP_OIS_BOE")[1]).index[-1] == "25Y"
    assert store.coverage("USD_SWAP_ICE")[1] <= rs.ICE_SWAP_LAST_DATE
    fixings = store.fixings(EXAMPLE_ASOF).set_index("fixing")
    assert set(fixings.index) == {"SOFR", "EFFR", "OBFR", "ESTR", "SONIA"}
    assert fixings.loc["SOFR", "date"] <= EXAMPLE_ASOF and 0.0 < fixings.loc["SOFR", "rate"] < 0.1
    assert store.fixing("SOFR", D("2020-01-01")) is None  # the bundled fixings start in the snapshot year


def test_store_treasury_curve_is_the_examples_curve(store):
    """The page's Treasury curve (from the store) equals the curve the tests
    and golden fixtures use (read straight from the snapshot CSV)."""

    quotes = treasury_quotes(store.curve("UST_PAR", EXAMPLE_ASOF).to_dict())
    from_store = bootstrap(EXAMPLE_ASOF, quotes, "USD UST")
    reference = ust_curve(EXAMPLE_ASOF)
    assert [(q.kind, q.tenor) for q in from_store.quotes] == [(q.kind, q.tenor) for q in reference.quotes]
    np.testing.assert_array_equal(from_store.dfs, reference.dfs)
    assert store.origin("UST_PAR", EXAMPLE_ASOF) == "US Treasury, bundled snapshot"


def test_higher_priority_rows_win(store):
    day = EXAMPLE_ASOF
    upload = rs.parse_treasury_csv(f'Date,"10 Yr"\n{day:%m/%d/%Y},6.00\n').assign(origin="upload: mine.csv")
    merged = rh.build_store([store.frame, upload], "test")
    assert merged.curve("UST_PAR", day)["10Y"] == 0.06
    assert merged.curve("UST_PAR", day)["2Y"] == store.curve("UST_PAR", day)["2Y"]  # other tenors untouched
    assert "upload: mine.csv" in merged.origin("UST_PAR", day)
    assert rh.build_store([], "empty").datasets() == []


def test_curve_moves(store):
    move = store.curve_move("UST_PAR", D("2022-01-03"), D("2022-10-21"))
    assert (move.start, move.end) == (D("2022-01-03"), D("2022-10-21"))
    assert dict(zip(move.tenors, move.change_bp))["2Y"] == 371.0  # 0.78% -> 4.49%
    assert move.series[0] == "FRED DGS1MO" and move.what == "par yields"
    nearest = store.curve_move("UST_PAR", D("2022-01-01"), D("2022-10-22"))  # weekend and holiday ends
    assert (nearest.start, nearest.end) == (D("2021-12-31"), D("2022-10-21"))
    with pytest.raises(ValueError, match="within 4 days"):
        store.curve_move("UST_PAR", D("2011-06-15"), D("2011-06-22"))  # mid-month before the daily 2015+ sample
    with pytest.raises(ValueError, match="must be after"):
        store.curve_move("UST_PAR", D("2022-10-21"), D("2022-01-03"))
    with pytest.raises(ValueError, match="no ICE"):
        store.curve_move("USD_SWAP_ICE", D("2022-01-03"), D("2022-10-21"))
    with pytest.raises(ValueError, match="unknown dataset"):
        rh.dataset_info("JPY_TONA")


def test_gbp_moves_are_derived_par_ois_rates(store):
    move = store.curve_move("GBP_OIS_BOE", D("2022-09-21"), D("2022-09-28"))
    assert move.what.startswith("par OIS rates derived") and move.tenors[0] == "1M" and move.tenors[-1] == "25Y"
    a, b = store.ois_par("GBP_OIS_BOE", move.start), store.ois_par("GBP_OIS_BOE", move.end)
    assert dict(zip(move.tenors, move.change_bp))["10Y"] == round((b["10Y"] - a["10Y"]) * 1e4, 2)
    assert dict(zip(move.tenors, move.change_bp))["2Y"] > 50  # the gilt-crisis week repriced UK rates sharply
    early = store.curve_move("GBP_OIS_BOE", D("2013-05-02"), D("2013-09-05"))
    assert early.tenors[-1] == "5Y"  # the BoE OIS curve only reached five years before 2016


def test_ois_par_reprices_the_boe_curve(store):
    day = store.coverage("GBP_OIS_BOE")[1]
    par = store.ois_par("GBP_OIS_BOE", day)
    from derivx.rates.curves import CurveQuote
    from derivx.rates.market_quotes import zero_rate_curve

    curve = bootstrap(day, [CurveQuote("OIS", t, r) for t, r in par.items()], "GBP SONIA", "linear_zero")
    zeros = store.curve("GBP_OIS_BOE", day, "zero_cc")
    from derivx.rates.conventions import parse_tenor

    boe = zero_rate_curve(day, [parse_tenor(t).years for t in zeros.index], list(zeros))
    for inst in curve.instruments:
        error_bp = abs(curve.zero_rate(inst.pillar) - boe.zero_rate(inst.pillar)) * 1e4
        assert error_bp < (1e-6 if inst.tenor.years <= 10 else 0.5), (inst.tenor, error_bp)
    with pytest.raises(ValueError, match="no OIS conventions"):
        store.ois_par("UST_PAR", day)


def test_parse_history_upload_long_and_wide():
    long = pd.DataFrame({"date": ["2024-01-02", "2024-01-02", "2024-02-01", "2024-02-01"],
                         "tenor": ["2Y", "10Y", "2Y", "10Y"], "rate": [4.2, 3.9, 4.4, 4.0]})  # fmt: skip
    frame, issues, units = rh.parse_history_upload(long, "USD", name="mine.csv")
    assert units == "percent" and any("percent" in i.message for i in issues)
    assert set(frame["dataset"]) == {"USER_SWAP:USD"} and set(frame["origin"]) == {"upload: mine.csv"}
    store = rh.build_store([frame])
    move = store.curve_move("USER_SWAP:USD", D("2024-01-02"), D("2024-02-01"))
    assert dict(zip(move.tenors, move.change_bp)) == {"2Y": 20.0, "10Y": 10.0}
    wide = pd.DataFrame({"Date": ["2024-01-02", "2024-02-01"], "2Y": [0.042, 0.044], "10Y": [0.039, None]})
    frame, _, units = rh.parse_history_upload(wide, "GBP", units="decimal")
    assert units == "decimal" and len(frame) == 3 and set(frame["dataset"]) == {"USER_SWAP:GBP"}


@pytest.mark.parametrize(
    "table, message",
    [
        (pd.DataFrame({"when": ["2024-01-02"], "2Y": [4.0]}), "'date' column"),
        (pd.DataFrame({"date": ["2024-01-02", "2024-02-01"], "2Q": [4.0, 4.1]}), "2Q"),
        (pd.DataFrame({"date": ["2024-01-02", "2024-02-01"], "2Y": ["4.0", "n/a"]}), "not a number"),
        (pd.DataFrame({"date": ["2024-01-02", "2024-02-01"], "2Y": [45.0, 46.0]}), "outside"),
        (pd.DataFrame({"date": ["2024-01-02"], "2Y": [4.0]}), "two dates"),
        (pd.DataFrame({"date": ["not a date", "2024-02-01"], "2Y": [4.0, 4.1]}), "dates"),
        # Final review: a 'rate' column without 'tenor' (one series) raised pandas' melt ValueError on the
        # SCENARIOS tab; a 'tenor' column without 'rate' was read as a tenor named 'tenor'.
        (pd.DataFrame({"date": ["2024-01-02", "2024-02-01"], "Rate": [4.0, 4.1]}), "'rate' column but no 'tenor'"),
        (pd.DataFrame({"date": ["2024-01-02"], "tenor": ["2Y"], "2Y": [4.0]}), "'tenor' column but no 'rate'"),
    ],
)
def test_parse_history_upload_errors(table, message):
    units = "percent" if message == "outside" else "auto"
    frame, issues, _ = rh.parse_history_upload(table, "USD", units=units)
    assert frame.empty and any(i.level == "error" and message in i.message for i in issues), issues


def test_cached_downloads_join_the_store(tmp_path):
    cache = DiskCache(tmp_path)
    assert rh.cache_signature(cache) == () and rh.cached_frames(cache) == []
    frame = rs.parse_fred_csv((FIXTURES / "fred_dswp.csv").read_text())
    cache.store(
        cache.key("FRED", {"ids": ["DSWP"]}), frame, b"x", {"source": "FRED", "fetched_at": "2026-10-20T00:00:00+00:00"}
    )
    signature = rh.cache_signature(cache)
    assert len(signature) == 1 and signature[0][1] == "2026-10-20T00:00:00+00:00"
    (tmp_path / "broken.json").write_text("{not json")
    frames = rh.cached_frames(cache)
    assert len(frames) == 1 and set(frames[0]["origin"]) == {"downloaded 2026-10-20T00:00:00+00:00"}
    store = rh.build_store([rh.bundled_frame()] + frames)
    assert D("2013-05-01") in store.dates("USD_SWAP_ICE")


def test_refresh_live_turns_failures_into_notices(tmp_path):
    frame = rs.parse_nyfed_json((FIXTURES / "nyfed_sofr_last3.json").read_text())
    ok = rs.Fetched(frame, "NY Fed", "u", "2026-10-20T00:00:00+00:00", "0" * 64, notices=["fine print"])
    seen = {}

    def good(**kw):
        seen.update(kw)
        return ok

    def bad(**kw):
        raise rs.SourceBlocked("HTTP 403 from example.org")

    result = rh.refresh_live(D("2026-10-20"), cache=DiskCache(tmp_path), requests={"good": good, "bad": bad})
    assert result.ok == {"good": True, "bad": False}
    assert any("bad: unavailable (HTTP 403" in n for n in result.notices) and "good: fine print" in result.notices
    assert seen["max_age"] == 3600.0 and seen["timeout"] == 8.0 and seen["retries"] == 1
    names = rh.live_requests(D("2026-10-20"))
    assert {"US Treasury par curve (FRED H.15)", "UK OIS curve (BoE)", "SOFR (NY Fed)", "ESTR (ECB)"} <= set(names)


def test_live_requests_run_offline_against_the_cache(tmp_path):
    result = rh.refresh_live(D("2026-10-20"), cache=DiskCache(tmp_path))  # offline in tests, empty cache
    assert result.ok and not any(result.ok.values())
    assert all("offline: no cached" in n for n in result.notices)


def test_treasury_downloads_in_the_background(tmp_path, monkeypatch):
    cache = DiskCache(tmp_path)
    monkeypatch.setenv("DERIVX_OFFLINE", "0")
    release = threading.Event()
    calls = []

    def slow(year, cache=None):
        calls.append(year)
        release.wait(5)
        frame = rs.parse_treasury_csv((FIXTURES / "treasury_2026.csv").read_text())
        cache.store(cache.key("US Treasury", {"year": year}), frame, b"csv", {"source": "US Treasury"})

    monkeypatch.setattr(rs, "treasury_par_curve", slow)
    monkeypatch.setattr(rh, "_PENDING", {})
    first = rh.treasury_in_background(2026, cache)
    second = rh.treasury_in_background(2026, cache)
    assert "background" in first and "background" in second and calls == [2026]  # started once
    release.set()
    rh._PENDING[(2026, str(cache.root))][0].result(timeout=5)
    assert rh.treasury_in_background(2026, cache) is None  # cached and fresh
    monkeypatch.setenv("DERIVX_OFFLINE", "1")
    assert rh.treasury_in_background(2025, cache) is None  # offline: never starts


# --------------------------------------------------------------------------
# Review regressions
# --------------------------------------------------------------------------


def test_bundled_boe_curves_are_dated_by_london_business_days(store):
    """58 month-end curves were dated on weekends or UK holidays, so the GBP
    prefill on, e.g., Friday 2013-08-30 took the July curve."""

    from derivx.rates.conventions import get_calendar

    london = get_calendar("LONDON")
    days = store.dates("GBP_OIS_BOE")
    assert days[0] == D("2009-01-30") and all(london.is_business_day(day) for day in days)
    assert store.latest_on_or_before("GBP_OIS_BOE", D("2013-08-30")) == D("2013-08-30")
    assert store.latest_on_or_before("GBP_OIS_BOE", D("2013-03-28")) == D("2013-03-28")


def test_failing_live_requests_never_raise(tmp_path):
    """Any exception from a request (not only SourceError/ValueError/OSError)
    is a notice: the LIVE page must not show a traceback."""

    def broken(**kw):
        raise SyntaxError("unclosed token: line 2, column 788")  # what a truncated BoE sheet raised

    result = rh.refresh_live(D("2026-10-20"), cache=DiskCache(tmp_path), requests={"UK OIS curve (BoE)": broken})
    assert result.ok == {"UK OIS curve (BoE)": False}
    assert "unclosed token" in result.notices[0] and "unavailable" in result.notices[0]


def test_background_treasury_download_restarts_after_max_age(tmp_path, monkeypatch):
    """It used to start only once per server process: after its first
    success the Treasury CSV was never downloaded again."""

    from concurrent.futures import Future

    cache = DiskCache(tmp_path)
    monkeypatch.setenv("DERIVX_OFFLINE", "0")
    calls = []
    monkeypatch.setattr(rs, "treasury_par_curve", lambda year, cache=None: calls.append(year))
    done = Future()
    done.set_result(None)
    now = rs.utc_now().timestamp()
    key = (2026, str(cache.root))
    monkeypatch.setattr(rh, "_PENDING", {key: (done, now - 60.0)})
    assert rh.treasury_in_background(2026, cache, max_age=3600.0) is None and calls == []  # finished a minute ago
    rh._PENDING[key] = (done, now - 7200.0)
    assert "background" in rh.treasury_in_background(2026, cache, max_age=3600.0)  # two hours ago: again
    rh._PENDING[key][0].result(timeout=5)
    assert calls == [2026]


def test_auto_units_refuse_magnitudes_that_could_be_either():
    """2020-21 rates in percent (0.15-0.30) were read as decimals (15-30%):
    inside the accepted range, so every move came out 100x too large."""

    from derivx.rates.quotes import ambiguous_units, clean_quotes

    zirp = pd.DataFrame({"date": ["2021-01-04", "2021-02-01"], "2Y": [0.15, 0.20], "5Y": [0.25, 0.30]})
    frame, issues, _ = rh.parse_history_upload(zirp, "USD")
    assert frame.empty and any(i.level == "error" and "choose the units" in i.message for i in issues)
    frame, _, units = rh.parse_history_upload(zirp, "USD", units="percent")
    assert units == "percent" and frame["value_decimal"].max() == pytest.approx(0.003)
    rows = [{"kind": "OIS", "tenor": t, "rate": r} for t, r in (("1Y", 0.12), ("2Y", 0.2), ("5Y", 0.35))]
    quotes, issues, _ = clean_quotes(rows)
    assert quotes == [] and "could be percent (0.2%) or decimals (20%)" in issues[-1].message
    assert ambiguous_units([0.04, 0.045]) is None and ambiguous_units([4.1, 4.5]) is None  # clear-cut cases
    assert ambiguous_units([450.0]) is None and ambiguous_units([]) is None


def test_uploaded_treasury_origin_is_named_once():
    text = (FIXTURES / "treasury_2026.csv").read_text()
    store = rh.build_store([rh.treasury_upload(text, "t.csv")])
    assert store.origin("UST_PAR", D("2026-10-19")) == "upload: t.csv"  # was "upload: t.csv, upload: t.csv"


def test_store_fingerprint_follows_the_data_not_the_label(store):
    same = rh.build_store([store.frame], "another label")
    assert same.fingerprint == store.fingerprint and len(store.fingerprint) == 64
    changed = store.frame.copy()
    changed.loc[changed.index[-1], "value_decimal"] += 1e-6  # one value, same number of rows
    other = rh.build_store([changed], store.label)
    assert len(other.frame) == len(store.frame) and other.fingerprint != store.fingerprint
