"""The daily option-chain store (``derivx.data.snapshots``): the compact
storage schema and its size, the ``data-snapshots`` layout and index,
append-only idempotency, the collection window (weekends, NYSE holidays,
sessions not yet closed), retries that never write a partial or
mislabelled file, and the ``derivx snapshots`` commands. No network: the
chains are the recorded fake Yahoo responses or the SYNTHETIC history."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from _fake_yahoo import FakeTicker
from _snapshot_store import fake_fetch, fake_session_fetch, real_store, spot_time, synthetic_store

from derivx import cli
from derivx.data import snapshots as snap
from derivx.data.chain_schema import load_chain_file
from derivx.data.market_data import apply_liquidity_filters
from derivx.data.samples import load_sample_chain

ROOT = Path(__file__).resolve().parents[1]
AFTER_CLOSE = pd.Timestamp("2026-10-20T21:35:00Z")  # 17:35 New York (EDT), a Tuesday


# -- compact storage schema ------------------------------------------------------


def test_compact_schema_keeps_the_lab_columns_and_the_limits():
    chain = load_sample_chain().single()
    frame, counts = snap.compact_chain(chain)
    assert list(frame.columns) == list(snap.STORE_COLUMNS)
    days = (frame["expiry"] - frame["trade_date"]).dt.days
    ratio = frame["strike"] / frame["underlying_price"]
    assert days.min() >= 1 and days.max() <= 730
    assert ratio.min() >= 0.4 and ratio.max() <= 1.6
    assert counts["rows_in"] == len(chain) == 10040 and counts["rows_kept"] == len(frame) == 8352
    assert counts["rows_kept"] + counts["dropped_expiry_window"] + counts["dropped_moneyness_band"] == len(chain)
    assert (counts["expiries_in"], counts["expiries_kept"]) == (31, 28)
    # prices, strikes and volumes are as published; only Yahoo's own IV is rounded (to 1e-5)
    merged = frame.merge(chain, on=["expiry", "option_type", "strike"], suffixes=("", "_raw"))
    assert len(merged) == len(frame)
    for col in ("bid", "ask", "volume", "underlying_price"):
        pd.testing.assert_series_equal(merged[col], merged[f"{col}_raw"], check_names=False)
    assert (merged["implied_vol"] - merged["implied_vol_raw"]).abs().max() <= 0.5e-5 + 1e-12
    # the liquid quotes the lab fits are kept, except deep wings and the two expiries beyond two years
    liquid = apply_liquidity_filters(chain, min_volume=10, max_spread=0.4, min_mid=0.1)
    kept = apply_liquidity_filters(frame, min_volume=10, max_spread=0.4, min_mid=0.1)
    assert (len(liquid), len(kept)) == (3520, 3228)


def test_a_stored_file_reloads_through_the_chain_schema_with_its_lineage(tmp_path):
    snapshot = snap.build_snapshot(fake_session_fetch("2026-10-21"), session="2026-10-21")
    store = snap.SnapshotStore(tmp_path)
    assert store.add(snapshot) == "added"
    path = tmp_path / "options" / "SPY" / "2026" / "SPY_2026-10-21.csv.gz"
    ds = load_chain_file(path.name, path.read_bytes())
    chain = ds.single()
    assert ds.ticker == "SPY" and ds.trade_dates == [pd.Timestamp("2026-10-21")]
    assert ds.fetched_at_utc is not None and set(chain["source"]) == {"yahoo"}
    assert (chain["mid"] - (chain["bid"] + chain["ask"]) / 2).abs().max() < 1e-12  # recomputed on load
    assert chain["spot_time_utc"].iloc[0] == "2026-10-21T20:00:00Z"
    manifest = json.loads(path.with_name("SPY_2026-10-21.json").read_text())
    assert manifest["session"] == "2026-10-21" and manifest["fetch_status"]["status"] == "ok"
    assert manifest["sha256"] == snap.sha256_hex(path.read_bytes()) and manifest["bytes"] == path.stat().st_size
    assert manifest["manifest"]["kind"] == "snapshot" and manifest["limits"]["max_days"] == 730
    assert manifest["compaction"]["rows_kept"] == manifest["rows"] == len(chain)


def test_stored_bytes_are_deterministic_and_small_enough_for_a_year_well_under_100_mb():
    a = snap.build_snapshot(fake_session_fetch("2026-10-21"))
    b = snap.build_snapshot(fake_session_fetch("2026-10-21"))
    assert a.data == b.data and a.manifest["sha256"] == b.manifest["sha256"]
    assert gzip.decompress(a.data).startswith(b"trade_date,expiry,option_type,strike,bid,ask,implied_vol")
    full = gzip.compress((ROOT / "data" / "examples" / "spy_2026-10-20.csv.gz").read_bytes())
    assert len(a.data) < 170_000 < len(full)
    assert 252 * len(a.data) < 50e6  # a year of SPY sessions at this size


def test_limits_are_validated():
    with pytest.raises(ValueError, match="bracket 1"):
        snap.StoreLimits(moneyness=(1.1, 1.6))
    with pytest.raises(ValueError, match="min_days"):
        snap.StoreLimits(min_days=10, max_days=5)
    narrow = snap.StoreLimits(max_days=60, moneyness=(0.9, 1.1))
    frame, _ = snap.compact_chain(load_sample_chain().single(), narrow)
    assert (frame["expiry"] - frame["trade_date"]).dt.days.max() <= 60


def test_partial_failed_and_mislabelled_fetches_are_never_stored():
    partial = fake_fetch(FakeTicker(failures={"2026-10-23": -1}))("SPY", min_days=1.0, backoff=snap_backoff())
    assert partial.status.status == "partial"
    with pytest.raises(ValueError, match="PARTIAL"):
        snap.build_snapshot(partial)
    failed = fake_fetch(FakeTicker(failures={"options": -1}))("SPY", min_days=1.0, backoff=snap_backoff())
    with pytest.raises(ValueError, match="RATE_LIMITED"):
        snap.build_snapshot(failed)
    with pytest.raises(ValueError, match="served the 2026-10-21 session, not 2026-10-22"):
        snap.build_snapshot(fake_session_fetch("2026-10-21"), session="2026-10-22")


def snap_backoff():
    from derivx.data.market_data import Backoff

    return Backoff(retries=1, sleep=lambda s: None)


# -- store layout, index and idempotency -----------------------------------------------


def test_layout_and_index_and_append_only_idempotency(tmp_path):
    store = real_store(tmp_path, ["2026-10-20", "2026-10-21"])
    files = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert files == [
        "options/SPY/2026/SPY_2026-10-20.csv.gz",
        "options/SPY/2026/SPY_2026-10-20.json",
        "options/SPY/2026/SPY_2026-10-21.csv.gz",
        "options/SPY/2026/SPY_2026-10-21.json",
        "options/index.csv",
    ]
    index = store.index()
    assert list(index.columns) == list(snap.INDEX_COLUMNS)
    assert list(index["session"]) == ["2026-10-20", "2026-10-21"] and set(index["status"]) == {"ok"}
    assert list(index["path"]) == ["options/SPY/2026/SPY_2026-10-20.csv.gz", "options/SPY/2026/SPY_2026-10-21.csv.gz"]
    assert store.verify() == []
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    # the same session again (even with other quotes) is kept: published history never changes
    other = snap.build_snapshot(fake_session_fetch("2026-10-21"), snap.StoreLimits(max_days=30))
    assert store.add(other) == "kept"
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    cov = store.coverage("SPY")
    assert (cov.n, cov.first, cov.last, cov.gaps) == (2, pd.Timestamp("2026-10-20").date(),
                                                       pd.Timestamp("2026-10-21").date(), ())  # fmt: skip


def test_merge_copies_only_new_sessions_and_verify_finds_tampering(tmp_path):
    published = real_store(tmp_path / "branch", ["2026-10-20"])
    staging = real_store(tmp_path / "staging", ["2026-10-20", "2026-10-22"])
    original = published.file(snap.relative_path("SPY", "2026-10-20")).read_bytes()
    done = published.merge_from(staging)
    assert done == [("SPY", "2026-10-20", "kept"), ("SPY", "2026-10-22", "added")]
    assert published.file(snap.relative_path("SPY", "2026-10-20")).read_bytes() == original
    assert list(published.index()["session"]) == ["2026-10-20", "2026-10-22"] and published.verify() == []
    assert published.merge_from(staging) == [("SPY", "2026-10-20", "kept"), ("SPY", "2026-10-22", "kept")]
    cov = published.coverage("SPY")
    assert [d.isoformat() for d in cov.gaps] == ["2026-10-21"]
    # a staged file whose bytes do not match its manifest is refused
    bad = real_store(tmp_path / "bad", ["2026-10-23"])
    path = bad.file(snap.relative_path("SPY", "2026-10-23"))
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(snap.SnapshotError, match="do not match"):
        published.merge_from(bad)
    # verify: a changed file, a missing file and an unindexed file
    published.file(snap.relative_path("SPY", "2026-10-22")).write_bytes(b"x")
    (tmp_path / "branch" / "options" / "SPY" / "2026" / "SPY_2026-10-20.csv.gz").unlink()
    extra = real_store(tmp_path / "extra", ["2026-10-26"])
    for rel in (snap.relative_path("SPY", "2026-10-26"), snap.manifest_path(snap.relative_path("SPY", "2026-10-26"))):
        published.file(rel).write_bytes(extra.file(rel).read_bytes())
    problems = " | ".join(published.verify())
    assert "SPY_2026-10-22.csv.gz: SHA-256 differs" in problems
    assert "SPY_2026-10-20.csv.gz: listed in the index but missing" in problems
    assert "SPY_2026-10-26.csv.gz: stored but not in the index" in problems
    with pytest.raises(snap.SnapshotError, match="SHA-256 differs"):
        published.load("SPY", ["2026-10-22"])
    rows_before = published.index().iloc[:2].copy()
    published.rebuild_index()  # adds the unindexed file, never rewrites existing rows
    after = published.index()
    assert list(after["session"]) == ["2026-10-20", "2026-10-22", "2026-10-26"]
    pd.testing.assert_frame_equal(after.iloc[:2].reset_index(drop=True), rows_before.reset_index(drop=True))


def test_store_paths_cannot_escape_the_store(tmp_path):
    store = snap.SnapshotStore(tmp_path)
    for bad in ("../x.csv.gz", "/etc/passwd", "elsewhere/SPY.csv.gz"):
        with pytest.raises(snap.SnapshotError, match="not a path inside"):
            store.file(bad)
    with pytest.raises(ValueError, match="ticker"):
        snap.relative_path("SP Y;rm", "2026-10-20")


def test_load_merges_sessions_into_one_validated_history(tmp_path):
    store = synthetic_store(tmp_path, 4)
    ds = store.load("SYNTH")
    assert len(ds.chains) == 4 and ds.ticker == "SYNTH" and "collected history" in ds.source
    assert ds.provenance()["row_sources"] == ["synthetic"]
    assert len(ds.files) == 4 and all(f.startswith("options/SYNTH/2026/") for f in ds.files)
    two = store.load("SYNTH", [d.date() for d in ds.trade_dates[:2]])
    assert len(two.chains) == 2
    with pytest.raises(snap.SnapshotError, match="no stored session 2001-01-02"):
        store.load("SYNTH", ["2001-01-02"])


# -- when a session is due -----------------------------------------------------------


@pytest.mark.parametrize(
    "now, due",
    [
        ("2026-10-20T21:35:00Z", "2026-10-20"),  # 17:35 EDT: the cron time in summer
        ("2026-12-01T21:35:00Z", "2026-12-01"),  # 16:35 EST: the same cron in winter
        ("2026-10-24T03:30:00Z", "2026-10-23"),  # a Friday run delayed past midnight New York
        ("2026-10-20T20:20:00Z", "2026-10-19"),  # 16:20 EDT, before 16:25: the previous session ... is too old
    ],
)
def test_due_session_after_the_close_in_edt_and_est(now, due):
    session, reason = snap.due_session(pd.Timestamp(now))
    if now == "2026-10-20T20:20:00Z":
        assert session is None and "has not closed yet" in reason
    else:
        assert session is not None and session.isoformat() == due, reason


@pytest.mark.parametrize(
    "now, why",
    [
        ("2026-11-26T21:35:00Z", "2026-11-26 (Thursday) is not an NYSE trading day"),  # Thanksgiving
        ("2026-12-25T21:35:00Z", "2026-12-25 (Friday) is not an NYSE trading day"),  # Christmas
        ("2026-10-24T21:35:00Z", "2026-10-24 (Saturday) is not an NYSE trading day"),
        ("2026-10-21T14:00:00Z", "the 2026-10-21 session has not closed yet"),  # 10:00 EDT
        ("2026-10-21T12:00:00Z", "the 2026-10-21 session has not closed yet"),  # 08:00, before the open
    ],
)
def test_nothing_is_due_on_non_trading_days_or_during_a_session(now, why):
    session, reason = snap.due_session(pd.Timestamp(now))
    assert session is None and why in reason and "collection window" in reason


# -- the collector ----------------------------------------------------------------------


def test_collect_saves_the_due_session_then_keeps_it_without_any_request(tmp_path):
    fake = FakeTicker()
    fetch = fake_fetch(fake)
    report = snap.collect(["SPY"], tmp_path / "staging", fetch=fetch, now=AFTER_CLOSE, sleep=lambda s: None)
    (out,) = report.outcomes
    assert (out.state, out.session, report.exit_code) == ("saved", "2026-10-20", 0)
    assert out.rows < out.rows_in and out.path == "options/SPY/2026/SPY_2026-10-20.csv.gz"
    assert fake.requests[0] == "options" and "2026-10-20" not in fake.requests  # the day's own expiry: not requested
    n = len(fake.requests)
    published = snap.SnapshotStore(tmp_path / "staging", "the data branch")
    again = snap.collect(["SPY"], tmp_path / "staging2", published, fetch=fetch, now=AFTER_CLOSE)
    assert again.outcomes[0].state == "kept" and len(fake.requests) == n and again.exit_code == 0
    assert not (tmp_path / "staging2").exists()
    md = report.markdown()
    assert "| SPY | 2026-10-20 | **SAVED** |" in md and "Storage limits" in md


def test_collect_on_a_holiday_skips_without_a_request_or_a_file(tmp_path):
    fake = FakeTicker()
    report = snap.collect(["SPY", "QQQ"], tmp_path / "staging", fetch=fake_fetch(fake),
                          now=pd.Timestamp("2026-11-26T21:35:00Z"))  # fmt: skip
    assert [o.state for o in report.outcomes] == ["skipped", "skipped"] and report.exit_code == 0
    assert fake.requests == [] and not (tmp_path / "staging").exists()
    assert "not an NYSE trading day" in report.markdown()


def test_collect_retries_then_fails_visibly_and_writes_nothing(tmp_path):
    waits = []
    fake = FakeTicker(failures={"options": -1})
    fetch = fake_fetch(fake)
    report = snap.collect(["SPY"], tmp_path / "staging", fetch=fetch, now=AFTER_CLOSE, attempts=3, wait_s=60,
                          sleep=waits.append)  # fmt: skip
    (out,) = report.outcomes
    assert out.state == "failed" and out.attempts == 3 and fetch.calls == 3 and waits == [60.0, 120.0]
    assert "RATE_LIMITED" in out.reason and report.exit_code == 1
    assert not (tmp_path / "staging").exists()
    assert "**FAILED**" in report.markdown() and "nothing partial or mislabelled was written" in report.markdown()


def test_collect_recovers_on_a_later_attempt(tmp_path):
    fake = FakeTicker(failures={"options": 5})  # the first fetch's 5 tries of the listing fail, the second's work
    report = snap.collect(["SPY"], tmp_path / "staging", fetch=fake_fetch(fake), now=AFTER_CLOSE,
                          sleep=lambda s: None)  # fmt: skip
    out = report.outcomes[0]
    assert out.state == "saved" and out.attempts == 2 and "after 1 failed attempt" in out.reason


def test_collect_never_stores_a_partial_chain_or_another_session(tmp_path):
    partial = snap.collect(["SPY"], tmp_path / "a", fetch=fake_fetch(FakeTicker(failures={"2026-10-23": -1})),
                           now=AFTER_CLOSE, attempts=2, sleep=lambda s: None)  # fmt: skip
    assert partial.outcomes[0].state == "failed" and "PARTIAL" in partial.outcomes[0].reason
    assert not (tmp_path / "a").exists()
    stale = FakeTicker(spot_time=spot_time("2026-10-19"))  # Yahoo still serving Monday's chain on Tuesday evening
    wrong = snap.collect(["SPY"], tmp_path / "b", fetch=fake_fetch(stale), now=AFTER_CLOSE, attempts=1)
    out = wrong.outcomes[0]
    assert out.state == "failed" and "served the 2026-10-19 session, not 2026-10-20" in out.reason
    assert not (tmp_path / "b").exists() and wrong.exit_code == 1


def test_one_failing_ticker_fails_the_run_but_the_complete_ones_are_kept(tmp_path):
    good = fake_fetch(FakeTicker())
    bad = fake_fetch(FakeTicker(failures={"options": -1}))

    def fetch(ticker, **kw):
        return (good if ticker == "SPY" else bad)(ticker, **kw)

    report = snap.collect(["SPY", "QQQ"], tmp_path / "s", fetch=fetch, now=AFTER_CLOSE, attempts=1)
    assert [o.state for o in report.outcomes] == ["saved", "failed"] and report.exit_code == 1
    assert snap.SnapshotStore(tmp_path / "s").tickers() == ["SPY"]


# -- walk-forward history needed -------------------------------------------------------------


def test_scorable_origins_counts_pairs_one_horizon_apart():
    days = ["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-21"]  # 10-20 missing
    assert snap.scorable_origins(days, 1) == 3  # 14->15, 15->16, 16->19 (19->20 and 21->22 missing)
    assert snap.scorable_origins(days, 2) == 3  # 14->16, 15->19, 19->21
    assert snap.scorable_origins(days[:1], 1) == 0
    assert snap.origins_needed() == 9 and snap.origins_needed(2) == 6


# -- command line ---------------------------------------------------------------------------


def test_cli_collect_on_a_holiday_writes_the_summary_and_exits_0(tmp_path, capsys):
    summary = tmp_path / "summary.md"
    code = cli.main(["snapshots", "collect", "--tickers", "SPY", "--staging", str(tmp_path / "s"), "--summary",
                     str(summary), "--report", str(tmp_path / "r.json"), "--now", "2026-11-26T21:35:00Z"])  # fmt: skip
    assert code == 0 and "SKIPPED" in capsys.readouterr().out
    assert "not an NYSE trading day" in summary.read_text()
    assert json.loads((tmp_path / "r.json").read_text())["exit_code"] == 0


def test_cli_rejects_bad_tickers(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["snapshots", "collect", "--tickers", "SPY;rm -rf", "--staging", "x"])
    assert exc.value.code == 2 and "give tickers as SPY" in capsys.readouterr().err


def test_cli_merge_index_and_list_round_trip(tmp_path, capsys):
    branch = tmp_path / "branch"
    real_store(tmp_path / "staging", ["2026-10-20", "2026-10-22"])
    assert cli.main(["snapshots", "merge", str(tmp_path / "staging"), str(branch)]) == 0
    out = capsys.readouterr().out
    assert "added SPY 2026-10-20" in out and "2 file(s) added" in out
    assert cli.main(["snapshots", "merge", str(tmp_path / "staging"), str(branch)]) == 0
    assert "0 file(s) added" in capsys.readouterr().out
    assert cli.main(["snapshots", "index", "--store", str(branch), "--check"]) == 0
    assert cli.main(["snapshots", "list", "--local", str(branch), "--sessions"]) == 0
    out = capsys.readouterr().out
    assert "SPY: 2 session(s) 2026-10-20 .. 2026-10-22; 1 missing trading day(s): 2026-10-21" in out
    assert "SPY: 2026-10-20 2026-10-22" in out
    (branch / "options" / "SPY" / "2026" / "SPY_2026-10-22.csv.gz").write_bytes(b"tampered")
    assert cli.main(["snapshots", "index", "--store", str(branch), "--check"]) == 1
    assert "SHA-256 differs" in capsys.readouterr().err
    assert cli.main(["snapshots", "list", "--local", str(tmp_path / "nothing")]) == 2


def test_download_tool_writes_the_compact_store(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import download_spy_chains as tool
    finally:
        sys.path.remove(str(ROOT / "tools"))
    fetch = fake_fetch(FakeTicker())

    def wrapped(ticker, **kw):
        from derivx.data.market_data import Backoff

        kw["backoff"] = Backoff(retries=1)
        kw.pop("max_expiries", None)
        kw.pop("expiries", None)
        return fetch(ticker, min_days=1.0, **kw)

    code = tool.main(["--store", str(tmp_path)], fetch=wrapped, now=pd.Timestamp("2026-10-20T21:00:00Z"))
    assert code == 0 and "added 8,352 of" in capsys.readouterr().out
    assert snap.SnapshotStore(tmp_path).verify() == [] and snap.SnapshotStore(tmp_path).sessions("SPY")
    assert tool.main(["--store", str(tmp_path)], fetch=wrapped, now=pd.Timestamp("2026-10-20T21:00:00Z")) == 0
    assert "kept" in capsys.readouterr().out
