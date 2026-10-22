"""ANALYTICS lab, *Collected history* (the daily snapshots on the
data-snapshots branch): coverage and provenance, the walk-forward held back
until the history has enough origin days (and run once it has), the synced
cache as the default source, a sync that fails with a reason, and any
collected session as the MARKET DATA tab's active dataset. Stores are built
from the SYNTHETIC history or the recorded fake Yahoo chain; no network."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _snapshot_store import real_store, synthetic_store
from streamlit.testing.v1 import AppTest

from derivx.data import snapshots as snap
from derivx.ui import analytics_lab as lab
from derivx.ui import lab_history

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"


def _lab(**state) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=300)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    at.sidebar.radio[0].set_value("[ ANALYTICS ]").run()
    assert not at.exception
    return at


def _collected(at: AppTest) -> AppTest:
    at.radio(key="lab_wf_source").set_value(lab_history.SOURCE_LABEL).run()
    assert not at.exception
    return at


def _texts(at: AppTest) -> str:
    return " ".join(str(m.value) for m in [*at.markdown, *at.warning, *at.info, *at.error, *at.success, *at.caption])


def test_without_a_synced_copy_the_panel_explains_how_to_get_one():
    at = _collected(_lab())
    text = _texts(at)
    assert "No collected history here yet" in text and "has not been synced on this server" in text
    assert os.environ["DERIVX_CACHE_DIR"] not in text  # no server path on the page
    assert not [b for b in at.button if b.key == "lab_hist_use"]
    # SYNC NOW goes to the network, which the tests block: the failure says why, and nothing breaks
    at.button(key="lab_hist_sync").click().run()
    assert not at.exception
    assert "Sync failed" in _texts(at) and "raw.githubusercontent.com unreachable" in _texts(at)


def test_too_few_sessions_show_coverage_and_how_many_are_needed(tmp_path, monkeypatch):
    synthetic_store(tmp_path / "branch", 4)
    monkeypatch.setenv(lab_history.LOCAL_ENV, str(tmp_path / "branch"))
    at = _collected(_lab())
    text = _texts(at)
    assert "COLLECTED HISTORY" in text and "4 session(s) 2026-10-14 .. 2026-10-19" in text
    assert "local checkout" in text and "no missing trading days" in text
    assert "give 3 origin day(s) with the session 1 trading day(s) later" in text
    assert "waits for 9" in text and "at least 6 more trading day(s)" in text
    assert not [b for b in at.button if b.key == "lab_hist_use"]
    assert lab.K_WF_DATA not in at.session_state


def test_enough_sessions_run_the_walk_forward_with_provenance(tmp_path, monkeypatch):
    synthetic_store(tmp_path / "branch", 10)
    monkeypatch.setenv(lab_history.LOCAL_ENV, str(tmp_path / "branch"))
    at = _collected(_lab())
    at.button(key="lab_hist_use").click().run()
    assert not at.exception
    entry = at.session_state[lab.K_WF_DATA]
    assert entry["origin"] == "collected" and len(entry["dataset"].chains) == 10
    assert entry["history"]["sessions"] == 10 and len(entry["history"]["index_sha256"]) == 64
    at.number_input(key="lab_wf_maxdays").set_value(20).run()
    run = at.button(key="lab_wf_run")
    assert not run.disabled
    run.click().run()
    assert not at.exception
    stored = at.session_state[lab.K_WF]
    assert not stored["result"].empty and stored["result"].daily["target"].nunique() == 9
    assert stored["manifest"]["inputs"]["collected_history"]["sessions"] == 10
    # a 2-day horizon leaves 8 origin days: the run waits
    at.number_input(key="lab_wf_horizon").set_value(2).run()
    assert at.button(key="lab_wf_run").disabled
    assert any("give 8 origin day(s)" in w.value for w in at.warning)


def test_the_synced_cache_is_the_default_source(tmp_path, monkeypatch):
    monkeypatch.delenv(lab_history.LOCAL_ENV, raising=False)
    root = snap.sync_root(snap.default_repo(), snap.default_branch())
    synthetic_store(root, 3)
    (root / snap.SYNC_META).write_text(json.dumps({"synced_at": "2026-10-21T22:40:00Z"}))
    at = _collected(_lab())
    text = _texts(at)
    assert "source: ZealousEar/derivx-fic-analytics@data-snapshots" in text
    assert "synced 2026-10-21T22:40:00Z" in text and "3 session(s)" in text


@pytest.mark.parametrize("day", ["2026-10-21", "2026-10-20"])
def test_market_data_loads_any_collected_session(tmp_path, monkeypatch, day):
    real_store(tmp_path / "branch", ["2026-10-20", "2026-10-21"])
    monkeypatch.setenv(lab_history.LOCAL_ENV, str(tmp_path / "branch"))
    at = _lab()
    at.selectbox(key="lab_hist_md_day_SPY").set_value(day).run()
    at.button(key="lab_hist_md_load").click().run()
    assert not at.exception
    active = at.session_state[lab.K_DATA]
    ds = active["dataset"]
    assert active["origin"] == "collected" and [d.date().isoformat() for d in ds.trade_dates] == [day]
    assert "collected history" in ds.source and ds.ticker == "SPY"
    assert active["status"]["fetch_status"]["quote_date"] == day
    assert any("Collected session" in s.value and "quotes kept" in s.value for s in at.success)
    banner = " ".join(m.value for m in at.markdown if "ACTIVE DATASET" in m.value)
    assert f"session {day}" in banner and "collected history" in banner


def test_the_banner_says_what_the_rows_are_from_their_manifests(tmp_path, monkeypatch):
    synthetic_store(tmp_path / "synthetic", 3)
    monkeypatch.setenv(lab_history.LOCAL_ENV, str(tmp_path / "synthetic"))
    text = _texts(_collected(_lab()))
    assert "SYNTHETIC (GENERATED, NOT MARKET DATA)" in text and "Yahoo Finance delayed quotes (real" not in text
    real_store(tmp_path / "real", ["2026-10-20", "2026-10-21"])
    monkeypatch.setenv(lab_history.LOCAL_ENV, str(tmp_path / "real"))
    assert "Yahoo Finance delayed quotes (real, one file per session)" in _texts(_collected(_lab()))


def test_the_walk_forward_runs_on_the_most_recent_sessions(tmp_path, monkeypatch):
    store = synthetic_store(tmp_path / "branch", 14)
    monkeypatch.setenv(lab_history.LOCAL_ENV, str(tmp_path / "branch"))
    at = _collected(_lab())
    assert at.selectbox(key=lab_history.WINDOW_KEY).value == 30  # the default: every one of the 14 here
    at.selectbox(key=lab_history.WINDOW_KEY).set_value(10).run()
    assert not at.exception
    stats = " ".join(m.value for m in at.markdown if "dx-lab-stats" in m.value)
    assert "<span>SESSIONS</span><b>14</b>" in stats and "<span>IN THE RUN</span><b>10</b>" in stats
    at.button(key="lab_hist_use").click().run()
    assert not at.exception
    entry = at.session_state[lab.K_WF_DATA]
    days = [d.isoformat() for d in store.sessions("SYNTH")]
    assert [d.date().isoformat() for d in entry["dataset"].trade_dates] == days[-10:]
    assert (entry["history"]["sessions"], entry["history"]["sessions_collected"]) == (10, 14)
    assert entry["history"]["first"] == days[-10] and "SYNTHETIC" in entry["history"]["data"]


def test_a_window_too_short_for_the_walk_forward_says_to_pick_more_sessions(tmp_path, monkeypatch):
    store = synthetic_store(tmp_path / "branch", 14)
    index = store.index()
    gone = index.iloc[[-3]]  # a missing day inside the 10 most recent sessions leaves 8 origin days in them
    for path in gone["path"]:
        (store.root / path).unlink()
    store.write_index(index.drop(gone.index))
    monkeypatch.setenv(lab_history.LOCAL_ENV, str(store.root))
    at = _collected(_lab())
    at.selectbox(key=lab_history.WINDOW_KEY).set_value(10).run()
    assert any("The most recent 10 of the 13 collected sessions" in w.value and "pick more SESSIONS" in w.value
               for w in at.warning)  # fmt: skip
    assert not [b for b in at.button if b.key == "lab_hist_use"]
    at.selectbox(key=lab_history.WINDOW_KEY).set_value("all").run()
    assert [b for b in at.button if b.key == "lab_hist_use"]  # 13 sessions with one gap: 11 origin days
