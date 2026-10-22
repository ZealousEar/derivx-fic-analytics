"""tools/compute_metrics.py history mode: the walk-forward summary of a
collected store (a local checkout or the synced copy) is reproducible byte
for byte, labels SYNTHETIC rows, records every file's SHA-256, and says how
many origin days are missing when the history is too short."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _snapshot_store import synthetic_store

from derivx.data import snapshots as snap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import compute_metrics as cm  # noqa: E402

FAST = ["--max-days", "20"]


def _outputs(path: Path):
    return {p.name: p.read_bytes() for p in sorted(path.iterdir()) if p.name != "manifest.json"}


def test_history_mode_is_reproducible_and_complete(tmp_path, capsys):
    store = synthetic_store(tmp_path / "branch", 10)
    args = ["--store", str(store.root), "--ticker", "SYNTH", *FAST]
    assert cm.main([*args, "--outdir", str(tmp_path / "a")]) == 0
    assert cm.main([*args, "--outdir", str(tmp_path / "b")]) == 0
    a, b = _outputs(tmp_path / "a"), _outputs(tmp_path / "b")
    assert a == b  # byte for byte, everything but the manifest's run time
    assert {"walkforward_summary.csv", "walkforward_daily.csv", "walkforward_by_regime.csv", "sessions.csv",
            "summary.json", "METRICS_SUMMARY.md"} <= set(a)  # fmt: skip
    summary = json.loads(a["summary.json"])
    assert summary["mode"] == "history" and summary["walk_forward"]["status"] == "ok"
    assert summary["coverage"] == {"sessions": 10, "missing_trading_days": [], "origin_days": 9,
                                   "origin_days_needed": 9}  # fmt: skip
    assert summary["walk_forward"]["target_days"] == 9
    assert summary["settings"]["discounting"]["mode"] == "curve"
    md = a["METRICS_SUMMARY.md"].decode()
    assert md.startswith("# Walk-forward on collected history") and "SYNTHETIC DATA, NOT MARKET DATA" in md
    assert "| svi_sticky_strike |" in md and "Origin days at the 1-day horizon: 9 (needed: 9)" in md
    # one discount curve per session: the summary names the first and the last, summary.json keeps them all
    curves = summary["settings"]["discounting"]["curves"]
    labels = sorted({c["label"] for c in curves.values()})
    assert len(curves) == 10 and len(labels) == 9  # the last session discounts on the latest curve, of the day before
    assert f"Discounting: 9 curves, {labels[0]} .. {labels[-1]} (trade date to expiry; 10 trade date(s))" in md
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    files = manifest["data"]["files"]
    assert len(files) == 10 and all(len(f["sha256"]) == 64 for f in files)
    assert files == [{"path": p, "sha256": h} for p, h in zip(store.index()["path"], store.index()["sha256"])]
    assert "walk-forward: ok" in capsys.readouterr().out


def test_a_short_history_says_how_many_origin_days_are_missing(tmp_path, capsys):
    store = synthetic_store(tmp_path / "branch", 5)
    assert cm.main(["--store", str(store.root), "--ticker", "SYNTH", "--outdir", str(tmp_path / "o"), *FAST]) == 0
    summary = json.loads((tmp_path / "o" / "summary.json").read_text())
    wf = summary["walk_forward"]
    assert wf["status"] == "not run" and "5 session(s) give 4 origin day(s)" in wf["reason"]
    assert "collect at least 5 more trading day(s)" in wf["reason"]
    assert not (tmp_path / "o" / "walkforward_summary.csv").exists()
    assert "not run: 5 session(s)" in (tmp_path / "o" / "METRICS_SUMMARY.md").read_text()


def test_history_mode_reads_the_synced_copy_offline_and_a_session_range(tmp_path, monkeypatch):
    root = snap.sync_root("owner/repo", "data-snapshots")
    synthetic_store(root, 12)
    out = tmp_path / "o"
    code = cm.main(["--history", "--offline", "--repo", "owner/repo", "--ticker", "SYNTH", "--first", "2026-10-06",
                    "--outdir", str(out), *FAST])  # fmt: skip
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["collection"]["source"] == "owner/repo@data-snapshots"
    assert summary["collection"]["first"] == "2026-10-06" and summary["coverage"]["sessions"] == 10
    assert cm.main(["--history", "--offline", "--repo", "nobody/none", "--outdir", str(out)]) == 1
    assert cm.main(["--store", str(root), "--ticker", "SYNTH", "--first", "2030-01-01", "--outdir", str(out)]) == 1


def test_outputs_do_not_depend_on_where_or_when_the_store_was_synced(tmp_path, capsys):
    """Another checkout directory, or the same synced copy synced again later,
    gives the same outputs byte for byte; where and when go to the manifest."""

    import shutil

    store = synthetic_store(tmp_path / "branch", 10)
    copy = tmp_path / "elsewhere" / "data-snapshots"
    shutil.copytree(store.root, copy)
    runs = {}
    for name, root in (("a", store.root), ("b", copy)):
        assert cm.main(["--store", str(root), "--ticker", "SYNTH", *FAST, "--outdir", str(tmp_path / name)]) == 0
        runs[name] = _outputs(tmp_path / name)
    assert runs["a"] == runs["b"]
    manifest = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert manifest["data"]["sync"] == {"local_checkout": str(copy)}
    # the synced copy, read twice with a re-sync in between (sync.json records a new time)
    root = snap.sync_root("owner/repo", "data-snapshots")
    shutil.copytree(store.root, root)
    outs = []
    for i, when in enumerate(("2026-10-21T22:40:00Z", "2026-10-22T22:41:00Z")):
        (root / snap.SYNC_META).write_text(json.dumps({"synced_at": when}))
        out = tmp_path / f"synced{i}"
        args = ["--history", "--offline", "--repo", "owner/repo", "--ticker", "SYNTH", *FAST, "--outdir", str(out)]
        assert cm.main(args) == 0
        outs.append(_outputs(out))
        assert json.loads((out / "manifest.json").read_text())["data"]["sync"]["synced_at"] == when
    assert outs[0] == outs[1]
    assert json.loads(outs[0]["summary.json"])["collection"]["source"] == "owner/repo@data-snapshots"
