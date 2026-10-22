"""tools/compute_metrics.py: reproducible single-snapshot metrics on the
committed real SPY sample (AL-01, critic-1, VM-09). The committed summary in
docs/metrics/ must be what the code produces from the committed data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivx.data.samples import load_sample_chain

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import compute_metrics as cm  # noqa: E402

COMMITTED = ROOT / "docs" / "metrics" / "spy_2026-10-20"
EXPIRIES = ["2026-11-20", "2027-01-15"]


@pytest.fixture(scope="module")
def quick():
    return cm.compute(load_sample_chain(), expiries=EXPIRIES)


def test_committed_summary_belongs_to_the_committed_sample():
    summary = json.loads((COMMITTED / "summary.json").read_text())
    assert summary["data"]["data_sha256"] == load_sample_chain().fingerprint
    assert summary["data"] == cm.jsonable(load_sample_chain().provenance())  # regenerated with the current code
    assert summary["data"]["trade_dates"] == ["2026-10-20"] and summary["snapshots"] == 1
    mc = summary["model_comparison"]
    assert mc["slices_scored"] + mc["slices_skipped"] == 31  # every expiry scored or reported
    assert summary["walk_forward"]["status"] == "not run"
    assert "one trading day apart" in summary["walk_forward"]["reason"]
    md = (COMMITTED / "METRICS_SUMMARY.md").read_text()
    assert "outperforms" not in md and all(s in md for s in mc["statements"])
    manifest = json.loads((COMMITTED / "manifest.json").read_text())
    assert manifest["data"]["data_sha256"] == summary["data"]["data_sha256"]
    # AL-14: discounted with the Treasury curve of the trade date, not an unstated 0% rate
    disc = summary["settings"]["discounting"]
    assert disc["mode"] == "curve" and "rate_cc" not in summary["settings"]
    assert disc["curves"]["2026-10-20"]["label"].startswith("US Treasury par curve of 2026-10-19")
    assert "Discounting: US Treasury par curve of 2026-10-19" in md and "American-style" in md
    scores = pd.read_csv(COMMITTED / "model_scores.csv")
    assert (scores["discount_method"] == "curve").all() and scores["rate_pct"].between(4.0, 4.8).all()


def test_scores_reproduce_the_committed_numbers(quick):
    committed = pd.read_csv(COMMITTED / "model_scores.csv")
    committed = committed[committed["expiry"].isin(EXPIRIES)].reset_index(drop=True)
    mine = quick["scores"].copy()
    mine["expiry"] = pd.to_datetime(mine["expiry"]).dt.strftime("%Y-%m-%d")
    assert len(mine) == len(committed) == 6
    for col in ("rmse_bp", "rmse_cv_bp", "max_abs_error_bp", "forward"):
        np.testing.assert_allclose(mine[col].to_numpy(float), committed[col].to_numpy(float), rtol=1e-4, atol=1e-3)
    assert mine["status"].tolist() == committed["status"].tolist()


def test_report_is_worded_by_sign_and_explains_the_missing_walk_forward(quick):
    summary = cm.jsonable(quick["summary"])
    md = cm.report_markdown(summary)
    for key, res in summary["model_comparison"]["pairwise"].items():
        a, b = key.split("_vs_")
        winner = cm.LABEL[a] if res["a_better"] > res["b_better"] else cm.LABEL[b]
        assert f"{winner} has the lower out-of-sample RMSE" in md
    assert "a walk-forward test needs daily snapshots" in md


def test_main_writes_every_output(tmp_path):
    code = cm.main(["--outdir", str(tmp_path), "--expiries", "2026-11-20", "--no-calibration"])
    assert code == 0
    names = {p.name for p in tmp_path.iterdir()}
    assert {"model_scores.csv", "skipped.csv", "summary.json", "METRICS_SUMMARY.md", "manifest.json"} <= names
    bad = tmp_path / "bad.csv"
    bad.write_text("strike,iv\n100,0.2\n")
    assert cm.main(["--chain", str(bad), "--outdir", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()


def test_a_multi_snapshot_synthetic_history_is_labelled_and_scored_by_days():
    """Files of synthetic rows are named as such even when the dataset is
    just 'N file(s)', a pooled run is not called 'single-snapshot', and the
    walk-forward table counts days won (not only quote-level wins)."""

    from derivx.data.chain_schema import merge_datasets, validate_chain
    from derivx.data.synthetic import synthetic_history

    days = synthetic_history(2)
    files = [
        validate_chain(df[df["expiry"] <= td + pd.Timedelta(days=8)], f"d{i}.csv")
        for i, (td, df) in enumerate(days.items())
    ]
    result = cm.compute(merge_datasets(files, source="2 file(s)"), calibration=False, max_days=8)
    summary = cm.jsonable(result["summary"])
    md = cm.report_markdown(summary)
    assert summary["data"]["row_sources"] == ["synthetic"]
    assert "**SYNTHETIC DATA, NOT MARKET DATA**" in md and "single-snapshot" not in md
    assert summary["walk_forward"]["status"] == "ok" and "| days won |" in md
    svi = next(r for r in summary["walk_forward"]["summary"] if r["predictor"] == "svi_sticky_strike")
    assert svi["day_win_n"] == 1 and svi["benchmark"] == "persistence"
