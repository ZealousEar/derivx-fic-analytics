"""Run manifests and result bundles (G14 (2), (3))."""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd

from derivx.validation.manifest import build_manifest, git_revision, jsonable, manifest_bytes, results_zip


def test_manifest_records_code_packages_inputs_data_model_and_seed():
    m = build_manifest(
        "calibration",
        inputs={"filters": {"min_volume": 10}},
        data={"source": "bundled sample", "fetched_at_utc": "2026-10-20T20:59:39Z", "data_sha256": "ab" * 32},
        model={"enforce_arbitrage": True},
        seed=7,
        created_at=dt.datetime(2026, 10, 20, 21, 0, tzinfo=dt.timezone.utc),
    )
    assert m["kind"] == "calibration" and m["created_at_utc"] == "2026-10-20T21:00:00+00:00"
    sha = m["code"]["sha"]
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)
    assert sha == git_revision()["sha"]
    for pkg in ("python", "numpy", "pandas", "scipy", "streamlit", "yfinance"):
        assert pkg in m["packages"]
    assert m["data"]["data_sha256"] == "ab" * 32 and m["inputs"]["filters"]["min_volume"] == 10
    assert m["model"] == {"enforce_arbitrage": True} and m["seed"] == 7
    json.loads(manifest_bytes(m))  # strict JSON (no NaN)


def test_jsonable_handles_numpy_timestamps_and_non_finite():
    out = jsonable(
        {"a": np.float64(np.nan), "b": np.int64(3), "c": pd.Timestamp("2026-10-20"), "d": np.array([1.0, np.inf])}
    )
    assert out == {"a": None, "b": 3, "c": "2026-10-20T00:00:00", "d": [1.0, None]}


def test_results_zip_is_deterministic_and_holds_manifest_and_csvs():
    m = build_manifest("x", inputs={}, data={}, model={}, created_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    tables = {"params": pd.DataFrame({"a": [1.0, 2.0]}), "metrics": pd.DataFrame({"rmse_bp": [3.5]})}
    z1, z2 = results_zip(m, tables), results_zip(m, tables)
    assert z1 == z2
    with zipfile.ZipFile(io.BytesIO(z1)) as zf:
        assert sorted(zf.namelist()) == ["manifest.json", "metrics.csv", "params.csv"]
        assert json.loads(zf.read("manifest.json"))["kind"] == "x"
        assert zf.read("params.csv").decode().splitlines() == ["a", "1.0", "2.0"]


def test_outside_a_checkout_the_commit_comes_from_the_image_environment(monkeypatch):
    from derivx.validation import manifest

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")  # the Docker image has neither git nor .git

    monkeypatch.setattr(manifest.subprocess, "run", no_git)
    try:
        for value, expected in [
            ("0A1751437A7BC779B670DD3ACCBFBFE0A0AEEB2C", "0a1751437a7bc779b670dd3accbfbfe0a0aeeb2c"),
            ("unknown", "unknown"),  # the Dockerfile default when no GIT_SHA build arg is given
            ("abc; rm -rf /", "unknown"),  # anything but a commit id is not recorded
        ]:
            monkeypatch.setenv(manifest.GIT_SHA_ENV, value)
            manifest.git_revision.cache_clear()
            assert manifest.git_revision() == {"sha": expected, "dirty": None}
        monkeypatch.delenv(manifest.GIT_SHA_ENV)
        manifest.git_revision.cache_clear()
        assert manifest.git_revision() == {"sha": "unknown", "dirty": None}
    finally:
        manifest.git_revision.cache_clear()
