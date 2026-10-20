"""``derivx`` command line: pull into the cache, offline reads, exports and
errors (recorded fixtures stand in for the network)."""

from __future__ import annotations

import io
import subprocess
import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from derivx import cli
from derivx.data import rates_history as rh
from derivx.data import rates_sources as rs

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "rates"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def online(monkeypatch):
    """Serve recorded fixtures for every source the CLI asks for."""

    routes = {
        "fredgraph.csv": "fred_ust.csv",
        "newyorkfed": "nyfed_sofr_last3.json",
        "data/EST": "ecb_estr.csv",
        "data/YC": "ecb_yc_par.csv",
        "yield-curves": "boe_latest.zip",
        "treasury.gov": "treasury_2026.csv",
    }
    urls = []

    def fake(request, timeout):
        urls.append(request.full_url)
        for needle, name in routes.items():
            if needle in request.full_url:
                return _Response((FIXTURES / name).read_bytes())
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(rs, "_urlopen", fake)
    monkeypatch.setenv("DERIVX_OFFLINE", "0")
    return urls


def test_pull_downloads_caches_and_exports(online, tmp_path, capsys):
    out = tmp_path / "dgs.csv"
    code = cli.main(["pull", "fred", "--ids", "DGS2,DGS10", "--start", "2026-10-13", "--cache", str(tmp_path / "c"),
                     "--out", str(out), "--show", "2"])  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0 and "FRED: 8 rows downloaded" in text and "sha256" in text
    assert "DGS10" in text and "last 5.09 percent (par_yield)" in text
    written = pd.read_csv(out)
    assert list(written.columns) == list(rs.LONG_COLUMNS) and len(written) == 8
    # Offline: the same request from the cache, no network.
    online.clear()
    code = cli.main(["pull", "fred", "--ids", "DGS2,DGS10", "--start", "2026-10-13", "--cache", str(tmp_path / "c"),
                     "--offline", "--out", str(tmp_path / "dgs.parquet")])  # fmt: skip
    assert code == 0 and online == [] and "from the cache" in capsys.readouterr().out
    assert len(pd.read_parquet(tmp_path / "dgs.parquet")) == 8


@pytest.mark.parametrize(
    "argv, expect",
    [
        (["pull", "nyfed", "--rate", "sofr", "--last", "3"], "NY Fed: 3 rows"),
        (["pull", "ecb-estr", "--last", "3"], "ESTR"),
        (["pull", "ecb-yc", "--tenors", "3M,2Y,10Y,30Y", "--last", "2"], "PY_10Y"),
        (["pull", "boe-ois", "--archive", "latest"], "BoE OIS spot 25Y"),
        (["pull", "treasury", "--year", "2026"], "US Treasury: 70 rows"),
    ],
)
def test_pull_every_source(online, tmp_path, capsys, argv, expect):
    assert cli.main(argv + ["--cache", str(tmp_path)]) == 0
    assert expect in capsys.readouterr().out


def test_pull_errors(tmp_path, capsys):
    assert cli.main(["pull", "fred", "--ids", "DGS10", "--offline", "--cache", str(tmp_path)]) == cli.EXIT_SOURCE_ERROR
    assert "offline: no cached FRED data" in capsys.readouterr().err
    assert cli.main(["pull", "nyfed", "--rate", "sofr", "--last", "0", "--cache", str(tmp_path)]) == 2
    assert "last_n must be at least 1" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["pull", "fred", "--ids", "DGS10", "--start", "13/10/2026"])
    assert exit_info.value.code == 2 and "use YYYY-MM-DD" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["pull", "nowhere"])


def test_pull_latest(monkeypatch, tmp_path, capsys):
    result = rh.LiveRefresh(ok={"SOFR (NY Fed)": True, "UK OIS curve (BoE)": False}, notices=["UK OIS curve: down"])
    monkeypatch.setattr(rh, "refresh_live", lambda **kw: result)
    assert cli.main(["pull", "latest", "--cache", str(tmp_path)]) == cli.EXIT_SOURCE_ERROR
    text = capsys.readouterr().out
    assert "ok   SOFR (NY Fed)" in text and "FAIL UK OIS curve (BoE)" in text and "note: UK OIS curve: down" in text


def test_commands_register_themselves_and_run_as_a_module():
    assert "pull" in cli.COMMANDS
    parser = cli.build_parser()
    assert parser.parse_args(["pull", "fred", "--ids", "DGS2"]).ids == ["DGS2"]
    grouped = parser.parse_args(["pull", "fred", "--ids", "ust,DGS10,ice"]).ids
    assert grouped == list(rs.UST_FRED_IDS) + list(rs.ICE_SWAP_FRED_IDS)  # groups expand, duplicates drop
    done = subprocess.run(
        [sys.executable, "-m", "derivx", "pull", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "DERIVX_OFFLINE": "1"},
        timeout=120,
    )
    assert done.returncode == 0 and "boe-ois" in done.stdout and "treasury" in done.stdout
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'derivx = "derivx.cli:main"' in pyproject
