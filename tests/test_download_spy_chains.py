"""tools/download_spy_chains.py never labels today's chain with a past date
(AL-02) and saves each session once, with its fetch status (AL-14)."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from _fake_yahoo import Clock, FakeTicker

from derivx.data.chain_schema import load_chain_file
from derivx.data.market_data import Backoff, fetch_option_chain

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import download_spy_chains as tool  # noqa: E402

NOW = pd.Timestamp("2026-10-20T21:00:00Z")


def _fetcher(fake):
    calls = []

    def fetch(ticker, **kw):
        calls.append(ticker)
        kw["backoff"] = Backoff(sleep=lambda s: None)
        return fetch_option_chain(ticker, client_factory=lambda s: fake, now=Clock(), request_pause=0.0, **kw)

    return fetch, calls


@pytest.mark.parametrize("flag", ["--start", "--end", "--date"])
def test_past_dates_are_refused_without_any_request(flag, tmp_path, capsys):
    fetch, calls = _fetcher(FakeTicker())
    code = tool.main([flag, "2026-09-01", "--outdir", str(tmp_path)], fetch=fetch, now=NOW)
    assert code == 1 and calls == [] and not any(tmp_path.iterdir())
    assert "only the current chain" in capsys.readouterr().err


def test_saves_one_file_per_session_with_true_timestamps(tmp_path, capsys):
    fetch, calls = _fetcher(FakeTicker())
    args = ["--max-expiries", "3", "--outdir", str(tmp_path), "--format", "csv.gz"]
    assert tool.main(args, fetch=fetch, now=NOW) == 0
    data, meta = tmp_path / "spy_2026-10-20.csv.gz", tmp_path / "spy_2026-10-20.json"
    assert data.exists() and meta.exists()
    ds = load_chain_file(data.name, data.read_bytes())
    chain = ds.single()
    assert chain["expiry"].nunique() == 3 and ds.fetched_at_utc is not None and ds.ticker == "SPY"
    status = json.loads(meta.read_text())
    assert status["fetch_status"]["quote_date"] == "2026-10-20" and status["fetch_status"]["status"] == "ok"
    assert status["manifest"]["kind"] == "snapshot" and "yfinance" in status["manifest"]["packages"]
    # a second run the same session keeps the first file
    before = data.read_bytes()
    assert tool.main(args, fetch=fetch, now=NOW) == 0
    assert data.read_bytes() == before and "kept the existing file" in capsys.readouterr().out
    assert gzip.decompress(before).startswith(b"trade_date,expiry,option_type,strike")


def test_partial_and_failed_fetches(tmp_path):
    fetch, _ = _fetcher(FakeTicker(failures={"2026-10-21": -1}))
    assert tool.main(["--max-expiries", "3", "--outdir", str(tmp_path)], fetch=fetch, now=NOW) == 3
    assert (tmp_path / "spy_2026-10-20.parquet").exists()
    fetch, _ = _fetcher(FakeTicker(failures={"options": -1}))
    out = tmp_path / "none"
    assert tool.main(["--outdir", str(out)], fetch=fetch, now=NOW) == 1 and not out.exists()


def test_current_session_rolls_back_over_weekends():
    assert tool.current_session(pd.Timestamp("2026-10-18T15:00:00Z")) == pd.Timestamp("2026-10-16")
    assert tool.refuse_past_dates(["2026-10-16"], pd.Timestamp("2026-10-18T15:00:00Z")) is None
