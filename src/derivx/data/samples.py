"""The bundled option-chain sample (offline data for the ANALYTICS lab).

``data/examples/spy_<session>.csv.gz`` is ONE real SPY option chain
from Yahoo Finance (delayed quotes, fetched through yfinance by
``tools/download_spy_chains.py``), unfiltered: every listed expiry, calls
and puts, with each row's real fetch time (``fetched_at_utc``) and the spot
read in the same response. The sidecar ``.json`` is the fetch status
(request counts, failed expiries, notes). It is a single snapshot: there is
no history in it, so it supports calibration and model comparison but not
a walk-forward test.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from derivx.data.chain_schema import ChainDataset, load_chain_file

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"
SAMPLE_SESSION = "2026-10-20"
SAMPLE_CHAIN = DATA_DIR / f"spy_{SAMPLE_SESSION}.csv.gz"
SAMPLE_STATUS = DATA_DIR / f"spy_{SAMPLE_SESSION}.json"
SAMPLE_SOURCE = "bundled sample: real SPY chain from Yahoo Finance (delayed quotes), one snapshot"


@lru_cache(maxsize=1)
def _sample() -> ChainDataset:
    ds = load_chain_file(SAMPLE_CHAIN.name, SAMPLE_CHAIN.read_bytes())
    ds.source = SAMPLE_SOURCE
    ds.files = [f"data/examples/{SAMPLE_CHAIN.name}"]
    return ds


def load_sample_chain() -> ChainDataset:
    """The bundled sample, validated (a fresh copy: callers may modify it)."""

    ds = _sample()
    return ChainDataset(
        chains={k: v.copy() for k, v in ds.chains.items()},
        source=ds.source,
        ticker=ds.ticker,
        fetched_at_utc=ds.fetched_at_utc,
        warnings=list(ds.warnings),
        renamed=dict(ds.renamed),
        files=list(ds.files),
    )


def sample_fetch_status() -> Optional[Dict[str, object]]:
    """The recorded fetch status of the sample (None if the file is absent)."""

    if not SAMPLE_STATUS.exists():
        return None
    return json.loads(SAMPLE_STATUS.read_text())
