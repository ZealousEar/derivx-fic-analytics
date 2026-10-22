"""Helpers that build snapshot stores (the ``data-snapshots`` layout) for
tests, from the recorded fake Yahoo chain or from the SYNTHETIC history,
without the network."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from _fake_yahoo import Clock, FakeTicker

from derivx.data.market_data import Backoff, ChainFetch, FetchStatus, fetch_option_chain
from derivx.data.snapshots import SnapshotStore, build_snapshot
from derivx.data.synthetic import synthetic_history


def fake_fetch(fake: FakeTicker, start: str = "2026-10-20T21:00:00Z"):
    """A ``fetch`` for :func:`derivx.data.snapshots.collect` that serves
    ``fake`` (no sleeps, a deterministic clock); ``fetch.calls`` counts runs."""

    def fetch(ticker, **kw):
        fetch.calls += 1
        kw["backoff"] = Backoff(retries=kw["backoff"].retries, sleep=lambda s: None)
        return fetch_option_chain(ticker, client_factory=lambda s: fake, now=Clock(start), request_pause=0.0, **kw)

    fetch.calls = 0
    return fetch


def spot_time(session: str) -> int:
    """``regularMarketTime`` of a session's 16:00 New York close (EDT)."""

    return int(pd.Timestamp(f"{session}T20:00:00Z").timestamp())


def fake_session_fetch(session: str) -> ChainFetch:
    """The recorded chain served as if it were ``session``'s."""

    fake = FakeTicker(spot_time=spot_time(session))
    return fetch_option_chain(
        "SPY", client_factory=lambda s: fake, now=Clock(f"{session}T21:00:00Z"), request_pause=0.0, min_days=1.0
    )


def chain_fetch(chain: pd.DataFrame, ticker: str) -> ChainFetch:
    """A complete fetch status around an existing chain (one session)."""

    session = pd.Timestamp(chain["trade_date"].iloc[0]).date().isoformat()
    status = FetchStatus(
        ticker=ticker,
        status="ok",
        started_at_utc=f"{session}T20:30:00Z",
        finished_at_utc=f"{session}T20:30:10Z",
        quote_date=session,
        spot=float(chain["underlying_price"].iloc[0]),
        spot_time_utc=f"{session}T20:00:00Z",
        spot_drift=0.0,
        expiries_requested=sorted(pd.to_datetime(chain["expiry"]).dt.strftime("%Y-%m-%d").unique()),
        rows=len(chain),
    )
    return ChainFetch(chain, status)


def synthetic_store(root: Path, n_days: int, seed: int = 20261020, end: str = "2026-10-19") -> SnapshotStore:
    """A store holding ``n_days`` SYNTHETIC sessions of ticker SYNTH
    (labelled ``source = synthetic`` in every row)."""

    store = SnapshotStore(root, "test store")
    for chain in synthetic_history(n_days, seed=seed, end=end).values():
        store.add(build_snapshot(chain_fetch(chain, "SYNTH")))
    return store


def real_store(root: Path, sessions: Iterable[str], label: Optional[str] = None) -> SnapshotStore:
    """A store of the recorded SPY chain filed under each of ``sessions``."""

    store = SnapshotStore(root, label or "test store")
    for session in sessions:
        store.add(build_snapshot(fake_session_fetch(session), session=session))
    return store
