"""Collected history in the ANALYTICS lab: the daily option-chain snapshots
that ``.github/workflows/collect-data.yml`` publishes on the
``data-snapshots`` branch (:mod:`derivx.data.snapshots`).

Where the lab reads them: the copy ``derivx snapshots sync`` (or the SYNC
NOW button here) mirrored into the download cache from
``raw.githubusercontent.com/<repo>/<branch>`` (repository and branch are
inputs, default ``$DERIVX_SNAPSHOT_REPO`` / ``$DERIVX_SNAPSHOT_BRANCH`` or
ZealousEar/derivx-fic-analytics @ data-snapshots), or a local checkout of
the branch when ``$DERIVX_SNAPSHOT_DIR`` names one. A page render never
downloads: only SYNC NOW does, and every file is checked against the
index's SHA-256.

Two uses:

* WALK-FORWARD VALIDATION, source *Collected history*: coverage and
  provenance of the collected sessions, and the walk-forward once they hold
  enough origin days (:func:`derivx.data.snapshots.origins_needed`: three
  per vol-regime tercile, 9 at the defaults; an origin day needs the
  session ``horizon`` trading days later). Until then the panel says how
  many sessions are collected and how many origin days are still needed.
* MARKET DATA, *Collected history*: any collected session as the active
  dataset, so calibration, the vol surface and model comparison run on it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from derivx.data.chain_schema import ChainDataset
from derivx.data.snapshots import (
    Coverage,
    SnapshotError,
    SnapshotStore,
    default_branch,
    default_repo,
    origins_needed,
    scorable_origins,
    sync,
)
from derivx.ui.lab_common import banner, stats
from derivx.ui.state import PERSIST

LOCAL_ENV = "DERIVX_SNAPSHOT_DIR"
REPO_KEY = "lab_hist_repo"
BRANCH_KEY = "lab_hist_branch"
TICKER_KEY = "lab_hist_ticker"
MSG_KEY = "lab_hist_message"
SOURCE_LABEL = "Collected history"


def _source() -> Tuple[str, str]:
    """(repository, branch) as typed on the WALK-FORWARD tab (defaults
    from the environment)."""

    repo = str(st.session_state.get(REPO_KEY) or default_repo()).strip()
    branch = str(st.session_state.get(BRANCH_KEY) or default_branch()).strip()
    return repo, branch


def synced_store() -> Tuple[Optional[SnapshotStore], Dict[str, Any], str]:
    """(store or None, provenance, why there is none): the local checkout
    of ``$DERIVX_SNAPSHOT_DIR``, else the synced copy of the chosen
    repository and branch in the cache (read offline, never downloaded)."""

    local = os.environ.get(LOCAL_ENV, "").strip()
    label = f"local checkout of the data branch (${LOCAL_ENV})"  # server paths are never shown on the page
    if local:
        root = Path(local).expanduser()
        if not (root / "options" / "index.csv").exists():
            return None, {}, f"${LOCAL_ENV} names a directory without options/index.csv (not a data-branch checkout)"
        return SnapshotStore(root, label), {"source": label}, ""
    repo, branch = _source()
    try:
        result = sync(repo, branch, offline=True)
    except SnapshotError:
        return None, {"repo": repo, "branch": branch}, f"{repo}@{branch} has not been synced on this server"
    except ValueError as exc:
        return None, {"repo": repo, "branch": branch}, str(exc)
    info = {"source": f"{repo}@{branch}", "repo": repo, "branch": branch, "synced_at": result.synced_at,
            "url": result.url}  # fmt: skip
    return result.store, info, ""


def _sync_now(repo: str, branch: str) -> None:
    try:
        with st.spinner(f"Syncing {repo}@{branch} from raw.githubusercontent.com..."):
            result = sync(repo, branch, offline=False)
    except (SnapshotError, ValueError) as exc:
        text = str(exc)
        if "HTTP 404" in text:
            text += (
                ". The branch has no index yet: collection starts once .github/workflows/collect-data.yml is on "
                "the default branch and its first scheduled run saves a session."
            )
        st.session_state[MSG_KEY] = ("error", f"Sync failed: {text}")
        return
    parts = [f"{result.downloaded} file(s) downloaded, {result.cached} already cached"]
    parts += result.notices
    if result.failed:
        parts.append(f"{len(result.failed)} file(s) not kept: " + "; ".join(result.failed[:3]))
    st.session_state[MSG_KEY] = ("warning" if result.failed or result.offline else "success",
                                 f"Synced {repo}@{branch}: " + ". ".join(parts) + ".")  # fmt: skip


def _message() -> None:
    msg = st.session_state.get(MSG_KEY)
    if msg:
        level, text = msg
        getattr(st, level if level in ("error", "warning", "success") else "info")(text)


@st.cache_data(max_entries=8, show_spinner=False)
def _load(root: str, label: str, ticker: str, sessions: Tuple[str, ...], index_sha: str) -> ChainDataset:
    """Stored sessions as one validated dataset (cached on the store's
    index content: a re-sync that adds a session is a new key)."""

    return SnapshotStore(Path(root), label).load(ticker, sessions)


def _index_sha(store: SnapshotStore) -> str:
    from derivx.data.snapshots import sha256_hex

    return sha256_hex(store.index_path.read_bytes()) if store.index_path.exists() else ""


def load(store: SnapshotStore, ticker: str, sessions: Optional[Tuple[str, ...]] = None) -> ChainDataset:
    days = sessions or tuple(d.isoformat() for d in store.sessions(ticker))
    return _load(str(store.root), store.label, ticker, tuple(days), _index_sha(store))


def coverage_parts(cov: Coverage, info: Dict[str, Any], horizon: int = 1) -> Tuple[list, int, int]:
    """(banner parts, origin days at ``horizon``, origin days needed)."""

    have = scorable_origins(cov.sessions, horizon)
    need = origins_needed()
    gaps = (
        "no missing trading days"
        if not cov.gaps
        else f"{len(cov.gaps)} missing trading day(s): {', '.join(d.isoformat() for d in cov.gaps[:4])}"
        + ("..." if len(cov.gaps) > 4 else "")
    )
    parts = [
        cov.ticker,
        f"{cov.n} session(s) {cov.first} .. {cov.last}" if cov.n else "no sessions",
        gaps,
        f"source: {info.get('source')}",
        f"synced {info['synced_at']}" if info.get("synced_at") else "read in place",
        f"{cov.rows:,} quotes, {cov.bytes / 1e6:.2f} MB",
        "Yahoo Finance delayed quotes (real, one file per session)",
    ]
    return parts, have, need


def gate_text(cov: Coverage, have: int, need: int, horizon: int = 1) -> str:
    """Why the walk-forward waits (empty once enough origin days exist)."""

    if have >= need:
        return ""
    return (
        f"{cov.n} session(s) collected ({cov.first} .. {cov.last}) give {have} origin day(s) with the session "
        f"{horizon} trading day(s) later; the walk-forward on collected history waits for {need} (three origin "
        f"days in each vol-regime tercile), so at least {need - have} more trading day(s) of collection."
        if cov.n
        else f"No session collected yet: the walk-forward waits for {need} origin days ({need + horizon} "
        "consecutive sessions)."
    )


def render_walkforward_source(on_use: Callable[[ChainDataset, Dict[str, Any]], None]) -> None:
    """The WALK-FORWARD tab's *Collected history* controls; ``on_use(ds,
    provenance)`` takes the history once it is long enough."""

    c1, c2, c3 = st.columns([2, 1, 1])
    c1.text_input("Repository (OWNER/NAME)", value=default_repo(), key=REPO_KEY, persist_state=PERSIST,
                  help="The GitHub repository whose data-snapshots branch holds the collected chains.")  # fmt: skip
    c2.text_input("Branch", value=default_branch(), key=BRANCH_KEY, persist_state=PERSIST)
    with c3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("Sync now", key="lab_hist_sync", width="stretch"):
            _sync_now(*_source())
            st.rerun()  # the MARKET DATA tab, drawn earlier in this run, reads the new copy too
    if os.environ.get(LOCAL_ENV, "").strip():
        st.caption(f"${LOCAL_ENV} is set: the lab reads that local checkout of the data branch; SYNC NOW only "
                   "refreshes the cached copy of the repository above.")  # fmt: skip
    _message()
    store, info, why = synced_store()
    if store is None:
        st.info(
            f"No collected history here yet ({why}). SYNC NOW downloads the index and the daily files (about 150 "
            "KB a session) from the public data-snapshots branch, which .github/workflows/collect-data.yml fills "
            "once per trading day after the close; or run `derivx snapshots sync`."
        )
        return
    tickers = store.tickers()
    if not tickers:
        st.info(f"{info.get('source')}: the index lists no sessions yet.")
        return
    ticker = st.selectbox("Ticker", tickers, key=TICKER_KEY, persist_state=PERSIST)
    cov = store.coverage(ticker)
    parts, have, need = coverage_parts(cov, info)
    banner("COLLECTED HISTORY", parts)
    stats([("SESSIONS", f"{cov.n}"), ("ORIGIN DAYS (1D)", f"{have}"), ("NEEDED", f"{need}"),
           ("MISSING DAYS", f"{len(cov.gaps)}")])  # fmt: skip
    waiting = gate_text(cov, have, need)
    if waiting:
        st.warning(waiting)
        return
    if st.button(f"Use the collected history ({cov.n} sessions)", key="lab_hist_use", width="stretch"):
        with st.spinner(f"Loading {cov.n} collected sessions (each checked against the index)..."):
            try:
                ds = load(store, ticker)
            except SnapshotError as exc:
                st.error(f"The collected history could not be loaded: {exc}")
                return
        on_use(ds, {**info, "ticker": ticker, "sessions": cov.n, "index_sha256": _index_sha(store)})


def render_market_data_loader(activate: Callable[..., None]) -> None:
    """MARKET DATA: pick one collected session as the active dataset."""

    store, info, why = synced_store()
    if store is None or not store.tickers():
        st.caption(
            "Not synced yet: SYNC NOW on WALK-FORWARD VALIDATION (source: Collected history) or `derivx snapshots "
            f"sync` mirrors the data-snapshots branch ({why or 'no sessions listed'})."
        )
        return
    tickers = store.tickers()
    c1, c2, c3 = st.columns([1, 1, 1])
    ticker = c1.selectbox("Ticker", tickers, key="lab_hist_md_ticker", persist_state=PERSIST)
    days = [d.isoformat() for d in store.sessions(ticker)]
    day = c2.selectbox("Session", list(reversed(days)), key=f"lab_hist_md_day_{ticker}",
                       help="Every collected session (newest first).")  # fmt: skip
    with c3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        clicked = st.button("Load session", key="lab_hist_md_load", width="stretch")
    st.caption(f"{store.coverage(ticker).describe()}; source {info.get('source')}"
               + (f", synced {info['synced_at']}" if info.get("synced_at") else "") + ".")  # fmt: skip
    if clicked and day:
        try:
            ds = load(store, ticker, (day,))
            manifest = store.manifest(ticker, day)
        except SnapshotError as exc:
            st.error(f"Session {day} could not be loaded: {exc}")
            return
        status = {"fetch_status": manifest.get("fetch_status"), "file": manifest.get("file")}
        kept = manifest.get("compaction") or {}
        note = (
            f"Collected session {day} ({kept.get('rows_kept', '?'):,} of {kept.get('rows_in', '?'):,} quotes kept "
            "by the storage limits)."
            if isinstance(kept.get("rows_kept"), int) and isinstance(kept.get("rows_in"), int)
            else f"Collected session {day}."
        )
        activate(ds, "collected", status, note=note)


def sessions_frame(store: SnapshotStore, ticker: str) -> pd.DataFrame:
    """The index rows of one ticker (for exports and tests)."""

    index = store.index()
    return index[index["ticker"].astype(str) == ticker].reset_index(drop=True)
