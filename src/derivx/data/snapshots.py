"""Daily option-chain snapshots, collected one session at a time: a compact
storage schema, an append-only store (the layout of the ``data-snapshots``
branch that ``.github/workflows/collect-data.yml`` pushes to every trading
day), the collector that workflow runs, and sync of the published store for
the ANALYTICS lab and ``tools/compute_metrics.py``.

Why a store: Yahoo Finance serves only the *current* chain (no history), so
a real walk-forward history exists only if each session is saved on its own
day. Every file keeps the lineage of :func:`derivx.data.market_data.
fetch_option_chain`: the session the quotes belong to (``trade_date``, from
the spot's ``regularMarketTime``), each expiry's real fetch time
(``fetched_at_utc``), the spot and its time from the same response, and the
fetch status (requests, retries, failed expiries, spot drift).

Storage schema (:data:`STORE_COLUMNS`, CSV gzip-compressed, re-loadable by
:func:`derivx.data.chain_schema.load_chain_file` as it is): ``trade_date,
expiry, option_type, strike, bid, ask, implied_vol, underlying_price,
volume, last_trade_date, fetched_at_utc, spot_time_utc, ticker, source``:
what the lab reads (liquidity filters, quote hygiene with its staleness
rule, re-inverted vols, lineage). Left out: ``mid`` (``(bid + ask) / 2``,
recomputed on load), ``open_interest``, ``last_price`` and
``contract_symbol`` (the OCC symbol is ticker, expiry, type and strike).

Rows kept (:class:`StoreLimits`): expiries from 1 to 730 calendar days
after the session (the day's own expiry, which Yahoo still lists after the
close, is not requested), strikes with ``0.4 <= K / S <= 1.6`` (S the
snapshot's spot). Yahoo's ``implied_vol`` (its own figure from the last
trade, which the fits do not use: they re-invert vols from bid/ask mids) is
rounded to 1e-5 (0.1 vol bp). Bids, asks, strikes, volumes, spots and
times are stored exactly as Yahoo published them. On the bundled
2026-10-20 SPY chain this keeps 8,352 of 10,040 quotes and 28 of 31
expiries (the same-day one and the two beyond two years go; 3,228 of the
3,520 quotes that pass the lab's default liquidity filters stay) in 153 KB,
against 352 KB for the full file. The real pull of 2026-10-21 stored 8,102
of 9,543 quotes in 148 KB: about 37 MB for a year of 252 sessions.

Layout (POSIX paths relative to the store root)::

    options/index.csv                                  one row per stored file
    options/<TICKER>/<YYYY>/<TICKER>_<session>.csv.gz  the quotes
    options/<TICKER>/<YYYY>/<TICKER>_<session>.json    manifest: fetch status,
                                                       compaction counts, sha256,
                                                       run manifest

``index.csv`` columns (:data:`INDEX_COLUMNS`): ``session, ticker, rows,
expiries, sha256`` (of the ``.csv.gz`` bytes), ``bytes, fetched_at``
(UTC, when the fetch finished), ``spot, status`` (the fetch status: only
complete fetches are stored, so ``ok``) and ``path``.

Append-only: a session already in the store is never rewritten or
re-indexed (:meth:`SnapshotStore.add` keeps it), so a re-run, a manual
dispatch or two overlapping runs cannot change published history. Only
complete fetches whose session is the one due are stored: a partial fetch
(some expiries failed after the retries) or a chain of another session is
never written (:func:`build_snapshot`, :func:`collect`).

When a session is due (:func:`due_session`): the most recent NYSE session
(:mod:`derivx.data.calendar`) whose options close (16:15 New York) was at
least 10 minutes and at most 12 hours ago. A weekend, an exchange holiday
or a run during the trading day collects nothing and makes no commit.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import re
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from derivx.data.calendar import EXCHANGE_TZ, is_trading_day, missing_trading_days, previous_trading_day, trading_days
from derivx.data.chain_schema import ChainDataset, ChainSchemaError, chain_to_csv, load_chain_file, merge_datasets

if TYPE_CHECKING:  # yfinance is imported only when a chain is fetched
    from derivx.data.market_data import ChainFetch

STORE_VERSION = 1
STORE_COLUMNS = (
    "trade_date",
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
    "implied_vol",
    "underlying_price",
    "volume",
    "last_trade_date",
    "fetched_at_utc",
    "spot_time_utc",
    "ticker",
    "source",
)
INDEX_COLUMNS = ("session", "ticker", "rows", "expiries", "sha256", "bytes", "fetched_at", "spot", "status", "path")
OPTIONS_DIR = "options"
INDEX_FILE = "index.csv"
DATA_BRANCH = "data-snapshots"
DEFAULT_REPO = "ZealousEar/derivx-fic-analytics"
REPO_ENV = "DERIVX_SNAPSHOT_REPO"
BRANCH_ENV = "DERIVX_SNAPSHOT_BRANCH"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{branch}"
SOURCE_LABEL = "Yahoo Finance via yfinance (delayed quotes)"
TERMS = "Yahoo Finance terms of use apply (personal and research use)"
OPTIONS_CLOSE = dt.time(16, 15)  # SPY / index-ETF options close at 16:15 New York
COLLECT_DELAY = dt.timedelta(minutes=10)  # delayed quotes settle after the close
COLLECT_WINDOW = dt.timedelta(hours=12)  # a session is collected up to 12 h after its close
GZIP_LEVEL = 9
TICKER_RE = re.compile(r"^[A-Z0-9.^=\-]{1,15}$")
SESSION_RE = re.compile(r"^(?P<ticker>[A-Z0-9.^=\-]{1,15})_(?P<session>\d{4}-\d{2}-\d{2})\.csv\.gz$")


BRANCH_README = """# data-snapshots

Daily option-chain snapshots for DerivX, one file per trading session, written by the scheduled workflow
`.github/workflows/collect-data.yml` on the default branch (`derivx snapshots collect`). Append-only: a
stored session is never rewritten.

- `options/index.csv`: one row per file (session, ticker, rows, expiries, sha256, bytes, fetched_at, spot,
  status, path).
- `options/<TICKER>/<YYYY>/<TICKER>_<session>.csv.gz`: the quotes in the compact storage schema of
  `derivx.data.snapshots` (expiries 1-730 days, strikes 0.4-1.6 x spot, Yahoo's implied vol rounded to 1e-5;
  bids, asks, strikes, volumes and times as published).
- `options/<TICKER>/<YYYY>/<TICKER>_<session>.json`: the manifest (fetch status with the real fetch time,
  spot and its time, retries, compaction counts, SHA-256, code and package versions).

Read it with `derivx snapshots sync` (or the ANALYTICS lab's Collected history source).

Source: Yahoo Finance delayed quotes via yfinance. Yahoo's terms of use apply (personal and research use);
the files are kept for reproducible research and are not market data for trading.
"""


class SnapshotError(RuntimeError):
    """A store or a sync could not be read or written (the message says why)."""


# --------------------------------------------------------------------------
# Compact storage schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreLimits:
    """Which quotes a stored snapshot keeps (see the module docs):
    ``min_days`` / ``max_days`` calendar days from the session to expiry,
    ``moneyness`` = (lowest, highest) strike / spot, ``iv_decimals`` the
    rounding of Yahoo's own implied vol (per-annum decimal)."""

    min_days: int = 1
    max_days: int = 730
    moneyness: Tuple[float, float] = (0.4, 1.6)
    iv_decimals: int = 5

    def __post_init__(self) -> None:
        lo, hi = (float(x) for x in self.moneyness)
        if not (0.0 < lo < 1.0 < hi):
            raise ValueError(f"the moneyness band must bracket 1 (strike / spot), got {self.moneyness}")
        if not 0 <= int(self.min_days) <= int(self.max_days):
            raise ValueError(f"need 0 <= min_days <= max_days, got {self.min_days} and {self.max_days}")
        if not 2 <= int(self.iv_decimals) <= 12:
            raise ValueError("iv_decimals must be between 2 and 12")
        object.__setattr__(self, "moneyness", (lo, hi))

    def as_dict(self) -> Dict[str, object]:
        return {
            "min_days": int(self.min_days),
            "max_days": int(self.max_days),
            "moneyness_min": self.moneyness[0],
            "moneyness_max": self.moneyness[1],
            "iv_decimals": int(self.iv_decimals),
        }


def compact_chain(chain: pd.DataFrame, limits: StoreLimits = StoreLimits()) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """The storage form of one snapshot (``chain`` in the chain schema, one
    trade date): the :data:`STORE_COLUMNS` it has, the rows within
    ``limits``, sorted by expiry, type and strike. Returns (frame, counts):
    rows and expiries in and kept, and the rows dropped by each rule."""

    missing = [c for c in ("trade_date", "expiry", "option_type", "strike", "underlying_price") if c not in chain]
    if missing:
        raise ValueError(f"the chain has no {', '.join(missing)} column(s)")
    if chain.empty:
        raise ValueError("the chain has no quotes")
    trade = pd.to_datetime(chain["trade_date"]).dt.normalize()
    if trade.nunique() != 1:
        raise ValueError(f"a stored snapshot holds one session, this chain has {trade.nunique()}")
    expiry = pd.to_datetime(chain["expiry"]).dt.normalize()
    days = (expiry - trade).dt.days
    ratio = pd.to_numeric(chain["strike"], errors="coerce") / pd.to_numeric(chain["underlying_price"], errors="coerce")
    in_days = (days >= int(limits.min_days)) & (days <= int(limits.max_days))
    in_band = (ratio >= limits.moneyness[0]) & (ratio <= limits.moneyness[1])
    keep = in_days & in_band
    out = chain.loc[keep, [c for c in STORE_COLUMNS if c in chain.columns]].copy()
    if "implied_vol" in out:
        out["implied_vol"] = pd.to_numeric(out["implied_vol"], errors="coerce").round(int(limits.iv_decimals))
    order = [c for c in ("expiry", "option_type", "strike") if c in out]
    out = out.sort_values(order, kind="stable").reset_index(drop=True)
    counts = {
        "rows_in": int(len(chain)),
        "rows_kept": int(len(out)),
        "dropped_expiry_window": int((~in_days).sum()),
        "dropped_moneyness_band": int((in_days & ~in_band).sum()),
        "expiries_in": int(expiry.nunique()),
        "expiries_kept": int(pd.to_datetime(out["expiry"]).nunique()) if len(out) else 0,
    }
    return out, counts


def store_bytes(frame: pd.DataFrame) -> bytes:
    """Deterministic ``.csv.gz`` bytes of a stored frame (no timestamp or
    file name in the gzip header, so equal quotes give equal bytes)."""

    return gzip.compress(chain_to_csv(frame), compresslevel=GZIP_LEVEL, mtime=0)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SYNTHETIC_LABEL = "SYNTHETIC (generated, not market data)"


UNRECORDED = "unrecorded"  # :meth:`SnapshotStore.row_sources` of a manifest without them (or unreadable)


def source_label(row_sources: Iterable[str]) -> str:
    """What a stored file holds, from its rows' ``source`` values: Yahoo's
    delayed quotes (``yahoo``), SYNTHETIC rows (named as such whatever else
    is there), a source the manifests do not record (:data:`UNRECORDED`),
    or the values themselves."""

    sources = sorted({str(s) for s in row_sources})
    if any("synthetic" in s.lower() for s in sources):
        return SYNTHETIC_LABEL
    if sources == ["yahoo"]:
        return SOURCE_LABEL
    if UNRECORDED in sources:
        return "rows of a source some manifests do not record (see each file's source column)"
    return "rows from " + (", ".join(sources) if sources else "an unnamed source")


# --------------------------------------------------------------------------
# Snapshots and the store
# --------------------------------------------------------------------------


def relative_path(ticker: str, session: str) -> PurePosixPath:
    """``options/<TICKER>/<YYYY>/<TICKER>_<session>.csv.gz``."""

    symbol = _ticker(ticker)
    day = _session(session)
    return PurePosixPath(OPTIONS_DIR, symbol, day[:4], f"{symbol}_{day}.csv.gz")


def manifest_path(data_path: PurePosixPath) -> PurePosixPath:
    return data_path.with_name(data_path.name[: -len(".csv.gz")] + ".json")


def layout_problem(ticker: object, session: object, path: object) -> Optional[str]:
    """Why an index row's ``path`` is not where the layout files its ticker
    and session (None when it is): a row that names another session's file
    would file that chain under a false date."""

    try:
        expected = str(relative_path(str(ticker), str(session)))
    except ValueError as exc:
        return f"{path}: index row ({ticker}, {session}) is not a ticker and session ({exc})"
    if str(path) != expected:
        return f"{path}: the index files ({ticker}, {session}) here, the layout puts it at {expected}"
    return None


def _ticker(ticker: str) -> str:
    symbol = str(ticker or "").strip().upper()
    if not TICKER_RE.match(symbol):
        raise ValueError(f"ticker must be 1-15 letters, digits or . ^ = -, got {ticker!r}")
    return symbol


def _session(session: object) -> str:
    try:
        return pd.Timestamp(str(session)).date().isoformat()
    except ValueError:
        raise ValueError(f"{session!r} is not a session date (YYYY-MM-DD)") from None


@dataclass
class Snapshot:
    """One session's compact chain, ready to store: ``data`` are the
    ``.csv.gz`` bytes, ``manifest`` the JSON sidecar."""

    ticker: str
    session: str
    frame: pd.DataFrame
    data: bytes
    manifest: Dict[str, Any]

    @property
    def path(self) -> PurePosixPath:
        return relative_path(self.ticker, self.session)

    def index_row(self) -> Dict[str, object]:
        return index_row(self.manifest, str(self.path))


def index_row(manifest: Mapping[str, Any], path: str) -> Dict[str, object]:
    status = manifest.get("fetch_status") or {}
    return {
        "session": manifest["session"],
        "ticker": manifest["ticker"],
        "rows": int(manifest["rows"]),
        "expiries": int(manifest["expiries"]),
        "sha256": manifest["sha256"],
        "bytes": int(manifest["bytes"]),
        "fetched_at": status.get("finished_at_utc") or manifest.get("fetched_at_utc"),
        "spot": manifest.get("spot"),
        "status": status.get("status", ""),
        "path": path,
    }


def build_snapshot(
    result: "ChainFetch", limits: StoreLimits = StoreLimits(), session: Optional[str] = None
) -> Snapshot:
    """The stored form of a fetch. Refuses (``ValueError``) a fetch that is
    not complete (status other than ``ok``: a partial chain is never
    stored), a chain of more than one session, and, with ``session``, a
    chain of any other session (it would be stored under a false date)."""

    from derivx.validation.manifest import build_manifest

    status = result.status
    if status.status != "ok":
        failed = "; ".join(f"{e}: {why}" for e, why in list(status.failed_expiries.items())[:3])
        more = "..." if len(status.failed_expiries) > 3 else ""
        raise ValueError(
            f"{status.ticker}: not stored, the fetch is not complete ({status.summary()})"
            + (f"; failed expiries: {failed}{more}" if failed else "")
        )
    served = str(status.quote_date)
    if session is not None and served != _session(session):
        raise ValueError(
            f"{status.ticker}: Yahoo served the {served} session, not {_session(session)}: nothing stored "
            "(a chain is never filed under another session's date)"
        )
    frame, counts = compact_chain(result.chain, limits)
    if frame.empty:
        raise ValueError(f"{status.ticker}: no quote is within the storage limits {limits.as_dict()}")
    data = store_bytes(frame)
    row_sources = sorted({str(s) for s in frame["source"].dropna()}) if "source" in frame else []
    manifest: Dict[str, Any] = {
        "store_version": STORE_VERSION,
        "ticker": status.ticker,
        "session": served,
        "file": relative_path(status.ticker, served).name,
        "sha256": sha256_hex(data),
        "bytes": len(data),
        "rows": int(len(frame)),
        "expiries": int(pd.to_datetime(frame["expiry"]).nunique()),
        "spot": status.spot,
        "spot_time_utc": status.spot_time_utc,
        "fetched_at_utc": status.finished_at_utc,
        "source": source_label(row_sources),
        "row_sources": row_sources,
        "terms": TERMS,
        "columns": list(frame.columns),
        "limits": limits.as_dict(),
        "compaction": counts,
        "fetch_status": status.as_dict(),
        "manifest": build_manifest(
            "snapshot",
            inputs={"ticker": status.ticker, "expiries_requested": status.expiries_requested},
            data={
                "source": source_label(row_sources),
                "fetched_at_utc": status.finished_at_utc,
                "data_sha256": sha256_hex(data),
            },  # fmt: skip
            model={"store_version": STORE_VERSION, "limits": limits.as_dict()},
        ),
    }
    return Snapshot(status.ticker, served, frame, data, manifest)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; stored files are plain readable data
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    from derivx.provenance import jsonable

    return (json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n").encode("utf-8")


def _with_rows(index: pd.DataFrame, rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """``index`` with ``rows`` appended (no concat with an empty frame:
    pandas would drop its column types)."""

    added = pd.DataFrame(list(rows), columns=list(INDEX_COLUMNS))
    if index.empty:
        return added
    if added.empty:
        return index
    return pd.concat([index, added], ignore_index=True)


@dataclass(frozen=True)
class Coverage:
    """Sessions of one ticker in a store: ``gaps`` are the NYSE trading
    days between the first and the last that have no file."""

    ticker: str
    sessions: Tuple[dt.date, ...]
    gaps: Tuple[dt.date, ...]
    rows: int
    bytes: int

    @property
    def n(self) -> int:
        return len(self.sessions)

    @property
    def first(self) -> Optional[dt.date]:
        return self.sessions[0] if self.sessions else None

    @property
    def last(self) -> Optional[dt.date]:
        return self.sessions[-1] if self.sessions else None

    def describe(self) -> str:
        if not self.sessions:
            return f"{self.ticker}: no sessions collected"
        gaps = (
            "no missing trading days"
            if not self.gaps
            else f"{len(self.gaps)} missing trading day(s): {', '.join(d.isoformat() for d in self.gaps[:6])}"
            + ("..." if len(self.gaps) > 6 else "")
        )
        return (
            f"{self.ticker}: {self.n} session(s) {self.first} .. {self.last}; {gaps}; {self.rows:,} quotes, "
            f"{self.bytes / 1e6:.2f} MB"
        )


class SnapshotStore:
    """A store on disk (a checkout of the ``data-snapshots`` branch, a
    staging directory or the sync cache); see the module docs for the
    layout. Reads never touch the network."""

    def __init__(self, root: Path | str, label: str = ""):
        self.root = Path(root)
        self.label = label or str(self.root)

    @property
    def index_path(self) -> Path:
        return self.root / OPTIONS_DIR / INDEX_FILE

    def file(self, relpath: PurePosixPath | str) -> Path:
        rel = PurePosixPath(str(relpath))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != OPTIONS_DIR:
            raise SnapshotError(f"{relpath!r} is not a path inside the store's {OPTIONS_DIR}/ directory")
        return self.root.joinpath(*rel.parts)

    def has(self, ticker: str, session: str) -> bool:
        return self.file(relative_path(ticker, session)).exists()

    # -- index ---------------------------------------------------------------

    def index(self) -> pd.DataFrame:
        """``index.csv`` (empty, with the columns, when there is none yet)."""

        if not self.index_path.exists():
            return pd.DataFrame({c: pd.Series(dtype=object) for c in INDEX_COLUMNS})
        try:
            frame = pd.read_csv(self.index_path, dtype={"session": str, "ticker": str, "sha256": str, "path": str})
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise SnapshotError(f"{self.index_path}: not a snapshot index ({exc})") from None
        missing = [c for c in INDEX_COLUMNS if c not in frame.columns]
        if missing:
            raise SnapshotError(f"{self.index_path}: missing column(s) {missing}")
        return frame[list(INDEX_COLUMNS)]

    def write_index(self, frame: pd.DataFrame) -> None:
        table = frame[list(INDEX_COLUMNS)].sort_values(["ticker", "session"], kind="stable").reset_index(drop=True)
        _atomic_write(self.index_path, table.to_csv(index=False, lineterminator="\n").encode("utf-8"))

    def manifests(self) -> List[Tuple[PurePosixPath, Dict[str, Any]]]:
        """(data path, manifest) of every stored file, from the JSON sidecars."""

        out: List[Tuple[PurePosixPath, Dict[str, Any]]] = []
        base = self.root / OPTIONS_DIR
        if not base.exists():
            return out
        for meta in sorted(base.glob("*/*/*.json")):
            data = meta.with_name(meta.name[: -len(".json")] + ".csv.gz")
            if not data.exists():
                continue
            try:
                manifest = json.loads(meta.read_text())
            except (OSError, ValueError) as exc:
                raise SnapshotError(f"{meta}: unreadable manifest ({exc})") from None
            out.append((PurePosixPath(*data.relative_to(self.root).parts), manifest))
        return out

    def rebuild_index(self) -> pd.DataFrame:
        """The index from the manifests on disk (rows of files that already
        have one keep it, so an existing row is never rewritten)."""

        current = self.index()
        known = set(zip(current["ticker"].astype(str), current["session"].astype(str)))
        rows = [index_row(m, str(p)) for p, m in self.manifests() if (str(m["ticker"]), str(m["session"])) not in known]
        self.write_index(_with_rows(current, rows))
        return self.index()

    def verify(self) -> List[str]:
        """Problems: index rows that name a path other than their ticker's
        and session's, rows without their file or with another SHA-256, and
        stored files the index does not list."""

        problems = []
        index = self.index()
        listed = set()
        for row in index.itertuples(index=False):
            listed.add(str(row.path))
            misplaced = layout_problem(row.ticker, row.session, row.path)
            if misplaced:
                problems.append(misplaced)
                continue
            try:
                path = self.file(str(row.path))
            except SnapshotError as exc:
                problems.append(str(exc))
                continue
            if not path.exists():
                problems.append(f"{row.path}: listed in the index but missing")
            elif sha256_hex(path.read_bytes()) != str(row.sha256):
                problems.append(f"{row.path}: SHA-256 differs from the index")
        for path, _ in self.manifests():
            if str(path) not in listed:
                problems.append(f"{path}: stored but not in the index")
        return problems

    # -- writing -------------------------------------------------------------

    def add(self, snapshot: Snapshot, update_index: bool = True) -> str:
        """Store one snapshot: ``'added'``, or ``'kept'`` when the store
        already holds that session (it is never rewritten)."""

        data_path = self.file(snapshot.path)
        if data_path.exists():
            return "kept"
        _atomic_write(self.file(manifest_path(snapshot.path)), _json_bytes(snapshot.manifest))
        _atomic_write(data_path, snapshot.data)
        if update_index:
            self._append_rows([snapshot.index_row()])
        return "added"

    def _append_rows(self, rows: Sequence[Mapping[str, object]]) -> None:
        current = self.index()
        known = set(zip(current["ticker"].astype(str), current["session"].astype(str)))
        new = [r for r in rows if (str(r["ticker"]), str(r["session"])) not in known]
        if new:
            self.write_index(_with_rows(current, new))

    def merge_from(self, other: "SnapshotStore") -> List[Tuple[str, str, str]]:
        """Copy the files of ``other`` (a staging store) that this store does
        not hold, then append their index rows: (ticker, session,
        'added' | 'kept') per file of ``other``. Files are checked against
        their manifest's SHA-256 before they are copied."""

        out: List[Tuple[str, str, str]] = []
        todo = []
        for rel, manifest in other.manifests():  # check everything before anything is written
            ticker, session = str(manifest["ticker"]), str(manifest["session"])
            if self.file(rel).exists():
                out.append((ticker, session, "kept"))
                continue
            data = other.file(rel).read_bytes()
            if sha256_hex(data) != manifest.get("sha256"):
                raise SnapshotError(f"{rel}: its bytes do not match its manifest's SHA-256: nothing copied")
            todo.append((rel, manifest, data))
            out.append((ticker, session, "added"))
        for rel, manifest, data in todo:
            _atomic_write(self.file(manifest_path(rel)), other.file(manifest_path(rel)).read_bytes())
            _atomic_write(self.file(rel), data)
        if todo:
            self._append_rows([index_row(manifest, str(rel)) for rel, manifest, _ in todo])
            readme = self.root / "README.md"
            if not readme.exists():
                _atomic_write(readme, BRANCH_README.encode("utf-8"))
        return out

    # -- reading -------------------------------------------------------------

    def tickers(self) -> List[str]:
        return sorted(set(self.index()["ticker"].astype(str)))

    def manifest(self, ticker: str, session: object) -> Dict[str, Any]:
        """The JSON manifest of one stored session (fetch status, limits,
        compaction counts, run manifest)."""

        path = self.file(manifest_path(relative_path(ticker, _session(session))))
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
            raise SnapshotError(f"{path.name}: no readable manifest in {self.label} ({reason})") from None
        if not isinstance(manifest, dict):
            raise SnapshotError(f"{path.name}: not a manifest (a JSON object) in {self.label}")
        return manifest

    def row_sources(self, ticker: str, sessions: Optional[Iterable[object]] = None) -> List[str]:
        """The distinct ``source`` values of the rows of ``ticker``'s stored
        files (all, or ``sessions``), from their manifests' ``row_sources``;
        a manifest without them (written before they were kept) or an
        unreadable one counts as :data:`UNRECORDED`, never as a guess."""

        days = [d.isoformat() for d in self.sessions(ticker)] if sessions is None else [_session(s) for s in sessions]
        out: set = set()
        for day in days:
            try:
                listed = self.manifest(ticker, day).get("row_sources")
            except SnapshotError:
                listed = None
            out |= {str(s) for s in listed} if isinstance(listed, list) and listed else {UNRECORDED}
        return sorted(out)

    def sessions(self, ticker: str) -> List[dt.date]:
        index = self.index()
        picked = index[index["ticker"].astype(str) == _ticker(ticker)]
        return sorted(pd.Timestamp(s).date() for s in picked["session"].astype(str))

    def coverage(self, ticker: str, last: Optional[int] = None) -> Coverage:
        """Coverage of ``ticker``'s sessions (``last``: only the most recent
        ``last`` of them)."""

        index = self.index()
        picked = index[index["ticker"].astype(str) == _ticker(ticker)].sort_values("session", kind="stable")
        if last is not None:
            picked = picked.iloc[-max(int(last), 1) :]
        days = sorted(pd.Timestamp(s).date() for s in picked["session"].astype(str))
        gaps = tuple(d.date() for d in missing_trading_days(days))
        return Coverage(
            _ticker(ticker),
            tuple(days),
            gaps,
            int(pd.to_numeric(picked["rows"]).sum()) if len(picked) else 0,
            int(pd.to_numeric(picked["bytes"]).sum()) if len(picked) else 0,
        )

    def load(self, ticker: str, sessions: Optional[Iterable[object]] = None, verify: bool = True) -> ChainDataset:
        """The stored sessions of ``ticker`` (all, or ``sessions``) as one
        validated :class:`ChainDataset`, each file checked against the
        index's SHA-256 (``verify``), its place in the layout and its own
        trade date (a file holding another session is refused)."""

        symbol = _ticker(ticker)
        index = self.index()
        picked = index[index["ticker"].astype(str) == symbol]
        if sessions is not None:
            wanted = {_session(s) for s in sessions}
            unknown = sorted(wanted - set(picked["session"].astype(str)))
            if unknown:
                raise SnapshotError(f"{symbol}: no stored session {', '.join(unknown[:6])} in {self.label}")
            picked = picked[picked["session"].astype(str).isin(wanted)]
        if picked.empty:
            raise SnapshotError(f"{symbol}: no stored sessions in {self.label}")
        datasets = []
        for row in picked.sort_values("session").itertuples(index=False):
            misplaced = layout_problem(row.ticker, row.session, row.path)
            if misplaced:
                raise SnapshotError(misplaced)
            path = self.file(str(row.path))
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise SnapshotError(f"{row.path}: cannot be read ({exc.strerror or exc})") from None
            if verify and sha256_hex(data) != str(row.sha256):
                raise SnapshotError(f"{row.path}: SHA-256 differs from the index (corrupt or edited file)")
            try:
                ds = load_chain_file(path.name, data, ticker=symbol)
            except ChainSchemaError as exc:
                raise SnapshotError(f"{row.path}: {exc}") from None
            held = [d.date().isoformat() for d in ds.trade_dates]
            if held != [_session(row.session)]:
                raise SnapshotError(f"{row.path}: holds the session(s) {', '.join(held)}, not {row.session}")
            ds.files = [str(row.path)]
            datasets.append(ds)
        merged = merge_datasets(datasets, source=f"collected history: {self.label}")
        merged.source = f"collected history ({self.label}): {len(datasets)} session(s)"
        return merged


# --------------------------------------------------------------------------
# Collector
# --------------------------------------------------------------------------


def _ny(when: Optional[pd.Timestamp]) -> pd.Timestamp:
    ts = pd.Timestamp.now(tz="UTC") if when is None else pd.Timestamp(when)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(EXCHANGE_TZ)


def session_ready_at(session: dt.date) -> pd.Timestamp:
    """When a session's chain can be collected: the options close (16:15
    New York) plus :data:`COLLECT_DELAY`, as a New York time."""

    close = pd.Timestamp(dt.datetime.combine(session, OPTIONS_CLOSE)).tz_localize(EXCHANGE_TZ)
    return close + pd.Timedelta(COLLECT_DELAY)


def due_session(now: Optional[pd.Timestamp] = None) -> Tuple[Optional[dt.date], str]:
    """(session to collect now, or None; why). The most recent NYSE session
    whose options closed at least :data:`COLLECT_DELAY` and at most
    :data:`COLLECT_WINDOW` before ``now``; None on a weekend or exchange
    holiday, during a session and in the hours before it opens."""

    ny = _ny(now)
    today = ny.date()
    if is_trading_day(today) and ny >= session_ready_at(today):
        candidate = today
    else:
        candidate = previous_trading_day(today - dt.timedelta(days=1) if is_trading_day(today) else today).date()
    ready = session_ready_at(candidate)
    age = ny - ready
    stamp = ny.strftime("%Y-%m-%d %H:%M %Z")
    if age <= pd.Timedelta(COLLECT_WINDOW):
        return candidate, f"the {candidate} session closed at 16:15 New York; collecting at {stamp}"
    if not is_trading_day(today):
        why = f"{today} ({ny.strftime('%A')}) is not an NYSE trading day (weekend or exchange holiday)"
    else:
        why = f"the {today} session has not closed yet (options close 16:15 New York, collected from 16:25)"
    hours = age.total_seconds() / 3600.0
    return None, (
        f"nothing to collect at {stamp}: {why}; the last session, {candidate}, closed {hours:.1f} h ago, more than "
        f"the {COLLECT_WINDOW.total_seconds() / 3600:g} h collection window"
    )


@dataclass
class Outcome:
    """What the collector did for one ticker: ``state`` is ``saved``,
    ``kept`` (the session is already stored), ``skipped`` (no session due)
    or ``failed`` (a session was due and nothing was saved)."""

    ticker: str
    session: Optional[str]
    state: str
    reason: str
    rows: int = 0
    bytes: int = 0
    rows_in: int = 0
    attempts: int = 0
    requests: int = 0
    retries: int = 0
    seconds: float = 0.0
    path: Optional[str] = None


@dataclass
class CollectReport:
    """The outcome of one collector run (see :func:`collect`)."""

    now_utc: str
    session: Optional[str]
    reason: str
    outcomes: List[Outcome] = field(default_factory=list)
    limits: Dict[str, object] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def failed(self) -> List[Outcome]:
        return [o for o in self.outcomes if o.state == "failed"]

    @property
    def saved(self) -> List[Outcome]:
        return [o for o in self.outcomes if o.state == "saved"]

    @property
    def exit_code(self) -> int:
        """0: every ticker saved or already stored, or no session due; 1: a
        session was due and some ticker produced nothing."""

        return 1 if self.failed else 0

    def as_dict(self) -> Dict[str, object]:
        return {**asdict(self), "exit_code": self.exit_code}

    def markdown(self) -> str:
        """The job summary (GitHub Actions ``$GITHUB_STEP_SUMMARY``)."""

        head = "## Option-chain collection" + (" (dry run: nothing is committed)" if self.dry_run else "")
        lines = [head, "", f"Run at {self.now_utc}. {self.reason[:1].upper() + self.reason[1:]}.", ""]
        if self.outcomes:
            lines += ["| ticker | session | result | quotes stored | size | attempts | requests / retries | seconds |",
                      "|---|---|---|---|---|---|---|---|"]  # fmt: skip
            for o in self.outcomes:
                size = f"{o.bytes / 1e3:,.0f} KB" if o.bytes else ""
                stored = f"{o.rows:,} of {o.rows_in:,}" if o.rows else ""
                lines.append(
                    f"| {o.ticker} | {o.session or ''} | **{o.state.upper()}** | {stored} | {size} | {o.attempts} | "
                    f"{o.requests} / {o.retries} | {o.seconds:.1f} |"
                )
            lines.append("")
            for o in self.outcomes:
                if o.reason:
                    lines.append(f"- {o.ticker}: {o.reason}")
        if self.failed:
            lines += ["", f"**FAILED**: {len(self.failed)} ticker(s) produced nothing for a trading day; nothing "
                      "partial or mislabelled was written. The job fails so this run shows red."]  # fmt: skip
        lines += ["", f"Storage limits: {self.limits}."]
        return "\n".join(lines) + "\n"


def _fetch_default(ticker: str, **kwargs: Any) -> "ChainFetch":
    from derivx.data.market_data import fetch_option_chain

    return fetch_option_chain(ticker, **kwargs)


def collect(
    tickers: Sequence[str],
    staging: Path | str,
    store: Optional[SnapshotStore] = None,
    *,
    fetch: Callable[..., "ChainFetch"] = _fetch_default,
    now: Optional[pd.Timestamp] = None,
    limits: StoreLimits = StoreLimits(),
    attempts: int = 3,
    wait_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    retries_per_request: int = 4,
    dry_run: bool = False,
) -> CollectReport:
    """Collect the due session (:func:`due_session`) of each ticker into
    the ``staging`` store, skipping tickers whose session ``store`` (the
    published store, e.g. a checkout of the data branch) or ``staging``
    already holds. A fetch that fails, is partial, serves another session
    or raises is retried ``attempts`` times in all, waiting ``wait_s``
    seconds, then twice that, between attempts (each Yahoo request also
    retries ``retries_per_request`` times with backoff); after that the
    ticker is ``failed`` and nothing is written for it, and the next
    ticker is collected. Expiries are requested from ``limits.min_days``
    after the *session* (the fetch's own ``min_days`` counts from the New
    York date of the fetch, a day later for a run after midnight)."""

    from derivx.data.market_data import Backoff

    symbols = [_ticker(t) for t in tickers]
    if not symbols:
        raise ValueError("give at least one ticker")
    if int(attempts) < 1:
        raise ValueError("attempts must be >= 1")
    clock = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    clock = clock.tz_localize("UTC") if clock.tzinfo is None else clock.tz_convert("UTC")
    session, reason = due_session(clock)
    stamp = clock.isoformat(timespec="seconds").replace("+00:00", "Z")
    report = CollectReport(stamp, session.isoformat() if session else None, reason, limits=limits.as_dict(),
                           dry_run=dry_run)  # fmt: skip
    staged = SnapshotStore(staging, "staging")
    run_started = time.monotonic()
    for symbol in symbols:
        if session is None:
            report.outcomes.append(Outcome(symbol, None, "skipped", "no session due"))
            continue
        day = session.isoformat()
        if store is not None and store.has(symbol, day):
            report.outcomes.append(Outcome(symbol, day, "kept", f"{store.label} already holds the {day} session"))
            continue
        if staged.has(symbol, day):
            report.outcomes.append(Outcome(symbol, day, "kept", f"the staging store already holds the {day} session"))
            continue
        outcome = Outcome(symbol, day, "failed", "")
        started = time.monotonic()
        for attempt in range(1, int(attempts) + 1):
            outcome.attempts = attempt
            # the fetch counts min_days from its own New York date: shift it so the window starts at the session
            fetch_day = _ny(clock + pd.Timedelta(seconds=time.monotonic() - run_started)).date()
            min_days = max(0.0, float(limits.min_days) - (fetch_day - session).days)
            try:
                result = fetch(symbol, min_days=min_days, backoff=Backoff(retries=int(retries_per_request)))
                status = result.status
                outcome.requests += int(status.requests)
                outcome.retries += int(status.retries)
                snapshot = build_snapshot(result, limits, session=day)
            except Exception as exc:  # a refused fetch (ValueError) or anything the fetch raised: retry, then fail
                text = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
                outcome.reason = f"attempt {attempt}: {text}"
                if attempt < int(attempts):
                    sleep(float(wait_s) * 2 ** (attempt - 1))
                continue
            staged.add(snapshot)
            outcome.state = "saved"
            outcome.rows = snapshot.manifest["rows"]
            outcome.rows_in = snapshot.manifest["compaction"]["rows_in"]
            outcome.bytes = snapshot.manifest["bytes"]
            outcome.path = str(snapshot.path)
            outcome.reason = (
                f"{snapshot.manifest['rows']:,} of {outcome.rows_in:,} quotes over {snapshot.manifest['expiries']} "
                f"expiries, spot {status.spot}, fetched {status.finished_at_utc}"
                + (f" (after {attempt - 1} failed attempt(s))" if attempt > 1 else "")
            )
            break
        outcome.seconds = round(time.monotonic() - started, 1)
        report.outcomes.append(outcome)
    return report


# --------------------------------------------------------------------------
# History needed by the walk-forward test
# --------------------------------------------------------------------------


def scorable_origins(sessions: Iterable[object], horizon: int = 1) -> int:
    """Sessions that have the session exactly ``horizon`` NYSE trading
    days later in ``sessions`` (the walk-forward test's origin days)."""

    days = sorted({pd.Timestamp(str(s)).normalize() for s in sessions})
    if len(days) < 2:
        return 0
    have = set(days)
    calendar = trading_days(days[0], days[-1])
    position = {d: i for i, d in enumerate(calendar)}
    count = 0
    for d in days:
        i = position.get(d)
        if i is not None and i + int(horizon) < len(calendar) and calendar[i + int(horizon)] in have:
            count += 1
    return count


def origins_needed(min_regime_days: int = 3) -> int:
    """Scored origin days a walk-forward on collected history waits for:
    ``min_regime_days`` in each of the three vol-regime terciles (with the
    defaults, 9 origin days: 10 consecutive sessions at a 1-day horizon)."""

    return 3 * int(min_regime_days)


# --------------------------------------------------------------------------
# Sync of the published store
# --------------------------------------------------------------------------


def default_repo() -> str:
    return os.environ.get(REPO_ENV, "").strip() or DEFAULT_REPO


def default_branch() -> str:
    return os.environ.get(BRANCH_ENV, "").strip() or DATA_BRANCH


def _repo(repo: str) -> str:
    text = str(repo).strip().strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
        raise ValueError(f"repository must be OWNER/NAME, got {repo!r}")
    return text


def _branch(branch: str) -> str:
    text = str(branch).strip()
    if not re.fullmatch(r"[A-Za-z0-9._/-]{1,100}", text) or ".." in text:
        raise ValueError(f"not a branch name: {branch!r}")
    return text


def sync_root(repo: str, branch: str, cache_root: Optional[Path] = None) -> Path:
    """Where a synced copy lives: ``<cache>/snapshots/<owner>__<name>/<branch>``
    (the cache is ``data/cache`` or ``$DERIVX_CACHE_DIR``)."""

    from derivx.data.cache import cache_dir

    base = Path(cache_root) if cache_root is not None else cache_dir()
    return base / "snapshots" / _repo(repo).replace("/", "__") / _branch(branch).replace("/", "__")


@dataclass
class SyncResult:
    """A synced store and what the sync did: files downloaded, already
    cached, failed (with reasons); ``offline`` when only the cache was
    read; ``url`` the store it mirrors."""

    store: SnapshotStore
    url: str
    downloaded: int = 0
    cached: int = 0
    failed: List[str] = field(default_factory=list)
    offline: bool = False
    notices: List[str] = field(default_factory=list)
    synced_at: Optional[str] = None


SYNC_META = "sync.json"


def _readable_manifest(store: SnapshotStore, rel: PurePosixPath) -> bool:
    """True when the JSON sidecar of a cached file is there and parses (a
    cached file without it is downloaded again)."""

    try:
        return isinstance(json.loads(store.file(manifest_path(rel)).read_text()), dict)
    except (OSError, ValueError):
        return False


def sync(
    repo: Optional[str] = None,
    branch: Optional[str] = None,
    *,
    tickers: Optional[Sequence[str]] = None,
    offline: Optional[bool] = None,
    cache_root: Optional[Path] = None,
    base_url: Optional[str] = None,
    timeout: float = 20.0,
    retries: int = 2,
    workers: int = 8,
) -> SyncResult:
    """Mirror the published store (``index.csv`` and the files it lists)
    from ``raw.githubusercontent.com/<repo>/<branch>`` (or ``base_url``)
    into the cache, downloading only files the cache lacks or holds with
    another SHA-256 and checking every download against the index.
    ``offline`` (default: ``$DERIVX_OFFLINE``) reads only the cache. When
    the index cannot be downloaded the cached copy is used with a notice;
    with no cached copy :class:`SnapshotError` is raised."""

    from derivx.data import rates_sources as rs

    repo_name = _repo(repo or default_repo())
    branch_name = _branch(branch or default_branch())
    root = sync_root(repo_name, branch_name, cache_root)
    url = (base_url or RAW_URL.format(repo=repo_name, branch=urllib.parse.quote(branch_name))).rstrip("/")
    label = f"{repo_name}@{branch_name}"
    store = SnapshotStore(root, label)
    result = SyncResult(store, url)
    meta_path = root / SYNC_META
    if meta_path.exists():
        try:
            result.synced_at = json.loads(meta_path.read_text()).get("synced_at")
        except (OSError, ValueError):
            result.synced_at = None
    offline = rs.is_offline() if offline is None else offline
    if offline:
        result.offline = True
        if not store.index_path.exists():
            raise SnapshotError(f"offline: {label} has not been synced into {root} yet; run `derivx snapshots sync`")
        result.notices.append(f"offline: the cached copy of {label} synced {result.synced_at or 'at an unknown time'}")
        return result
    try:
        index_bytes = rs.http_get(f"{url}/{OPTIONS_DIR}/{INDEX_FILE}", timeout=timeout, retries=retries)
        remote = pd.read_csv(io.BytesIO(index_bytes), dtype={"session": str, "ticker": str, "sha256": str, "path": str})
        missing = [c for c in INDEX_COLUMNS if c not in remote.columns]
        if missing:
            raise SnapshotError(f"the index at {url} has no column(s) {missing}")
    except (rs.SourceError, SnapshotError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        if not store.index_path.exists():
            raise SnapshotError(f"{label}: the index could not be downloaded ({exc}) and nothing is cached") from None
        result.offline = True
        result.notices.append(f"{label}: index download failed ({exc}); using the copy synced {result.synced_at}")
        return result
    remote = remote[list(INDEX_COLUMNS)]
    if tickers:
        remote = remote[remote["ticker"].astype(str).isin({_ticker(t) for t in tickers})]
    todo = []
    have_rows = []
    for row in remote.itertuples(index=False):
        rel = PurePosixPath(str(row.path))
        misplaced = layout_problem(row.ticker, row.session, row.path)
        if misplaced:
            result.failed.append(f"{misplaced} (not kept)")
            continue
        local = store.file(rel)
        if local.exists() and sha256_hex(local.read_bytes()) == str(row.sha256) and _readable_manifest(store, rel):
            result.cached += 1
            have_rows.append(row._asdict())
        else:
            todo.append(row)

    def download(row: Any) -> Tuple[Any, Optional[str]]:
        rel = PurePosixPath(str(row.path))
        try:
            data = rs.http_get(f"{url}/{rel}", timeout=timeout, retries=retries)
            meta = rs.http_get(f"{url}/{manifest_path(rel)}", timeout=timeout, retries=retries)
        except rs.SourceError as exc:
            return row, f"{rel}: {exc}"
        except Exception as exc:  # anything else one download raises fails that file, never the whole sync
            return row, f"{rel}: {type(exc).__name__}: {exc}"
        if sha256_hex(data) != str(row.sha256):
            return row, f"{rel}: the download's SHA-256 differs from the index (not kept)"
        try:
            if not isinstance(json.loads(meta), dict):
                raise ValueError("not an object")
        except ValueError:
            return row, f"{manifest_path(rel)}: not a JSON manifest (not kept)"
        _atomic_write(store.file(manifest_path(rel)), meta)
        _atomic_write(store.file(rel), data)
        return row, None

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(todo)))) as pool:
            for row, problem in pool.map(download, todo):
                if problem is None:
                    result.downloaded += 1
                    have_rows.append(row._asdict())
                else:
                    result.failed.append(problem)
    local_index = pd.DataFrame(have_rows, columns=list(INDEX_COLUMNS))
    if tickers:  # keep the cached rows of the other tickers
        others = store.index()
        others = others[~others["ticker"].astype(str).isin({_ticker(t) for t in tickers})]
        local_index = _with_rows(others, have_rows)
    store.write_index(local_index)
    result.synced_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds").replace("+00:00", "Z")
    _atomic_write(meta_path, _json_bytes({"url": url, "repo": repo_name, "branch": branch_name,
                                          "synced_at": result.synced_at, "files": int(len(local_index)),
                                          "index_sha256": sha256_hex(index_bytes)}))  # fmt: skip
    return result


def coverage_table(store: SnapshotStore, tickers: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per ticker: sessions, first, last, missing trading days, quotes, MB."""

    rows = []
    for symbol in tickers or store.tickers():
        cov = store.coverage(symbol)
        rows.append(
            {
                "ticker": cov.ticker,
                "sessions": cov.n,
                "first": cov.first,
                "last": cov.last,
                "missing_trading_days": len(cov.gaps),
                "gaps": ", ".join(d.isoformat() for d in cov.gaps[:10]) + ("..." if len(cov.gaps) > 10 else ""),
                "quotes": cov.rows,
                "mb": round(cov.bytes / 1e6, 3),
            }
        )
    return pd.DataFrame(
        rows, columns=["ticker", "sessions", "first", "last", "missing_trading_days", "gaps", "quotes", "mb"]
    )


__all__ = [
    "COLLECT_WINDOW",
    "CollectReport",
    "Coverage",
    "DATA_BRANCH",
    "INDEX_COLUMNS",
    "Outcome",
    "STORE_COLUMNS",
    "Snapshot",
    "SnapshotError",
    "SnapshotStore",
    "StoreLimits",
    "SyncResult",
    "build_snapshot",
    "collect",
    "compact_chain",
    "coverage_table",
    "due_session",
    "origins_needed",
    "relative_path",
    "scorable_origins",
    "store_bytes",
    "sync",
    "sync_root",
]
