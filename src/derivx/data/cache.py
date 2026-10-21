"""On-disk cache for downloaded market data.

Each entry is one request (a source plus its parameters): the normalised
long-format frame as Parquet and a JSON sidecar with the URL (credentials
removed), the UTC fetch time, the SHA-256 of the raw payload and the date
range. Writes go to a temporary file first and are then renamed, so readers
in other sessions never see half a file.

The cache lives in ``data/cache/`` (gitignored) unless ``DERIVX_CACHE_DIR``
points elsewhere. An unwritable directory disables writes instead of
failing: callers still get their data, only not cached.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
CACHE_ENV = "DERIVX_CACHE_DIR"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def cache_dir() -> Path:
    """The cache directory: ``$DERIVX_CACHE_DIR`` or ``data/cache``."""

    override = os.environ.get(CACHE_ENV)
    return Path(override).expanduser() if override else DEFAULT_CACHE_DIR


@dataclass
class CacheEntry:
    """One cached request. ``meta`` holds ``source``, ``params``, ``url``,
    ``fetched_at`` (ISO 8601 UTC), ``sha256`` (of the raw payload),
    ``rows``, ``first`` and ``last`` (as-of dates)."""

    key: str
    path: Path
    meta: Dict[str, object]
    _frame: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def fetched_at(self) -> dt.datetime:
        return dt.datetime.fromisoformat(str(self.meta["fetched_at"]))

    def age_seconds(self, now: Optional[dt.datetime] = None) -> float:
        return ((now or utc_now()) - self.fetched_at).total_seconds()

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = pd.read_parquet(self.path)
        return self._frame.copy()


class DiskCache:
    """Request-keyed cache of long-format frames (see module docstring)."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else cache_dir()
        self.write_error: Optional[str] = None

    @staticmethod
    def key(source: str, params: Mapping[str, object]) -> str:
        """Stable key from the source name and its request parameters."""

        canonical = json.dumps({k: params[k] for k in sorted(params)}, default=str, separators=(",", ":"))
        return f"{source}-{hashlib.sha1(canonical.encode()).hexdigest()[:16]}"

    def _paths(self, key: str):
        return self.root / f"{key}.parquet", self.root / f"{key}.json"

    def load(self, key: str) -> Optional[CacheEntry]:
        data, meta = self._paths(key)
        if not (data.exists() and meta.exists()):
            return None
        try:
            info = json.loads(meta.read_text())
        except (OSError, ValueError):
            return None
        return CacheEntry(key, data, info)

    def entries(self, source: Optional[str] = None) -> List[CacheEntry]:
        """Every readable entry (of one ``source`` if given), newest first."""

        if not self.root.is_dir():
            return []
        out = []
        for meta in self.root.glob("*.json"):
            entry = self.load(meta.stem)
            if entry is not None and (source is None or entry.meta.get("source") == source):
                out.append(entry)
        return sorted(out, key=lambda e: str(e.meta.get("fetched_at", "")), reverse=True)

    def store(self, key: str, frame: pd.DataFrame, payload: bytes, meta: Mapping[str, object]) -> Optional[CacheEntry]:
        """Write an entry; None (and ``write_error`` set) if the directory
        is not writable."""

        info = dict(meta)
        info.setdefault("fetched_at", utc_now().isoformat(timespec="seconds"))
        info["sha256"] = sha256(payload)
        info["rows"] = int(len(frame))
        if len(frame) and "asof" in frame:
            info["first"], info["last"] = str(min(frame["asof"])), str(max(frame["asof"]))
        data, meta_path = self._paths(key)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._atomic(data, lambda tmp: frame.to_parquet(tmp, index=False))
            self._atomic(meta_path, lambda tmp: Path(tmp).write_text(json.dumps(info, indent=1, default=str)))
        except OSError as exc:
            self.write_error = f"cache directory {self.root} is not writable ({exc})"
            return None
        return CacheEntry(key, data, info, frame.copy())

    def _atomic(self, path: Path, write) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-", suffix=path.suffix)
        os.close(fd)
        try:
            write(tmp)
            os.chmod(tmp, 0o644)  # mkstemp creates 0600; the cache is plain public data
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
