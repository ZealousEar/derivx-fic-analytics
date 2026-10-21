"""Run manifests and export bundles for rates results (G14).

A :class:`RunManifest` records what produced a result, so it can be
reproduced and audited:

* ``code``     - git commit of the checkout and whether it had uncommitted
  changes (``sha`` is ``"unknown"`` outside a git checkout);
* ``packages`` - Python and package versions;
* ``inputs``   - what the user entered (trades, shocks, settings);
* ``data``     - valuation date, each curve's quotes with their SHA-256 and
  source label, and the history store behind them: its label, the SHA-256
  of its rows, and per dataset the publisher, the URLs it is downloaded
  from and the dates it covers;
* ``model``    - model settings (interpolation, vol models, bumps, ...);
* ``seed``     - the random seed, ``None`` when nothing random was used
  (every rates computation here is deterministic).

The keys match the analytics lab's manifest (``manifest_version``,
``kind``, ``created_at_utc``, ``code``, ``packages``, ``inputs``, ``data``,
``model``, ``seed``). :func:`bundle_zip` packs a manifest with CSV tables
(fixed timestamps, so the same inputs give the same bytes apart from
``created_at_utc``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import platform
import subprocess
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from derivx.data import rates_sources as rs
from derivx.data.rates_history import ORIGIN_SNAPSHOT, HistoryStore, dataset_info, load_manifest
from derivx.rates.curves import Curve, Market

MANIFEST_VERSION = 1
PACKAGES = ("derivx", "numpy", "scipy", "pandas", "pyarrow", "plotly", "streamlit")
DATASET_URLS: Dict[str, Sequence[str]] = {
    "UST_PAR": (rs.TREASURY_CSV_URL, rs.FRED_CSV_URL),
    "USD_SWAP_ICE": (rs.FRED_CSV_URL,),
    "GBP_OIS_BOE": (rs.BOE_URL,),
    "EUR_AAA_ECB": (rs.ECB_URL,),
    "FIXINGS": (rs.NYFED_URL, rs.ECB_URL, rs.FRED_CSV_URL),
}


@lru_cache(maxsize=1)
def git_revision() -> Dict[str, object]:
    """``{"sha", "dirty"}`` of the checkout this package runs from."""

    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5, check=True).stdout.strip()  # fmt: skip
        return {"sha": sha or "unknown", "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"sha": "unknown", "dirty": None}


@lru_cache(maxsize=1)
def package_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def jsonable(value: object) -> object:
    """``value`` with dates, numpy types and non-finite floats made JSON
    safe (NaN and infinities become None)."""

    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: object) -> str:
    """SHA-256 of a value's canonical JSON (sorted keys)."""

    return sha256_bytes(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class RunManifest:
    """What produced one result (see the module docstring)."""

    kind: str
    inputs: Mapping[str, object] = field(default_factory=dict)
    data: Mapping[str, object] = field(default_factory=dict)
    model: Mapping[str, object] = field(default_factory=dict)
    seed: Optional[int] = None
    created_at_utc: str = ""
    code: Mapping[str, object] = field(default_factory=dict)
    packages: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return jsonable(
            {
                "manifest_version": MANIFEST_VERSION,
                "kind": self.kind,
                "created_at_utc": self.created_at_utc,
                "code": dict(self.code),
                "packages": dict(self.packages),
                "platform": {"system": platform.system(), "machine": platform.machine()},
                "inputs": dict(self.inputs),
                "data": dict(self.data),
                "model": dict(self.model),
                "seed": self.seed,
            }
        )

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False).encode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RunManifest":
        if payload.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(f"manifest version {payload.get('manifest_version')!r} is not {MANIFEST_VERSION}")
        return cls(
            kind=str(payload["kind"]),
            inputs=dict(payload.get("inputs") or {}),
            data=dict(payload.get("data") or {}),
            model=dict(payload.get("model") or {}),
            seed=payload.get("seed"),
            created_at_utc=str(payload.get("created_at_utc", "")),
            code=dict(payload.get("code") or {}),
            packages=dict(payload.get("packages") or {}),
        )


def run_manifest(
    kind: str,
    inputs: Optional[Mapping[str, object]] = None,
    data: Optional[Mapping[str, object]] = None,
    model: Optional[Mapping[str, object]] = None,
    seed: Optional[int] = None,
    created_at: Optional[dt.datetime] = None,
) -> RunManifest:
    when = created_at or dt.datetime.now(dt.timezone.utc)
    return RunManifest(kind, dict(inputs or {}), dict(data or {}), dict(model or {}), seed,
                       when.isoformat(timespec="seconds"), git_revision(), package_versions())  # fmt: skip


# --------------------------------------------------------------------------
# Market data provenance
# --------------------------------------------------------------------------


def curve_record(key: str, curve: Curve, source: Optional[str] = None) -> Dict[str, object]:
    """One curve: conventions, interpolation, quotes (decimal rates; futures
    rates for futures) and their SHA-256, and where they came from."""

    quotes = [[q.kind, q.tenor, q.rate] for q in curve.quotes]
    index = getattr(curve, "index", None)
    return {
        "key": key,
        "asof": curve.asof,
        "conventions": curve.conventions.name if curve.conventions is not None else None,
        "index": index.name if index is not None else None,
        "basis_index": getattr(curve, "basis_index", None),
        "interpolation": curve.interpolation,
        "convexity": None if curve.convexity is None else {
            "mean_reversion": curve.convexity.mean_reversion, "sigma": curve.convexity.sigma},
        "quotes": quotes,
        "quotes_sha256": sha256_json(quotes),
        "source": source or "",
    }  # fmt: skip


def _sample_urls() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    try:
        files = load_manifest()["files"]
    except (OSError, ValueError, KeyError):
        return out
    for entry in files:
        out.setdefault(str(entry["dataset"]), []).extend(str(u) for u in entry.get("urls", []))
    return out


def store_record(store: HistoryStore, datasets: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """The history store: label, SHA-256 of its rows and, per dataset, the
    publisher, download URLs (the committed sample's exact requests where
    the rows come from it) and the dates covered."""

    sample = _sample_urls()
    rows = []
    for key in datasets if datasets is not None else store.datasets():
        if not store.has(key):
            continue
        info = dataset_info(key)
        frame = store.rows(key)
        origins = sorted({str(o) for o in frame["origin"]}) if "origin" in frame else []
        urls = list(DATASET_URLS.get(key, ()))
        if ORIGIN_SNAPSHOT in origins:
            urls += sample.get(key, [])
        rows.append({
            "dataset": key,
            "title": info.title,
            "publisher": info.source,
            "origins": origins,
            "urls": urls,
            "first": min(frame["asof"]) if len(frame) else None,
            "last": max(frame["asof"]) if len(frame) else None,
            "rows": int(len(frame)),
        })  # fmt: skip
    return {"label": store.label, "rows_sha256": store.fingerprint, "datasets": rows}


def market_record(
    market: Market,
    sources: Optional[Mapping[str, str]] = None,
    store: Optional[HistoryStore] = None,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """``data`` section for a result priced on ``market``: valuation date,
    every curve (:func:`curve_record`, ``sources`` maps curve keys to their
    source labels), the history store (:func:`store_record`) and one
    SHA-256 over all of it."""

    sources = dict(sources or {})
    curves = [curve_record(k, market.get_curve(k), sources.get(k)) for k in market.curve_keys
              if k in ("curve", "govt") or not market.projections.is_discount(k)]  # fmt: skip
    record: Dict[str, object] = {
        "valuation_date": market.asof,
        "curves": curves,
        "index_fixings": {k: len(v) for k, v in market.index_fixings.items()},
    }
    if store is not None:
        record["store"] = store_record(store, datasets)
    record["data_sha256"] = sha256_json(record)
    return record


# --------------------------------------------------------------------------
# Bundles
# --------------------------------------------------------------------------


def bundle_zip(manifest: RunManifest, tables: Mapping[str, pd.DataFrame]) -> bytes:
    """ZIP with ``manifest.json`` and ``<name>.csv`` per table."""

    buf = io.BytesIO()
    stamp = (2000, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("manifest.json", date_time=stamp)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, manifest.to_json())
        for name, table in tables.items():
            info = zipfile.ZipInfo(f"{name}.csv", date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, table.to_csv(index=False))
    return buf.getvalue()


__all__ = [
    "RunManifest",
    "bundle_zip",
    "curve_record",
    "git_revision",
    "jsonable",
    "market_record",
    "package_versions",
    "run_manifest",
    "sha256_json",
    "store_record",
]
