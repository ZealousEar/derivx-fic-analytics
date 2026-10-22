"""Run manifests (G14): what produced a result, so it can be reproduced.

A manifest is a JSON-friendly dict with the code version (git commit and
whether the tree had uncommitted changes, from :mod:`derivx.provenance`,
shared with the Rates manifests), package versions, the inputs and
filters, the data's source, fetch time and SHA-256, the model settings and
the random seed (None when nothing random was used). :func:`results_zip`
bundles it with CSV tables of the result (parameters, metrics, ...).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import platform
import sys
import zipfile
from functools import lru_cache
from typing import Dict, Mapping, Optional

import pandas as pd

from derivx.provenance import GIT_SHA_ENV, git_revision, jsonable
from derivx.provenance import package_versions as _versions

PACKAGES = ("derivx", "numpy", "scipy", "pandas", "streamlit", "plotly", "yfinance", "pyarrow")
MANIFEST_VERSION = 1


@lru_cache(maxsize=1)
def package_versions() -> Dict[str, str]:
    return _versions(PACKAGES)


def build_manifest(
    kind: str,
    *,
    inputs: Mapping[str, object],
    data: Mapping[str, object],
    model: Mapping[str, object],
    seed: Optional[int] = None,
    created_at: Optional[dt.datetime] = None,
) -> Dict[str, object]:
    """Manifest for one run of ``kind`` (e.g. 'calibration'). ``data``
    should carry ``source``, ``fetched_at_utc`` and ``data_sha256`` (see
    :meth:`derivx.data.chain_schema.ChainDataset.provenance`)."""

    when = created_at or dt.datetime.now(dt.timezone.utc)
    return jsonable(
        {
            "manifest_version": MANIFEST_VERSION,
            "kind": kind,
            "created_at_utc": when.isoformat(timespec="seconds"),
            "code": {**git_revision(), "package": "derivx"},
            "packages": package_versions(),
            "platform": {"system": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0]},
            "inputs": dict(inputs),
            "data": dict(data),
            "model": dict(model),
            "seed": seed,
        }
    )


def manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    return json.dumps(jsonable(manifest), indent=2, sort_keys=False, allow_nan=False).encode("utf-8")


def results_zip(manifest: Mapping[str, object], tables: Mapping[str, pd.DataFrame]) -> bytes:
    """ZIP with ``manifest.json`` and one CSV per table (``<name>.csv``).
    Entries carry a fixed timestamp so the same inputs give the same bytes."""

    buf = io.BytesIO()
    stamp = (2000, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("manifest.json", date_time=stamp), manifest_bytes(manifest))
        for name, table in tables.items():
            info = zipfile.ZipInfo(f"{name}.csv", date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, table.to_csv(index=False))
    return buf.getvalue()


__all__ = [
    "GIT_SHA_ENV",
    "build_manifest",
    "git_revision",
    "jsonable",
    "manifest_bytes",
    "package_versions",
    "results_zip",
]
