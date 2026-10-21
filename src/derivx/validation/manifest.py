"""Run manifests (G14): what produced a result, so it can be reproduced.

A manifest is a JSON-friendly dict with the code version (git commit and
whether the tree had uncommitted changes), package versions, the inputs and
filters, the data's source, fetch time and SHA-256, the model settings and
the random seed (None when nothing random was used). :func:`results_zip`
bundles it with CSV tables of the result (parameters, metrics, ...).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import zipfile
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

PACKAGES = ("derivx", "numpy", "scipy", "pandas", "streamlit", "plotly", "yfinance", "pyarrow")
MANIFEST_VERSION = 1


GIT_SHA_ENV = "DERIVX_GIT_SHA"  # set by the Docker image (build arg GIT_SHA), which has no .git


@lru_cache(maxsize=1)
def git_revision() -> Dict[str, object]:
    """Commit of the checkout this package runs from: ``{"sha", "dirty"}``.

    Outside a git checkout (an installed wheel, or the Docker image, which
    has no .git) the sha comes from the ``DERIVX_GIT_SHA`` environment
    variable when it holds a 40-hex commit id (``dirty`` is then None:
    unknown), else it is 'unknown'."""

    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return {"sha": sha or "unknown", "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        env_sha = os.environ.get(GIT_SHA_ENV, "").strip().lower()
        return {"sha": env_sha if re.fullmatch(r"[0-9a-f]{40}", env_sha) else "unknown", "dirty": None}


@lru_cache(maxsize=1)
def package_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def jsonable(value):
    """``value`` with timestamps, numpy scalars/arrays and non-finite floats
    converted to JSON-safe types (NaN/inf -> None)."""

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
