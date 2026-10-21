"""Code provenance shared by every run manifest.

Both manifest flavours, the analytics lab's (:mod:`derivx.validation.manifest`)
and the Rates tabs' and CLI's (:mod:`derivx.rates.manifest`), record the code
version with :func:`git_revision` and make values JSON-safe with
:func:`jsonable`, so a run anywhere records the same ``code`` block. In the
Docker image, which has no ``.git``, the commit comes from the
``DERIVX_GIT_SHA`` environment variable (the image's ``GIT_SHA`` build arg).

Light on purpose (standard library, numpy and pandas only): the CLI imports
it without loading the option-chain or Streamlit code.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import platform
import re
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

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


def package_versions(names: Iterable[str]) -> Dict[str, str]:
    """``{"python": ..., name: version}`` for the installed distributions
    (``"not installed"`` for a missing one)."""

    out = {"python": platform.python_version()}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def jsonable(value: object) -> object:
    """``value`` with dates and timestamps (ISO strings), numpy scalars and
    arrays, and non-finite floats (NaN and infinities become None) made
    JSON-safe; anything else unknown becomes its ``str``."""

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


__all__ = ["GIT_SHA_ENV", "git_revision", "jsonable", "package_versions"]
