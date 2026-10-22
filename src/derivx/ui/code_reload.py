"""Re-import the package when its source changes under a running server.

Streamlit Community Cloud pulls each new commit into the app's running
process instead of restarting it. The main script is read again on the next
run, but modules already imported stay cached in ``sys.modules``, and
Streamlit reloads a changed module only when its file watcher sees the
change (none when watching is off). A new page that imports a name the
cached old module lacks then fails with ``ImportError`` until the app is
rebooted, which is how the first deploy of the rebuilt app broke.

:func:`drop_changed_modules` runs at the top of every script run, before
any other ``derivx`` import. It compares the source file of every loaded
module of the package with the timestamp recorded at the end of the
previous run (:func:`record_loaded_modules`); if any differs, every module
of the package is dropped so that this run imports the current code
consistently, and :data:`generation` counts the reloads so each session can
drop state made by the old code. Unchanged files cost one ``stat`` per
module per run. Dependency changes (requirements.txt) still need a reboot:
this only refreshes the package's own code.

Only the standard library is used here, so the check works even when every
other module of the package is stale.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PACKAGE = "derivx"
_HOLDER = "_derivx_code_reload_state"  # a module kept in sys.modules, so the state survives reruns


@dataclass
class _State:
    mtimes: Dict[str, Dict[str, int]] = field(default_factory=dict)  # package -> {module: source mtime_ns}
    generation: int = 0  # reloads so far in this process


def _state() -> _State:
    holder = sys.modules.get(_HOLDER)
    if holder is None:
        holder = types.ModuleType(_HOLDER)
        holder.state = _State()  # type: ignore[attr-defined]
        sys.modules[_HOLDER] = holder
    return getattr(holder, "state")


def _in_package(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _source_mtime(module: types.ModuleType) -> Optional[int]:
    path = getattr(module, "__file__", None)
    if not path:
        return None
    try:
        return os.stat(path).st_mtime_ns
    except OSError:  # removed or moved by the update: the module is stale
        return -1


def _loaded(package: str) -> Dict[str, types.ModuleType]:
    return {name: mod for name, mod in list(sys.modules.items()) if mod is not None and _in_package(name, package)}


def generation() -> int:
    """How many times the package was reloaded in this process."""

    return _state().generation


def drop_changed_modules(package: str = PACKAGE) -> List[str]:
    """Drop every loaded module of ``package`` if the source of any module
    recorded by :func:`record_loaded_modules` changed since; returns the
    dropped names (empty when nothing changed)."""

    state = _state()
    recorded = state.mtimes.get(package, {})
    loaded = _loaded(package)
    changed = [name for name, mod in loaded.items() if name in recorded and _source_mtime(mod) != recorded[name]]
    if not changed:
        return []
    for name in loaded:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    state.mtimes[package] = {}
    state.generation += 1
    return sorted(loaded)


def record_loaded_modules(package: str = PACKAGE) -> None:
    """Record the source timestamp of every loaded module of ``package`` not
    recorded yet (call at the end of each run, after every import)."""

    recorded = _state().mtimes.setdefault(package, {})
    for name, mod in _loaded(package).items():
        if name not in recorded:
            mtime = _source_mtime(mod)
            if mtime is not None:
                recorded[name] = mtime
