"""The app re-imports its own code when a deploy changes it under a running
server (Streamlit Community Cloud pulls commits into the live process)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from derivx.ui import code_reload

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pkg_dir(tmp_path, monkeypatch):
    """A two-module package on sys.path, removed again afterwards."""

    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text("VALUE = 1\n")
    (pkg / "page.py").write_text("from fakepkg.core import VALUE\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield pkg
    for name in [n for n in sys.modules if n == "fakepkg" or n.startswith("fakepkg.")]:
        del sys.modules[name]
    code_reload._state().mtimes.pop("fakepkg", None)


def _rewrite(path: Path, text: str) -> None:
    before = path.stat().st_mtime_ns
    path.write_text(text)
    os.utime(path, ns=(before + 10**9, before + 10**9))  # a deploy always gives a new mtime


def test_unchanged_code_is_kept(pkg_dir):
    import fakepkg.page  # noqa: F401

    code_reload.record_loaded_modules("fakepkg")
    assert code_reload.drop_changed_modules("fakepkg") == []
    assert "fakepkg.core" in sys.modules


def test_a_changed_module_drops_the_whole_package_and_reimports_current_code(pkg_dir):
    import fakepkg.page  # noqa: F401

    code_reload.record_loaded_modules("fakepkg")
    generation = code_reload.generation()
    _rewrite(pkg_dir / "core.py", "VALUE = 1\nNEW_NAME = 2\n")
    (pkg_dir / "page.py").write_text("from fakepkg.core import NEW_NAME\n")
    # Without the check the cached old core lacks NEW_NAME: the deploy's ImportError.
    sys.modules.pop("fakepkg.page")
    with pytest.raises(ImportError, match="NEW_NAME"):
        importlib.import_module("fakepkg.page")
    assert code_reload.drop_changed_modules("fakepkg") == ["fakepkg", "fakepkg.core"]
    assert not any(n == "fakepkg" or n.startswith("fakepkg.") for n in sys.modules)
    assert code_reload.generation() == generation + 1
    assert importlib.import_module("fakepkg.page").NEW_NAME == 2


def test_modules_imported_after_the_last_record_do_not_trigger_a_reload(pkg_dir):
    import fakepkg.core  # noqa: F401

    code_reload.record_loaded_modules("fakepkg")
    import fakepkg.page  # noqa: F401  (imported later: recorded at the end of the next run)

    _rewrite(pkg_dir / "page.py", "from fakepkg.core import VALUE\nEXTRA = 3\n")
    assert code_reload.drop_changed_modules("fakepkg") == []


def test_a_removed_module_file_counts_as_changed(pkg_dir):
    import fakepkg.page  # noqa: F401

    code_reload.record_loaded_modules("fakepkg")
    (pkg_dir / "page.py").unlink()
    assert "fakepkg.page" in code_reload.drop_changed_modules("fakepkg")


def test_the_app_reloads_its_code_when_a_deploy_changes_it_under_a_running_server():
    """Run in a fresh interpreter: a real reload swaps every derivx module,
    which would leave this test process holding the old ones."""

    script = textwrap.dedent(
        f"""
        import sys
        from streamlit.testing.v1 import AppTest
        from derivx.ui import code_reload

        app = AppTest.from_file({str(ROOT / "src" / "derivx" / "ui" / "streamlit_app.py")!r}, default_timeout=300)
        app.run()
        assert not app.exception, app.exception
        app.session_state["marker"] = "made by the old code"
        app.run()
        before = code_reload.generation()
        old_module = sys.modules["derivx.calibration.production"]
        # A deploy rewrote production.py while the server kept running.
        code_reload._state().mtimes["derivx"]["derivx.calibration.production"] = 0
        app.run()
        assert not app.exception, app.exception
        assert code_reload.generation() == before + 1
        assert sys.modules["derivx.calibration.production"] is not old_module
        assert "marker" not in app.session_state  # state from the old code is dropped once
        app.session_state["marker"] = "made by the new code"
        app.run()
        assert app.session_state["marker"] == "made by the new code" and code_reload.generation() == before + 1
        print("ok")
        """
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-3000:]
