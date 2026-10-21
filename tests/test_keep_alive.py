"""Keep-alive check (UIE-20): it passes only when the app's own UI renders.

The page texts below are what headless Chromium showed on 2026-10-20 for the
live app (whose UI Streamlit Cloud serves in an iframe at ``/~/+/``) and for
a nonexistent ``*.streamlit.app`` (both answer HTTP 303, which the old curl
check counted as "alive"). No browser or network is used here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("derivx_keep_alive", ROOT / "tools" / "keep_alive" / "main.py")
keep_alive = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = keep_alive
_spec.loader.exec_module(keep_alive)

State, Snapshot, classify, check_app = keep_alive.State, keep_alive.Snapshot, keep_alive.classify, keep_alive.check_app

APP_URL = "https://derivx.streamlit.app/"
APP = Snapshot(
    APP_URL,
    (
        "",
        "/ NAVIGATION\n\n[ EQUITY ]\n\n[ FX ]\n\n[ RATES ]\n\n[ ANALYTICS ]\n\n"
        "/ EQUITY PARAMS\nDERIVX\nFIC DERIVATIVES ANALYTICS",
    ),
)
WRAPPER_ONLY = Snapshot(APP_URL, ("Hosted with Streamlit\n\nCreated by zealousear",))
NOT_FOUND = Snapshot(
    "https://share.streamlit.io/errors/not_found",
    (
        "Explore\nDiscuss\nSign in\nSign up\nYou do not have access to this app or it does not exist\n\n"
        "Please sign in to continue.\n\nIf you believe this is a bug, contact support.",
    ),
)
AUTH_REDIRECT = Snapshot("https://share.streamlit.io/-/auth/app?redirect_uri=https%3A%2F%2Fderivx.streamlit.app", ("",))
ASLEEP = Snapshot(
    APP_URL,
    (
        "Zzzz\n\nThis app has gone to sleep due to inactivity. Would you like to wake it back up?\n\n"
        "Yes, get this app back up!",
    ),
)
BOOTING = Snapshot(APP_URL, ("Your app is in the oven",))
CRASHED = Snapshot(APP_URL, ("", "/ NAVIGATION\n[ EQUITY ]\nValueError: boom\nTraceback (most recent call last):"))
CLOUD_ERROR = Snapshot(APP_URL, ("Oh no.\n\nError running app. If this keeps happening, please contact support.",))


@pytest.mark.parametrize(
    "snap, state",
    [
        (APP, State.UP),
        (WRAPPER_ONLY, State.UNKNOWN),  # the Cloud wrapper before the app iframe renders is not "up"
        (NOT_FOUND, State.NOT_FOUND),
        (AUTH_REDIRECT, State.UNKNOWN),
        (ASLEEP, State.ASLEEP),
        (BOOTING, State.BOOTING),
        (CRASHED, State.APP_ERROR),  # a traceback wins over the sidebar heading
        (CLOUD_ERROR, State.APP_ERROR),
    ],
)
def test_classify_real_page_texts(snap, state):
    assert classify(snap) is state


def test_the_marker_is_configurable():
    assert classify(APP, marker="RATES") is State.UP
    assert classify(APP, marker="SOME OTHER APP") is State.UNKNOWN


class FakePage:
    """Replays snapshots; a click on the wake button moves to the next one."""

    def __init__(self, snapshots: List[Snapshot], clickable: bool = True):
        self.snapshots = list(snapshots)
        self.clickable = clickable
        self.opened: List[str] = []
        self.clicks = 0

    def open(self, url: str) -> None:
        self.opened.append(url)

    def snapshot(self) -> Snapshot:
        return self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]

    def click_wake(self) -> bool:
        self.clicks += 1
        return self.clickable


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _check(page, timeout=60.0):
    clock = FakeClock()
    return check_app(page, APP_URL, timeout=timeout, poll=5.0, clock=clock, sleep=clock.sleep), clock


def test_an_awake_app_passes_at_once():
    page = FakePage([APP])
    result, clock = _check(page)
    assert (result.status, result.state, result.woke) == (0, State.UP, False)
    assert page.opened == [APP_URL] and clock.now == 0.0


def test_a_sleeping_app_is_woken_and_passes_only_once_it_renders():
    page = FakePage([ASLEEP, BOOTING, BOOTING, WRAPPER_ONLY, APP])
    result, clock = _check(page)
    assert (result.status, result.state, result.woke) == (0, State.UP, True)
    assert page.clicks == 1 and clock.now == 20.0
    assert "after waking" in result.message


def test_a_missing_app_fails_immediately():
    result, clock = _check(FakePage([NOT_FOUND]))
    assert (result.status, result.state) == (1, State.NOT_FOUND)
    assert clock.now == 0.0


@pytest.mark.parametrize("stuck", [AUTH_REDIRECT, WRAPPER_ONLY, BOOTING, CRASHED, CLOUD_ERROR])
def test_an_app_that_never_renders_fails_after_the_timeout(stuck):
    result, clock = _check(FakePage([stuck]), timeout=30.0)
    assert result.status == 1 and result.state is classify(stuck)
    assert clock.now == 30.0 and "within 30s" in result.message


def test_a_wake_button_that_cannot_be_clicked_fails():
    page = FakePage([ASLEEP], clickable=False)
    result, _ = _check(page, timeout=20.0)
    assert (result.status, result.state, result.woke) == (1, State.ASLEEP, False)
    assert page.clicks == 5  # retried on every poll


def test_an_empty_url_fails_without_opening_a_browser():
    page = FakePage([APP])
    assert check_app(page, "").status == 1 and page.opened == []


def test_main_reports_a_driver_start_failure_as_a_failed_check(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError("no chrome here")

    monkeypatch.setattr(keep_alive, "SeleniumPage", broken)
    assert keep_alive.main() == 1
    assert "no chrome here" in capsys.readouterr().out


def test_main_closes_the_browser_and_returns_the_check_status(monkeypatch):
    page = FakePage([APP])
    page.closed = False

    def close():
        page.closed = True

    page.close = close
    monkeypatch.setattr(keep_alive, "SeleniumPage", lambda *a, **k: page)
    monkeypatch.setenv("STREAMLIT_APP_URL", "https://example.streamlit.app/")
    assert keep_alive.main() == 0
    assert page.opened == ["https://example.streamlit.app/"] and page.closed


def test_the_workflow_is_one_browser_check_that_never_commits():
    workflows = ROOT / ".github" / "workflows"
    text = (workflows / "keep-alive.yml").read_text()
    assert not (workflows / "wake-streamlit.yml").exists()
    assert "tools/keep_alive/main.py" in text and "schedule:" in text
    for forbidden in ("git commit", "git push", "contents: write", "last-ping", "curl"):
        assert forbidden not in text
    assert not (ROOT / ".github" / "last-ping.txt").exists()


def test_the_workflow_marker_is_text_the_app_shell_draws():
    workflow = (ROOT / ".github" / "workflows" / "keep-alive.yml").read_text()
    marker = workflow.split("STREAMLIT_APP_MARKER:", 1)[1].split("\n", 1)[0].strip()
    assert marker == keep_alive.DEFAULT_MARKER
    assert marker in (ROOT / "src" / "derivx" / "ui" / "streamlit_app.py").read_text()
