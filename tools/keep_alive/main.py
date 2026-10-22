"""Wake the Streamlit Community Cloud app and fail unless it actually renders.

An HTTP status cannot tell a live app from a missing one: both
https://derivx.streamlit.app and a nonexistent ``*.streamlit.app`` answer
``303`` to share.streamlit.io's auth page, so a ``curl`` check always passes.
Only a real browser follows that redirect and opens the app's websocket, and
only such a session counts as traffic that keeps a Community Cloud app
awake. This script therefore drives headless Chrome:

1. open ``STREAMLIT_APP_URL``;
2. if Streamlit Cloud shows its sleep page, click "Yes, get this app back up";
3. wait until the app itself renders: the page or one of its frames (Cloud
   serves the app in an iframe at ``/~/+/``) shows ``STREAMLIT_APP_MARKER``
   (default ``NAVIGATION``, the app's sidebar heading);
4. exit 0 only then. A "not found" page exits 1 at once; a sign-in wall, an
   app error page, a Streamlit exception element (an uncaught error in the
   script, redacted or not) or anything else that does not become the app
   within the timeout exits 1 too, so the workflow run fails when the app is
   down.

Environment: ``STREAMLIT_APP_URL``, ``STREAMLIT_APP_MARKER``,
``STREAMLIT_WAKE_TIMEOUT`` (seconds to wait for the app to render, default
300: waking a sleeping app takes a minute or two), ``STREAMLIT_PAGE_LOAD_TIMEOUT``
(seconds, default 60), and optionally ``CHROME_BINARY`` / ``CHROMEDRIVER_PATH``
(otherwise Selenium Manager finds Chrome and a matching driver).
"""

from __future__ import annotations

import enum
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, Tuple

DEFAULT_URL = "https://derivx.streamlit.app/"
DEFAULT_MARKER = "NAVIGATION"
WAKE_BUTTON_TEXT = "Yes, get this app back up"
WAKE_BUTTON_XPATH = f"//button[contains(., '{WAKE_BUTTON_TEXT}')]"
# Streamlit draws every uncaught script exception in this element, whatever
# client.showErrorDetails is set to (the app itself never calls st.exception).
EXCEPTION_SELECTOR = '[data-testid="stException"]'

# Page texts of Streamlit Community Cloud, matched case-insensitively.
NOT_FOUND_TEXTS = ("you do not have access to this app or it does not exist",)
APP_ERROR_TEXTS = (
    "error running app",
    "this app has encountered an error",  # Streamlit's redacted exception (showErrorDetails "type"/"none")
    "traceback (most recent call last)",
    "error installing requirements",
)
# Streamlit's full exception element (showErrorDetails "full"/"stacktrace") prints a line that is just
# "Traceback:" above the frames, not Python's "Traceback (most recent call last):".
TRACEBACK_LINE = re.compile(r"^\s*traceback:\s*$", re.IGNORECASE | re.MULTILINE)
BOOTING_TEXTS = ("your app is in the oven", "spinning up", "please wait")
SIGN_IN_TEXTS = ("sign in to continue", "please sign in")


class State(str, enum.Enum):
    """What the browser shows. Only ``UP`` is healthy."""

    UP = "up"
    ASLEEP = "asleep"
    BOOTING = "booting"
    NOT_FOUND = "not_found"
    SIGN_IN = "sign_in"
    APP_ERROR = "app_error"
    UNKNOWN = "unknown"


FATAL = (State.NOT_FOUND,)  # waiting will not change this; a sign-in wall may be a transient redirect


@dataclass(frozen=True)
class Snapshot:
    """The browser's current URL, the visible text of the page and of each of
    its iframes, and how many Streamlit exception elements they hold."""

    url: str
    texts: Tuple[str, ...]
    exceptions: int = 0


def classify(snap: Snapshot, marker: str = DEFAULT_MARKER) -> State:
    """State of the app from one snapshot. An error page, a Streamlit
    exception element or a traceback wins over the marker (a crashed script
    can still draw its sidebar)."""

    text = "\n".join(snap.texts)
    low = text.lower()
    if "/errors/not_found" in snap.url or any(t in low for t in NOT_FOUND_TEXTS):
        return State.NOT_FOUND
    if WAKE_BUTTON_TEXT.lower() in low:
        return State.ASLEEP
    if snap.exceptions > 0 or any(t in low for t in APP_ERROR_TEXTS) or TRACEBACK_LINE.search(text):
        return State.APP_ERROR
    if marker and marker in text:
        return State.UP
    if any(t in low for t in BOOTING_TEXTS):
        return State.BOOTING
    if any(t in low for t in SIGN_IN_TEXTS):
        return State.SIGN_IN
    return State.UNKNOWN


class Page(Protocol):
    def open(self, url: str) -> None: ...

    def snapshot(self) -> Snapshot: ...

    def click_wake(self) -> bool: ...


@dataclass(frozen=True)
class WakeResult:
    status: int  # process exit code: 0 only when the app rendered
    state: State
    woke: bool
    message: str


def check_app(
    page: Page,
    url: str = DEFAULT_URL,
    marker: str = DEFAULT_MARKER,
    timeout: float = 300.0,
    poll: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WakeResult:
    """Open ``url``, wake the app if it sleeps, and wait up to ``timeout``
    seconds for it to render ``marker``."""

    if not url:
        return WakeResult(1, State.UNKNOWN, False, "STREAMLIT_APP_URL is empty")
    page.open(url)
    deadline = clock() + timeout
    woke = False
    state = State.UNKNOWN
    snap = Snapshot(url, ())
    while True:
        snap = page.snapshot()
        state = classify(snap, marker)
        if state is State.UP:
            how = "after waking it" if woke else "already awake"
            return WakeResult(0, state, woke, f"App rendered ({how}): found {marker!r} at {snap.url}")
        if state in FATAL:
            return WakeResult(1, state, woke, f"App is not reachable ({state.value}) at {snap.url}")
        if state is State.ASLEEP and not woke:
            woke = page.click_wake()
            print("Sleep page found; " + ("clicked the wake button." if woke else "the wake button is not clickable."))
        if clock() >= deadline:
            last = f"last state {state.value} at {snap.url}"
            return WakeResult(1, state, woke, f"App did not render {marker!r} within {timeout:.0f}s; {last}")
        sleep(poll)


class SeleniumPage:
    """:class:`Page` on headless Chrome (Selenium imported only here, so the
    logic above is testable without a browser)."""

    def __init__(
        self, page_load_timeout: float = 60.0, binary: Optional[str] = None, driver_path: Optional[str] = None
    ) -> None:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        for arg in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"):
            options.add_argument(arg)
        options.add_argument("--window-size=1920,1080")
        if binary:
            options.binary_location = binary
        service = Service(executable_path=driver_path) if driver_path else Service()
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_page_load_timeout(page_load_timeout)

    def open(self, url: str) -> None:
        self.driver.get(url)

    def _body_text(self) -> str:
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.common.by import By

        try:
            return self.driver.find_element(By.TAG_NAME, "body").text
        except WebDriverException:
            return ""

    def _exception_count(self) -> int:
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.common.by import By

        try:
            return len(self.driver.find_elements(By.CSS_SELECTOR, EXCEPTION_SELECTOR))
        except WebDriverException:
            return 0

    def snapshot(self) -> Snapshot:
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.common.by import By

        texts = [self._body_text()]
        exceptions = self._exception_count()
        for i in range(len(self.driver.find_elements(By.TAG_NAME, "iframe"))):
            try:
                self.driver.switch_to.frame(self.driver.find_elements(By.TAG_NAME, "iframe")[i])
                texts.append(self._body_text())
                exceptions += self._exception_count()
            except (WebDriverException, IndexError):
                pass
            finally:
                self.driver.switch_to.default_content()
        return Snapshot(self.driver.current_url, tuple(texts), exceptions)

    def click_wake(self) -> bool:
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.common.by import By

        try:
            buttons = self.driver.find_elements(By.XPATH, WAKE_BUTTON_XPATH)
            if not buttons:
                return False
            buttons[0].click()
            return True
        except WebDriverException:
            return False

    def close(self) -> None:
        self.driver.quit()


def main(argv: Sequence[str] = ()) -> int:
    url = os.environ.get("STREAMLIT_APP_URL", DEFAULT_URL)
    marker = os.environ.get("STREAMLIT_APP_MARKER", DEFAULT_MARKER)
    timeout = float(os.environ.get("STREAMLIT_WAKE_TIMEOUT", "300"))
    try:
        page = SeleniumPage(
            float(os.environ.get("STREAMLIT_PAGE_LOAD_TIMEOUT", "60")),
            os.environ.get("CHROME_BINARY"),
            os.environ.get("CHROMEDRIVER_PATH"),
        )
    except Exception as exc:  # noqa: BLE001 - any driver start-up failure is a failed check
        print(f"Failed to start Chrome WebDriver: {exc}")
        return 1
    try:
        print(f"Opening {url}")
        result = check_app(page, url, marker, timeout)
    except Exception as exc:  # noqa: BLE001 - report, do not hide, a browser failure
        result = WakeResult(1, State.UNKNOWN, False, f"Browser error: {exc}")
    finally:
        page.close()
    print(result.message)
    return result.status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
