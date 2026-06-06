"""Utility script to wake the Streamlit Community Cloud app if it is sleeping."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

STREAMLIT_URL = os.environ.get("STREAMLIT_APP_URL", "https://derivx.streamlit.app/")
WAKE_BUTTON_XPATH = "//button[contains(text(),'Yes, get this app back up')]"
WAIT_SECONDS = int(os.environ.get("STREAMLIT_WAKE_WAIT", "20"))
# Paths optionally provided by the CI setup-chrome step so the chromedriver
# always matches the installed browser. When unset, Selenium Manager (bundled
# with Selenium >= 4.6) resolves a matching driver automatically.
CHROME_BINARY = os.environ.get("CHROME_BINARY")
CHROMEDRIVER_PATH = os.environ.get("CHROMEDRIVER_PATH")
PAGE_LOAD_TIMEOUT = int(os.environ.get("STREAMLIT_PAGE_LOAD_TIMEOUT", "60"))


@dataclass
class WakeResult:
    status: int
    message: str


def configure_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    if CHROME_BINARY:
        options.binary_location = CHROME_BINARY
    # Prefer an explicit chromedriver path (from CI); otherwise let Selenium
    # Manager download a driver that matches the installed Chrome.
    service = Service(executable_path=CHROMEDRIVER_PATH) if CHROMEDRIVER_PATH else Service()
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def wake_streamlit_app() -> WakeResult:
    driver = configure_driver()
    try:
        print(f"Opening {STREAMLIT_URL}")
        driver.get(STREAMLIT_URL)
        wait = WebDriverWait(driver, WAIT_SECONDS)

        try:
            button = wait.until(EC.element_to_be_clickable((By.XPATH, WAKE_BUTTON_XPATH)))
        except TimeoutException:
            return WakeResult(0, "Wake button not found; app is likely already awake.")

        print("Wake button located. Clicking...")
        button.click()

        try:
            wait.until(EC.invisibility_of_element_located((By.XPATH, WAKE_BUTTON_XPATH)))
            return WakeResult(0, "Wake button clicked and disappeared; app should be waking up.")
        except TimeoutException:
            return WakeResult(1, "Wake button clicked but did not disappear; manual check required.")

    except Exception as exc:  # pylint: disable=broad-except
        return WakeResult(1, f"Unexpected error: {exc}")
    finally:
        # Give the app a moment to start before closing the browser to avoid cutting the request short.
        time.sleep(2)
        driver.quit()


def main() -> int:
    if not STREAMLIT_URL:
        print("STREAMLIT_APP_URL is not set.")
        return 1

    result = wake_streamlit_app()
    print(result.message)
    return result.status


if __name__ == "__main__":
    sys.exit(main())
