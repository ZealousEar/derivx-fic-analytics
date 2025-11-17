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
from webdriver_manager.chrome import ChromeDriverManager

STREAMLIT_URL = os.environ.get("STREAMLIT_APP_URL", "https://derivx.streamlit.app/")
WAKE_BUTTON_XPATH = "//button[contains(text(),'Yes, get this app back up')]"
WAIT_SECONDS = int(os.environ.get("STREAMLIT_WAKE_WAIT", "20"))
CHROMEDRIVER_VERSION = os.environ.get("CHROMEDRIVER_VERSION")
CHROMEDRIVER_MAJOR_VERSION = os.environ.get("CHROMEDRIVER_MAJOR_VERSION")


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
    if CHROMEDRIVER_VERSION:
        driver_manager = ChromeDriverManager(version=CHROMEDRIVER_VERSION)
    elif CHROMEDRIVER_MAJOR_VERSION:
        driver_manager = ChromeDriverManager(version=CHROMEDRIVER_MAJOR_VERSION)
    else:
        driver_manager = ChromeDriverManager()
    service = Service(driver_manager.install())
    return webdriver.Chrome(service=service, options=options)


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
