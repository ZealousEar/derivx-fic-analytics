from __future__ import annotations

import logging

import streamlit as st

from derivx.ui import code_reload

# Streamlit Community Cloud pulls new commits into this running process: drop the package's cached modules when
# their source changed, before importing the pages, so a deploy never mixes old and new code (see code_reload).
if code_reload.drop_changed_modules():
    st.cache_data.clear()
    st.cache_resource.clear()

from derivx.ui import (  # noqa: E402 - after the reload check, so the pages import the current code
    analytics_lab,
    fx_page,
    pricer_page,
    rates_bonds,
    rates_book,
    rates_capfloor,
    rates_curve,
    rates_research,
    rates_scenarios,
    rates_swap,
    rates_swaption,
)
from derivx.ui.plot_limits import mount_zoom_guard  # noqa: E402
from derivx.ui.state import keep_widget_values, restore_widget_values  # noqa: E402
from derivx.ui.theme import apply_app_style  # noqa: E402

st.set_page_config(page_title="DERIVX / FIC ANALYTICS", layout="wide", initial_sidebar_state="auto")
if st.session_state.setdefault("_derivx_code_generation", code_reload.generation()) != code_reload.generation():
    st.session_state.clear()  # this session's state was made by code that has since been replaced
    st.session_state["_derivx_code_generation"] = code_reload.generation()
restore_widget_values()  # before any page draws: inputs of a mode switched back to show their kept values

# Minimalist mode selector - architectural, no emojis
st.sidebar.markdown("### / NAVIGATION")
choice = st.sidebar.radio(
    "MODE",
    ["[ EQUITY ]", "[ FX ]", "[ RATES ]", "[ ANALYTICS ]"],
    format_func=lambda x: x,
    label_visibility="collapsed",
    key="app_mode",
)
mode = (choice or "").replace("[", "").replace("]", "").strip()  # Extract mode name (a radio always has one)

apply_app_style()

try:
    if mode == "EQUITY":
        pricer_page.render(mode)
    elif mode == "FX":
        fx_page.render()
    elif mode == "RATES":
        st.markdown("## / INTEREST RATES")
        tabs = st.tabs(
            ["/ CURVE", "/ SWAP", "/ SWAPTION", "/ CAP-FLOOR", "/ BONDS", "/ BOOK", "/ SCENARIOS", "/ RESEARCH"]
        )
        with tabs[0]:
            market = rates_curve.render()
        if market is None:
            for tab in tabs[1:]:
                with tab:
                    st.info("Fix the quotes on the CURVE tab to price instruments.")
        else:
            tab_trades = []
            with tabs[1]:
                tab_trades += rates_swap.render(market)
            with tabs[2]:
                tab_trades += rates_swaption.render(market)
            with tabs[3]:
                tab_trades += rates_capfloor.render(market)
            with tabs[4]:
                tab_trades += rates_bonds.render(market)
            with tabs[5]:
                book = rates_book.render(market)
            positions, scope = rates_book.risk_scope(book, tab_trades)  # one position set for SCENARIOS and RESEARCH
            with tabs[6]:
                rates_scenarios.render(market, positions, scope)
            with tabs[7]:
                rates_research.render(market, positions, scope)
    elif mode == "ANALYTICS":
        analytics_lab.render()
finally:
    # Even when a page raises (Streamlit still shows its error), the other modes' widget values are kept
    # (UIE-04, B2); a failure in the helper itself must not hide the page's error, so it only logs.
    try:
        keep_widget_values()
    except Exception:  # noqa: BLE001 - defensive: never mask the page's own exception
        logging.getLogger(__name__).exception("keep_widget_values failed")
    mount_zoom_guard()
    code_reload.record_loaded_modules()  # every module this run imported, for the next run's check
