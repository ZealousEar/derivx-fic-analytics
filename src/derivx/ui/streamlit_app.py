from __future__ import annotations

import streamlit as st

from derivx.ui import (
    analytics_lab,
    fx_page,
    pricer_page,
    rates_bonds,
    rates_book,
    rates_capfloor,
    rates_curve,
    rates_fx,
    rates_research,
    rates_scenarios,
    rates_swap,
    rates_swaption,
)
from derivx.ui.plot_limits import mount_zoom_guard
from derivx.ui.state import keep_widget_values, restore_widget_values
from derivx.ui.theme import apply_app_style

st.set_page_config(page_title="DERIVX / FIC ANALYTICS", layout="wide", initial_sidebar_state="auto")
restore_widget_values()  # before any page draws: inputs of a mode switched back to show their kept values

# Minimalist mode selector - architectural, no emojis
st.sidebar.markdown("### / NAVIGATION")
choice = st.sidebar.radio(
    "MODE",
    ["[ EQUITY ]", "[ FX ]", "[ RATES ]", "[ ANALYTICS ]"],
    format_func=lambda x: x,
    label_visibility="collapsed",
)
mode = (choice or "").replace("[", "").replace("]", "").strip()  # Extract mode name (a radio always has one)

apply_app_style()

if mode == "EQUITY":
    pricer_page.render(mode)
elif mode == "FX":
    fx_page.render()
elif mode == "RATES":
    st.markdown("## / INTEREST RATES")
    tabs = st.tabs(["/ CURVE", "/ SWAP", "/ SWAPTION", "/ CAP-FLOOR", "/ BONDS", "/ BOOK", "/ SCENARIOS", "/ RESEARCH"])
    with tabs[0]:
        market = rates_curve.render()
    if market is None:
        for tab in tabs[1:]:
            with tab:
                st.info("Fix the quotes on the CURVE tab to price instruments.")
    else:
        reporting = rates_fx.sidebar(market)  # one reporting currency for the BOOK, SCENARIOS and RESEARCH totals
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
            book = rates_book.render(market, reporting)
        positions, scope = rates_book.risk_scope(book, tab_trades)  # one position set for SCENARIOS and RESEARCH
        with tabs[6]:
            rates_scenarios.render(market, positions, scope, reporting)
        with tabs[7]:
            rates_research.render(market, positions, scope, reporting)
elif mode == "ANALYTICS":
    analytics_lab.render()

keep_widget_values()  # widget values of the modes not drawn this run survive the switch (UIE-04)
mount_zoom_guard()
