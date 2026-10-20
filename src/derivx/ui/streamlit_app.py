from __future__ import annotations

import streamlit as st

from derivx.ui import (
    analytics_lab,
    fx_page,
    pricer_page,
    rates_bonds,
    rates_capfloor,
    rates_curve,
    rates_research,
    rates_scenarios,
    rates_swap,
    rates_swaption,
)
from derivx.ui.plot_limits import mount_zoom_guard
from derivx.ui.state import keep_widget_values
from derivx.ui.theme import apply_app_style

st.set_page_config(page_title="DERIVX / FIC ANALYTICS", layout="wide", initial_sidebar_state="expanded")

# Minimalist mode selector - architectural, no emojis
st.sidebar.markdown("### / NAVIGATION")
mode = st.sidebar.radio(
    "MODE",
    ["[ EQUITY ]", "[ FX ]", "[ RATES ]", "[ ANALYTICS ]"],
    format_func=lambda x: x,
    label_visibility="collapsed"
)
mode = mode.replace("[", "").replace("]", "").strip()  # Extract mode name

apply_app_style()

if mode == "EQUITY":
    pricer_page.render(mode)
elif mode == "FX":
    fx_page.render()
elif mode == "RATES":
    st.markdown("## / INTEREST RATES")
    tabs = st.tabs(["/ CURVE", "/ SWAP", "/ SWAPTION", "/ CAP-FLOOR", "/ BONDS", "/ SCENARIOS", "/ RESEARCH"])
    with tabs[0]:
        market = rates_curve.render()
    if market is None:
        for tab in tabs[1:]:
            with tab:
                st.info("Fix the quotes on the CURVE tab to price instruments.")
    else:
        positions = []
        with tabs[1]:
            positions += rates_swap.render(market)
        with tabs[2]:
            positions += rates_swaption.render(market)
        with tabs[3]:
            positions += rates_capfloor.render(market)
        with tabs[4]:
            positions += rates_bonds.render(market)
        with tabs[5]:
            rates_scenarios.render(market, positions)
        with tabs[6]:
            rates_research.render(market, positions)
elif mode == "ANALYTICS":
    analytics_lab.render()

keep_widget_values()  # widget values of the modes not drawn this run survive the switch (UIE-04)
mount_zoom_guard()
