"""Shared chart styling and config for the Streamlit pages."""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go

from derivx.ui.plot_limits import GUARD_AVAILABLE

ACCENT = "#FF4444"
FOREGROUND = "#FFFFFF"
GRID = "#333333"
SERIES = ["#FFFFFF", "#FF4444", "#35C1FF", "#FFA500", "#7FFFD4", "#C792EA"]

PLOTLY_2D_CONFIG = {
    "displaylogo": False,
    "scrollZoom": GUARD_AVAILABLE,
    "modeBarButtonsToRemove": [
        "toImage",
        "lasso2d",
        "select2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}


def dark_layout(
    fig: go.Figure,
    height: int = 380,
    xtitle: Optional[str] = None,
    ytitle: Optional[str] = None,
    legend: bool = True,
) -> go.Figure:
    """Apply the app's black, gridded chart style."""

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color=FOREGROUND, family="Helvetica"),
        height=height,
        margin=dict(l=60, r=20, t=30, b=50),
        showlegend=legend,
        legend=dict(bgcolor="#000000", bordercolor=FOREGROUND, borderwidth=1, font=dict(family="IBM Plex Mono")),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, title_text=xtitle)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_text=ytitle)
    return fig


APP_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    
    .stApp {
        background: #000000;
        color: #FFFFFF;
    }
    
    .stSidebar {
        background: #000000 !important;
        border-right: 2px solid #FFFFFF;
    }
    
    .stSidebar [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }
    
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 3px solid #FFFFFF;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        border-bottom: 2px solid #CCCCCC;
        font-size: 1.5rem;
    }
    
    h3 {
        border-bottom: 1px solid #999999;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    .stMetric {
        background: #FFFFFF;
        color: #000000;
        padding: 20px;
        border: 3px solid #000000;
        border-radius: 0;
        box-shadow: 5px 5px 0px #000000;
    }
    
    .stMetric label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    .stButton button {
        background: #000000;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
        border-radius: 0;
        padding: 0.75rem 2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: #FFFFFF;
        color: #000000;
        border: 2px solid #000000;
    }
    
    [data-testid="stHorizontalBlock"] {
        gap: 1.5rem;
    }
    
    /* Industrial brutalist inputs */
    input, select {
        border: 2px solid #FFFFFF !important;
        border-radius: 0 !important;
        background: #000000 !important;
        color: #FFFFFF !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Quote marks / design elements */
    .quote-mark {
        font-size: 3rem;
        font-weight: 700;
        color: #FF4444;
        line-height: 0.8;
    }
    
    </style>
    """

HEADER_HTML = """
    <div style='background: #000000; border-top: 4px solid #FF4444; 
    padding: 1.5rem 0 1rem 0; margin: -1rem 0 3rem 0;'>
        <h1 style='margin: 0; font-size: 3rem; border: none !important; letter-spacing: 0.1em; font-weight: 700;'>
            DERIVX
        </h1>
        <div style='margin-top: 0.5rem; display: flex; gap: 1rem; align-items: center;'>
            <span style='font-size: 0.75rem; letter-spacing: 0.2em; font-weight: 600; opacity: 0.8;'>
                FIC DERIVATIVES ANALYTICS
            </span>
            <span style='background: #FF4444; color: #000000; padding: 0.2rem 0.5rem; 
            font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;'>
                v2.0
            </span>
        </div>
    </div>
"""


def apply_app_style() -> None:
    """Inject the app-wide CSS and draw the page header."""

    import streamlit as st

    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(HEADER_HTML, unsafe_allow_html=True)
