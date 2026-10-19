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
