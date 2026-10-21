"""Shared chart styling, colour scales and the app-wide CSS for the Streamlit pages.

The app is drawn white on black. The committed ``.streamlit/config.toml``
sets Streamlit's dark base theme to the same colours, so Streamlit draws its
own widgets (labels, tabs, alerts, expanders, data editors, uploaders) for a
dark background; the CSS here only adds the app's square, bordered look and
fixes what the theme does not cover (metric cards, buttons, the header).

Contrast: every text colour in this module meets WCAG 2.1 AA on the
background it is drawn on (4.5:1 for body text, 3:1 for large text), and the
heatmap scales keep white in-cell text at 4.5:1 or more on every cell colour.
``tests/test_theme.py`` checks the constants; the rendered page was measured
with a browser script (contrast of every visible text node).
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

import plotly.graph_objects as go

from derivx import __version__
from derivx.ui.plot_limits import GUARD_AVAILABLE

ACCENT = "#FF4444"
FOREGROUND = "#FFFFFF"
BACKGROUND = "#000000"
MUTED = "#B3B3B3"  # secondary text on black: 10.0:1
GRID = "#333333"
SERIES = ["#FFFFFF", "#FF4444", "#35C1FF", "#FFA500", "#7FFFD4", "#C792EA"]

# Heatmap colour scales for charts that print values in the cells: white text
# (CELL_TEXT) stays at 4.5:1 or more on every colour of these scales.
DIVERGING_COLORS = [[0.0, "#C62828"], [0.5, "#111111"], [1.0, "#1565C0"]]  # negative red, zero black, positive blue
PNL_COLORS = [[0.0, "#DC2626"], [0.5, "#111111"], [1.0, "#15803D"]]  # loss red, zero black, gain green
SEQUENTIAL_COLORS = [[0.0, "#140B34"], [0.33, "#3B1C8C"], [0.66, "#1F5AA6"], [1.0, "#0E7C6B"]]  # low dark, high teal
CELL_TEXT = dict(color=FOREGROUND)

# Legends sit in one horizontal row (wrapping as needed) at the bottom of the figure, below the x-axis title:
# anchored to the container, so Plotly grows the bottom margin to fit tick labels, title and legend without
# overlap, and the plot keeps its full width on a phone (a side legend took a third of a 390px chart).
LEGEND_BELOW = dict(orientation="h", yref="container", y=0.0, yanchor="bottom", x=0.0, xanchor="left")

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
    """Apply the app's black, gridded chart style, with the legend below the
    plot (:data:`LEGEND_BELOW`)."""

    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color=FOREGROUND, family="Helvetica"),
        height=height,
        margin=dict(l=60, r=20, t=30, b=50),
        showlegend=legend,
        legend=dict(
            **LEGEND_BELOW,
            bgcolor=BACKGROUND,
            bordercolor=FOREGROUND,
            borderwidth=1,
            font=dict(family="IBM Plex Mono", color=FOREGROUND),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, title_text=xtitle)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_text=ytitle)
    return fig


# -- contrast (WCAG 2.1) ---------------------------------------------------------


def _rgb(color: str) -> Tuple[float, float, float]:
    """``#RRGGBB`` (or ``#RGB``) as 0..255 floats."""

    h = color.strip()
    if not re.fullmatch(r"#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})", h):
        raise ValueError(f"expected a #RRGGBB colour, got {color!r}")
    h = h[1:]
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(float(int(h[i : i + 2], 16)) for i in (0, 2, 4))  # type: ignore[return-value]


def relative_luminance(color: str) -> float:
    """WCAG 2.1 relative luminance of an sRGB colour, 0 (black) .. 1 (white)."""

    def channel(v: float) -> float:
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in _rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio of two colours, 1 .. 21 (AA: 4.5 body text, 3 large text)."""

    la, lb = relative_luminance(a), relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def scale_colors(scale: Sequence[Sequence[object]], n: int = 101) -> list:
    """``n`` colours sampled along a Plotly colour scale (linear in RGB between
    stops, as Plotly draws it), as ``#rrggbb``."""

    stops = [(float(t), _rgb(str(c))) for t, c in scale]
    out = []
    for i in range(n):
        t = i / (n - 1)
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                out.append("#%02x%02x%02x" % tuple(round(a + (b - a) * w) for a, b in zip(c0, c1)))
                break
    return out


def min_cell_contrast(scale: Sequence[Sequence[object]], text: str = FOREGROUND) -> float:
    """Lowest contrast of ``text`` over every colour of a heatmap scale."""

    return min(contrast_ratio(text, c) for c in scale_colors(scale))


# -- app CSS ---------------------------------------------------------------------

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

/* One sans-serif face for text. Zero specificity (:where), so Streamlit's own widget fonts, code blocks,
   Material icons and the monospace number classes (dx-*) keep theirs, and children of a monospace number
   inherit it. */
:where(.stApp) :where(:not(code, pre, code *, pre *, [data-testid="stIconMaterial"], .dx-mono, .dx-mono *,
  .dx-hero-value, .dx-greek-value, .dx-cell b, .dx-lab-stat b, .dx-lab-banner, .dx-lab-banner *)) {
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.stApp { background: #000000; color: #FFFFFF; }

/* No light band over the page: Streamlit's top bar takes the page colour. Content starts right below it. */
[data-testid="stHeader"] { background: #000000; }
[data-testid="stMainBlockContainer"] { padding-top: 3.75rem; padding-bottom: 4rem; }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
}

/* Sidebar: black with a white rule. Section text is uppercase, but widget labels keep the case they are
   written in (CSS uppercase would turn the label 'VOL.σ' into 'VOL.Σ' and 'USD per 1 EUR' into 'USD PER 1 EUR'). */
[data-testid="stSidebar"] { background: #000000 !important; border-right: 2px solid #FFFFFF; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { text-transform: none; }

h1, h2, h3 {
  color: #FFFFFF !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.05em;
  border-bottom: 3px solid #FFFFFF; padding-bottom: 0.5rem; margin-bottom: 1.25rem;
}
h2 { border-bottom: 2px solid #CCCCCC; font-size: 1.5rem; }
h3 { border-bottom: 1px solid #999999; font-size: 1.2rem; font-weight: 500; }

/* Metric cards: white, square, the value in monospace sized to the card (container query units), so four
   cards sit in one row next to the sidebar at 1280px and long values shrink instead of being cut off. Labels
   wrap onto a second line rather than ending in an ellipsis. */
[data-testid="stMetric"] {
  background: #FFFFFF; color: #000000; padding: 0.85rem 0.9rem; border: 3px solid #000000; border-radius: 0;
  box-shadow: 5px 5px 0 #000000; container-type: inline-size; height: 100%;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
  color: #000000 !important; white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important;
}
[data-testid="stMetricLabel"] p {
  font-weight: 700 !important; font-size: 0.75rem !important; line-height: 1.3 !important;
  text-transform: uppercase; letter-spacing: 0.12em;
}
[data-testid="stMetricValue"] {
  color: #000000 !important; font-family: 'IBM Plex Mono', monospace !important; font-weight: 700 !important;
  font-size: clamp(1.05rem, 12cqi, 1.9rem) !important; line-height: 1.25 !important;
}
[data-testid="stMetricValue"] > div {
  font-family: inherit !important; white-space: nowrap !important; text-overflow: clip !important;
  overflow-x: auto !important; overflow-y: hidden !important;
}
[data-testid="stMetricDelta"] { background: #E8E8E8 !important; color: #000000 !important; }
[data-testid="stMetricDelta"] * { color: #000000 !important; white-space: normal !important;
  overflow: visible !important; text-overflow: clip !important; }

/* Buttons (actions, downloads, the file uploader's browse button): black, square, white border; inverted
   on hover. Disabled buttons stay readable (#A6A6A6 on black, 8.5:1). */
.stButton button, .stDownloadButton button, .stFormSubmitButton button, [data-testid="stFileUploader"] button {
  background: #000000; color: #FFFFFF; border: 2px solid #FFFFFF; border-radius: 0; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.stButton button { padding: 0.6rem 1.25rem; }
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover,
[data-testid="stFileUploader"] button:hover { background: #FFFFFF; color: #000000; border-color: #FFFFFF; }
.stButton button:disabled, .stDownloadButton button:disabled { color: #A6A6A6; border-color: #666666; }

/* File uploader: a dashed drop zone on black. */
[data-testid="stFileUploaderDropzone"] { background: #000000 !important; border: 2px dashed #FFFFFF !important;
  border-radius: 0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] * { color: #E0E0E0 !important; }
[data-testid="stFileUploaderFile"] { border: 1px solid #FFFFFF; }

/* Multiselect chips: white on the dark accent (#B91C1C, 6.5:1); white on #FF4444 is only 3.4:1. */
[data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] [data-tag] { background-color: #B91C1C !important; }
[data-baseweb="tag"] *, [data-testid="stMultiSelectTagsContainer"] [data-tag] * { color: #FFFFFF !important; }

/* Text and number inputs (not the search box inside multiselects): square white frame, monospace. */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input,
[data-testid="stSelectbox"] input, select {
  border: 2px solid #FFFFFF !important; border-radius: 0 !important; background: #000000 !important;
  color: #FFFFFF !important; font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stNumberInput"] input:disabled, [data-testid="stTextInput"] input:disabled {
  color: #A6A6A6 !important; -webkit-text-fill-color: #A6A6A6 !important; border-color: #666666 !important;
}
/* Disabled widgets' labels: Streamlit fades them to 40%; keep them legible (#A6A6A6). */
[data-testid="stWidgetLabel"][aria-disabled="true"] p, label[aria-disabled="true"] p,
[data-disabled="true"] [data-testid="stWidgetLabel"] p { color: #A6A6A6 !important; }

/* Page header */
.dx-header { border-top: 4px solid #FF4444; padding: 1rem 0 0.25rem; margin: 0 0 1.75rem; }
.dx-header-title { font-size: clamp(2rem, 7vw, 3rem); font-weight: 700; letter-spacing: 0.1em; line-height: 1.1;
  color: #FFFFFF; }
.dx-header-sub { margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center; }
.dx-header-sub span:first-child { font-size: 0.75rem; letter-spacing: 0.2em; font-weight: 600; color: #CCCCCC; }
.dx-header-tag { background: #FF4444; color: #000000; padding: 0.15rem 0.5rem; font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.1em; }
</style>
"""

HEADER_HTML = f"""
<div class='dx-header'>
  <div class='dx-header-title'>DERIVX</div>
  <div class='dx-header-sub'>
    <span>FIC DERIVATIVES ANALYTICS</span><span class='dx-header-tag'>v{__version__}</span>
  </div>
</div>
"""


def apply_app_style() -> None:
    """Inject the app-wide CSS and draw the page header."""

    import streamlit as st

    st.markdown(APP_CSS + HEADER_HTML, unsafe_allow_html=True)  # one element: no empty gap above the header
