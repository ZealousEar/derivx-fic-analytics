"""The app theme: readable on black (WCAG 2.1 AA), heatmap cells readable,
metric rows that do not wrap, and a committed dark Streamlit theme.

The rendered page was also measured in a browser (contrast of every visible
text node at 390, 1280 and 1920 px); these tests pin the constants and the
CSS rules that measurement depends on."""

import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from derivx import __version__
from derivx.ui import theme

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomli ships with pytest there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "derivx" / "ui" / "streamlit_app.py"
HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


def test_contrast_ratio_matches_the_wcag_definition():
    assert theme.contrast_ratio("#FFFFFF", "#000000") == pytest.approx(21.0)
    assert theme.contrast_ratio("#000000", "#000000") == pytest.approx(1.0)
    # Streamlit's light-theme label colour on the black page, the audit's UIE-08 evidence: 1.68:1
    assert theme.contrast_ratio("#31333F", "#000000") == pytest.approx(1.68, abs=0.01)
    assert theme.contrast_ratio("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)  # just below AA
    with pytest.raises(ValueError, match="#RRGGBB"):
        theme.contrast_ratio("red", "#000000")


def test_theme_text_colours_pass_aa_on_black():
    for colour in (theme.FOREGROUND, theme.MUTED, theme.ACCENT, *theme.SERIES):
        assert theme.contrast_ratio(colour, theme.BACKGROUND) >= 4.5, colour


@pytest.mark.parametrize("name", ["DIVERGING_COLORS", "PNL_COLORS", "SEQUENTIAL_COLORS"])
def test_white_cell_text_is_readable_on_every_heatmap_colour(name):
    scale = getattr(theme, name)
    assert theme.min_cell_contrast(scale, theme.CELL_TEXT["color"]) >= 4.5


def test_every_colour_in_the_app_css_meets_aa_on_its_background():
    """Text colours the CSS sets, each with the background it is drawn on."""

    pairs = {
        "#FFFFFF": "#000000",  # page text, buttons, inputs
        "#000000": "#FFFFFF",  # metric cards
        "#E0E0E0": "#000000",  # uploader instructions
        "#A6A6A6": "#000000",  # disabled labels, inputs and buttons
        "#CCCCCC": "#000000",  # header subtitle
    }
    colours = set(c.upper() for c in HEX.findall(theme.APP_CSS))
    for text, background in pairs.items():
        assert text in colours, f"{text} is no longer used: update this test"
        assert theme.contrast_ratio(text, background) >= 4.5
    # multiselect chips: white on the dark accent
    assert "#B91C1C" in colours and theme.contrast_ratio("#FFFFFF", "#B91C1C") >= 4.5
    # the metric delta chip: black on light grey, and the header tag: black on the accent
    assert theme.contrast_ratio("#000000", "#E8E8E8") >= 4.5
    assert theme.contrast_ratio("#000000", theme.ACCENT) >= 4.5


def test_css_does_not_fight_streamlit_layout_or_fonts():
    css = theme.APP_CSS
    # a column gap larger than Streamlit's makes st.columns(4) wrap 3 + 1 next to the sidebar (UIE-09)
    assert "stHorizontalBlock" not in css
    # a universal font rule beats inherited monospace (the hero price, UIE-09); only a zero-specificity rule
    # that skips code, icons and the monospace classes is allowed
    assert not re.search(r"(^|[},])\s*\*\s*\{", css, flags=re.MULTILINE)  # a bare `* {` selector
    assert ":where(.stApp)" in css and "dx-hero-value" in css
    # sidebar widget labels keep their case (uppercase CSS turned 'VOL.σ' into 'VOL.Σ')
    assert re.search(r'stSidebar"\] \[data-testid="stWidgetLabel"\] p \{ text-transform: none; \}', css)


def test_committed_streamlit_theme_is_dark_with_the_app_colours():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    t = config["theme"]
    assert t["base"] == "dark"
    assert t["backgroundColor"].upper() == theme.BACKGROUND
    assert t["textColor"].upper() == theme.FOREGROUND
    assert t["primaryColor"].upper() == theme.ACCENT


def test_header_shows_the_package_version_not_a_hard_coded_one():
    assert f"v{__version__}" in theme.HEADER_HTML
    assert "v2.0" not in theme.HEADER_HTML


def test_app_injects_the_theme_once_and_opens_with_an_automatic_sidebar():
    src = APP.read_text()
    assert 'initial_sidebar_state="auto"' in src  # collapsed on a phone, open on a laptop (UIE-09)
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    styles = [m.value for m in at.markdown if "<style>" in m.value and "stMetricValue" in m.value]
    assert len(styles) == 1


def _heatmaps(at: AppTest) -> list:
    out = []
    for chart in at.get("plotly_chart"):
        spec = json.loads(chart.proto.spec)
        out += [t for t in spec["data"] if t.get("type") == "heatmap" and t.get("texttemplate")]
    return out


@pytest.mark.parametrize("mode", ["[ EQUITY ]", "[ FX ]", "[ RATES ]"])
def test_heatmaps_that_print_values_use_readable_cell_colours(mode):
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(mode).run()
    assert not at.exception
    maps = _heatmaps(at)
    assert maps, "expected at least one heatmap with values in the cells"
    for trace in maps:
        text = trace.get("textfont", {}).get("color")
        assert text == theme.FOREGROUND, f"cell text colour {text!r}: set textfont=CELL_TEXT"
        scale = trace["colorscale"]
        assert theme.min_cell_contrast(scale, text) >= 4.5, scale
