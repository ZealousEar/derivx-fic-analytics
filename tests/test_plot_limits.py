import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from derivx.ui.plot_limits import GUARD_AVAILABLE, axis_limits, bound_axes, bound_scene

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"


def test_axis_limits_pads_floors_and_sets_min_span():
    lim = axis_limits([60.0, 140.0], pad=0.05, floor=0.0)
    assert lim["min"] == pytest.approx(56.0)
    assert lim["max"] == pytest.approx(144.0)
    assert lim["min_span"] == pytest.approx(0.02 * 88.0)

    # A floor never lets a strike/vol axis go negative, but keeps the width.
    lim = axis_limits([1.0, 41.0], pad=0.5, floor=0.0)
    assert lim["min"] == 0.0
    assert lim["max"] - lim["min"] >= 40.0


def test_axis_limits_handles_flat_and_non_finite_data():
    flat = axis_limits([0.2, 0.2, np.nan, np.inf])
    assert flat["min"] < 0.2 < flat["max"]
    assert flat["min_span"] > 0
    empty = axis_limits([np.nan])
    assert empty["min"] < empty["max"]


def test_bound_axes_sets_declarative_limits_and_meta():
    fig = go.Figure(go.Scatter(x=[80, 120], y=[0.2, 0.3]))
    x_lim = axis_limits([80, 120], floor=0.0)
    bound_axes(fig, xaxis=x_lim)
    assert fig.layout.xaxis.minallowed == x_lim["min"]
    assert fig.layout.xaxis.maxallowed == x_lim["max"]
    assert list(fig.layout.xaxis.range) == [x_lim["min"], x_lim["max"]]
    assert fig.layout.meta["zoom"]["axes"]["xaxis"] == x_lim


def test_bound_scene_records_zoom_distance_and_pan_limits():
    fig = go.Figure(go.Surface(z=[[1, 2], [3, 4]]))
    bound_scene(fig, eye={"x": 1.6, "y": 1.6, "z": 1.2}, zoom=(0.5, 2.0), pan=0.3)
    scene = fig.layout.meta["zoom"]["scene"]
    r0 = np.sqrt(1.6**2 + 1.6**2 + 1.2**2)
    assert fig.layout.scene.aspectmode == "manual"
    assert scene["distance"] == pytest.approx([r0 / 2.0, r0 / 0.5])
    assert scene["zoom"] == [0.5, 2.0] and scene["pan"] == 0.3


def _charts(at: AppTest) -> list:
    return [(json.loads(c.proto.spec), json.loads(c.proto.config)) for c in at.get("plotly_chart")]


def _assert_bounded(spec: dict, config: dict) -> None:
    layout = spec["layout"]
    zoom = layout["meta"]["zoom"]
    assert config["scrollZoom"] == GUARD_AVAILABLE
    if "scene" in zoom:
        assert layout["scene"]["aspectmode"] == "manual"
        return
    for name, lim in zoom["axes"].items():
        assert layout[name]["minallowed"] == lim["min"]
        assert layout[name]["maxallowed"] == lim["max"]


@pytest.mark.parametrize("mode", ["[ EQUITY ]", "[ FX ]", "[ RATES ]"])
def test_every_pricer_chart_is_zoom_bounded(mode):
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(mode).run()
    assert not at.exception
    charts = _charts(at)
    # Equity: scenario grid + risk surface + smile; FX: smile + tenor x delta surface + scenario grid;
    # Rates: every chart across its five tabs.
    assert len(charts) == 3 if mode != "[ RATES ]" else len(charts) >= 5
    for spec, config in charts:
        _assert_bounded(spec, config)
    if mode != "[ RATES ]":
        named = {spec["layout"]["meta"]["chart"]: spec["layout"] for spec, _ in charts}
        expected = {"[ EQUITY ]": {"scenario", "surface", "smile"}, "[ FX ]": {"fx_smile", "fx_surface", "fx_scenario"}}
        assert set(named) == expected[mode]
        smile = named["smile" if mode == "[ EQUITY ]" else "fx_smile"]
        assert smile["xaxis"]["minallowed"] >= 0.0  # no negative strikes
        assert smile["yaxis"]["minallowed"] >= 0.0


def test_fx_smile_in_delta_space_is_zoom_bounded():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("[ FX ]").run()
    at.radio(key="fx_space").set_value("DELTA").run()
    assert not at.exception
    charts = _charts(at)
    assert len(charts) == 3
    for spec, config in charts:
        _assert_bounded(spec, config)
    smile = next(spec for spec, _ in charts if spec["layout"]["meta"]["chart"] == "fx_smile")
    n = len(smile["layout"]["xaxis"]["ticktext"])
    assert smile["layout"]["xaxis"]["minallowed"] == -0.5 and smile["layout"]["xaxis"]["maxallowed"] == n - 0.5


def test_every_analytics_chart_is_zoom_bounded():
    """Calibration smile, walk-forward and model comparison: drawn from the
    bundled sample and a short SYNTHETIC history (no network)."""

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("[ ANALYTICS ]").run()
    at.button(key="lab_cal_run").click().run()
    at.button(key="lab_mc_run").click().run()
    at.radio(key="lab_wf_source").set_value("SYNTHETIC demo history").run()
    at.slider(key="lab_wf_synth_days").set_value(4).run()
    at.button(key="lab_wf_synth_go").click().run()
    at.number_input(key="lab_wf_maxdays").set_value(30).run()
    at.button(key="lab_wf_run").click().run()
    assert not at.exception
    charts = _charts(at)
    assert len(charts) == 4  # calibration smile, its bucket vegas, walk-forward, model comparison
    for spec, config in charts:
        _assert_bounded(spec, config)
    named = {spec["layout"]["meta"]["chart"]: spec["layout"] for spec, _ in charts}
    assert set(named) == {"lab_calibration", "lab_bucket_vega", "lab_walkforward", "lab_model_comparison"}
    # the research tabs' charts (surface, local vol, parameter history, rates) are checked in test_lab_research_ui
    for name in ("lab_calibration", "lab_model_comparison"):
        assert named[name]["xaxis"]["minallowed"] >= 0.0 and named[name]["yaxis"]["minallowed"] >= 0.0
    assert named["lab_walkforward"]["yaxis"]["minallowed"] >= 0.0


def test_failed_guard_mount_falls_back_instead_of_raising(monkeypatch):
    from derivx.ui import plot_limits

    def broken(**_):
        raise TypeError("components v2 regression")

    monkeypatch.setattr(plot_limits, "_GUARD", broken)
    monkeypatch.setattr(plot_limits, "GUARD_AVAILABLE", True)
    assert plot_limits.mount_zoom_guard() is False
    assert plot_limits.GUARD_AVAILABLE is False  # next run uses safe configs
