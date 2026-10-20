import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from derivx.calibration.production import calibrate_with_constraints
from derivx.models.comparison import compare_models
from derivx.ui.plot_limits import GUARD_AVAILABLE, axis_limits, bound_axes, bound_scene
from derivx.vol.svi import svi_iv

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


def _chain(trade_date: dt.date) -> pd.DataFrame:
    strikes = np.linspace(80, 120, 9)
    params = {"a": 0.02, "b": 0.3, "rho": -0.2, "m": 0.0, "s": 0.4}
    df = pd.DataFrame(
        {
            "trade_date": pd.Timestamp(trade_date),
            "expiry": pd.Timestamp(trade_date + dt.timedelta(days=30)),
            "strike": strikes,
            "implied_vol": svi_iv(strikes, params, F=100.0, T=1.0),  # params: total variance at T = 1
            "underlying_price": 100.0,
            "bid": 1.0,
            "ask": 1.2,
            "mid": 1.1,
            "volume": 50,
            "open_interest": 100,
            "option_type": np.where(strikes < 100.0, "put", "call"),
        }
    )
    df["moneyness"] = df["strike"] / 100.0
    return df


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
    # Equity/FX: surface + smile; Rates: every chart across its five tabs.
    assert len(charts) == 2 if mode != "[ RATES ]" else len(charts) >= 5
    for spec, config in charts:
        _assert_bounded(spec, config)
    if mode != "[ RATES ]":
        smile = charts[1][0]["layout"]
        assert smile["xaxis"]["minallowed"] >= 0.0  # no negative strikes
        assert smile["yaxis"]["minallowed"] >= 0.0


def test_every_analytics_chart_is_zoom_bounded():
    start = dt.date(2024, 1, 2)
    chain = _chain(start)
    # Five days: with exactly three the tab's window slider has min == max and
    # Streamlit raises (a separate, pre-existing bug).
    dates = [start + dt.timedelta(days=i) for i in range(5)]
    days = {pd.Timestamp(d): _chain(d) for d in dates}
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["market_chain"] = chain
    at.session_state["calibration_results"] = calibrate_with_constraints(chain)
    at.session_state["walkforward_chains"] = days
    at.session_state["walkforward_metrics"] = pd.DataFrame(
        {
            "trade_date": list(days)[:2],
            "next_date": list(days)[1:3],
            "rmse": [0.01, 0.02],
            "hit_rate": [0.5, 0.6],
            "regime": "mid",
        }
    )
    at.session_state["model_comp_metrics"] = compare_models(chain)
    at.session_state["model_comp_slice"] = chain
    at.run()
    at.sidebar.radio[0].set_value("[ ANALYTICS ]").run()
    assert not at.exception
    charts = _charts(at)
    assert len(charts) == 3  # calibration smile, walk-forward, model comparison
    for spec, config in charts:
        _assert_bounded(spec, config)


def test_failed_guard_mount_falls_back_instead_of_raising(monkeypatch):
    from derivx.ui import plot_limits

    def broken(**_):
        raise TypeError("components v2 regression")

    monkeypatch.setattr(plot_limits, "_GUARD", broken)
    monkeypatch.setattr(plot_limits, "GUARD_AVAILABLE", True)
    assert plot_limits.mount_zoom_guard() is False
    assert plot_limits.GUARD_AVAILABLE is False  # next run uses safe configs
