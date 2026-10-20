"""The Walk-Forward tab runs for any history of two or more snapshots, its
calibration window is bounded by the history, and a short history never
crashes the widgets (AL-11, ui-engineering-V1, AL-06)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from derivx.data.chain_schema import ChainDataset
from derivx.data.synthetic import synthetic_history
from derivx.ui import analytics_lab as lab

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"


def _with_history(n_days: int) -> AppTest:
    ds = ChainDataset(chains=synthetic_history(n_days), source="SYNTHETIC demo history (test)", ticker="SYNTH")
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.session_state[lab.K_WF_DATA] = {"dataset": ds, "origin": "synthetic", "seed": 20261020}
    at.run()
    at.sidebar.radio[0].set_value("[ ANALYTICS ]").run()
    assert not at.exception
    return at


@pytest.mark.parametrize("n_days", [2, 3, 5])
def test_walkforward_tab_runs_for_any_supported_day_count(n_days):
    at = _with_history(n_days)
    at.number_input(key="lab_wf_maxdays").set_value(20).run()
    sliders = [s for s in at.slider if s.label == "Calibration window (snapshots pooled)"]
    if n_days == 2:  # one origin: a fixed one-snapshot window, shown as a caption
        assert not sliders
    else:
        assert sliders[0].max == n_days - 1  # horizon 1 leaves n - 1 origins
        sliders[0].set_value(sliders[0].max).run()
    at.button(key="lab_wf_run").click().run()
    assert not at.exception
    result = at.session_state[lab.K_WF]["result"]
    assert not result.empty and result.settings.window == (1 if n_days == 2 else n_days - 1)
    assert result.slices["origin"].nunique() == 1  # the largest window leaves exactly one scored origin


def test_a_horizon_beyond_the_history_explains_why_nothing_is_scored():
    at = _with_history(3)
    at.number_input(key="lab_wf_horizon").set_value(3).run()
    at.button(key="lab_wf_run").click().run()
    assert not at.exception
    assert any("No forecast could be scored" in w.value for w in at.warning)
    reasons = " ".join(at.dataframe[-1].value["reason"].astype(str))
    assert "no snapshot 3 trading day(s) later" in reasons
