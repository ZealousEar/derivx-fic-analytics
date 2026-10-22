"""ANALYTICS lab research tabs in AppTest, offline (VM-18, AL-20): smile
risk on a calibrated expiry, the SSVI surface with Dupire local vol, the
daily parameter history, and the lab's tools on rates curves and rates-vol
smiles, including implied against realised vol. Every new chart is
zoom-bounded and every result has a manifest."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from derivx.ui import lab_rates, lab_vol
from derivx.validation import realised_implied as ri
from derivx.validation.vol_history import synthetic_vol_history

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"


@pytest.fixture(autouse=True)
def no_yahoo(monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))


def _lab() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value("[ ANALYTICS ]").run()
    assert not at.exception
    return at


def _charts(at: AppTest) -> dict:
    out = {}
    for c in at.get("plotly_chart"):
        spec = json.loads(c.proto.spec)
        out[spec["layout"]["meta"]["chart"]] = (spec["layout"], json.loads(c.proto.config))
    return out


def _bounded(layout: dict) -> None:
    zoom = layout["meta"]["zoom"]
    for name, lim in zoom["axes"].items():
        assert layout[name]["minallowed"] == lim["min"] and layout[name]["maxallowed"] == lim["max"]


def _values(v) -> np.ndarray:
    """A Plotly array from a chart spec: a list, or Plotly's base64 'bdata' typed-array encoding."""

    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
    return np.asarray(v, dtype=float)


def _texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning]
    return "\n".join(str(p) for p in parts + [e.value for e in at.error])


def test_smile_risk_and_ssvi_surface_on_the_bundled_chain():
    at = _lab()
    at.button(key="lab_cal_run").click().run()
    assert not at.exception
    text = _texts(at)
    assert "SMILE RISK" in text and "Vega by quote bucket" in text
    charts = _charts(at)
    layout, _ = charts["lab_bucket_vega"]
    _bounded(layout)
    spec = next(
        json.loads(c.proto.spec)
        for c in at.get("plotly_chart")
        if json.loads(c.proto.spec)["layout"]["meta"]["chart"] == "lab_bucket_vega"
    )
    bars = _values(spec["data"][0]["y"])
    assert bars.size == 7 and np.isfinite(bars).all()
    vegas = [d.value for d in at.dataframe if "dV_dp" in d.value.columns]
    assert {tuple(v["parameter"]) for v in vegas} == {("a", "b", "rho", "m", "sigma"), ("alpha", "rho", "nu")}
    at.button(key="lab_surf_run").click().run()
    assert not at.exception and not at.error
    charts = _charts(at)
    for name in ("lab_ssvi_smile", "lab_ssvi_surface", "lab_local_vol"):
        _bounded(charts[name][0])
    stored = at.session_state[lab_vol.K_SURF]
    report = stored["result"]["fit"].surface.arbitrage_report()
    assert report["butterfly_ok"] and report["calendar_ok"] and report["eta_condition_ok"]
    assert (
        stored["manifest"]["kind"] == "ssvi_surface" and stored["manifest"]["inputs"]["discounting"]["mode"] == "curve"
    )
    assert "ARBITRAGE-FREE" in "\n".join(m.value for m in at.markdown)
    at.slider(key="lab_surf_days").set_value(100).run()  # an expiry between the listed ones
    assert not at.exception and "SSVI smile at 100 days" in _texts(at)


def test_parameter_history_on_the_synthetic_walk_forward_history():
    at = _lab()
    at.radio(key="lab_wf_source").set_value("SYNTHETIC demo history").run()
    at.slider(key="lab_wf_synth_days").set_value(4).run()
    at.button(key="lab_wf_synth_go").click().run()
    at.number_input(key="lab_wf_maxdays").set_value(25).run()
    at.button(key="lab_stab_run").click().run()
    assert not at.exception
    stored = at.session_state[lab_vol.K_STAB]
    assert set(stored["history"]["model"]) == {"svi", "sabr"}
    assert set(stored["summary"]["parameter"]) >= {"atm_vol", "skew", "alpha", "rho", "nu"}
    assert stored["manifest"]["inputs"]["discounting"]["mode"] == "parity"  # SYNTH chains are European Black-76
    _bounded(_charts(at)["lab_param_history"][0])


def test_rates_curves_tab_runs_the_lab_tools_on_the_treasury_history():
    at = _lab()
    assert at.selectbox(key="lab_rates_ds").value == "UST_PAR"
    at.number_input(key="lab_rates_w").set_value(120).run()
    at.button(key="lab_rates_wf_run").click().run()
    assert not at.exception
    stored = at.session_state[lab_rates.K_CURVE_WF]
    summary = stored["result"].summary.set_index("predictor")
    assert set(summary.index) == {"random_walk", "ns_ar1", "ns_curve", "pca_ar1"}
    assert 0.5 < summary.loc["random_walk", "rmse_bp"] < 15
    assert stored["manifest"]["data"]["dataset"] == "UST_PAR" and "bundled snapshot" in "; ".join(
        stored["manifest"]["data"]["origins"]
    )
    _bounded(_charts(at)["lab_curve_wf"][0])
    at.number_input(key="lab_rates_mc_n").set_value(3).run()
    at.button(key="lab_rates_mc_run").click().run()
    assert not at.exception
    scores = at.session_state[lab_rates.K_CURVE_MC]["scores"]
    assert scores["trade_date"].nunique() == 3 and set(scores["model"]) == {"ns", "ns_dl", "svensson", "spline"}
    assert any("lower out-of-sample RMSE" in m.value and "dates" in m.value for m in at.markdown)
    _bounded(_charts(at)["lab_curve_models"][0])


def test_an_uploaded_swap_curve_history_joins_the_rates_store():
    at = _lab()
    days = pd.bdate_range("2026-01-02", periods=80)
    rng = np.random.default_rng(0)
    level = 3.5 + np.cumsum(rng.normal(0, 0.03, days.size))
    frame = pd.DataFrame({"date": days.strftime("%Y-%m-%d")})
    for tenor, add in (("1Y", -0.2), ("2Y", -0.1), ("5Y", 0.0), ("10Y", 0.2), ("30Y", 0.4)):
        frame[tenor] = np.round(level + add, 4)
    at.selectbox(key="lab_rates_up_ccy").set_value("USD").run()
    at.file_uploader(key="lab_rates_up_file").set_value(
        ("my_swaps.csv", frame.to_csv(index=False).encode(), "text/csv")
    ).run()
    assert not at.exception
    ds = at.selectbox(key="lab_rates_ds")
    assert "Your uploaded USD swap-curve history" in ds.options  # AppTest lists the formatted labels
    ds.set_value("USER_SWAP:USD").run()
    assert any("<b>RATES HISTORY</b>" in m.value and "upload: my_swaps.csv" in m.value for m in at.markdown)


def test_rates_vol_history_tab_on_synthetic_and_uploaded_grids():
    at = _lab()
    assert "No free source publishes dated swaption or cap vol quotes" in "\n".join(i.value for i in at.info)
    at.radio(key="lab_vh_source").set_value("SYNTHETIC demo history").run()
    at.slider(key="lab_vh_days").set_value(8).run()
    at.button(key="lab_vh_gen").click().run()
    assert any("SYNTHETIC: not market data" in m.value for m in at.markdown if "dx-lab-banner" in m.value)
    for button in ("lab_vh_wf_run", "lab_vh_mc_run", "lab_vh_stab_run"):
        at.button(key=button).click().run()
        assert not at.exception
    charts = _charts(at)
    for name in ("lab_volhist_wf", "lab_volhist_params"):
        _bounded(charts[name][0])
    wf = at.session_state[lab_rates.K_VH_WF]["result"].summary.set_index("predictor")
    assert np.isfinite(wf.loc["sabr_sticky_moneyness", "skill"])
    # an upload: the same grids as a CSV (vols in bp), without forwards
    hist = synthetic_vol_history(5, expiries=("1Y",), tenors=("10Y",))
    csv = hist.frame().drop(columns="forward").rename(columns={"vol_bp": "vol"}).to_csv(index=False).encode()
    at.radio(key="lab_vh_source").set_value("Upload a dated vol history").run()
    at.file_uploader(key="lab_vh_file").set_value(("grids.csv", csv, "text/csv")).run()
    assert not at.exception
    banner = next(m.value for m in at.markdown if "<b>VOL HISTORY</b>" in m.value)
    assert "source: upload: grids.csv" in banner and "no forwards" in banner and "SYNTHETIC" not in banner
    bad = pd.DataFrame({"expiry": ["1Y"], "tenor": ["10Y"], "offset_bp": [0], "vol": [80]}).to_csv(index=False)
    at.file_uploader(key="lab_vh_file").set_value(("bad.csv", bad.encode(), "text/csv")).run()
    assert any("needs a 'date' column" in e.value for e in at.error)


def test_rates_vol_history_compares_implied_with_realised_vol():
    """AL-20: implied ATM normal vols against the realised normal vol of the
    underlying, from the vol history's own forwards and from the Treasury
    curve history (flagged as a PROXY for swap rates)."""

    at = _lab()
    at.radio(key="lab_vh_source").set_value("SYNTHETIC demo history").run()
    at.slider(key="lab_vh_days").set_value(40).run()
    at.button(key="lab_vh_gen").click().run()
    assert at.selectbox(key="lab_vh_rvi_src").value == ri.HISTORY_FORWARDS
    at.button(key="lab_vh_rvi_run").click().run()
    assert not at.exception and not at.error
    stored = at.session_state[lab_rates.K_VH_RVI]
    summary = stored["result"].summary.set_index("point")
    assert len(summary) == 6 and (summary["n_dates"] == 40 - 17).all()  # 21-day windows, >= 80% covered
    assert np.isfinite(summary["trailing_mean_bp"]).all() and (summary["trailing_mean_bp"] > 0).all()
    banner = next(m.value for m in at.markdown if "<b>REALISED VS IMPLIED</b>" in m.value)
    assert "SYNTHETIC implied vols: not market data" in banner and "WEEKENDS calendar" in banner
    assert stored["manifest"]["kind"] == "realised_vs_implied" and stored["manifest"]["inputs"]["window"] == 21
    layout = _charts(at)["lab_volhist_rvi"][0]
    _bounded(layout)
    assert layout["yaxis"]["minallowed"] >= 0.0
    assert any("BREAKEVEN, BP/DAY" in m.value and "SPREAD, BP/YR" in m.value for m in at.markdown)

    at.selectbox(key="lab_vh_rvi_w").set_value(10).run()
    assert any("Settings changed" in i.value for i in at.info)
    at.selectbox(key="lab_vh_rvi_src").set_value("UST_PAR").run()
    assert at.selectbox(key="lab_vh_rvi_cal_UST_PAR").value == "USGOV"
    assert any("STALE REALISED VS IMPLIED" in m.value and "vol history" in m.value for m in at.markdown)
    at.button(key="lab_vh_rvi_run").click().run()
    assert not at.exception and not at.error
    stored = at.session_state[lab_rates.K_VH_RVI]
    result = stored["result"]
    assert "PROXY" in result.underlying and any("PROXY underlying" in w.value for w in at.warning)
    assert stored["rates"]["dataset"] == "UST_PAR" and "bundled snapshot" in "; ".join(stored["rates"]["origins"])
    assert result.settings.window == 10 and result.settings.calendar == "USGOV"
    trailing = result.summary.set_index("point")["trailing_mean_bp"]
    assert ((trailing > 20) & (trailing < 300)).all()  # realised normal vol of forward Treasury yields, bp a year
    assert set(result.summary["point"]) == {"3Mx2Y", "3Mx10Y", "1Yx2Y", "1Yx10Y", "5Yx2Y", "5Yx10Y"}
    # ICE LIBOR swaps stop in 2016: the section says so instead of scoring nothing
    at.selectbox(key="lab_vh_rvi_src").set_value("USD_SWAP_ICE").run()
    assert not at.exception
    assert any("cannot be the underlying of this vol history" in w.value and "2013" in w.value for w in at.warning)
