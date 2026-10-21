"""Dated swaption-vol history (AL-20): upload parsing, the labelled
SYNTHETIC demo, the SABR walk-forward against persistence, the smile-model
comparison and the SABR parameter history with jump flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from derivx.models.comparison import summarise_comparison
from derivx.validation.stability import parameter_jumps
from derivx.validation.vol_history import (
    SYNTHETIC_SOURCE,
    compare_vol_models,
    parse_vol_history,
    run_vol_walk_forward,
    sabr_parameter_history,
    synthetic_vol_history,
)


@pytest.fixture(scope="module")
def synth():
    return synthetic_vol_history(12, expiries=("1Y", "5Y"), tenors=("10Y",), noise_bp=2.0)


def _upload_frame(history, forwards: bool = True) -> pd.DataFrame:
    frame = history.frame().rename(columns={"vol_bp": "vol"})
    return frame if forwards else frame.drop(columns="forward")


def test_upload_round_trip_and_errors(synth):
    hist, issues = parse_vol_history(_upload_frame(synth))
    assert hist is not None and not [i for i in issues if i.level == "error"]
    assert hist.dates == synth.dates and hist.points == synth.points and not hist.synthetic
    pd.testing.assert_frame_equal(hist.frame(), synth.frame())
    pct = _upload_frame(synth).assign(forward=lambda f: f["forward"] * 100)
    hist_pct, issues = parse_vol_history(pct)
    assert any("percent" in i.message for i in issues)
    pd.testing.assert_frame_equal(hist_pct.frame(), synth.frame())
    assert parse_vol_history(_upload_frame(synth).drop(columns="date"))[0] is None
    one_day = _upload_frame(synth)
    one_day = one_day[one_day["date"] == synth.dates[0]]
    none, issues = parse_vol_history(one_day)
    assert none is None and any("at least two" in i.message for i in issues)
    bad = _upload_frame(synth).astype({"date": str})
    bad.loc[3, "date"] = "yesterday"
    assert "unreadable date" in parse_vol_history(bad)[1][0].message


def test_sabr_smoothing_beats_persistence_on_noisy_quotes(synth):
    assert synth.synthetic and synth.source == SYNTHETIC_SOURCE
    result = run_vol_walk_forward(synth)
    s = result.summary.set_index("predictor")
    assert s.loc["sabr_sticky_moneyness", "skill"] > 0.05  # 2bp i.i.d. quote noise: a fitted smile filters it
    assert s.loc["sabr_sticky_moneyness", "benchmark"] == "persistence"
    assert s.loc["sabr_sticky_strike", "n_quotes"] > 0  # forwards given: sticky strike is scored too
    assert set(result.by_point["point"]) == {"1Yx10Y", "5Yx10Y"}
    no_fwd, _ = parse_vol_history(_upload_frame(synth, forwards=False))
    s2 = run_vol_walk_forward(no_fwd).summary.set_index("predictor")
    assert s2.loc["sabr_sticky_strike", "n_quotes"] == 0 and np.isnan(s2.loc["sabr_sticky_strike", "rmse_bp"])
    # normal SABR with beta 0 depends on the rate level only through sqrt(F K) ln(F/K) ~ F - K
    assert s2.loc["sabr_sticky_moneyness", "rmse_bp"] == pytest.approx(
        s.loc["sabr_sticky_moneyness", "rmse_bp"], rel=1e-3
    )


def test_model_comparison_and_parameter_history(synth):
    scores, skipped = compare_vol_models(synth, synth.dates[-1:])
    summary = summarise_comparison(scores).set_index("model")
    assert set(summary.index) == {"normal_sabr", "lognormal_sabr", "spline"}
    assert summary.loc["normal_sabr", "median_rmse_cv_bp"] < summary.loc["spline", "median_rmse_cv_bp"]
    no_fwd, _ = parse_vol_history(_upload_frame(synth, forwards=False))
    _, skipped = compare_vol_models(no_fwd, no_fwd.dates[-1:])
    assert (skipped["model"] == "lognormal_sabr").all() and skipped["reason"].str.contains("no forward").all()
    hist, _ = sabr_parameter_history(synth)
    changes, summ = parameter_jumps(hist)
    assert set(summ["parameter"]) == {"atm_normal_bp", "alpha", "rho", "nu"}
    assert not changes.loc[changes["parameter"] == "atm_normal_bp", "jump"].any()  # a smooth random walk
    # a 40% jump in every vol of one point on one date is flagged on the way in and on the way out
    frame = _upload_frame(synth)
    hit = (frame["date"] == synth.dates[6]) & (frame["expiry"] == "1Y")
    frame.loc[hit, "vol"] *= 1.4
    jumped, _ = parse_vol_history(frame)
    changes, _ = parameter_jumps(sabr_parameter_history(jumped)[0])
    flagged = changes[(changes["parameter"] == "atm_normal_bp") & changes["jump"]]
    assert set(flagged["expiry"]) == {"1Yx10Y"} and len(flagged) == 2
