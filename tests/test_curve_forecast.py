"""Walk-forward of yield-curve forecasts (AL-20): factor models beat the
random walk only when the factors mean-revert, never use data after the
origin, and refuse to score across gaps in a daily history."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from derivx.data import rates_history as rh
from derivx.rates.curve_models import DL_TAU, ns_loadings
from derivx.validation.curve_forecast import PREDICTORS, CurveForecastSettings, run_curve_walk_forward

TENORS = ("3M", "1Y", "2Y", "5Y", "10Y", "30Y")
YEARS = np.array([0.25, 1.0, 2.0, 5.0, 10.0, 30.0])


def _history(phi: float, n: int = 400, seed: int = 3) -> pd.DataFrame:
    """SYNTHETIC curves from Nelson-Siegel factors following an AR(1)
    (phi < 1: mean reversion; phi = 1: random walks) plus 0.5bp noise."""

    rng = np.random.default_rng(seed)
    mean = np.array([0.04, -0.01, 0.005])
    beta = mean + np.array([0.01, 0.01, 0.01])
    X = ns_loadings(YEARS, DL_TAU)
    rows = []
    for _ in range(n):
        beta = mean + phi * (beta - mean) + np.array([6e-4, 6e-4, 8e-4]) * rng.standard_normal(3)
        rows.append(X @ beta + 5e-5 * rng.standard_normal(YEARS.size))
    days = pd.bdate_range("2024-01-02", periods=n).date
    return pd.DataFrame(rows, index=days, columns=TENORS)


def test_factor_forecasts_beat_the_random_walk_only_when_factors_mean_revert():
    cfg = CurveForecastSettings(horizon=5, window=120)
    rev = run_curve_walk_forward(_history(0.9), cfg).summary.set_index("predictor")
    assert rev.loc["ns_ar1", "skill"] > 0.1 and rev.loc["ns_ar1", "dm_t"] < -2.0
    assert rev.loc["pca_ar1", "skill"] > 0.1
    rw = run_curve_walk_forward(_history(1.0), cfg).summary.set_index("predictor")
    assert abs(rw.loc["ns_ar1", "skill"]) < 0.1 and abs(rw.loc["pca_ar1", "skill"]) < 0.1  # no skill to find
    assert set(rw.index) == set(PREDICTORS) and np.isnan(rw.loc["random_walk", "skill"])


def test_forecasts_use_no_data_after_the_origin():
    levels = _history(0.9, n=200)
    cfg = CurveForecastSettings(horizon=1, window=60)
    base = run_curve_walk_forward(levels, cfg).errors
    changed = levels.copy()
    changed.iloc[150:] += 0.01  # a 100bp jump from row 150 on
    after = run_curve_walk_forward(changed, cfg).errors
    cut = pd.Timestamp(levels.index[149])
    errs = [c for c in base.columns if c.startswith("err_bp_")]
    early = base.loc[base["target"] <= cut, ["origin", "tenor", *errs]]
    pd.testing.assert_frame_equal(early, after.loc[after["target"] <= cut, ["origin", "tenor", *errs]])
    # only the regime labels (terciles of the whole sample, documented as labels) may change
    assert not after.loc[after["target"] > cut, errs].equals(base.loc[base["target"] > cut, errs])


def test_a_monthly_stretch_is_skipped_not_scored_as_daily():
    daily = _history(0.9, n=120)
    monthly = daily.iloc[::21].copy()
    monthly.index = [d - dt.timedelta(days=400) for d in monthly.index]
    levels = pd.concat([monthly, daily]).sort_index()
    result = run_curve_walk_forward(levels, CurveForecastSettings(horizon=1, window=40))
    assert not result.skipped.empty and result.skipped["reason"].str.contains("not a daily window").all()
    first_scored = result.errors["origin"].min()
    assert first_scored >= pd.Timestamp(daily.index[39])  # the first window fully inside the daily stretch


def test_runs_on_the_treasury_history_with_regimes():
    store = rh.bundled_store()
    levels = store.complete("UST_PAR")[list(("3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"))]
    levels = levels[levels.index >= dt.date(2024, 1, 1)]
    result = run_curve_walk_forward(levels, CurveForecastSettings(horizon=1, window=120))
    assert not result.empty and set(result.by_regime["regime"]) >= {"low", "mid", "high"}
    s = result.summary.set_index("predictor")
    assert 1.0 < s.loc["random_walk", "rmse_bp"] < 15.0  # daily Treasury moves: a few bp
    assert s.loc["ns_curve", "rmse_bp"] > s.loc["ns_ar1", "rmse_bp"]  # dropping the par-curve residual costs
    with pytest.raises(ValueError, match="window"):
        CurveForecastSettings(window=10)
