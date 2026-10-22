"""Daily smile-parameter history and jump detection (VM-18): SVI and SABR
parameters per (trade date, expiry), day-to-day changes on consecutive
snapshots only, and robust jump flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from derivx.data.synthetic import synthetic_history
from derivx.validation.backtest import run_walk_forward
from derivx.validation.stability import PARAMETERS, parameter_history, parameter_jumps


@pytest.fixture(scope="module")
def history():
    return synthetic_history(4)


def test_parameter_history_fits_svi_and_sabr_on_every_slice(history):
    hist, skipped = parameter_history(history, rate=0.04, max_days=60)
    assert set(hist["model"]) == {"svi", "sabr"}
    assert set(hist["trade_date"]) == set(history)
    for model, params in PARAMETERS.items():
        got = hist[hist["model"] == model].groupby(["trade_date", "expiry"])["parameter"].apply(set)
        assert (got == set(params)).all()
    atm = hist[(hist["model"] == "svi") & (hist["parameter"] == "atm_vol")]["value"]
    assert atm.between(0.08, 0.40).all()  # the synthetic 30-day level is about 17%
    assert (hist["days"] <= 60).all() and (hist["rmse_bp"] < 60).all()
    with pytest.raises(ValueError, match="subset"):
        parameter_history(history, models=("heston",))


def test_jumps_flag_a_planted_break_and_skip_gaps():
    days = pd.bdate_range("2026-09-01", periods=12)
    rng = np.random.default_rng(1)
    rows = []
    for exp in (pd.Timestamp("2026-12-18"), pd.Timestamp("2027-03-19")):
        level = 0.2
        for i, d in enumerate(days):
            level += 0.001 * rng.standard_normal() + (0.05 if (exp.month == 12 and i == 7) else 0.0)
            if exp.month == 3 and i == 4:
                continue  # this expiry was not fitted on one day: no change across the hole
            rows.append(
                {
                    "trade_date": d,
                    "expiry": exp,
                    "days": 0,
                    "tau": 0.5,
                    "forward": 100.0,
                    "model": "sabr",
                    "parameter": "alpha",
                    "value": level,
                    "rmse_bp": 1.0,
                }
            )
    changes, summary = parameter_jumps(pd.DataFrame(rows), z=4.0)
    jumps = changes[changes["jump"]]
    assert len(jumps) == 1
    assert jumps.iloc[0]["expiry"] == pd.Timestamp("2026-12-18") and jumps.iloc[0]["to_date"] == days[7]
    march = changes[changes["expiry"] == pd.Timestamp("2027-03-19")]
    assert len(march) == len(days) - 3 and (march["trading_days"] == 1).all()  # 11 steps minus the 2 across the hole
    assert summary.loc[0, "jumps"] == 1 and summary.loc[0, "n_changes"] == len(changes)


def test_walk_forward_slices_carry_the_origin_svi_parameters(history):
    result = run_walk_forward(history, rate=0.04, max_days=40)
    assert not result.empty
    cols = [f"svi_{p}" for p in ("a", "b", "rho", "m", "sigma")]
    assert set(cols) <= set(result.slices.columns) and np.isfinite(result.slices[cols].to_numpy(float)).all()
