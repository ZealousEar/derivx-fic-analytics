import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from derivx.validation.backtest import measure_prediction_error, walk_forward_test
from derivx.vol.svi import svi_iv


def _daily_chain(trade_date: dt.date, shift: float = 0.0, expiry: Optional[dt.date] = None) -> pd.DataFrame:
    expiry = expiry or (dt.date(2024, 2, 1))
    strikes = np.linspace(90, 110, 5)
    forward = 100.0 + shift
    params = {"a": 0.02, "b": 0.25, "rho": -0.1, "m": 0.0, "s": 0.35}
    ivs = np.array([svi_iv(k, params, F=forward) for k in strikes]) + 0.0002 * shift
    df = pd.DataFrame(
        {
            "trade_date": pd.Timestamp(trade_date),
            "expiry": pd.Timestamp(expiry),
            "strike": strikes,
            "implied_vol": ivs,
            "underlying_price": np.full_like(strikes, forward),
            "bid": np.full_like(strikes, 1.0),
            "ask": np.full_like(strikes, 1.2),
            "volume": np.full_like(strikes, 50),
            "open_interest": np.full_like(strikes, 100),
        }
    )
    df["moneyness"] = df["strike"] / forward
    return df


def test_walk_forward_produces_metrics():
    start = dt.date(2024, 1, 1)
    chains = {
        pd.Timestamp(start + dt.timedelta(days=i)): _daily_chain(start + dt.timedelta(days=i), shift=0.5 * i)
        for i in range(3)
    }
    metrics = walk_forward_test(chains, window=1)
    assert not metrics.empty
    assert {"rmse", "hit_rate", "n_points"}.issubset(metrics.columns)


def test_measure_prediction_error_computes_hit_rate():
    pred_frame = pd.DataFrame(
        {
            "predicted_iv": [0.2, 0.25],
            "implied_vol": [0.21, 0.24],
            "prev_iv": [0.19, 0.26],
        }
    )
    metrics = measure_prediction_error(pred_frame)
    assert metrics["rmse"] >= 0
    assert 0 <= metrics["hit_rate"] <= 1 or np.isnan(metrics["hit_rate"])
