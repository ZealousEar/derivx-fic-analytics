import datetime as dt

import numpy as np
import pandas as pd

from derivx.models.comparison import compare_models
from derivx.vol.svi import svi_iv


def _chain(trade_date: dt.date) -> pd.DataFrame:
    expiry = trade_date + dt.timedelta(days=30)
    strikes = np.linspace(90, 110, 7)
    forward = 100.0
    params = {"a": 0.02, "b": 0.25, "rho": -0.1, "m": 0.0, "s": 0.35}
    ivs = np.array([svi_iv(k, params, F=forward) for k in strikes])
    df = pd.DataFrame(
        {
            "trade_date": pd.Timestamp(trade_date),
            "expiry": pd.Timestamp(expiry),
            "strike": strikes,
            "implied_vol": ivs,
            "underlying_price": np.full_like(strikes, forward),
        }
    )
    return df


def test_compare_models_returns_results():
    chain = _chain(dt.date(2024, 1, 2))
    results = compare_models(chain)
    assert set(results["model"]) == {"svi", "sabr", "local_vol"}
    assert (results["rmse"] >= 0).all() or results["rmse"].isna().any()
