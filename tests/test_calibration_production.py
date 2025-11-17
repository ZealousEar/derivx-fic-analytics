import datetime as dt

import numpy as np
import pandas as pd

from derivx.calibration.production import CalibratedSlice, benchmark_vs_baseline, calibrate_with_constraints
from derivx.vol.svi import svi_iv


def _synthetic_chain(trade_date: dt.date, days_to_expiry: int = 30) -> pd.DataFrame:
    expiry = trade_date + dt.timedelta(days=days_to_expiry)
    strikes = np.linspace(80, 120, 9)
    forward = 100.0
    params = {"a": 0.02, "b": 0.3, "rho": -0.2, "m": 0.0, "s": 0.4}
    ivs = np.array([svi_iv(k, params, F=forward) for k in strikes])
    data = {
        "trade_date": pd.Timestamp(trade_date),
        "expiry": pd.Timestamp(expiry),
        "strike": strikes,
        "implied_vol": ivs + 0.0005 * np.random.default_rng(0).standard_normal(len(strikes)),
        "underlying_price": np.full_like(strikes, forward),
        "bid": np.full_like(strikes, 1.0),
        "ask": np.full_like(strikes, 1.2),
        "volume": np.full_like(strikes, 50),
        "open_interest": np.full_like(strikes, 100),
        "option_type": ["call"] * len(strikes),
    }
    df = pd.DataFrame(data)
    df["moneyness"] = df["strike"] / forward
    return df


def test_calibrate_with_constraints_recovers_params():
    trade_date = dt.date(2024, 1, 2)
    chain = _synthetic_chain(trade_date)
    results = calibrate_with_constraints(chain, enforce_arbitrage=False)
    assert len(results) == 1
    result = results[0]
    assert result.n_points == len(chain)
    assert result.rmse < 0.01
    assert "condition_number" in result.diagnostics


def test_benchmark_vs_baseline_reports_metrics():
    trade_date = dt.date(2024, 1, 2)
    chain = _synthetic_chain(trade_date)
    result = calibrate_with_constraints(chain, enforce_arbitrage=False)[0]
    metrics = benchmark_vs_baseline(result, chain)
    assert set(metrics.keys()) == {"rmse_svi", "rmse_spline", "relative_improvement"}
    assert metrics["rmse_svi"] >= 0
