import datetime as dt

import numpy as np
import pandas as pd

from derivx.surface.term_structure import SurfaceModel, calibrate_surface, compute_cross_greeks
from derivx.vol.svi import svi_iv


def _chain(trade_date: dt.date, expiry: dt.date) -> pd.DataFrame:
    strikes = np.linspace(90, 110, 7)
    forward = 100.0
    params = {"a": 0.02, "b": 0.28, "rho": -0.2, "m": 0.0, "s": 0.4}
    ivs = np.array([svi_iv(k, params, F=forward) for k in strikes])
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


def test_calibrate_surface_builds_model():
    trade_date = dt.date(2024, 1, 2)
    chains = {
        pd.Timestamp(dt.date(2024, 2, 1)): _chain(trade_date, dt.date(2024, 2, 1)),
        pd.Timestamp(dt.date(2024, 3, 1)): _chain(trade_date, dt.date(2024, 3, 1)),
    }
    surface = calibrate_surface(chains)
    assert isinstance(surface, SurfaceModel)
    assert len(surface.slices) == 2
    assert surface.calendar_violations == 0


def test_compute_cross_greeks_returns_rows():
    trade_date = dt.date(2024, 1, 2)
    chains = {pd.Timestamp(dt.date(2024, 2, 1)): _chain(trade_date, dt.date(2024, 2, 1))}
    surface = calibrate_surface(chains)
    greeks = compute_cross_greeks(surface, strikes=[95, 100, 105])
    assert {"vega", "volga", "vanna"}.issubset(greeks.columns)
    assert len(greeks) > 0
