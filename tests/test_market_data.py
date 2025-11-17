import datetime as dt

import numpy as np
import pandas as pd
import pytest

from derivx.data.market_data import apply_liquidity_filters, compute_moneyness


def test_apply_liquidity_filters_drops_illiquid_quotes():
    data = pd.DataFrame(
        {
            "bid": [1.0, 0.0, 0.5, 0.6],
            "ask": [1.2, 0.5, 1.5, 0.8],
            "volume": [50, 100, 5, 30],
            "strike": [100, 105, 110, 115],
            "underlying_price": [120, 120, 120, 120],
        }
    )
    filtered = apply_liquidity_filters(data, min_volume=10, max_spread=0.2, min_mid=0.5)
    assert len(filtered) == 1
    assert filtered.iloc[0]["strike"] == 100


def test_compute_moneyness_adds_column():
    df = pd.DataFrame({"strike": [100, 105], "underlying_price": [120, 120]})
    out = compute_moneyness(df)
    assert "moneyness" in out.columns
    assert np.allclose(out["moneyness"], [100 / 120, 105 / 120])


def test_compute_moneyness_requires_underlying():
    df = pd.DataFrame({"strike": [100]})
    with pytest.raises(ValueError):
        compute_moneyness(df)
