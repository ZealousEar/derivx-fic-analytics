import numpy as np

from derivx.rates.term_structure import build_zero_curve


def test_curve_monotone_positive_dfs():
    quotes = {"1M": 0.02, "6M": 0.025, "1Y": 0.03, "5Y": 0.035}
    curve = build_zero_curve(quotes)
    dfs = curve.dfs
    assert np.all(dfs > 0)
    # Discount factors should be non-increasing in time
    assert np.all(np.diff(dfs) <= 1e-12)


