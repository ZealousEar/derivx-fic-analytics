import numpy as np

from derivx.rates.term_structure import ZeroCurve
from derivx.rates.black_ir import price_caplet_black


def test_black_caplet_price_sanity():
    F, K, T, sigma = 0.03, 0.03, 1.0, 0.2
    df = np.exp(-0.03 * T)
    price = price_caplet_black(F, K, T, sigma, df)
    assert price >= 0.0
    # ATM caplet should have positive value
    assert price > 0.0


