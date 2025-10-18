import numpy as np

from derivx.core.bs import bs_price_greeks, put_call_parity_residual


def test_put_call_parity_residual_small():
    rng = np.random.default_rng(0)
    S = rng.uniform(50, 150, size=100)
    K = rng.uniform(50, 150, size=100)
    T = rng.uniform(0.05, 2.0, size=100)
    r = rng.uniform(-0.01, 0.05, size=100)
    q = rng.uniform(0.0, 0.03, size=100)
    sigma = rng.uniform(0.05, 0.6, size=100)
    C, *_ = bs_price_greeks(S, K, T, r, sigma, call=True, q=q)
    P, *_ = bs_price_greeks(S, K, T, r, sigma, call=False, q=q)
    res = np.abs(put_call_parity_residual(C, P, S, K, T, r, q))
    assert np.all(res < 1e-8)


