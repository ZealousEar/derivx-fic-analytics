import numpy as np

from derivx.core.bs import bs_price_greeks
from derivx.vol.implied import implied_vol


def test_round_trip_price_to_iv_to_price_call():
    rng = np.random.default_rng(1)
    for _ in range(50):
        S = float(rng.uniform(50, 150))
        K = float(rng.uniform(50, 150))
        T = float(rng.uniform(0.05, 2.0))
        r = float(rng.uniform(-0.01, 0.05))
        q = float(rng.uniform(0.0, 0.03))
        sigma = float(rng.uniform(0.05, 0.6))

        price, *_ = bs_price_greeks(S, K, T, r, sigma, call=True, q=q)
        iv = implied_vol(float(price), S, K, T, r, call=True, q=q)
        price_back, *_ = bs_price_greeks(S, K, T, r, iv, call=True, q=q)
        assert abs(float(price_back) - float(price)) < 1e-6


