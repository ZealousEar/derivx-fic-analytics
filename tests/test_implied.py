"""Implied vols: round trips, deep ITM via parity, status codes, no vol cap,
QuantLib golden values for Black and Bachelier inversions, lognormal <->
normal conversion, and speed."""

from __future__ import annotations

import time

import numpy as np
import pytest

from derivx.core.bs import bs_price_greeks
from derivx.vol.implied import (
    bachelier_price,
    batch_implied_vols,
    black_price,
    black_vega,
    hagan_normal_vol_approx,
    implied_black_vol,
    implied_normal_vol,
    implied_vol,
    implied_vol_bs,
    lognormal_to_normal_vol,
    normal_to_lognormal_vol,
    vega_weights,
)


def test_round_trip_price_to_iv_to_price_call():
    """Every quote either round-trips or is reported as having no time value
    (deep ITM, time value below float precision), in which case the OTM put
    at the same strike recovers the vol."""

    rng = np.random.default_rng(1)
    no_time_value = 0
    for _ in range(50):
        S = float(rng.uniform(50, 150))
        K = float(rng.uniform(50, 150))
        T = float(rng.uniform(0.05, 2.0))
        r = float(rng.uniform(-0.01, 0.05))
        q = float(rng.uniform(0.0, 0.03))
        sigma = float(rng.uniform(0.05, 0.6))

        price, *_ = bs_price_greeks(S, K, T, r, sigma, call=True, q=q)
        res = implied_vol_bs(float(price), S, K, T, r, q=q, is_call=True)
        if res.status == "ok":
            iv = float(res.vol)
            price_back, *_ = bs_price_greeks(S, K, T, r, iv, call=True, q=q)
            assert abs(float(price_back) - float(price)) < 1e-6
            if float(bs_price_greeks(S, K, T, r, sigma, call=True, q=q)[3]) > 1e-3:  # identifiable
                assert iv == pytest.approx(sigma, abs=1e-8)
        else:
            assert res.status == "no_time_value"
            no_time_value += 1
            put, *_ = bs_price_greeks(S, K, T, r, sigma, call=False, q=q)
            assert implied_vol(float(put), S, K, T, r, call=False, q=q) == pytest.approx(sigma, abs=1e-8)
    assert no_time_value < 10


def test_deep_itm_call_is_nan_with_reason_not_a_tiny_vol():
    """critic-3: the old solver returned the bracket edge 1e-6 as a valid vol."""

    S, K, T, r, sigma = 100.0, 60.0, 0.1, 0.05, 0.2
    call = float(bs_price_greeks(S, K, T, r, sigma, call=True)[0])
    put = float(bs_price_greeks(S, K, T, r, sigma, call=False)[0])
    res = implied_vol_bs(call, S, K, T, r, is_call=True)
    assert np.isnan(res.vol) and res.status == "no_time_value"
    assert np.isnan(implied_vol(call, S, K, T, r, call=True))
    assert implied_vol(put, S, K, T, r, call=False) == pytest.approx(0.2, abs=1e-10)
    # a moderately ITM call is inverted through parity to the same vol as the put
    c2 = float(bs_price_greeks(S, 90.0, 1.0, r, 0.3, call=True)[0])
    assert implied_vol(c2, S, 90.0, 1.0, r, call=True) == pytest.approx(0.3, abs=1e-12)


def test_status_codes_for_prices_outside_no_arbitrage_bounds():
    F, K, T, df = 100.0, 100.0, 1.0, 0.95
    prices = [-1.0, 0.0, 10.0, 96.0, 10.0, 10.0, np.nan]
    Ks = [K, K, K, K, K, -5.0, K]
    Ts = [T, T, T, T, 0.0, T, T]
    res = implied_black_vol(prices, F, Ks, Ts, is_call=True, df=df)
    assert list(res.status) == [
        "below_intrinsic",
        "no_time_value",
        "ok",
        "above_upper_bound",  # a call is worth less than df F = 95
        "expired",
        "invalid_input",
        "invalid_input",
    ]
    assert np.isnan(res.vol[[0, 1, 3, 4, 5, 6]]).all() and np.isfinite(res.vol[2])
    assert res.reason_counts() == {
        "above_upper_bound": 1,
        "below_intrinsic": 1,
        "expired": 1,
        "invalid_input": 2,
        "no_time_value": 1,
        "ok": 1,
    }
    itm_below = implied_black_vol(19.0 * df, 120.0, 100.0, 1.0, is_call=True, df=df)  # below intrinsic 20 df
    assert itm_below.status == "below_intrinsic"


@pytest.mark.parametrize("vol", [4.9, 6.0, 8.0])
def test_no_artificial_vol_cap(vol):
    price = black_price(100.0, 100.0, 1.0, vol)
    assert float(implied_black_vol(price, 100.0, 100.0, 1.0).vol) == pytest.approx(vol, rel=1e-9)


def test_vol_beyond_float_resolution_is_reported_not_capped():
    # at 2500% vol an ATM call is worth F (1 - 1e-35): indistinguishable from the bound in a double
    price = black_price(100.0, 100.0, 1.0, 25.0)
    res = implied_black_vol(price, 100.0, 100.0, 1.0)
    assert res.status == "above_upper_bound" and np.isnan(res.vol)


def test_deep_otm_and_short_dated_round_trip():
    cases = [
        (100.0, 300.0, 0.5, 0.2),
        (100.0, 40.0, 0.5, 0.25),
        (1.08, 1.0835, 1 / 365, 0.08),
        (100.0, 100.0, 1.0, 0.005),
    ]
    for F, K, T, vol in cases:
        is_call = K >= F
        p = black_price(F, K, T, vol, is_call)
        assert p > 0
        assert float(implied_black_vol(p, F, K, T, is_call).vol) == pytest.approx(vol, rel=1e-9)


def test_vectorised_mixed_calls_and_puts():
    F = 100.0
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vols = np.array([0.3, 0.25, 0.2, 0.18, 0.17])
    types = np.array(["put", "P", "call", "C", "call"])
    prices = black_price(F, K, 0.5, vols, types, df=0.98)
    res = implied_black_vol(prices, F, K, 0.5, types, df=0.98)
    np.testing.assert_allclose(res.vol, vols, rtol=1e-11)
    flags = np.array([False, False, True, True, True])
    out = batch_implied_vols(prices, 100.0 * 0.98, K, 0.5, -np.log(0.98) / 0.5, call=flags)  # F = S e^{rT} = 100
    np.testing.assert_allclose(out, vols, rtol=1e-10)
    with pytest.raises(ValueError, match="'call'/'put'"):
        black_price(F, K, 0.5, vols, ["call", "put", "straddle", "call", "call"])


def test_shifted_black_for_negative_forwards():
    F, K, T, s = -0.002, np.array([-0.01, 0.0, 0.01]), 2.0, 0.02
    p = black_price(F, K, T, 0.2, True, shift=s)
    res = implied_black_vol(p, F, K, T, True, shift=s)
    np.testing.assert_allclose(res.vol, 0.2, rtol=1e-11)
    assert implied_black_vol(0.001, F, K, T, True).status[0] == "invalid_input"  # unshifted Black cannot


def test_black_inversion_matches_quantlib_golden(golden):
    for c in golden("vol")["black_implied"]:
        price = black_price(c["F"], c["K"], c["T"], c["vol"], c["is_call"], c["df"], c["shift"])
        assert price == pytest.approx(c["price"], rel=1e-12)
        res = implied_black_vol(c["price"], c["F"], c["K"], c["T"], c["is_call"], c["df"], c["shift"])
        assert res.status == "ok"
        assert float(res.vol) == pytest.approx(c["ql_vol"], rel=1e-9)
        assert float(res.vol) == pytest.approx(c["vol"], rel=1e-9)


def test_bachelier_inversion_matches_quantlib_golden(golden):
    for c in golden("vol")["bachelier_implied"]:
        price = bachelier_price(c["F"], c["K"], c["T"], c["vol"], c["is_call"], c["df"])
        assert price == pytest.approx(c["price"], rel=1e-12)
        res = implied_normal_vol(c["price"], c["F"], c["K"], c["T"], c["is_call"], c["df"])
        assert res.status == "ok"
        assert float(res.vol) == pytest.approx(c["ql_vol"], rel=1e-8)
        assert float(res.vol) == pytest.approx(c["vol"], rel=1e-10)


def test_inversions_match_quantlib_live():
    ql = pytest.importorskip("QuantLib")
    rng = np.random.default_rng(3)
    for _ in range(100):
        F = float(rng.uniform(50, 150))
        K = float(F * np.exp(rng.normal(0, 0.3)))
        T = float(rng.uniform(0.05, 5))
        vol = float(rng.uniform(0.05, 1.5))
        call = bool(rng.random() < 0.5)
        t = ql.Option.Call if call else ql.Option.Put
        p = ql.blackFormula(t, K, F, vol * np.sqrt(T), 0.97)
        ref = ql.blackFormulaImpliedStdDev(t, K, F, p, 0.97, 0.0, ql.nullDouble(), 1e-14, 1000) / np.sqrt(T)
        res = implied_black_vol(p, F, K, T, call, 0.97)
        if res.status == "ok":
            assert float(res.vol) == pytest.approx(ref, rel=1e-7)
        Fn, Kn, vn = float(rng.normal(0.02, 0.02)), float(rng.normal(0.02, 0.02)), float(rng.uniform(0.001, 0.02))
        pn = ql.bachelierBlackFormula(t, Kn, Fn, vn * np.sqrt(T), 0.97)
        rn = implied_normal_vol(pn, Fn, Kn, T, call, 0.97)
        ref_n = ql.bachelierBlackFormulaImpliedVol(t, Kn, Fn, T, pn, 0.97)
        if rn.status == "ok" and ref_n > 0:
            assert float(rn.vol) == pytest.approx(ref_n, rel=1e-6)
        elif rn.status == "ok":  # QuantLib returns 0 for deep OTM prices (~1e-30); derivx still recovers the vol
            assert float(rn.vol) == pytest.approx(vn, rel=1e-8)


def test_bachelier_handles_negative_rates_and_deep_otm():
    F, K, T = -0.005, np.array([-0.02, -0.01, -0.005, 0.0, 0.01]), 1.0
    for vol in (0.0005, 0.005, 0.02):
        p = bachelier_price(F, K, T, vol, K >= F)
        np.testing.assert_allclose(implied_normal_vol(p, F, K, T, K >= F).vol, vol, rtol=1e-10)
    deep = bachelier_price(0.0, 0.2, 1.0, 0.005, True)  # 40 standard deviations OTM
    assert 0 < deep < 1e-300 or deep == 0.0


def test_lognormal_normal_conversion_round_trip_and_hagan_approximation():
    F, T = 0.03, np.array([0.25, 1.0, 5.0, 10.0])[:, None]
    K = np.array([0.01, 0.02, 0.03, 0.04, 0.06])[None, :]
    vol = 0.3
    vn = lognormal_to_normal_vol(F, K, T, vol)
    np.testing.assert_allclose(normal_to_lognormal_vol(F, K, T, vn), vol, rtol=1e-10)
    np.testing.assert_allclose(black_price(F, K, T, vol, K >= F), bachelier_price(F, K, T, vn, K >= F), rtol=1e-12)
    approx = hagan_normal_vol_approx(F, K, T, vol)
    assert np.max(np.abs(approx - vn)) * 1e4 < 0.05  # bp of normal vol
    # smooth at the money (no 0/0)
    near = hagan_normal_vol_approx(F, F * (1 + np.array([-1e-9, 0.0, 1e-9])), 1.0, vol)
    assert np.all(np.isfinite(near)) and np.ptp(near) < 1e-10


def test_vegas_and_weights():
    v = black_vega(100.0, np.array([80.0, 100.0, 120.0]), 1.0, 0.2, df=0.97)
    h = 1e-6
    fd = (black_price(100.0, 100.0, 1.0, 0.2 + h, df=0.97) - black_price(100.0, 100.0, 1.0, 0.2 - h, df=0.97)) / (2 * h)
    assert v[1] == pytest.approx(fd, rel=1e-8)
    w = vega_weights(100.0, np.array([60.0, 80.0, 100.0, 120.0, 150.0]), 0.25, 0.2)
    assert w.mean() == pytest.approx(1.0) and w.argmax() == 2


def test_vectorised_solver_is_fast():
    rng = np.random.default_rng(0)
    n = 10_000
    F = rng.uniform(50, 150, n)
    K = F * np.exp(rng.normal(0, 0.3, n))
    T = rng.uniform(0.02, 3, n)
    vol = rng.uniform(0.05, 1.0, n)
    call = rng.random(n) < 0.5
    p = black_price(F, K, T, vol, call)
    t0 = time.perf_counter()
    res = implied_black_vol(p, F, K, T, call)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0  # the old scalar loop took ~23 s for 10k quotes
    ok = res.ok
    assert ok.mean() > 0.98
    np.testing.assert_allclose(black_price(F[ok], K[ok], T[ok], res.vol[ok], call[ok]), p[ok], rtol=1e-9, atol=1e-12)


def test_price_within_rounding_of_the_bound_gets_no_vol():
    """Review: an ITM put whose OTM-equivalent call sits at F to float
    precision (std 40) came back 'ok' with 325% vol instead of 808%; the
    ITM -> OTM parity subtraction loses more than 4 eps of the bound."""

    F, K, T, vol = 3.064916374950069, 37.523517370675755, 24.97128674243225, 8.082468319588973
    df = 0.97
    p = black_price(F, K, T, vol, False, df)
    res = implied_black_vol(p, F, K, T, False, df)
    assert res.status == "above_upper_bound" and np.isnan(res.vol)
    # the same vol at a shorter tenor is still identifiable and round-trips
    p2 = black_price(F, K, 1.0, vol, False, df)
    assert float(implied_black_vol(p2, F, K, 1.0, False, df).vol) == pytest.approx(vol, rel=1e-8)


def test_subnormal_otm_price_has_no_identifiable_vol():
    p = black_price(0.2613582326694147, 0.6404068907796372, 0.0014690485187696634, 0.6204575005611779, True)
    assert 0 < p < np.finfo(float).tiny
    res = implied_black_vol(p, 0.2613582326694147, 0.6404068907796372, 0.0014690485187696634, True)
    assert res.status == "no_time_value"
    assert implied_normal_vol(5e-324, 0.0095, 0.00147, 0.0298, False).status == "no_time_value"
