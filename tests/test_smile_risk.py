"""Smile shape (ATM skew, curvature) and smile-consistent deltas, checked
by finite differences of prices under each smile dynamic."""

from __future__ import annotations

import numpy as np
import pytest
from _vol_chains import synth_chain

from derivx.calibration.production import calibrate_with_constraints
from derivx.vol.implied import black_price
from derivx.vol.sabr import hagan_sabr_iv
from derivx.vol.smile_risk import sabr_deltas, smile_deltas, smile_term_structure, svi_atm_skew_curvature
from derivx.vol.svi import svi_iv

SVI = {"a": 0.01, "b": 0.1, "rho": -0.6, "m": 0.02, "sigma": 0.15}


def test_atm_skew_and_curvature_match_finite_differences():
    T, h = 0.5, 1e-4
    out = svi_atm_skew_curvature(SVI, T)
    vol = lambda k: svi_iv(np.exp(k), SVI, 1.0, T)  # noqa: E731
    assert out["atm_vol"] == pytest.approx(vol(0.0))
    assert out["skew"] == pytest.approx((vol(h) - vol(-h)) / (2 * h), rel=1e-6)
    assert out["curvature"] == pytest.approx((vol(h) - 2 * vol(0.0) + vol(-h)) / h**2, rel=1e-4)
    assert out["skew"] < 0


def test_smile_term_structure_by_expiry():
    slices = calibrate_with_constraints(synth_chain(days=(30, 91, 365)), rate=0.04, div_yield=0.012)
    table = smile_term_structure(slices)
    assert list(table["tau"]) == sorted(table["tau"])
    assert (table["skew"] < 0).all()
    # the synthetic SSVI smile flattens with maturity
    assert abs(table["skew"].iloc[0]) > abs(table["skew"].iloc[-1])


def _sticky_moneyness_price(F, K, T, df):
    k = np.log(K / F)
    return black_price(F, K, T, svi_iv(np.exp(k), SVI, 1.0, T), True, df)


def test_sticky_strike_and_sticky_moneyness_deltas():
    F, T, df, h = 100.0, 0.5, 0.98, 1e-3
    K = np.array([85.0, 100.0, 110.0])
    vol = svi_iv(K / F, SVI, 1.0, T)
    k = np.log(K / F)
    hk = 1e-6
    dvol_dk = (svi_iv(np.exp(k + hk), SVI, 1.0, T) - svi_iv(np.exp(k - hk), SVI, 1.0, T)) / (2 * hk)
    d = smile_deltas(F, K, T, vol, dvol_dk, df)
    fd_strike = (black_price(F + h, K, T, vol, True, df) - black_price(F - h, K, T, vol, True, df)) / (2 * h)
    fd_moneyness = (_sticky_moneyness_price(F + h, K, T, df) - _sticky_moneyness_price(F - h, K, T, df)) / (2 * h)
    np.testing.assert_allclose(d["sticky_strike"], fd_strike, rtol=1e-7)
    np.testing.assert_allclose(d["sticky_moneyness"], fd_moneyness, rtol=1e-5)
    assert np.all(d["sticky_moneyness"] > d["sticky_strike"])  # negative skew: vol rises as F falls


def test_sabr_hagan_and_bartlett_deltas():
    F, T, df, h = 100.0, 1.0, 1.0, 1e-3
    a, b, r, n = 0.2, 1.0, -0.5, 0.8
    K = np.array([90.0, 100.0, 110.0])

    def price(f, alpha):
        return black_price(f, K, T, hagan_sabr_iv(f, K, T, alpha, b, r, n), True, df)

    d = sabr_deltas(F, K, T, a, b, r, n, df=df)
    hagan_fd = (price(F + h, a) - price(F - h, a)) / (2 * h)
    # Bartlett: alpha moves with F by d alpha = rho nu / F^beta dF
    da = r * n / F**b * h
    bartlett_fd = (price(F + h, a + da) - price(F - h, a - da)) / (2 * h)
    np.testing.assert_allclose(d["hagan"], hagan_fd, rtol=1e-5)
    np.testing.assert_allclose(d["bartlett"], bartlett_fd, rtol=1e-4)
    assert np.all(d["bartlett"] < d["hagan"])  # rho < 0: alpha falls as F rises
    with pytest.raises(ValueError):
        smile_deltas(F, K, 0.0, 0.2, 0.0)
