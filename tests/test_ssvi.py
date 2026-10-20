"""SSVI surface: arbitrage-free parameter set, fit recovery, raw-SVI slice
equivalence, and Dupire local vol checked against finite differences."""

from __future__ import annotations

import numpy as np
import pytest
from _vol_chains import SURFACE

from derivx.vol.ssvi import SsviSurface, fit_ssvi
from derivx.vol.svi import butterfly_density, svi_total_variance


def test_slices_are_raw_svi_and_the_surface_is_arbitrage_free():
    k = np.linspace(-1.5, 1.5, 61)
    for T in (0.05, 0.3, 1.0, 2.0):
        raw = svi_total_variance(k, SURFACE.slice_svi(T))
        np.testing.assert_allclose(raw, SURFACE.total_variance(k, T), rtol=1e-12)
    report = SURFACE.arbitrage_report()
    assert report["eta_condition_ok"] and report["gamma_ok"] and report["theta_monotone"]
    assert report["butterfly_ok"] and report["calendar_ok"] and report["butterfly_min_g"] > 0


def test_atm_total_variance_term_structure():
    np.testing.assert_allclose(SURFACE.total_variance(0.0, SURFACE.expiries), SURFACE.theta, rtol=1e-14)
    T_mid = 0.5 * (SURFACE.expiries[1] + SURFACE.expiries[2])
    assert SURFACE.theta_at(T_mid) == pytest.approx(0.5 * (SURFACE.theta[1] + SURFACE.theta[2]))
    # beyond the last expiry the ATM vol is flat
    assert np.sqrt(SURFACE.theta_at(2.0) / 2.0) == pytest.approx(np.sqrt(SURFACE.theta[-1] / SURFACE.expiries[-1]))


def test_fit_recovers_a_known_surface():
    data = []
    for T in SURFACE.expiries:
        k = np.linspace(-3, 2, 15) * 0.18 * np.sqrt(T)
        data.append((T, k, SURFACE.implied_vol(k, T)))
    fit = fit_ssvi(data, loss="linear")
    assert fit.success and fit.rmse_bp < 0.01
    assert fit.surface.rho == pytest.approx(SURFACE.rho, abs=1e-5)
    assert fit.surface.eta == pytest.approx(SURFACE.eta, rel=1e-4)
    assert fit.surface.gamma == pytest.approx(SURFACE.gamma, abs=1e-4)
    np.testing.assert_allclose(fit.surface.theta, SURFACE.theta, rtol=1e-5)


def test_fit_stays_arbitrage_free_on_noisy_quotes():
    rng = np.random.default_rng(5)
    data = []
    for T in SURFACE.expiries:
        k = np.linspace(-3, 2, 15) * 0.18 * np.sqrt(T)
        data.append((T, k, SURFACE.implied_vol(k, T) + 0.003 * rng.standard_normal(k.size)))
    fit = fit_ssvi(data)
    rep = fit.surface.arbitrage_report()
    assert rep["eta_condition_ok"] and rep["butterfly_ok"] and rep["calendar_ok"]
    assert 15 < fit.rmse_bp < 50  # 30 vol bp of noise


def test_dupire_local_vol_matches_finite_differences_of_total_variance():
    s = SURFACE
    T, h = 0.4, 1e-6
    k = np.linspace(-0.4, 0.3, 8)
    dw_dT = (s.total_variance(k, T + h) - s.total_variance(k, T - h)) / (2 * h)
    np.testing.assert_allclose(s.dw_dT(k, T), dw_dT, rtol=1e-6)
    g = butterfly_density(k, s.slice_svi(T))
    np.testing.assert_allclose(s.local_vol(k, T), np.sqrt(dw_dT / g), rtol=1e-6)
    # short-dated ATM local vol is close to ATM implied vol
    T0 = 0.5 * s.expiries[0]
    assert s.local_vol(0.0, T0) == pytest.approx(float(s.implied_vol(0.0, T0)), rel=0.05)
    # with a negative skew, local vol is steeper than implied vol (the 'twice the skew' rule)
    lv = s.local_vol(np.array([-0.05, 0.05]), 1.0)
    iv = s.implied_vol(np.array([-0.05, 0.05]), 1.0)
    assert (lv[0] - lv[1]) > 1.5 * (iv[0] - iv[1]) > 0


def test_invalid_surfaces_are_rejected():
    with pytest.raises(ValueError, match="non-decreasing"):
        SsviSurface(np.array([0.5, 1.0]), np.array([0.02, 0.01]), -0.5, 1.0, 0.4)
    with pytest.raises(ValueError, match="increasing"):
        SsviSurface(np.array([1.0, 0.5]), np.array([0.01, 0.02]), -0.5, 1.0, 0.4)
    with pytest.raises(ValueError, match="rho"):
        SsviSurface(np.array([1.0]), np.array([0.02]), -1.0, 1.0, 0.4)
    with pytest.raises(ValueError, match="quotes"):
        fit_ssvi([(0.5, np.array([0.0, 0.1]), np.array([0.2, 0.2]))])
