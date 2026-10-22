"""Hagan SABR: QuantLib golden values, smoothness at the money, the beta = 1
limit, T scaling, input guards, normal vols, ATM alpha, calibration at any
price level, and the SABR cube."""

from __future__ import annotations

import numpy as np
import pytest

from derivx.vol.implied import lognormal_to_normal_vol
from derivx.vol.sabr import (
    SabrCube,
    calibrate_sabr_smile,
    hagan_sabr_iv,
    hagan_sabr_normal_vol,
    sabr_alpha_from_atm,
    sabr_vol,
    z_over_x,
)

RATES = dict(F=0.03, T=5.0, alpha=0.05, beta=0.5, nu=0.4, rho=-0.3)


def test_sabr_matches_quantlib_golden(golden):
    cases = golden("vol")["sabr"]
    assert len(cases) >= 20
    for c in cases:
        K = np.array(c["strikes"])
        mine = sabr_vol(c["F"], K, c["T"], c["alpha"], c["beta"], c["rho"], c["nu"], c["shift"], c["vol_type"])
        ref = np.array(c["vols"])
        near_atm = np.abs((K + c["shift"]) / (c["F"] + c["shift"]) - 1.0) < 1e-5
        # QuantLib evaluates ln((sqrt(1-2 rho z+z^2)+z-rho)/(1-rho)) directly, which loses digits for
        # |z| ~ 1e-6; derivx uses a log1p form, so the tight bound applies away from those strikes.
        np.testing.assert_allclose(mine[~near_atm], ref[~near_atm], rtol=1e-12, atol=0)
        np.testing.assert_allclose(mine[near_atm], ref[near_atm], rtol=1e-9, atol=0)


def test_sabr_matches_quantlib_live_on_random_inputs():
    ql = pytest.importorskip("QuantLib")
    rng = np.random.default_rng(7)
    for _ in range(200):
        F = float(np.exp(rng.uniform(np.log(1e-3), np.log(1e4))))
        beta = float(rng.choice([0.0, 0.3, 0.5, 1.0]))
        T = float(rng.uniform(0.01, 10.0))
        atm = float(rng.uniform(0.05, 0.6))
        alpha = atm * F ** (1 - beta)
        rho, nu = float(rng.uniform(-0.9, 0.9)), float(rng.uniform(0.0, 1.5))
        K = F * np.exp(rng.normal(0, 0.3, 5))
        for vt, qvt in (("lognormal", ql.ShiftedLognormal), ("normal", ql.Normal)):
            try:
                mine = sabr_vol(F, K, T, alpha, beta, rho, nu, 0.0, vt)
            except ValueError:  # expansion breakdown (vol <= 0): QuantLib returns the negative number
                ref = [ql.sabrVolatility(float(k), F, T, alpha, beta, nu, rho, qvt) for k in K]
                assert min(ref) <= 0
                continue
            ref = np.array([ql.sabrVolatility(float(k), F, T, alpha, beta, nu, rho, qvt) for k in K])
            np.testing.assert_allclose(mine, ref, rtol=1e-9)


def test_smile_is_continuous_and_smooth_through_the_forward():
    F = RATES["F"]
    eps = np.array([-1e-4, -1e-6, -1e-8, 0.0, 1e-8, 1e-6, 1e-4])
    v = hagan_sabr_iv(F, F * (1 + eps), RATES["T"], RATES["alpha"], RATES["beta"], RATES["rho"], RATES["nu"])
    atm = v[3]
    # no jump: within 0.01 vol bp of ATM 1e-6 away (the old code jumped by ~90 vol bp here) ...
    assert np.all(np.abs(v[1:6] - atm) < 0.01e-4)
    # ... and 1e-4 away the change is just the smile slope (no kink: one-sided slopes agree)
    slope_left = (v[3] - v[1]) / (F * 1e-6)
    slope_right = (v[5] - v[3]) / (F * 1e-6)
    assert slope_left == pytest.approx(slope_right, rel=1e-3)
    assert v[6] - atm == pytest.approx(slope_right * F * 1e-4, rel=2e-2)
    assert v[0] - atm == pytest.approx(-slope_left * F * 1e-4, rel=2e-2)
    zx = z_over_x(np.array([-2e-6, -1e-6 + 1e-12, 0.0, 1e-6 - 1e-12, 2e-6]), -0.3)
    assert np.all(np.abs(np.diff(zx)) < 1e-6)


def test_off_atm_vol_scales_with_T():
    """VM-01: the correction bracket multiplies T on every branch, so the
    vol is exactly affine in T at every strike (and depends on T off ATM)."""

    K = np.array([0.01, 0.02, 0.029, 0.03, 0.04, 0.06])
    p = {k: RATES[k] for k in ("alpha", "beta", "rho", "nu")}
    v0, v1, v2 = (hagan_sabr_iv(0.03, K, T, **p) for T in (0.0, 1.0, 2.0))
    np.testing.assert_allclose(v2 - v1, v1 - v0, rtol=1e-10)
    assert np.all(np.abs(v1 - v0) > 1e-4)


def test_beta_one_is_the_lognormal_limit_and_needs_no_special_case():
    K = np.array([60.0, 80.0, 100.0, 120.0, 160.0])
    v1 = hagan_sabr_iv(100.0, K, 1.0, 0.2, 1.0, -0.3, 0.6)
    v_near = hagan_sabr_iv(100.0, K, 1.0, 0.2, 1.0 - 1e-9, -0.3, 0.6)
    np.testing.assert_allclose(v1, v_near, rtol=1e-8)
    # closed form at beta = 1: alpha z/x(z) (1 + T (rho nu alpha/4 + (2 - 3 rho^2) nu^2/24)), z = nu/alpha ln(F/K)
    z = 0.6 / 0.2 * np.log(100.0 / K)
    ref = 0.2 * z_over_x(z, -0.3) * (1 + (-0.3 * 0.6 * 0.2 / 4 + (2 - 3 * 0.09) * 0.36 / 24))
    np.testing.assert_allclose(v1, ref, rtol=1e-14)
    # nu = 0, beta = 1 is Black-Scholes: a flat smile at alpha
    np.testing.assert_allclose(hagan_sabr_iv(100.0, K, 1.0, 0.2, 1.0, -0.3, 0.0), 0.2, rtol=1e-14)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(alpha=0.0), "alpha"),
        (dict(alpha=-0.2), "alpha"),
        (dict(rho=1.0), "rho"),
        (dict(rho=-1.5), "rho"),
        (dict(beta=1.2), "beta"),
        (dict(nu=-0.1), "nu"),
        (dict(T=-1.0), "T"),
        (dict(F=-0.002), r"F \+ shift"),
        (dict(K=0.0), r"K \+ shift"),
        (dict(K=-0.002, shift=0.001), r"K \+ shift"),
        (dict(F=float("nan")), "finite"),
    ],
)
def test_invalid_inputs_raise_clear_errors(kwargs, match):
    args = dict(F=0.01, K=0.012, T=1.0, alpha=0.02, beta=0.5, rho=0.0, nu=0.3, shift=0.0)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        hagan_sabr_iv(**args)


def test_expansion_breakdown_raises_instead_of_returning_a_negative_vol():
    alpha = 0.3 * 0.03**0.5
    with pytest.raises(ValueError, match="breaks down"):
        hagan_sabr_iv(0.03, [0.03, 0.0301], 10.0, alpha, 0.5, -0.95, 1.5)


def test_shifted_sabr_prices_negative_rates():
    v = hagan_sabr_iv(-0.002, np.array([-0.01, -0.002, 0.0, 0.01]), 2.0, 0.01, 0.5, 0.1, 0.5, shift=0.02)
    assert np.all(np.isfinite(v)) and np.all(v > 0)


def test_vectorised_over_strikes_matches_scalar_calls():
    K = np.linspace(0.01, 0.06, 11)
    vec = hagan_sabr_iv(0.03, K, 5.0, 0.05, 0.5, -0.3, 0.4)
    loop = [hagan_sabr_iv(0.03, float(k), 5.0, 0.05, 0.5, -0.3, 0.4) for k in K]
    assert isinstance(loop[0], float)
    np.testing.assert_allclose(vec, loop, rtol=0, atol=0)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_normal_expansion_agrees_with_lognormal_by_price_matching(beta):
    """Both Hagan expansions approximate the same SABR prices: the normal
    vol of the lognormal-expansion price is within a fraction of a bp of the
    normal expansion for moderate nu^2 T."""

    F, T = 0.03, 2.0
    alpha = 0.008 / F**beta  # ~80 bp normal at the money
    K = F + np.array([-0.015, -0.005, 0.0, 0.005, 0.015])
    p = dict(alpha=alpha, beta=beta, rho=-0.2, nu=0.3, shift=0.01)
    normal = hagan_sabr_normal_vol(F, K, T, **p)
    via_price = lognormal_to_normal_vol(F, K, T, hagan_sabr_iv(F, K, T, **p), shift=0.01)
    np.testing.assert_allclose(normal * 1e4, via_price * 1e4, atol=0.5)  # bp


@pytest.mark.parametrize("vol_type, atm", [("lognormal", 0.25), ("normal", 0.0085)])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_alpha_from_atm_reprices_the_atm_vol(vol_type, atm, beta):
    F, T = 0.03, 5.0
    alpha = sabr_alpha_from_atm(F, T, atm, beta, -0.3, 0.4, 0.01, vol_type)
    assert alpha > 0
    assert sabr_vol(F, F, T, alpha, beta, -0.3, 0.4, 0.01, vol_type) == pytest.approx(atm, rel=1e-12)


SMILE_PARAMS = dict(beta=0.5, rho=-0.6, nu=1.5)


@pytest.mark.parametrize("F", [1.08, 100.0, 766.0, 5800.0, 15800.0])
@pytest.mark.parametrize("T", [7 / 365, 30 / 365, 1.0, 5.0])
def test_calibration_recovers_parameters_at_any_price_level(F, T):
    """vol-models-V1 / AL-09 / pricer-page-V2: start point and bounds scale
    with F^(1-beta), so a 25% ATM smile is recovered at index and EM FX
    levels (the old absolute alpha <= 5 bound capped ATM vol at 5/sqrt(F))."""

    nu = SMILE_PARAMS["nu"] / np.sqrt(max(T, 0.25))
    true = dict(alpha=0.25 * F**0.5, beta=0.5, rho=-0.6, nu=nu)
    width = 0.25 * np.sqrt(T)
    K = F * np.exp(np.linspace(-2.5 * width, 1.5 * width, 21))
    vols = hagan_sabr_iv(F, K, T, **true)
    fit = calibrate_sabr_smile(K, vols, T, F, beta=0.5)
    assert fit.success
    assert not fit.any_at_bound
    assert fit.rmse_bp < 0.01
    assert fit.alpha == pytest.approx(true["alpha"], rel=1e-5)
    assert fit.rho == pytest.approx(-0.6, abs=1e-5)
    assert fit.nu == pytest.approx(nu, rel=1e-5)
    np.testing.assert_allclose(hagan_sabr_iv(F, K, T, **fit), vols, atol=1e-8)


def test_calibration_result_reports_fixed_free_and_errors():
    F, T = 1.08, 0.25
    K = F * np.exp(np.linspace(-0.15, 0.15, 13))
    vols = hagan_sabr_iv(F, K, T, 0.1, 1.0, -0.2, 1.0)
    fit = calibrate_sabr_smile(K, vols, T, F, beta=1.0)  # FX-standard lognormal SABR
    assert fit.free == ("alpha", "rho", "nu") and fit.fixed == ("beta", "shift")
    assert dict(fit) == fit.params and set(fit) == {"alpha", "beta", "rho", "nu", "shift"}
    assert fit.rmse_bp == pytest.approx(fit.rmse * 1e4) and fit.rmse_vol_points == pytest.approx(fit.rmse * 100)
    assert fit.n_points == 13 and fit.n_dropped == 0
    np.testing.assert_allclose(fit.residuals, fit.model_vols - fit.market_vols)
    assert np.isfinite(fit.jacobian_condition)
    free_beta = calibrate_sabr_smile(K, vols, T, F, beta=0.7, fix_beta=False)
    assert free_beta.beta == pytest.approx(1.0, abs=1e-3) or free_beta.rmse_bp < 0.05
    assert "beta" in free_beta.free


def test_alpha_from_atm_fits_only_rho_and_nu_and_matches_atm():
    F, T = 100.0, 0.5
    K = F * np.exp(np.linspace(-0.3, 0.2, 15))
    vols = hagan_sabr_iv(F, K, T, 0.2 * 10, 0.5, -0.4, 0.9)
    fit = calibrate_sabr_smile(
        K, vols, T, F, beta=0.5, alpha_from_atm=True, atm_vol=float(hagan_sabr_iv(F, F, T, 2.0, 0.5, -0.4, 0.9))
    )
    assert fit.free == ("rho", "nu") and fit.alpha_from_atm
    assert fit.vol(F) == pytest.approx(hagan_sabr_iv(F, F, T, 2.0, 0.5, -0.4, 0.9), rel=1e-12)
    assert fit.rho == pytest.approx(-0.4, abs=1e-6) and fit.nu == pytest.approx(0.9, rel=1e-6)


def test_normal_vol_calibration_with_shift_for_negative_rates():
    F, T, shift = -0.001, 2.0, 0.02
    K = F + np.linspace(-0.008, 0.02, 15)
    true = dict(alpha=0.006, beta=0.0, rho=0.2, nu=0.4, shift=shift)
    nv = hagan_sabr_normal_vol(F, K, T, **true)
    fit = calibrate_sabr_smile(K, nv, T, F, beta=0.0, shift=shift, vol_type="normal", weights="vega")
    assert fit.vol_type == "normal" and fit.rmse_bp < 1e-3  # bp of normal vol
    assert fit.alpha == pytest.approx(0.006, rel=1e-5) and fit.rho == pytest.approx(0.2, abs=1e-4)


def test_bad_quotes_are_dropped_and_robust_loss_resists_an_outlier():
    F, T = 100.0, 0.5
    K = F * np.exp(np.linspace(-0.3, 0.2, 15))
    clean = hagan_sabr_iv(F, K, T, 0.2, 1.0, -0.5, 1.2)
    vols = clean.copy()
    vols[3] = np.nan  # a missing quote no longer kills the calibration
    vols[10] += 0.08  # a stale wing quote, 8 vol points off
    l2 = calibrate_sabr_smile(K, vols, T, F, beta=1.0)
    robust = calibrate_sabr_smile(K, vols, T, F, beta=1.0, loss="soft_l1", f_scale=0.002)
    assert l2.n_dropped == robust.n_dropped == 1
    good = np.ones(K.size, dtype=bool)
    good[[3, 10]] = False

    def err(fit):
        return np.max(np.abs(hagan_sabr_iv(F, K[good], T, **fit) - clean[good])) * 1e4

    assert err(robust) < 5.0  # vol bp
    assert err(l2) > 4 * err(robust)


def test_parameter_at_bound_is_flagged():
    F, T = 100.0, 1.0
    K = F * np.exp(np.linspace(-0.3, 0.3, 11))
    flat = np.full(K.size, 0.2)  # no smile: nu ends at its lower bound 0
    fit = calibrate_sabr_smile(K, flat, T, F, beta=1.0)
    assert fit.at_bound["nu"] and fit.any_at_bound
    assert fit.rmse_bp < 1e-3


def test_calibration_input_errors():
    K = np.linspace(90, 110, 3)
    with pytest.raises(ValueError, match="need at least 4"):
        calibrate_sabr_smile(K, np.full(3, 0.2), 1.0, 100.0)
    with pytest.raises(ValueError, match="T must be > 0"):
        calibrate_sabr_smile(np.linspace(90, 110, 9), np.full(9, 0.2), 0.0, 100.0)
    with pytest.raises(ValueError, match="same length"):
        calibrate_sabr_smile(np.linspace(90, 110, 9), np.full(8, 0.2), 1.0, 100.0)


def _cube_fit(expiry, tenor, F):
    K = F + np.linspace(-0.01, 0.015, 11)
    nv = hagan_sabr_normal_vol(F, K, expiry, 0.006 + 0.001 * tenor / 10, 0.0, -0.1 + 0.02 * expiry, 0.3, shift=0.02)
    return calibrate_sabr_smile(K, nv, expiry, F, beta=0.0, shift=0.02, vol_type="normal")


def test_sabr_cube_reprices_its_nodes_and_interpolates():
    grid = {(e, t): _cube_fit(e, t, 0.03 + 0.001 * t) for e in (1.0, 5.0) for t in (2.0, 10.0)}
    cube = SabrCube(grid)
    for (e, t), fit in grid.items():
        assert cube.vol(e, t, fit.F, fit.F) == pytest.approx(fit.vol(fit.F), rel=1e-10)
        p = cube.params_at(e, t, fit.F)
        assert p["rho"] == pytest.approx(fit.rho) and p["nu"] == pytest.approx(fit.nu)
    mid = cube.vol(3.0, 6.0, 0.036, 0.036)
    corners = [grid[k].vol(grid[k].F) for k in grid]
    assert min(corners) <= mid <= max(corners)
    with pytest.raises(ValueError, match="full expiry x tenor grid"):
        SabrCube({(1.0, 2.0): grid[(1.0, 2.0)], (5.0, 10.0): grid[(5.0, 10.0)]})


def test_normal_beta_zero_alpha_is_the_polynomial_root_bit_for_bit(monkeypatch):
    """A3: the rates cube's normal SABR (beta = 0) solves its ATM equation,
    a linear one, directly instead of through the cubic solver; the alpha
    and every error are the same as the general path's, bit for bit."""

    import derivx.vol.sabr as sabr

    rng = np.random.default_rng(7)
    cases = [(rng.uniform(-0.02, 0.08), rng.uniform(0.0, 30.0), rng.uniform(1e-5, 0.03), rng.uniform(-0.9999, 0.9999),
              rng.uniform(0.0, 8.0), float(rng.choice([0.0, 0.03]))) for _ in range(3000)]  # fmt: skip
    cases += [(0.01, 1.0, 0.01, 1.0, 0.5, 0.0), (0.01, 1.0, -0.01, 0.0, 0.5, 0.0), (-0.05, 1.0, 0.01, 0.0, 0.5, 0.0),
              (0.02, float("nan"), 0.01, 0.0, 0.5, 0.03), (0.02, 1.0, 0.01, 0.0, -0.1, 0.03)]  # fmt: skip

    def solve(F, T, atm, rho, nu, shift):
        try:
            return sabr.sabr_alpha_from_atm(F, T, atm, 0.0, rho, nu, shift, "normal")
        except ValueError as exc:
            return str(exc)

    fast = [solve(*c) for c in cases]
    monkeypatch.setattr(sabr, "_linear_atm_alpha", lambda *args, **kwargs: None)
    general = [solve(*c) for c in cases]
    assert fast == general
    assert sum(isinstance(v, str) for v in fast) >= 5  # the invalid inputs still raise, with the same message
