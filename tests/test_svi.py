"""Raw SVI in total variance: QuantLib golden vols and densities, butterfly
(Gatheral g), Lee and calendar checks, the a + b sigma sqrt(1-rho^2) >= 0
constraint, robust loss, arbitrage enforcement and input guards."""

from __future__ import annotations

import numpy as np
import pytest

from derivx.vol.svi import (
    basic_no_arb_checks,
    butterfly_density,
    check_butterfly,
    check_calendar,
    fit_svi_strike_slice,
    lee_wing_check,
    svi_derivatives,
    svi_iv,
    svi_total_variance,
)

VOGT = {"a": -0.0410, "b": 0.1331, "rho": 0.3060, "m": 0.3586, "sigma": 0.4153}  # Gatheral-Jacquier Ex. 3.1
CLEAN = {"a": 0.04, "b": 0.4, "rho": -0.4, "m": 0.0, "sigma": 0.2}


def test_svi_fit_on_synthetic_slice():
    rng = np.random.default_rng(2)
    F = 100.0
    Ks = np.linspace(60, 140, 15)
    # True params (a,b,rho,m,s)
    true = {"a": 0.02, "b": 0.3, "rho": -0.2, "m": 0.0, "s": 0.4}
    T = 1.0  # params are total variance at T = 1
    ivs = svi_iv(Ks, true, F, T)
    # Add small noise
    ivs_noisy = ivs + 0.0005 * rng.standard_normal(size=ivs.shape)

    est = fit_svi_strike_slice(Ks, ivs_noisy, F, T)
    ivs_fit = svi_iv(Ks, est, F, T)
    rmse = float(np.sqrt(np.mean((ivs_fit - ivs) ** 2)))
    assert rmse < 0.0005  # within the 5 vol bp quote noise of the true smile (was < 50 bp)
    assert 0.0002 < est.rmse_bp * 1e-4 < 0.0007  # and the residual is the noise, not a misfit


def test_exact_quotes_recover_the_parameters_and_scale_with_T():
    F, Ks = 100.0, np.linspace(60, 140, 15)
    true = {"a": 0.02, "b": 0.3, "rho": -0.2, "m": 0.0, "sigma": 0.4}  # total variance at T = 1
    ivs = svi_iv(Ks, true, F, 1.0)
    est = fit_svi_strike_slice(Ks, ivs, F, 1.0)
    for name, value in true.items():
        assert est.params[name] == pytest.approx(value, abs=1e-6), name
    assert est.rmse_bp < 1e-4
    # the same vols read at T = 0.1 / 2 are a total variance 0.1x / 2x as large: a and b scale, the rest do not
    for T in (0.1, 2.0):
        scaled = fit_svi_strike_slice(Ks, ivs, F, T).params
        assert scaled["a"] == pytest.approx(T * true["a"], abs=1e-6) and scaled["b"] == pytest.approx(
            T * true["b"], abs=1e-6
        )
        assert scaled["rho"] == pytest.approx(true["rho"], abs=1e-5) and scaled["sigma"] == pytest.approx(
            true["sigma"], abs=1e-5
        )


def test_svi_no_arb_basic_sanity():
    params_list = [
        {"a": 0.02, "b": 0.2, "rho": 0.0, "m": 0.0, "s": 0.2},
        {"a": 0.01, "b": 0.1, "rho": -0.5, "m": 0.1, "s": 0.3},
    ]
    diag = basic_no_arb_checks(params_list)
    assert diag["viol_b"] == 0
    assert diag["viol_rho"] == 0
    assert diag["viol_butterfly"] == 0 and diag["viol_lee"] == 0 and diag["min_g"] > 0


def test_svi_vols_and_density_sign_match_quantlib_golden(golden):
    for case in golden("vol")["svi"]:
        K = np.array(case["strikes"])
        np.testing.assert_allclose(svi_iv(K, case["params"], case["F"], case["T"]), case["vols"], rtol=1e-13)
        g = butterfly_density(np.log(K / case["F"]), case["params"])
        dens = np.array(case["density"])
        clear = np.abs(dens) > 1e-6  # away from the sign changes, where the numerical density is ~0
        assert np.array_equal(np.sign(g[clear]), np.sign(dens[clear]))
        if case["name"] == "vogt":
            assert (dens < 0).any()
        else:
            assert (dens > 0).all()


def test_vogt_slice_has_butterfly_arbitrage_and_says_where():
    bf = check_butterfly(VOGT)
    assert not bf.ok
    assert bf.min_g == pytest.approx(-0.0329, abs=5e-4)
    assert 0.8 < bf.k_at_min < 1.0
    (lo, hi), *_ = bf.negative_ranges
    assert 0.6 < lo < 0.7 and 1.2 < hi < 1.3
    assert lee_wing_check(VOGT)["lee_ok"]  # the wings are fine: only the density check catches it
    clean = check_butterfly(CLEAN)
    assert clean.ok and clean.min_g > 0.1 and clean.negative_ranges == []


def test_g_matches_breeden_litzenberger_density_sign():
    """g(k) has the sign of d2C/dK2 (the risk-neutral density)."""

    from derivx.vol.implied import black_price

    K = np.exp(np.linspace(0.3, 1.5, 61))
    h = 1e-4
    vol = lambda x: svi_iv(x, VOGT, 1.0, 1.0)  # noqa: E731
    c = lambda x: black_price(1.0, x, 1.0, vol(x))  # noqa: E731
    d2c = (c(K * (1 + h)) - 2 * c(K) + c(K * (1 - h))) / (K * h) ** 2
    g = butterfly_density(np.log(K), VOGT)
    clear = np.abs(g) > 1e-3
    assert np.array_equal(np.sign(d2c[clear]), np.sign(g[clear]))


def test_derivatives_match_finite_differences():
    k = np.linspace(-1, 1, 9)
    w, w1, w2 = svi_derivatives(k, CLEAN)
    h = 1e-5
    up, dn = svi_total_variance(k + h, CLEAN), svi_total_variance(k - h, CLEAN)
    np.testing.assert_allclose(w1, (up - dn) / (2 * h), rtol=1e-7)
    np.testing.assert_allclose(w2, (up - 2 * w + dn) / h**2, rtol=1e-4)


def test_lee_wing_check_flags_steep_wings():
    steep = dict(CLEAN, b=1.8, rho=0.3)
    lee = lee_wing_check(steep)
    assert lee["lee_right_slope"] == pytest.approx(1.8 * 1.3)
    assert not lee["lee_ok"]


def test_calendar_check_detects_a_wing_crossing():
    """VM-12: a steep short slice can cross a flat longer slice in the wing
    while ATM total variance still increases."""

    short = {"a": 0.005, "b": 0.12, "rho": -0.9, "m": 0.02, "sigma": 0.05}  # 3M, steep put skew
    long = {"a": 0.03, "b": 0.03, "rho": 0.0, "m": 0.0, "sigma": 0.1}  # 6M, flat
    assert svi_total_variance(0.0, long) > svi_total_variance(0.0, short)  # ATM looks fine
    (check,) = check_calendar([(0.25, short), (0.5, long)], np.linspace(-0.5, 0.5, 201))
    assert not check.ok and check.min_dw < 0 and check.k_at_min < 0
    assert check.negative_ranges[0][0] == pytest.approx(-0.5)
    ok = check_calendar([(0.25, CLEAN), (0.5, {k: v * 2 if k in ("a", "b") else v for k, v in CLEAN.items()})])
    assert ok[0].ok


def test_basic_no_arb_checks_counts_real_violations():
    """AL-16 / VM-04: the old check counted only b <= 0 and |rho| >= 1,
    which the optimiser bounds made impossible."""

    steep = dict(CLEAN, b=1.8, rho=0.3)
    diag = basic_no_arb_checks([VOGT, CLEAN, steep, dict(T=0.5, **CLEAN), dict(T=1.0, **CLEAN)])
    # Vogt's slice, and the steep one: a Lee slope above 2 makes g negative in the far wing too
    assert diag["viol_butterfly"] == 2
    assert check_butterfly(steep).negative_ranges[-1][1] == pytest.approx(check_butterfly(steep).k_max)
    assert diag["viol_lee"] == 1
    assert diag["viol_calendar"] == 0  # the same total variance at T = 0.5 and 1 is not a decrease
    diag2 = basic_no_arb_checks([dict(T=0.5, **CLEAN), dict(T=1.0, **dict(CLEAN, a=0.01))])
    assert diag2["viol_calendar"] == 1
    assert diag["min_g"] < 0


def test_fit_is_in_total_variance_and_scales_with_T():
    """PP-14 / VM-16: the same vols at T = 0.25 and T = 1 give a and b
    scaled by T, and identical rho, m, sigma."""

    F = 100.0
    K = F * np.exp(np.linspace(-0.5, 0.4, 17))
    vols = svi_iv(K, CLEAN, F, 1.0)
    f1 = fit_svi_strike_slice(K, vols, F, 1.0, loss="linear")
    fq = fit_svi_strike_slice(K, vols, F, 0.25, loss="linear")
    assert fq.a == pytest.approx(0.25 * f1.a, rel=1e-6) and fq.b == pytest.approx(0.25 * f1.b, rel=1e-6)
    for p in ("rho", "m", "sigma"):
        assert fq[p] == pytest.approx(f1[p], rel=1e-5, abs=1e-8)
    assert f1.T == 1.0 and fq.T == 0.25
    np.testing.assert_allclose(svi_iv(K, fq, F), vols, atol=1e-9)  # T read from the fit


def test_negative_a_is_allowed():
    """VM-08: a >= 0 used to pin rho at -0.999 and miss by 261 vol bp."""

    true = {"a": -0.08, "b": 0.48, "rho": -0.7, "m": 0.05, "sigma": 0.25}
    F, T = 100.0, 1.0
    K = F * np.exp(np.linspace(-0.6, 0.4, 15))
    fit = fit_svi_strike_slice(K, svi_iv(K, true, F, T), F, T)
    assert fit.a < 0
    assert fit.max_abs_error_bp < 0.01
    assert not any(fit.at_bound.values())
    for p, v in true.items():
        assert fit[p] == pytest.approx(v, abs=1e-6)
    assert fit.min_variance >= 0


def test_robust_loss_scale_matches_vol_residuals():
    """VM-07 / analytics-lab-V2: Huber/soft-L1 at f_scale = 1 was plain L2;
    with f_scale in vol units a 15-vol-point stale quote is down-weighted."""

    rng = np.random.default_rng(1)
    F, T = 100.0, 0.25
    true = {"a": 0.0025, "b": 0.025, "rho": -0.5, "m": 0.0, "sigma": 0.15}
    K = F * np.exp(np.linspace(-0.3, 0.2, 15))
    clean = svi_iv(K, true, F, T)
    noisy = clean + 0.001 * rng.standard_normal(K.size)
    noisy[2] += 0.15
    good = np.arange(K.size) != 2
    l2 = fit_svi_strike_slice(K, noisy, F, T, loss="linear")
    robust = fit_svi_strike_slice(K, noisy, F, T, loss="soft_l1", f_scale=0.003)
    err_l2 = np.max(np.abs(l2.vol(K[good]) - clean[good])) * 1e4
    err_rb = np.max(np.abs(robust.vol(K[good]) - clean[good])) * 1e4
    assert err_rb < 30 and err_l2 > 5 * err_rb
    assert robust.n_downweighted >= 1 and l2.n_downweighted == 0


def test_enforce_no_arbitrage_repairs_an_arbitrageable_fit():
    K = np.exp(np.linspace(-1.0, 1.4, 25))
    vols = svi_iv(K, VOGT, 1.0, 1.0)
    free = fit_svi_strike_slice(K, vols, 1.0, 1.0, loss="linear")
    assert not free.butterfly.ok and not free.arbitrage_free and not free.penalised
    fixed = fit_svi_strike_slice(K, vols, 1.0, 1.0, loss="linear", enforce_no_arbitrage=True)
    assert fixed.penalised and fixed.butterfly.ok and fixed.lee["lee_ok"] and fixed.arbitrage_free
    assert fixed.butterfly.min_g >= 0
    assert fixed.rmse_bp < 100  # the price of removing the arbitrage, reported, not hidden
    clean_K = 100 * np.exp(np.linspace(-0.6, 0.4, 15))
    untouched = fit_svi_strike_slice(clean_K, svi_iv(clean_K, CLEAN, 100, 1.0), 100, 1.0, enforce_no_arbitrage=True)
    assert not untouched.penalised and untouched.rmse_bp < 1e-3


def test_fit_reports_diagnostics_and_weights():
    F, T = 100.0, 0.5
    K = F * np.exp(np.linspace(-0.4, 0.3, 13))
    vols = svi_iv(K, {k: v * 0.5 if k in ("a", "b") else v for k, v in CLEAN.items()}, F, T)
    fit = fit_svi_strike_slice(K, vols, F, T, weights="vega")
    assert fit.weights.mean() == pytest.approx(1.0) and fit.weights.argmax() not in (0, K.size - 1)
    assert np.isfinite(fit.jacobian_condition) and fit.jacobian_condition > 1
    assert set(fit.param_se) == {"min_variance", "b", "rho", "m", "sigma"}
    assert fit.summary()["butterfly_ok"] is True
    assert dict(fit) == fit.params and len(fit) == 5


def test_too_few_points_and_bad_inputs_raise():
    K = np.linspace(90, 110, 5)
    with pytest.raises(ValueError, match="need at least 6"):
        fit_svi_strike_slice(K, np.full(5, 0.2), 100.0, 1.0)  # 5 points for 5 params (vol-models-V3)
    with pytest.raises(ValueError, match="T"):
        fit_svi_strike_slice(np.linspace(90, 110, 9), np.full(9, 0.2), 100.0, None)
    with pytest.raises(ValueError, match="F"):
        fit_svi_strike_slice(np.linspace(90, 110, 9), np.full(9, 0.2), None, 1.0)
    with pytest.raises(ValueError, match="T .* is required"):
        svi_iv(100.0, CLEAN, 100.0)
    with pytest.raises(ValueError, match="F .* is required"):
        svi_iv(100.0, CLEAN, None, 1.0)
    with pytest.raises(ValueError, match="negative minimum total variance"):
        svi_iv(100.0, dict(CLEAN, a=-0.2), 100.0, 1.0)
    with pytest.raises(ValueError, match="rho"):
        svi_iv(100.0, dict(CLEAN, rho=1.0), 100.0, 1.0)
    nan_quote = np.r_[svi_iv(np.linspace(80, 120, 9), CLEAN, 100.0, 1.0), np.nan]
    fit = fit_svi_strike_slice(np.r_[np.linspace(80, 120, 9), 125.0], nan_quote, 100.0, 1.0)
    assert fit.n_dropped == 1 and fit.n_points == 9


def test_multi_start_is_never_worse_than_a_single_start():
    """analytics-lab-V2: one fixed start risked a local minimum; the fit now
    keeps the best of several quasi-explicit starts."""

    rng = np.random.default_rng(11)
    F, T = 100.0, 0.1
    K = F * np.exp(np.linspace(-0.25, 0.15, 21))
    true = {"a": -0.001, "b": 0.03, "rho": -0.8, "m": 0.03, "sigma": 0.08}
    vols = svi_iv(K, true, F, T) + 0.002 * rng.standard_normal(K.size)
    one = fit_svi_strike_slice(K, vols, F, T, loss="linear", n_starts=1)
    many = fit_svi_strike_slice(K, vols, F, T, loss="linear", n_starts=6)
    assert many.n_starts > one.n_starts == 1
    assert many.rmse <= one.rmse * (1 + 1e-9)


def test_short_dated_small_b_is_not_flagged_at_its_bound():
    """Review: b = 0.002 on a 1-day slice was flagged 'at bound' because the
    tolerance was 0.1% of the [0, 4] box (0.004)."""

    T = 1 / 365
    true = {"a": 0.15**2 * T * 0.5, "b": 0.002, "rho": -0.6, "m": 0.002, "sigma": 0.004}
    F = 765.0
    K = F * np.exp(np.linspace(-0.02, 0.015, 25))
    fit = fit_svi_strike_slice(K, svi_iv(K, true, F, T), F, T, loss="linear")
    assert fit.b == pytest.approx(0.002, rel=1e-4)
    assert not fit.at_bound["b"] and not any(fit.at_bound.values())
    flat = fit_svi_strike_slice(K, np.full(K.size, 0.15), F, T, loss="linear")
    assert flat.at_bound["b"]  # a flat smile really has b at 0


def test_ssvi_slice_start_is_butterfly_free_by_construction():
    """The enforcement fallback/start: SSVI slices with theta phi (1 + |rho|)
    < 4 and theta phi^2 (1 + |rho|) <= 4 have g >= 0 (Gatheral-Jacquier
    2014, Theorem 4.2), whatever the data."""

    from derivx.vol.svi import SVI_PARAMS, _fit_ssvi_slice, _x_to_raw

    rng = np.random.default_rng(5)
    for _ in range(20):
        T = rng.uniform(0.01, 3.0)
        k = np.sort(rng.uniform(-0.6, 0.4, 15))
        vols = np.abs(0.2 + 0.3 * rng.standard_normal(15) * np.abs(k)) + 0.05
        x, _ = _fit_ssvi_slice(k, vols, T, np.ones(15), "linear", None)
        p = dict(zip(SVI_PARAMS, (float(v) for v in _x_to_raw(x))))
        assert check_butterfly(p).ok and lee_wing_check(p)["lee_ok"]


def test_enforcement_reports_how_the_arbitrage_was_removed():
    K = np.exp(np.linspace(-1.0, 1.4, 25))
    vols = svi_iv(K, VOGT, 1.0, 1.0)
    fixed = fit_svi_strike_slice(K, vols, 1.0, 1.0, loss="linear", enforce_no_arbitrage=True)
    assert fixed.arbitrage_repair in ("constrained", "ssvi_slice") and fixed.success
    assert fixed.rmse_bp < 70  # Vogt's smile is itself arbitrageable: some error is the price of removing it
    free = fit_svi_strike_slice(K, vols, 1.0, 1.0, loss="linear")
    assert free.arbitrage_repair == "none" and not free.penalised
    with pytest.raises(ValueError, match="loss"):
        fit_svi_strike_slice(K, vols, 1.0, 1.0, loss="l1")


def test_constrained_refit_silences_only_scipys_bound_clip_notice(monkeypatch):
    """UIE-18: CI turns RuntimeWarnings into errors. SciPy 1.11's SLSQP warns
    when a step lands a rounding error past a bound (it clips the point back
    first), which failed the walk-forward on the oldest supported versions.
    That notice is silenced inside the refit; any other warning still
    surfaces."""

    import warnings

    import derivx.vol.svi as svi

    K = np.exp(np.linspace(-1.0, 1.4, 25))
    vols = svi_iv(K, VOGT, 1.0, 1.0)
    expected = fit_svi_strike_slice(K, vols, 1.0, 1.0, loss="linear", enforce_no_arbitrage=True)
    real, calls = svi.minimize, []

    def warning_minimize(message):
        def run(*args, **kwargs):
            calls.append(message)
            warnings.warn(message, RuntimeWarning)
            return real(*args, **kwargs)

        return run

    notice = "Values in x were outside bounds during a minimize step, clipping to bounds"
    monkeypatch.setattr(svi, "minimize", warning_minimize(notice))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fixed = fit_svi_strike_slice(K, vols, 1.0, 1.0, loss="linear", enforce_no_arbitrage=True)
    assert calls and fixed.params == expected.params and fixed.arbitrage_repair == expected.arbitrage_repair
    monkeypatch.setattr(svi, "minimize", warning_minimize("overflow encountered in the objective"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="overflow"):
            fit_svi_strike_slice(K, vols, 1.0, 1.0, loss="linear", enforce_no_arbitrage=True)
