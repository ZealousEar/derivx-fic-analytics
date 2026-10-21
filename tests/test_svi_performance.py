"""Speed of the production SVI calibration without loss of accuracy (B3):
the analytic constraint gradient, the equivalent plain least-squares form of
the robust loss, the box-wide start points, the enforcement shortcut, the
progress callback, and on the committed real SPY sample the results of the
previous (slower) solver to 0.001 vol bp in sample, with a generous time
bound on its slowest slice."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares

import derivx.vol.svi as svi
from derivx.calibration.production import calibrate_chain
from derivx.data import rates_history as rh
from derivx.data.market_data import apply_liquidity_filters
from derivx.data.samples import load_sample_chain
from derivx.vol.discounting import treasury_discounting

# Committed sample, app defaults (filters 10 / 0.10 / 0.40, Treasury curve, 3 sd and 0.70-1.30 K/F bands,
# 5-fold CV, enforcement). In-sample and CV RMSE (vol bp) of the solver before B3 (scipy TRF from four
# near-quote starts, SLSQP from the SSVI slice with finite-difference constraint gradients), 2026-10-21.
PREVIOUS = {
    "2026-10-22": (5.9581, 6.0027),  # 2 days, 34 quotes: the slice that took 13-20 s
    "2026-10-30": (8.8420, 7.9620),
    "2026-11-20": (4.5421, 4.2147),
    "2027-06-17": (7.3010, 7.3296),
}


@pytest.fixture(scope="module")
def spy():
    ds = load_sample_chain()
    chain = apply_liquidity_filters(ds.single(), min_volume=10, max_spread=0.40, min_mid=0.10)
    return chain, treasury_discounting(rh.bundled_store(), ds.trade_dates, mode="curve", rate=0.0)


def _calibrate(spy, expiries, **kw):
    chain, disc = spy
    sub = chain[pd.to_datetime(chain["expiry"]).dt.strftime("%Y-%m-%d").isin(expiries)]
    return calibrate_chain(
        sub, enforce_arbitrage=True, std_band=3.0, moneyness_band=(0.7, 1.3), discount=disc, min_days=2.0, **kw
    )


def test_constraint_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    k = np.linspace(-0.6, 0.5, 57)
    for _ in range(20):
        x = np.array([rng.uniform(1e-3, 0.05), rng.uniform(0.01, 0.6), rng.uniform(-0.9, 0.9), 0.0, 0.0])
        x[3], x[4] = rng.uniform(-0.2, 0.2), rng.uniform(0.02, 0.5)
        g, grad = svi._g_and_grad(k, x)
        np.testing.assert_allclose(g, svi._g_raw(k, *svi._x_to_raw(x)), rtol=1e-12, atol=1e-14)
        for j in range(5):
            h = 1e-6 * max(abs(x[j]), 1e-3)
            up, dn = x.copy(), x.copy()
            up[j] += h
            dn[j] -= h
            fd = (svi._g_and_grad(k, up)[0] - svi._g_and_grad(k, dn)[0]) / (2 * h)
            np.testing.assert_allclose(grad[:, j], fd, rtol=1e-5, atol=1e-6 * np.max(np.abs(fd)) + 1e-9)


def test_plain_residuals_reproduce_the_robust_objective_and_its_jacobian():
    """r~ = r sqrt(rho(z) / z) makes 0.5 sum r~^2 equal to scipy's soft-L1
    cost, so MINPACK minimises exactly the same objective."""

    rng = np.random.default_rng(3)
    r0 = rng.normal(0.0, 0.01, 30)
    A = rng.normal(size=(30, 5))

    def resid(x):
        return r0 + A @ x

    def jac(x):
        return A.copy()  # SciPy <= 1.15 rescales the returned Jacobian in place for robust losses

    fun, grad = svi._robust_residuals(resid, jac, "soft_l1", 0.005)
    x = rng.normal(0.0, 1e-3, 5)
    scipy_cost = least_squares(resid, x, jac=jac, loss="soft_l1", f_scale=0.005, max_nfev=1).cost
    start_cost = 0.5 * 0.005**2 * np.sum(svi._loss_rho(resid(x) ** 2 / 0.005**2, "soft_l1")[0])
    assert 0.5 * np.sum(fun(x) ** 2) == pytest.approx(start_cost, rel=1e-12)
    assert scipy_cost <= start_cost  # scipy's cost is the same objective (after one step)
    fd = np.column_stack([(fun(x + 1e-7 * e) - fun(x - 1e-7 * e)) / 2e-7 for e in np.eye(5)])
    np.testing.assert_allclose(grad(x), fd, rtol=1e-6, atol=1e-9)
    with pytest.raises(ValueError, match="soft_l1"):
        svi._robust_residuals(resid, jac, "cauchy", 0.005)


def test_box_wide_starts_reach_the_far_optimum_of_a_v_shaped_smile(spy):
    """On the 2-day SPY smile the free optimum is far out (m at its bound,
    sigma ~ 1); from near-quote starts TRF crawled there in 1,000+ steps."""

    s = _calibrate(spy, ["2026-10-22"], cv_folds=0).slices[0]
    K, v, F, T = s.fit.strikes, s.fit.market_vols, s.forward, s.tau
    free = svi.fit_svi_strike_slice(K, v, F, T)
    assert free.at_bound["m"] and free.rmse_bp <= 5.8881  # the previous solver: 5.88802 vol bp
    assert free.nfev < 400


def test_enforcement_skips_free_starts_only_when_the_free_optimum_is_clearly_arbitrageable(spy, monkeypatch):
    s = _calibrate(spy, ["2026-10-22"], cv_folds=0).slices[0]
    K, v, F, T = s.fit.strikes, s.fit.market_vols, s.forward, s.tau
    quick = svi.fit_svi_strike_slice(K, v, F, T, enforce_no_arbitrage=True)
    assert quick.n_starts == 1 and quick.arbitrage_repair == "constrained" and quick.arbitrage_free
    monkeypatch.setattr(svi, "CERTIFICATE_GAP", 1.0)  # never skip: the free starts run as before
    full = svi.fit_svi_strike_slice(K, v, F, T, enforce_no_arbitrage=True)
    assert full.n_starts > 1 and full.arbitrage_repair == "constrained"
    assert quick.rmse_bp == pytest.approx(full.rmse_bp, abs=1e-3)  # the same constrained optimum


def test_the_slsqp_constraint_grid_stays_under_the_multi_threaded_blas_size():
    """From about 210 constraint rows SciPy's SLSQP runs its subproblems on
    multi-threaded BLAS, 2-10x slower on this small problem (the 396-point
    grid used before doubled a whole-sample calibration's wall time)."""

    rng = np.random.default_rng(11)
    for _ in range(50):
        xs = [np.array([1e-3, 0.1, rng.uniform(-0.9, 0.9), rng.uniform(-1, 1), rng.uniform(1e-3, 1)]) for _ in "ab"]
        grid = svi._constraint_grid(xs, (rng.uniform(-3, -0.1), rng.uniform(0.1, 2)))
        assert len(grid) <= 136 and np.all(np.diff(grid) > 0)


def test_calibrate_chain_reports_progress(spy):
    seen = []
    result = _calibrate(spy, ["2026-11-20", "2026-12-18"], cv_folds=0, progress=lambda *a: seen.append(a))
    assert [d for d, _, _ in seen] == [0, 1, 2] and {t for _, t, _ in seen} == {2}
    assert [lab for _, _, lab in seen] == ["2026-11-20", "2026-12-18", "done"] and len(result.slices) == 2


def test_real_sample_keeps_the_previous_accuracy_and_its_slowest_slice_is_fast(spy):
    """The same fits as the slower solver (0.001 vol bp in sample, 0.01 out
    of sample), arbitrage-free; the 2-day slice with 5-fold CV and
    enforcement took 13-20 s and now about 1 s (bound: 8 s, for slow CI)."""

    t0 = time.perf_counter()
    first = _calibrate(spy, ["2026-10-22"], cv_folds=5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 8.0, f"2-day slice took {elapsed:.1f} s"
    result = _calibrate(spy, list(PREVIOUS), cv_folds=5)
    got = {str(s.expiry.date()): s for s in result.slices}
    assert set(got) == set(PREVIOUS)
    assert got["2026-10-22"].params == first.slices[0].params  # deterministic
    for expiry, (rmse_bp, cv_bp) in PREVIOUS.items():
        s = got[expiry]
        assert s.diagnostics["rmse_bp"] == pytest.approx(rmse_bp, abs=1e-3), expiry
        assert s.benchmark["rmse_svi_cv"] * 1e4 == pytest.approx(cv_bp, abs=1e-2), expiry
        assert s.diagnostics["butterfly_ok"] and s.diagnostics["lee_ok"], expiry
