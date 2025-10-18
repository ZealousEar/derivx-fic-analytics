import numpy as np

from derivx.vol.svi import fit_svi_strike_slice, svi_iv, basic_no_arb_checks


def test_svi_fit_on_synthetic_slice():
    rng = np.random.default_rng(2)
    F = 100.0
    Ks = np.linspace(60, 140, 15)
    # True params (a,b,rho,m,s)
    true = {"a": 0.02, "b": 0.3, "rho": -0.2, "m": 0.0, "s": 0.4}
    ivs = np.array([svi_iv(K, true, F) for K in Ks])
    # Add small noise
    ivs_noisy = ivs + 0.0005 * rng.standard_normal(size=ivs.shape)

    est = fit_svi_strike_slice(Ks, ivs_noisy, F)
    ivs_fit = np.array([svi_iv(K, est, F) for K in Ks])
    rmse = float(np.sqrt(np.mean((ivs_fit - ivs) ** 2)))
    assert rmse < 0.005  # <= 0.5 vol points


def test_svi_no_arb_basic_sanity():
    params_list = [
        {"a": 0.02, "b": 0.2, "rho": 0.0, "m": 0.0, "s": 0.2},
        {"a": 0.01, "b": 0.1, "rho": -0.5, "m": 0.1, "s": 0.3},
    ]
    diag = basic_no_arb_checks(params_list)
    assert diag["viol_b"] == 0
    assert diag["viol_rho"] == 0


