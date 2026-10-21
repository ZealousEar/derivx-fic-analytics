"""Production SVI calibration: grouping, maturity handling, hygiene,
maturity-aware bands, real no-arbitrage diagnostics, conditioning and the
out-of-sample spline baseline."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from _vol_chains import SURFACE, synth_chain, true_vol

from derivx.calibration.production import (
    CalibratedSlice,
    benchmark_vs_baseline,
    calibrate_chain,
    calibrate_with_constraints,
)
from derivx.vol.svi import svi_iv


def _synthetic_chain(trade_date: dt.date, days_to_expiry: int = 30) -> pd.DataFrame:
    expiry = trade_date + dt.timedelta(days=days_to_expiry)
    strikes = np.linspace(80, 120, 9)
    forward = 100.0
    params = {"a": 0.02, "b": 0.3, "rho": -0.2, "m": 0.0, "s": 0.4}
    ivs = svi_iv(strikes, params, F=forward, T=1.0)  # params: total variance at T = 1
    data = {
        "trade_date": pd.Timestamp(trade_date),
        "expiry": pd.Timestamp(expiry),
        "strike": strikes,
        "implied_vol": ivs + 0.0005 * np.random.default_rng(0).standard_normal(len(strikes)),
        "underlying_price": np.full_like(strikes, forward),
        "bid": np.full_like(strikes, 1.0),
        "ask": np.full_like(strikes, 1.2),
        "volume": np.full_like(strikes, 50),
        "open_interest": np.full_like(strikes, 100),
        "option_type": np.where(strikes < forward, "put", "call"),
    }
    df = pd.DataFrame(data)
    df["moneyness"] = df["strike"] / forward
    return df


def test_calibrate_with_constraints_recovers_params():
    trade_date = dt.date(2024, 1, 2)
    chain = _synthetic_chain(trade_date)
    # quoted vols: the constant bid/ask of this helper are placeholders, not prices
    results = calibrate_with_constraints(chain, enforce_arbitrage=False, hygiene={"recompute_iv": False})
    assert len(results) == 1
    result = results[0]
    assert result.n_points == len(chain)
    assert result.tau == pytest.approx(30 / 365) and result.forward == pytest.approx(100.0)
    assert result.rmse < 0.0005  # the 5 vol bp quote noise (was < 100 bp)
    assert "condition_number" in result.diagnostics
    # the fitted smile is the true one to within the noise (single-slice SVI parameters themselves are not
    # identified that precisely; the exact-quote test below checks them)
    err = result.iv(chain["strike"].to_numpy()) - _true_vols(chain)
    assert np.sqrt(np.mean(err**2)) < 0.0005 and np.abs(err).max() < 0.001


def _true_vols(chain: pd.DataFrame) -> np.ndarray:
    return svi_iv(chain["strike"].to_numpy(), {"a": 0.02, "b": 0.3, "rho": -0.2, "m": 0.0, "s": 0.4}, F=100.0, T=1.0)


def test_exact_quotes_recover_total_variance_parameters_at_the_slice_maturity():
    chain = _synthetic_chain(dt.date(2024, 1, 2))
    chain["implied_vol"] = _true_vols(chain)
    result = calibrate_with_constraints(chain, enforce_arbitrage=False, hygiene={"recompute_iv": False})[0]
    tau = 30 / 365  # the helper's params are total variance at T = 1, so a and b scale by tau
    expected = {"a": 0.02 * tau, "b": 0.3 * tau, "rho": -0.2, "m": 0.0, "sigma": 0.4}
    for name, value in expected.items():
        assert result.params[name] == pytest.approx(value, abs=1e-6), name
    assert result.rmse < 1e-8


def test_benchmark_vs_baseline_reports_metrics():
    trade_date = dt.date(2024, 1, 2)
    chain = _synthetic_chain(trade_date)
    result = calibrate_with_constraints(chain, enforce_arbitrage=False, hygiene={"recompute_iv": False})[0]
    metrics = benchmark_vs_baseline(result, chain)
    assert {"rmse_svi_in_sample", "rmse_svi_cv", "rmse_spline_cv", "relative_improvement"} <= set(metrics)
    assert metrics["rmse_svi_in_sample"] == result.rmse and metrics["n_scored"] == len(chain) - 2  # no end points
    assert metrics["relative_improvement"] == pytest.approx(1 - metrics["rmse_svi_cv"] / metrics["rmse_spline_cv"])
    # held-out errors exceed the in-sample fit and are of the order of the 5 vol bp quote noise
    assert metrics["rmse_svi_cv"] > metrics["rmse_svi_in_sample"]
    for key in ("rmse_svi_cv", "rmse_spline_cv"):
        assert 0.0001 < metrics[key] < 0.001, key


def test_on_exact_svi_quotes_svi_predicts_held_out_strikes_and_the_spline_does_not():
    chain = _synthetic_chain(dt.date(2024, 1, 2))
    chain["implied_vol"] = _true_vols(chain)
    result = calibrate_with_constraints(chain, enforce_arbitrage=False, hygiene={"recompute_iv": False})[0]
    metrics = benchmark_vs_baseline(result)
    assert metrics["rmse_svi_cv"] < 1e-8  # the model is right, so unseen strikes are predicted exactly
    assert metrics["rmse_spline_cv"] > 5e-5  # the spline interpolates across each held-out strike with error
    assert metrics["relative_improvement"] > 0.999


@pytest.fixture(scope="module")
def run():
    chain = synth_chain(days=(0, 7, 30, 91, 182, 365), iv_noise=0.001, seed=4)
    return calibrate_chain(chain, rate=0.04, div_yield=0.012)


def test_chain_calibration_uses_parity_forwards_and_skips_expired(run):
    """VM-05 / VM-13 / AL-08: the same-day expiry is skipped with a reason;
    each slice uses its own parity forward and fits the true smile."""

    assert [s.reason.split(" ")[0] for s in run.skipped] == ["expired"]
    assert len(run.slices) == 5
    for s in run.slices:
        true_F = 580.0 * np.exp((0.04 - 0.012) * s.tau)
        assert s.forward == pytest.approx(true_F, rel=1e-5)
        assert "parity" in s.diagnostics["forward_source"]
        err = np.abs(s.iv(s.strikes) - true_vol(s.strikes, s.forward, s.tau))
        assert err.max() < 0.003  # 30 vol bp against 10 vol bp quote noise
        assert s.diagnostics["removed"]["itm"] > 10
        assert s.n_points >= 8


def test_real_no_arbitrage_diagnostics_are_reported(run):
    for s in run.slices:
        d = s.diagnostics
        assert d["butterfly_ok"] and d["butterfly_min_g"] > 0 and d["lee_ok"]
        assert np.isfinite(d["condition_number"])
    assert run.slices[0].diagnostics["calendar_ok"] is None  # first expiry: nothing before it
    assert all(s.diagnostics["calendar_ok"] for s in run.slices[1:])
    assert len(run.calendar) == 4 and run.calendar["calendar_ok"].all()
    table = run.table()
    assert list(table["points"]) == [s.n_points for s in run.slices]
    assert np.isfinite(table["spline_cv_rmse_bp"]).all() and (table["spline_cv_rmse_bp"] > 0).all()


def test_butterfly_violation_is_reported_and_enforcement_removes_it():
    """VM-04 / AL-16: a slice quoting Vogt's arbitrageable smile."""

    vogt = {"a": -0.0410, "b": 0.1331, "rho": 0.3060, "m": 0.3586, "sigma": 0.4153}
    K = 100 * np.exp(np.linspace(-1.0, 1.4, 25))
    chain = pd.DataFrame(
        {
            "trade_date": pd.Timestamp("2026-10-19"),
            "expiry": pd.Timestamp("2027-10-19"),
            "strike": K,
            "implied_vol": svi_iv(K, vogt, 100.0, 1.0),
            "underlying_price": 100.0,
        }
    )
    free = calibrate_with_constraints(chain, enforce_arbitrage=False, std_band=None, weights=None, loss="linear")[0]
    assert free.diagnostics["butterfly_ok"] is False and free.diagnostics["butterfly_min_g"] < 0
    assert free.diagnostics["butterfly_negative_k"]
    fixed = calibrate_with_constraints(chain, enforce_arbitrage=True, std_band=None, weights=None, loss="linear")[0]
    assert fixed.diagnostics["butterfly_ok"] and fixed.diagnostics["arbitrage_penalised"]


def test_calendar_violation_between_expiries_is_reported():
    """VM-12: a 3M steep-skew slice whose put wing sits above a flat 6M slice."""

    short = {"a": 0.005, "b": 0.12, "rho": -0.9, "m": 0.02, "sigma": 0.05}
    long = {"a": 0.03, "b": 0.03, "rho": 0.0, "m": 0.0, "sigma": 0.1}
    rows = []
    for days, p in ((91, short), (182, long)):
        T = days / 365
        K = 100 * np.exp(np.linspace(-0.4, 0.3, 15))
        rows.append(
            pd.DataFrame(
                {
                    "trade_date": pd.Timestamp("2026-10-19"),
                    "expiry": pd.Timestamp("2026-10-19") + pd.Timedelta(days=days),
                    "strike": K,
                    "implied_vol": svi_iv(K, p, 100.0, T),
                    "underlying_price": 100.0,
                }
            )
        )
    result = calibrate_chain(pd.concat(rows), std_band=None, enforce_arbitrage=False)
    assert len(result.slices) == 2
    assert result.slices[1].diagnostics["calendar_ok"] is False
    assert result.slices[1].diagnostics["calendar_min_dw"] < 0
    assert not result.calendar["calendar_ok"].iloc[0]


def test_groups_by_trade_date_and_expiry():
    """vol-models-V4: two trade dates with the same expiry are two slices,
    each with its own spot and maturity."""

    day1 = synth_chain(days=(30,), spot=100.0)
    day2 = synth_chain(days=(29,), spot=110.0, trade_date=pd.Timestamp("2026-10-20"))
    out = calibrate_with_constraints(pd.concat([day1, day2]), rate=0.04, div_yield=0.012)
    assert len(out) == 2
    assert {s.trade_date for s in out} == {pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-20")}
    assert {s.expiry for s in out} == {pd.Timestamp("2026-11-18")}
    taus = sorted(s.tau for s in out)
    assert taus == pytest.approx([29 / 365, 30 / 365])
    spots = sorted(s.diagnostics["spot"] for s in out)
    assert spots == [100.0, 110.0]


def test_missing_trade_date_is_a_clear_error_not_a_crash():
    """VM-13: chain.get('trade_date', Timestamp).iloc raised AttributeError."""

    chain = synth_chain(days=(30,)).drop(columns=["trade_date"])
    with pytest.raises(ValueError, match="trade_date"):
        calibrate_with_constraints(chain)
    assert len(calibrate_with_constraints(chain, trade_date="2026-10-19")) == 1


def test_min_points_must_exceed_the_svi_parameters_and_sparse_slices_are_skipped():
    """vol-models-V3: 5 points for 5 parameters interpolated exactly."""

    chain = synth_chain(days=(30,), n_strikes=7, width_sd=(-1.5, 1.5))
    with pytest.raises(ValueError, match="min_points"):
        calibrate_chain(chain, min_points=5)
    result = calibrate_chain(chain, min_points=8)
    assert not result.slices and "need 8" in result.skipped[0].reason


def test_jacobian_condition_depends_on_the_quotes():
    """VM-10: the old 'Cond #' was cond([1, k, k^2]) of the strike grid only."""

    base = synth_chain(days=(91,))
    noisy = synth_chain(days=(91,), iv_noise=0.02, seed=9)
    quoted = {"recompute_iv": False}  # the noise is in the quoted vols
    c1 = calibrate_with_constraints(base, hygiene=quoted)[0].diagnostics["condition_number"]
    c2 = calibrate_with_constraints(noisy, hygiene=quoted)[0].diagnostics["condition_number"]
    assert np.isfinite(c1) and np.isfinite(c2) and abs(np.log(c1 / c2)) > 0.05


def test_itm_junk_does_not_move_the_fit():
    """VM-06: garbage ITM IVs used to drag the smile by ~280 vol bp."""

    chain = synth_chain(days=(91,))
    T = 91 / 365
    F = 580.0 * np.exp(0.028 * T)
    itm = ((chain["option_type"] == "call") & (chain["strike"] < F)) | (
        (chain["option_type"] == "put") & (chain["strike"] > F)
    )
    chain.loc[itm, "implied_vol"] += np.random.default_rng(0).normal(0, 0.03, int(itm.sum()))
    s = calibrate_with_constraints(chain, rate=0.04, div_yield=0.012, hygiene={"recompute_iv": False})[0]
    assert s.diagnostics["iv_source"] == "quoted"
    assert np.max(np.abs(s.iv(s.strikes) - true_vol(s.strikes, s.forward, T))) < 1e-3


def test_spline_baseline_is_out_of_sample_and_finite():
    """AL-07 / VM-09: the spline used to be scored on its own knots (RMSE 0,
    improvement -3.7e15) or failed on duplicate strikes (NaN)."""

    chain = synth_chain(days=(30,), iv_noise=0.001, seed=2)
    s = calibrate_with_constraints(chain, rate=0.04, div_yield=0.012, hygiene={"recompute_iv": False})[0]
    b = s.benchmark
    assert b["n_scored"] == s.n_points - 2
    assert 0 < b["rmse_spline_cv"] < 0.01 and 0 < b["rmse_svi_cv"] < 0.01
    assert -1 < b["relative_improvement"] < 1
    assert b["rmse_svi_cv"] >= b["rmse_svi_in_sample"] * 0.8


def test_moneyness_band_is_on_forward_moneyness():
    chain = synth_chain(days=(182,), width_sd=(-3, 3))
    s = calibrate_with_constraints(chain, moneyness_band=(0.9, 1.1), rate=0.04, div_yield=0.012)[0]
    m = s.strikes / s.forward
    assert m.min() >= 0.9 and m.max() <= 1.1
    assert isinstance(s, CalibratedSlice) and s.clean is not None and s.fit is not None
    with pytest.raises(ValueError, match="moneyness_band"):
        calibrate_with_constraints(chain, moneyness_band=(1.1, 0.9))


def test_surface_truth_is_arbitrage_free():
    report = SURFACE.arbitrage_report()
    assert report["butterfly_ok"] and report["calendar_ok"] and report["eta_condition_ok"]


# --- review fixes -------------------------------------------------------------------------------


def test_forward_is_interpolated_when_parity_fails_for_an_expiry():
    """On a real SPY chain after liquidity filters, the long expiries lost
    their call/put pairs and fell back to spot exp((r - q) T) = spot (no rate
    input), 4-8% below the market forward, with spurious calendar
    violations. The forward is now interpolated from the parity forwards of
    the other expiries."""

    chain = synth_chain(days=(30, 91, 182, 365))
    T = 365 / 365
    F_true = 580.0 * np.exp((0.04 - 0.012) * T)
    last = pd.to_datetime(chain["expiry"]) == chain["expiry"].max()
    itm = ((chain["option_type"] == "call") & (chain["strike"] < F_true)) | (
        (chain["option_type"] == "put") & (chain["strike"] > F_true)
    )
    chain = chain[~(last & itm)]  # only OTM quotes left: no pair for parity
    run = calibrate_chain(chain, cv_folds=0)  # rate unknown: 0, as in the app
    s = run.slices[-1]
    assert s.diagnostics["forward_method"] == "interpolated" and "parity forward" in s.diagnostics["forward_source"]
    assert s.forward == pytest.approx(F_true, rel=2e-4)  # the spot is 2.8% away
    assert all(x.diagnostics["forward_method"] == "parity" for x in run.slices[:-1])
    assert run.calendar["calendar_ok"].all()


def test_pipeline_fits_mid_vols_by_default():
    chain = synth_chain(days=(91,))
    chain["implied_vol"] += np.where(chain["option_type"] == "call", 0.02, -0.02)  # quoted vols jump at F
    s = calibrate_with_constraints(chain, rate=0.04, div_yield=0.012)[0]
    assert s.diagnostics["iv_source"] == "mid" and s.diagnostics["rate"] == 0.04
    assert np.max(np.abs(s.iv(s.strikes) - true_vol(s.strikes, s.forward, s.tau))) < 5e-4
    assert run_table_has_sources(calibrate_chain(chain, rate=0.04, div_yield=0.012, cv_folds=0).table())


def run_table_has_sources(table: pd.DataFrame) -> bool:
    return {"forward_method", "iv_source"} <= set(table.columns) and (table["iv_source"] == "mid").all()


def _spy_fixture():
    import json
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "spy_2026-10-20_otm_slices.json"
    return json.loads(path.read_text())["slices"]


def test_enforcement_on_real_spy_slices_is_arbitrage_free_and_costs_little():
    """Real SPY smiles (fixture) whose free SVI fit has butterfly arbitrage.
    The old penalised refit turned an 11.7 vol bp fit of the 2026-11-13
    slice into 35.6 vol bp and still left g < 0; in cross-validation some
    folds ended near 450 vol bp."""

    from derivx.vol.svi import fit_svi_strike_slice

    for sl in _spy_fixture():
        K, iv = np.array(sl["strikes"]), np.array(sl["implied_vols"])
        free = fit_svi_strike_slice(K, iv, sl["F"], sl["T"])
        assert not free.arbitrage_free, sl["expiry"]
        fixed = fit_svi_strike_slice(K, iv, sl["F"], sl["T"], enforce_no_arbitrage=True)
        assert fixed.arbitrage_free and fixed.butterfly.min_g >= 0, sl["expiry"]
        assert fixed.arbitrage_repair in ("constrained", "ssvi_slice") and fixed.penalised
        assert fixed.rmse_bp < free.rmse_bp + 1.5, (sl["expiry"], free.rmse_bp, fixed.rmse_bp)


def test_cross_validation_with_enforcement_stays_close_to_in_sample_on_real_slice():
    sl = next(s for s in _spy_fixture() if s["expiry"] == "2026-11-13")
    K = np.array(sl["strikes"])
    chain = pd.DataFrame(
        {
            "trade_date": pd.Timestamp("2026-10-20"),
            "expiry": pd.Timestamp(sl["expiry"]),
            "strike": K,
            "implied_vol": sl["implied_vols"],
            "underlying_price": sl["F"],  # no option_type: the carry forward (r = q = 0) is this F
        }
    )
    s = calibrate_with_constraints(chain, std_band=None, enforce_arbitrage=True)[0]
    assert s.forward == pytest.approx(sl["F"]) and s.diagnostics["butterfly_ok"]
    b = s.benchmark
    assert b["rmse_svi_cv"] < 1.5 * b["rmse_svi_in_sample"]  # was 347 vs 35.6 vol bp


def test_cross_validation_refits_never_start_from_the_full_sample_fit(monkeypatch):
    import derivx.calibration.production as prod

    calls = []
    real = prod.fit_svi_strike_slice

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(prod, "fit_svi_strike_slice", spy)
    calibrate_chain(synth_chain(days=(30,)), rate=0.04, div_yield=0.012, cv_folds=3)
    assert len(calls) == 4  # the fit plus one refit per fold
    assert all(c.get("x0") is None for c in calls[1:]) and all(c["n_starts"] == prod.CV_STARTS for c in calls[1:])
