"""Walk-forward validation (AL-03, AL-05, AL-06, AL-17, analytics-lab-V3,
AL-19): pairing by option type, real horizons, a calibration window that
is used, benchmarks and the noise floor, and constant-maturity regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from derivx.data.synthetic import synthetic_history
from derivx.validation.backtest import (
    BENCHMARK,
    PREDICTORS,
    WalkForwardSettings,
    constant_maturity_atm_vol,
    diebold_mariano,
    regime_buckets,
    run_walk_forward,
    walk_forward_test,
    wilson_interval,
)
from derivx.vol.quotes import clean_option_slice

EXPIRY = pd.Timestamp("2027-01-15")
STRIKES = np.arange(80.0, 120.01, 2.5)


def _quoted(trade_date: str, spot: float = 100.0, call_gap: float = 0.05, expiry=EXPIRY, level: float = 0.20):
    """Calls and puts at every strike with QUOTED vols only (no bid/ask, so
    nothing is re-inverted): the call vol is ``call_gap`` above the put vol
    at the same strike, as on Yahoo. Forward = spot (no parity prices)."""

    k = np.log(STRIKES / spot)
    put_iv = level - 0.15 * k + 0.4 * k**2
    rows = []
    for typ, iv in (("put", put_iv), ("call", put_iv + call_gap)):
        rows.append(
            pd.DataFrame(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "expiry": expiry,
                    "option_type": typ,
                    "strike": STRIKES,
                    "implied_vol": iv,
                    "underlying_price": spot,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_calls_are_never_scored_against_puts():
    history = {pd.Timestamp(d): _quoted(d) for d in ("2026-10-19", "2026-10-20")}
    res = run_walk_forward(history, enforce_arbitrage=False)
    q = res.quotes
    # one OTM quote per strike (puts below F, calls above, the two averaged at F), matched on (strike, type)
    assert len(q) == STRIKES.size and q["strike"].is_unique
    assert set(q.loc[q["strike"] < 100, "option_type"]) == {"put"}
    assert set(q.loc[q["strike"] > 100, "option_type"]) == {"call"}
    # nothing moved, so 'no change' is exact: a call-vs-put pairing would show the 5 vol pt gap
    assert np.abs(q["err_persistence"]).max() == 0.0
    assert res.summary.set_index("predictor").loc["persistence", "rmse_bp"] == 0.0


def test_a_strike_whose_otm_side_flips_is_not_scored():
    history = {
        pd.Timestamp("2026-10-19"): _quoted("2026-10-19", spot=100.6),
        pd.Timestamp("2026-10-20"): _quoted("2026-10-20", spot=99.4),
    }
    q = run_walk_forward(history, enforce_arbitrage=False).quotes
    assert 100.0 not in set(q["strike"])  # a put on the origin day, a call on the target day
    assert len(q) == STRIKES.size - 1


def test_scores_use_the_stated_horizon_in_trading_days():
    full = synthetic_history(5)
    days = list(full)
    gappy = {d: full[d] for d in days if d != days[2]}  # a missing day
    one = run_walk_forward(gappy, horizon=1, max_days=60)
    assert set(one.slices["horizon_td"]) == {1}
    assert days[1] not in set(one.slices["origin"])
    reason = one.skipped[(one.skipped["origin"] == days[1]) & (one.skipped["stage"] == "horizon")]["reason"].iloc[0]
    assert "no snapshot 1 trading day(s) later" in reason and "2 trading day(s) later" in reason
    two = run_walk_forward(gappy, horizon=2, max_days=60)
    pairs = set(zip(two.slices["origin"], two.slices["target"]))
    assert (days[1], days[3]) in pairs and set(two.slices["horizon_td"]) == {2}


def test_the_calibration_window_changes_the_fit():
    history = synthetic_history(4)
    one = run_walk_forward(history, window=1, max_days=60)
    two = run_walk_forward(history, window=2, max_days=60)
    days = list(history)
    assert days[0] in set(one.slices["origin"]) and days[0] not in set(two.slices["origin"])
    assert "needs 2 snapshots" in two.skipped.loc[two.skipped["origin"] == days[0], "reason"].iloc[0]
    a = one.slices.set_index(["origin", "expiry"])
    b = two.slices.set_index(["origin", "expiry"])
    common = a.index.intersection(b.index)
    assert (a.loc[common, "window_snapshots"] == 1).all() and b.loc[common, "window_snapshots"].isin([1, 2]).all()
    common = common[(b.loc[common, "window_snapshots"] == 2).to_numpy()]  # expiries listed on both pooled days
    assert len(common) > 3
    assert (b.loc[common, "n_calibration"] > a.loc[common, "n_calibration"]).all()
    diff = np.abs(b.loc[common, "rmse_bp_svi_sticky_strike"] - a.loc[common, "rmse_bp_svi_sticky_strike"])
    assert diff.max() > 0.1  # pooled quotes give a different smile
    np.testing.assert_allclose(b.loc[common, "rmse_bp_persistence"], a.loc[common, "rmse_bp_persistence"])


def test_benchmarks_skill_noise_floor_and_win_rate():
    res = run_walk_forward(synthetic_history(5), max_days=60)
    s = res.summary.set_index("predictor")
    assert list(s.index) == list(PREDICTORS)
    for name, bench in BENCHMARK.items():
        assert s.loc[name, "benchmark"] == (bench or "")
        if bench:
            expected = 1.0 - s.loc[name, "rmse_bp"] / s.loc[bench, "rmse_bp"]
            assert s.loc[name, "skill"] == pytest.approx(expected)
            assert 0.0 <= s.loc[name, "win_lo"] <= s.loc[name, "win_rate"] <= s.loc[name, "win_hi"] <= 1.0
            assert s.loc[name, "win_n"] > 100
    # bid/ask in vol of the target quotes: 2% price spreads are tens of vol bp
    assert 1.0 < s.loc["persistence", "noise_bp"] < 200.0
    assert np.isnan(s.loc["persistence", "skill"])
    q = res.quotes
    pooled = np.sqrt(np.mean(q["err_svi_sticky_strike"] ** 2)) * 1e4
    assert s.loc["svi_sticky_strike", "rmse_bp"] == pytest.approx(pooled)  # pooled over quotes, not slice means
    assert (res.daily["target"].diff().dropna() > pd.Timedelta(0)).all()


def test_a_zero_error_benchmark_gives_no_skill_not_infinity():
    history = {pd.Timestamp(d): _quoted(d) for d in ("2026-10-19", "2026-10-20")}
    s = run_walk_forward(history, enforce_arbitrage=False).summary.set_index("predictor")
    assert np.isnan(s.loc["svi_sticky_strike", "skill"])


def test_constant_maturity_atm_vol_interpolates_total_variance():
    def slice_at(days: int, vol: float):
        df = _quoted("2026-10-20", call_gap=0.0, expiry=pd.Timestamp("2026-10-20") + pd.Timedelta(days=days), level=vol)
        return clean_option_slice(df)

    s20, s40 = slice_at(20, 0.20), slice_at(40, 0.30)
    vol, how = constant_maturity_atm_vol([s20, s40], 30.0)
    assert vol == pytest.approx(np.sqrt((0.20**2 * 20 + 0.30**2 * 40) / 2 / 30), rel=1e-9)
    assert "between 20d and 40d" in how
    vol, how = constant_maturity_atm_vol([s40], 30.0)
    assert np.isnan(vol) and "no expiries bracket 30 days" in how


def test_regime_buckets_are_terciles_with_counts():
    out = regime_buckets(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert out["bucket"].tolist() == ["low", "low", "mid", "mid", "high", "high"]
    assert regime_buckets(pd.Series([0.1, 0.2]))["bucket"].tolist() == ["n/a", "n/a"]
    res = run_walk_forward(synthetic_history(6), max_days=60, min_regime_days=2)
    assert set(res.regimes["bucket"]) <= {"low", "mid", "high"}
    assert res.regimes["atm_vol_cm"].between(0.05, 0.6).all()
    br = res.by_regime
    assert br["origin_days"].sum() == res.slices["origin"].nunique()
    assert (br["sufficient"] == (br["origin_days"] >= 2)).all()


def test_statistics_helpers():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)
    assert all(np.isnan(x) for x in wilson_interval(0, 0))
    assert diebold_mariano([1.0, 2.0, 3.0]) == pytest.approx(2.0 / (1.0 / np.sqrt(3.0)))
    assert np.isnan(diebold_mariano([1.0, 1.0])) and np.isnan(diebold_mariano([1.0, 1.0, 1.0]))


def test_runs_are_deterministic():
    history = synthetic_history(4)
    a = run_walk_forward(history, max_days=45)
    b = run_walk_forward(history, max_days=45)
    pd.testing.assert_frame_equal(a.quotes, b.quotes)
    pd.testing.assert_frame_equal(a.summary, b.summary)
    pd.testing.assert_frame_equal(walk_forward_test(history, max_days=45), a.slices)


def test_bad_histories_and_settings_are_clear_errors():
    two_days = pd.concat([_quoted("2026-10-19"), _quoted("2026-10-20")])
    with pytest.raises(ValueError, match="holds 2 trade dates"):
        run_walk_forward({pd.Timestamp("2026-10-19"): two_days})
    with pytest.raises(ValueError, match="has trade_date"):
        run_walk_forward({pd.Timestamp("2026-10-21"): _quoted("2026-10-20")})
    with pytest.raises(ValueError, match="horizon"):
        WalkForwardSettings(horizon=0)
    with pytest.raises(ValueError, match="window"):
        WalkForwardSettings(window=0)
    with pytest.raises(ValueError, match="min_points"):
        WalkForwardSettings(min_points=5)
    single = run_walk_forward({pd.Timestamp("2026-10-20"): _quoted("2026-10-20")})
    assert single.empty and (single.skipped["stage"] == "horizon").all()


def test_synthetic_history_is_seeded_and_labelled():
    a, b = synthetic_history(3, seed=1), synthetic_history(3, seed=1)
    for d in a:
        pd.testing.assert_frame_equal(a[d], b[d])
    c = synthetic_history(3, seed=2)
    assert not a[list(a)[-1]]["implied_vol"].equals(c[list(c)[-1]]["implied_vol"])
    frame = a[list(a)[0]]
    assert set(frame["source"]) == {"synthetic"} and set(frame["ticker"]) == {"SYNTH"}


def _dm_reference(d, h):
    """Independent Diebold-Mariano with the Harvey-Leybourne-Newbold factor
    (rectangular kernel, lags 0 .. h-1), written out term by term."""

    d = np.asarray(d, dtype=float)
    n, mean = d.size, d.mean()
    acov = [sum((d[t] - mean) * (d[t - j] - mean) for t in range(j, n)) / n for j in range(h)]
    var_mean = (acov[0] + 2 * sum(acov[1:])) / n
    return mean / np.sqrt(var_mean) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)


def test_diebold_mariano_accounts_for_overlapping_horizons():
    rng = np.random.default_rng(3)
    shocks = rng.standard_normal(41)
    d = 0.3 + shocks[1:] + 0.8 * shocks[:-1]  # MA(1): what 2-day-ahead forecasts on daily origins produce
    for h in (1, 2, 3):
        assert diebold_mariano(d, horizon=h) == pytest.approx(_dm_reference(d, h), rel=1e-12)
    assert diebold_mariano(d, horizon=1) == pytest.approx(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
    # positively autocorrelated differentials: the naive t overstates significance
    assert abs(diebold_mariano(d, horizon=2)) < abs(diebold_mariano(d, horizon=1))
    assert np.isnan(diebold_mariano([1.0, 2.0, 4.0], horizon=3))  # too few days for the horizon


def test_summary_scores_days_and_uses_the_run_horizon_for_dm():
    history = synthetic_history(6)
    for horizon in (1, 2):
        res = run_walk_forward(history, horizon=horizon, max_days=45)
        s = res.summary.set_index("predictor")
        d = res.daily.set_index("target").sort_index()
        mse = res.quotes.groupby("target")[["err_svi_sticky_strike", "err_persistence"]].apply(lambda g: (g**2).mean())
        diff = (mse["err_svi_sticky_strike"] - mse["err_persistence"]).sort_index()
        assert s.loc["svi_sticky_strike", "dm_t"] == pytest.approx(diebold_mariano(diff, horizon=horizon))
        wins = int((d["rmse_bp_svi_sticky_strike"] < d["rmse_bp_persistence"]).sum())
        assert s.loc["svi_sticky_strike", "day_win_n"] == len(d) == s.loc["svi_sticky_strike", "n_days"]
        assert s.loc["svi_sticky_strike", "day_win_rate"] == pytest.approx(wins / len(d))
        lo, hi = wilson_interval(wins, len(d))
        assert (s.loc["svi_sticky_strike", "day_win_lo"], s.loc["svi_sticky_strike", "day_win_hi"]) == (lo, hi)
        assert s.loc["persistence", "day_win_n"] == 0 and np.isnan(s.loc["persistence", "day_win_rate"])
