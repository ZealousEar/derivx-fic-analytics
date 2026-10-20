"""Curve RV: structures and weights, the audit's 2026-10-16 levels, rolling
statistics against a direct computation, OU half-lives on simulated data,
carry of long structures and the ranking table."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from derivx.data import rates_history as rh
from derivx.rates import pca, rv
from derivx.rates.conventions import parse_tenor

D = dt.date.fromisoformat


@pytest.fixture(scope="module")
def ust() -> pd.DataFrame:
    return rh.bundled_store().complete("UST_PAR")


def _daily_series(values, start="2020-01-01") -> pd.Series:
    days = pd.bdate_range(start, periods=len(values)).date
    return pd.Series(np.asarray(values, dtype=float), index=list(days))


def test_quoted_structures_on_the_audit_date(ust):
    """G12 evidence (FRED CMT, 2026-10-16): 2s10s 35bp, 5s30s 54bp,
    2s5s10s fly -5bp."""

    day = D("2026-10-16")
    for name, expected in (("2s10s", 35.0), ("5s30s", 54.0), ("2s5s10s", -5.0)):
        series = rv.structure_series(ust, rv.structure_weights(rv.PRESETS[name]))
        assert series.loc[day] == pytest.approx(expected, abs=1e-9), name
        assert series.name == name


def test_weight_schemes(ust):
    assert rv.structure_weights(["10Y", "2Y"]) == {"2Y": -1.0, "10Y": 1.0}
    assert rv.structure_weights(["2Y", "5Y", "10Y"]) == {"2Y": -1.0, "5Y": 2.0, "10Y": -1.0}
    mat = rv.structure_weights(["2Y", "5Y", "10Y"], "maturity")
    assert mat == pytest.approx({"2Y": -1.25, "5Y": 2.0, "10Y": -0.75})
    years = np.array([parse_tenor(t).years for t in mat])
    w = np.array(list(mat.values()))
    assert w.sum() == pytest.approx(0.0) and w @ years == pytest.approx(0.0)  # parallel- and slope-neutral
    result = pca.pca_curve_changes(ust, start=D("2015-01-02"), end=D("2026-10-16"))
    neutral = rv.structure_weights(["2Y", "5Y", "10Y"], "PCA", result)
    np.testing.assert_allclose(pca.factor_exposure(result, neutral, 2), 0.0, atol=1e-12)
    assert rv.structure_name(["2Y", "5Y", "10Y"]) == "2s5s10s" and rv.structure_name(["3M", "10Y"]) == "3M/10Y"
    with pytest.raises(ValueError, match="need a PCA"):
        rv.structure_weights(["2Y", "10Y"], "PCA")
    with pytest.raises(ValueError, match="two legs"):
        rv.structure_weights(["2Y", "5Y", "10Y", "30Y"])
    with pytest.raises(ValueError, match="must be one of"):
        rv.structure_weights(["2Y", "10Y"], "beta")


def test_parse_custom_weights():
    assert rv.parse_weights("2y:-1, 5Y:2; 10Y:-1") == {"2Y": -1.0, "5Y": 2.0, "10Y": -1.0}
    for text, message in (("2Y", "tenor:weight"), ("2Y:x, 5Y:1", "number"), ("2Y:1", "at least two"),
                          ("2Y:1, 2Y:2", "twice"), ("2Y:0, 5Y:0", "non-zero"), ("2Q:1, 5Y:1", "tenor")):  # fmt: skip
        with pytest.raises(ValueError, match=message):
            rv.parse_weights(text)


def test_structure_series_needs_every_leg():
    levels = pd.DataFrame({"2Y": [0.04, np.nan, 0.041], "10Y": [0.045, 0.046, 0.044]},
                          index=[D("2026-01-02"), D("2026-01-05"), D("2026-01-06")])  # fmt: skip
    series = rv.structure_series(levels, {"2Y": -1.0, "10Y": 1.0})
    assert list(series.index) == [D("2026-01-02"), D("2026-01-06")]
    np.testing.assert_allclose(series.to_numpy(), [50.0, 30.0])
    with pytest.raises(ValueError, match="not in the history"):
        rv.structure_series(levels, {"2Y": -1.0, "30Y": 1.0})


def test_rolling_stats_against_a_direct_window():
    rng = np.random.default_rng(3)
    series = _daily_series(np.cumsum(rng.normal(size=900)))
    stats = rv.rolling_stats(series, 1.0)
    for day in (series.index[400], series.index[-1]):
        window = series[(series.index > day - dt.timedelta(days=365)) & (series.index <= day)]
        row = stats.loc[day]
        assert row["n"] == len(window)
        assert row["mean"] == pytest.approx(window.mean())
        assert row["sd"] == pytest.approx(window.std(ddof=1))
        assert row["z"] == pytest.approx((series.loc[day] - window.mean()) / window.std(ddof=1))
        assert row["pct"] == pytest.approx(100.0 * np.mean(window.to_numpy() <= series.loc[day]))
    assert stats["mean"].iloc[: rv.MIN_WINDOW_OBS - 1].isna().all()
    snap = rv.snapshot(series).set_index("window")
    assert snap.loc["1Y", "z"] == pytest.approx(stats["z"].iloc[-1]) and snap.loc["1Y", "sampling"] == "daily"
    assert snap.loc["5Y", "n"] == len(series)  # shorter than five years: every observation
    month_ends = [dt.date(2020 + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1) for m in range(1, 25)]
    assert rv.sampling(month_ends) == "sparse"


def test_half_life_of_a_simulated_ou_process():
    rng = np.random.default_rng(11)
    phi, n = 0.97, 20000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 5.0 + phi * (x[t - 1] - 5.0) + rng.normal()
    fit = rv.half_life(_daily_series(x))
    assert fit.mean_reverting
    assert fit.days == pytest.approx(math.log(2) / -math.log(phi), rel=0.15)  # 22.8 business days
    assert fit.long_run_mean == pytest.approx(5.0, abs=0.5)
    trend = _daily_series(np.exp(np.linspace(0.0, 3.0, 300)))  # explosive: no mean reversion
    assert not rv.half_life(trend).mean_reverting and math.isinf(rv.half_life(trend).days)
    with pytest.raises(ValueError, match="oscillates"):
        rv.half_life(_daily_series(np.tile([1.0, -1.0], 100)))
    with pytest.raises(ValueError, match="needs at least"):
        rv.half_life(_daily_series(x[:10]))
    gappy = pd.Series(x[:200], index=[D("2020-01-01") + dt.timedelta(days=30 * i) for i in range(200)])
    with pytest.raises(ValueError, match="only 0 daily changes"):
        rv.half_life(gappy)


def test_long_structure_carry_signs():
    carry = {"2Y": 10.0, "5Y": 6.0, "10Y": 4.0}
    assert rv.long_carry({"2Y": -1.0, "10Y": 1.0}, carry) == pytest.approx(6.0)  # steepener: long 2Y, short 10Y
    assert rv.long_carry({"2Y": -1.0, "5Y": 2.0, "10Y": -1.0}, carry) == pytest.approx(2.0)
    assert math.isnan(rv.long_carry({"3M": -1.0, "10Y": 1.0}, carry))


def test_rank_structures(ust):
    structures = {name: rv.structure_weights(legs) for name, legs in rv.PRESETS.items()}
    daily = ust[ust.index >= D("2014-12-31")]
    carry = {"2Y": 5.0, "5Y": 3.0, "10Y": 2.0, "30Y": 1.0}
    table = rv.rank_structures(daily, structures, "3Y", carry, 3).set_index("structure")
    assert list(table.index) == list(rv.PRESETS)
    assert table.loc["2s10s", "value_bp"] == pytest.approx(38.0)  # 2026-10-19
    series = rv.structure_series(daily, structures["2s10s"])
    window = series[series.index > series.index[-1] - dt.timedelta(days=int(round(365.25 * 3)))]
    assert table.loc["2s10s", "n"] == len(window)
    vol = rv.daily_changes(window).std(ddof=1)
    assert table.loc["2s10s", "daily_vol_bp"] == pytest.approx(vol)
    assert table.loc["2s10s", "carry_roll_long_bp"] == pytest.approx(3.0)
    assert table.loc["2s10s", "carry_to_vol"] == pytest.approx(3.0 / (vol * math.sqrt(63)))
    assert math.isnan(table.loc["3M10Y", "carry_roll_long_bp"])
    assert table.loc["2s5s10s", "half_life_days"] > 0
    with pytest.raises(ValueError, match="window must be one of"):
        rv.rank_structures(daily, structures, "2Y")
    # On the whole store (month ends 2008-2014, then daily) each structure keeps its own daily stretch.
    whole = rv.rank_structures(ust, structures, "3Y", carry, 3).set_index("structure")
    assert whole.loc["2s10s", "n"] == table.loc["2s10s", "n"] and (whole["sampling"] == "daily").all()


def test_daily_run_drops_the_month_ends_before_a_daily_stretch():
    month_ends = [D("2026-05-29"), D("2026-06-30"), D("2026-07-31")]
    start = D("2026-08-10")  # ten days after the last month end
    daily = [start + dt.timedelta(days=k) for k in range(30) if (start + dt.timedelta(days=k)).weekday() < 5]
    series = pd.Series(np.arange(len(month_ends + daily), dtype=float), index=month_ends + daily)
    run = rv.daily_run(series)
    assert run.index[0] == start and len(run) == len(daily)
    # A month end one business day before the daily stretch is a daily change, so it stays.
    joined = pd.Series(1.0, index=[D("2026-07-31")] + [D("2026-08-03"), D("2026-08-04")])
    assert rv.daily_run(joined).index[0] == D("2026-07-31")


def test_swap_spreads_on_common_dates(ust):
    days = [d for d in ust.index if d >= D("2025-06-02")]
    rng = np.random.default_rng(5)
    spread_bp = 20.0 + np.cumsum(rng.normal(scale=0.5, size=len(days)))
    swaps = pd.DataFrame({"10Y": ust.loc[days, "10Y"].to_numpy() + spread_bp * 1e-4}, index=days)
    table = rv.swap_spreads(swaps.iloc[::1], ust, "1Y").set_index("tenor")
    assert list(table.index) == ["10Y"] and table.loc["10Y", "last_date"] == days[-1]
    assert table.loc["10Y", "value_bp"] == pytest.approx(spread_bp[-1])
    window = spread_bp[np.array([d > days[-1] - dt.timedelta(days=365) for d in days])]
    assert table.loc["10Y", "mean_bp"] == pytest.approx(window.mean())
    assert table.loc["10Y", "z"] == pytest.approx((spread_bp[-1] - window.mean()) / window.std(ddof=1))
    assert rv.swap_spreads(swaps, ust.drop(columns=["10Y"]), "1Y").empty  # no common tenor
