"""Realised against implied ATM normal vol (AL-20): forward swap rates read
off a curve history (QuantLib golden), the realised normal-vol estimator
(business days, gaps, no look-ahead), and the comparison against a vol
history's ATM quotes."""

from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivx.data import rates_history as rh
from derivx.rates.swaption_vol import VolGrid
from derivx.validation.realised_implied import (
    CurveBasis,
    RealisedSettings,
    curve_forwards,
    default_calendar,
    forward_swap_rates,
    history_forwards,
    implied_atm,
    log_discounts,
    realised_normal_vol,
    realised_vs_implied,
    store_underlying,
    swap_schedule,
)
from derivx.validation.vol_history import VolHistory, synthetic_vol_history

ROOT = Path(__file__).resolve().parents[1]


def _levels(tenors, rates, day=dt.date(2026, 1, 15)) -> pd.DataFrame:
    return pd.DataFrame([list(rates)], columns=list(tenors), index=[day])


def _bdays(start: str, n: int, calendar: str = "USGOV") -> list:
    from derivx.rates.conventions import Calendar

    cal, out, day = Calendar(calendar), [], pd.Timestamp(start).date()
    while len(out) < n:
        if cal.is_business_day(day):
            out.append(day)
        day += dt.timedelta(days=1)
    return out


# -- forward swap rates -----------------------------------------------------------


def _check_case(case: dict) -> None:
    levels = _levels(case["tenors"], case["rates"])
    basis = CurveBasis(case["kind"], case["frequency"])
    dfs = np.exp(log_discounts(levels, basis, case["times"])[0])
    # 1e-12: QuantLib's iterative bootstrap stops at that accuracy (derivx solves each node in closed form)
    np.testing.assert_allclose(dfs, case["dfs"], rtol=0, atol=1e-12)
    for f in case["forwards"]:
        got = forward_swap_rates(levels, basis, f["expiry"], f["tenor"]).iloc[0]
        assert got == pytest.approx(f["rate"], abs=1e-12), (case["name"], f)


def test_forward_swap_rates_match_quantlib(golden):
    """Par bootstrap (semi-annual and annual, deposits below one period),
    zero curves linear in the rate, stubs and single periods: discount
    factors and forward swap rates equal QuantLib's to 1e-12."""

    cases = golden("realised_implied")["cases"]
    assert {c["name"] for c in cases} == {"par_semi", "par_annual", "zero"}
    for case in cases:
        _check_case(case)


def test_forward_swap_rates_match_live_quantlib():
    pytest.importorskip("QuantLib")
    sys.path.insert(0, str(ROOT / "tools"))
    from golden.realised_implied import generate

    for case in generate()["cases"]:
        _check_case(case)


def test_flat_curves_and_par_repricing():
    flat = _levels(["6M", "1Y", "2Y", "5Y", "10Y"], [0.04] * 5)
    for e, t in ((0.25, 1.0), (1.0, 5.0), (2.0, 1.5), (0.75, 9.0)):  # whole semi-annual periods: the flat par rate
        assert forward_swap_rates(flat, CurveBasis("par", 2), e, t).iloc[0] == pytest.approx(0.04, abs=1e-14)
    z = 0.035  # flat continuously compounded: the annual-pay par rate is e^z - 1
    zero = _levels(["1Y", "10Y"], [z, z])
    assert forward_swap_rates(zero, CurveBasis("zero", 1), 2.0, 5.0).iloc[0] == pytest.approx(math.exp(z) - 1, 1e-13)
    # A spot-starting swap at a quoted maturity reprices its quote; between quotes the par rate is linear.
    par = _levels(["3M", "1Y", "2Y", "5Y", "10Y"], [0.030, 0.032, 0.035, 0.038, 0.041])
    basis = CurveBasis("par", 2)
    for tenor, quote in (("2Y", 0.035), ("5Y", 0.038), ("10Y", 0.041)):
        assert forward_swap_rates(par, basis, 0.0, float(tenor[:-1])).iloc[0] == pytest.approx(quote, abs=1e-14)
    assert forward_swap_rates(par, basis, 0.0, 3.0).iloc[0] == pytest.approx(0.035 + 0.003 / 3, abs=1e-14)
    assert np.exp(log_discounts(par, basis, [0.25])[0, 0]) == pytest.approx(1 / (1 + 0.030 * 0.25), abs=1e-15)


def test_schedules_and_refusals():
    times, acc = swap_schedule(1.0, 1.25, 2)
    np.testing.assert_allclose(times, [1.25, 1.75, 2.25])
    np.testing.assert_allclose(acc, [0.25, 0.5, 0.5])
    times, acc = swap_schedule(0.5, 0.25, 1)
    np.testing.assert_allclose(times, [0.75])
    np.testing.assert_allclose(acc, [0.25])
    par = _levels(["1Y", "2Y", "5Y", "10Y"], [0.032, 0.035, 0.038, 0.041])
    with pytest.raises(ValueError, match="past the curve's last node"):
        forward_swap_rates(par, CurveBasis("par", 1), 5.0, 10.0)
    with pytest.raises(ValueError, match="missing rates"):
        forward_swap_rates(_levels(["1Y", "2Y"], [0.03, np.nan]), CurveBasis("par", 1), 0.0, 1.0)
    with pytest.raises(ValueError, match="'par' or 'zero'"):
        CurveBasis("spline", 1)
    out, notes, skipped = curve_forwards(par, CurveBasis("par", 1), [("3M", "2Y"), ("5Y", "10Y"), ("1Y", "5Y")])
    assert set(out) == {"3Mx2Y", "1Yx5Y"}
    assert "before the curve's first maturity (1Y): short end flat" in notes["3Mx2Y"] and "1Yx5Y" not in notes
    assert skipped == [{"point": "5Yx10Y", "reason": skipped[0]["reason"]}] and "15y" in skipped[0]["reason"]


# -- realised normal vol ----------------------------------------------------------------


def test_realised_vol_of_a_known_path_is_exact():
    days = _bdays("2026-01-02", 80)
    step = 6.0 / 1e4  # 6bp a business day, alternating sign
    path = pd.Series(0.04 + step * (np.arange(80) % 2), index=days)
    cfg = RealisedSettings(window=21, calendar="USGOV")
    out = realised_normal_vol(path, days, cfg)
    full = out[out["coverage"] >= 1.0]
    np.testing.assert_allclose(full["realised_bp"], 6.0 * math.sqrt(252), rtol=1e-12)
    assert (full["n_steps"] == 21).all()
    # Early dates: the window starts before the data, coverage below 80% -> not scored.
    assert np.isnan(out.loc[out["coverage"] < 0.8, "realised_bp"]).all() and out["coverage"].iloc[0] == 0.0
    # The step over Martin Luther King Day (Fri 16 -> Tue 20 January) is one USGOV business day but two weekdays.
    assert dt.date(2026, 1, 19) not in days
    weekdays = realised_normal_vol(path, [dt.date(2026, 2, 13)], RealisedSettings(window=21, calendar="WEEKENDS"))
    usgov = realised_normal_vol(path, [dt.date(2026, 2, 13)], cfg)
    assert usgov["realised_bp"].iloc[0] == pytest.approx(6.0 * math.sqrt(252), rel=1e-12)
    # weekday count: 21 business days hold 19 steps of 1 day and one of 2 -> variance 20 x 36 over 21 days
    assert weekdays["realised_bp"].iloc[0] == pytest.approx(math.sqrt(252 * 20 * 36 / 21), rel=1e-12)


def test_realised_vol_estimates_a_brownian_normal_vol():
    rng = np.random.default_rng(7)
    days = _bdays("2016-01-04", 2521, "WEEKENDS")
    sigma = 90.0  # bp a year
    path = pd.Series(0.03 + np.cumsum(rng.normal(0, sigma / 1e4 / math.sqrt(252), len(days))), index=days)
    out = realised_normal_vol(path, [days[-1]], RealisedSettings(window=2520))
    assert out["realised_bp"].iloc[0] == pytest.approx(sigma, rel=0.03)
    assert out["n_steps"].iloc[0] == 2520


def test_no_look_ahead_gaps_and_holiday_observations():
    days = _bdays("2026-03-02", 120)
    rng = np.random.default_rng(1)
    path = pd.Series(0.04 + np.cumsum(rng.normal(0, 5e-4, len(days))), index=days)
    t = days[60]
    cfg = RealisedSettings(window=21, calendar="USGOV")
    trail, fwd = realised_normal_vol(path, [t], cfg), realised_normal_vol(path, [t], cfg, "forward")
    after, before = path.copy(), path.copy()
    after[after.index > t] += rng.normal(0, 1e-2, int((after.index > t).sum()))
    before[before.index < t] += rng.normal(0, 1e-2, int((before.index < t).sum()))
    assert realised_normal_vol(after, [t], cfg)["realised_bp"].iloc[0] == pytest.approx(trail["realised_bp"].iloc[0])
    assert realised_normal_vol(before, [t], cfg, "forward")["realised_bp"].iloc[0] == pytest.approx(
        fwd["realised_bp"].iloc[0]
    )
    assert fwd["n_steps"].iloc[0] == 21 and trail["n_steps"].iloc[0] == 21
    # The last 21 business days have no full forward window.
    assert np.isnan(realised_normal_vol(path, [days[-5]], cfg, "forward")["realised_bp"].iloc[0])
    # A 10-calendar-day hole: its step is dropped, so the window is covered less and not scored.
    holed = path.drop(days[50:58])
    got = realised_normal_vol(holed, [days[60]], cfg)
    assert got["coverage"].iloc[0] < 0.8 and np.isnan(got["realised_bp"].iloc[0])
    loose = realised_normal_vol(holed, [days[60]], RealisedSettings(window=21, calendar="USGOV", max_gap_days=15))
    assert loose["coverage"].iloc[0] == 1.0 and np.isfinite(loose["realised_bp"].iloc[0])
    # A value dated on a holiday of the calendar (Memorial Day) is dropped.
    memorial = dt.date(2026, 5, 25)
    later = days[next(i for i, d in enumerate(days) if d > memorial) + 10]
    extra = pd.concat([path, pd.Series([0.5], index=[memorial])]).sort_index()
    assert realised_normal_vol(extra, [later], cfg)["realised_bp"].iloc[0] == pytest.approx(
        realised_normal_vol(path, [later], cfg)["realised_bp"].iloc[0], rel=1e-12
    )
    assert realised_normal_vol(extra, [later], RealisedSettings(window=21))["realised_bp"].iloc[0] > 1000  # weekdays
    # A quote dated on the holiday reads the trailing window ending the business day before.
    assert realised_normal_vol(path, [memorial], cfg)["realised_bp"].iloc[0] == pytest.approx(
        realised_normal_vol(path, [dt.date(2026, 5, 22)], cfg)["realised_bp"].iloc[0], rel=1e-12
    )


def test_settings_are_validated():
    for kwargs, match in (
        ({"window": 1}, "at least 2 business days"),
        ({"calendar": "MARS"}, "Unknown calendar"),
        ({"max_gap_days": 0}, "max_gap_days"),
        ({"min_coverage": 0.0}, "min_coverage"),
        ({"days_per_year": -1.0}, "days_per_year"),
    ):
        with pytest.raises(ValueError, match=match):
            RealisedSettings(**kwargs)
    with pytest.raises(ValueError, match="direction"):
        realised_normal_vol(pd.Series(dtype=float), [], direction="sideways")
    assert default_calendar("USD", "govt") == "USGOV" and default_calendar("USD", "user") == "USD"
    assert default_calendar("EUR") == "TARGET" and default_calendar("XYZ") == "WEEKENDS"


# -- realised against implied -------------------------------------------------------------


def _flat_history(days, implied_bp=100.0, forward=0.04) -> VolHistory:
    grids, forwards = {}, {}
    for d in days:
        rows = tuple(("1Y", "10Y", o, (implied_bp + 0.01 * o) / 1e4) for o in (-100.0, 0.0, 100.0))
        rows += tuple(("2Y", "5Y", o, 90.0 / 1e4) for o in (-100.0, 100.0))  # no ATM quote at 2Yx5Y
        grids[d] = VolGrid(rows, "normal", 0.0, "test grid")
        forwards[d] = {("1Y", "10Y"): forward}
    return VolHistory(grids, forwards, "test grid")


def test_realised_vs_implied_on_a_known_path():
    days = _bdays("2026-01-02", 120)
    step = 80.0 / math.sqrt(252) / 1e4  # 80bp a year realised, alternating
    rate = pd.Series(0.04 + step * (np.arange(120) % 2), index=days)
    hist = _flat_history(days[30:100])
    assert set(implied_atm(hist)["point"]) == {"1Yx10Y"}
    res = realised_vs_implied(hist, {"1Yx10Y": rate}, RealisedSettings(window=21, calendar="USGOV"), what="test")
    row = res.summary.set_index("point").loc["1Yx10Y"]
    assert row["n_dates"] == 70 and row["n_forward"] == 70
    assert row["implied_mean_bp"] == pytest.approx(100.0) and row["trailing_mean_bp"] == pytest.approx(80.0)
    assert row["spread_mean_bp"] == pytest.approx(20.0) and row["ratio_mean"] == pytest.approx(1.25)
    assert row["premium_mean_bp"] == pytest.approx(20.0) and row["implied_above_share"] == 1.0
    assert row["rmse_implied_bp"] == pytest.approx(20.0) and row["rmse_trailing_bp"] == pytest.approx(0.0, abs=1e-9)
    daily = res.daily
    assert daily["breakeven_daily_bp"].iloc[0] == pytest.approx(100.0 / math.sqrt(252))
    assert daily["rate_pct"].notna().all() and set(daily["point"]) == {"1Yx10Y"}
    assert res.skipped.to_dict("records") == [{"point": "2Yx5Y", "reason": res.skipped["reason"].iloc[0]}]
    assert not res.empty
    # A rate history that ends before the vol dates: nothing scored, the reason names both ranges.
    old = pd.Series(rate.to_numpy(), index=[d - dt.timedelta(days=400) for d in days])
    stale = realised_vs_implied(hist, {"1Yx10Y": old}, RealisedSettings(window=21, calendar="USGOV"))
    assert stale.empty and "covers no 21-business-day window" in "; ".join(stale.skipped["reason"])


def test_history_forwards_and_the_store_underlyings():
    synth = synthetic_vol_history(30, expiries=("1Y",), tenors=("10Y",))
    fwd, skipped = history_forwards(synth)
    assert list(fwd) == ["1Yx10Y"] and len(fwd["1Yx10Y"]) == 30 and not skipped
    res = realised_vs_implied(synth, fwd, RealisedSettings(window=10))
    row = res.summary.iloc[0]
    # SYNTHETIC forwards move 5bp a business day: about 79bp a year realised
    assert 50 < row["trailing_mean_bp"] < 110 and row["n_dates"] == 30 - 8  # 80% of 10 days covered
    no_fwd = VolHistory(dict(synth.grids), {}, "upload")
    assert history_forwards(no_fwd) == (
        {},
        [{"point": "1Yx10Y", "reason": "the vol history gives no forwards for this point"}],
    )

    store = rh.bundled_store()
    ust = store_underlying(store, "UST_PAR")
    assert ust.proxy and ust.basis == CurveBasis("par", 2, ust.basis.description) and ust.calendar == "USGOV"
    assert "PROXY" in ust.what and ust.levels.notna().all().all() and len(ust.levels) > 2500
    eur = store_underlying(store, "EUR_AAA_ECB")
    assert eur.proxy and eur.basis.kind == "zero" and eur.calendar == "TARGET" and eur.measure == "zero_cc"
    gbp = store_underlying(store, "GBP_OIS_BOE")
    assert not gbp.proxy and gbp.basis.kind == "zero" and gbp.calendar == "LONDON"
    with pytest.raises(ValueError, match="not a curve"):
        store_underlying(store, "FIXINGS")
    # The 1Y x 10Y forward par yield on the last Treasury curve sits between the curve's 1Y and 20Y yields.
    out, notes, skipped = curve_forwards(ust.levels.tail(5), ust.basis, [("1Y", "10Y"), ("5Y", "30Y")])
    last = ust.levels.iloc[-1]
    assert min(last["1Y"], last["20Y"]) < out["1Yx10Y"].iloc[-1] < max(last["1Y"], last["20Y"]) + 0.01
    assert [s["point"] for s in skipped] == ["5Yx30Y"] and not notes


def test_an_uploaded_swap_curve_is_the_like_for_like_underlying():
    days = pd.bdate_range("2026-01-02", periods=90)
    rng = np.random.default_rng(3)
    level = 3.6 + np.cumsum(rng.normal(0, 0.05, days.size))  # percent, 5bp a day
    frame = pd.DataFrame({"date": days.strftime("%Y-%m-%d")})
    for tenor, add in (("1Y", -0.2), ("2Y", -0.1), ("5Y", 0.0), ("10Y", 0.2), ("15Y", 0.3), ("30Y", 0.4)):
        frame[tenor] = np.round(level + add, 4)
    upload, issues, units = rh.parse_history_upload(frame, "USD", "auto", "swaps.csv")
    assert units == "percent" and not upload.empty
    store = rh.build_store([rh.bundled_frame(), upload], "test")
    u = store_underlying(store, "USER_SWAP:USD")
    assert not u.proxy and u.basis.kind == "par" and u.basis.frequency == 1 and u.calendar == "USD"
    hist = _flat_history([d.date() for d in days[30:60]])
    out, notes, skipped = curve_forwards(u.levels, u.basis, hist.points)
    assert set(out) == {"1Yx10Y", "2Yx5Y"} and not skipped
    res = realised_vs_implied(hist, out, RealisedSettings(window=21, calendar=u.calendar), what=u.what)
    row = res.summary.set_index("point").loc["1Yx10Y"]
    # a parallel 5bp-a-day random walk: the forward moves with it, about 5 x sqrt(252) = 79bp a year
    assert 50 < row["trailing_mean_bp"] < 110 and row["n_dates"] == 30
