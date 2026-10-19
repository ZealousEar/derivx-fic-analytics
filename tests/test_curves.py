import datetime as dt

import numpy as np
import pytest

from derivx.rates.curves import CurveQuote, Market, bill_bey, bill_price, bootstrap, flat_curve
from derivx.rates.examples import EXAMPLE_ASOF, example_curve, ust_curve
from derivx.rates.quotes import clean_quotes, detect_units
from derivx.rates.term_structure import build_zero_curve

D = dt.date.fromisoformat


def test_bootstraps_match_quantlib(golden):
    """OIS (USD/EUR/GBP, two interpolations), single-curve IRS and Treasury
    bill/bond curves against QuantLib's piecewise bootstraps."""

    data = golden("curves")
    for case in data["cases"]:
        asof = D(case.get("asof", data["asof"]))
        curve = bootstrap(asof, [CurveQuote(*q) for q in case["quotes"]], case["conventions"], case["interpolation"])
        worst_bp = max(abs(curve.zero_rate(D(day)) - z) for day, z in case["zeros"]) * 1e4
        assert worst_bp < 1e-6, (case["conventions"], case["interpolation"], worst_bp)


@pytest.mark.parametrize("interpolation", ["loglinear", "linear_zero", "monotone_convex"])
def test_every_quote_reprices(interpolation):
    for curve in (example_curve(interpolation), ust_curve(interpolation=interpolation)):
        assert max(abs(r["error_bp"]) for r in curve.repricing()) < 1e-8
        assert curve.df(curve.asof) == 1.0


def test_monotone_convex_forwards_are_continuous_and_local():
    curve = example_curve("monotone_convex")
    for t in curve.ts[1:-1]:  # forwards agree on both sides of every pillar
        assert curve.instantaneous_forward(t - 1e-4) == pytest.approx(curve.instantaneous_forward(t + 1e-4), abs=2e-5)
    bumped = curve.rebuilt({len(curve.quotes) - 1: 10.0})  # move only the 30Y quote
    assert bumped.df(1.0) == pytest.approx(curve.df(1.0), abs=1e-12)


def test_negative_rates_and_flat_forward_extrapolation():
    quotes = [CurveQuote("OIS", t, r) for t, r in [("1M", -0.006), ("1Y", -0.005), ("5Y", -0.002), ("10Y", 0.001)]]
    curve = bootstrap(EXAMPLE_ASOF, quotes, "EUR ESTR")
    assert curve.df(2.0) > 1.0  # negative rates are kept, not clipped
    assert max(abs(r["error_bp"]) for r in curve.repricing()) < 1e-8
    last = curve.ts[-1]
    assert curve.fwd_rate(last + 5, last + 10) == pytest.approx(curve.fwd_rate(last - 0.5, last), abs=1e-12)


def test_rebuilt_parallel_bump_moves_every_quote():
    curve = example_curve()
    up = curve.rebuilt(1.0)
    for r0, r1 in zip(curve.repricing(), up.repricing()):
        assert r1["quote"] - r0["quote"] == pytest.approx(1e-4)
        assert abs(r1["error_bp"]) < 1e-8
    later = curve.rebuilt(asof=EXAMPLE_ASOF + dt.timedelta(days=30))
    assert later.asof == EXAMPLE_ASOF + dt.timedelta(days=30)


def test_bootstrap_rejects_bad_inputs_clearly():
    with pytest.raises(ValueError, match="at least one"):
        bootstrap(EXAMPLE_ASOF, [], "USD SOFR")
    with pytest.raises(ValueError, match="share the pillar"):
        bootstrap(EXAMPLE_ASOF, [CurveQuote("OIS", "12M", 0.04), CurveQuote("OIS", "1Y", 0.04)], "USD SOFR")
    with pytest.raises(ValueError, match="interpolation"):
        bootstrap(EXAMPLE_ASOF, [CurveQuote("OIS", "1Y", 0.04)], "USD SOFR", "spline")
    with pytest.raises(ValueError, match="conventions"):
        bootstrap(EXAMPLE_ASOF, [CurveQuote("OIS", "1Y", 0.04)], "JPY TONA")


def test_bill_yield_round_trip_both_sides_of_half_a_year():
    for days in (28, 91, 182, 183, 270, 364):
        price = bill_price(0.0437, days)
        assert bill_bey(price, days) == pytest.approx(0.0437, abs=1e-12)


def test_clean_quotes_detects_units_and_reports_problems():
    rows = [{"kind": "OIS", "tenor": "1 Mo", "rate": 4.11}, {"kind": "OIS", "tenor": "2Y", "rate": "4.5%"}]
    quotes, issues, units = clean_quotes(rows)
    assert units == "percent" and [q.rate for q in quotes] == pytest.approx([0.0411, 0.045])
    assert detect_units([450, 460]) == "bp" and detect_units([0.04]) == "decimal"
    bad = [
        {"kind": "OIS", "tenor": "ON", "rate": 0.04},
        {"kind": "FRA", "tenor": "1Y", "rate": 0.04},
        {"kind": "OIS", "tenor": "1Q", "rate": 0.04},
        {"kind": "OIS", "tenor": "2Y", "rate": "n/a"},
        {"kind": "OIS", "tenor": "5Y", "rate": 0.9},
        {"kind": "OIS", "tenor": "5y", "rate": 0.04},
        {"kind": "OIS", "tenor": "5Y", "rate": 0.04},
        {"kind": "OIS", "tenor": "", "rate": ""},
    ]
    quotes, issues, _ = clean_quotes(bad)
    errors = [i.message for i in issues if i.level == "error"]
    assert len(errors) == 5 and any("FRA" in e for e in errors) and any("repeats" in e for e in errors)
    assert [q.tenor for q in quotes] == ["ON", "5Y"]


def test_market_uses_fixings_for_periods_before_asof():
    base = flat_curve(EXAMPLE_ASOF, 0.05)
    later_asof = EXAMPLE_ASOF + dt.timedelta(days=60)
    market = Market(curve=flat_curve(later_asof, 0.02), fixings=base)
    start, end = EXAMPLE_ASOF + dt.timedelta(days=30), EXAMPLE_ASOF + dt.timedelta(days=120)
    growth = market.compounded_growth(start, end)
    expected = np.exp(0.05 * 30 / 365) * np.exp(0.02 * 60 / 365)
    assert growth == pytest.approx(expected, rel=1e-12)
    assert market.projected_rate(start, end, "ACT/365F", fixing_date=start) == pytest.approx(
        base.forward_rate(start, end, "ACT/365F")
    )


def test_legacy_build_zero_curve_now_reprices_its_quotes():
    curve = build_zero_curve({"1W": 5.12, "1M": 5.2, "2Y": 5.3, "5Y": 5.5, "10Y": 5.6}, asof=EXAMPLE_ASOF)
    assert max(abs(r["error_bp"]) for r in curve.repricing()) < 1e-8  # percent quotes detected
    assert len(curve.ts) == 6 and curve.discount(0.0) == 1.0  # short-end pillars are kept
    with pytest.raises(ValueError, match="outside"):  # mixed units are refused, not guessed
        build_zero_curve({"1W": 0.0512, "2Y": 5.3, "5Y": 0.055})
    with pytest.raises(ValueError, match="tenor"):
        build_zero_curve({"1Q": 0.05})
