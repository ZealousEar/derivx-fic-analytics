"""Published curves -> bootstrap quotes: Treasury par yields, the BoE OIS
spot curve as par OIS rates (QuantLib golden values and a live QuantLib
check), fixings as ON deposits."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

from derivx.data.rates_sources import parse_boe_ois_zip
from derivx.rates.curves import CurveQuote, bootstrap
from derivx.rates.market_quotes import (
    OIS_PAR_TENORS,
    fixing_quote,
    ois_par_quotes,
    par_from_zero_rows,
    treasury_quotes,
    zero_rate_curve,
)

ROOT = Path(__file__).resolve().parents[1]
BOE_FIXTURE = ROOT / "tests" / "fixtures" / "rates" / "boe_latest.zip"


def _boe_rows():
    frame = parse_boe_ois_zip(BOE_FIXTURE.read_bytes())
    return {
        day: list(zip(rows["tenor_years"], rows["value_decimal"]))
        for day, rows in frame.sort_values("tenor_years").groupby("asof")
    }


def test_par_ois_rates_match_quantlib_golden(golden):
    data = golden("market_data")
    rows = _boe_rows()
    assert len(data["cases"]) == 3
    for case in data["cases"]:
        day = dt.date.fromisoformat(case["asof"])
        ours = {q.tenor: q.rate for q in par_from_zero_rows(day, rows[day], "GBP SONIA")}
        for tenor, rate in case["par"].items():
            assert ours[tenor] == pytest.approx(rate, abs=1e-13), (day, tenor)  # 1e-9 bp
        assert "25Y" in ours  # derived, but past QuantLib's linear-zero extrapolation point


def test_par_ois_rates_match_quantlib_live():
    pytest.importorskip("QuantLib")
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    from golden import market_data

    for case in market_data.generate()["cases"]:
        day = dt.date.fromisoformat(case["asof"])
        ours = {q.tenor: q.rate for q in par_from_zero_rows(day, _boe_rows()[day], "GBP SONIA")}
        np.testing.assert_allclose([ours[t] for t in case["par"]], list(case["par"].values()), rtol=0, atol=1e-13)


def test_bootstrapping_the_par_rates_gives_back_the_boe_curve():
    day, rows = sorted(_boe_rows().items())[-1]
    quotes = par_from_zero_rows(day, rows, "GBP SONIA")
    assert [q.tenor for q in quotes] == list(OIS_PAR_TENORS)
    curve = bootstrap(day, quotes, "GBP SONIA", "linear_zero")
    boe = zero_rate_curve(day, [y for y, _ in rows], [z for _, z in rows])
    assert max(abs(r["error_bp"]) for r in curve.repricing()) < 1e-8
    for inst in curve.instruments:
        gap_bp = abs(curve.zero_rate(inst.pillar) - boe.zero_rate(inst.pillar)) * 1e4
        assert gap_bp < (1e-6 if inst.tenor.years <= 10 else 0.5), (inst.tenor, gap_bp)


def test_zero_rate_curve():
    day = dt.date(2026, 10, 19)
    curve = zero_rate_curve(day, [0.5, 1.0, 2.0], [0.04, 0.042, 0.045])
    assert curve.interpolation == "linear_zero" and curve.zero_rate(1.0) == pytest.approx(0.042)
    assert curve.zero_rate(0.1) == pytest.approx(0.04)  # flat before the first maturity
    assert curve.df(1.5) == pytest.approx(np.exp(-0.0435 * 1.5), rel=1e-12)
    for bad in (([1.0, 0.5], [0.04, 0.04]), ([1.0], [np.nan]), ([1.0, 2.0], [0.04]), ([0.0001], [0.04]), ([], [])):
        with pytest.raises(ValueError):
            zero_rate_curve(day, *bad)


def test_ois_par_quotes_skip_tenors_past_the_curve():
    day = dt.date(2026, 10, 19)
    curve = zero_rate_curve(day, [1 / 12, 1.0, 5.0], [0.04, 0.04, 0.04])
    tenors = [q.tenor for q in ois_par_quotes(curve, "GBP SONIA", ["1M", "1Y", "5Y", "7Y"])]
    assert tenors == ["1M", "1Y", "5Y"]  # 7Y would need two years of extrapolation
    flat = {q.tenor: q.rate for q in ois_par_quotes(curve, "GBP SONIA", ["1Y"])}
    assert flat["1Y"] == pytest.approx((np.exp(0.04 * 365 / 365) - 1.0), rel=1e-3)  # annual ACT/365F on a flat 4%


def test_treasury_quotes_and_fixings():
    quotes = treasury_quotes({"1 Mo": 0.041, "1.5 Month": 0.0414, "1 Yr": 0.045, "2 Yr": 0.0477, "30 Yr": float("nan")})
    assert quotes == [
        CurveQuote("BILL", "1M", 0.041),
        CurveQuote("BILL", "6W", 0.0414),
        CurveQuote("BILL", "1Y", 0.045),
        CurveQuote("BOND", "2Y", 0.0477),
    ]
    assert fixing_quote(0.0388) == CurveQuote("DEPO", "ON", 0.0388)
    with pytest.raises(ValueError):
        fixing_quote(float("nan"))
