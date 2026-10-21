"""Vanilla swaps: QuantLib golden values, pricing identities, risk, hedging,
seasoned valuation and the Streamlit tab."""

from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from derivx.rates.conventions import Calendar
from derivx.rates.curves import Curve, CurveQuote, Market, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, example_curve, sofr_example_rows
from derivx.rates.swaps import (
    VanillaSwap,
    benchmark_swaps,
    curve_bumps,
    hedge_with_benchmarks,
    key_rate_matrix,
    par_swap_rate,
    swap_position,
)
from derivx.risk.sensitivities import bumped_market, key_rate_pv01, ladder, parallel_pv01

ROOT = Path(__file__).resolve().parents[1]
NOTIONAL = 10_000_000.0
REL = 1e-10


@lru_cache(maxsize=None)
def _curve(asof: dt.date = EXAMPLE_ASOF) -> Curve:
    return example_curve(asof=asof)


def _market(valuation: dt.date = EXAMPLE_ASOF) -> Market:
    """Example SOFR market; after EXAMPLE_ASOF the same quotes re-dated,
    with past fixings from the original curve."""

    if valuation == EXAMPLE_ASOF:
        return Market(curve=_curve())
    return Market(curve=_curve(valuation), fixings=_curve())


def _case_swap(case: dict) -> VanillaSwap:
    """Mirror of tools/golden/swaps.py:case_swap."""

    return VanillaSwap.from_conventions(
        "USD SOFR",
        case["tenor"],
        case["fixed_rate"],
        notional=case["notional"],
        receive_fixed=case["receive_fixed"],
        kind=case["kind"],
        forward_start=case["forward_start"],
        spread=case["spread"],
        trade_date=dt.date.fromisoformat(case["trade_date"]),
    )


def _spec_market(spec: dict) -> Market:
    """Mirror of tools/golden/swaps.py:spec_market."""

    def curve(asof: dt.date, shift: float) -> Curve:
        quotes = [CurveQuote(r["kind"], r["tenor"], r["rate"] + shift) for r in sofr_example_rows()]
        return bootstrap(asof, quotes, spec["conv"], spec["interp"])

    trade, valuation = (dt.date.fromisoformat(spec[k]) for k in ("trade_date", "valuation_date"))
    fixings = curve(trade, spec["shift"])
    if valuation == trade:
        return Market(curve=fixings)
    return Market(curve=curve(valuation, spec["shift"] + spec["valuation_shift"]), fixings=fixings)


def _spec_swap(spec: dict) -> VanillaSwap:
    """Mirror of tools/golden/swaps.py:spec_swap."""

    fields = dict(spec["overrides"])
    fields.update({k: dt.date.fromisoformat(spec[k]) for k in ("start", "maturity") if spec[k]})
    return VanillaSwap.from_conventions(
        spec["conv"],
        spec["tenor"],
        spec["fixed_rate"],
        notional=spec["notional"],
        receive_fixed=spec["receive_fixed"],
        kind=spec["kind"],
        forward_start=spec["forward_start"],
        spread=spec["spread"],
        trade_date=dt.date.fromisoformat(spec["trade_date"]),
        **fields,
    )


def _iso(dates) -> list:
    return [d.isoformat() for d in dates]


def _check_case(case: dict, swap: Optional[VanillaSwap] = None, market: Optional[Market] = None) -> None:
    """derivx against one QuantLib reference: dates, NPV, fair rate, signed
    fixed-leg BPS, float-leg NPV and every live cash flow."""

    swap = swap or _case_swap(case)
    valuation = dt.date.fromisoformat(case["valuation_date"]) if market is None else market.asof
    market = market or _market(valuation)
    label = f"{swap.label} ({swap.calendar}, {swap.float_index}) at {valuation}"
    fixed_dates, float_dates = swap.schedule_dates(valuation)
    assert _iso(fixed_dates) == case["fixed_dates"], label
    assert _iso(float_dates) == case["float_dates"], label
    sign = 1.0 if swap.receive_fixed else -1.0
    assert swap.pv(market) == pytest.approx(case["npv"], rel=REL, abs=1e-6), label
    assert swap.par_rate(market) == pytest.approx(case["fair_rate"], rel=REL), label
    assert sign * swap.annuity(market) == pytest.approx(case["fixed_leg_bps"], rel=REL), label
    assert swap.pv_float(market) == pytest.approx(case["float_leg_npv"], rel=REL, abs=1e-6), label
    flows = swap.cashflows(market)
    for leg, leg_sign, key in (("fixed", sign, "fixed_flows"), ("float", -sign, "float_flows")):
        ours = flows[flows["leg"] == leg]
        assert _iso(ours["pay"]) == [d for d, _ in case[key]], label
        expected = [leg_sign * a for _, a in case[key]]
        np.testing.assert_allclose(ours["amount"], expected, rtol=REL, atol=1e-7, err_msg=label)


# --------------------------------------------------------------------------
# QuantLib golden values (tests/golden/swaps.json, tools/golden/swaps.py)
# --------------------------------------------------------------------------


def test_golden_fixture_uses_the_current_example_curve(golden):
    np.testing.assert_allclose(_curve().dfs, golden("swaps")["curve_dfs"], rtol=1e-13)


def test_spot_and_forward_swaps_match_quantlib(golden):
    cases = golden("swaps")["cases"]
    assert {c["kind"] for c in cases} == {"OIS", "IRS"}
    assert {c["tenor"] for c in cases} == {"1Y", "18M", "5Y", "10Y", "30Y"}
    assert any(c["spread"] for c in cases) and {c["forward_start"] for c in cases} == {None, "1Y"}
    for case in cases:
        _check_case(case)


def test_seasoned_swaps_with_fixings_match_quantlib(golden):
    cases = golden("swaps")["seasoned"]
    assert len(cases) == 8
    for case in cases:
        _check_case(case)


def test_independent_quantlib_cases(golden):
    """QuantLib dates and indices built from the conventions preset alone:
    EUR ESTR with negative rates, GBP SONIA (spot lag 0, term index fixed on
    the period start), linear-zero and monotone-convex curves, odd and
    weekend dates, tenors past the last quote, and seasoned swaps valued on
    a Saturday or a holiday."""

    cases = golden("swaps")["independent"]
    specs = [c["spec"] for c in cases]
    assert {s["conv"] for s in specs} == {"USD SOFR", "EUR ESTR", "GBP SONIA"}
    assert {s["interp"] for s in specs} == {"loglinear", "linear_zero", "monotone_convex"}
    assert any(s["fixed_rate"] < 0 for s in specs)
    for case in cases:
        spec = case["spec"]
        market = _spec_market(spec)
        np.testing.assert_allclose(market.curve.dfs, case["curve_dfs"], rtol=1e-13)
        _check_case(case, _spec_swap(spec), market)


def test_spot_ois_against_make_ois_defaults(golden):
    """ql.MakeOIS defaults differ from the SOFR conventions only in the
    payment lag (0 vs 2 business days); dates otherwise agree."""

    cal = Calendar("USD")
    for case in golden("swaps")["make_ois"]:
        lag = 2 if case["payment_lag"] is None else case["payment_lag"]
        ours = VanillaSwap.from_conventions(
            "USD SOFR", case["tenor"], case["fixed_rate"], notional=case["notional"], trade_date=EXAMPLE_ASOF
        )
        same = replace(ours, payment_lag=0 if case["payment_lag"] is None else lag)
        market = _market()
        fixed_dates, _ = same.schedule_dates(EXAMPLE_ASOF)
        assert _iso(fixed_dates) == case["schedule"]
        assert _iso(p.pay for p in same.periods(EXAMPLE_ASOF) if p.leg == "fixed") == case["fixed_pay_dates"]
        assert same.pv(market) == pytest.approx(case["npv"], rel=REL)
        assert same.par_rate(market) == pytest.approx(case["fair_rate"], rel=REL)
        assert same.annuity(market) == pytest.approx(case["fixed_leg_bps"], rel=REL)
        if case["payment_lag"] is None:
            # The convention gap: MakeOIS pays on the period end, SOFR swaps 2 days later.
            ours_pays = [p.pay for p in ours.periods(EXAMPLE_ASOF) if p.leg == "fixed"]
            assert [cal.add_business_days(dt.date.fromisoformat(d), 2) for d in case["fixed_pay_dates"]] == ours_pays
            assert 0 < abs(ours.par_rate(market) - case["fair_rate"]) < 1e-6


def _ql_helpers():
    ql = pytest.importorskip("QuantLib")
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    from golden import swaps as gs
    from golden.common import ql_curve, set_evaluation_date

    return ql, gs, ql_curve, set_evaluation_date


def test_random_swaps_match_live_quantlib():
    """Random tenors, forward starts, rates, spreads, valuation dates and
    seasoning against QuantLib (skipped without QuantLib)."""

    ql, gs, ql_curve, set_evaluation_date = _ql_helpers()
    rng = np.random.default_rng(20261019)
    cal = Calendar("USD")
    try:
        for i in range(40):
            trade = cal.adjust(EXAMPLE_ASOF + dt.timedelta(days=int(rng.integers(0, 900))), "F")
            seasoned = i % 3 == 0
            valuation = cal.adjust(trade + dt.timedelta(days=int(rng.integers(20, 800))), "F") if seasoned else trade
            months = int(rng.integers(1, 361))
            fwd = int(rng.integers(0, 37)) if i % 2 else 0
            swap = VanillaSwap.from_conventions(
                "USD SOFR",
                f"{months}M",
                float(rng.uniform(0.0, 0.07)),
                notional=float(rng.uniform(1e5, 1e9)),
                receive_fixed=bool(rng.integers(0, 2)),
                kind=("OIS", "IRS")[i % 4 >= 2],
                forward_start=f"{fwd}M" if fwd else None,
                spread=float(rng.uniform(-0.002, 0.002)),
                trade_date=trade,
            )
            fixings = example_curve(asof=trade)
            curve = example_curve(asof=valuation) if seasoned else fixings
            market = Market(curve=curve, fixings=fixings if seasoned else None)
            set_evaluation_date(valuation)
            ql.IndexManager.instance().clearHistories()
            handle = ql_curve(curve)
            if seasoned:
                gs.add_fixings(swap, valuation, fixings, handle)
            ref = gs.ql_values(swap, valuation, handle)
            label = f"case {i}: {swap.label} traded {trade} valued {valuation}"
            assert swap.pv(market) == pytest.approx(ref["npv"], rel=REL, abs=1e-12 * swap.notional), label
            assert swap.par_rate(market) == pytest.approx(ref["fair_rate"], rel=REL), label
            sign = 1.0 if swap.receive_fixed else -1.0
            assert sign * swap.annuity(market) == pytest.approx(ref["fixed_leg_bps"], rel=REL), label
    finally:
        ql.IndexManager.instance().clearHistories()
        set_evaluation_date(EXAMPLE_ASOF)


def test_random_presets_match_live_quantlib():
    """Random swaps on every conventions preset and interpolation, with
    weekend and holiday trade and valuation dates, against QuantLib dates
    and indices built from the preset alone (skipped without QuantLib)."""

    _, gs, _, _ = _ql_helpers()
    rng = np.random.default_rng(42)
    for i in range(18):
        conv = ("USD SOFR", "EUR ESTR", "GBP SONIA")[i % 3]
        trade = EXAMPLE_ASOF + dt.timedelta(days=int(rng.integers(0, 1200)))
        seasoned = i % 2 == 0
        valuation = trade + dt.timedelta(days=int(rng.integers(1, 700))) if seasoned else trade
        terms = dict(
            interp=("loglinear", "linear_zero", "monotone_convex")[(i // 3) % 3],
            shift=-0.045 if conv == "EUR ESTR" else 0.0,
            trade_date=trade.isoformat(),
            valuation_date=valuation.isoformat(),
            valuation_shift=float(rng.uniform(-0.004, 0.004)) if seasoned else 0.0,
            forward_start=f"{int(rng.integers(1, 37))}M" if i % 4 == 1 else None,
            receive_fixed=bool(rng.integers(0, 2)),
            spread=float(rng.uniform(-0.002, 0.002)),
            notional=float(rng.uniform(1e5, 1e9)),
        )
        spec = gs.spec(
            conv, ("OIS", "IRS")[i % 4 >= 2], f"{int(rng.integers(1, 481))}M", float(rng.uniform(-0.01, 0.06)), **terms
        )
        _check_case(gs.reference(spec), _spec_swap(spec), _spec_market(spec))


# --------------------------------------------------------------------------
# Pricing identities
# --------------------------------------------------------------------------

SHAPES = [
    ("OIS", "1Y", None),
    ("OIS", "18M", "6M"),
    ("OIS", "7Y", None),
    ("OIS", "13Y", "2Y"),
    ("IRS", "5Y", None),
    ("IRS", "30Y", "1Y"),
]


@pytest.mark.parametrize("kind,tenor,fwd", SHAPES)
def test_par_swap_is_worth_zero(kind, tenor, fwd):
    market = _market()
    swap = VanillaSwap.from_conventions(
        "USD SOFR", tenor, 0.0, NOTIONAL, kind=kind, forward_start=fwd, spread=0.0012, trade_date=EXAMPLE_ASOF
    )
    at_par = replace(swap, fixed_rate=swap.par_rate(market))
    assert abs(at_par.pv(market)) < 1e-9 * NOTIONAL
    assert at_par.pv_fixed(market) == pytest.approx(-at_par.pv_float(market), rel=1e-12)


@pytest.mark.parametrize("kind,tenor,fwd", SHAPES)
def test_payer_is_minus_receiver_and_pv_is_rate_gap_times_annuity(kind, tenor, fwd):
    market = _market()
    rec = VanillaSwap.from_conventions(
        "USD SOFR", tenor, 0.0415, NOTIONAL, kind=kind, forward_start=fwd, spread=-0.0005, trade_date=EXAMPLE_ASOF
    )
    pay = replace(rec, receive_fixed=False)
    assert pay.pv(market) == pytest.approx(-rec.pv(market), rel=1e-13)
    np.testing.assert_allclose(pay.cashflows(market)["amount"], -rec.cashflows(market)["amount"], rtol=1e-13)
    gap = (rec.fixed_rate - rec.par_rate(market)) * rec.annuity(market) * 1e4
    assert rec.pv(market) == pytest.approx(gap, rel=1e-10)
    assert pay.annuity(market) == pytest.approx(rec.annuity(market), rel=1e-15) and rec.annuity(market) > 0
    assert rec.pv(market) == pytest.approx(rec.pv_fixed(market) + rec.pv_float(market), rel=1e-13)


def test_par_rates_reprice_the_curve_quotes():
    market = _market()
    for quote in market.curve.quotes:
        if quote.kind == "OIS":
            assert par_swap_rate(market, quote.tenor) == pytest.approx(quote.rate, abs=1e-14)
    irs_quotes = [
        CurveQuote("DEPO", "3M", 0.041),
        CurveQuote("IRS", "2Y", 0.0385),
        CurveQuote("IRS", "5Y", 0.038),
        CurveQuote("IRS", "10Y", 0.0392),
        CurveQuote("IRS", "30Y", 0.0401),
    ]
    irs = Market(curve=bootstrap(EXAMPLE_ASOF, irs_quotes, "USD SOFR"))
    for quote in irs_quotes[1:]:
        assert par_swap_rate(irs, quote.tenor, kind="IRS") == pytest.approx(quote.rate, abs=1e-14)


def test_explicit_maturity_and_start_dates():
    market = _market()
    by_tenor = VanillaSwap.from_conventions("USD SOFR", "5Y", 0.036, trade_date=EXAMPLE_ASOF)
    by_date = replace(by_tenor, tenor=None, maturity=dt.date(2031, 10, 21))
    assert by_date.schedule_dates(EXAMPLE_ASOF) == by_tenor.schedule_dates(EXAMPLE_ASOF)
    assert by_date.pv(market) == pytest.approx(by_tenor.pv(market), rel=1e-15)
    explicit = replace(by_tenor, start=dt.date(2026, 10, 21), trade_date=None)
    assert explicit.pv(market) == pytest.approx(by_tenor.pv(market), rel=1e-15)
    # A Saturday start rolls to Monday. Up to a month past one period is a
    # single long period; beyond that, a front stub (as for curve instruments).
    odd = VanillaSwap(tenor=None, start=dt.date(2026, 10, 24), maturity=dt.date(2027, 10, 29))
    assert odd.schedule_dates(EXAMPLE_ASOF)[0] == [dt.date(2026, 10, 26), dt.date(2027, 10, 29)]
    stub = replace(odd, maturity=dt.date(2027, 11, 15))
    assert stub.schedule_dates(EXAMPLE_ASOF)[0] == [dt.date(2026, 10, 26), dt.date(2026, 11, 16), dt.date(2027, 11, 15)]


@pytest.mark.parametrize(
    "fields",
    [
        {"notional": -1.0},
        {"float_index": "LIBOR"},
        {"tenor": "10D"},
        {"tenor": None},
        {"fixed_day_count": "ACT/ACT ICMA"},
        {"calendar": "MARS"},
        {"start": dt.date(2027, 1, 4), "forward_start": "1Y"},
        {"fixed_rate": float("nan")},
        {"bdc": "XX"},
        {"notional": 0.0},
    ],
)
def test_invalid_swaps_are_rejected(fields):
    with pytest.raises(ValueError):
        VanillaSwap(**fields)


def test_maturity_before_start_is_rejected():
    swap = VanillaSwap(tenor=None, maturity=dt.date(2027, 1, 4), forward_start="1Y", trade_date=EXAMPLE_ASOF)
    with pytest.raises(ValueError, match="after its start"):
        swap.pv(_market())


# --------------------------------------------------------------------------
# Risk and hedging
# --------------------------------------------------------------------------


def test_par_receiver_pv01_is_minus_annuity():
    market = _market()
    for tenor in ("2Y", "5Y", "10Y", "30Y"):
        swap = benchmark_swaps(market, [tenor], notional=NOTIONAL)[0]
        assert swap.receive_fixed and abs(swap.pv(market)) < 1e-9 * NOTIONAL
        pv01 = parallel_pv01(swap.pv, market)
        assert pv01 < 0
        # Quoted tenor: +-1bp moves its par rate by exactly +-1bp, so the
        # central difference is minus the mean bumped annuity; the gap to the
        # annuity itself is convexity, (1bp * duration)^2 relative.
        up, down = (bumped_market(market, "curve", bp) for bp in (1.0, -1.0))
        assert pv01 == pytest.approx(-(swap.annuity(up) + swap.annuity(down)) / 2.0, rel=1e-9), tenor
        assert pv01 == pytest.approx(-swap.annuity(market), rel=1e-5), tenor
    off_quote = VanillaSwap.from_conventions("USD SOFR", "4Y", 0.0, NOTIONAL, trade_date=EXAMPLE_ASOF)
    off_quote = replace(off_quote, fixed_rate=off_quote.par_rate(market))
    assert parallel_pv01(off_quote.pv, market) == pytest.approx(-off_quote.annuity(market), rel=1e-3)


@pytest.mark.parametrize("conv", ["GBP SONIA", "EUR ESTR", "USD SOFR"])
def test_spot_irs_first_period_fixes_on_the_trade_date(conv):
    """The term index fixes spot-lag days before each period, so a spot
    swap's first rate is not a past fixing: a par IRS on an IRS-built curve
    has PV01 = -annuity and loads only on its own quote."""

    quotes = [
        CurveQuote("DEPO", "6M", 0.045),
        CurveQuote("IRS", "2Y", 0.041),
        CurveQuote("IRS", "5Y", 0.039),
        CurveQuote("IRS", "10Y", 0.040),
    ]
    market = Market(curve=bootstrap(EXAMPLE_ASOF, quotes, conv))
    swap = VanillaSwap.from_conventions(conv, "5Y", 0.0, NOTIONAL, kind="IRS", trade_date=EXAMPLE_ASOF)
    swap = replace(swap, fixed_rate=swap.par_rate(market))
    first = next(p for p in swap.periods(EXAMPLE_ASOF) if p.leg == "float")
    assert first.fixing == EXAMPLE_ASOF and swap.fixing_lag == swap.spot_lag
    assert parallel_pv01(swap.pv, market) == pytest.approx(-swap.annuity(market), rel=1e-6)
    rows = ladder(key_rate_pv01(swap.pv, market))
    assert abs(rows[2] + swap.annuity(market)) < 1e-6 * swap.annuity(market)
    assert np.abs(np.delete(rows, 2)).max() < 1e-6 * swap.annuity(market)


@pytest.mark.parametrize("kind,tenor,fwd", [("OIS", "10Y", None), ("OIS", "8Y", "2Y"), ("IRS", "12Y", None)])
def test_key_rate_ladder_sums_to_parallel_pv01(kind, tenor, fwd):
    market = _market()
    swap = VanillaSwap.from_conventions(
        "USD SOFR", tenor, 0.041, NOTIONAL, receive_fixed=False, kind=kind, forward_start=fwd, trade_date=EXAMPLE_ASOF
    )
    rows = key_rate_pv01(swap.pv, market)
    total = parallel_pv01(swap.pv, market)
    assert total > 0  # payer gains when rates rise
    assert ladder(rows).sum() == pytest.approx(total, rel=1e-4)
    meta, matrix = key_rate_matrix([swap.pv], market)
    np.testing.assert_allclose(matrix[:, 0], ladder(rows), rtol=1e-12, atol=1e-9)
    assert [m["label"] for m in meta] == [r["label"] for r in rows]


@pytest.mark.parametrize(
    "swap",
    [
        VanillaSwap.from_conventions("USD SOFR", "10Y", 0.0425, 5e7, trade_date=EXAMPLE_ASOF),
        VanillaSwap.from_conventions(
            "USD SOFR", "8Y", 0.031, 2e7, receive_fixed=False, forward_start="2Y", trade_date=EXAMPLE_ASOF
        ),
        VanillaSwap.from_conventions("USD SOFR", "25Y", 0.039, 1e7, forward_start="5Y", trade_date=EXAMPLE_ASOF),
    ],
    ids=["10Y", "2Yx8Y", "5Yx25Y"],
)
def test_benchmark_hedge_flattens_the_ladder(swap):
    market = _market()
    result = hedge_with_benchmarks(swap_position(swap), market)
    assert result["risk_removed"] > 0.99
    residual = np.array([r["residual"] for r in result["ladder"]])
    np.testing.assert_allclose(residual, result["residual"], rtol=1e-12, atol=1e-9)
    target = np.array([r["position"] for r in result["ladder"]])
    assert result["risk_removed"] == pytest.approx(1.0 - residual @ residual / (target @ target), rel=1e-12)
    assert abs(result["hedged_pv01"]) < 0.02 * abs(result["position_pv01"])
    assert result["position_pv01"] + result["hedge_pv01"] == pytest.approx(result["hedged_pv01"], abs=1e-6)
    assert [h["benchmark"] for h in result["hedges"]] == ["2Y", "5Y", "10Y", "30Y"]
    # The hedge positions are par swaps (worth zero) that carry the reported PV01.
    book = [swap_position(swap)] + result["positions"]
    book_pv = lambda m: sum(p.pv(m) for p in book)  # noqa: E731
    assert sum(p.pv(market) for p in result["positions"]) == pytest.approx(0.0, abs=1e-8 * swap.notional)
    tol = 1e-3 * abs(result["position_pv01"])
    assert parallel_pv01(book_pv, market) == pytest.approx(result["hedged_pv01"], abs=tol)
    directions = {1.0: "Receive fixed", -1.0: "Pay fixed", 0.0: "-"}
    for h in result["hedges"]:
        assert h["direction"] == directions[float(np.sign(h["notional"]))]
    assert len(result["positions"]) == sum(h["notional"] != 0 for h in result["hedges"])


def test_curve_bumps_are_reusable_and_match_the_sensitivities():
    market = _market()
    swaps = [
        VanillaSwap.from_conventions("USD SOFR", t, 0.038, NOTIONAL, kind=k, trade_date=EXAMPLE_ASOF)
        for t, k in (("7Y", "OIS"), ("12Y", "IRS"))
    ]
    bumps = curve_bumps(market)
    for swap in swaps:
        assert bumps.parallel_pv01(swap.pv) == parallel_pv01(swap.pv, market)
    np.testing.assert_array_equal(
        bumps.key_rate_matrix([s.pv for s in swaps]),
        np.column_stack([ladder(key_rate_pv01(s.pv, market)) for s in swaps]),
    )
    position = swap_position(swaps[0])
    shared = hedge_with_benchmarks(position, market, bumps=bumps)
    assert shared["ladder"] == hedge_with_benchmarks(position, market)["ladder"]
    with pytest.raises(ValueError):
        curve_bumps(market, parallel=False).parallel_pv01(swaps[0].pv)
    with pytest.raises(ValueError):
        curve_bumps(market, which="bogus")


def test_hedge_drops_benchmarks_the_ladder_does_not_load_on():
    market = _market()
    swap = VanillaSwap.from_conventions("USD SOFR", "5Y", 0.0325, NOTIONAL, False, trade_date=EXAMPLE_ASOF)
    result = hedge_with_benchmarks(swap_position(swap), market, ["5Y", "10Y", "30Y"])
    by_tenor = {h["benchmark"]: h for h in result["hedges"]}
    assert by_tenor["5Y"]["direction"] == "Receive fixed" and by_tenor["5Y"]["notional"] > 0.9 * NOTIONAL
    assert by_tenor["10Y"]["notional"] == by_tenor["30Y"]["notional"] == 0.0
    assert by_tenor["30Y"]["direction"] == "-" and len(result["positions"]) == 1


def test_hedge_rejects_bad_benchmarks():
    position = swap_position(VanillaSwap(trade_date=EXAMPLE_ASOF))
    with pytest.raises(ValueError):
        hedge_with_benchmarks(position, _market(), [])
    with pytest.raises(ValueError):
        hedge_with_benchmarks(position, _market(), ["5Y", "5y"])


def test_swap_position_wraps_pv_and_cash():
    market = _market()
    swap = VanillaSwap.from_conventions("USD SOFR", "3Y", 0.037, NOTIONAL, trade_date=EXAMPLE_ASOF)
    position = swap_position(swap)
    assert position.kind == "swap" and position.vol_model is None
    assert position.pv(market) == swap.pv(market)
    horizon = dt.date(2028, 1, 1)
    assert position.cash(market, EXAMPLE_ASOF, horizon) == swap.cash(market, EXAMPLE_ASOF, horizon) != 0.0
    assert position.details["notional"] == NOTIONAL and position.details["tenor"] == "3Y"


# --------------------------------------------------------------------------
# Seasoned swaps and cash
# --------------------------------------------------------------------------


def _forward_curve(curve: Curve, asof: dt.date) -> Curve:
    """The curve's own forward view from ``asof``: same log-linear nodes, so
    every forward rate after ``asof`` is unchanged."""

    later = [d for d in curve.pillar_dates if d > asof]
    base = curve.df(asof)
    times = [0.0] + [(d - asof).days / 365.0 for d in later]
    dfs = [1.0] + [curve.df(d) / base for d in later]
    return Curve(asof, times, dfs, "loglinear", later)


@pytest.mark.parametrize("kind", ["OIS", "IRS"])
@pytest.mark.parametrize("horizon", [dt.date(2027, 2, 17), dt.date(2028, 1, 18), dt.date(2029, 6, 1)])
def test_seasoned_swap_rolls_forward_consistently(kind, horizon):
    m0 = _market()
    swap = VanillaSwap.from_conventions(
        "USD SOFR", "5Y", 0.0372, NOTIONAL, kind=kind, spread=0.001, trade_date=EXAMPLE_ASOF
    )
    m1 = Market(curve=_forward_curve(m0.curve, horizon), fixings=m0.curve)
    assert swap.schedule_dates(horizon) == swap.schedule_dates(EXAMPLE_ASOF)
    flows0 = swap.cashflows(m0)
    window = (flows0["pay"] > EXAMPLE_ASOF) & (flows0["pay"] <= horizon)
    paid_pv = float(flows0.loc[window, "pv"].sum())
    # Value today = PV of what is paid by the horizon + discounted value then.
    assert swap.pv(m0) == pytest.approx(paid_pv + m0.curve.df(horizon) * swap.pv(m1), rel=1e-11, abs=1e-6)
    assert swap.cash(m1, EXAMPLE_ASOF, horizon) == pytest.approx(float(flows0.loc[window, "amount"].sum()), abs=1e-6)
    flows1 = swap.cashflows(m1)
    assert (flows1["pay"] > horizon).all() and len(flows1) == int((~window).sum())
    np.testing.assert_allclose(flows1["amount"], flows0.loc[~window, "amount"], rtol=1e-11)


def test_cash_counts_coupons_in_the_window():
    market = _market()
    swap = VanillaSwap.from_conventions("USD SOFR", "3Y", 0.04, NOTIONAL, trade_date=EXAMPLE_ASOF)
    flows = swap.cashflows(market)
    first_pay = flows["pay"].min()
    first = float(flows.loc[flows["pay"] == first_pay, "amount"].sum())
    assert swap.cash(market, EXAMPLE_ASOF, first_pay) == pytest.approx(first, rel=1e-14)
    assert swap.cash(market, EXAMPLE_ASOF, first_pay - dt.timedelta(days=1)) == 0.0
    assert swap.cash(market, first_pay, first_pay + dt.timedelta(days=30)) == 0.0
    assert swap.cash(market, EXAMPLE_ASOF, dt.date(2040, 1, 1)) == pytest.approx(flows["amount"].sum(), rel=1e-14)
    # Receiver of 4% against ~3.6% SOFR: the first net coupon is received.
    assert first > 0


def test_swap_without_trade_date_redates_with_the_market():
    floating = VanillaSwap(tenor="2Y")
    later = dt.date(2027, 3, 1)
    assert floating.schedule_dates(later)[0][0] == Calendar("USD").add_business_days(later, 2)
    pinned = replace(floating, trade_date=EXAMPLE_ASOF)
    assert pinned.schedule_dates(later)[0][0] == dt.date(2026, 10, 21)


# --------------------------------------------------------------------------
# Streamlit tab
# --------------------------------------------------------------------------

APP = """
import datetime as dt
import streamlit as st
from derivx.rates.curves import CurveQuote, Market, bootstrap, flat_curve
from derivx.rates.examples import example_market, sofr_example_rows
from derivx.ui.rates_swap import render

conv = st.session_state.get("swap_test_conv")
if st.session_state.get("swap_test_flat"):
    market = Market(curve=flat_curve(dt.date(2026, 10, 19), 0.03))
elif conv:
    shift = -0.045 if conv == "EUR ESTR" else 0.0
    quotes = [CurveQuote(r["kind"], r["tenor"], r["rate"] + shift) for r in sofr_example_rows()]
    market = Market(curve=bootstrap(dt.date(2026, 10, 19), quotes, conv, "linear_zero"))
else:
    market = example_market()
st.session_state["swap_test_positions"] = [p.name for p in render(market)]
"""


def _app(**state) -> AppTest:
    at = AppTest.from_string(APP, default_timeout=120)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _assert_charts_bounded(at: AppTest) -> None:
    charts = at.get("plotly_chart")
    assert charts
    for chart in charts:
        layout = json.loads(chart.proto.spec)["layout"]
        zoom = layout["meta"]["zoom"]["axes"]
        for axis in ("xaxis", "yaxis"):
            lim = zoom[axis]
            assert layout[axis]["minallowed"] == lim["min"] < lim["max"] == layout[axis]["maxallowed"]
            assert layout[axis]["range"] == [lim["min"], lim["max"]] and 0 < lim["min_span"] <= lim["max"] - lim["min"]


def test_swap_tab_default_inputs():
    at = _app()
    assert not at.error
    _assert_charts_bounded(at)
    widgets = ("text_input", "number_input", "radio", "checkbox", "multiselect")
    keys = {w.key for kind in widgets for w in getattr(at, kind)}
    assert keys and all(k.startswith("swap_") for k in keys)
    labels = [m.label for m in at.metric]
    assert {"PV ($)", "PAR RATE (%)", "ANNUITY ($ PER BP)", "PV01 ($, +1BP)", "LADDER RISK REMOVED (%)"} <= set(labels)
    pv = next(m for m in at.metric if m.label == "PV ($)").value
    assert abs(float(pv.replace(",", ""))) < 1.0  # par by default
    assert at.session_state["swap_test_positions"] == ["REC 3.5600% 5Y OIS"]
    assert len(at.dataframe) == 2


def test_swap_tab_edge_inputs():
    at = _app()
    at.text_input(key="swap_tenor").set_value("abc").run()
    assert not at.exception and at.error
    assert at.session_state["swap_test_positions"] == []
    at.text_input(key="swap_tenor").set_value("10D").run()
    assert not at.exception and at.error

    at.text_input(key="swap_tenor").set_value("12Y").run()
    at.radio(key="swap_side").set_value("Pay fixed").run()
    at.radio(key="swap_kind").set_value("IRS").run()
    at.text_input(key="swap_fwd").set_value("2Y").run()
    at.checkbox(key="swap_use_par").uncheck().run()
    at.number_input(key="swap_rate_pct").set_value(4.25).run()
    at.number_input(key="swap_spread_bp").set_value(-25.0).run()
    assert not at.exception and not at.error
    assert at.session_state["swap_test_positions"] == ["PAY 4.2500% 2Y fwd 12Y TERM"]

    at.multiselect(key="swap_hedge_tenors").set_value([]).run()
    assert not at.exception and at.info
    at.multiselect(key="swap_hedge_tenors").set_value(["5Y", "10Y", "15Y"]).run()
    at.checkbox(key="swap_add_hedges").check().run()
    assert not at.exception
    assert len(at.session_state["swap_test_positions"]) == 4

    at.text_input(key="swap_tenor").set_value("50Y").run()
    at.text_input(key="swap_fwd").set_value("30Y").run()
    assert not at.exception and not at.error and at.warning  # pays beyond the last curve quote


def test_swap_tab_maturity_date_and_flat_curve():
    at = _app()
    at.radio(key="swap_end_mode").set_value("Maturity date").run()
    at.date_input(key="swap_maturity").set_value(dt.date(2029, 3, 15)).run()
    assert not at.exception and not at.error
    at.text_input(key="swap_fwd").set_value("5Y").run()
    assert not at.exception and at.error  # maturity before the forward start

    flat = _app(swap_test_flat=True)
    assert not flat.error and flat.info
    assert flat.session_state["swap_test_positions"] == ["REC 3.0038% 5Y OIS"]


@pytest.mark.parametrize("conv", ["EUR ESTR", "GBP SONIA"])
def test_swap_tab_other_conventions(conv):
    """Negative EUR rates and GBP conventions on a linear-zero curve: a par
    receiver loses when rates rise."""

    at = _app(swap_test_conv=conv)
    at.radio(key="swap_kind").set_value("IRS").run()
    assert not at.exception and not at.error
    _assert_charts_bounded(at)
    metrics = {m.label: float(m.value.replace(",", "")) for m in at.metric}
    assert metrics["PV01 ($, +1BP)"] < 0 < metrics["ANNUITY ($ PER BP)"]
    assert (metrics["PAR RATE (%)"] < 0) == (conv == "EUR ESTR")
