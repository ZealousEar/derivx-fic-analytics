"""Multi-curve: dual-curve bootstrap (deposits, FRAs, EURIBOR and SOFR
futures with Hull-White convexity, IRS, 3s6s basis), pricers taking the
discount and projection curves apart, per-curve risk and hedges, and
per-curve scenarios. QuantLib references in tests/golden/multicurve.json
(tools/golden/multicurve.py), plus live QuantLib checks when it is
installed."""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from derivx.rates.capfloor import CapFloor, capfloor_position
from derivx.rates.curves import Curve, CurveQuote, Market, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, example_market, sofr_example_rows
from derivx.rates.indices import (
    HullWhite,
    get_index,
    hw_overnight_futures_convexity,
    hw_period_moments,
    hw_term_futures_convexity,
    imm_date,
    parse_contract,
)
from derivx.rates.multicurve import (
    PROJECTION_KINDS,
    ProjectionSpec,
    bootstrap_projection,
    build_curve_set,
    forward_grid,
    illustrative_quotes,
)
from derivx.rates.quotes import clean_quotes
from derivx.rates.stir import ForwardRateAgreement, RateFuture, fra_position, future_position
from derivx.rates.swaps import (
    BasisSwap,
    VanillaSwap,
    basis_position,
    curve_bumps,
    hedge_multi_curve,
    multi_curve_ladders,
    par_swap_rate,
    risk_curves,
    swap_position,
)
from derivx.risk.scenarios import (
    historical,
    historical_sources,
    ibor_ois_basis,
    parallel,
    run_scenario,
    scenario_grid,
    shocked_market,
    tenor_basis,
    twist,
)
from derivx.risk.sensitivities import bumped_market, key_rate_pv01, parallel_pv01

D = dt.date.fromisoformat
NOTIONAL = 10_000_000.0
REL = 1e-12


# --------------------------------------------------------------------------
# Curve sets (mirrors tools/golden/multicurve.py)
# --------------------------------------------------------------------------


def _eur_ois(asof: dt.date, shift: float) -> Curve:
    """Mirror of tools/golden/multicurve.py:eur_ois."""

    quotes = [CurveQuote(r["kind"], r["tenor"], r["rate"] + shift) for r in sofr_example_rows()]
    return bootstrap(asof, quotes, "EUR ESTR")


def _build(asof: dt.date, quotes: dict, interp6: str = "loglinear", interp3: str = "loglinear", hw=None) -> Market:
    """Mirror of tools/golden/multicurve.py:build."""

    hw = hw or HullWhite(0.03, 0.0085)
    ois = _eur_ois(asof, quotes["ois_shift"])
    c6 = bootstrap_projection(asof, [CurveQuote(*q) for q in quotes["6M"]], "EURIBOR 6M", ois, interp6)
    c3 = bootstrap_projection(asof, [CurveQuote(*q) for q in quotes["3M"]], "EURIBOR 3M", ois, interp3, basis=c6,
                              convexity=hw)  # fmt: skip
    return Market(curve=ois, projections={"EURIBOR 6M": c6, "EURIBOR 3M": c3})


@pytest.fixture(scope="module")
def data(request):
    import json
    from pathlib import Path

    path = Path(__file__).parent / "golden" / "multicurve.json"
    return json.loads(path.read_text())["data"]


def _case(data: dict, name: str = "base", interp: str = "loglinear") -> dict:
    return next(c for c in data["curves"] if c["set"] == name and c["interpolation"] == interp)


@lru_cache(maxsize=None)
def _markets() -> dict:
    """The fixture's curve sets (loglinear), built once."""

    import json
    from pathlib import Path

    data = json.loads((Path(__file__).parent / "golden" / "multicurve.json").read_text())["data"]
    return {name: _build(D(s["asof"]), s["quotes"]) for name, s in data["sets"].items()}


@pytest.fixture(scope="module")
def eur():
    return _markets()["base"]


def _worst_bp(curve: Curve, zeros) -> float:
    return max(abs(curve.zero_rate(D(day)) - z) for day, z in zeros) * 1e4


# --------------------------------------------------------------------------
# Bootstraps against QuantLib
# --------------------------------------------------------------------------


def test_dual_curve_bootstraps_match_quantlib(data):
    """EURIBOR 6M (deposit, FRAs, IRS on OIS discounting) and EURIBOR 3M
    (deposit, futures with convexity, 3s6s basis; and deposit, FRAs, IRS)
    against QuantLib's own bootstraps: same pillars, zero rates within
    1e-8bp."""

    for case in data["curves"]:
        asof, quotes, interp = D(case["asof"]), case["quotes"], case["interpolation"]
        market = _build(asof, quotes, interp, interp)
        c6, c3 = market.projection("EURIBOR 6M"), market.projection("EURIBOR 3M")
        label = (case["set"], interp)
        assert [d.isoformat() for d in c6.pillar_dates] == case["pillars_6m"], label
        assert [d.isoformat() for d in c3.pillar_dates] == case["pillars_3m"], label
        assert _worst_bp(c6, case["zeros_6m"]) < 1e-8, label
        assert _worst_bp(c3, case["zeros_3m"]) < 1e-8, label
        fra_irs = bootstrap_projection(asof, [CurveQuote(*q) for q in quotes["3M_fra_irs"]], "EURIBOR 3M",
                                       market.curve, interp)  # fmt: skip
        assert _worst_bp(fra_irs, case["zeros_3m_fra_irs"]) < 1e-8, label
        ours = {i.quote.tenor: i.convexity_adjustment for i in c3.instruments if i.kind == "FUT"}
        for contract, ca in case["convexity_3m"]:
            assert ours[contract] == pytest.approx(ca, rel=1e-12, abs=1e-15), (label, contract)


def test_sofr_futures_in_the_ois_curve_match_quantlib(data):
    """SR1 (averaged) and SR3 (compounded) futures, two of them part-fixed,
    with OIS swaps: QuantLib's SofrFutureRateHelper/OISRateHelper curve,
    and futures prices against OvernightIndexFuture."""

    sofr = data["sofr"]
    asof = D(sofr["asof"])
    fixings = {D(d): r for d, r in sofr["fixings"]}
    quotes = [CurveQuote(*q) for q in sofr["quotes"]]
    hw = HullWhite(*data["hw"])
    curve = bootstrap(asof, quotes, "USD SOFR", convexity=hw, fixings=fixings)
    assert [d.isoformat() for d in curve.pillar_dates] == sofr["pillars"]
    assert _worst_bp(curve, sofr["zeros"]) < 1e-8
    assert max(abs(r["error_bp"]) for r in curve.repricing()) < 1e-8
    ours = {(i.kind, i.quote.tenor): i.convexity_adjustment for i in curve.instruments}
    for kind, contract, ca in sofr["convexity"]:
        assert ours[(kind, contract)] == pytest.approx(ca, rel=1e-12, abs=1e-15)
    # With SR1 months that end on a weekend or holiday and part-fixed contracts, every future reprices
    # under QuantLib's OvernightIndexFuture, and RateFuture prices them the same way.
    repriced = bootstrap(asof, [CurveQuote(*q) for q in sofr["repriced_quotes"]], "USD SOFR", convexity=hw,
                         fixings=fixings)  # fmt: skip
    assert max(abs(r["error_bp"]) for r in repriced.repricing()) < 1e-8
    market = Market(curve=repriced, index_fixings={"SOFR": fixings})
    quoted = {(q[0], q[1]): 100.0 * (1.0 - q[2]) for q in sofr["repriced_quotes"]}
    for kind, contract, price, _ in sofr["prices"]:
        assert price == pytest.approx(quoted[(kind, contract)], abs=1e-9), (kind, contract)
        fut = RateFuture(kind, contract, 96.0, convexity=hw)
        assert fut.price(market) == pytest.approx(price, abs=1e-9), (kind, contract)
    dec = next(i for i in repriced.instruments if i.quote.tenor == "DEC26" and i.kind == "SR1")
    assert dec.end == dt.date(2027, 1, 1) and dec.pillar == dt.date(2027, 1, 4)  # New Year, then a weekend
    # Rebuilt and bumped curves keep the convexity settings and the fixings.
    assert curve.rebuilt(1.0).instruments[1].convexity_adjustment != 0.0
    assert max(abs(r["error_bp"]) for r in curve.rebuilt(1.0).repricing()) < 1e-8


def test_hull_white_term_convexity_matches_quantlib(data):
    for price, t, T, sigma, a, expected in data["convexity"]:
        ours = hw_term_futures_convexity((100.0 - price) / 100.0, t, T, sigma, a)
        assert ours == pytest.approx(expected, rel=1e-10, abs=1e-15), (price, t, T, sigma, a)


def test_overnight_futures_convexity_closed_forms():
    """The overnight (SR1/SR3) adjustment: Ho-Lee limits, quadrature of the
    defining integrals, agreement with the term formula as the period
    shrinks, and signs (futures rate above forward, compounding above
    averaging)."""

    from scipy.integrate import quad

    s, t1, tau = 0.01, 2.0, 0.25
    var, cov = hw_period_moments(t1, t1 + tau, s, 0.0)
    assert var == pytest.approx(s * s * (tau * tau * t1 + tau**3 / 3.0), rel=1e-13)
    assert cov == pytest.approx(s * s * tau * t1 * t1 / 2.0, rel=1e-13)
    for a in (0.02, 0.4):
        B = lambda x: (1.0 - math.exp(-a * x)) / a  # noqa: E731
        t2 = t1 + tau
        g = lambda u: B(t2 - u) - (B(t1 - u) if u < t1 else 0.0)  # noqa: E731
        v_quad = s * s * (quad(lambda u: g(u) ** 2, 0.0, t1)[0] + quad(lambda u: g(u) ** 2, t1, t2)[0])
        c_quad = s * s * quad(lambda u: B(t1 - u) * g(u), 0.0, t1)[0]
        var, cov = hw_period_moments(t1, t2, s, a)
        assert var == pytest.approx(v_quad, rel=1e-10) and cov == pytest.approx(c_quad, rel=1e-10)
        short = 1e-4
        term = hw_term_futures_convexity(0.03, t1, t1 + short, s, a)
        assert hw_overnight_futures_convexity(0.03, t1, t1 + short, s, a, "compound") == pytest.approx(term, rel=1e-4)
        assert hw_overnight_futures_convexity(0.03, t1, t1 + short, s, a, "average") == pytest.approx(term, rel=1e-3)
    comp = hw_overnight_futures_convexity(0.03, 1.0, 1.25, 0.01, 0.03, "compound")
    avg = hw_overnight_futures_convexity(0.03, 1.0, 1.25, 0.01, 0.03, "average")
    assert comp > avg > 0.0
    assert hw_overnight_futures_convexity(0.03, -0.1, 0.15, 0.01, 0.03) < hw_overnight_futures_convexity(
        0.03, 0.0, 0.25, 0.01, 0.03
    )  # a period under way has less randomness left
    assert hw_overnight_futures_convexity(0.03, 1.0, 1.25, 0.0, 0.03) == 0.0
    with pytest.raises(ValueError, match="mean reversion"):
        HullWhite(5.0, 0.01)
    with pytest.raises(ValueError, match="rate units"):
        HullWhite(0.03, 0.85)
    with pytest.raises(ValueError, match="averaging"):
        hw_overnight_futures_convexity(0.03, 1.0, 1.25, 0.01, 0.03, "geometric")


# --------------------------------------------------------------------------
# Pricers against QuantLib
# --------------------------------------------------------------------------


def _swap_from(spec, trade: dt.date) -> VanillaSwap:
    _, index, tenor, fwd, rate, receive, spread, _ = spec
    return VanillaSwap.from_index(index, tenor, rate, notional=NOTIONAL, receive_fixed=receive, forward_start=fwd,
                                  spread=spread, trade_date=trade)  # fmt: skip


def _set_market(data: dict, name: str, interp6: str = "loglinear") -> Market:
    if interp6 == "loglinear":
        return _markets()[name]
    return _build(D(data["sets"][name]["asof"]), data["sets"][name]["quotes"], interp6)


def test_euribor_swaps_on_ois_discounting_match_quantlib(data):
    for case in data["swaps"]:
        spec = case["spec"]
        market = _set_market(data, spec[0], spec[7])
        swap = _swap_from(spec, D(case["trade_date"]))
        fixed_dates, float_dates = swap.schedule_dates(market.asof)
        assert [d.isoformat() for d in fixed_dates] == case["fixed_dates"], spec
        assert [d.isoformat() for d in float_dates] == case["float_dates"], spec
        sign = 1.0 if swap.receive_fixed else -1.0
        assert swap.pv(market) == pytest.approx(case["npv"], rel=REL, abs=1e-8), spec
        assert swap.par_rate(market) == pytest.approx(case["fair_rate"], rel=REL), spec
        assert sign * swap.annuity(market) == pytest.approx(case["fixed_leg_bps"], rel=REL), spec
        assert swap.pv_float(market) == pytest.approx(case["float_leg_npv"], rel=REL, abs=1e-8), spec


def test_seasoned_euribor_swaps_use_published_fixings(data):
    market = _markets()["seasoned"]
    for case in data["seasoned_swaps"]:
        spec = case["spec"]
        fixings = {D(d): r for d, r in case["fixings"]}
        assert fixings, "a seasoned swap has past fixings"
        swap = _swap_from(spec, D(case["trade_date"]))
        with pytest.raises(ValueError, match="index_fixings"):
            swap.pv(market)  # a past fixing is never guessed from today's curve
        priced = market.replace(index_fixings={spec[1]: fixings})
        assert swap.pv(priced) == pytest.approx(case["npv"], rel=REL, abs=1e-8), spec
        assert swap.par_rate(priced) == pytest.approx(case["fair_rate"], rel=REL), spec


def test_basis_swaps_match_quantlib(data):
    for case in data["basis"]:
        name, tenor, fwd, spread, receive = case["spec"]
        market = _set_market(data, name)
        swap = BasisSwap(NOTIONAL, spread, tenor, forward_start=fwd, receive_spread_leg=receive, trade_date=market.asof)
        periods = swap.periods(market.asof)
        spread_dates = [p.start for p in periods if p.leg == "spread"][:1] + [
            p.end for p in periods if p.leg == "spread"
        ]
        other_dates = [p.start for p in periods if p.leg == "other"][:1] + [p.end for p in periods if p.leg == "other"]
        assert [d.isoformat() for d in spread_dates] == case["spread_dates"]
        assert [d.isoformat() for d in other_dates] == case["other_dates"]
        assert swap.pv(market) == pytest.approx(case["npv"], rel=REL, abs=1e-8), case["spec"]
        assert swap.par_spread(market) == pytest.approx(case["fair_spread"], rel=REL, abs=1e-15), case["spec"]
        assert swap.annuity(market) == pytest.approx(case["spread_leg_bps"], rel=REL), case["spec"]


def test_euribor_caps_with_ois_discounting_match_quantlib(data):
    for case in data["caps"]:
        name, index, kind, tenor, fwd, _, model, vol = case["spec"]
        market = _set_market(data, name)
        cap = CapFloor.from_index(index, NOTIONAL, kind, case["strike"], tenor, vol, model, forward_start=fwd)
        periods = cap.periods(market.asof)
        assert [periods[0].start.isoformat()] + [p.end.isoformat() for p in periods] == case["dates"]
        assert cap.atm_strike(market) == pytest.approx(case["atm"], rel=REL)
        assert cap.pv(market) == pytest.approx(case["npv"], rel=REL, abs=1e-8), case["spec"]


def test_fras_match_quantlib(data):
    for case in data["fras"]:
        name, index, months, rate, pay_fixed = case["spec"]
        market = _set_market(data, name)
        fra = ForwardRateAgreement(index, rate, months_to_start=months, notional=NOTIONAL, pay_fixed=pay_fixed)
        assert fra.value_date(market.asof).isoformat() == case["value_date"]
        assert fra.forward(market) == pytest.approx(case["forward"], rel=REL)
        assert fra.settlement(market) == pytest.approx(case["amount"], rel=REL)
        assert fra.pv(market) == pytest.approx(case["npv"], rel=REL, abs=1e-8)


def test_euribor_futures_price_from_the_3m_curve(data, eur):
    hw = HullWhite(*data["hw"])
    for contract, fwd in data["euribor_futures"]:
        fut = RateFuture("FUT", contract, 97.0, contracts=-5, convexity=hw)
        assert fut.forward(eur) == pytest.approx(fwd, rel=REL)
        start, end = fut.period(eur.asof)
        t1, t2 = (start - eur.asof).days / 365.0, (end - eur.asof).days / 365.0
        rate = fut.futures_rate(eur)
        assert rate - fwd == pytest.approx(hw_term_futures_convexity(rate, t1, t2, hw.sigma, hw.mean_reversion),
                                           rel=1e-9)  # fmt: skip
        assert fut.pv(eur) == pytest.approx(-5 * 2500.0 * (100.0 * (1 - rate) - 97.0), rel=1e-12)
    # The curve's own futures reprice their quotes.
    c3 = eur.projection("EURIBOR 3M")
    for inst in (i for i in c3.instruments if i.kind == "FUT"):
        fut = RateFuture("FUT", inst.quote.tenor, 97.0, convexity=hw)
        assert fut.futures_rate(eur) == pytest.approx(inst.quote.rate, abs=1e-12)


def test_random_multi_curve_trades_match_live_quantlib(eur):
    """Random EURIBOR swaps, basis swaps and caps priced by live QuantLib
    on copies of the derivx curves."""

    ql = pytest.importorskip("QuantLib")
    from derivx.rates.conventions import add_months

    def handle(curve):
        dates = [ql.Date(eur.asof.day, eur.asof.month, eur.asof.year)] + [
            ql.Date(d.day, d.month, d.year) for d in curve.pillar_dates
        ]
        ts = ql.DiscountCurve(dates, [float(x) for x in curve.dfs], ql.Actual365Fixed())
        ts.enableExtrapolation()
        return ql.YieldTermStructureHandle(ts)

    qd = lambda d: ql.Date(d.day, d.month, d.year)  # noqa: E731
    ql.Settings.instance().evaluationDate = qd(eur.asof)
    ois = handle(eur.curve)
    indices = {"EURIBOR 6M": ql.Euribor6M(handle(eur.projection("EURIBOR 6M"))),
               "EURIBOR 3M": ql.Euribor3M(handle(eur.projection("EURIBOR 3M")))}  # fmt: skip
    rng = random.Random(7)
    for _ in range(6):
        name = rng.choice(list(indices))
        years = rng.choice([2, 4, 7, 12, 25])
        fwd = rng.choice([None, "6M", "3Y"])
        rate = 0.02 + 0.01 * rng.random()
        swap = VanillaSwap.from_index(name, f"{years}Y", rate, notional=NOTIONAL, forward_start=fwd,
                                      receive_fixed=rng.random() < 0.5, trade_date=eur.asof)  # fmt: skip
        fixed, flt = swap.schedule_dates(eur.asof)
        out = ql.VanillaSwap(
            ql.Swap.Receiver if swap.receive_fixed else ql.Swap.Payer, NOTIONAL,
            ql.Schedule(ql.DateVector([qd(d) for d in fixed])), rate, ql.Thirty360(ql.Thirty360.BondBasis),
            ql.Schedule(ql.DateVector([qd(d) for d in flt])), indices[name], 0.0, ql.Actual360(),
        )  # fmt: skip
        out.setPricingEngine(ql.DiscountingSwapEngine(ois))
        assert swap.pv(eur) == pytest.approx(out.NPV(), rel=REL, abs=1e-8)
        assert swap.par_rate(eur) == pytest.approx(out.fairRate(), rel=1e-12)
        # A cap on the same index and dates
        cap = CapFloor.from_index(name, NOTIONAL, "cap", rate, f"{years}Y", 0.009, "bachelier")
        periods = cap.periods(eur.asof)
        leg = ql.IborLeg([NOTIONAL], ql.Schedule(ql.DateVector([qd(periods[0].start)] + [qd(p.end) for p in periods])),
                         indices[name], ql.Actual360(), ql.ModifiedFollowing, [2])  # fmt: skip
        inst = ql.Cap(leg, [rate])
        surface = ql.ConstantOptionletVolatility(qd(eur.asof), ql.TARGET(), ql.Following, 0.009, ql.Actual365Fixed(),
                                                 ql.Normal, 0.0)  # fmt: skip
        inst.setPricingEngine(ql.BachelierCapFloorEngine(ois, ql.OptionletVolatilityStructureHandle(surface)))
        assert cap.pv(eur) == pytest.approx(inst.NPV(), rel=REL, abs=1e-8)
        assert add_months(eur.asof, 1) > eur.asof


# --------------------------------------------------------------------------
# Structure: single curve, laziness, fixings, validation
# --------------------------------------------------------------------------


def test_single_curve_set_prices_exactly_as_term(eur):
    """A curve set whose projection is the discount curve is single-curve:
    a EURIBOR swap or cap on it equals the TERM (single-curve) trade, and
    follows the discount curve when it is rebuilt."""

    ois = eur.curve
    single = Market(curve=ois, projections={"EURIBOR 6M": ois})
    assert not single.is_multi_curve and eur.is_multi_curve
    proj = VanillaSwap.from_index("EURIBOR 6M", "7Y", 0.02, trade_date=eur.asof)
    term = replace(proj, projection=None)
    assert proj.pv(single) == pytest.approx(term.pv(Market(curve=ois)), rel=1e-14)
    cap = CapFloor.from_index("EURIBOR 6M", NOTIONAL, "cap", 0.02, "5Y", 0.009)
    assert cap.pv(single) == pytest.approx(replace(cap, projection=None).pv(Market(curve=ois)), rel=1e-14)
    bumped = single.rebuilt({"curve": 1.0})
    assert bumped.projection("EURIBOR 6M") is bumped.curve
    assert parallel_pv01(proj.pv, single) == pytest.approx(parallel_pv01(term.pv, Market(curve=ois)), rel=1e-12)
    # Multi-curve differs: EURIBOR projects above the OIS curve.
    assert proj.par_rate(eur) > proj.par_rate(single) + 10e-4


def test_every_quote_reprices_and_curve_sets_build_in_dependency_order(eur):
    for key in ("EURIBOR 6M", "EURIBOR 3M"):
        curve = eur.projection(key)
        assert max(abs(r["error_bp"]) for r in curve.repricing()) < 1e-8
        for interp in ("linear_zero", "monotone_convex"):
            other = bootstrap_projection(eur.asof, curve.quotes, key, eur.curve, interp, basis=curve.basis,
                                         convexity=curve.convexity)  # fmt: skip
            assert max(abs(r["error_bp"]) for r in other.repricing()) < 1e-8, (key, interp)
    assert eur.curve_keys == ("curve", "EURIBOR 6M", "EURIBOR 3M")
    c6, c3 = eur.projection("EURIBOR 6M"), eur.projection("EURIBOR 3M")
    built = build_curve_set(
        eur.curve,
        [ProjectionSpec("EURIBOR 3M", tuple(c3.quotes), basis_index="EURIBOR 6M"), ProjectionSpec("EURIBOR 6M",
                                                                                                    tuple(c6.quotes))],
        convexity=c3.convexity,
    )  # fmt: skip
    assert built.curve_keys == eur.curve_keys
    assert np.allclose(built.projection("EURIBOR 3M").dfs, c3.dfs, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="say which index"):
        build_curve_set(eur.curve, [ProjectionSpec("EURIBOR 3M", tuple(c3.quotes))])


def test_rebuilt_markets_are_lazy_and_rebuild_projections_on_the_new_discount_curve(eur):
    moved = eur.rebuilt({"curve": 5.0})
    assert not moved.projections.is_built("EURIBOR 6M") and not moved.projections.is_built("EURIBOR 3M")
    c6 = moved.projection("EURIBOR 6M")
    assert moved.projections.is_built("EURIBOR 6M") and not moved.projections.is_built("EURIBOR 3M")
    assert c6.discount is moved.curve and c6.quotes == eur.projection("EURIBOR 6M").quotes
    assert max(abs(r["error_bp"]) for r in c6.repricing()) < 1e-8  # quotes held, repriced on the new OIS curve
    c3 = moved.projection("EURIBOR 3M")
    assert c3.basis is c6 and max(abs(r["error_bp"]) for r in c3.repricing()) < 1e-8
    # A bump of the 6M quotes leaves the discount curve alone and moves the 3M curve built on it.
    six = eur.rebuilt({"EURIBOR 6M": 1.0})
    assert six.curve is eur.curve and six.projection("EURIBOR 3M").basis is six.projection("EURIBOR 6M")
    fwd = lambda m, k: m.projection(k).forward(dt.date(2030, 1, 15))  # noqa: E731
    # Par rates up 1bp on an annual 30/360 leg lift the ACT/360 forwards by about 360/365 bp.
    assert fwd(six, "EURIBOR 3M") - fwd(eur, "EURIBOR 3M") == pytest.approx(1e-4 * 360 / 365, rel=0.02)
    # Only-later quotes moved: the earlier pillars are reused exactly.
    later = eur.projection("EURIBOR 6M").rebuilt({14: 1.0})
    assert np.array_equal(later.dfs[:15], eur.projection("EURIBOR 6M").dfs[:15])  # t = 0 and 14 pillars
    assert later.dfs[15] != eur.projection("EURIBOR 6M").dfs[15]
    with pytest.raises(ValueError, match="no \\['EURIBOR 1M'\\] curve"):
        eur.rebuilt({"EURIBOR 1M": 1.0})


def test_missing_projection_and_fixings_raise_clear_errors(eur):
    swap = VanillaSwap.from_index("EURIBOR 6M", "5Y", 0.02, trade_date=eur.asof)
    with pytest.raises(ValueError, match="no EURIBOR 6M projection curve"):
        swap.pv(Market(curve=eur.curve))
    with pytest.raises(ValueError, match="index_fixings"):
        eur.index_rate("EURIBOR 6M", dt.date(2026, 4, 21), dt.date(2026, 10, 21), "ACT/360", dt.date(2026, 4, 17))
    published = eur.replace(index_fixings={"euribor 6m": {dt.date(2026, 4, 17): 0.0251}})
    assert published.index_rate("EURIBOR 6M", dt.date(2026, 4, 21), dt.date(2026, 10, 21), "ACT/360",
                                dt.date(2026, 4, 17)) == 0.0251  # fmt: skip
    today = eur.replace(index_fixings={"EURIBOR 6M": {eur.asof: 0.0301}})
    start = get_index("EURIBOR 6M").value_date(eur.asof)
    end = get_index("EURIBOR 6M").maturity_date(start)
    assert today.index_rate("EURIBOR 6M", start, end, "ACT/360", eur.asof) == 0.0301  # today's fixing, if published
    assert eur.index_rate("EURIBOR 6M", start, end, "ACT/360", eur.asof) == pytest.approx(
        eur.projection("EURIBOR 6M").forward(start), rel=1e-15
    )
    with pytest.raises(ValueError, match="reserved"):
        Market(curve=eur.curve, projections={"curve": eur.curve})
    with pytest.raises(ValueError, match="dated"):
        bootstrap_projection(dt.date(2026, 10, 20), eur.projection("EURIBOR 6M").quotes, "EURIBOR 6M", eur.curve)
    with pytest.raises(ValueError, match="EUR index"):
        bootstrap_projection(EXAMPLE_ASOF, eur.projection("EURIBOR 6M").quotes, "EURIBOR 6M",
                             example_market().curve)  # fmt: skip


@pytest.mark.parametrize(
    "kind, tenor, rate, match",
    [
        ("FRA", "3X6", 0.02, "spans 3 months"),
        ("FRA", "6-12", 0.02, "months to start"),
        ("FUT", "DEC26", 0.02, "no futures on EURIBOR 6M"),
        ("BASIS", "5Y", 0.001, "other index"),
        ("DEPO", "ON", 0.02, "weeks, months or years"),
        ("OIS", "5Y", 0.02, "expected one of"),
    ],
)
def test_projection_instruments_reject_bad_quotes(eur, kind, tenor, rate, match):
    with pytest.raises(ValueError, match=match):
        bootstrap_projection(eur.asof, [CurveQuote(kind, tenor, rate)], "EURIBOR 6M", eur.curve)


def test_futures_contracts_dates_and_validation(eur):
    assert parse_contract("DEC26") == (2026, 12) == parse_contract("Z26") == parse_contract("2026-12")
    assert parse_contract("z6", EXAMPLE_ASOF) == (2026, 12) and parse_contract("H5", EXAMPLE_ASOF) == (2035, 3)
    assert imm_date(2026, 12) == dt.date(2026, 12, 16) and imm_date(2027, 3) == dt.date(2027, 3, 17)
    for bad in ("DEC", "Q2026", "2026-13", "XYZ26"):
        with pytest.raises(ValueError, match="contract"):
            parse_contract(bad)
    with pytest.raises(ValueError, match="already fixed"):
        bootstrap_projection(eur.asof, [CurveQuote("FUT", "SEP26", 0.02)], "EURIBOR 3M", eur.curve)
    with pytest.raises(ValueError, match="expired"):
        bootstrap(EXAMPLE_ASOF, [CurveQuote("SR1", "SEP26", 0.04)], "USD SOFR")
    with pytest.raises(ValueError, match="USD SOFR curve"):
        bootstrap(EXAMPLE_ASOF, [CurveQuote("SR3", "DEC26", 0.02)], "EUR ESTR")
    with pytest.raises(ValueError, match="overnight fixing"):  # started: needs SOFR fixings
        bootstrap(EXAMPLE_ASOF, [CurveQuote("SR1", "OCT26", 0.039), CurveQuote("OIS", "1Y", 0.037)], "USD SOFR")
    with pytest.raises(ValueError, match="SOFR future"):
        RateFuture("SR3", "DEC26", 96.0, index="EURIBOR 3M")
    with pytest.raises(ValueError, match="non-zero"):
        RateFuture("FUT", "DEC26", 96.0, contracts=0)
    with pytest.raises(ValueError, match="USD SOFR market"):
        RateFuture("SR3", "DEC26", 96.0).price(eur)


def test_clean_quotes_reads_futures_prices_fra_labels_and_basis_spreads():
    rows = [
        {"instrument": "DEPO", "tenor": "3M", "rate": 2.83},
        {"instrument": "FUT", "tenor": "z6", "rate": 97.275},
        {"instrument": "FRA", "tenor": "6 x 9", "rate": 2.5},
        {"instrument": "IRS", "tenor": "2Y", "rate": 2.6},
        {"instrument": "BASIS", "tenor": "5Y", "rate": 0.08},
    ]
    quotes, issues, units = clean_quotes(rows, kinds=PROJECTION_KINDS, asof=EXAMPLE_ASOF)
    assert units == "percent" and not [i for i in issues if i.level == "error"]
    assert [(q.kind, q.tenor) for q in quotes] == [("DEPO", "3M"), ("FUT", "DEC26"), ("FRA", "6X9"), ("IRS", "2Y"),
                                                   ("BASIS", "5Y")]  # fmt: skip
    assert quotes[1].rate == pytest.approx(0.02725) and quotes[4].rate == pytest.approx(0.0008)
    bad = rows + [{"instrument": "FUT", "tenor": "DEC26", "rate": 97.0}, {"instrument": "FUT", "tenor": "MAR27",
                                                                          "rate": 3.2}]  # fmt: skip
    _, issues, _ = clean_quotes(bad, kinds=PROJECTION_KINDS, asof=EXAMPLE_ASOF)
    errors = [i.message for i in issues if i.level == "error"]
    assert any("repeats" in e for e in errors) and any("100 - rate" in e for e in errors)
    _, issues, _ = clean_quotes([{"instrument": "BASIS", "tenor": "5Y", "rate": 8.0}], kinds=PROJECTION_KINDS)
    assert any("units" in i.message for i in issues if i.level == "error")
    quotes, _, _ = clean_quotes(
        [{"instrument": "BASIS", "tenor": "5Y", "rate": 8.0}], units="bp", kinds=PROJECTION_KINDS
    )
    assert quotes[0].rate == pytest.approx(0.0008)


def test_illustrative_quotes_follow_the_discount_curve_and_assumed_basis(eur):
    c6 = eur.projection("EURIBOR 6M")
    quotes = illustrative_quotes(eur.curve, "EURIBOR 6M")
    curve = bootstrap_projection(eur.asof, quotes, "EURIBOR 6M", eur.curve)
    grid = forward_grid(Market(curve=eur.curve, projections={"EURIBOR 6M": curve}), 6, 20.0)
    basis_bp = (grid["EURIBOR 6M"] - grid["ois"]) * 1e4
    assert 8.0 < basis_bp.min() and basis_bp.max() < 30.0  # around the assumed 24 -> 10bp
    assert max(abs(r["error_bp"]) for r in curve.repricing()) < 1e-8
    assert c6.index.name == "EURIBOR 6M"


# --------------------------------------------------------------------------
# Per-curve risk and hedges
# --------------------------------------------------------------------------


def test_key_rate_risk_is_bucketed_per_curve(eur):
    """A par IRS vs 6M has risk only to its own 6M quote (quote risk); an
    off-par one also to the OIS curve; the curve ladders sum to the
    parallel PV01 of moving every curve together; signs follow +1bp."""

    par_rate = par_swap_rate(eur, "10Y", index="EURIBOR 6M")
    par = VanillaSwap.from_index("EURIBOR 6M", "10Y", par_rate, notional=NOTIONAL, trade_date=eur.asof)
    ladders = multi_curve_ladders([par.pv], eur)
    assert set(ladders) == set(risk_curves(eur)) == {"curve", "EURIBOR 6M", "EURIBOR 3M"}
    assert np.allclose(ladders["curve"][1], 0.0, atol=1e-6) and np.allclose(ladders["EURIBOR 3M"][1], 0.0)
    six = ladders["EURIBOR 6M"][1][:, 0]
    i10 = [r["label"] for r in ladders["EURIBOR 6M"][0]].index("10Y IRS")
    assert six[i10] == pytest.approx(-par.annuity(eur), rel=1e-6) and np.allclose(np.delete(six, i10), 0.0, atol=1e-6)

    off = replace(par, fixed_rate=par_rate + 0.01)
    ladders = multi_curve_ladders([off.pv], eur)
    disc = ladders["curve"][1][:, 0]
    assert disc.sum() < 0  # a receiver above par gains less as OIS discounting rises: its PV is positive
    total = sum(m[:, 0].sum() for _, m in ladders.values())
    parallel_all = (off.pv(eur.rebuilt({"curve": 1.0, "EURIBOR 6M": 1.0, "EURIBOR 3M": 1.0}))
                    - off.pv(eur.rebuilt({"curve": -1.0, "EURIBOR 6M": -1.0, "EURIBOR 3M": -1.0}))) / 2.0  # fmt: skip
    assert total == pytest.approx(parallel_all, rel=1e-6)
    # The generic sensitivities take projection keys too.
    rows = key_rate_pv01(off.pv, eur, "EURIBOR 6M")
    assert [r["pv01"] for r in rows] == pytest.approx(list(ladders["EURIBOR 6M"][1][:, 0]), rel=1e-12, abs=1e-9)
    with pytest.raises(ValueError, match="which must be one of"):
        key_rate_pv01(off.pv, eur, "EURIBOR 12M")


def test_a_3m_trade_carries_6m_and_basis_risk_and_hedges_flatten_every_curve(eur):
    swap = VanillaSwap.from_index("EURIBOR 3M", "7Y", 0.0255, notional=NOTIONAL, receive_fixed=False,
                                  trade_date=eur.asof)  # fmt: skip
    position = swap_position(swap)
    ladders = multi_curve_ladders([swap.pv], eur)
    # Near par, a 7Y swap vs 3M is a 7Y swap vs 6M less a 7Y 3s6s basis swap: a payer gains on the 7Y IRS
    # quote (the 3M curve rides on the 6M one) and loses on the 7Y basis (a wider spread lowers 3M vs 6M).
    six = dict(zip((r["label"] for r in ladders["EURIBOR 6M"][0]), ladders["EURIBOR 6M"][1][:, 0]))
    three = dict(zip((r["label"] for r in ladders["EURIBOR 3M"][0]), ladders["EURIBOR 3M"][1][:, 0]))
    assert six["7Y IRS"] > 1000.0 and three["7Y BASIS"] < -1000.0
    assert max(abs(v) for k, v in six.items() if k != "7Y IRS") < 1.0
    assert max(abs(v) for k, v in three.items() if k != "7Y BASIS") < 1.0
    benchmarks = {"curve": ["2Y", "5Y", "10Y"], "EURIBOR 6M": ["2Y", "5Y", "7Y", "10Y"], "EURIBOR 3M": ["5Y", "7Y"]}
    bumps = {k: curve_bumps(eur, k, parallel=False) for k in risk_curves(eur)}
    hedge = hedge_multi_curve(position, eur, benchmarks, bumps=bumps)
    assert hedge["risk_removed"] > 0.99
    assert {r["curve"] for r in hedge["hedges"] if r["notional"]} >= {"EURIBOR 6M", "EURIBOR 3M"}
    total = sum(abs(r["position"]) for rows in hedge["ladders"].values() for r in rows)
    for key, rows in hedge["ladders"].items():
        assert abs(sum(r["residual"] for r in rows)) < 1e-3 * total, key
    # Hedge positions are real trades: the hedged book's ladders are the residuals.
    book = [position] + hedge["positions"]
    ladders = multi_curve_ladders([lambda m: sum(p.pv(m) for p in book)], eur, bumps=bumps)
    for key, rows in hedge["ladders"].items():
        assert ladders[key][1][:, 0] == pytest.approx([r["residual"] for r in rows], abs=1e-4)
    with pytest.raises(ValueError, match="distinct"):
        hedge_multi_curve(position, eur, {"curve": ["5Y", "5Y"]})


def test_basis_swap_and_cap_positions_risk(eur):
    basis = BasisSwap(NOTIONAL, 0.0, "10Y", trade_date=eur.asof)
    basis = replace(basis, spread=basis.par_spread(eur))
    assert basis.pv(eur) == pytest.approx(0.0, abs=1e-6)
    ladders = multi_curve_ladders([basis_position(basis).pv], eur)
    i10 = [r["label"] for r in ladders["EURIBOR 3M"][0]].index("10Y BASIS")
    # It receives a fixed spread: a wider market spread (the quote +1bp) costs one annuity.
    assert ladders["EURIBOR 3M"][1][i10, 0] == pytest.approx(-basis.annuity(eur), rel=1e-6)
    cap = capfloor_position(CapFloor.from_index("EURIBOR 6M", NOTIONAL, "cap", 0.026, "5Y", 0.009), asof=eur.asof)
    ladders = multi_curve_ladders([cap.pv], eur)
    assert ladders["EURIBOR 6M"][1].sum() > 0 and np.allclose(ladders["EURIBOR 3M"][1], 0.0)
    assert ladders["curve"][1].sum() != 0.0  # OIS discounting


# --------------------------------------------------------------------------
# Scenarios per curve
# --------------------------------------------------------------------------


def test_shocks_move_curves_separately_and_basis_spreads_only_when_asked(eur):
    par = shocked_market(eur, parallel(10.0))
    assert par.curve.quotes[5].rate - eur.curve.quotes[5].rate == pytest.approx(1e-3)
    basis_before = [q.rate for q in eur.projection("EURIBOR 3M").quotes if q.kind == "BASIS"]
    assert [q.rate for q in par.projection("EURIBOR 3M").quotes if q.kind == "BASIS"] == basis_before
    assert par.projection("EURIBOR 6M").quotes[10].rate - eur.projection("EURIBOR 6M").quotes[10].rate == pytest.approx(
        1e-3
    )

    wide = shocked_market(eur, ibor_ois_basis(5.0))
    assert wide.curve is eur.curve
    assert wide.projection("EURIBOR 6M").quotes[10].rate - eur.projection("EURIBOR 6M").quotes[
        10
    ].rate == pytest.approx(5e-4)
    only6 = shocked_market(eur, ibor_ois_basis(5.0, indices=["EURIBOR 6M"]))
    assert [q.rate for q in only6.projection("EURIBOR 3M").quotes] == [
        q.rate for q in eur.projection("EURIBOR 3M").quotes
    ]

    tb = shocked_market(eur, tenor_basis(2.0))
    assert tb.curve is eur.curve and tb.projection("EURIBOR 6M").quotes == eur.projection("EURIBOR 6M").quotes
    moved = [b.rate - a.rate for a, b in zip(eur.projection("EURIBOR 3M").quotes, tb.projection("EURIBOR 3M").quotes)]
    kinds = [q.kind for q in eur.projection("EURIBOR 3M").quotes]
    assert all((m == pytest.approx(2e-4)) if k == "BASIS" else m == 0.0 for m, k in zip(moved, kinds))
    assert not parallel(5.0, curve=False, govt=False).moves_rates and ibor_ois_basis(1.0).moves_rates


def test_explain_by_curve_and_basis_pnl(eur):
    swap6 = swap_position(VanillaSwap.from_index("EURIBOR 6M", "10Y", 0.028, notional=NOTIONAL, trade_date=eur.asof))
    basis = basis_position(BasisSwap(NOTIONAL, 0.0008, "10Y", trade_date=eur.asof))
    fra = fra_position(ForwardRateAgreement.from_label("6X12", "EURIBOR 6M", 0.025, notional=NOTIONAL,
                                                       trade_date=eur.asof))  # fmt: skip
    fut = future_position(RateFuture("FUT", "MAR27", 97.3, contracts=20, convexity=HullWhite()))
    positions = [swap6, basis, fra, fut]
    for shock in (twist(5.0, -10.0, 10.0), ibor_ois_basis(3.0), tenor_basis(1.0)):
        rows = run_scenario(positions, eur, shock)
        for row in rows:
            assert sum(row["delta_by_curve"].values()) == pytest.approx(row["delta"], rel=1e-12, abs=1e-12)
            assert abs(row["residual"]) < 1e-3 * max(1.0, abs(row["pnl"])), (shock.name, row)
    tb = {r["position"]: r for r in run_scenario(positions, eur, tenor_basis(1.0))}
    assert tb[swap6.name]["pnl"] == 0.0 and set(tb[basis.name]["delta_by_curve"]) == {"EURIBOR 3M"}
    assert tb[basis.name]["pnl"] < 0  # receiving 3M + a fixed 8bp: the market spread widened
    wide = {r["position"]: r for r in run_scenario(positions, eur, ibor_ois_basis(3.0))}
    assert wide[swap6.name]["delta_by_curve"]["EURIBOR 3M"] == 0.0 and wide[swap6.name]["pnl"] < 0
    assert "curve" not in wide[swap6.name]["delta_by_curve"]
    assert wide[fut.name]["pnl"] < 0 and wide[fra.name]["pnl"] > 0  # long futures lose, FRA buyer gains
    grid = scenario_grid(positions, eur, [parallel(25.0), ibor_ois_basis(5.0), tenor_basis(2.0)])
    assert grid.shape == (3, 5) and np.isfinite(grid.values).all()


def test_carry_rolls_projection_curves_and_uses_their_fixings(eur):
    swap = swap_position(VanillaSwap.from_index("EURIBOR 6M", "5Y", 0.027, notional=NOTIONAL, trade_date=eur.asof))
    rows = run_scenario([swap], eur, parallel(0.0, horizon_days=200))
    assert math.isfinite(rows[0]["carry"]) and rows[0]["cash"] != 0.0  # the first fixing paid in the window
    rolled = shocked_market(eur, parallel(0.0, horizon_days=200))
    assert "EURIBOR 6M" in rolled.projection_fixings and rolled.projection("EURIBOR 6M").asof == rolled.asof


def test_historical_replays_move_projections_with_the_swap_curve_and_say_so(eur):
    shock = historical("2022 hiking cycle")
    table = historical_sources(shock)
    proj = table[table["curve"] == "PROJECTION"]
    assert list(proj["currency"]) == ["EUR"] and (proj["flag"] == "proxy").all()
    assert "basis are held" in proj["note"].iloc[0]
    moved = shocked_market(eur, shock)
    d_ois = moved.curve.quotes[-1].rate - eur.curve.quotes[-1].rate
    d_6m = moved.projection("EURIBOR 6M").quotes[-1].rate - eur.projection("EURIBOR 6M").quotes[-1].rate
    assert d_ois != 0.0 and d_6m == pytest.approx(d_ois, abs=1e-12)  # same 30Y move: basis held


def test_bumped_market_matches_rebuilt_and_keeps_single_curve_behaviour():
    usd = example_market()
    assert usd.curve_keys == ("curve", "govt") and not usd.is_multi_curve
    bumped = bumped_market(usd, "curve", {3: 1.0})
    assert bumped.fixings is usd.curve and bumped.govt is usd.govt
    assert np.array_equal(bumped.dfs if hasattr(bumped, "dfs") else bumped.curve.dfs, usd.curve.rebuilt({3: 1.0}).dfs)
