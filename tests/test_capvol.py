"""Caplet vols stripped from flat cap vols: QuantLib's OptionletStripper1,
repricing of every quoted cap, caplet smiles by strike and vega buckets."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from derivx.rates.capfloor import CapFloor
from derivx.rates.capvol import (
    CapletVolCurve,
    CapletVolSurface,
    quote_vegas,
    strip_caplet_vols,
    strip_surface,
    vega_by_expiry,
)
from derivx.rates.curves import CurveQuote, Market, bootstrap
from derivx.rates.examples import example_market, sofr_example_rows
from derivx.rates.indices import HullWhite
from derivx.rates.multicurve import bootstrap_projection
from derivx.rates.vol_examples import CAP_STRIKES_PCT, illustrative_cap_atm, illustrative_cap_strikes

D = __import__("datetime").date.fromisoformat
USD_TERMS = {"index_months": 3, "day_count": "ACT/360", "calendar": "USD", "fixing_lag": 2}
EUR_TERMS = {"index_months": 6, "day_count": "ACT/360", "calendar": "TARGET", "fixing_lag": 2,
             "projection": "EURIBOR 6M"}  # fmt: skip


@lru_cache(maxsize=None)
def _eur() -> Market:
    """The tools/golden/multicurve.py base set (stored in swaptions.json)."""

    sets = json.loads((Path(__file__).parent / "golden" / "swaptions.json").read_text())["data"]["sets"]
    asof, quotes = D(sets["base"]["asof"]), sets["base"]["quotes"]
    ois = bootstrap(asof, [CurveQuote(r["kind"], r["tenor"], r["rate"] + quotes["ois_shift"])
                           for r in sofr_example_rows()], "EUR ESTR")  # fmt: skip
    c6 = bootstrap_projection(asof, [CurveQuote(*q) for q in quotes["6M"]], "EURIBOR 6M", ois)
    c3 = bootstrap_projection(asof, [CurveQuote(*q) for q in quotes["3M"]], "EURIBOR 3M", ois, basis=c6,
                              convexity=HullWhite(0.03, 0.0085))  # fmt: skip
    return Market(curve=ois, projections={"EURIBOR 6M": c6, "EURIBOR 3M": c3})


@pytest.fixture(scope="module")
def usd() -> Market:
    return example_market()


# --------------------------------------------------------------------------
# QuantLib
# --------------------------------------------------------------------------


def test_piecewise_constant_strip_matches_quantlib_optionlet_stripper1(golden, usd):
    """With a quote at every caplet maturity, QuantLib's OptionletStripper1
    strips caplet by caplet: derivx's piecewise-constant strip gives the
    same caplet vols (normal and shifted lognormal, single- and dual-curve),
    and the same fixing dates."""

    for case in golden("capvol")["stripper1"]:
        market = usd if case["market"] == "usd" else _eur()
        terms = USD_TERMS if case["market"] == "usd" else EUR_TERMS
        flat = np.array(case["flat_vols"])
        expected = np.array(case["caplet_vols"])  # expiries x strikes
        for j, strike in enumerate(case["strikes"]):
            result = strip_caplet_vols(market, case["tenors"], flat[:, j], strike, "constant", case["model"],
                                       case["shift"], terms)  # fmt: skip
            np.testing.assert_allclose(result.curve.vols, expected[:, j], rtol=1e-11, atol=1e-14)
            assert result.max_error < 1e-15
        cap = CapFloor(1.0, "cap", case["strikes"][0], case["tenors"][-1], 0.01, case["model"], case["shift"], **terms)
        fixes = [d.isoformat() for d in cap.caplets(market)["fix"]]
        assert fixes == case["fixing_dates"]


@pytest.mark.parametrize("method", ["constant", "linear"])
def test_stripped_vols_reprice_quantlib_cap_prices(golden, usd, method):
    """Coarse quotes (1Y ... 30Y; ATM and fixed strikes): caplet vols
    stripped from the flat vols reprice QuantLib's flat-vol NPVs."""

    for block in golden("capvol")["repricing"]:
        quotes = block["quotes"]
        if method == "linear":
            # Linear caplet vols zigzag: the 20Y-30Y fall of these flat vols needs a negative 30Y node.
            with pytest.raises(ValueError, match="30Y cap .*try piecewise-constant"):
                strip_caplet_vols(usd, [q["tenor"] for q in quotes], [q["vol"] for q in quotes], block["strike"],
                                  method, terms=USD_TERMS)  # fmt: skip
            quotes = quotes[:-1]
        tenors, vols = [q["tenor"] for q in quotes], [q["vol"] for q in quotes]
        result = strip_caplet_vols(usd, tenors, vols, block["strike"], method, "bachelier", 0.0, USD_TERMS)
        for q, row in zip(quotes, result.table.itertuples()):
            assert row.strike == pytest.approx(q["strike"], abs=1e-15)
            assert row.target_pv == pytest.approx(q["npv"], rel=1e-11)
            cap = CapFloor(1.0, "cap", q["strike"], q["tenor"], result.curve, **USD_TERMS)
            assert cap.pv(usd) == pytest.approx(q["npv"], rel=1e-11), (method, block["strike"], q["tenor"])


def test_stripped_vols_reprice_in_live_quantlib(golden, usd):
    """QuantLib prices each quoted cap caplet by caplet at the stripped
    vols (skipped without QuantLib)."""

    ql = pytest.importorskip("QuantLib")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from golden.capvol import _usd_index
    from golden.common import CALENDARS, qd, ql_curve, set_evaluation_date

    set_evaluation_date(usd.asof)
    disc = ql_curve(usd.curve)
    index = _usd_index(disc)
    block = golden("capvol")["repricing"][1]
    tenors, vols = [q["tenor"] for q in block["quotes"]], [q["vol"] for q in block["quotes"]]
    curve = strip_caplet_vols(usd, tenors, vols, block["strike"], "constant", terms=USD_TERMS).curve
    for q in block["quotes"]:
        cap = CapFloor(1.0, "cap", q["strike"], q["tenor"], curve, **USD_TERMS)
        table = cap.caplets(usd)
        periods = cap.periods(usd.asof)
        schedule = ql.Schedule(ql.DateVector([qd(periods[0].start)] + [qd(p.end) for p in periods]))
        leg = ql.IborLeg([1.0], schedule, index, ql.Actual360(), ql.ModifiedFollowing, [2])
        total = 0.0
        for coupon, vol in zip(leg, table["vol"]):
            single = ql.Cap(ql.Leg([coupon]), [q["strike"]])
            surface = ql.ConstantOptionletVolatility(qd(usd.asof), CALENDARS["USD"], ql.Following, float(vol),
                                                     ql.Actual365Fixed(), ql.Normal, 0.0)  # fmt: skip
            single.setPricingEngine(ql.BachelierCapFloorEngine(disc, ql.OptionletVolatilityStructureHandle(surface)))
            total += single.NPV()
        assert total == pytest.approx(q["npv"], rel=1e-11), q["tenor"]


# --------------------------------------------------------------------------
# Stripping
# --------------------------------------------------------------------------


@pytest.mark.parametrize("currency", ["USD", "EUR", "GBP"])
@pytest.mark.parametrize("method", ["constant", "linear"])
def test_illustrative_cap_quotes_strip_on_the_example_curve(usd, currency, method):
    atm = illustrative_cap_atm(currency)
    result = strip_caplet_vols(usd, list(atm["maturity"]), atm["vol_bp"].to_numpy() * 1e-4, method=method)
    assert result.max_error < 1e-15
    assert len(result.curve.nodes) == len(atm) and min(result.curve.vols) > 0
    assert list(result.table["maturity"]) == list(atm["maturity"])
    # ATM strikes are each cap's own forward swap rate.
    for row in result.table.itertuples():
        cap = CapFloor(1.0, "cap", 0.03, row.maturity, 0.01)
        assert row.strike == pytest.approx(cap.atm_strike(usd), abs=1e-15)


def test_surface_strips_each_strike_and_gives_a_smile_per_expiry(usd):
    grid = illustrative_cap_strikes("USD")
    strikes = np.array(CAP_STRIKES_PCT["USD"]) / 100
    surface, results = strip_surface(usd, list(grid["maturity"]), strikes, grid.iloc[:, 1:].to_numpy() * 1e-4)
    assert isinstance(surface, CapletVolSurface) and len(results) == len(strikes)
    assert max(r.max_error for r in results) < 1e-15
    K, vols = surface.smile(4.9)
    assert np.all(K == strikes) and vols.argmin() in (2, 3, 4)  # a smile around the reference strike
    # Caps on the quoted strikes reprice; between strikes the vol is linear in strike.
    for j, k in enumerate(strikes):
        cap = CapFloor(1.0, "cap", float(k), "7Y", surface)
        flat = CapFloor(1.0, "cap", float(k), "7Y", float(grid.iloc[5, j + 1]) * 1e-4)
        assert cap.pv(usd) == pytest.approx(flat.pv(usd), rel=1e-12)
    mid = 0.5 * (strikes[2] + strikes[3])
    t = np.array([1.0, 6.0])
    expected = 0.5 * (results[2].curve.vol_at(t) + results[3].curve.vol_at(t))
    np.testing.assert_allclose(surface.caplet_vols(t, mid), expected, rtol=1e-14)
    np.testing.assert_allclose(surface.caplet_vols(t, 0.001), results[0].curve.vol_at(t), rtol=1e-14)  # flat


def test_curve_interpolation_rules():
    curve = CapletVolCurve((1.0, 2.0, 5.0), (0.010, 0.012, 0.009), "constant", "bachelier")
    np.testing.assert_allclose(curve.vol_at([0.2, 1.0, 1.0001, 2.0, 3.0, 5.0, 9.0]),
                               [0.010, 0.010, 0.012, 0.012, 0.009, 0.009, 0.009])  # fmt: skip
    linear = replace_method(curve, "linear")
    np.testing.assert_allclose(linear.vol_at([0.2, 1.5, 3.5, 9.0]), [0.010, 0.011, 0.0105, 0.009])
    assert curve.lowest == 0.009
    with pytest.raises(ValueError, match="increasing"):
        CapletVolCurve((2.0, 1.0), (0.01, 0.01), "constant", "bachelier")
    with pytest.raises(ValueError, match="non-negative"):
        CapletVolCurve((1.0,), (-0.01,), "constant", "bachelier")
    with pytest.raises(ValueError, match="method"):
        CapletVolCurve((1.0,), (0.01,), "cubic", "bachelier")


def replace_method(curve: CapletVolCurve, method: str) -> CapletVolCurve:
    return CapletVolCurve(curve.nodes, curve.vols, method, curve.model, curve.shift, curve.strike)


def test_quotes_that_cannot_be_stripped_raise_with_the_maturity(usd):
    with pytest.raises(ValueError, match="5Y cap .* fall too fast"):
        strip_caplet_vols(usd, ["2Y", "3Y", "5Y"], [0.0110, 0.0100, 0.0040])
    with pytest.raises(ValueError, match="distinct"):
        strip_caplet_vols(usd, ["2Y", "2Y"], [0.01, 0.011])
    with pytest.raises(ValueError, match="positive"):
        strip_caplet_vols(usd, ["2Y"], [-0.01])
    with pytest.raises(ValueError, match="one flat vol"):
        strip_caplet_vols(usd, ["2Y", "3Y"], [0.01])
    with pytest.raises(ValueError, match="same caplet"):
        strip_caplet_vols(usd, ["12M", "1Y"], [0.01, 0.011])
    with pytest.raises(ValueError, match="method"):
        strip_caplet_vols(usd, ["2Y"], [0.01], method="spline")


def test_a_cap_must_use_the_model_its_vols_were_stripped_for(usd):
    curve = strip_caplet_vols(usd, ["1Y", "2Y"], [0.0095, 0.0100]).curve
    with pytest.raises(ValueError, match="stripped for the bachelier model"):
        CapFloor(1e6, "cap", 0.035, "2Y", curve, "shifted_black", 0.02)
    black = strip_caplet_vols(usd, ["1Y", "2Y"], [0.20, 0.21], model="shifted_black", shift=0.02).curve
    assert CapFloor(1e6, "cap", 0.035, "2Y", black, "shifted_black", 0.02).pv(usd) > 0
    with pytest.raises(ValueError, match="shift"):
        CapFloor(1e6, "cap", 0.035, "2Y", black, "shifted_black", 0.03)


def test_dual_curve_strip_uses_the_projection_curve():
    market = _eur()
    result = strip_caplet_vols(market, ["2Y", "5Y", "10Y"], [0.0080, 0.0085, 0.0078], terms=EUR_TERMS)
    cap = CapFloor.from_index("EURIBOR 6M", 1.0, "cap", float(result.table["strike"].iloc[1]), "5Y", result.curve)
    assert cap.pv(market) == pytest.approx(result.table["target_pv"].iloc[1], rel=1e-12)
    single = CapFloor(1.0, "cap", 0.03, "5Y", 0.0085, index_months=6, calendar="TARGET")
    assert cap.atm_strike(market) != pytest.approx(single.atm_strike(market), abs=1e-6)  # EURIBOR, not €STR


# --------------------------------------------------------------------------
# Vega buckets
# --------------------------------------------------------------------------


def test_vega_by_expiry_and_by_quote(usd):
    atm = illustrative_cap_atm("USD")
    mats, flat = list(atm["maturity"])[:7], atm["vol_bp"].to_numpy()[:7] * 1e-4
    curve = strip_caplet_vols(usd, mats, flat).curve
    cap = CapFloor(10_000_000.0, "cap", 0.036, "6Y", curve)
    per_caplet = vega_by_expiry(cap, usd)
    assert per_caplet["vega"].sum() == pytest.approx(cap.vega(usd), rel=1e-12)
    buckets = vega_by_expiry(cap, usd, edges=[1, 2, 5])
    assert list(buckets["bucket"]) == ["0-1Y", "1-2Y", "2-5Y", ">5Y"]
    assert buckets["vega"].sum() == pytest.approx(cap.vega(usd), rel=1e-12)
    by_quote = quote_vegas(cap, usd, mats, flat)
    assert list(by_quote["maturity"]) == mats
    assert by_quote.loc[by_quote["maturity"] == "10Y", "vega"].iloc[0] == pytest.approx(0.0, abs=1e-6)  # beyond 6Y+
    # +1bp on every quoted flat vol moves every caplet vol by about 1bp: the sum is close to the caplet vega.
    assert by_quote["vega"].sum() == pytest.approx(cap.vega(usd), rel=0.02)
    assert (by_quote["vega"].iloc[:5] != 0).all()
