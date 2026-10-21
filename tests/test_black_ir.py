"""Legacy Black-76 caplet (``derivx.rates.black_ir``) against QuantLib's
``blackFormula`` (golden fixture and live), the closed form at the money,
no-arbitrage bounds and limits, and input guards. (The old test only
asserted that an at-the-money caplet is worth more than zero.)"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from derivx.rates.black_ir import price_cap_floor, price_caplet_black
from derivx.rates.term_structure import ZeroCurve


def test_caplets_match_quantlib_black_formula_golden(golden):
    cases = golden("black_ir")["cases"]
    assert len(cases) == 36
    for c in cases:
        got = price_caplet_black(c["F"], c["K"], c["T"], c["sigma"], c["df"])
        assert got == pytest.approx(c["caplet"], rel=1e-12, abs=1e-16), c
        # put-call parity against QuantLib's floorlet: caplet - floorlet = df (F - K)
        assert got - c["floorlet"] == pytest.approx(c["df"] * (c["F"] - c["K"]), rel=1e-10, abs=1e-15), c


def test_caplets_match_quantlib_live():
    ql = pytest.importorskip("QuantLib")
    rng = np.random.default_rng(11)
    for _ in range(200):
        F, T, sigma = rng.uniform(0.001, 0.1), rng.uniform(0.01, 30.0), rng.uniform(0.01, 1.5)
        K, df = F * rng.uniform(0.3, 3.0), math.exp(-rng.uniform(0.0, 0.08) * T)
        ref = ql.blackFormula(ql.Option.Call, K, F, sigma * math.sqrt(T), df)
        assert price_caplet_black(F, K, T, sigma, df) == pytest.approx(ref, rel=1e-11, abs=1e-16)


@pytest.mark.parametrize("F, T, sigma", [(0.03, 1.0, 0.2), (0.05, 10.0, 0.35), (0.002, 0.25, 0.9)])
def test_at_the_money_closed_form(F, T, sigma):
    df = math.exp(-0.04 * T)
    expected = df * F * (2.0 * norm.cdf(0.5 * sigma * math.sqrt(T)) - 1.0)
    assert price_caplet_black(F, F, T, sigma, df) == pytest.approx(expected, rel=1e-13)
    # and ~ df F sigma sqrt(T / 2 pi) for a small total vol
    if sigma * math.sqrt(T) < 0.3:
        assert price_caplet_black(F, F, T, sigma, df) == pytest.approx(
            df * F * sigma * math.sqrt(T / (2 * math.pi)), rel=0.01
        )


def test_bounds_monotonicity_and_limits():
    F, K, T, df = 0.04, 0.035, 2.0, 0.92
    vols = np.linspace(0.01, 3.0, 40)
    prices = np.array([price_caplet_black(F, K, T, v, df) for v in vols])
    assert np.all(np.diff(prices) > 0)  # increasing in vol (vega > 0)
    assert np.all(prices >= df * max(F - K, 0.0)) and np.all(prices <= df * F)
    assert price_caplet_black(F, K, T, 50.0, df) == pytest.approx(df * F, rel=1e-6)  # infinite vol: df F
    fwds = np.linspace(0.01, 0.08, 30)
    assert np.all(np.diff([price_caplet_black(f, K, T, 0.3, df) for f in fwds]) > 0)  # delta > 0
    # expiry and zero vol give the discounted intrinsic (the old code returned 0 for an ITM caplet)
    assert price_caplet_black(F, K, 0.0, 0.3, df) == pytest.approx(df * (F - K), rel=1e-14)
    assert price_caplet_black(F, K, T, 0.0, df) == pytest.approx(df * (F - K), rel=1e-14)
    assert price_caplet_black(K, F, 0.0, 0.3, df) == 0.0


@pytest.mark.parametrize(
    "args, message",
    [
        ((-0.001, 0.03, 1.0, 0.2, 0.97), "Bachelier"),
        ((0.03, 0.0, 1.0, 0.2, 0.97), "Bachelier"),
        ((0.03, 0.03, -0.5, 0.2, 0.97), "T must be"),
        ((0.03, 0.03, 1.0, -0.2, 0.97), "volatility"),
        ((0.03, 0.03, 1.0, 0.2, 0.0), "df must be"),
        ((0.03, 0.03, 1.0, 0.2, float("nan")), "df must be"),
    ],
)
def test_invalid_inputs_raise_with_a_reason(args, message):
    with pytest.raises(ValueError, match=message):
        price_caplet_black(*args)


def test_legacy_cap_strip_is_deprecated():
    curve = ZeroCurve(ts=np.array([0.0, 1.0, 5.0]), dfs=np.array([1.0, 0.96, 0.80]))
    vols = ZeroCurve(ts=curve.ts, dfs=np.full(3, 0.2))
    with pytest.warns(DeprecationWarning, match="capfloor.CapFloor"):
        res = price_cap_floor(curve, 0.03, [1.0, 2.0], vols)
    assert res["PV"] == pytest.approx(sum(res["buckets"].values())) and res["PV"] > 0
