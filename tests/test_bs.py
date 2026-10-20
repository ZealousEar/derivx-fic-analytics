"""Black-Scholes-Merton core: QuantLib golden prices and Greeks, the T = 0
and sigma = 0 limits, input validation, vectorised call/put flags and the
one-day theta."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

from derivx.core.bs import (
    DESK_UNITS,
    GREEK_NAMES,
    bs_greeks,
    bs_price,
    bs_price_greeks,
    bs_theta_1d,
    call_flags,
    desk_greeks,
    finite_difference_estimates,
    finite_difference_greeks,
    forward_price,
    no_arbitrage_bounds,
    put_call_parity_residual,
)

ROOT = Path(__file__).resolve().parents[1]
FIRST_ORDER = ("price", "delta", "gamma", "vega", "theta", "rho", "phi")
SECOND_ORDER = ("vanna", "volga", "charm")


def _atol(name: str, S: float) -> float:
    """Absolute floor for each Greek's natural scale (prices ~ S, delta ~ 1,
    gamma ~ 1/S)."""

    return {"delta": 1e-11, "gamma": 1e-11 / S, "vanna": 1e-8, "charm": 1e-8, "volga": 1e-7 * S}.get(name, 1e-11 * S)


def _ours(case: dict):
    return bs_greeks(case["S"], case["K"], case["T"], case["r"], case["sigma"], q=case["q"], call=case["call"])


def test_prices_and_first_order_greeks_match_quantlib_analytic_engine(golden):
    """ql.AnalyticEuropeanEngine: calls and puts, 1 day to 30 years, q != 0,
    negative r, q > r, vols 5%..150%, equity / index / FX levels."""

    cases = golden("bs")["cases"]
    assert len(cases) > 200 and any(c["q"] > c["r"] for c in cases) and any(c["r"] < 0 for c in cases)
    assert {c["days"] for c in cases} >= {1, 7, 3650, 10950}
    for case in cases:
        g = _ours(case)
        for name in FIRST_ORDER:
            assert float(getattr(g, name)) == pytest.approx(case[name], rel=1e-8, abs=_atol(name, case["S"])), (
                name,
                case,
            )


def test_second_order_greeks_match_differences_of_quantlib(golden):
    """Vanna, volga and charm against central differences of QuantLib's
    analytic delta and vega (independent of derivx's formulas)."""

    for case in golden("bs")["cases"]:
        g = _ours(case)
        for name in SECOND_ORDER:
            assert float(getattr(g, name)) == pytest.approx(case[name], rel=1e-5, abs=_atol(name, case["S"])), (
                name,
                case,
            )


def test_one_day_theta_by_repricing_matches_quantlib_date_roll(golden):
    """theta_1d = NPV after moving QuantLib's evaluation date one day minus
    the NPV today (PP-16), and it differs from theta / 365 near expiry."""

    n = 0
    for case in golden("bs")["cases"]:
        if "theta_1d" not in case:
            continue
        n += 1
        ours = float(
            bs_theta_1d(case["S"], case["K"], case["T"], case["r"], case["sigma"], q=case["q"], call=case["call"])
        )
        assert ours == pytest.approx(case["theta_1d"], rel=1e-8, abs=1e-11 * case["S"])
    assert n > 150
    # Hull-style ATM example: 1 day before expiry the true overnight decay is about 2x theta / 365.
    one_day = float(bs_theta_1d(100.0, 100.0, 1 / 365, 0.05, 0.2))
    analytic = float(bs_greeks(100.0, 100.0, 1 / 365, 0.05, 0.2).theta) / 365
    assert one_day == pytest.approx(-0.42449, abs=1e-4)
    assert one_day / analytic == pytest.approx(1.968, abs=1e-3)


def test_textbook_reference_price():
    """Hull, Options, Futures and Other Derivatives: S = K = 100, r = 5%,
    sigma = 20%, T = 1 call = 10.4506, put = 5.5735."""

    assert float(bs_price(100, 100, 1, 0.05, 0.2)) == pytest.approx(10.450583572185565, abs=1e-12)
    assert float(bs_price(100, 100, 1, 0.05, 0.2, call=False)) == pytest.approx(5.573526022256971, abs=1e-12)


def _natural_scale(name: str, S: float) -> float:
    return {"delta": 1.0, "gamma": 1.0 / S, "vanna": 1.0, "charm": 1.0}.get(name, S)


def test_analytic_greeks_match_finite_differences_of_the_pricer():
    """Typical inputs: every Greek agrees with the pricer's finite
    difference within the difference's own error estimate, and that estimate
    is tight (so the check would see a wrong Greek)."""

    rng = np.random.default_rng(20261020)
    for _ in range(200):
        S = float(rng.uniform(20, 300))
        K = S * float(np.exp(rng.uniform(-0.5, 0.5)))
        T = float(rng.uniform(0.02, 5.0))
        r, q = float(rng.uniform(-0.01, 0.08)), float(rng.uniform(0.0, 0.06))
        sigma = float(rng.uniform(0.05, 0.8))
        call = bool(rng.integers(0, 2))
        g = bs_greeks(S, K, T, r, sigma, q=q, call=call)
        fd = finite_difference_estimates(S, K, T, r, sigma, q=q, call=call)
        values = finite_difference_greeks(S, K, T, r, sigma, q=q, call=call)
        for name in GREEK_NAMES:
            a = float(getattr(g, name))
            assert values[name] == fd[name].value
            assert abs(a - fd[name].value) <= 2 * fd[name].error + 1e-10 * abs(a), (name, a, fd[name])
            tight = 1e-4 if name in ("gamma", "vanna", "volga", "charm") else 1e-8
            assert fd[name].error <= tight * _natural_scale(name, S), (name, fd[name])


@pytest.mark.parametrize(
    "S, K, T, r, q, sigma, call",
    [
        (100.0, 80.0, 10.0, -0.01, 0.03, 0.01, True),  # old volga step sigma / 2: FD 0.0353 vs 0.0516
        (100.0, 50.0, 0.25, 0.05, 0.0, 0.2, True),  # deep ITM: gamma 4e-13, old FD noise -7e-11
        (100.0, 50.0, 1 / 365, 0.05, 0.0, 5.0, True),  # one day, 500% vol: old theta step 4% of T
        (100.0, 100.0, 50.0, 0.0, 0.0, 1e-4, True),  # on the forward, tiny vol: old q bump > sigma sqrt(T)
        (100.0, 100.0, 1e-4, 0.05, 0.0, 1e-4, True),  # 53 minutes, 0.01% vol: old vol bump 10% of sigma
        (100.0, 97.0, 1e-4, 0.05, 0.0, 1.0, False),
        (100.0, 50.0, 50.0, 0.5, -0.5, 1e-4, True),  # F = 7e23 S: prices ~1e12, rounding dominates
        (1.08, 1.2, 7 / 365, 0.04, 0.025, 0.08, False),
        (5800.0, 4640.0, 30.0, 0.042, 0.013, 0.3, True),
    ],
)
def test_finite_difference_check_has_no_false_alarms_at_extreme_inputs(S, K, T, r, q, sigma, call):
    """The model check used to report '[!]' for about 1 in 5 valid inputs
    (fixed bump sizes): scale-aware steps, Richardson extrapolation and an
    error estimate fix that without loosening the check."""

    g = bs_greeks(S, K, T, r, sigma, q=q, call=call)
    fd = finite_difference_estimates(S, K, T, r, sigma, q=q, call=call)
    for name in GREEK_NAMES:
        a = float(getattr(g, name))
        assert abs(a - fd[name].value) <= 2 * fd[name].error + 1e-10 * abs(a), (name, a, fd[name])


def test_finite_difference_check_sees_small_greek_errors():
    """A 1e-5 relative error in any Greek is outside the error estimate for
    an ordinary option (the old 5e-4 / 2e-5 tolerances could not see it)."""

    g = bs_greeks(100.0, 105.0, 0.75, 0.03, 0.25, q=0.01, call=False)
    fd = finite_difference_estimates(100.0, 105.0, 0.75, 0.03, 0.25, q=0.01, call=False)
    for name in GREEK_NAMES:
        wrong = float(getattr(g, name)) * (1 + 1e-5)
        assert abs(wrong - fd[name].value) > 2 * fd[name].error + 1e-10 * abs(wrong), name
    with pytest.raises(ValueError, match="kink"):
        finite_difference_estimates(100.0, 100.0, 0.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="kink"):
        finite_difference_estimates(100.0, 100.0, 1.0, 0.05, 0.0)


def test_expiry_limits_are_intrinsic_with_a_step_delta_and_no_nan():
    """PP-06 / ui-engineering-V4: T = 0 gives intrinsic value, a 0/1 delta
    (1/2 exactly at the strike) and zero gamma, vega and theta."""

    S = np.array([80.0, 100.0, 120.0])
    for call, intrinsic, delta in (
        (True, [0.0, 0.0, 20.0], [0.0, 0.5, 1.0]),
        (False, [20.0, 0.0, 0.0], [-1.0, -0.5, 0.0]),
    ):
        g = bs_greeks(S, 100.0, 0.0, 0.05, 0.2, q=0.03, call=call)
        np.testing.assert_array_equal(g.price, intrinsic)
        np.testing.assert_array_equal(g.delta, delta)
        for name in ("gamma", "vega", "theta", "rho", "phi", "vanna", "volga", "charm"):
            np.testing.assert_array_equal(getattr(g, name), 0.0)
    # One second before expiry the ATM gamma is huge but finite; at expiry it is 0, not 19,947.
    assert float(bs_greeks(100, 100, 0.0, 0.05, 0.2).gamma) == 0.0


def test_zero_vol_limits_are_the_discounted_forward_intrinsic():
    S, K, T, r, q = 100.0, 90.0, 2.0, 0.05, 0.01
    call = bs_greeks(S, K, T, r, 0.0, q=q)
    put = bs_greeks(S, K, T, r, 0.0, q=q, call=False)
    df, dq = np.exp(-r * T), np.exp(-q * T)
    assert float(call.price) == pytest.approx(S * dq - K * df, rel=1e-14)
    assert float(put.price) == 0.0
    assert float(call.delta) == pytest.approx(dq) and float(put.delta) == 0.0
    assert float(call.rho) == pytest.approx(K * T * df) and float(call.phi) == pytest.approx(-S * T * dq)
    assert float(call.theta) == pytest.approx(-r * K * df + q * S * dq)
    for name in ("gamma", "vega", "vanna", "volga"):
        assert float(getattr(call, name)) == 0.0 and float(getattr(put, name)) == 0.0
    # The zero-vol limit is the limit of tiny vols.
    tiny = bs_greeks(S, K, T, r, 1e-9, q=q)
    for name in FIRST_ORDER:
        assert float(getattr(tiny, name)) == pytest.approx(float(getattr(call, name)), abs=1e-9)


def test_on_the_forward_zero_vol_keeps_the_finite_limits():
    S, T, r, q = 100.0, 1.0, 0.05, 0.02
    K = float(forward_price(S, T, r, q))
    g = bs_greeks(S, K, T, r, 0.0, q=q)
    near = bs_greeks(S, K, T, r, 1e-4, q=q)
    assert float(g.delta) == pytest.approx(0.5 * np.exp(-q * T))
    assert float(g.vega) == pytest.approx(float(near.vega), rel=1e-6)
    assert float(g.vanna) == pytest.approx(float(near.vanna), rel=1e-3)
    assert float(g.gamma) == 0.0 and float(g.charm) == 0.0  # Dirac / unbounded: documented as 0
    assert all(np.isfinite(float(v)) for v in g.as_dict().values())


@pytest.mark.parametrize(
    "bad, message",
    [
        (dict(S=0.0), "S \\(spot\\) must be > 0"),
        (dict(S=-100.0), "S \\(spot\\) must be > 0"),
        (dict(K=-50.0), "K \\(strike\\) must be > 0"),
        (dict(T=-1.0), "T \\(years to expiry\\) must be >= 0"),
        (dict(sigma=-0.2), "sigma \\(vol, decimal\\) must be >= 0"),
        (dict(r=np.nan), "r must be finite"),
        (dict(q=np.inf), "q must be finite"),
        (dict(T=1000.0, q=-1.0), "overflows"),
    ],
)
def test_invalid_inputs_raise_instead_of_being_clamped(bad, message):
    """PP-05 / PP-18: no silent clamping to 1e-12."""

    args = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, q=0.0)
    args.update(bad)
    with pytest.raises(ValueError, match=message):
        bs_greeks(**args)


def test_call_flag_is_vectorised_for_a_mixed_book():
    """PP-18: call may be an array of flags or 'call' / 'put' strings."""

    flags = np.array([True, False, True, False])
    K = np.array([90.0, 90.0, 110.0, 110.0])
    mixed = bs_greeks(100.0, K, 1.0, 0.03, 0.25, q=0.01, call=flags)
    for i, flag in enumerate(flags):
        one = bs_greeks(100.0, K[i], 1.0, 0.03, 0.25, q=0.01, call=bool(flag))
        for name in ("price",) + GREEK_NAMES:
            assert float(getattr(mixed, name)[i]) == float(getattr(one, name))
    np.testing.assert_array_equal(call_flags(["call", "PUT", "c", "p"]), [True, False, True, False])
    with pytest.raises(ValueError, match="call must be"):
        call_flags(["call", "straddle"])
    with pytest.raises(ValueError, match="call must be"):
        call_flags(np.array([1.0, 0.0]))


def test_legacy_tuple_and_desk_units():
    price, delta, gamma, vega, theta, rho = bs_price_greeks(100, 100, 1, 0.05, 0.2)
    g = bs_greeks(100, 100, 1, 0.05, 0.2)
    assert (float(price), float(delta), float(vega)) == (float(g.price), float(g.delta), float(g.vega))
    desk = desk_greeks(g, 100.0)
    assert set(desk) | {"theta_1d"} == set(DESK_UNITS)
    assert float(desk["vega_1vol"]) == pytest.approx(0.3752403469)  # per vol point
    assert float(desk["theta_day"]) == pytest.approx(-6.414027546 / 365)
    assert float(desk["rho_1bp"]) == pytest.approx(53.23248155e-4)
    assert float(desk["delta_1pct"]) == pytest.approx(0.6368306512)  # S = 100: 1% of spot is 1.00
    assert float(desk["gamma_1pct"]) == pytest.approx(0.018762017)


def test_bounds_and_parity_residual_flag_real_violations():
    lower, upper = no_arbitrage_bounds(100.0, 90.0, 1.0, 0.05, q=0.02)
    assert float(lower) == pytest.approx(100 * np.exp(-0.02) - 90 * np.exp(-0.05))
    assert float(upper) == pytest.approx(100 * np.exp(-0.02))
    p_lo, p_hi = no_arbitrage_bounds(100.0, 90.0, 1.0, 0.05, q=0.02, call=False)
    assert float(p_lo) == 0.0 and float(p_hi) == pytest.approx(90 * np.exp(-0.05))
    price = float(bs_price(100.0, 90.0, 1.0, 0.05, 0.3, q=0.02))
    assert float(lower) < price < float(upper)
    # Parity of two independent quotes: zero for consistent prices, the mispricing otherwise.
    put = float(bs_price(100.0, 90.0, 1.0, 0.05, 0.3, q=0.02, call=False))
    assert abs(float(put_call_parity_residual(price, put, 100.0, 90.0, 1.0, 0.05, 0.02))) < 1e-12
    assert float(put_call_parity_residual(price + 0.25, put, 100.0, 90.0, 1.0, 0.05, 0.02)) == pytest.approx(0.25)


def test_theta_1d_validates_days_and_reprices_to_expiry():
    with pytest.raises(ValueError, match="days"):
        bs_theta_1d(100, 100, 1.0, 0.05, 0.2, days=-1)
    # Less than a day left: the option is repriced at expiry (intrinsic).
    T = 0.5 / 365
    decay = float(bs_theta_1d(100, 95, T, 0.05, 0.2))
    assert decay == pytest.approx(5.0 - float(bs_price(100, 95, T, 0.05, 0.2)))
    assert float(bs_theta_1d(100, 100, 0.0, 0.05, 0.2)) == 0.0


def _ql():
    ql = pytest.importorskip("QuantLib")
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    from golden import bs as gbs

    return ql, gbs


def test_random_options_match_live_quantlib():
    """Random calls and puts against a live QuantLib engine (skipped
    without QuantLib)."""

    ql, gbs = _ql()
    rng = np.random.default_rng(7)
    try:
        for _ in range(60):
            S = float(rng.choice([1.08, 100.0, 5800.0])) * float(rng.uniform(0.8, 1.25))
            K = S * float(np.exp(rng.uniform(-0.6, 0.6)))
            days = int(rng.integers(1, 7300))
            r, q = float(rng.uniform(-0.02, 0.09)), float(rng.uniform(-0.01, 0.07))
            sigma = float(rng.uniform(0.03, 1.2))
            call = bool(rng.integers(0, 2))
            ref = gbs._case(S, K, days, r, q, sigma, call)
            g = bs_greeks(S, K, days / 365.0, r, sigma, q=q, call=call)
            for name in FIRST_ORDER:
                assert float(getattr(g, name)) == pytest.approx(ref[name], rel=1e-8, abs=_atol(name, S)), name
            for name in SECOND_ORDER:
                assert float(getattr(g, name)) == pytest.approx(ref[name], rel=1e-5, abs=_atol(name, S)), name
    finally:
        gbs.set_evaluation_date(dt.date(2026, 10, 20))
