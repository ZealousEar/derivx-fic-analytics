"""Garman-Kohlhagen FX pricing: QuantLib golden prices, Greeks, deltas,
strikes from delta and ATM strikes; premium conventions; expiry limits."""

from __future__ import annotations

import math

import numpy as np
import pytest

from derivx.core.bs import bs_greeks
from derivx.fx.pricing import (
    FxMarket,
    atm_strike,
    base_notional,
    fx_delta,
    fx_greeks,
    gk_price,
    max_pa_call_delta,
    premium_amount,
    premium_styles,
    scenario_pnl,
    strike_from_delta,
)


def _market(case: dict) -> FxMarket:
    return FxMarket(case["S"], case["rd"], case["rf"], case["tau_e"], case["tau_d"])


def test_garman_kohlhagen_matches_quantlib_engine(golden):
    """ql.GarmanKohlagenProcess + AnalyticEuropeanEngine: price, spot delta,
    gamma, vega, domestic rho and foreign rho (dividendRho), one time T."""

    cases = golden("fx")["gk"]
    assert len(cases) >= 300 and any(c["rd"] < 0 for c in cases) and any(c["rf"] > c["rd"] for c in cases)
    for c in cases:
        T = c["days"] / 365.0
        g = fx_greeks(FxMarket(c["S"], c["rd"], c["rf"], T, T), c["K"], c["sigma"], c["call"])
        scale = c["S"]
        assert g.price == pytest.approx(c["price"], rel=1e-9, abs=1e-12 * scale), c
        assert g.delta_spot == pytest.approx(c["delta"], rel=1e-9, abs=1e-12), c
        assert g.gamma == pytest.approx(c["gamma"], rel=1e-8, abs=1e-12 / scale), c
        assert g.vega_1vol == pytest.approx(0.01 * c["vega"], rel=1e-9, abs=1e-14 * scale), c
        assert g.rho_d_1bp == pytest.approx(1e-4 * c["rho"], rel=1e-9, abs=1e-15 * scale), c
        assert g.rho_f_1bp == pytest.approx(1e-4 * c["rho_f"], rel=1e-9, abs=1e-15 * scale), c
        if "theta_1d" in c:  # QuantLib moved one day: both times shrink by 1/365
            assert g.theta_1d == pytest.approx(c["theta_1d"], rel=1e-7, abs=1e-12 * scale), c


def test_two_time_price_matches_quantlib_black_formula(golden):
    """Vol time (trade -> expiry) != rate time (spot -> delivery): the price is
    DF_d(tau_d) Black(F(tau_d), K, sigma sqrt(tau_e))."""

    for c in golden("fx")["black"]:
        price = gk_price(_market(c), c["K"], c["sigma"], c["call"])
        assert price == pytest.approx(c["price"], rel=1e-11, abs=1e-14 * c["S"]), c


def test_deltas_match_quantlib_black_delta_calculator(golden):
    n = 0
    for case in golden("fx")["delta_calc"]:
        m = _market(case)
        for delta_type, call, K, expected in case["deltas"]:
            assert fx_delta(m, K, case["sigma"], call, delta_type) == pytest.approx(expected, rel=1e-12, abs=1e-14)
            n += 1
    assert n >= 300


def test_strike_from_delta_matches_quantlib_including_premium_adjusted_calls(golden):
    """Every delta type, calls and puts; QuantLib's inverse normal is an
    approximation (~1e-9 relative), ours inverts exactly: our strikes
    reproduce the target delta to 1e-13. Where QuantLib finds no strike (a
    premium-adjusted call delta above its peak) we raise."""

    solved = no_strike = 0
    for case in golden("fx")["delta_calc"]:
        m, vol = _market(case), case["sigma"]
        for delta_type, call, target, expected in case["strike_from_delta"]:
            if expected is None:
                assert delta_type.endswith("_pa") and call
                with pytest.raises(ValueError, match="above its maximum"):
                    strike_from_delta(m, target, vol, call, delta_type)
                no_strike += 1
                continue
            K = strike_from_delta(m, target, vol, call, delta_type)
            assert K == pytest.approx(expected, rel=2e-8), (delta_type, call, target, case)
            assert fx_delta(m, K, vol, call, delta_type) == pytest.approx(target, abs=1e-13)
            solved += 1
    assert solved > 200 and no_strike > 0


def test_atm_strikes_match_quantlib(golden):
    for case in golden("fx")["delta_calc"]:
        m = _market(case)
        for delta_type, atm_type, expected in case["atm"]:
            K = atm_strike(m, case["sigma"], atm_type, delta_type.endswith("_pa"))
            assert K == pytest.approx(expected, rel=1e-13), (delta_type, atm_type)


def test_atm_dns_is_delta_neutral_in_its_convention():
    m = FxMarket(1.1215, 0.04, 0.02, 0.25, 0.26)
    for delta_type, pa in (("spot", False), ("forward", False), ("spot_pa", True), ("forward_pa", True)):
        K = atm_strike(m, 0.075, "DNS", pa)
        straddle = fx_delta(m, K, 0.075, True, delta_type) + fx_delta(m, K, 0.075, False, delta_type)
        assert straddle == pytest.approx(0.0, abs=1e-15)


def test_premium_adjusted_call_delta_is_not_monotone_and_the_right_branch_is_taken():
    m = FxMarket(150.0, 0.045, 0.001, 2.0, 2.0)
    vol = 0.15
    k_peak, d_peak = max_pa_call_delta(m, vol, "spot_pa")
    left, right = 0.7 * k_peak, 1.3 * k_peak
    d_left, d_right = fx_delta(m, [left, right], vol, True, "spot_pa")
    assert d_left < d_peak and d_right < d_peak  # the delta rises, peaks, falls
    target = 0.5 * (d_left + d_right)
    K = strike_from_delta(m, target, vol, True, "spot_pa")
    assert K > k_peak  # the market strike sits right of the peak
    assert fx_delta(m, K, vol, True, "spot_pa") == pytest.approx(target, abs=1e-14)
    with pytest.raises(ValueError, match="above its maximum"):
        strike_from_delta(m, d_peak + 1e-4, vol, True, "spot_pa")
    # unadjusted deltas have no such peak and their strikes are always to the right
    assert strike_from_delta(m, target, vol, True, "spot") > K


def test_premium_conventions_reproduce_the_audit_example():
    """PP-12: EURUSD S = 1.08, K = 1.10, 3M, rd 5%, rf 3%, 8%: 109.7 USD pips,
    1.0154% EUR, premium-adjusted spot delta 0.3639, gamma per 1% 0.0942."""

    m = FxMarket(1.08, 0.05, 0.03, 0.25, 0.25)
    g = fx_greeks(m, 1.10, 0.08, True)
    styles = premium_styles(g.price, m.S, 1.10, 0.0001)
    assert styles["dom_pips"] == pytest.approx(109.666, abs=1e-3)
    assert styles["pct_for"] == pytest.approx(1.0154, abs=1e-4)
    assert styles["pct_dom"] == pytest.approx(100 * g.price / 1.10)
    assert styles["for_pips"] == pytest.approx(g.price / (1.08 * 1.10) * 1e4)
    assert g.delta_spot == pytest.approx(0.3740, abs=1e-4) and g.delta_spot_pa == pytest.approx(0.3639, abs=1e-4)
    assert g.gamma_1pct == pytest.approx(0.0942, abs=1e-4)
    assert g.delta_forward == pytest.approx(g.delta_spot / m.df_f)
    assert g.delta_forward_pa == pytest.approx(g.delta_spot_pa / m.df_f)
    # notional in either currency and premium in either currency
    assert base_notional(1_000_000, True, 1.10) == 1_000_000
    assert base_notional(1_100_000, False, 1.10) == pytest.approx(1_000_000)
    usd = premium_amount(g.price, m.S, 1_000_000, in_base=False)
    eur = premium_amount(g.price, m.S, 1_000_000, in_base=True)
    assert usd == pytest.approx(10_966.63, abs=0.01) and eur == pytest.approx(usd / 1.08)


def test_two_time_greeks_by_finite_differences():
    """Rhos, delta and vega with tau_e != tau_d, against bumps of gk_price."""

    m, K, vol = FxMarket(1.1215, 0.04, 0.02, 92 / 365, 94 / 365), 1.13, 0.075
    g = fx_greeks(m, K, vol, False)
    h = 1e-6

    def bump(**kw):
        return (
            gk_price(m.shifted(**{k: getattr(m, k) + v for k, v in kw.items()}), K, vol, False)
            - gk_price(m.shifted(**{k: getattr(m, k) - v for k, v in kw.items()}), K, vol, False)
        ) / (2 * h)

    assert g.rho_d_1bp == pytest.approx(1e-4 * bump(rd=h), rel=1e-6)
    assert g.rho_f_1bp == pytest.approx(1e-4 * bump(rf=h), rel=1e-6)
    assert g.delta_spot == pytest.approx(bump(S=h), rel=1e-6)
    dv = (gk_price(m, K, vol + h, False) - gk_price(m, K, vol - h, False)) / (2 * h)
    assert g.vega_1vol == pytest.approx(0.01 * dv, rel=1e-6)
    # with equal times it is the core Black-Scholes-Merton model with q = rf
    m1 = FxMarket(1.1215, 0.04, 0.02, 0.5, 0.5)
    core = bs_greeks(1.1215, K, 0.5, 0.04, vol, q=0.02, call=False)
    assert fx_greeks(m1, K, vol, False).price == pytest.approx(float(core.price), rel=1e-14)


def test_expiry_day_values_the_forward_intrinsic_and_zeroes_the_risk():
    """T = 0 (pricer-page-V1 / PP-06): no absurd Greeks; the delivery is
    still ahead, so the payoff is the discounted forward intrinsic."""

    m = FxMarket(1.12, 0.04, 0.02, 0.0, 2 / 365)
    g = fx_greeks(m, 1.10, 0.08, True)
    assert g.price == pytest.approx(m.df_d * (m.forward - 1.10))
    assert g.delta_spot == pytest.approx(m.df_f) and g.gamma == 0 and g.vega_1vol == 0 and g.volga_1vol == 0
    otm = fx_greeks(m, 1.15, 0.08, True)
    assert otm.price == 0 and otm.delta_spot == 0 and otm.rho_d_1bp == 0
    assert np.isfinite(list(g.as_dict().values())).all()
    with pytest.raises(ValueError, match="sigma > 0 and tau_e > 0"):
        strike_from_delta(m, 0.25, 0.08)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(S=0.0, rd=0.0, rf=0.0, tau_e=0.25, tau_d=0.25), "spot must be > 0"),
        (dict(S=1.1, rd=float("nan"), rf=0.0, tau_e=0.25, tau_d=0.25), "rd must be finite"),
        (dict(S=1.1, rd=0.0, rf=0.0, tau_e=-0.1, tau_d=0.25), "times must be >= 0"),
        (dict(S=1.1, rd=5000.0, rf=0.0, tau_e=1.0, tau_d=1.0), "rates are decimals"),
    ],
)
def test_market_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FxMarket(**kwargs)


def test_bad_strikes_vols_and_deltas_raise_clear_errors():
    m = FxMarket(1.1, 0.04, 0.02, 0.25, 0.25)
    with pytest.raises(ValueError, match="strikes must be finite and > 0"):
        gk_price(m, -1.0, 0.1)
    with pytest.raises(ValueError, match="vols must be finite and >= 0"):
        gk_price(m, 1.1, -0.1)
    with pytest.raises(ValueError, match="call delta must be > 0"):
        strike_from_delta(m, -0.25, 0.1, True)
    with pytest.raises(ValueError, match="not below its maximum"):
        strike_from_delta(m, -0.999, 0.1, False, "spot")  # |spot delta| < DF_f
    with pytest.raises(ValueError, match="delta type"):
        fx_delta(m, 1.1, 0.1, True, "smile")


def test_scenario_pnl_is_full_revaluation_of_the_notional():
    m = FxMarket(1.1215, 0.04, 0.02, 92 / 365, 94 / 365)
    grid = scenario_pnl(m, 1.13, 0.075, True, -2_000_000, [-0.02, 0.0, 0.02], [-0.01, 0.0, 0.01, -0.2], 7)
    assert grid.dropped_vol_shocks == (-0.2,)  # 7.5% - 20 pts would be negative: dropped, not floored
    later = m.shifted(tau_e=m.tau_e - 7 / 365, tau_d=m.tau_d - 7 / 365)
    expected = -2_000_000 * (gk_price(later.shifted(S=1.1215 * 1.02), 1.13, 0.085, True) - grid.base_value)
    assert grid.pnl[2, 2] == pytest.approx(expected)
    assert grid.pnl[1, 1] > 0  # short the option: a week of decay is a gain
    assert math.isclose(grid.base_value, gk_price(m, 1.13, 0.075, True))


def test_random_markets_match_live_quantlib():
    """Random spots, rates (negative too), vols and two distinct times:
    BlackDeltaCalculator deltas, strikes from delta and ATM strikes, and
    blackFormula prices (skipped without QuantLib)."""

    ql = pytest.importorskip("QuantLib")
    kinds = {
        "spot": ql.DeltaVolQuote.Spot,
        "forward": ql.DeltaVolQuote.Fwd,
        "spot_pa": ql.DeltaVolQuote.PaSpot,
        "forward_pa": ql.DeltaVolQuote.PaFwd,
    }
    rng = np.random.default_rng(20261020)
    for _ in range(60):
        S = float(rng.choice([0.65, 1.12, 1.33, 0.92, 158.0, 177.0]))
        rd, rf = (float(x) for x in rng.uniform(-0.01, 0.08, 2))
        te = float(rng.uniform(1 / 365, 3.0))
        td = te + float(rng.integers(0, 5)) / 365
        vol = float(rng.uniform(0.03, 0.4))
        m = FxMarket(S, rd, rf, te, td)
        K = S * float(np.exp(rng.normal(0, vol * np.sqrt(te))))
        for name, kind in kinds.items():
            for call in (True, False):
                calc = ql.BlackDeltaCalculator(
                    ql.Option.Call if call else ql.Option.Put, kind, S, m.df_d, m.df_f, vol * np.sqrt(te)
                )
                assert fx_delta(m, K, vol, call, name) == pytest.approx(calc.deltaFromStrike(K), rel=1e-11, abs=1e-14)
                target = (1 if call else -1) * float(rng.uniform(0.05, 0.45))
                try:
                    expected = calc.strikeFromDelta(target)
                except RuntimeError:
                    with pytest.raises(ValueError):
                        strike_from_delta(m, target, vol, call, name)
                    continue
                assert strike_from_delta(m, target, vol, call, name) == pytest.approx(expected, rel=2e-8)
            calc = ql.BlackDeltaCalculator(ql.Option.Call, kind, S, m.df_d, m.df_f, vol * np.sqrt(te))
            K_dns = atm_strike(m, vol, "DNS", name.endswith("_pa"))
            assert K_dns == pytest.approx(calc.atmStrike(ql.DeltaVolQuote.AtmDeltaNeutral), rel=1e-13)
        price = ql.blackFormula(ql.Option.Put, K, m.forward, vol * np.sqrt(te), m.df_d)
        assert gk_price(m, K, vol, False) == pytest.approx(price, rel=1e-10, abs=1e-14 * S)
