"""FX smiles from ATM / RR / BF quotes: the market-strangle iteration,
vanna-volga through its pillars, SABR (beta = 1 fixed) on the quotes,
smile delta strikes, rescaling, interpolation in time and the arbitrage
check."""

from __future__ import annotations

import datetime as dt
import math
import pickle

import numpy as np
import pytest

from derivx.fx.conventions import fx_dates, market_delta_type
from derivx.fx.examples import ILLUSTRATIVE_QUOTES, ILLUSTRATIVE_RATES
from derivx.fx.market_data import bundled_ecb_rates
from derivx.fx.pricing import FxMarket, atm_strike, fx_delta, gk_price, strike_from_delta
from derivx.fx.smile import (
    SURFACE_DELTAS,
    SmileConventions,
    TenorQuote,
    build_smile,
    density_check,
    interpolate_quote,
    sabr_beta1_vol,
    smile_by_delta,
    smile_strikes,
)
from derivx.vol.sabr import sabr_vol

EURUSD_3M = FxMarket(1.1215, 0.0375, 0.02, 92 / 365, 94 / 365)
USDJPY_2Y = FxMarket(158.13, 0.0075, 0.0375, 2.0, 2.0 + 3 / 365)
Q25 = TenorQuote("3M", 0.075, -0.0023, 0.0013)
Q10 = TenorQuote("3M", 0.075, -0.0023, 0.0013, -0.0045, 0.0050)
JPY = TenorQuote("2Y", 0.109, -0.0171, 0.0030)
CASES = [
    (EURUSD_3M, Q25, "spot"),
    (USDJPY_2Y, JPY, "forward_pa"),
    (FxMarket(0.6985, 0.0375, 0.036, 1 / 365, 3 / 365), TenorQuote("ON", 0.0875, -0.0034, 0.0002), "spot"),
    (FxMarket(0.8403, 0.0375, 0.02, 0.5, 0.5), TenorQuote("6M", 0.0565, 0.0022, 0.0014), "spot_pa"),
]


def _strangle_values(m, quote, smile, level, delta_type):
    """The market strangle from first principles: strikes with delta +-X at
    the flat vol ATM + BF, priced flat and on the smile."""

    flat = quote.atm + quote.bf(level)
    kc = strike_from_delta(m, level / 100, flat, True, delta_type)
    kp = strike_from_delta(m, -level / 100, flat, False, delta_type)
    on_flat = gk_price(m, kc, flat, True) + gk_price(m, kp, flat, False)
    on_smile = gk_price(m, kc, smile.vol(kc), True) + gk_price(m, kp, smile.vol(kp), False)
    return on_flat, on_smile


@pytest.mark.parametrize("model", ["VV", "SABR"])
@pytest.mark.parametrize("m, quote, delta_type", CASES)
def test_market_strangle_iteration_reprices_the_broker_fly(model, m, quote, delta_type):
    """The smile-strangle fly is iterated until the smile prices the market
    strangle like the flat vol ATM + BF does; it is not the quoted fly."""

    smile = build_smile(m, quote, SmileConventions(delta_type, "DNS", "market"), model)
    on_flat, on_smile = _strangle_values(m, quote, smile, 25, delta_type)
    vega = smile.market_strangles[25].vega
    assert abs(on_smile - on_flat) / vega < 1e-9  # under 0.00001 vol bp
    assert abs(smile.errors["BF25"]) < 1e-9 and abs(smile.errors["RR25"]) < 1e-9 and abs(smile.errors["ATM"]) < 1e-12
    assert abs(smile.smile_strangles[25] - quote.bf25) > 1e-8  # the smile strangle differs from the broker fly


@pytest.mark.parametrize("model", ["VV", "SABR"])
@pytest.mark.parametrize("m, quote, delta_type", CASES)
def test_smile_strangle_convention_matches_the_vols_at_the_smiles_own_strikes(model, m, quote, delta_type):
    smile = build_smile(m, quote, SmileConventions(delta_type, "DNS", "smile"), model)
    kc, kp = smile.strike_at_delta(0.25, True), smile.strike_at_delta(-0.25, False)
    vc, vp = smile.vol(kc), smile.vol(kp)
    assert vc - vp == pytest.approx(quote.rr25, abs=1e-10)
    assert 0.5 * (vc + vp) - quote.atm == pytest.approx(quote.bf25, abs=1e-10)
    # the smile's own delta strikes are fixed points: delta(K, sigma(K)) = target
    assert fx_delta(m, kc, vc, True, delta_type) == pytest.approx(0.25, abs=1e-12)
    assert fx_delta(m, kp, vp, False, delta_type) == pytest.approx(-0.25, abs=1e-12)


@pytest.mark.parametrize("strangle", ["market", "smile"])
@pytest.mark.parametrize("m, quote, delta_type", CASES)
def test_vanna_volga_reproduces_its_pillar_vols(strangle, m, quote, delta_type):
    """Castagna-Mercurio weights are 1 at their own pillar and 0 at the
    others, so the smile goes through 25P, ATM and 25C to rounding."""

    smile = build_smile(m, quote, SmileConventions(delta_type, "DNS", strangle), "VV")
    for label in ("25P", "ATM", "25C"):
        K, vol = smile.pillars[label]
        assert smile.vol(K) == pytest.approx(vol, abs=1e-13), label
    assert smile.pillars["25P"][0] < smile.atm_strike < smile.pillars["25C"][0]
    k_lo, k_hi = smile.pillars["25P"][0], smile.pillars["25C"][0]
    inner = np.linspace(k_lo, k_hi, 9)
    assert np.all(np.isfinite(smile.vol(inner)))


def test_atm_vol_sits_at_the_dns_strike_and_beta_is_fixed():
    for delta_type in ("spot", "spot_pa"):
        smile = build_smile(EURUSD_3M, Q10, SmileConventions(delta_type, "DNS", "market"), "SABR")
        K = atm_strike(EURUSD_3M, Q10.atm, "DNS", delta_type.endswith("_pa"))
        assert smile.atm_strike == pytest.approx(K) and smile.vol(K) == pytest.approx(Q10.atm, abs=1e-14)
        assert smile.params["beta"] == 1.0 and smile.fixed == ("beta",)
        straddle = fx_delta(EURUSD_3M, K, Q10.atm, True, delta_type) + fx_delta(
            EURUSD_3M, K, Q10.atm, False, delta_type
        )
        assert straddle == pytest.approx(0.0, abs=1e-14)


def test_sabr_with_ten_delta_quotes_is_a_least_squares_fit_with_reported_errors():
    smile = build_smile(EURUSD_3M, Q10, SmileConventions("spot", "DNS", "market"), "SABR")
    assert set(smile.errors) == {"ATM", "RR25", "BF25", "RR10", "BF10"}
    assert set(smile.used_quotes) == {"ATM", "RR25", "BF25", "RR10", "BF10"}
    assert smile.fit_rmse == pytest.approx(np.sqrt(np.mean([v**2 for v in smile.errors.values()])))
    assert max(abs(v) for v in smile.errors.values()) < 5e-4  # within 5 vol bp on these round quotes
    vv = build_smile(EURUSD_3M, Q10, SmileConventions("spot", "DNS", "market"), "VV")
    assert vv.used_quotes == ("ATM", "RR25", "BF25") and any("checked, not fitted" in n for n in vv.notes)
    assert np.isfinite(vv.errors["RR10"]) and np.isfinite(vv.errors["BF10"])


@pytest.mark.parametrize("code", ["EURUSD", "USDJPY", "EURGBP"])
def test_illustrative_quotes_are_fitted_by_sabr_to_their_rounding(code):
    """The illustrative defaults come from a SABR smile rounded to 0.01 vol
    points, so the beta = 1 fit misses no quote by more than ~0.6 vol bp."""

    from derivx.fx.conventions import get_pair

    pair, ecb, trade = get_pair(code), bundled_ecb_rates(), dt.date(2026, 10, 20)
    for row in ILLUSTRATIVE_QUOTES[code][4::3]:
        d = fx_dates(code, trade, tenor=row[0])
        m = FxMarket(ecb.cross(code), ILLUSTRATIVE_RATES[pair.quote], ILLUSTRATIVE_RATES[pair.base], d.tau_e, d.tau_d)
        smile = build_smile(
            m,
            TenorQuote(row[0], *[x / 100 for x in row[1:]]),
            SmileConventions(market_delta_type(code, d), "DNS", "market"),
            "SABR",
        )
        assert max(abs(v) for v in smile.errors.values()) < 0.7e-4, (code, row[0], smile.errors)


def test_sabr_beta_one_kernel_matches_the_checked_sabr_module():
    K = np.linspace(0.8, 1.3, 11)
    for rho, nu in ((-0.3, 0.8), (0.0, 0.0), (0.5, 2.0)):
        fast = sabr_beta1_vol(1.05, K, 0.5, 0.08, rho, nu)
        assert fast == pytest.approx(sabr_vol(1.05, K, 0.5, 0.08, 1.0, rho, nu), rel=1e-14)


def test_rescaled_smile_equals_a_smile_built_at_that_spot():
    unit = FxMarket(1.0, EURUSD_3M.rd, EURUSD_3M.rf, EURUSD_3M.tau_e, EURUSD_3M.tau_d)
    conv = SmileConventions("spot", "DNS", "market")
    for model in ("SABR", "VV"):
        scaled = build_smile(unit, Q25, conv, model).rescaled(EURUSD_3M.S)
        direct = build_smile(EURUSD_3M, Q25, conv, model)
        K = np.linspace(1.05, 1.2, 13)
        np.testing.assert_allclose(scaled.vol(K), direct.vol(K), atol=1e-9)
        assert scaled.atm_strike == pytest.approx(direct.atm_strike, rel=1e-14)
        assert scaled.market_strangles[25].value == pytest.approx(direct.market_strangles[25].value, rel=1e-12)
        assert [p.strike for p in scaled.points] == pytest.approx([p.strike for p in direct.points], rel=1e-9)
        assert pickle.loads(pickle.dumps(scaled)).vol(1.12) == pytest.approx(scaled.vol(1.12))  # cacheable


def test_surface_rows_and_premium_adjusted_call_limits():
    smile = build_smile(USDJPY_2Y, JPY, SmileConventions("forward_pa", "DNS", "market"), "SABR")
    rows = smile_by_delta(smile)
    assert [r[0] for r in rows] == [label for label, _, _ in SURFACE_DELTAS]
    strikes = [r[1] for r in rows]
    assert all(np.diff(strikes) > 0)  # put wing -> ATM -> call wing, increasing strikes
    atm = dict((r[0], r[2]) for r in rows)["ATM"]
    assert atm == pytest.approx(JPY.atm, abs=1e-14)
    with pytest.raises(ValueError, match="above its maximum"):  # the 2Y PA call delta peaks near 73%
        smile.strike_at_delta(0.9, True)


def test_interpolation_in_time():
    q1, q2 = TenorQuote("1M", 0.07, -0.002, 0.001), TenorQuote("3M", 0.08, -0.004, 0.002, -0.008, 0.006)
    pts = [(30 / 365, q1), (91 / 365, q2)]
    mid, note = interpolate_quote(pts, 60 / 365, "2026-12-19")
    w = q1.atm**2 * 30 / 365 + (q2.atm**2 * 91 / 365 - q1.atm**2 * 30 / 365) * 30 / 61
    assert mid.atm == pytest.approx(math.sqrt(w / (60 / 365)))
    assert mid.rr25 == pytest.approx(-0.002 + (-0.002) * 30 / 61) and not mid.has_10d  # 10D only if both have it
    assert "interpolated between 1M and 3M" in note
    edge, note = interpolate_quote(pts, 200 / 365, "x")
    assert edge.atm == q2.atm and "flat extrapolation" in note
    exact, note = interpolate_quote(pts, 30 / 365, "x")
    assert exact.atm == q1.atm and note == "the 1M quotes"
    falling = [(30 / 365, TenorQuote("1M", 0.12, 0, 0)), (91 / 365, TenorQuote("3M", 0.06, 0, 0))]
    assert "calendar arbitrage" in interpolate_quote(falling, 60 / 365, "x")[1]


def test_density_check_flags_a_butterfly_arbitrage_and_passes_a_sane_smile():
    sane = build_smile(EURUSD_3M, Q10, SmileConventions("spot", "DNS", "market"), "SABR")
    assert density_check(sane, smile_strikes(sane))[0]
    wild = build_smile(
        FxMarket(1.1215, 0.0375, 0.02, 2.0, 2.0), TenorQuote("2Y", 0.10, 0.0, 0.03), SmileConventions(), "VV"
    )
    ok, density, K = density_check(wild, smile_strikes(wild, n_sd=4.0))
    assert not ok and density < 0 and K > 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(atm=0.0, rr25=0.0, bf25=0.0), "ATM vol must be > 0"),
        (dict(atm=7.5, rr25=0.0, bf25=0.0), "above 300%"),
        (dict(atm=0.075, rr25=float("nan"), bf25=0.0), "RR25 must be a number"),
        (dict(atm=0.075, rr25=0.0, bf25=0.0, rr10=0.01), "both RR10 and BF10"),
    ],
)
def test_quote_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TenorQuote("3M", **kwargs)


def test_impossible_quotes_and_expiry_raise_clear_errors():
    with pytest.raises(ValueError, match="pillar vols are not positive"):
        build_smile(EURUSD_3M, TenorQuote("3M", 0.05, -0.2, 0.0), SmileConventions("spot", "DNS", "smile"), "VV")
    with pytest.raises(ValueError, match="no smile at expiry"):
        build_smile(FxMarket(1.1, 0.04, 0.02, 0.0, 2 / 365), Q25, SmileConventions(), "SABR")
    with pytest.raises(ValueError, match="smile model"):
        build_smile(EURUSD_3M, Q25, SmileConventions(), "SVI")
    with pytest.raises(ValueError, match="strangle convention"):
        SmileConventions(strangle="broker")


@pytest.mark.parametrize(
    "m, quote, conventions, before",
    [
        (
            FxMarket(150.0, 0.0814, 0.1405, 5.0, 5.0 + 2 / 365),
            TenorQuote("5Y", 0.0473, 0.0101, 0.0133, 0.0210, 0.0288),
            SmileConventions("spot_pa", "DNS", "smile"),
            377_447,
        ),
        (
            FxMarket(150.0, 0.0775, 0.0304, 2.0, 2.0 + 2 / 365),
            TenorQuote("2Y", 0.5292, 0.1118, 0.1062, 0.1913, 0.3908),
            SmileConventions("spot", "SPOT", "market"),
            271_484,
        ),
    ],
)
def test_sabr_fit_on_extreme_quotes_stays_within_its_budget(monkeypatch, m, quote, conventions, before):
    """Review: on quotes no SABR smile fits (found by fuzzing), every trial
    smile's delta strikes spun through 100 fixed-point cycles: one tenor took
    20-50 s (``before`` strike solves) and the page hung. Trial smiles now
    fail fast and the fit has a deterministic evaluation budget, which never
    binds on market-like quotes (the illustrative smiles are unchanged)."""

    import derivx.fx.smile as smile_module

    calls = [0]
    solve = smile_module.strike_from_delta

    def counted(*args, **kwargs):
        calls[0] += 1
        return solve(*args, **kwargs)

    monkeypatch.setattr(smile_module, "strike_from_delta", counted)
    smile = build_smile(m, quote, conventions, "SABR")
    assert calls[0] < 40_000 < before / 5
    assert any("misses" in note for note in smile.notes)  # the misfit is reported, not hidden
    assert smile.errors["ATM"] == pytest.approx(0.0, abs=1e-12)
