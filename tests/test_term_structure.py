"""Vol surface from SVI slices: calendar checks over every k, time
interpolation in total variance, and Black-76 cross-Greeks checked against
finite differences and QuantLib."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from _vol_chains import synth_chain, true_vol

from derivx.surface.term_structure import SurfaceModel, black76_cross_greeks, calibrate_surface, compute_cross_greeks
from derivx.vol.implied import black_price
from derivx.vol.svi import svi_iv


def _chain(trade_date: dt.date, expiry: dt.date) -> pd.DataFrame:
    strikes = np.linspace(88, 112, 13)
    forward = 100.0
    params = {"a": 0.02, "b": 0.28, "rho": -0.2, "m": 0.0, "s": 0.4}
    ivs = svi_iv(strikes, params, F=forward, T=1.0)  # params: total variance at T = 1
    df = pd.DataFrame(
        {
            "trade_date": pd.Timestamp(trade_date),
            "expiry": pd.Timestamp(expiry),
            "strike": strikes,
            "implied_vol": ivs,
            "underlying_price": np.full_like(strikes, forward),
            "bid": np.full_like(strikes, 1.0),
            "ask": np.full_like(strikes, 1.2),
            "volume": np.full_like(strikes, 50),
            "open_interest": np.full_like(strikes, 100),
        }
    )
    df["moneyness"] = df["strike"] / forward
    return df


def test_calibrate_surface_builds_model():
    trade_date = dt.date(2024, 1, 2)
    chains = {
        pd.Timestamp(dt.date(2024, 2, 1)): _chain(trade_date, dt.date(2024, 2, 1)),
        pd.Timestamp(dt.date(2024, 3, 1)): _chain(trade_date, dt.date(2024, 3, 1)),
    }
    surface = calibrate_surface(chains)
    assert isinstance(surface, SurfaceModel)
    assert len(surface.slices) == 2
    assert surface.calendar_violations == 0
    # exact SVI quotes: each slice reproduces them, on the stated forward and ACT/365F year fraction
    for slc, days in zip(surface.slices, (30, 59)):
        quotes = chains[pd.Timestamp(slc.expiry)]
        assert slc.tau == pytest.approx(days / 365.0) and slc.forward == pytest.approx(100.0)
        np.testing.assert_allclose(slc.iv(quotes["strike"].to_numpy()), quotes["implied_vol"], atol=1e-7)
        np.testing.assert_allclose(
            surface.implied_vol(quotes["strike"].to_numpy(), slc.tau), quotes["implied_vol"], atol=1e-7
        )


def test_compute_cross_greeks_returns_rows():
    trade_date = dt.date(2024, 1, 2)
    chains = {pd.Timestamp(dt.date(2024, 2, 1)): _chain(trade_date, dt.date(2024, 2, 1))}
    surface = calibrate_surface(chains)
    greeks = compute_cross_greeks(surface, strikes=[95, 100, 105])
    assert {"vega", "volga", "vanna"}.issubset(greeks.columns)
    assert greeks["strike"].tolist() == [95.0, 100.0, 105.0]
    # Black-76 at the smile vol of each strike (the quotes were exact SVI vols), no discounting at rate 0
    true_vols = svi_iv(
        np.array([95.0, 100.0, 105.0]), {"a": 0.02, "b": 0.28, "rho": -0.2, "m": 0.0, "s": 0.4}, 100.0, 1.0
    )
    np.testing.assert_allclose(greeks["vol"], true_vols, atol=1e-7)
    ref = black76_cross_greeks(100.0, greeks["strike"].to_numpy(), 30 / 365.0, greeks["vol"].to_numpy(), 1.0)
    for name in ("delta", "vega", "volga", "vanna", "charm"):
        np.testing.assert_allclose(greeks[name], ref[name], rtol=1e-12, atol=1e-12)
    assert greeks["vega"].idxmax() == 1  # vega peaks at the money


def _price(F, K, T, vol, r):
    return black_price(F, K, T, vol, True, np.exp(-r * T))


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_cross_greeks_match_finite_differences(K):
    """VM-11: vanna had the wrong sign and a missing 1/sqrt(T) (it was
    exactly -sqrt(T) times the right value), and d1 added the rate to a
    forward."""

    F, T, vol, r = 100.0, 0.5, 0.2, 0.05
    df = np.exp(-r * T)
    g = black76_cross_greeks(F, K, T, vol, df)
    hF, hs, ht = 1e-3, 1e-4, 1e-5
    vega = (_price(F, K, T, vol + hs, r) - _price(F, K, T, vol - hs, r)) / (2 * hs)
    volga = (_price(F, K, T, vol + hs, r) - 2 * _price(F, K, T, vol, r) + _price(F, K, T, vol - hs, r)) / hs**2
    vanna = (
        _price(F + hF, K, T, vol + hs, r)
        - _price(F + hF, K, T, vol - hs, r)
        - _price(F - hF, K, T, vol + hs, r)
        + _price(F - hF, K, T, vol - hs, r)
    ) / (4 * hF * hs)

    def delta(t):
        return (_price(F + hF, K, t, vol, r) - _price(F - hF, K, t, vol, r)) / (2 * hF)

    charm = -(delta(T + ht) - delta(T - ht)) / (2 * ht)
    assert g["delta"] == pytest.approx(delta(T), rel=1e-7)
    assert g["vega"] == pytest.approx(vega, rel=1e-7)
    assert g["volga"] == pytest.approx(volga, rel=1e-4, abs=1e-6)
    assert g["vanna"] == pytest.approx(vanna, rel=1e-5)
    assert g["charm"] == pytest.approx(charm, rel=1e-4, abs=1e-6)
    # sign: an OTM call's delta rises with vol (vanna > 0), an ITM call's falls
    assert np.sign(g["vanna"]) == np.sign(np.log(K / F) + 0.5 * vol**2 * T)
    # the rate only discounts: at r = 0 the ATM-forward vega is F n(d1) sqrt(T)
    atm = black76_cross_greeks(F, F, T, vol, 1.0)
    d1 = 0.5 * vol * np.sqrt(T)
    assert atm["vega"] == pytest.approx(F * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(T))


def test_cross_greeks_match_quantlib_golden(golden):
    for c in golden("vol")["greeks"]:
        g = black76_cross_greeks(c["F"], c["K"], c["T"], c["vol"], c["df"])
        assert black_price(c["F"], c["K"], c["T"], c["vol"], True, c["df"]) == pytest.approx(c["price"], rel=1e-12)
        scale = c["F"]
        assert g["delta"] == pytest.approx(c["delta"], rel=1e-6)
        assert g["vega"] == pytest.approx(c["vega"], rel=1e-6)
        assert g["volga"] == pytest.approx(c["volga"], rel=1e-3, abs=1e-6 * scale)
        assert g["vanna"] == pytest.approx(c["vanna"], rel=1e-4, abs=1e-7)
        assert g["charm"] == pytest.approx(c["charm"], rel=1e-3, abs=1e-6)


def test_cross_greeks_match_quantlib_live():
    ql = pytest.importorskip("QuantLib")
    F, K, T, vol, df = 100.0, 110.0, 0.75, 0.25, 0.97
    calc = ql.BlackCalculator(ql.PlainVanillaPayoff(ql.Option.Call, K), F, vol * np.sqrt(T), df)
    g = black76_cross_greeks(F, K, T, vol, df)
    assert g["delta"] == pytest.approx(calc.deltaForward(), rel=1e-12)
    assert g["vega"] == pytest.approx(calc.vega(T), rel=1e-12)


def test_surface_greeks_use_the_forward_and_report_charm():
    chain = synth_chain(days=(30, 91, 182), iv_noise=0.0)
    surface = calibrate_surface(dict(tuple(chain.groupby("expiry"))), rate=0.04, div_yield=0.012)
    table = compute_cross_greeks(surface, strikes=[540, 580, 620], rate=0.04)
    assert set(table.columns) >= {"delta", "vega", "volga", "vanna", "charm", "forward", "vol"}
    row = table.iloc[0]
    ref = black76_cross_greeks(row["forward"], row["strike"], row["tau"], row["vol"], np.exp(-0.04 * row["tau"]))
    assert row["vanna"] == pytest.approx(float(ref["vanna"]))


def test_surface_interpolates_total_variance_in_time_and_forwards():
    chain = synth_chain(days=(30, 91, 182, 365))
    surface = calibrate_surface(dict(tuple(chain.groupby("expiry"))), rate=0.04, div_yield=0.012)
    assert surface.calendar_violations == 0 and len(surface.calendar) == 3
    s0, s1 = surface.slices[1], surface.slices[2]
    # at a slice expiry the surface is that slice
    assert surface.implied_vol(600.0, s1.tau) == pytest.approx(float(s1.iv(600.0)), rel=1e-6)
    # in between: total variance linear in T at fixed k, so it is non-decreasing
    k = np.linspace(-0.3, 0.2, 11)
    T_mid = 0.5 * (s0.tau + s1.tau)
    w_mid = surface.total_variance(k, T_mid)
    assert np.all(w_mid >= surface.total_variance(k, s0.tau) - 1e-12)
    assert np.all(w_mid <= surface.total_variance(k, s1.tau) + 1e-12)
    # forwards: carry rate recovered from parity forwards
    assert surface.forward(0.75) == pytest.approx(580.0 * np.exp(0.028 * 0.75), rel=1e-5)
    # vs the true SSVI smile at an off-grid expiry
    F = surface.forward(T_mid)
    K = F * np.exp(np.array([-0.15, 0.0, 0.1]))
    assert np.max(np.abs(surface.implied_vol(K, T_mid) - true_vol(K, F, T_mid))) < 0.002
    with pytest.raises(ValueError, match="one trade date"):
        calibrate_surface({1: chain, 2: chain.assign(trade_date=pd.Timestamp("2026-10-20"))})


def test_surface_counts_wing_calendar_violations():
    """VM-12: the old check compared ATM-spot total variance only."""

    from derivx.vol.svi import svi_iv as iv

    short = {"a": 0.005, "b": 0.12, "rho": -0.9, "m": 0.02, "sigma": 0.05}
    long = {"a": 0.03, "b": 0.03, "rho": 0.0, "m": 0.0, "sigma": 0.1}
    chains = {}
    for days, p in ((91, short), (182, long)):
        K = 100 * np.exp(np.linspace(-0.4, 0.3, 15))
        chains[days] = pd.DataFrame(
            {
                "trade_date": pd.Timestamp("2026-10-19"),
                "expiry": pd.Timestamp("2026-10-19") + pd.Timedelta(days=days),
                "strike": K,
                "implied_vol": iv(K, p, 100.0, days / 365),
                "underlying_price": 100.0,
            }
        )
    surface = calibrate_surface(chains, std_band=None, enforce_arbitrage=False)
    assert surface.calendar_violations == 1
