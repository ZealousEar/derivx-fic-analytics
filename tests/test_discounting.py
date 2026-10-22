"""Discounting of option premiums (AL-14): the discount factor per expiry
comes from the Treasury curve of the trade date or from put-call parity
(European options), never from an unstated 0% rate, and every expiry says
which one it used."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
from _vol_chains import TRADE_DATE, synth_chain, true_vol

from derivx.calibration.production import calibrate_chain
from derivx.data import rates_history as rh
from derivx.data.market_data import apply_liquidity_filters, hygiene_report
from derivx.data.samples import load_sample_chain
from derivx.data.synthetic import synthetic_history
from derivx.rates.curves import bootstrap
from derivx.rates.market_quotes import treasury_quotes
from derivx.validation.backtest import WalkForwardSettings
from derivx.vol.discounting import (
    CurveDiscount,
    Discounting,
    european_style,
    parity_regression,
    treasury_discount,
    treasury_discounting,
)
from derivx.vol.quotes import clean_option_slice

RATE, DIV = 0.045, 0.013


def _flat_curve(asof: dt.date, rate: float) -> CurveDiscount:
    times = (0.0, 0.5, 1.0, 2.0, 5.0)
    return CurveDiscount(asof, times, tuple(float(np.exp(-rate * t)) for t in times), "loglinear", f"flat {rate}")


def _slice(days: int, **kw) -> pd.DataFrame:
    chain = synth_chain(days=(days,), rate=RATE, div_yield=DIV, **kw)
    return chain


def test_parity_regression_recovers_forward_and_discount_factor_of_european_quotes():
    g = _slice(365)
    S, T = 580.0, 1.0
    est = parity_regression(g, T, S, reference_rate=0.04, spot=S)
    assert est.ok, est.reason
    assert est.discount_factor == pytest.approx(np.exp(-RATE * T), abs=2e-6)
    assert est.forward == pytest.approx(S * np.exp((RATE - DIV) * T), rel=1e-6)
    assert est.rate == pytest.approx(RATE, abs=2e-6) and est.rate_se < 1e-4


def test_parity_gates_reject_noisy_short_expiries_and_american_bias():
    # noisy synthetic chains: the slope cannot pin the rate within a month, but can at 6 months
    h = synthetic_history(1)
    td, chain = next(iter(h.items()))
    S = float(chain["underlying_price"].iloc[0])
    by_days = {}
    for exp, g in chain.groupby("expiry"):
        tau = (exp - td).days / 365.0
        by_days[(exp - td).days] = parity_regression(g, tau, S * np.exp(0.028 * tau), reference_rate=0.04, spot=S)
    short = by_days[min(by_days)]
    assert not short.ok and "standard error" in short.reason
    long = by_days[max(by_days)]
    assert long.ok and abs(long.rate - 0.04) < 3 * long.rate_se + 1e-4
    # an early-exercise premium on in-the-money puts (growing with K - F) is not a discount factor
    g = _slice(182).copy()
    F = 580.0 * np.exp((RATE - DIV) * 0.5)
    itm_put = (g["option_type"] == "put") & (g["strike"] > F)
    eep = 0.02 * (g["strike"] - F).clip(lower=0.0)
    g.loc[itm_put, ["bid", "ask"]] = g.loc[itm_put, ["bid", "ask"]].add(eep[itm_put], axis=0)
    est = parity_regression(g, 182 / 365, F, reference_rate=RATE, spot=580.0)
    assert not est.ok and "American exercise" in est.reason and est.rate < RATE - 0.01


def test_vols_are_re_inverted_on_the_curve_discount_factor_not_zero():
    """The bug: vols were re-inverted with df = exp(-r T) from a rate input
    that defaulted to 0%, about 30-50 vol bp low at one year for a 4.5%
    rate. With the trade date's curve they match the true smile."""

    chain = _slice(365)
    curve = _flat_curve(TRADE_DATE.date(), RATE)
    disc = Discounting("curve", 0.0, ((TRADE_DATE.date().isoformat(), curve),))
    good = clean_option_slice(chain, discount=disc)
    assert good.discount_method == "curve" and good.discount_factor == pytest.approx(np.exp(-RATE), rel=1e-12)
    assert good.rate == pytest.approx(RATE, abs=1e-12) and good.discount_source == f"flat {RATE}"
    truth = true_vol(good.quotes["strike"], good.forward, good.tau)
    assert np.max(np.abs(good.quotes["implied_vol"] - truth)) * 1e4 < 1.0
    zero = clean_option_slice(chain, rate=0.0)
    bias = (zero.quotes["implied_vol"] - true_vol(zero.quotes["strike"], zero.forward, zero.tau)) * 1e4
    assert zero.discount_method == "rate" and np.median(bias) < -20.0


def test_parity_mode_uses_the_parity_dfs_where_they_pass_and_says_why_elsewhere():
    chain = synth_chain(days=(30, 182, 365), rate=RATE, div_yield=DIV)
    curve = _flat_curve(TRADE_DATE.date(), 0.040)  # a reference 50bp below the chain's own rate
    disc = Discounting("parity", 0.0, ((TRADE_DATE.date().isoformat(), curve),))
    report = hygiene_report(chain, discount=disc, min_days=2)
    assert set(report["discount_method"]) == {"parity"}
    np.testing.assert_allclose(report["rate_pct"], 100 * RATE, atol=1e-3)
    np.testing.assert_allclose(report["parity_rate_pct"], report["rate_pct"])
    strict = Discounting("parity", 0.0, ((TRADE_DATE.date().isoformat(), curve),), max_gap=0.002)
    report = hygiene_report(chain, discount=strict, min_days=2)
    assert set(report["discount_method"]) == {"curve"}
    np.testing.assert_allclose(report["rate_pct"], 4.0, atol=1e-9)
    assert report["discount_source"].str.contains("parity not used: implied rate 4.50% is +50bp", regex=False).all()


def test_treasury_curve_of_the_trade_date_from_the_bundled_store():
    store = rh.bundled_store()
    curve = treasury_discount(store, pd.Timestamp("2026-10-20"))
    assert curve.asof == dt.date(2026, 10, 19)  # the latest curve published on or before the trade date
    assert curve.label == "US Treasury par curve of 2026-10-19 (US Treasury, bundled snapshot), bootstrapped"
    ref = bootstrap(curve.asof, treasury_quotes(store.curve("UST_PAR", curve.asof).to_dict()), "USD UST")
    expiry = dt.date(2027, 10, 20)
    assert curve.discount("2026-10-20", expiry) == pytest.approx(ref.df(expiry) / ref.df(dt.date(2026, 10, 20)))
    with pytest.raises(ValueError, match="before the curve date"):
        curve.discount("2026-10-16", expiry)
    assert treasury_discount(store, pd.Timestamp("2031-01-02")) is None
    with pytest.raises(ValueError, match="no US Treasury curve within 7 days"):
        treasury_discounting(store, [pd.Timestamp("2031-01-02")])
    fallback = treasury_discounting(store, [pd.Timestamp("2031-01-02")], mode="parity", rate=0.03)
    df, method, source = fallback.reference("2031-01-02", "2032-01-02", 1.0)
    assert method == "rate" and df == pytest.approx(np.exp(-0.03)) and "no Treasury curve within 7 days" in source


def test_american_spy_chain_falls_back_to_the_treasury_curve_on_every_expiry():
    ds = load_sample_chain()
    chain = apply_liquidity_filters(ds.single(), min_volume=10, max_spread=0.4, min_mid=0.1)
    disc = treasury_discounting(rh.bundled_store(), ds.trade_dates, mode="parity")
    report = hygiene_report(chain, discount=disc, min_days=2)
    live = report[report["days"] >= 2]
    assert (live["discount_method"] == "curve").all()  # early exercise biases every parity slope
    assert live["parity_rate_pct"].min() < -10.0  # the American bias the module docs quote
    assert 4.0 < live["rate_pct"].min() and live["rate_pct"].max() < 4.8  # Treasury zero rates, 1 week to 2.3 years


def test_discounting_validates_and_describes_itself():
    with pytest.raises(ValueError, match="mode must be one of"):
        Discounting("libor")
    with pytest.raises(ValueError, match="needs at least one curve"):
        Discounting("curve")
    with pytest.raises(ValueError, match="cc decimal"):
        Discounting.flat(4.5)  # percent typed as a decimal
    assert european_style("^SPX") and european_style("xsp") and not european_style("SPY")
    disc = Discounting("parity", 0.01, (("2026-10-19", _flat_curve(dt.date(2026, 10, 19), 0.04)),))
    settings = WalkForwardSettings(discount=disc)
    desc = json.loads(json.dumps(settings.as_dict()))
    assert desc["discount"]["mode"] == "parity" and desc["discount"]["curves"]["2026-10-19"]["label"] == "flat 0.04"
    assert WalkForwardSettings().as_dict()["discount"] == {"mode": "rate"}


def test_calibration_records_the_discounting_per_slice():
    chain = synth_chain(days=(91,), rate=RATE, div_yield=DIV, n_strikes=25)
    disc = Discounting("curve", 0.0, ((TRADE_DATE.date().isoformat(), _flat_curve(TRADE_DATE.date(), RATE)),))
    run = calibrate_chain(chain, discount=disc, cv_folds=0)
    table = run.table()
    assert table["discount_method"].tolist() == ["curve"] and table["rate_pct"].iloc[0] == pytest.approx(4.5)
    assert run.slices[0].diagnostics["rmse_bp"] < 2.0
