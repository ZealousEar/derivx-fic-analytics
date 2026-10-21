"""Quote hygiene and put-call-parity forwards for one chain slice."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _vol_chains import TRADE_DATE, synth_chain

from derivx.vol.quotes import apply_std_band, atm_vol, clean_option_slice, infer_forward, year_fraction

SPOT, RATE, DIV = 580.0, 0.04, 0.012


def _slice(days=30, **kw):
    chain = synth_chain(days=(days,), **kw)
    return chain


def test_year_fraction_is_act_365_on_dates():
    assert year_fraction("2026-10-19", "2026-11-18") == pytest.approx(30 / 365)
    assert year_fraction(pd.Timestamp("2026-10-19 16:30", tz="America/New_York"), "2026-10-19") == 0.0


def test_forward_from_put_call_parity_is_insensitive_to_the_assumed_rate():
    """VM-05: the forward used to be the spot. Parity near the money barely
    depends on the discount rate, so even r = 0 recovers it."""

    T = 91 / 365
    chain = _slice(91)
    true_F = SPOT * np.exp((RATE - DIV) * T)
    F, source, n = infer_forward(chain, SPOT, T, rate=RATE)
    assert F == pytest.approx(true_F, rel=1e-6) and "parity" in source and n == 5
    F0, _, _ = infer_forward(chain, SPOT, T, rate=0.0)
    assert F0 == pytest.approx(true_F, rel=2e-4)
    assert abs(F0 / SPOT - 1) > 5 * abs(F0 / true_F - 1)  # far better than spot


def test_forward_falls_back_to_carry_with_a_reason():
    T = 0.5
    calls_only = _slice(182).query("option_type == 'call'")
    F, source, n = infer_forward(calls_only, SPOT, T, rate=RATE, div_yield=DIV)
    assert F == pytest.approx(SPOT * np.exp((RATE - DIV) * T)) and source.startswith("carry") and n == 0
    junk = _slice(182)
    junk["bid"], junk["ask"] = 1.0, 1.2  # constant prices: parity estimates disagree
    F2, source2, _ = infer_forward(junk, SPOT, T, rate=RATE, div_yield=DIV)
    assert F2 == pytest.approx(SPOT * np.exp((RATE - DIV) * T)) and "disagree" in source2
    with pytest.raises(ValueError, match="spot"):
        infer_forward(junk, 0.0, T)


def test_clean_slice_keeps_one_otm_quote_per_strike_and_reports_removals():
    """VM-06 / AL-10: ITM duplicates, IV placeholders, zero bids, crossed and
    stale quotes are removed with a reason; puts below F, calls above."""

    chain = _slice(30)
    n_strikes = chain["strike"].nunique()
    # inject junk
    i = chain.index
    chain.loc[i[3], "implied_vol"] = 1e-5  # Yahoo placeholder
    chain.loc[i[5], "implied_vol"] = 7.0
    chain.loc[i[40], "ask"] = chain.loc[i[40], "bid"] - 0.05  # crossed
    chain.loc[i[42], "lastTradeDate"] = TRADE_DATE - pd.Timedelta(days=300)  # stale
    dup = chain.iloc[[30]].assign(bid=chain.iloc[30]["bid"] - 0.2)  # wider duplicate
    chain = pd.concat([chain, dup], ignore_index=True)
    clean = clean_option_slice(chain, rate=RATE, recompute_iv=False)  # the quoted-vol path
    q = clean.quotes
    assert q["strike"].is_unique and q["strike"].is_monotonic_increasing
    F = clean.forward
    assert (q.loc[q["strike"] < F, "option_type"] == "put").all()
    assert (q.loc[q["strike"] > F, "option_type"] == "call").all()
    np.testing.assert_allclose(q["k"], np.log(q["strike"] / F))
    counts = clean.counts
    assert counts["itm"] >= n_strikes - 4
    assert counts["iv_placeholder"] >= 1 and counts["crossed"] >= 1 and counts["stale"] >= 1
    assert counts.get("duplicate", 0) + counts.get("itm", 0) > 0
    assert set(clean.removed["reason"]) <= {"iv_placeholder", "zero_bid", "crossed", "stale", "itm", "duplicate"}
    assert len(q) + len(clean.removed) == len(chain)
    assert clean.summary()["n_kept"] == len(q)
    assert (q["implied_vol"] > 1e-4).all() and (q["implied_vol"] <= 5).all()


def test_expired_and_same_day_rows_are_dropped():
    chain = synth_chain(days=(0,))
    clean = clean_option_slice(chain)
    assert clean.quotes.empty and clean.counts == {"expired": len(chain)}
    week = clean_option_slice(synth_chain(days=(7,)), min_days=10)
    assert week.quotes.empty and "expired" in week.counts


def test_no_option_type_means_one_smile_without_otm_selection():
    chain = _slice(30).query("option_type == 'call'").drop(columns=["option_type"])
    clean = clean_option_slice(chain)
    assert (clean.quotes["option_type"] == "smile").all()
    assert any("no option_type" in n for n in clean.notes)


def test_slice_errors_are_explicit():
    chain = _slice(30)
    with pytest.raises(ValueError, match="trade_date"):
        clean_option_slice(chain.drop(columns=["trade_date"]))
    two_days = pd.concat([chain, chain.assign(trade_date=TRADE_DATE + pd.Timedelta(days=1))])
    with pytest.raises(ValueError, match="trade dates"):
        clean_option_slice(two_days)
    two_exp = pd.concat([chain, synth_chain(days=(91,))])
    with pytest.raises(ValueError, match="one expiry"):
        clean_option_slice(two_exp)
    with pytest.raises(ValueError, match="underlying prices"):
        clean_option_slice(pd.concat([chain, chain.assign(underlying_price=600.0)]))


def test_standardised_band_scales_with_maturity():
    """VM-13: the same K/F band is many standard deviations at 1W and a
    fraction of one at 1Y; the standardised band keeps |k| <= n sd."""

    short = apply_std_band(clean_option_slice(_slice(7, width_sd=(-6, 6))), 3.0)
    long = apply_std_band(clean_option_slice(_slice(365, width_sd=(-6, 6))), 3.0)
    for clean in (short, long):
        width = 3.0 * atm_vol(clean.quotes) * np.sqrt(clean.tau)
        assert np.abs(clean.quotes["k"]).max() <= width
        assert "outside_band" in clean.counts
    assert np.abs(short.quotes["k"]).max() < 0.25 * np.abs(long.quotes["k"]).max()


def test_recompute_iv_replaces_quoted_vols_with_mid_vols_on_the_parity_forward():
    """VM-15 / AL-10: Yahoo's IV comes from the last trade and its own rates;
    recomputing from mids with our solver drops the placeholders and gives
    bid/ask vols for spread weights."""

    from _vol_chains import true_vol

    chain = _slice(91)
    chain["implied_vol"] = 1e-5  # every quoted vol is a placeholder
    clean = clean_option_slice(chain, rate=RATE, recompute_iv=True)
    q = clean.quotes
    assert len(q) > 20 and "iv_placeholder" not in clean.counts
    # mids carry a 1% half-spread, so mid vols sit within a few vol bp of the truth
    err = q["implied_vol"] - true_vol(q["strike"], clean.forward, clean.tau)
    assert np.max(np.abs(err)) < 5e-4
    assert (q["iv_bid"].dropna() < q["implied_vol"][q["iv_bid"].notna()]).all()
    assert (q["iv_ask"] > q["implied_vol"]).all()
    with pytest.raises(ValueError, match="recompute_iv"):
        clean_option_slice(chain.drop(columns=["bid"]), recompute_iv=True)


def test_spread_weights_in_the_pipeline():
    from derivx.calibration.production import calibrate_chain, spread_weights

    chain = _slice(91)
    run = calibrate_chain(chain, rate=RATE, div_yield=DIV, weights="spread", hygiene={"recompute_iv": True})
    s = run.slices[0]
    w = spread_weights(s.clean.quotes)
    assert np.isfinite(w).all() and (w > 0).all()
    assert s.fit.weights.mean() == pytest.approx(1.0)
    assert s.diagnostics["rmse_bp"] < 5
    with pytest.raises(ValueError, match="recompute_iv"):
        calibrate_chain(chain, weights="spread", hygiene={"recompute_iv": False})


# --- review fixes -------------------------------------------------------------------------------


def test_parity_forward_survives_one_bad_strike():
    """One stale pair near the money used to push the 5-strike range over 1%
    and drop the whole expiry to the carry forward (the spot when r = q = 0)."""

    T = 182 / 365
    chain = _slice(182)
    true_F = SPOT * np.exp((RATE - DIV) * T)
    k_bad = chain["strike"].iloc[(chain["strike"] - true_F).abs().argsort().iloc[0]]
    bad = (chain["strike"] == k_bad) & (chain["option_type"] == "call")
    chain.loc[bad, ["bid", "ask"]] += 10.0  # a stale call quote: its parity estimate is 1.7% off
    F, source, n = infer_forward(chain, SPOT, T, rate=RATE)
    assert F == pytest.approx(true_F, rel=1e-6)
    assert source.startswith("put-call parity") and "1 inconsistent" in source and n == 4


def test_default_recomputes_vols_from_mids_and_says_so():
    """Yahoo's quoted vols disagree between the call and the put at the same
    strike (by 1 to 10 vol points on a real SPY chain); with bid/ask in the
    chain the default re-inverts the mids, so the smile does not jump at F."""

    from _vol_chains import true_vol

    chain = _slice(91)
    chain["implied_vol"] += np.where(chain["option_type"] == "call", 0.03, -0.03)  # inconsistent quotes
    clean = clean_option_slice(chain, rate=RATE)
    q = clean.quotes
    assert clean.iv_source == "mid" and clean.summary()["iv_source"] == "mid"
    assert np.max(np.abs(q["implied_vol"] - true_vol(q["strike"], clean.forward, clean.tau))) < 5e-4
    quoted = clean_option_slice(chain, rate=RATE, recompute_iv=False)
    assert quoted.iv_source == "quoted"
    err = quoted.quotes["implied_vol"] - true_vol(quoted.quotes["strike"], quoted.forward, quoted.tau)
    assert np.max(np.abs(err)) > 0.02
    no_quotes = clean_option_slice(chain.drop(columns=["bid", "ask", "mid"]), rate=RATE)
    assert no_quotes.iv_source == "quoted"  # 'auto' needs executable quotes
    with pytest.raises(ValueError, match="recompute_iv"):
        clean_option_slice(chain, recompute_iv="always")


def test_min_days_counts_whole_days():
    """(3 / 365) * 365 < 3 in floating point: a 3-day expiry was dropped by min_days=3."""

    from derivx.models.comparison import select_expiry

    chain = _slice(3)
    assert not clean_option_slice(chain, min_days=3).quotes.empty
    assert clean_option_slice(chain, min_days=4).quotes.empty
    assert select_expiry(chain, target_days=3, min_days=3) == TRADE_DATE + pd.Timedelta(days=3)


def test_timezone_aware_trade_date_keeps_its_local_day():
    evening = pd.Timestamp("2026-10-19 21:30", tz="America/New_York")  # already the 20th in UTC
    assert year_fraction(evening, "2026-10-20") == pytest.approx(1 / 365)


def test_interpolate_forward_is_log_linear_with_flat_carry_beyond():
    from derivx.vol.quotes import interpolate_forward

    S, c = 100.0, 0.03
    known = [(0.25, S * np.exp(c * 0.25)), (1.0, S * np.exp(c * 1.0))]
    for T in (0.1, 0.5, 2.0):
        assert interpolate_forward(T, S, known) == pytest.approx(S * np.exp(c * T), rel=1e-12)
    with pytest.raises(ValueError, match="known forward"):
        interpolate_forward(1.0, S, [])
