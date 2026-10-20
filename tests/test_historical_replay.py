"""Historical replay with per-curve data: the Treasury curve moves by
Treasury par yields, each swap curve by its own swap/OIS history (ICE USD
swaps before 2016-10-28, the BoE SONIA curve, the user's upload) or a
flagged proxy; any window of the history store can be replayed."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from derivx.data import rates_history as rh
from derivx.rates.curves import CurveQuote, Instrument, Market, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, example_market
from derivx.risk.scenarios import (
    EPISODES,
    Shock,
    historical,
    historical_sources,
    history_window,
    run_scenario,
    shocked_market,
    standard_library,
)
from derivx.risk.sensitivities import Position

D = dt.date.fromisoformat


@pytest.fixture(scope="module")
def store():
    return rh.bundled_store()


@pytest.fixture(scope="module")
def gbp_market(store):
    """A real GBP SONIA curve (BoE, par OIS derived) with the Treasury curve."""

    par = store.ois_par("GBP_OIS_BOE", EXAMPLE_ASOF)
    curve = bootstrap(EXAMPLE_ASOF, [CurveQuote("OIS", t, r) for t, r in par.items()], "GBP SONIA")
    return Market(curve=curve, govt=example_market().govt)


def _flags(shock: Shock) -> dict:
    return {s.selector: (s.dataset, s.flag) for s in shock.sources}


def test_every_episode_says_which_data_each_curve_replays():
    expected_usd = {"2013 taper tantrum": ("USD_SWAP_ICE", "real")}
    for name, start, end in EPISODES:
        flags = _flags(historical(name))
        assert flags["govt:USD"] == ("UST_PAR", "real")
        assert flags["curve:USD"] == expected_usd.get(name, ("UST_PAR", "proxy")), name
        assert flags["curve:GBP"] == ("GBP_OIS_BOE", "real"), name
        assert flags["curve:EUR"] == ("EUR_AAA_ECB", "proxy"), name
        table = historical_sources(historical(name))
        assert list(table.columns) == ["curve", "currency", "data", "start", "end", "flag", "note"]
        assert all(abs((D(d) - start).days) <= 4 for d in table["start"])
        assert all(abs((D(d) - end).days) <= 4 for d in table["end"])


def test_usd_swap_curve_replays_ice_swap_rates_before_2016(store):
    shock = historical("2013 taper tantrum")
    ice = store.curve_move("USD_SWAP_ICE", D("2013-05-02"), D("2013-09-05"))
    ust = store.curve_move("UST_PAR", D("2013-05-02"), D("2013-09-05"))
    assert dict(shock.profiles)["curve:USD"] == ice.profile
    assert shock.profile == ust.profile  # the default (and the Treasury curve) stays on Treasury moves
    swap_10y = dict(zip(ice.tenors, ice.change_bp))["10Y"]
    assert swap_10y != dict(zip(ust.tenors, ust.change_bp))["10Y"]  # swaps did not move one-for-one with UST
    market = example_market()
    labels = market.curve.labels
    assert dict(zip(labels, shock.quote_bumps(market.curve)))["10Y OIS"] == swap_10y
    assert dict(zip(market.govt.labels, shock.quote_bumps(market.govt)))["10Y BOND"] == 132.0


def test_gbp_curve_replays_the_boe_curve(store, gbp_market):
    shock = historical("Sept 2022 gilt crisis week")
    move = store.curve_move("GBP_OIS_BOE", D("2022-09-21"), D("2022-09-28"))
    bumps = dict(zip([q.tenor for q in gbp_market.curve.quotes], shock.quote_bumps(gbp_market.curve)))
    for tenor, bp in zip(move.tenors, move.change_bp):
        assert bumps[tenor] == pytest.approx(bp, abs=1e-12), tenor
    moved = shocked_market(gbp_market, shock)
    rates = {q.tenor: q.rate for q in moved.curve.quotes}
    base = {q.tenor: q.rate for q in gbp_market.curve.quotes}
    assert (rates["10Y"] - base["10Y"]) * 1e4 == pytest.approx(dict(zip(move.tenors, move.change_bp))["10Y"])
    govt_bumps = shock.quote_bumps(gbp_market.govt)  # the Treasury curve still moves by Treasury yields
    np.testing.assert_array_equal(govt_bumps, historical("Sept 2022 gilt crisis week").bp_at(
        [i.tenor.years for i in gbp_market.govt.instruments],
        (gbp_market.govt.instruments[0].tenor.years, gbp_market.govt.instruments[-1].tenor.years),
    ))  # fmt: skip


def test_pnl_uses_the_curve_specific_moves(gbp_market):
    quote = gbp_market.curve.quotes[15]  # 10Y
    swap = Position(
        "REC GBP 10Y",
        lambda m: 1e6 * Instrument(quote, m.asof, m.curve.conventions).npv(m.curve),
        kind="swap",
    )
    shock = historical("Sept 2022 gilt crisis week")
    real = run_scenario([swap], gbp_market, shock)[0]
    proxy = run_scenario([swap], gbp_market, Shock("UST only", shock.profile))[0]
    assert real["pnl"] < 0 and real["pnl"] != pytest.approx(proxy["pnl"], rel=0.05)
    assert abs(real["residual"]) < 0.02 * abs(real["pnl"])


def test_history_window_replays_any_dates(store):
    shock = history_window(D("2022-01-31"), D("2022-10-31"))
    assert shock.name == "History 2022-01-31 to 2022-10-31" and "2022-01-31" in shock.description
    ust = store.curve_move("UST_PAR", D("2022-01-31"), D("2022-10-31"), 7)
    assert shock.profile == ust.profile
    half = history_window(D("2022-01-31"), D("2022-10-31"), scale=0.5)
    assert half.name.endswith("x0.5")
    for (sel, full), (_, scaled) in zip(shock.profiles, half.profiles):
        assert [b / 2 for _, b in full] == [b for _, b in scaled], sel
    in_year = history_window(D("2026-09-01"), EXAMPLE_ASOF)  # daily data of the snapshot year
    assert _flags(in_year)["curve:GBP"] == ("GBP_OIS_BOE", "real")
    with pytest.raises(ValueError, match="must be after"):
        history_window(EXAMPLE_ASOF, D("2026-09-01"))
    with pytest.raises(ValueError, match="within 7 days"):
        history_window(D("2011-06-15"), D("2026-09-01"))  # mid-month 2011 is not in the sample (daily from 2015)


def test_uploaded_history_comes_first_and_missing_markets_fall_back(store):
    upload = pd.DataFrame({"date": ["2023-03-08", "2023-03-15"], "2Y": [4.95, 4.10], "10Y": [3.85, 3.45]})
    frame, _, _ = rh.parse_history_upload(upload, "USD", "percent", "mine.csv")
    mine = rh.build_store([store.frame, frame])
    shock = historical("2023 SVB", store=mine)
    assert _flags(shock)["curve:USD"] == ("USER_SWAP:USD", "upload")
    assert dict(shock.profiles)["curve:USD"] == ((2.0, -85.0), (10.0, -40.0))
    ust_only = rh.build_store([store.frame[store.frame["dataset"] == "UST_PAR"]])
    flags = _flags(historical("2023 SVB", store=ust_only))
    assert flags["curve:GBP"] == ("UST_PAR", "cross-currency proxy") and flags["curve:EUR"][1] == "cross-currency proxy"
    with pytest.raises(ValueError, match="no US Treasury"):
        historical("2023 SVB", store=rh.build_store([store.frame[store.frame["dataset"] == "FIXINGS"]]))


def test_profiles_are_validated_and_count_as_moves():
    only_curve = Shock("x", profiles=(("curve:USD", ((1.0, 5.0),)),))
    assert only_curve.moves_rates and only_curve.profile_for("curve", "USD") == ((1.0, 5.0),)
    assert only_curve.profile_for("curve", "GBP") == () and only_curve.profile_for() == ()
    wildcard = Shock("y", profiles=(("curve:*", ((1.0, 2.0),)),))
    assert wildcard.profile_for("curve", "EUR") == ((1.0, 2.0),)
    with pytest.raises(ValueError, match="selectors"):
        Shock("z", profiles=(("curve", ((1.0, 1.0),)),))
    with pytest.raises(ValueError, match="same tenor"):
        Shock("z", profiles=(("curve:USD", ((1.0, 1.0), (1.0, 2.0))),))
    assert hash(historical("2023 SVB")) == hash(historical("2023 SVB"))  # cache keys


def test_standard_library_carries_the_sources(store):
    library = standard_library(store)
    episodes = [s for s in library if s.sources]
    assert [s.name for s in episodes] == [name for name, _, _ in EPISODES]
