"""Value at risk: historical VaR / ES on synthetic data with known
quantiles, the delta-normal cross-check, as-of alignment of histories,
which history drives which curve, and sensitivity against full
revaluation P&L on the committed history."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from derivx.data import rates_history as rh
from derivx.rates.bonds import benchmark_bond, bond_position
from derivx.rates.curves import CurveQuote, Market, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, example_market
from derivx.rates.multicurve import illustrative_quotes
from derivx.rates.swaps import VanillaSwap, swap_position
from derivx.risk import var
from derivx.risk.scenarios import quote_risk

D = dt.date.fromisoformat


@pytest.fixture(scope="module")
def store():
    return rh.bundled_store()


@pytest.fixture(scope="module")
def market():
    return example_market()


@pytest.fixture(scope="module")
def book(market):
    swap = VanillaSwap.from_conventions(market.curve.conventions, "10Y", 0.0373, 1e7, trade_date=market.asof)
    settle = market.govt.conventions.cal.add_business_days(market.asof, 1)
    return [swap_position(swap), bond_position(benchmark_bond(market.govt, 10, settle), -5e6)]


def test_var_and_es_on_a_known_loss_ladder():
    rng = np.random.default_rng(0)
    pnl = -rng.permutation(np.arange(1.0, 1001.0))  # losses 1, 2, ..., 1000
    assert var.var_es(pnl, 0.99) == pytest.approx((991.0, np.mean(np.arange(991.0, 1001.0))))
    assert var.var_es(pnl, 0.975) == pytest.approx((976.0, np.mean(np.arange(976.0, 1001.0))))
    # 250 days at 99%: the 3rd worst loss; ES weights the 3rd by the fractional half day.
    losses = np.arange(1.0, 251.0)
    v, es = var.var_es(-losses, 0.99)
    assert v == 248.0 and es == pytest.approx((250.0 + 249.0 + 0.5 * 248.0) / 2.5)
    assert es >= v
    # Gains everywhere: VaR is negative (no clamping).
    assert var.var_es(np.arange(1.0, 101.0), 0.95)[0] == pytest.approx(-5.0)


def test_var_and_es_of_a_large_normal_sample():
    rng = np.random.default_rng(42)
    sigma = 1e5
    pnl = rng.normal(scale=sigma, size=400_000)
    for a in (0.95, 0.975, 0.99):
        v, es = var.var_es(pnl, a)
        z = norm.ppf(a)
        assert v == pytest.approx(z * sigma, rel=0.01)
        assert es == pytest.approx(sigma * norm.pdf(z) / (1 - a), rel=0.01)
    dn = var.delta_normal(pnl, 0.99)
    assert dn.sigma == pytest.approx(pnl.std(ddof=1))
    assert dn.var == pytest.approx(2.3263478740 * dn.sigma, rel=1e-9)
    assert dn.es == pytest.approx(2.6652142203 * dn.sigma, rel=1e-9)


def test_var_input_errors():
    with pytest.raises(ValueError, match="at least 100"):
        var.var_es(np.arange(50.0), 0.99)
    with pytest.raises(ValueError, match="confidence"):
        var.var_es(np.arange(500.0), 99.0)
    with pytest.raises(ValueError, match="finite"):
        var.var_es([1.0, np.nan] * 100, 0.95)
    with pytest.raises(ValueError, match="two finite"):
        var.delta_normal([1.0], 0.95)


def test_as_of_alignment_across_holidays():
    days = [D("2026-07-01"), D("2026-07-02"), D("2026-07-06"), D("2026-07-07")]  # no 07-03 in this market
    levels = pd.DataFrame({"2Y": [0.040, 0.041, 0.043, 0.042]}, index=days)
    previous = [D("2026-07-01"), D("2026-07-02"), D("2026-07-03"), D("2026-07-06")]
    dates = [D("2026-07-02"), D("2026-07-03"), D("2026-07-06"), D("2026-07-07")]
    changes, observed = var.aligned_changes(levels, previous, dates)
    np.testing.assert_allclose(changes[:, 0], [10.0, 0.0, 20.0, -10.0])  # the holiday moves nothing, nothing is lost
    assert observed.all()
    stale, ok = var.aligned_changes(levels, [D("2026-07-20")], [D("2026-07-21")])
    assert not ok[0] and stale[0, 0] == 0.0  # the last observation is two weeks old: missing, not a zero move
    assert var.daily_pairs([D("2026-01-02"), D("2026-01-05"), D("2026-02-27")]) == [(D("2026-01-02"), D("2026-01-05"))]


def test_usd_scenarios_and_their_sources(market, store):
    scen = var.build_scenarios(market, store, 500)
    assert scen.n == 500 and scen.dates[-1] == EXAMPLE_ASOF and scen.asof == EXAMPLE_ASOF
    assert all((b - a).days <= 5 for a, b in zip(scen.previous, scen.dates))
    flags = {s.key: (s.dataset, s.flag, s.coverage) for s in scen.sources}
    assert flags["govt"] == ("UST_PAR", "real", 1.0)
    assert flags["curve"] == ("UST_PAR", "proxy", 1.0)  # no free SOFR swap history: swap spreads held
    assert scen.bumps["curve"].shape == (500, len(market.curve.quotes))
    # Treasury quotes move by the par yield change at their own tenor.
    levels = store.complete("UST_PAR")
    i = scen.dates.index(D("2026-10-16"))
    j = [q.tenor for q in market.govt.quotes].index("10Y")
    expected = (levels.loc[D("2026-10-16"), "10Y"] - levels.loc[scen.previous[i], "10Y"]) * 1e4
    assert scen.bumps["govt"][i, j] == pytest.approx(expected, abs=1e-9)
    table = scen.sources_table()
    assert list(table.columns) == ["curve", "currency", "history", "dataset", "flag", "coverage_pct", "note"]
    assert "swap spreads held" in table.set_index("curve").loc["curve", "note"]


def test_gbp_uses_the_bank_of_england_curve_where_it_covers_the_window(store):
    par = store.ois_par("GBP_OIS_BOE", EXAMPLE_ASOF)
    curve = bootstrap(EXAMPLE_ASOF, [CurveQuote("OIS", t, r) for t, r in par.items()], "GBP SONIA")
    gbp = Market(curve=curve, govt=example_market().govt)
    short = var.build_scenarios(gbp, store, 30)
    src = {s.key: s for s in short.sources}["curve"]
    assert (src.dataset, src.flag) == ("GBP_OIS_BOE", "real") and src.coverage >= 0.9
    # Par OIS changes derived from the BoE spot curve, at matching tenors.
    i = len(short.dates) - 1
    a, b = store.ois_par("GBP_OIS_BOE", store.latest_on_or_before("GBP_OIS_BOE", short.previous[i])), par
    j = [q.tenor for q in gbp.curve.quotes].index("10Y")
    assert short.bumps["curve"][i, j] == pytest.approx((b["10Y"] - a["10Y"]) * 1e4, abs=1e-9)
    long = var.build_scenarios(gbp, store, 250)  # the bundled BoE sample is daily only from 2026-08-19
    src = {s.key: s for s in long.sources}["curve"]
    assert (src.dataset, src.flag) == ("UST_PAR", "cross-currency proxy") and "upload GBP" in src.note


def test_eur_projection_curves_move_with_the_swap_curve(store):
    from derivx.rates.multicurve import bootstrap_projection

    discount = bootstrap(EXAMPLE_ASOF, [CurveQuote("OIS", t, r) for t, r in
                                        {"1Y": 0.02, "2Y": 0.021, "5Y": 0.023, "10Y": 0.026, "30Y": 0.028}.items()],
                         "EUR ESTR")  # fmt: skip
    six = bootstrap_projection(EXAMPLE_ASOF, illustrative_quotes(discount, "EURIBOR 6M"), "EURIBOR 6M", discount)
    eur = Market(curve=discount, projections={"EURIBOR 6M": six}, govt=example_market().govt)
    scen = var.build_scenarios(eur, store, 30)
    flags = {s.key: (s.dataset, s.flag) for s in scen.sources}
    assert flags["curve"] == ("EUR_AAA_ECB", "proxy") and flags["EURIBOR 6M"] == ("EUR_AAA_ECB", "proxy")
    basis = [k for k, inst in enumerate(six.instruments) if inst.kind == "BASIS"]
    assert not basis or not np.any(scen.bumps["EURIBOR 6M"][:, basis])


def test_an_uploaded_swap_history_drives_the_swap_curve(market, store):
    days = store.dates("UST_PAR")[-80:]
    rows = [{"date": d.isoformat(), "2Y": 3.5 + 0.01 * k, "10Y": 3.7 - 0.01 * k} for k, d in enumerate(days)]
    frame, issues, _ = rh.parse_history_upload(pd.DataFrame(rows), "USD", "percent", "mine.csv")
    assert not [i for i in issues if i.level == "error"]
    mine = rh.build_store([store.frame, frame], "with upload")
    scen = var.build_scenarios(market, mine, 50)
    src = {s.key: s for s in scen.sources}["curve"]
    assert (src.dataset, src.flag) == ("USER_SWAP:USD", "upload")
    j10 = [q.tenor for q in market.curve.quotes].index("10Y")
    assert scen.bumps["curve"][-1, j10] == pytest.approx(-1.0, abs=1e-9)  # -0.01% a day at 10Y


def test_sensitivity_pnl_against_full_revaluation(market, store, book):
    scen = var.build_scenarios(market, store, 60)
    risk = {k: quote_risk(book, market, k) for k in scen.bumps}
    delta = var.sensitivity_pnl(scen, risk, gamma=False)
    np.testing.assert_allclose(delta, scen.bumps["curve"] @ risk["curve"].pv01 + scen.bumps["govt"] @ risk["govt"].pv01)
    second = var.sensitivity_pnl(scen, risk)
    full = var.full_revaluation_pnl(book, market, scen)
    assert np.max(np.abs(full - second)) < 0.01 * np.max(np.abs(full))
    assert np.max(np.abs(full - second)) < np.max(np.abs(full - delta))  # gamma brings it closer
    table = var.var_table(second, [p.name for p in book], 0.95, delta).set_index("position")
    total = second.sum(axis=1)
    assert table.loc[var.TOTAL, "var"] == pytest.approx(var.var_es(total, 0.95)[0])
    assert table.loc[var.TOTAL, "dn_sigma"] == pytest.approx(delta.sum(axis=1).std(ddof=1))
    worst = var.worst_days(scen, total, 5)
    assert list(worst["pnl"]) == sorted(total)[:5]
    with pytest.raises(ValueError, match="no key-rate risk"):
        var.sensitivity_pnl(scen, {"curve": risk["curve"]})


def test_risk_report_lists_every_sensitivity(market, store, book):
    from derivx.risk.scenarios import vegas

    risk = {k: quote_risk(book, market, k) for k in ("curve", "govt")}
    scen = var.build_scenarios(market, store, 60)
    names = [p.name for p in book]
    table = var.var_table(var.sensitivity_pnl(scen, risk), names, 0.95)
    report = var.risk_report(names, risk, table, vegas(book, market), [p.vol_model for p in book])
    counts = report["section"].value_counts()
    quotes = len(market.curve.quotes) + len(market.govt.quotes)
    assert counts["pv01"] == counts["gamma"] == quotes * len(book)
    assert "vega" not in counts  # a swap and a bond have no vol model
    assert counts["var"] == (len(table.columns) - 1) * len(table)
    pv01 = report[(report["section"] == "pv01") & (report["curve"] == "govt") & (report["item"] == "10Y BOND")]
    i = risk["govt"].labels.index("10Y BOND")
    assert pv01.set_index("position").loc[names[1], "value"] == pytest.approx(risk["govt"].pv01[i, 1])
    assert risk["govt"].pv01[i, 1] > 0  # short the 10Y bond: gains when its quote rises
    with_vol = var.risk_report(names, risk, table, [12.5, 0.0], ["normal", None])
    vega = with_vol[with_vol["section"] == "vega"]
    assert list(vega["position"]) == [names[0]] and vega["unit"].iloc[0] == "P&L per +1bp normal vol"


def test_scenario_errors(market, store):
    with pytest.raises(ValueError, match="at least 20"):
        var.build_scenarios(market, store, 5)
    old = rh.build_store([store.frame[store.frame["asof"] <= D("2013-06-28")]], "old")
    with pytest.raises(ValueError, match="daily US Treasury changes"):
        var.build_scenarios(market, old, 250, end=D("2013-06-28"))
