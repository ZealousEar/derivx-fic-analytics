"""FX as a risk factor of the converted book: the VaR's FX moves on
synthetic data with known answers (exact converted P&L, the FX-held and
FX-only rows, the delta-normal check with FX returns, as-of alignment over
a holiday, the reporting currency's side of each pair), on the committed
ECB history for a mixed EUR / USD book, and FX in historical replays
(episodes, windows, the hold option, scaling, the CLI). No network."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from derivx import cli
from derivx.data import rates_history as rh
from derivx.data import rates_sources as rs
from derivx.rates import curve_io
from derivx.rates import portfolio as pf
from derivx.rates.examples import EXAMPLE_ASOF, example_market
from derivx.risk import reporting as rp
from derivx.risk import scenarios as sc
from derivx.risk import var

D = dt.date.fromisoformat


def _moves(eur_per_usd_ratio: np.ndarray, gbp_ratio=None) -> var.FxMoves:
    """FX moves with USD per EUR moving by ``eur_per_usd_ratio`` (GBP and
    EUR flat unless given)."""

    n = len(eur_per_usd_ratio)
    return var.FxMoves(
        {"EUR": np.ones(n), "USD": np.asarray(eur_per_usd_ratio, float),
         "GBP": np.ones(n) if gbp_ratio is None else np.asarray(gbp_ratio, float)},
        np.ones(n, dtype=bool),
    )  # fmt: skip


def _usd(eurusd: float = 1.10, gbpusd: float = 1.30) -> rp.Reporting:
    return rp.Reporting("USD", rp.entered_spots({"EURUSD": eurusd, "GBPUSD": gbpusd}))


def test_fx_only_var_of_a_euro_position_is_the_known_quantile():
    n = 1000
    x = np.random.default_rng(3).permutation((np.arange(1, n + 1) - 500.5) / 50_000.0)  # EURUSD moves, +/-1%
    fx = _moves(1.0 + x)
    rep = _usd(1.10)
    table = var.var_table(np.zeros((n, 1)), ["EUR SWAP"], 0.99, None, ["EUR"], rep, fx, [1e6]).set_index("position")
    only = table.loc["ALL IN USD (FX ONLY)"]
    losses = np.sort(-(1e6 * 1.10 * x))[::-1]
    assert only["var"] == pytest.approx(losses[9], rel=1e-12)  # the 10th largest of 1,000 losses
    assert only["es"] == pytest.approx(losses[:10].mean(), rel=1e-12)
    assert table.loc["ALL IN USD", "var"] == pytest.approx(only["var"], rel=1e-12)  # no rate P&L: FX is all of it
    assert table.loc["ALL IN USD (FX HELD)", "var"] == 0.0
    assert table.loc["TOTAL", "var"] == 0.0  # local P&L does not move with FX


def test_converted_pnl_is_exact_and_local_totals_are_unchanged():
    rng = np.random.default_rng(7)
    n = 300
    pnl = rng.normal(scale=[5e4, 2e4, 3e4], size=(n, 3))
    base = [2e6, -1e6, 5e5]
    ccys = ["EUR", "USD", "GBP"]
    usd_ratio, gbp_ratio = 1.0 + rng.normal(scale=0.005, size=n), 1.0 + rng.normal(scale=0.004, size=n)
    fx = _moves(usd_ratio, gbp_ratio)
    rep = _usd(1.10, 1.30)
    cols = var.book_columns(pnl, ["A", "B", "C"], ccys, rep, fx, base)
    held = var.book_columns(pnl, ["A", "B", "C"], ccys, rep)
    spot = np.array([1.10, 1.0, 1.30])
    # value in USD of 1 EUR = USD/EUR; of 1 GBP = USD/EUR / (GBP/EUR): their moves on each day
    x = np.column_stack([usd_ratio - 1.0, np.zeros(n), usd_ratio / gbp_ratio - 1.0])
    exact = ((np.array(base) + pnl) * spot * (1.0 + x) - np.array(base) * spot).sum(axis=1)
    np.testing.assert_allclose(cols["ALL IN USD"][0], exact, rtol=1e-12)
    np.testing.assert_array_equal(cols["ALL IN USD (FX HELD)"][0], held["ALL IN USD"][0])
    np.testing.assert_allclose(cols["ALL IN USD (FX ONLY)"][0], (np.array(base) * spot * x).sum(axis=1), rtol=1e-12)
    for label in ("TOTAL USD", "TOTAL EUR", "TOTAL GBP", "A", "B", "C"):
        np.testing.assert_array_equal(cols[label][0], held[label][0])  # local and per-currency P&L untouched
    # linear (first order): the cross term p S x is left out
    linear = var.book_columns(pnl, ["A", "B", "C"], ccys, rep, fx, base, linear=True)
    np.testing.assert_allclose(
        linear["ALL IN USD"][0], (pnl * spot).sum(axis=1) + (np.array(base) * spot * x).sum(axis=1), rtol=1e-12
    )
    with pytest.raises(ValueError, match="base"):
        var.book_columns(pnl, ["A", "B", "C"], ccys, rep, fx, None)


def test_delta_normal_includes_fx_returns_as_risk_factors():
    rng = np.random.default_rng(11)
    n = 500
    delta = rng.normal(scale=4e4, size=(n, 1))
    x = rng.normal(scale=0.006, size=n)
    table = var.var_table(delta, ["EUR SWAP"], 0.99, delta, ["EUR"], _usd(1.10), _moves(1.0 + x), [3e6])
    row = table.set_index("position").loc["ALL IN USD"]
    first = delta[:, 0] * 1.10 + 3e6 * 1.10 * x
    assert row["dn_sigma"] == pytest.approx(first.std(ddof=1), rel=1e-12)
    assert row["dn_var"] == pytest.approx(2.3263478740 * first.std(ddof=1), rel=1e-9)
    held = table.set_index("position").loc["ALL IN USD (FX HELD)"]
    assert held["dn_sigma"] == pytest.approx((delta[:, 0] * 1.10).std(ddof=1), rel=1e-12)
    assert row["dn_sigma"] > held["dn_sigma"]


def test_reporting_in_euros_moves_a_dollar_position_the_other_way():
    x = np.array([0.01, -0.02, 0.0, 0.005])  # EURUSD moves
    fx = _moves(1.0 + x)
    rep = rp.Reporting("EUR", rp.entered_spots({"EURUSD": 1.10, "GBPUSD": 1.30}))
    cols = var.book_columns(np.zeros((4, 1)), ["UST"], ["USD"], rep, fx, [1e6])
    # the value in EUR of one dollar is 1 / EURUSD: it moves by 1 / (1 + x) - 1
    np.testing.assert_allclose(cols["ALL IN EUR (FX ONLY)"][0], 1e6 / 1.10 * (1.0 / (1.0 + x) - 1.0), rtol=1e-12)
    # the market move of EURUSD reads the same whichever currency reports
    for reporting, ccys in (("EUR", ["USD"]), ("USD", ["EUR"])):
        np.testing.assert_allclose(var.pair_moves(fx, ccys, reporting)["EURUSD"], 100 * x, rtol=1e-12, atol=1e-15)


def _store_with_fx(rates: dict) -> rh.HistoryStore:
    """The bundled history with its FX rows replaced by ``{date: USD per EUR}``
    (GBP, JPY, CHF at fixed rates)."""

    frame = rh.bundled_frame()
    frame = frame[frame["dataset"] != rh.FX_DATASET]
    rows = []
    for day, usd in rates.items():
        for ccy, value in (("USD", usd), ("GBP", 0.85), ("JPY", 170.0), ("CHF", 0.93)):
            rows.append((day, f"EXR.D.{ccy}.EUR.SP00.A", value))
    fx = rs.long_frame([r[0] for r in rows], "ECB", [r[1] for r in rows], [None] * len(rows), [r[2] for r in rows],
                       ["fx"] * len(rows), ["fx_per_eur"] * len(rows), [rh.FX_DATASET] * len(rows))  # fmt: skip
    return rh.build_store([frame, fx.assign(origin="test")], "synthetic FX")


def test_fx_moves_align_as_of_over_a_holiday_without_dropping_or_doubling():
    days = [D("2026-10-13"), D("2026-10-14"), D("2026-10-15"), D("2026-10-16"), D("2026-10-19")]
    rates = {D("2026-10-13"): 1.10, D("2026-10-14"): 1.12, D("2026-10-16"): 1.09, D("2026-10-19"): 1.11}  # no 15th
    moves = var.fx_moves(_store_with_fx(rates), days[:-1], days[1:])
    usd = moves.ratio["USD"]
    np.testing.assert_allclose(usd, [1.12 / 1.10, 1.0, 1.09 / 1.12, 1.11 / 1.09], rtol=1e-15)
    assert np.prod(usd) == pytest.approx(1.11 / 1.10, rel=1e-14)  # every move once
    assert moves.observed.all() and moves.coverage == 1.0
    stale = var.fx_moves(_store_with_fx({D("2026-10-01"): 1.10}), days[:-1], days[1:])
    assert not stale.observed.any() and (stale.ratio["USD"] == 1.0).all()  # too old: no move, not a fake one


def test_scenarios_carry_the_fx_history_and_can_hold_it():
    market = example_market()
    store = rh.bundled_store()
    scen = var.build_scenarios(market, store, 250)
    assert scen.fx is not None and scen.fx.coverage > 0.95 and "move with the ECB reference rates" in scen.fx_note
    table = scen.sources_table()
    fx_row = table[table["curve"] == var.FX_KEY].iloc[0]
    assert fx_row["dataset"] == "FX_ECB" and fx_row["flag"] == "real" and "USD" in fx_row["currency"]
    t = store.fx_table()
    both = [(i, p, d) for i, (p, d) in enumerate(zip(scen.previous, scen.dates)) if d in t.index and p in t.index]
    expected = [t.loc[d, "USD"] / t.loc[p, "USD"] for _, p, d in both]
    observed = [scen.fx.ratio["USD"][i] for i, _, _ in both]
    np.testing.assert_allclose(observed, expected, rtol=1e-15)
    held = var.build_scenarios(market, store, 250, fx=False)
    assert held.fx is None and held.fx_note == "FX spots held (as asked)"
    assert var.FX_KEY not in set(held.sources_table()["curve"])
    no_fx = rh.build_store([rh.bundled_frame().loc[lambda f: f["dataset"] != rh.FX_DATASET]], "no FX")
    bare = var.build_scenarios(market, no_fx, 250)
    assert bare.fx is None and "no ECB reference rates" in bare.fx_note


def test_a_mixed_book_on_the_committed_history():
    """A EUR swap book with a US Treasury, reported in USD: the FX-only VaR
    equals the VaR of the euro book's value times the daily EURUSD moves of
    the ECB history, read straight from the committed sample."""

    market, _, _ = curve_io.default_market(rh.bundled_store(), EXAMPLE_ASOF, "EUR ESTR")
    trades, _ = pf.parse_blotter(pf.example_blotter("EUR"))
    risk = pf.price_book(trades, market, pf.build_vols(market, trades)[0], vega=False)
    scen = var.build_scenarios(market, rh.bundled_store(), 500)
    pnl = var.sensitivity_pnl(scen, risk.risk)
    rep = rp.default_reporting(market.asof, "USD")
    currencies = list(risk.currencies)
    assert set(currencies) == {"EUR"}
    table = var.var_table(pnl, risk.ids, 0.99, None, currencies, rep, scen.fx, list(risk.pv)).set_index("position")
    t = rh.bundled_store().fx_table()
    eurusd = rep.fx.quote("EURUSD")
    held_days = list(t.index)

    def usd_on(day):  # the ECB rate of the day, or of the latest TARGET day before it (e.g. Boxing Day)
        return t.loc[max(d for d in held_days if d <= day), "USD"]

    x = np.array([usd_on(d) / usd_on(p) - 1.0 for p, d in zip(scen.previous, scen.dates)])
    assert np.count_nonzero(x == 0.0) < 0.05 * len(x)  # US-only business days: no FX move, the next day takes it
    fx_pnl = float(np.sum(risk.pv)) * eurusd * x
    assert table.loc["ALL IN USD (FX ONLY)", "var"] == pytest.approx(var.var_es(fx_pnl, 0.99)[0], rel=1e-9)
    assert table.loc["ALL IN USD (FX HELD)", "var"] == pytest.approx(table.loc["TOTAL", "var"] * eurusd, rel=1e-12)
    # FX moves the converted tail (ES); the 99% VaR itself can sit on a TARGET holiday with no FX move
    # (2025-04-21, Easter Monday, is the 5th worst day here), where both read the same
    assert table.loc["ALL IN USD", "es"] != pytest.approx(table.loc["ALL IN USD (FX HELD)", "es"], rel=1e-6)
    worst = var.worst_days(scen, pnl.sum(axis=1), 5, currencies, "USD")
    assert "EURUSD move (%)" in worst.columns
    i = scen.dates.index(worst["date"].iloc[0])
    assert worst["EURUSD move (%)"].iloc[0] == pytest.approx(100 * x[i], rel=1e-12)


# -- FX in historical replays -------------------------------------------------------------


def test_episodes_replay_the_ecb_fx_moves_of_their_dates():
    t = rh.bundled_store().fx_table()
    shock = sc.historical("2022 hiking cycle")
    fx = dict(shock.fx)
    a, b = t.loc[D("2022-01-03")], t.loc[D("2022-10-21")]
    assert fx["EURUSD"] == pytest.approx(100 * (b["USD"] / a["USD"] - 1), rel=1e-12)
    assert fx["GBPUSD"] == pytest.approx(100 * ((b["USD"] / b["GBP"]) / (a["USD"] / a["GBP"]) - 1), rel=1e-12)
    assert -15.0 < fx["EURUSD"] < -14.0 and -18.5 < fx["GBPUSD"] < -17.0  # the dollar's 2022 rally
    sources = sc.historical_sources(shock)
    row = sources[sources["curve"] == "FX SPOTS"].iloc[0]
    assert (row["flag"], row["start"], row["end"], row["currency"]) == ("real", "2022-01-03", "2022-10-21",
                                                                        "EURUSD,GBPUSD")  # fmt: skip
    assert "ECB reference rates 2022-01-03 to 2022-10-21" in shock.description
    assert dict(sc.historical("2013 taper tantrum").fx)["GBPUSD"] != 0.0  # the episode windows of 2013 are bundled
    doubled = dict(sc.historical("2022 hiking cycle", 2.0).fx)
    assert doubled["EURUSD"] == pytest.approx(2 * fx["EURUSD"], rel=1e-12)
    held = sc.historical("2022 hiking cycle", fx=False)
    assert held.fx == () and not held.moves_fx and "FX spots held" in held.description
    assert all(s.moves_fx for s in sc.standard_library() if s.name in sc.historical_episodes())
    assert not any(s.moves_fx for s in sc.standard_library(fx=False))


def test_a_window_without_fx_history_holds_fx_and_says_why():
    no_fx = rh.build_store([rh.bundled_frame().loc[lambda f: f["dataset"] != rh.FX_DATASET]], "no FX")
    shock = sc.history_window(D("2022-01-03"), D("2022-10-21"), store=no_fx)
    assert shock.fx == ()
    row = sc.historical_sources(shock).loc[lambda f: f["curve"] == "FX SPOTS"].iloc[0]
    assert row["flag"] == "held" and "no ECB euro foreign exchange reference rates history" in row["note"]
    window = sc.history_window(D("2020-03-09"), D("2020-03-18"))
    assert dict(window.fx)["GBPUSD"] < -9.0  # the 2020 dash for cash


def test_replayed_fx_drives_only_the_converted_total_of_the_scenario_grid():
    market, _, _ = curve_io.default_market(rh.bundled_store(), EXAMPLE_ASOF, "EUR ESTR")
    trades, _ = pf.parse_blotter(pf.example_blotter("EUR"))
    positions = list(pf.price_book(trades, market, pf.build_vols(market, trades)[0], vega=False).positions)
    rep = rp.default_reporting(market.asof, "USD")
    moved, held = sc.historical("2022 hiking cycle"), sc.historical("2022 hiking cycle", fx=False)
    grid = sc.scenario_grid(positions, market, [moved, held], reporting=rep)
    a, b = grid.iloc[0], grid.iloc[1]
    assert a["TOTAL"] == pytest.approx(b["TOTAL"], rel=1e-12)  # local P&L does not move with FX
    base = sum(float(p.pv(market)) for p in positions)
    eurusd = rep.fx.quote("EURUSD")
    expected = b["TOTAL"] * eurusd + (base + b["TOTAL"]) * eurusd * dict(moved.fx)["EURUSD"] / 100
    assert a["ALL IN USD"] == pytest.approx(expected, rel=1e-9)
    assert b["ALL IN USD"] == pytest.approx(b["TOTAL"] * eurusd, rel=1e-12)


def test_cli_scenario_replays_fx_unless_held(tmp_path, capsys):
    blotter = tmp_path / "mixed.csv"
    pd.concat([pf.example_blotter("EUR"), pf.example_blotter("USD")]).to_csv(blotter, index=False)
    args = ["scenario", "--trades", str(blotter), "--asof", "2026-10-19", "--conventions", "all",
            "--illustrative-vols", "--shock", "historical:2022-hiking-cycle"]  # fmt: skip
    outs = {}
    for hold in (False, True):
        out = tmp_path / f"grid_{hold}.csv"
        code = cli.main(args + ["--out", str(out)] + (["--hold-fx"] if hold else []))
        assert code in (0, 3), capsys.readouterr().err
        outs[hold] = pd.read_csv(out).iloc[0]
    capsys.readouterr()
    moved, held = outs[False], outs[True]
    assert moved["TOTAL EUR"] == pytest.approx(held["TOTAL EUR"], rel=1e-12)
    assert moved["TOTAL USD"] == pytest.approx(held["TOTAL USD"], rel=1e-12)
    assert moved["ALL IN USD"] != pytest.approx(held["ALL IN USD"], rel=1e-6)
