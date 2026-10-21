"""Currency-aware totals (review finding R2 / A1: scenario totals added EUR
trade P&L to USD bond P&L without FX conversion).

Every aggregation of a book is per currency, and a book holding a currency
other than the reporting one is also converted at stated FX spots (the ECB
euro reference rates of the bundled snapshot, or rates the user gives):
FX spots and conversion, FX shocks, scenario grids and the P&L explain, the
VaR, factor risk, hedges (sized per currency), summaries and the risk
report, the CLI, and a book holding EUR, USD and GBP trades at once (each
currency on its own market)."""

from __future__ import annotations

import datetime as dt
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from derivx import cli
from derivx.data.rates_history import bundled_store
from derivx.fx.market_data import bundled_ecb_rates
from derivx.rates import curve_io, exports
from derivx.rates import portfolio as pf
from derivx.rates.examples import EXAMPLE_ASOF
from derivx.rates.vol_examples import illustrative_cap_atm, illustrative_swaption_grid
from derivx.risk import reporting as rp
from derivx.risk import var as var_mod
from derivx.risk.scenarios import Shock, fx_library, fx_spot, parallel, run_scenario, scenario_grid
from derivx.risk.sensitivities import Position

CONVENTIONS = {"USD": "USD SOFR", "EUR": "EUR ESTR", "GBP": "GBP SONIA"}


def _spots(eurusd: float = 1.10, gbpusd: float = 1.25) -> rp.FxSpots:
    return rp.entered_spots({"EURUSD": eurusd, "GBPUSD": gbpusd}, EXAMPLE_ASOF, "test rates")


@lru_cache(maxsize=None)
def _book(ccy: str, with_bond: bool = False):
    """(market, priced book) of the EXAMPLE book of ``ccy`` on that currency's
    market for EXAMPLE_ASOF, ILLUSTRATIVE vols; ``with_bond`` adds a long
    10mm US Treasury (USD) to it."""

    market, _, _ = curve_io.default_market(bundled_store(), EXAMPLE_ASOF, CONVENTIONS[ccy])
    table = pf.example_blotter(ccy).astype(object)
    if with_bond:
        bond = {c: None for c in pf.COLUMNS} | dict(id=f"UST_{ccy}", type="bond", currency="USD", direction="long",
                                                    notional="10mm", maturity="2035-08-15", rate=4.25)  # fmt: skip
        table = pd.concat([table, pd.DataFrame([bond])], ignore_index=True).astype(object)
    trades, issues = pf.parse_blotter(table)
    assert not [i for i in issues if i.level == "error"]
    vols, notes = pf.build_vols(market, trades, illustrative_swaption_grid(ccy), illustrative_cap_atm(ccy), "test",
                                hull_white=curve_io.HW_DEFAULT)  # fmt: skip
    assert not notes
    risk = pf.price_book(trades, market, vols)
    assert not risk.skipped
    return market, risk


# --------------------------------------------------------------------------
# FX spots
# --------------------------------------------------------------------------


def test_ecb_spots_take_the_latest_day_on_or_before_the_valuation_date():
    rates = bundled_ecb_rates()
    days = sorted(rates.history)
    on = rp.ecb_spots(rates, days[1])
    assert on.asof == days[1] and on.quote("EURUSD") == rates.history[days[1]]["USD"]
    assert on.quote("EURGBP") == rates.history[days[1]]["GBP"]
    between = rp.ecb_spots(rates, days[1] - dt.timedelta(days=1))  # a weekend: the Friday before
    assert between.asof == max(d for d in days if d < days[1])
    early = rp.ecb_spots(rates, dt.date(2020, 3, 18))  # before the snapshot: its first day, dated so
    assert early.asof == days[0]
    assert rp.Reporting("USD", early).stale_days(dt.date(2020, 3, 18)) == (days[0] - dt.date(2020, 3, 18)).days
    assert "ECB" in on.source and "bundled" in on.source


def test_spot_conversions_are_consistent_crosses():
    spots = _spots(1.10, 1.25)
    assert spots.quote("EURGBP") == pytest.approx(1.10 / 1.25, rel=1e-15)
    for a in rp.CURRENCIES:
        for b in rp.CURRENCIES:
            assert spots.value(a, b) * spots.value(b, a) == pytest.approx(1.0, rel=1e-15)
            for c in rp.CURRENCIES:
                assert spots.value(a, b) * spots.value(b, c) == pytest.approx(spots.value(a, c), rel=1e-14)
    usd = rp.Reporting("USD", spots)
    assert usd.rate("USD") == 1.0 and usd.rate("EUR") == 1.10 and usd.rate("GBP") == 1.25
    assert usd.pairs(["EUR", "GBP", "USD"]) == ["EURUSD", "GBPUSD"]
    assert usd.describe(["EUR", "USD"]).startswith("EURUSD 1.10000 (test rates, 2026-10-19")
    assert usd.describe(["USD"]) == "every amount is in USD: no conversion"
    gbp = rp.Reporting("GBP", spots)
    assert gbp.pairs(["EUR", "USD"]) == ["EURGBP", "GBPUSD"] and gbp.rate("EUR") == pytest.approx(1.10 / 1.25)


@pytest.mark.parametrize(
    "quotes, message",
    [({"EURUSD": 1.1}, "give GBPUSD"), ({"EURUSD": -1, "GBPUSD": 1.2}, "positive"),
     ({"EURUSD": float("nan"), "GBPUSD": 1.2}, "positive")],
)  # fmt: skip
def test_entered_spots_are_validated(quotes, message):
    with pytest.raises(ValueError, match=message):
        rp.entered_spots(quotes)


def test_reporting_currency_and_position_currencies():
    with pytest.raises(ValueError, match="use one of USD, EUR, GBP"):
        rp.Reporting("JPY", _spots())
    market, risk = _book("EUR", with_bond=True)
    assert rp.market_currency(market) == "EUR"
    assert rp.position_currencies(risk.positions, market) == list(risk.currencies)
    assert risk.currencies.count("USD") == 1 and risk.currencies[-1] == "USD"  # the Treasury
    bond = Position("UST", lambda m: 1.0, kind="bond")
    other = Position("X", lambda m: 1.0, kind="swap")
    tagged = Position("Y", lambda m: 1.0, kind="swap", details={"currency": "GBP"})
    assert rp.position_currencies([bond, other, tagged], market) == ["USD", "EUR", "GBP"]


# --------------------------------------------------------------------------
# FX shocks
# --------------------------------------------------------------------------


def test_fx_shocks_are_validated_and_move_only_fx():
    shock = fx_spot({"eurusd": 10})
    assert shock.fx == (("EURUSD", 10.0),) and shock.moves_fx and not shock.moves_local and not shock.moves_rates
    assert shock.name == "FX EURUSD +10%"
    for bad, text in (({"USDJPY": 5}, "use EURUSD or GBPUSD"), ({"EURUSD": -100}, "above -100%"),
                      ({}, "at least one FX move")):  # fmt: skip
        with pytest.raises(ValueError, match=text):
            fx_spot(bad)
    with pytest.raises(ValueError, match="twice"):
        Shock("x", fx=(("EURUSD", 1.0), ("eurusd", 2.0)))
    combined = Shock("both", profile=((0.0, 25.0),), fx={"GBPUSD": -5})
    assert combined.moves_local and combined.moves_fx and combined.fx == (("GBPUSD", -5.0),)


def test_translation_of_each_currency_into_the_reporting_one():
    shock = (("EURUSD", 10.0), ("GBPUSD", -5.0))
    assert list(rp.translation(["USD", "EUR", "GBP"], "USD", shock)) == pytest.approx([0.0, 0.10, -0.05])
    assert list(rp.translation(["USD", "EUR", "GBP"], "EUR", shock)) == pytest.approx(
        [1 / 1.1 - 1, 0.0, 0.95 / 1.1 - 1]
    )
    assert list(rp.translation(["USD", "EUR", "GBP"], "GBP", shock)) == pytest.approx(
        [1 / 0.95 - 1, 1.1 / 0.95 - 1, 0.0]
    )


@pytest.mark.parametrize(
    "currencies, reporting, pairs",
    [(["EUR", "USD"], "USD", ["EURUSD"]), (["USD"], "USD", []), (["USD"], "EUR", ["EURUSD"]),
     (["EUR", "GBP", "USD"], "USD", ["EURUSD", "GBPUSD"]), (["GBP"], "GBP", []),
     (["GBP"], "EUR", ["EURUSD", "GBPUSD"])],
)  # fmt: skip
def test_fx_library_moves_the_spots_the_conversion_uses(currencies, reporting, pairs):
    shocks = fx_library(currencies, reporting)
    assert [s.fx[0][0] for s in shocks[::2]] == pairs and [s.fx[0][1] for s in shocks[:2]] == (
        [10.0, -10.0][: len(shocks)]
    )


# --------------------------------------------------------------------------
# Totals
# --------------------------------------------------------------------------


def test_total_rows_never_add_currencies_unconverted():
    frame = pd.DataFrame({"id": ["a", "b", "c", "d"], "currency": ["EUR", "USD", "GBP", "EUR"],
                          "pv": [100.0, 200.0, 300.0, 50.0]})  # fmt: skip
    rep = rp.Reporting("USD", _spots(1.10, 1.25))
    table = rp.add_total_rows(frame, ["pv"], "id", reporting=rep, note="note")
    totals = table.set_index("id")
    assert list(table["id"])[4:] == ["TOTAL USD", "TOTAL EUR", "TOTAL GBP", "ALL IN USD"]
    assert totals.loc["TOTAL EUR", "pv"] == 150.0 and totals.loc["TOTAL EUR", "currency"] == "EUR"
    assert totals.loc["TOTAL GBP", "pv"] == 300.0 and totals.loc["TOTAL USD", "pv"] == 200.0
    assert totals.loc["ALL IN USD", "pv"] == pytest.approx(200.0 + 150.0 * 1.10 + 300.0 * 1.25, rel=1e-15)
    assert "EURUSD 1.10000" in totals.loc["ALL IN USD", "note"] and "GBPUSD 1.25000" in totals.loc["ALL IN USD", "note"]
    single = rp.add_total_rows(frame[frame["currency"] == "EUR"], ["pv"], "id", reporting=rp.Reporting("EUR", _spots()))
    assert list(single["id"])[-1] == "TOTAL" and single["pv"].iloc[-1] == 150.0  # one currency, reported in it
    eur_in_usd = rp.add_total_rows(frame[frame["currency"] == "EUR"], ["pv"], "id", reporting=rep)
    assert list(eur_in_usd["id"])[-2:] == ["TOTAL", "ALL IN USD"]
    unconverted = rp.add_total_rows(frame, ["pv"], "id")  # no FX spots: per currency only, no book total
    assert list(unconverted["id"])[4:] == ["TOTAL USD", "TOTAL EUR", "TOTAL GBP"]


def test_grid_totals_convert_and_translate_fx_shocks():
    grid = pd.DataFrame([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]], index=["rates", "fx"], columns=["e", "u", "g"])
    base = [1000.0, 2000.0, 3000.0]
    rep = rp.Reporting("USD", _spots(1.10, 1.25))
    fx = [(), (("EURUSD", 10.0), ("GBPUSD", -5.0))]
    table = rp.add_total_columns(grid, ["EUR", "USD", "GBP"], rep, base, fx)
    assert list(table.columns) == ["e", "u", "g", "TOTAL USD", "TOTAL EUR", "TOTAL GBP", "ALL IN USD"]
    assert table.loc["rates", "ALL IN USD"] == pytest.approx(20.0 + 10.0 * 1.10 + 30.0 * 1.25)
    assert table.loc["fx", "TOTAL EUR"] == 0.0 and table.loc["fx", "TOTAL GBP"] == 0.0
    assert table.loc["fx", "ALL IN USD"] == pytest.approx(1000.0 * 1.10 * 0.10 + 3000.0 * 1.25 * -0.05)
    with pytest.raises(ValueError, match="today's value"):
        rp.add_total_columns(grid, ["EUR", "USD", "GBP"], rep, None, fx)


def test_scenario_grid_on_a_eur_book_with_a_treasury():
    market, risk = _book("EUR", with_bond=True)
    rep = rp.Reporting("USD", _spots(1.10, 1.25))
    positions = list(risk.positions)
    shocks = [parallel(25.0), fx_spot({"EURUSD": 10.0}), Shock("both", profile=((0.0, 25.0),), fx={"EURUSD": 10})]
    grid = scenario_grid(positions, market, shocks, rep)
    names = [p.name for p in positions]
    assert list(grid.columns) == names + ["TOTAL USD", "TOTAL EUR", "ALL IN USD"]
    eur = [n for n, c in zip(names, risk.currencies) if c == "EUR"]
    base = {p.name: float(p.pv(market)) for p in positions}
    row = grid.loc["Parallel +25bp"]
    assert row["TOTAL EUR"] == pytest.approx(row[eur].sum(), rel=1e-12)
    assert row["ALL IN USD"] == pytest.approx(row["TOTAL USD"] + 1.10 * row["TOTAL EUR"], rel=1e-12)
    fx = grid.loc["FX EURUSD +10%"]
    assert (fx[names] == 0.0).all()
    assert fx["ALL IN USD"] == pytest.approx(0.10 * 1.10 * sum(base[n] for n in eur), rel=1e-12)
    both = grid.loc["both"]
    assert both["TOTAL EUR"] == pytest.approx(row["TOTAL EUR"], rel=1e-12)  # FX moves no local value
    after = sum(base[n] + both[n] for n in eur)
    assert both["ALL IN USD"] == pytest.approx(row["ALL IN USD"] + 0.10 * 1.10 * after, rel=1e-12)
    usd_only = scenario_grid(positions[:-1][:0] + [positions[-1]], market, shocks[:2], rp.Reporting("USD", _spots()))
    assert list(usd_only.columns) == [positions[-1].name, "TOTAL"]  # all dollars reported in dollars: no conversion


def test_explain_rows_carry_their_currency_and_the_fx_translation():
    market, risk = _book("EUR", with_bond=True)
    rep = rp.Reporting("USD", _spots(1.10, 1.25))
    shock = Shock("both", profile=((0.0, 25.0),), fx={"EURUSD": 10}, horizon_days=30)
    rows = run_scenario(list(risk.positions), market, shock)
    assert [r["currency"] for r in rows] == list(risk.currencies)
    frame = pd.DataFrame(rows).drop(columns="delta_by_curve")
    ccys = list(frame["currency"])
    extra = rp.explain_fx(frame["shocked_pv"], frame["cash"], ccys, rep, shock.fx)
    eur = frame[frame["currency"] == "EUR"]
    assert extra["fx"] == pytest.approx(0.10 * 1.10 * (eur["shocked_pv"] + eur["cash"]).sum(), rel=1e-12)
    values = ["base_pv", "shocked_pv", "cash", "pnl", "delta", "gamma", "cross_gamma", "vol", "carry", "residual"]
    frame["fx"] = np.nan
    table = rp.add_total_rows(frame, values, "position", reporting=rep, extra=extra).set_index("position")
    book = table.loc["ALL IN USD"]
    explained = sum(book[c] for c in ("delta", "gamma", "cross_gamma", "vol", "carry", "residual", "fx"))
    assert book["pnl"] == pytest.approx(explained, rel=1e-12)
    assert book["pnl"] == pytest.approx(book["shocked_pv"] + book["cash"] - book["base_pv"], rel=1e-12)


# --------------------------------------------------------------------------
# A book in EUR, USD and GBP
# --------------------------------------------------------------------------


def test_a_eur_usd_gbp_book_is_totalled_per_currency_and_converted():
    books = [_book("USD")[1], _book("EUR", with_bond=True)[1], _book("GBP")[1]]
    rep = rp.Reporting("EUR", _spots(1.10, 1.25))
    table = pf.summary_table(books, rep).set_index("id")
    assert list(table.index)[-4:] == ["TOTAL USD", "TOTAL EUR", "TOTAL GBP", "ALL IN EUR"]
    trades = table[~table.index.str.startswith(("TOTAL", "ALL IN"))]
    for ccy in ("USD", "EUR", "GBP"):
        assert table.loc[f"TOTAL {ccy}", "pv"] == pytest.approx(
            trades[trades["currency"] == ccy]["pv"].sum(), rel=1e-12
        )
    expected = sum(trades[trades["currency"] == c]["pv"].sum() * rep.rate(c) for c in ("USD", "EUR", "GBP"))
    assert table.loc["ALL IN EUR", "pv"] == pytest.approx(expected, rel=1e-12)
    assert table.loc["ALL IN EUR", "currency"] == "EUR"
    report = exports.risk_report(books, reporting=rep)
    pv = report[report["section"] == "pv"].set_index("id")
    assert pv.loc["ALL IN EUR", "value"] == pytest.approx(expected, rel=1e-12)
    assert set(report[report["section"] == "krd"]["currency"]) == {"USD", "EUR", "GBP"}
    buckets = report[report["section"] == "vega_bucket"]
    assert set(buckets["currency"]) == {"USD", "EUR", "GBP"}


def test_var_of_a_eur_usd_gbp_book_per_currency_and_converted():
    store = bundled_store()
    parts = []
    for ccy, bond in (("USD", False), ("EUR", True), ("GBP", False)):
        market, risk = _book(ccy, bond)
        scen = var_mod.build_scenarios(market, store, 250)
        parts.append((scen, var_mod.sensitivity_pnl(scen, risk.risk), risk))
    assert parts[0][0].dates == parts[1][0].dates == parts[2][0].dates  # the same scenario days for every market
    pnl = np.hstack([p for _, p, _ in parts])
    names = [t.id for _, _, r in parts for t in r.trades]
    ccys = [c for _, _, r in parts for c in r.currencies]
    rep = rp.Reporting("USD", _spots(1.10, 1.25))
    table = var_mod.var_table(pnl, names, 0.99, currencies=ccys, reporting=rep).set_index("position")
    assert list(table.index)[-4:] == ["TOTAL USD", "TOTAL EUR", "TOTAL GBP", "ALL IN USD"]
    for ccy in ("USD", "EUR", "GBP"):
        picks = [j for j, c in enumerate(ccys) if c == ccy]
        assert table.loc[f"TOTAL {ccy}", "var"] == var_mod.var_es(pnl[:, picks].sum(axis=1), 0.99)[0]
        assert table.loc[f"TOTAL {ccy}", "currency"] == ccy
    converted = (pnl * rep.rates(ccys)).sum(axis=1)
    assert table.loc["ALL IN USD", "var"] == var_mod.var_es(converted, 0.99)[0]
    unconverted = var_mod.var_es(pnl.sum(axis=1), 0.99)[0]  # what adding euros, pounds and dollars would give
    assert table.loc["ALL IN USD", "var"] != pytest.approx(unconverted, rel=1e-3)
    report = var_mod.risk_report(names, {}, table.reset_index(), currencies=ccys)
    var_rows = report[report["section"] == "var"]
    assert set(var_rows[var_rows["position"] == "ALL IN USD"]["currency"]) == {"USD"}
    assert set(var_rows[var_rows["position"] == "TOTAL GBP"]["currency"]) == {"GBP"}


def test_hedges_are_sized_per_currency():
    market, risk = _book("EUR", with_bond=True)
    bench = pf.default_benchmarks(risk, market)
    assert set(bench) == {"curve", "EURIBOR 6M", "EURIBOR 3M", "govt"}
    costs = pf.assumed_bid_ask(market, bench)
    instruments = pf.hedge_instruments(market, bench, bid_ask_bp=costs)
    result = pf.hedge_book(risk, market, instruments, 5.0)
    assert set(result.by_currency) == {"USD", "EUR"}
    assert set(result.by_currency["USD"].curves) == {"govt"}
    assert set(result.table["currency"]) == {"USD", "EUR"}
    for ccy, part in result.by_currency.items():
        rows = result.ladder[result.ladder["currency"] == ccy]
        assert part.risk_before == pytest.approx(float(np.linalg.norm(rows["book"])), rel=1e-12)
        assert part.risk_after == pytest.approx(float(np.linalg.norm(rows["residual"])), rel=1e-12)
        assert part.cost == pytest.approx(result.table[result.table["currency"] == ccy]["cost"].sum(), rel=1e-12)
    with pytest.raises(ValueError, match="does not add across currencies"):
        _ = result.risk_before
    # The dollar block is the hedge of the Treasury ladder alone, with the dollar costs and move.
    usd = [j for j, h in enumerate(instruments) if h.curve == "govt"]
    rows = np.flatnonzero(np.array([k == "govt" for k, qr in risk.risk.items() for _ in qr.labels]))
    target = risk.stacked()[rows]
    H = result.unit_ladders[np.ix_(rows, usd)]
    unit = np.array([0.5 * instruments[j].bid_ask_bp * instruments[j].quote_dv01 for j in usd])
    alone = pf.optimise_hedge(target, H, unit, 5.0)
    np.testing.assert_allclose(result.multiples[usd], alone, rtol=1e-9, atol=1e-12)
    single = pf.hedge_book(_book("USD")[1], _book("USD")[0], pf.hedge_instruments(
        _book("USD")[0], pf.default_benchmarks(_book("USD")[1], _book("USD")[0])), None)  # fmt: skip
    assert list(single.by_currency) == ["USD"] and math.isfinite(single.risk_after)  # one currency: one total


def test_reserved_ids_include_the_converted_total():
    trade, issues = pf.parse_trade({"id": "ALL IN USD", "type": "swap", "currency": "USD", "direction": "pay",
                                    "notional": "1mm", "tenor": "5Y", "index": "OIS"}, 1)  # fmt: skip
    assert trade is None and any("reserved" in i.message for i in issues)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mixed")
    frames = [pf.example_blotter(c) for c in ("USD", "EUR", "GBP")]
    path = folder / "mixed.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def test_cli_risk_prices_a_eur_usd_gbp_book_on_each_currency_market(mixed, tmp_path, capsys):
    out, manifest = tmp_path / "risk.csv", tmp_path / "run.json"
    code = cli.main(["risk", "--trades", str(mixed), "--asof", "2026-10-19", "--conventions", "all",
                     "--illustrative-vols", "--report-ccy", "GBP", "--fx", "EURUSD=1.10,GBPUSD=1.25", "--hedge",
                     "2Y,5Y,10Y,30Y", "--out", str(out), "--manifest", str(manifest)])  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0 and "21 trades priced, 0 skipped" in text
    assert "ALL IN GBP converts at EURGBP 0.88000, GBPUSD 1.25000 (FX rates given with --fx)" in text
    for ccy in ("USD", "EUR", "GBP"):
        assert f"hedge, {ccy} curves: residual risk" in text and f"({ccy} market)" in text
    report = pd.read_csv(out)
    pv = report[report["section"] == "pv"].set_index("id")
    totals = {c: pv.loc[f"TOTAL {c}", "value"] for c in ("USD", "EUR", "GBP")}
    expected = totals["USD"] / 1.25 + totals["EUR"] * 1.10 / 1.25 + totals["GBP"]
    assert pv.loc["ALL IN GBP", "value"] == pytest.approx(expected, rel=1e-9)
    payload = __import__("json").loads(manifest.read_text())
    assert payload["inputs"]["reporting"]["reporting_currency"] == "GBP"
    assert payload["inputs"]["reporting"]["fx"]["EURUSD"] == pytest.approx(1.10)
    assert set(payload["data"]["markets"]) == {"USD", "EUR", "GBP"}
    # One market only: the other currencies are skipped with the hint, never priced on the wrong curve.
    assert cli.main(["risk", "--trades", str(mixed), "--asof", "2026-10-19", "--illustrative-vols"]) == cli.EXIT_TRADES
    assert "or --conventions all to price every currency on its own market" in capsys.readouterr().err


def test_cli_scenario_with_fx_shocks_on_a_eur_usd_gbp_book(mixed, tmp_path, capsys):
    grid, explain = tmp_path / "grid.csv", tmp_path / "explain.csv"
    code = cli.main(["scenario", "--trades", str(mixed), "--asof", "2026-10-19", "--conventions", "all",
                     "--illustrative-vols", "--shock", "parallel:+25", "--shock", "fx:EURUSD=+10,GBPUSD=-5",
                     "--fx", "EURUSD=1.10,GBPUSD=1.25", "--out", str(grid), "--explain", str(explain)])  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0 and "FX EURUSD +10%, GBPUSD -5%" in text
    table = pd.read_csv(grid).set_index("scenario")
    assert {"TOTAL USD", "TOTAL EUR", "TOTAL GBP", "ALL IN USD"} <= set(table.columns)
    rates = table.loc["Parallel +25bp"]
    assert rates["ALL IN USD"] == pytest.approx(rates["TOTAL USD"] + 1.10 * rates["TOTAL EUR"] + 1.25 * rates[
        "TOTAL GBP"], rel=1e-10)  # fmt: skip
    rows = pd.read_csv(explain)
    fx = rows[rows["scenario"] == "FX EURUSD +10%, GBPUSD -5%"].set_index("position")
    trades = fx[~fx.index.str.startswith(("TOTAL", "ALL IN"))]
    value = {c: (trades[trades["currency"] == c]["shocked_pv"] + trades[trades["currency"] == c]["cash"]).sum()
             for c in ("EUR", "GBP")}  # fmt: skip
    translation = value["EUR"] * 1.10 * 0.10 + value["GBP"] * 1.25 * -0.05
    assert fx.loc["ALL IN USD", "fx"] == pytest.approx(translation, rel=1e-9)
    assert table.loc["FX EURUSD +10%, GBPUSD -5%", "ALL IN USD"] == pytest.approx(translation, rel=1e-9)
    assert (trades["pnl"] == 0.0).all()
    assert cli.main(["scenario", "--trades", str(mixed), "--asof", "2026-10-19", "--shock", "fx:USDJPY=+3"]) == 2
    assert "write EURUSD=+10" in capsys.readouterr().err
