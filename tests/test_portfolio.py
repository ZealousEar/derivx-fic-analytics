"""Trade blotter and portfolio risk: validation with row-level messages, the
book priced by the single-trade pricers (QuantLib golden values), risk that
adds up trade by trade, vega buckets, and the hedge optimiser's claims."""

from __future__ import annotations

import datetime as dt
import io
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivx.data.rates_history import bundled_store
from derivx.rates import curve_io
from derivx.rates import portfolio as pf
from derivx.rates.bonds import FixedRateBond, bond_position
from derivx.rates.capfloor import CapFloor
from derivx.rates.curves import CurveQuote, Market, bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, example_market, sofr_example_rows
from derivx.rates.stir import RateFuture
from derivx.rates.swaps import VanillaSwap, key_rate_matrix
from derivx.rates.swaption_vol import cube_vol, vega_buckets
from derivx.rates.swaptions import Swaption
from derivx.rates.vol_examples import illustrative_cap_atm, illustrative_swaption_grid
from derivx.risk import var as var_mod
from derivx.risk.scenarios import parallel, quote_risk, run_scenario
from derivx.risk.sensitivities import key_rate_pv01, parallel_pv01, rate_pv01_gamma
from derivx.vol.implied import lognormal_to_normal_vol

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def _usd():
    """Mirror of tools/golden/portfolio.py:book (example market, ILLUSTRATIVE
    vols, the default Hull-White convexity)."""

    market = example_market()
    trades, issues = pf.parse_blotter(pf.example_blotter("USD"))
    assert not issues
    cap_source = "ILLUSTRATIVE USD flat ATM cap normal vols"
    vols, notes = pf.build_vols(market, trades, illustrative_swaption_grid("USD"), illustrative_cap_atm("USD"),
                                cap_source, hull_white=curve_io.HW_DEFAULT)  # fmt: skip
    assert not notes
    return market, tuple(trades), vols, pf.price_book(trades, market, vols)


@lru_cache(maxsize=None)
def _eur():
    market, _, _ = curve_io.default_market(bundled_store(), EXAMPLE_ASOF, "EUR ESTR")
    trades, issues = pf.parse_blotter(pf.example_blotter("EUR"))
    assert not issues
    vols, notes = pf.build_vols(market, trades, illustrative_swaption_grid("EUR"), illustrative_cap_atm("EUR"),
                                hull_white=curve_io.HW_DEFAULT)  # fmt: skip
    assert not notes
    return market, tuple(trades), vols, pf.price_book(trades, market, vols)


def _row(**cells) -> dict:
    base = {c: None for c in pf.COLUMNS}
    base.update(id="T1", currency="USD", notional="10mm")
    base.update(cells)
    return base


def _issues(*rows) -> list:
    _, issues = pf.parse_blotter(pd.DataFrame(list(rows)))
    return issues


# --------------------------------------------------------------------------
# Blotter
# --------------------------------------------------------------------------


@pytest.mark.parametrize("currency", ["USD", "EUR", "GBP"])
def test_example_books_parse_cleanly_and_cover_the_types(currency):
    trades, issues = pf.parse_blotter(pf.example_blotter(currency))
    assert issues == [] and len(trades) >= 4
    assert len({t.id for t in trades}) == len(trades)
    if currency == "USD":
        assert {t.type for t in trades} == {"swap", "future", "bond", "cap", "floor", "swaption"}
    if currency == "EUR":
        assert {"basis swap", "fra", "future", "cap", "swaption"} <= {t.type for t in trades}


def test_blotter_round_trips_through_csv():
    trades, _ = pf.parse_blotter(pf.example_blotter("EUR"))
    table, issues = pf.read_blotter_csv(pf.blotter_csv(trades))
    again, more = pf.parse_blotter(table)
    assert issues == [] and more == []
    assert [replace(t, row=0) for t in again] == [replace(t, row=0) for t in trades]
    # a hand-written file with other column order, upper-case headers and a BOM
    text = '﻿ID,Type,Direction,Currency,Notional,Tenor,Index,Rate\nX1,swap,rec,USD,"1,000,000",5Y,OIS,3.5%\n'
    table, issues = pf.read_blotter_csv(text.encode("utf-8"))
    (trade,), more = pf.parse_blotter(table)
    assert issues == [] and more == []
    assert (trade.direction, trade.notional, trade.get("rate"), trade.get("tenor")) == ("receive", 1e6, 3.5, "5Y")


@pytest.mark.parametrize("text, value", [("10mm", 1e7), ("2.5bn", 2.5e9), ("500k", 5e5), ("1,000,000", 1e6),
                                         ("7.5M", 7.5e6), (25_000_000, 2.5e7)])  # fmt: skip
def test_notionals_in_desk_shorthand(text, value):
    trades, issues = pf.parse_blotter(pd.DataFrame([_row(type="swap", direction="pay", tenor="5Y", index="OIS",
                                                         notional=text)]))  # fmt: skip
    assert issues == [] and trades[0].notional == value


@pytest.mark.parametrize(
    "cells, level, text",
    [
        (dict(type="swpa", direction="pay"), "error", "type 'swpa' is not one of"),
        (dict(type="swap", direction="pay", tenor="5Y", maturity="2031-10-21", index="OIS"), "error", "not both"),
        (dict(type="swap", direction="pay", index="OIS"), "error", "not neither"),
        (dict(type="swap", direction="long", tenor="5Y", index="OIS"), "error", "a swap is receive or pay"),
        (dict(type="swap", direction="pay", tenor="5Y", index="OIS", rate=450), "error", "outside -10% to 30%"),
        (dict(type="swap", direction="pay", tenor="5Y", index="LIBOR"), "error", "use OIS, IRS or a term index"),
        (dict(type="swap", direction="pay", tenor="5Y", index="EURIBOR 6M"), "error", "is a EUR index, not USD"),
        (dict(type="swap", direction="pay", tenor="5Y", index="OIS", strike=3), "warning", "strike is not used by a"),
        (dict(type="swap", direction="pay", tenor="5Y", index="OIS", notional=-5), "error", "must be positive"),
        (dict(type="swap", direction="pay", tenor="5Y", index="OIS", notional="lots"), "error", "is not a number"),
        (dict(type="fra", direction="pay", index="EURIBOR 6M", currency="EUR"), "error", "tenor is missing"),
        (
            dict(type="fra", direction="pay", tenor="6x3", index="EURIBOR 6M", currency="EUR"),
            "error",
            "months to start x months to end",
        ),
        (
            dict(type="bond", direction="long", maturity="2035-08-15", rate=4.25, currency="EUR"),
            "error",
            "its currency is USD",
        ),
        (dict(type="bond", direction="long", maturity="15/08/2035", rate=4.25), "error", "use YYYY-MM-DD"),
        (
            dict(type="bond", direction="long", maturity="2035-08-15", rate=4.25, price=99, spread=10),
            "error",
            "a clean price or a z-spread, not both",
        ),
        (
            dict(type="future", direction="short", tenor="DEC26", index="SR3", price=96.2, notional=1.5),
            "error",
            "whole number of contracts",
        ),
        (
            dict(type="future", direction="short", tenor="DEC26", index="ED", price=96.2, notional=10),
            "error",
            "use SR1, SR3 or EURIBOR 3M",
        ),
        (dict(type="swaption", direction="long", expiry="1Y", tenor="5Y"), "error", "option is missing"),
        (
            dict(type="swaption", direction="long", expiry="1Y", tenor="5Y", option="call"),
            "error",
            "use payer, receiver or straddle",
        ),
        (dict(type="cap", direction="long", tenor="5Y", index="3M", model="shifted black"), "error", "needs a shift"),
        (dict(type="cap", direction="long", tenor="5Y", index="3M", vol=-5), "error", "vol must be zero or positive"),
        (dict(type="cap", direction="long", tenor="5Y", index="2M"), "error", "1M, 3M, 6M or 12M (a single-curve"),
        (dict(type="cap", direction="long", tenor="5Y", index="OIS 2M"), "error", "use OIS 1M, OIS 3M, OIS 6M or OIS"),
        (dict(type="cap", direction="long", tenor="5Y", index="3M", currency="CHF"), "error", "give USD, EUR or GBP"),
    ],
)
def test_validation_names_the_row_and_what_to_do(cells, level, text):
    issues = _issues(_row(**cells))
    found = [i for i in issues if text in i.message]
    assert found, [str(i) for i in issues]
    assert found[0].level == level and found[0].row == 1 and str(found[0]).startswith("row 1 (T1): ")


def test_rows_with_errors_are_left_out_and_ids_are_checked():
    rows = [
        _row(id="A", type="swap", direction="pay", tenor="5Y", index="OIS"),
        _row(id="A", type="swap", direction="pay", tenor="2Y", index="OIS"),
        _row(id="B", type="swap", direction="sideways", tenor="2Y", index="OIS"),
        {c: None for c in pf.COLUMNS},  # blank row: skipped
        _row(id=None, type="swap", direction="receive", tenor="10Y", index="OIS"),
    ]
    trades, issues = pf.parse_blotter(pd.DataFrame(rows))
    assert [t.id for t in trades] == ["A", "T5"]
    messages = [str(i) for i in issues]
    assert "row 2 (A): id A repeats row 1" in messages
    assert any(m.startswith("row 3 (B): direction 'sideways'") for m in messages)
    assert "row 5 (T5): no id: named T5" in messages
    _, issues = pf.parse_blotter(pd.DataFrame([{"id": "X", "type": "swap"}]))
    assert issues[0].row == 0 and "missing columns: currency, direction, notional" in issues[0].message


def test_column_guide_lists_every_column_and_type():
    guide = pf.column_guide()
    assert list(guide["column"]) == list(pf.COLUMNS)
    swaption = dict(zip(guide["column"], guide["swaption"]))
    assert swaption["expiry"] == "required" and swaption["vol"] == "optional" and swaption["rate"] == ""


# --------------------------------------------------------------------------
# Pricing: the book is the sum of the single-trade pricers
# --------------------------------------------------------------------------


def test_book_matches_the_quantlib_golden_values(golden):
    data = golden("portfolio")
    market, trades, _, risk = _usd()
    assert market.asof.isoformat() == data["asof"] and not risk.skipped
    for trade, position, pv in zip(risk.trades, risk.positions, risk.pv):
        case = data["trades"][trade.id]
        assert case["type"] == trade.type
        assert pv == pytest.approx(case["npv"], rel=1e-9, abs=1e-6), trade.id
        inst = position.details["instrument"]
        for name in ("fixed_rate", "strike", "vol"):
            if name in case:  # the inputs the QuantLib copy was given are still the book's
                assert getattr(inst, name) == pytest.approx(case[name], rel=1e-12), (trade.id, name)
    assert float(risk.pv.sum()) == pytest.approx(data["total"], rel=1e-12)
    total = risk.summary().iloc[-1]
    assert total["id"] == "TOTAL" and total["pv"] == pytest.approx(data["total"], rel=1e-12)


def test_book_matches_live_quantlib():
    pytest.importorskip("QuantLib")
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    from golden import portfolio as gp

    live = gp.generate()
    _, _, _, risk = _usd()
    for trade, pv in zip(risk.trades, risk.pv):
        assert pv == pytest.approx(live["trades"][trade.id]["npv"], rel=1e-9, abs=1e-6), trade.id


def test_blotter_rows_become_the_trades_they_describe():
    """Each row priced through the book equals the pricer built by hand from
    its terms, so directions, units and conventions map as documented."""

    market, trades, vols, risk = _usd()
    by_id = dict(zip(risk.ids, risk.pv))
    asof = market.asof
    swap = VanillaSwap.from_conventions("USD SOFR", "5Y", 0.036, 50e6, receive_fixed=True, trade_date=asof)
    assert by_id["US1"] == pytest.approx(swap.pv(market), rel=1e-12)
    pay = VanillaSwap.from_conventions("USD SOFR", "10Y", 0.037, 25e6, receive_fixed=False, trade_date=asof)
    assert by_id["US2"] == pytest.approx(pay.pv(market), rel=1e-12)
    fwd = VanillaSwap.from_conventions("USD SOFR", "2Y", 0.0, 20e6, receive_fixed=False, forward_start="1Y",
                                       trade_date=asof)  # fmt: skip
    assert by_id["US4"] == pytest.approx(0.0, abs=1e-6)  # blank rate: par on the valuation date
    assert risk.positions[3].details["instrument"].fixed_rate == pytest.approx(fwd.par_rate(market), rel=1e-12)
    future = RateFuture("SR3", "DEC26", 96.20, -100.0, None, curve_io.HW_DEFAULT)
    assert by_id["US5"] == pytest.approx(future.pv(market), rel=1e-12) and by_id["US5"] < 0  # short, price rose
    bond = FixedRateBond(0.0425, dt.date(2035, 8, 15))
    assert by_id["US6"] == pytest.approx(bond_position(bond, 1e7).pv(market), rel=1e-12)
    # US7/US8 are on SOFR compounded in arrears ("OIS 3M"), the way USD caps trade
    cap = CapFloor.from_overnight("USD SOFR", 25e6, "cap", 0.04, "5Y", vols.caplets["OIS 3M"])
    floor = CapFloor.from_overnight("USD SOFR", -25e6, "floor", 0.03, "5Y", 95e-4)
    assert risk.positions[6].details["instrument"] == cap.pinned(asof)
    assert by_id["US7"] == pytest.approx(cap.pv(market), rel=1e-12)
    assert by_id["US8"] == pytest.approx(floor.pv(market), rel=1e-12) and by_id["US8"] < 0  # sold floor
    probe = Swaption(2e7, "1Y", "10Y", 0.0, "payer").inputs(market)
    vol = cube_vol(vols.cubes["OIS"], probe.expiry_years, "10Y", probe.forward, probe.forward, "bachelier")
    payer = Swaption(2e7, "1Y", "10Y", probe.forward, "payer", vol)
    assert by_id["US9"] == pytest.approx(payer.pv(market), rel=1e-12)
    sold = Swaption(-2e7, "2Y", "5Y", 0.0325, "receiver", risk.positions[9].details["instrument"].vol)
    assert by_id["US10"] == pytest.approx(sold.pv(market), rel=1e-12) and by_id["US10"] < 0


def test_book_risk_is_the_sum_of_the_single_trade_risk():
    market, _, _, risk = _usd()
    for key, qr in risk.risk.items():
        for k, position in enumerate(risk.positions):
            single = np.array([r["pv01"] for r in key_rate_pv01(position.pv, market, which=key)])
            assert qr.pv01[:, k] == pytest.approx(single, rel=1e-9, abs=1e-7), (key, position.name)
        totals = risk.ladder_totals()
        assert totals[totals["curve"] == key]["pv01"].to_numpy() == pytest.approx(qr.pv01.sum(axis=1), rel=1e-12)
    for k, (trade, position) in enumerate(zip(risk.trades, risk.positions)):
        if trade.type == "bond":
            expected = parallel_pv01(position.pv, market, which="govt")
        else:
            expected, gamma = rate_pv01_gamma(position.pv, market)
            assert risk.gamma[k] == pytest.approx(gamma, rel=1e-6, abs=1e-6)
        assert risk.pv01[k] == pytest.approx(expected, rel=1e-9, abs=1e-7), trade.id
    assert risk.stacked().size == sum(len(qr.labels) for qr in risk.risk.values())


def test_book_key_rate_risk_is_the_scenario_engines():
    """The book's ladders are the same numbers the scenario engine and the
    VaR measure from their own bumps, so they plug into both."""

    market, _, _, risk = _usd()
    positions = list(risk.positions)
    for key, qr in risk.risk.items():
        theirs = quote_risk(positions, market, key)
        assert qr.pv01 == pytest.approx(theirs.pv01, rel=1e-9, abs=1e-7)
        assert qr.gamma == pytest.approx(theirs.gamma, rel=1e-6, abs=1e-5)
        assert qr.labels == theirs.labels and qr.asof == theirs.asof and qr.computed.all()
    rows = run_scenario(positions, market, parallel(10.0), risk.risk)
    assert sum(r["delta"] for r in rows) == pytest.approx(10.0 * risk.ladder_totals()["pv01"].sum(), rel=1e-9)
    scen = var_mod.build_scenarios(market, bundled_store(), 250)
    pnl = var_mod.sensitivity_pnl(scen, risk.risk)
    assert pnl.shape == (scen.n, len(positions)) and np.all(np.isfinite(pnl))


def test_vega_and_vega_buckets():
    market, _, vols, risk = _usd()
    buckets = risk.vega_buckets
    for k, (trade, position) in enumerate(zip(risk.trades, risk.positions)):
        inst = position.details["instrument"]
        mine = buckets[buckets["id"] == trade.id]
        if trade.type == "swaption":
            assert risk.vega[k] == pytest.approx(inst.greeks(market)["vega_normal_bp"], rel=1e-12)
            grid = vega_buckets([inst], vols.cubes["OIS"], market)
            assert mine["vega"].sum() == pytest.approx(grid.sum(), rel=1e-12)
            assert set(mine["method"]) == {"cube"}
        elif trade.type in ("cap", "floor"):
            assert risk.vega[k] == pytest.approx(inst.vega(market), rel=1e-12)  # normal model: vega per bp
            assert mine["vega"].sum() == pytest.approx(risk.vega[k], rel=1e-12)  # allocated, nothing lost
            assert set(mine["bucket"]) == {"CAP OIS 3M"}
        else:
            assert risk.vega[k] == 0.0 and mine.empty
    matrix = risk.vega_matrix()
    assert float(matrix.to_numpy().sum()) == pytest.approx(buckets["vega"].sum(), rel=1e-12)
    assert list(matrix.index) == list(vols.cubes["OIS"].expiries) and list(matrix.columns)[-1] == "CAP OIS 3M"


def test_flat_vol_and_black_vegas_are_in_normal_bp():
    market, _, _, _ = _usd()
    rows = [
        _row(id="S", type="swaption", direction="long", expiry="2Y", tenor="10Y", option="straddle", vol=30,
             model="black"),
        _row(id="C", type="cap", direction="short", tenor="4Y", index="3M", strike=3.8, vol=25, model="black"),
    ]  # fmt: skip
    trades, issues = pf.parse_blotter(pd.DataFrame(rows))
    assert issues == []
    risk = pf.price_book(trades, market)
    swaption, cap = (p.details["instrument"] for p in risk.positions)
    assert risk.vega[0] == pytest.approx(swaption.greeks(market)["vega_normal_bp"], rel=1e-12)
    mine = risk.vega_buckets[risk.vega_buckets["id"] == "S"]
    assert set(mine["method"]) == {"allocated"} and mine["vega"].sum() == pytest.approx(risk.vega[0], rel=1e-12)
    assert set(mine["expiry"]) == {"2Y"} and set(mine["bucket"]) == {"10Y"}  # on grid points: no split
    # Black cap: each caplet's normal-equivalent vega is its Black vega over d(normal vol)/d(Black vol).
    table = cap.caplets(market)
    live = table[~table["fixed"]]
    h = 1e-6
    total = 0.0
    for row in live.itertuples():
        up = float(lognormal_to_normal_vol(row.forward, cap.strike, row.expiry, row.vol + h))
        down = float(lognormal_to_normal_vol(row.forward, cap.strike, row.expiry, row.vol - h))
        total += row.vega / 0.01 * (2 * h) / (up - down) * 1e-4  # caplet vega per vol point -> per bp of normal
    assert risk.vega[1] == pytest.approx(total, rel=1e-5) and risk.vega[1] < 0  # sold cap


def test_book_caps_on_the_overnight_rate_compounded_in_arrears():
    """``OIS 3M`` caps are on the swap curve's overnight rate compounded in
    arrears (SOFR here: OIS day count, payment lag 2), like the CAP-FLOOR
    tab's; a Black one's normal-equivalent vega uses each caplet's
    Lyashenko-Mercurio option time."""

    market = example_market()
    rows = [
        _row(id="C", type="cap", direction="long", tenor="5Y", index="ois 3m", strike=4.0, vol=95),
        _row(id="B", type="floor", direction="short", tenor="3Y", index="3M OIS", strike=3.2, vol=30, model="black"),
        _row(id="T", type="cap", direction="long", tenor="5Y", index="3M", strike=4.0, vol=95),
    ]
    trades, issues = pf.parse_blotter(pd.DataFrame(rows))
    assert issues == [] and [t.get("index") for t in trades] == ["OIS 3M", "OIS 3M", "3M"]
    risk = pf.price_book(trades, market)
    cap, floor, term = (p.details["instrument"] for p in risk.positions)
    assert cap == CapFloor.from_overnight("USD SOFR", 1e7, "cap", 0.04, "5Y", 95e-4).pinned(market.asof)
    assert floor.compounded and floor.payment_lag == 2 and not term.compounded
    assert risk.pv[0] != pytest.approx(risk.pv[2], rel=1e-3)  # in arrears is not the term-rate approximation
    notes = "\n".join(risk.notes)
    assert "T is on a 3M term rate projected off the OIS curve: an approximation" in notes and "SOFR" in notes
    assert "C is on" not in notes and "B is on" not in notes
    table = floor.caplets(market)
    h = 1e-6
    total = 0.0
    for row in table[~table["fixed"]].itertuples():
        up = float(lognormal_to_normal_vol(row.forward, floor.strike, row.variance_time, row.vol + h))
        down = float(lognormal_to_normal_vol(row.forward, floor.strike, row.variance_time, row.vol - h))
        total += row.vega / 0.01 * (2 * h) / (up - down) * 1e-4
    assert risk.vega[1] == pytest.approx(total, rel=1e-5) and risk.vega[1] < 0  # sold floor
    gbp = pf.parse_blotter(pf.example_blotter("GBP"))[0]
    sonia = next(t for t in gbp if t.id == "GB3")
    assert sonia.get("index") == "OIS 3M"
    gbp_market = Market(bootstrap(market.asof, [CurveQuote(r["kind"], r["tenor"], r["rate"])
                                                 for r in sofr_example_rows()], "GBP SONIA"))  # fmt: skip
    terms = pf.cap_terms("OIS 3M", gbp_market)
    assert terms == dict(index_months=3, day_count="ACT/365F", calendar="LONDON", bdc="MF", fixing_lag=0,
                         payment_lag=0, rate_type="compounded")  # fmt: skip


def test_eur_book_on_the_multi_curve_market():
    market, trades, _, risk = _eur()
    assert not risk.skipped and set(risk.risk) == {"curve", "EURIBOR 6M", "EURIBOR 3M", "govt"}
    by_id = {t.id: k for k, t in enumerate(risk.trades)}
    basis = by_id["EU3"]  # blank spread: par 3s6s, worth nothing; rate moves leave it alone
    assert abs(risk.pv[basis]) < 1e-4 and abs(risk.pv01[basis]) < 1e-3
    three = risk.risk["EURIBOR 3M"]
    kinds = [market.get_curve("EURIBOR 3M").instruments[i].kind for i in range(len(three.labels))]
    spread_risk = sum(three.pv01[i, basis] for i, kind in enumerate(kinds) if kind == "BASIS")
    assert spread_risk == pytest.approx(-risk.positions[basis].details["instrument"].annuity(market), rel=1e-3)
    assert abs(risk.pv[by_id["EU4"]]) < 1e-6  # FRA at par
    swap = VanillaSwap.from_index("EURIBOR 6M", "10Y", 0.0365, 50e6, receive_fixed=True, trade_date=market.asof)
    assert risk.pv[by_id["EU1"]] == pytest.approx(swap.pv(market), rel=1e-12)
    sixes = risk.risk["EURIBOR 6M"].pv01[:, by_id["EU1"]]
    assert sixes.sum() < 0  # a receiver loses when EURIBOR quotes rise
    assert {"CAP EURIBOR 6M"} <= set(risk.vega_buckets["bucket"])


def test_trades_that_cannot_be_priced_are_skipped_with_the_reason():
    market = example_market()
    rows = [
        _row(id="E", currency="EUR", type="swap", direction="pay", tenor="5Y", index="OIS"),
        _row(id="V", type="swaption", direction="long", expiry="1Y", tenor="5Y", option="payer"),
        _row(id="C", type="cap", direction="long", tenor="5Y", index="3M"),
        _row(id="S", type="swap", direction="pay", tenor="5Y", index="OIS", start="2026-01-02"),
        _row(id="B", type="bond", direction="long", maturity="2025-11-15", rate=2.0),
        _row(id="P", type="swap", direction="pay", tenor="5Y", index="IRS"),
    ]
    trades, issues = pf.parse_blotter(pd.DataFrame(rows))
    assert issues == []
    risk = pf.price_book(trades, market)
    reasons = {t.id: r for t, r in risk.skipped}
    assert "switch the swap curve to EUR ESTR" in reasons["E"]
    assert "no swaption vol cube for OIS underlyings: give the trade a flat vol" in reasons["V"]
    assert "no stripped caplet vols for 3M caps" in reasons["C"]
    assert "before the valuation date" in reasons["S"] and "seasoned" in reasons["S"]
    assert "matured on 2025-11-15" in reasons["B"]
    assert risk.ids == ["P"] and abs(risk.pv[0]) < 1e-6  # single-curve IRS at par is priced


# --------------------------------------------------------------------------
# Hedging
# --------------------------------------------------------------------------


def _kkt(target, H, costs, move, x, tol=1e-6):
    """Optimality of move ||t + Hx|| + sum c|x|: for every instrument the
    marginal risk reduction balances (active) or stays under (inactive) its
    cost."""

    r = target + H @ x
    grad = move * H.T @ r / np.linalg.norm(r)
    scale = max(float(np.max(np.abs(move * H.T @ target / np.linalg.norm(target)))), 1.0)
    for j in range(x.size):
        if x[j] != 0.0:
            assert grad[j] + costs[j] * np.sign(x[j]) == pytest.approx(0.0, abs=tol * scale), j
        else:
            assert abs(grad[j]) <= costs[j] + tol * scale, j


def test_least_squares_hedge_without_costs():
    rng = np.random.default_rng(7)
    target = rng.normal(size=12) * 1000
    H = rng.normal(size=(12, 4)) * 300
    x = pf.optimise_hedge(target, H)
    assert H.T @ (target + H @ x) == pytest.approx(np.zeros(4), abs=1e-6)  # residual orthogonal to every hedge
    assert np.array_equal(pf.optimise_hedge(target, H, np.zeros(4), 5.0), x)  # no costs: least squares
    assert np.array_equal(pf.optimise_hedge(np.zeros(12), H, np.ones(4), 5.0), np.zeros(4))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cost_penalised_hedge_is_optimal(seed):
    rng = np.random.default_rng(seed)
    target = rng.normal(size=10) * 1000
    H = rng.normal(size=(10, 5)) * 250
    costs = rng.uniform(50, 500, size=5)
    move = 2.0
    x = pf.optimise_hedge(target, H, costs, move)
    _kkt(target, H, costs, move, x)
    best = pf.hedge_objective(target, H, x, costs, move)
    lstsq = pf.optimise_hedge(target, H)
    assert best <= pf.hedge_objective(target, H, lstsq, costs, move) + 1e-9
    assert best <= pf.hedge_objective(target, H, np.zeros(5), costs, move) + 1e-9
    for _ in range(50):
        other = x + rng.normal(size=5) * 0.05 * (np.abs(x) + 0.1)
        assert best <= pf.hedge_objective(target, H, other, costs, move) + 1e-9


def test_extreme_costs_and_moves():
    rng = np.random.default_rng(11)
    target = rng.normal(size=8) * 1000
    H = rng.normal(size=(8, 3)) * 200
    assert np.array_equal(pf.optimise_hedge(target, H, np.full(3, 1e12), 5.0), np.zeros(3))  # never worth it
    near = pf.optimise_hedge(target, H, np.full(3, 1e-3), 1e6)
    assert near == pytest.approx(pf.optimise_hedge(target, H), rel=1e-6)
    with pytest.raises(ValueError, match="positive number of bp"):
        pf.optimise_hedge(target, H, np.ones(3), 0.0)
    with pytest.raises(ValueError, match="one non-negative number per hedge"):
        pf.optimise_hedge(target, H, -np.ones(3), 5.0)


@pytest.mark.parametrize("move", [None, 5.0])
def test_hedge_leaves_the_residual_risk_it_claims(move):
    """Revaluing the book together with the sized hedges gives exactly the
    residual ladder the optimiser reports, and less risk than the book."""

    market, _, _, risk = _usd()
    bench = pf.default_benchmarks(risk, market)
    assert bench == {"curve": list(pf.HEDGE_TENORS), "govt": list(pf.HEDGE_TENORS)}
    instruments = pf.hedge_instruments(market, bench, bid_ask_bp=pf.assumed_bid_ask(market, bench))
    result = pf.hedge_book(risk, market, instruments, move)
    assert result.risk_after < 0.3 * result.risk_before
    assert result.risk_removed == pytest.approx(1.0 - (result.risk_after / result.risk_before) ** 2, rel=1e-12)
    hedged = []
    for key, qr in risk.risk.items():
        _, matrix = key_rate_matrix([p.pv for p in result.positions], market, key)
        hedged.append(qr.pv01.sum(axis=1) + matrix.sum(axis=1))
    actual = np.concatenate(hedged)
    assert actual == pytest.approx(result.ladder["residual"].to_numpy(), rel=1e-8, abs=1e-6)
    assert float(np.linalg.norm(actual)) == pytest.approx(result.risk_after, rel=1e-8)
    trades = result.table[result.table["notional"] != 0.0]
    assert len(trades) == len(result.positions)
    if move is None:
        assert result.move_bp is None and result.worst_before is None
    else:  # the penalty drops the small Treasury hedges and costs less
        plain = pf.hedge_book(risk, market, instruments, None)
        assert result.cost < plain.cost and result.risk_after >= plain.risk_after
        assert len(trades) < int((plain.table["notional"] != 0.0).sum())
        assert result.worst_after == pytest.approx(move * result.risk_after)
        assert set(trades["direction"]) <= {"Receive fixed", "Pay fixed", "Buy", "Sell"}


def test_hedge_instruments_are_par_and_validated():
    market, _, _, _ = _usd()
    units = pf.hedge_instruments(market, {"curve": ["5Y"], "govt": ["10Y"]}, bid_ask_bp={("govt", "10Y"): 0.5})
    swap, bond = units
    assert abs(swap.position.pv(market)) < 1e-6 and swap.quote_dv01 == pytest.approx(
        swap.position.details["notional"] * 1e-4 * 4.6, rel=0.05)  # fmt: skip
    assert bond.label == "10Y UST" and bond.bid_ask_bp == 0.5 and swap.bid_ask_bp == 0.0
    assert bond.position.pv(market) == pytest.approx(1e6, rel=2e-3)  # a par bond (dirty, valued today)
    with pytest.raises(ValueError, match="must be distinct"):
        pf.hedge_instruments(market, {"curve": ["5Y", "5Y"]})
    with pytest.raises(ValueError, match="whole years"):
        pf.hedge_instruments(market, {"govt": ["18M"]})
    with pytest.raises(ValueError, match="at least one hedge instrument"):
        pf.hedge_instruments(market, {"curve": []})
    no_govt = Market(curve=market.curve)
    with pytest.raises(ValueError, match="no Treasury curve"):
        pf.hedge_instruments(no_govt, {"govt": ["2Y"]})


def test_summary_and_ladder_tables():
    _, _, _, risk = _usd()
    summary = risk.summary()
    assert list(summary.columns) == ["id", "type", "description", "currency", "pv", "pv01", "gamma", "vega",
                                     "vol_source"]  # fmt: skip
    assert summary.iloc[-1]["pv01"] == pytest.approx(float(risk.pv01.sum()), rel=1e-12)
    assert summary.iloc[0]["description"] == "RECEIVE 5Y OIS 3.6%"
    ladder = risk.ladder()
    assert len(ladder) == len(risk.trades) * sum(len(qr.labels) for qr in risk.risk.values())
    assert set(ladder["currency"]) == {"USD"}
    buf = io.StringIO()
    risk.vega_buckets.to_csv(buf, index=False)
    assert buf.getvalue().startswith("id,expiry,bucket,vega,method")


# --------------------------------------------------------------------------
# Review findings (regressions)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cells, text",
    [
        (dict(notional="1,5mm"), "a comma is read only as a thousands separator"),  # was read as 15mm
        (dict(spread="2,5"), "a comma is read only as a thousands separator"),  # was read as 25bp
        (dict(rate="3,5"), "a comma is read only as a thousands separator"),
        (dict(notional="1,00,000"), "a comma is read only as a thousands separator"),
    ],
)
def test_decimal_commas_are_refused_not_misread(cells, text):
    row = _row(type="swap", direction="pay", tenor="5Y", index="OIS", **cells)
    trades, issues = pf.parse_blotter(pd.DataFrame([row]))
    assert trades == [] and any(text in i.message and i.level == "error" for i in issues), [str(i) for i in issues]


@pytest.mark.parametrize("text, value", [("1,000,000", 1e6), ("10,000mm", 1e10), ("2,500.5k", 2.5005e6)])
def test_thousands_separators_still_read(text, value):
    trades, issues = pf.parse_blotter(pd.DataFrame([_row(type="swap", direction="pay", tenor="5Y", index="OIS",
                                                         notional=text)]))  # fmt: skip
    assert issues == [] and trades[0].notional == value


@pytest.mark.parametrize("price", [0, 150, "96,2", 69.99, 110.01])
def test_futures_prices_outside_the_rate_range_are_refused(price):
    row = _row(type="future", direction="short", tenor="DEC26", index="SR3", price=price, notional=10)
    trades, issues = pf.parse_blotter(pd.DataFrame([row]))
    assert trades == [] and any(i.level == "error" and "price" in i.message for i in issues), [str(i) for i in issues]
    ok = _row(type="future", direction="short", tenor="DEC26", index="SR3", price=96.2, notional=10)
    assert pf.parse_blotter(pd.DataFrame([ok]))[1] == []


def test_bond_clean_price_must_be_positive():
    row = _row(type="bond", direction="long", maturity="2035-08-15", rate=4.25, price=0)
    _, issues = pf.parse_blotter(pd.DataFrame([row]))
    assert any("bond clean price must be positive" in i.message for i in issues)


@pytest.mark.parametrize(
    "cells, level, text",
    [
        (dict(vol=10000), "error", "above 2000bp for the normal model"),
        (dict(vol=500, model="black"), "error", "above 300% for the black model"),
        (dict(vol=0.9), "warning", "looks like a decimal"),
        (dict(vol=0.25, model="black"), "warning", "looks like a decimal"),
    ],
)
def test_implausible_vols_say_which_units(cells, level, text):
    row = _row(type="cap", direction="long", tenor="5Y", index="3M", strike=4, **cells)
    issues = _issues(row)
    found = [i for i in issues if text in i.message]
    assert found and found[0].level == level, [str(i) for i in issues]
    assert pf.parse_blotter(pd.DataFrame([_row(type="cap", direction="long", tenor="5Y", index="3M", vol=0)]))[1] == []


@pytest.mark.parametrize("name", ["TOTAL", "total", "Book", "TOTAL USD"])
def test_ids_of_the_book_totals_are_reserved(name):
    row = _row(id=name, type="swap", direction="pay", tenor="5Y", index="OIS")
    trades, issues = pf.parse_blotter(pd.DataFrame([row]))
    assert trades == [] and any("is reserved for the book's totals" in i.message for i in issues)


@pytest.mark.parametrize(
    "payload, text",
    [
        (b"", "the file is empty"),
        (b"   \n\n", "the file is empty"),
        (b"foo,bar\n1,2\n", "no blotter columns in the header (foo, bar)"),  # crashed the RATES page
        (b"\xff\xfe\x00garbage", "not UTF-8 text"),
        (b"id,type,ID\nA,swap,B\n", "columns id appear more than once"),
    ],
)
def test_unreadable_blotter_files_are_one_clear_error(payload, text):
    table, issues = pf.read_blotter_csv(payload)
    assert list(table.columns) == list(pf.COLUMNS) and table.empty
    assert len(issues) == 1 and issues[0].level == "error" and text in issues[0].message, [str(i) for i in issues]
    assert pf.parse_blotter(table) == ([], [])


def test_a_blotter_file_without_the_required_columns_says_which():
    table, issues = pf.read_blotter_csv(b"id\nX1\nX2\n")  # one column: used to crash the RATES page
    assert table.empty and len(issues) == 1 and issues[0].level == "error"
    assert "missing columns: type, currency, direction, notional" in issues[0].message
    _, issues = pf.read_blotter_csv(b"id,type,direction,notional,tenor,index\nA,swap,pay,1mm,5Y,OIS\n")
    assert [i.message.split(" (")[0] for i in issues] == ["missing columns: currency"]
    _, issues = pf.parse_blotter(pd.DataFrame([_row(type=None, direction="pay")]))
    assert issues[0].message.startswith("type is missing: give one of swap, basis swap")


def test_blotter_files_with_other_delimiters_and_spellings():
    text = "ID;Type;Currency;Direction;Notional;Tenor;Index;Expiry;Option;Model;Vol\n" \
           "A1;IRS;usd;rec;10mm;5Y;OIS;;;;\nA2;Swaption;USD;Buy;5mm;10Y;OIS;1Y;Payer;Bachelier;90\n" \
           "A3;swpa;USD;rec;1mm;5Y;OIS;;;;\n"  # fmt: skip
    table, issues = pf.read_blotter_csv(text.encode())
    assert issues == []
    # recognised spellings become the editor's options, so its select cells show them; others stay as typed
    assert list(table["type"]) == ["swap", "swaption", "swpa"]
    assert list(table["direction"]) == ["receive", "long", "receive"]
    assert list(table["currency"]) == ["USD"] * 3 and table.loc[1, "option"] == "payer"
    assert table.loc[1, "model"] == "normal" and table.loc[1, "vol"] == "90"
    trades, more = pf.parse_blotter(table)
    assert [t.id for t in trades] == ["A1", "A2"] and any("type 'swpa'" in i.message for i in more)
    tab = "id\ttype\tcurrency\tdirection\tnotional\ttenor\tindex\nT\tswap\tUSD\tpay\t1mm\t2Y\tOIS\n"
    (trade,), _ = pf.parse_blotter(pf.read_blotter_csv(tab.encode())[0])
    assert (trade.direction, trade.notional) == ("pay", 1e6)


def test_least_squares_hedge_leaves_out_redundant_combinations():
    """A hedge that is (almost) the average of two others: the exact least
    squares buys it against them with huge offsetting notionals for no risk;
    the hedge leaves that mix out and keeps the residual."""

    rng = np.random.default_rng(5)
    a, b = rng.normal(size=20) * 300, rng.normal(size=20) * 300
    H = np.column_stack([a, b, 0.5 * (a + b) + rng.normal(size=20) * 1e-4])
    target = rng.normal(size=20) * 1000
    exact, *_ = np.linalg.lstsq(H, -target, rcond=None)
    assert np.max(np.abs(exact)) > 1e3  # the defect: offsetting multiples of thousands of units
    x = pf.optimise_hedge(target, H)
    assert np.max(np.abs(x)) < 10.0
    best, mine = np.linalg.norm(target + H @ exact), np.linalg.norm(target + H @ x)
    assert best <= mine <= 1.05 * best
    assert np.array_equal(pf.optimise_hedge(target, np.column_stack([H, np.zeros(20)]))[:3], x)  # dead hedge: 0
    assert pf.optimise_hedge(target, np.zeros((20, 2))).tolist() == [0.0, 0.0]


def test_book_hedge_on_every_tenor_has_sane_notionals():
    """USD book hedged with 1Y-30Y OIS and Treasuries without costs: the
    exact least squares sold $358mm 10Y, bought $850mm 15Y and sold $495mm
    20Y Treasuries against a $10mm bond (bid/ask $445k) to take out 0.02% more
    risk."""

    market, _, _, risk = _usd()
    tenors = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
    bench = {"curve": tenors, "govt": tenors}
    instruments = pf.hedge_instruments(market, bench, bid_ask_bp=pf.assumed_bid_ask(market, bench))
    H = pf.unit_ladders(instruments, market, list(risk.risk))
    result = pf.hedge_book(risk, market, instruments, None, ladders=H)
    exact, *_ = np.linalg.lstsq(H, -risk.stacked(), rcond=None)
    assert np.linalg.norm(risk.stacked() + H @ exact) == pytest.approx(result.risk_after, rel=1e-3)
    gross = float(result.table["notional"].abs().sum())
    assert gross < 250e6 and float(np.abs(exact).sum()) * 1e6 > 1e9
    assert result.cost < 20_000.0


def test_trades_priced_beyond_the_market_data_say_so():
    """The SWAP and SWAPTION tabs warn when a trade runs past the curve's
    last quote or outside the vol cube; the book priced such trades silently
    (a 60Y swap on a 30Y curve, a 15Y-expiry swaption on a 10Y cube)."""

    market, _, vols, risk = _usd()
    assert risk.notes == ()  # the example book stays inside its data
    rows = [
        _row(id="S60", type="swap", direction="pay", tenor="60Y", index="OIS"),
        _row(id="FWD", type="swap", direction="pay", tenor="5Y", start="30Y", index="OIS"),
        _row(id="B75", type="bond", direction="long", maturity="2075-08-15", rate=4.0),
        _row(id="C40", type="cap", direction="long", tenor="40Y", index="OIS 3M", strike=4.0),
        _row(id="W15", type="swaption", direction="long", expiry="15Y", tenor="10Y", option="payer"),
        _row(id="IN", type="swap", direction="pay", tenor="30Y", index="OIS"),
    ]
    trades, issues = pf.parse_blotter(pd.DataFrame(rows))
    assert issues == []
    book = pf.price_book(trades, market, vols, vega=False)
    notes = "\n".join(book.notes)
    assert "S60 pays until 2086-10-21, after the last quote of the USD SOFR discount curve (2056-10-25)" in notes
    assert "FWD pays until" in notes and "B75 pays until 2075-08-15, after the last quote of the US Treasury" in notes
    assert "C40 has caplets fixing after the last stripped caplet" in notes
    assert "W15 is outside the swaption cube (10Y x 30Y at most)" in notes
    assert "IN " not in notes and "not market data" in notes
