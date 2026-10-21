"""QuantLib values of the EXAMPLE USD book (tests/test_portfolio.py).

Every trade of :func:`derivx.rates.portfolio.example_blotter` ("USD") is
priced by QuantLib on exact copies of the example market's log-linear SOFR
and Treasury curves (:func:`golden.common.ql_curve`), from the terms of the
derivx trade the blotter row becomes:

* swaps - ``ql.OvernightIndexedSwap`` on the derivx schedule dates
  (:func:`golden.swaps.ql_swap`);
* the SR3 future - contracts x $2,500 x (``ql.OvernightIndexFuture`` price
  with the derivx Hull-White convexity adjustment as its quote - traded
  price);
* the bond - ``ql.FixedRateBond`` with a ``DiscountingBondEngine`` on the
  Treasury curve copy;
* caps and floors - QuantLib Bachelier caplets on the derivx schedule, each
  at the derivx caplet vol (stripped vols or a flat vol: the vol is an
  input here; stripping is checked in tests/test_capvol.py);
* swaptions - ``ql.Swaption`` on QuantLib's own SOFR dates with a Bachelier
  engine at the derivx trade's vol (the SABR cube is checked in
  tests/test_swaption_vol.py).

Vols for trades without a flat vol come from the ILLUSTRATIVE USD swaption
grid and flat ATM cap quotes, as in the test. The book total is the sum.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

import QuantLib as ql

from derivx.rates import portfolio as pf
from derivx.rates.curve_io import HW_DEFAULT
from derivx.rates.examples import example_market
from derivx.rates.vol_examples import illustrative_cap_atm, illustrative_swaption_grid

from .bonds import ql_bond
from .capfloor import ql_value
from .common import qd, ql_curve, set_evaluation_date
from .swaps import ql_swap
from .swaptions import NOTIONAL as SWAPTION_NOTIONAL
from .swaptions import _engine, _usd_swaption, _vol_handle

POINT_VALUE = 2500.0  # SR3: $25 per bp per contract


def book():
    market = example_market()
    trades, issues = pf.parse_blotter(pf.example_blotter("USD"))
    assert not [i for i in issues if i.level == "error"], issues
    vols, notes = pf.build_vols(market, trades, illustrative_swaption_grid("USD"), illustrative_cap_atm("USD"),
                                "ILLUSTRATIVE USD flat ATM cap normal vols", hull_white=HW_DEFAULT)  # fmt: skip
    assert not notes, notes
    return market, trades, vols


def _future(position, market) -> float:
    fut = position.details["instrument"]
    start, end = fut.period(market.asof)
    handle = ql_curve(market.curve)
    adjustment = fut.convexity_adjustment(market)
    future = ql.OvernightIndexFuture(ql.Sofr(handle), qd(start), qd(end), ql.QuoteHandle(ql.SimpleQuote(adjustment)),
                                     ql.RateAveraging.Compound)  # fmt: skip
    return fut.contracts * POINT_VALUE * (future.NPV() - fut.trade_price), adjustment


def _bond(position, market) -> float:
    bond, notional = position.details["bond"], position.details["notional"]
    settle = bond.settlement_date(market.asof)
    qbond, _ = ql_bond(bond, settle, ql.Following)
    qbond.setPricingEngine(ql.DiscountingBondEngine(ql_curve(market.govt)))
    return qbond.NPV() * notional / bond.face


def _cap(position, market) -> float:
    cf = position.details["instrument"]
    table = cf.caplets(market)
    vols = {float(t): float(v) for t, v in zip(table["expiry"], table["vol"])}
    sign = 1.0 if cf.notional > 0 else -1.0  # QuantLib caps take positive nominals: a sold cap is minus a bought one
    return sign * ql_value(replace(cf, notional=abs(cf.notional), vol=vols), market)["npv"]


def _swaption(position, market) -> float:
    """QuantLib NPV of the swaption (each leg of a straddle), at the derivx
    trade's vol, on QuantLib's own dates for the blotter's expiry tenor."""

    trade = position.details["instrument"]
    expiry = str(position.details["blotter"].get("expiry"))
    disc = ql_curve(market.curve)
    vols = _vol_handle(market.asof, ql.UnitedStates(ql.UnitedStates.SOFR), trade.model, trade.vol, trade.shift)
    out = 0.0
    for payer in {"payer": (True,), "receiver": (False,), "straddle": (True, False)}[trade.kind]:
        swaption = _usd_swaption(market.asof, expiry, trade.tenor, trade.strike, payer, disc)
        assert swaption.exercise().dates()[0] == qd(trade.expiry_date), (expiry, trade.expiry_date)
        swaption.setPricingEngine(_engine(disc, vols, trade.model))
        out += swaption.NPV()
    return out * trade.notional / SWAPTION_NOTIONAL


def generate() -> dict:
    market, trades, vols = book()
    set_evaluation_date(market.asof)
    rows: Dict[str, Dict[str, object]] = {}
    for trade in trades:
        ql.IndexManager.instance().clearHistories()
        position = pf.trade_position(trade, market, vols)
        inst = position.details["instrument"]
        row: Dict[str, object] = {"type": trade.type}
        if trade.type == "swap":
            row["npv"] = ql_swap(inst, market.asof, ql_curve(market.curve)).NPV()
            row["fixed_rate"] = inst.fixed_rate
        elif trade.type == "future":
            row["npv"], row["convexity"] = _future(position, market)
        elif trade.type == "bond":
            row["npv"] = _bond(position, market)
        elif trade.type in ("cap", "floor"):
            row["npv"] = _cap(position, market)
            row["strike"] = inst.strike
        else:
            row["npv"] = _swaption(position, market)
            row["strike"], row["vol"] = inst.strike, inst.vol
        rows[trade.id] = row
    return {"asof": market.asof.isoformat(), "trades": rows, "total": sum(r["npv"] for r in rows.values())}
