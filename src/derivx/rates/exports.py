"""Export tables of rates results (CSV-ready, long or tidy).

* :func:`curve_nodes` / :func:`curve_grid` / :func:`market_tables` - every
  curve's quotes and pillars (discount factor, continuously compounded zero
  rate, forward over each pillar interval) and a dated grid (discount
  factors, zero rates, 3M simple forwards).
* :func:`book_cashflows` - the remaining cash flows of a book's swaps,
  basis swaps and bonds, and each live caplet / floorlet.
* :func:`risk_report` - a book's PV, PV01, gamma, key-rate PV01s and
  gammas per curve, vega and vega buckets in one long table.

Units are in the column names: ``_pct`` columns are percent, ``_bp`` basis
points, ``df`` and decimals otherwise; times are ACT/365F years from the
valuation date; amounts are in the instrument's currency, signed to the
holder.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from derivx.rates.curves import DISCOUNT_KEY, GOVT_KEY, Curve, Market
from derivx.rates.portfolio import BookRisk

NODE_COLUMNS = ["curve", "pillar", "years", "df", "zero_cc_pct", "forward_cc_pct", "quote_kind", "quote_tenor",
                "quote", "quote_units"]  # fmt: skip
GRID_COLUMNS = ["curve", "date", "years", "df", "zero_cc_pct", "fwd_3m_simple_act360_pct"]


def _quote_units(kind: str) -> str:
    if kind in ("SR1", "SR3", "FUT"):
        return "futures rate pct (100 - price)"
    if kind == "BASIS":
        return "spread pct"
    return "rate pct"


def curve_nodes(curve: Curve, key: str = DISCOUNT_KEY) -> pd.DataFrame:
    """One row per pillar: date, years, discount factor, zero rate (cont.,
    ACT/365F, %), the continuously compounded forward from the previous
    pillar (%) and the quote solved there (in %). A projection curve's
    "discount factors" are its pseudo-discount factors (forwards come from
    their ratios; cash flows discount on the OIS curve)."""

    pillars = list(curve.pillar_dates or [])
    quotes = list(curve.quotes)
    rows = []
    for k in range(1, len(curve.ts)):
        t, df = float(curve.ts[k]), float(curve.dfs[k])
        prev_t, prev_df = float(curve.ts[k - 1]), float(curve.dfs[k - 1])
        quote = quotes[k - 1] if k - 1 < len(quotes) else None
        rows.append({
            "curve": key,
            "pillar": pillars[k - 1] if k - 1 < len(pillars) else None,
            "years": t,
            "df": df,
            "zero_cc_pct": -math.log(df) / t * 100.0,
            "forward_cc_pct": math.log(prev_df / df) / (t - prev_t) * 100.0,
            "quote_kind": quote.kind if quote else None,
            "quote_tenor": quote.tenor if quote else None,
            "quote": quote.rate * 100.0 if quote else None,
            "quote_units": _quote_units(quote.kind) if quote else None,
        })  # fmt: skip
    return pd.DataFrame(rows, columns=NODE_COLUMNS)


def curve_grid(curve: Curve, key: str = DISCOUNT_KEY, step_days: int = 7, long_step_days: int = 30) -> pd.DataFrame:
    """Dated grid: weekly to two years, monthly after, to the last pillar:
    discount factor, zero rate and the simple ACT/360 forward for the 91
    days from each date (%)."""

    horizon = int(float(curve.ts[-1]) * 365)
    days = np.unique(np.concatenate([np.arange(1, min(731, horizon + 1), step_days),
                                     np.arange(731, horizon + 1, long_step_days)])).astype(int)  # fmt: skip
    dates = [curve.asof + dt.timedelta(days=int(d)) for d in days]
    ends = [d + dt.timedelta(days=91) for d in dates]
    dfs = np.asarray(curve.df(dates), dtype=float)
    end_dfs = np.asarray(curve.df(ends), dtype=float)
    years = days / 365.0
    return pd.DataFrame({
        "curve": key,
        "date": dates,
        "years": years,
        "df": dfs,
        "zero_cc_pct": -np.log(dfs) / years * 100.0,
        "fwd_3m_simple_act360_pct": (dfs / end_dfs - 1.0) / (91 / 360.0) * 100.0,
    }, columns=GRID_COLUMNS)  # fmt: skip


def market_tables(market: Market) -> Dict[str, pd.DataFrame]:
    """``curve_nodes`` and ``curve_grid`` for every curve of the market
    (a projection that is the discount curve itself is left out)."""

    nodes, grids = [], []
    for key in market.curve_keys:
        if key not in (DISCOUNT_KEY, GOVT_KEY) and market.projections.is_discount(key):
            continue
        curve = market.get_curve(key)
        nodes.append(curve_nodes(curve, key))
        grids.append(curve_grid(curve, key))
    return {"curve_nodes": pd.concat(nodes, ignore_index=True), "curve_grid": pd.concat(grids, ignore_index=True)}


CASHFLOW_COLUMNS = ["id", "type", "leg", "fixing", "start", "end", "pay", "accrual", "rate_pct", "amount", "df", "pv",
                    "currency"]  # fmt: skip


def book_cashflows(risk: BookRisk, market: Market) -> pd.DataFrame:
    """Remaining cash flows of the book: swaps and basis swaps by leg (float
    rates are forwards or fixings), bonds' coupons and principal (discounted
    on the Treasury curve at the trade's z-spread), and caps' and floors'
    live caplets (expected payoff = PV / DF). FRAs, futures and swaptions
    have no fixed schedule of flows and are not listed."""

    rows: List[Dict[str, object]] = []
    for trade, position, ccy in zip(risk.trades, risk.positions, risk.currencies):
        inst = position.details["instrument"]
        if trade.type in ("swap", "basis swap"):
            flows = inst.cashflows(market)
            for f in flows.itertuples():
                rows.append({"id": trade.id, "type": trade.type, "leg": f.leg, "fixing": getattr(f, "fixing", None),
                             "start": f.start, "end": f.end, "pay": f.pay, "accrual": f.accrual,
                             "rate_pct": f.rate * 100.0, "amount": f.amount, "df": f.df, "pv": f.pv,
                             "currency": ccy})  # fmt: skip
        elif trade.type == "bond":
            notional, z = position.details["notional"], position.details["z_spread"]
            curve = market.govt
            scale = notional / inst.face
            for c in inst.coupons(market.asof):
                amount = (c.amount + (inst.face if c.end == inst.maturity else 0.0)) * scale
                t = curve.time(c.pay)
                df = float(curve.df(c.pay)) * math.exp(-z * t)
                rows.append({"id": trade.id, "type": "bond", "leg": "coupon+principal" if c.end == inst.maturity
                             else "coupon", "fixing": None, "start": c.start, "end": c.end, "pay": c.pay,
                             "accrual": c.accrual, "rate_pct": inst.coupon * 100.0, "amount": amount, "df": df,
                             "pv": amount * df, "currency": ccy})  # fmt: skip
        elif trade.type in ("cap", "floor"):
            for c in inst.caplets(market).itertuples():
                rows.append({"id": trade.id, "type": trade.type, "leg": f"{trade.type}let", "fixing": c.fix,
                             "start": c.start, "end": c.end, "pay": c.pay, "accrual": c.tau,
                             "rate_pct": c.forward * 100.0, "amount": c.pv / c.df if c.df else np.nan, "df": c.df,
                             "pv": c.pv, "currency": ccy})  # fmt: skip
    return pd.DataFrame(rows, columns=CASHFLOW_COLUMNS)


REPORT_COLUMNS = ["section", "id", "curve", "item", "value", "unit", "currency"]


def risk_report(risk: BookRisk, extra: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The book's risk in one long table: ``section`` is ``pv``, ``pv01``,
    ``gamma``, ``krd`` (key-rate PV01 of one quote), ``krd_gamma``,
    ``vega`` or ``vega_bucket``; ``id`` the trade (``TOTAL`` rows sum the
    trades of one currency); ``unit`` states the convention."""

    rows: List[Dict[str, object]] = []
    units = {"pv": "currency", "pv01": "P&L per +1bp on every rate quote of every curve",
             "gamma": "P&L per (1bp)^2 of that move", "vega": "P&L per +1bp normal vol"}  # fmt: skip
    for k, trade in enumerate(risk.trades):
        for section, values in (("pv", risk.pv), ("pv01", risk.pv01), ("gamma", risk.gamma), ("vega", risk.vega)):
            rows.append(
                {
                    "section": section,
                    "id": trade.id,
                    "curve": "",
                    "item": trade.type,
                    "value": float(values[k]),
                    "unit": units[section],
                    "currency": risk.currencies[k],
                }
            )
    for ccy in sorted(set(risk.currencies)):
        picks = [k for k, c in enumerate(risk.currencies) if c == ccy]
        for section, values in (("pv", risk.pv), ("pv01", risk.pv01), ("gamma", risk.gamma), ("vega", risk.vega)):
            rows.append(
                {
                    "section": section,
                    "id": "TOTAL",
                    "curve": "",
                    "item": f"{len(picks)} trades",
                    "value": float(np.sum(values[picks])),
                    "unit": units[section],
                    "currency": ccy,
                }
            )
    ladder = risk.ladder()
    for row in ladder.itertuples():
        rows.append(
            {
                "section": "krd",
                "id": row.id,
                "curve": row.curve,
                "item": row.quote,
                "value": row.pv01,
                "unit": "P&L per +1bp of this quote (curves rebuilt)",
                "currency": row.currency,
            }
        )
        rows.append(
            {
                "section": "krd_gamma",
                "id": row.id,
                "curve": row.curve,
                "item": row.quote,
                "value": row.gamma,
                "unit": "P&L per (1bp)^2 of this quote",
                "currency": row.currency,
            }
        )
    for row in risk.vega_buckets.itertuples():
        rows.append(
            {
                "section": "vega_bucket",
                "id": row.id,
                "curve": "",
                "item": f"{row.expiry} x {row.bucket}",
                "value": row.vega,
                "unit": f"P&L per +1bp normal vol ({row.method})",
                "currency": risk.currencies[risk.ids.index(row.id)],
            }
        )
    frame = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    return pd.concat([frame, extra], ignore_index=True) if extra is not None else frame
