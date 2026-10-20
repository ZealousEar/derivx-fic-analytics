"""Multi-curve risk views shared by the Rates tabs: curve names, cached
bumped curve sets and per-curve key-rate ladder charts.

Every ladder is the P&L for +1bp on one quote of one curve (central
difference, curves rebuilt; a discount-curve bump also rebuilds the
projection curves on it with their quotes held), signed to the holder.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from derivx.rates.curves import DISCOUNT_KEY, GOVT_KEY, SPREAD_KINDS, Curve, Market
from derivx.rates.swaps import CurveBumps, curve_bumps, key_rate_matrix
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, SERIES, dark_layout

CACHE_ENTRIES = 12
NOISE = 0.005  # currency per bp: smaller buckets are rebuild rounding, shown as zero
FIXING_NAMES = {"USD": "SOFR", "EUR": "€STR", "GBP": "SONIA"}
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}


def money(market: Market, key: str = DISCOUNT_KEY) -> str:
    """Currency symbol of values priced on a market's curve ``key`` (the
    swap curve's currency by default; the Treasury curve is USD)."""

    if key == GOVT_KEY:
        return "$"
    conv = market.curve.conventions
    currency = conv.currency if conv is not None else "USD"
    return CURRENCY_SYMBOLS.get(currency, currency + " ")


def ladder_sums(market: Market, ladders: Dict[str, Tuple[List[Dict[str, object]], np.ndarray]]) -> Tuple[float, float]:
    """(sum of the rate-quote buckets, sum of the tenor basis spread
    buckets) of per-curve ladders ``{curve key: (rows, values)}``. The first
    is the P&L for +1bp on every rate quote (the PV01 with basis held); the
    second for +1bp on every basis spread. Adding them would mix the two."""

    rates = spreads = 0.0
    for key, (rows, values) in ladders.items():
        curve = market.get_curve(key)
        for row, value in zip(rows, np.ravel(values)):
            if curve.instruments[int(row["index"])].kind in SPREAD_KINDS:
                spreads += float(value)
            else:
                rates += float(value)
    return rates, spreads


def curve_key(curve: Optional[Curve]) -> tuple:
    """Identifies a curve by what it was built from."""

    if curve is None:
        return ()
    if not curve.quotes:
        return (id(curve),)
    index = getattr(curve, "index", None)
    basis = getattr(curve, "basis", None)
    return (
        curve.asof,
        tuple(curve.quotes),
        curve.interpolation,
        curve.conventions,
        index.name if index is not None else None,
        curve.convexity,
        tuple(sorted(curve.overnight_fixings.items())) if curve.overnight_fixings else (),
        curve_key(basis) if basis is not None else (),
    )


def market_key(market: Market) -> tuple:
    """Identifies a market's curves, fixings and vol shifts."""

    projections = tuple(
        (k, "discount" if market.projections.is_discount(k) else curve_key(market.projections[k]))
        for k in market.projections
    )
    fixings = tuple((k, tuple(sorted(v.items()))) for k, v in sorted(market.index_fixings.items()))
    return (
        curve_key(market.curve),
        curve_key(market.govt),
        curve_key(market.fixings),
        projections,
        fixings,
        market.vol_shift_normal,
        market.vol_shift_lognormal,
    )


def curve_title(market: Market, key: str) -> str:
    """Display name of a curve: ``€STR OIS (DISCOUNT)``, ``EURIBOR 6M
    (PROJECTION)`` or ``TREASURY``."""

    if key == GOVT_KEY:
        return "TREASURY"
    if key == DISCOUNT_KEY:
        conv = market.curve.conventions
        fixing = FIXING_NAMES.get(conv.currency, conv.name) if conv is not None else "OIS"
        return f"{fixing} OIS (DISCOUNT)" if market.is_multi_curve else f"{fixing} OIS"
    return f"{key} (PROJECTION)"


def risk_keys(market: Market) -> Tuple[str, ...]:
    """Curves a swap or cap is bucketed on: discount and projections with
    quotes (a projection that is the discount curve is not a curve of its
    own)."""

    out = []
    for key in market.curve_keys:
        if key == GOVT_KEY:
            continue
        if key != DISCOUNT_KEY and market.projections.is_discount(key):
            continue
        if market.get_curve(key).quotes:
            out.append(key)
    return tuple(out)


def _store(name: str) -> Dict[tuple, object]:
    return st.session_state.setdefault(f"_rates_risk_{name}", {})


def cached(name: str, key: tuple, compute: Callable[[], object]) -> object:
    """Keep the last few results of a slow computation across reruns (per
    session: results hold curves of this session's market)."""

    entries = _store(name)
    if key not in entries:
        if len(entries) >= CACHE_ENTRIES:
            entries.pop(next(iter(entries)))
        entries[key] = compute()
    return entries[key]


def bumps(market: Market, key: str, parallel: bool = True) -> CurveBumps:
    """Markets with each quote of curve ``key`` moved by +/-1bp, built once
    per market and shared by the tabs (projection curves build lazily)."""

    return cached("bumps", (market_key(market), key, parallel), lambda: curve_bumps(market, key, parallel=parallel))


def ladders(
    market: Market, pv: Callable[[Market], float], trade_key: tuple, curves: Optional[Sequence[str]] = None
) -> Dict[str, Tuple[List[Dict[str, object]], np.ndarray]]:
    """Key-rate PV01 ladder of one trade on each curve: {key: (rows,
    values)}; ``trade_key`` identifies the trade for the cache."""

    curves = tuple(curves) if curves is not None else risk_keys(market)

    def compute() -> Dict[str, Tuple[List[Dict[str, object]], np.ndarray]]:
        out = {}
        for key in curves:
            rows, matrix = key_rate_matrix([pv], market, key, bumps=bumps(market, key))
            out[key] = (rows, matrix[:, 0])
        return out

    return cached("ladders", (market_key(market), trade_key, curves), compute)


def rate_pv01(market: Market, pv: Callable[[Market], float], trade_key: tuple) -> float:
    """P&L for +1bp on every rate quote of every risk curve (discount and
    projections rebuilt; tenor basis spreads held), central difference."""

    def moved(bp: float) -> Market:
        bumps = {}
        for key in risk_keys(market):
            curve = market.get_curve(key)
            bumps[key] = [0.0 if inst.kind in SPREAD_KINDS else bp for inst in curve.instruments]
        return market.rebuilt(bumps)

    return cached("rate_pv01", (market_key(market), trade_key), lambda: (pv(moved(1.0)) - pv(moved(-1.0))) / 2.0)


def projection_chain(market: Market, index: Optional[str]) -> Tuple[str, ...]:
    """The discount curve, ``index`` and the projection curves it is built
    on (a 3M curve from 3s6s basis rides on the 6M curve)."""

    out = [DISCOUNT_KEY]
    while index:
        out.append(index)
        index = getattr(market.get_curve(index), "basis_index", None)
    return tuple(out)


def ladder_chart(
    labels: Sequence[str],
    values: Sequence[float],
    residual: Optional[Sequence[float]] = None,
    name: str = "TRADE",
    unit: str = "$",
) -> go.Figure:
    """Bars of a key-rate ladder (P&L per +1bp, in currency ``unit``),
    optionally next to the residual after a hedge; both axes bounded.
    Buckets below half a cent are rounding noise of the rebuilds and are
    drawn as zero."""

    clean = lambda vs: [0.0 if abs(float(v)) < NOISE else float(v) for v in vs]  # noqa: E731
    labels, values = list(labels), clean(values)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            name=name,
            marker=dict(color=[SERIES[0] if v >= 0 else ACCENT for v in values] if residual is None else SERIES[0]),
        )
    )
    shown = [-1.0, 1.0, *values]  # at least +/-$1, so an all-zero ladder reads as zero, not as nano-dollars
    if residual is not None:
        res = clean(residual)
        fig.add_trace(go.Bar(x=labels, y=res, name="AFTER HEDGE", marker=dict(color=ACCENT)))
        shown += res
        fig.update_layout(barmode="group")
    dark_layout(fig, height=320, xtitle="CURVE QUOTE", ytitle=f"PV01 ({unit} PER +1BP)", legend=residual is not None)
    return bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(labels) - 0.5, "min_span": 1.0},
        yaxis=axis_limits(shown, pad=0.12),
    )


def loads_on(values: Sequence[float], tolerance: float = 0.005) -> bool:
    """True if a ladder has risk worth showing (any bucket above half a cent)."""

    return bool(np.any(np.abs(np.asarray(values, dtype=float)) > tolerance))
