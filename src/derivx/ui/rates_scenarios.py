"""SCENARIOS tab of the Rates page: shock curve quotes and vols, roll the
market forward, explain each position's P&L and run the standard scenario
library over the portfolio. Historical shocks (named episodes or any window
of the history store) move each curve by its own history where a free one
exists and say per curve which data they used.

On a multi-curve market the shock moves the discount and projection curves
separately (MOVE PROJECTION CURVES), the IBOR-OIS and tenor basis shocks
move only the EURIBOR curves, and the explain splits delta by curve."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
from dataclasses import replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.data import rates_history as rh
from derivx.rates.curves import Curve, Market
from derivx.risk.scenarios import (
    EPISODES,
    EXPLAIN_COLUMNS,
    LOGNORMAL_VOL_UNIT,
    NORMAL_VOL_UNIT,
    TOTAL,
    QuoteRisk,
    Shock,
    butterfly,
    carry_market,
    historical,
    historical_episodes,
    historical_sources,
    history_window,
    horizon_date,
    ibor_ois_basis,
    key_rates,
    parallel,
    quote_risk,
    run_scenario,
    scenario_grid,
    shocked_curves,
    standard_library,
    tenor_basis,
    twist,
)
from derivx.risk.sensitivities import Position
from derivx.ui import rates_data, rates_risk
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SERIES, dark_layout

SHOCK_TYPES = ("Parallel", "Twist", "Butterfly", "Key rates", "Historical", "History window")
BASIS_TYPES = ("IBOR-OIS basis", "Tenor basis")  # multi-curve markets only
HISTORY_UPLOAD_KEY = "rates_user_history"
FLAG_LABELS = {
    "real": "REAL: the curve's own market",
    "upload": "YOUR UPLOAD",
    "proxy": "PROXY: another market's moves (spreads held)",
    "cross-currency proxy": "CROSS-CURRENCY PROXY",
}
KEY_TENORS = ("3M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y")
BP_LIMIT = 500.0
NORMAL_VOL_LIMIT = 200.0  # bp of normal vol
LOGNORMAL_VOL_LIMIT = 50.0  # lognormal vol points
MAX_HORIZON_DAYS = 730
CACHE_ENTRIES = 4
CURVE_LABELS = {"curve": "SWAP CURVE", "govt": "TREASURY CURVE"}
EXPLAIN_LABELS = {
    "delta": "DELTA",
    "gamma": "GAMMA",
    "cross_gamma": "CROSS GAMMA",
    "vol": "VOL",
    "carry": "CARRY",
    "residual": "RESIDUAL",
}
MONEY = st.column_config.NumberColumn(format="%,.2f")


def _table_labels(c: str = "$") -> Dict[str, str]:
    """Explain table columns, amounts in currency ``c``."""

    return {
        "position": "POSITION",
        "kind": "KIND",
        "base_pv": f"BASE PV ({c})",
        "shocked_pv": f"HORIZON PV ({c})",
        "cash": f"CASH ({c})",
        "pnl": f"TOTAL P&L ({c})",
        **{k: f"{v} ({c})" for k, v in EXPLAIN_LABELS.items()},
    }


def _book_currency(positions: Sequence[Position], market: Market) -> Tuple[str, bool]:
    """(currency label of the book's values, whether it mixes currencies):
    Treasury bonds are in dollars, swaps and caps in the swap curve's
    currency. Values are added without FX conversion."""

    symbols = sorted({rates_risk.money(market, "govt" if p.kind == "bond" else "curve") for p in positions})
    return (symbols[0], False) if len(symbols) == 1 else (" + ".join(symbols), True)


def _cached(name: str, key: tuple, compute: Callable[[], object]) -> object:
    """Keep the last few results of a slow computation across reruns."""

    entries = st.session_state.setdefault(f"scn_cache_{name}", {})
    if key not in entries:
        if len(entries) >= CACHE_ENTRIES:
            entries.pop(next(iter(entries)))
        entries[key] = compute()
    return entries[key]


def _market_key(market: Market) -> tuple:
    return rates_risk.market_key(market)


def _curve_label(market: Market, which: str) -> str:
    return CURVE_LABELS.get(which, which) if not market.is_multi_curve else rates_risk.curve_title(market, which)


def _positions_key(positions: Sequence[Position], market: Market) -> tuple:
    """Identifies the portfolio; today's PVs catch trades whose terms are not
    in their name or details. Raises ValueError if a PV is not finite."""

    pvs = [float(p.pv(market)) for p in positions]
    bad = [p.name for p, pv in zip(positions, pvs) if not math.isfinite(pv)]
    if bad:
        raise ValueError(f"no finite value for {', '.join(bad)}")
    return tuple((p.name, p.kind, p.vol_model, repr(p.details), pv) for p, pv in zip(positions, pvs))


def _has_quotes(curve: Optional[Curve]) -> bool:
    return curve is not None and bool(curve.quotes)


def _num(value: float, decimals: int = 2) -> str:
    """Thousands-separated number without a "-0.00" for tiny negatives."""

    return f"{round(value, decimals) + 0.0:,.{decimals}f}"


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _bp(container, label: str, value: float, key: str) -> float:
    return container.number_input(
        label, min_value=-BP_LIMIT, max_value=BP_LIMIT, value=value, step=1.0, format="%.2f", key=key
    )


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _profile_inputs(kind: str, c2, c3, store: rh.HistoryStore, market: Market) -> Shock:
    """The quote-move part of the shock; raises ValueError on bad inputs."""

    if kind == "IBOR-OIS basis":
        indices = [k for k in market.projections if not market.projections.is_discount(k)]
        picked = c3.multiselect("PROJECTION CURVES", indices, default=indices, key="scn_basis_indices")
        if not picked:
            raise ValueError("pick at least one projection curve")
        bp = _bp(c2, "BASIS MOVE (BP)", 10.0, "scn_basis_bp")
        return ibor_ois_basis(bp, None if len(picked) == len(indices) else picked)
    if kind == "Tenor basis":
        return tenor_basis(_bp(c2, "BASIS SPREAD MOVE (BP)", 5.0, "scn_tenor_bp"))
    if kind == "Parallel":
        return parallel(_bp(c2, "SHIFT (BP)", 25.0, "scn_par_bp"))
    if kind == "Twist":
        pivot = c2.number_input("PIVOT (YRS)", min_value=0.25, max_value=30.0, value=5.0, step=0.25, key="scn_tw_pivot")
        a, b, c, d = st.columns(4)
        short_bp = _bp(a, "SHORT END (BP)", -25.0, "scn_tw_short")
        long_bp = _bp(b, "LONG END (BP)", 25.0, "scn_tw_long")
        short_years = c.number_input(
            "SHORT END AT (YRS, 0 = SHORTEST QUOTE)",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=0.25,
            key="scn_tw_short_years",
        )
        long_years = d.number_input(
            "LONG END AT (YRS, 0 = LONGEST QUOTE)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            key="scn_tw_long_years",
        )
        return twist(pivot, short_bp, long_bp, short_years or None, long_years or None)
    if kind == "Butterfly":
        body_years = c2.number_input(
            "BODY (YRS)", min_value=0.25, max_value=30.0, value=5.0, step=0.25, key="scn_fly_body_years"
        )
        a, b, c, d = st.columns(4)
        body_bp = _bp(a, "BODY (BP)", 25.0, "scn_fly_body")
        wing_bp = _bp(b, "WINGS (BP)", 0.0, "scn_fly_wing")
        left = c.number_input(
            "SHORT WING (YRS)", min_value=0.0, max_value=30.0, value=2.0, step=0.25, key="scn_fly_left"
        )
        right = d.number_input(
            "LONG WING (YRS)", min_value=0.5, max_value=50.0, value=10.0, step=1.0, key="scn_fly_right"
        )
        return butterfly(wing_bp, body_bp, body_years, wings=(left, right))
    if kind == "Key rates":
        cols = st.columns(len(KEY_TENORS))
        moves = {t: _bp(col, f"{t} (BP)", 0.0, f"scn_kr_{t}") for t, col in zip(KEY_TENORS, cols)}
        return key_rates(moves)
    if kind == "Historical":
        name = c2.selectbox("EPISODE", historical_episodes(), key="scn_hist")
        scale = c3.number_input("SCALE (x)", min_value=-3.0, max_value=3.0, value=1.0, step=0.25, key="scn_hist_scale")
        return historical(name, scale, store=store)
    dates = store.dates("UST_PAR")[::-1]
    end_default = 0
    start_default = next((i for i, d in enumerate(dates) if (dates[0] - d).days >= 28), len(dates) - 1)
    start = c2.selectbox("WINDOW START", dates, index=start_default, key="scn_win_start")
    end = c3.selectbox("WINDOW END", dates, index=end_default, key="scn_win_end")
    scale = st.number_input("SCALE (x)", min_value=-3.0, max_value=3.0, value=1.0, step=0.25, key="scn_win_scale")
    return history_window(start, end, scale, store=store)


def _history_store(market: Market) -> rh.HistoryStore:
    """The CURVE tab's store plus the user's uploaded swap history."""

    base = rates_data.active_store()
    uploads = st.session_state.get(HISTORY_UPLOAD_KEY, {})
    if not uploads:
        return base
    token = ",".join(sorted(f"{ccy}:{u['sha'][:10]}" for ccy, u in uploads.items()))
    memo = st.session_state.setdefault("_scn_store_memo", {})
    key = (id(base), token)
    if key not in memo:
        memo.clear()
        memo[key] = rh.build_store([base.frame] + [u["frame"] for u in uploads.values()], f"{base.label} + {token}")
    return memo[key]


def _history_upload(market: Market) -> None:
    """Expander to upload the user's own swap-curve history (kept in session
    state per currency until cleared)."""

    currency = market.curve.conventions.currency if market.curve.conventions else "USD"
    uploads = st.session_state.setdefault(HISTORY_UPLOAD_KEY, {})
    with st.expander(f"Your {currency} swap-curve history (CSV upload, used by historical shocks)"):
        st.caption(
            "Columns date, tenor, rate (long) or date plus one column per tenor (wide: 2Y, 5Y, 10Y ...); par swap "
            "rates. Where it covers a window it replaces the proxy for the swap curve."
        )
        c1, c2 = st.columns([3, 1])
        file = c1.file_uploader("SWAP HISTORY CSV", type=["csv"], key=rates_data.widget_key("scn_hist_upload"))
        units = c2.radio("UNITS", ["auto", "percent", "decimal", "bp"], key="scn_hist_units")
        if file is not None:
            payload = file.getvalue()
            sha = hashlib.sha256(payload + units.encode()).hexdigest()
            if uploads.get(currency, {}).get("sha") != sha:
                try:
                    raw = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")  # Excel adds a BOM
                except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
                    st.error(f"{file.name}: could not read the file ({exc})")
                    raw = None
                if raw is not None:
                    frame, issues, used = rh.parse_history_upload(raw, currency, units, file.name)
                    for issue in issues:
                        (st.error if issue.level == "error" else st.caption)(f"{file.name}: {issue.message}")
                    if len(frame):
                        uploads[currency] = {"name": file.name, "sha": sha, "frame": frame}
        mine = uploads.get(currency)
        if mine is not None:
            days = sorted(set(mine["frame"]["asof"]))
            st.caption(
                f"Using {mine['name']}: {len(days)} dates from {days[0]} to {days[-1]}, "
                f"{mine['frame']['tenor'].nunique()} tenors."
            )
            st.button(
                "CLEAR SWAP HISTORY",
                key="scn_hist_clear",
                on_click=rates_data.clear_upload,
                args=(HISTORY_UPLOAD_KEY, "scn_hist_upload", currency),
            )


def _sources_table(shock: Shock, market: Market) -> None:
    """Which data each moved curve of a historical shock replays."""

    table = historical_sources(shock)
    if table.empty:
        return
    currency = market.curve.conventions.currency if market.curve.conventions else None
    shown = table[(table["curve"] == "TREASURY") | (table["currency"] == currency)]
    shown = shown.assign(flag=shown["flag"].map(FLAG_LABELS).fillna(shown["flag"]))
    shown.columns = [c.upper() for c in shown.columns]
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(
        "Each curve's quotes move by its data's change at the same tenor (linear in between, flat beyond). REAL "
        "swap/OIS history: ICE USD swap rates (FRED, to 2016-10-28), the Bank of England OIS curve (GBP, par OIS "
        "rates derived from its spot curve) or your upload. Otherwise a flagged proxy moves the swap curve by "
        "government yield changes, which holds swap spreads unchanged."
    )


def _shock_inputs(market: Market, store: rh.HistoryStore) -> Optional[Shock]:
    """Read the shock builder; None (with the reason shown) if invalid."""

    c1, c2, c3 = st.columns(3)
    multi = market.is_multi_curve
    kind = c1.selectbox("SHOCK TYPE", SHOCK_TYPES + (BASIS_TYPES if multi else ()), key="scn_type")
    try:
        shock = _profile_inputs(kind, c2, c3, store, market)
    except ValueError as exc:
        st.error(f"Cannot build this shock: {exc}")
        return None
    has_govt = _has_quotes(market.govt)
    basis = kind in BASIS_TYPES
    if multi:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        move_proj = c6.checkbox(
            "MOVE PROJECTION CURVES", value=True, key="scn_move_proj", disabled=basis,
            help="Move the EURIBOR curves' rate quotes with the swap curve (basis spreads held). Untick to move the "
            "OIS discount curve alone: the projection curves are then rebuilt on it with their quotes held.",
        )  # fmt: skip
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        move_proj = None
    move_curve = c1.checkbox("MOVE SWAP CURVE", value=True, key="scn_move_curve", disabled=basis)
    move_govt = c2.checkbox("MOVE TREASURY CURVE", value=has_govt, key="scn_move_govt", disabled=not has_govt or basis)
    normal = c3.number_input(
        "NORMAL VOL SHOCK (BP)",
        min_value=-NORMAL_VOL_LIMIT,
        max_value=NORMAL_VOL_LIMIT,
        value=0.0,
        step=1.0,
        key="scn_vol_normal",
        help="Added to every normal (Bachelier) vol, in bp.",
    )
    lognormal = c4.number_input(
        "LOGNORMAL VOL SHOCK (VOL PTS)",
        min_value=-LOGNORMAL_VOL_LIMIT,
        max_value=LOGNORMAL_VOL_LIMIT,
        value=0.0,
        step=0.5,
        key="scn_vol_lognormal",
        help="Added to every Black / shifted Black vol, in vol points (1 = 1%).",
    )
    horizon = c5.number_input(
        "HORIZON (DAYS)",
        min_value=0,
        max_value=MAX_HORIZON_DAYS,
        value=0,
        step=1,
        key="scn_horizon",
        help="Calendar days to roll the market forward; the shock hits at the horizon.",
    )
    curves = {} if basis else dict(curve=move_curve, govt=move_govt and has_govt, projections=move_proj)
    shock = replace(
        shock,
        **curves,
        normal_vol_bp=float(normal),
        lognormal_vol_pts=float(lognormal),
        horizon_days=int(horizon),
    )
    st.caption(f"{shock.name}: {shock.description}.")
    if kind in ("Historical", "History window"):
        _sources_table(shock, market)
    if not (shock.moves_rates or normal or lognormal or horizon):
        st.warning("This scenario moves nothing: pick a curve, a non-zero shock, a vol shock or a horizon.")
    return shock


def _lowest_vol(vol: object) -> Optional[float]:
    """Lowest of a flat vol, an {expiry: vol} term structure or stripped
    caplet vols."""

    if hasattr(vol, "lowest"):
        return float(vol.lowest)
    values = list(vol.values()) if isinstance(vol, Mapping) else [vol]
    try:
        return min(float(v) for v in values)
    except (TypeError, ValueError):
        return None


def _floored_vols(positions: Sequence[Position], market: Market, shock: Shock) -> List[str]:
    """Positions whose lowest quoted vol the vol shock takes below zero."""

    moves = {
        "normal": market.vol_shift_normal + shock.normal_vol_bp * NORMAL_VOL_UNIT,
        "lognormal": market.vol_shift_lognormal + shock.lognormal_vol_pts * LOGNORMAL_VOL_UNIT,
    }
    out = []
    for p in positions:
        lowest = _lowest_vol(p.details.get("vol")) if p.vol_model in moves else None
        if lowest is not None and lowest + moves[p.vol_model] < 0.0:
            out.append(p.name)
    return out


# --------------------------------------------------------------------------
# Charts and tables
# --------------------------------------------------------------------------


def _curve_chart(
    base: Curve, shocked: Curve, view: str, bumps: np.ndarray, source: Optional[Tuple[str, dt.date]] = None
) -> go.Figure:
    years = base.years
    if view == "PAR QUOTES":
        x = years
        y_base = np.array([q.rate for q in base.quotes]) * 100.0
        y_shocked = np.array([q.rate for q in shocked.quotes]) * 100.0
        mode = "lines+markers"
    else:
        x = np.unique(np.concatenate([np.linspace(0.05, 2.0, 40), np.linspace(2.0, max(years.max(), 2.0), 80)]))
        y_base = np.array([base.zero_rate(t) for t in x]) * 100.0
        y_shocked = np.array([shocked.zero_rate(t) for t in x]) * 100.0
        mode = "lines"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_base, mode=mode, name="BASE", line=dict(color=SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=x, y=y_shocked, mode=mode, name="SHOCKED", line=dict(color=ACCENT, width=2)))
    fig.add_trace(
        go.Scatter(
            x=years,
            y=bumps,
            mode="lines+markers",
            name="SHOCK (BP, RIGHT AXIS)",
            yaxis="y2",
            line=dict(color=SERIES[2], width=1, dash="dot"),
            marker=dict(size=6),
        )
    )
    dark_layout(fig, height=380, xtitle="TENOR (YEARS)", ytitle="RATE (%)")
    fig.update_layout(
        yaxis2=dict(title_text="SHOCK (BP)", overlaying="y", side="right", showgrid=False, zeroline=False)
    )
    label, asof = source if source is not None else ("curve as passed to this tab", base.asof)
    rates_data.badge(fig, label, asof)
    return bound_axes(
        fig,
        xaxis=axis_limits(np.concatenate([x, years]), pad=0.02, floor=0.0),
        yaxis=axis_limits(np.concatenate([y_base, y_shocked]), pad=0.15),
        yaxis2=axis_limits(np.concatenate([bumps, [-1.0, 1.0]]), pad=0.15),  # at least +/-1bp when flat
    )


def _waterfall(total: Dict[str, float], c: str = "$") -> go.Figure:
    labels = [EXPLAIN_LABELS[c] for c in EXPLAIN_COLUMNS] + ["TOTAL P&L"]
    values = [total[c] for c in EXPLAIN_COLUMNS] + [total["pnl"]]
    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=["relative"] * len(EXPLAIN_COLUMNS) + ["total"],
            increasing=dict(marker=dict(color=SERIES[4])),
            decreasing=dict(marker=dict(color=ACCENT)),
            totals=dict(marker=dict(color=SERIES[2])),
            connector=dict(line=dict(color=SERIES[0], width=1, dash="dot")),
            hovertemplate="%{x}: %{y:,.2f}<extra></extra>",
        )
    )
    dark_layout(fig, height=360, xtitle="P&L COMPONENT", ytitle=f"P&L ({c})", legend=False)
    levels = np.concatenate([np.cumsum([0.0] + values[:-1]), [values[-1], -1.0, 1.0]])  # at least +/-$1
    return bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(labels) - 0.5, "min_span": 1.0},
        yaxis=axis_limits(levels, pad=0.1),
    )


def _explain_table(rows: List[Dict[str, object]], c: str = "$") -> pd.DataFrame:
    frame = pd.DataFrame(rows).drop(columns=["delta_by_curve"], errors="ignore")
    total = {"position": TOTAL, "kind": "", **frame.drop(columns=["position", "kind"]).sum().to_dict()}
    frame = pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
    labels = _table_labels(c)
    return frame[list(labels)].rename(columns=labels)


def _grid(positions: Sequence[Position], market: Market, shocks: Sequence[Shock]) -> Tuple[pd.DataFrame, List[str]]:
    """Scenario grid rows for the shocks that revalue, and the reasons the
    others do not (e.g. a Black cap pushed to negative rates)."""

    tables, failed = [], []
    for shock in shocks:
        try:
            tables.append(scenario_grid(positions, market, [shock]))
        except ValueError as exc:
            failed.append(f"{shock.name}: {exc}")
    return (pd.concat(tables) if tables else pd.DataFrame()), failed


# --------------------------------------------------------------------------
# Tab
# --------------------------------------------------------------------------


def _risk(positions: Sequence[Position], market: Market, shock: Shock, key: tuple) -> Dict[str, QuoteRisk]:
    """Full key-rate risk of each moved curve on the market the shock hits
    (rolled to the horizon), kept across changes to the shock profile."""

    def compute(which: str) -> QuoteRisk:
        hit = carry_market(market, shock.horizon_days) if shock.horizon_days else market
        return quote_risk(positions, hit, which)

    out = {}
    for which in shocked_curves(market):
        if shock.applies_to(which) and _has_quotes(market.get_curve(which)):
            out[which] = _cached(f"risk_{which}", (key, shock.horizon_days), lambda w=which: compute(w))
    return out


def _show_curves(market: Market, shock: Shock) -> None:
    st.markdown("#### / BASE VS SHOCKED CURVE")
    available = [w for w in shocked_curves(market) if _has_quotes(market.get_curve(w))]
    c1, c2 = st.columns(2)
    which = c1.radio("CURVE", available, format_func=lambda w: _curve_label(market, w), horizontal=True,
                     key="scn_chart_curve")  # fmt: skip
    view = c2.radio("SHOW", ["PAR QUOTES", "ZERO RATES"], horizontal=True, key="scn_chart_view")
    base = market.get_curve(which)
    bumps = shock.quote_bumps(base, which) if shock.applies_to(which) else np.zeros(len(base.quotes))
    moves = {w: list(shock.quote_bumps(market.get_curve(w), w)) for w in available if shock.applies_to(w)}
    moves = {w: b for w, b in moves.items() if np.any(b)}
    shocked = market.rebuilt(moves).get_curve(which) if moves else base
    badge = rates_data.badge_for(which)
    if badge is not None and badge[1] != base.asof:
        badge = None
    st.plotly_chart(
        _curve_chart(base, shocked, view, bumps, badge),
        width="stretch",
        config=PLOTLY_2D_CONFIG,
        key="scn_curve_chart",
    )
    st.caption(
        "Each quote moves by the shock profile at its tenor (linear between profile points, flat beyond) and the "
        "curve is rebuilt, so the shocked curve reprices the shocked quotes. Shown on today's date; zero rates are "
        "continuously compounded ACT/365F."
        + (" A projection curve is rebuilt on the shocked OIS curve too, so its zero rates move with an OIS shock "
           "even where its own quotes do not; BASIS quotes are spreads (bp), shown in %." if market.is_multi_curve
           else "")
    )  # fmt: skip


def _show_explain(
    rows: List[Dict[str, object]], market: Market, shock: Shock, positions: Sequence[Position] = ()
) -> None:
    c, mixed = _book_currency(positions, market) if positions else (rates_risk.money(market), False)
    if mixed:
        st.warning(
            f"The book holds {market.curve.conventions.currency} swaps or caps and US Treasury bonds: totals add "
            f"amounts in {c} without FX conversion. Read the P&L per position, or keep one currency in the book."
        )
    table = _explain_table(rows, c)
    total = {k: float(sum(r[k] for r in rows)) for k in ("pnl", *EXPLAIN_COLUMNS)}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"TOTAL P&L ({c})", _num(total["pnl"]))
    m2.metric(f"EXPLAINED ({c})", _num(total["pnl"] - total["residual"]))
    m3.metric(f"RESIDUAL ({c})", _num(total["residual"]))
    share = abs(total["residual"]) / abs(total["pnl"]) * 100.0 if abs(total["pnl"]) > 1e-9 else None
    m4.metric("RESIDUAL (% OF P&L)", "n/a" if share is None else f"{share:.3f}")
    st.plotly_chart(_waterfall(total, c), width="stretch", config=PLOTLY_2D_CONFIG, key="scn_waterfall")
    horizon = horizon_date(market, shock.horizon_days)
    st.caption(
        f"Horizon {horizon} ({shock.horizon_days} days). P&L = value at the horizon + cash received - value today, "
        "signed to you. Carry = value with today's quotes rebuilt at the horizon + cash - value today (fixings and "
        "cash before the horizon follow today's curve). The shock then hits the rolled market and is explained by "
        "its Greeks: delta = key-rate PV01 x quote move; gamma = 1/2 diagonal key-rate gamma x move^2; cross gamma "
        "= the rest of the second-order term along the shock; vol = vega x vol shock."
    )
    money = {label: MONEY for label in list(_table_labels(c).values())[2:]}
    st.dataframe(table, hide_index=True, width="stretch", column_config=money)
    st.download_button("Download P&L explain CSV", _csv(table), file_name="scenario_explain.csv", key="scn_explain_csv")
    curves = sorted({k for r in rows for k in r.get("delta_by_curve", {})}, key=list(market.curve_keys).index)
    if market.is_multi_curve and curves:
        st.markdown("#### / DELTA BY CURVE")
        by_curve = pd.DataFrame(
            [{"POSITION": r["position"], **{f"{_curve_label(market, k)} ({rates_risk.money(market, k)})":
                                           r["delta_by_curve"].get(k, 0.0) for k in curves}} for r in rows]
        )  # fmt: skip
        st.dataframe(by_curve, hide_index=True, width="stretch",
                     column_config={c: MONEY for c in by_curve.columns[1:]})  # fmt: skip
        st.caption("The delta column split by the curve whose quotes moved (key-rate PV01 x quote move, per curve).")


def _episode_flags(store: rh.HistoryStore, market: Market) -> str:
    """One line per episode: the data its swap-curve move comes from."""

    currency = market.curve.conventions.currency if market.curve.conventions else "USD"
    parts = []
    for name, _, _ in EPISODES:
        rows = historical_sources(historical(name, store=store))
        row = rows[(rows["curve"] == "SWAP") & (rows["currency"] == currency)]
        if not row.empty:
            r = row.iloc[0]
            parts.append(f"{name} ({r['start']} to {r['end']}): {r['data']} [{FLAG_LABELS.get(r['flag'], r['flag'])}]")
    return "; ".join(parts)


def _show_grid(
    positions: Sequence[Position], market: Market, shock: Optional[Shock], key: tuple, store: rh.HistoryStore
) -> None:
    """Standard library grid, headed by the custom ``shock`` when given."""

    st.markdown("### / SCENARIO GRID")
    library = standard_library(store)
    if market.is_multi_curve:
        library += [ibor_ois_basis(10.0), tenor_basis(5.0)]
    key = (key, tuple(library))  # the historical shocks depend on the store
    if not _has_quotes(market.govt):
        library = [replace(s, govt=False) for s in library]
    grid, failed = _cached("grid", key, lambda: _grid(positions, market, library))
    if shock is not None:
        custom, _ = _cached("custom", (key, shock), lambda: _grid(positions, market, [shock]))
        grid = pd.concat([custom.rename(index=lambda name: f"CUSTOM: {name}"), grid])
    if failed:
        st.warning("Not revalued: " + "; ".join(failed))
    if grid.empty:
        return
    table = grid.rename_axis("SCENARIO").reset_index()
    st.dataframe(table, hide_index=True, width="stretch", column_config={c: MONEY for c in table.columns[1:]})
    st.download_button("Download scenario grid CSV", _csv(table), file_name="scenario_grid.csv", key="scn_grid_csv")
    st.caption(
        "Full-revaluation P&L in $ per position. The library is instantaneous and moves both curves: parallel "
        "+/-25 and +/-100bp; 2s10s steepener/flattener (2Y -/+12.5bp, 10Y +/-12.5bp, pivot 5Y, flat beyond); "
        "2s5s10s fly (5Y +25bp, 2Y and 10Y unchanged); historical episodes, the Treasury curve moving by US "
        "Treasury par yield changes and the swap curve by: " + _episode_flags(store, market) + "."
        + (" Multi-curve: the projection curves move with the swap curve in every rate scenario (basis held; no "
           "free EURIBOR history); IBOR-OIS basis +10bp moves only the EURIBOR rate quotes, tenor basis +5bp only "
           "the 3s6s spreads." if market.is_multi_curve else "")
    )  # fmt: skip


def render(market: Market, positions: List[Position]) -> None:
    """Scenario tab: shock builder, P&L explain and scenario grid."""

    st.markdown("### / SCENARIOS")
    if not market.curve.quotes:
        st.error("Scenarios shock curve quotes: the swap curve must be bootstrapped from quotes.")
        return
    _history_upload(market)
    store = _history_store(market)
    shock = _shock_inputs(market, store)
    if shock is None:
        return
    try:
        _show_curves(market, shock)
    except ValueError as exc:
        st.error(f"Cannot apply this shock: {exc}")
        return

    st.markdown("### / P&L EXPLAIN")
    if not positions:
        st.info("No positions yet: price a swap, cap/floor or bond on the other Rates tabs to explain its P&L here.")
        return
    floored = _floored_vols(positions, market, shock)
    if floored:
        st.warning(
            f"The vol shock takes the vol of {', '.join(floored)} below zero. The pricers floor vols at zero, so the "
            "P&L stops at the zero-vol value and the VOL column (vega x shock) overstates it."
        )
    try:
        key = (_market_key(market), _positions_key(positions, market))
    except ValueError as exc:
        st.error(f"Cannot value the portfolio today: {exc}")
        return
    try:
        risk = _risk(positions, market, shock, key)
        rows = _cached("explain", (key, shock), lambda: run_scenario(positions, market, shock, risk))
    except ValueError as exc:
        st.error(f"Cannot revalue the portfolio under this scenario: {exc}")
        rows = None
    if rows is not None:
        _show_explain(rows, market, shock, positions)
    _show_grid(positions, market, shock if rows is not None else None, key, store)
