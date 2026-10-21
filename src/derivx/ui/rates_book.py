"""BOOK tab of the Rates page: a trade blotter priced on the page's curves
and vols, with the book's risk, scenario P&L, VaR and suggested hedges.

The blotter (add, edit and delete rows; upload or download CSV) starts from
an EXAMPLE book of illustrative trades for the swap curve's currency.
Each row is priced by its single-trade pricer (:mod:`derivx.rates.portfolio`),
swaptions without a flat vol on the SWAPTION tab's SABR cube and caps and
floors without one on caplet vols stripped from the CAP-FLOOR tab's cap
quotes (ILLUSTRATIVE unless you entered your own). The book's positions are
what SCENARIOS and RESEARCH run on by default (RISK SCOPE in the sidebar):
one set of positions, the trades priced on the other tabs still available.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.rates import portfolio as pf
from derivx.rates.curves import GOVT_KEY, Market
from derivx.rates.exports import book_cashflows, market_tables, risk_report
from derivx.rates.indices import HullWhite
from derivx.risk import var as var_mod
from derivx.risk.sensitivities import Position
from derivx.ui import (
    rates_capfloor,
    rates_data,
    rates_exports,
    rates_research,
    rates_risk,
    rates_scenarios,
    rates_swaption,
)
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.state import data_editor as persistent_editor  # edits survive mode switches (UIE-04)
from derivx.ui.theme import ACCENT, CELL_TEXT, DIVERGING_COLORS, PLOTLY_2D_CONFIG, SERIES, dark_layout

STATE_KEY = "book_state"
CURRENT_KEY = "book_current"
UPLOAD_SHA_KEY = "book_upload_sha"
SCOPES = ("BOOK", "TAB TRADES", "BOOK + TAB TRADES")
SCOPE_KEY = "rates_risk_scope"
MONEY = st.column_config.NumberColumn(format="%,.2f")
VAR_LOOKBACK = 500
VAR_CONFIDENCE = 0.99
HW_DEFAULTS = (3.0, 85.0)  # the CURVE tab's defaults: mean reversion %, normal short-rate vol bp
PATTERNS = ("", "/", "x", ".", "+", "-")
SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}


# --------------------------------------------------------------------------
# Blotter state
# --------------------------------------------------------------------------


def _cell(value: object) -> str:
    """A blotter cell as text (the editor's columns are text: ``10mm`` and
    ``DEC26`` stay as typed), blanks empty."""

    if pf.is_blank(value):
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value).strip()


def _text_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({c: frame[c] if c in frame.columns else None for c in pf.COLUMNS})
    return table.map(_cell).astype(object)


def _example(currency: str) -> Dict[str, object]:
    table = _text_table(pf.example_blotter(currency))
    return {"table": table, "loaded": table, "source": f"EXAMPLE {currency} book (illustrative trades, not anyone's "
            "positions)", "example": currency, "gen": 0}  # fmt: skip


def _state(currency: str) -> Dict[str, object]:
    state = st.session_state.get(STATE_KEY)
    if state is None:
        state = _example(currency)
        st.session_state[STATE_KEY] = state
    elif state.get("example") and state["example"] != currency and not _edited(state):
        # an untouched example book follows the swap curve's currency
        gen = int(state["gen"]) + 1
        state.clear()
        state.update(_example(currency), gen=gen)
        st.session_state.pop(CURRENT_KEY, None)
    return state


def _loaded(state: Mapping[str, object]) -> pd.DataFrame:
    """The table as loaded (example, upload or empty), before any edit."""

    return state.get("loaded", state["table"])


def _edited(state: Mapping[str, object]) -> bool:
    current = st.session_state.get(CURRENT_KEY)
    return current is not None and not _same(current, _loaded(state))


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return _text_table(a).fillna("").equals(_text_table(b).fillna(""))


def _replace(table: pd.DataFrame, source: str, example: Optional[str] = None) -> None:
    state = st.session_state[STATE_KEY]
    table = _text_table(table)
    state.update(table=table, loaded=table, source=source, example=example, gen=int(state["gen"]) + 1)
    st.session_state.pop(CURRENT_KEY, None)


def _load_example(currency: str) -> None:
    _replace(pf.example_blotter(currency), _example(currency)["source"], currency)


def _clear() -> None:
    _replace(pd.DataFrame(columns=list(pf.COLUMNS)), "empty book (add rows)")


def _column_config(currency: str) -> Dict[str, object]:
    text = {c: st.column_config.TextColumn(c.upper(), help=pf.COLUMN_HELP[c]) for c in pf.COLUMNS}
    text["type"] = st.column_config.SelectboxColumn("TYPE", options=list(pf.TRADE_TYPES), help=pf.COLUMN_HELP["type"])
    text["currency"] = st.column_config.SelectboxColumn("CURRENCY", options=list(pf.CURRENCIES), default=currency,
                                                        help=pf.COLUMN_HELP["currency"])  # fmt: skip
    text["direction"] = st.column_config.SelectboxColumn("DIRECTION", options=["receive", "pay", "long", "short"],
                                                         help=pf.COLUMN_HELP["direction"])  # fmt: skip
    text["option"] = st.column_config.SelectboxColumn("OPTION", options=["payer", "receiver", "straddle"],
                                                      help=pf.COLUMN_HELP["option"])  # fmt: skip
    text["model"] = st.column_config.SelectboxColumn("MODEL", options=list(pf.MODEL_NAMES),
                                                     help=pf.COLUMN_HELP["model"])  # fmt: skip
    return text


def _blotter(currency: str) -> Tuple[pd.DataFrame, str]:
    """Draw the blotter editor and its controls; (table, source label)."""

    state = _state(currency)
    key = f"book_editor_{state['gen']}"
    current = st.session_state.get(CURRENT_KEY)
    if key not in st.session_state and current is not None and not _same(current, state["table"]):
        # the editor was not drawn (another page was open): start it again from the last edited table
        state.update(table=_text_table(current), gen=int(state["gen"]) + 1)
        key = f"book_editor_{state['gen']}"
    c1, c2, c3 = st.columns(3)
    c1.button(f"LOAD EXAMPLE {currency} BOOK", key="book_load_example", on_click=_load_example, args=(currency,))
    c2.button("CLEAR THE BOOK", key="book_clear", on_click=_clear)
    file = c3.file_uploader("UPLOAD BLOTTER CSV", type=["csv"], key=rates_data.widget_key("book_upload"))
    if file is None:
        st.session_state.pop(UPLOAD_SHA_KEY, None)  # an emptied uploader takes the same file again
    else:
        payload = file.getvalue()
        sha = hashlib.sha256(payload).hexdigest()
        seen = st.session_state.get(UPLOAD_SHA_KEY)
        if seen is None or seen[0] != sha:
            table, issues = pf.read_blotter_csv(payload)
            failed = [i.message if i.row == 0 else str(i) for i in issues if i.level == "error"]
            st.session_state[UPLOAD_SHA_KEY] = (sha, failed)
            for issue in issues:
                if issue.level != "error":
                    st.warning(f"{file.name}: {issue}")
            if not failed:
                _replace(table, f"uploaded: {file.name}")
                state = st.session_state[STATE_KEY]
                key = f"book_editor_{state['gen']}"
            seen = st.session_state[UPLOAD_SHA_KEY]
        if seen[1]:  # stays up while the file that could not be read is in the uploader
            st.error(f"{file.name} was not loaded: " + "; ".join(seen[1]))
    edited = persistent_editor(_text_table(state["table"]), key=key, num_rows="dynamic", width="stretch",
                            hide_index=True, column_config=_column_config(currency))  # fmt: skip
    st.session_state[CURRENT_KEY] = edited
    source = str(state["source"]) + (" (edited by you)" if not _same(edited, _loaded(state)) else "")
    st.caption(
        f"Blotter: {source}. One trade per row; rates, strikes and shifts in %, spreads in bp, vols in bp (normal) "
        "or % (Black), notionals positive (10mm, 2.5bn and 500k work) with the direction giving the side. Trades "
        "given by tenor start at spot from the valuation date. Blank vols use the market vols (SWAPTION cube, "
        "caplet vols stripped from the CAP-FLOOR quotes)."
    )
    with st.expander("COLUMN GUIDE (which columns each trade type reads)"):
        st.dataframe(pf.column_guide(), hide_index=True, width="stretch")
    st.download_button("DOWNLOAD BLOTTER CSV", _text_table(edited).to_csv(index=False), file_name="blotter.csv",
                       mime="text/csv", key="book_blotter_csv", on_click="ignore")  # fmt: skip
    return edited, source


def _show_issues(issues: Sequence[pf.BlotterIssue]) -> None:
    errors = [str(i) for i in issues if i.level == "error"]
    warnings = [str(i) for i in issues if i.level == "warning"]
    notes = [str(i) for i in issues if i.level == "info"]
    if errors:
        st.error("Rows left out of the book: " + "; ".join(errors))
    if warnings:
        st.warning("; ".join(warnings))
    if notes:
        st.caption("; ".join(notes))


# --------------------------------------------------------------------------
# Market vols and pricing
# --------------------------------------------------------------------------


def _hull_white() -> HullWhite:
    a = st.session_state.get("rates_hw_a", HW_DEFAULTS[0])
    sigma = st.session_state.get("rates_hw_sigma", HW_DEFAULTS[1])
    return HullWhite(float(a) / 100.0, float(sigma) / 1e4)


def _vols(market: Market, trades: Sequence[pf.Trade]) -> Tuple[pf.VolSet, tuple, List[str], Dict[str, object]]:
    """(vols, cache key, notes, calibration tables) for the trades without a
    flat vol, from the SWAPTION and CAP-FLOOR tabs' quotes."""

    cube_keys, cap_keys = pf.needed_vols(trades)
    cubes, caplets, sources = {}, {}, {}
    key: List[object] = []
    notes: List[str] = []
    calibration: Dict[str, pd.DataFrame] = {}
    for under in cube_keys:
        try:
            cube, source, cube_key = rates_swaption.book_cube(market, under)
        except ValueError as exc:
            notes.append(f"no SABR cube for {under} swaptions ({exc}): they are left out unless given a flat vol")
            continue
        cubes[under], sources[f"cube:{under}"] = cube, source
        key.append(("cube", under, cube_key))
        calibration[f"sabr_{under.replace(' ', '_').lower()}"] = cube.fit_table()
    for cap in cap_keys:
        try:
            vols, source, cap_key = rates_capfloor.book_caplets(market, cap)
        except ValueError as exc:
            notes.append(f"no caplet vols for {cap} caps ({exc}): they are left out unless given a flat vol")
            continue
        caplets[cap], sources[f"caplets:{cap}"] = vols, source
        key.append(("caplets", cap, cap_key))
        calibration[f"caplets_{cap.replace(' ', '_').lower()}"] = _caplet_table(vols)
    hw = _hull_white()
    return pf.VolSet(cubes, caplets, sources, hw), (tuple(key), hw), notes, calibration


def _caplet_table(vols: object) -> pd.DataFrame:
    """Stripped caplet vols: one row per node (and strike, for a surface),
    normal vol in bp."""

    curves = list(getattr(vols, "curves", ())) or [vols]
    rows = []
    for curve in curves:
        for node, vol in zip(curve.nodes, curve.vols):
            rows.append({"strike_pct": None if curve.strike is None else curve.strike * 100.0, "node_years": node,
                         "caplet_vol": vol, "units": "normal, rate units" if curve.model == "bachelier" else
                         "lognormal, decimal", "method": curve.method})  # fmt: skip
    return pd.DataFrame(rows)


def _priced(market: Market, trades: Sequence[pf.Trade], vols: pf.VolSet, vols_key: tuple) -> pf.BookRisk:
    key = (rates_risk.market_key(market), tuple(trades), vols_key)
    return rates_risk.cached(
        "book_risk", key, lambda: pf.price_book(trades, market, vols, bumps=lambda m, k: rates_risk.bumps(m, k))
    )


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def _krd_chart(risk: pf.BookRisk, key: str, unit: str, source: Tuple[str, object]) -> go.Figure:
    """Key-rate PV01 of one curve stacked by trade, with the book total."""

    qr = risk.risk[key]
    labels = list(qr.labels)
    fig = go.Figure()
    loaded = [j for j in range(len(risk.trades)) if np.any(np.abs(qr.pv01[:, j]) > rates_risk.NOISE)]
    for n, j in enumerate(loaded):
        values = [0.0 if abs(v) < rates_risk.NOISE else float(v) for v in qr.pv01[:, j]]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                name=risk.trades[j].id,
                marker=dict(
                    color=SERIES[n % len(SERIES)], pattern=dict(shape=PATTERNS[(n // len(SERIES)) % len(PATTERNS)])
                ),
            )
        )
    total = qr.pv01.sum(axis=1)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=total,
            mode="markers",
            name="BOOK",
            marker=dict(color=ACCENT, size=10, symbol="diamond", line=dict(color="#000000", width=1)),
        )
    )
    fig.update_layout(barmode="relative")
    dark_layout(fig, height=460, xtitle="CURVE QUOTE", ytitle=f"PV01 ({unit} PER +1BP)")
    rates_data.badge(fig, *source)
    loaded_values = qr.pv01[:, loaded] if loaded else np.zeros((len(labels), 1))
    highs = np.clip(loaded_values, 0.0, None).sum(axis=1)
    lows = np.clip(loaded_values, None, 0.0).sum(axis=1)
    return bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(labels) - 0.5, "min_span": 1.0},
        yaxis=axis_limits(np.concatenate([highs, lows, total, [-1.0, 1.0]]), pad=0.12),
    )  # fmt: skip


def _vega_heatmap(matrix: pd.DataFrame, unit: str, source: Tuple[str, object]) -> go.Figure:
    """Book vega per +1bp normal by expiry (rows, shortest on top) and swap
    tenor / cap index (columns); black at zero."""

    z = matrix.to_numpy(dtype=float)[::-1]
    rows = list(matrix.index)[::-1]
    peak = max(float(np.max(np.abs(z))) if z.size else 1.0, 1e-12)
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(matrix.columns), y=rows, colorscale=DIVERGING_COLORS,
            zmid=0.0, zmin=-peak, zmax=peak, text=[[f"{v:,.0f}" for v in row] for row in z], texttemplate="%{text}",
            textfont=CELL_TEXT, hovertemplate="%{y} x %{x}: %{text}<extra></extra>",
            colorbar=dict(title=dict(text=f"{unit}/BP")),
        )
    )  # fmt: skip
    dark_layout(fig, height=380, xtitle="SWAP TENOR / CAP INDEX", ytitle="OPTION / CAPLET EXPIRY", legend=False)
    fig.update_layout(hovermode="closest")
    rates_data.badge(fig, *source)
    return bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(matrix.columns) - 0.5, "min_span": 1.0},
        yaxis={"min": -0.5, "max": len(matrix.index) - 0.5, "min_span": 1.0},
    )


def _scenario_chart(totals: pd.Series, unit: str, source: Tuple[str, object]) -> go.Figure:
    names, values = list(totals.index)[::-1], [float(v) for v in totals.to_numpy()][::-1]  # first on top
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", name="BOOK P&L",
                           marker=dict(color=[SERIES[2] if v >= 0 else ACCENT for v in values]),
                           hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))  # fmt: skip
    dark_layout(fig, height=max(320, 26 * len(names) + 80), xtitle=f"BOOK P&L ({unit})", legend=False)
    fig.update_layout(margin=dict(l=220))
    rates_data.badge(fig, *source)
    return bound_axes(
        fig,
        xaxis=axis_limits(values + [-1.0, 1.0], pad=0.1),
        yaxis={"min": -0.5, "max": len(names) - 0.5, "min_span": 1.0},
    )


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _num(value: float, decimals: int = 0) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{round(value, decimals) + 0.0:,.{decimals}f}"


def _symbol(currency: str) -> str:
    return SYMBOLS.get(currency, currency + " ")


def _summary(risk: pf.BookRisk) -> pd.DataFrame:
    table = risk.summary()
    cols = {"id": "ID", "type": "TYPE", "description": "DESCRIPTION", "currency": "CCY", "pv": "PV", "pv01": "PV01",
            "gamma": "GAMMA", "vega": "VEGA (NORMAL BP)", "vol_source": "VOL"}  # fmt: skip
    return table.rename(columns=cols)


def _show_summary(risk: pf.BookRisk) -> pd.DataFrame:
    for ccy in sorted(set(risk.currencies)):
        picks = [k for k, c in enumerate(risk.currencies) if c == ccy]
        c = _symbol(ccy)
        cells = [
            (f"BOOK PV ({c})", _num(float(np.sum(risk.pv[picks])))),
            (f"BOOK PV01 ({c}/BP)", _num(float(np.sum(risk.pv01[picks])))),
            (f"BOOK VEGA ({c}/BP)", _num(float(np.sum(risk.vega[picks])))),
        ]
        for col, (label, value) in zip(st.columns(3), cells):
            col.metric(label, value)
    if risk.mixed:
        st.warning(
            "The book holds trades in more than one currency (US Treasuries are in dollars): totals are per "
            "currency, and the scenario, VaR and hedge totals below add them without FX conversion."
        )
    table = _summary(risk)
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={k: MONEY for k in ("PV", "PV01", "GAMMA", "VEGA (NORMAL BP)")})  # fmt: skip
    st.caption(
        "Per trade, in its currency and signed to the holder: PV; PV01 = P&L for +1bp on every rate quote of every "
        "curve (OIS, projection and Treasury curves rebuilt, tenor basis spreads held), central difference; GAMMA = "
        "P&L per (1bp)^2 of that move; VEGA = P&L per +1bp of normal vol (Black-model trades: of the normal vol "
        "with the same price). VOL: cube = the SWAPTION tab's SABR cube at the trade's strike, stripped = caplet "
        "vols stripped from cap quotes, flat = the trade's own vol. TOTAL rows add the trades of one currency."
    )
    for trade, reason in risk.skipped:
        st.warning(f"{trade.id} ({trade.type}) not priced: {reason}")
    if risk.notes:
        st.warning("Priced beyond the market data: " + "; ".join(risk.notes) + ".")
    return table


def _show_krd(risk: pf.BookRisk, market: Market) -> pd.DataFrame:
    st.markdown("### / KEY-RATE PV01 BY CURVE")
    curves = [k for k in risk.risk if np.any(np.abs(risk.risk[k].pv01) > rates_risk.NOISE)] or list(risk.risk)
    which = st.radio("CURVE", curves, horizontal=True, key="book_krd_curve",
                     format_func=lambda k: rates_risk.curve_title(market, k))  # fmt: skip
    unit = _symbol(risk.curve_currency[which])
    badge = rates_data.badge_for(which) or (f"{rates_risk.curve_title(market, which)} as built on the CURVE tab",
                                            market.asof)  # fmt: skip
    st.plotly_chart(_krd_chart(risk, which, unit, badge), width="stretch", config=PLOTLY_2D_CONFIG,
                    key="book_krd_chart")  # fmt: skip
    totals = risk.ladder_totals()
    part = totals[totals["curve"] == which]
    rate_sum = float(part["pv01"].sum())
    st.caption(
        f"P&L for +1bp on one quote of the {rates_risk.curve_title(market, which)} at a time (central difference, "
        "the curve and the curves built on it rebuilt), stacked by trade; the diamonds are the book. The buckets "
        f"sum to {unit}{_num(rate_sum, 2)}. Buckets under half a cent are rebuild noise and drawn as zero."
    )
    ladder = risk.ladder()
    wide = ladder.pivot_table(index=["curve", "quote", "years"], columns="id", values="pv01", sort=False).reset_index()
    wide["BOOK"] = wide[[t.id for t in risk.trades]].sum(axis=1) if risk.trades else 0.0
    st.download_button("DOWNLOAD KEY-RATE LADDER CSV (EVERY CURVE, BY TRADE)", wide.to_csv(index=False),
                       file_name="book_key_rates.csv", mime="text/csv", key="book_krd_csv",
                       on_click="ignore")  # fmt: skip
    return wide


def _show_vega(risk: pf.BookRisk, market: Market, vols: pf.VolSet) -> pd.DataFrame:
    st.markdown("### / VEGA BY EXPIRY x TENOR")
    if not len(risk.vega_buckets):
        st.info("No options in the book: no vega.")
        return pd.DataFrame()
    matrix = risk.vega_matrix()
    ccy = market.curve.conventions.currency if market.curve.conventions is not None else "USD"
    sources = sorted(set(vols.sources.values())) or ["flat vols of the trades"]
    st.plotly_chart(_vega_heatmap(matrix, _symbol(ccy), ("; ".join(sources), market.asof)), width="stretch",
                    config=PLOTLY_2D_CONFIG, key="book_vega_heatmap")  # fmt: skip
    st.caption(
        "P&L per +1bp of normal vol, summed over the book. Swaptions on the SABR cube: by the cube's ATM quote it "
        "moves (each ATM point bumped 1bp, smile shape held). Flat-vol swaptions: their vega split bilinearly "
        "between the nearest expiries and tenors of the grid. Caps and floors: each caplet's vega split linearly "
        "between the nearest expiries, in a column per cap index. Vol quotes: " + "; ".join(sources) + "."
    )
    st.download_button("DOWNLOAD VEGA BUCKETS CSV", risk.vega_buckets.to_csv(index=False), file_name="book_vega.csv",
                       mime="text/csv", key="book_vega_csv", on_click="ignore")  # fmt: skip
    return matrix.rename_axis("expiry").reset_index()


def _show_scenarios(risk: pf.BookRisk, market: Market, store) -> pd.DataFrame:
    st.markdown("### / SCENARIO P&L OF THE BOOK")
    positions = list(risk.positions)
    try:
        grid, failed, _ = rates_scenarios.library_grid(positions, market, store)
    except ValueError as exc:
        st.error(f"Cannot revalue the book under the scenario library: {exc}")
        return pd.DataFrame()
    if failed:
        st.warning("Not revalued: " + "; ".join(failed))
    if grid.empty:
        return pd.DataFrame()
    ccy = "/".join(sorted(set(risk.currencies)))
    unit = _symbol(ccy) if not risk.mixed else ccy + " "
    source = (rates_data.badge_for("curve") or ("curves of the CURVE tab", market.asof))[0]
    st.plotly_chart(_scenario_chart(grid[rates_scenarios.TOTAL], unit.strip(), (source, market.asof)),
                    width="stretch", config=PLOTLY_2D_CONFIG, key="book_scenario_chart")  # fmt: skip
    table = grid.rename_axis("SCENARIO").reset_index()
    with st.expander("SCENARIO P&L BY TRADE"):
        st.dataframe(table, hide_index=True, width="stretch", column_config={c: MONEY for c in table.columns[1:]})
    st.download_button("DOWNLOAD BOOK SCENARIO GRID CSV", table.to_csv(index=False), file_name="book_scenarios.csv",
                       mime="text/csv", key="book_scenario_csv", on_click="ignore")  # fmt: skip
    st.caption(
        "Full revaluation of every trade (value after the shock minus value today, quotes shocked and curves "
        "rebuilt). " + rates_scenarios.library_caption(store, market) + " Custom shocks, horizons and the P&L "
        "explain by trade: SCENARIOS tab with RISK SCOPE = BOOK."
    )
    return table


def _show_var(risk: pf.BookRisk, market: Market, store) -> pd.DataFrame:
    st.markdown("### / VALUE AT RISK OF THE BOOK (99%, 1 DAY)")
    try:
        _, scen = rates_research.var_scenarios(market, store, VAR_LOOKBACK)
        pnl = var_mod.sensitivity_pnl(scen, risk.risk)
        delta = var_mod.sensitivity_pnl(scen, risk.risk, gamma=False)
        table = var_mod.var_table(pnl, risk.ids, VAR_CONFIDENCE, delta)
    except ValueError as exc:
        st.warning(f"No VaR on {market.asof}: {exc}")
        return pd.DataFrame()
    total = table.iloc[-1]
    unit = _symbol(risk.currencies[0]) if risk.currencies and not risk.mixed else ""
    cells = [(f"HIST VAR 1D ({unit or 'mixed'})", _num(total["var"])),
             (f"HIST ES 1D ({unit or 'mixed'})", _num(total["es"])),
             (f"D-NORMAL VAR ({unit or 'mixed'})", _num(total["dn_var"]))]  # fmt: skip
    for col, (label, value) in zip(st.columns(3), cells):
        col.metric(label, value)
    if scen.short:
        st.warning(f"The daily history before {market.asof} holds only {scen.n} daily changes, not {VAR_LOOKBACK}.")
    shown = table.rename(columns={"position": "ID", "var": "HIST VAR", "es": "HIST ES", "worst_loss": "WORST LOSS",
                                  "mean_pnl": "MEAN P&L", "sd_pnl": "SD P&L", "dn_sigma": "DELTA-NORMAL SD",
                                  "dn_var": "DELTA-NORMAL VAR", "dn_es": "DELTA-NORMAL ES"})  # fmt: skip
    with st.expander("VAR BY TRADE"):
        st.dataframe(shown, hide_index=True, width="stretch", column_config={c: MONEY for c in shown.columns[1:]})
    st.caption(
        f"One-day 99% value at risk of the book from {scen.n} daily changes ({scen.previous[0]} to {scen.dates[-1]}) "
        "of each curve's own history or a flagged proxy, applied to today's quotes: P&L = key-rate PV01 x move + "
        "1/2 diagonal gamma x move^2 (the book's ladders above). Losses are positive; the book's VaR is not the sum "
        "of the trades'. Vols held. Lookback, confidence, full revaluation and the history used per curve: RESEARCH "
        "> VAR with RISK SCOPE = BOOK."
    )
    return shown


def _hedge_inputs(risk: pf.BookRisk, market: Market) -> Optional[Tuple[Dict[str, List[str]], Dict, Optional[float]]]:
    """(tenors per curve, bid/ask per (curve, tenor), move size or None)."""

    loaded = pf.default_benchmarks(risk, market)
    if not loaded:
        st.info("The book has no key-rate risk to hedge.")
        return None
    cols = st.columns(min(3, len(loaded)))
    bench: Dict[str, List[str]] = {}
    for n, key in enumerate(loaded):
        label = "TREASURIES" if key == GOVT_KEY else rates_risk.curve_title(market, key)
        tenors = cols[n % len(cols)].multiselect(f"HEDGE ON {label}", ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y",
                                                                       "20Y", "30Y"], default=list(pf.HEDGE_TENORS),
                                                 key=f"book_hedge_tenors_{key.replace(' ', '_').lower()}")  # fmt: skip
        if tenors:
            bench[key] = tenors
    if not bench:
        st.info("Pick hedge tenors on at least one curve.")
        return None
    assumed = pf.assumed_bid_ask(market, bench)
    c1, c2 = st.columns(2)
    penalty = c1.checkbox("COST PENALTY (BID/ASK)", value=True, key="book_hedge_penalty")
    move = c2.number_input("HEDGE AGAINST CURVE MOVES OF (BP)", min_value=0.1, max_value=100.0, value=5.0, step=0.5,
                           key="book_hedge_move", disabled=not penalty)  # fmt: skip
    costs = dict(assumed)
    if penalty:
        titles = {k: rates_risk.curve_title(market, k) for k in bench}
        keys = {title: k for k, title in titles.items()}
        table = pd.DataFrame([{"CURVE": titles[k], "TENOR": t, "BID/ASK (BP)": v} for (k, t), v in assumed.items()])
        digest = hashlib.sha1(repr(sorted(assumed)).encode()).hexdigest()[:8]
        edited = persistent_editor(
            table,
            hide_index=True,
            width="stretch",
            disabled=["CURVE", "TENOR"],
            key=f"book_hedge_costs_{digest}",
            column_config={"BID/ASK (BP)": st.column_config.NumberColumn(min_value=0.0, max_value=50.0, format="%.3f")},
        )
        costs = {(keys[r["CURVE"]], r["TENOR"]): float(r["BID/ASK (BP)"] if pd.notna(r["BID/ASK (BP)"]) else 0.0)
                 for r in edited.to_dict("records")}  # fmt: skip
        st.caption(
            "Bid/ask in bp of each hedge's own quote (par rate, basis spread or Treasury yield): ASSUMED defaults, "
            "not market data; edit them. Crossing half of it costs half the bid/ask x the hedge's DV01."
        )
    return bench, costs, float(move) if penalty else None


def _show_hedge(risk: pf.BookRisk, market: Market) -> Tuple[List[Position], pd.DataFrame, pd.DataFrame]:
    st.markdown("### / SUGGESTED HEDGES")
    inputs = _hedge_inputs(risk, market)
    if inputs is None:
        return [], pd.DataFrame(), pd.DataFrame()
    bench, costs, move = inputs
    mkey = rates_risk.market_key(market)
    try:
        instruments = pf.hedge_instruments(market, bench, bid_ask_bp=costs)
        bench_key = tuple((k, tuple(v)) for k, v in bench.items())
        ladders = rates_risk.cached("book_hedge_ladders", (mkey, bench_key, tuple(risk.risk)),
                                    lambda: pf.unit_ladders(instruments, market, list(risk.risk),
                                                            lambda m, k: rates_risk.bumps(m, k)))  # fmt: skip
        result = pf.hedge_book(risk, market, instruments, move, ladders=ladders)
    except ValueError as exc:
        st.error(f"Cannot size hedges: {exc}")
        return [], pd.DataFrame(), pd.DataFrame()
    unit = _symbol(risk.currencies[0]) if risk.currencies and not risk.mixed else ""
    cells = [  # short labels and plain numbers: a card at 1280px holds about 20 characters
        (f"RESIDUAL RISK ({unit}/BP)" if unit else "RESIDUAL (MIXED/BP)", _num(result.risk_after)),
        (f"UNHEDGED RISK ({unit}/BP)" if unit else "UNHEDGED (MIXED/BP)", _num(result.risk_before)),
        (f"BID/ASK COST ({unit or 'MIXED'})", _num(result.cost)),
    ]
    for col, (label, value) in zip(st.columns(3), cells):
        col.metric(label, value)
    table = result.table.rename(columns={"curve": "CURVE", "instrument": "HEDGE", "notional": "NOTIONAL",
                                         "direction": "DIRECTION", "par": "PAR", "bid_ask_bp": "BID/ASK (BP)",
                                         "cost": "COST", "pv01": "PV01"})  # fmt: skip
    table["CURVE"] = [rates_risk.curve_title(market, k) for k in table["CURVE"]]
    table["PAR"] = table["PAR"] * 100.0
    table = table.rename(columns={"PAR": "PAR (%)"})
    st.dataframe(table[table["NOTIONAL"] != 0.0], hide_index=True, width="stretch",
                 column_config={"NOTIONAL": MONEY, "COST": MONEY, "PV01": MONEY,
                                "PAR (%)": st.column_config.NumberColumn(format="%.4f")})  # fmt: skip
    rule = ""
    if result.move_bp is not None:
        rule = (f" With the cost penalty the hedge minimises the worst first-order loss on a {result.move_bp:g}bp "
                f"curve move ({_num(result.worst_before)} unhedged, {_num(result.worst_after)} hedged: the move size "
                "x the root-sum-square ladder) plus the bid/ask crossed, so a hedge is only added where the risk it "
                "takes out is worth its cost.")  # fmt: skip
    st.caption(
        "Par instruments traded today, sized to flatten the book's stacked key-rate ladder (every quote of every "
        "curve): OIS swaps on the discount curve, swaps or tenor basis swaps on a projection curve, par Treasuries "
        "on the Treasury curve; NOTIONAL > 0 receives fixed, receives the spread leg or buys the bond. RESIDUAL RISK "
        "is the root sum of squares of the ladder left (currency per bp), UNHEDGED RISK the same for the book: "
        f"{100 * result.risk_removed:.2f}% of the squared risk is removed. Buckets between hedge tenors cannot be "
        "taken out exactly, and a mix of hedges whose ladders almost cancel (a 15Y bond between the 10Y and 20Y "
        f"quotes, say) is not traded: it would take out next to no risk with huge offsetting notionals.{rule}"
    )
    rows = result.ladder.copy()
    rows["curve"] = [rates_risk.curve_title(market, k) for k in rows["curve"]]
    keys = [k for k in risk.risk if k in bench]
    if keys:
        which = st.radio("RESIDUAL LADDER ON", keys, horizontal=True, key="book_hedge_ladder_curve",
                         format_func=lambda k: rates_risk.curve_title(market, k))  # fmt: skip
        part = result.ladder[result.ladder["curve"] == which]
        st.plotly_chart(rates_risk.ladder_chart(part["quote"], part["book"], part["residual"], name="BOOK",
                                                unit=_symbol(risk.curve_currency[which])), width="stretch",
                        config=PLOTLY_2D_CONFIG, key="book_hedge_chart")  # fmt: skip
    st.download_button("DOWNLOAD HEDGE CSV (HEDGES, THEN BOOK / HEDGE / RESIDUAL LADDER)",
                       table.to_csv(index=False) + "\n" + rows.to_csv(index=False), file_name="book_hedge.csv",
                       mime="text/csv", key="book_hedge_csv", on_click="ignore")  # fmt: skip
    positions: List[Position] = []
    if st.checkbox("ADD THE HEDGES TO THE BOOK'S POSITIONS (SCENARIOS AND RESEARCH)", value=False,
                   key="book_add_hedges"):  # fmt: skip
        positions = list(result.positions)
    return positions, table, result.ladder


# --------------------------------------------------------------------------
# Tab
# --------------------------------------------------------------------------


def render(market: Market) -> List[Position]:
    """Draw the BOOK tab; returns the book's positions (and the hedges, if
    added)."""

    st.markdown("### / BOOK")
    conv = market.curve.conventions
    currency = conv.currency if conv is not None else "USD"
    edited, source = _blotter(currency)
    trades, issues = pf.parse_blotter(edited)
    _show_issues(issues)
    if not trades:
        st.info("The book is empty: add rows, load the example book or upload a blotter CSV.")
        return []
    vols, vols_key, notes, calibration = _vols(market, trades)
    for note in notes:
        st.warning(note)
    with st.spinner("Pricing the book and its key-rate risk..."):
        try:
            risk = _priced(market, trades, vols, vols_key)
        except ValueError as exc:
            st.error(f"Cannot price the book: {exc}")
            return []
    if not risk.trades:
        for trade, reason in risk.skipped:
            st.warning(f"{trade.id} ({trade.type}) not priced: {reason}")
        st.info("No trade of the book can be priced on this market.")
        return []
    positions = list(risk.positions)
    try:
        rates_scenarios.seed_risk(positions, market, risk.risk)
    except ValueError:
        pass
    summary = _show_summary(risk)
    ladder = _show_krd(risk, market)
    vega = _show_vega(risk, market, vols)
    store = rates_scenarios.history_store(market)
    scenarios = _show_scenarios(risk, market, store)
    var_table = _show_var(risk, market, store)
    hedges, hedge_table, hedge_ladder = _show_hedge(risk, market)
    st.markdown("### / EXPORT")
    tables = {"blotter": _text_table(edited), "book_summary": summary, "key_rates_by_trade": ladder,
              "vega_buckets": risk.vega_buckets, "vega_matrix": vega, "cashflows": book_cashflows(risk, market),
              "scenario_grid": scenarios, "var_table": var_table, "hedges": hedge_table, "hedge_ladder": hedge_ladder,
              "risk_report": risk_report(risk), **market_tables(market), **calibration}  # fmt: skip
    blotter_bytes = _text_table(edited).to_csv(index=False).encode("utf-8")
    rates_exports.export_row(
        "book", market, tables,
        {"blotter": {"source": source, "sha256": hashlib.sha256(blotter_bytes).hexdigest(),
                     "trades": [t.to_row() for t in risk.trades]},
         "skipped": [{"id": t.id, "reason": r} for t, r in risk.skipped],
         "hedge": {"added_to_positions": bool(hedges)}},
        {"risk": "central differences on full revaluation, +/-1bp per quote, curves rebuilt",
         "vols": dict(vols.sources),
         "hull_white": {"mean_reversion": vols.hull_white.mean_reversion, "sigma": vols.hull_white.sigma},
         "var": {"lookback_days": VAR_LOOKBACK, "confidence": VAR_CONFIDENCE}},
        key="book_export", store=store,
    )  # fmt: skip
    st.caption(
        "The ZIP holds every table above as CSV (blotter, risk by trade, key rates, vega, cash flows, "
        "scenarios, VaR, hedges, curve nodes and grid, vol calibration) with manifest.json."
    )
    return positions + hedges


def risk_scope(book: Sequence[Position], tab_trades: Sequence[Position]) -> Tuple[List[Position], str]:
    """The sidebar RISK SCOPE: which positions SCENARIOS and RESEARCH run
    on. Returns (positions, scope label)."""

    st.sidebar.markdown("### / RISK SCOPE")
    rates_data.sticky(SCOPE_KEY, SCOPES)
    scope = st.sidebar.radio(
        "SCENARIOS AND RESEARCH RUN ON", SCOPES, key=SCOPE_KEY,
        help="BOOK: the trades of the BOOK tab. TAB TRADES: the trade priced on each of the SWAP, SWAPTION, CAP-FLOOR "
        "and BONDS tabs. BOOK + TAB TRADES: both together.",
    )  # fmt: skip
    rates_data.remember(SCOPE_KEY)
    if scope == SCOPES[0]:
        positions = list(book)
    elif scope == SCOPES[1]:
        positions = list(tab_trades)
    else:
        positions = list(book) + list(tab_trades)
    st.sidebar.caption(f"{len(book)} book positions, {len(tab_trades)} tab trades.")
    return positions, scope
