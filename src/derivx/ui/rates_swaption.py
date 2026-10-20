"""SWAPTION tab of the Rates page: a European swaption on the curve set,
its premium and Greeks, the SABR smile per expiry x tenor, the ATM normal
vol matrix and the position's vega by expiry x tenor.

Vol quotes are not free: the tab starts from an ILLUSTRATIVE normal-vol grid
(labelled on every chart and table) that users edit, paste or upload. The
underlying is an OIS swap on the discount curve's conventions or, on a
multi-curve market, a swap against EURIBOR projected on its own curve and
discounted on €STR."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.rates.conventions import parse_tenor
from derivx.rates.curves import Market
from derivx.rates.swaption_vol import (
    DEFAULT_BETA,
    DEFAULT_SHIFT,
    GridPoint,
    SwaptionVolCube,
    VolGrid,
    butterfly_check,
    calendar_check,
    calibrate_cube,
    cube_vol,
    grid_points,
    parse_vol_grid,
    vega_buckets,
)
from derivx.rates.swaptions import Swaption, swaption_position
from derivx.rates.vol_examples import illustrative_swaption_grid
from derivx.risk.scenarios import theta
from derivx.risk.sensitivities import Position, rate_pv01_gamma
from derivx.ui import rates_data, rates_risk
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SERIES, dark_layout

MODEL_LABELS = {"Normal (bp)": "bachelier", "Black (%)": "black", "Shifted Black (%)": "shifted_black"}
FIT_LABELS = {"Normal SABR (beta 0)": "normal", "Shifted-lognormal SABR (beta 0.5)": "lognormal"}
KIND_LABELS = {"Payer": "payer", "Receiver": "receiver", "Straddle": "straddle"}
EXPIRIES = ("1M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y")
TENORS = ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")
VOL_UNITS = {"bachelier": ("BP", 1e-4, 1000.0), "black": ("%", 0.01, 300.0), "shifted_black": ("%", 0.01, 300.0)}
DEFAULT_FLAT = {"bachelier": 95.0, "black": 28.0, "shifted_black": 16.0}
GRID_KEY = "swn_user_grid"  # session state: {currency: {"name", "sha", "grid"}}
SMILE_OFFSETS_BP = np.linspace(-250.0, 250.0, 101)
MONEY = st.column_config.NumberColumn(format="%,.2f")


def _currency(market: Market) -> str:
    conv = market.curve.conventions
    return conv.currency if conv is not None else "USD"


def _num(value: float, decimals: int = 2) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{round(value, decimals) + 0.0:,.{decimals}f}"


# --------------------------------------------------------------------------
# Quotes
# --------------------------------------------------------------------------


def _read_upload(payload: bytes, name: str, vol_type: str, shift: float) -> Tuple[Optional[VolGrid], List[str]]:
    """Parse an uploaded or pasted CSV (comma, semicolon or tab separated:
    the delimiter is sniffed) into a vol grid and messages."""

    try:
        raw = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", dtype=str, keep_default_na=False,
                          na_values=[""], sep=None, engine="python")  # fmt: skip
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        return None, [f"{name}: could not read the file ({exc})"]
    units = "bp" if vol_type == "normal" else "percent"
    grid, issues = parse_vol_grid(raw, vol_type, units, shift, source=f"uploaded: {name}")
    return grid, [f"{name}: {i.message}" for i in issues]


def _quotes_expander(currency: str) -> Tuple[Optional[VolGrid], List[str]]:
    """Upload / paste / edit the vol grid. Returns (grid or None, messages);
    the user's grid persists per currency in session state."""

    uploads = st.session_state.setdefault(GRID_KEY, {})
    messages: List[str] = []
    mine = uploads.get(currency)
    base = mine["grid"] if mine else illustrative_swaption_grid(currency)
    slot = st.empty()  # filled once the quotes in use are known (an upload is read inside the expander)
    with st.expander("VOL QUOTES: EDIT, PASTE OR UPLOAD", expanded=False):
        st.caption(
            "Swaption vols by option expiry, swap tenor and strike offset from the ATM forward swap rate (bp). "
            "CSV layouts: wide (expiry, tenor, then one column per offset: -200, -100, ..., ATM, ..., +200) or "
            "long (expiry, tenor, offset_bp, vol). Every expiry x tenor needs an ATM quote."
        )
        c1, c2, c3 = st.columns([2, 1, 1])
        file = c1.file_uploader("VOL GRID CSV", type=["csv"], key=rates_data.widget_key("swn_grid_upload"))
        vol_type = c2.radio(
            "QUOTED AS",
            ["normal", "lognormal"],
            horizontal=True,
            key="swn_quote_type",
            format_func=lambda v: "Normal (bp)" if v == "normal" else "Lognormal (%)",
        )
        quote_shift = 0.0
        if vol_type == "lognormal":
            quote_shift = (
                c3.number_input(
                    "QUOTE SHIFT (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.5, key="swn_quote_shift"
                )
                / 100.0
            )
        pasted = st.text_area("OR PASTE CSV TEXT", value="", key="swn_grid_paste", height=80)
        use_paste = st.button("USE PASTED QUOTES", key="swn_grid_paste_use", disabled=not pasted.strip())
        payload, name = None, None
        if file is not None:
            payload, name = file.getvalue(), file.name
        elif use_paste and pasted.strip():
            payload, name = pasted.encode("utf-8"), "pasted quotes"
        if payload is not None:
            sha = hashlib.sha256(payload + f"{vol_type}{quote_shift}".encode()).hexdigest()
            if (mine or {}).get("sha") != sha:
                grid, notes = _read_upload(payload, name, vol_type, quote_shift)
                messages += notes
                if grid is not None:
                    uploads[currency] = {"name": name, "sha": sha, "grid": grid}
                    mine, base = uploads[currency], grid
        if mine is not None:
            st.caption(f"Using {mine['name']}: {len(base.expiries)} expiries x {len(base.tenors)} tenors.")
            st.button(
                "RESET TO ILLUSTRATIVE QUOTES",
                key="swn_grid_clear",
                on_click=rates_data.clear_upload,
                args=(GRID_KEY, "swn_grid_upload", currency),
            )
        units, scale = ("BP", 1e4) if base.vol_type == "normal" else ("%", 100.0)
        table = base.wide(scale)
        digest = hashlib.sha1(table.to_csv(index=False).encode()).hexdigest()[:10]
        st.caption(
            f"Vols in {units}"
            + (f" of lognormal vol on F + {base.shift:.2%}" if base.vol_type == "lognormal" else " of normal vol")
            + ". Edit cells to change the quotes."
        )
        edited = st.data_editor(
            table, hide_index=True, width="stretch", key=f"swn_grid_{currency}_{digest}", disabled=["expiry", "tenor"]
        )
    if edited.equals(table):
        slot.caption(f"Vol quotes in use: {base.source}.")
        return base, messages
    slot.caption(f"Vol quotes in use: {base.source} (edited by you).")
    grid, issues = parse_vol_grid(
        edited,
        base.vol_type,
        "bp" if base.vol_type == "normal" else "percent",
        base.shift,
        source=f"{base.source} (edited by you)",
    )
    messages += [i.message for i in issues]
    return grid, messages


@st.cache_data(show_spinner=False, max_entries=12)
def _calibrated(
    rows: Tuple[Tuple[str, str, float, float], ...],
    vol_type: str,
    quote_shift: float,
    source: str,
    points: Tuple[Tuple[str, str, float, float, float, float], ...],
    fit: str,
    beta: float,
    shift: float,
    weights: Optional[str],
    index: Optional[str],
) -> SwaptionVolCube:
    """SABR at every point; keyed on the quotes, the market's forwards and
    the settings only (no session state)."""

    grid = VolGrid(rows, vol_type, quote_shift, source)
    return calibrate_cube(grid, [GridPoint(*p) for p in points], fit, beta, shift, weights, index)


def _cube(
    grid: VolGrid, market: Market, index: Optional[str], fit: str, beta: float, shift: float, weights: Optional[str]
) -> SwaptionVolCube:
    key = (rates_risk.market_key(market), grid.rows, index)
    points = rates_risk.cached("swn_points", key, lambda: grid_points(grid, market, index))
    flat = tuple((p.expiry, p.tenor, p.expiry_years, p.tenor_years, p.forward, p.annuity) for p in points)
    with st.spinner("Calibrating SABR at every expiry x tenor..."):
        return _calibrated(grid.rows, grid.vol_type, grid.shift, grid.source, flat, fit, beta, shift, weights, index)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def _smile_chart(
    cube: SwaptionVolCube, trade: Swaption, T: float, F: float, trade_normal: float, source: str, asof
) -> go.Figure:
    """The cube's smile at the trade's expiry x tenor (normal bp against the
    strike offset), the quotes when the point is quoted, and the trade at
    the normal vol it is priced at (``trade_normal``, rate units: the cube's
    vol, or a flat vol's normal equivalent). Raises ``ValueError`` when the
    cube has no smile there."""

    L = parse_tenor(trade.tenor).years
    offsets = SMILE_OFFSETS_BP[F + SMILE_OFFSETS_BP * 1e-4 + cube.shift > 1e-4]
    if offsets.size < 2:
        raise ValueError(f"the forward {F:.3%} is too close to the SABR shift ({-cube.shift:.2%}) to draw a smile")
    model = np.asarray(cube.normal_vol(T, L, F, F + offsets * 1e-4), dtype=float) * 1e4
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=offsets, y=model, mode="lines", name="SABR (CUBE)", line=dict(color=SERIES[0], width=2)))
    shown = list(model)
    quoted = trade.expiry in cube.expiries and trade.tenor in cube.tenors
    if quoted:
        point = cube.point(trade.expiry, trade.tenor)
        fig.add_trace(
            go.Scatter(
                x=point.offsets_bp,
                y=point.market_normal * 1e4,
                mode="markers",
                name="QUOTES",
                marker=dict(color=SERIES[3], size=9, symbol="circle-open"),
            )
        )
        shown += list(point.market_normal * 1e4)
    trade_offset = (trade.strike - F) * 1e4
    xs = list(offsets)
    if math.isfinite(trade_normal):
        trade_vol = trade_normal * 1e4
        fig.add_trace(
            go.Scatter(
                x=[trade_offset],
                y=[trade_vol],
                mode="markers",
                name="TRADE STRIKE",
                marker=dict(color=ACCENT, size=12, symbol="diamond"),
            )
        )
        xs.append(trade_offset)
        shown.append(trade_vol)
    dark_layout(fig, height=420, xtitle="STRIKE - ATM FORWARD (BP)", ytitle="NORMAL VOL (BP)")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.0), margin=dict(b=100))  # off the curve
    rates_data.badge(fig, source, asof)
    return bound_axes(
        fig,
        xaxis=axis_limits(xs, pad=0.03),
        yaxis=axis_limits(shown, pad=0.12, floor=0.0),
    )


def _heatmap(z: np.ndarray, expiries, tenors, title: str, fmt: str, diverging: bool, source: str, asof) -> go.Figure:
    """Expiry x tenor matrix, shortest expiry on top; a diverging map is
    black at zero (red below, blue above)."""

    z = np.asarray(z, dtype=float)[::-1]  # plotly draws the first row at the bottom
    rows = list(expiries)[::-1]
    finite = z[np.isfinite(z)]
    peak = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-12)
    if diverging:
        scale = [[0.0, ACCENT], [0.5, "#111111"], [1.0, SERIES[2]]]
        kwargs = dict(colorscale=scale, zmid=0.0, zmin=-peak, zmax=peak, textfont=dict(color=SERIES[0]))
    else:
        kwargs = dict(colorscale="Viridis")
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(tenors),
            y=rows,
            text=[[fmt.format(v) for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="%{y} x %{x}: %{text}<extra></extra>",
            colorbar=dict(title=dict(text=title)),
            **kwargs,
        )
    )
    dark_layout(fig, height=360, xtitle="SWAP TENOR", ytitle="OPTION EXPIRY", legend=False)
    fig.update_layout(hovermode="closest")
    rates_data.badge(fig, source, asof)
    return bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(tenors) - 0.5, "min_span": 1.0},
        yaxis={"min": -0.5, "max": len(expiries) - 0.5, "min_span": 1.0},
    )


# --------------------------------------------------------------------------
# Tab
# --------------------------------------------------------------------------


def _risk(trade: Swaption, market: Market) -> Dict[str, float]:
    """Curve PV01/gamma and the full one-day theta (curves rebuilt), cached."""

    def compute() -> Dict[str, float]:
        position = swaption_position(trade, market)
        pv01, gamma = rate_pv01_gamma(position.pv, market)
        return {"pv01": pv01, "gamma": gamma, "theta": float(theta([position], market)[0])}

    return rates_risk.cached("swn_risk", (rates_risk.market_key(market), repr(trade)), compute)


def _inputs(market: Market) -> Optional[dict]:
    """Trade, vol and SABR inputs, three to a row (wider rows wrap next to
    the sidebar)."""

    indices = [k for k in market.projections if not market.projections.is_discount(k)]
    c1, c2, c3 = st.columns(3)
    kind = KIND_LABELS[c1.radio("TYPE", list(KIND_LABELS), horizontal=True, key="swn_kind")]
    side = c2.radio("POSITION", ["Long", "Short"], horizontal=True, key="swn_side")
    notional = c3.number_input(
        f"NOTIONAL ({rates_risk.money(market)})",
        min_value=1e3,
        max_value=1e11,
        value=1e7,
        step=1e6,
        format="%.0f",
        key="swn_notional",
    )
    c1, c2, c3 = st.columns(3)
    under = c1.radio(
        "UNDERLYING",
        indices + ["OIS"] if indices else ["OIS"],
        horizontal=True,
        key=f"swn_index_{_currency(market)}",
        help="OIS: an OIS swap on the discount curve's conventions (SOFR, €STR, SONIA). EURIBOR: fixed annual "
        "30/360 against EURIBOR, projected on its curve and discounted on €STR.",
    )
    expiry = c2.selectbox("OPTION EXPIRY", EXPIRIES, index=EXPIRIES.index("1Y"), key="swn_expiry")
    tenor = c3.selectbox("SWAP TENOR", TENORS, index=TENORS.index("5Y"), key="swn_tenor")
    c1, c2, c3 = st.columns(3)
    strike_mode = c1.radio("STRIKE", ["ATM ± bp", "Absolute %"], horizontal=True, key="swn_strike_mode")
    strike_pct = offset_bp = None
    if strike_mode == "Absolute %":
        strike_pct = c2.number_input(
            "STRIKE (%)", min_value=-5.0, max_value=30.0, value=3.5, step=0.125, format="%.4f", key="swn_strike_pct"
        )
    else:
        offset_bp = c2.number_input(
            "ATM OFFSET (BP)",
            min_value=-500.0,
            max_value=500.0,
            value=0.0,
            step=25.0,
            format="%.1f",
            key="swn_offset_bp",
        )
    model = MODEL_LABELS[c3.selectbox("VOL UNIT / MODEL", list(MODEL_LABELS), key="swn_model")]
    c1, c2, c3 = st.columns(3)
    source = c1.radio("VOL", ["SABR cube", "Flat vol"], horizontal=True, key="swn_vol_source")
    label, unit, top = VOL_UNITS[model]
    flat = None
    if source == "Flat vol":
        flat = (
            c2.number_input(
                f"FLAT VOL ({label})",
                min_value=0.0,
                max_value=top,
                value=DEFAULT_FLAT[model],
                step=1.0,
                format="%.2f",
                key=f"swn_flat_vol_{model}",
            )
            * unit
        )
    shift = 0.0
    if model == "shifted_black":
        shift = (
            c3.number_input(
                "MODEL SHIFT (%)",
                min_value=0.0,
                max_value=20.0,
                value=3.0,
                step=0.25,
                format="%.2f",
                key="swn_model_shift",
            )
            / 100.0
        )
    c1, c2, c3 = st.columns(3)
    fit = FIT_LABELS[c1.selectbox("SABR FIT", list(FIT_LABELS), key="swn_fit")]
    beta = c2.number_input(
        "SABR BETA",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_BETA[fit],
        step=0.1,
        format="%.2f",
        key=f"swn_beta_{fit}",
    )
    sabr_shift = (
        c3.number_input(
            "SABR SHIFT (%)",
            min_value=0.0,
            max_value=20.0,
            value=DEFAULT_SHIFT * 100,
            step=0.5,
            format="%.2f",
            key="swn_sabr_shift",
        )
        / 100.0
    )
    weights = "vega" if st.checkbox("VEGA-WEIGHTED FIT", value=False, key="swn_vega_weights") else None
    return dict(
        kind=kind,
        sign=1.0 if side == "Long" else -1.0,
        notional=float(notional),
        expiry=expiry,
        tenor=tenor,
        index=None if under == "OIS" else under,
        strike_pct=strike_pct,
        offset_bp=offset_bp,
        model=model,
        shift=shift,
        flat=flat,
        fit=fit,
        beta=float(beta),
        sabr_shift=float(sabr_shift),
        weights=weights,
    )


def render(market: Market) -> List[Position]:
    """Draw the tab and return the swaption as a position."""

    st.markdown("### / SWAPTION")
    currency = _currency(market)
    ui = _inputs(market)
    grid, messages = _quotes_expander(currency)
    for message in messages:
        (st.caption if "fewer than" in message else st.error)(message)
    if grid is None:
        st.error("Fix the vol quotes above to price swaptions.")
        return []
    try:
        cube = _cube(grid, market, ui["index"], ui["fit"], ui["beta"], ui["sabr_shift"], ui["weights"])
        probe = Swaption(1.0, ui["expiry"], ui["tenor"], 0.0, index=ui["index"])
        inputs = probe.inputs(market)
        F, T = inputs.forward, inputs.expiry_years
        K = ui["strike_pct"] / 100.0 if ui["strike_pct"] is not None else F + ui["offset_bp"] * 1e-4
        vol = ui["flat"] if ui["flat"] is not None else cube_vol(cube, T, ui["tenor"], F, K, ui["model"], ui["shift"])
        trade = Swaption(
            ui["sign"] * ui["notional"],
            ui["expiry"],
            ui["tenor"],
            K,
            ui["kind"],
            vol,
            ui["model"],
            ui["shift"],
            index=ui["index"],
        )
        g = trade.greeks(market)
        risk = _risk(trade, market)
    except ValueError as err:
        st.error(f"Cannot price this swaption: {err}")
        return []

    c = rates_risk.money(market)
    label, unit, _ = VOL_UNITS[ui["model"]]

    cells = [
        (f"PV ({c})", _num(g["pv"])),
        ("PREMIUM (BP)", _num(g["premium_bp"], 3)),
        ("FWD SWAP RATE (%)", _num(g["forward"] * 100, 4)),
        ("STRIKE (%)", _num(K * 100, 4)),
        (f"VOL ({label})", _num(vol / unit, 2)),
        ("NORMAL VOL EQUIV (BP)", _num(g["normal_vol"] * 1e4, 2)),
        (f"DELTA ({c}/BP FWD)", _num(g["delta_fwd"])),
        (f"PV01 ({c}/BP CURVE)", _num(risk["pv01"])),
        (f"GAMMA ({c}/BP²)", _num(g["gamma_fwd"], 4)),
        (f"VEGA ({c}/BP NORMAL)", _num(g["vega_normal_bp"])),
        (f"VEGA ({c}/VOL PT)", _num(g["vega_lognormal_pt"])),
        (f"THETA ({c}, 1 BUS. DAY)", _num(risk["theta"])),
    ]
    for start in range(0, len(cells), 3):
        for col, (name, value) in zip(st.columns(3), cells[start : start + 3]):
            if name == "STRIKE (%)":
                col.metric(name, value, delta=f"{g['atm_offset_bp']:+.1f}bp vs ATM", delta_color="off")
            else:
                col.metric(name, value)
    swap = trade.underlying(market)
    under = ui["index"] or f"{(market.curve.conventions.name if market.curve.conventions else 'USD SOFR')} OIS"
    st.caption(
        f"{trade.label}. Expiry {inputs.expiry_date} ({T:.4f}Y ACT/365F); underlying {under} swap "
        f"{inputs.start} to {inputs.maturity}, fixed every {swap.fixed_months}M {swap.fixed_day_count}; annuity "
        f"{c}{_num(g['annuity_bp'])} per bp. Physically settled; every flow discounted on the OIS curve"
        + (f", the float leg projected on {ui['index']}." if ui["index"] else ".")
    )
    st.caption(
        f"Signed to you ({'long' if ui['sign'] > 0 else 'short'}), in {c} for the notional. DELTA: P&L for +1bp on "
        "the forward swap rate, annuity held (delta notional "
        f"{_num(g['delta_notional'], 0)} of payer swap). PV01: +1bp on every rate quote, curves rebuilt (moves the "
        "annuity too). GAMMA: change in DELTA per +1bp. VEGA per +1bp of normal vol and per +1 vol point of "
        f"lognormal vol (shift {trade.shift:.2%}), each at the vol of that type with the same price. THETA: value "
        f"on the next business day with the curve quotes and vol unchanged (curves rolled) minus today; time decay "
        f"alone (forward and annuity held) is {_num(g['decay'])}. Greeks are at a fixed vol (sticky strike)."
    )
    if ui["model"] != "bachelier" and ui["flat"] is None:
        st.caption(
            f"The cube is in normal vol: the {label} vol shown has the cube's price at this strike (exact price "
            "matching), so the premium is the same in every model; delta and gamma depend on the model."
        )

    st.markdown("### / SABR SMILE")
    try:
        smile = _smile_chart(cube, trade, T, F, g["normal_vol"], grid.source, market.asof)
    except ValueError as err:
        st.warning(f"No smile chart for {ui['expiry']}x{ui['tenor']}: {err}")
    else:
        st.plotly_chart(smile, width="stretch", config=PLOTLY_2D_CONFIG, key="swn_smile_chart")
        if ui["flat"] is not None:
            st.caption(
                "The trade is priced at its flat vol (the diamond, as a normal vol); the line is the cube's smile."
            )
        if trade.strike + cube.shift <= 0:
            st.caption(f"The strike is below the SABR shift ({-cube.shift:.2%}): the cube has no vol there.")
    quoted = ui["expiry"] in cube.expiries and ui["tenor"] in cube.tenors
    if quoted:
        point = cube.point(ui["expiry"], ui["tenor"])
        params = pd.DataFrame(
            [
                {
                    "ALPHA": point.alpha,
                    "BETA": point.beta,
                    "RHO": point.rho,
                    "NU": point.nu,
                    "SHIFT (%)": point.shift * 100,
                    "RMSE (BP)": point.rmse_bp,
                    "MAX ERROR (BP)": point.max_abs_bp,
                    "STATUS": point.status,
                }
            ]
        )
        st.dataframe(
            params,
            hide_index=True,
            width="stretch",
            column_config={k: st.column_config.NumberColumn(format="%.4f") for k in params.columns[:-1]},
        )
        residuals = pd.DataFrame(
            {
                "OFFSET (BP)": point.offsets_bp,
                "STRIKE (%)": point.strikes * 100,
                "QUOTE (BP NORMAL)": point.market_normal * 1e4,
                "SABR (BP NORMAL)": point.model_normal * 1e4,
                "RESIDUAL (BP)": point.residual_bp,
            }
        )
        st.dataframe(
            residuals,
            hide_index=True,
            width="stretch",
            column_config={k: st.column_config.NumberColumn(format="%.4f") for k in residuals.columns},
        )
        check = butterfly_check(point)
        if check["negative_offsets_bp"]:
            st.warning(
                f"Butterfly arbitrage in the fitted smile: negative density at offsets "
                f"{check['negative_offsets_bp'][0]:.0f} to {check['negative_offsets_bp'][-1]:.0f}bp."
            )
        if point.status != "fitted":
            st.warning(f"{point.expiry}x{point.tenor}: {point.status} ({point.message})")
    else:
        st.caption(
            f"{ui['expiry']}x{ui['tenor']} is between quoted points: the smile is the cube's interpolation "
            "(ATM in variance-time, rho and nu bilinear, alpha from the interpolated ATM vol)."
        )
    fit_name = next(k for k, v in FIT_LABELS.items() if v == ui["fit"])
    st.caption(
        f"{fit_name} with beta {cube.beta:.2f} and shift {cube.shift:.2%}, fitted per expiry x tenor; alpha solved "
        "from the ATM quote so ATM reprices exactly. Residuals are model minus quote in bp of normal vol. "
        f"Source: {grid.source}."
    )
    table = cube.fit_table()
    failed = table[table["status"] != "fitted"]
    if len(failed):
        st.warning(
            f"{len(failed)} point(s) not fitted (flat smile used): "
            + ", ".join(f"{r.expiry}x{r.tenor} ({r.status})" for r in failed.itertuples())
        )
    drops = calendar_check(cube)
    if drops:
        st.warning(
            "ATM total variance falls with expiry at "
            + ", ".join(f"{d['tenor']} {d['from']}->{d['to']}" for d in drops[:6])
            + " (different underlyings, so a diagnostic rather than a strict arbitrage)."
        )
    with st.expander("CALIBRATION AT EVERY POINT"):
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "DOWNLOAD SABR CALIBRATION CSV",
            table.to_csv(index=False),
            file_name="swaption_sabr.csv",
            mime="text/csv",
            on_click="ignore",
            key="swn_fit_csv",
        )

    st.markdown("### / ATM NORMAL VOL (BP)")
    st.plotly_chart(
        _heatmap(cube.atm_normal * 1e4, cube.expiries, cube.tenors, "BP", "{:.1f}", False, grid.source, market.asof),
        width="stretch",
        config=PLOTLY_2D_CONFIG,
        key="swn_atm_heatmap",
    )
    st.caption(
        "ATM normal vols of the quotes. Between them the cube interpolates total variance (vol² x T) "
        "linearly in expiry and linearly in tenor; flat vol outside the grid."
    )

    if ui["flat"] is None:
        st.markdown(f"### / VEGA BY EXPIRY x TENOR ({c} PER +1BP NORMAL)")
        buckets = rates_risk.cached(
            "swn_vega",
            (rates_risk.market_key(market), repr(trade), grid.rows, cube.fit, cube.beta, cube.shift, ui["weights"]),
            lambda: vega_buckets([trade], cube, market),
        )
        st.plotly_chart(
            _heatmap(buckets, cube.expiries, cube.tenors, c, "{:,.0f}", True, grid.source, market.asof),
            width="stretch",
            config=PLOTLY_2D_CONFIG,
            key="swn_vega_heatmap",
        )
        st.caption(
            f"P&L for +1bp of normal vol on one ATM quote at a time, smile shape (rho, nu) held, the trade's vol "
            f"moving with the cube at its strike. Buckets sum to {_num(float(buckets.sum()))} against a vega of "
            f"{_num(g['vega_normal_bp'])} per +1bp normal."
        )
    else:
        st.caption("Flat vol: no vega by quote (the vol does not come from the cube).")

    with st.expander("IMPLIED VOL FROM A PREMIUM"):
        shown_premium = float(round(abs(g["premium_bp"]), 3))
        # Keyed on the trade and its premium: the input starts again from the premium shown above when either
        # changes, instead of keeping an earlier trade's premium (implied vols for a stale premium).
        tag = hashlib.sha1(f"{trade!r}|{shown_premium}".encode()).hexdigest()[:10]
        premium = st.number_input(  # no upper bound: deep in-the-money long swaptions exceed 5,000bp
            "PREMIUM (BP OF NOTIONAL, POSITIVE)",
            min_value=0.0,
            value=shown_premium,
            step=1.0,
            format="%.3f",
            key=f"swn_implied_premium_{tag}",
        )
        pv = ui["sign"] * premium * ui["notional"] * 1e-4
        cols = st.columns(2)
        for col, (name, model, scale, fmt) in zip(
            cols,
            [("IMPLIED NORMAL VOL (BP)", "bachelier", 1e4, 2), ("IMPLIED LOGNORMAL VOL (%)", "shifted_black", 100, 3)],
        ):
            try:
                value = trade.implied_vol(pv, market, model=model, shift=trade.shift) * scale
                col.metric(name, _num(value, fmt))
            except ValueError as err:
                col.metric(name, "n/a")
                col.caption(str(err))
        st.caption(
            f"Lognormal vol of F + {trade.shift:.2%} (the model shift). Premium per unit of notional, "
            "paid for the option; the solver works on the out-of-the-money side, so deep in-the-money "
            "premiums keep their digits."
        )
    return [swaption_position(trade, market)]
