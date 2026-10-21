"""EQUITY option pricer page: premium, Greeks in desk units, an
implied-vol solver, a spot x vol scenario P&L grid, a risk surface and the
smile. FX options, with their market conventions, have their own page
(:mod:`derivx.ui.fx_page`); this module also provides the card styles
(``PAGE_CSS``) both pages share.

Model: Black-Scholes-Merton on spot (:mod:`derivx.core.bs`), with ``q`` the
dividend yield. Units: spot, strike and premiums in price units (strike
currency per one unit of underlying), T in years (ACT/365F), r, q and vols
as per-annum decimals. Greeks are for a long option on one unit of
underlying, signed as the P&L (or change in delta) for the stated move: +1%
spot, +1 vol point, one calendar day, +1bp.

Nothing here is random. The default smile is an ILLUSTRATIVE model smile
from user-set parameters; a MARKET smile is offered only from a chain loaded
in the ANALYTICS lab (``st.session_state['market_chain']``). Every chart is
zoom-bounded by :mod:`derivx.ui.plot_limits` and styled by
:mod:`derivx.ui.theme`. Widget keys are namespaced (``eq_*``; the FX page
uses ``fx_*``) and persist across mode switches (:mod:`derivx.ui.state`).
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.calibration.production import ChainCalibration, calibrate_chain
from derivx.core.bs import DAYS_PER_YEAR, BSGreeks, bs_greeks, bs_theta_1d, desk_greeks
from derivx.ui import pricer_calc as calc
from derivx.ui.plot_limits import GUARD_AVAILABLE, axis_limits, bound_axes, bound_scene
from derivx.ui.state import PERSIST
from derivx.ui.theme import ACCENT, FOREGROUND, PLOTLY_2D_CONFIG, PNL_COLORS, SERIES, dark_layout
from derivx.vol.discounting import Discounting
from derivx.vol.implied import STATUS, implied_vol_bs
from derivx.vol.svi import svi_iv

DEFAULT_CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.2))
# Zoom and pan are bounded by derivx.ui.plot_limits. Wheel zoom (pinch or
# Ctrl/Cmd + wheel) and 3D pan need its in-page guard, so both are off without
# it. The 3D zoom tool only moves the camera, which an orthographic view does
# not show, so it is dropped.
ZOOM_HINT = "Pinch or Ctrl/⌘ + scroll to zoom (bounded) · home icon resets the view"
PLOTLY_3D_CONFIG = {
    "displaylogo": False,
    "scrollZoom": GUARD_AVAILABLE,
    "modeBarButtonsToRemove": [
        "toImage",
        "lasso3d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "zoom3d",
    ]
    + ([] if GUARD_AVAILABLE else ["pan3d"]),
    "modeBarButtonsToAdd": ["orbitRotation", "resetCameraDefault3d"] + (["pan3d"] if GUARD_AVAILABLE else []),
}
SURFACE_COLORS = [
    [0.0, "#00113D"],
    [0.1, "#003B82"],
    [0.25, "#0E7BCF"],
    [0.45, "#35C1FF"],
    [0.6, "#7FFFD4"],
    [0.75, "#FFE66D"],
    [0.9, "#FF9E4A"],
    [1.0, "#E5243B"],
]
CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "HKD", "OTHER")
PRESET_FIELDS = ("spot", "strike", "T", "r", "q", "vol")
VOL_MAX = 5.0  # VOL.σ input bound (500%)


@dataclass(frozen=True)
class ModeSpec:
    """Per-mode settings of the pricer page (EQUITY; FX has its own page,
    :mod:`derivx.ui.fx_page`)."""

    name: str
    prefix: str  # widget-key namespace
    price_format: str
    price_step: float
    price_decimals: int  # premium decimals in the hero card
    r_label: str
    q_label: str
    q_greek: str  # label of dV/dq
    underlying: str  # what one unit of underlying is
    smile_model: str
    default_position: float
    position_help: str
    presets: Mapping[str, Mapping[str, float]]


MODES: Dict[str, ModeSpec] = {
    "EQUITY": ModeSpec(
        name="EQUITY",
        prefix="eq",
        price_format="%.2f",
        price_step=1.0,
        price_decimals=4,
        r_label="RATE.R (DECIMAL)",
        q_label="DIV YIELD.Q (DECIMAL)",
        q_greek="RHO (DIV.Q)",
        underlying="one share / index point",
        smile_model="SVI",
        default_position=100.0,
        position_help="Signed units of underlying the options are on: +100 = long one 100-share contract, "
        "-100 = short. Scales the scenario P&L only.",
        presets={
            "DEFAULT": {"spot": 100.0, "strike": 100.0, "T": 1.0, "r": 0.05, "q": 0.0, "vol": 0.2},
            "HIGH VOL": {"spot": 100.0, "strike": 100.0, "T": 2.0, "r": 0.05, "q": 0.0, "vol": 0.6},
            "DEEP ITM": {"spot": 150.0, "strike": 80.0, "T": 1.0, "r": 0.05, "q": 0.0, "vol": 0.3},
            "FAR OTM": {"spot": 80.0, "strike": 120.0, "T": 0.5, "r": 0.05, "q": 0.0, "vol": 0.4},
        },
    ),
}

# The 3D risk surface is orthographic: Plotly fixes the visible height of the
# scene and scales its visible width with the canvas's width / height, so a
# tall, narrow canvas cuts off the left and right of the default view (the
# z-axis title and ticks, the end spot ticks; UIE-09). The chart therefore
# fills a container whose CSS (in PAGE_CSS) keeps width / height at
# SURFACE_MIN_ASPECT up to SURFACE_HEIGHT px: 560 px tall from 504 px of
# width up, 320 px tall at 288 px (a 320 px phone). tests/test_pricer_page.py
# checks the ratio against the labels' extent measured in Chromium.
SURFACE_HEIGHT = 560  # px, chart height on a wide screen
SURFACE_MIN_ASPECT = 0.9  # chart width / height below SURFACE_HEIGHT * SURFACE_MIN_ASPECT px of width
SURFACE_MARGIN = dict(l=0, r=0, t=10, b=50)  # px; the colour bar sits in the bottom margin

PAGE_CSS = """
<style>
.dx-hero { background:#FFFFFF; color:#000000; border:4px solid #000000; padding:2.2rem 1.25rem 1.25rem;
  margin:1rem 0 2rem; position:relative; }
.dx-hero-tag { position:absolute; top:-15px; left:20px; background:#B91C1C; color:#FFFFFF; padding:5px 15px;
  font-weight:700; font-size:0.8rem; letter-spacing:0.1em; }
.dx-mono, .dx-hero-value, .dx-cell b, .dx-greek-value { font-family:'IBM Plex Mono', monospace !important; }
.dx-hero-value { text-align:center; font-size:clamp(1.9rem, 8vw, 4.2rem); font-weight:700; line-height:1.1;
  white-space:nowrap; overflow-x:auto; }
.dx-hero-ccy { font-size:0.4em; font-weight:600; letter-spacing:0.08em; }
.dx-hero-unit { text-align:center; font-size:0.85rem; margin-top:0.4rem; color:#333333; }
.dx-hero-grid { display:flex; flex-wrap:wrap; gap:0.5rem 1.25rem; justify-content:center; margin-top:1.1rem; }
.dx-cell { display:flex; flex-direction:column; align-items:center; min-width:6.5rem; }
.dx-cell span { font-size:0.7rem; letter-spacing:0.12em; color:#333333; white-space:nowrap; }
.dx-cell b { font-size:1rem; white-space:nowrap; }
.dx-greeks { display:grid; grid-template-columns:repeat(auto-fill, minmax(10.5rem, 1fr)); gap:0.75rem;
  margin:0.5rem 0 1.25rem; }
.dx-greek { background:#FFFFFF; color:#000000; border:3px solid #000000; padding:0.7rem 0.8rem; }
.dx-greek-name { font-size:0.75rem; font-weight:700; letter-spacing:0.12em; }
.dx-greek-value { font-size:1.45rem; font-weight:700; white-space:nowrap; overflow-x:auto; }
.dx-greek-unit, .dx-greek-sub { font-size:0.75rem; color:#333333; }
.dx-greek-sub { margin-top:0.2rem; }
.dx-badge { display:inline-block; padding:2px 8px; font-weight:700; font-size:0.75rem; letter-spacing:0.1em;
  border:2px solid currentColor; margin-right:0.5rem; }
"""
PAGE_CSS += (
    "".join(
        f".st-key-{s.prefix}_surface_box {{ aspect-ratio:{SURFACE_MIN_ASPECT}; max-height:{SURFACE_HEIGHT}px; }}\n"
        for s in MODES.values()
    )
    + "</style>\n"
)


def _k(spec: ModeSpec, name: str) -> str:
    return f"{spec.prefix}_{name}"


def _defaults(spec: ModeSpec) -> Dict[str, object]:
    return {
        "preset": "DEFAULT",
        **spec.presets["DEFAULT"],
        "type": "CALL",
        "ccy": "USD",
        "position": spec.default_position,
        "scn_spot": 10.0,
        "scn_vol": 5.0,
        "scn_days": 0.0,
        "surf_axes": calc.SURFACE_AXES[0],
        "surf_z": "VALUE",
        "surf_res": 25,
        "surf_sd": 3.0,
        "smile_src": "ILLUSTRATIVE",
        "smile_rho": -0.6 if spec.smile_model == "SVI" else -0.1,
        "smile_curv": 1.0 if spec.smile_model == "SVI" else 0.6,
        "smile_sd": 3.0,
    }


def _init_state(spec: ModeSpec) -> None:
    """Seed widget values once (widgets are created without ``value=``)."""

    for name, value in _defaults(spec).items():
        st.session_state.setdefault(_k(spec, name), value)
    st.session_state.setdefault(_k(spec, "iv_premium"), None)
    st.session_state.setdefault(_k(spec, "camera_epoch"), 0)


def apply_preset(spec: ModeSpec) -> None:
    """on_change of the PRESET radio: runs before the next script run, so
    writing the inputs' session state here sets the widgets (UIE-03)."""

    for name, value in spec.presets[st.session_state[_k(spec, "preset")]].items():
        st.session_state[_k(spec, name)] = value
    st.session_state[_k(spec, "camera_epoch")] = st.session_state.get(_k(spec, "camera_epoch"), 0) + 1


def _set_vol(spec: ModeSpec, vol: float) -> None:
    st.session_state[_k(spec, "vol")] = float(vol)


def _num(label: str, spec: ModeSpec, name: str, container=None, **kwargs) -> Optional[float]:
    box = container if container is not None else st.sidebar
    return box.number_input(label, key=_k(spec, name), persist_state=PERSIST, **kwargs)


def _sidebar_inputs(spec: ModeSpec) -> Tuple[Optional[calc.Inputs], Dict[str, object]]:
    sb = st.sidebar
    sb.markdown(f"### / {spec.name} PARAMS")
    preset = sb.radio(
        "PRESET",
        list(spec.presets),
        horizontal=True,
        key=_k(spec, "preset"),
        on_change=apply_preset,
        args=(spec,),
        persist_state=PERSIST,
        help="Loads a market into the inputs below (DEEP ITM / FAR OTM are for a call).",
    )
    S = _num("SPOT.S", spec, "spot", min_value=0.0, step=spec.price_step, format=spec.price_format)
    K = _num("STRIKE.K", spec, "strike", min_value=0.0, step=spec.price_step, format=spec.price_format)
    T = _num("MATURITY.T (YEARS)", spec, "T", min_value=0.0, max_value=50.0, step=0.25, format="%.4f")
    r = _num(spec.r_label, spec, "r", min_value=-0.5, max_value=0.5, step=0.0025, format="%.4f")
    q = _num(spec.q_label, spec, "q", min_value=-0.5, max_value=0.5, step=0.0025, format="%.4f")
    sigma = _num("VOL.σ (DECIMAL)", spec, "vol", min_value=0.0, max_value=VOL_MAX, step=0.01, format="%.4f")
    kind = sb.selectbox("TYPE", ["CALL", "PUT"], key=_k(spec, "type"), persist_state=PERSIST)
    ccy = None
    if spec.name == "EQUITY":
        ccy = sb.selectbox(
            "PREMIUM CCY",
            CURRENCIES,
            key=_k(spec, "ccy"),
            persist_state=PERSIST,
            help="Currency of the strike and premium; 'OTHER' shows no currency.",
        )
    position = _num("POSITION (UNITS, SIGNED)", spec, "position", step=1.0, format="%.0f", help=spec.position_help)
    edited = [f for f in PRESET_FIELDS if st.session_state.get(_k(spec, f)) != spec.presets[preset][f]]
    if edited:
        sb.caption(f"Preset {preset} edited: {', '.join(edited)} changed.")
    extras = {"ccy": ccy, "position": float(position or 0.0)}
    errors = calc.input_errors(S, K, T, r, q, sigma)
    if errors:
        for msg in errors:
            st.error(msg)
        return None, extras
    return calc.Inputs(float(S), float(K), float(T), float(r), float(q), float(sigma), kind == "CALL"), extras


def _ccy_label(spec: ModeSpec, ccy: Optional[str]) -> Tuple[str, str]:
    """(currency suffix for the premium, unit line)."""

    suffix = "" if ccy in (None, "OTHER") else ccy
    return suffix, f"{suffix or 'price units'} per {spec.underlying}"


def _hero(spec: ModeSpec, inp: calc.Inputs, g: BSGreeks, ccy: Optional[str]) -> None:
    suffix, unit = _ccy_label(spec, ccy)
    F = inp.forward
    std = inp.sigma * math.sqrt(inp.T)
    sd_text = f"{math.log(inp.K / F) / std:+.2f} SD" if std > 0 else "n/a (σ√T = 0)"
    dec = 2
    cells = [
        ("SPOT", calc.fmt_exact(inp.S, dec)),
        ("FORWARD", calc.fmt_fixed(F, dec)),
        ("STRIKE", calc.fmt_exact(inp.K, dec)),
        ("K / F", f"{inp.K / F:.4f}"),
        ("ln(K/F) / σ√T", sd_text),
        ("TENOR", calc.tenor_text(inp.T)),
        ("DELTA", calc.fmt_sig(float(g.delta))),
    ]
    grid = "".join(f"<div class='dx-cell'><span>{html.escape(a)}</span><b>{html.escape(b)}</b></div>" for a, b in cells)
    price = calc.fmt_fixed(float(g.price), spec.price_decimals)
    tag = f"{'CALL' if inp.call else 'PUT'} / PREMIUM"
    st.markdown(
        f"""<div class='dx-hero'><div class='dx-hero-tag'>{tag}</div>
        <div class='dx-hero-value'>{price}<span class='dx-hero-ccy'> {html.escape(suffix)}</span></div>
        <div class='dx-hero-unit'>{html.escape(unit)} · long, Black-Scholes-Merton, F = S·e<sup>(r−q)T</sup></div>
        <div class='dx-hero-grid'>{grid}</div></div>""",
        unsafe_allow_html=True,
    )


def _greek_card(name: str, value: str, unit: str, sub: str = "") -> str:
    sub_html = f"<div class='dx-greek-sub'>{html.escape(sub)}</div>" if sub else ""
    return (
        f"<div class='dx-greek'><div class='dx-greek-name'>{html.escape(name)}</div>"
        f"<div class='dx-greek-value'>{html.escape(value)}</div>"
        f"<div class='dx-greek-unit'>{html.escape(unit)}</div>{sub_html}</div>"
    )


def _greeks_panel(spec: ModeSpec, inp: calc.Inputs, g: BSGreeks, ccy: Optional[str]) -> None:
    suffix, _ = _ccy_label(spec, ccy)
    money = suffix or "price units"
    d = {k: float(v) for k, v in desk_greeks(g, inp.S).items()}
    theta_1d = float(bs_theta_1d(inp.S, inp.K, inp.T, inp.r, inp.sigma, q=inp.q, call=inp.call))
    f = calc.fmt_sig
    first = [
        _greek_card("DELTA Δ", f(d["delta"]), "per 1.00 of spot", f"{f(d['delta_1pct'])} {money} per +1% spot"),
        _greek_card("GAMMA Γ", f(d["gamma_1pct"]), "Δ change per +1% spot", f"{f(d['gamma'])} per 1.00 of spot"),
        _greek_card("VEGA ν", f(d["vega_1vol"]), f"{money} per +1 vol pt"),
        _greek_card("THETA Θ", f(d["theta_day"]), f"{money} per calendar day", f"1-day repriced: {f(theta_1d)}"),
        _greek_card("RHO (RATE.R)", f(d["rho_1bp"]), f"{money} per +1bp of r", f"{f(d['rho_1pct'])} per +1%"),
        _greek_card(spec.q_greek, f(d["phi_1bp"]), f"{money} per +1bp of q"),
    ]
    second = [
        _greek_card("VANNA", f(d["vanna_1vol"]), "Δ change per +1 vol pt"),
        _greek_card("VOLGA", f(d["volga_1vol"]), "vega change per +1 vol pt"),
        _greek_card("CHARM", f(d["charm_day"]), "Δ change per calendar day"),
    ]
    st.markdown("### / GREEKS")
    st.caption(
        f"Long one option on {spec.underlying}; each value is the P&L (or change in delta) for the stated move. "
        "Theta is analytic dV/dt / 365; the 1-day value reprices one calendar day later (they differ near expiry)."
    )
    st.markdown(f"<div class='dx-greeks'>{''.join(first)}</div>", unsafe_allow_html=True)
    st.markdown("#### / SECOND ORDER")
    st.markdown(f"<div class='dx-greeks'>{''.join(second)}</div>", unsafe_allow_html=True)


def _iv_solver(spec: ModeSpec, inp: calc.Inputs, ccy: Optional[str]) -> None:
    st.markdown("### / IMPLIED VOL")
    lo, hi = calc.price_bounds(inp)
    suffix, _ = _ccy_label(spec, ccy)
    cols = st.columns([2, 3])
    with cols[0]:
        premium = st.number_input(
            f"MARKET PREMIUM ({suffix or 'price units'})",
            key=_k(spec, "iv_premium"),
            value=None,
            min_value=0.0,
            format="%.6f",
            placeholder="enter a premium",
            persist_state=PERSIST,
            help="Premium of one option on one unit of underlying, same S, K, T, r, q and type as the inputs.",
        )
    bounds = (
        f"No-arbitrage bounds: {calc.fmt_sig(lo)} ≤ premium < {calc.fmt_sig(hi)} "
        f"({'max(S·e^−qT − K·e^−rT, 0) and S·e^−qT' if inp.call else 'max(K·e^−rT − S·e^−qT, 0) and K·e^−rT'})."
    )
    with cols[1]:
        if premium is None:
            st.caption("Enter a premium to solve for the Black-Scholes vol. " + bounds)
            return
        res = implied_vol_bs(float(premium), inp.S, inp.K, inp.T, inp.r, q=inp.q, is_call=inp.call)
        status = str(res.status)
        if status == "ok":
            iv = float(res.vol)
            st.markdown(f"**IMPLIED VOL {iv:.4%}** ({iv:.6f} decimal) · {bounds}")
            if iv <= VOL_MAX:
                st.button("PRICE AT THIS VOL", key=_k(spec, "use_iv"), on_click=_set_vol, args=(spec, iv))
            else:
                st.caption(f"Above the VOL.σ input range (max {VOL_MAX:g} = {VOL_MAX:.0%}).")
        elif status == "below_intrinsic":
            st.error(
                f"Premium {calc.fmt_sig(premium)} is below the lower bound {calc.fmt_sig(lo)}: "
                f"no vol reproduces it. {bounds}"
            )
        elif status == "above_upper_bound":
            st.error(
                f"Premium {calc.fmt_sig(premium)} is at or above the upper bound {calc.fmt_sig(hi)}: "
                f"no vol reproduces it. {bounds}"
            )
        else:
            st.warning(f"No implied vol: {STATUS.get(status, status)}. {bounds}")


def _scenario(spec: ModeSpec, inp: calc.Inputs, position: float, ccy: Optional[str]) -> None:
    st.markdown("### / SCENARIO P&L")
    suffix, _ = _ccy_label(spec, ccy)
    c1, c2, c3 = st.columns(3)
    max_spot = _num("MAX SPOT SHOCK (%)", spec, "scn_spot", container=c1, min_value=0.5, max_value=90.0, step=0.5)
    max_vol = _num("MAX VOL SHOCK (VOL PTS)", spec, "scn_vol", container=c2, min_value=0.5, max_value=100.0, step=0.5)
    days = _num("HORIZON (CALENDAR DAYS)", spec, "scn_days", container=c3, min_value=0.0, max_value=3650.0, step=1.0)
    grid = calc.scenario_grid(inp, position, float(max_spot) / 100.0, float(max_vol) / 100.0, float(days))
    table = grid.table()
    x_labels, y_labels = grid.spot_labels, grid.vol_labels
    fig = go.Figure(
        go.Heatmap(
            z=grid.pnl,
            x=x_labels,
            y=y_labels,
            text=[[calc.fmt_compact(v) for v in row] for row in grid.pnl],
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=10, color=FOREGROUND),
            colorscale=PNL_COLORS,
            zmid=0.0,
            colorbar=dict(
                title=dict(text=f"P&L {suffix}".strip(), side="right"),
                orientation="h",
                thickness=10,
                y=-0.32,
                yanchor="top",
            ),
            hovertemplate="spot %{x}<br>vol %{y} pts<br>P&L %{z:,.4f}<extra></extra>",
        )
    )
    dark_layout(fig, height=400, xtitle="SPOT SHOCK (relative)", ytitle="VOL SHOCK (VOL PTS)", legend=False)
    fig.update_layout(hovermode="closest", meta={"chart": "scenario"}, margin=dict(l=60, r=10, t=20, b=110))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    n_x, n_y = len(x_labels), len(y_labels)
    bound_axes(
        fig,
        xaxis=axis_limits([-0.5, n_x - 0.5], pad=0.0, min_span_frac=1.0 / n_x),
        yaxis=axis_limits([-0.5, n_y - 0.5], pad=0.0, min_span_frac=1.0 / n_y),
    )
    st.plotly_chart(fig, width="stretch", key=_k(spec, "scenario_chart"), config=PLOTLY_2D_CONFIG)
    st.caption(
        f"Full revaluation P&L in {suffix or 'price units'} of {calc.fmt_sig(position)} units "
        f"({'long' if position >= 0 else 'short'}), "
        f"spot shocks relative, vol shocks in vol points, {grid.horizon_days:g} calendar days later "
        f"(base premium {calc.fmt_sig(grid.base_value)}). Cell labels are rounded to 2 significant figures; "
        "hover or open the table for exact values."
        + (
            f" Vol shocks below {(-inp.sigma * 100) or 0.0:.2f} pts are omitted (negative vol)."
            if grid.dropped_vol_shocks
            else ""
        )
        + (
            " The horizon reaches the expiry, so every cell is valued at intrinsic and the vol shocks do not matter."
            if grid.horizon_days >= inp.T * DAYS_PER_YEAR
            else ""
        )
    )
    with st.expander("SCENARIO GRID AS A TABLE"):
        st.dataframe(
            table,
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%,.4f") for c in table.columns},
        )
    st.download_button(
        "Download scenario grid CSV",
        table.to_csv().encode("utf-8"),
        file_name=f"{spec.prefix}_scenario_pnl.csv",
        key=_k(spec, "scenario_csv"),
    )


def _surface(spec: ModeSpec, inp: calc.Inputs) -> None:
    st.markdown("### / RISK SURFACE")
    c1, c2 = st.columns(2)
    axes = c1.radio("AXES", calc.SURFACE_AXES, horizontal=True, key=_k(spec, "surf_axes"), persist_state=PERSIST)
    z_label = c2.selectbox("SHOW", tuple(calc.SURFACE_Z), key=_k(spec, "surf_z"), persist_state=PERSIST)
    c3, c4 = st.columns(2)
    n = c3.slider("RESOLUTION", 15, 60, key=_k(spec, "surf_res"), persist_state=PERSIST)
    n_sd = c4.slider(
        "SPOT WIDTH (SD)",
        1.0,
        5.0,
        step=0.5,
        key=_k(spec, "surf_sd"),
        persist_state=PERSIST,
        help="Spot axis: spot and strike plus this many standard deviations σ√T of log-spot either side.",
    )
    if axes == calc.SURFACE_AXES[1] and inp.T <= 0:
        st.info("The option has expired (T = 0): showing SPOT × VOL, where the value is the intrinsic value.")
        axes = calc.SURFACE_AXES[0]
    grid = calc.surface_grid(inp, axes, z_label, n=int(n), n_sd=float(n_sd))
    kind = "CALL" if inp.call else "PUT"
    y_title = "VOL (decimal)" if axes == calc.SURFACE_AXES[0] else "TIME TO EXPIRY (YEARS)"
    y_axis_title = "VOL (decimal)" if axes == calc.SURFACE_AXES[0] else "T (YEARS)"  # short, to fit a phone
    z_title = f"{kind} {z_label}"
    fig = go.Figure(
        data=[
            go.Surface(
                z=grid.z,
                x=grid.x,
                y=grid.y,
                colorscale=SURFACE_COLORS,
                hovertemplate=f"spot %{{x:.4f}}<br>{y_title} %{{y:.4f}}<br>{z_title} %{{z:.6g}}<extra></extra>",
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color=FOREGROUND, family="IBM Plex Mono", size=10),
                    title=dict(text=z_label, font=dict(color=FOREGROUND, size=12), side="right"),
                    orientation="h",
                    len=0.8,
                    thickness=10,
                    y=0.0,
                    yanchor="top",
                ),
                contours={"z": {"show": True, "highlightcolor": FOREGROUND, "project": {"z": True}, "width": 2}},
                lighting=dict(ambient=0.9, diffuse=0.7, specular=0.3, roughness=0.5),
            ),
            go.Scatter3d(
                x=[grid.point[0]],
                y=[grid.point[1]],
                z=[grid.point[2]],
                mode="markers",
                marker=dict(size=6, color=ACCENT, line=dict(width=2, color=FOREGROUND)),
                name="CURRENT",
                hovertemplate=f"current: spot %{{x:.4f}}<br>{y_title} %{{y:.4f}}<br>{z_title} %{{z:.6g}}"
                "<extra></extra>",
            ),
        ]
    )
    z_lo, z_hi = float(np.min(grid.z)), float(np.max(grid.z))
    pad = max((z_hi - z_lo) * 0.1, 1e-9 * max(abs(z_hi), 1.0))
    axis = dict(backgroundcolor="#000000", gridcolor="#333333", showbackground=True)
    scene = dict(
        bgcolor="#000000",
        dragmode="turntable",
        xaxis=dict(title=dict(text="SPOT", font=dict(color=FOREGROUND)), range=[grid.x[0], grid.x[-1]], **axis),
        yaxis=dict(title=dict(text=y_axis_title, font=dict(color=FOREGROUND)), range=[grid.y[0], grid.y[-1]], **axis),
        zaxis=dict(title=dict(text=z_title, font=dict(color=FOREGROUND)), range=[z_lo - pad, z_hi + pad], **axis),
    )
    # The camera is always the default view; Plotly keeps the user's rotation
    # while uirevision is unchanged and resets it for a new preset or view.
    scene["camera"] = {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": DEFAULT_CAMERA["eye"],
        "projection": {"type": "orthographic"},
    }
    revision = f"{spec.prefix}_{axes}_{z_label}_{st.session_state.get(_k(spec, 'camera_epoch'), 0)}"
    fig.update_layout(
        scene=scene,
        paper_bgcolor="#000000",
        height=SURFACE_HEIGHT,  # what the chart falls back to without the container's CSS
        margin=SURFACE_MARGIN,
        font=dict(color=FOREGROUND, family="Helvetica"),
        showlegend=False,
        uirevision=revision,
        meta={"chart": "surface"},
    )
    fig.update_scenes(camera_projection_type="orthographic")
    bound_scene(fig, eye=DEFAULT_CAMERA["eye"])
    with st.container(key=_k(spec, "surface_box")):  # sized by PAGE_CSS: see SURFACE_MIN_ASPECT
        st.plotly_chart(fig, width="stretch", height="stretch", key=_k(spec, "surface_3d"), config=PLOTLY_3D_CONFIG)
    unit = "premium in price units" if z_label == "VALUE" else grid.z_unit
    st.caption(f"{z_title} ({unit}) for one option on one unit; the red marker is the current input.")
    if GUARD_AVAILABLE:
        st.caption(ZOOM_HINT)


@st.cache_data(show_spinner=False, max_entries=8)
def _market_calibration(chain: pd.DataFrame, rate: float, discount: Optional[Discounting] = None) -> ChainCalibration:
    """SVI fits of every slice of a loaded chain (cached on the chain's
    content, the rate and the discounting; no per-session nonce). The
    ANALYTICS lab publishes its chain with its discounting (the Treasury
    curve of the trade date, or put-call parity per expiry), which then
    sets each expiry's discount factor; ``rate`` is used otherwise."""

    return calibrate_chain(chain, rate=rate, cv_folds=0, enforce_arbitrage=True, discount=discount)


def _market_chain() -> Optional[pd.DataFrame]:
    chain = st.session_state.get("market_chain")
    if isinstance(chain, pd.DataFrame) and not chain.empty and {"strike", "expiry"} <= set(chain.columns):
        return chain
    return None


def _smile_figure(name: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(meta={"chart": name})
    return fig


def _finish_smile(spec: ModeSpec, fig: go.Figure, xs, ys) -> None:
    dark_layout(fig, height=400, xtitle="STRIKE", ytitle="IMPLIED VOL (decimal)")
    fig.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        margin=dict(l=60, r=20, t=30, b=60),  # the legend sits below the STRIKE title (theme.LEGEND_BELOW)
    )
    bound_axes(fig, xaxis=axis_limits(xs, floor=0.0), yaxis=axis_limits(ys, pad=0.1, floor=0.0))
    st.plotly_chart(fig, width="stretch", key=_k(spec, "smile_2d"), config=PLOTLY_2D_CONFIG)
    if GUARD_AVAILABLE:
        st.caption(ZOOM_HINT)


def _params_table(params: Dict[str, float], fixed=()) -> None:
    rows = [
        {"parameter": k + (" (fixed)" if k in fixed else ""), "value": calc.fmt_sig(v, 6)} for k, v in params.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _illustrative_smile(spec: ModeSpec, inp: calc.Inputs, reason: str = "") -> None:
    c1, c2, c3 = st.columns(3)
    if spec.smile_model == "SVI":
        rho = c1.slider("SKEW ρ (SSVI)", -0.95, 0.95, step=0.05, key=_k(spec, "smile_rho"), persist_state=PERSIST)
        curv = c2.slider(
            "CURVATURE η (SSVI)",
            0.05,
            3.0,
            step=0.05,
            key=_k(spec, "smile_curv"),
            persist_state=PERSIST,
            help="Arbitrage free when η (1 + |ρ|) ≤ 2 (Gatheral-Jacquier).",
        )
    else:
        rho = c1.slider("SABR ρ", -0.95, 0.95, step=0.05, key=_k(spec, "smile_rho"), persist_state=PERSIST)
        curv = c2.slider("SABR ν (VOL OF VOL)", 0.05, 3.0, step=0.05, key=_k(spec, "smile_curv"), persist_state=PERSIST)
    n_sd = c3.slider("WIDTH (SD)", 1.0, 5.0, step=0.5, key=_k(spec, "smile_sd"), persist_state=PERSIST)
    if inp.T <= 0 or inp.sigma <= 0:
        st.info("No smile: it needs T > 0 and VOL.σ > 0 (total variance σ²T is 0 at expiry or zero vol).")
        return
    F = inp.forward
    try:
        if spec.smile_model == "SVI":
            smile = calc.illustrative_svi(F, inp.T, inp.sigma, float(rho), float(curv), float(n_sd))
        else:
            smile = calc.illustrative_sabr(F, inp.T, inp.sigma, float(rho), float(curv), n_sd=float(n_sd))
    except ValueError as exc:
        st.warning(f"No smile for these parameters: {exc}")
        return
    st.markdown(
        f"<span class='dx-badge' style='color:{SERIES[3]}'>ILLUSTRATIVE</span>"
        f"{smile.model} model smile from your parameters, not market data{html.escape(reason)}.",
        unsafe_allow_html=True,
    )
    fig = _smile_figure("smile")
    fig.add_trace(
        go.Scatter(
            x=smile.strikes,
            y=smile.vols,
            mode="lines",
            name=f"{smile.model} (illustrative)",
            line=dict(color=ACCENT, width=3),
        )
    )
    vol_k = smile.vol_at(inp.K) if smile.strikes[0] <= inp.K <= smile.strikes[-1] else None
    if vol_k is not None and not (np.isfinite(vol_k) and vol_k > 0):
        vol_k = None  # no usable model vol at K (never expected inside the plotted range)
    fig.add_trace(
        go.Scatter(
            x=[F],
            y=[inp.sigma],
            mode="markers",
            name="ATM-F (VOL.σ)",
            marker=dict(size=11, color=FOREGROUND, symbol="x"),
        )
    )
    if vol_k is not None:
        fig.add_trace(
            go.Scatter(
                x=[inp.K], y=[vol_k], mode="markers", name="K", marker=dict(size=10, color=SERIES[2], symbol="diamond")
            )
        )
    _finish_smile(spec, fig, smile.strikes, np.r_[smile.vols, inp.sigma])
    parts = [
        f"Strikes F·exp(±{n_sd:g}·σ√T) around the forward F = {calc.fmt_sig(F, 6)}; VOL.σ is the ATM-forward vol.",
    ]
    if vol_k is not None:
        p_k = float(bs_greeks(inp.S, inp.K, inp.T, inp.r, vol_k, q=inp.q, call=inp.call).price)
        p_flat = float(bs_greeks(inp.S, inp.K, inp.T, inp.r, inp.sigma, q=inp.q, call=inp.call).price)
        parts.append(
            f"At K the smile vol is {vol_k:.2%} (premium {calc.fmt_sig(p_k)} vs {calc.fmt_sig(p_flat)} "
            "at the flat VOL.σ)."
        )
    else:
        parts.append("K is outside the plotted range.")
    st.caption(" ".join(parts))
    for note in smile.notes:
        st.warning(note)
    st.markdown(
        f"**[ {smile.model} PARAMETERS ]** "
        + ("raw SVI of the SSVI slice, total variance for T" if smile.model == "SVI" else "α solved from the ATM vol")
    )
    _params_table(smile.params, smile.fixed)


def _market_smile(spec: ModeSpec, inp: calc.Inputs, chain: pd.DataFrame) -> None:
    meta = st.session_state.get("market_chain_meta") or {}
    ticker = str(meta.get("ticker") or (chain["ticker"].iloc[0] if "ticker" in chain.columns else "unknown ticker"))
    try:
        discount = st.session_state.get("market_chain_discount")
        discount = discount if isinstance(discount, Discounting) else None
        result = _market_calibration(chain, float(inp.r), discount)
    except ValueError as exc:
        _illustrative_smile(spec, inp, reason=f" (the loaded chain could not be fitted: {exc})")
        return
    if not result.slices:
        reasons = "; ".join(f"{s.expiry.date()}: {s.reason}" for s in result.skipped[:3]) or "no usable slice"
        _illustrative_smile(spec, inp, reason=f" (no fittable market slice: {reasons})")
        return
    labels = [f"{s.expiry.date()} ({s.clean.days}d, trade {s.trade_date.date()})" for s in result.slices]
    one_month = int(np.argmin([abs(s.clean.days - 30) for s in result.slices]))  # default: nearest to 1M
    choice = st.selectbox("EXPIRY", labels, index=one_month, key=_k(spec, "smile_expiry"), persist_state=PERSIST)
    sl = result.slices[labels.index(choice)]
    fit, clean = sl.fit, sl.clean
    atm = float(svi_iv(sl.forward, sl.params, F=sl.forward, T=sl.tau))
    n_sd = 3.0
    grid = calc.smile_strikes(sl.forward, atm, sl.tau, n_sd)
    grid = np.unique(np.r_[grid, np.linspace(sl.strikes.min(), sl.strikes.max(), 61)])
    model = np.asarray(svi_iv(grid, sl.params, F=sl.forward, T=sl.tau), dtype=float)
    source = meta.get("source") or meta.get("status") or "loaded chain"
    st.markdown(
        f"<span class='dx-badge' style='color:{SERIES[2]}'>MARKET</span>"
        f"{html.escape(ticker)} · trade date {sl.trade_date.date()} · expiry {sl.expiry.date()} · "
        f"source: {html.escape(str(source))}",
        unsafe_allow_html=True,
    )
    fig = _smile_figure("smile")
    fig.add_trace(
        go.Scatter(
            x=sl.strikes,
            y=sl.observed_ivs,
            mode="markers",
            name=f"MARKET {ticker} {sl.expiry.date()}",
            marker=dict(size=7, color=FOREGROUND, line=dict(width=1, color="#000000")),
        )
    )
    fig.add_trace(go.Scatter(x=grid, y=model, mode="lines", name="SVI fit", line=dict(color=ACCENT, width=3)))
    fig.add_trace(
        go.Scatter(
            x=[sl.forward], y=[atm], mode="markers", name="ATM-F", marker=dict(size=11, color=SERIES[2], symbol="x")
        )
    )
    _finish_smile(spec, fig, grid, np.r_[model, sl.observed_ivs])
    d = sl.diagnostics
    vols = "re-inverted from bid/ask mids" if clean.iv_source == "mid" else "as quoted"
    st.caption(
        f"T = {sl.tau:.4f}y ({clean.days} days ACT/365F), spot {calc.fmt_sig(clean.spot, 6)}, forward "
        f"{calc.fmt_sig(sl.forward, 6)} ({d.get('forward_source')}), vols {vols} "
        + (
            f"discounted as on the ANALYTICS lab: {d.get('discount_method')} DF {clean.discount_factor:.6f} = "
            f"{clean.rate:.3%} cc ({d.get('discount_source')}). "
            if clean.discount_method != "rate"
            else f"with r = RATE.R = {inp.r:.2%}. "
        )
        + f"SVI fit RMSE {fit.rmse_bp:.1f} vol bp (max {fit.max_abs_error_bp:.1f}) over "
        f"{fit.n_points} OTM quotes; butterfly {'ok' if fit.butterfly.ok else 'FAILS'}, Lee wings "
        f"{'ok' if fit.lee['lee_ok'] else 'FAIL'}, arbitrage repair: {fit.arbitrage_repair}. The pricer inputs above "
        "are separate from this chain."
    )
    at_bound = [k for k, v in fit.at_bound.items() if v]
    if at_bound:
        st.warning(
            f"Fitted SVI parameters at their bounds: {', '.join(at_bound)}. The fit is constrained there; "
            "read them with care."
        )
    st.markdown("**[ SVI PARAMETERS ]** raw SVI, total variance for T")
    _params_table({f"SVI {k}": float(v) for k, v in sl.params.items()})


def _smile(spec: ModeSpec, inp: calc.Inputs) -> None:
    st.markdown(f"### / {spec.smile_model} SMILE")
    chain = _market_chain()
    if chain is not None:
        src = st.radio(
            "SMILE SOURCE",
            ("ILLUSTRATIVE", "MARKET"),
            horizontal=True,
            key=_k(spec, "smile_src"),
            persist_state=PERSIST,
        )
    else:
        src = "ILLUSTRATIVE"
        st.caption("Load an option chain in the ANALYTICS lab to see a MARKET smile here.")
    if src == "MARKET" and chain is not None:
        _market_smile(spec, inp, chain)
    else:
        _illustrative_smile(spec, inp)


def _model_check(inp: calc.Inputs, g: BSGreeks) -> None:
    with st.expander("MODEL CHECKS: analytic Greeks vs finite differences, no-arbitrage bounds"):
        lo, hi = calc.price_bounds(inp)
        price = float(g.price)
        inside = lo - 1e-12 * max(1.0, hi) <= price <= hi
        st.markdown(
            f"Price {calc.fmt_sig(price, 8)} within [{calc.fmt_sig(lo, 8)}, {calc.fmt_sig(hi, 8)}]: "
            f"{'[OK]' if inside else '[!]'}"
        )
        if inp.T <= 0 or inp.sigma <= 0:
            st.caption("Finite differences skipped: at T = 0 or σ = 0 the price has a kink at the forward.")
            return
        table = calc.model_check(inp, g)
        status = "[OK]" if bool(table["ok"].all()) else "[!]"
        st.markdown(f"Analytic Greeks vs central differences of the pricer {status} (raw units: per 1.00, per year)")
        shown = {c: table[c].map(lambda v: f"{v:.10g}") for c in ("analytic", "finite difference")}
        shown.update({c: table[c].map(lambda e: f"{e:.3e}") for c in ("FD error estimate", "|difference|")})
        st.dataframe(table.assign(**shown), hide_index=True, width="stretch")
        st.caption(
            "Each finite difference is Richardson-extrapolated over halving steps scaled to σ√T, with an error "
            "estimate (truncation plus rounding). ok: |analytic − FD| ≤ 2 × that estimate. A large estimate means "
            "the difference cannot resolve that Greek here (e.g. a gamma of 1e-13 deep in the money)."
        )


def render(mode: str) -> None:
    """Draw the pricer page for ``mode`` (a key of :data:`MODES`: EQUITY)."""

    spec = MODES[mode.upper()]
    _init_state(spec)
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(f"## / {spec.name} OPTIONS")
    inp, extras = _sidebar_inputs(spec)
    if inp is None:
        st.info("Fix the inputs in the sidebar to price the option.")
        return
    for note in calc.input_warnings(inp.T, inp.r, inp.q, inp.sigma):
        st.warning(note)
    try:
        g = bs_greeks(inp.S, inp.K, inp.T, inp.r, inp.sigma, q=inp.q, call=inp.call)
    except ValueError as exc:
        st.error(f"Cannot price these inputs: {exc}")
        return
    ccy = extras["ccy"]
    _hero(spec, inp, g, ccy)
    _greeks_panel(spec, inp, g, ccy)
    _iv_solver(spec, inp, ccy)
    _scenario(spec, inp, float(extras["position"]), ccy)
    _surface(spec, inp)
    _smile(spec, inp)
    _model_check(inp, g)
