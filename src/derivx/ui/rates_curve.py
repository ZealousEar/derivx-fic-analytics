"""CURVE tab: build the swap curve from editable quotes and show the
Treasury curve. Returns the Market every other Rates tab prices off."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.rates.curves import CONVENTIONS, INTERPOLATIONS, Curve, CurveQuote, Market, bootstrap
from derivx.rates.examples import sofr_example_rows, ust_par_history, ust_par_rows
from derivx.rates.quotes import clean_quotes
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SERIES, dark_layout

SWAP_CURVES = ["USD SOFR", "EUR ESTR", "GBP SONIA"]
INTERP_LABELS = {
    "loglinear": "Log-linear DF (flat forwards)",
    "linear_zero": "Linear zero rates",
    "monotone_convex": "Monotone convex (Hagan-West)",
}
SWAP_KINDS = ["DEPO", "OIS", "IRS"]


@st.cache_data(show_spinner=False)
def _build(asof: dt.date, quotes: Tuple[Tuple[str, str, float], ...], conventions: str, interpolation: str) -> Curve:
    return bootstrap(asof, [CurveQuote(*q) for q in quotes], conventions, interpolation)


@st.cache_data(show_spinner=False)
def _treasury(day: dt.date, interpolation: str) -> Curve:
    _, rows = ust_par_rows(day)
    quotes, _, _ = clean_quotes(rows)
    return bootstrap(day, quotes, "USD UST", interpolation)


def _default_table() -> pd.DataFrame:
    rows = sofr_example_rows()
    return pd.DataFrame(
        {
            "instrument": [r["kind"] for r in rows],
            "tenor": [r["tenor"] for r in rows],
            "rate": [round(100 * r["rate"], 4) for r in rows],
        }
    )


def _curve_grid(curve: Curve) -> pd.DataFrame:
    horizon = float(curve.ts[-1])
    days = np.unique(np.concatenate([np.arange(1, 731, 7), np.arange(731, int(horizon * 365) + 1, 30)]))
    dates = [curve.asof + dt.timedelta(days=int(d)) for d in days]
    fwd_3m = [curve.forward_rate(d, d + dt.timedelta(days=91), "ACT/360") for d in dates]
    return pd.DataFrame(
        {
            "date": dates,
            "years": days / 365.0,
            "df": curve.df(dates),
            "zero_cont": [curve.zero_rate(d) for d in dates],
            "fwd_3m": fwd_3m,
        }
    )


def _curve_chart(curve: Curve, grid: pd.DataFrame, title_zero: str) -> go.Figure:
    pillars = np.array([inst.tenor.years for inst in curve.instruments])
    quotes = np.array([inst.quote.rate for inst in curve.instruments]) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid["years"],
            y=grid["zero_cont"] * 100,
            mode="lines",
            name=title_zero,
            line=dict(color=SERIES[0], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid["years"], y=grid["fwd_3m"] * 100, mode="lines", name="3M FORWARD", line=dict(color=ACCENT, width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pillars,
            y=quotes,
            mode="markers",
            name="QUOTES",
            marker=dict(color=SERIES[2], size=8, line=dict(color="#000000", width=1)),
        )
    )
    dark_layout(fig, height=380, xtitle="TENOR (YEARS)", ytitle="RATE (%)")
    values = np.concatenate([grid["zero_cont"] * 100, grid["fwd_3m"] * 100, quotes])
    return bound_axes(fig, xaxis=axis_limits(grid["years"], pad=0.02, floor=0.0), yaxis=axis_limits(values, pad=0.15))


def _repricing_table(curve: Curve) -> pd.DataFrame:
    rows = curve.repricing()
    return pd.DataFrame(
        {
            "Tenor": [r["tenor"] for r in rows],
            "Instrument": [r["kind"] for r in rows],
            "Maturity": [r["maturity"] for r in rows],
            "Quote (%)": [100 * r["quote"] for r in rows],
            "Curve (%)": [100 * r["model"] for r in rows],
            "Error (bp)": [r["error_bp"] for r in rows],
        }
    )


def _show_issues(issues) -> bool:
    """Show validation messages; True if the curve can be built."""

    for issue in issues:
        if issue.level == "error":
            st.error(issue.message)
        elif issue.level == "warning":
            st.warning(issue.message)
        else:
            st.caption(issue.message)
    return not any(i.level == "error" for i in issues)


def render() -> Optional[Market]:
    """Draw the CURVE tab; the Market for the other tabs, or None if the
    quotes cannot be built into a curve (the reason is shown)."""

    history = ust_par_history()
    dates: List[dt.date] = list(history.index)
    st.sidebar.markdown("### / RATES MARKET")
    asof = st.sidebar.selectbox(
        "VALUATION DATE",
        dates[::-1],
        key="rates_asof",
        help="Dates covered by the shipped US Treasury par curve snapshot.",
    )
    conv = st.sidebar.selectbox("SWAP CURVE", SWAP_CURVES, key="rates_conv")
    interpolation = st.sidebar.selectbox(
        "INTERPOLATION", INTERPOLATIONS, format_func=INTERP_LABELS.get, key="rates_interp"
    )
    c = CONVENTIONS[conv]
    st.sidebar.caption(
        f"{conv}: spot T+{c.spot_lag}, {c.calendar} calendar, OIS annual {c.ois_day_count}, "
        f"payment lag {c.payment_lag}d; IRS fixed {c.irs_fixed_months}M {c.irs_fixed_day_count}."
    )

    st.markdown("### / SWAP CURVE")
    st.caption(
        "Illustrative quotes: there is no free feed of SOFR/€STR/SONIA swap rates, so edit, paste or upload your "
        "own (instrument DEPO, OIS or IRS; tenor like ON, 1W, 3M, 10Y; rate in %, decimals or bp)."
    )
    upload = st.file_uploader("Upload quotes (CSV with instrument, tenor, rate)", type=["csv"], key="rates_upload")
    table = _default_table()
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            raw.columns = [str(col).strip().lower() for col in raw.columns]
            if "kind" in raw.columns and "instrument" not in raw.columns:
                raw = raw.rename(columns={"kind": "instrument"})
            missing = {"instrument", "tenor", "rate"} - set(raw.columns)
            if missing:
                st.error(f"Uploaded file is missing columns: {', '.join(sorted(missing))}")
            else:
                table = raw[["instrument", "tenor", "rate"]]
        except Exception as exc:  # malformed CSV
            st.error(f"Could not read the uploaded file: {exc}")
    edited = st.data_editor(
        table,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key=f"rates_quotes_{upload.name if upload else 'default'}",
        column_config={
            "instrument": st.column_config.SelectboxColumn("INSTRUMENT", options=SWAP_KINDS, required=True),
            "tenor": st.column_config.TextColumn("TENOR", required=True),
            "rate": st.column_config.NumberColumn("RATE", format="%.4f"),
        },
    )
    units = st.radio("Rate units", ["auto", "percent", "decimal", "bp"], horizontal=True, key="rates_units")
    quotes, issues, used_units = clean_quotes(edited.to_dict("records"), units=units)
    if not _show_issues(issues):
        return None
    try:
        curve = _build(asof, tuple((q.kind, q.tenor, q.rate) for q in quotes), conv, interpolation)
    except ValueError as exc:
        st.error(f"Curve could not be built: {exc}")
        return None

    repricing = _repricing_table(curve)
    worst = float(repricing["Error (bp)"].abs().max())
    cols = st.columns(4)
    cols[0].metric("PILLARS", f"{len(curve.quotes)}")
    cols[1].metric("MAX REPRICING ERROR", f"{worst:.1e} bp")
    cols[2].metric("2Y ZERO", f"{100 * curve.zero_rate(2.0):.3f}%")
    cols[3].metric("10Y ZERO", f"{100 * curve.zero_rate(10.0):.3f}%")
    grid = _curve_grid(curve)
    st.plotly_chart(
        _curve_chart(curve, grid, "ZERO (CONT.)"), width="stretch", config=PLOTLY_2D_CONFIG, key="rates_curve_chart"
    )
    st.caption(
        "Every quote reprices exactly on the bootstrapped curve (error column). Zero rates are continuously "
        "compounded on ACT/365F; 3M forwards are simple ACT/360. Beyond the last pillar forwards are held flat. "
        "One curve both discounts and projects (single-curve)."
    )
    with st.expander("Repricing check and curve grid"):
        st.dataframe(repricing, width="stretch", hide_index=True)
        st.download_button(
            "Download curve grid (CSV)", grid.to_csv(index=False).encode(), "swap_curve.csv", key="rates_grid_download"
        )

    st.markdown("### / TREASURY CURVE")
    treasury = _treasury(asof, interpolation)
    t_grid = _curve_grid(treasury)
    st.plotly_chart(
        _curve_chart(treasury, t_grid, "ZERO (CONT.)"),
        width="stretch",
        config=PLOTLY_2D_CONFIG,
        key="rates_treasury_chart",
    )
    st.caption(
        f"US Treasury daily par yield curve for {asof} (home.treasury.gov): bills up to 1Y as bond-equivalent "
        "yields, 2Y-30Y as semiannual par coupons, settlement T+1. Bonds price off this curve."
    )
    with st.expander("Treasury repricing check"):
        st.dataframe(_repricing_table(treasury), width="stretch", hide_index=True)
    return Market(curve=curve, govt=treasury)
