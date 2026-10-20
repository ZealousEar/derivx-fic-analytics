"""CURVE tab: build the swap curve from editable quotes and show the
Treasury curve, both from the market data the sidebar DATA SOURCE picks.
Returns the Market every other Rates tab prices off."""

from __future__ import annotations

import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.data.rates_history import HistoryStore, dataset_info
from derivx.rates.curves import CONVENTIONS, INTERPOLATIONS, Curve, CurveQuote, Market, bootstrap
from derivx.rates.market_quotes import treasury_quotes
from derivx.rates.quotes import clean_quotes
from derivx.ui import rates_data
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SERIES, dark_layout

SWAP_CURVES = ["USD SOFR", "EUR ESTR", "GBP SONIA"]
INTERP_LABELS = {
    "loglinear": "Log-linear DF (flat forwards)",
    "linear_zero": "Linear zero rates",
    "monotone_convex": "Monotone convex (Hagan-West)",
}
SWAP_KINDS = list(rates_data.SWAP_KINDS)
DATE_WIDGETS = ("bond_settle", "bond_maturity", "bond_dated", "swap_maturity")  # defaults follow the valuation date
PREFILL_CAPTIONS = {
    "real": "Real market data: {label}. The Bank of England publishes a fitted (smoothed) SONIA OIS curve, not "
    "dealer quotes; its spot rates are converted to par OIS rates under GBP SONIA conventions, so bootstrapping "
    "them gives back the published curve at every pillar.",
    "proxy": "No free {ccy} OIS quotes exist, so the table starts from a labelled PROXY: {label}. Government "
    "yields differ from OIS rates by the swap spread; paste or upload your own quotes for a market curve.",
    "illustrative": "No free {ccy} swap quotes exist: the table starts from ILLUSTRATIVE quotes (not market "
    "data){fix}. Edit, paste or upload your own.",
    "upload": "Quotes from your {label}.",
}


@st.cache_data(show_spinner=False)
def _build(asof: dt.date, quotes: Tuple[Tuple[str, str, float], ...], conventions: str, interpolation: str) -> Curve:
    return bootstrap(asof, [CurveQuote(*q) for q in quotes], conventions, interpolation)


def _treasury_quotes(store: HistoryStore, day: dt.date) -> Tuple[Tuple[str, str, float], ...]:
    return tuple((q.kind, q.tenor, q.rate) for q in treasury_quotes(store.curve("UST_PAR", day).to_dict()))


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


def _curve_chart(curve: Curve, grid: pd.DataFrame, title_zero: str, source: str) -> go.Figure:
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
    rates_data.badge(fig, source, curve.asof)
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


def _sidebar(data: rates_data.RatesData) -> Tuple[dt.date, str, str]:
    dates = data.store.dates("UST_PAR")[::-1]
    if not dates:
        raise ValueError("the history store has no US Treasury curve")
    rates_data.sticky("rates_asof", dates)
    asof = st.sidebar.selectbox(
        "VALUATION DATE",
        dates,
        key="rates_asof",
        help=(
            f"Dates with a US Treasury par curve in the history store ({len(dates)} dates, {dates[-1]} to "
            f"{dates[0]}): every business day of the bundled year, month ends before it, and whatever the live "
            "cache or your uploads add."
        ),
    )
    rates_data.remember("rates_asof")
    if st.session_state.get("_rates_asof_seen", asof) != asof:
        for key in DATE_WIDGETS:  # dates chosen relative to the old valuation date start again from their defaults
            st.session_state.pop(key, None)
    st.session_state["_rates_asof_seen"] = asof
    rates_data.sticky("rates_conv", SWAP_CURVES)
    conv = st.sidebar.selectbox("SWAP CURVE", SWAP_CURVES, key="rates_conv")
    rates_data.remember("rates_conv")
    rates_data.sticky("rates_interp", INTERPOLATIONS)
    interpolation = st.sidebar.selectbox(
        "INTERPOLATION", INTERPOLATIONS, format_func=INTERP_LABELS.get, key="rates_interp"
    )
    rates_data.remember("rates_interp")
    c = CONVENTIONS[conv]
    st.sidebar.caption(
        f"{conv}: spot T+{c.spot_lag}, {c.calendar} calendar, OIS annual {c.ois_day_count}, "
        f"payment lag {c.payment_lag}d; IRS fixed {c.irs_fixed_months}M {c.irs_fixed_day_count}."
    )
    return asof, conv, interpolation


def _market_data_panel(store: HistoryStore, asof: dt.date) -> None:
    with st.expander(f"Market data on {asof}: fixings and sources"):
        fixings = store.fixings(asof)
        if fixings.empty:
            st.caption("No overnight fixings in the store on or before this date.")
        else:
            table = pd.DataFrame(
                {
                    "FIXING": [rates_data.FIXING_LABELS.get(f, f) for f in fixings["fixing"]],
                    "RATE DATE": fixings["date"],
                    "RATE (%)": (100 * fixings["rate"]).round(4),
                    "DISTRIBUTOR": fixings["source"],
                }
            )
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption(
                "Overnight fixings in percent (SOFR, EFFR, OBFR, ESTR ACT/360; SONIA ACT/365F), dated by the day "
                "the rate applies to; each is published the next business day."
            )
        coverage = []
        for key in store.datasets():
            span = store.coverage(key)
            if span:
                info = dataset_info(key)
                coverage.append({"DATASET": info.title, "SOURCE": info.source, "FROM": span[0], "TO": span[1],
                                 "DATES": len(store.dates(key))})  # fmt: skip
        st.dataframe(pd.DataFrame(coverage), hide_index=True, width="stretch")


def render() -> Optional[Market]:
    """Draw the CURVE tab; the Market for the other tabs, or None if the
    quotes cannot be built into a curve (the reason is shown)."""

    st.sidebar.markdown("### / RATES MARKET")
    data = rates_data.sidebar()
    asof, conv, interpolation = _sidebar(data)
    rates_data.show_notices(data.notices)

    st.markdown("### / SWAP CURVE")
    if data.mode == rates_data.MODES[2] and data.uploaded is not None:
        prefill = rates_data.uploaded_prefill(data.uploaded)
    else:
        prefill = rates_data.swap_prefill(data.store, conv, asof)
    currency = CONVENTIONS[conv].currency
    fixing_note = "" if "fixing" not in prefill.label else "; the ON quote is the real overnight fixing"
    st.caption(
        PREFILL_CAPTIONS[prefill.kind].format(label=prefill.label, ccy=currency, fix=fixing_note)
        + " Instrument DEPO, OIS or IRS; tenor like ON, 1W, 3M, 10Y; rate in %, decimals or bp."
    )
    for note in prefill.notes:
        st.warning(note)
    edited = st.data_editor(
        prefill.table,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key=f"rates_quotes_{prefill.key}",
        column_config={
            "instrument": st.column_config.SelectboxColumn("INSTRUMENT", options=SWAP_KINDS, required=True),
            "tenor": st.column_config.TextColumn("TENOR", required=True),
            "rate": st.column_config.NumberColumn("RATE", format="%.4f"),
            "source": st.column_config.TextColumn("SOURCE", disabled=True),
        },
    )
    # Prefilled tables are in percent; an uploaded file's units are read from its magnitudes unless set.
    uploaded = prefill.kind == "upload"
    units = st.radio(
        "Rate units",
        ["auto", "percent", "decimal", "bp"],
        index=0 if uploaded else 1,
        horizontal=True,
        key="rates_units_upload" if uploaded else "rates_units",
        help="auto reads the magnitudes (0.04 decimal, 4 percent, 400 bp); near-zero rates need the unit set.",
    )
    quotes, issues, used_units = clean_quotes(edited.drop(columns=["source"], errors="ignore").to_dict("records"),
                                              units=units)  # fmt: skip
    if not _show_issues(issues):
        return None
    try:
        curve = _build(asof, tuple((q.kind, q.tenor, q.rate) for q in quotes), conv, interpolation)
    except ValueError as exc:
        st.error(f"Curve could not be built: {exc}")
        return None
    source, changed = rates_data.provenance(prefill, edited)
    curve_badge = f"{conv} | {source}"

    repricing = _repricing_table(curve)
    worst = float(repricing["Error (bp)"].abs().max())
    cols = st.columns(4)
    cols[0].metric("PILLARS", f"{len(curve.quotes)}")
    cols[1].metric("MAX REPRICING ERROR", f"{worst:.1e} bp")
    cols[2].metric("2Y ZERO", f"{100 * curve.zero_rate(2.0):.3f}%")
    cols[3].metric("10Y ZERO", f"{100 * curve.zero_rate(10.0):.3f}%")
    grid = _curve_grid(curve)
    st.plotly_chart(
        _curve_chart(curve, grid, "ZERO (CONT.)", curve_badge),
        width="stretch",
        config=PLOTLY_2D_CONFIG,
        key="rates_curve_chart",
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
    treasury = _build(asof, _treasury_quotes(data.store, asof), "USD UST", interpolation)
    origin = data.store.origin("UST_PAR", asof)
    t_badge = f"US Treasury par yields ({origin})"
    st.plotly_chart(
        _curve_chart(treasury, _curve_grid(treasury), "ZERO (CONT.)", t_badge),
        width="stretch",
        config=PLOTLY_2D_CONFIG,
        key="rates_treasury_chart",
    )
    st.caption(
        f"US Treasury daily par yield curve for {asof} ({origin}): bills up to 1Y as bond-equivalent yields, "
        "2Y-30Y as semiannual par coupons, settlement T+1. FRED's H.15 DGS* series are the same constant-maturity "
        "par yields with fewer tenors. Bonds price off this curve."
    )
    with st.expander("Treasury repricing check"):
        st.dataframe(_repricing_table(treasury), width="stretch", hide_index=True)
    _market_data_panel(data.store, asof)
    rates_data.set_badges(curve=(curve_badge, asof), govt=(t_badge, asof))
    return Market(curve=curve, govt=treasury)
