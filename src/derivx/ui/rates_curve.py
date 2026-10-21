"""CURVE tab: build the curve set from editable quotes and show the
Treasury curve, all from the market data the sidebar DATA SOURCE picks.

The swap (OIS) curve discounts every trade. For EUR the tab also builds
EURIBOR 6M and 3M projection curves, each from its own quote table and
bootstrapped with the €STR curve discounting (multi-curve), with forward
and basis charts. USD SOFR and GBP SONIA stay single-curve, the market
standard for OIS. Returns the Market every other Rates tab prices off."""

from __future__ import annotations

import datetime as dt
import hashlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.data.rates_history import HistoryStore, dataset_info
from derivx.rates.curves import CONVENTIONS, INTERPOLATIONS, OIS_FUTURES, Curve, CurveQuote, Market, bootstrap
from derivx.rates.exports import market_tables
from derivx.rates.indices import CURRENCY_INDICES, FUTURES_KINDS, HullWhite, price_from_futures_rate
from derivx.rates.market_quotes import treasury_quotes
from derivx.rates.multicurve import (
    BASIS_OF,
    ILLUSTRATIVE_BASIS_BP,
    PROJECTION_KINDS,
    ProjectionCurve,
    bootstrap_projection,
    forward_grid,
    illustrative_quotes,
)
from derivx.rates.quotes import clean_quotes
from derivx.ui import rates_data, rates_exports
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.state import data_editor as persistent_editor  # edits survive mode switches (UIE-04)
from derivx.ui.theme import ACCENT, LEGEND_BELOW, PLOTLY_2D_CONFIG, SERIES, dark_layout

SWAP_CURVES = ["USD SOFR", "EUR ESTR", "GBP SONIA"]
HW_DEFAULTS = (3.0, 85.0)  # mean reversion (% per year), normal short-rate vol (bp per year)
FIXING_OF = {"USD": "SOFR", "EUR": "ESTR", "GBP": "SONIA"}
SINGLE_CURVE_NOTES = {
    "USD": "USD swaps are SOFR OIS since the LIBOR cessation: the SOFR curve above discounts and projects them "
    "(single-curve, the market standard). No term-rate projection curve is built: CME Term SOFR swaps are thin, "
    "with no free quotes or standard curve conventions. SOFR futures (SR1/SR3) can be added to the SOFR table.",
    "GBP": "GBP swaps are SONIA OIS since the LIBOR cessation: the SONIA curve above discounts and projects them "
    "(single-curve, the market standard).",
}
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


Quotes = Tuple[Tuple[str, str, float], ...]
HwKey = Optional[Tuple[float, float]]
FixingsKey = Tuple[Tuple[dt.date, float], ...]


@st.cache_data(show_spinner=False)
def _build(
    asof: dt.date,
    quotes: Quotes,
    conventions: str,
    interpolation: str,
    hw: HwKey = None,
    fixings: FixingsKey = (),
) -> Curve:
    convexity = HullWhite(*hw) if hw is not None else None
    return bootstrap(asof, [CurveQuote(*q) for q in quotes], conventions, interpolation, convexity=convexity,
                     fixings=dict(fixings))  # fmt: skip


@st.cache_data(show_spinner=False)
def _build_projection(
    discount: Tuple[dt.date, Quotes, str, str, HwKey, FixingsKey],
    index: str,
    quotes: Quotes,
    interpolation: str,
    hw: HwKey,
    basis: Optional[Tuple[str, Quotes, str]] = None,
) -> ProjectionCurve:
    """Projection curve on the discount curve ``discount`` (the arguments of
    :func:`_build`); ``basis`` = (index, quotes, interpolation) of the curve
    its BASIS quotes are against (built the same way, on the same discount)."""

    ois = _build(*discount)
    other = _build_projection(discount, basis[0], basis[1], basis[2], hw) if basis else None
    convexity = HullWhite(*hw) if hw is not None else None
    return bootstrap_projection(ois.asof, [CurveQuote(*q) for q in quotes], index, ois, interpolation, basis=other,
                                convexity=convexity)  # fmt: skip


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
    pillars = curve.years
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
    table = pd.DataFrame(
        {
            "Tenor": [r["tenor"] for r in rows],
            "Instrument": [r["kind"] for r in rows],
            "Maturity": [r["maturity"] for r in rows],
            "Quote (%)": [100 * r["quote"] for r in rows],
            "Curve (%)": [100 * r["model"] for r in rows],
            "Error (bp)": [r["error_bp"] for r in rows],
        }
    )
    futures = [r["kind"] in FUTURES_KINDS for r in rows]
    if any(futures):  # futures: price, and the convexity adjustment (futures rate - forward)
        table["Price"] = [price_from_futures_rate(r["quote"]) if f else np.nan for r, f in zip(rows, futures)]
        table["Convexity (bp)"] = [r["convexity_bp"] if f else np.nan for r, f in zip(rows, futures)]
    return table


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
    previous = st.session_state.get("rates_asof", st.session_state.get("_keep_rates_asof"))
    if previous is not None and previous not in dates:
        # e.g. a date only the live cache had, after switching back to the snapshot: say so, do not jump silently
        st.session_state.pop("rates_asof", None)
        data.notices.append(
            ("warning", f"VALUATION DATE {previous} is not in the {data.mode} history; showing {dates[0]} instead.")
        )
    rates_data.sticky("rates_asof", dates)
    asof = st.sidebar.selectbox(
        "VALUATION DATE",
        dates,
        key="rates_asof",
        help=(
            f"Dates with a US Treasury par curve in the history store ({len(dates)} dates, {dates[-1]} to "
            f"{dates[0]}): every business day since 2015 (FRED H.15, then the Treasury's 2026 CSV), month ends "
            "and the scenario episodes before that, and whatever the live cache or your uploads add."
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


def _convexity_inputs() -> HullWhite:
    """Hull-White parameters for futures convexity (sidebar)."""

    with st.sidebar.expander("FUTURES CONVEXITY (HULL-WHITE)"):
        a = st.number_input(
            "MEAN REVERSION (% PER YEAR)", min_value=0.0, max_value=100.0, value=HW_DEFAULTS[0], step=0.5,
            format="%.2f", key="rates_hw_a",
        )  # fmt: skip
        sigma = st.number_input(
            "SHORT-RATE VOL (BP PER YEAR, NORMAL)", min_value=0.0, max_value=500.0, value=HW_DEFAULTS[1], step=5.0,
            format="%.1f", key="rates_hw_sigma",
        )  # fmt: skip
        st.caption(
            "Futures are margined daily, so a futures rate sits above the forward: the adjustment is the "
            "Hull-White (one-factor) convexity of each contract, shown in the repricing tables. These are model "
            "inputs, not market data."
        )
    return HullWhite(a / 100.0, sigma / 1e4)


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


def _ois_fixings(store: HistoryStore, currency: str, asof: dt.date) -> FixingsKey:
    """Overnight fixings of the last four months (part-fixed SOFR futures
    need the days of their period before the valuation date)."""

    name = FIXING_OF.get(currency)
    if name is None:
        return ()
    history = store.fixing_history(name, asof - dt.timedelta(days=125), asof)
    return tuple(sorted(history.items()))


@st.cache_data(show_spinner=False)
def _projection_prefill(discount_args: tuple, index: str) -> Tuple[pd.DataFrame, Curve]:
    """ILLUSTRATIVE quotes for a projection table (not market data): priced
    off the prefilled OIS curve (:func:`_build` arguments, log-linear) plus
    an assumed basis, as a percent / price table, and the curve they build
    (the basis curve of the next table). Futures prices are set with the
    default Hull-White inputs: quotes are inputs, so the sidebar's
    interpolation and convexity settings change the curves built from them,
    never the quotes (or the editor, which would drop the user's edits)."""

    discount = _build(*discount_args)
    convexity = HullWhite(HW_DEFAULTS[0] / 100.0, HW_DEFAULTS[1] / 1e4)
    basis_index = BASIS_OF.get(index)
    basis = _projection_prefill(discount_args, basis_index)[1] if basis_index else None
    quotes = illustrative_quotes(discount, index, convexity, basis_curve=basis)
    curve = bootstrap_projection(discount.asof, quotes, index, discount, basis=basis, convexity=convexity)
    profile = ", ".join(f"{y:g}Y {b:g}bp" for y, b in ILLUSTRATIVE_BASIS_BP[index])
    label = f"ILLUSTRATIVE: OIS prefill + assumed basis ({profile})"
    rows = []
    for q in quotes:
        value = round(price_from_futures_rate(q.rate), 4) if q.kind in FUTURES_KINDS else round(100.0 * q.rate, 6)
        rows.append({"instrument": q.kind, "tenor": q.tenor, "rate": value, "source": label})
    return pd.DataFrame(rows, columns=["instrument", "tenor", "rate", "source"]), curve


def _projection_source(changed: int, rows: int) -> str:
    """Badge text for a projection curve's quotes."""

    if not changed:
        return "ILLUSTRATIVE quotes (not market data)"
    if changed >= rows:
        return "quotes entered by you"
    return f"ILLUSTRATIVE quotes, {changed} rows changed by you"


def _legend_below(fig: go.Figure) -> None:
    """Horizontal legend under the plot, so it never covers the lines on a
    narrow screen."""

    fig.update_layout(legend=LEGEND_BELOW)  # dark_layout's default; kept for charts styled without it


def _forward_chart(market: Market, source: str) -> go.Figure:
    """Simple 3M and 6M forwards on the OIS curve and on each projection
    curve, by start date."""

    fig = go.Figure()
    values, years = [], None
    styles = {3: ("dot", SERIES[2]), 6: ("solid", SERIES[3])}
    for months in (3, 6):
        grid = forward_grid(market, months, step_days=14)
        years = grid["years"]
        dash, color = styles[months]
        ois = grid["ois"] * 100
        fig.add_trace(go.Scatter(x=years, y=ois, mode="lines", name=f"OIS {months}M FWD",
                                 line=dict(color=SERIES[0], width=1.5, dash=dash)))  # fmt: skip
        values.append(ois)
        for key, fwd in grid.items():
            if key in ("years", "dates", "ois"):
                continue
            fig.add_trace(go.Scatter(x=years, y=fwd * 100, mode="lines", name=f"{key} FWD",
                                     line=dict(color=color, width=2, dash=dash)))  # fmt: skip
            values.append(fwd * 100)
    dark_layout(fig, height=440, xtitle="FORWARD START (YEARS)", ytitle="SIMPLE FORWARD, ACT/360 (%)")
    _legend_below(fig)
    rates_data.badge(fig, source, market.asof)
    return bound_axes(fig, xaxis=axis_limits(years, pad=0.02, floor=0.0), yaxis=axis_limits(np.concatenate(values),
                                                                                            pad=0.12))  # fmt: skip


def _basis_chart(market: Market, source: str) -> go.Figure:
    """Forward basis in bp: each projection's forward over the OIS forward
    of the same tenor, plus the quoted tenor basis spreads."""

    fig = go.Figure()
    values, xs = [], []
    colors = {3: SERIES[2], 6: SERIES[3]}
    for months in (3, 6):
        grid = forward_grid(market, months, step_days=14)
        for key, fwd in grid.items():
            if key in ("years", "dates", "ois"):
                continue
            spread = (fwd - grid["ois"]) * 1e4
            fig.add_trace(go.Scatter(x=grid["years"], y=spread, mode="lines", name=f"{key} - OIS {months}M",
                                     line=dict(color=colors[months], width=2)))  # fmt: skip
            values.append(spread)
            xs.append(grid["years"])
    for key in market.projections:
        curve = market.projections[key]
        basis = [(i.years, i.quote.rate * 1e4) for i in curve.instruments if i.kind == "BASIS"]
        if basis:
            other = getattr(curve, "basis_index", "")
            fig.add_trace(go.Scatter(x=[b[0] for b in basis], y=[b[1] for b in basis], mode="markers+lines",
                                     name=f"{key} vs {other} BASIS QUOTES (PAR)", line=dict(color=ACCENT, width=1),
                                     marker=dict(size=7, color=ACCENT)))  # fmt: skip
            values.append(np.array([b[1] for b in basis]))
            xs.append(np.array([b[0] for b in basis]))
    dark_layout(fig, height=420, xtitle="FORWARD START / TENOR (YEARS)", ytitle="BASIS (BP)")
    _legend_below(fig)
    rates_data.badge(fig, source, market.asof)
    return bound_axes(
        fig,
        xaxis=axis_limits(np.concatenate(xs), pad=0.02, floor=0.0),
        yaxis=axis_limits(np.concatenate(values + [np.array([0.0])]), pad=0.15),
    )


def _projection_section(
    asof: dt.date,
    currency: str,
    discount_args: tuple,
    prefill_args: tuple,
    interpolation: str,
    hw: HullWhite,
    edits: Dict[str, Tuple[int, int]],
) -> Optional[Dict[str, ProjectionCurve]]:
    """Editors and builds of the currency's projection curves; ``{}`` when it
    is single-curve, None if a table cannot be built (reason shown).
    ``edits`` receives {index: (rows edited or added by the user, rows)}."""

    indices = CURRENCY_INDICES.get(currency, ())
    st.markdown("### / PROJECTION CURVES (MULTI-CURVE)")
    if not indices:
        st.caption(SINGLE_CURVE_NOTES.get(currency, "Single-curve: the OIS curve discounts and projects."))
        return {}
    if not st.checkbox("BUILD EURIBOR PROJECTION CURVES", value=True, key="rates_mc_on"):
        st.info(
            "Single-curve: EURIBOR swaps and caps cannot be priced without their projection curves; OIS and TERM "
            "(single-curve) trades price off the €STR curve."
        )
        return {}
    st.caption(
        "EURIBOR trades over €STR: each tenor gets its own projection curve, bootstrapped with the €STR curve above "
        "discounting every cash flow (dual curve). EURIBOR 6M: deposit, FRAs (AxB months), swaps fixed annual "
        "30/360 vs 6M; EURIBOR 3M: deposit, three-month futures (IMM, price = 100 - rate, Hull-White convexity) "
        "and 3s6s basis swaps (3M + spread vs 6M flat, spread in the table's units). No free EURIBOR quotes "
        "exist: the tables start from ILLUSTRATIVE quotes; paste your own. Futures are entered as prices; the "
        "illustrative ones were set with the default Hull-White inputs, so the sidebar convexity inputs move the "
        "3M forwards under fixed prices."
    )
    built: Dict[str, ProjectionCurve] = {}
    tabs = st.tabs([f"/ {index}" for index in indices])
    for tab, index in zip(tabs, indices):
        with tab:
            basis_index = BASIS_OF.get(index)
            try:
                table, _ = _projection_prefill(prefill_args, index)
            except ValueError as exc:
                st.error(f"Could not prefill {index}: {exc}")
                return None
            digest = hashlib.sha1(table.to_csv(index=False).encode()).hexdigest()[:10]
            edited = persistent_editor(
                table, num_rows="dynamic", width="stretch", hide_index=True,
                key=f"rates_proj_{index.replace(' ', '_')}_{digest}",
                column_config={
                    "instrument": st.column_config.SelectboxColumn("INSTRUMENT", options=list(PROJECTION_KINDS),
                                                                   required=True),
                    "tenor": st.column_config.TextColumn("TENOR / FRA / CONTRACT", required=True),
                    "rate": st.column_config.NumberColumn("RATE / SPREAD / PRICE", format="%.4f"),
                    "source": st.column_config.TextColumn("SOURCE", disabled=True),
                },
            )  # fmt: skip
            units = st.radio("Rate units", ["auto", "percent", "decimal", "bp"], index=1, horizontal=True,
                             key=f"rates_proj_units_{index.replace(' ', '_')}")  # fmt: skip
            rows = edited.drop(columns=["source"], errors="ignore").to_dict("records")
            quotes, issues, _ = clean_quotes(rows, units=units, kinds=PROJECTION_KINDS, asof=asof)
            if not _show_issues(issues):
                return None
            changed = rates_data.edited_rows(table, edited)
            edits[index] = (changed, len(edited))
            if changed:
                st.caption(f"{changed} rows edited, added or deleted by you; the others are ILLUSTRATIVE.")
            basis_args = None
            if any(q.kind == "BASIS" for q in quotes):
                if basis_index is None:
                    others = ", ".join(k for k, v in BASIS_OF.items() if v == index)
                    st.error(
                        f"BASIS quotes go on the shorter tenor's curve ({others or 'none'}: its index plus the "
                        f"spread vs {index}), not on {index}; quote {index} with DEPO, FRA or IRS rows."
                    )
                    return None
                if basis_index not in built:
                    st.error(f"BASIS quotes on {index} need the {basis_index} curve built first.")
                    return None
                other = built[basis_index]
                basis_args = (basis_index, tuple((q.kind, q.tenor, q.rate) for q in other.quotes), other.interpolation)
            try:
                curve = _build_projection(discount_args, index, tuple((q.kind, q.tenor, q.rate) for q in quotes),
                                          interpolation, (hw.mean_reversion, hw.sigma), basis_args)  # fmt: skip
            except ValueError as exc:
                st.error(f"The {index} curve could not be built: {exc}")
                return None
            built[index] = curve
            repricing = _repricing_table(curve)
            cols = st.columns(3)
            cols[0].metric("PILLARS", f"{len(curve.quotes)}")
            cols[1].metric("MAX REPRICING ERROR", f"{float(repricing['Error (bp)'].abs().max()):.1e} bp")
            cols[2].metric("5Y FORWARD (%)", f"{100 * curve.forward(asof + dt.timedelta(days=5 * 365)):.3f}")
            with st.expander(f"{index} repricing check"):
                st.dataframe(repricing, width="stretch", hide_index=True)
    return built


def render() -> Optional[Market]:
    """Draw the CURVE tab; the Market for the other tabs, or None if the
    quotes cannot be built into a curve (the reason is shown)."""

    st.sidebar.markdown("### / RATES MARKET")
    data = rates_data.sidebar()
    asof, conv, interpolation = _sidebar(data)
    hw = _convexity_inputs()
    rates_data.show_notices(data.notices)

    st.markdown("### / SWAP CURVE")
    if data.mode == rates_data.MODES[2] and data.uploaded is not None:
        prefill = rates_data.uploaded_prefill(data.uploaded)
    else:
        prefill = rates_data.swap_prefill(data.store, conv, asof)
    currency = CONVENTIONS[conv].currency
    kinds = SWAP_KINDS + (list(OIS_FUTURES) if conv == "USD SOFR" else [])
    fixing_note = "" if "fixing" not in prefill.label else "; the ON quote is the real overnight fixing"
    futures_note = " SR1/SR3 SOFR futures: contract month as the tenor (DEC26), price in the rate column." if (
        conv == "USD SOFR") else ""  # fmt: skip
    st.caption(
        PREFILL_CAPTIONS[prefill.kind].format(label=prefill.label, ccy=currency, fix=fixing_note)
        + f" Instrument {', '.join(kinds)}; tenor like ON, 1W, 3M, 10Y; rate in %, decimals or bp."
        + futures_note
    )
    for note in prefill.notes:
        st.warning(note)
    edited = persistent_editor(
        prefill.table,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key=f"rates_quotes_{prefill.key}",
        column_config={
            "instrument": st.column_config.SelectboxColumn("INSTRUMENT", options=kinds, required=True),
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
                                              units=units, kinds=kinds, asof=asof)  # fmt: skip
    if not _show_issues(issues):
        return None
    has_futures = any(q.kind in OIS_FUTURES for q in quotes)
    hw_key = (hw.mean_reversion, hw.sigma) if has_futures else None
    fixings = _ois_fixings(data.store, currency, asof) if has_futures else ()
    discount_args = (asof, tuple((q.kind, q.tenor, q.rate) for q in quotes), conv, interpolation, hw_key, fixings)
    try:
        curve = _build(*discount_args)
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
    multi = bool(CURRENCY_INDICES.get(currency))
    st.caption(
        "Every quote reprices exactly on the bootstrapped curve (error column). Zero rates are continuously "
        "compounded on ACT/365F; 3M forwards are simple ACT/360. Beyond the last pillar forwards are held flat. "
        + ("This OIS curve discounts every trade; EURIBOR legs project on their own curves below."
           if multi else "This one curve both discounts and projects (single-curve, the OIS standard).")
    )  # fmt: skip
    with st.expander("Repricing check and curve grid"):
        st.dataframe(repricing, width="stretch", hide_index=True)
        st.download_button(
            "Download curve grid (CSV)", grid.to_csv(index=False).encode(), "swap_curve.csv", key="rates_grid_download"
        )

    # Projection prefills follow the prefilled OIS curve, log-linear, so neither editing an OIS quote nor changing
    # the interpolation or the convexity inputs replaces them (and the user's edits to them).
    prefill_args = (asof, discount_args[1], conv, "loglinear", hw_key, fixings)
    if multi:
        base_rows = prefill.table.drop(columns=["source"], errors="ignore").to_dict("records")
        base_quotes, base_issues, _ = clean_quotes(base_rows, units="auto" if uploaded else "percent", kinds=kinds,
                                                   asof=asof)  # fmt: skip
        if base_quotes and not any(i.level == "error" for i in base_issues):
            candidate = (asof, tuple((q.kind, q.tenor, q.rate) for q in base_quotes), conv, "loglinear", hw_key,
                         fixings)  # fmt: skip
            try:
                _build(*candidate)
                prefill_args = candidate
            except ValueError:
                pass
    edits: Dict[str, Tuple[int, int]] = {}
    projections = _projection_section(asof, currency, discount_args, prefill_args, interpolation, hw, edits)
    if projections is None:
        return None
    market = Market(curve=curve, projections=projections)
    proj_sources = {key: _projection_source(*edits.get(key, (0, 0))) for key in projections}
    if projections:
        names = " / ".join(projections)
        if all(src == _projection_source(1, 1) for src in proj_sources.values()):
            mc_quotes = "EURIBOR quotes entered by you"
        elif any(edits.get(key, (0, 0))[0] for key in projections):
            mc_quotes = "EURIBOR quotes partly edited by you, the rest ILLUSTRATIVE"
        else:
            mc_quotes = "ILLUSTRATIVE EURIBOR quotes (not market data)"
        mc_badge = f"{conv} + {names} | {mc_quotes}"
        st.markdown("#### / OIS VS PROJECTION FORWARDS")
        st.plotly_chart(_forward_chart(market, mc_badge), width="stretch", config=PLOTLY_2D_CONFIG,
                        key="rates_mc_forwards")  # fmt: skip
        st.markdown("#### / BASIS SPREADS")
        st.plotly_chart(_basis_chart(market, mc_badge), width="stretch", config=PLOTLY_2D_CONFIG,
                        key="rates_mc_basis")  # fmt: skip
        st.caption(
            "Forwards are simple ACT/360 rates for 3M and 6M periods starting on each date (period end ~ start plus "
            "the tenor in days): OIS = compounded €STR implied by the discount curve; EURIBOR from its projection "
            "curve. Basis = EURIBOR forward minus the OIS forward of the same period, in bp; markers are the quoted "
            "par 3s6s spreads (3M + spread vs 6M). Positive basis: EURIBOR above €STR (bank credit and term premium)."
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
    proj_badges = {key: (f"{key} | {proj_sources[key]}", asof) for key in projections}
    rates_data.set_badges(curve=(curve_badge, asof), govt=(t_badge, asof), **proj_badges)
    full = market.replace(govt=treasury)
    st.markdown("### / EXPORT")
    tables = dict(market_tables(full))
    tables["swap_repricing"] = repricing
    tables["treasury_repricing"] = _repricing_table(treasury)
    for key in projections:
        tables[f"{key.replace(' ', '_').lower()}_repricing"] = _repricing_table(projections[key])
    rates_exports.export_row(
        "curve set", full, tables,
        {"data_source": data.mode, "valuation_date": asof, "swap_curve": conv, "interpolation": interpolation,
         "quote_units": used_units, "swap_quotes_edited_rows": changed,
         "projection_quotes_edited_rows": {k: v[0] for k, v in edits.items()},
         "upload": None if data.uploaded is None else {"name": data.uploaded.get("name"),
                                                       "sha256": data.uploaded.get("sha")}},
        {"bootstrap": "every quote reprices exactly; curve time ACT/365F; flat forward beyond the last pillar",
         "interpolation": interpolation, "hull_white": {"mean_reversion": hw.mean_reversion, "sigma": hw.sigma,
                                                        "used": hw_key is not None}},
        key="rates_curve_export", store=data.store,
    )  # fmt: skip
    st.caption(
        "Curve nodes (pillar dates, discount factors, zero rates, forwards between pillars, the quote solved at "
        "each) and a dated grid of every curve, the repricing checks and manifest.json (code version, packages, "
        "quotes and their sources, the history store's hash and download URLs)."
    )
    return full
