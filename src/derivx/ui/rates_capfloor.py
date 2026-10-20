"""CAP / FLOOR tab: caps and floors on a term rate, priced with the
Bachelier (normal), Black or shifted Black model.

On a single-curve market the term rate projects on the swap curve. On a
multi-curve market (EUR) the cap is on EURIBOR 3M or 6M: forwards from its
projection curve, discounting on €STR, and key-rate risk bucketed per curve."""

from __future__ import annotations

import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from derivx.rates.capfloor import CapFloor, cap_floor_parity, capfloor_position
from derivx.rates.curves import Market
from derivx.risk.sensitivities import Position, key_rate_pv01, parallel_pv01
from derivx.ui import rates_risk
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SERIES, dark_layout

MODEL_LABELS = {"Normal (bp)": "bachelier", "Black (%)": "black", "Shifted Black (%)": "shifted_black"}
TENORS = ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]
FORWARD_STARTS = ["Spot", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y"]
# Vol inputs are in display units: bp for the normal model, % for the Black models.
VOL_UNITS = {"bachelier": ("bp", 1e-4, 2000.0), "black": ("%", 0.01, 300.0), "shifted_black": ("%", 0.01, 300.0)}
DEFAULT_FLAT_VOL = {"bachelier": 100.0, "black": 28.0, "shifted_black": 18.0}
DEFAULT_TERM_VOLS = {
    "bachelier": [(0.5, 85.0), (1.0, 95.0), (2.0, 105.0), (5.0, 100.0), (10.0, 90.0), (20.0, 80.0)],
    "black": [(0.5, 24.0), (1.0, 27.0), (2.0, 29.0), (5.0, 28.0), (10.0, 25.0), (20.0, 22.0)],
    "shifted_black": [(0.5, 15.0), (1.0, 17.0), (2.0, 18.0), (5.0, 18.0), (10.0, 16.0), (20.0, 14.0)],
}
CAPLET_FORMATS = {
    "EXPIRY (Y)": st.column_config.NumberColumn(format="%.4f"),
    "TAU": st.column_config.NumberColumn(format="%.6f"),
    "FORWARD (%)": st.column_config.NumberColumn(format="%.4f"),
    "STRIKE (%)": st.column_config.NumberColumn(format="%.4f"),
    "DF": st.column_config.NumberColumn(format="%.8f"),
    "PV": st.column_config.NumberColumn(format="%.2f"),
    "PV01 (/BP)": st.column_config.NumberColumn(format="%.2f"),
}


def _vol_table(model: str) -> Dict[float, float]:
    """Editable expiry -> vol table; returns the vols as decimals."""

    label, unit, top = VOL_UNITS[model]
    column = f"VOL ({label})"
    default = pd.DataFrame(DEFAULT_TERM_VOLS[model], columns=["EXPIRY (Y)", column])
    edited = st.data_editor(
        default,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"cap_vol_table_{model}",
        column_config={
            "EXPIRY (Y)": st.column_config.NumberColumn(min_value=0.0, max_value=60.0, step=0.25, format="%.2f"),
            column: st.column_config.NumberColumn(min_value=0.0, max_value=top, step=1.0, format="%.2f"),
        },
    )
    rows = edited.dropna()
    if rows.empty:
        raise ValueError("the vol table needs at least one row with an expiry and a vol")
    expiries = rows["EXPIRY (Y)"].astype(float)
    if expiries.duplicated().any():
        raise ValueError("vol table expiries must be distinct")
    return {float(t): float(v) * unit for t, v in zip(expiries, rows[column].astype(float))}


def _caplet_view(table: pd.DataFrame, model: str) -> pd.DataFrame:
    label, unit, _ = VOL_UNITS[model]
    per = "1BP" if model == "bachelier" else "VOL PT"
    return pd.DataFrame(
        {
            "FIXING": table["fix"],
            "START": table["start"],
            "END": table["end"],
            "PAY": table["pay"],
            "EXPIRY (Y)": table["expiry"],
            "TAU": table["tau"],
            "FORWARD (%)": table["forward"] * 100,
            "STRIKE (%)": table["strike"] * 100,
            f"VOL ({label})": table["vol"] / unit,
            "DF": table["df"],
            "PV": table["pv"],
            f"VEGA (/{per})": table["vega"],
            "STATUS": np.where(table["fixed"], "fixed", "live"),
        }
    )


def _date_limits(days: List[dt.date]) -> Dict[str, str]:
    pad = dt.timedelta(days=max(15, (max(days) - min(days)).days // 40))
    return {"min": (min(days) - pad).isoformat(), "max": (max(days) + pad).isoformat()}


def _strip_chart(table: pd.DataFrame, strike: float, kind: str) -> go.Figure:
    fixes = list(table["fix"])
    forwards = (table["forward"] * 100).tolist()
    pvs = table["pv"].tolist()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.55, 0.45],
        subplot_titles=("FORWARD VS STRIKE (%)", f"{kind.upper()}LET PV"),
    )
    fig.add_trace(
        go.Scatter(x=fixes, y=forwards, mode="lines+markers", name="FORWARD", line=dict(color=SERIES[0], width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fixes, y=[strike * 100] * len(fixes), mode="lines" if len(fixes) > 1 else "markers", name="STRIKE",
            line=dict(color=ACCENT, width=2, dash="dash"), marker=dict(color=ACCENT, symbol="line-ew-open", size=14),
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Bar(x=fixes, y=pvs, name="PV", marker=dict(color=SERIES[2])), row=2, col=1)
    dark_layout(fig, height=520)
    fig.update_xaxes(title_text="FIXING DATE", row=2, col=1)
    fig.update_yaxes(title_text="RATE (%)", row=1, col=1)
    fig.update_yaxes(title_text="PV", row=2, col=1)
    dates = _date_limits(fixes)
    bound_axes(
        fig,
        xaxis=dates,
        xaxis2=dates,
        yaxis=axis_limits([*forwards, strike * 100], pad=0.1),
        yaxis2=axis_limits([0.0, *pvs], pad=0.1, floor=min(0.0, *pvs)),
    )
    return fig


def _ladder_chart(rows: List[dict]) -> go.Figure:
    labels = [r["label"] for r in rows]
    values = [r["pv01"] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=labels, y=values, name="PV01",
            marker=dict(color=[SERIES[0] if v >= 0 else ACCENT for v in values]),
        )
    )
    dark_layout(fig, height=340, xtitle="CURVE QUOTE", ytitle="PV01 (PER +1BP)", legend=False)
    bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(labels) - 0.5, "min_span": 1.0},
        yaxis=axis_limits([0.0, *values], pad=0.15),
    )
    return fig


def render(market: Market) -> List[Position]:
    """Draw the tab and return the priced cap/floor as a position."""

    st.markdown("### / CAP & FLOOR")
    conv = market.curve.conventions
    calendar = conv.calendar if conv else "USD"
    day_count = conv.irs_float_day_count if conv else "ACT/360"
    fixing_lag = conv.spot_lag if conv else 2

    c1, c2, c3, c4, c5 = st.columns(5)
    kind = c1.radio("TYPE", ["Cap", "Floor"], horizontal=True, key="cap_kind").lower()
    notional = c2.number_input(
        "NOTIONAL", min_value=1.0, max_value=1e12, value=10_000_000.0, step=1_000_000.0, format="%.0f",
        key="cap_notional",
    )
    tenor = c3.selectbox("TENOR", TENORS, index=TENORS.index("5Y"), key="cap_tenor")
    fwd = c4.selectbox("FORWARD START", FORWARD_STARTS, key="cap_forward_start")
    indices = [k for k in market.projections if not market.projections.is_discount(k)]
    index_choice = c5.radio("INDEX", indices or ["3M", "6M"], horizontal=True, key="cap_index")
    projection = index_choice if indices else None
    index_months = market.get_curve(projection).index.months if projection else int(index_choice[:-1])

    c1, c2, c3, c4 = st.columns(4)
    strike_mode = c1.radio("STRIKE", ["ATM ± bp", "Absolute %"], horizontal=True, key="cap_strike_mode")
    if strike_mode == "Absolute %":
        strike_pct = c2.number_input(
            "STRIKE (%)", min_value=-5.0, max_value=30.0, value=4.0, step=0.125, format="%.4f", key="cap_strike_pct"
        )
    else:
        offset_bp = c2.number_input(
            "ATM OFFSET (BP)", min_value=-1000.0, max_value=1000.0, value=0.0, step=25.0, format="%.1f",
            key="cap_atm_offset_bp",
        )
    model = MODEL_LABELS[c3.selectbox("MODEL", list(MODEL_LABELS), key="cap_model")]
    shift = 0.0
    if model == "shifted_black":
        shift = c4.number_input(
            "SHIFT (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.25, format="%.2f", key="cap_shift_pct"
        ) / 100

    vol_label, vol_unit, vol_top = VOL_UNITS[model]
    vol_mode = st.radio("VOL INPUT", ["Flat", "Term structure"], horizontal=True, key="cap_vol_mode")
    try:
        if vol_mode == "Flat":
            vol = st.number_input(
                f"FLAT VOL ({vol_label})", min_value=0.0, max_value=vol_top, value=DEFAULT_FLAT_VOL[model],
                step=1.0, format="%.2f", key=f"cap_flat_vol_{model}",
            ) * vol_unit
        else:
            st.caption("Vol by option expiry (years to fixing), linear in expiry and flat outside the table.")
            vol = _vol_table(model)
        if projection:
            index = market.get_curve(projection).index
            terms = dict(
                forward_start=None if fwd == "Spot" else fwd, index_months=index.months, day_count=index.day_count,
                calendar=index.calendar, fixing_lag=index.fixing_lag, projection=projection,
            )  # fmt: skip
            calendar, day_count, fixing_lag = index.calendar, index.day_count, index.fixing_lag
        else:
            terms = dict(
                forward_start=None if fwd == "Spot" else fwd, index_months=index_months, day_count=day_count,
                calendar=calendar, fixing_lag=fixing_lag,
            )
        atm = CapFloor(notional, kind, 0.0, tenor, 0.0, **terms).atm_strike(market)
        strike = strike_pct / 100 if strike_mode == "Absolute %" else atm + offset_bp * 1e-4
        cf = CapFloor(notional, kind, strike, tenor, vol, model, shift, **terms)
        table = cf.caplets(market)
        position = capfloor_position(cf, asof=market.asof)
        pv = float(table["pv"].sum())
        vega = float(table["vega"].sum())
        premium_bp = cf.premium_bp_running(market)
        if market.is_multi_curve:
            trade = ("cap", repr(position.details["trade"]))
            curves = rates_risk.projection_chain(market, projection)
            ladders = rates_risk.ladders(market, position.pv, trade, curves)
            pv01 = rates_risk.rate_pv01(market, position.pv, trade)
            ladder = [dict(r, pv01=float(v)) for key in curves for r, v in zip(*ladders[key])]
        else:
            ladders = None
            pv01 = parallel_pv01(position.pv, market)
            ladder = key_rate_pv01(position.pv, market)
        parity = cap_floor_parity(cf, market)
    except ValueError as err:
        st.error(f"Cannot price this {kind}: {err}")
        return []

    per = "1BP" if model == "bachelier" else "VOL PT"
    m = st.columns(6)
    m[0].metric("PV", f"{pv:,.2f}")
    m[1].metric("PREMIUM (BP RUNNING)", f"{premium_bp:,.2f}")
    m[2].metric("PREMIUM (% UPFRONT)", f"{pv / notional * 100:.4f}")
    m[3].metric("ATM STRIKE (%)", f"{atm * 100:.4f}")
    m[4].metric(f"VEGA (/{per})", f"{vega:,.2f}")
    m[5].metric("PV01 (/BP)", f"{pv01:,.2f}")
    shift_text = (
        "+1bp on every rate quote of the OIS and EURIBOR curves, rebuilt (basis spreads held)"
        if ladders is not None else "+1bp on every curve quote with the curve rebuilt"
    )  # fmt: skip
    st.caption(
        f"{cf.label}, {len(table)} {kind}lets. PV, vega and PV01 are P&L to the holder in "
        f"currency for the notional; PV01 is {shift_text}. Running premium is "
        f"PV over the strip annuity (sum of notional x accrual x DF). Cap - floor - payer swap = "
        f"{parity['residual']:.2e}."
    )
    st.caption(
        f"Schedule: {index_months}M periods rolled back from maturity, {calendar} calendar, modified following, "
        f"{day_count} accrual; each rate fixes {fixing_lag} business days before its period starts and pays at "
        "the period end. A spot-starting cap leaves out the first period (it fixes at trade); a forward start is "
        "spot plus the forward period, modified following, as on the swap tab. Option time is ACT/365F to the "
        "fixing; "
        + (f"forwards from the {projection} projection curve, discounting on the OIS curve." if projection
           else "one curve projects and discounts.")
    )  # fmt: skip
    if vol_mode == "Flat" and vol == 0.0:
        st.warning("Zero vol: every caplet is worth only its intrinsic value on the forward.")
    pillars = market.curve.pillar_dates
    if pillars and table["pay"].max() > pillars[-1]:
        st.warning(f"The {kind} pays after the curve's last quote ({pillars[-1]}); forwards beyond it are held flat.")

    st.markdown(f"### / {kind.upper()}LETS")
    st.plotly_chart(
        _strip_chart(table, strike, kind), width="stretch", config=PLOTLY_2D_CONFIG, key="cap_strip_chart"
    )
    view = _caplet_view(table, model)
    two_dp = st.column_config.NumberColumn(format="%.2f")
    formats = {**CAPLET_FORMATS, **{c: two_dp for c in view.columns if c.startswith(("VOL (", "VEGA ("))}}
    st.dataframe(view, width="stretch", hide_index=True, column_config=formats)
    st.download_button(
        "DOWNLOAD CAPLETS CSV", view.to_csv(index=False), file_name=f"{kind}lets.csv", mime="text/csv",
        on_click="ignore", key="cap_caplets_csv",
    )

    st.markdown("### / KEY-RATE PV01" + (" BY CURVE" if ladders is not None else ""))
    if ladders is None:
        st.plotly_chart(_ladder_chart(ladder), width="stretch", config=PLOTLY_2D_CONFIG, key="cap_ladder_chart")
        ladder_view = pd.DataFrame({"QUOTE": [r["label"] for r in ladder], "PV01 (/BP)": [r["pv01"] for r in ladder]})
    else:
        for key in ladders:
            rows, values = ladders[key]
            st.markdown(f"#### {rates_risk.curve_title(market, key)}")
            chart_key = "cap_ladder_chart" if key == "curve" else f"cap_ladder_chart_{key.lower().replace(' ', '_')}"
            st.plotly_chart(rates_risk.ladder_chart([r["label"] for r in rows], values, name=kind.upper(),
                                                    unit=rates_risk.money(market)), width="stretch",
                            config=PLOTLY_2D_CONFIG, key=chart_key)  # fmt: skip
        ladder_view = pd.DataFrame(
            {
                "CURVE": [rates_risk.curve_title(market, key) for key in ladders for _ in ladders[key][0]],
                "QUOTE": [r["label"] for r in ladder],
                "PV01 (/BP)": [r["pv01"] for r in ladder],
            }
        )
    st.dataframe(ladder_view, width="stretch", hide_index=True, column_config=CAPLET_FORMATS)
    st.download_button(
        "DOWNLOAD KEY-RATE PV01 CSV", ladder_view.to_csv(index=False), file_name=f"{kind}_key_rate_pv01.csv",
        mime="text/csv", on_click="ignore", key="cap_ladder_csv",
    )
    if ladders is None:
        sums = f"Buckets sum to {sum(r['pv01'] for r in ladder):,.2f} against a parallel PV01 of {pv01:,.2f}."
    else:
        rate_sum, spread_sum = rates_risk.ladder_sums(market, ladders)
        sums = f"The rate-quote buckets sum to {rate_sum:,.2f} against a PV01 of {pv01:,.2f}"
        if any(r["label"].endswith(" BASIS") for r in ladder):
            sums += f"; the 3s6s basis-spread buckets (+1bp on a spread, not a rate) sum to {spread_sum:,.2f}"
        sums += "."
    st.caption(
        "P&L for +1bp on one curve quote at a time (central difference, curve rebuilt"
        + ("; an OIS bump rebuilds the EURIBOR curves on it, quotes held" if ladders is not None else "")
        + f"). {sums}"
    )
    return [position]
