"""Bond tab of the Rates page: a Treasury-style bond calculator with prices
in 32nds, yields, risk, spreads to the Treasury and swap curves, cash flows
and key-rate risk on the Treasury curve."""

from __future__ import annotations

import calendar
import datetime as dt
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.rates.bonds import (
    FixedRateBond,
    benchmark_bond,
    bill_bond_equivalent_yield,
    bill_discount_yield,
    bill_money_market_yield,
    bill_price_from_discount,
    bond_position,
    format_32nds,
    ois_par_rate,
    par_yield,
    parse_32nds,
)
from derivx.rates.conventions import add_months
from derivx.rates.curves import Curve, Market, get_conventions
from derivx.risk.sensitivities import Position, key_rate_pv01, parallel_pv01
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SERIES, dark_layout

BENCHMARKS = (2, 5, 10, 30)
FREQUENCY_LABELS = {2: "Semiannual", 1: "Annual", 4: "Quarterly"}
DAY_COUNT_OPTIONS = ("ACT/ACT ICMA", "30/360")
COUPON_BOUNDS_PCT = (0.0, 20.0)
YIELD_BOUNDS_PCT = (-5.0, 25.0)
MAX_YEARS = 50
CHART_SHIFT_BP = 200.0
QUOTE_KEYS = ("bond_input_mode", "bond_price_text", "bond_yield_pct")
MONEY = st.column_config.NumberColumn(format="%,.2f")
CASHFLOW_FORMATS = {
    "ACCRUAL (YRS)": st.column_config.NumberColumn(format="%.6f"),
    "COUPON ($)": MONEY,
    "PRINCIPAL ($)": MONEY,
    "TOTAL ($)": MONEY,
    "TIME (YRS)": st.column_config.NumberColumn(format="%.6f"),
    "DF (YIELD)": st.column_config.NumberColumn(format="%.8f"),
    "PV AT YIELD ($)": MONEY,
    "DF (TSY + Z)": st.column_config.NumberColumn(format="%.8f"),
    "PV ON CURVE ($)": MONEY,
}
SPREAD_FORMATS = {
    "SPREAD (BP)": st.column_config.NumberColumn(format="%.2f"),
    "REFERENCE RATE (%)": st.column_config.NumberColumn(format="%.4f"),
}


def _num(value: float, decimals: int = 2) -> str:
    """Thousands-separated number without a "-0.00" for tiny negatives."""

    return f"{round(value, decimals) + 0.0:,.{decimals}f}"


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _cached(name: str, key: tuple, compute: Callable[[], object]) -> object:
    """Keep the last result of a slow computation across reruns."""

    slot = f"bond_cache_{name}"
    cached = st.session_state.get(slot)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = compute()
    st.session_state[slot] = (key, value)
    return value


def _curve_key(curve: Curve) -> tuple:
    return (curve.asof, tuple(curve.quotes), curve.interpolation, curve.conventions)


def _spot(curve: Curve) -> dt.date:
    """Settlement date of the Treasury curve's own instruments (T+1)."""

    conv = curve.conventions or get_conventions("USD UST")
    cal = conv.cal
    return cal.add_business_days(cal.adjust(curve.asof, "F"), conv.spot_lag)


def _coupon_days(bond: FixedRateBond, settle: dt.date) -> str:
    if bond.end_of_month:
        return "month ends"
    ends = sorted({(c.end.month, c.end.day) for c in bond.coupons(settle)[-bond.frequency:]})
    return " / ".join(f"{calendar.month_abbr[month]} {day}" for month, day in ends)


def _custom_bond(asof: dt.date) -> Tuple[Optional[FixedRateBond], List[str]]:
    c1, c2, c3, c4 = st.columns(4)
    coupon_pct = c1.number_input(
        "COUPON (%)",
        min_value=COUPON_BOUNDS_PCT[0],
        max_value=COUPON_BOUNDS_PCT[1],
        value=4.25,
        step=0.125,
        format="%.4f",
        key="bond_coupon_pct",
    )
    maturity = c2.date_input(
        "MATURITY",
        value=dt.date(asof.year + 9, asof.month, 15),
        min_value=asof + dt.timedelta(days=1),
        max_value=dt.date(asof.year + MAX_YEARS, 12, 31),
        key="bond_maturity",
    )
    day_count = c3.selectbox(
        "DAY COUNT",
        DAY_COUNT_OPTIONS,
        key="bond_day_count",
        help="ACT/ACT ICMA for Treasuries; 30/360 (bond basis) for agencies and corporates.",
    )
    frequency = c4.selectbox("COUPONS", list(FREQUENCY_LABELS), format_func=FREQUENCY_LABELS.get, key="bond_frequency")
    errors: List[str] = []
    dated, first_coupon = None, None
    if st.checkbox(
        "ODD FIRST COUPON (SET THE DATED DATE)",
        value=False,
        key="bond_odd",
        help="Without a dated date the bond pays regular coupons back to settlement.",
    ):
        c1, c2 = st.columns(2)
        dated = c1.date_input(
            "DATED DATE (ACCRUAL START)",
            value=asof - dt.timedelta(days=45),
            min_value=dt.date(asof.year - MAX_YEARS, 1, 1),
            max_value=dt.date(asof.year + MAX_YEARS, 12, 31),
            key="bond_dated",
        )
        long_first = c2.radio(
            "FIRST COUPON",
            ["Short", "Long"],
            horizontal=True,
            key="bond_first_kind",
            help="Short: the first regular coupon date after the dated date. Long: one period later.",
        )
        if dated >= maturity:
            errors.append("The dated date must be before maturity.")
        elif long_first == "Long":
            regular = FixedRateBond(0.0, maturity, frequency=frequency).coupon_dates(dated)
            later = [d for d in regular if d > dated]
            if len(later) < 2:
                errors.append("A long first coupon needs at least two coupon dates after the dated date.")
            else:
                first_coupon = later[1]
    if errors:
        return None, errors
    try:
        bond = FixedRateBond(
            coupon_pct / 100.0,
            maturity,
            dated=dated,
            first_coupon=first_coupon,
            frequency=frequency,
            day_count=day_count,
        )
    except ValueError as exc:
        return None, [f"Cannot build this bond: {exc}"]
    return bond, []


def _bond_inputs(market: Market) -> Optional[Tuple[FixedRateBond, str, dt.date, float]]:
    """Read the bond, settlement and notional; returns (bond, name, settle,
    signed notional) or None after showing what is wrong."""

    asof = market.asof
    govt = market.govt
    c1, c2, c3, c4 = st.columns(4)
    source = c1.radio("BOND", ["Treasury benchmark", "Custom"], horizontal=True, key="bond_source")
    side = c3.radio("POSITION", ["Long", "Short"], horizontal=True, key="bond_side")
    notional = c4.number_input(
        "FACE AMOUNT ($)", min_value=1e3, max_value=1e11, value=1e7, step=1e6, format="%.0f", key="bond_notional"
    )
    spot = _spot(govt)
    if source == "Treasury benchmark":
        years = c2.selectbox("TENOR", BENCHMARKS, index=2, format_func=lambda y: f"{y}Y", key="bond_tenor")
        try:
            bond = benchmark_bond(govt, years, spot)
        except ValueError as exc:
            st.error(f"Cannot build the {years}Y benchmark: {exc}")
            return None
        name = f"{'LONG' if side == 'Long' else 'SHORT'} UST {years}Y {bond.label}"
        st.caption(
            f"On-the-run style {years}Y note: issued at spot ({spot}), coupon = the Treasury curve's {years}Y par "
            f"yield ({bond.coupon * 100:.4f}%, not rounded to 1/8), so it prices at par on the curve."
        )
    else:
        bond, errors = _custom_bond(asof)
        for err in errors:
            st.error(err)
        if bond is None:
            return None
        name = f"{'LONG' if side == 'Long' else 'SHORT'} {bond.label} BOND"

    settle = st.date_input(
        "SETTLEMENT DATE",
        value=spot,
        min_value=asof,
        max_value=add_months(asof, 12 * MAX_YEARS),
        key="bond_settle",
        help=f"Defaults to T+{bond.settle_lag} on the {bond.calendar} calendar.",
    )
    if settle >= bond.maturity or not bond.coupons(settle):
        st.error(f"Settlement {settle} is not before maturity ({bond.maturity}): pick an earlier date.")
        return None
    if not bond.cal.is_business_day(settle):
        st.warning(f"{settle} is not a {bond.calendar} business day; priced for that date anyway.")
    if govt.pillar_dates and bond.coupons(settle)[-1].pay > govt.pillar_dates[-1]:
        st.warning(
            f"The bond pays after the Treasury curve's last quote ({govt.pillar_dates[-1]}); forwards beyond it are "
            "held flat."
        )
    return bond, name, settle, float(notional) * (1.0 if side == "Long" else -1.0)


def _keep_quote_state() -> None:
    """Keep the quote widgets' values through reruns that do not draw them
    (Streamlit drops a widget's state once it is not rendered)."""

    for key in QUOTE_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def _sync_quote(bond: FixedRateBond, settle: dt.date, mode: str, fair_clean: float) -> None:
    """Show the last quote (``bond_quote`` = (price text, clean price)) in
    the widget of the mode just selected."""

    state = st.session_state
    state["bond_quote_mode"] = mode
    text, price = state["bond_quote"]
    if mode == "Price":
        state["bond_price_text"] = text
        return
    lo, hi = YIELD_BOUNDS_PCT
    for candidate in (price, fair_clean):
        try:
            y = bond.yield_from_price(candidate, settle)
        except ValueError:
            continue
        state["bond_yield_pct"] = min(max(y * 100.0, lo), hi)
        return


def _price_input(bond: FixedRateBond, settle: dt.date, fair_clean: float) -> Optional[Tuple[float, float]]:
    """(clean price, street yield) from the price or yield the user typed.

    The quote carries over when the mode switches (a price shows as its
    yield and back); a new bond starts at its fair value on the Treasury
    curve, and a new settlement date keeps the quote.
    """

    state = st.session_state
    if state.get("bond_last_bond") != bond or "bond_quote" not in state:
        text = format_32nds(fair_clean)
        state["bond_last_bond"] = bond
        state["bond_quote"] = (text, parse_32nds(text))
        state.pop("bond_quote_mode", None)
    c1, c2, _ = st.columns(3)
    mode = c1.radio("QUOTE", ["Price", "Yield"], horizontal=True, key="bond_input_mode")
    if state.get("bond_quote_mode") != mode:
        _sync_quote(bond, settle, mode, fair_clean)
    if mode == "Price":
        text = c2.text_input(
            "CLEAN PRICE (32NDS OR DECIMAL)",
            key="bond_price_text",
            help="99-16 = 99 16/32, 99-16+ = 99 16.5/32, 99-162 = 99 16.25/32 (third digit in eighths), or 99.5.",
        )
        try:
            clean = parse_32nds(text)
        except ValueError as exc:
            st.error(str(exc))
            return None
        if clean > 10 * bond.face:
            st.error(f"A clean price of {clean:g} per {bond.face:g} face is not plausible.")
            return None
        try:
            y = bond.yield_from_price(clean, settle)
        except ValueError as exc:
            st.error(f"Cannot solve for the yield: {exc}.")
            return None
        state["bond_quote"] = (text, clean)
        return clean, y
    lo, hi = YIELD_BOUNDS_PCT
    y_pct = c2.number_input(
        "YIELD TO MATURITY (%)", min_value=lo, max_value=hi, step=0.01, format="%.4f", key="bond_yield_pct"
    )
    y = y_pct / 100.0
    clean = bond.price_from_yield(y, settle)
    if abs(clean - state["bond_quote"][1]) > 1e-9:  # keep the typed price text while the yield is unchanged
        state["bond_quote"] = (f"{clean:.6f}", clean)
    return clean, y


def _spreads(bond: FixedRateBond, market: Market, settle: dt.date, clean: float, z: float) -> pd.DataFrame:
    govt, swaps = market.govt, market.curve
    swap_name = swaps.conventions.name if swaps.conventions else "SWAP"
    ytm = bond.yield_from_price(clean, settle)
    rows = [("Z-SPREAD VS TREASURY", z, "Treasury zero rate at maturity (cont. comp.)", govt.zero_rate(bond.maturity))]
    references = [
        ("G-SPREAD", "Treasury par yield at maturity", lambda: par_yield(govt, bond.maturity, settle)),
        (f"I-SPREAD VS {swap_name} OIS", "Spot OIS par rate to maturity", lambda: ois_par_rate(swaps, bond.maturity)),
    ]
    for label, note, rate in references:
        try:
            ref = rate()
            rows.append((label, ytm - ref, note, ref))
        except ValueError as exc:
            rows.append((label, math.nan, f"n/a: {exc}", math.nan))
    try:
        asw = bond.asset_swap_spread(clean, market, settle)
        swap_price = bond.price_from_curve(swaps, settle)
        rows.append((f"ASW PAR-PAR VS {swap_name}", asw, f"Clean price on the swap curve {swap_price:.4f}", math.nan))
    except ValueError as exc:
        rows.append((f"ASW PAR-PAR VS {swap_name}", math.nan, f"n/a: {exc}", math.nan))
    return pd.DataFrame(
        {
            "SPREAD": [r[0] for r in rows],
            "SPREAD (BP)": [r[1] * 1e4 for r in rows],
            "REFERENCE": [r[2] for r in rows],
            "REFERENCE RATE (%)": [r[3] * 100.0 for r in rows],
        }
    )


def _cashflow_table(
    bond: FixedRateBond, market: Market, settle: dt.date, y: float, z: float, scale: float
) -> pd.DataFrame:
    flows = bond.cashflows(settle)
    times = bond.yield_times(settle)
    df_yield = (1.0 + y / bond.frequency) ** (-bond.frequency * times)
    govt = market.govt
    t_pay = np.array([govt.time(d) for d in flows["pay"]])
    df_curve = govt.df_times(t_pay) * np.exp(-z * (t_pay - govt.time(settle))) / govt.df(settle)
    amount = flows["amount"].to_numpy() * scale
    return pd.DataFrame(
        {
            "ACCRUAL START": flows["start"],
            "COUPON DATE": flows["end"],
            "PAY DATE": flows["pay"],
            "ACCRUAL (YRS)": flows["accrual"],
            "COUPON ($)": flows["coupon"] * scale,
            "PRINCIPAL ($)": flows["principal"] * scale,
            "TOTAL ($)": amount,
            "TIME (YRS)": times,
            "DF (YIELD)": df_yield,
            "PV AT YIELD ($)": amount * df_yield,
            "DF (TSY + Z)": df_curve,
            "PV ON CURVE ($)": amount * df_curve,
        }
    )


def _price_yield_chart(bond: FixedRateBond, settle: dt.date, clean: float, y: float) -> go.Figure:
    shifts = np.linspace(-CHART_SHIFT_BP, CHART_SHIFT_BP, 81) * 1e-4
    yields = np.maximum(y + shifts, -0.9 * bond.frequency)
    prices = np.array([bond.price_from_yield(v, settle) for v in yields])
    tangent = clean + bond.dv01(y, settle) * 1e4 * (yields - y)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=yields * 100, y=prices, mode="lines", name="CLEAN PRICE", line=dict(color=SERIES[0], width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=yields * 100,
            y=tangent,
            mode="lines",
            name="DURATION TANGENT",
            line=dict(color=ACCENT, width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[y * 100],
            y=[clean],
            mode="markers",
            name="QUOTE",
            marker=dict(color=SERIES[2], size=10, line=dict(color="#000000", width=1)),
        )
    )
    dark_layout(fig, height=360, xtitle="YIELD TO MATURITY (%)", ytitle=f"CLEAN PRICE (PER {bond.face:g})")
    return bound_axes(
        fig,
        xaxis=axis_limits(yields * 100, pad=0.02),
        yaxis=axis_limits(np.concatenate([prices, tangent]), pad=0.05),
    )


def _ladder_chart(rows: List[Dict[str, object]]) -> go.Figure:
    labels = [r["label"] for r in rows]
    values = [float(r["pv01"]) for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=values, name="BOND", marker=dict(color=SERIES[0])))
    dark_layout(fig, height=340, xtitle="TREASURY CURVE QUOTE", ytitle="PV01 ($ PER +1BP)", legend=False)
    return bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(labels) - 0.5, "min_span": 1.0},
        yaxis=axis_limits([0.0, *values], pad=0.1),
    )


def _bill_calculator() -> None:
    with st.expander("T-BILL CALCULATOR"):
        c1, c2, c3 = st.columns(3)
        days = int(c1.number_input("DAYS TO MATURITY", min_value=1, max_value=364, value=91, key="bond_bill_days"))
        mode = c2.radio("QUOTE", ["Discount yield", "Price"], horizontal=True, key="bond_bill_mode")
        if mode == "Discount yield":
            disc = c3.number_input(
                "DISCOUNT YIELD (%)",
                min_value=-5.0,
                max_value=50.0,
                value=4.0,
                step=0.005,
                format="%.4f",
                key="bond_bill_discount",
            )
            price = bill_price_from_discount(disc / 100.0, days)
        else:
            price = c3.number_input(
                "PRICE (PER 100)",
                min_value=50.0,
                max_value=105.0,
                value=99.0,
                step=0.01,
                format="%.6f",
                key="bond_bill_price",
            )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PRICE (PER 100)", f"{price:.6f}")
        m2.metric("DISCOUNT YIELD (%)", f"{bill_discount_yield(price, days) * 100:.4f}")
        m3.metric("BOND-EQUIVALENT (%)", f"{bill_bond_equivalent_yield(price, days) * 100:.4f}")
        m4.metric("MONEY-MARKET (%)", f"{bill_money_market_yield(price, days) * 100:.4f}")
        st.caption(
            "Discount yield: (100 - P)/100 x 360/days. Bond-equivalent (investment) yield: simple ACT/365 up to "
            "182 days, the Treasury's semiannual formula beyond. Money-market yield: (100/P - 1) x 360/days."
        )


def render(market: Market) -> List[Position]:
    """Bond calculator tab; returns the bond position (none while an input
    is invalid). The T-bill calculator needs no curve and always shows."""

    st.markdown("### / BOND")
    _keep_quote_state()
    positions = _bond_calculator(market)
    _bill_calculator()
    return positions


def _bond_calculator(market: Market) -> List[Position]:
    if market.govt is None:
        st.error("The bond calculator needs the Treasury curve.")
        return []
    built = _bond_inputs(market)
    if built is None:
        return []
    bond, name, settle, notional = built
    govt = market.govt
    try:
        fair_clean = bond.price_from_curve(govt, settle)
    except ValueError as exc:
        st.error(f"Cannot price this bond on the Treasury curve: {exc}")
        return []
    quoted = _price_input(bond, settle, fair_clean)
    if quoted is None:
        return []
    clean, y = quoted
    accrued = bond.accrued(settle)
    dirty = clean + accrued
    try:
        z = bond.z_spread(clean, govt, settle)
        y_true = bond.yield_from_price(clean, settle, convention="true")
    except ValueError as exc:
        st.error(f"Cannot solve for the z-spread over the Treasury curve or the true yield: {exc}.")
        return []
    scale = notional / bond.face
    position = bond_position(bond, notional, z, name=name)

    st.caption(
        f"{bond.day_count}, {FREQUENCY_LABELS.get(bond.frequency, bond.frequency)} coupons on "
        f"{_coupon_days(bond, settle)}, paid the next {bond.calendar} business day; settlement {settle}. "
        f"Street-convention yield: compounded {bond.frequency}x a year with a fractional first period, cash flows "
        "on their scheduled coupon dates."
    )
    m = st.columns(4)
    m[0].metric("CLEAN (32NDS)", format_32nds(clean))
    m[1].metric("CLEAN PRICE (PER 100)", f"{clean:.6f}")
    m[2].metric("DIRTY PRICE (PER 100)", f"{dirty:.6f}")
    m[3].metric("ACCRUED (PER 100)", f"{accrued:.6f}")
    m = st.columns(4)
    m[0].metric("YIELD (%, STREET)", f"{y * 100:.4f}")
    m[1].metric("TRUE YIELD (%)", f"{y_true * 100:.4f}")
    m[2].metric("MACAULAY DURATION (YRS)", f"{bond.macaulay_duration(y, settle):.4f}")
    m[3].metric("MODIFIED DURATION (YRS)", f"{bond.modified_duration(y, settle):.4f}")
    m = st.columns(4)
    m[0].metric("CONVEXITY (YRS²)", f"{bond.convexity(y, settle):.4f}")
    m[1].metric("DV01 ($, +1BP)", _num(bond.dv01(y, settle, notional)))
    m[2].metric("YIELD VALUE OF 1/32 (BP)", f"{bond.yield_value_32nd(y, settle) * 1e4:.4f}")
    m[3].metric(
        "SETTLEMENT AMOUNT ($)",
        _num(dirty * abs(scale)),
        help="Dirty price x face amount: paid by the buyer (long), received by the seller (short).",
    )
    st.caption(
        "DV01 is the signed P&L of the position for +1bp of yield (dP/dy x 1bp): a long bond loses when yields "
        "rise. True yield discounts on the actual (business-day) payment dates. Convexity is (1/P) d2P/dy2 in "
        f"years squared. Coupon BPS (price change per +1bp of coupon): {bond.bps(y, settle):.6f} per 100."
    )

    st.markdown("### / SPREADS")
    spreads = _spreads(bond, market, settle, clean, z)
    st.dataframe(spreads, hide_index=True, width="stretch", column_config=SPREAD_FORMATS)
    st.download_button("Download spreads CSV", _csv(spreads), file_name="bond_spreads.csv", key="bond_spreads_csv")
    st.caption(
        "Z: constant spread over the Treasury curve's continuously compounded zero rates (ACT/365F) that reprices "
        "the dirty price. G: street yield minus the Treasury par yield at the bond's maturity (a par bond on the "
        "Treasury curve). I: street yield minus the swap curve's OIS par rate to maturity (annual ACT/360 for "
        "SOFR; the bases differ). ASW: par-par asset-swap spread, (swap-curve dirty price - market dirty price) / "
        "annuity of a floating leg on the bond's payment dates (OIS day count)."
    )

    st.markdown("### / CASH FLOWS")
    table = _cashflow_table(bond, market, settle, y, z, scale)
    st.dataframe(table, hide_index=True, width="stretch", column_config=CASHFLOW_FORMATS)
    st.download_button("Download cash flows CSV", _csv(table), file_name="bond_cashflows.csv", key="bond_cf_csv")
    st.caption(
        f"Amounts are for {_num(abs(notional), 0)} face, signed to you. Both PV columns sum to the position's "
        f"dirty value {_num(dirty * scale)}: at the yield, and on the Treasury curve plus the z-spread "
        f"({z * 1e4:.2f} bp), both discounted to settlement."
    )

    st.markdown("### / PRICE / YIELD")
    st.plotly_chart(
        _price_yield_chart(bond, settle, clean, y), width="stretch", config=PLOTLY_2D_CONFIG, key="bond_price_yield"
    )
    st.caption("Clean price against street yield; the dashed line is the duration (first-order) approximation.")

    st.markdown("### / KEY-RATE PV01 (TREASURY CURVE)")
    if not govt.quotes:
        st.info("Key-rate risk needs a Treasury curve bootstrapped from quotes.")
    else:
        key = (bond, notional, z, _curve_key(govt))
        rows = _cached("ladder", key, lambda: key_rate_pv01(position.pv, market, which="govt"))
        pv01 = _cached("pv01", key, lambda: parallel_pv01(position.pv, market, which="govt"))
        st.plotly_chart(_ladder_chart(rows), width="stretch", config=PLOTLY_2D_CONFIG, key="bond_ladder_chart")
        ladder = pd.DataFrame({"QUOTE": [r["label"] for r in rows], "PV01 ($, +1BP)": [r["pv01"] for r in rows]})
        st.download_button("Download key-rate CSV", _csv(ladder), file_name="bond_key_rates.csv", key="bond_kr_csv")
        st.caption(
            f"P&L for +1bp on one Treasury curve quote at a time, the curve rebuilt and the z-spread held. The "
            f"ladder sums to {_num(sum(r['pv01'] for r in rows))} against a parallel curve PV01 of {_num(pv01)} "
            f"and a yield DV01 of {_num(bond.dv01(y, settle, notional))}."
        )
    return [position]
