"""Swap tab of the Rates page: price one vanilla swap, show its cash flows
and key-rate risk, and size benchmark par swaps that hedge it."""
from __future__ import annotations

import datetime as dt
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.rates.conventions import parse_tenor
from derivx.rates.curves import CurveConventions, Market
from derivx.rates.swaps import (
    BENCHMARK_TENORS,
    DEFAULT_CONVENTIONS,
    CurveBumps,
    VanillaSwap,
    curve_bumps,
    hedge_with_benchmarks,
    key_rate_matrix,
    swap_position,
)
from derivx.risk.sensitivities import Position
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SERIES, dark_layout

HEDGE_TENORS = ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")
MAX_TENOR_YEARS = 50.0
MAX_FORWARD_YEARS = 30.0
RATE_BOUNDS_PCT = (-5.0, 25.0)
MONEY = st.column_config.NumberColumn(format="%,.2f")
CASHFLOW_FORMATS = {
    "ACCRUAL (YRS)": st.column_config.NumberColumn(format="%.6f"),
    "RATE (%)": st.column_config.NumberColumn(format="%.4f"),
    "AMOUNT ($)": MONEY,
    "DF": st.column_config.NumberColumn(format="%.8f"),
    "PV ($)": MONEY,
}
HEDGE_FORMATS = {
    "NOTIONAL ($)": st.column_config.NumberColumn(format="%,.0f"),
    "PAR RATE (%)": st.column_config.NumberColumn(format="%.4f"),
    "PV01 ($, +1BP)": MONEY,
}


def _tenor(text: str, label: str, max_years: float) -> Tuple[Optional[str], Optional[str]]:
    """(normalised tenor, error message) for a W/M/Y tenor typed by the user."""

    try:
        tenor = parse_tenor(text)
    except ValueError as exc:
        return None, f"{label}: {exc}"
    if tenor.unit == "D":
        return None, f"{label} must be in weeks, months or years (e.g. 1Y, 18M), not days"
    if tenor.years > max_years:
        return None, f"{label} must be at most {max_years:.0f}Y"
    return str(tenor), None


def _cached(name: str, key: tuple, compute: Callable[[], object]) -> object:
    """Keep the last result of a slow computation across reruns."""

    slot = f"swap_cache_{name}"
    cached = st.session_state.get(slot)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = compute()
    st.session_state[slot] = (key, value)
    return value


def _curve_key(market: Market) -> tuple:
    curve = market.curve
    return (curve.asof, tuple(curve.quotes), curve.interpolation, curve.conventions)


def _conventions_note(swap: VanillaSwap, conv: CurveConventions, market: Market) -> str:
    fixed_dates, _ = swap.schedule_dates(market.asof)
    if swap.float_index == "OIS":
        index = "overnight rate compounded daily"
    else:
        index = f"term rate fixed {swap.fixing_lag} business days before each period"
    lag = f", both legs paid {swap.payment_lag} business days after each period end" if swap.payment_lag else ""
    return (
        f"{conv.name} conventions: start {fixed_dates[0]}, maturity {fixed_dates[-1]}; fixed leg {swap.fixed_months}M "
        f"{swap.fixed_day_count}; float leg {index}, {swap.float_months}M {swap.float_day_count}{lag}; "
        f"{swap.calendar} calendar, {swap.bdc}. One curve projects and discounts both legs."
    )


def _cashflow_table(swap: VanillaSwap, market: Market) -> pd.DataFrame:
    flows = swap.cashflows(market)
    return pd.DataFrame(
        {
            "LEG": flows["leg"].str.upper(),
            "START": flows["start"],
            "END": flows["end"],
            "PAY": flows["pay"],
            "ACCRUAL (YRS)": flows["accrual"],
            "RATE (%)": flows["rate"] * 100.0,
            "AMOUNT ($)": flows["amount"],
            "DF": flows["df"],
            "PV ($)": flows["pv"],
        }
    )


def _bumps(market: Market) -> Tuple[Optional[CurveBumps], Optional[str]]:
    """(bumped curves for every risk number, why there are none). They are
    rebuilt only when the curve changes, so editing the swap just reprices."""

    if not market.curve.quotes:
        return None, "Key-rate risk and hedging need a curve bootstrapped from quotes."
    try:
        return _cached("bumps", _curve_key(market), lambda: curve_bumps(market)), None
    except ValueError as exc:
        return None, f"Cannot rebuild the curve with bumped quotes, so risk is not available: {exc}"


def _ladder_only(swap: VanillaSwap, market: Market, bumps: CurveBumps) -> List[Dict[str, object]]:
    rows, matrix = key_rate_matrix([swap.pv], market, bumps=bumps)
    return [{**r, "position": float(v), "hedge": 0.0, "residual": float(v)} for r, v in zip(rows, matrix[:, 0])]


def _ladder_chart(ladder: List[Dict[str, object]], hedged: bool) -> go.Figure:
    labels = [r["label"] for r in ladder]
    swap_values = [r["position"] for r in ladder]
    fig = go.Figure(go.Bar(x=labels, y=swap_values, name="SWAP", marker=dict(color=SERIES[0])))
    values = [0.0, *swap_values]
    if hedged:
        residual = [r["residual"] for r in ladder]
        fig.add_trace(go.Bar(x=labels, y=residual, name="AFTER HEDGE", marker=dict(color=ACCENT)))
        values += residual
    dark_layout(fig, height=340, xtitle="CURVE QUOTE", ytitle="PV01 ($ PER +1BP)", legend=hedged)
    fig.update_layout(barmode="group")
    return bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(labels) - 0.5, "min_span": 1.0},
        yaxis=axis_limits(values, pad=0.1),
    )


def _hedge_tables(hedge: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.DataFrame(
        {
            "BENCHMARK": [h["benchmark"] for h in hedge["hedges"]],
            "DIRECTION": [h["direction"] for h in hedge["hedges"]],
            "NOTIONAL ($)": [abs(h["notional"]) for h in hedge["hedges"]],
            "PAR RATE (%)": [h["par_rate"] * 100.0 for h in hedge["hedges"]],
            "PV01 ($, +1BP)": [h["pv01"] for h in hedge["hedges"]],
        }
    )
    ladder = pd.DataFrame(
        {
            "QUOTE": [r["label"] for r in hedge["ladder"]],
            "SWAP PV01 ($)": [r["position"] for r in hedge["ladder"]],
            "HEDGE PV01 ($)": [r["hedge"] for r in hedge["ladder"]],
            "RESIDUAL PV01 ($)": [r["residual"] for r in hedge["ladder"]],
        }
    )
    return rows, ladder


def _num(value: float, decimals: int = 2) -> str:
    """Thousands-separated number without a "-0.00" for tiny negatives."""

    return f"{round(value, decimals) + 0.0:,.{decimals}f}"


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _swap_inputs(market: Market, conv: CurveConventions) -> Optional[Tuple[VanillaSwap, float]]:
    """Read the inputs; returns (swap at its fixed rate, par rate) or None
    after showing what is wrong."""

    asof = market.asof
    c1, c2, c3 = st.columns(3)
    notional = c1.number_input(
        "NOTIONAL ($)", min_value=1e3, max_value=1e11, value=1e7, step=1e6, format="%.0f", key="swap_notional"
    )
    side = c2.radio("DIRECTION", ["Receive fixed", "Pay fixed"], horizontal=True, key="swap_side")
    kind = c3.radio(
        "CONVENTIONS",
        ["OIS", "IRS"],
        horizontal=True,
        key="swap_kind",
        help="OIS: compounded overnight rate against fixed, on the curve's OIS legs. "
        "IRS: term rate fixed in advance against fixed, on the curve's IRS legs.",
    )

    c1, c2, c3 = st.columns(3)
    end_mode = c1.radio("END", ["Tenor", "Maturity date"], horizontal=True, key="swap_end_mode")
    errors: List[str] = []
    tenor, maturity = None, None
    if end_mode == "Tenor":
        tenor, err = _tenor(c2.text_input("TENOR", "5Y", key="swap_tenor"), "Tenor", MAX_TENOR_YEARS)
        errors += [err] if err else []
    else:
        maturity = c2.date_input(
            "MATURITY",
            value=dt.date(asof.year + 5, asof.month, min(asof.day, 28)),
            min_value=asof + dt.timedelta(days=7),
            max_value=dt.date(asof.year + int(MAX_TENOR_YEARS), 12, 31),
            key="swap_maturity",
        )
    fwd_text = c3.text_input("FORWARD START (BLANK = SPOT)", "", key="swap_fwd").strip()
    forward_start = None
    if fwd_text:
        forward_start, err = _tenor(fwd_text, "Forward start", MAX_FORWARD_YEARS)
        errors += [err] if err else []

    c1, c2, c3 = st.columns(3)
    use_par = c1.checkbox("USE PAR RATE", value=True, key="swap_use_par")
    spread_bp = c3.number_input(
        "FLOAT SPREAD (BP)", min_value=-500.0, max_value=500.0, value=0.0, step=1.0, key="swap_spread_bp"
    )
    for err in errors:
        st.error(err)
    if errors:
        return None
    try:
        swap = VanillaSwap.from_conventions(
            conv,
            tenor,
            0.0,
            notional=float(notional),
            receive_fixed=side == "Receive fixed",
            kind=kind,
            forward_start=forward_start,
            spread=spread_bp * 1e-4,
            maturity=maturity,
            trade_date=asof,
        )
        par = swap.par_rate(market)
    except ValueError as exc:
        st.error(f"Cannot build this swap: {exc}")
        return None

    # The rate box shows the par rate while "use par" is ticked; set before
    # the widget is created, as Streamlit requires.
    lo, hi = RATE_BOUNDS_PCT
    if use_par or "swap_rate_pct" not in st.session_state:
        st.session_state["swap_rate_pct"] = min(max(par * 100.0, lo), hi)
    rate_pct = c2.number_input(
        "FIXED RATE (%)", min_value=lo, max_value=hi, step=0.01, format="%.4f", key="swap_rate_pct", disabled=use_par
    )
    if use_par and not lo <= par * 100.0 <= hi:
        st.warning(f"The par rate {par:.4%} is outside the {lo:.0f}% to {hi:.0f}% input range; priced at par anyway.")
    return replace(swap, fixed_rate=par if use_par else rate_pct / 100.0), par


def render(market: Market) -> List[Position]:
    """Swap pricer tab; returns the swap (and its hedges, if asked)."""

    st.markdown("### / SWAP")
    conv = market.curve.conventions or DEFAULT_CONVENTIONS
    built = _swap_inputs(market, conv)
    if built is None:
        return []
    swap, par = built
    positions = [swap_position(swap)]
    st.caption(_conventions_note(swap, conv, market))
    pillars = market.curve.pillar_dates
    if pillars and swap.periods(market.asof)[-1].pay > pillars[-1]:
        st.warning(f"The swap pays after the curve's last quote ({pillars[-1]}); forwards beyond it are held flat.")

    bumps, no_risk = _bumps(market)
    pv01 = None if bumps is None else bumps.parallel_pv01(swap.pv)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PV ($)", _num(swap.pv(market), 0))
    gap_bp = round((swap.fixed_rate - par) * 1e4, 2) + 0.0
    m2.metric("PAR RATE (%)", f"{par * 100:.4f}", delta=f"{gap_bp:+.2f} bp fixed - par", delta_color="off")
    m3.metric("ANNUITY ($ PER BP)", _num(swap.annuity(market)))
    m4.metric("PV01 ($, +1BP)", "n/a" if pv01 is None else _num(pv01))
    st.caption(
        "PV and every risk number are signed P&L to you. PV01 is the P&L for +1bp on every curve quote with the "
        "curve rebuilt (central difference): a receiver loses when rates rise. Annuity is the PV of 1bp a year on "
        "the remaining fixed leg."
    )

    st.markdown("#### / CASH FLOWS")
    table = _cashflow_table(swap, market)
    st.dataframe(table, hide_index=True, width="stretch", column_config=CASHFLOW_FORMATS)
    st.download_button("Download cash flows CSV", _csv(table), file_name="swap_cashflows.csv", key="swap_cf_csv")
    st.caption("Amounts are signed to you (+ received, - paid); float rates include the spread.")

    st.markdown("### / KEY-RATE PV01")
    if bumps is None:
        st.info(no_risk)
        return positions
    ladder_slot = st.container()

    st.markdown("### / HEDGE")
    tenors = st.multiselect(
        "BENCHMARK PAR SWAPS",
        HEDGE_TENORS,
        default=list(BENCHMARK_TENORS),
        key="swap_hedge_tenors",
        help="Spot par swaps with the conventions chosen above, sized to flatten the key-rate ladder.",
    )
    kind = "OIS" if swap.float_index == "OIS" else "IRS"
    hedge = None
    if tenors:
        hedge = _cached(
            "hedge",
            (swap, _curve_key(market), tuple(tenors)),
            lambda: hedge_with_benchmarks(swap_position(swap), market, tenors, kind=kind, bumps=bumps),
        )
        ladder = hedge["ladder"]
    else:
        ladder = _cached("ladder", (swap, _curve_key(market)), lambda: _ladder_only(swap, market, bumps))

    with ladder_slot:
        st.plotly_chart(
            _ladder_chart(ladder, hedge is not None), width="stretch", config=PLOTLY_2D_CONFIG, key="swap_ladder_chart"
        )
        st.caption(
            f"P&L for +1bp on one curve quote at a time. The ladder sums to "
            f"{_num(sum(r['position'] for r in ladder))} against a parallel PV01 of {_num(pv01)}."
        )

    if hedge is None:
        st.info("Pick at least one benchmark tenor to size a hedge.")
        return positions
    rows, ladder_table = _hedge_tables(hedge)
    h1, h2, h3 = st.columns(3)
    h1.metric("SWAP PV01 ($)", _num(hedge["position_pv01"]))
    h2.metric("HEDGED PV01 ($)", _num(hedge["hedged_pv01"]))
    h3.metric("LADDER RISK REMOVED (%)", f"{hedge['risk_removed'] * 100:.2f}")
    st.dataframe(rows, hide_index=True, width="stretch", column_config=HEDGE_FORMATS)
    st.download_button(
        "Download hedge CSV", _csv(rows) + b"\n" + _csv(ladder_table), file_name="swap_hedge.csv", key="swap_hedge_csv"
    )
    st.caption(
        "Least-squares notionals of the benchmark swaps that minimise the residual ladder (PV01 columns are ladder "
        "sums). Risk removed is the share of the squared ladder taken out; buckets between benchmarks cannot be "
        "hedged exactly."
    )
    if st.checkbox("ADD HEDGES TO THE PORTFOLIO", value=False, key="swap_add_hedges"):
        positions += hedge["positions"]
    return positions
