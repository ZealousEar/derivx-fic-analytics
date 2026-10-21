"""Swap tab of the Rates page: price one vanilla swap, show its cash flows
and key-rate risk, and size benchmark par swaps that hedge it.

On a multi-curve market (EUR with EURIBOR projection curves) the float leg
can be EURIBOR 6M or 3M, projected on its own curve and discounted on €STR;
key-rate risk is then bucketed per curve (discount and each projection) and
the hedge uses OIS swaps, swaps against EURIBOR and 3s6s basis swaps."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.rates.conventions import parse_tenor
from derivx.rates.curves import SPREAD_KINDS, CurveConventions, Market
from derivx.rates.swaps import (
    BENCHMARK_TENORS,
    DEFAULT_CONVENTIONS,
    CurveBumps,
    VanillaSwap,
    curve_bumps,
    hedge_multi_curve,
    hedge_set,
    hedge_with_benchmarks,
    key_rate_matrix,
    swap_position,
)
from derivx.risk.sensitivities import Position
from derivx.ui import rates_exports, rates_risk
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SERIES, dark_layout

HEDGE_TENORS = ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")
MAX_TENOR_YEARS = 50.0
MAX_FORWARD_YEARS = 30.0
RATE_BOUNDS_PCT = (-5.0, 25.0)
MONEY = st.column_config.NumberColumn(format="%,.2f")


def _cashflow_formats(c: str) -> Dict[str, object]:
    """Column formats of the cash-flow table, amounts in currency ``c``."""

    return {
        "ACCRUAL (YRS)": st.column_config.NumberColumn(format="%.6f"),
        "RATE (%)": st.column_config.NumberColumn(format="%.4f"),
        f"AMOUNT ({c})": MONEY,
        "DF": st.column_config.NumberColumn(format="%.8f"),
        f"PV ({c})": MONEY,
    }


def _hedge_formats(c: str) -> Dict[str, object]:
    return {
        f"NOTIONAL ({c})": st.column_config.NumberColumn(format="%,.0f"),
        "PAR RATE (%)": st.column_config.NumberColumn(format="%.4f"),
        f"PV01 ({c}, +1BP)": MONEY,
        f"BASIS PV01 ({c}, +1BP)": MONEY,
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
    elif swap.projection:
        index = f"{swap.projection} fixed {swap.fixing_lag} business days before each period"
    else:
        index = f"term rate fixed {swap.fixing_lag} business days before each period"
    lag = f", both legs paid {swap.payment_lag} business days after each period end" if swap.payment_lag else ""
    if swap.projection:
        curves = f"The {swap.projection} curve projects the float leg; the {conv.name} curve discounts both legs."
    elif market.is_multi_curve and swap.float_index == "TERM":
        curves = "The term leg projects on the OIS curve (single-curve approximation); it discounts both legs."
    else:
        curves = "One curve projects and discounts both legs."
    name = swap.projection or conv.name
    return (
        f"{name} conventions: start {fixed_dates[0]}, maturity {fixed_dates[-1]}; fixed leg {swap.fixed_months}M "
        f"{swap.fixed_day_count}; float leg {index}, {swap.float_months}M {swap.float_day_count}{lag}; "
        f"{swap.calendar} calendar, {swap.bdc}. {curves}"
    )


def _cashflow_table(swap: VanillaSwap, market: Market) -> pd.DataFrame:
    flows = swap.cashflows(market)
    c = rates_risk.money(market)
    return pd.DataFrame(
        {
            "LEG": flows["leg"].str.upper(),
            "START": flows["start"],
            "END": flows["end"],
            "PAY": flows["pay"],
            "ACCRUAL (YRS)": flows["accrual"],
            "RATE (%)": flows["rate"] * 100.0,
            f"AMOUNT ({c})": flows["amount"],
            "DF": flows["df"],
            f"PV ({c})": flows["pv"],
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


def _ladder_chart(ladder: List[Dict[str, object]], hedged: bool, c: str = "$") -> go.Figure:
    labels = [r["label"] for r in ladder]
    swap_values = [r["position"] for r in ladder]
    fig = go.Figure(go.Bar(x=labels, y=swap_values, name="SWAP", marker=dict(color=SERIES[0])))
    values = [0.0, *swap_values]
    if hedged:
        residual = [r["residual"] for r in ladder]
        fig.add_trace(go.Bar(x=labels, y=residual, name="AFTER HEDGE", marker=dict(color=ACCENT)))
        values += residual
    dark_layout(fig, height=340, xtitle="CURVE QUOTE", ytitle=f"PV01 ({c} PER +1BP)", legend=hedged)
    fig.update_layout(barmode="group")
    return bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(labels) - 0.5, "min_span": 1.0},
        yaxis=axis_limits(values, pad=0.1),
    )


def _hedge_tables(hedge: Dict[str, object], c: str = "$") -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.DataFrame(
        {
            "BENCHMARK": [h["benchmark"] for h in hedge["hedges"]],
            "DIRECTION": [h["direction"] for h in hedge["hedges"]],
            f"NOTIONAL ({c})": [abs(h["notional"]) for h in hedge["hedges"]],
            "PAR RATE (%)": [h["par_rate"] * 100.0 for h in hedge["hedges"]],
            f"PV01 ({c}, +1BP)": [h["pv01"] for h in hedge["hedges"]],
        }
    )
    ladder = pd.DataFrame(
        {
            "QUOTE": [r["label"] for r in hedge["ladder"]],
            f"SWAP PV01 ({c})": [r["position"] for r in hedge["ladder"]],
            f"HEDGE PV01 ({c})": [r["hedge"] for r in hedge["ladder"]],
            f"RESIDUAL PV01 ({c})": [r["residual"] for r in hedge["ladder"]],
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
        f"NOTIONAL ({rates_risk.money(market)})", min_value=1e3, max_value=1e11, value=1e7, step=1e6, format="%.0f",
        key="swap_notional",
    )  # fmt: skip
    side = c2.radio("DIRECTION", ["Receive fixed", "Pay fixed"], horizontal=True, key="swap_side")
    indices = [k for k in market.projections if not market.projections.is_discount(k)]
    kind = c3.radio(
        "CONVENTIONS",
        ["OIS", "IRS", *indices],
        horizontal=True,
        key="swap_kind",
        help="OIS: compounded overnight rate against fixed, on the curve's OIS legs. "
        "IRS: term rate fixed in advance against fixed, on the curve's IRS legs, projected on the OIS curve "
        "(single-curve). EURIBOR 6M/3M: fixed annual 30/360 against EURIBOR, projected on its own curve and "
        "discounted on OIS (multi-curve).",
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
        terms = dict(
            notional=float(notional),
            receive_fixed=side == "Receive fixed",
            forward_start=forward_start,
            spread=spread_bp * 1e-4,
            maturity=maturity,
            trade_date=asof,
        )
        if kind in indices:
            swap = VanillaSwap.from_index(kind, tenor, 0.0, **terms)
        else:
            swap = VanillaSwap.from_conventions(conv, tenor, 0.0, kind=kind, **terms)
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
    exports: Dict[str, pd.DataFrame] = {}
    positions = _render(market, exports)
    if positions:
        swap = positions[0].details
        rates_exports.export_row(
            "swap", market, exports, {"swap": swap, "hedges_added": len(positions) - 1},
            {"pricing": "OIS discounting; OIS legs compounded on the OIS curve, term legs on their projection curve",
             "risk": "+/-1bp per curve quote, curves rebuilt, central difference"},
            key="swap_export",
        )  # fmt: skip
    return positions


def _render(market: Market, exports: Dict[str, pd.DataFrame]) -> List[Position]:
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

    if market.is_multi_curve:
        return _render_multi_curve(swap, par, market, positions, exports)
    bumps, no_risk = _bumps(market)
    pv01 = None if bumps is None else bumps.parallel_pv01(swap.pv)
    c = rates_risk.money(market)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"PV ({c})", _num(swap.pv(market), 0))
    gap_bp = round((swap.fixed_rate - par) * 1e4, 2) + 0.0
    m2.metric("PAR RATE (%)", f"{par * 100:.4f}", delta=f"{gap_bp:+.2f} bp fixed - par", delta_color="off")
    m3.metric(f"ANNUITY ({c} PER BP)", _num(swap.annuity(market)))
    m4.metric(f"PV01 ({c}, +1BP)", "n/a" if pv01 is None else _num(pv01))
    st.caption(
        "PV and every risk number are signed P&L to you. PV01 is the P&L for +1bp on every curve quote with the "
        "curve rebuilt (central difference): a receiver loses when rates rise. Annuity is the PV of 1bp a year on "
        "the remaining fixed leg."
    )

    st.markdown("#### / CASH FLOWS")
    table = _cashflow_table(swap, market)
    exports["cashflows"] = table
    st.dataframe(table, hide_index=True, width="stretch", column_config=_cashflow_formats(c))
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

    exports["key_rate_pv01"] = pd.DataFrame(ladder)
    with ladder_slot:
        st.plotly_chart(
            _ladder_chart(ladder, hedge is not None, c),
            width="stretch",
            config=PLOTLY_2D_CONFIG,
            key="swap_ladder_chart",
        )
        st.caption(
            f"P&L for +1bp on one curve quote at a time. The ladder sums to "
            f"{_num(sum(r['position'] for r in ladder))} against a parallel PV01 of {_num(pv01)}."
        )

    if hedge is None:
        st.info("Pick at least one benchmark tenor to size a hedge.")
        return positions
    rows, ladder_table = _hedge_tables(hedge, c)
    exports["hedge"] = rows
    h1, h2, h3 = st.columns(3)
    h1.metric(f"SWAP PV01 ({c})", _num(hedge["position_pv01"]))
    h2.metric(f"HEDGED PV01 ({c})", _num(hedge["hedged_pv01"]))
    h3.metric("LADDER RISK REMOVED (%)", f"{hedge['risk_removed'] * 100:.2f}")
    st.dataframe(rows, hide_index=True, width="stretch", column_config=_hedge_formats(c))
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


# --------------------------------------------------------------------------
# Multi-curve
# --------------------------------------------------------------------------


def _slug(key: str) -> str:
    return key.lower().replace(" ", "_")


def _kinds(market: Market, key: str) -> List[str]:
    return [inst.kind for inst in market.get_curve(key).instruments]


def _default_benchmarks(market: Market, swap: VanillaSwap, key: str) -> List[str]:
    """OIS benchmarks always; on a projection curve only where the swap
    projects (its index and the curve it rides on)."""

    if key == "curve":
        return list(BENCHMARK_TENORS)
    chain, index = [], swap.projection
    while index:
        chain.append(index)
        index = getattr(market.get_curve(index), "basis_index", None)
    return list(BENCHMARK_TENORS) if key in chain else []


def _render_multi_curve(
    swap: VanillaSwap, par: float, market: Market, positions: List[Position], exports: Dict[str, pd.DataFrame]
) -> List[Position]:
    """Metrics, per-curve ladders and the multi-curve hedge."""

    curves = rates_risk.risk_keys(market)
    try:
        ladders = rates_risk.ladders(market, swap.pv, ("swap", swap), curves)
        pv01 = rates_risk.rate_pv01(market, swap.pv, ("swap", swap))
    except ValueError as exc:
        st.error(f"Cannot rebuild the curves with bumped quotes, so risk is not available: {exc}")
        return positions
    c = rates_risk.money(market)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"PV ({c})", _num(swap.pv(market), 0))
    gap_bp = round((swap.fixed_rate - par) * 1e4, 2) + 0.0
    m2.metric("PAR RATE (%)", f"{par * 100:.4f}", delta=f"{gap_bp:+.2f} bp fixed - par", delta_color="off")
    m3.metric(f"ANNUITY ({c} PER BP)", _num(swap.annuity(market)))
    m4.metric(f"PV01 ({c}, +1BP)", _num(pv01))
    st.caption(
        "PV and every risk number are signed P&L to you. PV01 is the P&L for +1bp on every rate quote of every "
        "curve (the OIS curve and each EURIBOR curve, rebuilt; tenor basis spreads held), central difference: a "
        "receiver loses when rates rise. Annuity is the PV of 1bp a year on the remaining fixed leg, on OIS "
        "discounting."
    )

    st.markdown("#### / CASH FLOWS")
    table = _cashflow_table(swap, market)
    exports["cashflows"] = table
    exports["key_rate_pv01"] = pd.DataFrame(
        [{"curve": key, **{k: r[k] for k in ("label", "tenor", "years")}, "pv01": float(v)}
         for key in curves for r, v in zip(*ladders[key])]
    )  # fmt: skip
    st.dataframe(table, hide_index=True, width="stretch", column_config=_cashflow_formats(c))
    st.download_button("Download cash flows CSV", _csv(table), file_name="swap_cashflows.csv", key="swap_cf_csv")
    st.caption("Amounts are signed to you (+ received, - paid); float rates include the spread. DF is the OIS curve.")

    st.markdown("### / KEY-RATE PV01 BY CURVE")
    ladder_slot = st.container()

    st.markdown("### / HEDGE")
    st.caption(
        "Par instruments per curve, sized together to flatten every curve's ladder: OIS swaps on the discount "
        "curve, swaps against EURIBOR on a curve built from swap quotes, 3s6s basis swaps on a curve built from "
        "basis quotes."
    )
    benchmarks: Dict[str, List[str]] = {}
    cols = st.columns(len(curves))
    for col, key in zip(cols, curves):
        # Projection benchmarks default to the swap's own index chain: the key follows the index so a new
        # index starts from its defaults.
        widget = "swap_hedge_tenors" if key == "curve" else f"swap_hedge_{_slug(key)}_{_slug(swap.projection or 'ois')}"
        kind = "OIS SWAPS" if key == "curve" else (
            "3S6S BASIS SWAPS" if getattr(market.get_curve(key), "basis_index", None) else f"SWAPS VS {key}"
        )  # fmt: skip
        picked = col.multiselect(f"{rates_risk.curve_title(market, key)}: {kind}", HEDGE_TENORS,
                                 default=_default_benchmarks(market, swap, key), key=widget)  # fmt: skip
        if picked:
            benchmarks[key] = picked
    hedge = None
    if benchmarks:
        frozen = tuple((k, tuple(v)) for k, v in benchmarks.items())
        try:
            bumps = {k: rates_risk.bumps(market, k) for k in curves}
            units = rates_risk.cached("hedge_set", (rates_risk.market_key(market), frozen),
                                      lambda: hedge_set(market, benchmarks, bumps=bumps))  # fmt: skip
            hedge = hedge_multi_curve(swap_position(swap), market, hedges=units, ladders=ladders)
        except ValueError as exc:
            st.error(f"Cannot size the hedge: {exc}")

    with ladder_slot:
        shown = 0
        for key in curves:
            rows, values = ladders[key]
            if key != "curve" and not rates_risk.loads_on(values):
                continue
            residual = [r["residual"] for r in hedge["ladders"][key]] if hedge else None
            st.markdown(f"#### {rates_risk.curve_title(market, key)}")
            chart_key = "swap_ladder_chart" if key == "curve" else f"swap_ladder_chart_{_slug(key)}"
            st.plotly_chart(rates_risk.ladder_chart([r["label"] for r in rows], values, residual, name="SWAP",
                                                    unit=c), width="stretch", config=PLOTLY_2D_CONFIG,
                            key=chart_key)  # fmt: skip
            shown += 1
        skipped = [k for k in curves if k != "curve" and not rates_risk.loads_on(ladders[k][1])]
        rate_sum, spread_sum = rates_risk.ladder_sums(market, ladders)
        st.caption(
            "P&L for +1bp on one quote of one curve at a time (central difference). Bumping an OIS quote rebuilds "
            "the EURIBOR curves on the moved OIS curve with their own quotes held, so the discount ladder carries "
            f"the trade's whole response to OIS. The rate-quote buckets sum to {_num(rate_sum)} {c} (the PV01 above)"
            + (f"; the 3s6s basis-spread buckets (+1bp on a spread, not a rate) to {_num(spread_sum)} {c}"
               if any(k in SPREAD_KINDS for key in curves for k in _kinds(market, key)) else "")
            + (f"; no risk to {', '.join(skipped)}." if skipped else ".")
        )  # fmt: skip

    if hedge is None:
        if not benchmarks:
            st.info("Pick at least one benchmark tenor to size a hedge.")
        return positions
    is_basis = ["fixed_rate" not in u.details for _, _, u in units.units]
    rows = pd.DataFrame(
        {
            "CURVE": [rates_risk.curve_title(market, h["curve"]) for h in hedge["hedges"]],
            "BENCHMARK": [h["benchmark"] for h in hedge["hedges"]],
            "DIRECTION": [h["direction"] for h in hedge["hedges"]],
            f"NOTIONAL ({c})": [abs(h["notional"]) for h in hedge["hedges"]],
            "PAR RATE (%)": [np.nan if b else h["par"] * 100.0 for h, b in zip(hedge["hedges"], is_basis)],
            "PAR SPREAD (BP)": [h["par"] * 1e4 if b else np.nan for h, b in zip(hedge["hedges"], is_basis)],
            f"PV01 ({c}, +1BP)": [h["pv01"] for h in hedge["hedges"]],
            f"BASIS PV01 ({c}, +1BP)": [h["basis_pv01"] for h in hedge["hedges"]],
        }
    )
    if not any(is_basis):
        rows = rows.drop(columns=["PAR SPREAD (BP)", f"BASIS PV01 ({c}, +1BP)"])
    rate_rows = [r for rows_k in hedge["ladders"].values() for r in rows_k if r["kind"] not in SPREAD_KINDS]
    h1, h2, h3 = st.columns(3)
    h1.metric(f"SWAP PV01 ({c})", _num(sum(r["position"] for r in rate_rows)))
    h2.metric(f"HEDGED PV01 ({c})", _num(sum(r["residual"] for r in rate_rows)))
    h3.metric("LADDER RISK REMOVED (%)", f"{hedge['risk_removed'] * 100:.2f}")
    formats = {**_hedge_formats(c), "PAR SPREAD (BP)": st.column_config.NumberColumn(format="%.2f")}
    st.dataframe(rows, hide_index=True, width="stretch", column_config=formats)
    exports["hedge"] = rows
    st.download_button("Download hedge CSV", _csv(rows), file_name="swap_hedge.csv", key="swap_hedge_csv")
    st.caption(
        "Least-squares notionals that minimise the stacked residual ladder of every curve (every quote of every "
        "curve, rate and basis buckets alike). PV01 is a hedge's P&L for +1bp on every rate quote (basis spreads "
        "held); BASIS PV01 for +1bp on every 3s6s spread. SWAP / HEDGED PV01 are the rate-quote ladder sums before "
        "and after the hedge."
    )
    if st.checkbox("ADD HEDGES TO THE PORTFOLIO", value=False, key="swap_add_hedges"):
        positions += hedge["positions"]
    return positions
