"""CAP / FLOOR tab: caps, floors and collars priced with the Bachelier
(normal), Black or shifted Black model.

The rate is the curve's overnight rate (SOFR, €STR, SONIA) compounded in
arrears, the way USD and GBP caps trade since LIBOR ended, with the
Lyashenko-Mercurio vol (or an effective vol to the last fixing); or a term
rate. On a multi-curve market (EUR) the term rate is EURIBOR 3M or 6M:
forwards from its projection curve, discounting on €STR, and key-rate risk
bucketed per curve. On a single-curve market a 3M/6M term rate projects on
the OIS curve: an approximation, labelled as one (no USD or GBP term-rate
cap market or curve).

Vols are one flat vol, a term structure, or caplet vols stripped from flat
cap vol quotes (ATM or by strike; ILLUSTRATIVE defaults, editable) so that
every quoted cap reprices. Risk: PV01 and gamma for +1bp on every rate
quote, vega in total and by caplet expiry (and by quoted cap when stripped),
and a one-day theta."""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from derivx.rates.capfloor import OVERNIGHT_NAMES, CapFloor, cap_floor_parity, capfloor_position
from derivx.rates.capvol import CapletVolSurface, quote_vegas, strip_caplet_vols, strip_surface, vega_by_expiry
from derivx.rates.conventions import parse_tenor
from derivx.rates.curves import Market
from derivx.rates.portfolio import cap_terms
from derivx.rates.vol_examples import ILLUSTRATIVE, illustrative_cap_atm, illustrative_cap_strikes
from derivx.risk.scenarios import theta
from derivx.risk.sensitivities import Position, key_rate_pv01, rate_pv01_gamma
from derivx.ui import rates_data, rates_exports, rates_risk
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SERIES, dark_layout
from derivx.vol.sabr import calibrate_sabr_smile, sabr_vol

MODEL_LABELS = {"Normal (bp)": "bachelier", "Black (%)": "black", "Shifted Black (%)": "shifted_black"}
VOL_MODES = ["Flat", "Term structure", "Stripped from cap quotes"]
STRIP_METHODS = {"Piecewise constant": "constant", "Linear": "linear"}
SMILE_SHIFT = 0.03  # shift of the normal SABR fitted to each caplet smile
MONEY_2DP = st.column_config.NumberColumn(format="%,.2f")
TENORS = ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]
BOOK_QUOTES_KEY = "_cap_book_quotes"  # session state: the cap quotes in use, for the BOOK tab
FORWARD_STARTS = ["Spot", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y"]
KINDS = ["Cap", "Floor", "Collar"]
TERM_CHOICES = ["3M term", "6M term"]  # single-curve term rates on the OIS curve (approximations)
RESETS = ["3M", "1M", "6M", "12M"]  # compounded periods
RFR_VOLS = {"Forward-looking (Lyashenko-Mercurio)": False, "Effective to last fixing": True}
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
    "OPTION TIME (Y)": st.column_config.NumberColumn(format="%.4f"),
    "TAU": st.column_config.NumberColumn(format="%.6f"),
    "FORWARD (%)": st.column_config.NumberColumn(format="%.4f"),
    "STRIKE (%)": st.column_config.NumberColumn(format="%.4f"),
    "FLOOR STRIKE (%)": st.column_config.NumberColumn(format="%.4f"),
    "DF": st.column_config.NumberColumn(format="%.8f"),
    "PV": st.column_config.NumberColumn(format="%.2f"),
    "CAP PV": st.column_config.NumberColumn(format="%.2f"),
    "FLOOR PV": st.column_config.NumberColumn(format="%.2f"),
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


def _caplet_view(table: pd.DataFrame, model: str, compounded: bool = False) -> pd.DataFrame:
    """The caplet table in display units: a compounded rate adds its first
    daily fixing, option (variance) time and effective vol; a collar its
    floor strike, floor vol and the PVs of both legs."""

    label, unit, _ = VOL_UNITS[model]
    per = "1BP" if model == "bachelier" else "VOL PT"
    cols: Dict[str, object] = {}
    if compounded:
        cols["FIRST FIXING"] = table["first_fix"]
    cols.update({"LAST FIXING" if compounded else "FIXING": table["fix"], "START": table["start"], "END": table["end"],
                 "PAY": table["pay"], "EXPIRY (Y)": table["expiry"]})  # fmt: skip
    if compounded:
        cols["OPTION TIME (Y)"] = table["variance_time"]
    cols.update({"TAU": table["tau"], "FORWARD (%)": table["forward"] * 100, "STRIKE (%)": table["strike"] * 100,
                 f"VOL ({label})": table["vol"] / unit})  # fmt: skip
    if compounded:
        cols[f"EFF VOL ({label})"] = table["effective_vol"] / unit
    if "floor_strike" in table:
        cols.update({"FLOOR STRIKE (%)": table["floor_strike"] * 100, f"FLOOR VOL ({label})": table["floor_vol"] / unit,
                     "CAP PV": table["cap_pv"], "FLOOR PV": table["floor_pv"]})  # fmt: skip
    cols.update({"DF": table["df"], "PV": table["pv"], f"VEGA (/{per})": table["vega"],
                 "STATUS": np.where(table["fixed"], "fixed", "live")})  # fmt: skip
    return pd.DataFrame(cols)


def _date_limits(days: List[dt.date]) -> Dict[str, str]:
    pad = dt.timedelta(days=max(15, (max(days) - min(days)).days // 40))
    return {"min": (min(days) - pad).isoformat(), "max": (max(days) + pad).isoformat()}


def _strip_chart(table: pd.DataFrame, strike: float, kind: str, floor_strike: Optional[float] = None) -> go.Figure:
    """Forward against the strike(s) by (last) fixing date, and each period's PV."""

    fixes = list(table["fix"])
    forwards = (table["forward"] * 100).tolist()
    pvs = table["pv"].tolist()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.55, 0.45],
        subplot_titles=("FORWARD VS STRIKE (%)", f"{kind.upper()}LET PV"),
    )
    fig.add_trace(
        go.Scatter(x=fixes, y=forwards, mode="lines+markers", name="FORWARD", line=dict(color=SERIES[0], width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fixes,
            y=[strike * 100] * len(fixes),
            mode="lines" if len(fixes) > 1 else "markers",
            name="STRIKE",
            line=dict(color=ACCENT, width=2, dash="dash"),
            marker=dict(color=ACCENT, symbol="line-ew-open", size=14),
        ),
        row=1,
        col=1,
    )
    levels = [*forwards, strike * 100]
    if floor_strike is not None:
        fig.data[-1].name = "CAP STRIKE"
        fig.add_trace(
            go.Scatter(
                x=fixes,
                y=[floor_strike * 100] * len(fixes),
                mode="lines" if len(fixes) > 1 else "markers",
                name="FLOOR STRIKE",
                line=dict(color=SERIES[3], width=2, dash="dot"),
                marker=dict(color=SERIES[3], symbol="line-ew-open", size=14),
            ),
            row=1,
            col=1,
        )
        levels.append(floor_strike * 100)
    fig.add_trace(go.Bar(x=fixes, y=pvs, name="PV", marker=dict(color=SERIES[2])), row=2, col=1)
    dark_layout(fig, height=520)
    fig.update_xaxes(title_text="FIXING DATE (LAST FIXING IF COMPOUNDED)", row=2, col=1)
    fig.update_yaxes(title_text="RATE (%)", row=1, col=1)
    fig.update_yaxes(title_text="PV", row=2, col=1)
    dates = _date_limits(fixes)
    bound_axes(
        fig,
        xaxis=dates,
        xaxis2=dates,
        yaxis=axis_limits(levels, pad=0.1),
        yaxis2=axis_limits([0.0, *pvs], pad=0.1, floor=min(0.0, *pvs)),
    )
    return fig


def _ladder_chart(rows: List[dict]) -> go.Figure:
    labels = [r["label"] for r in rows]
    values = [r["pv01"] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            name="PV01",
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


def _currency(market: Market) -> str:
    conv = market.curve.conventions
    return conv.currency if conv is not None else "USD"


def _maturity(text: object, row: int) -> str:
    label = str(text).strip().upper()
    tenor = parse_tenor(label)
    if tenor.months is None:
        raise ValueError(f"row {row}: cap maturity {text!r} must be months or years (e.g. 18M, 5Y)")
    return label


def _strike_column(name: str) -> float:
    try:
        return float(str(name).replace("%", "").strip()) / 100.0
    except ValueError:
        raise ValueError(f"column {name!r} is not a strike in % (e.g. 3.50%)") from None


def _quote_tables(market: Market, layout: str):
    """Editable flat cap vol quotes (normal bp); returns (maturities, flat
    vols decimal, strikes decimal or None for ATM, whether the user edited
    the defaults)."""

    ccy = _currency(market)
    if layout == "ATM":
        default = illustrative_cap_atm(ccy).rename(columns={"maturity": "MATURITY", "vol_bp": "FLAT VOL (BP)"})
        edited = st.data_editor(
            default, num_rows="dynamic", hide_index=True, width="stretch", key=f"cap_strip_atm_{ccy}",
            column_config={"FLAT VOL (BP)": st.column_config.NumberColumn(min_value=0.0, max_value=2000.0,
                                                                          format="%.1f")},
        )  # fmt: skip
        rows = edited.dropna()
        if rows.empty:
            raise ValueError("the cap quote table needs at least one maturity with a vol")
        mats = [_maturity(m, i + 1) for i, m in enumerate(rows["MATURITY"])]
        return mats, rows["FLAT VOL (BP)"].astype(float).to_numpy() * 1e-4, None, not edited.equals(default)
    default = illustrative_cap_strikes(ccy).rename(columns={"maturity": "MATURITY"})
    edited = st.data_editor(default, num_rows="dynamic", hide_index=True, width="stretch", key=f"cap_strip_grid_{ccy}")
    rows = edited.dropna(how="all", subset=list(edited.columns[1:]))
    if rows.empty:
        raise ValueError("the cap quote grid needs at least one maturity row")
    mats = [_maturity(m, i + 1) for i, m in enumerate(rows["MATURITY"])]
    strikes = np.array([_strike_column(c) for c in rows.columns[1:]])
    grid = rows.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) * 1e-4
    return mats, grid, strikes, not edited.equals(default)


def _quote_terms(terms: Dict[str, object]) -> Dict[str, object]:
    """Schedule of the quoted caps: the trade's index, calendar and day
    count, but spot-start (cap vols are quoted on spot-start caps, whatever
    forward start the trade has)."""

    return {k: v for k, v in terms.items() if k != "forward_start"}


def _strip(market: Market, mats, vols, strikes, method: str, terms: Dict[str, object]):
    """(stripped vols, strip results), cached per market and quotes."""

    key = (rates_risk.market_key(market), tuple(mats), tuple(np.ravel(vols).tolist()),
           None if strikes is None else tuple(np.ravel(strikes).tolist()), method,
           tuple(sorted(terms.items())))  # fmt: skip

    def compute():
        if strikes is None:
            result = strip_caplet_vols(market, mats, vols, None, method, terms=terms)
            return result.curve, [result]
        return strip_surface(market, mats, strikes, vols, method, terms=terms)

    return rates_risk.cached("cap_strip", key, compute)


def _stripped_inputs(market: Market, terms: Dict[str, object]) -> Dict[str, object]:
    """Quote editor and stripping settings; returns the stripped vols (a
    curve for ATM quotes, a surface by strike) and what they came from.
    ``terms`` are the quoted caps' schedule (:func:`_quote_terms`). The
    quotes in use are kept for the BOOK tab (:func:`book_caplets`)."""

    st.markdown("#### / CAP VOL QUOTES")
    c1, c2 = st.columns(2)
    layout = c1.radio("QUOTES", ["ATM", "By strike"], horizontal=True, key="cap_strip_layout")
    method = STRIP_METHODS[c2.radio("CAPLET VOLS", list(STRIP_METHODS), horizontal=True, key="cap_strip_method")]
    caps = (
        "spot-start caps on the compounded overnight rate, every caplet included, its vol read the same way as the "
        "trade's (forward-looking or effective)"
        if terms.get("rate_type") == "compounded" else "spot-start caps, first caplet excluded"
    )  # fmt: skip
    st.caption(
        f"Flat normal cap vols (bp) by cap maturity on the selected index: {caps}. "
        f"Defaults are {ILLUSTRATIVE}; edit the table or paste your broker's quotes."
    )
    currency = _currency(market)
    try:
        mats, vols, strikes, edited = _quote_tables(market, layout)
        source = f"{ILLUSTRATIVE}: {currency} flat cap normal vols" + (" (edited by you)" if edited else "")
        vol, results = _strip(market, mats, vols, strikes, method, terms)
    except ValueError as exc:
        st.session_state[BOOK_QUOTES_KEY] = {"currency": currency, "error": f"the CAP-FLOOR tab's cap quotes: {exc}"}
        raise
    st.session_state[BOOK_QUOTES_KEY] = {"currency": currency, "maturities": tuple(mats), "flat": np.array(vols),
                                         "strikes": strikes, "method": method, "source": source}  # fmt: skip
    return {"vol": vol, "results": results, "layout": layout, "method": method, "maturities": mats, "flat": vols,
            "strikes": strikes, "source": source}  # fmt: skip


def book_caplets(market: Market, key: str) -> Tuple[object, str, tuple]:
    """Caplet vols the BOOK prices caps and floors without a flat vol on,
    for cap index ``key`` (``"3M"`` or a term index): stripped from the cap
    quotes last used on this tab, or the ILLUSTRATIVE ATM quotes. Returns
    (vols, quotes source, cache key); raises ValueError when the quotes do
    not strip."""

    currency = _currency(market)
    state = st.session_state.get(BOOK_QUOTES_KEY)
    if state is not None and state.get("currency") == currency and state.get("error"):
        raise ValueError(state["error"])
    if state is None or state.get("currency") != currency:
        table = illustrative_cap_atm(currency)
        mats, vols, strikes, method = list(table["maturity"]), table["vol_bp"].to_numpy() * 1e-4, None, "constant"
        source = f"{ILLUSTRATIVE}: {currency} flat ATM cap normal vols"
    else:
        mats, vols, strikes = list(state["maturities"]), state["flat"], state["strikes"]
        method, source = state["method"], state["source"]
    terms = cap_terms(key, market)
    vol, _ = _strip(market, mats, vols, strikes, method, terms)
    cache_key = (tuple(mats), tuple(np.ravel(vols).tolist()), None if strikes is None else
                 tuple(np.ravel(strikes).tolist()), method, key)  # fmt: skip
    return vol, source, cache_key


def _show_strip(stripped: Dict[str, object], cf: CapFloor, market: Market, terms: Dict[str, object]) -> None:
    """Stripped caplet vols, repricing table, caplet smile and quote vegas;
    ``terms`` are the quoted (spot-start) caps' schedule."""

    results, method, source = stripped["results"], stripped["method"], stripped["source"]
    st.markdown("### / STRIPPED CAPLET VOLS")
    shown = results
    if stripped["strikes"] is not None:  # one strike at a time: eight overlaid step curves are unreadable
        labels = {f"{r.curve.strike:.2%}": r for r in results}
        nearest = int(np.argmin([abs(r.curve.strike - cf.strike) for r in results]))
        pick = st.selectbox("CAPLET VOLS AT STRIKE", list(labels), index=nearest, key="cap_strip_strike")
        shown = [labels[pick]]
    st.plotly_chart(_vol_strip_chart(shown, method, source, market.asof), width="stretch", config=PLOTLY_2D_CONFIG,
                    key="cap_vol_strip_chart")  # fmt: skip
    rows = []
    for result in results:
        for r in result.table.itertuples():
            rows.append({"MATURITY": r.maturity, "STRIKE (%)": r.strike * 100, "FLAT VOL (BP)": r.flat_vol * 1e4,
                         "LAST CAPLET EXPIRY (Y)": r.node, "CAPLETS": r.caplets,
                         "QUOTED PV (BP UPFRONT)": r.target_pv * 1e4, "STRIPPED PV (BP UPFRONT)": r.stripped_pv * 1e4,
                         "ERROR (BP UPFRONT)": r.error * 1e4})  # fmt: skip
    table = pd.DataFrame(rows)
    fmt = {c: st.column_config.NumberColumn(format="%.6f") for c in table.columns if c not in ("MATURITY", "CAPLETS")}
    st.dataframe(table, hide_index=True, width="stretch", column_config=fmt)
    worst = max(r.max_error for r in results) * 1e4
    rule = ("one caplet vol for the caplets each quote adds (piecewise constant in expiry)" if method == "constant"
            else "caplet vols linear in expiry between the quotes' last caplets (flat before the first)")  # fmt: skip
    st.caption(
        f"Bootstrapped maturity by maturity, {rule}, so every quoted cap reprices at its flat vol (largest error "
        f"{worst:.1e}bp upfront). "
        + ("ATM quotes strip on each cap's own ATM strike: the curve is the caplet vol at those strikes, with no "
           "smile between them." if stripped["strikes"] is None else
           "Each strike is stripped on its own; between strikes the caplet vol is linear in strike, flat outside.")
    )  # fmt: skip
    if stripped["strikes"] is not None:
        surface = stripped["vol"]
        nodes = list(surface.nodes)
        labels = {f"{m} ({t:.2f}Y)": t for m, t in zip(results[0].table["maturity"], nodes)}
        choice = st.selectbox("SMILE AT CAPLET EXPIRY", list(labels), index=min(4, len(labels) - 1),
                              key="cap_smile_expiry")  # fmt: skip
        expiry = labels[choice]
        longest = CapFloor(1.0, "cap", 0.03, results[0].table["maturity"].iloc[-1], 0.01, **terms).caplets(market)
        forward = float(longest.iloc[int(np.argmin(np.abs(longest["expiry"].to_numpy() - expiry)))]["forward"])
        fig, fit_note = _smile_chart(surface, expiry, forward, source, market.asof)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_2D_CONFIG, key="cap_smile_chart")
        st.caption(
            f"Caplet vols stripped at each quoted strike for the caplet fixing {expiry:.2f}Y ahead"
            + (f", and a normal SABR (beta 0, shift {SMILE_SHIFT:.0%}) fitted to them" if not fit_note else "")
            + "; the diamond is that caplet's forward."
            + (f" No SABR fit is drawn: {fit_note}." if fit_note else "")
        )
    elif st.checkbox("SHOW VEGA BY QUOTED CAP (RE-STRIPS EACH QUOTE)", value=False, key="cap_quote_vega"):
        key: tuple = (rates_risk.market_key(market), repr(cf), tuple(stripped["maturities"]),
               tuple(np.ravel(stripped["flat"]).tolist()), method)  # fmt: skip
        try:
            vegas = rates_risk.cached("cap_quote_vega", key, lambda: quote_vegas(
                cf, market, stripped["maturities"], stripped["flat"], None, method, terms))  # fmt: skip
        except ValueError as err:
            st.warning(f"No vega by quoted cap: {err}")
            return
        view = vegas.rename(columns={"maturity": "QUOTED CAP", "vega": "VEGA (PER +1BP FLAT VOL)",
                                     "difference": "DIFFERENCE"})  # fmt: skip
        st.dataframe(view, hide_index=True, width="stretch", column_config={"VEGA (PER +1BP FLAT VOL)": MONEY_2DP})
        one_sided = [r.maturity for r in vegas.itertuples() if r.difference != "central"]
        st.caption(
            f"P&L for +1bp on one quoted flat vol, caplet vols re-stripped (central difference). Sum "
            f"{vegas['vega'].sum():,.2f}; quotes beyond the cap's last caplet do not move it."
            + (f" One-sided at {', '.join(one_sided)}: the other side of the bump cannot be stripped (flat vols "
               "falling about as fast as non-negative caplet vols allow)." if one_sided else "")
        )  # fmt: skip


def _vol_strip_chart(results, method: str, source: str, asof) -> go.Figure:
    fig = go.Figure()
    shown: List[float] = []
    top = max(r.curve.nodes[-1] for r in results)
    t = np.linspace(1e-6, top + 0.5, 600)
    for k, result in enumerate(results):
        name = "ATM" if result.curve.strike is None else f"{result.curve.strike:.2%}"
        vols = result.curve.vol_at(t) * 1e4
        color = SERIES[k % len(SERIES)]
        fig.add_trace(go.Scatter(x=t, y=vols, mode="lines", name=f"CAPLET {name}", line=dict(color=color, width=2)))
        flat = result.table["flat_vol"].to_numpy() * 1e4
        fig.add_trace(go.Scatter(x=result.table["node"], y=flat, mode="markers", name=f"FLAT QUOTE {name}",
                                 marker=dict(color=color, size=8, symbol="circle-open")))  # fmt: skip
        shown += list(vols) + list(flat)
    dark_layout(fig, height=360, xtitle="CAPLET EXPIRY (YEARS TO FIXING)", ytitle="NORMAL VOL (BP)")
    rates_data.badge(fig, source, asof)
    return bound_axes(fig, xaxis=axis_limits([0.0, top + 0.5], pad=0.02, floor=0.0),
                      yaxis=axis_limits(shown, pad=0.1, floor=0.0))  # fmt: skip


def _smile_chart(surface: CapletVolSurface, expiry: float, forward: float, source: str, asof) -> Tuple[go.Figure, str]:
    """Stripped caplet vols at one expiry and a normal SABR fit; returns the
    figure and why no fit is drawn (empty when it is)."""

    strikes, vols = surface.smile(expiry)
    note = ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes * 100, y=vols * 1e4, mode="markers", name="STRIPPED CAPLET VOLS",
                             marker=dict(color=SERIES[3], size=9)))  # fmt: skip
    shown = list(vols * 1e4)
    xs = list(strikes * 100) + [forward * 100]
    try:
        fit = calibrate_sabr_smile(strikes, vols, expiry, forward, 0.0, SMILE_SHIFT, vol_type="normal")
        if not fit.success:
            raise ValueError(f"the fit did not converge ({fit.message})")
        grid = np.linspace(strikes.min(), strikes.max(), 120)
        line = sabr_vol(forward, grid, expiry, fit.alpha, 0.0, fit.rho, fit.nu, SMILE_SHIFT, "normal") * 1e4
        fig.add_trace(go.Scatter(x=grid * 100, y=line, mode="lines", name=f"NORMAL SABR (RMSE {fit.rmse_bp:.2f}BP)",
                                 line=dict(color=SERIES[0], width=2)))  # fmt: skip
        shown += list(line)
    except ValueError as err:  # too few strikes or SABR cannot fit: the stripped points are still shown
        note = str(err).rstrip(".")
    fig.add_trace(go.Scatter(x=[forward * 100], y=[float(np.interp(forward, strikes, vols)) * 1e4], mode="markers",
                             name="CAPLET FORWARD", marker=dict(color=ACCENT, size=12, symbol="diamond")))  # fmt: skip
    dark_layout(fig, height=340, xtitle="STRIKE (%)", ytitle="CAPLET NORMAL VOL (BP)")
    rates_data.badge(fig, source, asof)
    return bound_axes(fig, xaxis=axis_limits(xs, pad=0.05), yaxis=axis_limits(shown, pad=0.12, floor=0.0)), note


def _vega_chart(buckets: pd.DataFrame, per: str) -> go.Figure:
    fixes, values = list(buckets["fix"]), buckets["vega"].tolist()
    fig = go.Figure(go.Bar(x=fixes, y=values, name="VEGA", marker=dict(color=SERIES[2])))
    dark_layout(fig, height=300, xtitle="CAPLET FIXING DATE", ytitle=f"VEGA (PER +{per})", legend=False)
    return bound_axes(fig, xaxis=_date_limits(fixes), yaxis=axis_limits([0.0, *values], pad=0.12,
                                                                         floor=min(0.0, *values)))  # fmt: skip


def _greeks(position: Position, market: Market, trade_key: tuple) -> Dict[str, float]:
    """PV01 and gamma (+1bp on every rate quote, curves rebuilt) and the full
    one-day theta, cached per market and trade."""

    def compute() -> Dict[str, float]:
        pv01, gamma = rate_pv01_gamma(position.pv, market)
        return {"pv01": pv01, "gamma": gamma, "theta": float(theta([position], market)[0])}

    return rates_risk.cached("cap_greeks", (rates_risk.market_key(market), trade_key), compute)


def _index_choices(market: Market) -> Tuple[List[str], str]:
    """INDEX options (term indices of a multi-curve market or single-curve
    term rates, and the compounded overnight rate) and the overnight
    rate's name. Single-curve, the compounded rate comes first: USD and GBP
    caps trade on SOFR and SONIA compounded in arrears."""

    conv = market.curve.conventions
    overnight = OVERNIGHT_NAMES.get(conv.currency if conv is not None else "USD", "OIS")
    compounded = f"{overnight} compounded"
    indices = [k for k in market.projections if not market.projections.is_discount(k)]
    return (indices + [compounded], compounded) if indices else ([compounded] + TERM_CHOICES, compounded)


def render(market: Market) -> List[Position]:
    """Draw the tab and return the priced cap, floor or collar as a position."""

    st.markdown("### / CAP, FLOOR & COLLAR")
    conv = market.curve.conventions
    calendar = conv.calendar if conv else "USD"
    day_count = conv.irs_float_day_count if conv else "ACT/360"
    fixing_lag = conv.spot_lag if conv else 2

    c1, c2, c3, c4, c5 = st.columns(5)
    kind = c1.radio("TYPE", KINDS, horizontal=True, key="cap_kind").lower()
    notional = c2.number_input(
        "NOTIONAL",
        min_value=1.0,
        max_value=1e12,
        value=10_000_000.0,
        step=1_000_000.0,
        format="%.0f",
        key="cap_notional",
    )
    tenor = c3.selectbox("TENOR", TENORS, index=TENORS.index("5Y"), key="cap_tenor")
    fwd = c4.selectbox("FORWARD START", FORWARD_STARTS, key="cap_forward_start")
    choices, compounded_choice = _index_choices(market)
    index_choice = c5.radio("INDEX", choices, horizontal=True, key="cap_index")
    compounded = index_choice == compounded_choice
    projection = index_choice if index_choice in market.projections else None
    effective_vol = False
    if compounded:
        c1, c2 = st.columns([1, 2])
        index_months = int(c1.selectbox("RESET PERIOD", RESETS, key="cap_reset")[:-1])
        vol_input = c2.radio("VOL QUOTE", list(RFR_VOLS), horizontal=True, key="cap_rfr_vol")
        effective_vol = RFR_VOLS[vol_input]
    elif projection:
        index_months = market.get_curve(projection).index.months
    else:
        index_months = int(index_choice.split("M")[0])
    overnight = compounded_choice.split(" ")[0]
    if not compounded and not projection:
        st.warning(
            f"Term-rate approximation: a {index_months}M rate fixed in advance, projected off the {overnight} OIS "
            f"curve. The app has no {overnight} term-rate curve, and standard caps in this currency are on "
            f"{overnight} compounded in arrears: pick {compounded_choice} for those."
        )

    collar = kind == "collar"
    c1, c2, c3, c4, c5 = st.columns(5)
    strike_mode = c1.radio("STRIKE", ["ATM ± bp", "Absolute %"], horizontal=True, key="cap_strike_mode")
    strike_word = "CAP STRIKE" if collar else "STRIKE"
    if strike_mode == "Absolute %":
        strike_pct = c2.number_input(
            f"{strike_word} (%)", min_value=-5.0, max_value=30.0, value=4.0, step=0.125, format="%.4f",
            key="cap_strike_pct",
        )  # fmt: skip
        if collar:
            floor_pct = c3.number_input(
                "FLOOR STRIKE (%)", min_value=-5.0, max_value=30.0, value=3.0, step=0.125, format="%.4f",
                key="cap_floor_strike_pct",
            )  # fmt: skip
    else:
        offset_bp = c2.number_input(
            "CAP ATM OFFSET (BP)" if collar else "ATM OFFSET (BP)",
            min_value=-1000.0,
            max_value=1000.0,
            value=50.0 if collar else 0.0,
            step=25.0,
            format="%.1f",
            key="cap_collar_offset_bp" if collar else "cap_atm_offset_bp",
        )
        if collar:
            floor_offset_bp = c3.number_input(
                "FLOOR ATM OFFSET (BP)", min_value=-1000.0, max_value=1000.0, value=-50.0, step=25.0, format="%.1f",
                key="cap_floor_offset_bp",
            )  # fmt: skip
    model = MODEL_LABELS[c4.selectbox("MODEL", list(MODEL_LABELS), key="cap_model")]
    shift = 0.0
    if model == "shifted_black":
        shift = (
            c5.number_input(
                "SHIFT (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.25, format="%.2f", key="cap_shift_pct"
            )
            / 100
        )

    vol_label, vol_unit, vol_top = VOL_UNITS[model]
    vol_mode = st.radio("VOL INPUT", VOL_MODES, horizontal=True, key="cap_vol_mode")
    stripped = None
    try:
        if vol_mode == "Flat":
            vol = (
                st.number_input(
                    f"FLAT VOL ({vol_label})",
                    min_value=0.0,
                    max_value=vol_top,
                    value=DEFAULT_FLAT_VOL[model],
                    step=1.0,
                    format="%.2f",
                    key=f"cap_flat_vol_{model}",
                )
                * vol_unit
            )
        elif vol_mode == "Term structure":
            st.caption(
                "Vol by option expiry (years to the fixing; to the last daily fixing for a compounded rate), "
                "linear in expiry and flat outside the table."
            )
            vol = _vol_table(model)
        else:
            vol = 0.0  # replaced by the stripped caplet vols below
            if model != "bachelier":
                raise ValueError(
                    "cap vol quotes here are flat normal vols (bp), the way USD and EUR caps are quoted: pick the "
                    "Normal (bp) model to strip them, or enter a lognormal term structure"
                )
        terms: Dict[str, object]
        if compounded:
            terms = dict(
                forward_start=None if fwd == "Spot" else fwd, index_months=index_months,
                day_count=conv.ois_day_count if conv else "ACT/360", calendar=calendar, fixing_lag=fixing_lag,
                payment_lag=conv.payment_lag if conv else 2, rate_type="compounded", effective_vol=effective_vol,
            )  # fmt: skip
            day_count = str(terms["day_count"])
        elif projection:
            index = market.get_curve(projection).index
            terms = dict(
                forward_start=None if fwd == "Spot" else fwd, index_months=index.months, day_count=index.day_count,
                calendar=index.calendar, fixing_lag=index.fixing_lag, projection=projection,
            )  # fmt: skip
            calendar, day_count, fixing_lag = index.calendar, index.day_count, index.fixing_lag
        else:
            terms = dict(
                forward_start=None if fwd == "Spot" else fwd,
                index_months=index_months,
                day_count=day_count,
                calendar=calendar,
                fixing_lag=fixing_lag,
            )
        atm = CapFloor(notional, "cap", 0.0, tenor, 0.0, **terms).atm_strike(market)
        strike = strike_pct / 100 if strike_mode == "Absolute %" else atm + offset_bp * 1e-4
        floor_strike = None
        if collar:
            floor_strike = floor_pct / 100 if strike_mode == "Absolute %" else atm + floor_offset_bp * 1e-4
        if vol_mode == "Stripped from cap quotes":
            stripped = _stripped_inputs(market, _quote_terms(terms))
            vol = stripped["vol"]
        cf = CapFloor(notional, kind, strike, tenor, vol, model, shift, floor_strike=floor_strike, **terms)
        table = cf.caplets(market)
        position = capfloor_position(cf, asof=market.asof)
        pv = float(table["pv"].sum())
        vega = float(table["vega"].sum())
        premium_bp = cf.premium_bp_running(market)
        trade = ("cap", repr(position.details["trade"]))
        if market.is_multi_curve:
            curves = rates_risk.projection_chain(market, projection)
            ladders = rates_risk.ladders(market, position.pv, trade, curves)
            ladder = [dict(r, pv01=float(v)) for key in curves for r, v in zip(*ladders[key])]
        else:
            ladders = None
            ladder = rates_risk.cached("cap_ladder", (rates_risk.market_key(market), trade),
                                       lambda: key_rate_pv01(position.pv, market))  # fmt: skip
        greeks = _greeks(position, market, trade)
        pv01 = greeks["pv01"]
        decay = cf.decay(market)
        parity = cap_floor_parity(cf, market)
    except ValueError as err:
        st.error(f"Cannot price this {kind}: {err}")
        return []

    per = "1BP" if model == "bachelier" else "VOL PT"
    cells = [
        ("PV", f"{pv:,.2f}"),
        ("PREMIUM (BP RUNNING)", f"{premium_bp:,.2f}"),
        ("PREMIUM (% UPFRONT)", f"{pv / notional * 100:.4f}"),
        ("ATM STRIKE (%)", f"{atm * 100:.4f}"),
        (f"VEGA (/{per})", f"{vega:,.2f}"),
        ("PV01 (/BP)", f"{pv01:,.2f}"),
        ("GAMMA (/BP²)", f"{greeks['gamma']:,.4f}"),
        ("THETA (1 BUS. DAY)", f"{greeks['theta']:,.2f}"),
    ]
    for start in range(0, len(cells), 3):  # three to a row: wider rows wrap next to the sidebar
        for col, (label, value) in zip(st.columns(3), cells[start : start + 3]):
            col.metric(label, value)
    shift_text = (
        "+1bp on every rate quote of the OIS and EURIBOR curves, rebuilt (basis spreads held)"
        if ladders is not None else "+1bp on every curve quote with the curve rebuilt"
    )  # fmt: skip
    legs = (
        f" Long the cap at {strike:.4%} ({float(table['cap_pv'].sum()):,.2f}) and short the floor at "
        f"{floor_strike:.4%} ({float(table['floor_pv'].sum()):,.2f})."
        if collar and floor_strike is not None else ""
    )  # fmt: skip
    st.caption(
        f"{cf.label}, {len(table)} {kind}lets. PV, vega and PV01 are P&L to the holder in "
        f"currency for the notional; PV01 is {shift_text}. Running premium is "
        f"PV over the strip annuity (sum of notional x accrual x DF).{legs} Cap - floor - payer swap at "
        f"{'the cap strike' if collar else 'the strike'} = {parity['residual']:.2e}."
    )
    if compounded:
        vol_rule = (
            "the vol is the effective vol to the last fixing (variance vol² x time to the last fixing)"
            if effective_vol else
            "the vol is that of the period's forward-looking rate, decaying linearly to zero over the accrual "
            "period (Lyashenko-Mercurio): variance vol² x (t_start + (t_end - t_start)/3) before the period, "
            "shrinking as it fixes (the table's OPTION TIME; EFF VOL is the equivalent vol to the last fixing)"
        )  # fmt: skip
        st.caption(
            f"Schedule: {index_months}M periods rolled back from maturity, {calendar} calendar, modified following, "
            f"{day_count} accrual. Each caplet pays on {overnight} compounded in arrears over its period, as the "
            f"{overnight} OIS floating leg: daily fixings from the period start to the business day before its end, "
            + ("paid at the period end" if cf.payment_lag == 0 else
               f"paid {cf.payment_lag} business day{'s' if cf.payment_lag != 1 else ''} after the period end")
            + ", no lookback or observation shift. Every period is included (a compounded rate is unknown until its "
            "period ends); a forward start is spot plus the forward period, modified following. Option time is "
            f"ACT/365F; {vol_rule}. Forwards and discounting on the {overnight} OIS curve."
        )  # fmt: skip
    else:
        st.caption(
            f"Schedule: {index_months}M periods rolled back from maturity, {calendar} calendar, modified "
            f"following, {day_count} accrual; each rate fixes {fixing_lag} business days before its period starts "
            "and pays at the period end. A spot-starting cap leaves out the first period (it fixes at trade); a "
            "forward start is spot plus the forward period, modified following, as on the swap tab. Option time is "
            "ACT/365F to the fixing; "
            + (f"forwards from the {projection} projection curve, discounting on the OIS curve." if projection
               else "one curve projects and discounts.")
        )  # fmt: skip
    st.caption(
        "GAMMA: P&L per (1bp)² for a move of every rate quote (central second difference, curves rebuilt), so "
        "P&L(x bp) ~ PV01 x + gamma x²/2. THETA: value on the next business day with quotes and vols unchanged "
        f"(curves rolled, caplet vols read at their shorter expiry) minus today; time decay alone (forwards, "
        f"discount factors and vols held) is {decay:,.2f}."
    )
    if stripped is not None:
        _show_strip(stripped, cf, market, _quote_terms(terms))
    if vol_mode == "Flat" and vol == 0.0:
        st.warning("Zero vol: every caplet is worth only its intrinsic value on the forward.")
    pillars = market.curve.pillar_dates
    if pillars and table["pay"].max() > pillars[-1]:
        st.warning(f"The {kind} pays after the curve's last quote ({pillars[-1]}); forwards beyond it are held flat.")

    st.markdown(f"### / {kind.upper()}LETS")
    st.plotly_chart(_strip_chart(table, strike, kind, floor_strike), width="stretch", config=PLOTLY_2D_CONFIG,
                    key="cap_strip_chart")  # fmt: skip
    view = _caplet_view(table, model, compounded)
    two_dp = st.column_config.NumberColumn(format="%.2f")
    formats = {
        **CAPLET_FORMATS,
        **{c: two_dp for c in view.columns if c.startswith(("VOL (", "VEGA (", "EFF VOL (", "FLOOR VOL ("))},
    }
    st.dataframe(view, width="stretch", hide_index=True, column_config=formats)
    st.download_button(
        "DOWNLOAD CAPLETS CSV",
        view.to_csv(index=False),
        file_name=f"{kind}lets.csv",
        mime="text/csv",
        on_click="ignore",
        key="cap_caplets_csv",
    )
    buckets = vega_by_expiry(cf, market)
    if len(buckets):
        st.markdown(f"### / VEGA BY CAPLET EXPIRY (PER +{per})")
        st.plotly_chart(_vega_chart(buckets, per), width="stretch", config=PLOTLY_2D_CONFIG, key="cap_vega_chart")
        st.caption(
            f"Each live {kind}let's vega: P&L for +{'1bp of normal vol' if model == 'bachelier' else '1 vol point'} "
            f"on that caplet's vol alone. They sum to the vega {vega:,.2f}."
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
        "DOWNLOAD KEY-RATE PV01 CSV",
        ladder_view.to_csv(index=False),
        file_name=f"{kind}_key_rate_pv01.csv",
        mime="text/csv",
        on_click="ignore",
        key="cap_ladder_csv",
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
    tables = {"caplets": view, "key_rate_pv01": ladder_view, "vega_by_expiry": buckets,
              "summary": pd.DataFrame([{"label": cf.label, "pv": pv, "premium_bp_running": premium_bp, "vega": vega,
                                        "pv01": pv01, "gamma": greeks["gamma"], "theta_1d": greeks["theta"],
                                        "decay_1d": decay, "atm_strike": atm, **{f"parity_{k}": v for k, v in
                                                                                 parity.items()}}])}  # fmt: skip
    if stripped is not None:
        tables["caplet_vol_strip"] = pd.concat([r.table for r in stripped["results"]], ignore_index=True)
    vol_input = (
        vol
        if isinstance(vol, (int, float))
        else ("term structure " + repr(vol) if isinstance(vol, dict) else stripped["source"] if stripped else repr(vol))
    )
    rates_exports.export_row(
        "cap floor", market, tables,
        {"trade": {"kind": kind, "notional": notional, "tenor": tenor, "forward_start": fwd, "strike": strike,
                   "floor_strike": floor_strike, "model": model, "shift": shift, **terms}, "vol": vol_input,
         "vol_mode": vol_mode},
        {"pricing": f"{model} per caplet, "
         + (f"{overnight} compounded in arrears, "
            + ("effective vol to the last fixing" if effective_vol else "Lyashenko-Mercurio vol decay")
            if compounded else
            f"term rate from {'the ' + projection if projection else 'the OIS curve (approximation)'}")
         + ", OIS discounting", "strip_method": stripped["method"] if stripped else None,
         "risk": "+/-1bp per curve quote, curves rebuilt, central difference"},
        key="cap_export",
    )  # fmt: skip
    return [position]
