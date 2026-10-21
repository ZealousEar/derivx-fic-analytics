"""RESEARCH tab of the Rates page: curve relative value (spreads and flies
against their history), carry and roll-down by tenor, PCA of curve moves and
the book's value at risk.

Every view reads the history store the sidebar DATA SOURCE chose (plus a
swap history uploaded on the SCENARIOS tab) up to the valuation date, and
says which history and dates its numbers come from. Statistics use daily
data only: each view starts where the chosen history's most recent daily
stretch begins (the bundled history is daily for US Treasuries from 2015,
for the ECB AAA and BoE OIS curves only from 2026-08-19, month ends before).

Currencies: factor risk and VaR are per position in its own currency (US
Treasuries in USD, the rest in the swap curve's), totalled per currency
and, when the positions hold a currency other than the sidebar's REPORTING
CURRENCY, for the whole book converted at the stated FX spots (``ALL IN
<R>``). In the VaR the converted book also moves with FX: each scenario
day's ECB reference-rate change (the store's ``FX_ECB`` history, the same
dates as the curve changes) moves today's spots, with the FX-held and
FX-only VaR shown next to it; FX IN VAR = HELD holds the spots.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.data import rates_history as rh
from derivx.rates import carry, rv
from derivx.rates import pca as pca_mod
from derivx.rates import portfolio as pf
from derivx.rates.conventions import parse_tenor
from derivx.rates.curves import Market
from derivx.risk import var as var_mod
from derivx.risk.reporting import Reporting, add_total_rows
from derivx.risk.scenarios import factor_risk as position_factor_risk
from derivx.risk.scenarios import vegas as position_vegas
from derivx.risk.sensitivities import Position
from derivx.ui import rates_data, rates_exports, rates_fx, rates_risk, rates_scenarios
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import ACCENT, LEGEND_BELOW, PLOTLY_2D_CONFIG, SERIES, dark_layout

SUBTABS = ("/ RV", "/ CARRY & ROLL", "/ PCA", "/ VAR")
MEASURES = {
    "par_yield": "par yields",
    "par_yield_cc": "par yields, cont. comp.",
    "par_swap": "par swap rates",
    "zero_cc": "spot (zero) rates, cont. comp.",
}
WEIGHT_HELP = (
    "EQUAL: spread -1/+1, fly -1/+2/-1 (the quoted structure; as DV01 it is DV01-neutral, each wing half the body). "
    "MATURITY: DV01-neutral fly with wings weighted by maturity distance, also neutral to a move linear in "
    "maturity. PCA: neutral to PC1 (spread) or PC1 and PC2 (fly) of the history's daily changes."
)
CARRY_KINDS = ("TREASURY PAR BONDS", "SWAPS (RECEIVE FIXED)")
FX_CHOICES = ("HISTORICAL MOVES", "HELD")
FLAG_LABELS = {
    "real": "REAL: the curve's own market",
    "upload": "YOUR UPLOAD",
    "proxy": "PROXY: another market's moves",
    "cross-currency proxy": "CROSS-CURRENCY PROXY",
}
CACHE_ENTRIES = 6
MONEY = st.column_config.NumberColumn(format="%,.0f")
BP = st.column_config.NumberColumn(format="%.2f")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _download(label: str, data: bytes, file_name: str, key: str) -> None:
    """CSV download that does not rerun the page when clicked."""

    st.download_button(label, data, file_name=file_name, mime="text/csv", key=key, on_click="ignore")


def _num(value: Optional[float], decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{round(value, decimals) + 0.0:,.{decimals}f}"


def _cached(name: str, key: tuple, compute):
    """Last few results of a slow computation, per session (they hold this
    session's positions)."""

    entries = st.session_state.setdefault(f"_rs_cache_{name}", {})
    if key not in entries:
        if len(entries) >= CACHE_ENTRIES:
            entries.pop(next(iter(entries)))
        entries[key] = compute()
    return entries[key]


def _datasets(store: rh.HistoryStore) -> List[str]:
    """Curve histories in the store, US Treasuries first."""

    keys = [k for k in store.datasets() if rh.dataset_info(k).kind not in ("fixings", "fx") and len(store.dates(k)) > 1]
    return sorted(keys, key=lambda k: (k != var_mod.ANCHOR, k))


def _unique(names: Sequence[str]) -> List[str]:
    """Position names made unique (``NAME (2)`` for a repeat)."""

    seen: Dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


def _label(key: str) -> str:
    info = rh.dataset_info(key)
    return f"{info.title} ({MEASURES.get(info.measure, info.measure)})"


def _levels(store: rh.HistoryStore, dataset: str, end: dt.date) -> pd.DataFrame:
    """Decimal levels up to ``end`` with the tenors quoted on the last date,
    shortest first."""

    frame = store.complete(dataset)
    frame = frame[frame.index <= end]
    if frame.empty:
        return frame
    keep = [c for c in frame.columns if pd.notna(frame.iloc[-1][c])]
    keep.sort(key=lambda t: parse_tenor(t).years)
    return frame[keep]


def _daily(levels: pd.DataFrame, tenors: Sequence[str]) -> pd.DataFrame:
    """The most recent daily stretch of ``levels`` for ``tenors``."""

    start = pca_mod.recent_daily_start(levels, tenors)
    if start is None:
        return levels.iloc[:0]
    return levels[levels.index >= start]


def _source(store: rh.HistoryStore, dataset: str, day: dt.date) -> str:
    origin = store.origin(dataset, day)
    return f"{rh.dataset_info(dataset).title} ({origin})" if origin else rh.dataset_info(dataset).title


def _legend_below(fig: go.Figure) -> None:
    """Horizontal legend under the plot, so it never squeezes the plot on a
    narrow screen."""

    fig.update_layout(legend=LEGEND_BELOW)  # dark_layout's default; kept for charts styled without it


def _times(dates: Sequence[dt.date]) -> np.ndarray:
    """Dates as numpy datetime64 (Plotly serialises them ~25x faster than
    lists of ``datetime.date``)."""

    return pd.to_datetime(list(dates)).values


def _iso(day: dt.date) -> str:
    return pd.Timestamp(day).isoformat()


def _date_axis(dates: Sequence[dt.date]) -> Dict[str, str]:
    first, last = min(dates), max(dates)
    if first == last:
        first = first - dt.timedelta(days=1)
    return {"min": _iso(first), "max": _iso(last)}


def _keep_in_range(key: str, lo: dt.date, hi: dt.date) -> None:
    """Forget a stored date outside [lo, hi] (the history or the valuation
    date changed), so the widget starts again from its default."""

    value = st.session_state.get(key)
    if isinstance(value, dt.date) and not lo <= value <= hi:
        st.session_state.pop(key, None)


# The cached computations below are keyed on ``data`` = (store fingerprint, dataset, valuation date), which
# fixes the ``_levels`` frame they read (left out of the hash: hashing a few thousand dated rows on every rerun
# cost more than the cached work).


def _data_key(store: rh.HistoryStore, dataset: str, end: dt.date) -> tuple:
    return (store.fingerprint, dataset, end)


@st.cache_data(show_spinner=False, max_entries=16)
def _pca(data: tuple, tenors: Tuple[str, ...], start: Optional[dt.date], end: Optional[dt.date], _levels):
    return pca_mod.pca_curve_changes(_levels, tenors, None, start=start, end=end)


@st.cache_data(show_spinner=False, max_entries=8)
def _rolling(data: tuple, tenors: Tuple[str, ...], window: int, start: dt.date, end: dt.date, _levels):
    return pca_mod.rolling_loadings(_levels, tenors, window, step=max(1, window // 12), start=start, end=end)


@st.cache_data(show_spinner=False, max_entries=16)
def _stats(data: tuple, weights: Tuple[Tuple[str, float], ...], years: float, _daily) -> pd.DataFrame:
    return rv.rolling_stats(rv.structure_series(_daily, dict(weights)), years)


@st.cache_data(show_spinner=False, max_entries=16)
def _rank(
    data: tuple,
    structures: Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...],
    window: str,
    carry_bp: Optional[Tuple[Tuple[str, float], ...]],
    months: Optional[int],
    _levels,
) -> pd.DataFrame:
    return rv.rank_structures(
        _levels, {n: dict(w) for n, w in structures}, window, dict(carry_bp) if carry_bp else None, months
    )


def _chart(name: str, key: tuple, build, widget_key: str) -> None:
    """Draw a figure built once per ``key`` (building the Plotly figure of a
    few thousand dated points is most of a rerun's time)."""

    st.plotly_chart(_cached(f"fig_{name}", key, build), width="stretch", config=PLOTLY_2D_CONFIG, key=widget_key)


def _bond_rows(market: Market, horizon: str, repo: float) -> List[carry.BondCarry]:
    key = (rates_risk.market_key(market), horizon, round(repo, 10))
    return _cached("bonds", key, lambda: carry.treasury_carry(market, horizon, repo))


def _swap_rows(market: Market, horizon: str) -> List[carry.SwapCarry]:
    key = (rates_risk.market_key(market), horizon)
    return _cached("swaps", key, lambda: [carry.swap_carry(market, t, horizon) for t in _swap_tenors(market)])


def _swap_tenors(market: Market) -> List[str]:
    longest = max(market.curve.years) if market.curve.instruments else 30.0
    return [t for t in carry.SWAP_TENORS if parse_tenor(t).years <= longest + 1e-9]


def _repo_default(store: rh.HistoryStore, market: Market) -> Tuple[float, str]:
    """(repo % , where it comes from): the latest SOFR fixing on or before
    the valuation date (the GC repo benchmark), else the 1M bill yield."""

    got = store.fixing("SOFR", market.asof)
    if got is not None:
        day, rate, source = got
        return round(100.0 * rate, 4), f"SOFR fixing for {day} ({source}) as the GC repo rate"
    if market.govt is not None:
        bill = next((q for q in market.govt.quotes if q.tenor in ("1M", "4W")), None)
        if bill is not None:
            return round(100.0 * bill.rate, 4), (
                f"no SOFR fixing on or before {market.asof} in the history store: the 1M Treasury bill yield stands in "
                "for repo; type the repo rate you finance at"
            )
    return 0.0, "no SOFR fixing or bill yield for this date: type the repo rate you finance at"


# --------------------------------------------------------------------------
# RV
# --------------------------------------------------------------------------


def _rv_chart(stats: pd.DataFrame, name: str, years: float, source: str, asof: dt.date) -> go.Figure:
    dates = list(stats.index)
    x = _times(dates)
    mean, sd, value = (stats[c].to_numpy(dtype=float) for c in ("mean", "sd", "value"))
    fig = go.Figure()
    bands = [(2.0, "+/-2 SD", SERIES[3]), (1.0, "+/-1 SD", SERIES[2])]
    for k, label, colour in bands:
        for sign in (1.0, -1.0):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean + sign * k * sd,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=sign > 0,
                    line=dict(color=colour, width=1, dash="dot"),
                    hovertemplate="%{y:.1f}bp<extra>" + label + "</extra>",
                )
            )
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=f"{years:g}Y MEAN", line=dict(color=ACCENT, width=1)))
    fig.add_trace(go.Scatter(x=x, y=value, mode="lines", name=name, line=dict(color=SERIES[0], width=2)))
    dark_layout(fig, height=420, xtitle="DATE", ytitle=f"{name} (BP)")
    _legend_below(fig)
    rates_data.badge(fig, source, asof)
    shown = np.concatenate([value, mean + 2 * sd, mean - 2 * sd])
    return bound_axes(fig, xaxis=_date_axis(dates), yaxis=axis_limits(shown, pad=0.08))


Structure = Tuple[str, Optional[Dict[str, float]], Optional[Tuple[str, ...]], str]


def _structure_inputs(levels: pd.DataFrame, c2, c3) -> Structure:
    """(name, custom weights or None, preset legs or None, scheme)."""

    tenors = list(levels.columns)
    presets = [name for name, legs in rv.PRESETS.items() if all(t in tenors for t in legs)]
    choice = c2.selectbox("STRUCTURE", presets + ["CUSTOM"], key="rs_rv_structure")
    scheme = c3.radio("WEIGHTS", rv.WEIGHTINGS, horizontal=True, key="rs_rv_weights", help=WEIGHT_HELP)
    if choice != "CUSTOM":
        return choice, None, rv.PRESETS[choice], scheme
    text = st.text_input(
        "CUSTOM LEGS (TENOR:WEIGHT IN BP OF YIELD, COMMA SEPARATED)",
        value="2Y:-1, 5Y:2, 10Y:-1",
        key="rs_rv_custom",
        help="Value = sum of weight x yield, in bp. Positive weights gain when that yield rises.",
    )
    weights = rv.parse_weights(text)
    missing = [t for t in weights if t not in tenors]
    if missing:
        raise ValueError(f"tenors {missing} are not quoted in this history (it has {', '.join(tenors)})")
    return "CUSTOM", weights, None, scheme


def _carry_by_tenor(market: Market, horizon: str, repo: float) -> Dict[str, float]:
    rows = _bond_rows(market, horizon, repo)
    return {t: r.total_bp for t, r in zip(carry.BOND_TENORS, rows)}


def _render_rv(store: rh.HistoryStore, market: Market) -> None:
    datasets = _datasets(store)
    if not datasets:
        st.info("The history store has no curve history.")
        return
    c1, c2, c3, c4 = st.columns(4)
    dataset = c1.selectbox("HISTORY", datasets, format_func=_label, key="rs_rv_dataset")
    window = c4.radio("Z-SCORE WINDOW", list(rv.WINDOWS), horizontal=True, key="rs_rv_window")
    levels = _levels(store, dataset, market.asof)
    data = _data_key(store, dataset, market.asof)
    try:
        name, custom, legs, scheme = _structure_inputs(levels, c2, c3)
    except ValueError as exc:
        st.error(f"Cannot build this structure: {exc}")
        return
    pca = None
    if scheme == "PCA":
        pca_tenors = tuple(t for t in pca_mod.DEFAULT_TENORS if t in levels.columns) or tuple(levels.columns)
        try:
            pca = _pca(data, pca_tenors, pca_mod.recent_daily_start(levels, pca_tenors), market.asof, levels)
        except ValueError as exc:
            st.error(f"PCA weights need a PCA of this history: {exc}")
            return
    try:
        weights = custom if custom is not None else rv.structure_weights(legs, scheme, pca)
    except ValueError as exc:
        st.error(f"Cannot weight this structure: {exc}")
        return
    if custom is not None:
        name = "CUSTOM " + ", ".join(f"{t} {w:+g}" for t, w in weights.items())
    elif scheme != "EQUAL":
        name = f"{name} ({scheme})"
    daily = _daily(levels, list(weights))
    years = rv.WINDOWS[window]
    if len(daily) < rv.MIN_WINDOW_OBS + 1:
        st.warning(
            f"{_label(dataset)} has only {len(daily)} daily observations of these tenors up to {market.asof}: too few "
            "for statistics. Pick another history or a later valuation date, or use the LIVE source."
        )
        return
    stats = _stats(data, tuple(weights.items()), years, daily)
    series = stats["value"]
    snap = rv.snapshot(series, {window: years}).iloc[0]
    try:
        hl = rv.half_life(rv.trailing(series, years))
    except ValueError:
        hl = None
    m1, m2, m3 = st.columns(3)  # three cards a row: four wrap next to the sidebar
    m1.metric("LATEST (BP)", _num(snap["value_bp"], 1))
    m2.metric(f"{window} Z-SCORE", _num(snap["z"], 2))
    m3.metric(f"{window} PERCENTILE", _num(snap["pct"], 0))
    m4, m5, m6 = st.columns(3)
    m4.metric(f"{window} MEAN (BP)", _num(snap["mean_bp"], 1))
    m5.metric(f"{window} SD (BP)", _num(snap["sd_bp"], 1))
    m6.metric("HALF-LIFE (BUS. DAYS)", "none" if hl is None or not hl.mean_reverting else _num(hl.days, 0))
    last = series.index[-1]
    source = _source(store, dataset, last)
    title = name.split(" (")[0]
    _chart(
        "rv",
        (data, tuple(weights.items()), years, title, source),
        lambda: _rv_chart(stats.dropna(subset=["mean"]), title, years, source, last),
        "rs_rv_chart",
    )
    covered = (last - snap["from"]).days if snap["from"] is not None else 0
    if covered < 0.9 * 365.25 * years:
        st.warning(
            f"The daily history of these tenors starts {daily.index[0]}: the {window} window holds only "
            f"{int(snap['n'])} observations ({covered} days), so its mean, z-score and percentile describe a shorter "
            "period than the label says."
        )
    w_text = ", ".join(f"{t} {w:+.3f}" for t, w in weights.items())
    st.caption(
        f"{name}: value = sum of weight x {MEASURES.get(rh.dataset_info(dataset).measure, 'rate')} in bp, weights "
        f"{w_text}; {source}, daily from {daily.index[0]} to {last}. Bands: trailing {window} mean +/- 1 and 2 "
        "standard deviations (calendar window, divisor n - 1). Z-score = (latest - mean) / sd; percentile = share of "
        "the window at or below the latest value. Half-life: AR(1) fit of daily changes on the level over the window "
        "(Ornstein-Uhlenbeck), ln 2 / kappa business days; 'none' when the fit shows no mean reversion. Being long "
        "the structure means holding minus each weight in DV01 of that leg (a 2s10s steepener: long 2Y, short 10Y)."
    )
    _rv_ranking(store, market, dataset, levels, data, window, scheme, pca, custom, name)
    _swap_spreads(store, market, window)


def _swap_spreads(store: rh.HistoryStore, market: Market, window: str) -> None:
    """Swap minus Treasury yields per tenor where real USD swap history
    exists (an upload, or ICE swap rates pulled into the cache)."""

    st.markdown("#### / SWAP SPREADS")
    govt = _levels(store, var_mod.ANCHOR, market.asof)
    shown = []
    for dataset in (rh.user_dataset("USD"), "USD_SWAP_ICE"):
        if not store.has(dataset):
            continue
        table = rv.swap_spreads(_levels(store, dataset, market.asof), govt, window)
        table = table[table["n"] >= rv.MIN_WINDOW_OBS]
        if not table.empty:
            shown.append(table.assign(history=rh.dataset_info(dataset).title))
    if not shown:
        st.caption(
            "No USD swap history with daily data in the store, so no swap spreads: the free ICE swap rates (FRED "
            "DSWP*) end 2016-10-28 and only their episode windows are bundled. Upload your USD swap history on the "
            "SCENARIOS tab, or pull DSWP* with 'derivx pull fred' and use the LIVE source."
        )
        return
    table = pd.concat(shown, ignore_index=True)
    labels = {"history": "SWAP HISTORY", "tenor": "TENOR", "last_date": "LAST DATE", "from": "FROM", "n": "OBS",
              "value_bp": "LATEST (BP)", "mean_bp": f"{window} MEAN (BP)", "sd_bp": f"{window} SD (BP)",
              "z": "Z-SCORE", "pct": "PERCENTILE"}  # fmt: skip
    table = table[list(labels)].rename(columns=labels)
    st.dataframe(table, hide_index=True, width="stretch")
    _download("Download swap spreads CSV", _csv(table), file_name="swap_spreads.csv", key="rs_rv_spreads_csv")
    stale = table[[(market.asof - d).days > 10 for d in table["LAST DATE"]]]
    st.caption(
        "Swap spread = par swap rate minus the US Treasury par yield of the same tenor (bp), on the dates both are "
        "quoted, over each tenor's latest daily stretch."
        + (f" {len(stale)} of these end well before the valuation date (ICE swap rates stop on 2016-10-28)."
           if len(stale) else "")
    )  # fmt: skip


def _rv_ranking(
    store: rh.HistoryStore,
    market: Market,
    dataset: str,
    levels: pd.DataFrame,
    data: tuple,
    window: str,
    scheme: str,
    pca: Optional[pca_mod.PCAResult],
    custom: Optional[Dict[str, float]],
    custom_name: str,
) -> None:
    st.markdown("#### / RANKING")
    c1, c2 = st.columns(2)
    horizon = c1.radio("CARRY HORIZON", list(carry.HORIZONS), index=1, horizontal=True, key="rs_rv_horizon")
    sort = c2.radio("SORT BY", ["|Z-SCORE|", "CARRY / VOL"], horizontal=True, key="rs_rv_sort")
    structures: Dict[str, Dict[str, float]] = {}
    for preset, legs in rv.PRESETS.items():
        if all(t in levels.columns for t in legs):
            try:
                structures[preset] = rv.structure_weights(legs, scheme, pca)
            except ValueError:
                structures[preset] = rv.structure_weights(legs, "EQUAL")
    if custom is not None:
        structures[custom_name] = custom
    carry_bp, note = None, ""
    if dataset == var_mod.ANCHOR and market.govt is not None and market.govt.quotes:
        repo, _ = _repo_default(store, market)
        try:
            carry_bp = _carry_by_tenor(market, horizon, repo / 100.0)
        except ValueError as exc:
            note = f" Carry is not available: {exc}."
    else:
        note = " Carry needs the US Treasury curve: it is shown for the US Treasury history only."
    table = _rank(
        data,
        tuple((n, tuple(w.items())) for n, w in structures.items()),
        window,
        tuple(carry_bp.items()) if carry_bp else None,
        carry.HORIZONS[horizon],
        levels,
    )
    if table.empty:
        st.info("No structure of the presets is quoted in this history.")
        return
    if sort == "CARRY / VOL" and "carry_to_vol" in table:
        table = table.sort_values("carry_to_vol", ascending=False, na_position="last")
    else:
        table = table.reindex(table["z"].abs().sort_values(ascending=False, na_position="last").index)
    labels = {
        "structure": "STRUCTURE",
        "weights": "WEIGHTS",
        "value_bp": "LATEST (BP)",
        "mean_bp": f"{window} MEAN (BP)",
        "sd_bp": f"{window} SD (BP)",
        "z": "Z-SCORE",
        "pct": "PERCENTILE",
        "daily_vol_bp": "DAILY VOL (BP)",
        "half_life_days": "HALF-LIFE (BUS. DAYS)",
        "n": "OBS",
        "sampling": "SAMPLING",
        "carry_roll_long_bp": f"CARRY+ROLL LONG, {horizon} (BP)",
        "carry_to_vol": "CARRY / VOL",
    }
    shown = table.rename(columns=labels)
    st.dataframe(shown, hide_index=True, width="stretch", column_config={c: BP for c in shown.columns[2:8]})
    _download("Download RV ranking CSV", _csv(shown), file_name="rv_ranking.csv", key="rs_rv_csv")
    st.caption(
        f"Weights: {scheme} where the structure allows it. Daily vol = sd of the structure's daily changes over the "
        f"window. Carry + roll-down of being long the structure over {horizon}: minus the weighted carry + roll of "
        "long Treasury par bonds at each leg (CARRY & ROLL tab, SOFR financing), in bp of DV01; carry / vol divides it "
        f"by the daily vol x sqrt(21 x months) business days.{note}"
    )


# --------------------------------------------------------------------------
# Carry and roll-down
# --------------------------------------------------------------------------


def _carry_chart(table: pd.DataFrame, horizon: str, source: str, asof: dt.date, what: str) -> go.Figure:
    x = list(table["tenor"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=table["carry_bp"], name="CARRY", marker=dict(color=SERIES[2])))
    fig.add_trace(go.Bar(x=x, y=table["roll_bp"], name="ROLL-DOWN", marker=dict(color=SERIES[3])))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=table["total_bp"],
            mode="markers",
            name="CARRY + ROLL",
            marker=dict(color=SERIES[0], size=10, symbol="diamond", line=dict(color="#000000", width=1)),
        )
    )
    fig.update_layout(barmode="relative")
    dark_layout(fig, height=400, xtitle=f"{what} TENOR", ytitle=f"{horizon} CARRY + ROLL (BP PER DV01)")
    _legend_below(fig)
    rates_data.badge(fig, source, asof)
    stacked_pos = table[["carry_bp", "roll_bp"]].clip(lower=0).sum(axis=1)
    stacked_neg = table[["carry_bp", "roll_bp"]].clip(upper=0).sum(axis=1)
    shown = np.concatenate([stacked_pos, stacked_neg, table["total_bp"], [-0.5, 0.5]])
    return bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(x) - 0.5, "min_span": 1.0},
        yaxis=axis_limits(shown, pad=0.12),
    )


def _render_carry(store: rh.HistoryStore, market: Market) -> None:
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.radio("INSTRUMENT", CARRY_KINDS, key="rs_carry_kind")
    horizon = c2.radio("HORIZON", list(carry.HORIZONS), index=1, horizontal=True, key="rs_carry_horizon")
    repo_default, repo_note = _repo_default(store, market)
    bonds = kind == CARRY_KINDS[0]
    repo = c3.number_input(
        "REPO / FINANCING (%)",
        min_value=-5.0,
        max_value=25.0,
        value=float(repo_default),
        step=0.01,
        format="%.4f",
        key=f"rs_carry_repo_{market.asof.isoformat()}",
        disabled=not bonds,
        help="Simple ACT/360 rate the bond position is financed at (bonds only; a swap needs no financing).",
    )
    money = "$" if bonds else rates_risk.money(market)
    dv01 = c4.number_input(
        f"POSITION DV01 ({money} PER BP)",
        min_value=1.0,
        max_value=1e9,
        value=carry.DEFAULT_DV01,
        step=1000.0,
        format="%.0f",
        key="rs_carry_dv01",
        help="The currency columns are carry and roll of a position of this DV01: bp x DV01.",
    )
    try:
        if bonds:
            rows = _bond_rows(market, horizon, repo / 100.0)
            table = carry.bond_table(rows, carry.BOND_TENORS, dv01)
            badge = rates_data.badge_for("govt")
            source = badge[0] if badge else "US Treasury par curve"
            what = "TREASURY"
        else:
            table = carry.swap_table(_swap_rows(market, horizon), dv01)
            badge = rates_data.badge_for("curve")
            source = badge[0] if badge else "swap curve as passed to this tab"
            what = "SWAP"
    except ValueError as exc:
        st.error(f"Cannot compute carry and roll-down: {exc}")
        return
    chart_key = (rates_risk.market_key(market), kind, horizon, round(repo, 10) if bonds else None, source)
    _chart("carry", chart_key, lambda: _carry_chart(table, horizon, source, market.asof, what), "rs_carry_chart")
    if bonds:
        st.caption(
            f"Long par Treasuries (coupon = the curve's par yield for settlement T+1, price 100) financed in repo at "
            f"{repo:.4f}% ({repo_note}), horizon {horizon}: carry = coupon accrual - financing + pull to par at an "
            "unchanged yield; roll-down = price effect of the bond's yield sliding to its yield on today's Treasury "
            "curve held static in time to maturity (each remaining cash flow discounted at today's rate for its time "
            "from the horizon). bp = P&L / DV01 per 100 face; currency = bp x the DV01 above. Between quoted tenors "
            "the slide follows the bootstrapped curve's interpolation (the sidebar's INTERPOLATION), not a straight "
            "line between par yields."
        )
        labels = {
            "tenor": "TENOR", "label": "BOND", "ytm": "YIELD", "ytm_roll": "YIELD AT HORIZON (STATIC CURVE)",
            "income": "COUPON ACCRUAL (PER 100)", "financing": "FINANCING (PER 100)", "pull": "PULL TO PAR (PER 100)",
            "carry": "CARRY (PER 100)", "roll": "ROLL (PER 100)", "dv01": "DV01 (PER 100, PER BP)",
            "carry_bp": "CARRY (BP)", "roll_bp": "ROLL (BP)", "total_bp": "CARRY+ROLL (BP)",
            "carry_ccy": f"CARRY ({money})", "roll_ccy": f"ROLL ({money})", "total_ccy": f"CARRY+ROLL ({money})",
        }  # fmt: skip
        shown = table[list(labels)].rename(columns=labels)
        shown["YIELD"] = 100.0 * shown["YIELD"]
        shown["YIELD AT HORIZON (STATIC CURVE)"] = 100.0 * shown["YIELD AT HORIZON (STATIC CURVE)"]
        shown = shown.rename(columns={"YIELD": "YIELD (%)", "YIELD AT HORIZON (STATIC CURVE)": "YIELD AT HORIZON (%)"})
    else:
        st.caption(
            f"Receive fixed in par OIS swaps on the {rates_risk.curve_title(market, 'curve')} curve ({source}). "
            f"Carry + roll ({horizon}) = forward-starting par minus spot par, F({horizon}, T) - S(0, T): what a "
            "receiver of the forward-starting swap earns on a static curve. Roll-down = S(0, T + horizon) - S(0, T), "
            "the slide of the rate to the forward swap's end date from maturity T + horizon to T; carry = F(horizon, "
            f"T) - S(0, T + horizon), the fixed rate against the {horizon} floating rate over the horizon spread over "
            "the forward annuity. bp of swap rate; currency = bp x the DV01 above."
        )
        labels = {
            "tenor": "TENOR", "spot_par_pct": "SPOT PAR (%)", "forward_par_pct": f"{horizon} FWD PAR (%)",
            "long_par_pct": "T + H PAR (%)", "short_par_pct": f"{horizon} PAR (%)", "carry_bp": "CARRY (BP)",
            "roll_bp": "ROLL (BP)", "total_bp": "CARRY+ROLL (BP)", "carry_ccy": f"CARRY ({money})",
            "roll_ccy": f"ROLL ({money})", "total_ccy": f"CARRY+ROLL ({money})",
            "dv01_per_mm": f"DV01 PER 1MM ({money})",
        }  # fmt: skip
        shown = table[list(labels)].rename(columns=labels)
        if "ILLUSTRATIVE" in source:
            st.warning(
                "The swap curve is built from ILLUSTRATIVE quotes (not market data), so these swap carry and roll "
                "numbers are illustrative too: paste or upload real quotes on the CURVE tab."
            )
    money_columns = [col for col in shown.columns if col.endswith(f"({money})")]
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={col: (MONEY if col in money_columns else BP) for col in shown.columns if col in money_columns
                       or col.endswith("(BP)")},
    )  # fmt: skip
    _download(
        "Download carry and roll-down CSV", _csv(shown), file_name=f"carry_roll_{what.lower()}_{horizon}.csv",
        key="rs_carry_csv",
    )  # fmt: skip


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------


def _loadings_chart(result: pca_mod.PCAResult, source: str) -> go.Figure:
    fig = go.Figure()
    k = min(3, len(result.tenors))
    for j in range(k):
        name = f"{pca_mod.factor_name(j)} {pca_mod.FACTOR_LABELS[j]}"
        fig.add_trace(
            go.Scatter(
                x=list(range(len(result.tenors))),
                y=result.loadings[:, j],
                mode="lines+markers",
                name=name,
                line=dict(color=SERIES[[2, 1, 3][j]], width=2),
                customdata=list(result.tenors),
                hovertemplate="%{customdata}: %{y:.3f}<extra>" + name + "</extra>",
            )
        )
    dark_layout(fig, height=380, xtitle="TENOR", ytitle="LOADING (UNIT VECTOR)")
    _legend_below(fig)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(result.tenors))), ticktext=list(result.tenors))
    rates_data.badge(fig, source, result.end)
    return bound_axes(
        fig,
        xaxis={"min": -0.3, "max": len(result.tenors) - 0.7, "min_span": 1.0},
        yaxis=axis_limits(np.concatenate([result.loadings[:, :k].ravel(), [-0.1, 0.1]]), pad=0.12),
    )


def _explained_chart(result: pca_mod.PCAResult) -> go.Figure:
    names = [pca_mod.factor_name(j) for j in range(len(result.explained))]
    pct = 100.0 * result.explained
    fig = go.Figure(go.Bar(x=names, y=pct, name="EXPLAINED", marker=dict(color=SERIES[2])))
    fig.add_trace(go.Scatter(x=names, y=np.cumsum(pct), mode="lines+markers", name="CUMULATIVE",
                             line=dict(color=ACCENT, width=2)))  # fmt: skip
    dark_layout(fig, height=320, xtitle="FACTOR", ytitle="VARIANCE EXPLAINED (%)")
    _legend_below(fig)
    xaxis = {"min": -0.5, "max": len(names) - 0.5, "min_span": 1.0}
    return bound_axes(fig, xaxis=xaxis, yaxis={"min": 0.0, "max": 105.0, "min_span": 5.0})


def _scores_chart(result: pca_mod.PCAResult, source: str) -> go.Figure:
    k = min(3, len(result.tenors))
    dates = list(result.scores.index)
    # Raw (not demeaned) projections, so the cumulative sum is the curve's move since the start along each loading.
    cumulative = pd.DataFrame(result.data.to_numpy() @ result.loadings[:, :k], index=result.data.index).cumsum()
    fig = go.Figure()
    for j in range(k):
        fig.add_trace(
            go.Scatter(
                x=_times(dates),
                y=cumulative.iloc[:, j].to_numpy(dtype=float),
                mode="lines",
                name=f"{pca_mod.factor_name(j)} {pca_mod.FACTOR_LABELS[j]}",
                line=dict(color=SERIES[[2, 1, 3][j]], width=1.5),
            )
        )
    dark_layout(fig, height=380, xtitle="DATE", ytitle="CUMULATIVE MOVE (BP)")
    _legend_below(fig)
    rates_data.badge(fig, source, result.end)
    return bound_axes(fig, xaxis=_date_axis(dates), yaxis=axis_limits(cumulative.to_numpy().ravel(), pad=0.08))


def _heatmap(rolling: pca_mod.RollingPCA, factor: str) -> go.Figure:
    frame = rolling.loadings[factor]
    dates = list(frame.index)
    tenors = list(frame.columns)
    z = frame.to_numpy().T
    fig = go.Figure(
        go.Heatmap(
            x=_times(dates),
            y=list(range(len(tenors))),
            z=z,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title=dict(text="LOADING"), tickfont=dict(color="#FFFFFF")),
            customdata=[[t] * len(dates) for t in tenors],
            hovertemplate="%{x|%Y-%m-%d} %{customdata}: %{z:.3f}<extra></extra>",
        )
    )
    dark_layout(fig, height=320, xtitle=f"END OF {rolling.window}-CHANGE WINDOW", ytitle="TENOR", legend=False)
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(tenors))), ticktext=tenors)
    return bound_axes(fig, xaxis=_date_axis(dates), yaxis={"min": -0.5, "max": len(tenors) - 0.5, "min_span": 1.0})


def _factor_risk(
    positions: Sequence[Position],
    market: Market,
    result: pca_mod.PCAResult,
    store: rh.HistoryStore,
    rep: Optional[Reporting] = None,
) -> Optional[pd.DataFrame]:
    """P&L per +1 sd daily move of each factor, every curve's rate quotes
    moving along the loadings (the book's key-rate PV01s x loadings; basis
    spread quotes held, as in the PCA factor shock and the VaR), per
    position in its currency (CCY), totalled per currency and, with ``rep``,
    for the whole book converted (``ALL IN <R>``)."""

    if not positions:
        return None
    _, risk = rates_scenarios.book_risk(positions, market)
    k = min(3, len(result.tenors))
    per_sd = position_factor_risk(market, risk, result, k) * result.sigma_bp[:k, None]
    names = _unique([p.name for p in positions])
    columns = [f"{pca_mod.factor_name(j)} +1 SD" for j in range(k)]
    frame = pd.DataFrame(per_sd.T, columns=columns)
    frame.insert(0, "CCY", rates_fx.currencies(positions, market))
    frame.insert(0, "POSITION", names)
    return add_total_rows(frame, columns, "POSITION", "CCY", rep)


def _render_pca(
    store: rh.HistoryStore, market: Market, positions: Sequence[Position], rep: Optional[Reporting] = None
) -> None:
    datasets = _datasets(store)
    if not datasets:
        st.info("The history store has no curve history.")
        return
    c1, c2 = st.columns([1, 2])
    dataset = c1.selectbox("HISTORY", datasets, format_func=_label, key="rs_pca_dataset")
    levels = _levels(store, dataset, market.asof)
    data = _data_key(store, dataset, market.asof)
    options = list(levels.columns)
    default = [t for t in pca_mod.DEFAULT_TENORS if t in options] or options[: min(6, len(options))]
    tenors_key = f"rs_pca_tenors_{dataset}"
    if tenors_key in st.session_state and not set(st.session_state[tenors_key]) <= set(options):
        st.session_state.pop(tenors_key)
    tenors = c2.multiselect("TENORS", options, default=default, key=tenors_key)
    tenors = sorted(tenors, key=lambda t: parse_tenor(t).years)
    if len(tenors) < 3:
        st.warning("Pick at least three tenors (level, slope and curvature need three).")
        return
    try:
        first = pca_mod.recent_daily_start(levels, tenors)
    except ValueError as exc:
        st.error(str(exc))
        return
    if first is None:
        st.warning(f"{_label(dataset)} has no daily history of these tenors up to {market.asof}.")
        return
    last = levels.index[-1]
    c3, c4, c5 = st.columns(3)
    # Keyed on the bounds: a new valuation date, history or tenor set (another daily start) starts the
    # dates again from the whole daily history to the valuation date, instead of keeping a window chosen
    # for other data.
    start_key, end_key = f"rs_pca_start_{dataset}_{first}_{last}", f"rs_pca_end_{dataset}_{first}_{last}"
    _keep_in_range(start_key, first, last)
    _keep_in_range(end_key, first, last)
    start = c3.date_input("FROM", value=first, min_value=first, max_value=last, key=start_key)
    end = c4.date_input("TO", value=last, min_value=first, max_value=last, key=end_key)
    window = int(
        c5.number_input(
            "ROLLING WINDOW (DAILY CHANGES)", min_value=30, max_value=2520, value=252, step=21, key="rs_pca_window"
        )  # fmt: skip
    )
    try:
        result = _pca(data, tuple(tenors), start, end, levels)
    except ValueError as exc:
        st.error(f"Cannot run the PCA: {exc}")
        return
    source = _source(store, dataset, result.end)
    m = st.columns(3)
    for j in range(3):
        m[j].metric(f"{pca_mod.factor_name(j)} {pca_mod.FACTOR_LABELS[j]}", f"{100 * result.explained[j]:.1f}%")
    view = (data, tuple(tenors), start, end, source)
    _chart("pca_loadings", view, lambda: _loadings_chart(result, source), "rs_pca_loadings")
    _chart("pca_explained", view, lambda: _explained_chart(result), "rs_pca_explained")
    _chart("pca_scores", view, lambda: _scores_chart(result, source), "rs_pca_scores")
    dropped = f"; {result.dropped} spanning more than {pca_mod.MAX_GAP_DAYS} days left out" if result.dropped else ""
    st.caption(
        f"PCA of daily changes in bp of {source}, {result.start} to {result.end} ({result.n_obs} changes{dropped}"
        + f"): eigenvectors of the sample covariance (demeaned, divisor n - 1); the first three explain "
        f"{100 * result.explained[:3].sum():.1f}% of the variance. Signs: PC1 all up = level up, PC2 long end up vs "
        "short end = steepening, PC3 belly up vs the ends. A factor score is a bp move along its unit loading; the "
        f"last chart adds up each day's change projected on the loadings: the curve's move since {result.start} "
        "along each factor."
    )
    try:
        rolling = _rolling(data, tuple(tenors), window, start, end, levels)
    except ValueError as exc:
        st.warning(f"No rolling loadings: {exc}")
        rolling = None
    if rolling is not None:
        factor = st.radio("ROLLING LOADINGS OF", ["PC1", "PC2", "PC3"], horizontal=True, key="rs_pca_heat_factor")
        _chart("pca_heatmap", (view, window, factor), lambda: _heatmap(rolling, factor), "rs_pca_heatmap")
        stable = rolling.stability[factor]
        st.caption(
            f"Loadings of {factor} on windows of {window} daily changes, one every {max(1, window // 12)} changes. "
            f"Shape stability (|cosine| with the whole-period loading): median {stable.median():.2f}, lowest "
            f"{stable.min():.2f} (1 = same shape; a low value means the factor changed meaning, e.g. curvature "
            "trading places with PC4)."
        )
    k = min(3, len(result.tenors))
    loadings = pd.DataFrame(pca_mod.summary_rows(result, k))
    explained = pca_mod.explained_table(result)
    t1, t2 = st.columns(2)
    with t1:
        st.dataframe(loadings, hide_index=True, width="stretch")
        _download("Download loadings CSV", _csv(loadings), file_name="pca_loadings.csv", key="rs_pca_csv")
    with t2:
        st.dataframe(explained, hide_index=True, width="stretch")
        _download("Download explained variance CSV", _csv(explained), file_name="pca_explained.csv",
                           key="rs_pca_explained_csv")  # fmt: skip
    if rolling is not None:
        flat = rolling.loadings.copy()
        flat.columns = [f"{f}_{t}" for f, t in flat.columns]
        _download(
            "Download rolling loadings CSV", _csv(flat.reset_index()), file_name="pca_rolling_loadings.csv",
            key="rs_pca_rolling_csv",
        )  # fmt: skip
    st.caption(f"RMS error = RMS over the dates of each tenor's daily change minus its {k}-factor reconstruction (bp).")
    _render_factor_risk(positions, market, result, store, dataset, rep)


def _render_factor_risk(
    positions: Sequence[Position],
    market: Market,
    result: pca_mod.PCAResult,
    store: rh.HistoryStore,
    dataset: str,
    rep: Optional[Reporting] = None,
) -> None:
    st.markdown("#### / FACTOR RISK OF THE BOOK")
    if not positions:
        st.info(
            "No positions: add trades to the BOOK (or pick the tab trades under RISK SCOPE) to see their factor risk."
        )
        return
    rep = rep if rep is not None else rates_fx.reporting(market)
    try:
        frame = _cached(
            "factor_risk",
            (rates_scenarios.book_risk(positions, market)[0], result.tenors, result.start, result.end, dataset, rep),
            lambda: _factor_risk(positions, market, result, store, rep),
        )
    except ValueError as exc:
        st.error(f"Cannot compute factor risk: {exc}")
        return
    currencies = rates_fx.currencies(positions, market)
    c = rates_fx.table_unit(frame["CCY"])
    shown = frame.rename(columns={col: f"{col} ({c})" for col in frame.columns[2:]}) if c else frame
    st.dataframe(shown, hide_index=True, width="stretch", column_config={col: MONEY for col in shown.columns[2:]})
    _download("Download factor risk CSV", _csv(shown), file_name="pca_factor_risk.csv", key="rs_pca_risk_csv")
    sd = ", ".join(f"{pca_mod.factor_name(j)} {result.sigma_bp[j]:.2f}bp" for j in range(min(3, len(result.tenors))))
    st.caption(
        f"P&L for a +1 sd daily move of each factor ({sd} a day) on every curve of the book: key-rate PV01s (P&L per "
        "+1bp of each curve quote) times the loadings read at each quote's maturity (linear between the PCA tenors, "
        f"flat beyond); basis spread quotes held, as in the PCA factor shock on SCENARIOS. Every curve moves with this "
        f"{rh.dataset_info(dataset).title} factor: for the swap curve that is a proxy (swap spreads held). Vols held. "
        "Amounts in each position's currency (CCY)." + rates_fx.conversion_note(currencies, rep, market.asof)
    )


# --------------------------------------------------------------------------
# VaR
# --------------------------------------------------------------------------


def _histogram(pnl: np.ndarray, var: float, es: float, c: str, label: str) -> go.Figure:
    counts, edges = np.histogram(pnl, bins=min(60, max(10, len(pnl) // 10)))
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig = go.Figure(go.Bar(x=centres, y=counts, width=np.diff(edges), name="DAYS", marker=dict(color=SERIES[2])))
    top = float(counts.max()) * 1.1 if counts.size else 1.0
    for value, name, colour in ((-var, "-VAR", ACCENT), (-es, "-ES", SERIES[3])):
        fig.add_trace(
            go.Scatter(x=[value, value], y=[0.0, top], mode="lines", name=f"{name} ({_num(value, 0)})",
                       line=dict(color=colour, width=2, dash="dash"))
        )  # fmt: skip
    dark_layout(fig, height=380, xtitle=f"ONE-DAY P&L ({c}), {label}", ytitle="SCENARIO DAYS")
    _legend_below(fig)
    fig.update_layout(bargap=0.05)
    lo, hi = float(edges[0]), float(edges[-1])
    span = max(hi - lo, 1.0)
    return bound_axes(
        fig,
        xaxis={"min": min(lo, -es) - 0.02 * span, "max": hi + 0.02 * span, "min_span": span * 0.02},
        yaxis={"min": 0.0, "max": top, "min_span": max(top * 0.05, 1.0)},
    )


def _scenarios(
    market: Market, store: rh.HistoryStore, lookback: int, fx: bool = True
) -> Tuple[tuple, var_mod.VarScenarios]:
    """(cache key, scenarios): keyed on the market, the store's content and
    whether FX moves."""

    key = (rates_risk.market_key(market), store.fingerprint, lookback, fx)
    return key, _cached("var_scen", key, lambda: var_mod.build_scenarios(market, store, lookback, fx=fx))


def var_scenarios(
    market: Market, store: rh.HistoryStore, lookback: int, fx: bool = True
) -> Tuple[tuple, var_mod.VarScenarios]:
    """The VaR scenarios of this tab (cached on the market and the store's
    content), for the BOOK tab: (cache key, scenarios)."""

    return _scenarios(market, store, lookback, fx)


def fx_label(key: str) -> str:
    """Display name of a VaR source row (curves, and the FX spots)."""

    return "FX SPOTS (ECB REFERENCE RATES)" if key == var_mod.FX_KEY else key


def _render_var(
    store: rh.HistoryStore, market: Market, positions: Sequence[Position], rep: Optional[Reporting] = None
) -> None:
    if not positions:
        st.info(
            "No positions yet: add trades to the BOOK, or price a swap, swaption, cap/floor or bond on the other "
            "Rates tabs and pick them under RISK SCOPE."
        )
        return
    c1, c2, c3 = st.columns(3)
    lookback = c1.radio("LOOKBACK (DAILY CHANGES)", var_mod.LOOKBACKS, index=1, horizontal=True, key="rs_var_lookback")
    confidence = c2.radio(
        "CONFIDENCE", var_mod.CONFIDENCES, index=2, horizontal=True, key="rs_var_conf",
        format_func=lambda a: f"{100 * a:g}%",
    )  # fmt: skip
    fx_choice = c3.radio(
        "FX IN VAR", FX_CHOICES, horizontal=True, key="rs_var_fx",
        help="HISTORICAL MOVES: each scenario day's ECB reference-rate change moves today's FX spots, so the book "
        "converted to the reporting currency (ALL IN) carries FX risk; local P&L never moves with FX. HELD: spots "
        "held at today's.",
    )  # fmt: skip
    move_fx = fx_choice == FX_CHOICES[0]
    try:
        key, risk = rates_scenarios.book_risk(positions, market)
        scen_key, scen = _scenarios(market, store, int(lookback), move_fx)
    except ValueError as exc:
        st.warning(f"No VaR for this valuation date: {exc}")
        return
    if scen.short:
        st.warning(
            f"The daily US Treasury history up to {market.asof} holds only {scen.n} daily changes (from "
            f"{scen.previous[0]}), not the {scen.lookback} asked for: this VaR uses those {scen.n} days. Pick a later "
            "valuation date for the full window."
        )
    names = _unique([p.name for p in positions])
    rep = rep if rep is not None else rates_fx.reporting(market)
    currencies = rates_fx.currencies(positions, market)
    headline, c = rates_fx.headline(currencies, rep)
    if headline is None:
        st.warning(
            f"The positions are in {' and '.join(sorted(set(currencies)))} and there are no FX spots to convert "
            "them: the VaR is per currency only (TOTAL rows below). Enter rates under REPORTING CURRENCY."
        )

    converting = rep is not None and rep.needs_conversion(currencies)
    fx = scen.fx if converting else None
    base = _cached("var_base", key, lambda: [float(p.pv(market)) for p in positions]) if fx is not None else None
    if converting and move_fx and scen.fx is None:
        st.warning(f"{scen.fx_note}: the converted VaR holds the spots.")

    def measures():
        pnl = var_mod.sensitivity_pnl(scen, risk)
        delta = var_mod.sensitivity_pnl(scen, risk, gamma=False)
        return pnl, delta, var_mod.var_table(pnl, names, float(confidence), delta, currencies, rep, fx, base)

    try:
        pnl, delta, table = _cached("var_measures", (key, scen_key, float(confidence), rep), measures)
    except ValueError as exc:
        st.error(f"Cannot compute VaR: {exc}")
        return
    full_key = (key, scen_key)
    full_store = st.session_state.setdefault("_rs_var_full", {})
    sources = scen.sources_table()
    sources["flag"] = sources["flag"].map(FLAG_LABELS).fillna(sources["flag"])
    sources["curve"] = [fx_label(k) if k == var_mod.FX_KEY else rates_risk.curve_title(market, k)
                        for k in sources["curve"]]  # fmt: skip
    st.dataframe(sources.rename(columns=str.upper), hide_index=True, width="stretch")
    if st.button(f"RUN FULL REVALUATION ({scen.n} SCENARIOS)", key="rs_var_full"):
        bar = st.progress(0.0, text="Revaluing the book on every scenario day...")
        try:
            result = var_mod.full_revaluation_pnl(
                positions, market, scen, progress=lambda i, n: bar.progress(i / n) if i % 25 == 0 or i == n else None
            )
        except ValueError as exc:
            st.error(f"Full revaluation failed: {exc}")
            result = None
        bar.empty()
        if result is not None:
            if len(full_store) >= 4:
                full_store.pop(next(iter(full_store)))
            full_store[full_key] = result
    full = full_store.get(full_key)
    rows = table.set_index("position")
    total = rows.loc[headline] if headline is not None else None
    level = f"{100 * confidence:g}%"
    m = st.columns(3) + st.columns(3)  # three cards a row: four wrap next to the sidebar
    m[0].metric(f"HIST VAR {level} ({c})", "per currency" if total is None else _num(total["var"], 0))
    m[1].metric(f"HIST ES {level} ({c})", "per currency" if total is None else _num(total["es"], 0))
    m[2].metric(f"DELTA-NORMAL VAR ({c})", "per currency" if total is None else _num(total["dn_var"], 0))
    columns = var_mod.book_columns(full, names, currencies, rep, fx, base) if full is not None else None
    full_var, full_es = (var_mod.var_es(columns[headline][0], float(confidence))
                         if columns is not None and headline is not None else (None, None))  # fmt: skip
    m[3].metric(f"FULL-REVAL VAR ({c})", "not run" if full_var is None else _num(full_var, 0))
    m[4].metric(f"FULL-REVAL ES ({c})", "not run" if full_es is None else _num(full_es, 0))
    m[5].metric("SCENARIO DAYS", f"{scen.n:,}")
    if fx is not None and headline is not None:
        held = rows.loc[f"{headline} {var_mod.FX_HELD}"]
        only = rows.loc[f"{headline} {var_mod.FX_ONLY}"]
        f1, f2, f3 = st.columns(3)
        f1.metric(f"VAR FX HELD ({c})", _num(held["var"], 0))
        f2.metric(f"FX-ONLY VAR ({c})", _num(only["var"], 0))
        f3.metric(f"FX ADD-ON ({c})", _num(total["var"] - held["var"], 0))
    shown_pnl, label = (full, "FULL REVALUATION") if full is not None else (pnl, "DELTA + GAMMA")
    book_columns = var_mod.book_columns(shown_pnl, names, currencies, rep, fx, base)
    book_label = headline if headline is not None else next(k for k in book_columns if k.startswith("TOTAL"))
    book, book_ccy = book_columns[book_label]
    book_unit = c if headline is not None else rates_fx.symbol(book_ccy).strip()
    v, e = var_mod.var_es(book, float(confidence))
    hist_key = (key, scen_key, float(confidence), label, book_unit, rep)
    _chart("var_hist", hist_key, lambda: _histogram(book, v, e, book_unit, label), "rs_var_hist")
    unit = rates_fx.table_unit(table["currency"])
    u = f" ({unit})" if unit else ""
    labels = {
        "position": "POSITION", "currency": "CCY", "var": f"HIST VAR{u}", "es": f"HIST ES{u}",
        "worst_loss": f"WORST LOSS{u}", "mean_pnl": f"MEAN P&L{u}", "sd_pnl": f"SD P&L{u}",
        "dn_sigma": f"DELTA-NORMAL SD{u}", "dn_var": f"DELTA-NORMAL VAR{u}", "dn_es": f"DELTA-NORMAL ES{u}",
    }  # fmt: skip
    shown = table.rename(columns=labels)
    if full is not None:
        full_table = var_mod.var_table(full, names, float(confidence), None, currencies, rep, fx, base)
        shown[f"FULL-REVAL VAR{u}"] = full_table["var"].to_numpy()
        shown[f"FULL-REVAL ES{u}"] = full_table["es"].to_numpy()
    st.dataframe(shown, hide_index=True, width="stretch", column_config={col: MONEY for col in shown.columns[2:]})
    _download("Download VaR table CSV", _csv(shown), file_name="var_table.csv", key="rs_var_csv")
    try:
        vegas = _cached("vegas", key, lambda: position_vegas(positions, market))
    except ValueError:
        vegas = None
    try:
        buckets = _cached("vega_buckets", key, lambda: [pf.position_vega_buckets(p, market) for p in positions])
    except ValueError:
        buckets = None
    report = var_mod.risk_report(names, risk, table, vegas, [p.vol_model for p in positions], currencies, buckets)
    _download(
        "Download risk report CSV (key-rate PV01, gamma, vega and vega buckets, VaR)", _csv(report),
        file_name="risk_report.csv", key="rs_var_report_csv",
    )  # fmt: skip
    worst = var_mod.worst_days(scen, book, 10, currencies if fx is not None else None,
                               rep.currency if rep is not None else None)  # fmt: skip
    names_by_column = {"pnl": f"BOOK P&L ({book_unit})", "date": "DATE", "from": "FROM"}
    names_by_column.update({col: col.upper() for col in worst.columns if col.endswith("move (%)")})
    for curve_key in scen.bumps:
        title = rates_risk.curve_title(market, curve_key)
        names_by_column[f"{curve_key} max move (bp)"] = f"{title}: LARGEST QUOTE MOVE (BP)"
        names_by_column[f"{curve_key} at"] = f"{title}: AT"
    worst = worst.rename(columns=names_by_column)
    st.markdown("#### / WORST DAYS")
    one_dp = st.column_config.NumberColumn(format="%.1f")
    two_dp = st.column_config.NumberColumn(format="%+.2f")
    worst_format = {col: (MONEY if col.startswith("BOOK P&L") else one_dp) for col in worst.columns if "(BP)" in col
                    or col.startswith("BOOK P&L")}  # fmt: skip
    worst_format.update({col: two_dp for col in worst.columns if col.endswith("MOVE (%)")})
    st.dataframe(worst, hide_index=True, width="stretch", column_config=worst_format)
    mixed = len(set(currencies)) > 1 or (rep is not None and rep.needs_conversion(currencies))
    scenario_frame = pd.DataFrame({(f"{n} ({ccy})" if mixed else n): values
                                   for n, (values, ccy) in book_columns.items()})  # fmt: skip
    scenario_frame.insert(0, "from", list(scen.previous))
    scenario_frame.insert(0, "date", list(scen.dates))
    _download("Download scenario P&L CSV", _csv(scenario_frame), file_name="var_scenario_pnl.csv", key="rs_var_pnl_csv")
    rates_exports.export_row(
        "value at risk", market, {"var_table": shown, "scenario_pnl": scenario_frame, "risk_report": report,
                                  "worst_days": worst, "history_sources": sources},
        {"positions": [{"name": p.name, "kind": p.kind, "currency": ccy, "details": repr(p.details)}
                       for p, ccy in zip(positions, currencies)],
         "reporting": rates_fx.record(rep, currencies),
         "lookback_days": int(lookback), "confidence": float(confidence), "full_revaluation": full is not None,
         "fx_in_var": fx_choice},
        {"scenarios": "historical daily quote moves of each curve's own history or a flagged proxy, applied to "
         "today's curves", "pnl": label, "vols": "held", "var": "ceil(n (1 - confidence))-th largest loss",
         "es": "Acerbi-Tasche", "delta_normal": "z x sd of first-order P&L, mean zero",
         "fx": (scen.fx_note + "; converted P&L (V + p) S (1 + x) - V S") if fx is not None
         else "spots held at today's"},
        key="rs_var_export", store=store,
    )  # fmt: skip
    st.caption(
        f"One-day {level} VaR and expected shortfall of the book on the valuation date ({market.asof}), from {scen.n} "
        f"daily changes {scen.previous[0]} to {scen.dates[-1]}: each day's change of each curve's history (table "
        "above: which history drives which curve, and whether it is the curve's own market) moves today's quotes, "
        "linear in maturity between the history's tenors. HIST: P&L = key-rate PV01 x move + 1/2 diagonal gamma x "
        "move^2; FULL REVALUATION rebuilds the curves and reprices every position. VaR = the ceil(n x (1 - "
        "confidence))-th largest loss; ES = average loss beyond it (Acerbi-Tasche). Delta-normal: z x sd of the "
        "first-order P&L, mean taken as zero. Vols are held (no free vol history); no carry over the day. Losses "
        "are positive numbers, in each row's currency (CCY): a TOTAL row adds the P&L of one currency's positions "
        "per scenario."
        + rates_fx.conversion_note(currencies, rep, market.asof)
        + " "
        + (
            f"{scen.fx_note}: {rep.label} = each position's value after the day, converted at today's spot moved by "
            f"that day's ECB change, minus today's converted value; {rep.label} {var_mod.FX_HELD} holds the spots, "
            f"{rep.label} {var_mod.FX_ONLY} moves only FX (today's values), and the delta-normal check treats the FX "
            "returns as risk factors too. Local P&L and the TOTAL rows do not move with FX."
            if fx is not None and rep is not None
            else "FX spots are held at today's (FX IN VAR = HELD, or nothing to convert)."
        )
        + " The risk report CSV holds the key-rate PV01s and gammas, each option's vega "
        "and its vega by bucket (swaptions by option expiry x swap tenor: the SABR cube's ATM points for book "
        "swaptions priced on the cube, the vega split between the nearest grid points otherwise; caps by caplet "
        "expiry) and the VaR table."
    )


# --------------------------------------------------------------------------
# Tab
# --------------------------------------------------------------------------


def render(market: Market, positions: List[Position], scope: str = "", reporting: Optional[Reporting] = None) -> None:
    """The RESEARCH tab: RV, carry and roll-down, PCA and VaR sub-tabs (PCA
    factor risk and VaR on the positions in scope, totalled per currency and
    in the ``reporting`` currency; default: the sidebar's)."""

    st.markdown("### / RESEARCH")
    if scope:
        st.caption(f"FACTOR RISK AND VAR ON: {scope} ({len(positions)} positions); change it with RISK SCOPE in the "
                   "sidebar.")  # fmt: skip
    store = rates_scenarios.history_store(market)
    tabs = st.tabs(list(SUBTABS))
    with tabs[0]:
        _render_rv(store, market)
    with tabs[1]:
        _render_carry(store, market)
    with tabs[2]:
        _render_pca(store, market, positions, reporting)
    with tabs[3]:
        _render_var(store, market, positions, reporting)
