"""FX OPTIONS page: Garman-Kohlhagen with FX market conventions.

* Pairs (:data:`derivx.fx.conventions.PAIRS`) with their pip, precision
  (a tenth of a pip: 5 decimals, 3 for JPY pairs), spot lag, calendars and
  market premium / delta conventions.
* Dates: trade, spot, expiry and delivery dates from the tenor rules; the
  vol time runs trade -> expiry and the rate time spot -> delivery
  (ACT/365F). The premium is paid on the spot date.
* Carry from the two deposit rates (continuously compounded) or from
  forward points per tenor; the forward and its points are shown.
* A smile per tenor from ATM / RR25 / BF25 / RR10 / BF10 quotes in an
  editable table (ILLUSTRATIVE defaults: FX vol quotes are not free), in a
  chosen delta / ATM / strangle convention, with SABR (beta = 1 fixed) or
  vanna-volga (:mod:`derivx.fx.smile`); the option is priced at its smile
  vol unless a flat vol overrides it.
* Spot defaults to the ECB euro reference rate of the bundled snapshot; a
  button fetches the latest one (cached, short timeout, offline fallback).

Every chart is zoom-bounded by :mod:`derivx.ui.plot_limits` and styled by
:mod:`derivx.ui.theme`. Widget keys start with ``fx_``; the market inputs
(spot, strike, rates, quotes) are kept per pair (``fx_spot_EURUSD``, ...)
and persist across mode switches (:mod:`derivx.ui.state`).
"""

from __future__ import annotations

import datetime as dt
import html
import math
from dataclasses import dataclass, replace
from typing import List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.fx import market_data
from derivx.fx.conventions import (
    DELTA_TYPE_LABELS,
    DELTA_TYPES,
    OPTION_TENORS,
    PAIRS,
    CurrencyPair,
    FxDates,
    forward_points,
    fx_dates,
    implied_foreign_rate,
    market_delta_type,
)
from derivx.fx.examples import ILLUSTRATIVE_QUOTES, ILLUSTRATIVE_RATES
from derivx.fx.pricing import (
    ATM_TYPE_LABELS,
    ATM_TYPES,
    FxGreeks,
    FxMarket,
    atm_strike,
    base_notional,
    fx_greeks,
    premium_amount,
    premium_styles,
    scenario_pnl,
    strike_from_delta,
)
from derivx.fx.smile import (
    MODEL_LABELS,
    NOTE_TOL,
    SMILE_DELTAS,
    SMILE_MODELS,
    STRANGLE_CONVENTIONS,
    STRANGLE_LABELS,
    SURFACE_DELTAS,
    FxSmile,
    SmileConventions,
    TenorQuote,
    build_smile,
    density_check,
    interpolate_quote,
    smile_by_delta,
    smile_strikes,
)
from derivx.fx.smile_risk import quote_vegas
from derivx.fx.smile_risk import sabr_parameter_vegas as sabr_param_vegas
from derivx.ui import pricer_calc as calc
from derivx.ui.plot_limits import GUARD_AVAILABLE, axis_limits, bound_axes
from derivx.ui.pricer_page import PAGE_CSS, ZOOM_HINT
from derivx.ui.state import PERSIST
from derivx.ui.theme import ACCENT, FOREGROUND, PLOTLY_2D_CONFIG, PNL_COLORS, SEQUENTIAL_COLORS, SERIES, dark_layout

QUOTE_COLUMNS = ("ATM (%)", "RR25 (%)", "BF25 (%)", "RR10 (%)", "BF10 (%)")
POINTS_COLUMN = "FWD PTS (PIPS)"
EXPIRY_CHOICES = OPTION_TENORS + ("DATE",)
STRIKE_MODES = ("STRIKE", "DELTA", "ATM")
CARRY_MODES = ("RATES", "FORWARD POINTS")
VOL_SOURCES = ("SMILE", "FLAT")
SPACES = ("STRIKE", "DELTA")
DEFAULT_TENOR = "3M"
CARD_VALUE_CHARS = 14  # longest Greek-card value that fits the narrowest card of FX_CSS (desktop and phone)
FX_CSS = """
<style>
/* Greek cards: at least 14rem wide at 1.3rem, so CARD_VALUE_CHARS monospace characters fit (a 100m notional's
   gamma is EUR 10,336,703); longer values are shortened by _card_money, never cut off */
.dx-greeks { grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr)); }
.dx-greek-value { font-size: 1.3rem; }
/* the sidebar captions (spot source, as-of date, ILLUSTRATIVE labels) use the dark theme's caption colour:
   white at 60% opacity, 7.1:1 on black */
@media (max-width: 640px) {
  .dx-greek-value { font-size: 1.05rem; }
  .dx-greeks { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.5rem; }
  .dx-greek { padding: 0.55rem 0.6rem; }
}
</style>
"""
SHORT_LABELS = {
    "MARKET": "MARKET (PAIR DEFAULT)",
    "spot": "SPOT",
    "forward": "FORWARD",
    "spot_pa": "SPOT, PREMIUM-ADJ.",
    "forward_pa": "FORWARD, PREMIUM-ADJ.",
    "DNS": "ATM DNS",
    "ATMF": "ATMF",
    "SPOT": "ATM SPOT",
    "market": "MARKET STRANGLE",
    "smile": "SMILE STRANGLE",
    "SABR": "SABR (β = 1 FIXED)",
    "VV": "VANNA-VOLGA",
}
SCENARIO_SPOT_STEPS = calc.SHOCK_STEPS
SCENARIO_VOL_STEPS = calc.VOL_STEPS


def _k(name: str, pair: Optional[CurrencyPair] = None) -> str:
    return f"fx_{name}_{pair.code}" if pair is not None else f"fx_{name}"


# ---------------------------------------------------------------------------
# Defaults and state
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _bundled() -> Optional[market_data.EcbRates]:
    """The bundled ECB snapshot (a file shipped with the app), read once."""

    try:
        return market_data.bundled_ecb_rates()
    except market_data.FxDataError:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _live_ecb() -> market_data.EcbRates:
    """Latest ECB rates, cached for an hour for every session (keyed on
    nothing but the time to live: the rates are the same for everyone)."""

    return market_data.fetch_ecb_rates()


def _reference(pair: CurrencyPair) -> Tuple[Optional[float], Optional[market_data.EcbRates]]:
    """(reference spot, its source): the live ECB rates when fetched in this
    session, else the bundled snapshot."""

    for rates in (st.session_state.get(_k("ecb_live")), _bundled()):
        if rates is not None:
            try:
                return round(rates.cross(pair), pair.decimals), rates
            except market_data.FxDataError:
                continue
    return None, None


def _quotes_frame(pair: CurrencyPair, S: float, rd: float, rf: float, trade: dt.date) -> pd.DataFrame:
    rows = []
    for tenor, atm, rr25, bf25, rr10, bf10 in ILLUSTRATIVE_QUOTES[pair.code]:
        d = fx_dates(pair, trade, tenor=tenor)
        F = S * math.exp((rd - rf) * d.tau_d)
        rows.append([tenor, atm, rr25, bf25, rr10, bf10, round(forward_points(S, F, pair.pip), 2)])
    return pd.DataFrame(rows, columns=["TENOR", *QUOTE_COLUMNS, POINTS_COLUMN])


def is_default_rate(ccy: str, value_pct: Optional[float]) -> bool:
    """True when a rate input (percent) is still the ILLUSTRATIVE default of ``ccy``."""

    return value_pct is not None and math.isclose(float(value_pct), 100 * ILLUSTRATIVE_RATES[ccy], abs_tol=1e-9)


def _record_points(pair: CurrencyPair) -> None:
    """Remember the forward points just written to the quotes table and
    whether they came from the ILLUSTRATIVE default rates, so the page can
    say when the carry is still illustrative."""

    ss = st.session_state
    illustrative = is_default_rate(pair.quote, ss[_k("rd", pair)]) and is_default_rate(pair.base, ss[_k("rf", pair)])
    points = tuple(float(x) for x in ss[_k("quotes", pair)][POINTS_COLUMN])
    ss[_k("points_ref", pair)] = (points, illustrative)


def _init_pair(pair: CurrencyPair) -> None:
    ss = st.session_state
    ss.setdefault(_k("trade_date"), dt.date.today())
    spot, _ = _reference(pair)
    spot = spot if spot is not None else 1.0
    ss.setdefault(_k("spot", pair), spot)
    big_figure = pair.pip * 100
    ss.setdefault(_k("strike", pair), round(round(spot / big_figure) * big_figure, pair.decimals))
    ss.setdefault(_k("rd", pair), 100 * ILLUSTRATIVE_RATES[pair.quote])
    ss.setdefault(_k("rf", pair), 100 * ILLUSTRATIVE_RATES[pair.base])
    ss.setdefault(_k("premium_ccy", pair), pair.premium_ccy)
    ss.setdefault(_k("notional_ccy", pair), pair.base)
    ss.setdefault(_k("editor_epoch", pair), 0)
    if _k("quotes", pair) not in ss:
        ss[_k("quotes", pair)] = _quotes_frame(
            pair, float(ss[_k("spot", pair)]), ss[_k("rd", pair)] / 100, ss[_k("rf", pair)] / 100, ss[_k("trade_date")]
        )
        _record_points(pair)


def _init_state() -> None:
    defaults = {
        "pair": "EURUSD",
        "expiry": DEFAULT_TENOR,
        "type": "CALL",
        "side": "BUY",
        "strike_mode": "STRIKE",
        "delta": 25.0,
        "notional": 1_000_000.0,
        "delta_conv": "MARKET",
        "atm_type": "DNS",
        "strangle": "market",
        "model": "SABR",
        "carry": "RATES",
        "vol_src": "SMILE",
        "flat_vol": 8.0,
        "space": "STRIKE",
        "smile_sd": 2.5,
        "scn_spot": 3.0,
        "scn_vol": 2.0,
        "scn_days": 0.0,
    }
    for name, value in defaults.items():
        st.session_state.setdefault(_k(name), value)


def apply_editor_changes(base: pd.DataFrame, state: Optional[Mapping]) -> pd.DataFrame:
    """``base`` with the edited cells of a ``st.data_editor`` state applied
    (rows and columns are fixed: added or deleted rows are ignored).
    Idempotent: applying the same state twice gives the same frame."""

    out = base.copy()
    for row, changes in ((state or {}).get("edited_rows") or {}).items():
        i = int(row)
        if not 0 <= i < len(out):
            continue
        for col, value in changes.items():
            if col in out.columns and col != "TENOR":
                out.iloc[i, out.columns.get_loc(col)] = np.nan if value is None else value
    return out


def edited_tenors(pair: CurrencyPair, frame: pd.DataFrame) -> List[str]:
    """Tenors whose vol quotes differ from the illustrative defaults."""

    defaults = {row[0]: row[1:] for row in ILLUSTRATIVE_QUOTES[pair.code]}
    out = []
    for _, r in frame.iterrows():
        given = np.array([np.nan if r[c] is None else float(r[c]) for c in QUOTE_COLUMNS])
        ref = np.array(defaults.get(str(r["TENOR"]), (np.nan,) * len(QUOTE_COLUMNS)), dtype=float)
        if not np.array_equal(given, ref, equal_nan=True):
            out.append(str(r["TENOR"]))
    return out


def _merge_quote_edits(pair: CurrencyPair) -> None:
    """on_change of the quotes editor: keep the edits in the pair's base
    table (plain session state, so they survive mode switches)."""

    key = _editor_key(pair)
    st.session_state[_k("quotes", pair)] = apply_editor_changes(
        st.session_state[_k("quotes", pair)], st.session_state.get(key)
    )


def _reset_quotes(pair: CurrencyPair) -> None:
    ss = st.session_state
    ss[_k("quotes", pair)] = _quotes_frame(
        pair, float(ss[_k("spot", pair)]), ss[_k("rd", pair)] / 100, ss[_k("rf", pair)] / 100, ss[_k("trade_date")]
    )
    _record_points(pair)
    ss[_k("editor_epoch", pair)] = ss.get(_k("editor_epoch", pair), 0) + 1


def _points_from_rates(pair: CurrencyPair) -> None:
    ss = st.session_state
    base = ss[_k("quotes", pair)].copy()
    S, rd, rf = float(ss[_k("spot", pair)]), ss[_k("rd", pair)] / 100, ss[_k("rf", pair)] / 100
    for i, tenor in enumerate(base["TENOR"]):
        d = fx_dates(pair, ss[_k("trade_date")], tenor=str(tenor))
        base.iloc[i, base.columns.get_loc(POINTS_COLUMN)] = round(
            forward_points(S, S * math.exp((rd - rf) * d.tau_d), pair.pip), 2
        )
    ss[_k("quotes", pair)] = base
    _record_points(pair)
    ss[_k("editor_epoch", pair)] = ss.get(_k("editor_epoch", pair), 0) + 1


def _editor_key(pair: CurrencyPair) -> str:
    return f"fx_quotes_editor_{pair.code}_{st.session_state.get(_k('editor_epoch', pair), 0)}"


def _fetch_live(pair: CurrencyPair) -> None:
    """on_click of the ECB button: fetch (cached), set the spot, or keep the
    current spot and say why the fetch failed."""

    try:
        rates = _live_ecb()
    except market_data.FxDataError as exc:
        st.session_state[_k("ecb_error")] = f"{exc}. Kept the current spot; the bundled snapshot is the fallback."
        return
    st.session_state[_k("ecb_live")] = rates
    st.session_state.pop(_k("ecb_error"), None)
    try:
        st.session_state[_k("spot", pair)] = round(rates.cross(pair), pair.decimals)
    except market_data.FxDataError as exc:
        st.session_state[_k("ecb_error")] = str(exc)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FxInputs:
    pair: CurrencyPair
    S: float
    trade: dt.date
    dates: FxDates
    call: bool
    sign: float  # +1 buy, -1 sell
    strike_mode: str
    strike: Optional[float]
    delta_pct: float
    notional: float
    notional_in_base: bool
    premium_in_base: bool
    delta_conv: str
    atm_type: str
    strangle: str
    model: str
    carry: str
    rd: float  # decimals
    rf: float  # decimals
    vol_src: str
    flat_vol: float  # decimal


def _num(label: str, key: str, container=None, **kwargs):
    box = container if container is not None else st.sidebar
    return box.number_input(label, key=key, persist_state=PERSIST, **kwargs)


def _sidebar() -> Tuple[Optional[FxInputs], List[str]]:
    sb = st.sidebar
    ss = st.session_state
    sb.markdown("### / FX PARAMS")
    code = sb.selectbox(
        "CURRENCY PAIR",
        list(PAIRS),
        key=_k("pair"),
        persist_state=PERSIST,
        format_func=lambda c: PAIRS[c].label,
        help="BASE/QUOTE: the price of one unit of the base currency in the quote currency.",
    )
    pair = PAIRS[code]
    _init_pair(pair)
    step = pair.pip
    fmt = f"%.{pair.decimals}f"
    S = _num(f"SPOT ({pair.quote} per 1 {pair.base})", _k("spot", pair), min_value=0.0, step=step, format=fmt)
    ref, source = _reference(pair)
    if source is not None:
        edited = S is not None and ref is not None and abs(float(S) - ref) > 0.5 * 10.0**-pair.decimals
        sb.caption(("EDITED. Reference: " if edited else "") + f"{pair.fmt(ref)} = {html.escape(source.description)}.")
    sb.button(
        "FETCH LATEST ECB RATE",
        key=_k("fetch_ecb"),
        on_click=_fetch_live,
        args=(pair,),
        help=f"Fetches the ECB reference rates (timeout {market_data.DEFAULT_TIMEOUT:g} s, cached for an hour) and "
        "sets the spot. Offline, the bundled snapshot stays the default.",
    )
    if ss.get(_k("ecb_error")):
        sb.warning(ss[_k("ecb_error")])
    trade = sb.date_input("TRADE DATE", key=_k("trade_date"), persist_state=PERSIST)
    choice = sb.selectbox("EXPIRY", EXPIRY_CHOICES, key=_k("expiry"), persist_state=PERSIST)
    expiry_date = None
    if choice == "DATE":
        ss.setdefault(_k("expiry_date"), trade + dt.timedelta(days=91))
        if ss[_k("expiry_date")] < trade:  # the trade date moved past it: an expiry cannot precede the trade
            ss[_k("expiry_date")] = trade
        expiry_date = sb.date_input("EXPIRY DATE", key=_k("expiry_date"), min_value=trade, persist_state=PERSIST)
    # Labels and options of a keyed widget shared by every pair stay fixed: a pair-dependent label would make
    # Streamlit treat it as a new widget and the browser would show no selection.
    kind = sb.radio(
        "OPTION (ON THE BASE CURRENCY)", ("CALL", "PUT"), horizontal=True, key=_k("type"), persist_state=PERSIST
    )
    sb.caption(f"{pair.base} {kind} / {pair.quote} {'PUT' if kind == 'CALL' else 'CALL'}")
    side = sb.radio("SIDE", ("BUY", "SELL"), horizontal=True, key=_k("side"), persist_state=PERSIST)
    mode = sb.radio("STRIKE BY", STRIKE_MODES, horizontal=True, key=_k("strike_mode"), persist_state=PERSIST)
    K = None
    if mode == "STRIKE":
        K = _num(f"STRIKE ({pair.quote} per 1 {pair.base})", _k("strike", pair), min_value=0.0, step=step, format=fmt)
    elif mode == "DELTA":
        _num(
            "DELTA (%, of the option; quote convention)",
            _k("delta"),
            min_value=0.01,
            max_value=99.0,
            step=1.0,
            format="%.2f",
            help="25 = the 25-delta option of the chosen type, in the delta convention of the quotes.",
        )
    sb.markdown("**/ NOTIONAL AND PREMIUM**")
    ccys = [pair.base, pair.quote]
    notional = _num("NOTIONAL", _k("notional"), min_value=0.0, step=100_000.0, format="%.0f")
    n_ccy = sb.radio("NOTIONAL CCY", ccys, horizontal=True, key=_k("notional_ccy", pair), persist_state=PERSIST)
    p_ccy = sb.radio(
        "PREMIUM CCY",
        ccys,
        horizontal=True,
        key=_k("premium_ccy", pair),
        persist_state=PERSIST,
        help=f"Market default for {pair.label}: {pair.premium_ccy}. A premium in {pair.base} makes the hedge delta "
        "premium-adjusted.",
    )
    sb.markdown("**/ CARRY**")
    carry = sb.radio("CARRY FROM", CARRY_MODES, horizontal=True, key=_k("carry"), persist_state=PERSIST)
    rd = _num(
        f"{pair.quote} RATE (%, CONT. ACT/365F)",
        _k("rd", pair),
        min_value=-50.0,
        max_value=50.0,
        step=0.05,
        format="%.4f",
    )
    rf = None
    if carry == "RATES":
        rf = _num(
            f"{pair.base} RATE (%, CONT. ACT/365F)",
            _k("rf", pair),
            min_value=-50.0,
            max_value=50.0,
            step=0.05,
            format="%.4f",
        )
    defaults = [c for c, v in ((pair.quote, rd), (pair.base, rf)) if is_default_rate(c, v)]
    if defaults:
        sb.caption(
            f"ILLUSTRATIVE: the {' and '.join(defaults)} rate{'s are' if len(defaults) > 1 else ' is a'} round-number "
            f"default{'s' if len(defaults) > 1 else ''} (flat, not a fixing or a curve), not market data. Enter your "
            "deposit rates" + (" or forward points." if carry == "RATES" else "; the forward comes from the points.")
        )
    sb.markdown("**/ VOL QUOTE CONVENTIONS**")
    delta_conv = sb.selectbox(
        "DELTA",
        ("MARKET",) + DELTA_TYPES,
        key=_k("delta_conv"),
        persist_state=PERSIST,
        format_func=SHORT_LABELS.get,
        help="MARKET: spot delta to 1Y and forward delta beyond; premium-adjusted when the pair's market premium "
        "currency is the base currency (USDJPY, USDCHF, USDCAD and the EUR crosses).",
    )
    atm_type = sb.selectbox(
        "ATM",
        ATM_TYPES,
        key=_k("atm_type"),
        persist_state=PERSIST,
        format_func=SHORT_LABELS.get,
        help="DNS: delta-neutral straddle (premium-adjusted with premium-adjusted deltas); ATMF: K = forward; "
        "SPOT: K = spot.",
    )
    strangle = sb.selectbox(
        "BUTTERFLY",
        STRANGLE_CONVENTIONS,
        key=_k("strangle"),
        persist_state=PERSIST,
        format_func=SHORT_LABELS.get,
        help="MARKET: the broker fly (one vol ATM + BF for both strikes; the smile reprices that strangle). "
        "SMILE: BF = (σ(25C) + σ(25P)) / 2 − ATM on the smile.",
    )
    model = sb.selectbox(
        "SMILE MODEL", SMILE_MODELS, key=_k("model"), persist_state=PERSIST, format_func=SHORT_LABELS.get
    )
    sb.markdown("**/ VOL**")
    vol_src = sb.radio(
        "PRICE AT",
        VOL_SOURCES,
        horizontal=True,
        key=_k("vol_src"),
        persist_state=PERSIST,
        format_func=lambda v: "SMILE VOL" if v == "SMILE" else "FLAT VOL",
    )
    flat = None
    if vol_src == "FLAT":
        flat = _num("FLAT VOL (%)", _k("flat_vol"), min_value=0.0, max_value=300.0, step=0.05, format="%.3f")

    errors = []
    for label, value in (("SPOT", S), ("NOTIONAL", notional), ("RATE", rd)):
        if value is None or not np.isfinite(value):
            errors.append(f"{label} is empty: enter a number.")
    if S is not None and S <= 0:
        errors.append(f"SPOT must be greater than 0 (got {S:g}): {pair.label} trades near {pair.fmt(ref or 1.0)}.")
    if mode == "STRIKE" and (K is None or K <= 0):
        errors.append(f"STRIKE must be greater than 0 (got {K if K is not None else 'nothing'}).")
    if carry == "RATES" and (rf is None or not np.isfinite(rf)):
        errors.append(f"{pair.base} RATE is empty: enter a number.")
    if vol_src == "FLAT" and (flat is None or flat <= 0):
        errors.append("FLAT VOL must be greater than 0 (vol points, 8 = 8%).")
    try:
        dates = fx_dates(pair, trade, tenor=None if choice == "DATE" else choice, expiry=expiry_date)
    except ValueError as exc:
        errors.append(f"EXPIRY: {exc}.")
        dates = None
    if errors:
        return None, errors
    inputs = FxInputs(
        pair=pair,
        S=float(S),
        trade=trade,
        dates=dates,
        call=kind == "CALL",
        sign=1.0 if side == "BUY" else -1.0,
        strike_mode=mode,
        strike=float(K) if K is not None else None,
        delta_pct=float(ss[_k("delta")]),
        notional=float(notional),
        notional_in_base=n_ccy == pair.base,
        premium_in_base=p_ccy == pair.base,
        delta_conv=delta_conv,
        atm_type=atm_type,
        strangle=strangle,
        model=model,
        carry=carry,
        rd=float(rd) / 100.0,
        rf=float(rf) / 100.0 if rf is not None else float("nan"),
        vol_src=vol_src,
        flat_vol=float(flat) / 100.0 if flat is not None else float("nan"),
    )
    return inputs, []


# ---------------------------------------------------------------------------
# Quotes and smiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenorRow:
    tenor: str
    dates: FxDates
    quote: Optional[TenorQuote]
    points: Optional[float]
    error: Optional[str]


def parse_quote_rows(pair: CurrencyPair, trade: dt.date, frame: pd.DataFrame) -> List[TenorRow]:
    """Tenor rows of the quotes table (percent -> decimals), each with its
    dates and either a quote or the reason it is unusable."""

    rows = []
    for _, r in frame.iterrows():
        tenor = str(r["TENOR"])
        dates = fx_dates(pair, trade, tenor=tenor)
        vals = []
        for col in QUOTE_COLUMNS:
            v = r[col]
            vals.append(None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))
        pts = r.get(POINTS_COLUMN)
        pts = None if pts is None or (isinstance(pts, float) and math.isnan(pts)) else float(pts)
        atm, rr25, bf25, rr10, bf10 = vals
        error, quote = None, None
        if atm is None or rr25 is None or bf25 is None:
            error = f"{tenor}: ATM, RR25 and BF25 are required"
        elif not 0 < atm <= 300:
            error = f"{tenor}: ATM must be above 0 and at most 300 vol points (7.5 = 7.5%), got {atm:g}"
        elif any(v is not None and abs(v) > 100 for v in (rr25, bf25, rr10, bf10)):
            error = f"{tenor}: RR and BF are in vol points (0.25 = 0.25%); |value| above 100 is not a quote"
        else:
            try:
                quote = TenorQuote(
                    tenor,
                    atm / 100,
                    rr25 / 100,
                    bf25 / 100,
                    None if rr10 is None else rr10 / 100,
                    None if bf10 is None else bf10 / 100,
                )
            except ValueError as exc:
                error = str(exc).replace("decimal, 0.075 = 7.5%", "vol points, 7.5 = 7.5%")
        rows.append(TenorRow(tenor, dates, quote, pts, error))
    return rows


def _foreign_rate(inp: FxInputs, tau_d: float, points: Optional[float]) -> Tuple[float, Optional[str]]:
    """rf for one tenor: the input rate, or implied from its forward points."""

    if inp.carry == "RATES":
        return inp.rf, None
    if points is None:
        return float("nan"), "no forward points"
    if tau_d <= 0:
        return inp.rd, None  # delivery on the spot date: the forward is spot, rf does not matter
    F = inp.S + points * inp.pair.pip
    if F <= 0:
        return float("nan"), f"spot + {points:g} pips is not a positive forward"
    return implied_foreign_rate(inp.S, F, inp.rd, tau_d), None


def illustrative_carry(inp: FxInputs) -> List[str]:
    """The carry inputs of the selected expiry that are still ILLUSTRATIVE
    defaults (empty when every one was entered): the domestic rate, the
    foreign rate (RATES) or the forward points computed from the default
    rates (FORWARD POINTS; for an expiry between tenors, any tenor's)."""

    pair, ss = inp.pair, st.session_state
    out = []
    if is_default_rate(pair.quote, 100 * inp.rd):
        out.append(f"{pair.quote} rate {100 * inp.rd:.2f}%")
    if inp.carry == "RATES":
        if is_default_rate(pair.base, 100 * inp.rf):
            out.append(f"{pair.base} rate {100 * inp.rf:.2f}%")
        return out
    ref = ss.get(_k("points_ref", pair))
    if ref is None or not ref[1]:
        return out
    frame = ss[_k("quotes", pair)]
    unchanged = {}
    for tenor, now, then in zip(frame["TENOR"], frame[POINTS_COLUMN], ref[0]):
        now = np.nan if now is None else float(now)
        unchanged[str(tenor)] = bool(np.isfinite(now) and math.isclose(now, then, abs_tol=1e-9))
    tenor = inp.dates.tenor
    if unchanged.get(tenor, False) if tenor is not None else any(unchanged.values()):
        out.append("forward points (computed from the default rates)")
    return out


def _delta_type(inp: FxInputs, dates: FxDates) -> str:
    return market_delta_type(inp.pair, dates) if inp.delta_conv == "MARKET" else inp.delta_conv


@st.cache_data(show_spinner=False, max_entries=1024)
def _unit_smile(
    rd: float, rf: float, tau_e: float, tau_d: float, quote: Tuple, conventions: Tuple[str, str, str], model: str
) -> Tuple[Optional[FxSmile], Optional[str]]:
    """Smile of one tenor at spot 1 (cached on the inputs; rescaled to the
    spot by the caller, so a spot change rebuilds nothing)."""

    try:
        smile = build_smile(
            FxMarket(1.0, rd, rf, tau_e, tau_d), TenorQuote(*quote), SmileConventions(*conventions), model
        )
    except ValueError as exc:
        return None, str(exc)
    return smile, None


@st.cache_data(show_spinner=False, max_entries=1024)
def _unit_surface_row(
    rd: float, rf: float, tau_e: float, tau_d: float, quote: Tuple, conventions: Tuple[str, str, str], model: str
) -> Tuple[Tuple[str, float, float], ...]:
    """(label, strike at spot 1, vol) per surface delta of one tenor."""

    smile, _ = _unit_smile(rd, rf, tau_e, tau_d, quote, conventions, model)
    if smile is None:
        return tuple((label, float("nan"), float("nan")) for label, _, _ in SURFACE_DELTAS)
    return tuple(smile_by_delta(smile))


def _quote_key(q: TenorQuote) -> Tuple:
    return (q.tenor, q.atm, q.rr25, q.bf25, q.rr10, q.bf10)


@dataclass(frozen=True)
class Selected:
    """The selected expiry: its market, quote (as entered or interpolated),
    smile (at the real spot) and how the quote was obtained. ``market`` is
    None when the expiry has no forward (forward points missing or
    unusable): nothing can be priced then, and ``error`` says why."""

    market: Optional[FxMarket]
    quote: Optional[TenorQuote]
    smile: Optional[FxSmile]
    delta_type: str
    note: str
    error: Optional[str]
    carry_note: str = ""  # how the forward of an expiry between tenors came from the forward points


def points_between_tenors(inp: FxInputs, rows: List[TenorRow]) -> Tuple[Optional[float], str]:
    """Forward points (pips) of an expiry date that is not a tenor of the
    quotes table, with CARRY FROM FORWARD POINTS, and how they were obtained:
    the foreign rates implied by the tenors' points interpolated linearly in
    the rate time, held flat before the first and after the last tenor with
    points. ``(None, '')`` when no tenor has usable points; ``(0, '')`` when
    delivery is on the spot date (an expiry today: the forward is spot)."""

    d, base = inp.dates, inp.pair.base
    if d.tau_d <= 0:  # delivery on the spot date: the forward is spot whatever the points
        return 0.0, ""
    implied = [(r.dates.tau_d, _foreign_rate(inp, r.dates.tau_d, r.points)[0], r.tenor) for r in rows]
    implied = sorted((t, v, name) for t, v, name in implied if np.isfinite(v))
    if not implied:
        return None, ""
    t, v, names = zip(*implied)
    rf = float(np.interp(d.tau_d, t, v))
    points = forward_points(inp.S, inp.S * math.exp((inp.rd - rf) * d.tau_d), inp.pair.pip)
    if d.tau_d < t[0] or d.tau_d > t[-1]:
        edge, side = (names[0], "before the first") if d.tau_d < t[0] else (names[-1], "after the last")
        note = f"forward from the {edge} implied {base} rate held flat ({side} tenor with forward points)"
    else:
        j = min(int(np.searchsorted(t, d.tau_d)), len(t) - 1)
        lo, hi = names[max(j - 1, 0)], names[j]
        note = (
            f"forward from the {hi} forward points"
            if math.isclose(d.tau_d, t[j], abs_tol=1e-12)
            else f"forward from the implied {base} rates interpolated between {lo} and {hi} (linear in rate time)"
        )
    return points, note


def _selected(inp: FxInputs, rows: List[TenorRow]) -> Selected:
    d = inp.dates
    row = next((r for r in rows if r.tenor == d.tenor), None)
    carry_note = ""
    if row is not None:
        points = row.points
    elif inp.carry == "FORWARD POINTS":  # a date between tenors
        points, carry_note = points_between_tenors(inp, rows)
    else:
        points = None
    sel = _selected_market(inp, rows, row, points)
    return replace(sel, carry_note=carry_note) if carry_note and sel.market is not None else sel


def _selected_market(inp: FxInputs, rows: List[TenorRow], row: Optional[TenorRow], points: Optional[float]) -> Selected:
    d = inp.dates
    dtype = _delta_type(inp, d)
    valid = [r for r in rows if r.quote is not None]
    rf, rf_error = _foreign_rate(inp, d.tau_d, points)
    if rf_error:  # no forward: never price on a made-up foreign rate
        return Selected(None, None, None, dtype, "", f"{d.label}: {rf_error}")
    market = FxMarket(inp.S, inp.rd, rf, d.tau_e, d.tau_d)
    if row is not None and row.quote is not None:
        quote, note = row.quote, f"the {row.tenor} quotes"
    elif row is not None:
        return Selected(market, None, None, dtype, "", row.error)
    elif valid:
        quote, note = interpolate_quote([(r.dates.tau_e, r.quote) for r in valid], d.tau_e, d.expiry.isoformat())
    else:
        return Selected(market, None, None, dtype, "", "no valid vol quotes in the table")
    if d.tau_e <= 0:
        return Selected(market, quote, None, dtype, note, None)
    unit, error = _unit_smile(
        inp.rd, rf, d.tau_e, d.tau_d, _quote_key(quote), (dtype, inp.atm_type, inp.strangle), inp.model
    )
    if unit is None:
        return Selected(market, quote, None, dtype, note, f"{quote.tenor}: {error}")
    return Selected(market, quote, unit.rescaled(inp.S), dtype, note, None)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Priced:
    K: float
    vol: float
    vol_note: str
    greeks: FxGreeks
    n_base: float  # signed base-currency notional
    hedge_type: str


def _strike_and_vol(inp: FxInputs, sel: Selected) -> Tuple[float, float, str]:
    m, smile = sel.market, sel.smile
    if m is None:
        raise ValueError(
            f"no forward for this expiry ({sel.error}): enter its forward points (vol quotes table) or take the "
            "carry from the rates"
        )
    expired = m.tau_e <= 0
    use_smile = inp.vol_src == "SMILE" and not expired
    if use_smile and smile is None:
        raise ValueError(f"no smile for this expiry ({sel.error}); fix the quotes or price at a flat vol")
    flat = inp.flat_vol if inp.vol_src == "FLAT" else (sel.quote.atm if sel.quote is not None else float("nan"))
    if inp.strike_mode == "STRIKE":
        K = float(inp.strike)
    elif inp.strike_mode == "ATM":
        if use_smile:
            K = smile.atm_strike
        else:
            K = atm_strike(m, flat if np.isfinite(flat) else 0.0, inp.atm_type, sel.delta_type.endswith("_pa"))
    else:
        if expired:
            raise ValueError("at expiry an option has no delta to solve a strike from: enter a strike")
        target = (1.0 if inp.call else -1.0) * inp.delta_pct / 100.0
        if use_smile:
            K = smile.strike_at_delta(target, inp.call)
        else:
            K = strike_from_delta(m, target, flat, inp.call, sel.delta_type)
    if expired:
        return K, 0.0, "expired: the vol no longer matters"
    if use_smile:
        vol = smile.vol(K)
        if not (np.isfinite(vol) and vol > 0):
            raise ValueError(
                f"the {MODEL_LABELS[smile.model]} smile has no vol at K = {inp.pair.fmt(K)} (far wing); price at a "
                "flat vol or use SABR"
            )
        return K, float(vol), f"smile vol at K ({sel.note})"
    return K, flat, "flat vol (override)"


def _price(inp: FxInputs, sel: Selected) -> Priced:
    K, vol, note = _strike_and_vol(inp, sel)
    g = fx_greeks(sel.market, K, vol, inp.call)
    n_base = inp.sign * base_notional(inp.notional, inp.notional_in_base, K)
    premium_ccy = inp.pair.base if inp.premium_in_base else inp.pair.quote
    hedge = market_delta_type(inp.pair, inp.dates, premium_ccy=premium_ccy)
    return Priced(K, vol, note, g, n_base, hedge)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _money(x: float, ccy: str) -> str:
    if not np.isfinite(x):
        return "n/a"
    if x == 0:
        return f"{ccy} 0"
    if abs(x) < 1:
        return f"{ccy} {calc.fmt_sig(x)}"
    return f"{ccy} {x:,.0f}" if abs(x) >= 100 else f"{ccy} {x:,.2f}"


def _card_money(x: float, ccy: str) -> Tuple[str, Optional[str]]:
    """(headline, exact) for a Greek card: the exact amount when it fits the
    card (:data:`CARD_VALUE_CHARS`), else 4 significant figures with a k / M /
    B / T suffix and the exact amount for the line below (never cut off)."""

    exact = _money(x, ccy)
    if len(exact) <= CARD_VALUE_CHARS:
        return exact, None
    ax = abs(x)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if ax >= scale:
            short = f"{x / scale:.4g}{suffix}" if ax < 1e15 else f"{x:.2e}"
            return f"{ccy} {short}", exact
    return f"{ccy} {x:.2e}", exact


def _cell(label: str, value: str) -> str:
    return f"<div class='dx-cell'><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>"


def _hero(inp: FxInputs, sel: Selected, p: Priced) -> None:
    pair, m, g = inp.pair, sel.market, p.greeks
    premium_ccy = pair.base if inp.premium_in_base else pair.quote
    amount = premium_amount(g.price, m.S, abs(p.n_base), inp.premium_in_base)
    d = inp.dates
    pts = forward_points(m.S, m.forward, pair.pip)
    cells = [
        ("SPOT", calc.fmt_exact(m.S, pair.decimals)),
        (f"FORWARD ({d.delivery.isoformat()})", pair.fmt(m.forward)),
        ("FWD POINTS", f"{pts:+,.2f} pips"),
        ("STRIKE", calc.fmt_exact(p.K, pair.decimals) if inp.strike_mode == "STRIKE" else pair.fmt(p.K)),
        ("VOL", f"{100 * p.vol:.3f}%"),
        ("EXPIRY", f"{d.expiry.isoformat()} · {d.days_to_expiry}D"),
        (DELTA_TYPE_LABELS[p.hedge_type], f"{100 * g.delta(p.hedge_type):.2f}%"),
    ]
    grid = "".join(_cell(a, b) for a, b in cells)
    kind = f"{pair.base} {'CALL' if inp.call else 'PUT'} / {pair.quote} {'PUT' if inp.call else 'CALL'}"
    side = "PAY" if inp.sign > 0 else "RECEIVE"
    st.markdown(
        f"""<div class='dx-hero'><div class='dx-hero-tag'>{html.escape(kind)} · PREMIUM</div>
        <div class='dx-hero-value'>{amount:,.2f}<span class='dx-hero-ccy'> {premium_ccy}</span></div>
        <div class='dx-hero-unit'>{side} on the spot date {d.spot.isoformat()} for {pair.base}
        {abs(p.n_base):,.0f} notional ({inp.dates.label}, {html.escape(p.vol_note)}) · Garman-Kohlhagen</div>
        <div class='dx-hero-grid'>{grid}</div></div>""",
        unsafe_allow_html=True,
    )


def _premium_panel(inp: FxInputs, sel: Selected, p: Priced) -> None:
    pair, m, g = inp.pair, sel.market, p.greeks
    styles = premium_styles(g.price, m.S, p.K, pair.pip)
    n = abs(p.n_base)
    market_style = "pct_for" if inp.premium_in_base else "dom_pips"
    rows = [
        ("dom_pips", f"{pair.quote} PIPS", f"{styles['dom_pips']:,.2f}", f"{pair.quote} pips per 1 {pair.base}"),
        ("pct_for", f"% {pair.base}", f"{styles['pct_for']:.4f}%", f"of the {pair.base} notional"),
        ("for_pips", f"{pair.base} PIPS", f"{styles['for_pips']:,.2f}", f"{pair.base} 0.0001s per 1 {pair.quote}"),
        ("pct_dom", f"% {pair.quote}", f"{styles['pct_dom']:.4f}%", f"of the {pair.quote} notional (at K)"),
    ]
    table = pd.DataFrame(
        [
            {"QUOTE STYLE": label + ("  ← premium ccy" if key == market_style else ""), "VALUE": value, "UNIT": unit}
            for key, label, value, unit in rows
        ]
        + [
            {"QUOTE STYLE": f"AMOUNT IN {pair.quote}", "VALUE": f"{n * g.price:,.2f}", "UNIT": f"{pair.quote}"},
            {
                "QUOTE STYLE": f"AMOUNT IN {pair.base}",
                "VALUE": f"{n * g.price / m.S:,.2f}",
                "UNIT": f"{pair.base} at spot",
            },
        ]
    )
    st.markdown("### / PREMIUM")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        f"Value on the spot date of one option per 1 {pair.base}: {calc.fmt_sig(g.price, 6)} {pair.quote} "
        f"(pip {pair.pip:g}). Notional {pair.base} {n:,.0f}"
        + ("" if inp.notional_in_base else f" = {pair.quote} {inp.notional:,.0f} / K")
        + "."
    )


def _greeks_panel(inp: FxInputs, p: Priced) -> None:
    pair, g, n = inp.pair, p.greeks, p.n_base
    f = calc.fmt_sig
    hedge = p.hedge_type
    delta_amount = n * g.delta(hedge)
    amounts = [  # (name, amount, its currency, unit line, per-unit line)
        (
            "GAMMA Γ",
            n * g.gamma_1pct,
            pair.base,
            "spot-delta change per +1% spot",
            f"{f(g.gamma_1pct)} per 1 {pair.base}",
        ),
        ("VEGA ν", n * g.vega_1vol, pair.quote, "P&L per +1 vol pt", f"{f(g.vega_1vol)} per 1 {pair.base}"),
        ("THETA Θ", n * g.theta_1d, pair.quote, "P&L over 1 calendar day", "repriced, both times −1 day"),
        (f"RHO {pair.quote}", n * g.rho_d_1bp, pair.quote, f"P&L per +1bp of the {pair.quote} rate", ""),
        (f"RHO {pair.base}", n * g.rho_f_1bp, pair.quote, f"P&L per +1bp of the {pair.base} rate", ""),
        ("VANNA", n * g.vanna_1vol, pair.base, "spot-delta change per +1 vol pt", ""),
        ("VOLGA", n * g.volga_1vol, pair.quote, "vega change per +1 vol pt", ""),
    ]
    cards = [
        (
            "DELTA Δ",
            f"{100 * g.delta(hedge):.3f}%",
            DELTA_TYPE_LABELS[hedge].lower(),
            [f"{_money(delta_amount, pair.base)} {'long' if delta_amount >= 0 else 'short'}"],
        )
    ]
    for name, amount, ccy, unit, per_unit in amounts:
        headline, exact = _card_money(amount, ccy)
        cards.append((name, headline, unit, [s for s in (f"= {exact}" if exact else "", per_unit) if s]))
    html_cards = "".join(
        f"<div class='dx-greek'><div class='dx-greek-name'>{html.escape(a)}</div>"
        f"<div class='dx-greek-value'>{html.escape(b)}</div><div class='dx-greek-unit'>{html.escape(c)}</div>"
        + "".join(f"<div class='dx-greek-sub'>{html.escape(line)}</div>" for line in subs)
        + "</div>"
        for a, b, c, subs in cards
    )
    st.markdown("### / GREEKS")
    st.caption(
        f"For the {'long' if n >= 0 else 'short'} position on {pair.base} {abs(n):,.0f}; each value is the P&L (or "
        f"change in delta) for the stated move. The delta shown is the hedge convention for a premium in "
        f"{pair.base if inp.premium_in_base else pair.quote}; all four are below. Each rho bumps one deposit rate "
        "with spot and the other rate fixed, so the forward moves"
        + (" (the foreign rate is the one implied by the forward points)" if inp.carry != "RATES" else "")
        + "; theta keeps spot, rates and vol fixed."
    )
    st.markdown(f"<div class='dx-greeks'>{html_cards}</div>", unsafe_allow_html=True)
    table = pd.DataFrame(
        [
            {
                "DELTA TYPE": DELTA_TYPE_LABELS[t] + ("  ← hedge" if t == hedge else ""),
                "DELTA (%)": f"{100 * g.delta(t):.4f}",
                f"AMOUNT ({pair.base})": f"{n * g.delta(t):,.0f}",
            }
            for t in DELTA_TYPES
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")


def _dates_panel(inp: FxInputs, sel: Selected) -> None:
    d, pair = inp.dates, inp.pair
    defaults = illustrative_carry(inp)
    with st.expander("DATES AND CONVENTIONS"):
        rows = [
            ("Trade date", d.trade.isoformat()),
            ("Spot date (premium)", f"{d.spot.isoformat()} (T+{pair.spot_lag})"),
            ("Expiry", f"{d.expiry.isoformat()} ({d.days_to_expiry} days, vol time {d.tau_e:.6f}y ACT/365F)"),
            ("Delivery", f"{d.delivery.isoformat()} (rate time {d.tau_d:.6f}y from spot, ACT/365F)"),
            ("Calendars", f"expiry {pair.calendar.name}; settlement {pair.settlement_calendar.name}"),
            ("Quote delta convention", DELTA_TYPE_LABELS[sel.delta_type]),
            (
                "Rates",
                f"{pair.quote} {100 * sel.market.rd:.4f}%, {pair.base} {100 * sel.market.rf:.4f}% "
                f"(continuous ACT/365F{'; implied from the forward points' if inp.carry != 'RATES' else ''}"
                + (f"; {sel.carry_note}" if sel.carry_note else "")
                + (f"; ILLUSTRATIVE defaults: {', '.join(defaults)}" if defaults else "")
                + ")",
            ),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["DATE / CONVENTION", "VALUE"]), hide_index=True, width="stretch")


def date_warnings(pair: CurrencyPair, dates: FxDates) -> List[str]:
    """Warnings for a trade or expiry date that is not a business day of
    both currencies (FX options trade and expire on such days; the dates
    and the price are computed anyway, from the date as entered)."""

    cal, out = pair.calendar, []
    if not cal.is_business_day(dates.trade):
        out.append(
            f"TRADE DATE {dates.trade.isoformat()} is not a business day for {pair.label} ({cal.name}): spot and "
            f"the tenor dates are computed from it anyway (next business day {cal.adjust(dates.trade, 'F')})."
        )
    if dates.tenor is None and dates.expiry > dates.trade and not cal.is_business_day(dates.expiry):
        out.append(
            f"EXPIRY {dates.expiry.isoformat()} is not a business day for {pair.label} ({cal.name}): an FX option "
            f"expires on a business day of both currencies (next one {cal.adjust(dates.expiry, 'F')}). Priced as "
            "entered."
        )
    return out


def _quotes_editor(inp: FxInputs, rows: List[TenorRow]) -> None:
    pair = inp.pair
    st.markdown("### / VOL QUOTES")
    base = st.session_state[_k("quotes", pair)]
    edited = edited_tenors(pair, base)
    if edited:
        badge, color = "EDITED", SERIES[2]
        text = (
            f"Your quotes in {', '.join(edited)}; the other rows are the ILLUSTRATIVE defaults (not market data)."
            if len(edited) < len(base)
            else "Every row is your input."
        )
    else:
        badge, color = "ILLUSTRATIVE", SERIES[3]
        text = (
            "Default quotes are round numbers that look like a G10 smile (generated from a SABR smile), not market "
            "data: FX vol quotes are not freely available. Edit the cells or paste your own (vol points: 7.5 = 7.5%)."
        )
    st.markdown(
        f"<span class='dx-badge' style='color:{color}'>{badge}</span>{html.escape(text)}", unsafe_allow_html=True
    )
    shown = base.copy()
    shown.insert(1, "EXPIRY", [fx_dates(pair, inp.trade, tenor=str(t)).expiry.isoformat() for t in base["TENOR"]])
    columns = ["TENOR", "EXPIRY", *QUOTE_COLUMNS] + ([POINTS_COLUMN] if inp.carry == "FORWARD POINTS" else [])
    st.data_editor(
        shown,
        key=_editor_key(pair),
        on_change=_merge_quote_edits,
        args=(pair,),
        hide_index=True,
        width="stretch",
        height=36 * (len(shown) + 1) + 3,
        num_rows="fixed",
        disabled=["TENOR", "EXPIRY"],
        column_order=columns,
        column_config={
            **{c: st.column_config.NumberColumn(c, format="%.2f", step=0.01) for c in QUOTE_COLUMNS},
            POINTS_COLUMN: st.column_config.NumberColumn(POINTS_COLUMN, format="%.2f", step=0.01),
        },
    )
    c1, c2 = st.columns(2)
    c1.button("RESET TO ILLUSTRATIVE QUOTES", key=_k("reset_quotes"), on_click=_reset_quotes, args=(pair,))
    if inp.carry == "FORWARD POINTS":
        c2.button("SET POINTS FROM THE RATES", key=_k("points_from_rates"), on_click=_points_from_rates, args=(pair,))
    st.caption(
        "ATM is the vol at the ATM strike of the chosen ATM convention; RR = σ(call) − σ(put) and BF at 25 and 10 "
        "delta, in the chosen delta convention; leave RR10 and BF10 empty for a 25-delta smile."
        + (" Forward points are (F − S) / pip to each delivery date." if inp.carry == "FORWARD POINTS" else "")
    )
    for r in rows:
        if r.error:
            st.error(f"Quote row {r.error}.")


def _smile_chart(inp: FxInputs, sel: Selected, p: Optional[Priced]) -> None:
    smile = sel.smile
    pair = inp.pair
    st.markdown(f"### / SMILE · {inp.dates.label}")
    if inp.dates.tau_e <= 0:
        st.info(
            "EXPIRED (expiry = trade date): no time value is left, so there is no smile; the premium is the "
            "discounted intrinsic value of the forward."
        )
        return
    if smile is None:
        st.warning(f"No smile for this expiry: {sel.error}.")
        return
    c1, c2 = st.columns(2)
    space = c1.radio("SPACE", SPACES, horizontal=True, key=_k("space"), persist_state=PERSIST)
    n_sd = c2.slider("WIDTH (SD)", 1.0, 4.0, step=0.5, key=_k("smile_sd"), persist_state=PERSIST)
    st.markdown(
        f"<span class='dx-badge' style='color:{SERIES[3]}'>MODEL</span>{html.escape(MODEL_LABELS[smile.model])} "
        f"from {html.escape(sel.note)} (illustrative unless you entered market quotes), "
        f"{html.escape(DELTA_TYPE_LABELS[sel.delta_type].lower())}, {html.escape(ATM_TYPE_LABELS[inp.atm_type])}, "
        f"{html.escape(STRANGLE_LABELS[inp.strangle].lower())}.",
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.update_layout(meta={"chart": "fx_smile"})
    off_chart = None  # the priced strike when it lies outside the plotted strikes
    if space == "STRIKE":
        K = smile_strikes(smile, n_sd=float(n_sd))
        vols = 100 * smile.vol(K)
        fig.add_trace(
            go.Scatter(
                x=K,
                y=vols,
                mode="lines",
                name=f"{smile.model} smile",
                line=dict(color=ACCENT, width=3),
                hovertemplate=f"K %{{x:.{pair.decimals}f}}<br>vol %{{y:.3f}}%<extra></extra>",
            )
        )
        pts = [pt for pt in smile.points if np.isfinite(pt.strike)]
        fig.add_trace(
            go.Scatter(
                x=[pt.strike for pt in pts],
                y=[100 * pt.vol for pt in pts],
                mode="markers+text",
                text=[pt.label for pt in pts],
                textposition="top center",
                name="DELTA PILLARS",
                marker=dict(size=9, color=FOREGROUND),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[sel.market.forward],
                y=[100 * smile.vol(sel.market.forward)],
                mode="markers",
                name="F (FORWARD)",
                marker=dict(size=11, color=SERIES[2], symbol="x"),
            )
        )
        xs, ys = list(K), list(vols)
        if p is not None and K[0] <= p.K <= K[-1]:
            fig.add_trace(
                go.Scatter(
                    x=[p.K],
                    y=[100 * p.vol],
                    mode="markers",
                    name="K (PRICED)",
                    marker=dict(size=11, color=SERIES[3], symbol="diamond"),
                )
            )
            ys.append(100 * p.vol)
        elif p is not None:
            off_chart = (
                f"K (PRICED) = {pair.fmt(p.K)} is outside the plotted strikes ({pair.fmt(K[0])} to "
                f"{pair.fmt(K[-1])}): widen WIDTH (SD) to see it."
            )
        x_title, x_lim = f"STRIKE ({pair.quote} per 1 {pair.base})", axis_limits(xs, floor=0.0)
    else:
        data = [row for row in smile_by_delta(smile, SMILE_DELTAS) if np.isfinite(row[2])]
        labels = [lab for lab, _, _ in data]
        vols = [100 * v for _, _, v in data]
        strikes = [k for _, k, _ in data]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(labels))),
                y=vols,
                mode="lines+markers",
                name=f"{smile.model} smile",
                line=dict(color=ACCENT, width=3),
                marker=dict(size=8, color=FOREGROUND),
                customdata=[[lab, k] for lab, k in zip(labels, strikes)],
                hovertemplate=f"%{{customdata[0]}}: K %{{customdata[1]:.{pair.decimals}f}}"
                "<br>vol %{y:.3f}%<extra></extra>",
            )
        )
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels)
        n = len(labels)
        xs, ys = [-0.5, n - 0.5], vols
        x_title, x_lim = "DELTA (OTM option, quote convention)", axis_limits(xs, pad=0.0, min_span_frac=2.0 / n)
    dark_layout(fig, height=420, xtitle=x_title, ytitle="IMPLIED VOL (%)")
    fig.update_layout(
        hovermode="closest",
        margin=dict(l=60, r=20, t=30, b=60),  # the legend sits below the x title (theme.LEGEND_BELOW)
    )
    bound_axes(fig, xaxis=x_lim, yaxis=axis_limits(ys, pad=0.15, floor=0.0))
    st.plotly_chart(fig, width="stretch", key=_k("smile_chart"), config=PLOTLY_2D_CONFIG)
    if off_chart:
        st.caption(off_chart)
    if GUARD_AVAILABLE:
        st.caption(ZOOM_HINT)
    ok, min_density, k_min = density_check(smile, smile_strikes(smile, n_sd=float(n_sd)))
    if not ok:
        why = (
            "Hagan's SABR expansion is asymptotic in ν²T"
            if smile.model == "SABR"
            else "vanna-volga does not guarantee a positive density in the wings"
        )
        st.warning(
            f"Butterfly arbitrage: the smile implies a negative density near K = {pair.fmt(k_min)} (call prices are "
            f"not convex there; density {min_density:.3g} per unit of strike). {why}: check the quotes or narrow the "
            "width."
        )
    for note in smile.notes:
        st.warning(note)
    _smile_tables(inp, sel)
    if p is not None and inp.vol_src == "SMILE":
        _smile_risk(inp, sel, p)


@st.cache_data(show_spinner=False, max_entries=256)
def _unit_smile_risk(
    rd: float,
    rf: float,
    tau_e: float,
    tau_d: float,
    quote: Tuple,
    conventions: Tuple[str, str, str],
    model: str,
    k_over_s: float,
    call: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Quote and SABR parameter vegas at spot 1 for strike ``k_over_s``
    (GK is homogeneous of degree one in spot and strike: the caller scales
    by the spot). Cached on the inputs."""

    m, q, conv = FxMarket(1.0, rd, rf, tau_e, tau_d), TenorQuote(*quote), SmileConventions(*conventions)
    base, error = _unit_smile(rd, rf, tau_e, tau_d, quote, conventions, model)  # cached: the page's own smile
    if base is None:
        return None, None, str(error)
    try:
        table = quote_vegas(m, q, conv, model, k_over_s, call, base=base)
        params = sabr_param_vegas(base, k_over_s, call) if model == "SABR" else None
    except ValueError as exc:
        return None, None, str(exc)
    return table, params, None


def _smile_risk(inp: FxInputs, sel: Selected, p: Priced) -> None:
    """Vega of the priced position to each market quote of its tenor (and
    to the SABR parameters): VM-18."""

    m, pair = sel.market, inp.pair
    S = m.S
    table, params, error = _unit_smile_risk(
        inp.rd,
        m.rf,
        m.tau_e,
        m.tau_d,
        _quote_key(sel.quote),
        (sel.delta_type, inp.atm_type, inp.strangle),
        inp.model,
        p.K / S,
        inp.call,
    )
    st.markdown("#### / SMILE RISK · VEGA BY QUOTE")
    if table is None:
        st.warning(f"Smile risk is not available for this option: {error}.")
        return
    n = p.n_base
    rows = [
        {
            "QUOTE": r.quote,
            "QUOTED (%)": f"{100 * r.quoted:+.4f}" if r.quote != "ATM" else f"{100 * r.quoted:.4f}",
            f"PER 1 {pair.base} ({pair.quote})": f"{S * r.vega:+.6f}",
            f"POSITION ({pair.quote})": _money(n * S * r.vega, pair.quote),
            "FITTED": "yes" if r.used else "no (checked only)",
        }
        for r in table.itertuples()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"P&L per +1 vol point on one quote of the {sel.quote.tenor} tenor, the smile rebuilt from the bumped quotes "
        f"({MODEL_LABELS[inp.model]}, same conventions) and the option repriced at the same strike "
        f"{pair.fmt(p.K)}. ATM is the vega to a parallel move of the quoted smile (RR and BF are spreads to ATM); "
        "RR is the skew (vanna) exposure, BF the wing (volga) exposure. Central differences of ±5 vol bp."
    )
    if params is not None:
        prm = [
            {
                "SABR PARAMETER": r.parameter,
                "VALUE": f"{r.value:+.6f}",
                f"POSITION PER +0.01 ({pair.quote})": _money(n * S * r._4, pair.quote),
            }
            for r in params.itertuples()
        ]
        st.dataframe(pd.DataFrame(prm), hide_index=True, width="stretch")
        st.caption("SABR (β = 1) parameter vegas: P&L of the position per +0.01 of α, ρ or ν, the others fixed.")


def _smile_tables(inp: FxInputs, sel: Selected) -> None:
    smile, pair = sel.smile, inp.pair
    rows = []
    for pt in smile.points:
        rows.append(
            {
                "PILLAR": pt.label,
                "DELTA": "ATM" if pt.call is None else f"{100 * pt.delta:+.0f}%",
                "STRIKE": pair.fmt(pt.strike) if np.isfinite(pt.strike) else "n/a",
                "VOL (%)": f"{100 * pt.vol:.4f}" if np.isfinite(pt.vol) else "n/a",
            }
        )
    st.markdown("#### / STRIKES BY DELTA")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("The smile's own strikes: the strike K whose delta at the smile vol σ(K) is the pillar delta.")
    q = smile.quote
    inputs = {"ATM": q.atm, **{f"RR{lv}": q.rr(lv) for lv in q.levels}, **{f"BF{lv}": q.bf(lv) for lv in q.levels}}
    errors = [
        {
            "QUOTE": k,
            "INPUT (%)": f"{100 * inputs[k]:.4f}",
            "ERROR (VOL BP)": f"{1e4 * v:+.4f}" if np.isfinite(v) else "n/a",
            "FITTED": "yes" if k in smile.used_quotes else "no (checked)",
        }
        for k, v in smile.errors.items()
    ]
    with st.expander("SMILE CALIBRATION: parameters, quote errors, strangles", expanded=False):
        if smile.model == "SABR":
            prm = [
                ("α (alpha)", f"{smile.params['alpha']:.6f}", "solved: reproduces the ATM vol at the ATM strike"),
                ("β (beta)", f"{smile.params['beta']:.1f}", "FIXED (lognormal backbone, the FX convention)"),
                ("ρ (rho)", f"{smile.params['rho']:+.6f}", "fitted"),
                ("ν (vol of vol)", f"{smile.params['nu']:.6f}", "fitted"),
            ]
        else:
            prm = [
                (k, pair.fmt(v) if k.startswith("K_") else f"{100 * v:.4f}%", "pillar (exact)")
                for k, v in smile.params.items()
            ]
        st.dataframe(pd.DataFrame(prm, columns=["PARAMETER", "VALUE", "STATUS"]), hide_index=True, width="stretch")
        st.caption(
            f"Quote errors on the final smile (RMS of the fitted ones {1e4 * smile.fit_rmse:.4f} vol bp; a note "
            f"appears above {1e4 * NOTE_TOL:g} vol bp)."
        )
        st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")
        ms_rows = []
        for lv, ms in smile.market_strangles.items():
            on_smile = ms.smile_value(sel.market, smile.vol)
            ms_rows.append(
                {
                    "DELTA": f"{lv}",
                    "MARKET STRANGLE VOL (%)": f"{100 * ms.vol:.4f}",
                    "CALL / PUT STRIKES": f"{pair.fmt(ms.call_strike)} / {pair.fmt(ms.put_strike)}",
                    f"VALUE AT FLAT VOL ({pair.quote} PIPS)": f"{ms.value / pair.pip:,.4f}",
                    f"VALUE ON THE SMILE ({pair.quote} PIPS)": f"{on_smile / pair.pip:,.4f}",
                    "SMILE-STRANGLE BF (%)": f"{100 * smile.smile_strangles.get(lv, float('nan')):.4f}",
                    "QUOTED BF (%)": f"{100 * smile.quote.bf(lv):.4f}",
                }
            )
        st.dataframe(pd.DataFrame(ms_rows), hide_index=True, width="stretch")
        st.caption(
            "Market strangle: strikes at ±delta for the single vol ATM + BF, priced at that vol; the smile reprices "
            "it when the strangle convention is MARKET. The smile-strangle BF is (σ(call) + σ(put)) / 2 − ATM at the "
            "smile's own strikes."
        )


def _surface(inp: FxInputs, rows: List[TenorRow]) -> None:
    st.markdown("### / VOL SURFACE · TENOR × DELTA")
    valid = [r for r in rows if r.quote is not None]
    z, text, hover_k, tenors, no_forward = [], [], [], [], []
    labels = [lab for lab, _, _ in SURFACE_DELTAS]
    with st.spinner("Building the smiles"):
        for r in valid:
            rf, err = _foreign_rate(inp, r.dates.tau_d, r.points)
            if err:
                no_forward.append(r.tenor)
                continue
            row = _unit_surface_row(
                inp.rd,
                rf,
                r.dates.tau_e,
                r.dates.tau_d,
                _quote_key(r.quote),
                (_delta_type(inp, r.dates), inp.atm_type, inp.strangle),
                inp.model,
            )
            tenors.append(r.tenor)
            z.append([100 * v for _, _, v in row])
            text.append([f"{100 * v:.2f}" if np.isfinite(v) else "" for _, _, v in row])
            hover_k.append([inp.pair.fmt(k * inp.S) if np.isfinite(k) else "n/a" for _, k, _ in row])
    if not tenors:
        st.info(
            "No valid tenor quotes: fix the table above."
            + (f" Tenors without a forward: {', '.join(no_forward)}." if no_forward else "")
        )
        return
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=tenors,
            text=text,
            texttemplate="%{text}",
            customdata=hover_k,
            textfont=dict(family="IBM Plex Mono", size=8, color=FOREGROUND),
            colorscale=SEQUENTIAL_COLORS,
            colorbar=dict(
                title=dict(text="VOL %", side="right"), orientation="h", thickness=10, y=-0.25, yanchor="top"
            ),
            hovertemplate="%{y} %{x}<br>K %{customdata}<br>vol %{z:.3f}%<extra></extra>",
        )
    )
    dark_layout(fig, height=440, xtitle="DELTA (OTM option, quote convention)", ytitle="TENOR", legend=False)
    fig.update_layout(hovermode="closest", meta={"chart": "fx_surface"}, margin=dict(l=60, r=10, t=20, b=110))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", autorange=False)
    nx, ny = len(labels), len(tenors)
    bound_axes(
        fig,
        xaxis=axis_limits([-0.5, nx - 0.5], pad=0.0, min_span_frac=1.0 / nx),
        yaxis=axis_limits([-0.5, ny - 0.5], pad=0.0, min_span_frac=1.0 / ny),
    )
    st.plotly_chart(fig, width="stretch", key=_k("surface_chart"), config=PLOTLY_2D_CONFIG)
    st.caption(
        f"Smile vols (%) at the smile's own delta strikes, per tenor, {MODEL_LABELS[inp.model]}; hover for the strike. "
        "Blank cells: no strike has that delta (a premium-adjusted call delta above its maximum) or the model has no "
        "vol there. Built from the quotes table (illustrative unless edited)."
        + (
            f" Left out, no forward (forward points missing or unusable): {', '.join(no_forward)}."
            if no_forward
            else ""
        )
    )


def _scenario(inp: FxInputs, sel: Selected, p: Priced) -> None:
    pair = inp.pair
    st.markdown("### / SCENARIO P&L")
    c1, c2, c3 = st.columns(3)
    max_spot = _num("MAX SPOT SHOCK (%)", _k("scn_spot"), container=c1, min_value=0.1, max_value=50.0, step=0.5)
    max_vol = _num("MAX VOL SHOCK (VOL PTS)", _k("scn_vol"), container=c2, min_value=0.1, max_value=50.0, step=0.5)
    days = _num("HORIZON (DAYS)", _k("scn_days"), container=c3, min_value=0.0, max_value=3650.0, step=1.0)
    spot = float(max_spot) / 100 * np.asarray(SCENARIO_SPOT_STEPS)
    vols = float(max_vol) / 100 * np.asarray(SCENARIO_VOL_STEPS)
    grid = scenario_pnl(sel.market, p.K, p.vol, inp.call, p.n_base, spot, vols, float(days))
    x = [f"{t}%" for t in calc.shock_labels(100 * grid.spot_shocks)]
    y = calc.shock_labels(100 * grid.vol_shocks)
    fig = go.Figure(
        go.Heatmap(
            z=grid.pnl,
            x=x,
            y=y,
            text=[[calc.fmt_compact(v) for v in row] for row in grid.pnl],
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=10, color=FOREGROUND),
            colorscale=PNL_COLORS,
            zmid=0.0,
            colorbar=dict(
                title=dict(text=f"P&L {pair.quote}", side="right"),
                orientation="h",
                thickness=10,
                y=-0.32,
                yanchor="top",
            ),
            hovertemplate=f"spot %{{x}}<br>vol %{{y}} pts<br>P&L %{{z:,.2f}} {pair.quote}<extra></extra>",
        )
    )
    dark_layout(fig, height=400, xtitle="SPOT SHOCK (relative)", ytitle="VOL SHOCK (VOL PTS)", legend=False)
    fig.update_layout(hovermode="closest", meta={"chart": "fx_scenario"}, margin=dict(l=60, r=10, t=20, b=110))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    nx, ny = len(x), len(y)
    bound_axes(
        fig,
        xaxis=axis_limits([-0.5, nx - 0.5], pad=0.0, min_span_frac=1.0 / nx),
        yaxis=axis_limits([-0.5, ny - 0.5], pad=0.0, min_span_frac=1.0 / ny),
    )
    st.plotly_chart(fig, width="stretch", key=_k("scenario_chart"), config=PLOTLY_2D_CONFIG)
    st.caption(
        f"Full revaluation P&L in {pair.quote} of the {'long' if p.n_base >= 0 else 'short'} {pair.base} "
        f"{abs(p.n_base):,.0f} position, at a flat vol shifted from {100 * p.vol:.3f}% (the smile is not re-marked), "
        f"{grid.horizon_days:g} days later (both times shorter, rates unchanged)."
        + (f" Vol shocks below −{100 * p.vol:.2f} pts are omitted (negative vol)." if grid.dropped_vol_shocks else "")
        + (
            " The horizon reaches the expiry: every cell is the intrinsic value of the forward, so the vol shocks do "
            "not matter."
            if grid.horizon_days >= inp.dates.days_to_expiry
            else ""
        )
    )
    table = pd.DataFrame(grid.pnl, index=[f"{v} vol pts" for v in y], columns=x)
    with st.expander("SCENARIO GRID AS A TABLE"):
        st.dataframe(
            table, width="stretch", column_config={c: st.column_config.NumberColumn(format="%,.2f") for c in x}
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render() -> None:
    """Draw the FX OPTIONS page."""

    _init_state()
    st.markdown(PAGE_CSS + FX_CSS, unsafe_allow_html=True)
    st.markdown("## / FX OPTIONS")
    st.caption(
        "Garman-Kohlhagen with FX conventions: vol time trade → expiry, rate time spot → delivery (ACT/365F), "
        "premium paid on the spot date; smiles from ATM / risk-reversal / butterfly quotes."
    )
    inp, errors = _sidebar()
    if inp is None:
        for msg in errors:
            st.error(msg)
        st.info("Fix the inputs in the sidebar to price the option.")
        return
    for msg in date_warnings(inp.pair, inp.dates):
        st.warning(msg)
    rows = parse_quote_rows(inp.pair, inp.trade, st.session_state[_k("quotes", inp.pair)])
    sel = _selected(inp, rows)
    if inp.dates.tau_e <= 0:
        st.warning(
            "EXPIRES TODAY (expiry = trade date): the premium is the discounted intrinsic value of the forward to the "
            "delivery date, the spot delta is the payoff's step and gamma, vega, vanna and volga are 0."
        )
    priced: Optional[Priced] = None
    try:
        priced = _price(inp, sel)
    except ValueError as exc:
        st.error(f"Cannot price this option: {exc}.")
    if priced is not None:
        _hero(inp, sel, priced)
        if sel.carry_note:
            st.caption(f"FORWARD: {sel.carry_note}.")
        defaults = illustrative_carry(inp)
        if defaults:
            st.markdown(
                f"<span class='dx-badge' style='color:{SERIES[3]}'>ILLUSTRATIVE CARRY</span>"
                f"{html.escape(', '.join(defaults))}: defaults, not market data (the default rates are flat "
                "round numbers, continuously compounded ACT/365F); the forward, its points and the premium depend on "
                "them. Enter your deposit rates in the sidebar, or forward points in the vol quotes table (CARRY "
                "FROM: FORWARD POINTS).",
                unsafe_allow_html=True,
            )
        _premium_panel(inp, sel, priced)
        _greeks_panel(inp, priced)
        _dates_panel(inp, sel)
    _quotes_editor(inp, rows)
    _smile_chart(inp, sel, priced)
    _surface(inp, rows)
    if priced is not None:
        _scenario(inp, sel, priced)
