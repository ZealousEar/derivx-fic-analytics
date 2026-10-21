"""Reporting currency of the Rates page: the sidebar REPORTING CURRENCY, the
FX spots that convert into it, and the labels and captions that state them.

Every amount on the BOOK, SCENARIOS and RESEARCH tabs is in a position's own
currency: US Treasuries in USD, everything else in the swap curve's currency.
Totals are per currency; whenever the positions hold a currency other than
the reporting one (USD by default), the whole book is also shown converted
into it (``ALL IN USD``) at FX spots from the ECB euro reference rates of the
valuation date (the latest day on or before it in the ``FX_ECB`` history of
the sidebar DATA SOURCE's store: the bundled daily history since 2015, plus
the latest days in LIVE mode) or at rates the user types, and every
converted figure says which rates and date it used. Historical replays and
the value at risk move the spots by the ECB rates of their own dates (each
can hold them); other scenarios hold them unless the SCENARIOS tab shocks
them (EURUSD, GBPUSD).
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Sequence, Tuple

import streamlit as st

from derivx.fx.market_data import FxDataError, bundled_ecb_rates
from derivx.rates.curves import Market
from derivx.risk.reporting import (
    CURRENCIES,
    DEFAULT_REPORTING,
    SHOCK_PAIRS,
    FxSpots,
    Reporting,
    ecb_spots,
    entered_spots,
    ordered,
    position_currencies,
    store_spots,
    symbol,
)
from derivx.risk.sensitivities import Position
from derivx.ui import rates_data

REPORT_KEY = "rates_report_ccy"
SOURCE_KEY = "rates_fx_source"
PAIR_KEYS = {"EURUSD": "rates_fx_eurusd", "GBPUSD": "rates_fx_gbpusd"}
SOURCES = ("ECB REFERENCE RATES", "ENTER RATES")
PAIR_HELP = {"EURUSD": "US dollars per 1 euro", "GBPUSD": "US dollars per 1 pound"}


@st.cache_resource(show_spinner=False)
def _ecb():
    """The bundled ECB snapshot (read once per server); None if unreadable."""

    try:
        return bundled_ecb_rates()
    except FxDataError:
        return None


def _ecb_spots(asof: dt.date) -> Optional[FxSpots]:
    """The ECB reference rates of ``asof`` from the DATA SOURCE's history
    store (latest day on or before it), else from the bundled history."""

    try:
        return store_spots(rates_data.active_store(), asof)
    except ValueError:
        pass
    rates = _ecb()
    if rates is None:
        return None
    try:
        return ecb_spots(rates, asof)
    except ValueError:
        return None


def _entered(asof: dt.date, ecb: Optional[FxSpots]) -> Optional[FxSpots]:
    """The rates typed in the sidebar (the ECB rates until edited)."""

    quotes = {}
    for pair, key in PAIR_KEYS.items():
        value = st.session_state.get(key)
        if value is None and ecb is not None:
            value = ecb.quote(pair)
        if value is None:
            return None
        quotes[pair] = float(value)
    try:
        return entered_spots(quotes, asof, "FX rates entered by you")
    except ValueError:
        return None


def reporting(market: Market) -> Optional[Reporting]:
    """The reporting currency and FX spots the sidebar chose (from session
    state: tabs drawn before the sidebar section, and pages without it, get
    the same answer; default USD at the ECB rates). None when no FX spots
    are available (the ECB snapshot unreadable and no rates entered)."""

    currency = st.session_state.get(REPORT_KEY, st.session_state.get(f"_keep_{REPORT_KEY}", DEFAULT_REPORTING))
    currency = currency if currency in CURRENCIES else DEFAULT_REPORTING
    ecb = _ecb_spots(market.asof)
    source = st.session_state.get(SOURCE_KEY, st.session_state.get(f"_keep_{SOURCE_KEY}", SOURCES[0]))
    spots = _entered(market.asof, ecb) if source == SOURCES[1] else ecb
    return Reporting(currency, spots) if spots is not None else None


def _fx_caption(spots: FxSpots) -> str:
    quotes = ", ".join(f"{p} {spots.quote(p):.5f}" for p in ("EURUSD", "GBPUSD", "EURGBP"))
    return f"{quotes}: {spots.dated}."


def sidebar(market: Market) -> Optional[Reporting]:
    """Draw the REPORTING CURRENCY section; the reporting it sets."""

    st.sidebar.markdown("### / REPORTING CURRENCY")
    rates_data.sticky(REPORT_KEY, CURRENCIES)
    st.sidebar.selectbox(
        "REPORT BOOK TOTALS IN", CURRENCIES, key=REPORT_KEY,
        help="Totals stay per currency (US Treasuries are in USD, everything else in the swap curve's currency); a "
        "book holding another currency is also shown converted into this one (ALL IN ...).",
    )  # fmt: skip
    rates_data.remember(REPORT_KEY)
    rates_data.sticky(SOURCE_KEY, SOURCES)
    source = st.sidebar.radio("FX SPOTS", SOURCES, key=SOURCE_KEY, horizontal=True)
    rates_data.remember(SOURCE_KEY)
    ecb = _ecb_spots(market.asof)
    if source == SOURCES[1]:
        cols = st.sidebar.columns(2)
        for col, pair in zip(cols, SHOCK_PAIRS):
            key = PAIR_KEYS[pair]
            if key not in st.session_state and ecb is not None:
                st.session_state[key] = round(ecb.quote(pair), 5)
            col.number_input(f"{pair} ({PAIR_HELP[pair].upper()})", min_value=0.0001, max_value=1000.0, step=0.0001,
                             format="%.5f", key=key)  # fmt: skip
    chosen = reporting(market)
    if chosen is None:
        st.sidebar.warning("No FX spots: the ECB snapshot could not be read. Enter rates (ENTER RATES) to convert.")
        return None
    st.sidebar.caption(
        _fx_caption(chosen.fx)
        + (" ECB reference rates of the valuation date (daily history since 2015), indicative, not dealable quotes."
           if source == SOURCES[0] else "")
        + " Historical replays and the value at risk move the spots by the ECB rates of their dates (each can hold "
        "them); other scenarios hold them unless the SCENARIOS tab shocks them."
    )  # fmt: skip
    if source == SOURCES[0] and chosen.fx.asof is not None and chosen.fx.asof != market.asof:
        days = chosen.stale_days(market.asof)
        when = "before" if chosen.fx.asof < market.asof else "after"
        st.sidebar.warning(
            f"The ECB history has no rates for {market.asof} (not a TARGET business day, or beyond the days held): "
            f"converting at those of {chosen.fx.asof}, {days} days {when} the valuation date. Enter the rates of "
            f"{market.asof} (ENTER RATES) for a consistent conversion."
        )
    return chosen


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def currencies(positions: Sequence[Position], market: Market) -> List[str]:
    return position_currencies(positions, market)


def headline(currencies: Sequence[str], rep: Optional[Reporting]) -> Tuple[Optional[str], str]:
    """(label of the row or column holding the book's headline total, unit
    for card labels): a book in one currency, reported in it (or with no
    FX spots), is its own ``TOTAL`` in ``$``; otherwise the converted ``ALL
    IN <R>``, in ``IN $``. The label is None for an empty book or a book in
    several currencies without FX spots (no headline total exists)."""

    groups = ordered(currencies)
    if rep is not None and rep.needs_conversion(currencies):
        return rep.label, f"IN {rep.symbol.strip()}"
    if len(groups) == 1:
        return "TOTAL", symbol(groups[0]).strip()
    return None, "MIXED"


def table_unit(row_currencies: Sequence[str]) -> Optional[str]:
    """The currency symbol of a table whose rows are all in one currency
    (for its column headers), else None (headers without a currency and a
    CCY column)."""

    groups = ordered([c for c in row_currencies if c])
    return symbol(groups[0]).strip() if len(groups) == 1 else None


def conversion_note(currencies: Sequence[str], rep: Optional[Reporting], valuation: dt.date) -> str:
    """One sentence on how the totals were formed, for captions."""

    groups = ordered(currencies)
    if rep is None:
        if len(groups) > 1:
            return (f" Amounts in {' and '.join(groups)} are totalled per currency only: no FX spots to convert them "
                    "(enter rates under REPORTING CURRENCY).")  # fmt: skip
        return ""
    if not rep.needs_conversion(currencies):
        return ""
    per = " Totals are per currency;" if len(groups) > 1 else ""
    stale = ""
    if rep.fx.asof is not None and rep.fx.asof != valuation:
        stale = f" (FX of {rep.fx.asof}, not of the valuation date {valuation})"
    return (f"{per} {rep.label} converts {', '.join(c for c in groups if c != rep.currency)} at "
            f"{rep.describe(currencies)}{stale}.")  # fmt: skip


def record(rep: Optional[Reporting], currencies: Sequence[str]) -> dict:
    """The manifest entry of a conversion."""

    if rep is None:
        return {"reporting_currency": None, "fx": None, "position_currencies": ordered(currencies)}
    return {
        "reporting_currency": rep.currency,
        "position_currencies": ordered(currencies),
        "fx": {p: rep.fx.quote(p) for p in ("EURUSD", "GBPUSD", "EURGBP")},
        "fx_source": rep.fx.source,
        "fx_asof": rep.fx.asof,
        "converted": rep.needs_conversion(currencies),
    }
