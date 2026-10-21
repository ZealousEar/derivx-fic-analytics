"""Currency-aware totals of a rates book: subtotals per currency, and totals
converted to one reporting currency at stated FX spots.

Currencies
    A position's values are in the currency of the curve it is priced on:
    swaps, basis swaps, FRAs, futures, caps, floors and swaptions in the swap
    curve's currency (USD, EUR or GBP), US Treasuries in USD
    (:func:`position_currency`). Amounts in different currencies are never
    added as they are: every total is per currency (``TOTAL EUR``,
    ``TOTAL USD``; plain ``TOTAL`` when the book holds one currency), and the
    whole book is also shown converted to the reporting currency R
    (``ALL IN R``) whenever it holds a currency other than R.

FX spots (:class:`FxSpots`)
    ``per_eur[c]`` is the number of units of currency c per 1 EUR, the ECB's
    quotation (EUR itself is 1). The value in currency R of one unit of c is
    ``per_eur[R] / per_eur[c]``. Pairs are quoted in market convention
    (:data:`PAIRS`: EURUSD, GBPUSD, EURGBP = units of the second currency per
    1 unit of the first). The spots come from the ECB euro reference rates
    of the valuation date (:func:`store_spots`: the ``FX_ECB`` history of
    the rates history store, every TARGET business day since 2015 in the
    committed sample plus whatever the download cache adds; :func:`ecb_spots`
    for an :class:`~derivx.fx.market_data.EcbRates`; indicative reference
    rates, not dealable quotes) or from rates the user enters
    (:func:`entered_spots`); every converted total states the rates used and
    their as-of date (:meth:`Reporting.describe`).

FX shocks
    A scenario can move FX spots (``Shock.fx``: % moves of EURUSD and
    GBPUSD, i.e. of the euro's and the pound's value in dollars). A move
    ``m_c`` of each currency against USD (``m_USD = 0``) moves the value in
    R of one unit of c by ``x_c = (1 + m_c) / (1 + m_R) - 1``
    (:func:`translation`). Local-currency P&L does not depend on FX (curves
    and vols do not move with it); the converted P&L of a position in c is::

        pnl_R = pnl_c x S_c + (PV_c(horizon) + cash_c) x S_c x x_c

    with ``S_c`` today's spot: the first term is the local P&L at today's
    spot, the second the FX translation of the position's value at the
    horizon (zero for c = R). Without an FX shock, spots are held: no FX
    forward drift, FX carry or cross-currency basis over a horizon. A
    historical replay (:func:`derivx.risk.scenarios.historical`) moves the
    spots by the ECB reference-rate changes of its own dates, and the value
    at risk (:mod:`derivx.risk.var`) by those of each scenario day, unless
    asked to hold them.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from derivx.rates.curves import Market
from derivx.risk.sensitivities import Position

CURRENCIES = ("USD", "EUR", "GBP")
SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}
PAIRS = {frozenset(("EUR", "USD")): "EURUSD", frozenset(("GBP", "USD")): "GBPUSD", frozenset(("EUR", "GBP")): "EURGBP"}
SHOCK_PAIRS = ("EURUSD", "GBPUSD")  # FX shocks move each currency against the dollar
DEFAULT_REPORTING = "USD"
TOTAL = "TOTAL"
ALL_IN = "ALL IN"


def symbol(currency: str) -> str:
    """``$``, ``€`` or ``£`` (the code and a space for another currency)."""

    return SYMBOLS.get(currency, f"{currency} ")


def _code(currency: object) -> str:
    code = str(currency).strip().upper()
    if code not in CURRENCIES:
        raise ValueError(f"currency {currency!r}: use one of {', '.join(CURRENCIES)}")
    return code


def market_pair(a: str, b: str) -> str:
    """Market-convention code of the pair of two different currencies
    (``EURUSD``, ``GBPUSD``, ``EURGBP``)."""

    a, b = _code(a), _code(b)
    if a == b:
        raise ValueError(f"a currency pair needs two currencies, got {a} twice")
    return PAIRS[frozenset((a, b))]


def total_label(currency: str, mixed: bool) -> str:
    """``TOTAL`` for a book in one currency, ``TOTAL <CCY>`` for one
    currency's trades of a book in several."""

    return f"{TOTAL} {currency}" if mixed else TOTAL


def all_in_label(currency: str) -> str:
    """Label of the whole book converted to ``currency``: ``ALL IN USD``."""

    return f"{ALL_IN} {currency}"


# --------------------------------------------------------------------------
# FX spots
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FxSpots:
    """FX spots as units of each currency per 1 EUR (``per_eur``, EUR = 1),
    dated ``asof``, from ``source`` (a label for captions and manifests)."""

    per_eur: Tuple[Tuple[str, float], ...]
    asof: Optional[dt.date]
    source: str

    def __post_init__(self) -> None:
        rates = {"EUR": 1.0}
        for ccy, value in dict(self.per_eur).items():
            code = str(ccy).strip().upper()
            value = float(value)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"the {code} FX spot must be a positive number, got {value!r}")
            if code == "EUR" and abs(value - 1.0) > 1e-12:
                raise ValueError("EUR per 1 EUR is 1")
            rates[code] = value
        object.__setattr__(self, "per_eur", tuple(sorted(rates.items())))

    @property
    def rates(self) -> Dict[str, float]:
        return dict(self.per_eur)

    def value(self, currency: str, reporting: str) -> float:
        """Units of ``reporting`` per 1 unit of ``currency``."""

        rates = self.rates
        missing = [c for c in (currency, reporting) if c not in rates]
        if missing:
            raise ValueError(f"no FX spot for {', '.join(missing)} ({self.source})")
        return rates[reporting] / rates[currency]

    def quote(self, pair: str) -> float:
        """Market quote of ``pair`` (``EURUSD``: USD per 1 EUR)."""

        code = str(pair).strip().upper().replace("/", "")
        if code not in PAIRS.values():
            raise ValueError(f"pair {pair!r}: use one of {', '.join(PAIRS.values())}")
        return self.value(code[:3], code[3:])

    @property
    def dated(self) -> str:
        return f"{self.source}, {self.asof.isoformat()}" if self.asof is not None else self.source


def ecb_spots(rates, on: Optional[dt.date] = None) -> FxSpots:
    """FX spots from ECB euro reference rates (an
    :class:`~derivx.fx.market_data.EcbRates`): the rates of the latest day
    on or before ``on`` that the rates hold (default: their own as-of day);
    if they hold no day that early, the earliest day they hold (later than
    ``on``: say so where it is used, see :meth:`Reporting.stale_days`)."""

    history = dict(getattr(rates, "history", {}) or {})
    history.setdefault(rates.asof, dict(rates.per_eur))
    days = sorted(d for d, values in history.items() if all(c in values for c in CURRENCIES if c != "EUR"))
    if not days:
        raise ValueError("the ECB rates hold no day with both a USD and a GBP rate")
    if on is None:
        day = rates.asof if rates.asof in days else days[-1]
    else:
        earlier = [d for d in days if d <= on]
        day = earlier[-1] if earlier else days[0]
    origin = "bundled snapshot" if getattr(rates, "origin", "") == "bundled" else "fetched live"
    values = history[day]
    return FxSpots(tuple((c, float(values[c])) for c in CURRENCIES if c != "EUR"), day,
                   f"ECB euro reference rates ({origin})")  # fmt: skip


def store_spots(store, on: Optional[dt.date] = None) -> FxSpots:
    """FX spots from the ECB reference rates of a rates history store
    (:meth:`~derivx.data.rates_history.HistoryStore.fx_table`): the latest
    day on or before ``on`` (default: the latest day held); if the store
    holds no day that early, its first day (later than ``on``: say so
    where it is used, see :meth:`Reporting.stale_days`). The source names
    where that day's rows came from (bundled snapshot or a download)."""

    table = store.fx_table()
    if table.empty or any(c not in table.columns for c in CURRENCIES if c != "EUR"):
        raise ValueError("the history store holds no ECB reference rates with both a USD and a GBP rate")
    days = list(table.index)
    if on is None:
        day = days[-1]
    else:
        earlier = [d for d in days if d <= on]
        day = earlier[-1] if earlier else days[0]
    origins = sorted({str(o) for o in store.rows("FX_ECB").loc[lambda f: f["asof"] == day, "origin"]})
    origin = "bundled snapshot" if origins == ["bundled snapshot"] else (", ".join(origins) or "history store")
    row = table.loc[day]
    return FxSpots(tuple((c, float(row[c])) for c in CURRENCIES if c != "EUR"), day,
                   f"ECB euro reference rates ({origin})")  # fmt: skip


def entered_spots(quotes: Mapping[str, float], asof: Optional[dt.date] = None,
                  source: str = "FX rates entered by you") -> FxSpots:  # fmt: skip
    """FX spots from market quotes ``{"EURUSD": 1.16, "GBPUSD": 1.34}``
    (USD per 1 EUR and per 1 GBP); EURGBP is their cross."""

    values = {str(k).strip().upper().replace("/", ""): v for k, v in quotes.items()}
    missing = [p for p in SHOCK_PAIRS if p not in values]
    if missing:
        raise ValueError(f"give {' and '.join(missing)} (USD per 1 unit of the first currency)")
    for pair in SHOCK_PAIRS:
        value = float(values[pair])
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{pair} must be a positive number, got {values[pair]!r}")
    eurusd, gbpusd = float(values["EURUSD"]), float(values["GBPUSD"])
    return FxSpots((("USD", eurusd), ("GBP", eurusd / gbpusd)), asof, source)


# --------------------------------------------------------------------------
# Reporting currency
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reporting:
    """The reporting currency and the FX spots that convert into it."""

    currency: str
    fx: FxSpots

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", _code(self.currency))
        self.fx.value(self.currency, self.currency)  # the spots must hold the reporting currency

    @property
    def label(self) -> str:
        return all_in_label(self.currency)

    @property
    def symbol(self) -> str:
        return symbol(self.currency)

    def rate(self, currency: str) -> float:
        """Units of the reporting currency per 1 unit of ``currency``."""

        return 1.0 if currency == self.currency else self.fx.value(currency, self.currency)

    def rates(self, currencies: Sequence[str]) -> np.ndarray:
        return np.array([self.rate(c) for c in currencies], dtype=float)

    def needs_conversion(self, currencies: Sequence[str]) -> bool:
        """True if some amount is not in the reporting currency."""

        return any(c != self.currency for c in currencies)

    def pairs(self, currencies: Sequence[str]) -> List[str]:
        """Market pairs that convert ``currencies`` to the reporting one."""

        return sorted({market_pair(c, self.currency) for c in currencies if c != self.currency})

    def describe(self, currencies: Sequence[str]) -> str:
        """``EURUSD 1.16420 (ECB euro reference rates (bundled snapshot),
        2026-10-20)``: the rates the conversion of ``currencies`` uses."""

        pairs = self.pairs(currencies)
        if not pairs:
            return f"every amount is in {self.currency}: no conversion"
        quotes = ", ".join(f"{p} {self.fx.quote(p):.5f}" for p in pairs)
        return f"{quotes} ({self.fx.dated})"

    def stale_days(self, valuation: dt.date) -> int:
        """Calendar days between the FX spots' date and the valuation date
        (0 when the same day or undated)."""

        return 0 if self.fx.asof is None else abs((valuation - self.fx.asof).days)


def default_reporting(on: Optional[dt.date] = None, currency: str = DEFAULT_REPORTING, store=None) -> Reporting:
    """Reporting in ``currency`` at the ECB reference rates of the latest
    day on or before ``on`` in ``store`` (default: the committed history
    sample; no network)."""

    if store is None:
        from derivx.data.rates_history import bundled_store

        store = bundled_store()
    return Reporting(currency, store_spots(store, on))


# --------------------------------------------------------------------------
# Currencies of positions
# --------------------------------------------------------------------------


def market_currency(market: Market) -> str:
    """Currency of a market's swap curve (USD when it has no conventions)."""

    conv = market.curve.conventions
    return conv.currency if conv is not None else "USD"


def position_currency(position: Position, market: Market) -> str:
    """Currency of a position's values: its ``details["currency"]`` when set
    (the BOOK's trades), else USD for a bond (a US Treasury priced on the
    Treasury curve) and the swap curve's currency for everything else."""

    stated = position.details.get("currency") if isinstance(position.details, Mapping) else None
    if isinstance(stated, str) and stated in CURRENCIES:
        return stated
    return "USD" if position.kind == "bond" else market_currency(market)


def position_currencies(positions: Sequence[Position], market: Market) -> List[str]:
    return [position_currency(p, market) for p in positions]


def ordered(currencies: Sequence[str]) -> List[str]:
    """Distinct currencies in a fixed order (USD, EUR, GBP)."""

    have = set(currencies)
    return [c for c in CURRENCIES if c in have] + sorted(have - set(CURRENCIES))


# --------------------------------------------------------------------------
# FX shocks
# --------------------------------------------------------------------------


def fx_moves(fx: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    """Relative moves of each currency's value in USD (decimal; USD 0) from
    ``(pair, % move)`` of :data:`SHOCK_PAIRS` (EURUSD +10 -> EUR +0.10)."""

    moves = {"USD": 0.0}
    for pair, pct in fx:
        code = str(pair).strip().upper()
        if code not in SHOCK_PAIRS:
            raise ValueError(f"FX shock pair {pair!r}: use {' or '.join(SHOCK_PAIRS)}")
        moves[code[:3]] = float(pct) / 100.0
    return moves


def translation(currencies: Sequence[str], reporting: str, fx: Sequence[Tuple[str, float]]) -> np.ndarray:
    """``x_c = (1 + m_c) / (1 + m_R) - 1`` for each of ``currencies``: the
    relative move of its value in the reporting currency R under the FX
    shock ``fx`` (see the module docstring)."""

    moves = fx_moves(fx)
    base = 1.0 + moves.get(reporting, 0.0)
    return np.array([(1.0 + moves.get(c, 0.0)) / base - 1.0 for c in currencies], dtype=float)


# --------------------------------------------------------------------------
# Totals
# --------------------------------------------------------------------------


def converted(values: np.ndarray, currencies: Sequence[str], reporting: Reporting) -> np.ndarray:
    """Amounts (last axis: one column per currency in ``currencies``)
    converted to the reporting currency at today's spots."""

    values = np.asarray(values, dtype=float)
    return values * reporting.rates(currencies)


def add_total_rows(
    frame: pd.DataFrame,
    values: Sequence[str],
    label: str,
    currency: str = "currency",
    reporting: Optional[Reporting] = None,
    extra: Optional[Mapping[str, float]] = None,
    note: Optional[str] = None,
) -> pd.DataFrame:
    """``frame`` (one row per position, amounts in the row's ``currency``)
    with a total row per currency (``TOTAL`` / ``TOTAL <CCY>`` in column
    ``label``) and, when ``reporting`` is given and some row is in another
    currency, an ``ALL IN <R>`` row: each of ``values`` summed over every
    row converted at today's spot, plus ``extra[column]`` (an amount in R,
    e.g. the FX translation P&L of an FX shock). ``note`` (a column name)
    receives how each total was formed."""

    if frame.empty:
        return frame
    if note and note not in frame.columns:
        frame = frame.assign(**{note: ""})
    currencies = [str(c) for c in frame[currency]]
    groups = ordered(currencies)
    mixed = len(groups) > 1
    rows = []
    for ccy in groups:
        part = frame[frame[currency] == ccy]
        row: Dict[str, object] = {c: None for c in frame.columns}
        row.update({label: total_label(ccy, mixed), currency: ccy})
        row.update({c: float(np.nansum(pd.to_numeric(part[c]).to_numpy(dtype=float))) for c in values})
        if note:
            row[note] = f"{len(part)} {'position' if len(part) == 1 else 'positions'} in {ccy}"
        rows.append(row)
    if reporting is not None and reporting.needs_conversion(currencies):
        rates = reporting.rates(currencies)
        row = {c: None for c in frame.columns}
        row.update({label: reporting.label, currency: reporting.currency})
        for c in values:
            row[c] = float(np.nansum(pd.to_numeric(frame[c]).to_numpy(dtype=float) * rates))
        for c, amount in (extra or {}).items():
            row[c] = float(row.get(c) or 0.0) + float(amount)
        if note:
            row[note] = f"converted: {reporting.describe(currencies)}"
        rows.append(row)
    return pd.concat([frame, pd.DataFrame(rows, columns=frame.columns)], ignore_index=True)


def add_total_columns(
    grid: pd.DataFrame,
    currencies: Sequence[str],
    reporting: Optional[Reporting] = None,
    base: Optional[Sequence[float]] = None,
    fx: Optional[Sequence[Sequence[Tuple[str, float]]]] = None,
) -> pd.DataFrame:
    """A scenario grid (rows = scenarios, one column per position, P&L in
    the position's currency ``currencies[j]``) with a total column per
    currency and, when ``reporting`` is given and some position is in
    another currency (or a scenario moves FX), ``ALL IN <R>``: the P&L
    converted at today's spots plus, for a scenario with an FX shock
    (``fx[i]``, ``Shock.fx``), the FX translation of each position's value
    after the scenario (``base[j]`` today's value + its P&L; see the module
    docstring)."""

    table = grid.copy()
    positions = list(grid.columns)
    if len(positions) != len(currencies):
        raise ValueError(f"{len(positions)} position columns but {len(currencies)} currencies")
    groups = ordered(currencies)
    mixed = len(groups) > 1
    values = grid.to_numpy(dtype=float)
    for ccy in groups:
        picks = [j for j, c in enumerate(currencies) if c == ccy]
        table[total_label(ccy, mixed)] = values[:, picks].sum(axis=1) if picks else 0.0
    shocks = [tuple(s) for s in fx] if fx is not None else [()] * len(grid)
    if len(shocks) != len(grid):
        raise ValueError(f"{len(grid)} scenarios but {len(shocks)} FX shocks")
    if reporting is not None and reporting.needs_conversion(currencies):
        rates = reporting.rates(currencies)
        local = values * rates
        if any(shocks):
            if base is None:
                raise ValueError("FX shocks need today's value of each position (base)")
            after = (np.asarray(base, dtype=float)[None, :] + values) * rates
            moves = np.array([translation(currencies, reporting.currency, s) for s in shocks])
            local = local + after * moves
        table[reporting.label] = local.sum(axis=1)
    return table


def fx_translation(
    base: Sequence[float],
    pnl: Sequence[float],
    currencies: Sequence[str],
    reporting: Reporting,
    fx: Sequence[Tuple[str, float]],
) -> np.ndarray:
    """FX translation P&L of each position in the reporting currency:
    ``(base + pnl) x S_c x x_c`` (value after the scenario, at today's spot,
    times the relative FX move of its currency)."""

    after = (np.asarray(base, dtype=float) + np.asarray(pnl, dtype=float)) * reporting.rates(currencies)
    return after * translation(currencies, reporting.currency, fx)


def explain_fx(
    shocked_pv: Sequence[float],
    cash: Sequence[float],
    currencies: Sequence[str],
    reporting: Reporting,
    fx: Sequence[Tuple[str, float]],
) -> Dict[str, float]:
    """What an FX shock adds to the ``ALL IN`` row of a P&L explain, in the
    reporting currency: the horizon value and the cash are converted at the
    shocked spots (``S_c (1 + x_c)``), so each gains ``amount x S_c x x_c``,
    and the P&L gains their sum, shown as its own ``fx`` component."""

    moves = translation(currencies, reporting.currency, fx) * reporting.rates(currencies)
    pv = float(np.sum(np.asarray(shocked_pv, dtype=float) * moves))
    paid = float(np.sum(np.asarray(cash, dtype=float) * moves))
    return {"shocked_pv": pv, "cash": paid, "pnl": pv + paid, "fx": pv + paid}


__all__ = [
    "ALL_IN",
    "CURRENCIES",
    "DEFAULT_REPORTING",
    "FxSpots",
    "PAIRS",
    "Reporting",
    "SHOCK_PAIRS",
    "TOTAL",
    "add_total_columns",
    "add_total_rows",
    "all_in_label",
    "converted",
    "default_reporting",
    "ecb_spots",
    "entered_spots",
    "explain_fx",
    "store_spots",
    "fx_moves",
    "fx_translation",
    "market_currency",
    "market_pair",
    "ordered",
    "position_currencies",
    "position_currency",
    "symbol",
    "total_label",
    "translation",
]
