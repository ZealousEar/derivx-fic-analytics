"""Loaders: the live Yahoo chain as a plain frame, and rates-quote CSVs
(one curve, or a curve set with a ``curve`` column)."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from derivx.data.calendar import previous_trading_day, session_date
from derivx.data.market_data import FetchStatus, fetch_option_chain

if TYPE_CHECKING:
    from derivx.rates.curves import CurveQuote


class ChainFetchError(RuntimeError):
    """A live chain could not be fetched; ``status`` says why."""

    def __init__(self, status: FetchStatus):
        self.status = status
        super().__init__(f"{status.ticker}: {status.summary()}")


def load_equity_chain_from_yf(
    ticker: str,
    date: Optional[str | dt.date] = None,
    *,
    max_expiries: Optional[int] = None,
    allow_partial: bool = False,
    **fetch_kwargs,
) -> pd.DataFrame:
    """The current option chain of ``ticker`` from Yahoo Finance, in the
    schema of :mod:`derivx.data.chain_schema` (with ``trade_date`` = the
    session the quotes belong to and ``fetched_at_utc``).

    Yahoo has no history: ``date``, when given, must be the session being
    served now; an earlier date raises ``ValueError`` before any request,
    and a mismatch with the fetched session raises after it. Raises
    :class:`ChainFetchError` when the fetch fails, or is partial without
    ``allow_partial``."""

    if date is not None:
        wanted = pd.Timestamp(date).normalize()
        current = previous_trading_day(session_date(pd.Timestamp.now(tz="UTC")))
        if wanted < current:
            raise ValueError(
                f"Yahoo serves only the current chain ({current.date()} session): it cannot return {wanted.date()}. "
                "Build history by saving one snapshot per day (tools/download_spy_chains.py)."
            )
    result = fetch_option_chain(ticker, max_expiries=max_expiries, **fetch_kwargs)
    status = result.status
    if not status.ok or (status.partial and not allow_partial):
        raise ChainFetchError(status)
    if date is not None and status.quote_date != pd.Timestamp(date).date().isoformat():
        raise ValueError(f"asked for the {pd.Timestamp(date).date()} session but Yahoo served {status.quote_date}")
    return result.chain


def load_rates_quotes_from_csv(path: str, units: str = "decimal") -> List["CurveQuote"]:
    """Quotes for one curve from a CSV with columns ``instrument, tenor,
    rate``, as decimal :class:`~derivx.rates.curves.CurveQuote` objects in
    file order, ready for :func:`derivx.rates.term_structure.build_zero_curve`
    or :func:`derivx.rates.curves.bootstrap`.

    The instrument column is kept: each row is one of
    :data:`derivx.rates.curves.INSTRUMENT_KINDS` (``DEPO``, ``OIS``, ``IRS``,
    ...), and anything else (``BANANA``) is an error. (This loader used to
    return ``{tenor: rate}``, which dropped the instrument and so priced IRS
    rows as OIS.) ``units`` is what the rate column holds: ``decimal``
    (0.0512, the default, as in the bundled ``data/examples/rates_quotes.csv``),
    ``percent`` (5.12), ``bp`` (512) or ``auto`` (guessed, and refused when
    ambiguous); a rate outside -5%..30% after conversion is an error, so a
    percent file read as decimals fails instead of building.

    One quote per tenor: a tenor quoted twice raises ``ValueError``. (The
    loader once averaged them, e.g. a 2Y OIS at 4.0% and a 2Y IRS at 5.0%
    became 4.5%, blending two curves' quotes into one.) Quotes for several
    curves, or for a projection index, carry a ``curve`` column and go
    through :func:`load_curve_quotes_from_csv`. ``ValueError`` lists every
    problem.
    """

    from derivx.rates.indices import INDICES
    from derivx.rates.quotes import clean_quotes

    df = pd.read_csv(path)
    if not {"instrument", "tenor", "rate"}.issubset(set(df.columns)):
        raise ValueError("CSV must contain instrument, tenor, rate columns")
    if "curve" in df.columns:
        names = sorted(set(df["curve"].dropna().astype(str).str.strip().str.upper()))
        if len(names) > 1 or any(n in INDICES for n in names):
            raise ValueError(
                f"the CSV holds quotes for {', '.join(names)}: load a curve set (or a projection curve) with "
                "load_curve_quotes_from_csv"
            )
    quotes, issues, _ = clean_quotes(df.to_dict("records"), units=units)
    errors = [i.message for i in issues if i.level == "error"]
    if errors:
        raise ValueError("; ".join(errors))
    return quotes


def load_curve_quotes_from_csv(path: str, units: str = "auto") -> Dict[str, List[object]]:
    """Quotes for a curve set from a CSV with columns ``instrument, tenor,
    rate`` and an optional ``curve`` (the discount curve, e.g. ``EUR ESTR``,
    or a projection index, e.g. ``EURIBOR 6M``; one curve if absent):
    ``{curve: [CurveQuote, ...]}`` in decimals, each curve's rows validated
    with :func:`derivx.rates.quotes.clean_quotes` (its instruments, units,
    tenors, duplicates). ``ValueError`` lists every problem."""

    from derivx.rates.curves import INSTRUMENT_KINDS
    from derivx.rates.indices import INDICES
    from derivx.rates.multicurve import PROJECTION_KINDS
    from derivx.rates.quotes import clean_quotes

    df = pd.read_csv(path)
    if not {"instrument", "tenor", "rate"}.issubset(set(df.columns)):
        raise ValueError("CSV must contain instrument, tenor, rate columns (and optionally curve)")
    names = df["curve"].astype(str).str.strip().str.upper() if "curve" in df.columns else pd.Series(["CURVE"] * len(df))
    out: Dict[str, List[object]] = {}
    errors: List[str] = []
    for name in dict.fromkeys(names):
        rows = df[names.values == name].to_dict("records")
        kinds = PROJECTION_KINDS if name in INDICES else INSTRUMENT_KINDS
        quotes, issues, _ = clean_quotes(rows, units=units, kinds=kinds)
        errors += [f"{name}: {i.message}" for i in issues if i.level == "error"]
        out[name] = quotes
    if errors:
        raise ValueError("; ".join(errors))
    return out
