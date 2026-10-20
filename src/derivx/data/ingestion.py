"""Loaders: the live Yahoo chain as a plain frame, and a rates-quote CSV."""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

import pandas as pd

from derivx.data.calendar import previous_trading_day, session_date
from derivx.data.market_data import FetchStatus, fetch_option_chain


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


def load_rates_quotes_from_csv(path: str) -> Dict[str, float]:
    """Load rates quotes CSV with columns: instrument,tenor,rate (decimal)."""

    df = pd.read_csv(path)
    if not {"instrument", "tenor", "rate"}.issubset(set(df.columns)):
        raise ValueError("CSV must contain instrument, tenor, rate columns")
    # Aggregate by tenor: prefer IRS over OIS if duplicates, else average
    df_sorted = df.sort_values(by=["tenor", "instrument"])  # OIS then IRS
    agg = df_sorted.groupby("tenor")["rate"].mean()
    return {k: float(v) for k, v in agg.items()}
