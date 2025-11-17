"""Utilities for downloading and filtering real option chains via yfinance."""
from __future__ import annotations

import datetime as dt
import time
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from requests.exceptions import JSONDecodeError, RequestException

def _call_with_retries(func, retries: int = 3, delay: float = 0.75):
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return func()
        except (JSONDecodeError, RequestException, ValueError) as exc:
            last_exc = exc
            time.sleep(delay * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unreachable: retries exhausted without exception")

REQUIRED_COLUMNS = [
    "trade_date",
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
    "mid",
    "volume",
    "open_interest",
    "implied_vol",
    "underlying_price",
]


def _get_underlying_price(ticker: yf.Ticker) -> float:
    try:
        return float(ticker.fast_info["last_price"])
    except Exception:  # pragma: no cover - fallback path
        hist = ticker.history(period="1d")
        if hist.empty:
            return float("nan")
        return float(hist["Close"].iloc[-1])


def _normalize_chain(df: pd.DataFrame, option_type: str, expiry: str) -> pd.DataFrame:
    if df.empty:
        return df
    renamed = df.rename(
        columns={
            "openInterest": "open_interest",
            "impliedVolatility": "implied_vol",
            "lastPrice": "last_price",
            "inTheMoney": "in_the_money",
        }
    ).copy()
    renamed["option_type"] = option_type
    renamed["expiry"] = pd.Timestamp(expiry).tz_localize(None)
    if "bid" in renamed.columns and "ask" in renamed.columns:
        renamed["mid"] = (renamed["bid"] + renamed["ask"]) / 2.0
    else:
        renamed["mid"] = np.nan
    return renamed


def fetch_option_chain(
    ticker: str,
    trade_date: Optional[dt.date] = None,
    expiries: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Download the option chain for ``ticker`` as-of ``trade_date``.

    Parameters
    ----------
    ticker : str
        Underlying ticker, e.g. "SPY".
    trade_date : date, optional
        As-of date used for labelling; defaults to today.
    expiries : sequence of str, optional
        Subset of expiration dates (YYYY-MM-DD) to request. If ``None`` the
        entire list provided by Yahoo Finance is requested.
    """

    trade_date = trade_date or dt.date.today()
    try:
        yf_ticker = yf.Ticker(ticker)
        available_expiries = _call_with_retries(lambda: yf_ticker.options)
    except (JSONDecodeError, RequestException, ValueError):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    except Exception:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    if not available_expiries:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    if expiries:
        target_expiries = [exp for exp in expiries if exp in available_expiries]
        if not target_expiries:
            target_expiries = available_expiries
    else:
        target_expiries = available_expiries

    frames: list[pd.DataFrame] = []
    for expiry in target_expiries:
        try:
            chain = _call_with_retries(lambda: yf_ticker.option_chain(expiry))
        except Exception:
            continue
        frames.append(_normalize_chain(chain.calls, "call", expiry))
        frames.append(_normalize_chain(chain.puts, "put", expiry))

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    chain_df = pd.concat(frames, ignore_index=True)
    chain_df["trade_date"] = pd.Timestamp(trade_date)
    chain_df["underlying_price"] = _get_underlying_price(yf_ticker)
    for col in REQUIRED_COLUMNS:
        if col not in chain_df:
            chain_df[col] = np.nan
    return chain_df[REQUIRED_COLUMNS + [c for c in chain_df.columns if c not in REQUIRED_COLUMNS]]


def apply_liquidity_filters(
    chain: pd.DataFrame,
    min_volume: int = 10,
    max_spread: float = 0.2,
    min_mid: float = 0.1,
) -> pd.DataFrame:
    """Remove illiquid quotes from the option chain."""

    required = {"bid", "ask", "volume"}
    missing = required - set(chain.columns)
    if missing:
        raise ValueError(f"Chain missing columns: {missing}")

    filtered = chain.copy()
    if "mid" not in filtered:
        filtered["mid"] = (filtered["bid"] + filtered["ask"]) / 2.0

    cond = filtered["volume"].fillna(0) >= min_volume
    cond &= filtered["bid"].fillna(0) > 0
    cond &= filtered["ask"].fillna(0) > filtered["bid"].fillna(0)
    cond &= filtered["mid"].fillna(0) >= min_mid
    if max_spread is not None:
        spread = (filtered["ask"] - filtered["bid"]).abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_spread = spread / filtered["mid"].replace(0, np.nan)
        cond &= rel_spread <= max_spread
    return filtered.loc[cond].reset_index(drop=True)


def compute_moneyness(chain: pd.DataFrame, column: str = "moneyness") -> pd.DataFrame:
    """Add a simple strike/spot moneyness column."""

    if "underlying_price" not in chain:
        raise ValueError("Chain must contain 'underlying_price' column")
    result = chain.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        result[column] = result["strike"] / result["underlying_price"]
    return result
