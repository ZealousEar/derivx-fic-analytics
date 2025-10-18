from __future__ import annotations

from typing import Dict

import pandas as pd


def load_equity_chain_from_yf(ticker: str, date: str) -> pd.DataFrame:
    """Placeholder: fetch option chain via yfinance (to be implemented)."""

    raise NotImplementedError("yfinance integration deferred; use sample CSVs")


def load_rates_quotes_from_csv(path: str) -> Dict[str, float]:
    """Load rates quotes CSV with columns: instrument,tenor,rate (decimal)."""

    df = pd.read_csv(path)
    if not {"instrument", "tenor", "rate"}.issubset(set(df.columns)):
        raise ValueError("CSV must contain instrument, tenor, rate columns")
    # Aggregate by tenor: prefer IRS over OIS if duplicates, else average
    df_sorted = df.sort_values(by=["tenor", "instrument"])  # OIS then IRS
    agg = df_sorted.groupby("tenor")["rate"].mean()
    return {k: float(v) for k, v in agg.items()}


