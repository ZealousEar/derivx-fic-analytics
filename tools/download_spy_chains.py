"""Utility script to download filtered SPY option chains for a date range."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from derivx.data.market_data import (
    apply_liquidity_filters,
    compute_moneyness,
    fetch_option_chain,
)


def business_days(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += dt.timedelta(days=1)


def save_chain(df: pd.DataFrame, path: Path, ticker: str, trade_date: dt.date) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / f"{ticker.lower()}_chain_{trade_date:%Y%m%d}.parquet"
    df.to_parquet(out_file, index=False)
    return out_file


def download_history(
    ticker: str,
    start: dt.date,
    end: dt.date,
    outdir: Path,
    expiries: Optional[list[str]] = None,
) -> None:
    for trade_date in business_days(start, end):
        print(f"Fetching {ticker} chain for {trade_date:%Y-%m-%d}...")
        chain = fetch_option_chain(ticker, trade_date, expiries=expiries)
        if chain.empty:
            print("  no data returned")
            continue
        chain = apply_liquidity_filters(chain)
        if chain.empty:
            print("  all rows filtered out")
            continue
        chain = compute_moneyness(chain)
        outfile = save_chain(chain, outdir, ticker, trade_date)
        print(f"  saved {len(chain)} quotes -> {outfile}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY", help="Underlying ticker (default: SPY)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--outdir",
        default="data/market/spy",
        help="Directory to write parquet files (default: data/market/spy)",
    )
    parser.add_argument(
        "--expiries",
        nargs="*",
        help="Optional list of expiration dates to pull (YYYY-MM-DD).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if end < start:
        raise ValueError("end date must be >= start date")
    outdir = Path(args.outdir)
    download_history(args.ticker, start, end, outdir, expiries=args.expiries)


if __name__ == "__main__":
    main()
