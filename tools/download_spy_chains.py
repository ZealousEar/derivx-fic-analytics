"""Save the current option-chain snapshot of a ticker, one file per session,
to build a genuine dated history for the walk-forward test.

Yahoo Finance serves only the *current* chain: there is no historical
endpoint. This tool therefore never labels data with a past date. Each run
saves what Yahoo serves now, stamped with the session the quotes belong to
(``trade_date``), each expiry's real fetch time (``fetched_at_utc``) and the
spot from the same response. Past dates (``--start`` / ``--end`` / ``--date``
before the current session) are refused.

Run it once per trading day after the options close (16:15 New York), e.g.
from cron in New York time::

    CRON_TZ=America/New_York
    25 16 * * 1-5  cd /path/to/repo && python tools/download_spy_chains.py --ticker SPY --outdir data/market/spy

Output: ``<outdir>/<ticker>_<session>.<parquet|csv.gz>`` with every quote
(unfiltered, so other filters can be applied later) and
``<outdir>/<ticker>_<session>.json`` with the fetch status (requests,
retries, failed expiries, spot drift) and a run manifest (code version,
package versions). A file that already exists for the session is kept
unless ``--replace``. Exit status: 0 saved, 1 nothing saved (error, rate
limit, empty or refused), 3 saved but partial (some expiries failed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

from derivx.data.calendar import previous_trading_day, session_date
from derivx.data.chain_schema import chain_to_csv
from derivx.data.market_data import Backoff, ChainFetch, fetch_option_chain
from derivx.validation.manifest import build_manifest, manifest_bytes


def current_session(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """The latest session on or before ``now`` (UTC) in New York."""

    return previous_trading_day(session_date(now if now is not None else pd.Timestamp.now(tz="UTC")))


def refuse_past_dates(dates: Sequence[Optional[str]], now: Optional[pd.Timestamp] = None) -> Optional[str]:
    """An error message if any given date is before the current session."""

    session = current_session(now)
    for value in dates:
        if value is None:
            continue
        try:
            day = pd.Timestamp(dt.date.fromisoformat(value))
        except ValueError:
            return f"{value!r} is not a date in YYYY-MM-DD form"
        if day < session:
            return (
                f"refusing {day.date()}: Yahoo serves only the current chain (the {session.date()} session), so "
                "a past date would be today's quotes under a false label. Run this tool once per trading day "
                "to build history."
            )
    return None


def output_paths(outdir: Path, ticker: str, session: str, fmt: str) -> tuple[Path, Path]:
    stem = f"{ticker.lower()}_{session}"
    ext = "parquet" if fmt == "parquet" else "csv.gz"
    return outdir / f"{stem}.{ext}", outdir / f"{stem}.json"


def save_snapshot(result: ChainFetch, outdir: Path, fmt: str = "parquet", replace: bool = False) -> Optional[Path]:
    """Write the chain and its status; returns the chain file, or None when
    a file for this session exists and ``replace`` is False."""

    status = result.status
    data_path, meta_path = output_paths(outdir, status.ticker, str(status.quote_date), fmt)
    if data_path.exists() and not replace:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        result.chain.to_parquet(data_path, index=False)
    else:
        import gzip

        data_path.write_bytes(gzip.compress(chain_to_csv(result.chain), mtime=0))
    manifest = build_manifest(
        "snapshot",
        inputs={"ticker": status.ticker, "expiries_requested": status.expiries_requested},
        data={"source": "Yahoo Finance via yfinance (delayed quotes)", "fetched_at_utc": status.finished_at_utc},
        model={},
    )
    meta = {"fetch_status": status.as_dict(), "file": data_path.name, "manifest": manifest}
    meta_path.write_bytes(manifest_bytes(meta))
    return data_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default="SPY", help="underlying ticker (default SPY)")
    parser.add_argument("--outdir", default="data/market/spy", help="directory for the snapshot files")
    parser.add_argument("--max-expiries", type=int, default=None, help="request only the nearest N expiries")
    parser.add_argument("--expiries", nargs="*", help="request only these expiries (YYYY-MM-DD, listed by Yahoo)")
    parser.add_argument("--format", choices=("parquet", "csv.gz"), default="parquet")
    parser.add_argument("--replace", action="store_true", help="overwrite an existing file for the same session")
    parser.add_argument("--date", help="expected session (YYYY-MM-DD); refused if before the current session")
    parser.add_argument("--start", help=argparse.SUPPRESS)  # kept to explain the refusal
    parser.add_argument("--end", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    fetch: Callable[..., ChainFetch] = fetch_option_chain,
    now: Optional[pd.Timestamp] = None,
) -> int:
    args = parse_args(argv)
    refusal = refuse_past_dates([args.start, args.end, args.date], now)
    if refusal:
        print(f"error: {refusal}", file=sys.stderr)
        return 1
    result = fetch(args.ticker, expiries=args.expiries, max_expiries=args.max_expiries, backoff=Backoff(retries=4))
    status = result.status
    print(f"{status.ticker}: {status.summary()}")
    for note in status.notes:
        print(f"  note: {note}")
    for expiry, reason in status.failed_expiries.items():
        print(f"  failed {expiry}: {reason}")
    if not status.ok:
        return 1
    if args.date and status.quote_date != args.date:
        print(
            f"error: expected the {args.date} session, Yahoo served {status.quote_date}; nothing saved", file=sys.stderr
        )
        return 1
    path = save_snapshot(result, Path(args.outdir), fmt=args.format, replace=args.replace)
    if path is None:
        print(f"kept the existing file for the {status.quote_date} session (use --replace to overwrite)")
        return 0
    print(f"saved {status.rows:,} quotes -> {path}")
    return 3 if status.partial else 0


if __name__ == "__main__":
    sys.exit(main())
