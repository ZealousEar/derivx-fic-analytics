"""``derivx`` command line.

    derivx pull fred --ids DGS2,DGS10 --start 2015-01-01
    derivx pull fred --ids ust,ice --start 2000-01-01   # groups: ust, ice, fixings
    derivx pull treasury --year 2026
    derivx pull nyfed --rate sofr --last 30
    derivx pull ecb-estr --start 2026-01-01
    derivx pull ecb-yc --curve AAA --measure par --tenors 2Y,10Y,30Y --last 20
    derivx pull boe-ois --archive daily --start 2022-09-01 --end 2022-10-31
    derivx pull latest                 # every fast live source, as the app's LIVE mode
    derivx pull fred --ids DGS10 --offline --out dgs10.csv

``pull`` downloads into the cache (``data/cache`` or ``$DERIVX_CACHE_DIR``)
and prints what it got: rows, series, date range, fetch time and the
payload's SHA-256. ``--offline`` reads only the cache. ``--out`` writes the
normalised long format (CSV or Parquet).

Each subcommand registers itself with :func:`command`; later commands
(curve, risk, scenario) plug in the same way.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from derivx.data import rates_sources as rs
from derivx.data.cache import DiskCache

Register = Callable[[argparse.ArgumentParser], None]
COMMANDS: Dict[str, Register] = {}
EXIT_SOURCE_ERROR = 1


def command(name: str) -> Callable[[Register], Register]:
    """Decorator: ``register(parser)`` fills the subcommand's parser and
    sets ``func`` (``args -> exit code``) with ``parser.set_defaults``."""

    def wrap(register: Register) -> Register:
        COMMANDS[name] = register
        return register

    return wrap


def _date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a date; use YYYY-MM-DD") from None


def _list(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


FRED_GROUPS = {
    "UST": rs.UST_FRED_IDS,  # Treasury par (CMT) yields, 1M-30Y
    "ICE": rs.ICE_SWAP_FRED_IDS,  # ICE USD swap rates, 2000-07 to 2016-10-28
    "FIXINGS": ("SOFR", "EFFR", "OBFR", "IUDSOIA", "ECBESTRVOLWGTTRMDMNRT"),
}


def _fred_ids(text: str) -> List[str]:
    """Comma-separated FRED ids; ``ust``, ``ice`` and ``fixings`` expand to
    their series."""

    out: List[str] = []
    for part in _list(text):
        out += list(FRED_GROUPS.get(part.upper(), (part,)))
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------
# pull
# --------------------------------------------------------------------------


def _summary(fetched: rs.Fetched, show: int) -> str:
    frame = fetched.frame
    how = "from the cache" if fetched.from_cache else "downloaded"
    lines = [f"{fetched.source}: {len(frame)} rows {how} (fetched {fetched.fetched_at}, sha256 {fetched.sha256[:12]})"]
    if len(frame):
        for series, rows in frame.groupby("series", sort=False):
            first, last = min(rows["asof"]), max(rows["asof"])
            latest = rows.loc[rows["asof"] == last].iloc[-1]
            tenor = latest["tenor"] or "-"
            lines.append(
                f"  {series:<40} {tenor:>5}  {len(rows):>6} obs  {first} .. {last}  "
                f"last {latest['value_original']:g} {latest['unit_original']} ({latest['measure']})"
            )
    lines += [f"  note: {n}" for n in fetched.notices]
    if show and len(frame):
        lines.append(frame.tail(show).to_string(index=False))
    return "\n".join(lines)


def _write(frame: pd.DataFrame, out: Optional[Path]) -> Optional[str]:
    if out is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        frame.to_parquet(out, index=False)
    else:
        frame.to_csv(out, index=False)
    return f"wrote {len(frame)} rows to {out}"


def _fetch(args: argparse.Namespace) -> rs.Fetched:
    common = dict(
        offline=args.offline or None, max_age=args.max_age, cache=DiskCache(args.cache) if args.cache else None
    )
    if args.timeout is not None:
        common["timeout"] = args.timeout
    source = args.source
    if source == "fred":
        units = dict(u.split("=", 1) for u in args.units) if args.units else None
        return rs.fred_series(args.ids, args.start, args.end, units=units, vintage=args.vintage, **common)
    if source == "treasury":
        return rs.treasury_par_curve(args.year, **common)
    if source == "nyfed":
        if args.start is None and args.last is None:
            args.last = 30
        return rs.nyfed_rate(args.rate, last_n=args.last, start=args.start, end=args.end, **common)
    if source == "ecb-estr":
        return rs.ecb_estr(args.start, args.end, args.last, **common)
    if source == "ecb-yc":
        return rs.ecb_yield_curve(args.tenors, args.curve, args.measure, args.start, args.end, args.last, **common)
    if source == "boe-ois":
        return rs.boe_ois_curve(args.archive, args.start, args.end, **common)
    raise ValueError(f"unknown source {source!r}")


def _pull(args: argparse.Namespace) -> int:
    if args.source == "latest":
        from derivx.data.rates_history import refresh_live

        result = refresh_live(
            max_age=args.max_age if args.max_age is not None else 0.0,
            timeout=args.timeout or 8.0,
            offline=args.offline or None,
            cache=DiskCache(args.cache) if args.cache else None,
        )
        for name, ok in result.ok.items():
            print(f"{'ok  ' if ok else 'FAIL'} {name}")
        for notice in result.notices:
            print(f"  note: {notice}")
        return 0 if all(result.ok.values()) else EXIT_SOURCE_ERROR
    try:
        fetched = _fetch(args)
    except rs.SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_SOURCE_ERROR
    print(_summary(fetched, args.show))
    message = _write(fetched.frame, args.out)
    if message:
        print(message)
    return 0


@command("pull")
def _register_pull(parser: argparse.ArgumentParser) -> None:
    """download rates market data into the cache (FRED, Treasury, NY Fed, ECB, BoE)"""

    sources = parser.add_subparsers(dest="source", required=True, metavar="SOURCE")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = sources.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("--offline", action="store_true", help="read only the cache, never the network")
        sub.add_argument("--max-age", type=float, help="reuse a cached copy younger than this many seconds")
        sub.add_argument("--timeout", type=float, help="seconds per request attempt")
        sub.add_argument("--cache", type=Path, help="cache directory (default data/cache or $DERIVX_CACHE_DIR)")
        sub.add_argument("--out", type=Path, help="write the long-format rows to a .csv or .parquet file")
        sub.add_argument("--show", type=int, default=0, help="print the last N rows")
        sub.set_defaults(func=_pull)
        return sub

    fred = add("fred", "FRED series via the keyless fredgraph.csv (the JSON API if $FRED_API_KEY is set)")
    fred.add_argument(
        "--ids",
        type=_fred_ids,
        required=True,
        help="comma-separated series ids, e.g. DGS2,DGS10; 'ust', 'ice' and 'fixings' expand to their series",
    )
    fred.add_argument("--start", type=_date)
    fred.add_argument("--end", type=_date)
    fred.add_argument("--units", type=_list, help="units of unknown series, e.g. MYSERIES=percent")
    fred.add_argument("--vintage", type=_date, help="data as known on this date (ALFRED; needs $FRED_API_KEY)")
    treasury = add("treasury", "US Treasury daily par yield curve CSV for one year (slow endpoint)")
    treasury.add_argument("--year", type=int, default=dt.date.today().year)
    nyfed = add("nyfed", "NY Fed reference rates: sofr, effr, obfr, tgcr, bgcr or sofrai (averages and index)")
    nyfed.add_argument("--rate", choices=sorted(rs.NYFED_RATES), default="sofr")
    nyfed.add_argument("--last", type=int, help="the last N publications (default 30)")
    nyfed.add_argument("--start", type=_date)
    nyfed.add_argument("--end", type=_date)
    estr = add("ecb-estr", "euro short-term rate (ECB Data Portal)")
    estr.add_argument("--last", type=int)
    estr.add_argument("--start", type=_date)
    estr.add_argument("--end", type=_date)
    yc = add("ecb-yc", "euro area government yield curve (ECB, Svensson fit)")
    yc.add_argument("--curve", choices=sorted(rs.ECB_CURVES), default="AAA")
    yc.add_argument("--measure", choices=sorted(rs.ECB_MEASURES), default="par")
    yc.add_argument("--tenors", type=_list, default=list(rs.ECB_TENORS))
    yc.add_argument("--last", type=int)
    yc.add_argument("--start", type=_date)
    yc.add_argument("--end", type=_date)
    boe = add("boe-ois", "Bank of England fitted UK OIS (SONIA) spot curve")
    boe.add_argument("--archive", choices=sorted(rs.BOE_ARCHIVES), default="latest")
    boe.add_argument("--start", type=_date)
    boe.add_argument("--end", type=_date)
    add("latest", "every fast live source the app's LIVE (CACHED) mode uses")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="derivx", description="DerivX FIC analytics command line")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, register in COMMANDS.items():
        sub = commands.add_parser(name, help=(register.__doc__ or name).strip().splitlines()[0])
        register(sub)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return int(args.func(args))
    except ValueError as exc:  # bad combinations of arguments the parser cannot see
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
