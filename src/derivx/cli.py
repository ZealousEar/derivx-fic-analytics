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

Curves, books and scenarios (no network: the committed history sample plus
whatever ``pull`` cached)::

    derivx curve build --quotes q.csv --asof 2026-10-19 --interp loglinear --out curve.json
    derivx curve build --asof 2026-10-19 --conventions "EUR ESTR" --out eur.json   # the app's quotes
    derivx curve show --curve curve.json --nodes nodes.csv
    derivx book example --currency USD --out trades.csv
    derivx risk --trades trades.csv --curve curve.json --illustrative-vols --out risk.csv --manifest run.json
    derivx risk --trades trades.csv --asof 2026-10-19 --hedge 2Y,5Y,10Y,30Y --bid-ask 0.25 --move 5
    derivx scenario --trades trades.csv --asof 2026-10-19 --shock parallel:+25 --shock historical:2022-hiking-cycle

``curve build`` writes the curve set as JSON (quotes, nodes and a run
manifest); ``risk`` and ``scenario`` rebuild it from its quotes and refuse a
file whose quotes no longer give its nodes. Exit codes: 0 ok, 1 a data
source failed, 2 bad arguments, 3 some blotter rows or trades were left out
(the rest were priced).

Each subcommand registers itself with :func:`command`.
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


def _units(text: str) -> Dict[str, str]:
    """``ID=unit[,ID=unit]`` -> ``{ID: unit}`` (units as in
    :data:`derivx.data.rates_sources.UNIT_SCALE`, or ``index``)."""

    allowed = sorted(rs.UNIT_SCALE) + ["index"]
    out: Dict[str, str] = {}
    for part in _list(text):
        sid, sep, unit = part.partition("=")
        if not sep or not sid.strip() or unit.strip() not in allowed:
            raise argparse.ArgumentTypeError(f"{part!r}: use SERIES=UNIT with UNIT one of {', '.join(allowed)}")
        out[sid.strip().upper()] = unit.strip()
    return out


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
        return rs.fred_series(args.ids, args.start, args.end, units=args.units, vintage=args.vintage, **common)
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
    fred.add_argument("--units", type=_units, help="units of unknown series, e.g. MYSERIES=percent,OTHER=bp")
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
# Market inputs shared by curve, risk and scenario
# --------------------------------------------------------------------------


def _store(cache: Optional[Path]):
    """The committed history sample plus whatever ``derivx pull`` cached (no
    network)."""

    from derivx.data import rates_history as rh

    frames = [rh.bundled_frame()] + rh.cached_frames(DiskCache(cache) if cache else None)
    return rh.build_store(frames, "bundled snapshot + download cache")


def _hw(text: str):
    from derivx.rates.indices import HullWhite

    parts = _list(text)
    try:
        a, sigma = (float(p) for p in parts)
        return HullWhite(a / 100.0, sigma / 1e4)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: give MEAN_REVERSION_PCT,SIGMA_BP (e.g. 3,85): {exc}") from None


def _market_options(parser: argparse.ArgumentParser, curve_file: bool = True) -> None:
    group = parser.add_argument_group("market")
    if curve_file:
        group.add_argument("--curve", type=Path, help="curve set JSON from 'derivx curve build'")
    group.add_argument("--asof", type=_date, help="valuation date: build the market the app starts from that day")
    group.add_argument("--conventions", default="USD SOFR", choices=["USD SOFR", "EUR ESTR", "GBP SONIA"],
                       help="swap curve (with --asof)")  # fmt: skip
    group.add_argument("--interp", default="loglinear", help="loglinear, linear_zero or monotone_convex (with --asof)")
    group.add_argument("--hw", type=_hw, help="futures convexity: Hull-White MEAN_REVERSION_PCT,SIGMA_BP (3,85)")
    group.add_argument("--cache", type=Path, help="download cache directory (default data/cache or $DERIVX_CACHE_DIR)")


def _market(args: argparse.Namespace):
    """(market, {curve: source}, data record, notes) from --curve or --asof."""

    from derivx.rates import curve_io
    from derivx.rates.manifest import market_record, sha256_bytes

    if getattr(args, "curve", None) is not None:
        if args.asof is not None:
            raise ValueError("give --curve or --asof, not both")
        market, sources, manifest = curve_io.read_curve_set(args.curve)
        data = market_record(market, sources)
        data["curve_file"] = {"path": str(args.curve), "sha256": sha256_bytes(args.curve.read_bytes()),
                              "built": manifest.get("created_at_utc"), "data": manifest.get("data", {})}  # fmt: skip
        return market, sources, data, []
    if args.asof is None:
        raise ValueError("give the market: --curve curve.json or --asof YYYY-MM-DD")
    store = _store(args.cache)
    market, sources, notes = curve_io.default_market(store, args.asof, args.conventions, args.interp,
                                                     args.hw or curve_io.HW_DEFAULT)  # fmt: skip
    return market, sources, market_record(market, sources, store), notes


def _print_curves(market, sources) -> None:
    for key in market.curve_keys:
        if key not in ("curve", "govt") and market.projections.is_discount(key):
            continue
        curve = market.get_curve(key)
        worst = max((abs(r["error_bp"]) for r in curve.repricing()), default=0.0)
        print(
            f"  {key:<11} {len(curve.quotes):>3} quotes  max repricing error {worst:.1e}bp  2Y zero "
            f"{100 * curve.zero_rate(2.0):.4f}%  10Y zero {100 * curve.zero_rate(10.0):.4f}%  | "
            f"{sources.get(key, '')}"
        )


def _write_table(frame: pd.DataFrame, path: Optional[Path], what: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {what} ({len(frame)} rows) to {path}")


def _write_manifest(manifest, path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(manifest.to_json())
    print(f"wrote the run manifest to {path}")


# --------------------------------------------------------------------------
# curve
# --------------------------------------------------------------------------


def _curve_build(args: argparse.Namespace) -> int:
    from derivx.rates import curve_io
    from derivx.rates.curves import CONVENTIONS
    from derivx.rates.exports import market_tables
    from derivx.rates.manifest import market_record, run_manifest, sha256_bytes

    store = _store(args.cache)
    hw = args.hw or curve_io.HW_DEFAULT
    inputs = {"asof": args.asof, "conventions": args.conventions, "interpolation": args.interp, "units": args.units,
              "hull_white": {"mean_reversion": hw.mean_reversion, "sigma": hw.sigma}}  # fmt: skip
    if args.quotes is None:
        market, sources, notes = curve_io.default_market(store, args.asof, args.conventions, args.interp, hw)
        inputs["quotes"] = "the app's starting quotes for the date (see the sources)"
    else:
        payload = args.quotes.read_bytes()
        quotes, errors, notes = curve_io.read_quotes(payload, args.conventions, args.asof, args.units)
        if errors:
            for error in errors:
                print(f"error: {error}", file=sys.stderr)
            return 2
        sources = {key: f"{args.quotes.name}" for key in quotes}
        if "govt" not in quotes and args.treasury:
            try:
                quotes["govt"] = curve_io.treasury_curve_quotes(store, args.asof)
            except ValueError as exc:
                raise ValueError(f"{exc}: add TREASURY rows to the quotes file or pass --no-treasury") from None
            sources["govt"] = f"US Treasury par yields ({store.origin('UST_PAR', args.asof)})"
        currency = CONVENTIONS[args.conventions].currency
        market = curve_io.build_market(args.asof, quotes, args.conventions, args.interp, hw,
                                       curve_io.ois_fixings(store, currency, args.asof))  # fmt: skip
        inputs["quotes"] = {"path": str(args.quotes), "sha256": sha256_bytes(payload)}
    for note in notes:
        print(f"  note: {note}")
    manifest = run_manifest(
        "curve build", inputs, market_record(market, sources, store),
        {"bootstrap": "every quote reprices exactly; curve time ACT/365F", "interpolation": args.interp},
    )  # fmt: skip
    print(f"curve set on {market.asof}:")
    _print_curves(market, sources)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        curve_io.write_curve_set(args.out, market, sources, manifest.to_dict())
        print(f"wrote the curve set to {args.out}")
    tables = market_tables(market)
    _write_table(tables["curve_nodes"], args.nodes, "curve nodes")
    _write_table(tables["curve_grid"], args.grid, "curve grid (DFs, zero rates, forwards)")
    return 0


def _curve_show(args: argparse.Namespace) -> int:
    from derivx.rates import curve_io
    from derivx.rates.exports import market_tables

    market, sources, manifest = curve_io.read_curve_set(args.curve)
    code = manifest.get("code", {})
    print(f"curve set on {market.asof} (built {manifest.get('created_at_utc', '?')}, code {code.get('sha', '?')}); "
          "rebuilt from its quotes and checked against its nodes:")  # fmt: skip
    _print_curves(market, sources)
    tables = market_tables(market)
    _write_table(tables["curve_nodes"], args.nodes, "curve nodes")
    _write_table(tables["curve_grid"], args.grid, "curve grid (DFs, zero rates, forwards)")
    return 0


@command("curve")
def _register_curve(parser: argparse.ArgumentParser) -> None:
    """bootstrap a curve set from quotes and save it as JSON, or show one"""

    actions = parser.add_subparsers(dest="action", required=True, metavar="ACTION")
    build = actions.add_parser("build", help="bootstrap from a quotes CSV (or the app's quotes for --asof)",
                               description="Bootstrap the swap curve, any EURIBOR projection curves and the Treasury "
                               "curve. Quotes CSV: [curve,] instrument, tenor, rate.")  # fmt: skip
    build.add_argument("--quotes", type=Path, help="quotes CSV (default: the app's starting quotes for the date)")
    build.add_argument("--asof", type=_date, required=True, help="valuation date YYYY-MM-DD")
    build.add_argument("--conventions", default="USD SOFR", choices=["USD SOFR", "EUR ESTR", "GBP SONIA"])
    build.add_argument("--interp", default="loglinear", help="loglinear, linear_zero or monotone_convex")
    build.add_argument("--units", default="auto", choices=["auto", "percent", "decimal", "bp"],
                       help="rate units of the quotes file (auto reads them from the magnitudes)")  # fmt: skip
    build.add_argument(
        "--treasury",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add the US Treasury par curve for the date from the history store (default on)",
    )
    build.add_argument("--hw", type=_hw, help="futures convexity: Hull-White MEAN_REVERSION_PCT,SIGMA_BP (3,85)")
    build.add_argument("--cache", type=Path, help="download cache directory")
    build.add_argument("--out", type=Path, help="write the curve set JSON here")
    build.add_argument("--nodes", type=Path, help="write the curve nodes CSV (pillars, DFs, zero rates)")
    build.add_argument("--grid", type=Path, help="write a dated grid CSV (DFs, zero rates, 3M forwards)")
    build.set_defaults(func=_curve_build)
    show = actions.add_parser("show", help="load a curve set JSON, check it and export its nodes")
    show.add_argument("--curve", type=Path, required=True)
    show.add_argument("--nodes", type=Path)
    show.add_argument("--grid", type=Path)
    show.set_defaults(func=_curve_show)


# --------------------------------------------------------------------------
# book, risk and scenario
# --------------------------------------------------------------------------

EXIT_TRADES = 3  # some blotter rows had errors or could not be priced (the rest were)


def _trades(path: Path):
    from derivx.rates import portfolio as pf

    table, read_issues = pf.read_blotter_csv(path.read_bytes())
    trades, issues = pf.parse_blotter(table)
    issues = read_issues + issues
    for issue in issues:
        stream = sys.stderr if issue.level == "error" else sys.stdout
        print(f"{issue.level}: {path.name} {issue}", file=stream)
    return trades, any(i.level == "error" for i in issues)


def _vol_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("vols for trades without a flat vol")
    group.add_argument("--swaption-vols", type=Path, help="swaption normal vol grid CSV (bp; wide or long, as the app)")
    group.add_argument("--cap-vols", type=Path, help="flat ATM normal cap vols CSV: maturity, vol_bp")
    group.add_argument("--illustrative-vols", action="store_true",
                       help="use the built-in ILLUSTRATIVE swaption and cap quotes (not market data)")  # fmt: skip


def _vols(args: argparse.Namespace, market, trades):
    from derivx.rates import portfolio as pf
    from derivx.rates.swaption_vol import parse_vol_grid
    from derivx.rates.vol_examples import illustrative_cap_atm, illustrative_swaption_grid

    currency = market.curve.conventions.currency if market.curve.conventions else "USD"
    grid = cap_quotes = None
    cap_source = ""
    inputs = {}
    if args.swaption_vols is not None:
        raw = pd.read_csv(args.swaption_vols, dtype=str, keep_default_na=False, na_values=[""], sep=None,
                          engine="python")  # fmt: skip
        grid, issues = parse_vol_grid(raw, "normal", "bp", source=f"uploaded: {args.swaption_vols.name}")
        errors = [i.message for i in issues if i.level == "error"]
        if errors:
            raise ValueError(f"{args.swaption_vols}: " + "; ".join(errors))
        inputs["swaption_vols"] = str(args.swaption_vols)
    elif args.illustrative_vols:
        grid = illustrative_swaption_grid(currency)
        inputs["swaption_vols"] = grid.source
    if args.cap_vols is not None:
        cap_quotes = pd.read_csv(args.cap_vols)
        missing = {"maturity", "vol_bp"} - set(cap_quotes.columns)
        if missing:
            raise ValueError(f"{args.cap_vols}: needs columns maturity and vol_bp (missing {', '.join(missing)})")
        cap_source = f"uploaded: {args.cap_vols.name}"
        inputs["cap_vols"] = str(args.cap_vols)
    elif args.illustrative_vols:
        cap_quotes, cap_source = illustrative_cap_atm(currency), f"ILLUSTRATIVE {currency} flat ATM cap normal vols"
        inputs["cap_vols"] = cap_source
    hw = args.hw or None
    vols, notes = pf.build_vols(market, trades, grid, cap_quotes, cap_source, hull_white=hw)
    for note in notes:
        print(f"  note: {note}")
    return vols, inputs


def _skipped(risk) -> None:
    for trade, reason in risk.skipped:
        hint = ""
        if "vol cube" in reason or "caplet vols" in reason:
            hint = " (or pass --swaption-vols / --cap-vols / --illustrative-vols)"
        print(f"skipped: {trade.id} ({trade.type}): {reason}{hint}", file=sys.stderr)


def _money(value: float) -> str:
    return f"{round(value, 2) + 0.0:,.2f}"


def _risk(args: argparse.Namespace) -> int:
    from derivx.rates import exports
    from derivx.rates import portfolio as pf
    from derivx.rates.manifest import run_manifest, sha256_bytes

    trades, bad_rows = _trades(args.trades)
    market, sources, data, notes = _market(args)
    for note in notes:
        print(f"  note: {note}")
    vols, vol_inputs = _vols(args, market, trades)
    risk = pf.price_book(trades, market, vols)
    _skipped(risk)
    summary = risk.summary()
    shown = summary.copy()
    for column in ("pv", "pv01", "gamma", "vega"):
        shown[column] = shown[column].map(_money)
    print(f"book on {market.asof}: {len(risk.trades)} trades priced, {len(risk.skipped)} skipped; values in each "
          "trade's currency, signed to the holder; PV01 and gamma for +1bp on every rate quote of every curve, vega "
          "per +1bp normal vol")  # fmt: skip
    print(shown.to_string(index=False))
    totals = risk.ladder_totals()
    if args.ladder_all:
        print(totals.to_string(index=False))
    else:
        loaded = totals[totals["pv01"].abs() > 0.005]
        print("key-rate PV01 of the book (P&L per +1bp of one quote; buckets under half a cent left out):")
        print(loaded.assign(pv01=loaded["pv01"].map(_money)).drop(columns=["gamma"]).to_string(index=False))
    hedge = None
    if args.hedge:
        bench = {key: list(args.hedge) for key in pf.default_benchmarks(risk, market)}
        costs = {k: args.bid_ask for k in pf.assumed_bid_ask(market, bench)} if args.bid_ask is not None else {}
        instruments = pf.hedge_instruments(market, bench, bid_ask_bp=costs)
        hedge = pf.hedge_book(risk, market, instruments, args.move)
        table = hedge.table[hedge.table["notional"] != 0.0]
        print(f"hedge: residual risk {_money(hedge.risk_after)} vs {_money(hedge.risk_before)} per bp (root sum of "
              f"squares of the key-rate ladder, {100 * hedge.risk_removed:.2f}% of its square removed), bid/ask cost "
              f"{_money(hedge.cost)}")  # fmt: skip
        print(table.to_string(index=False))
    report = exports.risk_report(risk)
    _write_table(report, args.out, "risk report")
    _write_table(risk.ladder(), args.ladder, "key-rate ladder by trade")
    if hedge is not None:
        _write_table(hedge.ladder, args.hedge_out, "hedge ladder")
    manifest = run_manifest(
        "book risk",
        {"trades": {"path": str(args.trades), "sha256": sha256_bytes(args.trades.read_bytes()),
                    "rows_valid": len(trades)}, "vols": vol_inputs,
         "hedge": None if hedge is None else {"tenors": list(args.hedge), "move_bp": args.move,
                                              "bid_ask_bp": args.bid_ask}},
        data,
        {"bump_bp": pf.BP, "risk": "central differences on full revaluation, curves rebuilt",
         "vol_sources": dict(vols.sources)},
    )  # fmt: skip
    _write_manifest(manifest, args.manifest)
    return EXIT_TRADES if bad_rows or risk.skipped else 0


def _shock_years(text: str) -> float:
    from derivx.rates.conventions import parse_tenor

    try:
        return float(text)
    except ValueError:
        return parse_tenor(text).years


def parse_shock(spec: str, market, store) -> List[object]:
    """One ``--shock`` spec -> shocks (see ``derivx scenario --help``)."""

    from derivx.risk import scenarios as sc

    kind, _, rest = spec.strip().partition(":")
    kind = kind.strip().lower()
    parts = [p.strip() for p in rest.split(":")] if rest else []
    try:
        if kind == "parallel":
            return [sc.parallel(float(parts[0]))]
        if kind == "twist":
            pivot, short_bp, long_bp = _shock_years(parts[0]), float(parts[1]), float(parts[2])
            return [sc.twist(pivot, short_bp, long_bp)]
        if kind in ("fly", "butterfly"):
            body, body_bp = _shock_years(parts[0]), float(parts[1])
            wing_bp = float(parts[2]) if len(parts) > 2 else 0.0
            wings = (_shock_years(parts[3]), _shock_years(parts[4])) if len(parts) > 4 else (2.0, 10.0)
            return [sc.butterfly(wing_bp, body_bp, body, wings=wings)]
        if kind in ("keyrate", "keyrates"):
            moves = {}
            for item in _list(rest):
                tenor, sep, bp = item.partition("=")
                if not sep:
                    raise ValueError(f"{item!r}: write TENOR=BP, e.g. 2Y=+10")
                moves[tenor.strip().upper()] = float(bp)
            return [sc.key_rates(moves)]
        if kind == "pca":
            factor, n_sigma = int(parts[0]), float(parts[1]) if len(parts) > 1 else 2.0
            days = int(parts[2]) if len(parts) > 2 else 1
            return [sc.pca_factor(factor, n_sigma, store, end=market.asof, days=days)]
        if kind == "historical":
            wanted = parts[0].lower().replace("-", " ").replace("_", " ")
            names = {n.lower(): n for n in sc.historical_episodes()}
            if wanted not in names:
                raise ValueError(f"unknown episode {parts[0]!r}; one of: {'; '.join(sc.historical_episodes())}")
            return [sc.historical(names[wanted], float(parts[1]) if len(parts) > 1 else 1.0, store=store)]
        if kind == "window":
            start, end = dt.date.fromisoformat(parts[0]), dt.date.fromisoformat(parts[1])
            return [sc.history_window(start, end, float(parts[2]) if len(parts) > 2 else 1.0, store=store)]
        if kind == "basis":
            return [sc.ibor_ois_basis(float(parts[0]))]
        if kind == "tenorbasis":
            return [sc.tenor_basis(float(parts[0]))]
        if kind == "library":
            shocks = sc.standard_library(store)
            if market.is_multi_curve:
                shocks += [sc.ibor_ois_basis(10.0), sc.tenor_basis(5.0)]
            return shocks + sc.pca_library(store, end=market.asof)
    except (IndexError, ValueError) as exc:
        detail = str(exc) if isinstance(exc, ValueError) else "missing a field"
        raise ValueError(f"--shock {spec!r}: {detail} (see derivx scenario --help)") from None
    raise ValueError(f"--shock {spec!r}: unknown shock type {kind!r} (see derivx scenario --help)")


SHOCK_HELP = """shock specs (bp unless stated; tenors as years or 2Y/10Y):
  parallel:BP                         every quote BP
  twist:PIVOT:SHORT_BP:LONG_BP        zero at PIVOT, SHORT_BP at the shortest quote, LONG_BP at the longest
  fly:BODY:BODY_BP[:WING_BP[:LEFT:RIGHT]]   butterfly (wings 2Y and 10Y by default)
  keyrate:2Y=+10,10Y=-5               key-rate moves, linear in between, flat beyond
  pca:FACTOR[:N_SIGMA[:DAYS]]         PC1 level / PC2 slope / PC3 curvature of daily US Treasury changes
  historical:NAME[:SCALE]             a named episode, e.g. historical:2022-hiking-cycle
  window:START:END[:SCALE]            the curve changes between two dates of the history store
  basis:BP / tenorbasis:BP            IBOR-OIS basis / tenor basis spreads (multi-curve markets)
  library                             the app's scenario library (parallel, twists, fly, episodes, PCA)"""


def _scenario(args: argparse.Namespace) -> int:
    from dataclasses import replace as dc_replace

    from derivx.rates import portfolio as pf
    from derivx.rates.manifest import run_manifest, sha256_bytes
    from derivx.risk import scenarios as sc

    trades, bad_rows = _trades(args.trades)
    market, sources, data, notes = _market(args)
    for note in notes:
        print(f"  note: {note}")
    store = _store(args.cache)
    shocks = [s for spec in args.shock for s in parse_shock(spec, market, store)]
    if not shocks:
        raise ValueError("give at least one --shock (see derivx scenario --help)")
    settings = {"horizon_days": args.horizon, "normal_vol_bp": args.vol_normal, "lognormal_vol_pts": args.vol_lognormal}
    if market.govt is None:
        settings["govt"] = False
    shocks = [dc_replace(s, **settings) for s in shocks]
    vols, vol_inputs = _vols(args, market, trades)
    risk = pf.price_book(trades, market, vols, vega=False)
    _skipped(risk)
    positions = list(risk.positions)
    if not positions:
        raise ValueError("no trade could be priced")
    rows, failed = [], []
    for shock in shocks:
        try:
            rows.append(sc.scenario_grid(positions, market, [shock]))
        except ValueError as exc:
            failed.append(f"{shock.name}: {exc}")
    for reason in failed:
        print(f"not revalued: {reason}", file=sys.stderr)
    grid = pd.concat(rows) if rows else pd.DataFrame()
    if not grid.empty:
        print(f"full-revaluation P&L on {market.asof} (value at the horizon + cash - value today, each trade's "
              f"currency, signed to the holder; horizon {args.horizon} days):")  # fmt: skip
        print(grid.map(_money).to_string())
    _write_table(grid.rename_axis("scenario").reset_index(), args.out, "scenario grid")
    if args.explain is not None:
        explain = []
        for shock in shocks:
            try:
                for row in sc.run_scenario(positions, market, shock):
                    by_curve = row.pop("delta_by_curve")
                    explain.append({"scenario": shock.name, **row, **{f"delta_{k}": v for k, v in by_curve.items()}})
            except ValueError as exc:
                print(f"no explain for {shock.name}: {exc}", file=sys.stderr)
        _write_table(pd.DataFrame(explain), args.explain, "P&L explain")
    manifest = run_manifest(
        "scenario grid",
        {"trades": {"path": str(args.trades), "sha256": sha256_bytes(args.trades.read_bytes())},
         "shocks": list(args.shock), "horizon_days": args.horizon, "normal_vol_bp": args.vol_normal,
         "lognormal_vol_pts": args.vol_lognormal, "vols": vol_inputs},
        {**data, "scenario_sources": [{"shock": s.name, "sources": [vars(x) for x in s.sources]} for s in shocks]},
        {"revaluation": "quotes shocked, curves rebuilt, every trade revalued", "vol_sources": dict(vols.sources)},
    )  # fmt: skip
    _write_manifest(manifest, args.manifest)
    return EXIT_TRADES if bad_rows or risk.skipped or failed else 0


def _book_example(args: argparse.Namespace) -> int:
    from derivx.rates import portfolio as pf

    table = pf.example_blotter(args.currency)
    if args.out is None:
        print(table.to_csv(index=False), end="")
    else:
        _write_table(table, args.out, f"EXAMPLE {args.currency} book (illustrative trades)")
    return 0


def _book_check(args: argparse.Namespace) -> int:
    trades, bad_rows = _trades(args.trades)
    print(f"{len(trades)} valid trades" + (" (rows with errors left out)" if bad_rows else ""))
    return EXIT_TRADES if bad_rows else 0


@command("book")
def _register_book(parser: argparse.ArgumentParser) -> None:
    """write an example trade blotter or validate one"""

    actions = parser.add_subparsers(dest="action", required=True, metavar="ACTION")
    example = actions.add_parser("example", help="an EXAMPLE book of illustrative trades as CSV")
    example.add_argument("--currency", default="USD", choices=["USD", "EUR", "GBP"])
    example.add_argument("--out", type=Path, help="write it here (default: print)")
    example.set_defaults(func=_book_example)
    check = actions.add_parser("check", help="validate a blotter CSV and list its problems")
    check.add_argument("--trades", type=Path, required=True)
    check.set_defaults(func=_book_check)


@command("risk")
def _register_risk(parser: argparse.ArgumentParser) -> None:
    """price a trade blotter: PV, PV01, key-rate ladders, vega and hedges"""

    parser.add_argument("--trades", type=Path, required=True, help="blotter CSV (see derivx book example)")
    _market_options(parser)
    _vol_options(parser)
    parser.add_argument("--out", type=Path, help="write the long risk report CSV (PV, PV01, key rates, vega)")
    parser.add_argument("--ladder", type=Path, help="write the key-rate ladder by trade CSV")
    parser.add_argument("--ladder-all", action="store_true", help="print every key-rate bucket, zeros included")
    parser.add_argument("--hedge", type=_list, help="hedge with par instruments at these tenors, e.g. 2Y,5Y,10Y,30Y")
    parser.add_argument("--move", type=float, help="with --bid-ask: hedge against curve moves of this size (bp)")
    parser.add_argument("--bid-ask", type=float, help="bid/ask (bp of the hedge's quote) of every hedge")
    parser.add_argument("--hedge-out", type=Path, help="write the book / hedge / residual ladder CSV")
    parser.add_argument("--manifest", type=Path, help="write the run manifest JSON")
    parser.set_defaults(func=_risk)


@command("scenario")
def _register_scenario(parser: argparse.ArgumentParser) -> None:
    """full-revaluation scenario P&L of a trade blotter"""

    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = SHOCK_HELP
    parser.add_argument("--trades", type=Path, required=True, help="blotter CSV (see derivx book example)")
    parser.add_argument("--shock", action="append", default=[], help="a shock spec (repeat for more); see below")
    parser.add_argument("--horizon", type=int, default=0, help="calendar days to roll the market forward first")
    parser.add_argument("--vol-normal", type=float, default=0.0, help="normal vol shock (bp)")
    parser.add_argument("--vol-lognormal", type=float, default=0.0, help="lognormal vol shock (vol points)")
    _market_options(parser)
    _vol_options(parser)
    parser.add_argument("--out", type=Path, help="write the scenario grid CSV (scenarios x trades)")
    parser.add_argument("--explain", type=Path, help="write the P&L explain CSV (delta, gamma, vol, carry, residual)")
    parser.add_argument("--manifest", type=Path, help="write the run manifest JSON")
    parser.set_defaults(func=_scenario)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="derivx", description="DerivX FIC analytics command line")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, register in COMMANDS.items():
        summary = (register.__doc__ or name).strip().splitlines()[0]
        sub = commands.add_parser(name, help=summary, description=summary)
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
