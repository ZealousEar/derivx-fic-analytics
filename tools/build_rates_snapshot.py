"""Refresh the committed rates snapshot and history sample (data/examples).

Downloads, with the fetchers in ``derivx.data.rates_sources``:

* ``ust_par_yield_curve_<year>.csv`` - the Treasury's daily par yield curve
  CSV for the current year, as published (the offline default);
* ``rates_history/*.csv`` - small wide samples, values as published:
  month-end curves since 2008/2009, every business day of the current year
  where noted, and every curve within four days of each named scenario
  episode's start and end (so episodes replay offline exactly as they would
  on the full history). The US Treasury par curve (FRED H.15) also keeps
  every business day since 2015-01-02: the daily changes the PCA, RV and
  VaR views of the RESEARCH tab use offline;
* ``rates_history/manifest.json`` - per file: dataset, source, units,
  measure, tenor of every column, sampling, URLs, fetch time, SHA-256 and
  attribution.

    python tools/build_rates_snapshot.py            # everything
    python tools/build_rates_snapshot.py --skip-treasury
    python tools/build_rates_snapshot.py --only ust_par_fred.csv   # one file; the manifest keeps the others

Needs network access; nothing is written for a source that fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from derivx.data import rates_sources as rs  # noqa: E402
from derivx.data.cache import DiskCache, utc_now  # noqa: E402
from derivx.data.rates_history import DATASETS, EXAMPLES_DIR, MANIFEST, SAMPLE_DIR, file_sha256  # noqa: E402
from derivx.risk.scenarios import EPISODES  # noqa: E402

WINDOW_DAYS = 4
UST_FROM, ECB_FROM = dt.date(2008, 1, 1), dt.date(2008, 1, 1)
UST_DAILY_FROM = dt.date(2015, 1, 1)  # every business day from here (PCA, RV and VaR need daily changes)
FILES = ("ust_par_fred.csv", "usd_swap_ice.csv", "gbp_ois_boe.csv", "eur_aaa_ecb.csv", "eur_aaa_ecb_spot.csv",
         "fixings.csv")  # fmt: skip
BOE_TENORS = tuple(f"{m}M" for m in range(1, 12)) + ("1Y", "18M") + tuple(f"{y}Y" for y in range(2, 26))
DAILY_FROM = dt.date(utc_now().year, 1, 1)
RECENT_FROM = utc_now().date() - dt.timedelta(days=62)  # daily curves kept for about two months
DECIMALS = 4  # percent, i.e. 0.01bp
ATTRIBUTION = {
    "FRED": "Board of Governors of the Federal Reserve System (H.15) via FRED, Federal Reserve Bank of St. Louis",
    "ICE": "ICE Benchmark Administration via FRED (ICE swap rates; republished by FRED until 2016-10-28)",
    "BoE": "Source: Bank of England (yield curve archive)",
    "ECB": "Source: ECB Data Portal",
    "NY Fed": "Source: Federal Reserve Bank of New York",
    "Treasury": "U.S. Department of the Treasury, daily Treasury par yield curve rates",
}


def episode_days() -> List[dt.date]:
    out = []
    for _, start, end in EPISODES:
        for anchor in (start, end):
            out += [anchor + dt.timedelta(days=k) for k in range(-WINDOW_DAYS, WINDOW_DAYS + 1)]
    return out


def wide(frame: pd.DataFrame, code_of: Mapping[str, str]) -> pd.DataFrame:
    """Long rows -> date x code table of published values."""

    frame = frame.assign(code=[code_of.get(t, s) for t, s in zip(frame["tenor"], frame["series"])])
    table = frame.pivot_table(index="asof", columns="code", values="value_original", aggfunc="last")
    return table.sort_index()


def sample(table: pd.DataFrame, required: Sequence[str], daily_from: Optional[dt.date] = None) -> pd.DataFrame:
    """Month-end complete rows, plus every complete row in the episode
    windows and (optionally) from ``daily_from`` on."""

    complete = table.dropna(subset=[c for c in required if c in table.columns])
    if complete.empty:
        return complete
    index = pd.Series(complete.index, index=complete.index)
    month_end = index.groupby([d.year * 100 + d.month for d in complete.index]).max()
    keep = set(month_end) | (set(episode_days()) & set(complete.index))
    if daily_from is not None:
        keep |= {d for d in complete.index if d >= daily_from}
    return complete.loc[sorted(keep)]


def write(name: str, table: pd.DataFrame, decimals: Mapping[str, int], entry: Dict[str, object], files: list) -> None:
    out = table.copy()
    for col in out.columns:
        out[col] = out[col].map(lambda v, n=decimals.get(col, 6): "" if pd.isna(v) else f"{v:.{n}f}")
    out.index.name = "date"
    path = SAMPLE_DIR / name
    out.to_csv(path, lineterminator="\n")
    entry.update({"file": name, "rows": int(len(out)), "first": str(out.index[0]), "last": str(out.index[-1])})
    entry["sha256"] = file_sha256(path)
    files.append(entry)
    print(f"wrote {path.relative_to(ROOT)}: {len(out)} rows x {len(out.columns)} columns, {path.stat().st_size} bytes")


def build(args: argparse.Namespace) -> None:
    cache = DiskCache(Path(args.cache)) if args.cache else DiskCache()
    kw = dict(cache=cache, max_age=args.max_age)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    files: list = []
    fetched_at = utc_now().isoformat(timespec="seconds")
    last_year = DAILY_FROM.year - 1
    only = set(args.only or FILES)
    unknown = only - set(FILES)
    if unknown:
        raise SystemExit(f"--only takes {', '.join(FILES)}; got {', '.join(sorted(unknown))}")

    if not args.skip_treasury and not args.only:
        year = DAILY_FROM.year
        raw = rs.http_get(rs.TREASURY_CSV_URL.format(year=year), timeout=rs.TREASURY_TIMEOUT)
        got = rs.parse_treasury_csv(raw.decode("utf-8-sig"))  # validate before overwriting the snapshot
        path = EXAMPLES_DIR / f"ust_par_yield_curve_{year}.csv"
        path.write_bytes(raw.replace(b"\r\n", b"\n"))
        print(f"wrote {path.relative_to(ROOT)}: {got['asof'].nunique()} days to {max(got['asof'])}")

    # US Treasury par yields (FRED H.15 = Treasury CMT) for earlier years: month ends, episode windows and
    # every business day since UST_DAILY_FROM, as published (two decimals, percent).
    if "ust_par_fred.csv" in only:
        ust = rs.fred_series(rs.UST_FRED_IDS, UST_FROM, dt.date(last_year, 12, 31), **kw)
        table = wide(ust.frame, {})
        write(
            "ust_par_fred.csv",
            sample(table, rs.UST_FRED_IDS, UST_DAILY_FROM),
            {sid: 2 for sid in rs.UST_FRED_IDS},
            {"dataset": "UST_PAR", "source": "FRED", "unit": "percent", "measure": "par_yield",
             "series": {sid: tenor for sid, tenor in rs.UST_FRED_TENORS.items()},
             "sampling": f"month ends {UST_FROM.year}-{UST_DAILY_FROM.year - 1}, every business day from "
                         f"{UST_DAILY_FROM.year} to {last_year} (complete curves) and episode windows "
                         f"(+/-{WINDOW_DAYS} days)",
             "urls": [ust.url], "fetched_at": ust.fetched_at, "attribution": ATTRIBUTION["FRED"]},
            files,
        )  # fmt: skip

    # ICE USD swap rates: only the episode windows (the series ends 2016-10-28).
    if "usd_swap_ice.csv" in only:
        ice = rs.fred_series(rs.ICE_SWAP_FRED_IDS, dt.date(2000, 7, 1), rs.ICE_SWAP_LAST_DATE, **kw)
        table = wide(ice.frame, {})
        windows = table.loc[[d for d in table.index if d in set(episode_days())]].dropna()
        write(
            "usd_swap_ice.csv",
            windows,
            {},
            {"dataset": "USD_SWAP_ICE", "source": "FRED", "unit": "percent", "measure": "par_swap",
             "series": {sid: tenor for sid, tenor in rs.ICE_FRED_TENORS.items()},
             "sampling": f"episode windows (+/-{WINDOW_DAYS} days) before 2016-10-28 only; full history via "
                         "'derivx pull fred --ids DSWP1,...'",
             "urls": [ice.url], "fetched_at": ice.fetched_at, "attribution": ATTRIBUTION["ICE"]},
            files,
        )  # fmt: skip

    if only & {"gbp_ois_boe.csv", "eur_aaa_ecb.csv", "eur_aaa_ecb_spot.csv", "fixings.csv"}:
        _build_rest(only, kw, files, fetched_at)
    _write_manifest(files, fetched_at, keep_others=bool(args.only))


def _build_rest(only: set, kw: Mapping[str, object], files: list, fetched_at: str) -> None:
    """BoE, ECB and fixings samples (the files in ``only``)."""

    if "gbp_ois_boe.csv" in only:
        _boe(kw, files)
    if "eur_aaa_ecb.csv" in only:
        _ecb_par(kw, files)
    if "eur_aaa_ecb_spot.csv" in only:
        _ecb_spot(kw, files)
    if "fixings.csv" in only:
        _fixings(kw, files, fetched_at)


def _boe(kw: Mapping[str, object], files: list) -> None:
    # Bank of England OIS spot curve: month ends, episode windows, current year daily.
    month_end = rs.boe_ois_curve("month_end", **kw)
    daily = rs.boe_ois_curve("daily", **kw)
    latest = rs.boe_ois_curve("latest", **kw)
    frames = pd.concat([month_end.frame, daily.frame, latest.frame]).drop_duplicates(["asof", "tenor"], keep="last")
    frames = frames[frames["tenor"].isin(BOE_TENORS)]
    codes = {t: f"BoE OIS spot {t}" for t in BOE_TENORS}
    table = wide(frames, codes)[[codes[t] for t in BOE_TENORS if codes[t] in set(frames["series"])]]
    core = [codes[t] for t in DATASETS["GBP_OIS_BOE"].core_tenors]
    write(
        "gbp_ois_boe.csv",
        sample(table, core, RECENT_FROM),
        {c: DECIMALS for c in table.columns},
        {"dataset": "GBP_OIS_BOE", "source": "BoE", "unit": "percent", "measure": "zero_cc",
         "series": {codes[t]: t for t in BOE_TENORS},
         "sampling": f"month ends since 2009, episode windows (+/-{WINDOW_DAYS} days), every day from "
                     f"{RECENT_FROM}; maturities 1-11M, 1Y, 18M, 2Y-25Y (to 5Y only before 2016); "
                     f"rounded to 1e-{DECIMALS} percent (0.01bp)",
         "urls": [month_end.url, daily.url, latest.url], "fetched_at": latest.fetched_at,
         "attribution": ATTRIBUTION["BoE"]},
        files,
    )  # fmt: skip


def _ecb_par(kw: Mapping[str, object], files: list) -> None:
    # ECB euro area AAA par yields.
    ecb = rs.ecb_yield_curve(rs.ECB_TENORS, "AAA", "par", start=ECB_FROM, **kw)
    codes = {t: s for t, s in zip(ecb.frame["tenor"], ecb.frame["series"])}
    table = wide(ecb.frame, {})
    order = [codes[t] for t in rs.ECB_TENORS]
    write(
        "eur_aaa_ecb.csv",
        sample(table[order], order, RECENT_FROM),
        {c: DECIMALS for c in order},
        {"dataset": "EUR_AAA_ECB", "source": "ECB", "unit": "percent", "measure": "par_yield_cc",
         "series": {codes[t]: t for t in rs.ECB_TENORS},
         "sampling": f"month ends since {ECB_FROM.year}, episode windows (+/-{WINDOW_DAYS} days), every day from "
                     f"{RECENT_FROM}; rounded to 1e-{DECIMALS} percent (0.01bp)",
         "urls": [ecb.url], "fetched_at": ecb.fetched_at, "attribution": ATTRIBUTION["ECB"]},
        files,
    )  # fmt: skip


def _ecb_spot(kw: Mapping[str, object], files: list) -> None:
    # ECB AAA spot (zero) curve: the EUR proxy quotes are derived from it. Month ends at the core
    # maturities, every maturity (3M-30Y) on recent days.
    spot = rs.ecb_yield_curve(rs.ECB_SPOT_TENORS, "AAA", "spot", start=RECENT_FROM, **kw)
    spot_core = rs.ecb_yield_curve(rs.ECB_TENORS, "AAA", "spot", start=ECB_FROM, **kw)
    frame = pd.concat([spot_core.frame, spot.frame]).drop_duplicates(["asof", "tenor"], keep="last")
    codes = {t: s for t, s in zip(frame["tenor"], frame["series"])}
    order = [codes[t] for t in rs.ECB_SPOT_TENORS]
    table = wide(frame, {})[order]
    core = [codes[t] for t in rs.ECB_TENORS]
    write(
        "eur_aaa_ecb_spot.csv",
        sample(table, core, RECENT_FROM),
        {c: DECIMALS for c in order},
        {"dataset": "EUR_AAA_ECB", "source": "ECB", "unit": "percent", "measure": "zero_cc",
         "series": {codes[t]: t for t in rs.ECB_SPOT_TENORS},
         "sampling": f"month ends since {ECB_FROM.year} and episode windows at 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, "
                     f"15Y, 20Y, 30Y; every maturity 3M-30Y every day from {RECENT_FROM}; rounded to "
                     f"1e-{DECIMALS} percent (0.01bp)",
         "urls": [spot_core.url, spot.url], "fetched_at": spot.fetched_at, "attribution": ATTRIBUTION["ECB"]},
        files,
    )  # fmt: skip


def _fixings(kw: Mapping[str, object], files: list, fetched_at: str) -> None:
    # Overnight fixings for the current year.
    today = utc_now().date()
    parts = {
        "SOFR": rs.nyfed_rate("sofr", start=DAILY_FROM, end=today, **kw),
        "EFFR": rs.nyfed_rate("effr", start=DAILY_FROM, end=today, **kw),
        "OBFR": rs.nyfed_rate("obfr", start=DAILY_FROM, end=today, **kw),
        "ESTR": rs.ecb_estr(start=DAILY_FROM, **kw),
        "IUDSOIA": rs.fred_series(["IUDSOIA"], DAILY_FROM, **kw),
    }
    table = pd.concat([wide(f.frame, {}) for f in parts.values()], axis=1).sort_index()[list(parts)]
    write(
        "fixings.csv",
        table.dropna(how="all"),
        {"SOFR": 2, "EFFR": 2, "OBFR": 2, "ESTR": 3, "IUDSOIA": 4},
        {"dataset": "FIXINGS", "source": "NY Fed", "unit": "percent", "measure": "overnight_fixing",
         "series": {"SOFR": "ON", "EFFR": "ON", "OBFR": "ON", "ESTR": {"tenor": "ON", "source": "ECB"},
                    "IUDSOIA": {"tenor": "ON", "source": "FRED"}},
         "sampling": f"every publication from {DAILY_FROM}; dated by effective (rate) date",
         "urls": [f.url for f in parts.values()], "fetched_at": fetched_at,
         "attribution": "; ".join([ATTRIBUTION["NY Fed"], ATTRIBUTION["ECB"],
                                   "SONIA: Bank of England via FRED"])},
        files,
    )  # fmt: skip


def _write_manifest(files: list, fetched_at: str, keep_others: bool) -> None:
    """Write the manifest; with ``keep_others`` (``--only``) the entries of
    the files not rebuilt stay as they were, in their order."""

    built = {entry["file"]: entry for entry in files}
    if keep_others and MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text())["files"]
        files = [built.pop(entry["file"], entry) for entry in previous] + list(built.values())
    manifest = {
        "description": (
            "Committed rates history sample for offline use and tests (values as published, in the stated units). "
            "Rebuild with: python tools/build_rates_snapshot.py"
        ),
        "built_at": fetched_at,
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")


def main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-treasury", action="store_true", help="keep the Treasury snapshot as it is")
    parser.add_argument("--cache", help="cache directory (default: data/cache or $DERIVX_CACHE_DIR)")
    parser.add_argument(
        "--max-age", type=float, default=6 * 3600.0, help="reuse cached downloads younger than this many seconds"
    )
    parser.add_argument(
        "--only", nargs="+", metavar="FILE", help=f"rebuild only these sample files ({', '.join(FILES)})"
    )
    build(parser.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
