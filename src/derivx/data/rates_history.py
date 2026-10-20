"""Dated rates history: the committed sample, the download cache and user
uploads merged into one :class:`HistoryStore`.

Datasets (``DATASETS``) are curves of one kind from one publisher:

* ``UST_PAR``      - US Treasury par yield curve (Treasury CSV; FRED H.15
  ``DGS*`` carry the same constant-maturity par yields)
* ``USD_SWAP_ICE`` - ICE USD swap rates vs 3M LIBOR via FRED ``DSWP*``
  (2000-07-03 to 2016-10-28 only)
* ``GBP_OIS_BOE``  - Bank of England fitted UK OIS (SONIA) spot curve,
  continuously compounded; par OIS rates are derived on demand
* ``EUR_AAA_ECB``  - ECB euro area AAA government par yields (Svensson fit,
  continuously compounded)
* ``FIXINGS``      - SOFR, EFFR, OBFR (NY Fed), ESTR (ECB), SONIA (BoE via
  FRED) and the SOFR averages/index
* ``USER_SWAP:<CCY>`` - swap-curve history uploaded by the user

Rows of higher priority win where they overlap (same dataset, measure, date
and tenor): user uploads over downloads over the committed sample.

:meth:`HistoryStore.curve_move` measures a curve's change between two dates
in bp per tenor (nearest complete curves within a few days), which the
scenario engine replays as a historical shock.
"""

from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from derivx.data import rates_sources as rs
from derivx.data.cache import REPO_ROOT, DiskCache
from derivx.rates.conventions import parse_tenor
from derivx.rates.market_quotes import OIS_PAR_TENORS, par_from_zero_rows
from derivx.rates.quotes import RATE_RANGE, UNIT_SCALE, Issue, ambiguous_units, detect_units

EXAMPLES_DIR = REPO_ROOT / "data" / "examples"
SAMPLE_DIR = EXAMPLES_DIR / "rates_history"
MANIFEST = SAMPLE_DIR / "manifest.json"
TREASURY_SNAPSHOT = EXAMPLES_DIR / "ust_par_yield_curve_2026.csv"
ORIGIN_SNAPSHOT = "bundled snapshot"
EPISODE_TOLERANCE_DAYS = 4
PAR_TENORS = OIS_PAR_TENORS + ("30Y",)


@dataclass(frozen=True)
class Dataset:
    """One curve history: its currency, what the numbers are and where they
    come from. ``core_tenors`` must all be present for a curve to count as
    complete on a date."""

    key: str
    title: str
    currency: str
    kind: str  # "govt", "swap", "ois", "fixings" or "user"
    measure: str
    source: str
    core_tenors: Tuple[str, ...] = ()
    note: str = ""
    ois_conventions: str = ""  # zero-curve datasets: conventions their par OIS rates are derived under


DATASETS: Dict[str, Dataset] = {
    "UST_PAR": Dataset(
        "UST_PAR",
        "US Treasury par yield curve",
        "USD",
        "govt",
        "par_yield",
        "US Treasury daily par yield curve / FRED H.15 DGS*",
        ("1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
    ),
    "USD_SWAP_ICE": Dataset(
        "USD_SWAP_ICE",
        "ICE USD swap rates (vs 3M LIBOR)",
        "USD",
        "swap",
        "par_swap",
        "ICE Benchmark Administration via FRED DSWP*",
        ("1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "30Y"),
        "LIBOR swaps, published 2000-07-03 to 2016-10-28; used as the history of USD swap-curve moves",
    ),
    "GBP_OIS_BOE": Dataset(
        "GBP_OIS_BOE",
        "Bank of England UK OIS (SONIA) curve",
        "GBP",
        "ois",
        "zero_cc",
        "Bank of England yield-curve archive",
        ("1M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y"),
        "fitted spot curve (continuously compounded); par OIS rates derived under GBP SONIA conventions",
        "GBP SONIA",
    ),
    "EUR_AAA_ECB": Dataset(
        "EUR_AAA_ECB",
        "ECB euro area AAA government curve",
        "EUR",
        "govt",
        "par_yield_cc",
        "ECB Data Portal (YC, Svensson fit)",
        ("3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"),
        "continuously compounded par yields of AAA-rated euro area central government bonds",
        "EUR ESTR",
    ),
    "FIXINGS": Dataset(
        "FIXINGS",
        "Overnight fixings",
        "",
        "fixings",
        "overnight_fixing",
        "NY Fed (SOFR, EFFR, OBFR), ECB (ESTR), Bank of England via FRED (SONIA)",
    ),
}
CURRENCY_FIXING = {"USD": "SOFR", "EUR": "ESTR", "GBP": "SONIA"}  # the overnight rate each OIS curve compounds
FIXING_NAMES = {"SOFR": "SOFR", "EFFR": "EFFR", "OBFR": "OBFR", "ESTR": "ESTR", "ECBESTRVOLWGTTRMDMNRT": "ESTR",
                "IUDSOIA": "SONIA"}  # fmt: skip


def user_dataset(currency: str) -> str:
    return f"USER_SWAP:{currency.upper()}"


def dataset_info(key: str) -> Dataset:
    if key in DATASETS:
        return DATASETS[key]
    if key.startswith("USER_SWAP:"):
        ccy = key.split(":", 1)[1]
        return Dataset(key, f"Your uploaded {ccy} swap-curve history", ccy, "user", "par_swap", "user upload")
    raise ValueError(f"unknown dataset {key!r}; expected one of {sorted(DATASETS)} or USER_SWAP:<CCY>")


# --------------------------------------------------------------------------
# Committed sample
# --------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> Dict[str, object]:
    return json.loads(MANIFEST.read_text())


def _sample_file_frame(entry: Mapping[str, object]) -> pd.DataFrame:
    """One committed wide CSV (``date`` + one column per series, values as
    published) -> long format, using the manifest entry for units and
    tenors (per-series overrides allowed)."""

    raw = pd.read_csv(SAMPLE_DIR / str(entry["file"]), dtype=str)
    raw["date"] = pd.to_datetime(raw["date"], format="%Y-%m-%d").dt.date
    long = raw.melt(id_vars="date", var_name="series", value_name="value").dropna(subset=["value"])
    specs = {}
    for code, spec in dict(entry["series"]).items():
        specs[code] = {**entry, **(spec if isinstance(spec, Mapping) else {"tenor": spec})}
    unknown = sorted(set(long["series"]) - set(specs))
    if unknown:
        raise ValueError(f"{entry['file']}: columns {unknown} are not described in the manifest")

    def column(name: str) -> pd.Series:
        return long["series"].map({code: spec.get(name) for code, spec in specs.items()})

    return rs.long_frame(
        long["date"],
        column("source").astype(str),
        long["series"],
        column("tenor"),
        long["value"],
        column("unit").astype(str),
        column("measure").astype(str),
        column("dataset").astype(str),
    )


@lru_cache(maxsize=1)
def _bundled() -> pd.DataFrame:
    frames = [rs.parse_treasury_csv(TREASURY_SNAPSHOT.read_text())]
    frames += [_sample_file_frame(entry) for entry in load_manifest()["files"]]
    frame = pd.concat(frames, ignore_index=True)
    frame["origin"] = ORIGIN_SNAPSHOT
    return frame


def bundled_frame() -> pd.DataFrame:
    """The committed snapshot and history sample in long format (no
    network): the Treasury's 2026 daily par curve plus the files listed in
    ``data/examples/rates_history/manifest.json``."""

    return _bundled().copy()


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveMove:
    """A curve's change between two dates: ``change_bp[i]`` at
    ``years[i]`` (tenor ``tenors[i]``, series ``series[i]``), from the
    dataset's curve on ``start`` to the one on ``end`` (the dates actually
    used). ``what`` says which rates moved (par yields, derived par OIS)."""

    dataset: str
    start: dt.date
    end: dt.date
    tenors: Tuple[str, ...]
    years: Tuple[float, ...]
    change_bp: Tuple[float, ...]
    series: Tuple[str, ...]
    what: str

    @property
    def profile(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(zip(self.years, self.change_bp))


class HistoryStore:
    """Read-only view over long-format rows (see :mod:`rates_sources`) with
    an ``origin`` column. Build it with :func:`build_store`."""

    def __init__(self, frame: pd.DataFrame, label: str = ""):
        frame = frame.copy()
        if "origin" not in frame:
            frame["origin"] = ""
        frame = frame[np.isfinite(frame["value_decimal"].astype(float))]
        # One value per curve point, or per named fixing (ESTR from the ECB and via FRED are one series).
        fixings = frame["dataset"] == "FIXINGS"
        frame["item"] = frame["tenor"].astype(object)
        frame.loc[fixings, "item"] = [FIXING_NAMES.get(s, s) for s in frame.loc[fixings, "series"]]
        self.frame = frame.drop_duplicates(["dataset", "measure", "asof", "item"], keep="last").reset_index(drop=True)
        self.label = label
        self._rows: Dict[str, pd.DataFrame] = {k: g for k, g in self.frame.groupby("dataset", sort=False)}
        self._wide: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._complete: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._dates: Dict[Tuple[str, str], List[dt.date]] = {}
        self._moves: Dict[tuple, CurveMove] = {}
        self._par: Dict[Tuple[str, dt.date], Dict[str, float]] = {}

    # -- coverage ----------------------------------------------------------

    def datasets(self) -> List[str]:
        return sorted(self._rows)

    def has(self, dataset: str) -> bool:
        return dataset in self._rows

    def rows(self, dataset: str) -> pd.DataFrame:
        """The long-format rows of one dataset (empty if absent)."""

        return self._rows.get(dataset, self.frame.iloc[:0])

    def wide(self, dataset: str, measure: Optional[str] = None) -> pd.DataFrame:
        """Decimal values: one row per date, one column per tenor (shortest
        first); NaN where a tenor is missing."""

        measure = measure or dataset_info(dataset).measure
        key = (dataset, measure)
        if key not in self._wide:
            rows = self.rows(dataset)
            rows = rows[rows["measure"] == measure]
            if rows.empty:
                self._wide[key] = pd.DataFrame()
            else:
                table = rows.pivot_table(index="asof", columns="tenor", values="value_decimal", aggfunc="last")
                order = sorted(table.columns, key=lambda t: parse_tenor(t).years)
                self._wide[key] = table[order].sort_index()
        return self._wide[key]

    def complete(self, dataset: str, measure: Optional[str] = None) -> pd.DataFrame:
        """Rows of :meth:`wide` with every core tenor present (any tenor for
        a measure other than the dataset's own)."""

        info = dataset_info(dataset)
        measure = measure or info.measure
        key = (dataset, measure)
        if key not in self._complete:
            wide = self.wide(dataset, measure)
            core = list(info.core_tenors) if measure == info.measure else []
            if wide.empty:
                out = wide
            elif core and set(core) <= set(wide.columns):
                out = wide.dropna(subset=core, how="any")
            else:
                out = wide.dropna(how="all")
            self._dates[key] = list(out.index)  # before _complete: _days() reads it once _complete has the key
            self._complete[key] = out
        return self._complete[key]

    def _days(self, dataset: str, measure: Optional[str] = None) -> List[dt.date]:
        measure = measure or dataset_info(dataset).measure
        self.complete(dataset, measure)
        return self._dates[(dataset, measure)]

    def dates(self, dataset: str, measure: Optional[str] = None) -> List[dt.date]:
        """Dates with a complete curve, oldest first."""

        return list(self._days(dataset, measure))

    def coverage(self, dataset: str, measure: Optional[str] = None) -> Optional[Tuple[dt.date, dt.date]]:
        days = self._days(dataset, measure)
        return (days[0], days[-1]) if days else None

    def latest_on_or_before(self, dataset: str, day: dt.date, measure: Optional[str] = None) -> Optional[dt.date]:
        days = self._days(dataset, measure)
        i = bisect.bisect_right(days, day)
        return days[i - 1] if i else None

    def nearest(self, dataset: str, day: dt.date, tolerance_days: int = EPISODE_TOLERANCE_DAYS) -> dt.date:
        """Closest complete curve date to ``day`` (the earlier on a tie);
        ``ValueError`` if none lies within ``tolerance_days``."""

        days = self._days(dataset)
        info = dataset_info(dataset)
        if not days:
            raise ValueError(f"no {info.title} history in the store")
        i = bisect.bisect_left(days, day)
        best = min(days[max(0, i - 1) : i + 1], key=lambda d: (abs((d - day).days), d))
        if abs((best - day).days) > tolerance_days:
            raise ValueError(
                f"no {info.title} curve within {tolerance_days} days of {day} "
                f"(the store has {len(days)} dates from {days[0]} to {days[-1]})"
            )
        return best

    def origin(self, dataset: str, day: dt.date) -> str:
        """Where the rows of a dataset on ``day`` came from (source and
        origin, e.g. ``"US Treasury, bundled snapshot"``)."""

        rows = self.rows(dataset)
        rows = rows[rows["asof"] == day]
        pairs = sorted({f"{s}, {o}" if o and o != s else s for s, o in zip(rows["source"], rows["origin"])})
        return "; ".join(pairs)

    # -- curves on a date ----------------------------------------------------

    def curve(self, dataset: str, day: dt.date, measure: Optional[str] = None) -> pd.Series:
        """``{tenor: decimal}`` on ``day`` (missing tenors dropped);
        ``ValueError`` if the dataset has no curve that day."""

        wide = self.wide(dataset, measure)
        if wide.empty or day not in wide.index:
            raise ValueError(f"no {dataset_info(dataset).title} data on {day}")
        return wide.loc[day].dropna()

    def series_labels(self, dataset: str, day: dt.date) -> Dict[str, str]:
        """``{tenor: "<source> <series code>"}`` of a dataset's rows on ``day``."""

        rows = self.rows(dataset)
        rows = rows[rows["asof"] == day]
        return {t: f"{src} {code}" for t, src, code in zip(rows["tenor"], rows["source"], rows["series"])}

    def fixing(self, name: str, day: dt.date) -> Optional[Tuple[dt.date, float, str]]:
        """Latest overnight fixing ``name`` (SOFR, EFFR, OBFR, ESTR, SONIA)
        published for a date on or before ``day``: (rate date, decimal,
        source); None if the store has none."""

        rows = self.rows("FIXINGS")
        rows = rows[(rows["item"] == name) & (rows["measure"] == "overnight_fixing") & (rows["asof"] <= day)]
        if rows.empty:
            return None
        row = rows.sort_values("asof").iloc[-1]
        return row["asof"], float(row["value_decimal"]), str(row["source"])

    def fixing_history(self, name: str, start: dt.date, end: dt.date) -> Dict[dt.date, float]:
        """Overnight fixings ``name`` for rate dates in [start, end):
        ``{date: decimal}`` (what part-fixed SOFR futures need)."""

        rows = self.rows("FIXINGS")
        rows = rows[(rows["item"] == name) & (rows["measure"] == "overnight_fixing")]
        rows = rows[(rows["asof"] >= start) & (rows["asof"] < end)]
        return {d: float(v) for d, v in zip(rows["asof"], rows["value_decimal"])}

    def fixings(self, day: dt.date) -> pd.DataFrame:
        """The latest of every overnight fixing on or before ``day``."""

        rows = []
        for name in sorted(set(FIXING_NAMES.values())):
            got = self.fixing(name, day)
            if got is not None:
                rows.append({"fixing": name, "date": got[0], "rate": got[1], "source": got[2]})
        return pd.DataFrame(rows, columns=["fixing", "date", "rate", "source"])

    def ois_par(self, dataset: str, day: dt.date) -> Dict[str, float]:
        """Par OIS rates ``{tenor: decimal}`` derived from a dataset's
        continuously compounded zero (spot) curve on ``day``, under the
        dataset's OIS conventions (see :mod:`derivx.rates.market_quotes`).
        Tenors shorter than the first published maturity or paying past the
        last are left out."""

        key = (dataset, day)
        if key not in self._par:
            info = dataset_info(dataset)
            if not info.ois_conventions:
                raise ValueError(f"{info.title} has no OIS conventions to derive par rates under")
            zeros = self.curve(dataset, day, "zero_cc")
            rows = [(parse_tenor(t).years, v) for t, v in zeros.items()]
            first = min(y for y, _ in rows)
            tenors = [t for t in PAR_TENORS if parse_tenor(t).years >= first - 1e-9]
            quotes = par_from_zero_rows(day, rows, info.ois_conventions, tenors)
            self._par[key] = {q.tenor: q.rate for q in quotes}
        return self._par[key]

    # -- moves -------------------------------------------------------------

    def curve_move(
        self, dataset: str, start: dt.date, end: dt.date, tolerance_days: int = EPISODE_TOLERANCE_DAYS
    ) -> CurveMove:
        """Change in bp (to 0.01bp) of every tenor present on both of the
        nearest complete curves to ``start`` and ``end``. Zero-curve
        datasets move their derived par OIS rates."""

        if end <= start:
            raise ValueError(f"the window end ({end}) must be after its start ({start})")
        key = (dataset, start, end, tolerance_days)
        if key in self._moves:
            return self._moves[key]
        d0 = self.nearest(dataset, start, tolerance_days)
        d1 = self.nearest(dataset, end, tolerance_days)
        if d1 <= d0:
            raise ValueError(f"the window {start} to {end} maps to a single {dataset} curve date ({d0})")
        info = dataset_info(dataset)
        if info.measure == "zero_cc":
            a, b = self.ois_par(dataset, d0), self.ois_par(dataset, d1)
            codes = {t: f"{info.source}: par OIS {t} (derived)" for t in a}
            what = "par OIS rates derived from the fitted OIS spot curve"
        else:
            a, b = self.curve(dataset, d0).to_dict(), self.curve(dataset, d1).to_dict()
            codes = self.series_labels(dataset, d1)
            what = {"par_yield": "par yields", "par_yield_cc": "par yields", "par_swap": "par swap rates"}.get(
                info.measure, info.measure
            )
        common = sorted(set(a) & set(b), key=lambda t: parse_tenor(t).years)
        if not common:
            raise ValueError(f"{info.title}: no tenor is quoted on both {d0} and {d1}")
        move = CurveMove(
            dataset,
            d0,
            d1,
            tuple(common),
            tuple(parse_tenor(t).years for t in common),
            tuple(round((b[t] - a[t]) * 1e4, 2) for t in common),
            tuple(codes.get(t, t) for t in common),
            what,
        )
        self._moves[key] = move
        return move


def build_store(frames: Iterable[pd.DataFrame], label: str = "") -> HistoryStore:
    """Merge long-format frames, lowest priority first."""

    parts = [f for f in frames if f is not None and len(f)]
    frame = pd.concat(parts, ignore_index=True) if parts else rs.empty_frame().assign(origin="")
    return HistoryStore(frame, label)


@lru_cache(maxsize=1)
def bundled_store() -> HistoryStore:
    """The committed snapshot as a store (shared, read-only)."""

    return build_store([_bundled()], ORIGIN_SNAPSHOT)


# --------------------------------------------------------------------------
# Cache and live refresh
# --------------------------------------------------------------------------


def cache_signature(cache: Optional[DiskCache] = None) -> Tuple[Tuple[str, str], ...]:
    """(key, fetched_at) of every cache entry: changes when anything new is
    downloaded, so callers can key their own caches on it."""

    cache = cache or DiskCache()
    return tuple(sorted((e.key, str(e.meta.get("fetched_at", ""))) for e in cache.entries()))


def cached_frames(cache: Optional[DiskCache] = None) -> List[pd.DataFrame]:
    """Every cached download, oldest first, tagged with its origin."""

    cache = cache or DiskCache()
    out = []
    for entry in reversed(cache.entries()):
        try:
            frame = entry.frame
        except (OSError, ValueError):  # unreadable or truncated parquet: skip it
            continue
        if set(rs.LONG_COLUMNS) <= set(frame.columns):
            frame["origin"] = f"downloaded {entry.meta.get('fetched_at', '')}"
            out.append(frame[list(rs.LONG_COLUMNS) + ["origin"]])
    return out


@dataclass
class LiveRefresh:
    """Result of :func:`refresh_live`: which requests succeeded and the
    notices to show (failures, stale cache, background downloads)."""

    ok: Dict[str, bool] = field(default_factory=dict)
    notices: List[str] = field(default_factory=list)
    frames: List[pd.DataFrame] = field(default_factory=list)  # fresh frames the cache could not keep


def live_requests(today: dt.date) -> Dict[str, Callable[..., rs.Fetched]]:
    """The fast requests behind the live data source (each well under a
    second when the host is up)."""

    since = dt.date(today.year - 1, 1, 1)
    recent = today - dt.timedelta(days=45)
    return {
        "US Treasury par curve (FRED H.15)": lambda **kw: rs.fred_series(rs.UST_FRED_IDS, since, **kw),
        "UK OIS curve (BoE)": lambda **kw: rs.boe_ois_curve("latest", **kw),
        "SOFR (NY Fed)": lambda **kw: rs.nyfed_rate("sofr", start=recent, end=today, **kw),
        "EFFR/OBFR (NY Fed)": lambda **kw: _concat(
            rs.nyfed_rate("effr", start=recent, end=today, **kw), rs.nyfed_rate("obfr", start=recent, end=today, **kw)
        ),
        "ESTR (ECB)": lambda **kw: rs.ecb_estr(start=recent, **kw),
        "EUR AAA par curve (ECB)": lambda **kw: rs.ecb_yield_curve(start=since, **kw),
        "EUR AAA spot curve (ECB)": lambda **kw: rs.ecb_yield_curve(
            rs.ECB_SPOT_TENORS, measure="spot", start=recent, **kw
        ),
        "SONIA (BoE via FRED)": lambda **kw: rs.fred_series(["IUDSOIA"], recent, **kw),
    }


def _concat(*fetched: rs.Fetched) -> rs.Fetched:
    first = fetched[0]
    return rs.Fetched(
        pd.concat([f.frame for f in fetched], ignore_index=True),
        first.source,
        " ".join(f.url for f in fetched),
        first.fetched_at,
        first.sha256,
        from_cache=all(f.from_cache for f in fetched),
        stale=any(f.stale for f in fetched),
        notices=[n for f in fetched for n in f.notices],
    )


def refresh_live(
    today: Optional[dt.date] = None,
    max_age: float = 3600.0,
    timeout: float = 8.0,
    retries: int = 1,
    offline: Optional[bool] = None,
    cache: Optional[DiskCache] = None,
    requests: Optional[Mapping[str, Callable[..., rs.Fetched]]] = None,
) -> LiveRefresh:
    """Bring the cache up to date: every live request runs concurrently with
    a short ``timeout``; entries younger than ``max_age`` seconds are not
    fetched again. Failures become notices, never exceptions."""

    today = today or rs.utc_now().date()
    cache = cache or DiskCache()
    requests = dict(requests or live_requests(today))
    out = LiveRefresh()
    kwargs = dict(offline=offline, max_age=max_age, timeout=timeout, retries=retries, cache=cache)
    with ThreadPoolExecutor(max_workers=max(1, len(requests))) as pool:
        futures = {name: pool.submit(call, **kwargs) for name, call in requests.items()}
    for name, future in futures.items():
        try:
            fetched = future.result()
        except Exception as exc:  # noqa: BLE001 - one failing source must never break the page
            out.ok[name] = False
            out.notices.append(f"{name}: unavailable ({exc}); using the cache or the bundled snapshot")
            continue
        out.ok[name] = True
        out.notices += [f"{name}: {n}" for n in fetched.notices]
        if not fetched.from_cache and cache.write_error:
            out.frames.append(fetched.frame.assign(origin=f"downloaded {fetched.fetched_at}"))
    return out


_BACKGROUND = ThreadPoolExecutor(max_workers=1, thread_name_prefix="derivx-treasury")
_PENDING: Dict[Tuple[int, str], Tuple[Future, float]] = {}


def treasury_in_background(year: int, cache: Optional[DiskCache] = None, max_age: float = 3600.0) -> Optional[str]:
    """Start (at most once per ``max_age``) a background download of the
    Treasury's own par-curve CSV for ``year`` into the cache: the endpoint
    is slow, so the page never waits for it. Returns a notice while it runs
    or after it failed; None when the cache is fresh or once it finished."""

    if rs.is_offline():
        return None
    cache = cache or DiskCache()
    entry = cache.load(cache.key("US Treasury", {"year": int(year)}))
    if entry is not None and entry.age_seconds() <= max_age:
        return None
    key = (int(year), str(cache.root))
    pending = _PENDING.get(key)
    now = rs.utc_now().timestamp()
    running = f"US Treasury CSV for {year}: downloading in the background (slow endpoint); FRED H.15 used meanwhile"
    if pending is not None:
        future, started = pending
        if not future.done():
            return running
        if now - started < max_age:  # finished recently: do not start again yet
            if future.exception() is None:
                return None
            return f"US Treasury CSV for {year}: background download failed ({future.exception()}); FRED H.15 used"
    _PENDING[key] = (_BACKGROUND.submit(rs.treasury_par_curve, int(year), cache=cache), now)
    return running


# --------------------------------------------------------------------------
# User uploads
# --------------------------------------------------------------------------


def parse_history_upload(
    raw: pd.DataFrame, currency: str, units: str = "auto", name: str = "upload"
) -> Tuple[pd.DataFrame, List[Issue], str]:
    """A user's swap-curve history -> (long frame, issues, units used).

    Accepts long tables (``date``, ``tenor``, ``rate``) or wide ones
    (``date`` plus one column per tenor, e.g. ``2Y``, ``10Y``). Rates are par
    swap rates in ``units`` (``auto`` reads the magnitudes and says so).
    Errors (bad dates, tenors or rates, fewer than two dates) leave the
    frame empty.
    """

    issues: List[Issue] = []
    table = raw.copy()
    table.columns = [str(c).strip() for c in table.columns]
    lower = {c.lower(): c for c in table.columns}
    if "date" not in lower:
        return rs.empty_frame(), [Issue("error", "the history needs a 'date' column")], units
    table = table.rename(columns={lower["date"]: "date"})
    if {"tenor", "rate"} <= set(lower):
        long = table.rename(columns={lower["tenor"]: "tenor", lower["rate"]: "rate"})[["date", "tenor", "rate"]]
    else:
        long = table.melt(id_vars="date", var_name="tenor", value_name="rate")
    long = long.dropna(subset=["rate"])
    try:
        long["date"] = pd.to_datetime(long["date"].astype(str).str.strip(), format="ISO8601").dt.date
    except (ValueError, TypeError) as exc:
        return rs.empty_frame(), [Issue("error", f"unreadable dates ({exc}); use YYYY-MM-DD")], units
    labels = {}
    for tenor in long["tenor"].astype(str).unique():
        try:
            labels[tenor] = str(parse_tenor(tenor))
        except ValueError as exc:
            issues.append(Issue("error", f"column/tenor {tenor!r}: {exc}"))
    values = pd.to_numeric(long["rate"].astype(str).str.replace("%", "").str.strip(), errors="coerce")
    if values.isna().any():
        bad = long.loc[values.isna(), "rate"].iloc[0]
        issues.append(Issue("error", f"rate {bad!r} is not a number"))
    if any(i.level == "error" for i in issues):
        return rs.empty_frame(), issues, units
    if units == "auto":
        reason = ambiguous_units(values)
        if reason:
            return rs.empty_frame(), issues + [Issue("error", reason)], units
        units = detect_units(values)
        issues.append(Issue("info", f"Rates read as {units} and converted to decimals"))
    decimal = values * UNIT_SCALE[units]
    outside = (decimal < RATE_RANGE[0]) | (decimal > RATE_RANGE[1])
    if outside.any():
        issues.append(
            Issue("error", f"{int(outside.sum())} rates fall outside {RATE_RANGE[0]:.0%} to {RATE_RANGE[1]:.0%}")
        )
        return rs.empty_frame(), issues, units
    if long["date"].nunique() < 2:
        issues.append(Issue("error", "the history needs at least two dates"))
        return rs.empty_frame(), issues, units
    unit_name = {"decimal": "decimal", "percent": "percent", "bp": "bp"}[units]
    n = len(long)
    frame = rs.long_frame(
        list(long["date"]),
        f"upload: {name}",
        [f"{currency} swap {labels[t]}" for t in long["tenor"].astype(str)],
        [labels[t] for t in long["tenor"].astype(str)],
        list(values),
        [unit_name] * n,
        ["par_swap"] * n,
        [user_dataset(currency)] * n,
    )
    frame["origin"] = f"upload: {name}"
    return frame, issues, units


def treasury_upload(text: str, name: str) -> pd.DataFrame:
    """A Treasury-format par curve CSV uploaded by the user -> long frame."""

    frame = rs.parse_treasury_csv(text)
    frame["source"] = f"upload: {name}"
    frame["origin"] = f"upload: {name}"
    return frame
