"""Free, keyless rates market data: FRED, US Treasury, NY Fed, ECB, BoE.

Every fetcher returns a :class:`Fetched` whose ``frame`` is in one
normalised long format (:data:`LONG_COLUMNS`), one row per observation:

* ``asof``           - observation date (``datetime.date``)
* ``source``         - publisher, e.g. ``"FRED"``, ``"US Treasury"``
* ``series``         - the publisher's series code
* ``tenor``          - market tenor label (``"ON"``, ``"3M"``, ``"10Y"``)
* ``tenor_years``    - nominal tenor in years (months/12; ON = 1/365)
* ``value_original`` - the number as published
* ``unit_original``  - its unit: ``percent``, ``bp``, ``decimal`` or ``index``
* ``value_decimal``  - the rate as a decimal (``percent / 100``); for an
  ``index`` it is the index level; NaN where the unit is unknown
* ``measure``        - what the number is (see :data:`MEASURES`)
* ``dataset``        - the history-store dataset it belongs to

Units are converted explicitly by :func:`to_decimal`; nothing is guessed
from magnitudes. Requests use short timeouts and a few retries with
backoff, then fall back to the last cached copy (see
:mod:`derivx.data.cache`). ``offline=True`` (or ``DERIVX_OFFLINE=1``)
reads only the cache.

Sources and their terms (all free, no key):

* FRED (St. Louis Fed) ``fredgraph.csv``; the JSON API needs a free key
  (``FRED_API_KEY``) and adds ALFRED vintages. H.15 Treasury yields and NY
  Fed rates are public; the ICE swap rates (``DSWP*``, 2000-07 to
  2016-10-28) are (c) ICE Benchmark Administration, republished by FRED.
* US Treasury daily par yield curve CSV (public domain; slow endpoint).
* NY Fed Markets API: SOFR, EFFR, OBFR, TGCR, BGCR, SOFR averages/index
  ("Source: Federal Reserve Bank of New York").
* ECB Data Portal: euro short-term rate and the euro area yield curves
  ("Source: ECB").
* Bank of England yield-curve archive: the fitted UK OIS (SONIA) curve
  ("Source: Bank of England").
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from derivx.data.cache import DiskCache, sha256, utc_now
from derivx.data.xlsx import excel_date, read_workbook
from derivx.rates.conventions import parse_tenor

LONG_COLUMNS = (
    "asof",
    "source",
    "series",
    "tenor",
    "tenor_years",
    "value_original",
    "unit_original",
    "value_decimal",
    "measure",
    "dataset",
)
UNIT_SCALE = {"percent": 0.01, "bp": 1e-4, "decimal": 1.0}
MEASURES = {
    "par_yield": "US Treasury par yield: bills (1Y and shorter) bond-equivalent, notes/bonds semiannual par coupons",
    "par_swap": "par swap rate (ICE USD: semiannual 30/360 fixed vs 3M LIBOR)",
    "par_yield_cc": "ECB par yield, continuously compounded",
    "zero_cc": "zero-coupon (spot) rate, continuously compounded",
    "forward_cc": "instantaneous forward rate, continuously compounded",
    "overnight_fixing": "overnight reference rate fixing (SOFR/EFFR/OBFR/ESTR ACT/360, SONIA ACT/365F)",
    "compounded_average": "backward-looking compounded average of SOFR (ACT/360)",
    "index_level": "compounded index level (dimensionless)",
}
OFFLINE_ENV = "DERIVX_OFFLINE"
FRED_KEY_ENV = "FRED_API_KEY"
USER_AGENT = "derivx-fic-analytics/0.1 (research; rates market data)"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
TREASURY_CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
    "{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)
NYFED_URL = "https://markets.newyorkfed.org/api/rates/{group}/{rate}/{path}.json"
ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
BOE_URL = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/{name}"

# FRED series this module understands: id -> (dataset, tenor, unit, measure).
UST_FRED_TENORS = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y",
}
ICE_FRED_TENORS = {f"DSWP{n}": f"{n}Y" for n in (1, 2, 3, 4, 5, 7, 10, 30)}
FRED_SERIES: Dict[str, Tuple[str, Optional[str], str, str]] = {
    **{sid: ("UST_PAR", tenor, "percent", "par_yield") for sid, tenor in UST_FRED_TENORS.items()},
    **{sid: ("USD_SWAP_ICE", tenor, "percent", "par_swap") for sid, tenor in ICE_FRED_TENORS.items()},
    "SOFR": ("FIXINGS", "ON", "percent", "overnight_fixing"),
    "EFFR": ("FIXINGS", "ON", "percent", "overnight_fixing"),
    "OBFR": ("FIXINGS", "ON", "percent", "overnight_fixing"),
    "IUDSOIA": ("FIXINGS", "ON", "percent", "overnight_fixing"),  # SONIA, Bank of England via FRED
    "ECBESTRVOLWGTTRMDMNRT": ("FIXINGS", "ON", "percent", "overnight_fixing"),  # ESTR, ECB via FRED
    "SOFR30DAYAVG": ("FIXINGS", "30D", "percent", "compounded_average"),
    "SOFR90DAYAVG": ("FIXINGS", "90D", "percent", "compounded_average"),
    "SOFR180DAYAVG": ("FIXINGS", "180D", "percent", "compounded_average"),
    "SOFRINDEX": ("FIXINGS", None, "index", "index_level"),
}
UST_FRED_IDS = tuple(UST_FRED_TENORS)
ICE_SWAP_FRED_IDS = tuple(ICE_FRED_TENORS)
ICE_SWAP_LAST_DATE = dt.date(2016, 10, 28)  # ICE stopped licensing DSWP* to FRED after this date

NYFED_RATES = {
    "sofr": ("secured", "SOFR"),
    "tgcr": ("secured", "TGCR"),
    "bgcr": ("secured", "BGCR"),
    "sofrai": ("secured", "SOFRAI"),
    "effr": ("unsecured", "EFFR"),
    "obfr": ("unsecured", "OBFR"),
}
ESTR_KEY = "B.EU000A2X2A25.WT"  # volume-weighted trimmed mean rate
ECB_CURVES = {"AAA": "G_N_A", "ALL": "G_N_C"}  # AAA-rated / all euro area central government bonds
ECB_MEASURES = {"par": ("PY", "par_yield_cc"), "spot": ("SR", "zero_cc"), "forward": ("IF", "forward_cc")}
ECB_TENORS = ("3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")
ECB_SPOT_TENORS = ("3M", "6M", "9M") + tuple(f"{y}Y" for y in range(1, 31))  # dense enough to derive par rates
BOE_ARCHIVES = {
    "latest": "latest-yield-curve-data.zip",  # current month, daily
    "daily": "oisddata.zip",  # 2009 to date, daily (about 12 MB)
    "month_end": "oismonthedata.zip",  # 2009 to date, month ends
}

DEFAULT_TIMEOUT = 15.0  # seconds per attempt
TREASURY_TIMEOUT = 45.0  # the Treasury CSV endpoint routinely takes 15-25 s


class SourceError(RuntimeError):
    """A source could not deliver data (network, HTTP or parse failure)."""


class SourceBlocked(SourceError):
    """HTTP 403/407: the host or the network policy refuses the request.
    Not retried."""


@dataclass
class Fetched:
    """A fetch result: the long-format ``frame`` plus provenance."""

    frame: pd.DataFrame
    source: str
    url: str
    fetched_at: str
    sha256: str
    from_cache: bool = False
    stale: bool = False
    notices: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        how = "cache" if self.from_cache else "live"
        return f"{self.source} ({how}, fetched {self.fetched_at})"


# --------------------------------------------------------------------------
# Units and the long format
# --------------------------------------------------------------------------


def to_decimal(values: Iterable[float], unit: str) -> np.ndarray:
    """Convert published numbers to decimals: percent / 100, bp / 10,000,
    decimals unchanged; an ``index`` level is passed through; an unknown
    unit gives NaN (never a guess). Raises ``ValueError`` for an
    unsupported unit name."""

    arr = np.asarray(list(values), dtype=float)
    if unit in UNIT_SCALE:
        return arr * UNIT_SCALE[unit]
    if unit == "index":
        return arr
    if unit == "unknown":
        return np.full_like(arr, np.nan)
    raise ValueError(f"unknown unit {unit!r}; expected one of {sorted(UNIT_SCALE) + ['index', 'unknown']}")


def tenor_years(tenor: Optional[str]) -> float:
    """Nominal tenor in years (months / 12, weeks * 7 / 365, ON = 1/365);
    NaN for no tenor. ``30D``-style averaging windows count calendar days."""

    if tenor is None:
        return float("nan")
    if tenor.endswith("D") and tenor[:-1].isdigit():
        return int(tenor[:-1]) / 365.0
    return parse_tenor(tenor).years


def long_frame(
    asof: Sequence[dt.date],
    source: Union[str, Sequence[str]],
    series: Sequence[str],
    tenor: Sequence[Optional[str]],
    values: Sequence[float],
    unit: Sequence[str],
    measure: Sequence[str],
    dataset: Sequence[str],
) -> pd.DataFrame:
    """Assemble a long-format frame (``source`` is one name or one per row);
    drops missing values; converts units row by row with
    :func:`to_decimal`."""

    def col(values: Sequence[object]) -> np.ndarray:
        return values.to_numpy(dtype=object) if isinstance(values, pd.Series) else np.asarray(values, dtype=object)

    frame = pd.DataFrame(
        {
            "asof": col(asof),
            "source": source if isinstance(source, str) else col(source),
            "series": col(series),
            "tenor": col(tenor),
            "value_original": pd.to_numeric(pd.Series(col(values)), errors="coerce"),
            "unit_original": col(unit),
            "measure": col(measure),
            "dataset": col(dataset),
        }
    )
    frame = frame[frame["value_original"].notna()].reset_index(drop=True)
    years = {t: tenor_years(t) for t in frame["tenor"].unique()}
    frame["tenor_years"] = frame["tenor"].map(years).astype(float)
    decimal = np.empty(len(frame))
    for u in frame["unit_original"].unique():
        mask = (frame["unit_original"] == u).to_numpy()
        decimal[mask] = to_decimal(frame.loc[mask, "value_original"], u)
    frame["value_decimal"] = decimal
    return frame[list(LONG_COLUMNS)]


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in LONG_COLUMNS})


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _urlopen(request: urllib.request.Request, timeout: float):
    """Single network call; tests replace this to serve recorded fixtures."""

    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 (https URLs built here)


def http_get(url: str, timeout: float = DEFAULT_TIMEOUT, retries: int = 2, backoff: float = 1.0) -> bytes:
    """GET ``url``: up to ``retries`` extra attempts on timeouts, connection
    errors, HTTP 429 and 5xx, sleeping ``backoff * 2**attempt`` seconds in
    between. HTTP 403/407 raise :class:`SourceBlocked` at once; other 4xx
    raise :class:`SourceError`."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with _urlopen(request, timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 407):
                message = f"HTTP {exc.code} from {_host(url)}: blocked by the host or the network policy"
                raise SourceBlocked(message) from exc
            if exc.code != 429 and exc.code < 500:
                raise SourceError(f"HTTP {exc.code} from {_host(url)}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(backoff * 2**attempt)
    raise SourceError(f"{_host(url)} unreachable after {retries + 1} attempts ({_reason(last)})")


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc


def _reason(exc: Optional[Exception]) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return str(exc) or type(exc).__name__


def _redact(url: str) -> str:
    """URL without credentials (api_key=...), safe to log and cache."""

    parts = urllib.parse.urlsplit(url)
    query = [
        (k, "REDACTED" if k.lower() in ("api_key", "key", "token") else v)
        for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def is_offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in ("1", "true", "yes")


def _fetch(
    source: str,
    params: Mapping[str, object],
    urls: Sequence[str],
    parse: Callable[[List[bytes]], pd.DataFrame],
    offline: Optional[bool],
    max_age: Optional[float],
    timeout: float,
    retries: int,
    cache: Optional[DiskCache],
) -> Fetched:
    """Cache-first fetch shared by every source (see module docstring)."""

    cache = cache or DiskCache()
    key = cache.key(source, params)
    entry = cache.load(key)
    offline = is_offline() if offline is None else offline

    def cached(stale: bool, notices: List[str]) -> Fetched:
        return Fetched(
            entry.frame,
            source,
            str(entry.meta.get("url", "")),
            str(entry.meta["fetched_at"]),
            str(entry.meta.get("sha256", "")),
            from_cache=True,
            stale=stale,
            notices=notices,
        )

    if offline:
        if entry is None:
            raise SourceError(f"offline: no cached {source} data for {dict(params)}; fetch it once online")
        return cached(False, [f"offline: {source} from the cache fetched {entry.meta['fetched_at']}"])
    if entry is not None and max_age is not None and entry.age_seconds() <= max_age:
        return cached(False, [])
    try:
        payloads = [http_get(url, timeout=timeout, retries=retries) for url in urls]
        frame = parse(payloads)
    except (SourceError, ValueError, KeyError) as exc:
        if entry is None:
            if isinstance(exc, SourceError):
                raise
            raise SourceError(f"{source}: unexpected response ({exc})") from exc
        return cached(True, [f"{source} fetch failed ({exc}); using the cache fetched {entry.meta['fetched_at']}"])
    payload = b"".join(payloads)
    meta = {
        "source": source,
        "params": dict(params),
        "url": " ".join(_redact(u) for u in urls),
        "fetched_at": utc_now().isoformat(timespec="seconds"),
    }
    stored = cache.store(key, frame, payload, meta)
    notices = [] if stored is not None else [f"{source}: not cached ({cache.write_error})"]
    return Fetched(frame, source, meta["url"], str(meta["fetched_at"]), sha256(payload), notices=notices)


def _iso(day: Optional[dt.date]) -> Optional[str]:
    return day.isoformat() if day is not None else None


# --------------------------------------------------------------------------
# FRED
# --------------------------------------------------------------------------


def _fred_meta(sid: str, units: Mapping[str, str]) -> Tuple[str, Optional[str], str, str]:
    dataset, tenor, unit, measure = FRED_SERIES.get(sid, ("FRED", None, "unknown", "unknown"))
    return dataset, tenor, units.get(sid, unit), measure


def parse_fred_csv(text: str, units: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """fredgraph.csv (``observation_date`` or ``DATE`` then one column per
    series; ``.`` or blank = missing) -> long format. Units come from
    :data:`FRED_SERIES` or ``units``; unknown series get ``unknown``."""

    units = dict(units or {})
    try:
        raw = pd.read_csv(io.StringIO(text), na_values=["."], dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"not a FRED CSV ({exc})") from exc
    if raw.empty or raw.columns[0].lower() not in ("observation_date", "date"):
        raise ValueError(f"not a FRED CSV: first column is {raw.columns[0]!r}")
    dates = pd.to_datetime(raw.iloc[:, 0], format="%Y-%m-%d").dt.date
    rows: Dict[str, list] = {k: [] for k in ("asof", "series", "tenor", "value", "unit", "measure", "dataset")}
    for sid in raw.columns[1:]:
        dataset, tenor, unit, measure = _fred_meta(sid, units)
        values = raw[sid]
        n = len(values)
        rows["asof"] += list(dates)
        rows["series"] += [sid] * n
        rows["tenor"] += [tenor] * n
        rows["value"] += list(values)
        rows["unit"] += [unit] * n
        rows["measure"] += [measure] * n
        rows["dataset"] += [dataset] * n
    return long_frame(
        rows["asof"],
        "FRED",
        rows["series"],
        rows["tenor"],
        rows["value"],
        rows["unit"],
        rows["measure"],
        rows["dataset"],
    )


def parse_fred_json(text: str, sid: str, units: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """FRED API ``series/observations`` JSON for one series -> long format.
    With vintages, the latest ``realtime_start`` per date wins."""

    try:
        payload = json.loads(text)
        observations = payload["observations"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"not a FRED API observations payload ({exc})") from exc
    obs = sorted(observations, key=lambda o: (o["date"], o.get("realtime_start", "")))
    latest = {o["date"]: o for o in obs}
    dataset, tenor, unit, measure = _fred_meta(sid, dict(units or {}))
    dates = [dt.date.fromisoformat(d) for d in latest]
    values = [None if o["value"] in (".", "") else o["value"] for o in latest.values()]
    n = len(dates)
    return long_frame(dates, "FRED", [sid] * n, [tenor] * n, values, [unit] * n, [measure] * n, [dataset] * n)


def fred_series(
    ids: Sequence[str],
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    *,
    units: Optional[Mapping[str, str]] = None,
    vintage: Optional[dt.date] = None,
    offline: Optional[bool] = None,
    max_age: Optional[float] = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = 2,
    cache: Optional[DiskCache] = None,
) -> Fetched:
    """FRED series from ``start`` to ``end`` (inclusive; open-ended if None).

    Uses the keyless fredgraph.csv endpoint, or the JSON API when
    ``FRED_API_KEY`` is set (required for ``vintage``: the data as known on
    that date, from ALFRED). The key is sent only to FRED and never cached
    or logged.
    """

    ids = [str(i).strip().upper() for i in ids if str(i).strip()]
    if not ids:
        raise ValueError("give at least one FRED series id, e.g. DGS10")
    if start and end and end < start:
        raise ValueError(f"end {end} is before start {start}")
    key = os.environ.get(FRED_KEY_ENV, "").strip()
    if vintage is not None and not key:
        raise ValueError(f"FRED vintages (ALFRED) need a free API key in ${FRED_KEY_ENV}")
    params = {"ids": ids, "start": _iso(start), "end": _iso(end), "vintage": _iso(vintage), "units": dict(units or {})}
    if key:
        urls = []
        for sid in ids:
            query = {"series_id": sid, "api_key": key, "file_type": "json"}
            if start:
                query["observation_start"] = start.isoformat()
            if end:
                query["observation_end"] = end.isoformat()
            if vintage:
                query["realtime_start"] = query["realtime_end"] = vintage.isoformat()
            urls.append(f"{FRED_API_URL}?{urllib.parse.urlencode(query)}")

        def parse(payloads: List[bytes]) -> pd.DataFrame:
            return pd.concat([parse_fred_json(p.decode("utf-8"), sid, units) for p, sid in zip(payloads, ids)])

        params["api"] = "json"
    else:
        query = {"id": ",".join(ids)}
        if start:
            query["cosd"] = ",".join([start.isoformat()] * len(ids))
        if end:
            query["coed"] = ",".join([end.isoformat()] * len(ids))
        urls = [f"{FRED_CSV_URL}?{urllib.parse.urlencode(query, safe=',')}"]

        def parse(payloads: List[bytes]) -> pd.DataFrame:
            frame = parse_fred_csv(payloads[0].decode("utf-8"), units)
            frame = frame[frame["series"].isin(ids)].reset_index(drop=True)  # only what was asked for
            if len(frame) == 0:
                raise ValueError(f"FRED returned no data for {ids}")
            return frame

    fetched = _fetch("FRED", params, urls, parse, offline, max_age, timeout, retries, cache)
    fetched.frame = _clip(fetched.frame, start, end)
    missing = [sid for sid in ids if sid not in set(fetched.frame["series"])]
    if missing:
        fetched.notices.append(f"FRED returned no observations for {', '.join(missing)} in the requested window")
    return fetched


def _clip(frame: pd.DataFrame, start: Optional[dt.date], end: Optional[dt.date]) -> pd.DataFrame:
    if start is not None:
        frame = frame[frame["asof"] >= start]
    if end is not None:
        frame = frame[frame["asof"] <= end]
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# US Treasury
# --------------------------------------------------------------------------


def treasury_tenor(label: str) -> str:
    """Treasury CSV column (``1 Mo``, ``1.5 Month``, ``10 Yr``) -> tenor."""

    return str(parse_tenor(label))


def parse_treasury_csv(text: str) -> pd.DataFrame:
    """Treasury daily par yield curve CSV (``Date`` as MM/DD/YYYY, one
    column per tenor, percent) -> long format (dataset ``UST_PAR``)."""

    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"not a Treasury par curve CSV ({exc})") from exc
    if "Date" not in raw.columns or len(raw.columns) < 2:
        raise ValueError(f"not a Treasury par curve CSV: columns {list(raw.columns)[:5]}")
    dates = pd.to_datetime(raw["Date"], format="%m/%d/%Y").dt.date
    cols = [c for c in raw.columns if c != "Date"]
    tenors = {c: treasury_tenor(c) for c in cols}
    n = len(raw)
    asof, series, tenor, values = [], [], [], []
    for c in cols:
        asof += list(dates)
        series += [c] * n
        tenor += [tenors[c]] * n
        values += list(raw[c])
    count = len(values)
    return long_frame(
        asof, "US Treasury", series, tenor, values, ["percent"] * count, ["par_yield"] * count, ["UST_PAR"] * count
    )


def treasury_par_curve(
    year: int,
    *,
    offline: Optional[bool] = None,
    max_age: Optional[float] = None,
    timeout: float = TREASURY_TIMEOUT,
    retries: int = 2,
    cache: Optional[DiskCache] = None,
) -> Fetched:
    """The Treasury's daily par yield curve for one calendar ``year``.

    The endpoint is slow (15-25 s is normal), so the default timeout is 45 s
    with two retries. FRED's ``DGS*`` series carry the same par (CMT)
    yields a day later and return in well under a second.
    """

    if not 1990 <= int(year) <= utc_now().year:
        raise ValueError(f"year must be between 1990 and {utc_now().year}, got {year}")
    url = TREASURY_CSV_URL.format(year=int(year))
    return _fetch(
        "US Treasury",
        {"year": int(year)},
        [url],
        lambda p: parse_treasury_csv(p[0].decode("utf-8-sig")),
        offline,
        max_age,
        timeout,
        retries,
        cache,
    )


# --------------------------------------------------------------------------
# NY Fed
# --------------------------------------------------------------------------


def parse_nyfed_json(text: str) -> pd.DataFrame:
    """NY Fed Markets API ``refRates`` JSON -> long format. Rates are in
    percent; ``SOFRAI`` rows give the 30/90/180-day averages and the index."""

    try:
        rates = json.loads(text)["refRates"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"not a NY Fed refRates payload ({exc})") from exc
    asof, series, tenor, values, unit, measure = [], [], [], [], [], []
    for r in rates:
        day = dt.date.fromisoformat(r["effectiveDate"])
        kind = str(r["type"]).upper()
        if kind == "SOFRAI":
            for field_name, label in (("average30day", "30D"), ("average90day", "90D"), ("average180day", "180D")):
                asof.append(day)
                series.append(f"SOFR{label[:-1]}DAYAVG")
                tenor.append(label)
                values.append(r.get(field_name))
                unit.append("percent")
                measure.append("compounded_average")
            asof.append(day)
            series.append("SOFRINDEX")
            tenor.append(None)
            values.append(r.get("index"))
            unit.append("index")
            measure.append("index_level")
        else:
            asof.append(day)
            series.append(kind)
            tenor.append("ON")
            values.append(r.get("percentRate"))
            unit.append("percent")
            measure.append("overnight_fixing")
    return long_frame(asof, "NY Fed", series, tenor, values, unit, measure, ["FIXINGS"] * len(asof))


def nyfed_rate(
    rate: str,
    last_n: Optional[int] = None,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    *,
    offline: Optional[bool] = None,
    max_age: Optional[float] = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = 2,
    cache: Optional[DiskCache] = None,
) -> Fetched:
    """A NY Fed reference rate (``sofr``, ``effr``, ``obfr``, ``tgcr``,
    ``bgcr`` or ``sofrai`` for the SOFR averages and index): the last
    ``last_n`` publications, or ``start``..``end``."""

    name = str(rate).strip().lower()
    if name not in NYFED_RATES:
        raise ValueError(f"unknown NY Fed rate {rate!r}; expected one of {sorted(NYFED_RATES)}")
    if (last_n is None) == (start is None):
        raise ValueError("give either last_n or start (with an optional end)")
    group, _ = NYFED_RATES[name]
    if last_n is not None:
        if int(last_n) < 1:
            raise ValueError("last_n must be at least 1")
        path = f"last/{int(last_n)}"
    else:
        end = end or utc_now().date()
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        path = "search"
    url = NYFED_URL.format(group=group, rate=name, path=path)
    if last_n is None:
        url += "?" + urllib.parse.urlencode({"startDate": start.isoformat(), "endDate": end.isoformat()})
    params = {"rate": name, "last_n": last_n, "start": _iso(start), "end": _iso(end)}
    return _fetch(
        "NY Fed",
        params,
        [url],
        lambda p: parse_nyfed_json(p[0].decode("utf-8")),
        offline,
        max_age,
        timeout,
        retries,
        cache,
    )


# --------------------------------------------------------------------------
# ECB
# --------------------------------------------------------------------------


def _ecb_series(key: str) -> Tuple[str, Optional[str], str, str]:
    """(series label, tenor, measure, dataset) of an ECB series key."""

    if key.startswith("EST."):
        return "ESTR", "ON", "overnight_fixing", "FIXINGS"
    if key.startswith("YC."):
        parts = key.split(".")
        curve, item = parts[5], parts[-1]
        prefix, _, tenor = item.partition("_")
        measure = {v[0]: v[1] for v in ECB_MEASURES.values()}.get(prefix)
        if measure is None or not tenor:
            raise ValueError(f"unsupported ECB yield-curve series {key!r}")
        label = {v: k for k, v in ECB_CURVES.items()}.get(curve, curve)
        tenor = _ym(tenor) if re.fullmatch(r"\d+Y\d+M", tenor) else str(parse_tenor(tenor))
        return key, tenor, measure, f"EUR_{label}_ECB"
    raise ValueError(f"unsupported ECB series {key!r}")


def _ym(text: str) -> str:
    """ECB maturities like ``10Y6M`` -> ``126M``."""

    years, _, months = text.partition("Y")
    return f"{int(years) * 12 + int(months.rstrip('M') or 0)}M"


def parse_ecb_csv(text: str) -> pd.DataFrame:
    """ECB Data Portal ``format=csvdata`` (``KEY``, ``TIME_PERIOD``,
    ``OBS_VALUE`` in percent) for the ESTR or yield-curve flows -> long
    format."""

    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"not an ECB csvdata payload ({exc})") from exc
    missing = {"KEY", "TIME_PERIOD", "OBS_VALUE"} - set(raw.columns)
    if missing:
        raise ValueError(f"not an ECB csvdata payload: missing {sorted(missing)}")
    meta = {key: _ecb_series(key) for key in raw["KEY"].unique()}
    asof = list(pd.to_datetime(raw["TIME_PERIOD"], format="%Y-%m-%d").dt.date)
    n = len(raw)
    return long_frame(
        asof,
        "ECB",
        [meta[k][0] for k in raw["KEY"]],
        [meta[k][1] for k in raw["KEY"]],
        list(raw["OBS_VALUE"]),
        ["percent"] * n,
        [meta[k][2] for k in raw["KEY"]],
        [meta[k][3] for k in raw["KEY"]],
    )


def _ecb_query(start: Optional[dt.date], end: Optional[dt.date], last_n: Optional[int]) -> str:
    query = {"format": "csvdata", "detail": "dataonly"}
    if last_n is not None:
        query["lastNObservations"] = str(int(last_n))
    if start is not None:
        query["startPeriod"] = start.isoformat()
    if end is not None:
        query["endPeriod"] = end.isoformat()
    return urllib.parse.urlencode(query)


def ecb_estr(
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    last_n: Optional[int] = None,
    *,
    offline: Optional[bool] = None,
    max_age: Optional[float] = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = 2,
    cache: Optional[DiskCache] = None,
) -> Fetched:
    """The euro short-term rate (volume-weighted trimmed mean, percent)."""

    if start is None and last_n is None:
        last_n = 30
    url = ECB_URL.format(flow="EST", key=ESTR_KEY) + "?" + _ecb_query(start, end, last_n)
    params = {"start": _iso(start), "end": _iso(end), "last_n": last_n}
    return _fetch(
        "ECB",
        {"flow": "EST", **params},
        [url],
        lambda p: parse_ecb_csv(p[0].decode("utf-8")),
        offline,
        max_age,
        timeout,
        retries,
        cache,
    )


def ecb_yield_curve(
    tenors: Sequence[str] = ECB_TENORS,
    curve: str = "AAA",
    measure: str = "par",
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    last_n: Optional[int] = None,
    *,
    offline: Optional[bool] = None,
    max_age: Optional[float] = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = 2,
    cache: Optional[DiskCache] = None,
) -> Fetched:
    """Euro area government yield curve (Svensson fit, business daily since
    2004-09-06): ``curve`` ``AAA`` (AAA-rated issuers) or ``ALL``;
    ``measure`` ``par`` (continuously compounded par yields), ``spot``
    (continuously compounded zero rates) or ``forward`` (instantaneous)."""

    curve = curve.upper()
    if curve not in ECB_CURVES:
        raise ValueError(f"curve must be one of {sorted(ECB_CURVES)}, got {curve!r}")
    if measure not in ECB_MEASURES:
        raise ValueError(f"measure must be one of {sorted(ECB_MEASURES)}, got {measure!r}")
    labels = [str(parse_tenor(t)) for t in tenors]
    if not labels:
        raise ValueError("give at least one tenor")
    prefix = ECB_MEASURES[measure][0]
    items = "+".join(f"{prefix}_{_ecb_tenor(t)}" for t in labels)
    key = f"B.U2.EUR.4F.{ECB_CURVES[curve]}.SV_C_YM.{items}"
    if start is None and last_n is None:
        last_n = 30
    url = ECB_URL.format(flow="YC", key=key) + "?" + _ecb_query(start, end, last_n)
    params = {
        "flow": "YC",
        "curve": curve,
        "measure": measure,
        "tenors": labels,
        "start": _iso(start),
        "end": _iso(end),
        "last_n": last_n,
    }
    return _fetch(
        "ECB",
        params,
        [url],
        lambda p: parse_ecb_csv(p[0].decode("utf-8")),
        offline,
        max_age,
        timeout,
        retries,
        cache,
    )


def _ecb_tenor(label: str) -> str:
    """Tenor label -> ECB maturity code (3M, 1Y, 10Y, 2Y6M)."""

    tenor = parse_tenor(label)
    months = tenor.months
    if months is None:
        raise ValueError(f"ECB curves are quoted in months and years, not {label!r}")
    years, rest = divmod(months, 12)
    if not years:
        return f"{rest}M"
    return f"{years}Y{rest}M" if rest else f"{years}Y"


# --------------------------------------------------------------------------
# Bank of England
# --------------------------------------------------------------------------


def _boe_tenor(years: float) -> str:
    months = round(years * 12)
    if abs(years * 12 - months) > 1e-4:
        raise ValueError(f"BoE maturity {years!r} years is not a whole number of months")
    return f"{months // 12}Y" if months % 12 == 0 and months >= 12 else f"{months}M"


def parse_boe_ois_workbook(data: bytes) -> pd.DataFrame:
    """One BoE OIS workbook -> long format of the spot (zero) curve:
    percent, continuously compounded (BoE yield-curve FAQ), maturities in
    years from the curve date. The short-end sheet (1-60 months) and the
    long sheet (0.5-25 years) overlap; the short-end value is kept."""

    book = read_workbook(data)
    spot_sheets = sorted((name for name in book.sheets if "spot" in name.lower()), key=lambda n: "short" not in n)
    if not spot_sheets:
        raise ValueError(f"no spot-curve sheet in the BoE workbook (sheets: {list(book.sheets)})")
    values: Dict[Tuple[dt.date, str], Tuple[float, float]] = {}
    for name in spot_sheets:
        rows = book.sheets[name]
        header = next((r for r in rows if r and str(r[0]).strip().lower() == "years:"), None)
        if header is None:
            raise ValueError(f"BoE sheet {name!r} has no 'years:' maturity row")
        maturities = {c: float(y) for c, y in enumerate(header) if c and isinstance(y, float)}
        for row in rows:
            if not row or not isinstance(row[0], float) or isinstance(row[0], bool):
                continue
            day = excel_date(row[0], book.date1904)
            for c, years in maturities.items():
                value = row[c] if c < len(row) else None
                if isinstance(value, float) and math.isfinite(value):
                    values.setdefault((day, _boe_tenor(years)), (years, value))
    if not values:
        raise ValueError("the BoE workbook has no spot-curve observations")
    keys = sorted(values, key=lambda k: (k[0], values[k][0]))
    n = len(keys)
    return long_frame(
        [k[0] for k in keys],
        "BoE",
        [f"BoE OIS spot {k[1]}" for k in keys],
        [k[1] for k in keys],
        [values[k][1] for k in keys],
        ["percent"] * n,
        ["zero_cc"] * n,
        ["GBP_OIS_BOE"] * n,
    )


def parse_boe_ois_zip(data: bytes) -> pd.DataFrame:
    """A BoE yield-curve zip archive -> long format of every OIS workbook in
    it (GLC gilt workbooks in the same archive are skipped)."""

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a zip archive ({exc})") from exc
    members = [m for m in archive.namelist() if m.lower().endswith(".xlsx") and "ois" in m.lower()]
    if not members:
        raise ValueError(f"no OIS workbook in the BoE archive ({archive.namelist()})")
    frame = pd.concat([parse_boe_ois_workbook(archive.read(m)) for m in members], ignore_index=True)
    return (
        frame.drop_duplicates(["asof", "tenor"], keep="last")
        .sort_values(["asof", "tenor_years"])
        .reset_index(drop=True)
    )


def boe_ois_curve(
    archive: str = "latest",
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    *,
    offline: Optional[bool] = None,
    max_age: Optional[float] = None,
    timeout: float = 30.0,
    retries: int = 2,
    cache: Optional[DiskCache] = None,
) -> Fetched:
    """Bank of England fitted UK OIS (SONIA) spot curve: ``archive``
    ``latest`` (current month), ``daily`` (2009 to date, ~12 MB) or
    ``month_end``. Rows outside ``start``..``end`` are dropped after the
    download (the archives are fixed files)."""

    if archive not in BOE_ARCHIVES:
        raise ValueError(f"archive must be one of {sorted(BOE_ARCHIVES)}, got {archive!r}")
    url = BOE_URL.format(name=BOE_ARCHIVES[archive])
    fetched = _fetch(
        "BoE",
        {"archive": archive},
        [url],
        lambda p: parse_boe_ois_zip(p[0]),
        offline,
        max_age,
        timeout,
        retries,
        cache,
    )
    fetched.frame = _clip(fetched.frame, start, end)
    return fetched
