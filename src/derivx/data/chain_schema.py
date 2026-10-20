"""Schema of an option-chain snapshot, and validation of uploaded files.

One row per quote. Required columns (:data:`REQUIRED_COLUMNS`) and units:

* ``trade_date`` - the quote (valuation) date, the session the quotes belong
  to; ``tau`` is measured from it (ACT/365F, see :mod:`derivx.vol.quotes`).
* ``expiry`` - expiration date.
* ``option_type`` - 'call' / 'put' (or C / P, any case).
* ``strike`` - in price units of the underlying, > 0.
* ``bid`` / ``ask`` - premium per share in the underlying's currency
  (>= 0; a 0 bid means no executable bid).
* ``implied_vol`` - Black implied vol, per-annum decimal (0.18 = 18%); may be
  missing on a row (vols are re-inverted from bid/ask mids when possible).
* ``underlying_price`` - spot of the underlying at the snapshot, > 0; one
  value per (trade_date, expiry).

Optional columns (kept when present): ``mid``, ``volume``,
``open_interest``, ``last_price``, ``last_trade_date`` (time of the last
trade, UTC), ``ticker``, ``fetched_at_utc`` (when the quotes were fetched,
ISO-8601 UTC), ``spot_time_utc``, ``contract_symbol``, ``source``. Other
columns are dropped with a note. Common vendor spellings are renamed
(:data:`ALIASES`, e.g. ``impliedVolatility`` -> ``implied_vol``) and
reported.

:func:`validate_chain` never lets a bad file through silently: problems
that would make numbers wrong (missing columns, text in numeric columns,
non-positive strikes or spots, vols that look like percent, unknown option
types, unparseable dates, several spots in one snapshot, several tickers)
raise :class:`ChainSchemaError` with every problem listed; harmless
oddities (duplicate quotes, missing vols or bids) become warnings. A file
may hold several trade dates: :class:`ChainDataset` splits it into one
frame per trade date, and :func:`merge_datasets` refuses two files that
carry the same trade date.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: Dict[str, str] = {
    "trade_date": "quote (valuation) date, YYYY-MM-DD",
    "expiry": "expiration date, YYYY-MM-DD",
    "option_type": "call / put (or C / P)",
    "strike": "strike in price units of the underlying, > 0",
    "bid": "bid premium per share, >= 0",
    "ask": "ask premium per share, >= 0",
    "implied_vol": "Black implied vol, per-annum decimal (0.18 = 18%)",
    "underlying_price": "spot of the underlying at the snapshot, > 0",
}
OPTIONAL_COLUMNS: Dict[str, str] = {
    "mid": "(bid + ask) / 2, premium per share",
    "volume": "contracts traded in the session",
    "open_interest": "open contracts",
    "last_price": "last traded premium per share",
    "last_trade_date": "time of the last trade (UTC)",
    "ticker": "underlying symbol",
    "fetched_at_utc": "time the quotes were fetched, ISO-8601 UTC",
    "spot_time_utc": "time of the spot quote, ISO-8601 UTC",
    "contract_symbol": "OCC contract symbol",
    "source": "where the quotes came from",
}
COLUMN_ORDER = list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)

ALIASES: Dict[str, str] = {
    "impliedvolatility": "implied_vol",
    "iv": "implied_vol",
    "impliedvol": "implied_vol",
    "openinterest": "open_interest",
    "oi": "open_interest",
    "lasttradedate": "last_trade_date",
    "lastprice": "last_price",
    "contractsymbol": "contract_symbol",
    "type": "option_type",
    "optiontype": "option_type",
    "callput": "option_type",
    "cp": "option_type",
    "putcall": "option_type",
    "underlying": "underlying_price",
    "underlyingprice": "underlying_price",
    "spot": "underlying_price",
    "expiration": "expiry",
    "expirationdate": "expiry",
    "expirydate": "expiry",
    "quotedate": "trade_date",
    "tradedate": "trade_date",
    "asof": "trade_date",
    "fetchedat": "fetched_at_utc",
    "fetchedatutc": "fetched_at_utc",
    "spottimeutc": "spot_time_utc",
}
NUMERIC = ("strike", "bid", "ask", "implied_vol", "underlying_price", "mid", "volume", "open_interest", "last_price")
MAX_ROWS = 2_000_000
IV_PERCENT_MEDIAN = 3.0  # a median vol above 300% means the column is in percent


class ChainSchemaError(ValueError):
    """An option-chain file that cannot be used as is; ``problems`` lists
    every reason (``str()`` joins them)."""

    def __init__(self, problems: Iterable[str], source: str = "chain"):
        self.problems = [str(p) for p in problems]
        self.source = source
        super().__init__(f"{source}: " + "; ".join(self.problems))


@dataclass
class ChainDataset:
    """Validated quotes, one normalised frame per trade date, plus where
    they came from. ``fetched_at_utc`` is the latest fetch time recorded in
    the data (None when the file has no ``fetched_at_utc`` column)."""

    chains: Dict[pd.Timestamp, pd.DataFrame]
    source: str
    ticker: Optional[str] = None
    fetched_at_utc: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    renamed: Dict[str, str] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    _fingerprint: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def trade_dates(self) -> List[pd.Timestamp]:
        return sorted(self.chains)

    @property
    def rows(self) -> int:
        return int(sum(len(df) for df in self.chains.values()))

    def single(self) -> pd.DataFrame:
        """The one trade date's frame; raises when there are several."""

        if len(self.chains) != 1:
            dates = ", ".join(d.date().isoformat() for d in self.trade_dates[:6])
            raise ChainSchemaError(
                [
                    f"holds {len(self.chains)} trade dates ({dates}{'...' if len(self.chains) > 6 else ''}): "
                    "load it on the Walk-Forward tab, or keep one trade date"
                ],
                self.source,
            )
        return next(iter(self.chains.values()))

    def frame(self) -> pd.DataFrame:
        """All trade dates in one frame (sorted by trade date)."""

        return pd.concat([self.chains[d] for d in self.trade_dates], ignore_index=True)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the quotes (computed once: treat ``chains`` as read-only)."""

        if self._fingerprint is None:
            self._fingerprint = frames_hash(self.chains)
        return self._fingerprint

    def provenance(self) -> Dict[str, object]:
        """JSON-friendly description for banners and run manifests."""

        return {
            "source": self.source,
            "ticker": self.ticker,
            "trade_dates": [d.date().isoformat() for d in self.trade_dates],
            "fetched_at_utc": self.fetched_at_utc,
            "rows": self.rows,
            "expiries": int(sum(df["expiry"].nunique() for df in self.chains.values())),
            "files": list(self.files),
            "data_sha256": self.fingerprint,
        }


def frames_hash(frames: Mapping[object, pd.DataFrame] | pd.DataFrame) -> str:
    """SHA-256 of the content of one frame or of a mapping of frames
    (column names, values and the mapping keys; row order matters)."""

    h = hashlib.sha256()
    items = [("", frames)] if isinstance(frames, pd.DataFrame) else sorted(frames.items(), key=lambda kv: str(kv[0]))
    for key, df in items:
        h.update(str(key).encode())
        h.update("|".join(map(str, df.columns)).encode())
        h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _norm_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Lower-case, snake-case and alias the column names; returns the
    renamed frame and ``{original: new}`` for the names that changed."""

    renamed: Dict[str, str] = {}
    new_cols = []
    for col in df.columns:
        base = _norm_name(col)
        target = (
            base if base in REQUIRED_COLUMNS or base in OPTIONAL_COLUMNS else ALIASES.get(base.replace("_", ""), base)
        )
        if target != col and (target in REQUIRED_COLUMNS or target in OPTIONAL_COLUMNS):
            renamed[str(col)] = target
        new_cols.append(target)
    out = df.copy()
    out.columns = new_cols
    return out, renamed


def _examples(values: pd.Series, n: int = 3) -> str:
    return ", ".join(repr(v) for v in values.astype(str).unique()[:n])


def _parse_dates(series: pd.Series, name: str, problems: List[str]) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    else:
        with warnings.catch_warnings():  # "could not infer format": the mixed-format retry below handles it
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce")  # one format, inferred from the first value
        if (parsed.isna() & series.notna()).any():
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if not pd.api.types.is_datetime64_any_dtype(parsed):  # mixed UTC offsets
            parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    bad = parsed.isna() & series.notna()
    if bad.any():
        problems.append(f"{name}: {int(bad.sum())} value(s) are not dates (e.g. {_examples(series[bad])})")
    if series.isna().any():
        problems.append(f"{name}: {int(series.isna().sum())} empty value(s)")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)  # keep the local calendar date
    return parsed.dt.normalize().astype("datetime64[ns]")  # one unit everywhere (pandas 2.1 concat needs it)


def _option_types(series: pd.Series, problems: List[str]) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    out = pd.Series(
        np.where(raw.str.startswith("c"), "call", np.where(raw.str.startswith("p"), "put", "")), index=series.index
    )
    bad = (out == "") | series.isna()
    if bad.any():
        problems.append(f"option_type: {int(bad.sum())} value(s) are not call/put (e.g. {_examples(series[bad])})")
    return out


def validate_chain(df: pd.DataFrame, source: str = "chain", *, ticker: Optional[str] = None) -> ChainDataset:
    """Check and normalise one file's quotes (see the module docs).

    ``ticker`` is used when the data has no ``ticker`` column (it is never
    taken from a file name). Raises :class:`ChainSchemaError`."""

    if not isinstance(df, pd.DataFrame):
        raise ChainSchemaError(["not a table"], source)
    if df.empty:
        raise ChainSchemaError(["the file has no rows"], source)
    if len(df) > MAX_ROWS:
        raise ChainSchemaError([f"{len(df):,} rows: more than the {MAX_ROWS:,} rows this app accepts"], source)
    data, renamed = normalise_columns(df)
    problems: List[str] = []
    warnings: List[str] = []
    dup_cols = data.columns[data.columns.duplicated()].unique().tolist()
    if dup_cols:
        raise ChainSchemaError([f"columns appear more than once after renaming: {dup_cols}"], source)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        detail = "; ".join(f"{c} ({REQUIRED_COLUMNS[c]})" for c in missing)
        found = ", ".join(map(str, df.columns[:20]))
        raise ChainSchemaError([f"missing required column(s): {detail}. Found: {found}"], source)
    extra = [c for c in data.columns if c not in REQUIRED_COLUMNS and c not in OPTIONAL_COLUMNS]
    if extra:
        warnings.append(f"ignored column(s): {', '.join(map(str, extra[:12]))}{'...' if len(extra) > 12 else ''}")
        data = data.drop(columns=extra)

    for col in [c for c in NUMERIC if c in data.columns]:
        raw = data[col]
        num = pd.to_numeric(raw, errors="coerce")
        bad = num.isna() & raw.notna() & (raw.astype(str).str.strip() != "")
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} non-numeric value(s) (e.g. {_examples(raw[bad])})")
        data[col] = num.astype(float)
        if np.isinf(data[col]).any():
            problems.append(f"{col}: {int(np.isinf(data[col]).sum())} infinite value(s)")
    strike = data["strike"]
    if strike.isna().any():
        problems.append(f"strike: {int(strike.isna().sum())} empty value(s)")
    if (strike <= 0).any():
        problems.append(f"strike: {int((strike <= 0).sum())} value(s) <= 0")
    spot = data["underlying_price"]
    if spot.isna().any() or (spot <= 0).any():
        problems.append(f"underlying_price: {int((spot.isna() | (spot <= 0)).sum())} empty or non-positive value(s)")
    for col in ("bid", "ask"):
        if (data[col] < 0).any():
            problems.append(f"{col}: {int((data[col] < 0).sum())} negative value(s)")
        n_nan = int(data[col].isna().sum())
        if n_nan:
            warnings.append(f"{col}: {n_nan} empty value(s) (those quotes have no executable market)")
    iv = data["implied_vol"]
    finite = iv[np.isfinite(iv)]
    if (finite < 0).any():
        problems.append(f"implied_vol: {int((finite < 0).sum())} negative value(s)")
    if finite.size and float(finite.median()) > IV_PERCENT_MEDIAN:
        problems.append(
            f"implied_vol looks like percent (median {float(finite.median()):.4g}): give per-annum decimals "
            "(18% = 0.18), e.g. divide the column by 100"
        )
    if iv.isna().any():
        warnings.append(f"implied_vol: {int(iv.isna().sum())} empty value(s) (re-inverted from bid/ask where possible)")
    data["option_type"] = _option_types(data["option_type"], problems)
    data["trade_date"] = _parse_dates(data["trade_date"], "trade_date", problems)
    data["expiry"] = _parse_dates(data["expiry"], "expiry", problems)
    if "last_trade_date" in data.columns:
        data["last_trade_date"] = pd.to_datetime(data["last_trade_date"], errors="coerce", utc=True, format="mixed")
    tickers: List[str] = []
    if "ticker" in data.columns:
        tickers = sorted({str(t).strip().upper() for t in data["ticker"].dropna() if str(t).strip()})
        if len(tickers) > 1:
            problems.append(f"ticker: several underlyings in one file ({', '.join(tickers[:5])}); load one at a time")
    if problems:
        raise ChainSchemaError(problems, source)

    keys = ["trade_date", "expiry"]
    spots = data.groupby(keys)["underlying_price"].agg(lambda s: s.max() - s.min())
    multi = spots[spots > 1e-9 * data["underlying_price"].abs().max()]
    if not multi.empty:
        (td, exp) = multi.index[0]
        raise ChainSchemaError(
            [
                f"{len(multi)} (trade_date, expiry) group(s) carry several underlying prices, e.g. trade date "
                f"{td.date()} expiry {exp.date()}: one snapshot per trade date (merged snapshots?)"
            ],
            source,
        )
    dups = data.duplicated(["trade_date", "expiry", "option_type", "strike"]).sum()
    if dups:
        warnings.append(
            f"{int(dups)} duplicate quote(s) (same date, expiry, type and strike): hygiene keeps the tightest"
        )
    if "mid" not in data.columns:
        data["mid"] = (data["bid"] + data["ask"]) / 2.0
    expired = int((data["expiry"] < data["trade_date"]).sum())
    if expired:
        warnings.append(f"{expired} quote(s) expire before their trade date (dropped as expired before fitting)")
    tick = tickers[0] if tickers else (ticker.strip().upper() if ticker and ticker.strip() else None)
    if tick is not None:
        data["ticker"] = tick
    fetched = None
    if "fetched_at_utc" in data.columns:
        times = pd.to_datetime(data["fetched_at_utc"], errors="coerce", utc=True, format="mixed").dropna()
        if not times.empty:
            fetched = times.max().isoformat()
    cols = [c for c in COLUMN_ORDER if c in data.columns]
    data = data[cols]
    chains = {
        pd.Timestamp(td): g.sort_values(["expiry", "option_type", "strike"]).reset_index(drop=True)
        for td, g in data.groupby("trade_date", sort=True)
    }
    return ChainDataset(
        chains=chains,
        source=source,
        ticker=tick,
        fetched_at_utc=fetched,
        warnings=warnings,
        renamed=renamed,
        files=[source],
    )


def _sniff_sep(text_head: str) -> str:
    header = text_head.splitlines()[0] if text_head else ""
    counts = {sep: header.count(sep) for sep in (",", ";", "\t", "|")}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def read_chain_file(name: str, data: bytes) -> pd.DataFrame:
    """Read an uploaded chain: ``.parquet`` / ``.pq``, ``.csv`` / ``.txt``
    (comma, semicolon, tab or pipe separated, detected from the header) or
    either CSV gzip-compressed (``.gz``). Raises :class:`ChainSchemaError`
    with a readable reason (never a parser traceback)."""

    lower = name.lower()
    try:
        if lower.endswith(".gz"):
            data = gzip.decompress(data)
            lower = lower[:-3]
        if lower.endswith((".parquet", ".pq")):
            return pd.read_parquet(io.BytesIO(data))
        if lower.endswith((".csv", ".txt")):
            head = data[:4096].decode("utf-8-sig", errors="replace")
            return pd.read_csv(io.BytesIO(data), sep=_sniff_sep(head), encoding="utf-8-sig", low_memory=False)
    except ChainSchemaError:
        raise
    except Exception as exc:  # parser errors carry no server paths here, but keep only the message
        kind = "parquet" if lower.endswith((".parquet", ".pq")) else "CSV"
        msg = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
        raise ChainSchemaError([f"could not read the file as {kind}: {msg}"], name) from None
    raise ChainSchemaError(["unsupported file type: use .csv, .csv.gz or .parquet"], name)


def load_chain_file(name: str, data: bytes, *, ticker: Optional[str] = None) -> ChainDataset:
    """:func:`read_chain_file` then :func:`validate_chain`, labelled with
    the file name as the source."""

    return validate_chain(read_chain_file(name, data), source=f"upload: {name}", ticker=ticker)


def merge_datasets(datasets: List[ChainDataset], source: Optional[str] = None) -> ChainDataset:
    """Combine several files into one history. Refuses (with every clash
    listed) two files holding the same trade date or different tickers."""

    if not datasets:
        raise ChainSchemaError(["no files"], source or "upload")
    seen: Dict[pd.Timestamp, str] = {}
    clashes: List[str] = []
    for ds in datasets:
        for td in ds.trade_dates:
            if td in seen:
                clashes.append(f"trade date {td.date()} is in both {seen[td]} and {ds.source}")
            else:
                seen[td] = ds.source
    tickers = sorted({ds.ticker for ds in datasets if ds.ticker})
    if len(tickers) > 1:
        clashes.append(f"files are for different tickers: {', '.join(tickers)}")
    if clashes:
        raise ChainSchemaError(clashes + ["keep one snapshot per trade date"], source or "upload")
    chains: Dict[pd.Timestamp, pd.DataFrame] = {}
    warnings: List[str] = []
    renamed: Dict[str, str] = {}
    files: List[str] = []
    for ds in datasets:
        chains.update(ds.chains)
        warnings += [f"{ds.source}: {w}" for w in ds.warnings]
        renamed.update(ds.renamed)
        files += ds.files
    fetched = [ds.fetched_at_utc for ds in datasets if ds.fetched_at_utc]
    return ChainDataset(
        chains=dict(sorted(chains.items())),
        source=source or (datasets[0].source if len(datasets) == 1 else f"{len(datasets)} files"),
        ticker=tickers[0] if tickers else None,
        fetched_at_utc=max(fetched) if fetched else None,
        warnings=warnings,
        renamed=renamed,
        files=files,
    )


def chain_to_csv(df: pd.DataFrame) -> bytes:
    """CSV bytes in this schema (re-uploadable): dates as YYYY-MM-DD, times
    as ISO-8601 UTC."""

    out = df.reset_index(drop=True)
    for col in ("trade_date", "expiry"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    if "last_trade_date" in out.columns:
        ltd = pd.to_datetime(out["last_trade_date"], errors="coerce", utc=True)
        out["last_trade_date"] = ltd.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.to_csv(index=False).encode("utf-8")
