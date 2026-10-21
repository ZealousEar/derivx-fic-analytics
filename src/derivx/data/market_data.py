"""Live option chains from Yahoo Finance (via yfinance), with lineage, plus
liquidity filters and a per-expiry hygiene report.

Yahoo serves only the *current* chain: there is no historical endpoint, so
a snapshot is never labelled with a past date. Every row of
:func:`fetch_option_chain` carries (schema: :mod:`derivx.data.chain_schema`):

* ``trade_date`` - the session the quotes belong to: the New York date of
  the spot's ``regularMarketTime`` from the same response (else of the
  latest last-trade time, else the last trading day on or before the fetch);
* ``fetched_at_utc`` - when that expiry's response arrived (ISO-8601 UTC);
* ``underlying_price`` / ``spot_time_utc`` - Yahoo's ``regularMarketPrice``
  and its time, read from the same response as the quotes (after the close
  this is the 16:00 close; extended-hours prices are not used);
* ``source`` = 'yahoo', ``ticker``.

Units: strikes and premiums in the underlying's currency per share;
``implied_vol`` is Yahoo's own per-annum decimal vol (from its last trade
and rate assumptions; :mod:`derivx.vol.quotes` re-inverts mids instead).

Robust access (:class:`Backoff`): rate limits (HTTP 429 / yfinance
``YFRateLimitError``) and transient network errors are retried with
exponential backoff and jitter; other errors are not retried. Only the
requested expiries are downloaded (``max_expiries`` truncates the *request*,
not the result); unknown requested expiries raise ``ValueError``. Once an
expiry is still rate limited after its retries, the remaining expiries are
not requested. :class:`FetchStatus` says what happened: 'ok', 'partial'
(some expiries failed, listed with reasons), 'empty', 'rate_limited' or
'error', with the reason.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from derivx.data.calendar import previous_trading_day, session_date
from derivx.data.chain_schema import COLUMN_ORDER, REQUIRED_COLUMNS

try:  # yfinance is a runtime dependency, but keep the module importable without it
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
except ImportError:  # pragma: no cover
    yf = None
    YFRateLimitError = None

try:
    from requests.exceptions import RequestException
except ImportError:  # pragma: no cover
    RequestException = OSError

try:  # yfinance >= 0.2.5x talks to Yahoo through curl_cffi
    from curl_cffi.requests.exceptions import RequestException as CurlRequestException
except ImportError:  # pragma: no cover
    CurlRequestException = OSError

_TRANSIENT = (ConnectionError, TimeoutError, json.JSONDecodeError, RequestException, CurlRequestException)
_TRANSIENT_TEXT = ("timed out", "timeout", "connection", "temporarily", "502", "503", "504", "reset by peer")
_RATE_TEXT = ("too many requests", "rate limit", "429")
TICKER_RE = re.compile(r"^[A-Z0-9.^=\-]{1,15}$")

# Kept for callers of the old module: the columns every snapshot row has.
REQUIRED_COLUMNS_LIST = list(REQUIRED_COLUMNS)


def classify_error(exc: BaseException) -> str:
    """'rate_limited', 'transient' (worth retrying) or 'fatal'."""

    if YFRateLimitError is not None and isinstance(exc, YFRateLimitError):
        return "rate_limited"
    text = str(exc).lower()
    if any(s in text for s in _RATE_TEXT):
        return "rate_limited"
    if isinstance(exc, _TRANSIENT) or any(s in text for s in _TRANSIENT_TEXT):
        return "transient"
    return "fatal"


def _message(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text[:200]}" if text else type(exc).__name__


@dataclass
class Backoff:
    """Retry policy: up to ``retries`` retries of rate-limited or transient
    errors, waiting ``min(max_delay, base_delay 2^i) (0.5 + 0.5 u)`` seconds
    before retry ``i`` (u uniform in [0, 1): 'equal jitter', so concurrent
    clients spread out). Counts retries and seconds waited."""

    retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0
    sleep: Callable[[float], None] = time.sleep
    rng: Optional[np.random.Generator] = None
    retried: int = 0
    waited_s: float = 0.0
    attempts: int = 0

    def __post_init__(self) -> None:
        if self.retries < 0 or self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("retries, base_delay and max_delay must be >= 0")
        if self.rng is None:
            # jitter only spaces out retries; it never touches data
            self.rng = np.random.default_rng(time.time_ns())

    def delay(self, retry: int) -> float:
        cap = min(self.max_delay, self.base_delay * 2.0**retry)
        return float(cap * (0.5 + 0.5 * self.rng.random()))

    def call(self, func: Callable[[], Any]) -> Any:
        for retry in range(self.retries + 1):
            self.attempts += 1
            try:
                return func()
            except Exception as exc:
                if classify_error(exc) == "fatal" or retry == self.retries:
                    raise
                wait = self.delay(retry)
                self.sleep(wait)
                self.retried += 1
                self.waited_s += wait
        raise RuntimeError("unreachable")  # pragma: no cover


@dataclass
class FetchStatus:
    """What a fetch did. Times are ISO-8601 UTC; ``spot_drift`` is the
    relative spot change between the first and last expiry response."""

    ticker: str
    status: str = "error"
    reason: str = ""
    started_at_utc: str = ""
    finished_at_utc: str = ""
    quote_date: Optional[str] = None
    spot: Optional[float] = None
    spot_time_utc: Optional[str] = None
    spot_drift: Optional[float] = None
    expiries_available: int = 0
    expiries_requested: List[str] = field(default_factory=list)
    expiries_fetched: List[str] = field(default_factory=list)
    failed_expiries: Dict[str, str] = field(default_factory=dict)
    requests: int = 0
    retries: int = 0
    waited_s: float = 0.0
    rows: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when quotes came back (complete or partial)."""

        return self.status in ("ok", "partial")

    @property
    def partial(self) -> bool:
        return self.status == "partial"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        if self.status == "ok":
            return (
                f"{self.rows:,} quotes over {len(self.expiries_fetched)} expiries, session {self.quote_date}, "
                f"fetched {self.finished_at_utc}"
            )
        if self.status == "partial":
            return (
                f"PARTIAL: {len(self.expiries_fetched)} of {len(self.expiries_requested)} expiries "
                f"({len(self.failed_expiries)} failed). {self.reason}"
            )
        return f"{self.status.upper()}: {self.reason}"


@dataclass
class ChainFetch:
    """A fetched chain (empty unless ``status.ok``) and its status."""

    chain: pd.DataFrame
    status: FetchStatus


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _normalise(df: Optional[pd.DataFrame], option_type: str, expiry: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.rename(
        columns={
            "contractSymbol": "contract_symbol",
            "lastTradeDate": "last_trade_date",
            "lastPrice": "last_price",
            "openInterest": "open_interest",
            "impliedVolatility": "implied_vol",
        }
    ).copy()
    out["option_type"] = option_type
    out["expiry"] = pd.Timestamp(expiry)
    for col in ("bid", "ask", "strike", "implied_vol", "volume", "open_interest", "last_price"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    if "bid" in out.columns and "ask" in out.columns:
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
    return out


def fetch_option_chain(
    ticker: str,
    *,
    expiries: Optional[Sequence[str]] = None,
    max_expiries: Optional[int] = None,
    backoff: Optional[Backoff] = None,
    request_pause: float = 0.2,
    client_factory: Optional[Callable[[str], Any]] = None,
    now: Callable[[], pd.Timestamp] = _utc_now,
    min_days: Optional[float] = None,
) -> ChainFetch:
    """Download the current chain of ``ticker`` (see the module docs).

    ``expiries`` (YYYY-MM-DD, must all be listed by Yahoo) restricts the
    request; ``max_expiries`` keeps the nearest N of them. ``min_days``
    (None: request everything listed) skips listed expiries fewer than that
    many calendar days after the New York date of the fetch (after the
    close Yahoo still lists the day's expired expiry; an analysis that
    cannot fit short slices need not spend requests on them), before
    ``max_expiries`` is applied; the skipped expiries are named in
    ``status.notes``. ``backoff`` is
    the retry policy (default :class:`Backoff()`); ``request_pause`` seconds
    separate expiry requests. ``client_factory`` (tests) replaces
    ``yfinance.Ticker``. Never raises for network failures (see
    ``status``); raises ``ValueError`` for an invalid ticker or
    ``max_expiries`` and for unknown requested expiries."""

    symbol = (ticker or "").strip().upper()
    if not TICKER_RE.match(symbol):
        raise ValueError(f"ticker must be 1-15 letters, digits or . ^ = -, got {ticker!r}")
    if max_expiries is not None and int(max_expiries) < 1:
        raise ValueError(f"max_expiries must be >= 1, got {max_expiries}")
    policy = backoff or Backoff()
    status = FetchStatus(ticker=symbol, started_at_utc=_iso(now()))
    factory = client_factory or (yf.Ticker if yf is not None else None)
    if factory is None:  # pragma: no cover
        status.reason = "yfinance is not installed"
        status.finished_at_utc = _iso(now())
        return ChainFetch(pd.DataFrame(columns=COLUMN_ORDER), status)
    client = factory(symbol)

    def finish(state: str, reason: str) -> ChainFetch:
        status.status, status.reason = state, reason
        status.finished_at_utc = _iso(now())
        status.requests, status.retries, status.waited_s = policy.attempts, policy.retried, round(policy.waited_s, 3)
        return ChainFetch(pd.DataFrame(columns=COLUMN_ORDER), status)

    try:
        available = list(policy.call(lambda: client.options) or [])
    except Exception as exc:
        kind = classify_error(exc)
        tries = policy.attempts
        if kind == "rate_limited":
            return finish(
                "rate_limited",
                f"Yahoo rate-limited the expiry listing ({tries} attempt(s), {policy.waited_s:.1f}s of backoff). "
                "Try again in a few minutes, or load a saved chain.",
            )
        return finish("error", f"listing expiries failed after {tries} attempt(s): {_message(exc)}")
    status.expiries_available = len(available)
    if not available:
        return finish("empty", f"Yahoo lists no option expiries for {symbol} (not optionable, or unknown symbol)")
    if expiries:
        wanted = [str(e) for e in expiries]
        unknown = [e for e in wanted if e not in available]
        if unknown:
            raise ValueError(
                f"expiries not listed by Yahoo for {symbol}: {unknown}; available: {available[:8]}"
                + ("..." if len(available) > 8 else "")
            )
        targets = [e for e in available if e in set(wanted)]
    else:
        targets = list(available)
    if min_days is not None:
        today = session_date(pd.Timestamp(status.started_at_utc))
        short = [e for e in targets if (pd.Timestamp(e) - today).days < float(min_days)]
        if short:
            targets = [e for e in targets if e not in set(short)]
            status.notes.append(
                f"not requested: {len(short)} expiry(ies) fewer than {float(min_days):g} days after "
                f"{today.date()} (New York): {', '.join(short[:6])}{'...' if len(short) > 6 else ''}"
            )
        if not targets:
            return finish("empty", f"no listed expiry is at least {float(min_days):g} days after {today.date()}")
    if max_expiries is not None:
        targets = targets[: int(max_expiries)]
    status.expiries_requested = targets

    frames: List[pd.DataFrame] = []
    spots: List[tuple] = []
    stop: Optional[str] = None
    for i, expiry in enumerate(targets):
        if stop is not None:
            status.failed_expiries[expiry] = stop
            continue
        try:
            response = policy.call(lambda e=expiry: client.option_chain(e))
        except Exception as exc:
            kind = classify_error(exc)
            status.failed_expiries[expiry] = f"{kind}: {_message(exc)}"
            if kind == "rate_limited":
                stop = "not requested: Yahoo was still rate limiting after the retries on an earlier expiry"
            continue
        received = now()
        underlying = dict(getattr(response, "underlying", None) or {})
        calls = _normalise(getattr(response, "calls", None), "call", expiry)
        puts = _normalise(getattr(response, "puts", None), "put", expiry)
        if calls.empty and puts.empty:
            status.failed_expiries[expiry] = "empty response (no calls or puts)"
            continue
        part = pd.concat([calls, puts], ignore_index=True)
        part["fetched_at_utc"] = _iso(received)
        price = underlying.get("regularMarketPrice")
        price = float(price) if isinstance(price, (int, float)) and np.isfinite(price) and price > 0 else np.nan
        t_spot = underlying.get("regularMarketTime")
        part["underlying_price"] = price
        part["spot_time_utc"] = (
            _iso(pd.Timestamp(int(t_spot), unit="s", tz="UTC")) if isinstance(t_spot, (int, float)) else None
        )
        if np.isfinite(price):
            spots.append((price, t_spot))
        frames.append(part)
        status.expiries_fetched.append(expiry)
        if request_pause and i < len(targets) - 1:
            policy.sleep(request_pause)

    if not frames:
        reasons = sorted(set(status.failed_expiries.values()))
        if any(r.startswith("rate_limited") for r in reasons):
            return finish("rate_limited", "Yahoo rate-limited every expiry request; try again in a few minutes")
        if all(r.startswith("empty") for r in reasons):
            return finish("empty", "Yahoo returned no quotes for any requested expiry")
        return finish("error", "every expiry request failed: " + "; ".join(reasons[:3]))

    chain = pd.concat(frames, ignore_index=True)
    if chain["underlying_price"].isna().any():
        if spots:
            chain["underlying_price"] = chain["underlying_price"].fillna(spots[0][0])
            status.notes.append("some responses had no spot: used the first spot of this fetch for them")
        else:
            try:
                last = float(policy.call(lambda: client.fast_info["last_price"]))
            except Exception as exc:
                return finish("error", f"no spot price in Yahoo's responses and fast_info failed: {_message(exc)}")
            chain["underlying_price"] = last
            status.notes.append("spot from yfinance fast_info (a separate request after the chain)")
    if spots:
        status.spot = float(spots[0][0])
        status.spot_drift = float(spots[-1][0] / spots[0][0] - 1.0)
        if spots[0][1] is not None:
            status.spot_time_utc = _iso(pd.Timestamp(int(spots[0][1]), unit="s", tz="UTC"))
    else:
        status.spot = float(chain["underlying_price"].iloc[0])

    # The session the quotes belong to (never the server's 'today').
    fetched = pd.Timestamp(status.started_at_utc)
    if status.spot_time_utc:
        quote_date = session_date(status.spot_time_utc)
        how = "from the spot's regularMarketTime"
    elif "last_trade_date" in chain.columns and chain["last_trade_date"].notna().any():
        latest = pd.to_datetime(chain["last_trade_date"], utc=True, errors="coerce").max()
        quote_date = session_date(latest)
        how = "from the latest last-trade time"
    else:
        quote_date = previous_trading_day(session_date(fetched))
        how = "from the fetch time (last trading day on or before it)"
    if quote_date != session_date(fetched):
        status.notes.append(
            f"quotes belong to the {quote_date.date()} session ({how}); fetched {status.started_at_utc}"
        )
    chain["trade_date"] = quote_date
    for col in ("trade_date", "expiry"):  # nanosecond unit, as validate_chain gives (pandas 2.1 concat needs one unit)
        chain[col] = pd.to_datetime(chain[col]).astype("datetime64[ns]")
    chain["ticker"] = symbol
    chain["source"] = "yahoo"
    status.quote_date = quote_date.date().isoformat()
    status.rows = int(len(chain))
    cols = [c for c in COLUMN_ORDER if c in chain.columns]
    result = ChainFetch(chain[cols].reset_index(drop=True), status)
    if status.failed_expiries:
        failed = len(status.failed_expiries)
        finish("partial", f"{failed} expiry request(s) failed: see failed_expiries")
    else:
        finish("ok", "")
    return result


def apply_liquidity_filters(
    chain: pd.DataFrame,
    min_volume: int = 10,
    max_spread: Optional[float] = 0.2,
    min_mid: float = 0.1,
) -> pd.DataFrame:
    """Keep quotes with ``volume >= min_volume`` (a missing volume on a row
    counts as 0), ``bid > 0``, ``ask > bid``, ``mid >= min_mid`` (premium
    per share) and, unless ``max_spread`` is None, relative spread ``(ask -
    bid) / mid <= max_spread``. Calls and puts are both kept (put-call parity
    needs them); the OTM selection, dedupe, IV-placeholder, expiry and
    staleness rules are :mod:`derivx.vol.quotes` hygiene, applied per slice
    before any fit (see :func:`hygiene_report`).

    ``volume`` is an optional column of the chain schema: without it the
    volume filter cannot be applied, so ``min_volume`` must be 0 (raises
    ``ValueError`` otherwise, rather than silently dropping every quote or
    silently skipping the filter)."""

    missing = {"bid", "ask"} - set(chain.columns)
    if missing:
        raise ValueError(f"chain is missing column(s) {sorted(missing)} needed by the liquidity filters")
    if min_volume < 0 or min_mid < 0 or (max_spread is not None and max_spread <= 0):
        raise ValueError("min_volume and min_mid must be >= 0 and max_spread > 0 (or None)")
    if "volume" not in chain.columns and min_volume > 0:
        raise ValueError(
            f"this chain has no 'volume' column, so the min volume filter ({min_volume}) cannot be applied: "
            "set min volume to 0 to use it without a volume filter"
        )
    filtered = chain.copy()
    bid, ask = filtered["bid"].astype(float), filtered["ask"].astype(float)
    mid = (bid + ask) / 2.0
    cond = pd.Series(True, index=filtered.index)
    if min_volume > 0:
        cond &= filtered["volume"].fillna(0) >= min_volume
    cond &= bid.fillna(0) > 0
    cond &= ask.fillna(0) > bid.fillna(0)
    cond &= mid.fillna(0) >= min_mid
    if max_spread is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_spread = (ask - bid) / mid.replace(0, np.nan)
        cond &= rel_spread <= max_spread
    return filtered.loc[cond].reset_index(drop=True)


def compute_moneyness(chain: pd.DataFrame, column: str = "moneyness") -> pd.DataFrame:
    """Add spot moneyness ``K / S`` (not forward moneyness: the fits use
    ``k = ln(K/F)`` on put-call-parity forwards)."""

    if "underlying_price" not in chain:
        raise ValueError("Chain must contain 'underlying_price' column")
    result = chain.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        result[column] = result["strike"] / result["underlying_price"]
    return result


def hygiene_report(
    chain: pd.DataFrame,
    *,
    rate: float = 0.0,
    min_days: float = 1.0,
    hygiene: Optional[Dict[str, object]] = None,
    discount=None,
) -> pd.DataFrame:
    """One row per (trade_date, expiry): what the quote hygiene keeps (one
    OTM quote per strike) and removes, by reason, the forward with how it
    was found (put-call parity, interpolated from the other expiries'
    parity forwards, or carry) and the discount factor with how it was
    found (``discount``, a :class:`derivx.vol.discounting.Discounting`, else
    the flat ``rate``), exactly as the calibration step cleans the chain
    (:func:`derivx.calibration.production.clean_chain`). ``days`` are
    calendar days to expiry; ``rate_pct`` is the effective cc rate ``-ln(DF)
    / tau`` in percent; ``parity_rate_pct`` / ``parity_se_bp`` are the
    put-call-parity regression's implied rate and its standard error when
    it was run (parity mode); ``k_min`` / ``k_max`` are ln(K/F) of the kept
    quotes; ``error`` says why a slice could not be cleaned at all."""

    from derivx.calibration.production import clean_chain

    if chain is None or chain.empty:
        return pd.DataFrame()
    cleaned, bad = clean_chain(chain, rate=rate, min_days=min_days, hygiene=hygiene, discount=discount)
    rows: List[Dict[str, object]] = []
    for td, exp, group, clean in cleaned:
        q = clean.quotes
        par = clean.parity or {}
        rows.append(
            {
                "trade_date": td.date(),
                "expiry": exp.date(),
                "days": int(clean.days),
                "quotes_in": int(clean.n_input),
                "kept": int(len(q)),
                "forward": clean.forward,
                "forward_method": clean.forward_method,
                "discount_factor": clean.discount_factor,
                "rate_pct": 100.0 * clean.rate,
                "discount_method": clean.discount_method,
                "parity_rate_pct": 100.0 * float(par.get("parity_rate", np.nan)),
                "parity_se_bp": 1e4 * float(par.get("parity_rate_se", np.nan)),
                "discount_source": clean.discount_source,
                "iv_source": clean.iv_source,
                **{f"removed_{k}": v for k, v in clean.counts.items()},
                "k_min": float(q["k"].min()) if len(q) else np.nan,
                "k_max": float(q["k"].max()) if len(q) else np.nan,
                "error": "",
            }
        )
    for s in bad:
        rows.append(
            {
                "trade_date": s.trade_date.date(),
                "expiry": s.expiry.date(),
                "days": int((s.expiry - s.trade_date).days),
                "quotes_in": int(s.n_points),
                "kept": 0,
                "error": s.reason,
            }
        )
    out = pd.DataFrame(rows).sort_values(["trade_date", "expiry"]).reset_index(drop=True)
    removed = sorted(c for c in out.columns if c.startswith("removed_"))
    if removed:
        out[removed] = out[removed].fillna(0).astype(int)
    lead = [
        "trade_date",
        "expiry",
        "days",
        "quotes_in",
        "kept",
        "forward",
        "forward_method",
        "discount_factor",
        "rate_pct",
        "discount_method",
        "parity_rate_pct",
        "parity_se_bp",
        "iv_source",
    ]
    tail = [c for c in ("k_min", "k_max", "discount_source", "error") if c in out.columns]
    return out[[c for c in lead if c in out.columns] + removed + tail]
