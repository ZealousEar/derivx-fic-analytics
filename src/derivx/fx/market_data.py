"""ECB euro foreign exchange reference rates, for FX spot.

Source: the ECB Data Portal, dataset ``EXR``, series ``D.<CCY>.EUR.SP00.A``
(daily reference rate, units of the currency per 1 EUR). The rates are set
around 14:15 CET by a concertation of central banks and published around
16:00 CET on TARGET business days. They are reference rates for
information, not dealable quotes. Crosses (GBPUSD = USD per EUR / GBP per
EUR, USDJPY = JPY per EUR / USD per EUR, ...) are computed from one day's
rates and carry the ECB's rounding (four or five significant decimals).

* :func:`fetch_ecb_rates` - the latest rates over the network (short
  timeout; raises :class:`FxDataError` on any failure).
* :func:`bundled_ecb_rates` - the snapshot shipped in
  ``data/examples/ecb_fx_reference_rates.csv`` (the offline fallback).
* :func:`parse_ecb_csv` - parser of the Data Portal's ``csvdata`` format.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from derivx.fx.conventions import CurrencyPair, get_pair

ECB_CURRENCIES = ("USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/EXR/D.{codes}.EUR.SP00.A"
    "?lastNObservations={n}&format=csvdata&detail=dataonly"
)
ECB_SOURCE = "ECB euro foreign exchange reference rates (Data Portal, EXR, 14:15 CET)"
BUNDLED_CSV = Path(__file__).resolve().parents[3] / "data" / "examples" / "ecb_fx_reference_rates.csv"
DEFAULT_TIMEOUT = 4.0  # seconds: the page never waits longer than this for the ECB


class FxDataError(RuntimeError):
    """The ECB rates could not be fetched or parsed (message says why)."""


@dataclass(frozen=True)
class EcbRates:
    """ECB reference rates of one day: ``per_eur[ccy]`` units of ``ccy`` per
    1 EUR on ``asof`` (EUR itself is 1)."""

    asof: dt.date
    per_eur: Dict[str, float]
    origin: str  # 'live' (fetched now) or 'bundled' (shipped snapshot)
    fetched_at: Optional[str] = None  # UTC timestamp of a live fetch
    history: Dict[dt.date, Dict[str, float]] = field(default_factory=dict, repr=False, compare=False)

    def cross(self, pair: str | CurrencyPair) -> float:
        """Reference rate of ``pair`` (quote currency per 1 base) on ``asof``."""

        pair = get_pair(pair)
        rates = {"EUR": 1.0, **self.per_eur}
        missing = [c for c in (pair.base, pair.quote) if c not in rates]
        if missing:
            raise FxDataError(f"the ECB rates of {self.asof} have no {', '.join(missing)} rate")
        return rates[pair.quote] / rates[pair.base]

    @property
    def description(self) -> str:
        when = "bundled snapshot" if self.origin == "bundled" else f"fetched {self.fetched_at}"
        return f"{ECB_SOURCE}, {self.asof.isoformat()} ({when}); cross via EUR; indicative, not dealable"


def parse_ecb_csv(text: str, origin: str = "live", currencies=ECB_CURRENCIES) -> EcbRates:
    """Parse a Data Portal ``csvdata`` response (columns CURRENCY,
    CURRENCY_DENOM, TIME_PERIOD, OBS_VALUE). ``asof`` is the latest date on
    which every currency in ``currencies`` has a rate. Raises
    :class:`FxDataError` on a malformed or incomplete file."""

    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except csv.Error as exc:
        raise FxDataError(f"ECB response is not CSV: {exc}") from None
    need = {"CURRENCY", "CURRENCY_DENOM", "TIME_PERIOD", "OBS_VALUE"}
    if not rows or not need <= set(rows[0]):
        raise FxDataError(f"ECB response has no rates (expected columns {sorted(need)})")
    by_date: Dict[dt.date, Dict[str, float]] = {}
    for row in rows:
        if row["CURRENCY_DENOM"] != "EUR" or not row["OBS_VALUE"]:
            continue
        try:
            day = dt.date.fromisoformat(row["TIME_PERIOD"])
            value = float(row["OBS_VALUE"])
        except ValueError:
            raise FxDataError(
                f"ECB row has a bad date or value: {row['TIME_PERIOD']!r}, {row['OBS_VALUE']!r}"
            ) from None
        if not value > 0:
            raise FxDataError(f"ECB rate {row['CURRENCY']} on {day} is not positive: {value}")
        by_date.setdefault(day, {})[row["CURRENCY"]] = value
    complete = [d for d, rates in by_date.items() if set(currencies) <= set(rates)]
    if not complete:
        raise FxDataError(f"no date in the ECB response has rates for all of {', '.join(currencies)}")
    asof = max(complete)
    fetched = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if origin == "live" else None
    return EcbRates(asof, dict(by_date[asof]), origin, fetched, by_date)


def fetch_ecb_rates(timeout: float = DEFAULT_TIMEOUT, observations: int = 5) -> EcbRates:
    """Latest ECB reference rates (``observations`` days back, so a missing
    day for one currency still leaves a complete date). Raises
    :class:`FxDataError` with the reason on any network or format error."""

    url = ECB_URL.format(codes="+".join(ECB_CURRENCIES), n=int(observations))
    request = urllib.request.Request(url, headers={"Accept": "text/csv", "User-Agent": "derivx-fx/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise FxDataError(f"ECB Data Portal returned HTTP {exc.code} ({exc.reason})") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise FxDataError(f"ECB Data Portal not reachable: {reason}") from None
    return parse_ecb_csv(text, origin="live")


def bundled_ecb_rates() -> EcbRates:
    """The ECB rates shipped with the app (``data/examples``)."""

    try:
        text = BUNDLED_CSV.read_text(encoding="utf-8")
    except OSError as exc:
        raise FxDataError(f"bundled ECB snapshot not found at {BUNDLED_CSV.name}: {exc.strerror}") from None
    return parse_ecb_csv(text, origin="bundled")
