"""FX market conventions: currency pairs, spot, expiry and delivery dates,
forwards and forward points.

Terminology
-----------
A pair ``BASEQUOTE`` (EURUSD = EUR/USD) is the price of one unit of the
*base* currency (CCY1) in units of the *quote* currency (CCY2). In
Garman-Kohlhagen terms the base currency is the *foreign* currency (rate
``rf``) and the quote currency the *domestic* one (rate ``rd``): option
values come out in quote-currency units per one unit of base-currency
notional. A pip is the pair's quoting unit (0.0001; 0.01 for JPY pairs);
prices are shown to a tenth of a pip (5 decimals, 3 for JPY pairs).

Dates (Clark, *Foreign Exchange Option Pricing*, 2011, ch. 1)
-------------------------------------------------------------
* **Spot date**: T+2 business days (USDCAD: T+1). For a pair with USD the
  first day only has to be a business day of the other currency (a USD
  holiday on T+1 does not delay spot); the spot date itself must be a
  business day of both. For a cross without USD every day must be a
  business day of both currencies, and a spot date that is a USD holiday
  rolls forward. Calendars per currency: :data:`CURRENCY_CALENDARS`
  (QuantLib-checked, see :mod:`derivx.rates.conventions`).
* **Option tenors** ON / nD / nW: the expiry is n business days (ON = 1) or n
  weeks after the trade date, rolled to the next business day of both
  currencies; the delivery date is the spot date of the expiry.
* **Option tenors** nM / nY: the delivery date is the spot date plus the
  tenor, modified following, end-of-month rule (a spot date on the last
  business day of its month delivers on the last business day of the
  target month), on the settlement calendar (both currencies, plus USD for
  a cross); the expiry is the latest business day whose spot date is not
  after that delivery date, and the delivery date is the spot date of that
  expiry (the same date unless holidays intervene).
* **Year fractions**, ACT/365F: the vol time ``tau_e`` runs from the trade
  date to the expiry date; the rate time ``tau_d`` from the spot date (when
  the premium is paid) to the delivery date.

Forwards
--------
``F = S exp((rd - rf) tau_d)`` with continuously compounded ACT/365F rates,
or ``F = S + points * pip`` from quoted forward points (then ``rf`` is
implied from ``rd``). Forward points are ``(F - S) / pip``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from derivx.rates.conventions import Calendar, Tenor, parse_tenor

Date = dt.date
DAYS_PER_YEAR = 365.0  # ACT/365F

CURRENCY_CALENDARS: Dict[str, str] = {
    "USD": "USFED",
    "EUR": "TARGET",
    "GBP": "LONDON",
    "JPY": "TOKYO",
    "CHF": "ZURICH",
    "CAD": "TORONTO",
    "AUD": "SYDNEY",
    "NZD": "AUCKLAND+WELLINGTON",
}

OPTION_TENORS = ("ON", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y")
DELTA_TYPES = ("spot", "forward", "spot_pa", "forward_pa")
DELTA_TYPE_LABELS = {
    "spot": "SPOT DELTA",
    "forward": "FORWARD DELTA",
    "spot_pa": "SPOT DELTA, PREMIUM-ADJUSTED",
    "forward_pa": "FORWARD DELTA, PREMIUM-ADJUSTED",
}
SPOT_DELTA_MAX_TENOR = "1Y"  # G10 market convention: spot delta up to and including 1Y, forward delta beyond


def currency_calendar(ccy: str) -> Calendar:
    """Settlement calendar of one currency (``ValueError`` if unknown)."""

    try:
        return Calendar(CURRENCY_CALENDARS[ccy.upper()])
    except KeyError:
        raise ValueError(
            f"no settlement calendar for {ccy!r}; known currencies: {sorted(CURRENCY_CALENDARS)}"
        ) from None


@dataclass(frozen=True)
class CurrencyPair:
    """A currency pair and its market conventions.

    ``premium_ccy`` is the market's default premium currency; a premium paid
    in the base (foreign) currency makes the quoted deltas premium-adjusted
    (USDJPY, USDCHF, USDCAD and the EUR crosses), a premium in the quote
    currency leaves them unadjusted (EURUSD, GBPUSD, AUDUSD, NZDUSD)."""

    base: str
    quote: str
    pip: float
    premium_ccy: str
    spot_lag: int = 2

    def __post_init__(self) -> None:
        for ccy in (self.base, self.quote):
            currency_calendar(ccy)
        if self.premium_ccy not in (self.base, self.quote):
            raise ValueError(f"premium currency must be {self.base} or {self.quote}, got {self.premium_ccy}")
        if self.spot_lag not in (1, 2):
            raise ValueError("spot lag must be 1 or 2 business days")

    @property
    def code(self) -> str:
        return self.base + self.quote

    @property
    def label(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def decimals(self) -> int:
        """Decimals shown for spot, strikes and forwards: a tenth of a pip."""

        return int(round(-np.log10(self.pip))) + 1

    @property
    def premium_adjusted(self) -> bool:
        """Market default: deltas are premium-adjusted when the premium is
        paid in the base (foreign) currency."""

        return self.premium_ccy == self.base

    @property
    def has_usd(self) -> bool:
        return "USD" in (self.base, self.quote)

    @property
    def calendar(self) -> Calendar:
        """Joint calendar of the two currencies (expiry dates)."""

        return Calendar(f"{CURRENCY_CALENDARS[self.base]}+{CURRENCY_CALENDARS[self.quote]}")

    @property
    def settlement_calendar(self) -> Calendar:
        """Joint calendar for value dates: both currencies, plus USD for a cross."""

        if self.has_usd:
            return self.calendar
        return Calendar(f"{self.calendar.name}+{CURRENCY_CALENDARS['USD']}")

    def fmt(self, x: float, extra: int = 0) -> str:
        """``x`` to a tenth of a pip (plus ``extra`` decimals)."""

        return f"{float(x):,.{self.decimals + extra}f}"


def _pair(base: str, quote: str, premium: str, lag: int = 2) -> CurrencyPair:
    return CurrencyPair(base, quote, 0.01 if quote == "JPY" else 0.0001, premium, lag)


PAIRS: Dict[str, CurrencyPair] = {
    p.code: p
    for p in (
        _pair("EUR", "USD", "USD"),
        _pair("GBP", "USD", "USD"),
        _pair("AUD", "USD", "USD"),
        _pair("NZD", "USD", "USD"),
        _pair("USD", "JPY", "USD"),
        _pair("USD", "CHF", "USD"),
        _pair("USD", "CAD", "USD", lag=1),
        _pair("EUR", "GBP", "EUR"),
        _pair("EUR", "JPY", "EUR"),
        _pair("EUR", "CHF", "EUR"),
    )
}


def get_pair(code: str | CurrencyPair) -> CurrencyPair:
    if isinstance(code, CurrencyPair):
        return code
    key = str(code).upper().replace("/", "").replace(" ", "")
    if key not in PAIRS:
        raise ValueError(f"unknown currency pair {code!r}; available: {', '.join(PAIRS)}")
    return PAIRS[key]


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def spot_date(pair: str | CurrencyPair, trade_date: Date) -> Date:
    """Spot (value) date of a trade on ``trade_date`` (module docs)."""

    pair = get_pair(pair)
    joint = pair.calendar
    if pair.spot_lag == 1:
        return joint.add_business_days(trade_date, 1)
    if pair.has_usd:
        other = currency_calendar(pair.quote if pair.base == "USD" else pair.base)
        first = other.add_business_days(trade_date, 1)  # a USD holiday on T+1 does not count
        return joint.add_business_days(first, pair.spot_lag - 1)
    day = joint.add_business_days(trade_date, pair.spot_lag)
    usd = currency_calendar("USD")
    while not (usd.is_business_day(day) and joint.is_business_day(day)):
        day += dt.timedelta(days=1)
    return day


def delivery_date(pair: str | CurrencyPair, expiry: Date) -> Date:
    """Delivery (settlement) date of an option expiring on ``expiry``: the
    spot date of the expiry date."""

    return spot_date(pair, expiry)


def _expiry_for_delivery(pair: CurrencyPair, delivery: Date) -> Date:
    """Latest business day (both currencies) whose spot date is not after
    ``delivery``."""

    cal = pair.calendar
    expiry = cal.add_business_days(delivery, -pair.spot_lag)
    while spot_date(pair, expiry) > delivery:
        expiry = cal.add_business_days(expiry, -1)
    while True:
        nxt = cal.add_business_days(expiry, 1)
        if spot_date(pair, nxt) > delivery:
            return expiry
        expiry = nxt


def expiry_and_delivery(pair: str | CurrencyPair, trade_date: Date, tenor: str | Tenor) -> Tuple[Date, Date]:
    """``(expiry, delivery)`` of an option tenor (ON, 1W, 1M, 1Y, ...) traded
    on ``trade_date`` (rules in the module docs)."""

    pair = get_pair(pair)
    t = parse_tenor(tenor) if isinstance(tenor, str) else tenor
    cal = pair.calendar
    if t.unit == "D":
        if t.anchor not in ("today", "spot") or str(t) in ("TN", "SN"):
            raise ValueError(f"option tenor {t} is not supported: use ON, nD, nW, nM or nY")
        expiry = cal.add_business_days(trade_date, t.n)
    elif t.unit == "W":
        expiry = cal.adjust(trade_date + dt.timedelta(weeks=t.n), "F")
    else:
        target = pair.settlement_calendar.advance(spot_date(pair, trade_date), t, "MF", eom=True)
        expiry = _expiry_for_delivery(pair, target)
    return expiry, delivery_date(pair, expiry)


@dataclass(frozen=True)
class FxDates:
    """The dates of one FX option and its two year fractions (ACT/365F):
    ``tau_e`` (vol time, trade date to expiry) and ``tau_d`` (rate time,
    spot date to delivery)."""

    pair: CurrencyPair
    trade: Date
    spot: Date
    expiry: Date
    delivery: Date
    tenor: Optional[str] = None

    @property
    def tau_e(self) -> float:
        return (self.expiry - self.trade).days / DAYS_PER_YEAR

    @property
    def tau_d(self) -> float:
        return max((self.delivery - self.spot).days, 0) / DAYS_PER_YEAR

    @property
    def days_to_expiry(self) -> int:
        return (self.expiry - self.trade).days

    @property
    def label(self) -> str:
        return self.tenor or self.expiry.isoformat()


def fx_dates(
    pair: str | CurrencyPair, trade_date: Date, tenor: Optional[str] = None, expiry: Optional[Date] = None
) -> FxDates:
    """Dates of an option given a tenor or an explicit expiry date (not both).
    Raises ``ValueError`` for an expiry before the trade date."""

    pair = get_pair(pair)
    if (tenor is None) == (expiry is None):
        raise ValueError("give either a tenor or an expiry date")
    if tenor is not None:
        expiry, delivery = expiry_and_delivery(pair, trade_date, tenor)
        label = str(parse_tenor(tenor))
    else:
        if expiry < trade_date:
            raise ValueError(f"expiry {expiry} is before the trade date {trade_date}")
        delivery = delivery_date(pair, expiry)
        label = None
    return FxDates(pair, trade_date, spot_date(pair, trade_date), expiry, delivery, label)


def market_delta_type(pair: str | CurrencyPair, dates: FxDates, premium_ccy: Optional[str] = None) -> str:
    """Delta convention a G10 desk quotes for this option: spot delta up to
    and including the 1Y expiry, forward delta beyond; premium-adjusted when
    the premium is paid in the base currency (``premium_ccy``, default the
    pair's market premium currency)."""

    pair = get_pair(pair)
    premium_ccy = premium_ccy or pair.premium_ccy
    if premium_ccy not in (pair.base, pair.quote):
        raise ValueError(f"premium currency must be {pair.base} or {pair.quote}")
    one_year, _ = expiry_and_delivery(pair, dates.trade, SPOT_DELTA_MAX_TENOR)
    kind = "spot" if dates.expiry <= one_year else "forward"
    return kind + ("_pa" if premium_ccy == pair.base else "")


# ---------------------------------------------------------------------------
# Forwards
# ---------------------------------------------------------------------------


def _finite(name: str, x: float) -> float:
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be a finite number, got {x}")
    return x


def forward_from_rates(S: float, rd: float, rf: float, tau_d: float) -> float:
    """``S exp((rd - rf) tau_d)``: rates continuously compounded, ACT/365F."""

    S, rd, rf, tau_d = (_finite(n, x) for n, x in (("spot", S), ("rd", rd), ("rf", rf), ("tau_d", tau_d)))
    if S <= 0 or tau_d < 0:
        raise ValueError("forward needs spot > 0 and tau_d >= 0")
    return S * float(np.exp((rd - rf) * tau_d))


def forward_points(S: float, F: float, pip: float) -> float:
    """Forward points in pips: ``(F - S) / pip``."""

    return (float(F) - float(S)) / float(pip)


def forward_from_points(S: float, points: float, pip: float) -> float:
    """Outright forward from forward points (pips): ``S + points * pip``.
    Raises ``ValueError`` when the forward is not positive."""

    F = _finite("spot", S) + _finite("forward points", points) * float(pip)
    if F <= 0:
        raise ValueError(f"spot {S} plus {points} pips gives a forward of {F:g} <= 0: check the sign and the units")
    return F


def implied_foreign_rate(S: float, F: float, rd: float, tau_d: float) -> float:
    """Foreign rate that reproduces the forward ``F`` given ``rd`` (covered
    interest parity, continuous ACT/365F): ``rd - ln(F / S) / tau_d``."""

    S, F, rd, tau_d = (_finite(n, x) for n, x in (("spot", S), ("forward", F), ("rd", rd), ("tau_d", tau_d)))
    if S <= 0 or F <= 0:
        raise ValueError("spot and forward must be > 0")
    if tau_d <= 0:
        raise ValueError("no rate time between spot and delivery: the forward equals spot, rf is undetermined")
    return rd - float(np.log(F / S)) / tau_d
