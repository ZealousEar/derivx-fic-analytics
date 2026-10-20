"""Term rate indices, short-term rate futures and futures convexity.

**Indices.** A :class:`TermIndex` is an IBOR-style term rate fixed
``fixing_lag`` business days before its value date and running ``months``
from it (modified following, end-of-month rule) on its own day count. The
presets are QuantLib's ``ql.Euribor3M`` / ``ql.Euribor6M``: TARGET, T+2,
ACT/360, modified following, end of month. EURIBOR swaps pay an annual
30/360 fixed leg (bond basis) against the index; they are discounted on the
€STR curve (multi-curve), so each EURIBOR tenor has a projection curve of
its own (see :mod:`derivx.rates.multicurve`).

USD SOFR, €STR and SONIA OIS stay single-curve: since the LIBOR cessation
the overnight rate both discounts and projects those swaps, which is the
market standard. No USD term index is preset: CME Term SOFR swaps are thin
and have no free quotes or standard curve-building conventions, so a Term
SOFR projection curve would be guesswork.

**Futures.** Contracts are named by their reference month (``DEC26``,
``Z6``, ``Z26`` or ``2026-12``) and quoted as a price, 100 - rate:

* ``SR3`` - three-month SOFR: daily SOFR compounded from the third Wednesday
  of the contract month to the third Wednesday three months later;
* ``SR1`` - one-month SOFR: the arithmetic average of daily SOFR over the
  calendar month (weekend and holiday days take the previous fixing);
* ``FUT`` - three-month EURIBOR: the 3M EURIBOR fixing for a period that
  starts on the third Wednesday (IMM date) of the contract month and runs
  three months (modified following, no end-of-month rule), as QuantLib's
  ``FuturesRateHelper`` with ``ql.Euribor3M``.

The reference periods and the averaging/compounding match QuantLib's
``SofrFutureRateHelper`` / ``OvernightIndexFuture`` (``tests/test_multicurve.py``).

**Convexity.** Futures are margined daily, so the futures rate exceeds the
forward rate. :class:`HullWhite` (mean reversion ``a`` per year, normal
short-rate vol ``sigma`` in rate units per sqrt(year), 0.01 = 100bp) gives
the adjustment *futures rate - forward rate* (decimal):

* term-rate futures (``FUT``): the Kirikos-Novak (1997) formula QuantLib
  uses in ``HullWhite::convexityBias``, with ``t`` and ``T`` the ACT/365F
  times to the start and end of the rate period;
* overnight futures (``SR3``/``SR1``), whose rate is set over the reference
  period and paid at its end: the exact Gaussian result for the compounded
  (``1 + tau R = (1 + tau F) exp(V + C)``) or averaged
  (``R - F = (V / 2 + C) / tau``) rate, where ``V`` is the variance of the
  integrated short rate over the period and ``C`` its covariance with the
  integral up to the period start. Both tend to the term-futures value as
  the period shrinks.
"""

from __future__ import annotations

import calendar as _calendar
import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np

from .conventions import Calendar, Date, Tenor, add_months, year_fraction

FUTURES_KINDS = ("SR1", "SR3", "FUT")
FUTURES_POINT_VALUE = {"SR3": 2500.0, "SR1": 4167.0, "FUT": 2500.0}  # currency per 1.00 price point per contract
MONTH_CODES = "FGHJKMNQUVXZ"
MONTH_NAMES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
PRICE_RANGE = (70.0, 105.0)  # futures prices accepted (rates from -5% to +30%)


# --------------------------------------------------------------------------
# Term indices
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TermIndex:
    """A term rate index (IBOR style) and the swap conventions against it.

    ``fixing_lag`` is in business days of ``calendar`` between the fixing
    and the value date; the rate runs ``months`` from the value date
    (``bdc``, end-of-month rule ``eom``) on ``day_count``. ``discount`` is
    the OIS curve preset that discounts its swaps; ``swap_fixed_*`` are the
    fixed-leg conventions of a standard swap against it.
    """

    name: str
    currency: str
    months: int
    calendar: str
    fixing_lag: int
    day_count: str
    discount: str
    bdc: str = "MF"
    eom: bool = True
    swap_fixed_months: int = 12
    swap_fixed_day_count: str = "30/360"
    has_futures: bool = False

    @property
    def cal(self) -> Calendar:
        return Calendar(self.calendar)

    @property
    def tenor(self) -> Tenor:
        return Tenor(self.months, "M", label=f"{self.months}M")

    def spot(self, asof: Date) -> Date:
        """Value date of a fixing on ``asof`` (the next business day if
        ``asof`` is not one, then ``fixing_lag`` business days)."""

        cal = self.cal
        return cal.add_business_days(cal.adjust(asof, "F"), self.fixing_lag)

    def value_date(self, fixing: Date) -> Date:
        return self.cal.add_business_days(fixing, self.fixing_lag)

    def fixing_date(self, value: Date) -> Date:
        return self.cal.add_business_days(value, -self.fixing_lag)

    def maturity_date(self, value: Date) -> Date:
        """End of the rate period that starts on ``value``."""

        return self.cal.advance(value, self.tenor, self.bdc, eom=self.eom)

    def accrual(self, start: Date, end: Date) -> float:
        return year_fraction(start, end, self.day_count)


INDICES: Dict[str, TermIndex] = {
    "EURIBOR 3M": TermIndex("EURIBOR 3M", "EUR", 3, "TARGET", 2, "ACT/360", "EUR ESTR", has_futures=True),
    "EURIBOR 6M": TermIndex("EURIBOR 6M", "EUR", 6, "TARGET", 2, "ACT/360", "EUR ESTR"),
}
CURRENCY_INDICES: Dict[str, Tuple[str, ...]] = {"EUR": ("EURIBOR 6M", "EURIBOR 3M"), "USD": (), "GBP": ()}


def get_index(name: Union[str, TermIndex]) -> TermIndex:
    if isinstance(name, TermIndex):
        return name
    try:
        return INDICES[str(name).strip().upper()]
    except KeyError:
        raise ValueError(f"Unknown index {name!r}; expected one of {sorted(INDICES)}") from None


# --------------------------------------------------------------------------
# Futures contracts
# --------------------------------------------------------------------------


def imm_date(year: int, month: int) -> Date:
    """Third Wednesday of a month."""

    first = Date(year, month, 1)
    return first + dt.timedelta(days=(2 - first.weekday()) % 7 + 14)


_CODE_RE = re.compile(r"^([FGHJKMNQUVXZ])(\d{1,2})$")
_NAME_RE = re.compile(r"^([A-Z]{3})[-\s]?(\d{2}|\d{4})$")
_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})$")


def parse_contract(code: str, asof: Optional[Date] = None) -> Tuple[int, int]:
    """(year, month) of a futures contract named ``DEC26``, ``Dec 2026``,
    ``Z6``, ``Z26`` or ``2026-12``. A one-digit year is the first year with
    that last digit on or after ``asof``'s year (the next ten years)."""

    text = str(code).strip().upper()
    match = _CODE_RE.match(text)
    if match:
        month = MONTH_CODES.index(match.group(1)) + 1
        digits = match.group(2)
        if len(digits) == 2:
            year = 2000 + int(digits)
        else:
            if asof is None:
                raise ValueError(f"contract {code!r}: a one-digit year needs a valuation date; write {text[0]}2x")
            year = asof.year - asof.year % 10 + int(digits)
            if year < asof.year:
                year += 10
        return year, month
    match = _NAME_RE.match(text)
    if match and match.group(1) in MONTH_NAMES:
        digits = match.group(2)
        year = int(digits) if len(digits) == 4 else 2000 + int(digits)
        return year, MONTH_NAMES.index(match.group(1)) + 1
    match = _ISO_RE.match(text)
    if match and 1 <= int(match.group(2)) <= 12:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Unrecognised futures contract {code!r}: use DEC26, Z26 or 2026-12")


def contract_label(year: int, month: int) -> str:
    return f"{MONTH_NAMES[month - 1]}{year % 100:02d}"


def futures_period(kind: str, year: int, month: int, index: Optional[TermIndex] = None) -> Tuple[Date, Date]:
    """(start, end) of a contract's reference period (unadjusted for SR1,
    whose month may start or end on a weekend, as QuantLib's)."""

    kind = kind.upper()
    if kind == "SR3":
        start = imm_date(year, month)
        later = add_months(Date(year, month, 1), 3)
        return start, imm_date(later.year, later.month)
    if kind == "SR1":
        start = Date(year, month, 1)
        return start, Date(year, month, _calendar.monthrange(year, month)[1]) + dt.timedelta(days=1)
    if kind == "FUT":
        if index is None:
            raise ValueError("a term-rate future needs its index")
        start = imm_date(year, month)
        return start, index.cal.advance(start, index.tenor, index.bdc, eom=False)
    raise ValueError(f"Unknown futures kind {kind!r}; expected one of {FUTURES_KINDS}")


def futures_rate_from_price(price: float) -> float:
    """Futures rate (decimal) of a price quoted as 100 - rate in percent."""

    if not math.isfinite(price) or not PRICE_RANGE[0] <= price <= PRICE_RANGE[1]:
        raise ValueError(
            f"futures price {price!r} is outside {PRICE_RANGE[0]:g} to {PRICE_RANGE[1]:g}: quote futures as "
            "100 - rate (e.g. 96.25 for 3.75%)"
        )
    return (100.0 - price) / 100.0


def price_from_futures_rate(rate: float) -> float:
    return 100.0 * (1.0 - rate)


def overnight_futures_rate(
    kind: str,
    start: Date,
    end: Date,
    asof: Date,
    cal: Calendar,
    df,
    fixings: Optional[Mapping[Date, float]] = None,
    day_count: str = "ACT/360",
) -> float:
    """Rate an overnight future settles on, before convexity: SOFR
    compounded (``SR3``) or averaged (``SR1``) over [start, end), with
    fixings for days before ``asof`` (QuantLib's ``OvernightIndexFuture``).

    ``df(date)`` is the curve's discount factor. A fixing missing for a
    past business day raises ``ValueError``; a fixing for ``asof`` itself is
    used when present, otherwise the day is projected.
    """

    fixings = fixings or {}
    total = year_fraction(start, end, day_count)

    def past(day: Date) -> float:
        if day not in fixings:
            raise ValueError(
                f"{kind} reference period {start} to {end} has started: it needs the overnight fixing for {day}"
            )
        return float(fixings[day])

    if kind == "SR1":
        avg, d1 = 0.0, start
        fixing_day = cal.adjust(d1, "P")
        while d1 < end:
            d2 = cal.add_business_days(d1, 1)
            if fixing_day < asof:
                fwd = past(fixing_day)
            elif fixing_day == asof and fixing_day in fixings:
                fwd = float(fixings[fixing_day])
            else:
                fwd = (df(fixing_day) / df(d2) - 1.0) / year_fraction(fixing_day, d2, day_count)
            avg += fwd * year_fraction(d1, min(d2, end), day_count)
            fixing_day = d1 = d2
        return avg / total
    if kind != "SR3":
        raise ValueError(f"{kind} is not an overnight future")
    prod, forward_start = 1.0, start
    if asof > start:
        today = cal.adjust(asof, "F")
        forward_start = today
        d1, fixing_day = start, cal.adjust(start, "P")
        while d1 < today:
            d2 = cal.add_business_days(d1, 1)
            prod *= 1.0 + past(fixing_day) * year_fraction(d1, d2, day_count)
            fixing_day = d1 = d2
        if today < end and today in fixings:
            accrual_end = min(cal.add_business_days(today, 1), end)
            prod *= 1.0 + float(fixings[today]) * year_fraction(today, accrual_end, day_count)
            forward_start = accrual_end
    prod *= df(forward_start) / df(end)
    return (prod - 1.0) / total


# --------------------------------------------------------------------------
# Hull-White convexity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HullWhite:
    """One-factor Hull-White parameters for futures convexity:
    ``mean_reversion`` a (per year, >= 0) and ``sigma``, the normal vol of
    the short rate (rate units per sqrt(year): 0.0085 = 85bp)."""

    mean_reversion: float = 0.03
    sigma: float = 0.0085

    def __post_init__(self) -> None:
        for label, value in (("mean_reversion", self.mean_reversion), ("sigma", self.sigma)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Hull-White {label} must be a non-negative number, got {value!r}")
        if self.mean_reversion > 2.0:
            raise ValueError(f"Hull-White mean reversion {self.mean_reversion:g} per year is implausibly large")
        if self.sigma > 0.1:
            raise ValueError(f"Hull-White vol {self.sigma:g} is above 0.1 (1000bp): give it in rate units")


def _b(a: float, x: float) -> float:
    """B(x) = (1 - exp(-a x)) / a (x when a is zero)."""

    return x if a < np.finfo(float).eps else -math.expm1(-a * x) / a


def hw_term_futures_convexity(futures_rate: float, t: float, T: float, sigma: float, a: float) -> float:
    """Futures rate - forward rate of a term-rate future fixing at ``t``
    on [t, T] (years): QuantLib's ``HullWhite::convexityBias`` (Kirikos and
    Novak, 1997)."""

    if not (0.0 <= t <= T) or sigma < 0.0 or a < 0.0:
        raise ValueError(f"convexity needs 0 <= t <= T and non-negative a and sigma (t={t}, T={T})")
    delta = T - t
    b_delta = _b(a, delta)
    z = 0.5 * sigma * sigma * (_b(a, 2.0 * t) * b_delta + _b(a, t) ** 2)
    if delta < np.finfo(float).eps:
        return z
    return -math.expm1(-z * b_delta) * (futures_rate + 1.0 / delta)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def _integral(f, lo: float, hi: float) -> float:
    """Gauss-Legendre quadrature (exact to rounding for these smooth
    exponential integrands)."""

    if hi <= lo:
        return 0.0
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    return float(half * np.dot(_GL_W, f(mid + half * _GL_X)))


def hw_period_moments(t1: float, t2: float, sigma: float, a: float) -> Tuple[float, float]:
    """(V, C) for the short rate integrated over [t1, t2]: V its variance
    and C its covariance with the integral over [0, t1] (Hull-White, under
    the risk-neutral measure)."""

    if not 0.0 <= t1 <= t2:
        raise ValueError(f"need 0 <= t1 <= t2, got {t1}, {t2}")
    s2 = sigma * sigma
    b12 = _b(a, t2 - t1)

    def b_vec(x: np.ndarray) -> np.ndarray:
        return x if a < np.finfo(float).eps else -np.expm1(-a * x) / a

    # Shocks before t1 reach [t1, t2] damped by exp(-a (t1 - u)); later ones by B(u, t2).
    var = s2 * (
        b12 * b12 * _integral(lambda u: np.exp(-2.0 * a * (t1 - u)), 0.0, t1)
        + _integral(lambda u: b_vec(t2 - u) ** 2, t1, t2)
    )
    cov = s2 * b12 * _integral(lambda u: b_vec(t1 - u) * np.exp(-a * (t1 - u)), 0.0, t1)
    return var, cov


def hw_overnight_futures_convexity(
    futures_rate: float, t1: float, t2: float, sigma: float, a: float, averaging: str = "compound"
) -> float:
    """Futures rate - forward rate of an overnight future on [t1, t2]
    (ACT/365F years from today), settled on the rate set over the period:
    ``compound`` (SR3) or ``average`` (SR1). For a period already under way
    (``t1 < 0``) only the part still to come is random: the moments run
    from today, while the rate accrues over the whole period."""

    if not t2 > 0.0 or t2 <= t1 or sigma < 0.0 or a < 0.0:
        raise ValueError(f"convexity needs t2 > 0, t2 > t1 and non-negative a and sigma (t1={t1}, t2={t2})")
    tau = t2 - t1
    var, cov = hw_period_moments(max(t1, 0.0), t2, sigma, a)
    if averaging == "compound":
        return -math.expm1(-(var + cov)) * (futures_rate + 1.0 / tau)
    if averaging == "average":
        return (0.5 * var + cov) / tau
    raise ValueError(f"averaging must be 'compound' or 'average', got {averaging!r}")
