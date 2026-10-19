"""Market conventions: tenors, day counts, holiday calendars and schedules.

Calendars and day counts follow the QuantLib definitions so results can be
checked against it (see ``tests/test_conventions.py``):

* ``USGOV``  - US government bond market (SIFMA), QuantLib ``GovernmentBond``
* ``USD``    - SOFR fixing calendar, QuantLib ``UnitedStates.SOFR``
* ``TARGET`` - euro area TARGET2
* ``LONDON`` - UK settlement
* ``WEEKENDS`` - Saturdays and Sundays only

Joint calendars are written ``"USD+LONDON"``.
"""

from __future__ import annotations

import calendar as _calendar
import datetime as dt
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet, List, Optional, Sequence

Date = dt.date

DAY_COUNTS = ("ACT/360", "ACT/365F", "30/360", "30E/360", "ACT/ACT ISDA", "ACT/ACT ICMA")
BUSINESS_DAY_CONVENTIONS = ("F", "MF", "P", "MP", "U")


# --------------------------------------------------------------------------
# Tenors
# --------------------------------------------------------------------------

_SPECIAL_TENORS = {"ON": "today", "O/N": "today", "TN": "tomorrow", "T/N": "tomorrow", "SN": "spot", "S/N": "spot"}
_UNIT_WORDS = {
    "D": "D",
    "DAY": "D",
    "DAYS": "D",
    "W": "W",
    "WK": "W",
    "WEEK": "W",
    "WEEKS": "W",
    "M": "M",
    "MO": "M",
    "MON": "M",
    "MONTH": "M",
    "MONTHS": "M",
    "Y": "Y",
    "YR": "Y",
    "YEAR": "Y",
    "YEARS": "Y",
}
_TENOR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")


@dataclass(frozen=True)
class Tenor:
    """A period such as 3M or 10Y.

    ``unit`` is ``D`` (business days), ``W``, ``M`` or ``Y``. Overnight-style
    tenors (ON, TN, SN) are one business day with ``anchor`` saying where the
    period starts: ``today``, ``tomorrow`` or ``spot``.
    """

    n: int
    unit: str
    anchor: str = "spot"
    label: str = ""

    @property
    def years(self) -> float:
        """Approximate length in years, for sorting and display only."""

        per_unit = {"D": 1 / 365.0, "W": 7 / 365.0, "M": 1 / 12.0, "Y": 1.0}
        return self.n * per_unit[self.unit]

    @property
    def months(self) -> Optional[int]:
        if self.unit == "M":
            return self.n
        if self.unit == "Y":
            return 12 * self.n
        return None

    def __str__(self) -> str:
        return self.label or f"{self.n}{self.unit}"


def parse_tenor(text: str) -> Tenor:
    """Parse market tenor labels: ``ON``, ``TN``, ``SN``, ``2D``, ``1W``,
    ``3M``, ``18M``, ``10Y``, ``1.5Y``, Treasury-style ``1 Mo`` / ``30 Yr``.

    Raises ``ValueError`` with a readable message for anything else.
    """

    raw = str(text).strip()
    key = raw.upper().replace(" ", "")
    if key in _SPECIAL_TENORS:
        return Tenor(1, "D", anchor=_SPECIAL_TENORS[key], label=key.replace("/", ""))
    match = _TENOR_RE.match(raw)
    if not match:
        raise ValueError(f"Unrecognised tenor {text!r}: use ON/TN/SN, 2D, 1W, 3M, 10Y or '1 Mo'")
    value, word = float(match.group(1)), match.group(2).upper()
    unit = _UNIT_WORDS.get(word)
    if unit is None:
        raise ValueError(f"Unrecognised tenor unit in {text!r}")
    if value <= 0:
        raise ValueError(f"Tenor must be positive: {text!r}")
    if value != int(value):
        if unit == "Y" and (value * 12).is_integer():  # 1.5Y -> 18M
            return Tenor(int(value * 12), "M", label=f"{int(value * 12)}M")
        if unit == "M" and value == 1.5:  # Treasury "1.5 Month" is the 6-week bill
            return Tenor(6, "W", label="6W")
        raise ValueError(f"Fractional tenor {text!r} is not a whole number of months")
    n = int(value)
    return Tenor(n, unit, label=f"{n}{unit}")


# --------------------------------------------------------------------------
# Day counts
# --------------------------------------------------------------------------


def _thirty_360(d1: Date, d2: Date, european: bool) -> float:
    day1, day2 = d1.day, d2.day
    if european:
        day1, day2 = min(day1, 30), min(day2, 30)
    else:  # US bond basis (ISDA 30/360)
        day1 = min(day1, 30)
        if day1 == 30:
            day2 = min(day2, 30)
    days = 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (day2 - day1)
    return days / 360.0


def _act_act_isda(d1: Date, d2: Date) -> float:
    if d1 == d2:
        return 0.0
    if d1 > d2:
        return -_act_act_isda(d2, d1)
    total, start = 0.0, d1
    while start.year < d2.year:
        year_end = Date(start.year + 1, 1, 1)
        total += (year_end - start).days / (366.0 if _calendar.isleap(start.year) else 365.0)
        start = year_end
    return total + (d2 - start).days / (366.0 if _calendar.isleap(d2.year) else 365.0)


def year_fraction(
    d1: Date,
    d2: Date,
    day_count: str,
    ref_start: Optional[Date] = None,
    ref_end: Optional[Date] = None,
    frequency: Optional[int] = None,
) -> float:
    """Year fraction between ``d1`` and ``d2``.

    ``ACT/ACT ICMA`` needs the reference (regular coupon) period and the
    coupon ``frequency`` per year; the period must lie inside the reference
    period, as it does for a regular coupon or an accrual to settlement.
    """

    dc = day_count.upper()
    if dc == "ACT/360":
        return (d2 - d1).days / 360.0
    if dc in ("ACT/365F", "ACT/365"):
        return (d2 - d1).days / 365.0
    if dc == "30/360":
        return _thirty_360(d1, d2, european=False)
    if dc == "30E/360":
        return _thirty_360(d1, d2, european=True)
    if dc == "ACT/ACT ISDA":
        return _act_act_isda(d1, d2)
    if dc == "ACT/ACT ICMA":
        if ref_start is None or ref_end is None or not frequency:
            raise ValueError("ACT/ACT ICMA needs ref_start, ref_end and frequency")
        return (d2 - d1).days / (frequency * (ref_end - ref_start).days)
    raise ValueError(f"Unknown day count {day_count!r}; expected one of {DAY_COUNTS}")


# --------------------------------------------------------------------------
# Calendars
# --------------------------------------------------------------------------


def easter_sunday(year: int) -> Date:
    """Gregorian Easter (anonymous algorithm)."""

    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return Date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> Date:
    """n-th (1-based) ``weekday`` (Mon=0) of a month; n=-1 is the last."""

    if n > 0:
        first = Date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = Date(year, month, _calendar.monthrange(year, month)[1])
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed_sat_fri_sun_mon(day: Date) -> Date:
    if day.weekday() == 5:
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


def _us_holidays(year: int, kind: str) -> set:
    hols = set()
    new_year = Date(year, 1, 1)
    hols.add(new_year if new_year.weekday() != 6 else Date(year, 1, 2))
    if year >= 1983:
        hols.add(_nth_weekday(year, 1, 0, 3))  # Martin Luther King Jr. Day
    hols.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    good_friday = easter_sunday(year) - dt.timedelta(days=2)
    # Since 1996 the bond market only closes early on a Good Friday that is
    # also payrolls day (the first Friday of the month); SOFR is not fixed.
    if kind == "SOFR" or year < 1996 or good_friday.day > 7:
        hols.add(good_friday)
    hols.add(_nth_weekday(year, 5, 0, -1))  # Memorial Day
    if year >= 2022:
        hols.add(_observed_sat_fri_sun_mon(Date(year, 6, 19)))  # Juneteenth
    hols.add(_observed_sat_fri_sun_mon(Date(year, 7, 4)))  # Independence Day
    hols.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    if year >= 1971:
        hols.add(_nth_weekday(year, 10, 0, 2))  # Columbus Day
    veterans = Date(year, 11, 11)  # Monday if Sunday, never moved to Friday
    hols.add(veterans + dt.timedelta(days=1) if veterans.weekday() == 6 else veterans)
    hols.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    hols.add(_observed_sat_fri_sun_mon(Date(year, 12, 25)))  # Christmas
    special = {
        2004: [Date(2004, 6, 11)],  # President Reagan's funeral
        2012: [Date(2012, 10, 30)],  # Hurricane Sandy
        2018: [Date(2018, 12, 5)],  # President Bush's funeral
    }
    hols.update(special.get(year, []))
    return hols


def _target_holidays(year: int) -> set:
    easter = easter_sunday(year)
    hols = {Date(year, 1, 1), Date(year, 12, 25)}
    if year >= 2000:
        hols.update({easter - dt.timedelta(days=2), easter + dt.timedelta(days=1)})
        hols.update({Date(year, 5, 1), Date(year, 12, 26)})
    if year in (1998, 1999, 2001):
        hols.add(Date(year, 12, 31))
    return hols


def _london_holidays(year: int) -> set:
    easter = easter_sunday(year)
    new_year = Date(year, 1, 1)
    hols = {
        new_year + dt.timedelta(days={5: 2, 6: 1}.get(new_year.weekday(), 0)),
        easter - dt.timedelta(days=2),
        easter + dt.timedelta(days=1),
        _nth_weekday(year, 8, 0, -1),  # summer bank holiday
    }
    early_may = {1995: Date(1995, 5, 8), 2020: Date(2020, 5, 8)}
    hols.add(early_may.get(year, _nth_weekday(year, 5, 0, 1)))
    spring = {2002: Date(2002, 6, 4), 2012: Date(2012, 6, 4), 2022: Date(2022, 6, 2)}
    hols.add(spring.get(year, _nth_weekday(year, 5, 0, -1)))
    christmas = Date(year, 12, 25)
    if christmas.weekday() == 5:  # Sat: Mon 27 and Tue 28
        hols.update({Date(year, 12, 27), Date(year, 12, 28)})
    elif christmas.weekday() == 6:  # Sun: Mon 26 (Boxing Day) and Tue 27
        hols.update({Date(year, 12, 26), Date(year, 12, 27)})
    elif christmas.weekday() == 4:  # Fri: Boxing Day moves to Mon 28
        hols.update({christmas, Date(year, 12, 28)})
    else:
        hols.update({christmas, Date(year, 12, 26)})
    special = {
        1999: [Date(1999, 12, 31)],
        2002: [Date(2002, 6, 3)],  # Golden Jubilee
        2011: [Date(2011, 4, 29)],  # Royal Wedding
        2012: [Date(2012, 6, 5)],  # Diamond Jubilee
        2022: [Date(2022, 6, 3), Date(2022, 9, 19)],  # Platinum Jubilee, State Funeral
        2023: [Date(2023, 5, 8)],  # Coronation
    }
    hols.update(special.get(year, []))
    return hols


_RULES = {
    "USGOV": lambda y: _us_holidays(y, "GOV"),
    "USD": lambda y: _us_holidays(y, "SOFR"),
    "TARGET": _target_holidays,
    "LONDON": _london_holidays,
    "WEEKENDS": lambda y: set(),
}


@lru_cache(maxsize=None)
def _holidays(name: str, year: int) -> FrozenSet[Date]:
    return frozenset(_RULES[name](year))


class Calendar:
    """Business-day calendar (weekends plus named holiday rules)."""

    def __init__(self, name: str):
        parts = [p.strip().upper() for p in name.split("+") if p.strip()]
        unknown = [p for p in parts if p not in _RULES]
        if not parts or unknown:
            raise ValueError(f"Unknown calendar {name!r}; expected {sorted(_RULES)} or a '+' join")
        self.name = "+".join(parts)
        self._parts = tuple(parts)

    def __repr__(self) -> str:
        return f"Calendar({self.name!r})"

    def is_holiday(self, day: Date) -> bool:
        return any(day in _holidays(p, day.year) for p in self._parts)

    def is_business_day(self, day: Date) -> bool:
        return day.weekday() < 5 and not self.is_holiday(day)

    def holidays(self, start: Date, end: Date) -> List[Date]:
        """Weekday holidays in [start, end]."""

        out = []
        day = start
        while day <= end:
            if day.weekday() < 5 and self.is_holiday(day):
                out.append(day)
            day += dt.timedelta(days=1)
        return out

    def adjust(self, day: Date, bdc: str = "F") -> Date:
        bdc = bdc.upper()
        if bdc == "U" or self.is_business_day(day):
            return day
        step = -1 if bdc in ("P", "MP") else 1
        out = day
        while not self.is_business_day(out):
            out += dt.timedelta(days=step)
        if bdc in ("MF", "MP") and out.month != day.month:
            return self.adjust(day, "P" if bdc == "MF" else "F")
        return out

    def add_business_days(self, day: Date, n: int) -> Date:
        out, step = day, (1 if n >= 0 else -1)
        if n == 0:
            return self.adjust(day, "F")
        for _ in range(abs(n)):
            out += dt.timedelta(days=step)
            while not self.is_business_day(out):
                out += dt.timedelta(days=step)
        return out

    def business_days_between(self, start: Date, end: Date) -> int:
        """Business days in [start, end)."""

        count, day = 0, start
        while day < end:
            count += self.is_business_day(day)
            day += dt.timedelta(days=1)
        return count

    def advance(self, day: Date, tenor: Tenor | str, bdc: str = "F", eom: bool = False) -> Date:
        """Add a tenor: business days for ``D``; calendar periods for W/M/Y,
        then adjusted with ``bdc`` (end-of-month rule for M/Y if ``eom``)."""

        tenor = parse_tenor(tenor) if isinstance(tenor, str) else tenor
        if tenor.unit == "D":
            return self.add_business_days(day, tenor.n)
        if tenor.unit == "W":
            return self.adjust(day + dt.timedelta(weeks=tenor.n), bdc)
        out = add_months(day, tenor.months)
        if eom:
            if bdc.upper() == "U" and day.day == _calendar.monthrange(day.year, day.month)[1]:
                return add_months(day, tenor.months, eom=True)
            if bdc.upper() != "U" and self.is_end_of_month(day):
                return self.end_of_month(out)
        return self.adjust(out, bdc)

    def is_end_of_month(self, day: Date) -> bool:
        """True if the next business day is in a later month (QuantLib's
        ``isEndOfMonth``; also true for a weekend after the last business day)."""

        return self.adjust(day + dt.timedelta(days=1), "F").month != day.month

    def end_of_month(self, day: Date) -> Date:
        """Last business day of the month containing ``day``."""

        return self.adjust(Date(day.year, day.month, _calendar.monthrange(day.year, day.month)[1]), "P")


def get_calendar(name: str | Calendar) -> Calendar:
    return name if isinstance(name, Calendar) else Calendar(name)


def add_months(day: Date, months: int, eom: bool = False) -> Date:
    """Calendar month arithmetic, clipping to month end (or snapping to it)."""

    month_index = day.month - 1 + months
    year, month = day.year + month_index // 12, month_index % 12 + 1
    last = _calendar.monthrange(year, month)[1]
    return Date(year, month, last if eom else min(day.day, last))


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------


def _null_advance_months(day: Date, months: int, eom: bool) -> Date:
    """Month arithmetic on a calendar with no holidays (QuantLib's
    ``NullCalendar.advance``): calendar month ends stay month ends."""

    is_month_end = day.day == _calendar.monthrange(day.year, day.month)[1]
    return add_months(day, months, eom=eom and is_month_end)


def make_schedule(
    effective: Date,
    termination: Date,
    months: int,
    calendar: str | Calendar,
    bdc: str = "MF",
    termination_bdc: Optional[str] = None,
    backward: bool = True,
    eom: bool = False,
) -> List[Date]:
    """Adjusted schedule dates from ``effective`` to ``termination``.

    Mirrors QuantLib's ``Schedule`` with ``DateGeneration.Backward`` (front
    stub) or ``Forward`` (back stub): dates are generated every ``months`` from
    the unadjusted seed date, then adjusted with ``bdc`` (``termination_bdc``
    for the last date). With ``eom`` and a seed on the last business day of
    its month, inner dates go to month ends.
    """

    if termination <= effective:
        raise ValueError("termination must be after effective")
    if months <= 0:
        raise ValueError("months must be positive")
    cal = get_calendar(calendar)
    termination_bdc = termination_bdc or bdc
    seed = termination if backward else effective
    dates: List[Date] = [seed]
    k = 1
    while True:
        day = _null_advance_months(seed, -months * k if backward else months * k, eom)
        if (backward and day < effective) or (not backward and day > termination):
            break
        if cal.adjust(day, bdc) != cal.adjust(dates[-1], bdc):  # skip duplicates
            dates.append(day)
        k += 1
    if backward:
        if cal.adjust(dates[-1], bdc) != cal.adjust(effective, bdc):
            dates.append(effective)
        dates.reverse()
    elif cal.adjust(dates[-1], termination_bdc) != cal.adjust(termination, termination_bdc):
        dates.append(termination)

    snap = eom and cal.is_end_of_month(seed)

    def inner_date(day: Date) -> Date:
        if snap:  # month end, then adjusted (lands on the last business day)
            day = Date(day.year, day.month, _calendar.monthrange(day.year, day.month)[1])
        return cal.adjust(day, bdc)

    dates = [cal.adjust(dates[0], bdc), *map(inner_date, dates[1:-1]), cal.adjust(dates[-1], termination_bdc)]
    # Adjustment can push a stub date onto (or past) its neighbour.
    if len(dates) > 2 and dates[-2] >= dates[-1]:
        dates[-2:] = [dates[-1]]
    if len(dates) > 2 and dates[1] <= dates[0]:
        dates[:2] = [dates[0]]
    return dates


def periods(dates: Sequence[Date]) -> List[tuple]:
    """Consecutive (start, end) pairs of a schedule."""

    return list(zip(dates[:-1], dates[1:]))
