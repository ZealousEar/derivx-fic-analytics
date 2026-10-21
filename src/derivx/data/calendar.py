"""US equity-options trading calendar (NYSE / Cboe full-day closures).

Used to measure walk-forward horizons in trading days and to read the
session a Yahoo snapshot belongs to. Rules (pandas holiday rules, valid
from 2000): New Year's Day (Sunday -> Monday; a Saturday New Year is not
moved), Martin Luther King Jr. Day, Presidents' Day, Good Friday, Memorial
Day, Juneteenth (from 2022, nearest weekday), Independence Day (nearest
weekday), Labor Day, Thanksgiving and Christmas (nearest weekday).
Unscheduled closures (e.g. national days of mourning) and early closes
are not modelled.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

EXCHANGE_TZ = "America/New_York"

DateLike = Union[str, dt.date, dt.datetime, pd.Timestamp, np.datetime64]
ONE_DAY = np.timedelta64(1, "D")  # explicit unit: numpy >= 2.5 deprecates datetime64 + bare int


class NyseHolidayCalendar(AbstractHolidayCalendar):
    """Full-day NYSE holidays (see the module docs)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


@lru_cache(maxsize=1)
def _holidays() -> np.ndarray:
    days = NyseHolidayCalendar().holidays(start="2000-01-01", end="2040-12-31")
    return np.array(days.values.astype("datetime64[D]"))


def _day(value: DateLike) -> np.datetime64:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return np.datetime64(ts.normalize().date(), "D")


def holidays() -> List[dt.date]:
    """Exchange holidays from 2000 to 2040 (dates)."""

    return [pd.Timestamp(d).date() for d in _holidays()]


def is_trading_day(value: DateLike) -> bool:
    """True on a weekday that is not an exchange holiday."""

    return bool(np.is_busday(_day(value), holidays=_holidays()))


def trading_days_between(start: DateLike, end: DateLike) -> int:
    """Trading days from ``start`` (exclusive) to ``end`` (inclusive): 1
    from a Friday to the next Monday, 0 for the same day, negative when
    ``end`` is before ``start``."""

    a, b = _day(start), _day(end)
    if b >= a:
        return int(np.busday_count(a + ONE_DAY, b + ONE_DAY, holidays=_holidays()))
    return -int(np.busday_count(b + ONE_DAY, a + ONE_DAY, holidays=_holidays()))


def add_trading_days(start: DateLike, n: int) -> pd.Timestamp:
    """The ``n``-th trading day after ``start`` (``n`` >= 0), so that
    ``trading_days_between(start, add_trading_days(start, n)) == n`` also
    when ``start`` is a weekend or holiday (Saturday + 1 = Monday,
    Thanksgiving + 1 = the Friday after). With ``n = 0``: ``start`` itself
    if it is a trading day, else the next one."""

    if n < 0:
        raise ValueError("n must be >= 0")
    # roll a non-trading start back to the last session, then count forward
    day = np.busday_offset(_day(start), n, roll="backward" if n > 0 else "forward", holidays=_holidays())
    return pd.Timestamp(day)


def trading_days(start: DateLike, end: DateLike) -> List[pd.Timestamp]:
    """Trading days from ``start`` to ``end`` inclusive."""

    a, b = _day(start), _day(end)
    if b < a:
        return []
    days = np.arange(a, b + ONE_DAY, dtype="datetime64[D]")
    return [pd.Timestamp(d) for d in days[np.is_busday(days, holidays=_holidays())]]


def session_date(when: DateLike) -> pd.Timestamp:
    """New York calendar date of a time (naive times are taken as UTC)."""

    ts = pd.Timestamp(when)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return pd.Timestamp(ts.tz_convert(EXCHANGE_TZ).date())


def previous_trading_day(value: DateLike) -> pd.Timestamp:
    """The last trading day on or before ``value``."""

    day = np.busday_offset(_day(value), 0, roll="backward", holidays=_holidays())
    return pd.Timestamp(day)


def missing_trading_days(dates: Iterable[DateLike]) -> List[pd.Timestamp]:
    """Trading days between the first and last of ``dates`` that are not in
    ``dates`` (gaps in a daily history)."""

    have = sorted({_day(d) for d in dates})
    if len(have) < 2:
        return []
    present = set(have)
    return [d for d in trading_days(have[0], have[-1]) if np.datetime64(d.date(), "D") not in present]
