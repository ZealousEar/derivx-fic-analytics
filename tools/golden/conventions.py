"""QuantLib reference values for calendars, day counts, schedules and
date arithmetic (tests/test_conventions.py)."""

from __future__ import annotations

import datetime as dt
import random

import QuantLib as ql

from .common import BDC, CALENDARS, DAY_COUNTS, pyd, qd

UNITS = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}


def generate() -> dict:
    rng = random.Random(20261019)
    holidays = {
        name: [
            pyd(d).isoformat()
            for d in ql.Calendar.holidayList(cal, qd(dt.date(1990, 1, 1)), qd(dt.date(2080, 12, 31)), False)
        ]
        for name, cal in CALENDARS.items()
    }

    day_counts = []
    for _ in range(200):
        d1 = dt.date(1995, 1, 1) + dt.timedelta(days=rng.randrange(0, 20000))
        d2 = d1 + dt.timedelta(days=rng.randrange(0, 12000))
        for name, dc in DAY_COUNTS.items():
            day_counts.append([d1.isoformat(), d2.isoformat(), name, dc.yearFraction(qd(d1), qd(d2))])

    schedules = []
    while len(schedules) < 400:
        cal = rng.choice(sorted(CALENDARS))
        eff = dt.date(2000, 1, 1) + dt.timedelta(days=rng.randrange(0, 15000))
        if rng.random() < 0.4:  # month ends exercise the end-of-month rules
            nxt = (eff.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
            eff = nxt - dt.timedelta(days=1 + rng.randrange(0, 3))
        months = rng.choice([1, 3, 6, 12])
        days = rng.choice([30, 90, 181, 365, 400, 731, 1826, 3652, 10957])
        if days // (31 * months) > 40:  # keep the fixture small: at most ~40 periods
            continue
        term = eff + dt.timedelta(days=days + rng.choice([0, 1, -1, 3]))
        conv, tconv = rng.choice(sorted(BDC)), rng.choice(sorted(BDC))
        eom, backward = rng.random() < 0.5, rng.random() < 0.7
        rule = ql.DateGeneration.Backward if backward else ql.DateGeneration.Forward
        s = ql.Schedule(
            qd(eff), qd(term), ql.Period(months, ql.Months), CALENDARS[cal], BDC[conv], BDC[tconv], rule, eom
        )
        schedules.append(
            {
                "calendar": cal,
                "effective": eff.isoformat(),
                "termination": term.isoformat(),
                "months": months,
                "bdc": conv,
                "termination_bdc": tconv,
                "eom": eom,
                "backward": backward,
                "dates": [pyd(d).isoformat() for d in s],
            }
        )

    advances = []
    for _ in range(600):
        cal = rng.choice(sorted(CALENDARS))
        day = dt.date(2000, 1, 1) + dt.timedelta(days=rng.randrange(0, 15000))
        tenor = rng.choice(["2D", "1W", "2W", "1M", "3M", "6M", "1Y", "2Y", "10Y", "18M"])
        conv, eom = rng.choice(sorted(BDC)), rng.random() < 0.5
        n, unit = int(tenor[:-1]), tenor[-1]
        out = CALENDARS[cal].advance(qd(day), n, UNITS[unit], BDC[conv], eom)
        advances.append([cal, day.isoformat(), tenor, conv, eom, pyd(out).isoformat()])

    return {
        "holiday_range": ["1990-01-01", "2080-12-31"],
        "holidays": holidays,
        "day_counts": day_counts,
        "schedules": schedules,
        "advances": advances,
    }
