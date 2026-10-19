"""Helpers shared by the golden-fixture generators."""

from __future__ import annotations

import datetime as dt

import QuantLib as ql

from derivx.rates.curves import Curve


def qd(day: dt.date) -> ql.Date:
    return ql.Date(day.day, day.month, day.year)


def pyd(day: ql.Date) -> dt.date:
    return dt.date(day.year(), day.month(), day.dayOfMonth())


def set_evaluation_date(day: dt.date) -> None:
    ql.Settings.instance().evaluationDate = qd(day)


def ql_curve(curve: Curve) -> ql.YieldTermStructureHandle:
    """Exact QuantLib copy of a log-linear derivx curve: same pillar dates and
    discount factors on ACT/365F, log-linear in between, flat forward beyond."""

    if curve.interpolation != "loglinear" or curve.pillar_dates is None:
        raise ValueError("ql_curve mirrors bootstrapped log-linear curves only")
    dates = [qd(curve.asof)] + [qd(d) for d in curve.pillar_dates]
    ts = ql.DiscountCurve(dates, [float(x) for x in curve.dfs], ql.Actual365Fixed())
    ts.enableExtrapolation()
    return ql.YieldTermStructureHandle(ts)


DAY_COUNTS = {
    "ACT/360": ql.Actual360(),
    "ACT/365F": ql.Actual365Fixed(),
    "30/360": ql.Thirty360(ql.Thirty360.BondBasis),
    "30E/360": ql.Thirty360(ql.Thirty360.European),
    "ACT/ACT ISDA": ql.ActualActual(ql.ActualActual.ISDA),
}
CALENDARS = {
    "USGOV": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    "USD": ql.UnitedStates(ql.UnitedStates.SOFR),
    "TARGET": ql.TARGET(),
    "LONDON": ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
}
BDC = {"F": ql.Following, "MF": ql.ModifiedFollowing, "P": ql.Preceding, "MP": ql.ModifiedPreceding, "U": ql.Unadjusted}
