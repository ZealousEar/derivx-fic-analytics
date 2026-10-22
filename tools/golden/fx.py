"""QuantLib reference values for FX options (tests/test_fx_conventions.py,
tests/test_fx_pricing.py).

* ``holidays``: QuantLib holiday lists 1990-2080 of the calendars added for
  FX settlement (UnitedStates.FederalReserve, Japan, Switzerland,
  Canada.Settlement, Australia.Settlement, NewZealand Wellington and
  Auckland).
* ``joint_advance``: ``ql.JointCalendar(base, quote).advance(d, lag, Days)``
  per pair, for every calendar day of 2025-2027 as the trade date (weekends
  and holidays included), stored as days after the trade date.
* ``spot_dates`` / ``tenor_dates``: spot, expiry and delivery dates by the
  market rules of derivx.fx.conventions, computed here independently with
  QuantLib calendars and date arithmetic (``ql.JointCalendar``,
  ``Calendar.advance`` with ModifiedFollowing and the end-of-month rule).
* ``gk``: ``ql.GarmanKohlagenProcess`` + ``ql.AnalyticEuropeanEngine`` on
  flat continuously compounded ACT/365F curves (one time ``T`` for vol and
  rates): NPV, delta, gamma, vega, rho, dividendRho (the foreign rho) and the
  NPV one day later (evaluation date moved, curves anchored to it).
* ``black``: ``ql.blackFormula`` with the forward to delivery, the vol time
  and the domestic discount factor over the rate time (two times).
* ``delta_calc``: ``ql.BlackDeltaCalculator`` with distinct vol and rate
  times: deltas of four types for calls and puts, strikes from deltas
  (None where QuantLib finds no strike) and ATM strikes (AtmFwd, AtmSpot,
  AtmDeltaNeutral).
"""

from __future__ import annotations

import datetime as dt
import math
import random
from typing import Dict, List, Optional

import QuantLib as ql

from golden.common import pyd, qd, set_evaluation_date

EVAL = dt.date(2026, 10, 20)
DATES_START, DATES_DAYS = dt.date(2025, 1, 1), 3 * 365 + 1  # 2025-2027, every calendar day
FX_CALENDARS = {
    "USFED": ql.UnitedStates(ql.UnitedStates.FederalReserve),
    "TOKYO": ql.Japan(),
    "ZURICH": ql.Switzerland(),
    "TORONTO": ql.Canada(ql.Canada.Settlement),
    "SYDNEY": ql.Australia(ql.Australia.Settlement),
    "WELLINGTON": ql.NewZealand(ql.NewZealand.Wellington),
    "AUCKLAND": ql.NewZealand(ql.NewZealand.Auckland),
}
CCY = {
    "USD": FX_CALENDARS["USFED"],
    "EUR": ql.TARGET(),
    "GBP": ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
    "JPY": FX_CALENDARS["TOKYO"],
    "CHF": FX_CALENDARS["ZURICH"],
    "CAD": FX_CALENDARS["TORONTO"],
    "AUD": FX_CALENDARS["SYDNEY"],
    "NZD": ql.JointCalendar(FX_CALENDARS["AUCKLAND"], FX_CALENDARS["WELLINGTON"]),
}
PAIRS = {
    "EURUSD": ("EUR", "USD", 2),
    "GBPUSD": ("GBP", "USD", 2),
    "AUDUSD": ("AUD", "USD", 2),
    "NZDUSD": ("NZD", "USD", 2),
    "USDJPY": ("USD", "JPY", 2),
    "USDCHF": ("USD", "CHF", 2),
    "USDCAD": ("USD", "CAD", 1),
    "EURGBP": ("EUR", "GBP", 2),
    "EURJPY": ("EUR", "JPY", 2),
    "EURCHF": ("EUR", "CHF", 2),
}
TENORS = ("ON", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y")
DELTA_TYPES = {
    "spot": ql.DeltaVolQuote.Spot,
    "forward": ql.DeltaVolQuote.Fwd,
    "spot_pa": ql.DeltaVolQuote.PaSpot,
    "forward_pa": ql.DeltaVolQuote.PaFwd,
}
ATM_TYPES = {"ATMF": ql.DeltaVolQuote.AtmFwd, "SPOT": ql.DeltaVolQuote.AtmSpot, "DNS": ql.DeltaVolQuote.AtmDeltaNeutral}


def _joint(base: str, quote: str) -> ql.Calendar:
    return ql.JointCalendar(CCY[base], CCY[quote])


def _plus_days(day: ql.Date, n: int) -> ql.Date:
    return day + n


def ql_spot(code: str, trade: ql.Date) -> ql.Date:
    base, quote, lag = PAIRS[code]
    joint = _joint(base, quote)
    if lag == 1:
        return joint.advance(trade, 1, ql.Days)
    if "USD" in (base, quote):
        other = CCY[quote if base == "USD" else base]
        return joint.advance(other.advance(trade, 1, ql.Days), lag - 1, ql.Days)
    day = joint.advance(trade, lag, ql.Days)
    while not (CCY["USD"].isBusinessDay(day) and joint.isBusinessDay(day)):
        day = _plus_days(day, 1)
    return day


def ql_tenor_dates(code: str, trade: ql.Date, tenor: str):
    base, quote, lag = PAIRS[code]
    joint = _joint(base, quote)
    n, unit = (1, "D") if tenor == "ON" else (int(tenor[:-1]), tenor[-1])
    if unit == "D":
        expiry = joint.advance(trade, n, ql.Days)
    elif unit == "W":
        expiry = joint.adjust(_plus_days(trade, 7 * n), ql.Following)
    else:
        settle = joint if "USD" in (base, quote) else ql.JointCalendar(joint, CCY["USD"])
        period = ql.Period(n, ql.Months if unit == "M" else ql.Years)
        target = settle.advance(ql_spot(code, trade), period, ql.ModifiedFollowing, True)
        expiry = joint.advance(target, -lag, ql.Days)
        while ql_spot(code, expiry) > target:
            expiry = joint.advance(expiry, -1, ql.Days)
        while ql_spot(code, joint.advance(expiry, 1, ql.Days)) <= target:
            expiry = joint.advance(expiry, 1, ql.Days)
    return expiry, ql_spot(code, expiry)


def _flat(rate: float) -> ql.YieldTermStructureHandle:
    return ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), rate, ql.Actual365Fixed(), ql.Continuous))


def _gk_case(S: float, K: float, days: int, rd: float, rf: float, vol: float, call: bool) -> Dict[str, object]:
    set_evaluation_date(EVAL)
    process = ql.GarmanKohlagenProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        _flat(rf),
        _flat(rd),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.NullCalendar(), vol, ql.Actual365Fixed())),
    )
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, K)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(qd(EVAL + dt.timedelta(days=days))))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    out: Dict[str, object] = {
        "S": S,
        "K": K,
        "days": days,
        "rd": rd,
        "rf": rf,
        "sigma": vol,
        "call": call,
        "price": option.NPV(),
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega(),
        "rho": option.rho(),
        "rho_f": option.dividendRho(),
    }
    if days > 1:
        npv = option.NPV()
        set_evaluation_date(EVAL + dt.timedelta(days=1))
        out["theta_1d"] = option.NPV() - npv
        set_evaluation_date(EVAL)
    return out


# (spot, rd, rf, tau_e, tau_d, vol): FX levels, negative rates, rf > rd, short and long expiries
MARKETS = [
    (1.1215, 0.040, 0.020, 92 / 365, 94 / 365, 0.075),
    (1.08, 0.050, 0.030, 0.25, 0.25, 0.08),
    (150.2, 0.045, 0.001, 2.0, 2.0 + 3 / 365, 0.15),
    (0.6545, 0.040, 0.045, 1 / 365, 4 / 365, 0.12),
    (0.9237, -0.0075, 0.020, 1.0, 1.0 + 2 / 365, 0.06),
    (1.3, 0.000, 0.060, 5.0, 5.0, 0.30),
    (1.38, 0.030, 0.035, 7 / 365, 9 / 365, 0.25),
    (0.84, 0.040, 0.020, 0.5, 0.5, 0.05),
]


def _delta_case(S: float, rd: float, rf: float, te: float, td: float, vol: float) -> Dict[str, object]:
    dd, df = math.exp(-rd * td), math.exp(-rf * td)
    std = vol * math.sqrt(te)
    case: Dict[str, object] = {"S": S, "rd": rd, "rf": rf, "tau_e": te, "tau_d": td, "sigma": vol}
    strikes = [S * x for x in (0.8, 0.95, 1.0, 1.05, 1.25)]
    deltas, from_delta, atm = [], [], []
    for name, kind in DELTA_TYPES.items():
        for call in (True, False):
            calc = ql.BlackDeltaCalculator(ql.Option.Call if call else ql.Option.Put, kind, S, dd, df, std)
            for K in strikes:
                deltas.append([name, call, K, calc.deltaFromStrike(K)])
            for d in (0.05, 0.1, 0.25, 0.4, 0.45):
                target = d if call else -d
                strike: Optional[float]
                try:
                    strike = calc.strikeFromDelta(target)
                except RuntimeError:
                    strike = None
                from_delta.append([name, call, target, strike])
        calc = ql.BlackDeltaCalculator(ql.Option.Call, kind, S, dd, df, std)
        for atm_name, atm_kind in ATM_TYPES.items():
            atm.append([name, atm_name, calc.atmStrike(atm_kind)])
    case.update({"deltas": deltas, "strike_from_delta": from_delta, "atm": atm})
    return case


def generate() -> Dict[str, object]:
    rng = random.Random(20261020)
    holidays = {
        name: [pyd(d).isoformat() for d in cal.holidayList(qd(dt.date(1990, 1, 1)), qd(dt.date(2080, 12, 31)), False)]
        for name, cal in FX_CALENDARS.items()
    }
    # every calendar day from DATES_START for DATES_DAYS days: days from the trade date to the date
    joint_advance: Dict[str, List[int]] = {code: [] for code in PAIRS}
    spots: Dict[str, List[int]] = {code: [] for code in PAIRS}
    for i in range(DATES_DAYS):
        day = DATES_START + dt.timedelta(days=i)
        for code, (base, quote, lag) in PAIRS.items():
            joint_advance[code].append((pyd(_joint(base, quote).advance(qd(day), lag, ql.Days)) - day).days)
            spots[code].append((pyd(ql_spot(code, qd(day))) - day).days)
    tenor_dates = []
    for code in PAIRS:
        trades = [dt.date(2026, 10, 20), dt.date(2026, 12, 23), dt.date(2027, 4, 1), dt.date(2026, 2, 26)]
        trades += [dt.date(2025, 1, 1) + dt.timedelta(days=rng.randrange(0, 1095)) for _ in range(12)]
        for trade in trades:
            for tenor in TENORS:
                expiry, delivery = ql_tenor_dates(code, qd(trade), tenor)
                tenor_dates.append([code, trade.isoformat(), tenor, pyd(expiry).isoformat(), pyd(delivery).isoformat()])
    gk: List[Dict[str, object]] = []
    for S, markets in (
        (1.1215, [(0.040, 0.020, 0.075), (-0.005, 0.030, 0.12)]),
        (150.2, [(0.005, 0.045, 0.10), (0.045, 0.005, 0.15)]),
        (0.6545, [(0.040, 0.045, 0.11)]),
    ):
        for rd, rf, vol in markets:
            for days in (1, 7, 30, 91, 365, 730):
                for ratio in (0.9, 0.97, 1.0, 1.03, 1.1):
                    for call in (True, False):
                        gk.append(_gk_case(S, S * ratio, days, rd, rf, vol, call))
    black = []
    for S, rd, rf, te, td, vol in MARKETS:
        F = S * math.exp((rd - rf) * td)
        for ratio in (0.9, 1.0, 1.1):
            for call in (True, False):
                price = ql.blackFormula(
                    ql.Option.Call if call else ql.Option.Put, S * ratio, F, vol * math.sqrt(te), math.exp(-rd * td)
                )
                black.append(
                    {
                        "S": S,
                        "rd": rd,
                        "rf": rf,
                        "tau_e": te,
                        "tau_d": td,
                        "sigma": vol,
                        "K": S * ratio,
                        "call": call,
                        "price": price,
                    }
                )
    set_evaluation_date(EVAL)
    return {
        "holiday_range": ["1990-01-01", "2080-12-31"],
        "holidays": holidays,
        "dates_start": DATES_START.isoformat(),
        "joint_advance": joint_advance,
        "spot_dates": spots,
        "tenor_dates": tenor_dates,
        "gk": gk,
        "black": black,
        "delta_calc": [_delta_case(*mk) for mk in MARKETS],
    }
