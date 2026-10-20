"""Fixed-rate government bonds: cash flows, yields, risk and spreads.

Schedule
    Coupon dates run backward from maturity every ``12 / frequency`` months on
    unadjusted dates, with the end-of-month rule for a bond maturing on the
    last day of a month (Aug 31 / Feb 28 notes pay on month ends). Payments
    move to the next business day. ``dated`` (the accrual start) gives an odd
    first coupon: short when it falls inside a regular period, long when
    ``first_coupon`` is set to a later regular date. Without ``dated`` the
    bond is regular back to any settlement date.

Day count
    ``ACT/ACT ICMA`` measures each stretch of days against the regular
    (quasi-)coupon period it falls in, so a regular coupon is exactly
    ``coupon / frequency`` and an odd one is the matching fraction of it.
    ``30/360`` and the other day counts use their plain year fraction.

Yield (street convention, the market quote)
    ``dirty = sum_k CF_k / (1 + y/f) ** (f * T_k)``, compounding at the coupon
    frequency ``f``, with ``T_k`` the time from settlement to cash flow ``k``
    in the bond's day count. For a regular bond ``f * T_k = w + k - 1`` where
    ``w`` is the fraction of the current coupon period still to run (a
    fractional first period). ``convention="street"`` discounts on the
    scheduled coupon dates, ignoring weekends and holidays;
    ``convention="true"`` discounts on the actual payment dates, as QuantLib's
    ``FixedRateBond`` does with a Following payment convention.

Curve pricing and z-spread
    Cash flows paid after settlement are discounted on a curve at their
    payment dates, relative to the settlement date. The z-spread is added to
    the curve's continuously compounded zero rates (ACT/365F from the curve
    date): ``DF_z(t) = DF(t) * exp(-z * t)``.

Prices are per ``face`` (100) and every risk number is signed P&L to the
holder: a long bond has a negative DV01.
"""

from __future__ import annotations

import calendar as _calendar
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from derivx.risk.sensitivities import Position

from .conventions import BUSINESS_DAY_CONVENTIONS, DAY_COUNTS, Calendar, Date, add_months, year_fraction
from .curves import Curve, Market, bill_bey, bill_price, get_conventions, swap_schedule

YIELD_CONVENTIONS = ("street", "true")
FREQUENCIES = (1, 2, 4, 12)
TREASURY = "USD UST"
SWAPS = "USD SOFR"


def _is_month_end(day: Date) -> bool:
    return day.day == _calendar.monthrange(day.year, day.month)[1]


def _solve(f: Callable[[float], float], brackets: Sequence[Tuple[float, float]], what: str) -> float:
    for lo, hi in brackets:
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi <= 0:
            return brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=300)
    raise ValueError(f"no {what} reproduces this price")


@dataclass(frozen=True)
class Coupon:
    """One coupon period: unadjusted accrual dates, payment date, accrual
    (year fraction) and amount per bond face."""

    start: Date
    end: Date
    pay: Date
    accrual: float
    amount: float


@dataclass(frozen=True)
class FixedRateBond:
    """A fixed-coupon bond; defaults are US Treasury note conventions.

    ``coupon`` is a decimal (0.0425 = 4.25%), ``face`` the price basis (100).
    ``eom`` defaults to True for a maturity on the last day of a month.
    """

    coupon: float
    maturity: Date
    dated: Optional[Date] = None
    first_coupon: Optional[Date] = None
    face: float = 100.0
    frequency: int = 2
    day_count: str = "ACT/ACT ICMA"
    calendar: str = "USGOV"
    settle_lag: int = 1
    payment_bdc: str = "F"
    eom: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.frequency not in FREQUENCIES:
            raise ValueError(f"frequency must be one of {FREQUENCIES} coupons a year")
        if self.day_count.upper() not in DAY_COUNTS:
            raise ValueError(f"Unknown day count {self.day_count!r}; expected one of {DAY_COUNTS}")
        if self.payment_bdc.upper() not in BUSINESS_DAY_CONVENTIONS:
            raise ValueError(f"payment_bdc must be one of {BUSINESS_DAY_CONVENTIONS}")
        Calendar(self.calendar)  # raises for an unknown calendar
        if not (math.isfinite(self.coupon) and 0.0 <= self.coupon < 1.0):
            raise ValueError("coupon must be a decimal in [0, 1), e.g. 0.0425 for 4.25%")
        if not (math.isfinite(self.face) and self.face > 0):
            raise ValueError("face must be positive")
        if self.settle_lag < 0:
            raise ValueError("settle_lag must be non-negative")
        if self.eom and not _is_month_end(self.maturity):
            raise ValueError("the end-of-month rule needs a maturity on the last day of a month")
        if self.dated is not None and self.dated >= self.maturity:
            raise ValueError("dated date must be before maturity")
        if self.first_coupon is not None:
            if self.dated is None:
                raise ValueError("first_coupon needs a dated date")
            if not self.dated < self.first_coupon <= self.maturity:
                raise ValueError("first_coupon must be after the dated date and no later than maturity")
            if self._regular(self._period_index(self.first_coupon)) != self.first_coupon:
                raise ValueError(f"first_coupon {self.first_coupon} is not a regular coupon date of this bond")

    # -- schedule --------------------------------------------------------------

    @property
    def months(self) -> int:
        return 12 // self.frequency

    @property
    def end_of_month(self) -> bool:
        return _is_month_end(self.maturity) if self.eom is None else self.eom

    @property
    def cal(self) -> Calendar:
        return Calendar(self.calendar)

    @property
    def label(self) -> str:
        return f"{self.coupon * 100:.3f}% {self.maturity:%d-%b-%Y}".upper()

    def settlement_date(self, trade_date: Date) -> Date:
        return self.cal.add_business_days(trade_date, self.settle_lag)

    def _regular(self, k: int) -> Date:
        """k-th regular coupon date before maturity (k < 0: after it)."""

        return add_months(self.maturity, -self.months * k, eom=self.end_of_month)

    def _period_index(self, day: Date) -> int:
        """k with regular(k) <= day < regular(k - 1)."""

        k = ((self.maturity.year - day.year) * 12 + self.maturity.month - day.month) // self.months
        while self._regular(k) > day:
            k += 1
        while self._regular(k - 1) <= day:
            k -= 1
        return k

    def coupon_dates(self, since: Optional[Date] = None) -> List[Date]:
        """Unadjusted schedule: the dated date (if any), then coupon dates to
        maturity. Without a dated date the schedule starts one regular period
        before the one containing ``since``, which covers every coupon still
        unpaid at ``since``."""

        if self.dated is not None:
            first = self.first_coupon or self._regular(self._period_index(self.dated) - 1)
            return [self.dated] + [self._regular(k) for k in range(self._period_index(first), -1, -1)]
        if since is None:
            raise ValueError("a bond without a dated date needs `since` to generate its schedule")
        return [self._regular(k) for k in range(max(self._period_index(since) + 1, 1), -1, -1)]

    def year_fraction(self, d1: Date, d2: Date) -> float:
        """Day-count fraction between two dates; ACT/ACT ICMA sums the days in
        each regular coupon period over ``frequency`` x the period's days.

        Regular (quasi-)coupon dates are counted back from maturity on the
        bond's coupon day, also before the dated date. QuantLib differs in two
        corner cases: it steps an odd (short or long) first coupon's
        quasi-dates back from the first coupon date (an Aug 30 bond's Feb 28
        coupon then starts a period on Aug 28), and it measures a lone odd
        period from the dated date.
        """

        if self.day_count.upper() != "ACT/ACT ICMA":
            return year_fraction(d1, d2, self.day_count)
        if d1 == d2:
            return 0.0
        if d2 < d1:
            return -self.year_fraction(d2, d1)
        k, cur, total = self._period_index(d1), d1, 0.0
        while True:
            lo, hi = self._regular(k), self._regular(k - 1)
            total += (min(d2, hi) - cur).days / (self.frequency * (hi - lo).days)
            if d2 <= hi:
                return total
            cur, k = hi, k - 1

    def coupons(self, since: Date) -> Tuple[Coupon, ...]:
        """Coupons paid after ``since`` (payment date strictly later)."""

        return _coupons(self, since)

    def cashflows(self, settle: Date) -> pd.DataFrame:
        """Cash flows a buyer settling on ``settle`` receives, per face."""

        coupons = self.coupons(settle)
        df = pd.DataFrame(
            {
                "start": [c.start for c in coupons],
                "end": [c.end for c in coupons],
                "pay": [c.pay for c in coupons],
                "accrual": [c.accrual for c in coupons],
                "coupon": [c.amount for c in coupons],
                "principal": [self.face if c.end == self.maturity else 0.0 for c in coupons],
            }
        )
        df["amount"] = _amounts(self, coupons)
        return df

    def accrued(self, settle: Date) -> float:
        """Accrued interest per face: the current coupon's day-count fraction
        from its start to settlement."""

        coupons = self.coupons(settle)
        if not coupons or settle <= coupons[0].start:
            return 0.0
        first = coupons[0]
        return self.face * self.coupon * self.year_fraction(first.start, min(settle, first.end))

    # -- yield -----------------------------------------------------------------

    def _flows(self, settle: Date, convention: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _yield_flows(self, settle, convention)

    def yield_times(self, settle: Date, convention: str = "street") -> np.ndarray:
        """Time in years from settlement to each cash flow of
        :meth:`cashflows`, as the yield formula uses it."""

        return self._flows(settle, convention)[1].copy()

    def _discount(self, y: float, times: np.ndarray) -> np.ndarray:
        base = 1.0 + y / self.frequency
        if not base > 0:
            raise ValueError(f"yield must be above {-100 * self.frequency:.0f}%")
        with np.errstate(over="ignore"):  # inf at the solver's widest bracket is fine
            return base ** (-self.frequency * times)

    def price_from_yield(self, y: float, settle: Date, clean: bool = True, convention: str = "street") -> float:
        """Price per face at yield ``y`` (decimal, compounded ``frequency``
        times a year); clean unless ``clean=False``."""

        amounts, times, _ = self._flows(settle, convention)
        dirty = float(amounts @ self._discount(y, times))
        return dirty - self.accrued(settle) if clean else dirty

    def yield_from_price(self, price: float, settle: Date, clean: bool = True, convention: str = "street") -> float:
        """Yield to maturity (decimal) reproducing ``price`` per face."""

        target = price + self.accrued(settle) if clean else price
        if not (math.isfinite(target) and target > 0):
            raise ValueError("the dirty price must be positive")
        amounts, times, _ = self._flows(settle, convention)
        brackets = ((-0.5, 1.0), (-0.95 * self.frequency, 10.0))
        return _solve(lambda y: float(amounts @ self._discount(y, times)) - target, brackets, "yield")

    def macaulay_duration(self, y: float, settle: Date, convention: str = "street") -> float:
        """PV-weighted average time to the cash flows, in years."""

        amounts, times, _ = self._flows(settle, convention)
        pv = amounts * self._discount(y, times)
        return float(times @ pv / pv.sum())

    def modified_duration(self, y: float, settle: Date, convention: str = "street") -> float:
        """-(1/P) dP/dy = Macaulay / (1 + y/f)."""

        return self.macaulay_duration(y, settle, convention) / (1.0 + y / self.frequency)

    def convexity(self, y: float, settle: Date, convention: str = "street") -> float:
        """(1/P) d2P/dy2, in years squared."""

        amounts, times, _ = self._flows(settle, convention)
        f = self.frequency
        pv = amounts * self._discount(y, times)
        return float(pv @ (times * (f * times + 1.0)) / (f * (1.0 + y / f) ** 2) / pv.sum())

    def _dprice_dy(self, y: float, settle: Date, convention: str) -> float:
        amounts, times, _ = self._flows(settle, convention)
        return -float((amounts * times) @ self._discount(y, times)) / (1.0 + y / self.frequency)

    def dv01(self, y: float, settle: Date, notional: Optional[float] = None, convention: str = "street") -> float:
        """Price change for +1bp of yield (dP/dy x 1bp, negative for a long
        bond): per face, or in currency for a ``notional`` face amount."""

        scale = 1.0 if notional is None else notional / self.face
        return self._dprice_dy(y, settle, convention) * 1e-4 * scale

    def yield_value_32nd(self, y: float, settle: Date, convention: str = "street") -> float:
        """Yield change (decimal, positive) that moves the price by 1/32 of a
        point."""

        return (self.face / 100.0 / 32.0) / abs(self._dprice_dy(y, settle, convention))

    def bps(self, y: float, settle: Date, convention: str = "street") -> float:
        """Price change per face for +1bp on the coupon rate, discounted at the
        yield (QuantLib's ``BondFunctions.bps``)."""

        _, times, accruals = self._flows(settle, convention)
        return 1e-4 * self.face * float(accruals @ self._discount(y, times))

    # -- curves and spreads ----------------------------------------------------

    def price_from_curve(self, curve: Curve, settle: Date, z_spread: float = 0.0, clean: bool = True) -> float:
        """Price per face for settlement on ``settle`` off ``curve`` plus a
        continuously compounded z-spread."""

        if settle < curve.asof:
            raise ValueError(f"settlement {settle} is before the curve date {curve.asof}")
        coupons = self.coupons(settle)
        if not coupons:
            raise ValueError(f"the bond has no cash flows after {settle}")
        amounts = _amounts(self, coupons)
        times = np.array([curve.time(c.pay) for c in coupons])
        t_settle = curve.time(settle)
        dfs = curve.df_times(times) * np.exp(-z_spread * times)
        dirty = float(amounts @ dfs) / (curve.df(settle) * math.exp(-z_spread * t_settle))
        return dirty - self.accrued(settle) if clean else dirty

    def z_spread(self, price: float, curve: Curve, settle: Date, clean: bool = True) -> float:
        """Spread (decimal) over the curve's continuously compounded zero
        rates that reprices ``price``."""

        target = price + self.accrued(settle) if clean else price
        if not (math.isfinite(target) and target > 0):
            raise ValueError("the dirty price must be positive")
        brackets = ((-0.5, 0.5), (-2.0, 2.0))
        return _solve(lambda z: self.price_from_curve(curve, settle, z, clean=False) - target, brackets, "z-spread")

    def pv(self, market: Market, z_spread: float = 0.0, notional: Optional[float] = None) -> float:
        """Value on ``market.asof`` of the cash flows paid after it, on the
        Treasury curve ``market.govt`` plus ``z_spread``; per face, or in
        currency for a ``notional`` face amount."""

        curve = _govt(market)
        coupons = self.coupons(market.asof)
        if not coupons:
            return 0.0
        amounts = _amounts(self, coupons)
        times = np.array([curve.time(c.pay) for c in coupons])
        scale = 1.0 if notional is None else notional / self.face
        return float(amounts @ (curve.df_times(times) * np.exp(-z_spread * times))) * scale

    def cash(self, start: Date, end: Date, notional: Optional[float] = None) -> float:
        """Coupons and principal paid in (start, end]."""

        scale = 1.0 if notional is None else notional / self.face
        paid = [c for c in self.coupons(start) if c.pay <= end]
        return float(_amounts(self, paid).sum()) * scale

    def g_spread(self, price: float, market: Market, settle: Date, clean: bool = True) -> float:
        """Street yield minus the Treasury par yield at the bond's maturity
        (:func:`par_yield` on ``market.govt``)."""

        ytm = self.yield_from_price(price, settle, clean)
        return ytm - par_yield(_govt(market), self.maturity, settle)

    def i_spread(self, price: float, market: Market, settle: Date, clean: bool = True) -> float:
        """Street yield minus the par rate of a spot-starting OIS ending at the
        bond's maturity on ``market.curve`` (:func:`ois_par_rate`)."""

        return self.yield_from_price(price, settle, clean) - ois_par_rate(market.curve, self.maturity)

    def asset_swap_spread(self, price: float, market: Market, settle: Date, clean: bool = True) -> float:
        """Par-par asset-swap spread (decimal) on the swap curve ``market.curve``.

        The buyer pays par for the bond and swaps its coupons for floating
        plus a spread ``s`` on par, so ``s = (P_swap - P_mkt) / (face * A)``:
        ``P_mkt`` is the market dirty price, ``P_swap`` the dirty price of the
        bond's cash flows discounted on the swap curve, and ``A`` the annuity
        of the floating leg, whose periods run from settlement through the
        bond's payment dates and accrue on the curve's OIS day count (all
        discounted relative to settlement). Positive when the bond is cheap
        to swaps.
        """

        curve = market.curve
        day_count = (curve.conventions or get_conventions(SWAPS)).ois_day_count
        dirty_mkt = price + self.accrued(settle) if clean else price
        dirty_swap = self.price_from_curve(curve, settle, clean=False)
        ends = [c.pay for c in self.coupons(settle)]
        starts = [settle] + ends[:-1]
        accruals = np.array([year_fraction(s, e, day_count) for s, e in zip(starts, ends)])
        annuity = float(accruals @ curve.df(ends)) / curve.df(settle)
        return (dirty_swap - dirty_mkt) / (self.face * annuity)


def _amounts(bond: FixedRateBond, coupons: Sequence[Coupon]) -> np.ndarray:
    """Coupon plus any principal paid on each coupon's payment date, per face."""

    return np.array([c.amount + (bond.face if c.end == bond.maturity else 0.0) for c in coupons])


@lru_cache(maxsize=4096)
def _coupons(bond: FixedRateBond, since: Date) -> Tuple[Coupon, ...]:
    dates = bond.coupon_dates(since)
    cal = bond.cal
    out = []
    for start, end in zip(dates[:-1], dates[1:]):
        pay = cal.adjust(end, bond.payment_bdc)
        if pay > since:
            accrual = bond.year_fraction(start, end)
            out.append(Coupon(start, end, pay, accrual, bond.face * bond.coupon * accrual))
    return tuple(out)


@lru_cache(maxsize=4096)
def _yield_flows(bond: FixedRateBond, settle: Date, convention: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(amounts, times in years, coupon accruals) of the flows after
    settlement. Times step from flow to flow as QuantLib does: within a
    coupon, the coupon's day count from its accrual start."""

    if convention not in YIELD_CONVENTIONS:
        raise ValueError(f"convention must be one of {YIELD_CONVENTIONS}")
    coupons = bond.coupons(settle)
    if not coupons:
        raise ValueError(f"the bond has no cash flows after {settle}")
    t, last, times = 0.0, settle, []
    for c in coupons:
        day = c.pay if convention == "true" else c.end
        if last == c.start:
            t += bond.year_fraction(last, day)
        else:
            t += bond.year_fraction(c.start, day) - bond.year_fraction(c.start, last)
        last = day
        times.append(t)
    return _amounts(bond, coupons), np.array(times), np.array([c.accrual for c in coupons])


def _govt(market: Market) -> Curve:
    if market.govt is None:
        raise ValueError("the market has no Treasury (govt) curve")
    return market.govt


# --------------------------------------------------------------------------
# Reference rates
# --------------------------------------------------------------------------


def par_yield(curve: Curve, maturity: Date, settle: Date) -> float:
    """Coupon of a bond maturing on ``maturity`` that prices at par on
    ``curve`` for settlement on ``settle``.

    The bond is the curve's own par-bond instrument: dated at settlement (a
    short first coupon if ``maturity`` is not a whole number of periods
    away), coupons every ``bond_months`` back from maturity, month ends for
    a month-end maturity, so at a quoted tenor settling at spot it returns
    the quote.
    """

    conv = curve.conventions or get_conventions(TREASURY)
    unit = FixedRateBond(
        0.0,  # only the schedule and accruals are used
        maturity,
        dated=settle,
        frequency=12 // conv.bond_months,
        day_count=conv.bond_day_count,
        calendar=conv.calendar,
    )
    coupons = unit.coupons(settle)
    if not coupons:
        raise ValueError(f"maturity {maturity} is not after settlement {settle}")
    dfs = curve.df([c.pay for c in coupons]) / curve.df(settle)
    annuity = float(np.array([c.accrual for c in coupons]) @ dfs)
    return (1.0 - float(dfs[-1])) / annuity


def ois_par_rate(curve: Curve, maturity: Date) -> float:
    """Par rate of a spot-starting OIS ending on ``maturity`` with the curve's
    OIS conventions (``USD SOFR`` for a curve without them); at a quoted OIS
    tenor's end date it returns the quote."""

    conv = curve.conventions or get_conventions(SWAPS)
    cal = conv.cal
    spot = cal.add_business_days(cal.adjust(curve.asof, "F"), conv.spot_lag)
    if maturity <= spot:
        raise ValueError(f"maturity {maturity} is not after the swap spot date {spot}")
    dates = swap_schedule(spot, maturity, conv.ois_fixed_months, cal, conv.bdc)
    annuity = floating = 0.0
    for start, end in zip(dates[:-1], dates[1:]):
        df_pay = curve.df(cal.add_business_days(end, conv.payment_lag) if conv.payment_lag else end)
        annuity += year_fraction(start, end, conv.ois_day_count) * df_pay
        floating += (curve.df(start) / curve.df(end) - 1.0) * df_pay
    return floating / annuity


def benchmark_bond(curve: Curve, tenor_years: int, settle: Date) -> FixedRateBond:
    """On-the-run style Treasury issued on ``settle``: maturity ``tenor_years``
    later (a month-end issue pays on month ends), coupon = the curve's par
    yield there (unrounded), so it prices at par."""

    maturity = add_months(settle, 12 * tenor_years)
    conv = curve.conventions or get_conventions(TREASURY)
    return FixedRateBond(
        par_yield(curve, maturity, settle),
        maturity,
        dated=settle,
        frequency=12 // conv.bond_months,
        day_count=conv.bond_day_count,
        calendar=conv.calendar,
    )


def bond_position(bond: FixedRateBond, notional: float, z_spread: float = 0.0, name: Optional[str] = None) -> Position:
    """A holding of ``notional`` face (negative = short) valued on the
    Treasury curve at a constant z-spread."""

    side = "LONG" if notional >= 0 else "SHORT"
    return Position(
        name=name or f"{side} {bond.label} BOND",
        pv=lambda market: bond.pv(market, z_spread, notional),
        kind="bond",
        cash=lambda market, start, end: bond.cash(start, end, notional),
        details={"bond": bond, "notional": notional, "z_spread": z_spread},
    )


# --------------------------------------------------------------------------
# Price formats and T-bills
# --------------------------------------------------------------------------

_32NDS = re.compile(r"^(\d+)-(\d{1,2})([0-7+]?)$")


def parse_32nds(text: str) -> float:
    """Price from Treasury notation or a decimal: ``99-16`` = 99 + 16/32,
    ``99-16+`` = 99 + 16.5/32, ``99-162`` = 99 + 16.25/32 (the third digit
    counts eighths of a 32nd), ``99.5``."""

    raw = str(text).strip().replace(" ", "")
    match = _32NDS.match(raw)
    if match:
        whole, ticks, eighth = match.groups()
        if int(ticks) > 31:
            raise ValueError(f"{text!r}: 32nds must be 00 to 31")
        eighths = 4 if eighth == "+" else int(eighth or 0)
        value = int(whole) + (int(ticks) + eighths / 8.0) / 32.0
    else:
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"Unrecognised price {text!r}: use 99-16, 99-16+, 99-162 or 99.5") from None
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"price must be positive, got {text!r}")
    return value


def format_32nds(price: float) -> str:
    """Price in Treasury notation, rounded to the nearest 1/256
    (``99-16+``, ``99-162``)."""

    if not (math.isfinite(price) and price >= 0):
        raise ValueError("price must be a non-negative number")
    whole, rest = divmod(int(round(price * 256)), 256)
    ticks, eighths = divmod(rest, 8)
    return f"{whole}-{ticks:02d}{'+' if eighths == 4 else (str(eighths) if eighths else '')}"


def _check_bill(days: int, price: Optional[float] = None) -> None:
    if days <= 0:
        raise ValueError("days to maturity must be positive")
    if price is not None and not (math.isfinite(price) and price > 0):
        raise ValueError("the bill price must be positive")


def bill_price_from_discount(discount: float, days: int, face: float = 100.0) -> float:
    """T-bill price from its bank discount yield (ACT/360)."""

    _check_bill(days)
    return face * (1.0 - discount * days / 360.0)


def bill_discount_yield(price: float, days: int, face: float = 100.0) -> float:
    """Bank discount yield: (face - price) / face x 360 / days."""

    _check_bill(days, price)
    return (face - price) / face * 360.0 / days


def bill_price_from_bey(bey: float, days: int, face: float = 100.0) -> float:
    """T-bill price from its bond-equivalent (investment) yield."""

    _check_bill(days)
    return face * bill_price(bey, days)


def bill_bond_equivalent_yield(price: float, days: int, face: float = 100.0) -> float:
    """Bond-equivalent yield: simple ACT/365 up to 182 days, the Treasury's
    semiannual formula beyond."""

    _check_bill(days, price)
    return bill_bey(price / face, days)


def bill_money_market_yield(price: float, days: int, face: float = 100.0) -> float:
    """Money-market (CD-equivalent) yield: (face / price - 1) x 360 / days."""

    _check_bill(days, price)
    return (face / price - 1.0) * 360.0 / days

