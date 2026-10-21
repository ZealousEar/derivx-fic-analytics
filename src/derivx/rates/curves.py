"""Discount curves bootstrapped from market quotes, and the market state.

A :class:`Curve` is built so that every input instrument reprices exactly:
deposits, OIS swaps, single-curve fixed/float swaps (IRS), SOFR futures
(``SR1``/``SR3``, quoted as prices with a Hull-White convexity adjustment,
see :mod:`derivx.rates.indices`), Treasury bills quoted as bond-equivalent
yields and par Treasury notes/bonds. Instruments are solved in maturity
order for the discount factor at their last cash flow (their pillar); with a
non-local interpolation the passes repeat until the discount factors stop
moving.

Interpolation (``interpolation=``):

* ``loglinear``       - log discount factors linear in time (flat forwards)
* ``linear_zero``     - continuously compounded zero rates linear in time
* ``monotone_convex`` - Hagan-West monotone convex forwards (smooth, local)

Beyond the last pillar the instantaneous forward is held at its last value.
Curve time is ACT/365F from the valuation date.

A :class:`Market` holds the discount (OIS) curve, the projection curves of
term indices such as EURIBOR 3M/6M (bootstrapped against the discount curve
by :mod:`derivx.rates.multicurve`), the Treasury curve and fixings. A market
without projection curves is single-curve: the OIS curve discounts and
projects, the standard for SOFR, €STR and SONIA swaps.
"""

from __future__ import annotations

import collections.abc
import datetime as dt
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq

from .conventions import Calendar, Date, Tenor, _null_advance_months, make_schedule, parse_tenor, year_fraction
from .indices import (
    HullWhite,
    contract_label,
    futures_period,
    hw_overnight_futures_convexity,
    overnight_futures_rate,
    parse_contract,
)

INTERPOLATIONS = ("loglinear", "linear_zero", "monotone_convex")
INSTRUMENT_KINDS = ("DEPO", "OIS", "IRS", "BILL", "BOND", "SR1", "SR3")
OIS_FUTURES = ("SR1", "SR3")  # SOFR futures: instruments of the USD SOFR curve itself
SPREAD_KINDS = ("BASIS",)  # quotes that are spreads between two rates, not rate levels


# --------------------------------------------------------------------------
# Conventions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveConventions:
    """Instrument conventions for one curve (one currency / index)."""

    name: str
    currency: str
    calendar: str
    spot_lag: int  # business days from trade date to spot
    depo_day_count: str = "ACT/360"
    ois_fixed_months: int = 12
    ois_day_count: str = "ACT/360"
    payment_lag: int = 0  # OIS payment lag in business days after period end
    irs_fixed_months: int = 6
    irs_fixed_day_count: str = "30/360"
    irs_float_months: int = 3
    irs_float_day_count: str = "ACT/360"
    bond_months: int = 6
    bond_day_count: str = "ACT/ACT ICMA"
    bdc: str = "MF"

    @property
    def cal(self) -> Calendar:
        return Calendar(self.calendar)


CONVENTIONS: Dict[str, CurveConventions] = {
    "USD SOFR": CurveConventions("USD SOFR", "USD", "USD", spot_lag=2, payment_lag=2),
    "EUR ESTR": CurveConventions(
        "EUR ESTR",
        "EUR",
        "TARGET",
        spot_lag=2,
        payment_lag=1,
        irs_fixed_months=12,
        irs_fixed_day_count="30/360",
        irs_float_months=6,
    ),
    "GBP SONIA": CurveConventions(
        "GBP SONIA",
        "GBP",
        "LONDON",
        spot_lag=0,
        depo_day_count="ACT/365F",
        ois_day_count="ACT/365F",
        payment_lag=0,
        irs_fixed_months=12,
        irs_fixed_day_count="ACT/365F",
        irs_float_months=6,
        irs_float_day_count="ACT/365F",
    ),
    # Treasury par curve: bills as bond-equivalent yields, coupons semiannual.
    "USD UST": CurveConventions("USD UST", "USD", "USGOV", spot_lag=1, bdc="F"),
}


def get_conventions(name: Union[str, CurveConventions]) -> CurveConventions:
    if isinstance(name, CurveConventions):
        return name
    try:
        return CONVENTIONS[name]
    except KeyError:
        raise ValueError(f"Unknown conventions {name!r}; expected one of {sorted(CONVENTIONS)}") from None


# --------------------------------------------------------------------------
# Quotes and instruments
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveQuote:
    """One market quote: instrument kind, tenor label and rate (decimal).

    Futures (``SR1``, ``SR3``, ``FUT``) carry their contract (``DEC26``) as
    the tenor and their futures *rate*, (100 - price) / 100, as the rate, so
    +1bp on the quote is -0.01 on the price. A ``BASIS`` quote is a spread.
    """

    kind: str
    tenor: str
    rate: float

    def bumped(self, bp: float) -> "CurveQuote":
        return replace(self, rate=self.rate + bp * 1e-4)


def bill_price(bey: float, days: int) -> float:
    """Price per 1 face of a T-bill from its bond-equivalent yield.

    Up to half a year the BEY is simple interest on ACT/365; beyond it, the
    Treasury's quadratic formula (inverted in closed form here).
    """

    t = days / 365.0
    if days <= 182:
        return 1.0 / (1.0 + bey * t)
    a = days / 730.0 - 0.25
    return 1.0 / (1.0 + t * bey + a * bey * bey)


def bill_bey(price: float, days: int) -> float:
    """Bond-equivalent yield of a T-bill price per 1 face (inverse of
    :func:`bill_price`)."""

    t = days / 365.0
    if days <= 182:
        return (1.0 / price - 1.0) / t
    a, c = days / 730.0 - 0.25, 1.0 - 1.0 / price
    return (-t + math.sqrt(t * t - 4.0 * a * c)) / (2.0 * a)


@dataclass
class Instrument:
    """A curve instrument with its dates resolved for a valuation date.

    ``npv(curve)`` is the value per unit notional of receiving the quoted
    rate (zero when the curve reprices the quote); ``implied(curve)`` is the
    rate the curve implies, in the quote's units. SOFR futures need
    ``convexity`` (Hull-White parameters; None means no adjustment) and,
    once their reference period has started, the SOFR ``fixings`` (date ->
    decimal rate) for the days before the valuation date.
    """

    quote: CurveQuote
    asof: Date
    conv: CurveConventions
    convexity: Optional[HullWhite] = None
    fixings: Optional[Mapping[Date, float]] = None
    start: Date = field(init=False)
    end: Date = field(init=False)
    pillar: Date = field(init=False)
    _fixed: List[Tuple[Date, Date, Date, float]] = field(init=False, default_factory=list)
    _float: List[Tuple[Date, Date, Date]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        kind = self.quote.kind.upper()
        if kind not in INSTRUMENT_KINDS:
            raise ValueError(f"Unknown instrument {self.quote.kind!r}; expected one of {INSTRUMENT_KINDS}")
        self.kind = kind
        self.convexity_adjustment = 0.0
        cal = self.conv.cal
        if kind in OIS_FUTURES:
            self._init_future(cal)
            self._cache_times()
            return
        tenor = parse_tenor(self.quote.tenor)
        self.tenor = tenor
        spot = cal.add_business_days(cal.adjust(self.asof, "F"), self.conv.spot_lag)
        if kind == "DEPO":
            anchors = {
                "today": cal.adjust(self.asof, "F"),
                "tomorrow": cal.add_business_days(self.asof, 1),
                "spot": spot,
            }
            self.start = anchors[tenor.anchor]
            self.end = (
                cal.advance(self.start, tenor, self.conv.bdc)
                if tenor.unit != "D"
                else cal.add_business_days(self.start, tenor.n)
            )
            self.pillar = self.end
        elif kind in ("OIS", "IRS"):
            if tenor.unit == "D":
                raise ValueError(f"{kind} tenor must be weeks, months or years, got {tenor}")
            self.start = spot
            self.end = cal.advance(spot, tenor, self.conv.bdc)
            unadj_end = _unadjusted_end(spot, tenor)
            if kind == "OIS":
                dates = swap_schedule(spot, unadj_end, self.conv.ois_fixed_months, cal, self.conv.bdc)
                lag = self.conv.payment_lag
                for s, e in zip(dates[:-1], dates[1:]):
                    pay = cal.add_business_days(e, lag) if lag else e
                    self._fixed.append((s, e, pay, year_fraction(s, e, self.conv.ois_day_count)))
                    self._float.append((s, e, pay))
            else:
                fixed = swap_schedule(spot, unadj_end, self.conv.irs_fixed_months, cal, self.conv.bdc)
                for s, e in zip(fixed[:-1], fixed[1:]):
                    self._fixed.append((s, e, e, year_fraction(s, e, self.conv.irs_fixed_day_count)))
                flt = swap_schedule(spot, unadj_end, self.conv.irs_float_months, cal, self.conv.bdc)
                self._float = [(s, e, e) for s, e in zip(flt[:-1], flt[1:])]
            self.end = self._fixed[-1][1]
            self.pillar = max(p for _, _, p, _ in self._fixed)
        elif kind == "BILL":
            self.start = spot
            self.end = cal.adjust(_unadjusted_end(spot, tenor), "F")
            self.pillar = self.end
        else:  # BOND: par coupon bond settling at spot, coupons on unadjusted dates
            self.start = spot
            maturity = _unadjusted_end(spot, tenor)
            months = self.conv.bond_months
            freq = 12 // months
            # Regular coupon dates step back from maturity (month ends stay
            # month ends), so a regular period accrues exactly 1/freq; only
            # a settlement stub accrues a fraction of its reference period.
            regular = [maturity]
            while regular[-1] > spot:
                regular.append(_null_advance_months(maturity, -months * len(regular), eom=True))
            coupon_dates = [spot] + [d for d in reversed(regular) if d > spot]
            for s, e in zip(coupon_dates[:-1], coupon_dates[1:]):
                ref_start = regular[regular.index(e) + 1]
                if s == ref_start:
                    accrual = 1.0 / freq
                else:
                    accrual = year_fraction(s, e, self.conv.bond_day_count, ref_start, e, freq)
                self._fixed.append((s, e, cal.adjust(e, "F"), accrual))
            self.end = maturity
            self.pillar = self._fixed[-1][2]
        self._cache_times()

    def _init_future(self, cal: Calendar) -> None:
        """SR1/SR3: reference period, pillar at its end and the convexity
        adjustment (futures rate - forward rate)."""

        if self.conv.name != "USD SOFR":
            raise ValueError(
                f"{self.kind} {self.quote.tenor} is a SOFR future: use it on the USD SOFR curve, not {self.conv.name}"
            )
        year, month = parse_contract(self.quote.tenor, self.asof)
        self.start, self.end = futures_period(self.kind, year, month)
        if self.end <= self.asof:
            raise ValueError(f"{self.kind} {contract_label(year, month)} has expired (its period ended {self.end})")
        months = max(1, round((self.end - self.asof).days / 30.4375))
        self.tenor = Tenor(months, "M", label=contract_label(year, month))
        # An SR1 month ending on a weekend or holiday still takes the last fixing's forward up to the next
        # business day: solve the pillar there so the quote reprices exactly (QuantLib's helper puts it on the
        # calendar month end and leaves such a quote off by up to ~0.4bp once later pillars are solved).
        self.pillar = cal.adjust(self.end, "F") if self.kind == "SR1" else self.end
        if self.convexity is not None:
            t1, t2 = (self.start - self.asof).days / 365.0, (self.end - self.asof).days / 365.0
            averaging = "compound" if self.kind == "SR3" else "average"
            self.convexity_adjustment = hw_overnight_futures_convexity(
                self.quote.rate, t1, t2, self.convexity.sigma, self.convexity.mean_reversion, averaging
            )

    @property
    def years(self) -> float:
        """Tenor in years (time to the end of a futures period), for
        ordering quotes and placing them on a shock profile."""

        if self.kind in OIS_FUTURES:
            return (self.end - self.asof).days / 365.0
        return self.tenor.years

    def _cache_times(self) -> None:
        to_t = lambda d: (d - self.asof).days / 365.0  # noqa: E731
        self._t_start, self._t_end = to_t(self.start), to_t(self.end)
        self._t_pillar = to_t(self.pillar)
        self._fix_pay = np.array([to_t(p) for _, _, p, _ in self._fixed])
        self._fix_tau = np.array([tau for _, _, _, tau in self._fixed])
        self._flt_s = np.array([to_t(s) for s, _, _ in self._float])
        self._flt_e = np.array([to_t(e) for _, e, _ in self._float])
        self._flt_pay = np.array([to_t(p) for _, _, p in self._float])

    # -- valuation ---------------------------------------------------------

    def _annuity(self, curve: "Curve") -> float:
        return float(np.dot(self._fix_tau, curve.df_times(self._fix_pay)))

    def _float_pv(self, curve: "Curve") -> float:
        growth = curve.df_times(self._flt_s) / curve.df_times(self._flt_e)
        return float(np.dot(growth - 1.0, curve.df_times(self._flt_pay)))

    def _df2(self, curve: "Curve", a: float, b: float) -> Tuple[float, float]:
        da, db = curve.df_times([a, b])
        return float(da), float(db)

    def _future_rate(self, curve: "Curve") -> float:
        """Futures rate the curve implies: the compounded or averaged SOFR
        forward plus the convexity adjustment."""

        forward = overnight_futures_rate(
            self.kind, self.start, self.end, self.asof, self.conv.cal, curve.df, self.fixings, self.conv.ois_day_count
        )
        return forward + self.convexity_adjustment

    def implied(self, curve: "Curve") -> float:
        """Rate the curve implies for this instrument, in quote units."""

        if self.kind in OIS_FUTURES:
            return self._future_rate(curve)
        if self.kind == "DEPO":
            tau = year_fraction(self.start, self.end, self.conv.depo_day_count)
            d_start, d_end = self._df2(curve, self._t_start, self._t_end)
            return (d_start / d_end - 1.0) / tau
        if self.kind in ("OIS", "IRS"):
            return self._float_pv(curve) / self._annuity(curve)
        if self.kind == "BILL":
            d_start, d_end = self._df2(curve, self._t_start, self._t_end)
            return bill_bey(d_end / d_start, (self.end - self.start).days)
        # BOND: the coupon that prices it at par (dirty 100 at settlement).
        d_start, d_last = self._df2(curve, self._t_start, self._t_pillar)
        return (d_start - d_last) / self._annuity(curve)

    def npv(self, curve: "Curve") -> float:
        rate = self.quote.rate
        if self.kind in OIS_FUTURES:
            return rate - self._future_rate(curve)
        if self.kind == "DEPO":
            tau = year_fraction(self.start, self.end, self.conv.depo_day_count)
            d_start, d_end = self._df2(curve, self._t_start, self._t_end)
            return d_end * (1.0 + rate * tau) - d_start
        if self.kind in ("OIS", "IRS"):
            return rate * self._annuity(curve) - self._float_pv(curve)
        if self.kind == "BILL":
            d_start, d_end = self._df2(curve, self._t_start, self._t_end)
            return d_end - bill_price(rate, (self.end - self.start).days) * d_start
        d_start, d_last = self._df2(curve, self._t_start, self._t_pillar)
        return rate * self._annuity(curve) + d_last - d_start


def _add_months(day: Date, months: int) -> Date:
    from .conventions import add_months

    return add_months(day, months)


def _unadjusted_end(start: Date, tenor: Tenor) -> Date:
    if tenor.unit == "W":
        return start + dt.timedelta(weeks=tenor.n)
    if tenor.unit == "D":
        return start + dt.timedelta(days=tenor.n)
    return _add_months(start, tenor.months)


def swap_schedule(start: Date, end: Date, months: int, cal: Calendar, bdc: str) -> List[Date]:
    """Swap leg schedule (backward from the unadjusted end, front stub).

    A swap no longer than one coupon period (plus a few days of slack for
    month lengths) is a single period. Swap pricers use this too, so par
    swaps reprice the curve's own OIS/IRS quotes.
    """

    if (end - start).days <= months * 31:
        return [start, cal.adjust(end, bdc)]
    return make_schedule(start, end, months, cal, bdc, bdc, backward=True)


# --------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------


class _Interpolator:
    """-ln DF(t) from pillar times/DFs for one interpolation scheme
    (vectorised over t)."""

    def __init__(self, times: np.ndarray, dfs: np.ndarray, method: str):
        if method not in INTERPOLATIONS:
            raise ValueError(f"Unknown interpolation {method!r}; expected one of {INTERPOLATIONS}")
        self.t = np.asarray(times, dtype=float)
        self.y = -np.log(np.asarray(dfs, dtype=float))  # y = -ln DF = integral of f
        self.method = method
        self.dt = np.diff(self.t)
        self.fd = np.diff(self.y) / self.dt  # discrete forward on each interval
        if method == "monotone_convex":
            self.f_nodes = self._monotone_convex_nodes()
            self._mc_cases()
        if method == "linear_zero":
            z = np.empty_like(self.t)
            z[1:] = self.y[1:] / self.t[1:]
            z[0] = z[1]
            self.z = z
        self.f_last = self._forward_at_last()

    def _monotone_convex_nodes(self) -> np.ndarray:
        """Instantaneous forwards at the pillars (Hagan-West, no positivity
        collar so negative rates are allowed)."""

        fd, n = self.fd, len(self.fd)
        f = np.empty(n + 1)
        if n == 1:
            f[:] = fd[0]
            return f
        w = self.dt[:-1] / (self.dt[:-1] + self.dt[1:])
        f[1:-1] = w * fd[1:] + (1.0 - w) * fd[:-1]
        f[0] = fd[0] - 0.5 * (f[1] - fd[0])
        f[-1] = fd[-1] - 0.5 * (f[-2] - fd[-1])
        return f

    def _forward_at_last(self) -> float:
        if self.method == "monotone_convex":
            return float(self.f_nodes[-1])
        if self.method == "linear_zero" and len(self.t) > 2:
            slope = (self.z[-1] - self.z[-2]) / (self.t[-1] - self.t[-2])
            return float(self.z[-1] + self.t[-1] * slope)
        return float(self.fd[-1])

    def _mc_cases(self) -> None:
        """Per interval: the forward corrections g0, g1 at its ends and which
        of the Hagan-West shapes it takes (see :meth:`_g_integral`), so a
        lookup evaluates one shape per point instead of all four."""

        g0, g1 = self.f_nodes[:-1] - self.fd, self.f_nodes[1:] - self.fd
        with np.errstate(divide="ignore", invalid="ignore"):
            c1 = ((g0 < 0) & (-0.5 * g0 <= g1) & (g1 <= -2.0 * g0)) | ((g0 > 0) & (-0.5 * g0 >= g1) & (g1 >= -2.0 * g0))
            c2 = ((g0 < 0) & (g1 > -2.0 * g0)) | ((g0 > 0) & (g1 < -2.0 * g0))
            c3 = ((g0 > 0) & (g1 < 0) & (g1 > -0.5 * g0)) | ((g0 < 0) & (g1 > 0) & (g1 < -0.5 * g0))
            flat = (np.abs(g0) < 1e-15) & (np.abs(g1) < 1e-15)
        self.g0, self.g1 = g0, g1
        self.case = np.select([flat, c1, c2, c3], [0, 1, 2, 3], default=4)

    @staticmethod
    def _g_integral_case(case: int, g0: np.ndarray, g1: np.ndarray, x: np.ndarray) -> np.ndarray:
        """:meth:`_g_integral` for points that all take shape ``case`` (the
        same formulas, so the same numbers). Call it under ``np.errstate``
        ignoring divide, invalid and over, as :meth:`neg_log_df` does."""

        if case == 0:
            out = np.zeros_like(x)
        elif case == 1:
            out = g0 * (x - 2 * x**2 + x**3) + g1 * (-(x**2) + x**3)
        elif case == 2:
            eta2 = (g1 + 2.0 * g0) / (g1 - g0)
            out = np.where(x <= eta2, g0 * x, g0 * x + (g1 - g0) * (x - eta2) ** 3 / (3.0 * (1.0 - eta2) ** 2))
        elif case == 3:
            eta3 = 3.0 * g1 / (g1 - g0)
            out = np.where(
                x < eta3,
                g1 * x + (g0 - g1) * eta3 / 3.0 * (1.0 - (1.0 - x / eta3) ** 3),
                g1 * x + (g0 - g1) * eta3 / 3.0,
            )
        else:
            eta4 = g1 / (g1 + g0)
            a = -g0 * g1 / (g0 + g1)
            out = np.where(
                x <= eta4,
                a * x + (g0 - a) * eta4 / 3.0 * (1.0 - (1.0 - x / eta4) ** 3),
                a * x + (g0 - a) * eta4 / 3.0 + (g1 - a) * (x - eta4) ** 3 / (3.0 * (1.0 - eta4) ** 2),
            )
        return out if np.isfinite(out).all() else np.nan_to_num(out, nan=0.0)

    @staticmethod
    def _g_integral(g0: np.ndarray, g1: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Integral over [0, x] of the Hagan-West forward correction on a unit
        interval. It is zero at x = 1, so the pillars reprice exactly."""

        with np.errstate(divide="ignore", invalid="ignore"):
            c1 = ((g0 < 0) & (-0.5 * g0 <= g1) & (g1 <= -2.0 * g0)) | ((g0 > 0) & (-0.5 * g0 >= g1) & (g1 >= -2.0 * g0))
            c2 = ((g0 < 0) & (g1 > -2.0 * g0)) | ((g0 > 0) & (g1 < -2.0 * g0))
            c3 = ((g0 > 0) & (g1 < 0) & (g1 > -0.5 * g0)) | ((g0 < 0) & (g1 > 0) & (g1 < -0.5 * g0))
            flat = (np.abs(g0) < 1e-15) & (np.abs(g1) < 1e-15)
            # (i) cubic
            r1 = g0 * (x - 2 * x**2 + x**3) + g1 * (-(x**2) + x**3)
            # (ii) flat then rising/falling to g1
            eta2 = (g1 + 2.0 * g0) / (g1 - g0)
            r2 = np.where(x <= eta2, g0 * x, g0 * x + (g1 - g0) * (x - eta2) ** 3 / (3.0 * (1.0 - eta2) ** 2))
            # (iii) from g0 to flat g1
            eta3 = 3.0 * g1 / (g1 - g0)
            r3 = np.where(
                x < eta3,
                g1 * x + (g0 - g1) * eta3 / 3.0 * (1.0 - (1.0 - x / eta3) ** 3),
                g1 * x + (g0 - g1) * eta3 / 3.0,
            )
            # (iv) same signs: down to a minimum (or up to a maximum) and back
            eta4 = g1 / (g1 + g0)
            a = -g0 * g1 / (g0 + g1)
            r4 = np.where(
                x <= eta4,
                a * x + (g0 - a) * eta4 / 3.0 * (1.0 - (1.0 - x / eta4) ** 3),
                a * x + (g0 - a) * eta4 / 3.0 + (g1 - a) * (x - eta4) ** 3 / (3.0 * (1.0 - eta4) ** 2),
            )
            out = np.select([flat, c1, c2, c3], [0.0, r1, r2, r3], default=r4)
        return np.nan_to_num(out, nan=0.0)

    def neg_log_df(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        tt, y = self.t, self.y
        if self.method == "loglinear":
            # np.interp is flat outside the pillars (0 before t = 0, where y = 0); add the flat-forward tail.
            return np.interp(t, tt, y) + np.maximum(t - tt[-1], 0.0) * self.f_last
        out = np.zeros_like(t)
        beyond = t >= tt[-1]
        out[beyond] = y[-1] + self.f_last * (t[beyond] - tt[-1])
        inside = (t > 0.0) & ~beyond
        if inside.any():
            ti = t[inside]
            if self.method == "loglinear":
                out[inside] = np.interp(ti, tt, y)
            elif self.method == "linear_zero":
                out[inside] = np.interp(ti, tt, self.z) * ti
            else:
                k = np.searchsorted(tt, ti, side="right") - 1  # tt[k] <= t < tt[k + 1]
                t0, h, fd = tt[k], self.dt[k], self.fd[k]
                x = (ti - t0) / h
                cases = self.case[k]
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    if cases.min() == cases.max():
                        g = self._g_integral_case(int(cases[0]), self.g0[k], self.g1[k], x)
                    else:
                        g = np.empty_like(x)
                        for case in np.unique(cases):
                            on = cases == case
                            g[on] = self._g_integral_case(int(case), self.g0[k[on]], self.g1[k[on]], x[on])
                out[inside] = y[k] + fd * (ti - t0) + h * g
        return out


# --------------------------------------------------------------------------
# Curve
# --------------------------------------------------------------------------

DateOrTime = Union[Date, float, int]


class Curve:
    """Discount curve on a valuation date (``asof``).

    ``df``, ``zero_rate`` and ``forward_rate`` accept dates or times in years
    (ACT/365F from ``asof``). ``ts``/``dfs``/``discount``/``fwd_rate`` keep the
    old ``ZeroCurve`` interface working.
    """

    def __init__(
        self,
        asof: Date,
        times: Sequence[float],
        dfs: Sequence[float],
        interpolation: str = "loglinear",
        pillar_dates: Optional[Sequence[Date]] = None,
        quotes: Optional[Sequence[CurveQuote]] = None,
        conventions: Optional[CurveConventions] = None,
        instruments: Optional[Sequence[Instrument]] = None,
        convexity: Optional[HullWhite] = None,
        overnight_fixings: Optional[Mapping[Date, float]] = None,
    ):
        self.asof = asof
        self.ts = np.asarray(times, dtype=float)
        self.dfs = np.asarray(dfs, dtype=float)
        if self.ts[0] != 0.0 or abs(self.dfs[0] - 1.0) > 1e-14:
            raise ValueError("curve must start at t=0 with DF=1")
        if np.any(np.diff(self.ts) <= 0):
            raise ValueError("pillar times must be strictly increasing")
        if np.any(self.dfs <= 0):
            raise ValueError("discount factors must be positive")
        self.interpolation = interpolation
        self.pillar_dates = list(pillar_dates) if pillar_dates is not None else None
        self.quotes = list(quotes) if quotes is not None else []
        self.conventions = conventions
        self.instruments = list(instruments) if instruments is not None else []
        self.convexity = convexity
        self.overnight_fixings = dict(overnight_fixings) if overnight_fixings else {}
        self._interp = _Interpolator(self.ts, self.dfs, interpolation)

    def __repr__(self) -> str:
        name = self.conventions.name if self.conventions else "curve"
        return f"Curve({name!r}, asof={self.asof}, {len(self.ts) - 1} pillars, {self.interpolation})"

    @classmethod
    def _trial(cls, asof: Date, times: np.ndarray, dfs: np.ndarray, interpolation: str) -> "Curve":
        """Unvalidated curve for the bootstrap's inner loop (inputs are the
        solver's own, already consistent)."""

        curve = cls.__new__(cls)
        curve.asof, curve.ts, curve.dfs, curve.interpolation = asof, times, dfs, interpolation
        curve._interp = _Interpolator(times, dfs, interpolation)
        return curve

    # -- time and discounting ---------------------------------------------

    def time(self, x: DateOrTime) -> float:
        if isinstance(x, dt.date):
            return (x - self.asof).days / 365.0
        return float(x)

    def _df_scalar(self, x: DateOrTime) -> float:
        return math.exp(-float(self._interp.neg_log_df(self.time(x))[0]))

    def df_times(self, times) -> np.ndarray:
        """Vectorised discount factors at times in years."""

        return np.exp(-self._interp.neg_log_df(times))

    def df(self, x):
        """Discount factor at a date or time; arrays/lists give an array."""

        if isinstance(x, (dt.date, float, int, np.floating, np.integer)):
            return self._df_scalar(x)
        return self.df_times([self.time(v) for v in x])

    def discount(self, t: DateOrTime) -> float:
        return self._df_scalar(t)

    def zero_rate(self, x: DateOrTime, compounding: str = "continuous") -> float:
        t = self.time(x)
        if t <= 0:
            t = 1e-6
        df = self._df_scalar(t)
        if compounding == "continuous":
            return -math.log(df) / t
        if compounding == "simple":
            return (1.0 / df - 1.0) / t
        per_year = {"annual": 1, "semiannual": 2, "quarterly": 4}[compounding]
        return per_year * (df ** (-1.0 / (per_year * t)) - 1.0)

    def forward_rate(self, start: DateOrTime, end: DateOrTime, day_count: str = "ACT/360") -> float:
        """Simple forward rate between two dates (day count accrual)."""

        if isinstance(start, dt.date) and isinstance(end, dt.date):
            tau = year_fraction(start, end, day_count)
        else:
            tau = self.time(end) - self.time(start)
        return (self._df_scalar(start) / self._df_scalar(end) - 1.0) / tau

    def instantaneous_forward(self, x: DateOrTime, h: float = 1e-4) -> float:
        t = max(self.time(x), h)
        y = self._interp.neg_log_df([t - h, t + h])
        return float((y[1] - y[0]) / (2 * h))

    def fwd_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward between two times (legacy API)."""

        t1, t2 = float(t1), float(t2)
        if t2 <= t1:
            raise ValueError("fwd_rate needs t2 > t1")
        y = self._interp.neg_log_df([t1, t2])
        return float((y[1] - y[0]) / (t2 - t1))

    # -- quotes, repricing and rebuilding ------------------------------------

    def repricing(self) -> List[Dict[str, object]]:
        """Each input quote next to the rate this curve implies for it."""

        rows = []
        for inst in self.instruments:
            implied = inst.implied(self)
            rows.append(
                {
                    "kind": inst.kind,
                    "tenor": str(inst.tenor),
                    "maturity": inst.end,
                    "years": inst.years,
                    "quote": inst.quote.rate,
                    "model": implied,
                    "error_bp": (implied - inst.quote.rate) * 1e4,
                    "convexity_bp": getattr(inst, "convexity_adjustment", 0.0) * 1e4,
                }
            )
        return rows

    def rebuilt(
        self,
        bumps_bp: Union[float, Mapping[int, float], Sequence[float]] = 0.0,
        asof: Optional[Date] = None,
    ) -> "Curve":
        """Rebuild from the same quotes, bumped in basis points.

        ``bumps_bp`` is one number (parallel), a ``{quote index: bp}`` map, or
        a sequence aligned with ``quotes``. ``asof`` re-dates the curve with
        unchanged quotes (used for carry and roll-down): futures are dated
        contracts, so a later date drops those whose reference period has
        ended, and a contract then part-way through its period takes the
        overnight fixings from this curve's date to the new one from this
        curve's own forwards (realised, as carry assumes). See
        :meth:`rolled_inputs`.
        """

        if not self.quotes or self.conventions is None:
            raise ValueError("curve has no quotes to rebuild from")
        quotes = self.bumped_quotes(bumps_bp)
        same_date = asof is None or asof == self.asof
        fixings: Mapping[Date, float] = self.overnight_fixings
        if not same_date:
            quotes, fixings = self.rolled_inputs(quotes, asof)
        return bootstrap(
            asof or self.asof,
            quotes,
            self.conventions,
            self.interpolation,
            convexity=self.convexity,
            fixings=fixings,
            _known=self.unchanged_pillars(quotes) if same_date else (),
            _guess=self.dfs[1:] if same_date else (),
            _jacobian=self.solver_jacobian() if same_date else None,
        )

    def solver_jacobian(self) -> Optional[np.ndarray]:
        """The chord matrix of this curve's rebuilds (monotone convex only:
        a local interpolation solves pillar by pillar), computed once."""

        if self.interpolation != "monotone_convex" or not self.instruments:
            return None
        cached = getattr(self, "_solver_jacobian", None)
        if cached is None:
            cached = residual_jacobian(self.asof, self.instruments, self.ts, self.dfs, self.interpolation)
            self._solver_jacobian = cached
        return cached

    def rolled_inputs(self, quotes: Sequence[CurveQuote], asof: Date) -> Tuple[List[CurveQuote], Dict[Date, float]]:
        """``quotes`` (aligned with :attr:`quotes`) and the overnight fixings
        for a rebuild on the later date ``asof``. SOFR futures whose
        reference period has ended by ``asof`` are dropped (settled). For a
        contract whose period has begun, the business days from this curve's
        date up to ``asof`` that have no published fixing take this curve's
        overnight forward (ACT/360 simple rate to the next business day).
        ``ValueError`` if no quote is left."""

        fixings: Dict[Date, float] = dict(self.overnight_fixings)
        if asof <= self.asof or not any(inst.kind in OIS_FUTURES for inst in self.instruments):
            return list(quotes), fixings
        kept: List[CurveQuote] = []
        started = []
        for quote, inst in zip(quotes, self.instruments):
            if inst.kind in OIS_FUTURES:
                if inst.end <= asof:
                    continue
                if inst.start < asof:
                    started.append(inst)
            kept.append(quote)
        if not kept:
            raise ValueError(f"every quote of the {self.conventions.name} curve has settled by {asof}")
        if started:
            cal, day_count = self.conventions.cal, self.conventions.ois_day_count
            day = cal.adjust(max(min(cal.adjust(inst.start, "P") for inst in started), self.asof), "F")
            while day < asof:
                following = cal.add_business_days(day, 1)
                if day not in fixings:
                    fixings[day] = self.forward_rate(day, following, day_count)
                day = following
        return kept, fixings

    def unchanged_pillars(self, quotes: Sequence[CurveQuote]) -> np.ndarray:
        """Discount factors of the leading pillars whose quotes are the
        same in ``quotes`` (same order), which a local rebuild on the same
        date keeps exactly; empty for monotone convex or if nothing moves."""

        if self.interpolation == "monotone_convex" or len(quotes) != len(self.quotes):
            return np.array([])
        same = 0
        for old, new in zip(self.quotes, quotes):
            if old != new:
                break
            same += 1
        if same == len(self.quotes):
            return np.array([])
        return self.dfs[1 : same + 1]

    def bumped_quotes(self, bumps_bp: Union[float, Mapping[int, float], Sequence[float]] = 0.0) -> List[CurveQuote]:
        """The curve's quotes moved by ``bumps_bp`` (see :meth:`rebuilt`)."""

        n = len(self.quotes)
        if isinstance(bumps_bp, (int, float, np.floating, np.integer)):
            bumps = [float(bumps_bp)] * n
        elif isinstance(bumps_bp, Mapping):
            bumps = [float(bumps_bp.get(i, 0.0)) for i in range(n)]
        else:
            bumps = [float(b) for b in bumps_bp]
            if len(bumps) != n:
                raise ValueError(f"expected {n} bumps, got {len(bumps)}")
        return [q.bumped(b) for q, b in zip(self.quotes, bumps)]

    @property
    def labels(self) -> List[str]:
        return [f"{q.tenor} {q.kind}" for q in self.quotes]

    @property
    def years(self) -> np.ndarray:
        """Tenor in years of each quote's instrument (futures: to the end of
        the reference period)."""

        return np.array([inst.years for inst in self.instruments], dtype=float)


def _secant(f: Callable[[float], float], x0: float, max_iter: int = 30) -> Optional[float]:
    """Secant iteration to machine precision; None if it does not settle."""

    x1 = x0 * (1.0 + 1e-6)
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        if f1 == 0.0:
            return x1
        if f1 == f0 or not math.isfinite(f1):
            return None
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if not math.isfinite(x2) or x2 <= 0.0:
            return None
        if abs(x2 - x1) <= 4.0 * np.finfo(float).eps * abs(x2):
            return x2
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
    return None


def solve_pillars(
    asof: Date,
    instruments: Sequence[object],
    interpolation: str = "loglinear",
    tol: float = 1e-14,
    max_passes: int = 50,
    known: Sequence[float] = (),
    guess: Sequence[float] = (),
    jacobian: Optional[np.ndarray] = None,
) -> Tuple[List[object], np.ndarray, np.ndarray]:
    """Sort instruments by pillar and solve the discount factor at each one
    so that ``instrument.npv(curve)`` is zero (curve time ACT/365F).

    Works for any instrument with ``pillar``, ``kind``, ``quote`` and
    ``npv(curve)``: the discount curve's own instruments, or projection
    curve instruments that discount on a fixed OIS curve. ``known`` are
    discount factors already solved for the first pillars from the same
    instruments (a local interpolation solves pillars in order, so a rebuild
    that changes only later quotes reuses them exactly); ``guess`` are
    starting discount factors for every pillar (a rebuild starts from the
    curve it replaces). With monotone convex (non-local) interpolation, a
    rebuild that also passes the replaced curve's ``jacobian`` (see
    :func:`residual_jacobian`) solves every pillar at once by the chord
    method from ``guess``: a 1bp bump converges in a few iterations, about
    ten times faster than repeated passes, to the same tolerance; it falls
    back to the passes if it does not converge. Returns (sorted instruments,
    times with 0 first, discount factors with 1 first).
    """

    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation {interpolation!r}; expected one of {INTERPOLATIONS}")
    if not instruments:
        raise ValueError("need at least one quote")
    instruments = sorted(instruments, key=lambda i: i.pillar)
    pillars = [inst.pillar for inst in instruments]
    for a, b, ia, ib in zip(pillars[:-1], pillars[1:], instruments[:-1], instruments[1:]):
        if a == b:
            raise ValueError(f"{ia.quote.tenor} {ia.kind} and {ib.quote.tenor} {ib.kind} share the pillar date {a}")
    if pillars[0] <= asof:
        raise ValueError("every instrument must mature after the valuation date")
    times = np.array([0.0] + [(d - asof).days / 365.0 for d in pillars])
    dfs = np.exp(-0.03 * times)  # starting guess, refined below
    local = interpolation != "monotone_convex"
    n = len(instruments)
    if not local and jacobian is not None and len(guess) == n and np.shape(jacobian) == (n, n):
        solved = _chord(_residuals(asof, instruments, times, interpolation), np.log(np.asarray(guess, float)), jacobian,
                        tol)  # fmt: skip
        if solved is not None:
            return instruments, times, np.concatenate([[1.0], np.exp(solved)])
    first = 1
    if local and len(known):
        if len(known) >= len(instruments):
            raise ValueError("known discount factors cover every pillar: nothing to solve")
        dfs[1 : len(known) + 1] = known
        first = len(known) + 1

    def trial_curve(k: int, df_k: float, full: bool) -> Curve:
        # A local scheme only needs the pillars solved so far; monotone convex
        # uses every pillar once a first pass has estimated them.
        trial = dfs.copy()
        trial[k] = df_k
        upto = len(times) if full else k + 1
        return Curve._trial(asof, times[:upto], trial[:upto], interpolation)

    for passes in range(max_passes):
        previous = dfs.copy()
        full = not local and passes > 0
        for k, inst in enumerate(instruments, start=1):
            if k < first:
                continue
            span = times[k] - times[k - 1]
            base = dfs[k - 1]
            f = lambda x, k=k, inst=inst: inst.npv(trial_curve(k, x, full))  # noqa: E731
            # NPV is close to linear in the pillar DF, so a secant step from a
            # good guess (the previous pass, or the last forward carried on)
            # converges in a few evaluations; brentq is the safe fallback.
            if passes > 0:
                start = dfs[k]
            elif len(guess) == len(instruments):
                start = float(guess[k - 1])
            else:
                fwd = (math.log(dfs[k - 2] / base) / (times[k - 1] - times[k - 2])) if k > 1 else 0.03
                start = base * math.exp(-fwd * span)
            root = _secant(f, start)
            lo, hi = base * math.exp(-1.0 * span), base * math.exp(0.25 * span)
            if root is not None and lo < root < hi:
                dfs[k] = root
                continue
            f_lo, f_hi = f(lo), f(hi)
            widen = 0
            while f_lo * f_hi > 0 and widen < 8:
                lo, hi = lo * math.exp(-2.0 * span), hi * math.exp(1.0 * span)
                f_lo, f_hi = f(lo), f(hi)
                widen += 1
            if f_lo * f_hi > 0:
                raise ValueError(f"cannot fit {inst.quote.tenor} {inst.kind} at {inst.quote.rate:.4%}")
            dfs[k] = brentq(f, lo, hi, xtol=1e-16, rtol=4 * np.finfo(float).eps, maxiter=200)
        if local or (passes > 0 and np.max(np.abs(dfs - previous)) < tol):
            break
    return instruments, times, dfs


class _RecordingCurve(Curve):
    """Trial curve that notes every time it is asked for (the instruments of
    a bootstrap always ask for the same times, whatever the curve)."""

    def df_times(self, times) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        self._seen.append(t.ravel())
        return super().df_times(t)

    def _df_scalar(self, x: DateOrTime) -> float:
        self._seen.append(np.array([self.time(x)]))
        return super()._df_scalar(x)


class _GridCurve(Curve):
    """Trial curve that interpolates once on a grid of times and looks the
    instruments' times up in it (the same numbers: interpolation is
    elementwise); any other time is interpolated as usual."""

    def _set_grid(self, grid: np.ndarray) -> None:
        self._grid, self._values = grid, self._interp.neg_log_df(grid)

    def _lookup(self, t: np.ndarray) -> Optional[np.ndarray]:
        k = np.minimum(np.searchsorted(self._grid, t), len(self._grid) - 1)
        return self._values[k] if np.array_equal(self._grid[k], t) else None

    def df_times(self, times) -> np.ndarray:
        t = np.atleast_1d(np.asarray(times, dtype=float))
        found = self._lookup(t)
        return np.exp(-found) if found is not None else super().df_times(t)

    def _df_scalar(self, x: DateOrTime) -> float:
        found = self._lookup(np.array([self.time(x)]))
        return math.exp(-float(found[0])) if found is not None else super()._df_scalar(x)


def _residuals(
    asof: Date, instruments: Sequence[object], times: np.ndarray, interpolation: str
) -> Callable[[np.ndarray], np.ndarray]:
    """NPV of every instrument on the curve with log discount factors ``x``
    at ``times[1:]``. The first call notes the times the instruments use;
    later calls interpolate them all at once."""

    grid: List[np.ndarray] = []

    def resid(x: np.ndarray) -> np.ndarray:
        dfs = np.concatenate([[1.0], np.exp(x)])
        if not grid:
            curve = _RecordingCurve._trial(asof, times, dfs, interpolation)
            curve._seen = []
            out = np.array([inst.npv(curve) for inst in instruments], dtype=float)
            grid.append(np.unique(np.concatenate(curve._seen)) if curve._seen else np.array([0.0]))
            return out
        curve = _GridCurve._trial(asof, times, dfs, interpolation)
        curve._set_grid(grid[0])
        return np.array([inst.npv(curve) for inst in instruments], dtype=float)

    return resid


def residual_jacobian(
    asof: Date, instruments: Sequence[object], times: np.ndarray, dfs: np.ndarray, interpolation: str, h: float = 1e-7
) -> np.ndarray:
    """d npv_i / d ln DF_j of a bootstrapped curve's instruments at its
    pillars (forward differences of ``h`` in log DF): the chord matrix of
    its rebuilds."""

    resid = _residuals(asof, instruments, times, interpolation)
    x0 = np.log(np.asarray(dfs[1:], dtype=float))
    r0 = resid(x0)
    out = np.empty((len(x0), len(x0)))
    for j in range(len(x0)):
        x = x0.copy()
        x[j] += h
        out[:, j] = (resid(x) - r0) / h
    return out


def _chord(
    resid: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, jacobian: np.ndarray, tol: float, max_iter: int = 40
) -> Optional[np.ndarray]:
    """Root of ``resid`` by the chord method (fixed Jacobian) from ``x0``;
    None if it does not settle below ``tol``."""

    try:
        factor = lu_factor(np.asarray(jacobian, dtype=float), check_finite=True)
    except (ValueError, np.linalg.LinAlgError):
        return None
    x = x0.copy()
    for _ in range(max_iter):
        step = lu_solve(factor, resid(x))
        if not np.all(np.isfinite(step)):
            return None
        x -= step
        if float(np.max(np.abs(step))) < tol:
            return x
    return None


def bootstrap(
    asof: Date,
    quotes: Sequence[CurveQuote],
    conventions: Union[str, CurveConventions] = "USD SOFR",
    interpolation: str = "loglinear",
    tol: float = 1e-14,
    max_passes: int = 50,
    convexity: Optional[HullWhite] = None,
    fixings: Optional[Mapping[Date, float]] = None,
    _known: Sequence[float] = (),
    _guess: Sequence[float] = (),
    _jacobian: Optional[np.ndarray] = None,
) -> Curve:
    """Bootstrap a curve on which every quote reprices.

    ``convexity`` (Hull-White parameters) sets the convexity adjustment of
    SR1/SR3 futures quotes (none if omitted); ``fixings`` are the overnight
    fixings (date -> decimal) a future needs once its period has started.
    Raises ``ValueError`` for an empty quote set, two quotes sharing a
    pillar date, or quotes no positive discount factor can match.
    """

    conv = get_conventions(conventions)
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation {interpolation!r}; expected one of {INTERPOLATIONS}")
    if not quotes:
        raise ValueError("need at least one quote")
    built = [Instrument(q, asof, conv, convexity, fixings) for q in quotes]
    instruments, times, dfs = solve_pillars(asof, built, interpolation, tol, max_passes, _known, _guess, _jacobian)
    pillars = [inst.pillar for inst in instruments]
    return Curve(
        asof, times, dfs, interpolation, pillars, [i.quote for i in instruments], conv, instruments, convexity, fixings
    )


# --------------------------------------------------------------------------
# Market state
# --------------------------------------------------------------------------


CurveSource = Union[Curve, Callable[[], Curve]]
DISCOUNT_KEY, GOVT_KEY = "curve", "govt"


class ProjectionSet(collections.abc.Mapping):
    """Projection curves keyed by index name (``"EURIBOR 6M"``).

    A value is a curve or a zero-argument builder: a rebuilt market defers
    re-bootstrapping a projection curve until something prices off it, so
    bumping the discount curve costs nothing for curves a trade does not
    use. ``depends_on(key)`` lists what the curve was bootstrapped on:
    ``"curve"`` (the discount curve) and, for a curve built from tenor basis
    swaps, the other index. A projection that *is* the discount curve
    (single-curve, as a ``CurveSet`` whose projection is its discount curve)
    follows it when the discount curve is rebuilt.
    """

    def __init__(
        self,
        items: Optional[Mapping[str, CurveSource]] = None,
        depends: Optional[Mapping[str, Tuple[str, ...]]] = None,
        discount: Optional[Curve] = None,
        same_as_discount: Iterable[str] = (),
    ):
        self._sources: Dict[str, CurveSource] = {}
        self._built: Dict[str, Curve] = {}
        self._depends: Dict[str, Tuple[str, ...]] = {}
        self._same = {str(k).strip().upper() for k in same_as_discount}
        depends = None if depends is None else {str(k).strip().upper(): tuple(v) for k, v in depends.items()}
        for key, value in (items or {}).items():
            name = str(key).strip().upper()
            if name in (DISCOUNT_KEY, GOVT_KEY, DISCOUNT_KEY.upper(), GOVT_KEY.upper()):
                raise ValueError(f"{key!r} is reserved for the discount/Treasury curve; key projections by index name")
            self._sources[name] = value
            if isinstance(value, Curve):
                self._built[name] = value
                if discount is not None and value is discount:
                    self._same.add(name)
            if depends is not None and name in depends:
                self._depends[name] = tuple(depends[name])
            elif isinstance(value, Curve):
                self._depends[name] = _curve_depends(value) if name not in self._same else (DISCOUNT_KEY,)
            else:
                self._depends[name] = ()
        unknown = [d for deps in self._depends.values() for d in deps if d != DISCOUNT_KEY and d not in self._sources]
        if unknown:
            raise ValueError(f"projection curves depend on curves the market does not have: {sorted(set(unknown))}")

    def __getitem__(self, key: str) -> Curve:
        name = str(key).strip().upper()
        if name not in self._sources:
            raise KeyError(key)
        if name not in self._built:
            value = self._sources[name]
            self._built[name] = value() if callable(value) and not isinstance(value, Curve) else value
        return self._built[name]

    def __iter__(self):
        return iter(self.order())

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, key: object) -> bool:
        return str(key).strip().upper() in self._sources

    def __repr__(self) -> str:
        return f"ProjectionSet({list(self.order())})"

    def source(self, key: str) -> CurveSource:
        name = str(key).strip().upper()
        return self._built.get(name, self._sources[name])

    def rebased(self, discount: Curve) -> "ProjectionSet":
        """This set with every projection that is the discount curve
        pointing at ``discount`` (``self`` if they already do)."""

        if all(self._built.get(k) is discount for k in self._same if k in self._sources):
            return self
        sources = {k: (discount if k in self._same else self.source(k)) for k in self._sources}
        return ProjectionSet(sources, self._depends, same_as_discount=self._same)

    def depends_on(self, key: str) -> Tuple[str, ...]:
        return self._depends[str(key).strip().upper()]

    def is_discount(self, key: str) -> bool:
        return str(key).strip().upper() in self._same

    def is_built(self, key: str) -> bool:
        return str(key).strip().upper() in self._built

    def order(self) -> Tuple[str, ...]:
        """Keys with every curve after the projections it was built on."""

        out: List[str] = []
        pending = list(self._sources)
        while pending:
            ready = [k for k in pending if all(d == DISCOUNT_KEY or d in out for d in self._depends[k])]
            if not ready:
                raise ValueError(f"projection curves depend on each other in a cycle: {pending}")
            out += ready
            pending = [k for k in pending if k not in ready]
        return tuple(out)


def _curve_depends(curve: Curve) -> Tuple[str, ...]:
    """What a bootstrapped projection curve was built on (none for a plain
    curve)."""

    if not getattr(curve, "is_projection", False):
        return ()
    basis = getattr(curve, "basis_index", None)
    return (DISCOUNT_KEY,) + ((basis,) if basis else ())


@dataclass
class Market:
    """What instruments are priced off: the curve set and fixings.

    * ``curve`` - the discount curve (OIS: SOFR, €STR, SONIA). It also
      projects OIS legs and, single-curve, generic ``TERM`` legs.
    * ``projections`` - projection curves of term indices keyed by index
      name (``"EURIBOR 6M"``), bootstrapped with ``curve`` discounting (see
      :mod:`derivx.rates.multicurve`). A projection may be ``curve`` itself:
      that is the single-curve set, and it prices exactly as ``TERM``.
    * ``govt`` - the Treasury curve for bonds.

    Vol shifts are additive scenario shocks in each model's own units
    (normal vol in rate units, 1bp = 1e-4; lognormal vol as a decimal, 1 vol
    point = 0.01). Past rates: ``fixings`` is the curve that supplies
    discount-curve rates for periods before ``curve.asof`` (e.g. when the
    market has been rolled forward for carry); by default ``curve`` itself.
    ``index_fixings`` holds published fixings of term indices (``{index:
    {fixing date: decimal}}``); ``projection_fixings`` holds, for a rolled
    market, the projection curves of the earlier date, which supply fixings
    on or after that date.
    """

    curve: Curve
    govt: Optional[Curve] = None
    vol_shift_normal: float = 0.0
    vol_shift_lognormal: float = 0.0
    fixings: Optional[Curve] = None
    projections: Mapping[str, CurveSource] = field(default_factory=dict)
    projection_fixings: Mapping[str, CurveSource] = field(default_factory=dict)
    index_fixings: Mapping[str, Mapping[Date, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.projections, ProjectionSet):
            self.projections = ProjectionSet(self.projections, discount=self.curve)
        else:  # a single-curve projection follows a replaced discount curve
            self.projections = self.projections.rebased(self.curve)
        if not isinstance(self.projection_fixings, ProjectionSet):
            fixings = dict(self.projection_fixings)
            self.projection_fixings = ProjectionSet(fixings, depends={k: () for k in fixings})
        self.index_fixings = {
            str(k).strip().upper(): {d: float(r) for d, r in v.items()} for k, v in dict(self.index_fixings).items()
        }

    @property
    def asof(self) -> Date:
        return self.curve.asof

    @property
    def discount(self) -> Curve:
        """The discount curve (same as ``curve``)."""

        return self.curve

    @property
    def fixing_curve(self) -> Curve:
        return self.fixings if self.fixings is not None else self.curve

    @property
    def curve_keys(self) -> Tuple[str, ...]:
        """Every curve of the market: ``"curve"`` (discount), the projection
        indices (each after the curves it is built on) and ``"govt"``."""

        return (DISCOUNT_KEY, *self.projections.order(), *((GOVT_KEY,) if self.govt is not None else ()))

    @property
    def is_multi_curve(self) -> bool:
        return any(not self.projections.is_discount(k) for k in self.projections)

    def get_curve(self, key: str) -> Curve:
        """Curve by key: ``"curve"``, ``"govt"`` or a projection index."""

        if key == DISCOUNT_KEY:
            return self.curve
        if key == GOVT_KEY:
            if self.govt is None:
                raise ValueError("market has no 'govt' curve")
            return self.govt
        return self.projection(key)

    def projection(self, index: str) -> Curve:
        """Projection curve of a term index; ``ValueError`` if the market
        has none (pricing it on the discount curve instead would be a silent
        single-curve approximation)."""

        if index not in self.projections:
            have = ", ".join(self.projections.order()) or "none"
            raise ValueError(
                f"the market has no {index} projection curve (projection curves: {have}); build one with "
                f"derivx.rates.multicurve.bootstrap_projection, or pass projections={{'{index}': market.curve}} "
                "to project it single-curve on purpose"
            )
        curve = self.projections[index]
        if curve.asof != self.asof:
            raise ValueError(f"the {index} projection curve is dated {curve.asof}, the market {self.asof}")
        return curve

    def projected_rate(self, start: Date, end: Date, day_count: str, fixing_date: Optional[Date] = None) -> float:
        """Simple rate for an accrual period; a period that fixed before the
        valuation date takes its rate from the fixings curve."""

        fixing_date = fixing_date or start
        source = self.fixing_curve if fixing_date < self.asof else self.curve
        return source.forward_rate(start, end, day_count)

    def index_rate(
        self, index: str, start: Date, end: Date, day_count: str, fixing_date: Optional[Date] = None
    ) -> float:
        """Rate of a term index for the accrual period [start, end].

        A fixing published for ``fixing_date`` (``index_fixings``) is used
        when the fixing date is today or earlier; otherwise a past fixing
        comes from ``projection_fixings`` (the curve of an earlier valuation
        date, when the fixing date is on or after it) and a future one is
        the projection curve's forward over the period (par coupon, as
        QuantLib's ``IborCoupon``). ``ValueError`` if a past fixing is
        unknown.
        """

        name = str(index).strip().upper()
        fixing_date = fixing_date or start
        if fixing_date <= self.asof:
            published = self.index_fixings.get(name, {}).get(fixing_date)
            if published is not None:
                return published
        if fixing_date < self.asof:
            if name in self.projection_fixings:
                past = self.projection_fixings[name]
                if fixing_date >= past.asof:
                    return past.forward_rate(start, end, day_count)
            raise ValueError(
                f"{name} fixed on {fixing_date}, before the valuation date {self.asof}: add the fixing to the "
                f"market's index_fixings ({{'{name}': {{date: rate}}}})"
            )
        return self.projection(name).forward_rate(start, end, day_count)

    def compounded_growth(self, start: Date, end: Date) -> float:
        """Growth factor of an overnight-compounded period, using fixings for
        the part before the valuation date."""

        if end <= self.asof:
            return self.fixing_curve.df(start) / self.fixing_curve.df(end)
        if start >= self.asof:
            return self.curve.df(start) / self.curve.df(end)
        fixed = self.fixing_curve.df(start) / self.fixing_curve.df(self.asof)
        return fixed / self.curve.df(end)

    def replace(self, **changes) -> "Market":
        return replace(self, **changes)

    def rebuilt(self, bumps: Optional[Mapping[str, object]] = None, asof: Optional[Date] = None) -> "Market":
        """The market with curves rebuilt from bumped quotes and/or on a new
        valuation date.

        ``bumps`` maps curve keys (:attr:`curve_keys`) to bumps in bp in any
        form :meth:`Curve.rebuilt` takes. Projection curves bootstrapped on
        a curve that changes are rebuilt on the new one with their own quotes
        held (market-quote risk), lazily on first use. Past discount-curve
        fixings stay on today's curve; a rolled market (``asof`` later) also
        keeps today's projection curves for fixings in the window.
        """

        bumps = dict(bumps or {})
        keys = self.curve_keys
        unknown = [k for k in bumps if k not in keys]
        if unknown:
            raise ValueError(f"no {unknown} curve to bump; the market's curves are {list(keys)}")
        aliases = [k for k in bumps if k not in (DISCOUNT_KEY, GOVT_KEY) and self.projections.is_discount(k)]
        if aliases:
            raise ValueError(
                f"the {', '.join(aliases)} projection is the discount curve itself (single-curve): bump "
                f"{DISCOUNT_KEY!r} instead"
            )
        at = asof or self.asof
        rolled = at != self.asof
        if rolled and (self.fixings is not None or len(self.projection_fixings)):
            raise ValueError("cannot roll a market that already has a separate fixings curve")
        changed = set()

        def rebuild(key: str, curve: Optional[Curve]) -> Optional[Curve]:
            if curve is None or not (key in bumps or rolled):
                return curve
            if not curve.quotes:
                verb = f"roll the {key!r} curve to {at}" if rolled else f"bump the {key!r} curve"
                raise ValueError(f"cannot {verb}: it has no quotes to rebuild from")
            changed.add(key)
            return curve.rebuilt(bumps.get(key, 0.0), asof=at)

        new_curve = rebuild(DISCOUNT_KEY, self.curve)
        new_govt = rebuild(GOVT_KEY, self.govt)
        old = self.projections
        sources: Dict[str, CurveSource] = {}
        depends: Dict[str, Tuple[str, ...]] = {}
        same = []
        new_set: Dict[str, ProjectionSet] = {}
        for key in old.order():
            depends[key] = old.depends_on(key)
            if old.is_discount(key):
                same.append(key)
                sources[key] = new_curve
                if DISCOUNT_KEY in changed:
                    changed.add(key)
                continue
            if not (key in bumps or rolled or any(d in changed for d in depends[key])):
                sources[key] = old.source(key)
                continue
            changed.add(key)

            def build(key: str = key) -> Curve:
                base = old[key]
                if getattr(base, "is_projection", False):
                    # Only curves that changed are swapped in, so a bump of this curve's own quotes keeps the
                    # curves it was built on and reuses its unchanged leading pillars.
                    basis = base.basis_index
                    return base.rebuilt(
                        bumps.get(key, 0.0),
                        asof=at,
                        discount=new_curve if DISCOUNT_KEY in changed else None,
                        basis=new_set["set"][basis] if basis and basis in changed else None,
                    )
                if not base.quotes:
                    verb = f"roll the {key} curve to {at}" if rolled else f"bump the {key} curve"
                    raise ValueError(f"cannot {verb}: it has no quotes to rebuild from")
                return base.rebuilt(bumps.get(key, 0.0), asof=at)

            sources[key] = build
        new_set["set"] = ProjectionSet(sources, depends, same_as_discount=same)
        fixings = self.fixings
        if DISCOUNT_KEY in changed and fixings is None:
            fixings = self.curve  # past fixings stay where they were
        projection_fixings = self.projection_fixings
        if rolled and len(old):
            projection_fixings = ProjectionSet({k: old.source(k) for k in old.order()}, depends={k: () for k in old})
        return replace(
            self,
            curve=new_curve,
            govt=new_govt,
            fixings=fixings,
            projections=new_set["set"],
            projection_fixings=projection_fixings,
        )


CurveSet = Market  # the curve set: one discount curve and projection curves keyed by index


def flat_curve(asof: Date, rate: float, interpolation: str = "loglinear", horizon: float = 60.0) -> Curve:
    """Continuously compounded flat curve (tests and examples)."""

    times = np.array([0.0, horizon])
    return Curve(asof, times, np.exp(-rate * times), interpolation)


def curve_from_rows(
    asof: Date,
    rows: Iterable[Mapping[str, object]],
    conventions: Union[str, CurveConventions] = "USD SOFR",
    interpolation: str = "loglinear",
) -> Curve:
    """Bootstrap from rows with ``kind``/``instrument``, ``tenor`` and
    ``rate`` (decimal) keys."""

    quotes = []
    for row in rows:
        kind = str(row.get("kind", row.get("instrument", ""))).strip().upper()
        quotes.append(CurveQuote(kind, str(row["tenor"]).strip(), float(row["rate"])))
    return bootstrap(asof, quotes, conventions, interpolation)


PvFunction = Callable[[Market], float]
