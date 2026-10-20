"""Discount curves bootstrapped from market quotes.

A :class:`Curve` is built so that every input instrument reprices exactly:
deposits, OIS swaps, single-curve fixed/float swaps (IRS), Treasury bills
quoted as bond-equivalent yields and par Treasury notes/bonds. Instruments
are solved in maturity order for the discount factor at their last cash flow
(their pillar); with a non-local interpolation the passes repeat until the
discount factors stop moving.

Interpolation (``interpolation=``):

* ``loglinear``       - log discount factors linear in time (flat forwards)
* ``linear_zero``     - continuously compounded zero rates linear in time
* ``monotone_convex`` - Hagan-West monotone convex forwards (smooth, local)

Beyond the last pillar the instantaneous forward is held at its last value.
Curve time is ACT/365F from the valuation date.

This is a single-curve framework: one curve discounts and projects.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import brentq

from .conventions import Calendar, Date, Tenor, _null_advance_months, make_schedule, parse_tenor, year_fraction

INTERPOLATIONS = ("loglinear", "linear_zero", "monotone_convex")
INSTRUMENT_KINDS = ("DEPO", "OIS", "IRS", "BILL", "BOND")


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
    """One market quote: instrument kind, tenor label and rate (decimal)."""

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
    rate the curve implies, in the quote's units.
    """

    quote: CurveQuote
    asof: Date
    conv: CurveConventions
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
        cal = self.conv.cal
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

    def implied(self, curve: "Curve") -> float:
        """Rate the curve implies for this instrument, in quote units."""

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
                i = np.searchsorted(tt, ti, side="right")  # tt[i-1] <= t < tt[i]
                t0, h, fd = tt[i - 1], self.dt[i - 1], self.fd[i - 1]
                g0, g1 = self.f_nodes[i - 1] - fd, self.f_nodes[i] - fd
                x = (ti - t0) / h
                out[inside] = y[i - 1] + fd * (ti - t0) + h * self._g_integral(g0, g1, x)
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
        self._interp = _Interpolator(self.ts, self.dfs, interpolation)

    def __repr__(self) -> str:
        name = self.conventions.name if self.conventions else "curve"
        return f"Curve({name!r}, asof={self.asof}, {len(self.ts) - 1} pillars, {self.interpolation})"

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
                    "quote": inst.quote.rate,
                    "model": implied,
                    "error_bp": (implied - inst.quote.rate) * 1e4,
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
        unchanged quotes (used for carry and roll-down).
        """

        if not self.quotes or self.conventions is None:
            raise ValueError("curve has no quotes to rebuild from")
        n = len(self.quotes)
        if isinstance(bumps_bp, (int, float)):
            bumps = [float(bumps_bp)] * n
        elif isinstance(bumps_bp, Mapping):
            bumps = [float(bumps_bp.get(i, 0.0)) for i in range(n)]
        else:
            bumps = [float(b) for b in bumps_bp]
            if len(bumps) != n:
                raise ValueError(f"expected {n} bumps, got {len(bumps)}")
        quotes = [q.bumped(b) for q, b in zip(self.quotes, bumps)]
        return bootstrap(asof or self.asof, quotes, self.conventions, self.interpolation)

    @property
    def labels(self) -> List[str]:
        return [f"{q.tenor} {q.kind}" for q in self.quotes]


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


def bootstrap(
    asof: Date,
    quotes: Sequence[CurveQuote],
    conventions: Union[str, CurveConventions] = "USD SOFR",
    interpolation: str = "loglinear",
    tol: float = 1e-14,
    max_passes: int = 50,
) -> Curve:
    """Bootstrap a curve on which every quote reprices.

    Raises ``ValueError`` for an empty quote set, two quotes sharing a pillar
    date, or quotes no positive discount factor can match.
    """

    conv = get_conventions(conventions)
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation {interpolation!r}; expected one of {INTERPOLATIONS}")
    if not quotes:
        raise ValueError("need at least one quote")
    instruments = sorted((Instrument(q, asof, conv) for q in quotes), key=lambda i: i.pillar)
    pillars = [inst.pillar for inst in instruments]
    for a, b, ia, ib in zip(pillars[:-1], pillars[1:], instruments[:-1], instruments[1:]):
        if a == b:
            raise ValueError(f"{ia.quote.tenor} {ia.kind} and {ib.quote.tenor} {ib.kind} share the pillar date {a}")
    if pillars[0] <= asof:
        raise ValueError("every instrument must mature after the valuation date")
    times = np.array([0.0] + [(d - asof).days / 365.0 for d in pillars])
    dfs = np.exp(-0.03 * times)  # starting guess, refined below
    local = interpolation != "monotone_convex"

    def trial_curve(k: int, df_k: float, full: bool) -> Curve:
        # A local scheme only needs the pillars solved so far; monotone convex
        # uses every pillar once a first pass has estimated them.
        trial = dfs.copy()
        trial[k] = df_k
        upto = len(times) if full else k + 1
        return Curve(asof, times[:upto], trial[:upto], interpolation)

    for passes in range(max_passes):
        previous = dfs.copy()
        full = not local and passes > 0
        for k, inst in enumerate(instruments, start=1):
            span = times[k] - times[k - 1]
            base = dfs[k - 1]
            f = lambda x, k=k, inst=inst: inst.npv(trial_curve(k, x, full))  # noqa: E731
            # NPV is close to linear in the pillar DF, so a secant step from a
            # good guess (the previous pass, or the last forward carried on)
            # converges in a few evaluations; brentq is the safe fallback.
            if passes > 0:
                guess = dfs[k]
            else:
                fwd = (math.log(dfs[k - 2] / base) / (times[k - 1] - times[k - 2])) if k > 1 else 0.03
                guess = base * math.exp(-fwd * span)
            root = _secant(f, guess)
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
    return Curve(asof, times, dfs, interpolation, pillars, [i.quote for i in instruments], conv, instruments)


# --------------------------------------------------------------------------
# Market state
# --------------------------------------------------------------------------


@dataclass
class Market:
    """What instruments are priced off.

    ``curve`` discounts and projects swaps and caps; ``govt`` is the
    Treasury curve for bonds. Vol shifts are additive scenario shocks in each
    model's own units (normal vol in rate units, 1bp = 1e-4; lognormal vol
    as a decimal, 1 vol point = 0.01). ``fixings`` is the curve that supplies
    rates for periods that fixed before ``curve.asof`` (e.g. when the curve has
    been rolled forward for carry); by default ``curve`` itself.
    """

    curve: Curve
    govt: Optional[Curve] = None
    vol_shift_normal: float = 0.0
    vol_shift_lognormal: float = 0.0
    fixings: Optional[Curve] = None

    @property
    def asof(self) -> Date:
        return self.curve.asof

    @property
    def fixing_curve(self) -> Curve:
        return self.fixings if self.fixings is not None else self.curve

    def projected_rate(self, start: Date, end: Date, day_count: str, fixing_date: Optional[Date] = None) -> float:
        """Simple rate for an accrual period; a period that fixed before the
        valuation date takes its rate from the fixings curve."""

        fixing_date = fixing_date or start
        source = self.fixing_curve if fixing_date < self.asof else self.curve
        return source.forward_rate(start, end, day_count)

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
