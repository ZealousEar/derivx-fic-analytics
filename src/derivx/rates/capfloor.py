"""Caps and floors on a term rate: Black, shifted Black and Bachelier.

A cap is a strip of caplets. Caplet i accrues over [s_i, e_i] (``tau_i`` on
the cap's day count), fixes ``fixing_lag`` business days before s_i and pays
at e_i::

    caplet_i = N * tau_i * DF(e_i) * E[max(F_i - K, 0)]

``F_i`` is the simple forward for the period and the expectation is taken
under the chosen model with option time ACT/365F from the valuation date to
the fixing. A caplet that has already fixed is worth its intrinsic value;
one already paid is gone. ``DF`` is always the discount (OIS) curve,
``market.curve``. The forward comes from the projection curve of the cap's
``projection`` index (a EURIBOR cap, :meth:`CapFloor.from_index`:
multi-curve, OIS discounting) through :meth:`Market.index_rate`, or from
``market.curve`` itself when ``projection`` is None (single-curve, via
:meth:`Market.projected_rate`). On the same schedule this matches
QuantLib's ``Cap``/``Floor`` on an ``IborLeg`` with ``BlackCapFloorEngine``
(plain and displaced) and ``BachelierCapFloorEngine``
(``tests/test_capfloor.py``, ``tests/test_multicurve.py``).

Vols are in each model's own units: normal (Bachelier) vol in rate units
(1bp = 1e-4), lognormal vol as a decimal (0.20 = 20%): one flat vol, an
``{expiry: vol}`` term structure, or caplet vols stripped from cap quotes
(:mod:`derivx.rates.capvol`). Vegas are P&L per +1bp of normal vol or per
+1 vol point of lognormal vol; :meth:`CapFloor.decay` is the time decay to
the next business day with forwards, discount factors and vols held (the
full one-day theta, curves rolled, is :func:`derivx.risk.scenarios.theta`).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from derivx.rates.conventions import (
    BUSINESS_DAY_CONVENTIONS,
    Calendar,
    Date,
    add_months,
    make_schedule,
    parse_tenor,
    year_fraction,
)
from derivx.rates.curves import Market
from derivx.rates.indices import TermIndex, get_index
from derivx.risk.sensitivities import Position

MODELS = ("bachelier", "black", "shifted_black")
KINDS = ("cap", "floor")
INDEX_MONTHS = (1, 3, 6, 12)
VOL_UNIT = {"bachelier": 1e-4, "black": 0.01, "shifted_black": 0.01}  # 1bp normal, 1 vol point lognormal
VolInput = Union[float, Mapping[float, float], object]  # or a stripped vol object with caplet_vols(expiry, strike)

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_EPS = np.finfo(float).eps


# --------------------------------------------------------------------------
# Model kernels (undiscounted, per unit notional and unit accrual)
# --------------------------------------------------------------------------


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _cdf(x: float) -> float:
    return float(ndtr(x))


def _check_vol(sigma: float) -> None:
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"volatility must be a non-negative number, got {sigma!r}")


def _check_inputs(F: float, K: float, T: float) -> None:
    if not (math.isfinite(F) and math.isfinite(K) and math.isfinite(T)):
        raise ValueError(f"forward, strike and expiry must be finite numbers, got {F!r}, {K!r}, {T!r}")


def _shifted(F: float, K: float, shift: float) -> Tuple[float, float]:
    if not math.isfinite(shift):
        raise ValueError(f"shift must be a finite number, got {shift!r}")
    f, k = F + shift, K + shift
    if f <= 0.0 or k <= 0.0:
        if shift == 0.0:
            raise ValueError(
                f"Black needs a positive forward and strike (forward {F:.4%}, strike {K:.4%}); "
                "use the normal (Bachelier) or shifted Black model for zero or negative rates"
            )
        raise ValueError(
            f"shifted Black needs forward and strike above -{shift:.4%} (forward {F:.4%}, strike {K:.4%}); "
            "increase the shift or use the normal (Bachelier) model"
        )
    return f, k


def _black_d1(f: float, k: float, std: float) -> float:
    return math.log(f / k) / std + 0.5 * std


def shifted_black_price(F: float, K: float, T: float, sigma: float, shift: float, is_call: bool = True) -> float:
    """Black-76 on ``F + shift`` and ``K + shift`` (displaced diffusion)."""

    _check_inputs(F, K, T)
    f, k = _shifted(F, K, shift)
    _check_vol(sigma)
    w = 1.0 if is_call else -1.0
    std = sigma * math.sqrt(max(T, 0.0))
    if std == 0.0:
        return max(w * (f - k), 0.0)
    d1 = _black_d1(f, k, std)
    return w * (f * _cdf(w * d1) - k * _cdf(w * (d1 - std)))


def black_price(F: float, K: float, T: float, sigma: float, is_call: bool = True) -> float:
    """Black-76 call/put on a forward rate; needs ``F > 0`` and ``K > 0``."""

    return shifted_black_price(F, K, T, sigma, 0.0, is_call)


def bachelier_price(F: float, K: float, T: float, sigma_n: float, is_call: bool = True) -> float:
    """Bachelier (normal) call/put on a forward; any sign of ``F`` and ``K``."""

    _check_inputs(F, K, T)
    _check_vol(sigma_n)
    d = (F - K) if is_call else (K - F)
    std = sigma_n * math.sqrt(max(T, 0.0))
    if std == 0.0:
        return max(d, 0.0)
    h = d / std
    return std * _pdf(h) + d * _cdf(h)


def shifted_black_vega(F: float, K: float, T: float, sigma: float, shift: float) -> float:
    """d price / d sigma (per unit of lognormal vol; same for calls and puts)."""

    _check_inputs(F, K, T)
    f, k = _shifted(F, K, shift)
    _check_vol(sigma)
    if T <= 0.0:
        return 0.0
    std = sigma * math.sqrt(T)
    if std == 0.0:
        return f * math.sqrt(T) * _pdf(0.0) if f == k else 0.0
    return f * _pdf(_black_d1(f, k, std)) * math.sqrt(T)


def black_vega(F: float, K: float, T: float, sigma: float) -> float:
    """d price / d sigma of Black-76 (per unit of lognormal vol)."""

    return shifted_black_vega(F, K, T, sigma, 0.0)


def bachelier_vega(F: float, K: float, T: float, sigma_n: float) -> float:
    """d price / d sigma_n (per unit of normal vol; same for calls and puts)."""

    _check_inputs(F, K, T)
    _check_vol(sigma_n)
    if T <= 0.0:
        return 0.0
    std = sigma_n * math.sqrt(T)
    if std == 0.0:
        return math.sqrt(T) * _pdf(0.0) if F == K else 0.0
    return math.sqrt(T) * _pdf((F - K) / std)


def shifted_black_delta(F: float, K: float, T: float, sigma: float, shift: float, is_call: bool = True) -> float:
    """d price / d F."""

    _check_inputs(F, K, T)
    f, k = _shifted(F, K, shift)
    _check_vol(sigma)
    w = 1.0 if is_call else -1.0
    std = sigma * math.sqrt(max(T, 0.0))
    if std == 0.0:
        return w if w * (f - k) > 0.0 else 0.0
    return w * _cdf(w * _black_d1(f, k, std))


def black_delta(F: float, K: float, T: float, sigma: float, is_call: bool = True) -> float:
    """d price / d F of Black-76."""

    return shifted_black_delta(F, K, T, sigma, 0.0, is_call)


def bachelier_delta(F: float, K: float, T: float, sigma_n: float, is_call: bool = True) -> float:
    """d price / d F."""

    _check_inputs(F, K, T)
    _check_vol(sigma_n)
    w = 1.0 if is_call else -1.0
    std = sigma_n * math.sqrt(max(T, 0.0))
    if std == 0.0:
        return w if w * (F - K) > 0.0 else 0.0
    return w * _cdf(w * (F - K) / std)


def shifted_black_gamma(F: float, K: float, T: float, sigma: float, shift: float) -> float:
    """d2 price / d F2 (per unit of rate squared; same for calls and puts).
    Zero at zero time value away from the strike."""

    _check_inputs(F, K, T)
    f, k = _shifted(F, K, shift)
    _check_vol(sigma)
    std = sigma * math.sqrt(max(T, 0.0))
    if std == 0.0:
        return 0.0
    return _pdf(_black_d1(f, k, std)) / (f * std)


def bachelier_gamma(F: float, K: float, T: float, sigma_n: float) -> float:
    """d2 price / d F2 (per unit of rate squared; same for calls and puts)."""

    _check_inputs(F, K, T)
    _check_vol(sigma_n)
    std = sigma_n * math.sqrt(max(T, 0.0))
    if std == 0.0:
        return 0.0
    return _pdf((F - K) / std) / std


def model_delta(
    model: str, F: float, K: float, T: float, sigma: float, shift: float = 0.0, is_call: bool = True
) -> float:
    """d price / d F under one of :data:`MODELS` (undiscounted, per unit of
    rate): ``N(d1)`` for a Black call, ``N(d)`` for a Bachelier call."""

    if model == "bachelier":
        return bachelier_delta(F, K, T, sigma, is_call)
    return shifted_black_delta(F, K, T, sigma, _model_shift(model, shift), is_call)


def model_gamma(model: str, F: float, K: float, T: float, sigma: float, shift: float = 0.0) -> float:
    """d2 price / d F2 under one of :data:`MODELS` (undiscounted, per unit of
    rate squared)."""

    if model == "bachelier":
        return bachelier_gamma(F, K, T, sigma)
    return shifted_black_gamma(F, K, T, sigma, _model_shift(model, shift))


def _solve_vol(excess: Callable[[float], float], hi: float, cap: float, what: str) -> float:
    """Root of an increasing ``excess(vol)`` with ``excess(0) < 0``."""

    while excess(hi) < 0.0:
        hi *= 2.0
        if hi > cap:
            raise ValueError(f"price implies a {what} above {cap:g}: check the price and inputs")
    return float(brentq(excess, 0.0, hi, xtol=1e-16, rtol=4 * _EPS, maxiter=500))


def _check_price(price: float, lower: float, upper: float, scale: float, units: float = 1.0) -> bool:
    """Validate ``price`` against [lower, upper); True if it is the lower
    bound (zero vol) within rounding. Messages quote amounts times
    ``units`` (a signed notional turns unit prices into currency)."""

    if not math.isfinite(price):
        raise ValueError(f"price must be a finite number, got {price * units!r}")
    tol = 16 * _EPS * max(scale, abs(price), 1e-12)
    below, above = ("below", "above") if units > 0 else ("above", "below")
    if price < lower - tol:
        raise ValueError(
            f"price {price * units:.8g} is {below} its zero-vol (intrinsic) value {lower * units:.8g}: "
            "no vol reproduces it"
        )
    if price >= upper:
        raise ValueError(
            f"price {price * units:.8g} is at or {above} its infinite-vol value {upper * units:.8g}: "
            "no vol reproduces it"
        )
    return price <= lower + tol


def implied_black_vol(price: float, F: float, K: float, T: float, is_call: bool = True, shift: float = 0.0) -> float:
    """Lognormal (optionally shifted) vol that reprices an undiscounted
    ``price``. Raises ``ValueError`` outside the no-arbitrage bounds
    [intrinsic, F + shift) for a call or [intrinsic, K + shift) for a put."""

    _check_inputs(F, K, T)
    f, k = _shifted(F, K, shift)
    if T <= 0.0:
        raise ValueError("the option has expired (T <= 0): there is no time value to imply a vol from")
    intrinsic = max(f - k, 0.0) if is_call else max(k - f, 0.0)
    if _check_price(price, intrinsic, f if is_call else k, max(f, k)):
        return 0.0
    return _solve_vol(lambda s: shifted_black_price(F, K, T, s, shift, is_call) - price, 0.5, 100.0, "lognormal vol")


def implied_normal_vol(price: float, F: float, K: float, T: float, is_call: bool = True) -> float:
    """Bachelier vol (rate units) that reprices an undiscounted ``price``.
    Raises ``ValueError`` below intrinsic value."""

    _check_inputs(F, K, T)
    if T <= 0.0:
        raise ValueError("the option has expired (T <= 0): there is no time value to imply a vol from")
    intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
    if _check_price(price, intrinsic, math.inf, max(abs(F), abs(K))):
        return 0.0
    return _solve_vol(lambda s: bachelier_price(F, K, T, s, is_call) - price, 0.01, 10.0, "normal vol")


def model_price(
    model: str, F: float, K: float, T: float, sigma: float, shift: float = 0.0, is_call: bool = True
) -> float:
    """Undiscounted price under one of :data:`MODELS` (``shift`` is used by
    ``shifted_black`` only)."""

    if model == "bachelier":
        return bachelier_price(F, K, T, sigma, is_call)
    return shifted_black_price(F, K, T, sigma, _model_shift(model, shift), is_call)


def model_vega(model: str, F: float, K: float, T: float, sigma: float, shift: float = 0.0) -> float:
    """d price / d vol per unit of the model's vol (see :func:`model_price`)."""

    if model == "bachelier":
        return bachelier_vega(F, K, T, sigma)
    return shifted_black_vega(F, K, T, sigma, _model_shift(model, shift))


def _model_shift(model: str, shift: float) -> float:
    """Displacement a lognormal model uses; raises for an unknown model."""

    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    return shift if model == "shifted_black" else 0.0


# --------------------------------------------------------------------------
# Cap / floor
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CapletPeriod:
    """One accrual period of the strip: fixing, accrual and payment dates."""

    fix: Date
    start: Date
    end: Date
    pay: Date
    tau: float


@dataclass(frozen=True)
class _Strip:
    """Periods still to be paid, with market inputs resolved."""

    periods: List[CapletPeriod]
    tau: np.ndarray
    expiry: np.ndarray  # ACT/365F years to fixing; 0 once fixed
    forward: np.ndarray
    df: np.ndarray

    @property
    def live(self) -> np.ndarray:
        return self.expiry > 0.0


@dataclass(frozen=True)
class CapFloor:
    """A cap or floor on a term rate.

    ``strike`` is a decimal rate. ``vol`` is one number, a mapping
    ``{expiry in years: vol}`` interpolated linearly in expiry and flat
    outside, or stripped caplet vols (:class:`derivx.rates.capvol.CapletVolCurve`
    or ``CapletVolSurface``, read at the cap's strike), in the model's units. ``shift`` is the displacement of
    ``shifted_black`` (ignored by the other models). ``notional`` < 0 is a
    sold cap.

    The schedule starts at ``start`` (unadjusted) or, when it is None, at
    spot (``fixing_lag`` business days after the valuation date) plus
    ``forward_start`` adjusted with ``bdc``, as for
    :class:`~derivx.rates.swaps.VanillaSwap`; it runs ``tenor`` from there,
    rolls backward every ``index_months`` and is adjusted with ``bdc`` on
    ``calendar``. (QuantLib's ``MakeCapFloor`` adjusts a forward start with
    Following instead, which differs only when that pushes it into the next
    month.) ``exclude_first`` drops the first period, whose rate fixes on
    the trade date; ``None`` means QuantLib ``MakeCapFloor``'s rule: drop it
    unless ``forward_start`` is set (an explicit ``start`` drops it too).
    """

    notional: float
    kind: str
    strike: float
    tenor: str
    vol: VolInput
    model: str = "bachelier"
    shift: float = 0.0
    start: Optional[Date] = None
    forward_start: Optional[str] = None
    index_months: int = 3
    day_count: str = "ACT/360"
    calendar: str = "USD"
    bdc: str = "MF"
    fixing_lag: int = 2
    exclude_first: Optional[bool] = None
    projection: Optional[str] = None

    @classmethod
    def from_index(
        cls,
        index: Union[str, TermIndex],
        notional: float,
        kind: str,
        strike: float,
        tenor: str,
        vol: VolInput,
        model: str = "bachelier",
        shift: float = 0.0,
        **fields: object,
    ) -> "CapFloor":
        """Cap or floor on a term index (multi-curve): periods of the index
        tenor on its calendar and day count, fixed ``fixing_lag`` business
        days before each period, forwards from ``market.projection(index)``
        and discounting on ``market.curve``. ``fields`` overrides any other
        attribute (``forward_start``, ``start``, ``exclude_first``)."""

        index = get_index(index)
        terms = dict(
            index_months=index.months,
            day_count=index.day_count,
            calendar=index.calendar,
            bdc=index.bdc,
            fixing_lag=index.fixing_lag,
            projection=index.name,
        )
        return cls(notional, kind, strike, tenor, vol, model, shift, **{**terms, **fields})

    def __post_init__(self) -> None:
        if self.projection is not None:
            index = get_index(self.projection)
            object.__setattr__(self, "projection", index.name)
            if self.index_months != index.months or str(self.day_count).upper() != index.day_count:
                raise ValueError(
                    f"a {index.name} cap resets every {index.months}M on {index.day_count}, not every "
                    f"{self.index_months}M on {self.day_count}"
                )
        object.__setattr__(self, "kind", str(self.kind).lower())
        object.__setattr__(self, "model", str(self.model).lower())
        object.__setattr__(self, "bdc", str(self.bdc).upper())
        object.__setattr__(self, "day_count", str(self.day_count).upper())
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        if not math.isfinite(self.notional) or self.notional == 0.0:
            raise ValueError("notional must be a non-zero number")
        if not math.isfinite(self.strike):
            raise ValueError("strike must be a finite rate")
        if self.index_months not in INDEX_MONTHS:
            raise ValueError(f"index_months must be one of {INDEX_MONTHS}, got {self.index_months!r}")
        lag = self.fixing_lag
        whole = isinstance(lag, (int, float, np.number)) and not isinstance(lag, bool) and float(lag).is_integer()
        if not whole or lag < 0:
            raise ValueError(f"fixing_lag must be a whole number of business days >= 0, got {lag!r}")
        object.__setattr__(self, "fixing_lag", int(lag))
        if self.bdc not in BUSINESS_DAY_CONVENTIONS:
            raise ValueError(f"bdc must be one of {BUSINESS_DAY_CONVENTIONS}, got {self.bdc!r}")
        if self.day_count == "ACT/ACT ICMA":
            raise ValueError("ACT/ACT ICMA is a bond day count; use ACT/360 or ACT/365F for a rate index")
        year_fraction(dt.date(2000, 1, 1), dt.date(2000, 2, 1), self.day_count)  # unknown day count raises
        Calendar(self.calendar)
        if self._months(self.tenor, "tenor") <= 0:
            raise ValueError("tenor must be positive")
        if self.forward_start is not None:
            if self.start is not None:
                raise ValueError("give either start or forward_start, not both")
            self._months(self.forward_start, "forward_start")
        if self.model == "shifted_black" and (not math.isfinite(self.shift) or self.shift < 0.0):
            raise ValueError("shift must be zero or positive")
        if self.model == "black" and self.strike <= 0.0:
            raise ValueError(
                f"Black needs a positive strike (got {self.strike:.4%}); use the normal or shifted Black model"
            )
        if self.model == "shifted_black" and self.strike + self.shift <= 0.0:
            raise ValueError(
                f"shifted Black needs strike + shift > 0 (strike {self.strike:.4%}, shift {self.shift:.4%})"
            )
        if self.stripped_vol:
            self._check_stripped_vol()
        else:
            self._vol_points()  # validates vol

    # -- conventions -------------------------------------------------------

    @staticmethod
    def _months(text: str, what: str) -> int:
        tenor = parse_tenor(text)
        if tenor.months is None:
            raise ValueError(f"{what} must be in months or years (e.g. 6M, 5Y), got {text!r}")
        return tenor.months

    @property
    def is_cap(self) -> bool:
        return self.kind == "cap"

    @property
    def vol_model(self) -> str:
        """``normal`` or ``lognormal``: which market vol shift moves it."""

        return "normal" if self.model == "bachelier" else "lognormal"

    @property
    def label(self) -> str:
        span = f"{self.forward_start}x{self.tenor}" if self.forward_start else self.tenor
        return f"{span} {self.kind.upper()} {self.strike:.3%} {self.projection or f'{self.index_months}M'}"

    def _forwards(self, market: Market, rows: List[CapletPeriod]) -> np.ndarray:
        """:meth:`rate` of each period, the forwards of periods still to fix
        in one curve call."""

        asof = market.asof
        published = market.index_fixings.get(self.projection, {}) if self.projection is not None else {}
        future = [p.fix > asof or (p.fix == asof and p.fix not in published) for p in rows]
        out = np.empty(len(rows))
        live = [i for i, f in enumerate(future) if f]
        if live:
            curve = market.projection(self.projection) if self.projection is not None else market.curve
            starts = np.asarray(curve.df([rows[i].start for i in live]), dtype=float)
            ends = np.asarray(curve.df([rows[i].end for i in live]), dtype=float)
            out[live] = (starts / ends - 1.0) / np.array([rows[i].tau for i in live])
        for i, p in enumerate(rows):
            if not future[i]:
                out[i] = self.rate(market, p)
        return out

    def rate(self, market: Market, period: CapletPeriod) -> float:
        """The period's rate: a fixing or the forward (see the module
        docstring for which curve)."""

        if self.projection is not None:
            return market.index_rate(self.projection, period.start, period.end, self.day_count, period.fix)
        return market.projected_rate(period.start, period.end, self.day_count, period.fix)

    @property
    def stripped_vol(self) -> bool:
        """True when ``vol`` is a stripped caplet vol object."""

        return hasattr(self.vol, "caplet_vols")

    def _check_stripped_vol(self) -> None:
        """A stripped vol object belongs to one model and shift."""

        own_shift = float(self.shift) if self.model == "shifted_black" else 0.0
        if self.vol.model != self.model or float(self.vol.shift) != own_shift:
            raise ValueError(
                f"the caplet vols were stripped for the {self.vol.model} model (shift {self.vol.shift:.2%}); "
                f"price the cap with that model and shift, not {self.model} (shift {own_shift:.2%})"
            )

    def _vol_points(self) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.vol, Mapping):
            if not self.vol:
                raise ValueError("vol term structure is empty")
            pts = sorted((float(t), float(v)) for t, v in self.vol.items())
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            if np.any(~np.isfinite(xs)) or np.any(xs < 0.0) or np.any(np.diff(xs) <= 0.0):
                raise ValueError("vol expiries must be distinct, non-negative years")
        else:
            xs, ys = np.array([0.0]), np.array([float(self.vol)])
        if np.any(~np.isfinite(ys)) or np.any(ys < 0.0):
            raise ValueError("vols must be non-negative numbers")
        return xs, ys

    def vol_at(self, expiry) -> np.ndarray:
        """Model vol at option expiries (years), before any market vol shift."""

        if self.stripped_vol:
            return np.asarray(self.vol.caplet_vols(np.asarray(expiry, dtype=float), self.strike), dtype=float)
        xs, ys = self._vol_points()
        return np.interp(np.asarray(expiry, dtype=float), xs, ys)

    def _excludes_first(self) -> bool:
        return self.exclude_first if self.exclude_first is not None else self.forward_start is None

    def effective_date(self, asof: Date) -> Date:
        """Start of the schedule for a valuation date (the maturity is
        ``tenor`` after it): ``start`` as given, or spot plus
        ``forward_start`` adjusted with ``bdc``."""

        if self.start is not None:
            return self.start
        cal = Calendar(self.calendar)
        spot = cal.add_business_days(cal.adjust(asof, "F"), self.fixing_lag)
        return cal.advance(spot, self.forward_start, self.bdc) if self.forward_start else spot

    def schedule(self, asof: Date) -> List[Date]:
        """Adjusted schedule dates, first period included."""

        effective = self.effective_date(asof)
        termination = add_months(effective, self._months(self.tenor, "tenor"))
        return make_schedule(effective, termination, self.index_months, self.calendar, self.bdc, backward=True)

    def periods(self, asof: Date) -> List[CapletPeriod]:
        """The cap's periods (after ``exclude_first``), paid ones included."""

        cal = Calendar(self.calendar)
        dates = self.schedule(asof)
        rows = [
            CapletPeriod(cal.add_business_days(s, -self.fixing_lag), s, e, e, year_fraction(s, e, self.day_count))
            for s, e in zip(dates[:-1], dates[1:])
        ]
        if self._excludes_first():
            rows = rows[1:]
        if not rows:
            raise ValueError(
                f"a {self.tenor} cap on a {self.index_months}M index has no caplet once the first period is excluded"
            )
        return rows

    def pinned(self, asof: Date) -> "CapFloor":
        """Same trade with its dates fixed (explicit ``start``), so it keeps
        its schedule when valued on a later date."""

        if self.start is not None and self.exclude_first is not None:
            return self
        return replace(self, start=self.effective_date(asof), forward_start=None, exclude_first=self._excludes_first())

    # -- valuation ---------------------------------------------------------

    def _strip(self, market: Market) -> _Strip:
        asof = market.asof
        rows = [p for p in self.periods(asof) if p.pay > asof]
        forward = self._forwards(market, rows)
        df = np.asarray(market.curve.df([p.pay for p in rows]), dtype=float) if rows else np.array([])
        return _Strip(
            periods=rows,
            tau=np.array([p.tau for p in rows]),
            expiry=np.array([max((p.fix - asof).days, 0) / 365.0 for p in rows]),
            forward=forward,
            df=df,
        )

    def _market_vols(self, strip: _Strip, market: Market) -> np.ndarray:
        bump = market.vol_shift_normal if self.model == "bachelier" else market.vol_shift_lognormal
        return np.maximum(self.vol_at(strip.expiry) + bump, 0.0)

    def _unit_values(self, strip: _Strip, vols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Undiscounted option values and vegas (per unit vol) per period."""

        shift = _model_shift(self.model, self.shift)
        w = 1.0 if self.is_cap else -1.0
        price, vega = np.zeros(len(strip.periods)), np.zeros(len(strip.periods))
        for i, (p, t, f, v) in enumerate(zip(strip.periods, strip.expiry, strip.forward, vols)):
            if t <= 0.0:  # fixed: intrinsic, whatever the model
                price[i] = max(w * (f - self.strike), 0.0)
                continue
            try:
                price[i] = model_price(self.model, f, self.strike, t, v, shift, self.is_cap)
                vega[i] = model_vega(self.model, f, self.strike, t, v, shift)
            except ValueError as err:
                raise ValueError(f"caplet fixing {p.fix}: {err}") from None
        return price, vega

    def _scale(self, strip: _Strip) -> np.ndarray:
        return self.notional * strip.tau * strip.df

    def pv(self, market: Market) -> float:
        """Present value to the holder, in currency."""

        strip = self._strip(market)
        price, _ = self._unit_values(strip, self._market_vols(strip, market))
        return float(np.dot(self._scale(strip), price))

    def caplets(self, market: Market) -> pd.DataFrame:
        """One row per remaining caplet: dates, accrual, forward, strike,
        vol (after market shifts), discount factor, PV and vega (P&L per
        +1bp normal / +1 vol point lognormal)."""

        strip = self._strip(market)
        vols = self._market_vols(strip, market)
        price, vega = self._unit_values(strip, vols)
        scale = self._scale(strip)
        return pd.DataFrame(
            {
                "fix": [p.fix for p in strip.periods],
                "start": [p.start for p in strip.periods],
                "end": [p.end for p in strip.periods],
                "pay": [p.pay for p in strip.periods],
                "expiry": strip.expiry,
                "tau": strip.tau,
                "forward": strip.forward,
                "strike": np.full(len(strip.periods), self.strike),
                "vol": vols,
                "df": strip.df,
                "pv": scale * price,
                "vega": scale * vega * VOL_UNIT[self.model],
                "fixed": ~strip.live,
            }
        )

    def vega(self, market: Market) -> float:
        """P&L for +1bp normal vol (Bachelier) or +1 vol point (Black)."""

        strip = self._strip(market)
        _, vega = self._unit_values(strip, self._market_vols(strip, market))
        return float(np.dot(self._scale(strip), vega) * VOL_UNIT[self.model])

    def decay(self, market: Market, days: Optional[int] = None) -> float:
        """Time decay in currency: the PV with every live caplet's option time
        shortened by ``days`` calendar days (default: to the next business
        day of the cap's calendar), forwards, discount factors and vols held,
        minus today's PV. A caplet fixing within the window drops to its
        intrinsic value."""

        if days is None:
            days = (Calendar(self.calendar).add_business_days(market.asof, 1) - market.asof).days
        if days < 0:
            raise ValueError("decay days must be 0 or more")
        strip = self._strip(market)
        vols = self._market_vols(strip, market)
        now, _ = self._unit_values(strip, vols)
        later, _ = self._unit_values(replace(strip, expiry=np.maximum(strip.expiry - days / 365.0, 0.0)), vols)
        return float(np.dot(self._scale(strip), later - now))

    def annuity(self, market: Market) -> float:
        """Sum of notional x accrual x discount factor over remaining periods
        (currency per unit of rate)."""

        strip = self._strip(market)
        return float(np.sum(self._scale(strip)))

    def atm_strike(self, market: Market) -> float:
        """Forward swap rate of the strip: the strike at which the cap and
        the floor are worth the same."""

        strip = self._strip(market)
        if not strip.periods:
            raise ValueError("the cap has no remaining periods")
        weights = strip.tau * strip.df
        return float(np.dot(weights, strip.forward) / np.sum(weights))

    def premium_bp_running(self, market: Market) -> float:
        """PV spread over the remaining periods as a running rate, in bp."""

        annuity = self.annuity(market)
        if annuity == 0.0:
            raise ValueError("the cap has no remaining periods")
        return self.pv(market) / annuity * 1e4

    def swap_pv(self, market: Market) -> float:
        """PV of the payer swap on the same periods: receive the rate, pay
        the strike (cap - floor = payer swap)."""

        strip = self._strip(market)
        return float(np.dot(self._scale(strip), strip.forward - self.strike))

    def implied_flat_vol(self, price: float, market: Market) -> float:
        """The single vol (model units, market vol shifts not added) that
        reprices the cap to ``price`` in currency."""

        strip = self._strip(market)
        if not strip.periods:
            raise ValueError("the cap has no remaining periods")
        if not strip.live.any():
            raise ValueError("every remaining caplet has fixed: there is no time value to imply a vol from")
        scale = strip.tau * strip.df
        target = price / self.notional

        def value(sigma: float) -> float:
            unit, _ = self._unit_values(strip, np.full(len(strip.periods), sigma))
            return float(np.dot(scale, unit))

        lower = value(0.0)
        if self.model == "bachelier":
            upper = math.inf
        else:  # infinite vol: a call is worth F + shift, a put K + shift
            shift = _model_shift(self.model, self.shift)
            ceiling = strip.forward + shift if self.is_cap else np.full(len(strip.periods), self.strike + shift)
            intrinsic, _ = self._unit_values(strip, np.zeros(len(strip.periods)))
            upper = float(np.dot(scale, np.where(strip.live, ceiling, intrinsic)))
        if _check_price(target, lower, upper, float(np.sum(scale) * max(abs(self.strike), 0.01)), self.notional):
            return 0.0
        hi, cap = (0.01, 10.0) if self.model == "bachelier" else (0.5, 100.0)
        return _solve_vol(lambda s: value(s) - target, hi, cap, "flat vol")

    def cash(self, market: Market, start: Date, end: Date) -> float:
        """Option payments in (start, end], with rates from ``market``
        (fixings curve for periods fixed before its valuation date)."""

        w = 1.0 if self.is_cap else -1.0
        window = [p for p in self.periods(market.asof) if start < p.pay <= end]
        total = 0.0
        for p, rate in zip(window, self._forwards(market, window)):
            total += self.notional * p.tau * max(w * (rate - self.strike), 0.0)
        return total


def cap_floor_parity(cf: CapFloor, market: Market) -> Dict[str, float]:
    """Cap, floor and payer swap PVs on ``cf``'s schedule and strike, with the
    parity residual cap - floor - swap (zero up to rounding)."""

    cap = replace(cf, kind="cap").pv(market)
    floor = replace(cf, kind="floor").pv(market)
    swap = cf.swap_pv(market)
    return {"cap": cap, "floor": floor, "payer_swap": swap, "residual": cap - floor - swap}


def capfloor_position(cf: CapFloor, asof: Optional[Date] = None, name: Optional[str] = None) -> Position:
    """Wrap a cap/floor for the risk and scenario tools. With ``asof`` the
    schedule is pinned to that valuation date (see :meth:`CapFloor.pinned`)."""

    label = name or cf.label
    trade = cf.pinned(asof) if asof is not None else cf
    return Position(
        name=label,
        pv=trade.pv,
        kind="cap",
        vol_model=trade.vol_model,
        cash=trade.cash,
        details={
            "type": trade.kind,
            "notional": trade.notional,
            "strike": trade.strike,
            "tenor": cf.tenor,
            "forward_start": cf.forward_start,
            "start": trade.start,
            "index_months": trade.index_months,
            "model": trade.model,
            "vol": trade.vol,
            "shift": trade.shift,
            "projection": trade.projection,
            "trade": trade,
        },
    )
