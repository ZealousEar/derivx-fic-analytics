"""Projection curves for term indices, bootstrapped against an OIS curve.

Multi-curve pricing discounts every cash flow on the OIS (collateral)
curve and projects each term index (EURIBOR 3M, EURIBOR 6M) on its own
curve. A :class:`ProjectionCurve` is a pseudo-discount curve P_i whose
simple forwards are the index forwards, ``F = (P_i(s) / P_i(e) - 1) / tau``
on the index day count. :func:`bootstrap_projection` solves its pillars so
every quote reprices with the discount curve held fixed (dual-curve
bootstrap), from:

* ``DEPO`` - a deposit / index fixing from spot (e.g. ``6M``): the forward
  from spot over the tenor (QuantLib ``DepositRateHelper``);
* ``FRA``  - ``AxB`` (months to start x months to end, ``B - A`` = the index
  tenor), the index forward for the period starting ``A`` months after spot
  (QuantLib ``FraRateHelper``, indexed coupon);
* ``FUT``  - three-month EURIBOR futures by contract month (``DEC26``),
  quoted as the futures rate (100 - price) / 100; the forward is the
  futures rate minus the Hull-White convexity adjustment (QuantLib
  ``FuturesRateHelper``, adjustment from ``HullWhite.convexityBias``);
* ``IRS``  - par swap rate, fixed (annual 30/360 for EURIBOR) against the
  index (period = index tenor, ACT/360), discounted on the OIS curve
  (QuantLib ``SwapRateHelper(..., discountingCurve=ois)``);
* ``BASIS`` - tenor basis swap spread: this curve's index plus the spread
  against another index flat (``3s6s``: EURIBOR 3M + spread vs EURIBOR 6M),
  each leg at its own frequency, discounted on OIS (QuantLib
  ``IborIborBasisSwapRateHelper``). The other index's curve must be built
  first.

Spot is ``fixing_lag`` business days after the valuation date (moved to a
business day first). Rates are decimals; a ``BASIS`` quote is a spread.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .conventions import Date, Tenor, add_months, make_schedule, parse_tenor, year_fraction
from .curves import Curve, CurveQuote, Market, get_conventions, solve_pillars, swap_schedule
from .indices import (
    HullWhite,
    TermIndex,
    contract_label,
    futures_period,
    get_index,
    hw_term_futures_convexity,
    parse_contract,
    price_from_futures_rate,
)

PROJECTION_KINDS = ("DEPO", "FRA", "FUT", "IRS", "BASIS")
_FRA_RE = re.compile(r"^\s*(\d{1,3})\s*[Xx×]\s*(\d{1,3})\s*$")


# --------------------------------------------------------------------------
# Leg schedules shared with the swap pricers
# --------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def basis_leg_dates(index: TermIndex, start: Date, termination: Date) -> List[Date]:
    """Adjusted dates of one leg of a tenor basis swap: backward from the
    unadjusted ``termination`` every index tenor, modified following on
    the index calendar, no end-of-month rule (QuantLib ``MakeSchedule`` in
    ``IborIborBasisSwapRateHelper``)."""

    return make_schedule(start, termination, index.months, index.calendar, index.bdc, index.bdc, backward=True)


@lru_cache(maxsize=4096)
def irs_leg_dates(index: TermIndex, start: Date, termination: Date) -> Tuple[List[Date], List[Date]]:
    """(fixed, float) adjusted dates of a fixed/float swap against
    ``index``, as :class:`~derivx.rates.swaps.VanillaSwap` builds them."""

    cal = index.cal
    return (
        swap_schedule(start, termination, index.swap_fixed_months, cal, index.bdc),
        swap_schedule(start, termination, index.months, cal, index.bdc),
    )


def _unadjusted(start: Date, tenor: Tenor) -> Date:
    if tenor.unit == "W":
        return start + dt.timedelta(weeks=tenor.n)
    return add_months(start, tenor.months)


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------


Legs = Tuple[Tuple[Date, Date, float], ...]  # (start, end = pay date, accrual)


@dataclass(frozen=True)
class _Dates:
    start: Date
    end: Date
    tenor: Tenor
    fixed: Legs = ()
    floating: Legs = ()
    other: Legs = ()


@lru_cache(maxsize=8192)
def _resolve_dates(kind: str, text: str, asof: Date, index: TermIndex, basis_index: Optional[TermIndex]) -> _Dates:
    """Dates and accruals of a projection instrument (pure, cached: a
    rebuild on a bumped curve reuses them)."""

    cal, spot = index.cal, index.spot(asof)
    if kind == "DEPO":
        tenor = parse_tenor(text)
        if tenor.unit not in ("W", "M", "Y"):
            raise ValueError(f"{index.name} deposit tenor must be weeks, months or years, got {tenor}")
        return _Dates(spot, cal.advance(spot, tenor, index.bdc, eom=index.eom), tenor)
    if kind == "FRA":
        match = _FRA_RE.match(text)
        if not match:
            raise ValueError(f"FRA {text!r}: write months to start x months to end, e.g. 6X12")
        first, last = int(match.group(1)), int(match.group(2))
        if last - first != index.months:
            raise ValueError(
                f"FRA {first}X{last} spans {last - first} months; a {index.name} FRA spans {index.months} "
                f"(e.g. {first}X{first + index.months})"
            )
        start = cal.advance(spot, Tenor(first, "M"), index.bdc, eom=index.eom) if first else spot
        return _Dates(start, index.maturity_date(start), Tenor(last, "M", label=f"{first}X{last}"))
    if kind == "FUT":
        if not index.has_futures:
            raise ValueError(
                f"there are no futures on {index.name}: put three-month EURIBOR futures on the EURIBOR 3M curve"
            )
        year, month = parse_contract(text, asof)
        start, end = futures_period("FUT", year, month, index)
        if index.fixing_date(start) < asof:
            raise ValueError(f"FUT {contract_label(year, month)} has already fixed ({index.fixing_date(start)})")
        months = max(1, round((end - asof).days / 30.4375))
        return _Dates(start, end, Tenor(months, "M", label=contract_label(year, month)))
    tenor = parse_tenor(text)
    if tenor.unit not in ("W", "M", "Y"):
        raise ValueError(f"{kind} tenor must be weeks, months or years, got {tenor}")
    termination = _unadjusted(spot, tenor)
    if kind == "IRS":
        fixed, flt = irs_leg_dates(index, spot, termination)
        if len(flt) < 2:
            raise ValueError(f"IRS {tenor} is shorter than one {index.name} period")
        return _Dates(
            spot,
            max(fixed[-1], flt[-1]),
            tenor,
            fixed=tuple((s, e, year_fraction(s, e, index.swap_fixed_day_count)) for s, e in zip(fixed[:-1], fixed[1:])),
            floating=tuple((s, e, index.accrual(s, e)) for s, e in zip(flt[:-1], flt[1:])),
        )
    if basis_index is None:
        raise ValueError(
            f"BASIS {text} on {index.name} needs the other index's projection curve (build it first and pass basis=...)"
        )
    if basis_index.currency != index.currency or basis_index.months == index.months:
        raise ValueError(f"a tenor basis swap needs another {index.currency} tenor, got {basis_index.name}")
    base = basis_leg_dates(index, spot, termination)
    other = basis_leg_dates(basis_index, spot, termination)
    return _Dates(
        spot,
        max(base[-1], other[-1]),
        tenor,
        floating=tuple((s, e, index.accrual(s, e)) for s, e in zip(base[:-1], base[1:])),
        other=tuple((s, e, basis_index.accrual(s, e)) for s, e in zip(other[:-1], other[1:])),
    )


class ProjectionInstrument:
    """One quote of a projection curve, with dates resolved on ``asof``.

    ``npv(proj)`` is the value per unit notional of receiving the quoted
    rate (or spread) with ``proj`` as this index's projection curve and the
    fixed ``discount`` (and ``basis``) curves; zero when the curve reprices
    the quote. ``implied(proj)`` is the quote the curves imply.
    """

    def __init__(
        self,
        quote: CurveQuote,
        asof: Date,
        index: TermIndex,
        discount: Curve,
        basis: Optional[Curve] = None,
        basis_index: Optional[TermIndex] = None,
        convexity: Optional[HullWhite] = None,
        _dfs: Optional[Tuple[Mapping[Date, float], Mapping[Date, float]]] = None,
    ):
        kind = quote.kind.strip().upper()
        if kind not in PROJECTION_KINDS:
            raise ValueError(f"Unknown {index.name} instrument {quote.kind!r}; expected one of {PROJECTION_KINDS}")
        if kind == "BASIS" and basis is None:
            raise ValueError(
                f"BASIS {quote.tenor} on {index.name} needs the other index's projection curve "
                "(build it first and pass basis=...)"
            )
        self.quote, self.asof, self.index, self.kind = quote, asof, index, kind
        self.discount, self.basis, self.basis_index = discount, basis, basis_index
        self.convexity = convexity
        dates = _resolve_dates(kind, str(quote.tenor).strip().upper(), asof, index, basis_index)
        self.start, self.end, self.tenor = dates.start, dates.end, dates.tenor
        self.pillar = self.end
        self.fixed, self.floating, self.other = dates.fixed, dates.floating, dates.other
        self.convexity_adjustment = 0.0
        if kind == "FUT" and convexity is not None:
            t1, t2 = (self.start - asof).days / 365.0, (self.end - asof).days / 365.0
            self.convexity_adjustment = hw_term_futures_convexity(
                quote.rate, t1, t2, convexity.sigma, convexity.mean_reversion
            )
        self._cache(*(_dfs or ({}, {})))

    def _cache(self, discount_dfs: Mapping[Date, float], basis_dfs: Mapping[Date, float]) -> None:
        """Times (ACT/365F) and discount-curve factors, fixed while the
        projection curve is solved, as arrays for vectorised valuation
        (``discount_dfs`` / ``basis_dfs``: factors already looked up for
        these dates, see :func:`_prefetch`)."""

        def on(curve: Curve, table: Mapping[Date, float], dates: List[Date]) -> np.ndarray:
            if dates and all(d in table for d in dates):
                return np.array([table[d] for d in dates], dtype=float)
            return np.asarray(curve.df(dates), dtype=float)

        to_t = lambda d: (d - self.asof).days / 365.0  # noqa: E731
        self._t = np.array([to_t(self.start), to_t(self.end)])
        self._tau = self.index.accrual(self.start, self.end)
        if self.fixed:
            fixed_df = on(self.discount, discount_dfs, [e for _, e, _ in self.fixed])
            self._fixed_annuity = float(np.dot([tau for _, _, tau in self.fixed], fixed_df))
        self._flt_s = np.array([to_t(s) for s, _, _ in self.floating])
        self._flt_e = np.array([to_t(e) for _, e, _ in self.floating])
        self._flt_tau = np.array([tau for _, _, tau in self.floating])
        self._flt_df = on(self.discount, discount_dfs, [e for _, e, _ in self.floating])
        self._flt_annuity = float(np.dot(self._flt_tau, self._flt_df)) if self.floating else 0.0
        if self.other:
            # The other leg is fixed while this curve is solved: value it once.
            starts, ends = [s for s, _, _ in self.other], [e for _, e, _ in self.other]
            growth = on(self.basis, basis_dfs, starts) / on(self.basis, basis_dfs, ends)
            self._other_pv = float(np.dot(growth - 1.0, on(self.discount, discount_dfs, ends)))

    @property
    def years(self) -> float:
        """Tenor in years (a FRA's end, e.g. 1.0 for 6X12; a future's time
        to the end of its rate period), for ordering quotes and placing them
        on a shock profile."""

        if self.kind == "FUT":
            return (self.end - self.asof).days / 365.0
        return self.tenor.years

    # -- valuation -----------------------------------------------------------

    def _float_pv(self, proj: Curve) -> float:
        """Index leg value per unit notional: sum of tau F DF_ois (par
        coupons: each forward runs over its accrual period)."""

        growth = proj.df_times(self._flt_s) / proj.df_times(self._flt_e)
        return float(np.dot(growth - 1.0, self._flt_df))

    def implied(self, proj: Curve) -> float:
        """Quote the curves imply: rate (DEPO/FRA/IRS), futures rate (FUT)
        or spread (BASIS), decimal."""

        if self.kind in ("DEPO", "FRA", "FUT"):
            d_start, d_end = proj.df_times(self._t)
            return (d_start / d_end - 1.0) / self._tau + self.convexity_adjustment
        if self.kind == "IRS":
            return self._float_pv(proj) / self._fixed_annuity
        return (self._other_pv - self._float_pv(proj)) / self._flt_annuity

    def npv(self, proj: Curve) -> float:
        rate = self.quote.rate
        if self.kind in ("DEPO", "FRA", "FUT"):
            return rate - self.implied(proj)
        if self.kind == "IRS":
            return rate * self._fixed_annuity - self._float_pv(proj)
        return rate * self._flt_annuity + self._float_pv(proj) - self._other_pv


# --------------------------------------------------------------------------
# Curve
# --------------------------------------------------------------------------


class ProjectionCurve(Curve):
    """Projection curve of one term index: forwards come from it, discount
    factors from ``discount`` (the OIS curve it was bootstrapped on).
    ``basis`` / ``basis_index`` are the other index's curve and name for
    ``BASIS`` quotes; ``convexity`` the Hull-White parameters of its
    futures. ``df`` here is a projection pseudo-discount factor, not a price
    of money: discount cash flows on ``discount``."""

    is_projection = True

    def __init__(
        self,
        asof: Date,
        times: Sequence[float],
        dfs: Sequence[float],
        interpolation: str,
        pillar_dates: Sequence[Date],
        quotes: Sequence[CurveQuote],
        index: TermIndex,
        discount: Curve,
        instruments: Sequence[ProjectionInstrument],
        basis: Optional[Curve] = None,
        convexity: Optional[HullWhite] = None,
    ):
        conventions = discount.conventions or get_conventions(index.discount)
        super().__init__(asof, times, dfs, interpolation, pillar_dates, quotes, conventions, instruments, convexity)
        self.index = index
        self.discount = discount
        self.basis = basis
        uses_basis = basis is not None and any(q.kind.upper() == "BASIS" for q in quotes)
        self.basis_index: Optional[str] = basis.index.name if uses_basis else None

    def __repr__(self) -> str:
        return (
            f"ProjectionCurve({self.index.name!r}, asof={self.asof}, {len(self.ts) - 1} pillars, "
            f"{self.interpolation}, discount={self.discount.conventions.name if self.discount.conventions else '?'})"
        )

    @property
    def name(self) -> str:
        return self.index.name

    def rebuilt(
        self,
        bumps_bp: Union[float, Mapping[int, float], Sequence[float]] = 0.0,
        asof: Optional[Date] = None,
        discount: Optional[Curve] = None,
        basis: Optional[Curve] = None,
    ) -> "ProjectionCurve":
        """Rebuild from the same quotes bumped in bp, on ``discount`` (and
        ``basis``) when given, else on the curves it was built on. On a later
        ``asof`` (carry, roll-down) a future that has fixed by then is
        dropped: its rate is a fixing, no longer a quote of the curve."""

        quotes = self.bumped_quotes(bumps_bp)
        if asof is not None and asof > self.asof:
            quotes = [q for q, inst in zip(quotes, self.instruments) if not _fixed_by(inst, asof)]
            if not quotes:
                raise ValueError(f"every quote of the {self.index.name} curve has fixed by {asof}")
        discount = discount if discount is not None else self.discount
        basis = basis if basis is not None else self.basis
        same_inputs = (asof is None or asof == self.asof) and discount is self.discount and basis is self.basis
        return bootstrap_projection(
            asof or self.asof,
            quotes,
            self.index,
            discount,
            self.interpolation,
            basis=basis,
            convexity=self.convexity,
            _known=self.unchanged_pillars(quotes) if same_inputs else (),
            _guess=self.dfs[1:] if asof is None or asof == self.asof else (),
            _jacobian=self.solver_jacobian() if asof is None or asof == self.asof else None,
        )

    def forward(self, start: Date, end: Optional[Date] = None) -> float:
        """Index forward for a period starting ``start`` (to the index
        maturity unless ``end`` is given), on the index day count."""

        end = end or self.index.maturity_date(start)
        return self.forward_rate(start, end, self.index.day_count)


def _prefetch(
    quotes: Sequence[CurveQuote],
    asof: Date,
    index: TermIndex,
    discount: Curve,
    basis: Optional[Curve],
    basis_index: Optional[TermIndex],
) -> Tuple[Dict[Date, float], Dict[Date, float]]:
    """Discount (and basis curve) factors at every payment date the quotes'
    instruments use, each curve interpolated once (a rebuild of a bumped
    curve set builds every instrument again). Quotes that do not resolve are
    skipped: building their instrument raises the error."""

    pay: set = set()  # payment dates
    other: set = set()  # basis-leg period dates
    for quote in quotes:
        kind = quote.kind.strip().upper()
        if kind not in PROJECTION_KINDS:
            continue
        try:
            dates = _resolve_dates(kind, str(quote.tenor).strip().upper(), asof, index, basis_index)
        except ValueError:
            continue
        pay.update(e for _, e, _ in dates.fixed + dates.floating + dates.other)
        other.update(d for s, e, _ in dates.other for d in (s, e))

    def table(curve: Optional[Curve], days: set) -> Dict[Date, float]:
        if curve is None or not days:
            return {}
        days = sorted(days)
        return dict(zip(days, np.asarray(curve.df(days), dtype=float)))

    return table(discount, pay), table(basis, other)


def _fixed_by(inst: ProjectionInstrument, asof: Date) -> bool:
    """Whether a futures quote has fixed before ``asof`` (and so cannot be
    an instrument of a curve dated ``asof``)."""

    return inst.kind == "FUT" and inst.index.fixing_date(inst.start) < asof


def bootstrap_projection(
    asof: Date,
    quotes: Sequence[CurveQuote],
    index: Union[str, TermIndex],
    discount: Curve,
    interpolation: str = "loglinear",
    basis: Optional[Curve] = None,
    convexity: Optional[HullWhite] = None,
    tol: float = 1e-14,
    max_passes: int = 50,
    _known: Sequence[float] = (),
    _guess: Sequence[float] = (),
    _jacobian: Optional[np.ndarray] = None,
) -> ProjectionCurve:
    """Dual-curve bootstrap of ``index``'s projection curve given the
    ``discount`` curve (and ``basis``, the other index's curve, for
    ``BASIS`` quotes). ``convexity`` sets the futures adjustment (none if
    omitted). ``ValueError`` on bad quotes, pillars that coincide or a
    curve dated differently from ``asof``."""

    index = get_index(index)
    if discount.asof != asof:
        raise ValueError(f"the discount curve is dated {discount.asof}, the projection curve {asof}")
    if discount.conventions is not None and discount.conventions.currency != index.currency:
        raise ValueError(f"{index.name} is a {index.currency} index; the discount curve is {discount.conventions.name}")
    basis_index = None
    if basis is not None:
        basis_index = getattr(basis, "index", None)
        if basis_index is None:
            raise ValueError("the basis curve must be another index's projection curve")
        if basis.asof != asof:
            raise ValueError(f"the {basis_index.name} curve is dated {basis.asof}, the projection curve {asof}")
    if not quotes:
        raise ValueError(f"the {index.name} curve needs at least one quote")
    looked_up = _prefetch(quotes, asof, index, discount, basis, basis_index)
    built = [ProjectionInstrument(q, asof, index, discount, basis, basis_index, convexity, looked_up) for q in quotes]
    instruments, times, dfs = solve_pillars(asof, built, interpolation, tol, max_passes, _known, _guess, _jacobian)
    return ProjectionCurve(
        asof,
        times,
        dfs,
        interpolation,
        [i.pillar for i in instruments],
        [i.quote for i in instruments],
        index,
        discount,
        instruments,
        basis,
        convexity,
    )


# --------------------------------------------------------------------------
# Curve sets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionSpec:
    """Quotes and settings of one projection curve of a curve set."""

    index: str
    quotes: Tuple[CurveQuote, ...]
    interpolation: str = "loglinear"
    basis_index: Optional[str] = None


def build_curve_set(
    discount: Curve,
    specs: Sequence[ProjectionSpec],
    convexity: Optional[HullWhite] = None,
    govt: Optional[Curve] = None,
    index_fixings: Optional[Mapping[str, Mapping[Date, float]]] = None,
) -> Market:
    """Market with ``discount`` and one projection curve per spec, built
    in dependency order (a curve with ``BASIS`` quotes after its
    ``basis_index``)."""

    built: Dict[str, ProjectionCurve] = {}
    pending = list(specs)
    while pending:
        ready = [s for s in pending if s.basis_index is None or s.basis_index in built]
        if not ready:
            raise ValueError(
                "tenor basis curves need their other index: "
                + ", ".join(f"{s.index} -> {s.basis_index}" for s in pending)
            )
        for spec in ready:
            if spec.basis_index is None and any(q.kind.upper() == "BASIS" for q in spec.quotes):
                raise ValueError(f"the {spec.index} curve has BASIS quotes: say which index they are against")
            basis = built[spec.basis_index] if spec.basis_index else None
            built[get_index(spec.index).name] = bootstrap_projection(
                discount.asof, spec.quotes, spec.index, discount, spec.interpolation, basis=basis, convexity=convexity
            )
        pending = [s for s in pending if s not in ready]
    return Market(curve=discount, govt=govt, projections=built, index_fixings=index_fixings or {})


def forward_grid(market: Market, months: int, horizon_years: Optional[float] = None, step_days: int = 14) -> Dict:
    """Forwards on a grid of start dates (ACT/365F years from the valuation
    date): the OIS curve's simple ACT/360 forward for periods of ``months``
    months (30.4375 days each) and, for every projection index of that
    tenor, its forward over the same periods. Rates decimal."""

    asof = market.asof
    horizon = horizon_years or float(market.curve.ts[-1])
    days = np.arange(0, int(horizon * 365) - 31 * months, step_days)
    starts = [asof + dt.timedelta(days=int(d)) for d in days]
    out: Dict[str, object] = {"years": days / 365.0, "dates": starts}
    length = round(30.4375 * months)
    ends = [s + dt.timedelta(days=length) for s in starts]
    tau = length / 360.0  # ACT/360 accrual of every period (EURIBOR and €STR/SOFR count ACT/360)

    def forwards(curve: Curve) -> np.ndarray:
        return (np.asarray(curve.df(starts)) / np.asarray(curve.df(ends)) - 1.0) / tau

    out["ois"] = forwards(market.curve)
    for key in market.projections:
        curve = market.projections[key]
        index = getattr(curve, "index", None)
        if index is not None and index.months == months:
            out[key] = forwards(curve)
    return out


# --------------------------------------------------------------------------
# Illustrative quotes
# --------------------------------------------------------------------------


EURIBOR_INSTRUMENTS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "EURIBOR 6M": (
        ("DEPO", "6M"), ("FRA", "1X7"), ("FRA", "2X8"), ("FRA", "3X9"), ("FRA", "4X10"), ("FRA", "5X11"),
        ("FRA", "6X12"), ("FRA", "9X15"), ("FRA", "12X18"), ("IRS", "2Y"), ("IRS", "3Y"), ("IRS", "4Y"),
        ("IRS", "5Y"), ("IRS", "6Y"), ("IRS", "7Y"), ("IRS", "8Y"), ("IRS", "9Y"), ("IRS", "10Y"), ("IRS", "12Y"),
        ("IRS", "15Y"), ("IRS", "20Y"), ("IRS", "25Y"), ("IRS", "30Y"),
    ),
    "EURIBOR 3M": (
        ("DEPO", "3M"), ("FUT", "next"), ("FUT", "next"), ("FUT", "next"), ("FUT", "next"), ("FUT", "next"),
        ("FUT", "next"),
        # The same tenors as the 6M swaps: with pillars in common, the 3M forwards do not straddle a 6M pillar.
        ("BASIS", "2Y"), ("BASIS", "3Y"), ("BASIS", "4Y"), ("BASIS", "5Y"), ("BASIS", "6Y"), ("BASIS", "7Y"),
        ("BASIS", "8Y"), ("BASIS", "9Y"), ("BASIS", "10Y"), ("BASIS", "12Y"), ("BASIS", "15Y"), ("BASIS", "20Y"),
        ("BASIS", "25Y"), ("BASIS", "30Y"),
    ),
}  # fmt: skip
# Assumed instantaneous basis over the OIS curve, (years, bp): ILLUSTRATIVE, not market data.
ILLUSTRATIVE_BASIS_BP: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "EURIBOR 6M": ((0.0, 24.0), (2.0, 20.0), (10.0, 14.0), (30.0, 10.0)),
    "EURIBOR 3M": ((0.0, 14.0), (2.0, 12.0), (10.0, 8.0), (30.0, 6.0)),
}
BASIS_OF = {"EURIBOR 3M": "EURIBOR 6M"}  # the illustrative 3M curve is built from 3s6s basis swaps


def quarterly_contracts(asof: Date, index: TermIndex, count: int) -> List[str]:
    """The next ``count`` quarterly (Mar/Jun/Sep/Dec) futures that have not
    fixed, as contract labels."""

    out: List[str] = []
    year, month = asof.year, asof.month
    month = ((month - 1) // 3) * 3 + 3
    while len(out) < count:
        start, _ = futures_period("FUT", year, month, index)
        if index.fixing_date(start) >= asof:
            out.append(contract_label(year, month))
        month += 3
        if month > 12:
            year, month = year + 1, month - 12
    return out


def _synthetic(discount: Curve, profile: Sequence[Tuple[float, float]], horizon: float) -> Curve:
    """Projection pseudo-curve: the discount curve times exp(-integral of
    an instantaneous basis profile), on a daily grid."""

    days = np.arange(0, int(horizon * 365) + 2)
    t = days / 365.0
    xs, ys = np.array([p[0] for p in profile]), np.array([p[1] for p in profile]) * 1e-4
    basis = np.interp(t, xs, ys)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (basis[1:] + basis[:-1]) * np.diff(t))])
    dfs = discount.df_times(t) * np.exp(-integral)
    return Curve(discount.asof, t, dfs, "loglinear")


def illustrative_quotes(
    discount: Curve,
    index: Union[str, TermIndex],
    convexity: Optional[HullWhite] = None,
    basis_curve: Optional[Curve] = None,
    profile: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[CurveQuote]:
    """ILLUSTRATIVE quotes (not market data) for ``index``: priced off the
    discount curve plus an assumed basis (:data:`ILLUSTRATIVE_BASIS_BP`),
    so the curves they bootstrap are smooth and consistent with each other.
    ``basis_curve`` is the other index's curve for ``BASIS`` quotes.
    Futures quotes come back as futures rates (price = 100 - 100 x rate)."""

    index = get_index(index)
    profile = profile or ILLUSTRATIVE_BASIS_BP[index.name]
    synthetic = _synthetic(discount, profile, 32.0)
    basis_index = getattr(basis_curve, "index", None)
    contracts = iter(quarterly_contracts(discount.asof, index, 12)) if index.has_futures else iter(())
    out = []
    for kind, tenor in EURIBOR_INSTRUMENTS[index.name]:
        if kind == "BASIS" and basis_curve is None:
            continue
        if kind == "FUT":
            tenor = next(contracts)
        inst = ProjectionInstrument(CurveQuote(kind, tenor, 0.0), discount.asof, index, discount, basis_curve,
                                    basis_index, None)  # fmt: skip
        rate = inst.implied(synthetic)
        if kind == "FUT" and convexity is not None:
            t1, t2 = (inst.start - discount.asof).days / 365.0, (inst.end - discount.asof).days / 365.0
            # Solve futures rate = forward + adjustment(futures rate): the adjustment is nearly linear in the rate.
            fut = rate
            for _ in range(3):
                fut = rate + hw_term_futures_convexity(fut, t1, t2, convexity.sigma, convexity.mean_reversion)
            rate = fut
        out.append(CurveQuote(kind, tenor, round(rate, 8) if kind != "FUT" else _round_price(rate)))
    return out


def _round_price(futures_rate: float) -> float:
    """Futures rate of the price rounded to half a tick (0.0025)."""

    price = round(price_from_futures_rate(futures_rate) / 0.0025) * 0.0025
    return (100.0 - price) / 100.0
