"""Interest rate swaps: fixed/float (OIS, term index) and tenor basis.

A :class:`VanillaSwap` exchanges a fixed rate for a floating index:

* ``OIS``  - daily compounded overnight rate over each period, paid
  ``payment_lag`` business days after the period end (SOFR, ESTR, SONIA)
* ``TERM`` - simple term rate for each period, fixed ``fixing_lag``
  business days before the period start and paid at the period end

Every cash flow is discounted on ``market.curve`` (the OIS curve). OIS legs
are projected on it too (single-curve, the market standard for SOFR, €STR
and SONIA swaps). A TERM leg is projected on the projection curve of its
``projection`` index (``"EURIBOR 6M"``: multi-curve, see
:meth:`VanillaSwap.from_index`), or on ``market.curve`` when ``projection``
is None (the single-curve approximation). A :class:`BasisSwap` exchanges
one term index plus a spread for another (3s6s: EURIBOR 3M + spread vs
EURIBOR 6M), each leg on its own projection curve.

Schedules are generated backward from the maturity with the swap's
calendar and business-day convention by the curve's own
:func:`~derivx.rates.curves.swap_schedule` (basis swaps:
:func:`~derivx.rates.multicurve.basis_leg_dates`), so a par swap at a quoted
tenor reprices that quote.

Seasoned swaps take past rates from fixings: an OIS period's overnight
fixings come from ``market.fixing_curve`` up to the first fixing date on or
after ``market.asof`` (on a weekend or holiday the last fixing still covers
the days to the next business day, as in QuantLib); a TERM period whose
fixing date is before ``market.asof`` from ``market.fixing_curve``
(single-curve) or :meth:`Market.index_rate` (published ``index_fixings``).

Every value is in currency for the swap's notional and signed to the holder:
a receiver swap gains when rates fall, so its PV01 is negative.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from derivx.rates.conventions import BUSINESS_DAY_CONVENTIONS, Calendar, Date, add_months, parse_tenor, year_fraction
from derivx.rates.curves import (
    CONVENTIONS,
    SPREAD_KINDS,
    CurveConventions,
    Market,
    get_conventions,
    swap_schedule,
)
from derivx.rates.indices import TermIndex, get_index
from derivx.rates.multicurve import basis_leg_dates
from derivx.risk.sensitivities import Position, PvFunction, bumped_market, rate_curves, solve_hedge

FLOAT_INDICES = ("OIS", "TERM")
SWAP_KINDS = ("OIS", "IRS")
SWAP_DAY_COUNTS = ("ACT/360", "ACT/365F", "30/360", "30E/360", "ACT/ACT ISDA")
BENCHMARK_TENORS = ("2Y", "5Y", "10Y", "30Y")
DEFAULT_CONVENTIONS = CONVENTIONS["USD SOFR"]
CASHFLOW_COLUMNS = ["leg", "start", "end", "pay", "accrual", "rate", "amount", "df", "pv"]


@dataclass(frozen=True)
class SwapPeriod:
    """One accrual period of one leg (``fixing`` is set for TERM float
    periods only)."""

    leg: str
    start: Date
    end: Date
    pay: Date
    accrual: float
    fixing: Optional[Date] = None


@dataclass(frozen=True)
class VanillaSwap:
    """Fixed versus floating swap.

    Dates: the swap starts on ``start`` if given, otherwise at spot
    (``spot_lag`` business days after ``trade_date``) plus ``forward_start``.
    It ends on ``maturity`` if given, otherwise ``tenor`` after the start;
    both ends are adjusted with ``bdc``. ``trade_date=None`` means "spot from
    the valuation date", so the swap re-dates with the market; give a
    ``trade_date`` (or ``start``) to keep its dates fixed as the market moves
    forward, e.g. for carry. Rates and ``spread`` (added to the float rate)
    are decimals.
    """

    notional: float = 1e6
    fixed_rate: float = 0.0
    tenor: Optional[str] = "5Y"
    maturity: Optional[Date] = None
    start: Optional[Date] = None
    forward_start: Optional[str] = None
    receive_fixed: bool = True
    fixed_months: int = 12
    fixed_day_count: str = "ACT/360"
    float_months: int = 12
    float_day_count: str = "ACT/360"
    float_index: str = "OIS"
    spread: float = 0.0
    calendar: str = "USD"
    bdc: str = "MF"
    payment_lag: int = 2
    spot_lag: int = 2
    fixing_lag: int = 2
    trade_date: Optional[Date] = None
    projection: Optional[str] = None

    def __post_init__(self) -> None:
        if self.projection is not None:
            index = get_index(self.projection)
            object.__setattr__(self, "projection", index.name)
            if self.float_index != "TERM":
                raise ValueError(f"a {index.name} leg is a TERM leg (float_index='TERM'), not {self.float_index!r}")
            if self.float_months != index.months or self.float_day_count != index.day_count:
                raise ValueError(
                    f"a {index.name} leg resets every {index.months}M on {index.day_count}, not every "
                    f"{self.float_months}M on {self.float_day_count}"
                )
        if not np.isfinite(self.notional) or self.notional <= 0:
            raise ValueError("notional must be positive (use receive_fixed for the direction)")
        if not (np.isfinite(self.fixed_rate) and np.isfinite(self.spread)):
            raise ValueError("fixed_rate and spread must be finite decimals")
        if self.float_index not in FLOAT_INDICES:
            raise ValueError(f"float_index must be one of {FLOAT_INDICES}, got {self.float_index!r}")
        for label, dc in (("fixed_day_count", self.fixed_day_count), ("float_day_count", self.float_day_count)):
            if dc not in SWAP_DAY_COUNTS:
                raise ValueError(f"{label} must be one of {SWAP_DAY_COUNTS}, got {dc!r}")
        if self.fixed_months <= 0 or self.float_months <= 0:
            raise ValueError("fixed_months and float_months must be positive")
        if min(self.payment_lag, self.spot_lag, self.fixing_lag) < 0:
            raise ValueError("payment, spot and fixing lags must be non-negative")
        if self.bdc not in BUSINESS_DAY_CONVENTIONS:
            raise ValueError(f"bdc must be one of {BUSINESS_DAY_CONVENTIONS}, got {self.bdc!r}")
        if self.maturity is None:
            if not self.tenor:
                raise ValueError("give a tenor or a maturity date")
            if parse_tenor(self.tenor).unit == "D":
                raise ValueError(f"swap tenor must be weeks, months or years, got {self.tenor!r}")
        if self.forward_start and self.start is not None:
            raise ValueError("give either start or forward_start, not both")
        if self.forward_start and parse_tenor(self.forward_start).unit == "D":
            raise ValueError(f"forward start must be weeks, months or years, got {self.forward_start!r}")
        Calendar(self.calendar)  # validates the name

    @classmethod
    def from_conventions(
        cls,
        conv: Union[str, CurveConventions],
        tenor: Optional[str],
        fixed_rate: float,
        notional: float = 1e6,
        receive_fixed: bool = True,
        kind: str = "OIS",
        forward_start: Optional[str] = None,
        **fields: object,
    ) -> "VanillaSwap":
        """Market-standard swap for a curve's conventions.

        ``kind="OIS"`` uses ``ois_fixed_months`` and ``ois_day_count`` on both
        legs with the OIS ``payment_lag``; ``kind="IRS"`` uses the ``irs_*``
        fields with a TERM float leg paid on the period end and fixed
        ``spot_lag`` business days before it starts (2 for USD and EUR, 0 for
        GBP), so a spot swap's first period fixes on the trade date.
        ``fields`` overrides any other attribute (``spread``, ``trade_date``,
        ...).
        """

        conv = get_conventions(conv)
        kind = kind.upper()
        if kind == "OIS":
            legs = dict(
                fixed_months=conv.ois_fixed_months,
                fixed_day_count=conv.ois_day_count,
                float_months=conv.ois_fixed_months,
                float_day_count=conv.ois_day_count,
                float_index="OIS",
                payment_lag=conv.payment_lag,
            )
        elif kind == "IRS":
            legs = dict(
                fixed_months=conv.irs_fixed_months,
                fixed_day_count=conv.irs_fixed_day_count,
                float_months=conv.irs_float_months,
                float_day_count=conv.irs_float_day_count,
                float_index="TERM",
                payment_lag=0,
                fixing_lag=conv.spot_lag,
            )
        else:
            raise ValueError(f"kind must be one of {SWAP_KINDS}, got {kind!r}")
        base = dict(
            notional=notional,
            fixed_rate=fixed_rate,
            tenor=tenor,
            forward_start=forward_start,
            receive_fixed=receive_fixed,
            calendar=conv.calendar,
            bdc=conv.bdc,
            spot_lag=conv.spot_lag,
        )
        return cls(**{**base, **legs, **fields})

    @classmethod
    def from_index(
        cls,
        index: Union[str, TermIndex],
        tenor: Optional[str],
        fixed_rate: float,
        notional: float = 1e6,
        receive_fixed: bool = True,
        forward_start: Optional[str] = None,
        **fields: object,
    ) -> "VanillaSwap":
        """Market-standard swap against a term index (multi-curve): fixed
        leg on the index's swap conventions (EURIBOR: annual 30/360), float
        leg every index tenor on its day count, fixed ``fixing_lag`` business
        days before each period and paid at its end, spot ``fixing_lag``
        days after the trade date, index calendar, modified following. The
        float leg projects on ``market.projection(index)``; ``fields``
        overrides any other attribute."""

        index = get_index(index)
        base = dict(
            notional=notional,
            fixed_rate=fixed_rate,
            tenor=tenor,
            forward_start=forward_start,
            receive_fixed=receive_fixed,
            fixed_months=index.swap_fixed_months,
            fixed_day_count=index.swap_fixed_day_count,
            float_months=index.months,
            float_day_count=index.day_count,
            float_index="TERM",
            payment_lag=0,
            fixing_lag=index.fixing_lag,
            calendar=index.calendar,
            bdc=index.bdc,
            spot_lag=index.fixing_lag,
            projection=index.name,
        )
        return cls(**{**base, **fields})

    # -- dates ---------------------------------------------------------------

    @property
    def cal(self) -> Calendar:
        return Calendar(self.calendar)

    def spot_date(self, asof: Date) -> Date:
        cal = self.cal
        return cal.add_business_days(cal.adjust(self.trade_date or asof, "F"), self.spot_lag)

    def _anchor_start(self, asof: Date) -> Date:
        if self.start is not None:
            return self.start
        spot = self.spot_date(asof)
        return self.cal.advance(spot, self.forward_start, self.bdc) if self.forward_start else spot

    def effective_date(self, asof: Date) -> Date:
        return self.cal.adjust(self._anchor_start(asof), self.bdc)

    def termination_date(self, asof: Date) -> Date:
        """Unadjusted maturity (the schedule adjusts it with ``bdc``)."""

        if self.maturity is not None:
            return self.maturity
        tenor = parse_tenor(self.tenor)
        start = self._anchor_start(asof)
        return start + dt.timedelta(weeks=tenor.n) if tenor.unit == "W" else add_months(start, tenor.months)

    def schedule_dates(self, asof: Date) -> Tuple[List[Date], List[Date]]:
        """Adjusted (fixed, float) schedule dates."""

        periods = _periods(self, asof)
        out = []
        for leg in ("fixed", "float"):
            legs = [p for p in periods if p.leg == leg]
            out.append([legs[0].start] + [p.end for p in legs])
        return out[0], out[1]

    def periods(self, asof: Date) -> List[SwapPeriod]:
        return list(_periods(self, asof))

    # -- valuation -----------------------------------------------------------

    def _float_rates(self, market: Market, floats: Sequence[SwapPeriod]) -> np.ndarray:
        """Index rate of each float period: fixings for periods that fixed
        before (or, if published, on) the valuation date; otherwise the
        forward, computed for every such period in one curve call."""

        asof = market.asof
        if self.float_index == "OIS":
            split = self.cal.adjust(asof, "F")
            curve = market.curve
            future = [p.start >= split for p in floats]
        else:
            curve = market.projection(self.projection) if self.projection is not None else market.curve
            published = market.index_fixings.get(self.projection, {}) if self.projection is not None else {}
            future = [p.fixing > asof or (p.fixing == asof and p.fixing not in published) for p in floats]
        out = np.empty(len(floats))
        live = [i for i, f in enumerate(future) if f]
        if live:
            starts = np.asarray(curve.df([floats[i].start for i in live]), dtype=float)
            ends = np.asarray(curve.df([floats[i].end for i in live]), dtype=float)
            out[live] = (starts / ends - 1.0) / np.array([floats[i].accrual for i in live])
        for i, p in enumerate(floats):
            if future[i]:
                continue
            if self.float_index == "OIS":
                out[i] = (_compounded_growth(market, p.start, p.end, self.cal) - 1.0) / p.accrual
            elif self.projection is not None:
                out[i] = market.index_rate(self.projection, p.start, p.end, self.float_day_count, p.fixing)
            else:
                out[i] = market.projected_rate(p.start, p.end, self.float_day_count, p.fixing)
        return out

    def _flow_arrays(self, market: Market) -> Tuple[Tuple[SwapPeriod, ...], np.ndarray, np.ndarray]:
        """(periods, rates incl. spread, amounts signed to the holder) of
        every period, paid or not."""

        sign = 1.0 if self.receive_fixed else -1.0
        periods = _periods(self, market.asof)
        accrual = np.array([p.accrual for p in periods])
        is_float = np.array([p.leg == "float" for p in periods])
        index_rates = np.zeros(len(periods))
        index_rates[is_float] = self._float_rates(market, [p for p in periods if p.leg == "float"])
        rates = np.where(is_float, index_rates + self.spread, self.fixed_rate)
        amounts = np.where(is_float, -sign, sign) * self.notional * rates * accrual
        return periods, rates, amounts

    def _flows(self, market: Market) -> pd.DataFrame:
        """Every cash flow (paid or not), signed to the holder."""

        periods, rates, amounts = self._flow_arrays(market)
        rows = [(p.leg, p.start, p.end, p.pay, p.accrual, r, a) for p, r, a in zip(periods, rates, amounts)]
        return pd.DataFrame(rows, columns=CASHFLOW_COLUMNS[:7])

    def cashflows(self, market: Market) -> pd.DataFrame:
        """Cash flows still to be paid (pay date after ``market.asof``) with
        discount factors and PVs; amounts are signed to the holder."""

        flows = self._flows(market)
        flows = flows[flows["pay"] > market.asof].reset_index(drop=True)
        flows["df"] = np.asarray(market.curve.df(list(flows["pay"])), dtype=float) if len(flows) else []
        flows["pv"] = flows["amount"] * flows["df"]
        return flows

    def pv(self, market: Market) -> float:
        periods, _, amounts = self._flow_arrays(market)
        live = [i for i, p in enumerate(periods) if p.pay > market.asof]
        if not live:
            return 0.0
        dfs = np.asarray(market.curve.df([periods[i].pay for i in live]), dtype=float)
        return float(np.dot(amounts[live], dfs))

    def pv_fixed(self, market: Market) -> float:
        """Signed PV of the fixed leg (``pv = pv_fixed + pv_float``)."""

        flows = self.cashflows(market)
        return float(flows.loc[flows["leg"] == "fixed", "pv"].sum())

    def pv_float(self, market: Market) -> float:
        """Signed PV of the floating leg, spread included."""

        flows = self.cashflows(market)
        return float(flows.loc[flows["leg"] == "float", "pv"].sum())

    def annuity(self, market: Market) -> float:
        """PV of 1bp a year on the remaining fixed leg (currency, positive)."""

        flows = self.cashflows(market)
        fixed = flows[flows["leg"] == "fixed"]
        return float(self.notional * 1e-4 * (fixed["accrual"] * fixed["df"]).sum())

    def par_rate(self, market: Market) -> float:
        """Fixed rate that makes the remaining cash flows worth zero."""

        flows = self.cashflows(market)
        fixed, flt = flows[flows["leg"] == "fixed"], flows[flows["leg"] == "float"]
        annuity = float((fixed["accrual"] * fixed["df"]).sum())
        if annuity <= 0:
            raise ValueError("the swap has no fixed payments left to solve a par rate for")
        float_value = float((flt["rate"] * flt["accrual"] * flt["df"]).sum())
        return float_value / annuity

    def cash(self, market: Market, start: Date, end: Date) -> float:
        """Net cash paid to the holder with pay dates in (start, end]."""

        flows = self._flows(market)
        window = (flows["pay"] > start) & (flows["pay"] <= end)
        return float(flows.loc[window, "amount"].sum())

    # -- display -------------------------------------------------------------

    @property
    def label(self) -> str:
        side = "REC" if self.receive_fixed else "PAY"
        length = self.tenor if self.maturity is None else f"to {self.maturity.isoformat()}"
        fwd = f"{self.forward_start} fwd " if self.forward_start else ""
        return f"{side} {self.fixed_rate * 100:.4f}% {fwd}{length} {self.projection or self.float_index}"


@lru_cache(maxsize=512)
def _periods(swap: VanillaSwap, asof: Date) -> Tuple[SwapPeriod, ...]:
    cal = swap.cal
    effective, termination = swap.effective_date(asof), swap.termination_date(asof)

    def pay(day: Date) -> Date:
        return cal.add_business_days(day, swap.payment_lag) if swap.payment_lag else day

    if cal.adjust(termination, swap.bdc) <= effective:
        raise ValueError(f"swap maturity {termination} must be after its start {effective}")
    out = []
    for s, e in _pairs(swap_schedule(effective, termination, swap.fixed_months, cal, swap.bdc)):
        out.append(SwapPeriod("fixed", s, e, pay(e), year_fraction(s, e, swap.fixed_day_count)))
    for s, e in _pairs(swap_schedule(effective, termination, swap.float_months, cal, swap.bdc)):
        fixing = cal.add_business_days(s, -swap.fixing_lag) if swap.float_index == "TERM" else None
        out.append(SwapPeriod("float", s, e, pay(e), year_fraction(s, e, swap.float_day_count), fixing))
    return tuple(out)


def _pairs(dates: Sequence[Date]) -> List[Tuple[Date, Date]]:
    return list(zip(dates[:-1], dates[1:]))


def _compounded_growth(market: Market, start: Date, end: Date, cal: Calendar) -> float:
    """Overnight growth over [start, end]: fixings up to the first fixing date
    on or after the valuation date, the curve after it.

    Same as :meth:`Market.compounded_growth` on a business day; on a weekend
    or holiday the last fixing still runs to the next business day.
    """

    split = cal.adjust(market.asof, "F")
    fixings, curve = market.fixing_curve, market.curve
    if start >= split:
        return curve.df(start) / curve.df(end)
    if end <= split:
        return fixings.df(start) / fixings.df(end)
    return fixings.df(start) / fixings.df(split) * curve.df(split) / curve.df(end)


# --------------------------------------------------------------------------
# Tenor basis swaps
# --------------------------------------------------------------------------

BASIS_COLUMNS = ["leg", "fixing", "start", "end", "pay", "accrual", "rate", "amount", "df", "pv"]


@dataclass(frozen=True)
class BasisSwap:
    """Float/float tenor basis swap: ``spread_index`` plus ``spread`` against
    ``other_index`` flat (3s6s: EURIBOR 3M + spread vs EURIBOR 6M).

    Each leg resets every period of its own index, accrues on the index day
    count and pays at the period end; rates fix ``fixing_lag`` business days
    before each period (the index's). Dates: spot is the spread index's
    fixing lag after ``trade_date`` (``None``: the valuation date, so the
    swap re-dates with the market), plus ``forward_start``; ``tenor`` runs
    from there (or the swap ends on ``maturity``); legs are rolled backward
    from the unadjusted end, modified following (as QuantLib's
    ``IborIborBasisSwapRateHelper``), so a par swap reprices a ``BASIS``
    curve quote. Every flow is discounted on ``market.curve``. Values are in
    currency and signed to the holder: ``receive_spread_leg`` receives the
    spread index plus the spread.
    """

    notional: float = 1e6
    spread: float = 0.0
    tenor: Optional[str] = "5Y"
    maturity: Optional[Date] = None
    start: Optional[Date] = None
    forward_start: Optional[str] = None
    spread_index: str = "EURIBOR 3M"
    other_index: str = "EURIBOR 6M"
    receive_spread_leg: bool = True
    trade_date: Optional[Date] = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.notional) or self.notional <= 0:
            raise ValueError("notional must be positive (use receive_spread_leg for the direction)")
        if not np.isfinite(self.spread):
            raise ValueError("spread must be a finite decimal")
        a, b = get_index(self.spread_index), get_index(self.other_index)
        object.__setattr__(self, "spread_index", a.name)
        object.__setattr__(self, "other_index", b.name)
        if a.currency != b.currency or a.months == b.months:
            raise ValueError(f"a tenor basis swap needs two tenors of one currency, got {a.name} and {b.name}")
        if self.maturity is None:
            if not self.tenor or parse_tenor(self.tenor).unit not in ("W", "M", "Y"):
                raise ValueError(f"basis swap tenor must be weeks, months or years, got {self.tenor!r}")
        if self.forward_start and self.start is not None:
            raise ValueError("give either start or forward_start, not both")

    @property
    def indices(self) -> Tuple[TermIndex, TermIndex]:
        return get_index(self.spread_index), get_index(self.other_index)

    def effective_date(self, asof: Date) -> Date:
        if self.start is not None:
            return self.start
        index = self.indices[0]
        spot = index.spot(self.trade_date or asof)
        return index.cal.advance(spot, self.forward_start, index.bdc) if self.forward_start else spot

    def termination_date(self, asof: Date) -> Date:
        if self.maturity is not None:
            return self.maturity
        tenor = parse_tenor(self.tenor)
        start = self.effective_date(asof)
        return start + dt.timedelta(weeks=tenor.n) if tenor.unit == "W" else add_months(start, tenor.months)

    def periods(self, asof: Date) -> List[SwapPeriod]:
        return list(_basis_periods(self, asof))

    def _flow_arrays(self, market: Market) -> Tuple[Tuple[SwapPeriod, ...], np.ndarray, np.ndarray]:
        """(periods, rates incl. spread, amounts signed to the holder)."""

        sign = 1.0 if self.receive_spread_leg else -1.0
        periods = _basis_periods(self, market.asof)
        rates = np.zeros(len(periods))
        for leg, name in (("spread", self.spread_index), ("other", self.other_index)):
            idx = [i for i, p in enumerate(periods) if p.leg == leg]
            rates[idx] = _index_rates(market, name, [periods[i] for i in idx]) + (self.spread if leg == "spread" else 0)
        accrual = np.array([p.accrual for p in periods])
        signs = np.array([sign if p.leg == "spread" else -sign for p in periods])
        return periods, rates, signs * self.notional * rates * accrual

    def _flows(self, market: Market) -> pd.DataFrame:
        periods, rates, amounts = self._flow_arrays(market)
        rows = [(p.leg, p.fixing, p.start, p.end, p.pay, p.accrual, r, a) for p, r, a in zip(periods, rates, amounts)]
        return pd.DataFrame(rows, columns=BASIS_COLUMNS[:8])

    def cashflows(self, market: Market) -> pd.DataFrame:
        """Flows still to be paid, with OIS discount factors and PVs, signed
        to the holder (``leg`` is ``spread`` or ``other``)."""

        flows = self._flows(market)
        flows = flows[flows["pay"] > market.asof].reset_index(drop=True)
        flows["df"] = np.asarray(market.curve.df(list(flows["pay"])), dtype=float) if len(flows) else []
        flows["pv"] = flows["amount"] * flows["df"]
        return flows

    def pv(self, market: Market) -> float:
        periods, _, amounts = self._flow_arrays(market)
        live = [i for i, p in enumerate(periods) if p.pay > market.asof]
        if not live:
            return 0.0
        dfs = np.asarray(market.curve.df([periods[i].pay for i in live]), dtype=float)
        return float(np.dot(amounts[live], dfs))

    def annuity(self, market: Market) -> float:
        """PV of 1bp a year on the remaining spread leg (currency, positive)."""

        flows = self.cashflows(market)
        leg = flows[flows["leg"] == "spread"]
        return float(self.notional * 1e-4 * (leg["accrual"] * leg["df"]).sum())

    def par_spread(self, market: Market) -> float:
        """Spread on the spread leg that makes the remaining flows worth zero."""

        flows = self.cashflows(market)
        spread_leg, other = flows[flows["leg"] == "spread"], flows[flows["leg"] == "other"]
        annuity = float((spread_leg["accrual"] * spread_leg["df"]).sum())
        if annuity <= 0:
            raise ValueError("the basis swap has no spread-leg payments left to solve a par spread for")
        index_value = float(((spread_leg["rate"] - self.spread) * spread_leg["accrual"] * spread_leg["df"]).sum())
        other_value = float((other["rate"] * other["accrual"] * other["df"]).sum())
        return (other_value - index_value) / annuity

    def cash(self, market: Market, start: Date, end: Date) -> float:
        flows = self._flows(market)
        window = (flows["pay"] > start) & (flows["pay"] <= end)
        return float(flows.loc[window, "amount"].sum())

    @property
    def label(self) -> str:
        side = "REC" if self.receive_spread_leg else "PAY"
        a, b = self.indices
        length = self.tenor if self.maturity is None else f"to {self.maturity.isoformat()}"
        fwd = f"{self.forward_start} fwd " if self.forward_start else ""
        return f"{side} {a.months}s{b.months}s {fwd}{length} {self.spread * 1e4:+.2f}bp"


@lru_cache(maxsize=512)
def _basis_periods(swap: BasisSwap, asof: Date) -> Tuple[SwapPeriod, ...]:
    effective, termination = swap.effective_date(asof), swap.termination_date(asof)
    out = []
    for leg, index in (("spread", swap.indices[0]), ("other", swap.indices[1])):
        if index.cal.adjust(termination, index.bdc) <= effective:
            raise ValueError(f"basis swap maturity {termination} must be after its start {effective}")
        for s, e in _pairs(basis_leg_dates(index, effective, termination)):
            out.append(SwapPeriod(leg, s, e, e, index.accrual(s, e), index.fixing_date(s)))
    return tuple(out)


def _index_rates(market: Market, index: str, periods: Sequence[SwapPeriod]) -> np.ndarray:
    """Rates of a term index for accrual periods: published or past fixings
    one by one, forwards of the periods still to fix in one curve call."""

    asof = market.asof
    published = market.index_fixings.get(index, {})
    future = [p.fixing > asof or (p.fixing == asof and p.fixing not in published) for p in periods]
    out = np.empty(len(periods))
    live = [i for i, f in enumerate(future) if f]
    if live:
        curve = market.projection(index)
        starts = np.asarray(curve.df([periods[i].start for i in live]), dtype=float)
        ends = np.asarray(curve.df([periods[i].end for i in live]), dtype=float)
        out[live] = (starts / ends - 1.0) / np.array([periods[i].accrual for i in live])
    day_count = get_index(index).day_count
    for i, p in enumerate(periods):
        if not future[i]:
            out[i] = market.index_rate(index, p.start, p.end, day_count, p.fixing)
    return out


def basis_position(swap: BasisSwap, name: Optional[str] = None) -> Position:
    return Position(name=name or swap.label, pv=swap.pv, kind="basis swap", cash=swap.cash, details=asdict(swap))


# --------------------------------------------------------------------------
# Helpers: par rates, positions, hedging
# --------------------------------------------------------------------------


def _market_conventions(market: Market, conventions: Union[str, CurveConventions, None]) -> CurveConventions:
    if conventions is not None:
        return get_conventions(conventions)
    return market.curve.conventions or DEFAULT_CONVENTIONS


def par_swap_rate(
    market: Market,
    tenor: str,
    conventions: Union[str, CurveConventions, None] = None,
    kind: str = "OIS",
    forward_start: Optional[str] = None,
    index: Optional[str] = None,
) -> float:
    """Par rate of a spot (or forward-starting) swap traded on
    ``market.asof``; conventions default to the curve's own. With ``index``
    the swap is fixed against that term index (multi-curve)."""

    if index is not None:
        swap = VanillaSwap.from_index(index, tenor, 0.0, forward_start=forward_start, trade_date=market.asof)
    else:
        conv = _market_conventions(market, conventions)
        swap = VanillaSwap.from_conventions(
            conv, tenor, 0.0, kind=kind, forward_start=forward_start, trade_date=market.asof
        )
    return swap.par_rate(market)


def swap_position(swap: VanillaSwap, name: Optional[str] = None) -> Position:
    return Position(name=name or swap.label, pv=swap.pv, kind="swap", cash=swap.cash, details=asdict(swap))


@dataclass(frozen=True)
class CurveBumps:
    """Markets with one curve rebuilt from bumped quotes, built once and
    shared by any number of positions (a rebuild costs far more than a swap
    PV, especially on a monotone-convex curve).

    ``up[i]``/``down[i]`` move quote ``i`` alone by +/-``bp``; ``rows``
    describe the quotes. ``parallel`` (optional) moves every quote. The
    numbers match :mod:`derivx.risk.sensitivities` exactly.
    """

    rows: Tuple[Dict[str, object], ...]
    up: Tuple[Market, ...]
    down: Tuple[Market, ...]
    bp: float = 1.0
    parallel: Optional[Tuple[Market, Market]] = None

    def key_rate_matrix(self, pvs: Sequence[PvFunction]) -> np.ndarray:
        """Key-rate PV01s, quotes x positions (``key_rate_pv01`` per column)."""

        out = np.zeros((len(self.rows), len(pvs)))
        for i, (up, down) in enumerate(zip(self.up, self.down)):
            out[i] = [(pv(up) - pv(down)) / (2.0 * self.bp) for pv in pvs]
        return out

    def parallel_pv01(self, pv: PvFunction) -> float:
        """Same as ``parallel_pv01``: P&L for +bp on every quote."""

        if self.parallel is None:
            raise ValueError("built without parallel bumps")
        up, down = self.parallel
        return (pv(up) - pv(down)) / (2.0 * self.bp)


def curve_bumps(market: Market, which: str = "curve", bp: float = 1.0, parallel: bool = True) -> CurveBumps:
    """Rebuild ``market.<which>`` once per bumped quote (and twice more for
    the parallel move when ``parallel``)."""

    if which not in market.curve_keys:
        raise ValueError(f"which must be one of {list(market.curve_keys)}, got {which!r}")
    curve = market.get_curve(which)
    if curve is None or not curve.quotes:
        raise ValueError(f"market {which!r} curve has no quotes to bump")
    rows = tuple(
        {"index": i, "label": f"{q.tenor} {q.kind}", "tenor": q.tenor, "years": inst.years}
        for i, (q, inst) in enumerate(zip(curve.quotes, curve.instruments))
    )
    return CurveBumps(
        rows=rows,
        up=tuple(bumped_market(market, which, {i: bp}) for i in range(len(rows))),
        down=tuple(bumped_market(market, which, {i: -bp}) for i in range(len(rows))),
        bp=bp,
        parallel=(bumped_market(market, which, bp), bumped_market(market, which, -bp)) if parallel else None,
    )


def key_rate_matrix(
    pvs: Sequence[PvFunction],
    market: Market,
    which: str = "curve",
    bp: float = 1.0,
    bumps: Optional[CurveBumps] = None,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    """Key-rate PV01 ladders of several positions from one set of bumped
    curves: (rows describing each quote, matrix of quotes x positions).

    Same numbers as :func:`derivx.risk.sensitivities.key_rate_pv01` for each
    position, at a fraction of the rebuilds. ``bumps`` (from
    :func:`curve_bumps` on this market) reuses curves already rebuilt; its
    own ``bp`` then applies.
    """

    bumps = bumps if bumps is not None else curve_bumps(market, which, bp, parallel=False)
    return [dict(r) for r in bumps.rows], bumps.key_rate_matrix(pvs)


def benchmark_swaps(
    market: Market,
    tenors: Sequence[str] = BENCHMARK_TENORS,
    conventions: Union[str, CurveConventions, None] = None,
    kind: str = "OIS",
    notional: float = 1e6,
) -> List[VanillaSwap]:
    """Spot par receiver swaps traded on ``market.asof``."""

    conv = _market_conventions(market, conventions)
    out = []
    for tenor in tenors:
        swap = VanillaSwap.from_conventions(conv, tenor, 0.0, notional=notional, kind=kind, trade_date=market.asof)
        out.append(replace(swap, fixed_rate=swap.par_rate(market)))
    return out


def hedge_with_benchmarks(
    position: Position,
    market: Market,
    tenors: Sequence[str] = BENCHMARK_TENORS,
    conventions: Union[str, CurveConventions, None] = None,
    kind: str = "OIS",
    unit_notional: float = 1e6,
    bp: float = 1.0,
    bumps: Optional[CurveBumps] = None,
) -> Dict[str, object]:
    """Par swaps that flatten a position's key-rate ladder (least squares).

    Returns:

    * ``hedges`` - rows ``benchmark``, ``notional`` (signed: positive means
      receive fixed), ``direction``, ``par_rate`` and ``pv01`` of the hedge
    * ``positions`` - the hedge swaps as :class:`Position` objects
    * ``ladder`` - rows ``label``, ``tenor``, ``years``, ``position``,
      ``hedge`` and ``residual`` key-rate PV01
    * ``residual`` - the residual ladder as an array
    * ``position_pv01``, ``hedge_pv01``, ``hedged_pv01`` (ladder sums) and
      ``risk_removed`` (share of the squared ladder removed)

    Least-squares notionals under a cent are set to zero. ``bumps`` reuses
    curves rebuilt by :func:`curve_bumps`.
    """

    tenors = [str(t).strip().upper() for t in tenors]
    if not tenors:
        raise ValueError("pick at least one benchmark tenor")
    if len(set(tenors)) != len(tenors):
        raise ValueError(f"benchmark tenors must be distinct: {tenors}")
    units = benchmark_swaps(market, tenors, conventions, kind, unit_notional)
    rows, matrix = key_rate_matrix([position.pv] + [u.pv for u in units], market, bp=bp, bumps=bumps)
    target, unit_ladders = matrix[:, 0], matrix[:, 1:]
    solved = solve_hedge(target, unit_ladders, tenors)
    multiples = np.array([solved["multiples"][t] for t in tenors])
    # A benchmark the ladder does not load on gets least-squares noise: drop it.
    multiples[np.abs(multiples) * unit_notional < 0.01] = 0.0
    hedge_ladder = unit_ladders @ multiples
    residual = target + hedge_ladder
    base = float(target @ target)
    risk_removed = 1.0 - float(residual @ residual) / base if base > 0 else 1.0
    hedges, positions = [], []
    for tenor, unit, x, column in zip(tenors, units, multiples, unit_ladders.T):
        hedges.append(
            {
                "benchmark": tenor,
                "notional": float(x * unit_notional),
                "direction": "Receive fixed" if x > 0 else "Pay fixed" if x < 0 else "-",
                "par_rate": unit.fixed_rate,
                "pv01": float(x * column.sum()),
            }
        )
        if x != 0:
            swap = replace(unit, notional=abs(float(x)) * unit_notional, receive_fixed=bool(x > 0))
            positions.append(swap_position(swap, name=f"Hedge {swap.label}"))
    ladder_rows = [
        {**{k: r[k] for k in ("label", "tenor", "years")}, "position": t, "hedge": h, "residual": t + h}
        for r, t, h in zip(rows, target, hedge_ladder)
    ]
    return {
        "hedges": hedges,
        "positions": positions,
        "ladder": ladder_rows,
        "residual": residual,
        "position_pv01": float(target.sum()),
        "hedge_pv01": float(hedge_ladder.sum()),
        "hedged_pv01": float(residual.sum()),
        "risk_removed": risk_removed,
    }


# --------------------------------------------------------------------------
# Multi-curve risk and hedging
# --------------------------------------------------------------------------


def risk_curves(market: Market) -> Tuple[str, ...]:
    """Curves a swap book's key-rate risk is bucketed on: the discount curve
    and every projection curve with quotes of its own (the Treasury curve
    does not price swaps; a projection that is the discount curve moves
    with it)."""

    return rate_curves(market)


def benchmark_instruments(
    market: Market, key: str, tenors: Sequence[str], unit_notional: float = 1e6
) -> List[Tuple[str, Position]]:
    """Par hedge instruments for one curve, traded on ``market.asof``:
    OIS swaps for the discount curve; for a projection curve, swaps fixed
    against its index, or tenor basis swaps against its basis index when the
    curve is built from ``BASIS`` quotes. Returns (label, position) pairs;
    a positive multiple receives fixed (or the spread leg)."""

    out = []
    if key == "curve":
        for swap in benchmark_swaps(market, tenors, kind="OIS", notional=unit_notional):
            out.append((f"{swap.tenor} OIS", swap_position(swap)))
        return out
    curve = market.get_curve(key)
    basis_index = getattr(curve, "basis_index", None)
    for tenor in tenors:
        if basis_index:
            basis = BasisSwap(
                unit_notional, 0.0, tenor, spread_index=key, other_index=basis_index, trade_date=market.asof
            )
            out.append(
                (
                    f"{tenor} {get_index(key).months}s{get_index(basis_index).months}s",
                    basis_position(replace(basis, spread=basis.par_spread(market))),
                )
            )
        else:
            irs = VanillaSwap.from_index(key, tenor, 0.0, notional=unit_notional, trade_date=market.asof)
            out.append((f"{tenor} IRS {key}", swap_position(replace(irs, fixed_rate=irs.par_rate(market)))))
    return out


def multi_curve_ladders(
    pvs: Sequence[PvFunction],
    market: Market,
    curves: Optional[Sequence[str]] = None,
    bp: float = 1.0,
    bumps: Optional[Mapping[str, CurveBumps]] = None,
) -> Dict[str, Tuple[List[Dict[str, object]], np.ndarray]]:
    """Key-rate PV01 ladders on every curve: ``{curve key: (rows, quotes x
    positions)}``. Each entry is the P&L for +``bp`` on one quote of that
    curve with every curve built on it rebuilt (quotes held), so a EURIBOR
    swap's discount ladder carries its response through the projection
    curve, and its projection ladder its index risk."""

    curves = tuple(curves) if curves is not None else risk_curves(market)
    out = {}
    for key in curves:
        given = (bumps or {}).get(key)
        out[key] = key_rate_matrix(pvs, market, which=key, bp=bp, bumps=given)
    return out


@dataclass(frozen=True)
class HedgeSet:
    """Benchmark hedge instruments and their stacked key-rate ladders on
    one market (independent of the position hedged, so it can be reused).
    ``matrix`` is (every quote of every curve in ``curves``) x units."""

    units: Tuple[Tuple[str, str, Position], ...]
    curves: Tuple[str, ...]
    rows: Tuple[Tuple[Dict[str, object], ...], ...]
    matrix: np.ndarray
    unit_notional: float


def hedge_set(
    market: Market,
    benchmarks: Mapping[str, Sequence[str]],
    unit_notional: float = 1e6,
    bp: float = 1.0,
    bumps: Optional[Mapping[str, CurveBumps]] = None,
) -> HedgeSet:
    """Par benchmark instruments per curve (see :func:`benchmark_instruments`)
    and their key-rate ladders on every risk curve."""

    units: List[Tuple[str, str, Position]] = []
    for key, tenors in benchmarks.items():
        tenors = [str(t).strip().upper() for t in tenors]
        if len(set(tenors)) != len(tenors):
            raise ValueError(f"{key} benchmark tenors must be distinct: {tenors}")
        units += [(key, label, unit) for label, unit in benchmark_instruments(market, key, tenors, unit_notional)]
    if not units:
        raise ValueError("pick at least one benchmark")
    curves = risk_curves(market)
    ladders = multi_curve_ladders([u.pv for _, _, u in units], market, curves, bp, bumps)
    return HedgeSet(
        units=tuple(units),
        curves=curves,
        rows=tuple(tuple(ladders[k][0]) for k in curves),
        matrix=np.vstack([ladders[k][1] for k in curves]),
        unit_notional=unit_notional,
    )


def hedge_multi_curve(
    position: Position,
    market: Market,
    benchmarks: Optional[Mapping[str, Sequence[str]]] = None,
    unit_notional: float = 1e6,
    bp: float = 1.0,
    bumps: Optional[Mapping[str, CurveBumps]] = None,
    hedges: Optional[HedgeSet] = None,
    ladders: Optional[Mapping[str, Tuple[List[Dict[str, object]], np.ndarray]]] = None,
) -> Dict[str, object]:
    """Par instruments on several curves that flatten a position's stacked
    key-rate ladders (every quote of every risk curve) in least squares.

    ``benchmarks`` maps curve keys to tenors (see
    :func:`benchmark_instruments`); or pass ``hedges`` from
    :func:`hedge_set`, and the position's ``ladders`` ({curve: (rows,
    values)} as from :func:`multi_curve_ladders`), to reuse them. Returns
    ``hedges`` (rows ``curve``, ``benchmark``, ``notional`` signed: positive
    receives fixed or the spread leg, ``direction``, ``par`` rate or
    spread, ``pv01`` the sized hedge's P&L for +1bp on every rate quote,
    ``basis_pv01`` for +1bp on every tenor basis spread), ``positions``,
    ``ladders`` ({curve: rows ``label``, ``kind``, ``position``, ``hedge``,
    ``residual``}), the ladder sums ``position_pv01`` and ``hedged_pv01``
    per curve, and ``risk_removed`` (share of the squared stacked ladder
    removed).
    """

    if hedges is None:
        if benchmarks is None:
            raise ValueError("give benchmarks or a hedge set")
        hedges = hedge_set(market, benchmarks, unit_notional, bp, bumps)
    curves = hedges.curves
    if ladders is None:
        ladders = multi_curve_ladders([position.pv], market, curves, bp, bumps)
    columns = [np.asarray(ladders[k][1], dtype=float) for k in curves]  # (quotes x 1) matrices or plain vectors
    target = np.concatenate([c[:, 0] if c.ndim == 2 else c for c in columns])
    matrix = hedges.matrix
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("the hedge set was built on another market")
    names = [f"{key}: {label}" for key, label, _ in hedges.units]
    solved = solve_hedge(target, matrix, names)
    multiples = np.array([solved["multiples"][n] for n in names])
    multiples[np.abs(multiples) * hedges.unit_notional < 0.01] = 0.0
    hedge_ladder = matrix @ multiples
    residual = target + hedge_ladder
    base = float(target @ target)
    kinds = [_quote_kind(market, key, r) for key, rows_k in zip(curves, hedges.rows) for r in rows_k]
    spread = np.array([k in SPREAD_KINDS for k in kinds], dtype=bool)
    rows, positions = [], []
    for (key, label, unit), x, column in zip(hedges.units, multiples, matrix.T):
        details = unit.details
        swap = "fixed_rate" in details
        rows.append(
            {
                "curve": key,
                "benchmark": label,
                "notional": float(x * hedges.unit_notional),
                "direction": ("Receive fixed" if swap else "Receive spread leg") if x > 0
                else ("Pay fixed" if swap else "Pay spread leg") if x < 0 else "-",
                "par": float(details["fixed_rate"] if swap else details["spread"]),
                "pv01": float(x * column[~spread].sum()),
                "basis_pv01": float(x * column[spread].sum()),
            }
        )  # fmt: skip
        if x != 0:
            positions.append(_scaled(unit, abs(float(x)), x > 0))
    out_ladders, start = {}, 0
    position_pv01, hedged_pv01 = {}, {}
    for key, rows_k in zip(curves, hedges.rows):
        n = len(rows_k)
        t, h = target[start : start + n], hedge_ladder[start : start + n]
        out_ladders[key] = [
            {**{k: r[k] for k in ("label", "tenor", "years")}, "kind": kind, "position": a, "hedge": b,
             "residual": a + b}
            for r, a, b, kind in zip(rows_k, t, h, kinds[start : start + n])
        ]  # fmt: skip
        position_pv01[key], hedged_pv01[key] = float(t.sum()), float((t + h).sum())
        start += n
    return {
        "hedges": rows,
        "positions": positions,
        "ladders": out_ladders,
        "position_pv01": position_pv01,
        "hedged_pv01": hedged_pv01,
        "risk_removed": 1.0 - float(residual @ residual) / base if base > 0 else 1.0,
    }


def _quote_kind(market: Market, key: str, row: Mapping[str, object]) -> str:
    """Instrument kind of a ladder row (``DEPO``, ``IRS``, ``BASIS`` ...)."""

    return market.get_curve(key).instruments[int(row["index"])].kind


def _scaled(unit: Position, multiple: float, receive: bool) -> Position:
    """A hedge position: the unit benchmark with its notional scaled and its
    direction set."""

    trade = unit.details
    if "fixed_rate" in trade:
        swap = VanillaSwap(**{**trade, "notional": trade["notional"] * multiple, "receive_fixed": bool(receive)})
        return swap_position(swap, name=f"Hedge {swap.label}")
    basis = BasisSwap(**{**trade, "notional": trade["notional"] * multiple, "receive_spread_leg": bool(receive)})
    return basis_position(basis, name=f"Hedge {basis.label}")
