"""Vanilla fixed/float interest rate swaps on a single curve.

A :class:`VanillaSwap` exchanges a fixed rate for a floating index:

* ``OIS``  - daily compounded overnight rate over each period, paid
  ``payment_lag`` business days after the period end (SOFR, ESTR, SONIA)
* ``TERM`` - simple term rate for each period, fixed ``fixing_lag``
  business days before the period start and paid at the period end

Both legs are projected and discounted on ``market.curve`` (single-curve
framework). Schedules are generated backward from the maturity with the
swap's calendar and business-day convention by the curve's own
:func:`~derivx.rates.curves.swap_schedule`, so a par swap at a quoted tenor
reprices that quote.

Seasoned swaps take past rates from ``market.fixing_curve``: a TERM period
whose fixing date is before ``market.asof``, and the overnight fixings of
an OIS period up to the first fixing date on or after ``market.asof`` (on a
weekend or holiday the last fixing still covers the days to the next
business day, as in QuantLib).

Every value is in currency for the swap's notional and signed to the holder:
a receiver swap gains when rates fall, so its PV01 is negative.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from derivx.rates.conventions import BUSINESS_DAY_CONVENTIONS, Calendar, Date, add_months, parse_tenor, year_fraction
from derivx.rates.curves import CONVENTIONS, CurveConventions, Market, get_conventions, swap_schedule
from derivx.risk.sensitivities import CURVE_FIELDS, Position, PvFunction, bumped_market, solve_hedge

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

    def __post_init__(self) -> None:
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

    def _flows(self, market: Market) -> pd.DataFrame:
        """Every cash flow (paid or not), signed to the holder."""

        sign = 1.0 if self.receive_fixed else -1.0
        rows = []
        for p in _periods(self, market.asof):
            if p.leg == "fixed":
                rate, amount = self.fixed_rate, sign * self.notional * self.fixed_rate * p.accrual
            else:
                if self.float_index == "OIS":
                    interest = _compounded_growth(market, p.start, p.end, self.cal) - 1.0
                    index_rate = interest / p.accrual
                else:
                    index_rate = market.projected_rate(p.start, p.end, self.float_day_count, p.fixing)
                    interest = index_rate * p.accrual
                rate = index_rate + self.spread
                amount = -sign * self.notional * (interest + self.spread * p.accrual)
            rows.append((p.leg, p.start, p.end, p.pay, p.accrual, rate, amount))
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
        return float(self.cashflows(market)["pv"].sum())

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
        return f"{side} {self.fixed_rate * 100:.4f}% {fwd}{length} {self.float_index}"


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
) -> float:
    """Par rate of a spot (or forward-starting) swap traded on
    ``market.asof``; conventions default to the curve's own."""

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

    if which not in CURVE_FIELDS:
        raise ValueError(f"which must be one of {CURVE_FIELDS}")
    curve = getattr(market, which)
    if curve is None or not curve.quotes:
        raise ValueError(f"market {which!r} curve has no quotes to bump")
    rows = tuple(
        {"index": i, "label": f"{q.tenor} {q.kind}", "tenor": q.tenor, "years": inst.tenor.years}
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
