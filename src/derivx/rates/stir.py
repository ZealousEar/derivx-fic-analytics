"""Short-term rate trades: FRAs and rate futures on a multi-curve market.

* :class:`ForwardRateAgreement` - a FRA on a term index (EURIBOR): on the
  value date it settles ``N (F - K) tau / (1 + F tau)`` (discounted at the
  index rate for the period, as FRAs settle in advance), where ``F`` is the
  index fixing; before then its PV is that amount, with ``F`` the
  projection-curve forward, discounted on the OIS curve from the value date
  (QuantLib ``ForwardRateAgreement`` with the index's own period). A FRA
  whose value date has come is settled and worth nothing.
* :class:`RateFuture` - SOFR (``SR1``, ``SR3``) or three-month EURIBOR
  (``FUT``) futures: model price ``100 x (1 - (F + CA))`` with ``F`` the
  curve forward for the reference period (compounded, averaged or the term
  fixing, see :mod:`derivx.rates.indices`) and ``CA`` the Hull-White
  convexity adjustment. Futures are margined daily, so a position's value
  is its variation margin, undiscounted: contracts x point value x (model
  price - traded price). ``SR3`` is $25 per bp per contract, ``SR1`` $41.67,
  EURIBOR 3M EUR 25.

Signs: values are P&L to the holder in currency. A FRA that pays fixed (the
buyer) and a short future gain when rates rise.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from derivx.risk.sensitivities import Position

from .conventions import Date, Tenor
from .curves import Market
from .indices import (
    FUTURES_KINDS,
    FUTURES_POINT_VALUE,
    HullWhite,
    TermIndex,
    contract_label,
    futures_period,
    get_index,
    hw_overnight_futures_convexity,
    hw_term_futures_convexity,
    overnight_futures_rate,
    parse_contract,
    price_from_futures_rate,
)


@dataclass(frozen=True)
class ForwardRateAgreement:
    """FRA on ``index``: the period starts ``months_to_start`` months after
    spot (spot = fixing lag after ``trade_date``, or the valuation date if
    None) unless ``start`` is given, and runs one index tenor.
    ``pay_fixed=True`` is the buyer (pays ``fixed_rate``, receives the
    index, QuantLib ``Position.Long``)."""

    index: str
    fixed_rate: float
    months_to_start: Optional[int] = None
    start: Optional[Date] = None
    notional: float = 1e6
    pay_fixed: bool = True
    trade_date: Optional[Date] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", get_index(self.index).name)
        if not math.isfinite(self.notional) or self.notional <= 0:
            raise ValueError("notional must be positive (use pay_fixed for the direction)")
        if not math.isfinite(self.fixed_rate):
            raise ValueError("fixed_rate must be a finite decimal")
        if (self.start is None) == (self.months_to_start is None):
            raise ValueError("give either months_to_start (e.g. 6 for a 6X12) or a start date")
        if self.months_to_start is not None and self.months_to_start < 0:
            raise ValueError("months_to_start must be 0 or more")

    @classmethod
    def from_label(cls, label: str, index: str, fixed_rate: float, **fields: object) -> "ForwardRateAgreement":
        """FRA named ``AxB`` (months to start x months to end)."""

        first, _, last = str(label).upper().partition("X")
        try:
            a, b = int(first), int(last)
        except ValueError:
            raise ValueError(f"FRA {label!r}: write months to start x months to end, e.g. 6X12") from None
        ix = get_index(index)
        if b - a != ix.months:
            raise ValueError(f"a {ix.name} FRA spans {ix.months} months, not {b - a} ({label})")
        return cls(index, fixed_rate, months_to_start=a, **fields)

    @property
    def term_index(self) -> TermIndex:
        return get_index(self.index)

    def value_date(self, asof: Date) -> Date:
        if self.start is not None:
            return self.start
        ix = self.term_index
        spot = ix.spot(self.trade_date or asof)
        if not self.months_to_start:
            return spot
        return ix.cal.advance(spot, Tenor(self.months_to_start, "M"), ix.bdc, eom=ix.eom)

    def dates(self, asof: Date) -> Dict[str, Date]:
        value = self.value_date(asof)
        ix = self.term_index
        return {"fixing": ix.fixing_date(value), "value": value, "maturity": ix.maturity_date(value)}

    def forward(self, market: Market) -> float:
        """The index rate for the period: its fixing, or the projection
        forward."""

        d = self.dates(market.asof)
        return market.index_rate(self.index, d["value"], d["maturity"], self.term_index.day_count, d["fixing"])

    def settlement(self, market: Market) -> float:
        """Amount paid on the value date, signed to the holder."""

        d = self.dates(market.asof)
        tau = self.term_index.accrual(d["value"], d["maturity"])
        rate = self.forward(market)
        sign = 1.0 if self.pay_fixed else -1.0
        return self.notional * sign * (rate - self.fixed_rate) * tau / (1.0 + rate * tau)

    def pv(self, market: Market) -> float:
        value = self.value_date(market.asof)
        if value <= market.asof:
            return 0.0
        return self.settlement(market) * market.curve.df(value)

    def cash(self, market: Market, start: Date, end: Date) -> float:
        value = self.value_date(market.asof)
        return self.settlement(market) if start < value <= end else 0.0

    @property
    def label(self) -> str:
        side = "PAY" if self.pay_fixed else "REC"
        if self.months_to_start is not None:
            span = f"{self.months_to_start}X{self.months_to_start + self.term_index.months}"
        else:
            span = f"from {self.start.isoformat()}"
        return f"{side} {self.fixed_rate * 100:.4f}% {span} {self.index} FRA"


def fra_position(fra: ForwardRateAgreement, name: Optional[str] = None) -> Position:
    return Position(name=name or fra.label, pv=fra.pv, kind="fra", cash=fra.cash, details=asdict(fra))


@dataclass(frozen=True)
class RateFuture:
    """A futures position: ``contracts`` (negative = short) of a ``kind``
    (``SR1``, ``SR3``, ``FUT``) contract for ``contract`` (``DEC26``)
    bought or sold at ``trade_price``. ``index`` is the term index of a
    ``FUT`` (EURIBOR 3M); ``convexity`` the Hull-White parameters (None: no
    adjustment)."""

    kind: str
    contract: str
    trade_price: float
    contracts: float = 1.0
    index: Optional[str] = None
    convexity: Optional[HullWhite] = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().upper()
        object.__setattr__(self, "kind", kind)
        if kind not in FUTURES_KINDS:
            raise ValueError(f"kind must be one of {FUTURES_KINDS}, got {self.kind!r}")
        if kind == "FUT":
            index = get_index(self.index or "EURIBOR 3M")
            if not index.has_futures:
                raise ValueError(f"there are no futures on {index.name}")
            object.__setattr__(self, "index", index.name)
        elif self.index is not None:
            raise ValueError(f"{kind} is a SOFR future: it has no term index")
        if not math.isfinite(self.contracts) or self.contracts == 0:
            raise ValueError("contracts must be a non-zero number (negative = short)")
        if not math.isfinite(self.trade_price):
            raise ValueError("trade_price must be a finite price (100 - rate)")
        parse_contract(self.contract, Date(2000, 1, 1))  # validates the name; a one-digit year resolves when valued

    def period(self, asof: Date):
        year, month = parse_contract(self.contract, asof)
        index = get_index(self.index) if self.index else None
        return futures_period(self.kind, year, month, index)

    @property
    def point_value(self) -> float:
        """Currency per 1.00 of price per contract."""

        return FUTURES_POINT_VALUE[self.kind]

    def _overnight_fixings(self, market: Market, start: Date) -> Dict[Date, float]:
        """SOFR fixings for the days of the period before the valuation
        date: published ones (``index_fixings['SOFR']``, else the fixings the
        curves were bootstrapped with), else daily forwards of the fixings
        curve on or after its own date."""

        published = {
            **market.fixing_curve.overnight_fixings,
            **market.curve.overnight_fixings,
            **market.index_fixings.get("SOFR", {}),
        }
        if market.fixings is None or start >= market.asof:
            return published
        cal = market.curve.conventions.cal
        day = cal.adjust(start, "P")
        source = market.fixings
        while day < market.asof:
            nxt = cal.add_business_days(day, 1)
            if day not in published and day >= source.asof:
                published[day] = source.forward_rate(day, nxt, "ACT/360")
            day = nxt
        return published

    def forward(self, market: Market) -> float:
        """Curve rate for the reference period (before convexity). Once the
        rate is set (a ``FUT`` past its fixing, an ``SR1``/``SR3`` past its
        period) it is the final settlement rate, from fixings, so a position
        keeps its variation margin to the horizon of a carry scenario."""

        start, end = self.period(market.asof)
        if self.kind == "FUT":
            index = get_index(self.index)
            return market.index_rate(index.name, start, end, index.day_count, index.fixing_date(start))
        conv = market.curve.conventions
        if conv is None or conv.name != "USD SOFR":
            raise ValueError(f"{self.kind} is a SOFR future: price it on a USD SOFR market")
        fixings = self._overnight_fixings(market, start)
        return overnight_futures_rate(self.kind, start, end, market.asof, conv.cal, market.curve.df, fixings,
                                      conv.ois_day_count)  # fmt: skip

    def convexity_adjustment(self, market: Market, forward: Optional[float] = None) -> float:
        """Futures rate - forward rate (decimal), solved with the futures
        rate it depends on (three fixed-point steps; the dependence is weak)."""

        if self.convexity is None:
            return 0.0
        start, end = self.period(market.asof)
        fixed = get_index(self.index).fixing_date(start) < market.asof if self.kind == "FUT" else end <= market.asof
        if fixed:
            return 0.0  # the rate is set: no more convexity
        forward = self.forward(market) if forward is None else forward
        t1, t2 = (start - market.asof).days / 365.0, (end - market.asof).days / 365.0
        a, sigma = self.convexity.mean_reversion, self.convexity.sigma
        if self.kind == "FUT":
            ca = lambda r: hw_term_futures_convexity(r, max(t1, 0.0), t2, sigma, a)  # noqa: E731
        else:
            averaging = "compound" if self.kind == "SR3" else "average"
            ca = lambda r: hw_overnight_futures_convexity(r, t1, t2, sigma, a, averaging)  # noqa: E731
        rate = forward
        for _ in range(3):
            rate = forward + ca(rate)
        return rate - forward

    def futures_rate(self, market: Market) -> float:
        forward = self.forward(market)
        return forward + self.convexity_adjustment(market, forward)

    def price(self, market: Market) -> float:
        return price_from_futures_rate(self.futures_rate(market))

    def pv(self, market: Market) -> float:
        return self.contracts * self.point_value * (self.price(market) - self.trade_price)

    @property
    def label(self) -> str:
        side = "LONG" if self.contracts > 0 else "SHORT"
        name = {"SR1": "SR1", "SR3": "SR3", "FUT": "EURIBOR 3M"}[self.kind]
        try:
            contract = contract_label(*parse_contract(self.contract))
        except ValueError:
            contract = self.contract
        return f"{side} {abs(self.contracts):g} {name} {contract} @ {self.trade_price:.4f}"


def future_position(future: RateFuture, name: Optional[str] = None) -> Position:
    return Position(name=name or future.label, pv=future.pv, kind="future", details=asdict(future))
