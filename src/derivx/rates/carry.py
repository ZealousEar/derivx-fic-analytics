"""Carry and roll-down per tenor of Treasury par bonds and par swaps over a
horizon of 1, 3 or 6 months.

Every number is for a *long* position: a bond bought and financed in repo,
or a swap receiving fixed. bp figures are P&L divided by the position's DV01
(the yield or swap-rate rise that would wipe the P&L out); the currency
figures are for a position holding ``dv01`` of DV01 (P&L per bp), i.e.
``bp x dv01``.

Bonds (:func:`bond_carry`, per 100 face; yields in the street convention of
:mod:`derivx.rates.bonds`)
    Bought for settlement ``s0`` at the clean price ``P0`` (dirty
    ``D0 = P0 + AI(s0)``, yield ``y0``) and financed in repo at ``repo``
    (simple, ACT/360) until the horizon settlement ``s1``:

    * ``income = AI(s1) - AI(s0) + coupons paid in (s0, s1]`` (coupon accrual)
    * ``financing = D0 x repo x days(s0, s1) / 360``
    * ``pull = clean price at s1 at the unchanged yield y0 - P0`` (pull to par)
    * ``carry = income + pull - financing``: the P&L if the bond's yield
      stays at ``y0``
    * ``roll = D(s1, y_roll) - D(s1, y0)``: the price effect of the bond's
      yield sliding from ``y0`` to ``y_roll``, its yield at ``s1`` on the
      static curve. :func:`treasury_carry` takes ``y_roll`` from today's
      Treasury quotes rebuilt on the horizon date (each par yield held at
      its tenor: the bond, now shorter, prices off the curve slid down).

    bp = P&L / DV01 at ``s1`` and ``y0`` (per 100 face, per bp).

Swaps (:func:`swap_carry`, receive fixed; par rates in decimals)
    ``S(0,T)`` is the spot par rate of tenor T, ``F(h,T)`` the par rate of
    the T swap starting h after spot, ``S(0,T-h)`` the spot par rate of the
    remaining maturity:

    * ``total = F(h,T) - S(0,T)``: forward-starting par minus spot par. A
      receiver of the forward-starting swap earns this (in bp of swap rate)
      on a static curve: at the horizon it is the spot T swap, worth
      ``(F - S(0,T)) x annuity``.
    * ``roll = S(0,T) - S(0,T-h)``: the slide down the static curve.
    * ``carry = total - roll``: fixed against floating accrual over the
      horizon, about ``(S(0,T) - S(0,h)) x h / duration``.

    Swaps use the discount curve's OIS conventions (SOFR, ESTR or SONIA OIS)
    unless a term ``index`` (``"EURIBOR 6M"``) is given.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from derivx.rates.bonds import FixedRateBond, benchmark_bond
from derivx.rates.conventions import Date, add_months, parse_tenor, year_fraction
from derivx.rates.curves import Market
from derivx.rates.swaps import VanillaSwap

HORIZONS: Dict[str, int] = {"1M": 1, "3M": 3, "6M": 6}  # label -> months
BOND_TENORS = ("2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y")
SWAP_TENORS = ("2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")
REPO_DAY_COUNT = "ACT/360"
DEFAULT_DV01 = 10_000.0  # currency per bp of the position the currency columns are for


def _months(horizon: str) -> int:
    label = str(horizon).strip().upper()
    if label not in HORIZONS:
        raise ValueError(f"horizon must be one of {list(HORIZONS)}, got {horizon!r}")
    return HORIZONS[label]


def horizon_date(asof: Date, horizon: str, calendar) -> Date:
    """``horizon`` months after ``asof``, on the next business day."""

    return calendar.adjust(add_months(asof, _months(horizon)), "F")


# --------------------------------------------------------------------------
# Bonds
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BondCarry:
    """Carry and roll-down of one long bond position (see the module
    docstring); prices per 100 face, yields and repo in decimals."""

    label: str
    settle: Date
    horizon_settle: Date
    days: int
    clean: float
    dirty: float
    ytm: float
    repo: float
    ytm_roll: float
    income: float
    financing: float
    pull: float
    carry: float
    roll: float
    dv01: float  # price change per bp at the horizon settlement and ytm, per 100 face (positive)

    @property
    def total(self) -> float:
        return self.carry + self.roll

    @property
    def carry_bp(self) -> float:
        return self.carry / self.dv01

    @property
    def roll_bp(self) -> float:
        return self.roll / self.dv01

    @property
    def total_bp(self) -> float:
        return self.total / self.dv01


def bond_carry(
    bond: FixedRateBond,
    settle: Date,
    horizon_settle: Date,
    clean_price: float,
    repo: float,
    roll_yield: Optional[float] = None,
) -> BondCarry:
    """Carry from ``settle`` to ``horizon_settle`` of a bond bought at
    ``clean_price`` (per 100 face) and financed at ``repo`` (decimal,
    ACT/360); roll-down to ``roll_yield`` (decimal street yield at the
    horizon; None = no roll)."""

    if horizon_settle <= settle:
        raise ValueError(f"the horizon settlement {horizon_settle} must be after the settlement {settle}")
    if horizon_settle >= bond.maturity:
        raise ValueError(f"the bond matures on {bond.maturity}, before the horizon settlement {horizon_settle}")
    for name, value in (("clean price", clean_price), ("repo rate", repo)):
        if not math.isfinite(value):
            raise ValueError(f"the {name} must be a finite number")
    if not clean_price > 0:
        raise ValueError("the clean price must be positive (per 100 face)")
    if not -0.05 <= repo <= 0.25:
        raise ValueError(f"repo {repo:.4%} looks wrong: give it as a decimal (0.0425 = 4.25%)")
    y0 = bond.yield_from_price(clean_price, settle)
    y_roll = y0 if roll_yield is None else float(roll_yield)
    if not math.isfinite(y_roll):
        raise ValueError("the roll-down yield must be a finite number")
    ai0, ai1 = bond.accrued(settle), bond.accrued(horizon_settle)
    dirty0 = clean_price + ai0
    coupons = bond.cash(settle, horizon_settle)
    days = (horizon_settle - settle).days
    financing = dirty0 * repo * year_fraction(settle, horizon_settle, REPO_DAY_COUNT)
    clean1 = bond.price_from_yield(y0, horizon_settle)
    dirty1 = clean1 + ai1
    roll = bond.price_from_yield(y_roll, horizon_settle, clean=False) - dirty1
    income = ai1 - ai0 + coupons
    pull = clean1 - clean_price
    return BondCarry(
        label=bond.label,
        settle=settle,
        horizon_settle=horizon_settle,
        days=days,
        clean=float(clean_price),
        dirty=float(dirty0),
        ytm=float(y0),
        repo=float(repo),
        ytm_roll=float(y_roll),
        income=float(income),
        financing=float(financing),
        pull=float(pull),
        carry=float(income + pull - financing),
        roll=float(roll),
        dv01=abs(bond.dv01(y0, horizon_settle)),
    )


def treasury_carry(
    market: Market,
    horizon: str,
    repo: float,
    tenors: Sequence[str] = BOND_TENORS,
) -> List[BondCarry]:
    """Carry and roll-down of par bonds at each Treasury ``tenor`` (coupon =
    the curve's par yield for settlement T+1, price 100), financed at
    ``repo``; roll-down on today's Treasury quotes rebuilt on the horizon
    date."""

    curve = market.govt
    if curve is None or not curve.quotes:
        raise ValueError("carry and roll-down of Treasuries need the Treasury curve built from its par quotes")
    cal = curve.conventions.cal
    settle = cal.add_business_days(cal.adjust(market.asof, "F"), curve.conventions.spot_lag)
    at = horizon_date(market.asof, horizon, cal)
    rolled = curve.rebuilt(asof=at)
    out = []
    for tenor in tenors:
        months = parse_tenor(tenor).months
        if months is None or months % 12:
            raise ValueError(f"bond tenors are whole years, got {tenor!r}")
        bond = benchmark_bond(curve, months // 12, settle)
        horizon_settle = bond.settlement_date(at)
        price = bond.price_from_curve(curve, settle)
        roll_price = bond.price_from_curve(rolled, horizon_settle)
        roll_yield = bond.yield_from_price(roll_price, horizon_settle)
        out.append(bond_carry(bond, settle, horizon_settle, price, repo, roll_yield))
    return out


# --------------------------------------------------------------------------
# Swaps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapCarry:
    """Carry and roll-down of receiving fixed in a par swap (see the module
    docstring); rates in decimals."""

    tenor: str
    horizon: str
    spot_par: float  # S(0,T)
    forward_par: float  # F(h,T)
    rolled_par: float  # S(0,T-h)
    short_par: float  # S(0,h): the floating rate over the horizon, as a par rate
    annuity: float  # PV of 1bp a year on 1 notional of the spot swap (its DV01 per unit notional)

    @property
    def total_bp(self) -> float:
        return (self.forward_par - self.spot_par) * 1e4

    @property
    def roll_bp(self) -> float:
        return (self.spot_par - self.rolled_par) * 1e4

    @property
    def carry_bp(self) -> float:
        return self.total_bp - self.roll_bp


def _swap(market: Market, tenor: str, forward_start: Optional[str], index: Optional[str]) -> VanillaSwap:
    if index is not None:
        return VanillaSwap.from_index(index, tenor, 0.0, 1.0, forward_start=forward_start, trade_date=market.asof)
    conv = market.curve.conventions
    if conv is None:
        raise ValueError("swap carry needs a curve bootstrapped with conventions")
    return VanillaSwap.from_conventions(conv, tenor, 0.0, 1.0, forward_start=forward_start, trade_date=market.asof)


def swap_carry(market: Market, tenor: str, horizon: str, index: Optional[str] = None) -> SwapCarry:
    """Carry and roll-down of receiving the par ``tenor`` swap over
    ``horizon`` on ``market`` (OIS on the discount curve, or against the
    term ``index``)."""

    months = parse_tenor(tenor).months
    h = _months(horizon)
    if months is None or months <= h:
        raise ValueError(f"the swap tenor {tenor!r} must be months or years longer than the {horizon} horizon")
    spot = _swap(market, tenor, None, index)
    forward = _swap(market, tenor, f"{h}M", index)
    rolled = _swap(market, f"{months - h}M", None, index)
    short = _swap(market, f"{h}M", None, index)
    return SwapCarry(
        tenor=str(parse_tenor(tenor)),
        horizon=str(horizon).upper(),
        spot_par=spot.par_rate(market),
        forward_par=forward.par_rate(market),
        rolled_par=rolled.par_rate(market),
        short_par=short.par_rate(market),
        annuity=spot.annuity(market) * 1e4,
    )


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def bond_table(rows: Sequence[BondCarry], tenors: Sequence[str], dv01: float = DEFAULT_DV01) -> pd.DataFrame:
    """One row per bond: per-100 amounts, bp and currency for ``dv01``."""

    out = []
    for tenor, r in zip(tenors, rows):
        record = {"tenor": tenor, **{k: v for k, v in asdict(r).items()}}
        record.update(
            carry_bp=r.carry_bp,
            roll_bp=r.roll_bp,
            total_bp=r.total_bp,
            carry_ccy=r.carry_bp * dv01,
            roll_ccy=r.roll_bp * dv01,
            total_ccy=r.total_bp * dv01,
        )
        out.append(record)
    return pd.DataFrame(out)


def swap_table(rows: Sequence[SwapCarry], dv01: float = DEFAULT_DV01) -> pd.DataFrame:
    """One row per swap: par rates (percent), bp and currency for ``dv01``."""

    out = []
    for r in rows:
        out.append(
            {
                "tenor": r.tenor,
                "horizon": r.horizon,
                "spot_par_pct": 100 * r.spot_par,
                "forward_par_pct": 100 * r.forward_par,
                "rolled_par_pct": 100 * r.rolled_par,
                "short_par_pct": 100 * r.short_par,
                "carry_bp": r.carry_bp,
                "roll_bp": r.roll_bp,
                "total_bp": r.total_bp,
                "carry_ccy": r.carry_bp * dv01,
                "roll_ccy": r.roll_bp * dv01,
                "total_ccy": r.total_bp * dv01,
                "dv01_per_mm": r.annuity * 1e6 * 1e-4,
            }
        )
    return pd.DataFrame(out)
