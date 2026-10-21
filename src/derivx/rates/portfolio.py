"""Trade blotter and portfolio risk of a rates book.

A blotter is a table with one trade per row (CSV or ``st.data_editor``):

=========  ==================================================================
column     meaning (units)
=========  ==================================================================
id         unique trade id (text)
type       ``swap``, ``basis swap``, ``fra``, ``future``, ``bond``, ``cap``,
           ``floor`` or ``swaption``
currency   ``USD``, ``EUR`` or ``GBP``; a bond is a US Treasury (``USD``)
direction  swap: ``receive`` / ``pay`` the fixed rate; basis swap: ``receive``
           / ``pay`` the spread leg; FRA: ``pay`` (the buyer pays fixed) /
           ``receive``; future, bond, cap, floor, swaption: ``long`` / ``short``
notional   positive, in currency (bond: face amount; future: number of
           contracts); ``10mm``, ``2.5bn``, ``500k`` and ``1,000,000`` work
tenor      swap / basis swap / cap / floor / swaption underlying: ``5Y``;
           FRA: ``6X12``; future: the contract, ``DEC26``
start      forward start from spot (``1Y``) or a start date (YYYY-MM-DD, not
           before the valuation date); blank = spot
maturity   bond maturity date; a swap or basis swap end date instead of tenor
expiry     swaption option expiry (``1Y``)
index      swap: ``OIS`` (the currency's OIS conventions), ``IRS`` (term leg on
           the OIS curve, single-curve) or a term index (``EURIBOR 6M``);
           basis swap: ``EURIBOR 3M/EURIBOR 6M`` (spread leg / other leg); FRA:
           a term index; future: ``SR1``, ``SR3`` or ``EURIBOR 3M``; cap /
           floor: ``OIS 3M`` (the currency's overnight rate, SOFR / €STR /
           SONIA, compounded in arrears over 1M, 3M, 6M or 12M periods, as
           those caps trade), ``3M`` / ``6M`` (a single-curve term rate on
           the OIS curve: an approximation) or a term index; swaption:
           ``OIS`` or a term index
option     swaption: ``payer``, ``receiver`` or ``straddle``
rate       fixed rate (%) of a swap or FRA (blank = par on the valuation
           date); bond coupon (%)
spread     bp: swap float-leg spread; basis swap spread (blank = par); bond
           z-spread over the Treasury curve
strike     % for caps, floors and swaptions (blank = ATM)
price      future: traded price (100 - rate); bond: clean price per 100
           (sets the z-spread)
vol        flat vol: bp for the normal model, % for Black and shifted Black;
           blank = the market vols (swaptions: the SABR cube at the trade's
           strike; caps and floors: caplet vols stripped from cap quotes)
model      ``normal`` (default), ``black`` or ``shifted black``
shift      % displacement of ``shifted black``
=========  ==================================================================

:func:`parse_blotter` validates every row and reports what is wrong with
the row number and trade id; rows with errors are left out of the book.
Dates are fixed from the valuation date: a trade given by tenor starts at
spot from it (as the per-trade tabs), and a start before it is refused (the
book holds no past fixings for seasoned trades).

Each trade becomes the :class:`~derivx.risk.sensitivities.Position` of its
single-trade pricer (``swap_position``, ``basis_position``,
``fra_position``, ``future_position``, ``bond_position``,
``capfloor_position``, ``swaption_position``), so a book's numbers are the
sums of those pricers' numbers. :func:`price_book` reports, per trade, in
the trade's currency and signed to the holder:

* ``pv``    - present value (futures: variation margin to date)
* ``pv01``  - P&L for +1bp on every rate quote of every curve (OIS,
  projection and Treasury curves rebuilt; tenor basis spreads held),
  central difference; ``gamma`` the P&L per (1bp)^2 of the same move
* key-rate PV01s per curve (P&L for +1bp on one quote of one curve, the
  curve and those built on it rebuilt) with their diagonal gammas, as
  :class:`~derivx.risk.scenarios.QuoteRisk` (the scenario and VaR engines
  take them as they are)
* ``vega``  - P&L per +1bp of normal vol (Black-model trades: of the normal
  vol with the same price), and vega by bucket: cube-priced swaptions by
  the SABR cube's ATM points (:func:`~derivx.rates.swaption_vol.vega_buckets`),
  flat-vol swaptions allocated bilinearly to the nearest (expiry, tenor)
  grid points, caps and floors by caplet expiry, allocated linearly to the
  grid's expiries.

Amounts are never added across currencies: :meth:`BookRisk.summary` totals
each currency (``TOTAL USD``, ``TOTAL EUR``; ``TOTAL`` for a book in one
currency) and, given a :class:`~derivx.risk.reporting.Reporting`, adds the
whole book converted at stated FX spots (``ALL IN <R>``).

:func:`hedge_book` sizes par hedge instruments (OIS / term-index swaps,
tenor basis swaps, par Treasuries) to minimise ``move_bp * ||k + H x||_2 +
sum_j cost_j |x_j|`` for each currency's curves on their own (a EUR book
with US Treasuries is hedged in EUR on the EUR curves and in USD on the
Treasury curve; :attr:`HedgeResult.by_currency`): ``k`` is the stacked
key-rate ladder of that currency's curves (every quote, currency per bp),
``H`` the hedges' ladders,
``move_bp * ||r||_2`` the largest first-order loss of the residual book on
any curve move whose bp moves have a Euclidean norm of ``move_bp``, and
``cost_j`` half the bid/ask (bp) times the hedge's DV01 to its own quote.
Without costs it is the least-squares hedge, leaving out combinations of
hedges whose ladders almost cancel (they would be bought with huge
offsetting notionals for next to no risk; see :func:`least_squares_hedge`).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from derivx.rates.bonds import FixedRateBond, benchmark_bond, bond_position
from derivx.rates.capfloor import OVERNIGHT_NAMES, CapFloor, bachelier_vega, capfloor_position
from derivx.rates.capvol import strip_caplet_vols
from derivx.rates.conventions import Calendar, parse_tenor
from derivx.rates.curves import DISCOUNT_KEY, GOVT_KEY, SPREAD_KINDS, Market
from derivx.rates.indices import INDICES, HullWhite, get_index, parse_contract
from derivx.rates.stir import ForwardRateAgreement, RateFuture, fra_position, future_position
from derivx.rates.swaps import (
    BasisSwap,
    CurveBumps,
    VanillaSwap,
    _scaled,
    basis_position,
    benchmark_instruments,
    curve_bumps,
    swap_position,
)
from derivx.rates.swaption_vol import (
    DEFAULT_SHIFT,
    GRID_EXPIRIES,
    GRID_TENORS,
    SwaptionVolCube,
    VolGrid,
    calibrate_cube,
    cube_vol,
    grid_points,
    vega_buckets,
)
from derivx.rates.swaptions import KINDS as SWAPTION_KINDS
from derivx.rates.swaptions import Swaption, swaption_position
from derivx.risk.reporting import ALL_IN, Reporting, add_total_rows, ordered
from derivx.risk.scenarios import QuoteRisk
from derivx.risk.sensitivities import Position
from derivx.vol.implied import lognormal_to_normal_vol

TRADE_TYPES = ("swap", "basis swap", "fra", "future", "bond", "cap", "floor", "swaption")
COLUMNS = (
    "id", "type", "currency", "direction", "notional", "tenor", "start", "maturity", "expiry", "index", "option",
    "rate", "spread", "strike", "price", "vol", "model", "shift",
)  # fmt: skip
CURRENCIES = ("USD", "EUR", "GBP")
OIS_CONVENTIONS = {"USD": "USD SOFR", "EUR": "EUR ESTR", "GBP": "GBP SONIA"}
MODEL_NAMES = {"normal": "bachelier", "black": "black", "shifted black": "shifted_black"}
MODEL_LABELS = {v: k for k, v in MODEL_NAMES.items()}
CAP_MONTHS = (1, 3, 6, 12)
BP = 1.0  # risk bump, bp
NOISE = 0.005  # currency per bp: smaller hedge notionals and buckets are rebuild rounding
DEFAULT_UNIT_NOTIONAL = 1e6
TOTAL = "TOTAL"
RESERVED_IDS = (TOTAL, "BOOK")  # the book's total rows and columns; "TOTAL USD", "ALL IN USD" etc. too
FUTURE_PRICES = (70.0, 110.0)  # 100 - rate for rates of -10% to 30%, the range rates and strikes are checked on
MAX_VOL = {"normal": (2000.0, "bp"), "black": (300.0, "%"), "shifted black": (300.0, "%")}  # the CAP-FLOOR tab's

DIRECTIONS: Dict[str, Tuple[str, str]] = {  # (positive, negative) side of each type
    "swap": ("receive", "pay"),
    "basis swap": ("receive", "pay"),
    "fra": ("pay", "receive"),
    "future": ("long", "short"),
    "bond": ("long", "short"),
    "cap": ("long", "short"),
    "floor": ("long", "short"),
    "swaption": ("long", "short"),
}
_DIRECTION_ALIASES = {
    "receive": "receive", "rec": "receive", "receiver": "receive", "receive fixed": "receive", "r": "receive",
    "pay": "pay", "payer": "pay", "pay fixed": "pay", "p": "pay",
    "long": "long", "buy": "long", "bought": "long", "b": "long",
    "short": "short", "sell": "short", "sold": "short", "s": "short",
}  # fmt: skip
_TYPE_ALIASES = {
    "swap": "swap", "irs": "swap", "ois": "swap", "basis swap": "basis swap", "basis": "basis swap",
    "basisswap": "basis swap", "basis_swap": "basis swap", "fra": "fra", "future": "future", "futures": "future",
    "bond": "bond", "cap": "cap", "floor": "floor", "swaption": "swaption",
}  # fmt: skip
# Columns each type reads: required ones first, then optional ones.
REQUIRED: Dict[str, Tuple[str, ...]] = {
    "swap": ("index",),
    "basis swap": ("index",),
    "fra": ("tenor", "index"),
    "future": ("tenor", "index", "price"),
    "bond": ("maturity", "rate"),
    "cap": ("tenor", "index"),
    "floor": ("tenor", "index"),
    "swaption": ("expiry", "tenor", "option"),
}
OPTIONAL: Dict[str, Tuple[str, ...]] = {
    "swap": ("tenor", "maturity", "start", "rate", "spread"),
    "basis swap": ("tenor", "maturity", "start", "spread"),
    "fra": ("rate",),
    "future": (),
    "bond": ("spread", "price"),
    "cap": ("start", "strike", "vol", "model", "shift"),
    "floor": ("start", "strike", "vol", "model", "shift"),
    "swaption": ("index", "strike", "vol", "model", "shift"),
}
NUMERIC = ("notional", "rate", "spread", "strike", "price", "vol", "shift")
COLUMN_HELP = {
    "id": "Unique trade id",
    "type": "swap, basis swap, fra, future, bond, cap, floor or swaption",
    "currency": "USD, EUR or GBP (a bond is a USD Treasury)",
    "direction": "swap: receive/pay fixed; basis: receive/pay the spread leg; FRA: pay (buyer)/receive; others: "
    "long/short",
    "notional": "Positive, in currency (bond: face; future: contracts). 10mm, 2.5bn, 500k work",
    "tenor": "Swap/cap/swaption tenor (5Y); FRA 6X12; future contract DEC26",
    "start": "Forward start from spot (1Y) or a start date YYYY-MM-DD; blank = spot",
    "maturity": "Bond maturity date; or a swap end date instead of a tenor (YYYY-MM-DD)",
    "expiry": "Swaption option expiry (1Y)",
    "index": "Swap OIS / IRS / EURIBOR 6M; basis EURIBOR 3M/EURIBOR 6M; FRA index; future SR1/SR3/EURIBOR 3M; cap "
    "OIS 3M (SOFR/€STR/SONIA compounded in arrears), 3M/6M (term rate on the OIS curve, an approximation) or "
    "EURIBOR 3M/6M; swaption OIS or EURIBOR 6M",
    "option": "Swaption: payer, receiver or straddle",
    "rate": "Swap or FRA fixed rate in % (blank = par); bond coupon in %",
    "spread": "bp: swap float spread; basis spread (blank = par); bond z-spread",
    "strike": "Cap/floor/swaption strike in % (blank = ATM)",
    "price": "Future traded price (100 - rate); bond clean price per 100",
    "vol": "Flat vol: bp for normal, % for Black; blank = market vols (SABR cube / stripped caplet vols)",
    "model": "normal (default), black or shifted black",
    "shift": "Shifted Black displacement in %",
}


# --------------------------------------------------------------------------
# Trades and validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlotterIssue:
    """A problem in a blotter row: ``level`` is ``error`` (the row is left
    out), ``warning`` or ``info``; ``row`` is 1-based (0: the whole table)."""

    level: str
    row: int
    trade_id: str
    message: str

    def __str__(self) -> str:
        where = "blotter" if self.row == 0 else f"row {self.row}" + (f" ({self.trade_id})" if self.trade_id else "")
        return f"{where}: {self.message}"


@dataclass(frozen=True)
class Trade:
    """One validated blotter row. ``fields`` holds the type's other columns
    in blotter units (percent, bp, text, dates) so the row round-trips;
    ``sign`` is +1 for the first side in :data:`DIRECTIONS`."""

    id: str
    type: str
    currency: str
    direction: str
    notional: float
    fields: Tuple[Tuple[str, object], ...] = ()
    row: int = 0

    def get(self, name: str, default: object = None) -> object:
        return dict(self.fields).get(name, default)

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == DIRECTIONS[self.type][0] else -1.0

    def to_row(self) -> Dict[str, object]:
        """The trade as a blotter row (every column, blanks as None)."""

        row: Dict[str, object] = {c: None for c in COLUMNS}
        row.update(id=self.id, type=self.type, currency=self.currency, direction=self.direction, notional=self.notional)
        for name, value in self.fields:
            row[name] = value.isoformat() if isinstance(value, dt.date) else value
        return row

    @property
    def label(self) -> str:
        """Short description, e.g. ``RECEIVE 5Y OIS 3.60%``."""

        parts = [self.direction.upper()]
        for name in ("option", "expiry", "tenor", "maturity", "index"):
            value = self.get(name)
            if value is not None and value != "":
                parts.append(str(value.isoformat() if isinstance(value, dt.date) else value).upper())
        for name, unit in (("rate", "%"), ("strike", "% K"), ("price", " PX")):
            value = self.get(name)
            if value is not None:
                parts.append(f"{value:g}{unit}")
        return " ".join(parts)


def is_blank(value: object) -> bool:
    """True for an empty blotter cell (None, NaN, blank text)."""

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() in ("", "nan", "NaN", "None", "NaT", "<NA>")


_SUFFIX = {"K": 1e3, "M": 1e6, "MM": 1e6, "B": 1e9, "BN": 1e9}
_NUMBER_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(K|M|MM|B|BN)?$")
_GROUPED_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?(?:K|M|MM|B|BN)?$")


def parse_number(value: object, what: str, suffixes: bool = False) -> float:
    """A cell as a float: ``4.25``, ``"4.25%"``, ``"1,000,000"`` and, with
    ``suffixes``, ``10mm`` / ``2.5bn`` / ``500k``. A comma is only read as a
    thousands separator (groups of three digits): ``"1,5mm"`` or ``"3,5"``
    (a decimal comma) is refused rather than read as 15mm or 35.
    ``ValueError`` says which column could not be read."""

    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        number = float(value)
    else:
        text = str(value).strip().upper().replace("%", "").replace(" ", "")
        if "," in text:
            if not _GROUPED_RE.match(text):
                raise ValueError(
                    f"{what} {value!r}: a comma is read only as a thousands separator (1,000,000); write decimals "
                    "with a dot (1.5mm, 3.5)"
                )
            text = text.replace(",", "")
        match = _NUMBER_RE.match(text)
        if not match or (match.group(2) and not suffixes):
            raise ValueError(f"{what} {value!r} is not a number" + (" (e.g. 10mm, 2.5bn)" if suffixes else ""))
        number = float(match.group(1)) * (_SUFFIX[match.group(2)] if match.group(2) else 1.0)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return number


def _date(value: object, what: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    text = str(value).strip()[:10]
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{what} {value!r} is not a date: use YYYY-MM-DD") from None


def _tenor_label(value: object, what: str, units: Tuple[str, ...] = ("W", "M", "Y")) -> str:
    tenor = parse_tenor(str(value))
    if tenor.unit not in units:
        raise ValueError(f"{what} {value!r} must be in {', '.join(units)} (e.g. 5Y, 18M)")
    return str(tenor)


def _index_name(value: object) -> str:
    return " ".join(str(value).strip().upper().split())


def _parse_index(kind: str, value: object) -> str:
    """Normalised ``index`` cell of a trade type; ``ValueError`` if it is
    not one the type takes."""

    text = _index_name(value)
    if kind == "swap":
        if text in ("OIS", "IRS") or text in INDICES:
            return text
        raise ValueError(f"index {value!r}: use OIS, IRS or a term index ({', '.join(INDICES)})")
    if kind == "basis swap":
        legs = [p.strip() for p in re.split(r"/|\bVS\b", text) if p.strip()]
        if len(legs) != 2 or any(leg not in INDICES for leg in legs):
            raise ValueError(f"index {value!r}: give the spread leg and the other leg, e.g. EURIBOR 3M/EURIBOR 6M")
        return f"{legs[0]}/{legs[1]}"
    if kind == "fra":
        if text in INDICES:
            return text
        raise ValueError(f"index {value!r}: a FRA is on a term index ({', '.join(INDICES)})")
    if kind == "future":
        if text in ("SR1", "SR3", "FUT"):
            return text
        if text in INDICES and get_index(text).has_futures:
            return "FUT"
        raise ValueError(f"index {value!r}: use SR1, SR3 or EURIBOR 3M")
    if kind in ("cap", "floor"):
        if text in INDICES:
            return text
        match = re.match(r"^(\d{1,2})M$", text)
        if match and int(match.group(1)) in CAP_MONTHS:
            return text
        match = re.match(r"^(?:OIS\s*(\d{1,2})M|(\d{1,2})M\s*OIS)$", text)
        if match and int(match.group(1) or match.group(2)) in CAP_MONTHS:
            return f"OIS {int(match.group(1) or match.group(2))}M"
        raise ValueError(
            f"index {value!r}: use OIS 1M, OIS 3M, OIS 6M or OIS 12M (the overnight rate compounded in arrears), "
            f"1M, 3M, 6M or 12M (a single-curve term rate) or a term index ({', '.join(INDICES)})"
        )
    if kind == "swaption":
        if text in ("OIS", "") or text in INDICES:
            return text or "OIS"
        raise ValueError(f"index {value!r}: use OIS or a term index ({', '.join(INDICES)})")
    raise ValueError(f"a {kind} has no index")


def _parse_field(kind: str, name: str, value: object) -> object:
    """One type-specific cell in blotter units (raises ValueError)."""

    if name == "index":
        return _parse_index(kind, value)
    if name in ("rate", "spread", "strike", "price", "vol", "shift"):
        number = parse_number(value, name)
        if name == "vol" and number < 0:
            raise ValueError(f"vol must be zero or positive, got {number:g}")
        if name == "shift" and number < 0:
            raise ValueError(f"shift must be zero or positive (%), got {number:g}")
        if name == "rate" and kind == "bond" and not 0.0 <= number < 100.0:
            raise ValueError(f"bond coupon must be in [0, 100) %, got {number:g}")
        if name in ("rate", "strike") and kind != "bond" and not -10.0 <= number <= 30.0:
            raise ValueError(f"{name} {number:g}% is outside -10% to 30%: rates are in percent (3.5 = 3.5%)")
        if name == "price" and kind == "future" and not FUTURE_PRICES[0] <= number <= FUTURE_PRICES[1]:
            raise ValueError(
                f"futures price {number:g} is outside {FUTURE_PRICES[0]:g} to {FUTURE_PRICES[1]:g} (a rate of -10% "
                "to 30%): the price is 100 minus the rate in percent (96.20 for 3.80%)"
            )
        if name == "price" and kind == "bond" and number <= 0:
            raise ValueError(f"bond clean price must be positive (per 100 face), got {number:g}")
        return number
    if name == "maturity":
        return _date(value, "maturity")
    if name == "start":
        text = str(value).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}", text) or isinstance(value, (dt.date, pd.Timestamp)):
            return _date(value, "start")
        return _tenor_label(text, "start")
    if name == "expiry":
        return _tenor_label(value, "expiry")
    if name == "tenor":
        if kind == "fra":
            text = str(value).strip().upper().replace(" ", "")
            match = re.match(r"^(\d{1,3})X(\d{1,3})$", text)
            if not match or int(match.group(2)) <= int(match.group(1)):
                raise ValueError(f"FRA tenor {value!r}: write months to start x months to end, e.g. 6X12")
            return f"{int(match.group(1))}X{int(match.group(2))}"
        if kind == "future":
            text = str(value).strip().upper()
            parse_contract(text, dt.date(2000, 1, 1))
            return text
        units = ("M", "Y") if kind in ("cap", "floor", "swaption") else ("W", "M", "Y")
        return _tenor_label(value, "tenor", units)
    if name == "option":
        text = str(value).strip().lower()
        if text not in SWAPTION_KINDS:
            raise ValueError(f"option {value!r}: use payer, receiver or straddle")
        return text
    if name == "model":
        text = " ".join(str(value).strip().lower().replace("_", " ").split())
        text = {"bachelier": "normal", "lognormal": "black", "shifted": "shifted black"}.get(text, text)
        if text not in MODEL_NAMES:
            raise ValueError(f"model {value!r}: use normal, black or shifted black")
        return text
    raise ValueError(f"unknown column {name}")


def parse_trade(row: Mapping[str, object], number: int = 0) -> Tuple[Optional[Trade], List[BlotterIssue]]:
    """Validate one blotter row: (trade or None if it has errors, issues)."""

    issues: List[BlotterIssue] = []
    trade_id = "" if is_blank(row.get("id")) else str(row.get("id")).strip()

    def issue(level: str, message: str) -> None:
        issues.append(BlotterIssue(level, number, trade_id, message))

    upper = trade_id.upper()
    if trade_id and (upper in RESERVED_IDS or upper.startswith(TOTAL + " ") or upper.startswith(ALL_IN + " ")):
        issue("error", f"id {trade_id!r} is reserved for the book's totals: give the trade another id")

    raw_type = row.get("type")
    kind = _TYPE_ALIASES.get(" ".join(str(raw_type).strip().lower().split())) if not is_blank(raw_type) else None
    if kind is None:
        if is_blank(raw_type):
            issue("error", f"type is missing: give one of {', '.join(TRADE_TYPES)}")
        else:
            issue("error", f"type {raw_type!r} is not one of {', '.join(TRADE_TYPES)}")
        return None, issues
    currency = "" if is_blank(row.get("currency")) else str(row.get("currency")).strip().upper()
    if currency not in CURRENCIES:
        issue("error", f"currency {row.get('currency')!r}: give USD, EUR or GBP")
    elif kind == "bond" and currency != "USD":
        issue("error", f"a bond is a US Treasury priced on the Treasury curve: its currency is USD, not {currency}")
    raw_dir = row.get("direction")
    direction = _DIRECTION_ALIASES.get(str(raw_dir).strip().lower()) if not is_blank(raw_dir) else None
    if direction not in DIRECTIONS[kind]:
        issue("error", f"direction {raw_dir!r}: a {kind} is {' or '.join(DIRECTIONS[kind])}")
    notional = float("nan")
    if is_blank(row.get("notional")):
        issue("error", "notional is missing" + (" (number of contracts)" if kind == "future" else ""))
    else:
        try:
            notional = parse_number(row.get("notional"), "notional", suffixes=True)
            if notional <= 0:
                issue("error", f"notional must be positive (the direction gives the side), got {notional:g}")
            elif kind == "future" and not float(notional).is_integer():
                issue("error", f"a future's notional is a whole number of contracts, got {notional:g}")
        except ValueError as exc:
            issue("error", str(exc))
    fields: List[Tuple[str, object]] = []
    used = REQUIRED[kind] + OPTIONAL[kind]
    for name in used:
        value = row.get(name)
        if is_blank(value):
            if name in REQUIRED[kind]:
                issue("error", f"{name} is missing ({COLUMN_HELP[name]})")
            continue
        try:
            fields.append((name, _parse_field(kind, name, value)))
        except ValueError as exc:
            issue("error", str(exc))
    unused = [c for c in COLUMNS if c not in used and c not in ("id", "type", "currency", "direction", "notional")]
    ignored = [c for c in unused if c in row and not is_blank(row.get(c))]
    if ignored:
        verb = "is" if len(ignored) == 1 else "are"
        issue("warning", f"{', '.join(ignored)} {verb} not used by a {kind} and ignored")
    values = dict(fields)
    if kind in ("swap", "basis swap"):
        has_tenor, has_maturity = "tenor" in values, "maturity" in values
        if has_tenor == has_maturity:
            issue("error", f"give a {kind} a tenor (5Y) or a maturity date, not {'both' if has_tenor else 'neither'}")
    if kind == "bond" and "price" in values and "spread" in values:
        issue("error", "give a bond a clean price or a z-spread, not both")
    if kind in ("cap", "floor", "swaption"):
        model = values.get("model", "normal")
        if model == "shifted black" and "shift" not in values:
            issue("error", "shifted black needs a shift (%)")
        if model != "shifted black" and values.get("shift"):
            issue("warning", f"shift is only used by shifted black, not {model}; ignored")
            fields = [(n, v) for n, v in fields if n != "shift"]
        if "vol" in values:
            vol, (top, unit) = float(values["vol"]), MAX_VOL[model]
            units = "normal vols are in bp (95 = 95bp), Black vols in % (25 = 25%)"
            if vol > top:
                issue("error", f"vol {vol:g}{unit} is above {top:g}{unit} for the {model} model: {units}")
            elif 0.0 < vol < 1.0:
                issue("warning", f"vol {vol:g}{unit} for the {model} model looks like a decimal: {units}")
    if kind == "basis swap" and currency in CURRENCIES and "index" in values:
        legs = str(values["index"]).split("/")
        a, b = get_index(legs[0]), get_index(legs[1])
        if a.currency != b.currency or a.months == b.months:
            issue("error", f"a tenor basis swap needs two tenors of one currency, got {a.name} and {b.name}")
    index = values.get("index")
    if index in INDICES and currency in CURRENCIES and get_index(index).currency != currency:
        issue("error", f"{index} is a {get_index(index).currency} index, not {currency}")
    if not trade_id:
        trade_id = f"T{number}"
        issues.append(BlotterIssue("info", number, trade_id, f"no id: named {trade_id}"))
    if any(i.level == "error" for i in issues):
        return None, issues
    return Trade(trade_id, kind, currency, direction, float(notional), tuple(fields), number), issues


def parse_blotter(table: pd.DataFrame) -> Tuple[List[Trade], List[BlotterIssue]]:
    """Validate a blotter table (column names are matched without case or
    surrounding spaces). Returns the valid trades and every issue; a row
    with an error is left out, blank rows are skipped, a repeated id is an
    error."""

    issues: List[BlotterIssue] = []
    frame = table.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    unknown = [c for c in frame.columns if c not in COLUMNS]
    if unknown:
        issues.append(BlotterIssue("warning", 0, "", f"columns {', '.join(unknown)} are not blotter columns; ignored"))
    missing = [c for c in ("type", "currency", "direction", "notional") if c not in frame.columns]
    if missing:
        issues.append(BlotterIssue("error", 0, "", f"missing columns: {', '.join(missing)} (see the column guide)"))
        return [], issues
    trades: List[Trade] = []
    seen: Dict[str, int] = {}
    for number, row in enumerate(frame.to_dict("records"), start=1):
        if all(is_blank(v) for v in row.values()):
            continue
        trade, found = parse_trade(row, number)
        issues += found
        if trade is None:
            continue
        if trade.id in seen:
            issues.append(BlotterIssue("error", number, trade.id, f"id {trade.id} repeats row {seen[trade.id]}"))
            continue
        seen[trade.id] = number
        trades.append(trade)
    return trades, issues


CSV_DELIMITERS = ",;\t|"


def _canonical(column: str, value: object) -> object:
    """A recognised spelling of a categorical cell as the editor's option
    (``IRS`` -> ``swap``, ``rec`` -> ``receive``, ``Payer`` -> ``payer``,
    ``usd`` -> ``USD``); anything else is kept as typed for
    :func:`parse_blotter` to report."""

    if is_blank(value):
        return None
    text = " ".join(str(value).strip().split())
    if column == "type":
        return _TYPE_ALIASES.get(text.lower(), text)
    if column == "direction":
        return _DIRECTION_ALIASES.get(text.lower(), text)
    if column == "option":
        return text.lower() if text.lower() in SWAPTION_KINDS else text
    if column == "model":
        lower = text.lower().replace("_", " ")
        lower = {"bachelier": "normal", "lognormal": "black", "shifted": "shifted black"}.get(lower, lower)
        return lower if lower in MODEL_NAMES else text
    if column == "currency":
        return text.upper() if text.upper() in CURRENCIES else text
    return value


def read_blotter_csv(payload: bytes) -> Tuple[pd.DataFrame, List[BlotterIssue]]:
    """Uploaded blotter CSV -> (table of text cells in blotter columns,
    issues). The delimiter (comma, semicolon, tab or bar) is read from the
    header line. Cells stay text, so ``10mm`` or ``DEC26`` are read as typed
    (recognised spellings of type, direction, option, model and currency
    become the editor's options); :func:`parse_blotter` validates them. A
    file that cannot be read, or lacks the type, currency, direction or
    notional column, is one ``error`` issue and an empty table."""

    def failed(message: str) -> Tuple[pd.DataFrame, List[BlotterIssue]]:
        return blotter_frame([]), [BlotterIssue("error", 0, "", message)]

    try:
        text = bytes(payload).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return failed(f"could not read the file: it is not UTF-8 text ({exc})")
    if not text.strip():
        return failed("the file is empty: it needs a header row with the blotter columns (see the column guide)")
    header = text.lstrip().splitlines()[0]
    try:
        sep = csv.Sniffer().sniff(header, delimiters=CSV_DELIMITERS).delimiter
    except csv.Error:
        sep = ","  # one column, or no delimiter in the header
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_values=[""], sep=sep)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, csv.Error) as exc:
        return failed(f"could not read the file ({exc})")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    twice = sorted({c for c in raw.columns if list(raw.columns).count(c) > 1})
    if twice:
        return failed(f"columns {', '.join(twice)} appear more than once (column names ignore case)")
    known = [c for c in raw.columns if c in COLUMNS]
    if not known:
        return failed(
            f"no blotter columns in the header ({', '.join(raw.columns) or 'none'}): it needs type, currency, "
            "direction, notional and the columns each trade type reads (see the column guide)"
        )
    missing = [c for c in ("type", "currency", "direction", "notional") if c not in raw.columns]
    if missing:
        return failed(f"missing columns: {', '.join(missing)} (see the column guide)")
    issues = []
    unknown = [c for c in raw.columns if c not in COLUMNS]
    if unknown:
        issues.append(BlotterIssue("warning", 0, "", f"columns {', '.join(unknown)} are not blotter columns; ignored"))
    out = pd.DataFrame({c: raw[c] if c in raw.columns else None for c in COLUMNS}, index=raw.index)
    out = out.astype(object).where(out.notna(), None)
    for column in ("type", "direction", "option", "model", "currency"):
        out[column] = [_canonical(column, v) for v in out[column]]
    return out, issues


def blotter_frame(trades: Sequence[Trade]) -> pd.DataFrame:
    """Trades as a blotter table in :data:`COLUMNS` order (round-trips
    through :func:`parse_blotter`)."""

    return pd.DataFrame([t.to_row() for t in trades], columns=list(COLUMNS))


def blotter_csv(trades: Sequence[Trade]) -> bytes:
    return blotter_frame(trades).to_csv(index=False).encode("utf-8")


def column_guide() -> pd.DataFrame:
    """Which columns each trade type reads (``required`` / ``optional``)."""

    rows = []
    for column in COLUMNS:
        row = {"column": column, "meaning": COLUMN_HELP[column]}
        for kind in TRADE_TYPES:
            if column in ("id", "type", "currency", "direction", "notional"):
                row[kind] = "required" if column != "id" else "optional"
            elif column in REQUIRED[kind]:
                row[kind] = "required"
            elif column in REQUIRED[kind] + OPTIONAL[kind]:
                row[kind] = "optional"
            else:
                row[kind] = ""
        rows.append(row)
    return pd.DataFrame(rows)


EXAMPLE_ASOF = dt.date(2026, 10, 19)
_EXAMPLES: Dict[str, Tuple[Dict[str, object], ...]] = {
    "USD": (
        dict(id="US1", type="swap", direction="receive", notional="50mm", tenor="5Y", index="OIS", rate=3.60),
        dict(id="US2", type="swap", direction="pay", notional="25mm", tenor="10Y", index="OIS", rate=3.70),
        dict(id="US3", type="swap", direction="receive", notional="10mm", tenor="30Y", index="OIS", rate=3.85),
        dict(id="US4", type="swap", direction="pay", notional="20mm", tenor="2Y", start="1Y", index="OIS"),
        dict(id="US5", type="future", direction="short", notional="100", tenor="DEC26", index="SR3", price=96.20),
        dict(id="US6", type="bond", direction="long", notional="10mm", maturity="2035-08-15", rate=4.25),
        dict(id="US7", type="cap", direction="long", notional="25mm", tenor="5Y", index="OIS 3M", strike=4.00),
        dict(id="US8", type="floor", direction="short", notional="25mm", tenor="5Y", index="OIS 3M", strike=3.00,
             vol=95, model="normal"),
        dict(id="US9", type="swaption", direction="long", notional="20mm", expiry="1Y", tenor="10Y", option="payer",
             index="OIS"),
        dict(id="US10", type="swaption", direction="short", notional="20mm", expiry="2Y", tenor="5Y",
             option="receiver", index="OIS", strike=3.25),
    ),
    "EUR": (
        dict(id="EU1", type="swap", direction="receive", notional="50mm", tenor="10Y", index="EURIBOR 6M",
             rate=3.65),
        dict(id="EU2", type="swap", direction="pay", notional="40mm", tenor="5Y", index="OIS", rate=3.20),
        dict(id="EU3", type="basis swap", direction="receive", notional="100mm", tenor="5Y",
             index="EURIBOR 3M/EURIBOR 6M"),
        dict(id="EU4", type="fra", direction="pay", notional="200mm", tenor="6X12", index="EURIBOR 6M"),
        dict(id="EU5", type="future", direction="long", notional="50", tenor="MAR27", index="EURIBOR 3M",
             price=96.90),
        dict(id="EU6", type="cap", direction="long", notional="25mm", tenor="5Y", index="EURIBOR 6M", strike=3.50),
        dict(id="EU7", type="swaption", direction="long", notional="20mm", expiry="1Y", tenor="10Y",
             option="payer", index="EURIBOR 6M"),
    ),
    "GBP": (
        dict(id="GB1", type="swap", direction="receive", notional="50mm", tenor="5Y", index="OIS", rate=4.85),
        dict(id="GB2", type="swap", direction="pay", notional="25mm", tenor="2Y", index="OIS"),
        dict(id="GB3", type="cap", direction="long", notional="20mm", tenor="3Y", index="OIS 3M", strike=5.00,
             vol=90),
        dict(id="GB4", type="swaption", direction="long", notional="20mm", expiry="1Y", tenor="5Y",
             option="straddle", index="OIS", vol=85),
    ),
}  # fmt: skip


def example_blotter(currency: str = "USD") -> pd.DataFrame:
    """An EXAMPLE book for a currency: illustrative trades (not positions
    anyone holds), with rates set near the illustrative curves of
    :data:`EXAMPLE_ASOF`."""

    code = str(currency).strip().upper()
    if code not in _EXAMPLES:
        raise ValueError(f"no example book for {currency!r}; have {sorted(_EXAMPLES)}")
    rows = [{c: None for c in COLUMNS} | {"currency": code} | dict(r) for r in _EXAMPLES[code]]
    return pd.DataFrame(rows, columns=list(COLUMNS)).astype(object)


# --------------------------------------------------------------------------
# Market inputs beyond the curves
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VolSet:
    """Vols for trades without a flat vol: ``cubes`` keyed by swaption
    underlying (``"OIS"`` or a term index), ``caplets`` (stripped caplet
    vols, :mod:`derivx.rates.capvol`) keyed by cap index (``"3M"`` or a term
    index). ``sources`` says where each came from (``"ILLUSTRATIVE ..."``
    for the built-in quotes). ``hull_white`` sets the futures convexity
    adjustment (None: the curve's own, or none)."""

    cubes: Mapping[str, SwaptionVolCube] = field(default_factory=dict)
    caplets: Mapping[str, object] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)
    hull_white: Optional[HullWhite] = None


def cube_key(trade: Trade) -> str:
    return str(trade.get("index") or "OIS")


def cap_key(trade: Trade) -> str:
    return str(trade.get("index"))


def _check_currency(trade: Trade, market: Market) -> None:
    if trade.type == "bond":
        return
    conv = market.curve.conventions
    have = conv.currency if conv is not None else "USD"
    if trade.currency != have:
        raise ValueError(
            f"a {trade.currency} {trade.type}: the market is {conv.name if conv else have}; switch the swap curve to "
            f"{OIS_CONVENTIONS[trade.currency]} to price it"
        )


def _start(trade: Trade, market: Market) -> Dict[str, object]:
    start = trade.get("start")
    if isinstance(start, dt.date):
        if start < market.asof:
            raise ValueError(
                f"starts {start}, before the valuation date {market.asof}: the book holds no past fixings for "
                "seasoned trades; enter the remaining trade from the valuation date (spot or forward start)"
            )
        return {"start": start}
    return {"forward_start": start} if start else {}


def _swap(trade: Trade, market: Market) -> VanillaSwap:
    index = str(trade.get("index"))
    rate = trade.get("rate")
    terms = dict(notional=trade.notional, receive_fixed=trade.sign > 0, spread=float(trade.get("spread") or 0.0) * 1e-4,
                 maturity=trade.get("maturity"), trade_date=market.asof, **_start(trade, market))  # fmt: skip
    fixed = 0.0 if rate is None else float(rate) / 100.0
    if index in INDICES:
        market.projection(index)  # a clear error if the market has no such curve
        swap = VanillaSwap.from_index(index, trade.get("tenor"), fixed, **terms)
    else:
        swap = VanillaSwap.from_conventions(market.curve.conventions, trade.get("tenor"), fixed, kind=index, **terms)
    return swap if rate is not None else replace(swap, fixed_rate=swap.par_rate(market))


def _basis(trade: Trade, market: Market) -> BasisSwap:
    spread_index, other_index = str(trade.get("index")).split("/")
    for name in (spread_index, other_index):
        market.projection(name)
    spread = trade.get("spread")
    swap = BasisSwap(
        trade.notional, 0.0 if spread is None else float(spread) * 1e-4, trade.get("tenor"), trade.get("maturity"),
        spread_index=spread_index, other_index=other_index, receive_spread_leg=trade.sign > 0, trade_date=market.asof,
        **_start(trade, market),
    )  # fmt: skip
    return swap if spread is not None else replace(swap, spread=swap.par_spread(market))


def _fra(trade: Trade, market: Market) -> ForwardRateAgreement:
    index = str(trade.get("index"))
    market.projection(index)
    rate = trade.get("rate")
    fra = ForwardRateAgreement.from_label(
        str(trade.get("tenor")),
        index,
        0.0 if rate is None else float(rate) / 100,
        notional=trade.notional,
        pay_fixed=trade.sign > 0,
        trade_date=market.asof,
    )
    return fra if rate is not None else replace(fra, fixed_rate=fra.forward(market))  # fmt: skip


def _future(trade: Trade, market: Market, vols: VolSet) -> RateFuture:
    kind = str(trade.get("index"))
    index = "EURIBOR 3M" if kind == "FUT" else None
    curve = market.projection(index) if index else market.curve
    convexity = vols.hull_white if vols.hull_white is not None else curve.convexity
    return RateFuture(kind, str(trade.get("tenor")), float(trade.get("price")), trade.sign * trade.notional, index,
                      convexity)  # fmt: skip


def _bond(trade: Trade, market: Market) -> Tuple[FixedRateBond, float]:
    if market.govt is None:
        raise ValueError("the market has no Treasury (govt) curve to price bonds on")
    bond = FixedRateBond(float(trade.get("rate")) / 100.0, trade.get("maturity"))
    if bond.maturity <= market.asof:
        raise ValueError(f"the bond matured on {bond.maturity}, before the valuation date {market.asof}")
    price = trade.get("price")
    if price is not None:
        settle = bond.settlement_date(market.asof)
        return bond, bond.z_spread(float(price), market.govt, settle)
    return bond, float(trade.get("spread") or 0.0) * 1e-4


def _vol_units(model: str) -> float:
    return 1e-4 if model == "bachelier" else 0.01


def cap_terms(key: str, market: Market) -> Dict[str, object]:
    """Schedule fields of a cap on ``key``, as on the CAP-FLOOR tab: a term
    index (its calendar, day count and fixing lag, projected on its curve),
    ``"OIS 3M"``-style periods of the swap curve's overnight rate
    compounded in arrears (its OIS day count, spot lag and payment lag), or
    ``"3M"``-style months of a single-curve term rate on the swap curve's
    IRS float conventions."""

    if key in INDICES:
        market.projection(key)
        ix = get_index(key)
        return dict(index_months=ix.months, day_count=ix.day_count, calendar=ix.calendar, fixing_lag=ix.fixing_lag,
                    projection=ix.name)  # fmt: skip
    conv = market.curve.conventions
    if key.startswith("OIS "):
        return dict(index_months=int(key[4:-1]), day_count=conv.ois_day_count, calendar=conv.calendar,
                    bdc=conv.bdc, fixing_lag=conv.spot_lag, payment_lag=conv.payment_lag,
                    rate_type="compounded")  # fmt: skip
    return dict(index_months=int(key[:-1]), day_count=conv.irs_float_day_count, calendar=conv.calendar,
                fixing_lag=conv.spot_lag)  # fmt: skip


def _cap_terms(trade: Trade, market: Market) -> Dict[str, object]:
    return cap_terms(str(trade.get("index")), market)


def needed_vols(trades: Sequence[Trade]) -> Tuple[List[str], List[str]]:
    """(swaption cube keys, cap keys) the trades without a flat vol need."""

    cubes = sorted({cube_key(t) for t in trades if t.type == "swaption" and t.get("vol") is None})
    caps = sorted({cap_key(t) for t in trades if t.type in ("cap", "floor") and t.get("vol") is None})
    return cubes, caps


def build_vols(
    market: Market,
    trades: Sequence[Trade],
    swaption_grid: Optional[VolGrid] = None,
    cap_quotes: Optional[pd.DataFrame] = None,
    cap_source: str = "",
    hull_white: Optional[HullWhite] = None,
    fit: str = "normal",
    beta: Optional[float] = None,
    shift: float = DEFAULT_SHIFT,
    method: str = "constant",
) -> Tuple[VolSet, List[str]]:
    """Vols for the trades without a flat vol: a SABR cube per swaption
    underlying calibrated to ``swaption_grid`` (:func:`calibrate_cube`), and
    caplet vols per cap index stripped from ``cap_quotes`` (columns
    ``maturity``, ``vol_bp``: flat ATM normal cap vols, bp). Returns the
    vols and notes on anything that could not be built (those trades are
    then skipped with the reason)."""

    cube_keys, cap_keys = needed_vols(trades)
    cubes: Dict[str, SwaptionVolCube] = {}
    caplets: Dict[str, object] = {}
    sources: Dict[str, str] = {}
    notes: List[str] = []
    if cube_keys and swaption_grid is not None:
        for key in cube_keys:
            index = None if key == "OIS" else key
            try:
                cubes[key] = calibrate_cube(swaption_grid, grid_points(swaption_grid, market, index), fit, beta,
                                            shift, index=index)  # fmt: skip
                sources[f"cube:{key}"] = swaption_grid.source
            except ValueError as exc:
                notes.append(f"no {key} swaption cube: {exc}")
    if cap_keys and cap_quotes is not None:
        maturities = [str(m).strip().upper() for m in cap_quotes["maturity"]]
        flat = pd.to_numeric(cap_quotes["vol_bp"], errors="coerce").to_numpy(dtype=float) * 1e-4
        for key in cap_keys:
            try:
                terms = cap_terms(key, market)
                caplets[key] = strip_caplet_vols(market, maturities, flat, None, method, terms=terms).curve
                sources[f"caplets:{key}"] = cap_source
            except ValueError as exc:
                notes.append(f"no {key} caplet vols: {exc}")
    return VolSet(cubes, caplets, sources, hull_white), notes


def _cap(trade: Trade, market: Market, vols: VolSet) -> Tuple[CapFloor, str]:
    model = MODEL_NAMES[str(trade.get("model") or "normal")]
    shift = float(trade.get("shift") or 0.0) / 100.0
    terms = dict(_cap_terms(trade, market), **_start(trade, market))
    notional = trade.sign * trade.notional
    strike = trade.get("strike")
    if strike is None:
        strike_value = CapFloor(notional, trade.type, 0.0, trade.get("tenor"), 0.0, **terms).atm_strike(market)
    else:
        strike_value = float(strike) / 100.0
    vol = trade.get("vol")
    if vol is not None:
        return CapFloor(notional, trade.type, strike_value, trade.get("tenor"), float(vol) * _vol_units(model), model,
                        shift, **terms), "flat"  # fmt: skip
    key = cap_key(trade)
    stripped = vols.caplets.get(key)
    if stripped is None:
        raise ValueError(f"no stripped caplet vols for {key} caps: give the trade a flat vol, or strip cap quotes")
    if stripped.model != model or float(stripped.shift) != (shift if model == "shifted_black" else 0.0):
        raise ValueError(
            f"the {key} caplet vols were stripped for the {MODEL_LABELS[stripped.model]} model: use that model or "
            "give a flat vol"
        )
    return CapFloor(notional, trade.type, strike_value, trade.get("tenor"), stripped, model, shift, **terms), "stripped"


def _swaption(trade: Trade, market: Market, vols: VolSet) -> Tuple[Swaption, str]:
    model = MODEL_NAMES[str(trade.get("model") or "normal")]
    shift = float(trade.get("shift") or 0.0) / 100.0
    under = cube_key(trade)
    index = None if under == "OIS" else under
    if index:
        market.projection(index)
    notional = trade.sign * trade.notional
    kind = str(trade.get("option"))
    probe = Swaption(notional, trade.get("expiry"), trade.get("tenor"), 0.0, kind, index=index)
    inputs = probe.inputs(market)
    strike = trade.get("strike")
    K = inputs.forward if strike is None else float(strike) / 100.0
    vol = trade.get("vol")
    if vol is not None:
        sigma, source = float(vol) * _vol_units(model), "flat"
    else:
        cube = vols.cubes.get(under)
        if cube is None:
            raise ValueError(f"no swaption vol cube for {under} underlyings: give the trade a flat vol")
        sigma = cube_vol(cube, inputs.expiry_years, trade.get("tenor"), inputs.forward, K, model, shift)
        source = "cube"
    trade_obj = Swaption(notional, trade.get("expiry"), trade.get("tenor"), K, kind, sigma, model, shift, index)
    return trade_obj.pinned(market), source


def trade_position(trade: Trade, market: Market, vols: Optional[VolSet] = None) -> Position:
    """The trade's single-trade pricer as a :class:`Position` named by its
    id; ``details`` carries ``trade`` (the blotter row), ``instrument`` (the
    pricer object) and ``vol_source`` (``flat``, ``cube``, ``stripped`` or
    None). Raises ValueError (currency, missing curve or vols, dates)."""

    vols = vols or VolSet()
    _check_currency(trade, market)
    vol_source: Optional[str] = None
    if trade.type == "swap":
        instrument = _swap(trade, market)
        position = swap_position(instrument, trade.id)
    elif trade.type == "basis swap":
        instrument = _basis(trade, market)
        position = basis_position(instrument, trade.id)
    elif trade.type == "fra":
        instrument = _fra(trade, market)
        position = fra_position(instrument, trade.id)
    elif trade.type == "future":
        instrument = _future(trade, market, vols)
        position = future_position(instrument, trade.id)
    elif trade.type == "bond":
        instrument, z = _bond(trade, market)
        position = bond_position(instrument, trade.sign * trade.notional, z, trade.id)
    elif trade.type in ("cap", "floor"):
        instrument, vol_source = _cap(trade, market, vols)
        position = capfloor_position(instrument, asof=market.asof, name=trade.id)
        instrument = position.details["trade"]
    else:
        instrument, vol_source = _swaption(trade, market, vols)
        position = swaption_position(instrument, market, trade.id)
    position.details = {**position.details, "blotter": trade, "instrument": instrument, "vol_source": vol_source,
                        "currency": _currency_of(trade, market)}  # fmt: skip
    value = float(position.pv(market))
    if not math.isfinite(value):
        raise ValueError(f"no finite value on {market.asof}")
    return position


# --------------------------------------------------------------------------
# Book risk
# --------------------------------------------------------------------------

BumpProvider = Callable[[Market, str], CurveBumps]


def book_curves(market: Market) -> Tuple[str, ...]:
    """Curves the book is bucketed on: the discount curve, every projection
    curve with quotes of its own, and the Treasury curve."""

    out = []
    for key in market.curve_keys:
        if key not in (DISCOUNT_KEY, GOVT_KEY) and market.projections.is_discount(key):
            continue
        if market.get_curve(key).quotes:
            out.append(key)
    return tuple(out)


def _depends(trade: Trade, key: str) -> bool:
    """Whether a trade's value can move with curve ``key``: bonds price on
    the Treasury curve only, everything else never on it."""

    return (key == GOVT_KEY) == (trade.type == "bond")


def _default_bumps(market: Market, key: str) -> CurveBumps:
    return curve_bumps(market, key, BP, parallel=False)


def _rate_moved(market: Market, curves: Sequence[str], bp: float) -> Market:
    bumps = {}
    for key in curves:
        curve = market.get_curve(key)
        bumps[key] = [0.0 if inst.kind in SPREAD_KINDS else bp for inst in curve.instruments]
    return market.rebuilt(bumps)


def _tent(x: float, nodes: np.ndarray) -> np.ndarray:
    """Linear interpolation weights of ``x`` on ``nodes`` (flat beyond the
    ends), summing to 1."""

    w = np.zeros(nodes.size)
    if nodes.size == 1 or x <= nodes[0]:
        w[0] = 1.0
    elif x >= nodes[-1]:
        w[-1] = 1.0
    else:
        j = int(np.searchsorted(nodes, x, side="right")) - 1
        a = (x - nodes[j]) / (nodes[j + 1] - nodes[j])
        w[j], w[j + 1] = 1.0 - a, a
    return w


def cap_normal_vegas(cf: CapFloor, market: Market) -> pd.DataFrame:
    """Live caplets of a cap or floor with their expiry (years) and vega
    per +1bp of normal vol (Black models: of the normal vol with the same
    price, so every option's vega is in one unit; a caplet worth under
    1e-12 of the notional has no such vol and a vega of 0)."""

    if cf.kind == "collar":  # the long cap plus the short floor, each at its own strike and vol
        cap = cap_normal_vegas(replace(cf, kind="cap", floor_strike=None), market)
        floor = replace(cf, kind="floor", strike=cf.floor_strike, floor_strike=None, notional=-cf.notional)
        return cap.assign(vega=cap["vega"] + cap_normal_vegas(floor, market)["vega"])
    table = cf.caplets(market)
    table = table[~table["fixed"]].reset_index(drop=True)
    if cf.model == "bachelier" or table.empty:
        return pd.DataFrame({"expiry": table["expiry"], "vega": table["vega"]})
    shift = cf.shift if cf.model == "shifted_black" else 0.0
    F, vols = table["forward"].to_numpy(), table["vol"].to_numpy()
    T = table["variance_time"].to_numpy()  # the option time (Lyashenko-Mercurio for a compounded rate)
    K = np.full(F.size, cf.strike)
    normal = np.asarray(lognormal_to_normal_vol(F, K, T, vols, shift=shift), dtype=float)
    scale = cf.notional * table["tau"].to_numpy() * table["df"].to_numpy()
    worthless = np.abs(table["pv"].to_numpy()) <= 1e-12 * abs(cf.notional)
    bad = ~np.isfinite(normal) & ~worthless
    if bad.any():
        raise ValueError(f"no normal vol matches the {cf.model} price of the caplet fixing {table['fix'][bad].iloc[0]}")
    vega = np.array([0.0 if w else s * bachelier_vega(f, k, t, n) * 1e-4
                     for s, f, k, t, n, w in zip(scale, F, K, T, normal, worthless)])  # fmt: skip
    return pd.DataFrame({"expiry": table["expiry"].to_numpy(), "vega": vega})


def _labels_years(labels: Sequence[str]) -> np.ndarray:
    return np.array([parse_tenor(t).years for t in labels], dtype=float)


def _expiry_years(labels: Sequence[str], market: Market) -> np.ndarray:
    """ACT/365F years from the valuation date to each option expiry tenor's
    expiry date (as the swaption cube measures its expiries)."""

    return np.array([(Swaption(1.0, t, "1Y", 0.0).expiry_on(market) - market.asof).days / 365.0 for t in labels])


def cap_label(cf: CapFloor) -> str:
    """Vega bucket column of a cap: ``CAP <index>`` (a term index, ``OIS 3M``
    for the overnight rate compounded in arrears, ``3M`` for a single-curve
    term rate), as the blotter's ``index`` column names it."""

    if cf.projection:
        return f"CAP {cf.projection}"
    return f"CAP OIS {cf.index_months}M" if cf.compounded else f"CAP {cf.index_months}M"


def option_vega_rows(
    instrument: object,
    market: Market,
    expiries: Sequence[str] = GRID_EXPIRIES,
    tenors: Sequence[str] = GRID_TENORS,
    cube: Optional[SwaptionVolCube] = None,
    label: Optional[str] = None,
) -> Tuple[float, List[Dict[str, object]]]:
    """(vega per +1bp normal vol, bucket rows) of a swaption or a cap /
    floor / collar. Rows are ``expiry``, ``bucket`` (swap tenor, or ``CAP
    <index>``), ``vega``, ``method``: a swaption priced on ``cube`` by the
    cube's ATM points it moves (``cube``, :func:`vega_buckets`), any other
    swaption with its vega split bilinearly between the nearest grid
    expiries and tenors (``allocated``), a cap each caplet's vega split
    linearly between the nearest grid expiries (``allocated``, column
    ``label`` or :func:`cap_label`). Vegas sum to the trade's vega (cube
    buckets: to within the second-order terms of the bumps)."""

    rows: List[Dict[str, object]] = []
    exp_years = _expiry_years(expiries, market)
    if isinstance(instrument, Swaption):
        greeks = instrument.greeks(market)
        vega = float(greeks["vega_normal_bp"])
        if not math.isfinite(vega):
            vega = 0.0
        if cube is not None:
            grid = vega_buckets([instrument], cube, market)
            for i, e in enumerate(cube.expiries):
                for j, t in enumerate(cube.tenors):
                    if grid[i, j] != 0.0:
                        rows.append({"expiry": e, "bucket": t, "vega": float(grid[i, j]), "method": "cube"})
            return vega, rows
        inputs = instrument.inputs(market)
        we = _tent(inputs.expiry_years, exp_years)
        wt = _tent(parse_tenor(instrument.tenor).years, _labels_years(tenors))
        for i, e in enumerate(expiries):
            for j, t in enumerate(tenors):
                if we[i] * wt[j]:
                    rows.append({"expiry": e, "bucket": t, "vega": vega * we[i] * wt[j], "method": "allocated"})
        return vega, rows
    if not isinstance(instrument, CapFloor):
        raise ValueError(f"no vega buckets for a {type(instrument).__name__}: swaptions and caps / floors have them")
    caplets = cap_normal_vegas(instrument, market)
    bucket = label or cap_label(instrument)
    totals = np.zeros(exp_years.size)
    for expiry, value in zip(caplets["expiry"], caplets["vega"]):
        totals += _tent(float(expiry), exp_years) * float(value)
    rows = [{"expiry": e, "bucket": bucket, "vega": float(v), "method": "allocated"}
            for e, v in zip(expiries, totals) if v != 0.0]  # fmt: skip
    return float(caplets["vega"].sum()), rows


def _vega_rows(position: Position, trade: Trade, market: Market, vols: VolSet, expiries: Sequence[str],
               tenors: Sequence[str]) -> Tuple[float, List[Dict[str, object]]]:  # fmt: skip
    """(vega per +1bp normal, bucket rows) of one option trade of the book."""

    instrument = position.details["instrument"]
    if trade.type == "swaption":
        cube = vols.cubes.get(cube_key(trade)) if position.details["vol_source"] == "cube" else None
        return option_vega_rows(instrument, market, expiries, tenors, cube)
    return option_vega_rows(instrument, market, expiries, tenors, label=f"CAP {cap_key(trade)}")


def position_vega_buckets(position: Position, market: Market) -> List[Dict[str, object]]:
    """Vega bucket rows of any position (see :func:`option_vega_rows`): its
    own rows when it carries them (``details["vega_buckets"]``: the BOOK's
    trades, set by :func:`price_book`, and the SWAPTION tab's trade, both
    by the SABR cube's ATM points when priced on the cube), else those of
    the swaption or cap in ``details["trade"]`` on the standard grid
    (:data:`~derivx.rates.swaption_vol.GRID_EXPIRIES` x
    :data:`~derivx.rates.swaption_vol.GRID_TENORS`, allocated); none for a
    position without vol."""

    rows = position.details.get("vega_buckets") if isinstance(position.details, Mapping) else None
    if rows is not None:
        return [dict(r) for r in rows]
    instrument = position.details.get("trade") if isinstance(position.details, Mapping) else None
    if not isinstance(instrument, (Swaption, CapFloor)):
        return []
    return option_vega_rows(instrument, market)[1]


@dataclass
class BookRisk:
    """Risk of the priced trades on one market (see the module docstring
    for units and signs); ``skipped`` lists trades not priced with the
    reason, ``notes`` where trades are priced beyond the market data."""

    asof: dt.date
    trades: Tuple[Trade, ...]
    positions: Tuple[Position, ...]
    pv: np.ndarray
    pv01: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    risk: Dict[str, QuoteRisk]
    vega_buckets: pd.DataFrame
    vega_grid: Tuple[Tuple[str, ...], Tuple[str, ...]]
    skipped: Tuple[Tuple[Trade, str], ...]
    currencies: Tuple[str, ...]
    curve_currency: Dict[str, str]
    notes: Tuple[str, ...] = ()  # where trades are priced beyond the market data (:func:`extrapolation_notes`)

    @property
    def ids(self) -> List[str]:
        return [t.id for t in self.trades]

    @property
    def mixed(self) -> bool:
        return len(set(self.currencies)) > 1

    def trade_rows(self) -> pd.DataFrame:
        """One row per trade: id, type, description, currency, pv, pv01,
        gamma, vega, vol source."""

        rows = []
        for k, (trade, position) in enumerate(zip(self.trades, self.positions)):
            rows.append({"id": trade.id, "type": trade.type, "description": trade.label,
                         "currency": self.currencies[k], "pv": float(self.pv[k]), "pv01": float(self.pv01[k]),
                         "gamma": float(self.gamma[k]), "vega": float(self.vega[k]),
                         "vol_source": position.details.get("vol_source") or ""})  # fmt: skip
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    def summary(self, reporting: Optional[Reporting] = None) -> pd.DataFrame:
        """One row per trade, a TOTAL row per currency (``TOTAL`` for a book
        in one currency, ``TOTAL USD``, ``TOTAL EUR``... otherwise) and, with
        ``reporting`` when some trade is in another currency, ``ALL IN <R>``
        (every trade converted at the stated FX spots; the description says
        which): id, type, description, currency, pv, pv01, gamma, vega, vol
        source (see :func:`summary_table`)."""

        return summary_table([self], reporting)

    def ladder(self) -> pd.DataFrame:
        """Key-rate PV01s, long: curve, quote, kind, years, trade id,
        pv01 (P&L per +1bp of that quote), gamma (per (1bp)^2), currency."""

        rows = []
        for key, qr in self.risk.items():
            for i, label in enumerate(qr.labels):
                for j, trade in enumerate(self.trades):
                    rows.append({"curve": key, "quote": label, "years": float(qr.years[i]), "id": trade.id,
                                 "pv01": float(qr.pv01[i, j]), "gamma": float(qr.gamma[i, j]),
                                 "currency": self.curve_currency[key]})  # fmt: skip
        return pd.DataFrame(rows, columns=["curve", "quote", "years", "id", "pv01", "gamma", "currency"])

    def ladder_totals(self) -> pd.DataFrame:
        """Book key-rate PV01 per quote: curve, quote, years, pv01, gamma."""

        rows = []
        for key, qr in self.risk.items():
            for i, label in enumerate(qr.labels):
                rows.append({"curve": key, "quote": label, "years": float(qr.years[i]),
                             "pv01": float(qr.pv01[i].sum()), "gamma": float(qr.gamma[i].sum()),
                             "currency": self.curve_currency[key]})  # fmt: skip
        return pd.DataFrame(rows, columns=["curve", "quote", "years", "pv01", "gamma", "currency"])

    def stacked(self) -> np.ndarray:
        """The book's key-rate ladder on every quote of every curve, in
        :attr:`risk` order (what the hedge optimiser flattens)."""

        return np.concatenate([qr.pv01.sum(axis=1) for qr in self.risk.values()]) if self.risk else np.zeros(0)

    def vega_matrix(self) -> pd.DataFrame:
        """Book vega per +1bp normal by expiry (rows) and bucket (columns:
        swaption tenors, then ``CAP <index>``), summed over trades."""

        expiries, tenors = self.vega_grid
        caps = sorted({b for b in self.vega_buckets["bucket"] if b not in tenors}) if len(self.vega_buckets) else []
        table = pd.DataFrame(0.0, index=list(expiries), columns=list(tenors) + caps)
        for row in self.vega_buckets.itertuples():
            if row.expiry in table.index and row.bucket in table.columns:
                table.loc[row.expiry, row.bucket] += row.vega
        return table


SUMMARY_COLUMNS = ["id", "type", "description", "currency", "pv", "pv01", "gamma", "vega", "vol_source"]


def summary_table(risks: Sequence[BookRisk], reporting: Optional[Reporting] = None) -> pd.DataFrame:
    """The trades of one or more books (each priced on its own market, e.g.
    one per currency) in one table, with a TOTAL row per currency and, with
    ``reporting`` when some trade is in another currency, ``ALL IN <R>``:
    amounts are never added across currencies unconverted."""

    frame = pd.concat([r.trade_rows() for r in risks], ignore_index=True) if risks else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    table = add_total_rows(frame, ("pv", "pv01", "gamma", "vega"), "id", "currency", reporting, note="description")
    total = ~table.index.isin(frame.index)
    table.loc[total, ["type", "vol_source"]] = ""
    notes = [str(d).replace("positions in", "trades in").replace("position in", "trade in")
             for d in table.loc[total, "description"]]  # fmt: skip
    table.loc[total, "description"] = notes
    return table


def _trade_curves(trade: Trade) -> List[str]:
    """Curves a trade's cash flows are projected or discounted on."""

    index = str(trade.get("index") or "")
    if trade.type == "bond":
        return [GOVT_KEY]
    if trade.type == "basis swap":
        return [DISCOUNT_KEY] + index.split("/")
    if trade.type == "future":
        return ["EURIBOR 3M"] if index == "FUT" else [DISCOUNT_KEY]
    return [DISCOUNT_KEY] + ([index] if index in INDICES else [])


def _curve_name(market: Market, key: str) -> str:
    if key == DISCOUNT_KEY:
        conv = market.curve.conventions
        return f"{conv.name if conv is not None else 'swap'} discount"
    return "US Treasury" if key == GOVT_KEY else key


def _last_date(trade: Trade, position: Position, market: Market) -> dt.date:
    inst = position.details["instrument"]
    if trade.type in ("swap", "basis swap"):
        return inst.termination_date(market.asof)
    if trade.type == "fra":
        return inst.dates(market.asof)["maturity"]
    if trade.type == "future":
        return inst.period(market.asof)[1]
    if trade.type == "bond":
        return inst.maturity
    if trade.type in ("cap", "floor"):
        return max(inst.caplets(market)["pay"])
    return inst.inputs(market).maturity


def extrapolation_notes(trades: Sequence[Trade], positions: Sequence[Position], market: Market,
                        vols: Optional[VolSet] = None) -> List[str]:  # fmt: skip
    """Where the book is priced beyond its market data, as on the SWAP and
    SWAPTION tabs: a trade paying after the last quote of a curve it uses
    (forwards there are flat extrapolations, not market data), a
    cube-priced swaption outside the cube's expiries or tenors, a cap on
    stripped vols beyond the last stripped caplet (vols held flat) and a cap
    on a single-curve term rate (an approximation of the compounded
    overnight-rate caps that trade)."""

    vols = vols or VolSet()
    notes: List[str] = []
    for trade, position in zip(trades, positions):
        try:
            notes += _trade_notes(trade, position, market, vols)
        except ValueError:  # a priced trade's dates always resolve; never let a note stop the book
            continue
    return notes


def _trade_notes(trade: Trade, position: Position, market: Market, vols: VolSet) -> List[str]:
    """Notes of one priced trade (see :func:`extrapolation_notes`)."""

    notes: List[str] = []
    end = _last_date(trade, position, market)
    years = (end - market.asof).days / 365.0
    for key in _trade_curves(trade):
        curve = market.get_curve(key)
        last = float(curve.ts[-1])
        if years > last + 1.0 / 365.0:
            pillar = (curve.pillar_dates or [None])[-1]
            notes.append(
                f"{trade.id} pays until {end}, after the last quote of the {_curve_name(market, key)} curve "
                f"({pillar or f'{last:.1f}Y'}): forwards there are flat extrapolations, not market data"
            )
            break
    source = position.details.get("vol_source")
    inst = position.details["instrument"]
    if source == "cube":
        cube = vols.cubes.get(cube_key(trade))
        if cube is not None:
            expiry = inst.inputs(market).expiry_years
            outside = expiry > float(_expiry_years([cube.expiries[-1]], market)[0]) + 1e-9 or (
                parse_tenor(str(trade.get("tenor"))).years > parse_tenor(cube.tenors[-1]).years
            )
            if outside:
                notes.append(f"{trade.id} is outside the swaption cube ({cube.expiries[-1]} x {cube.tenors[-1]} "
                             "at most): its vol is held flat from the cube's edge")  # fmt: skip
    if trade.type in ("cap", "floor") and not inst.compounded and inst.projection is None:
        conv = market.curve.conventions
        overnight = OVERNIGHT_NAMES.get(conv.currency, conv.name) if conv is not None else "the overnight rate"
        notes.append(
            f"{trade.id} is on a {inst.index_months}M term rate projected off the OIS curve: an approximation "
            f"(caps in this currency trade on {overnight} compounded in arrears: index OIS {inst.index_months}M)"
        )
    if source == "stripped":
        stripped = vols.caplets.get(cap_key(trade))
        curve = (list(getattr(stripped, "curves", ())) or [stripped])[0] if stripped is not None else None
        live = inst.caplets(market)
        if curve is not None and len(live) and float(live["expiry"].max()) > float(curve.nodes[-1]) + 1e-9:
            notes.append(f"{trade.id} has caplets fixing after the last stripped caplet ({curve.nodes[-1]:.1f}Y): "
                         "their vols are held flat")  # fmt: skip
    return notes


def _currency_of(trade: Trade, market: Market) -> str:
    if trade.type == "bond":
        return "USD"
    conv = market.curve.conventions
    return conv.currency if conv is not None else trade.currency


def price_book(
    trades: Sequence[Trade],
    market: Market,
    vols: Optional[VolSet] = None,
    bumps: Optional[BumpProvider] = None,
    vega: bool = True,
) -> BookRisk:
    """Price every trade and measure the book's risk (see the module
    docstring). ``bumps(market, key)`` supplies the markets with one quote
    of curve ``key`` moved by +/-1bp (default
    :func:`~derivx.rates.swaps.curve_bumps`; the app passes its cached
    ones). Trades that cannot be priced on this market (another currency,
    a missing projection curve or vols) are skipped with the reason."""

    vols = vols or VolSet()
    bumps = bumps or _default_bumps
    priced: List[Trade] = []
    positions: List[Position] = []
    skipped: List[Tuple[Trade, str]] = []
    for trade in trades:
        try:
            positions.append(trade_position(trade, market, vols))
            priced.append(trade)
        except ValueError as exc:
            skipped.append((trade, str(exc)))
    n = len(positions)
    base = np.array([float(p.pv(market)) for p in positions])
    curves = book_curves(market)
    ccy = market.curve.conventions.currency if market.curve.conventions is not None else "USD"
    curve_currency = {key: ("USD" if key == GOVT_KEY else ccy) for key in curves}
    risk: Dict[str, QuoteRisk] = {}
    for key in curves:
        curve = market.get_curve(key)
        nq = len(curve.quotes)
        pv01, gamma = np.zeros((nq, n)), np.zeros((nq, n))
        live = [k for k, t in enumerate(priced) if _depends(t, key)]
        if live:
            moved = bumps(market, key)
            for i, (up_m, down_m) in enumerate(zip(moved.up, moved.down)):
                for k in live:
                    up, down = float(positions[k].pv(up_m)), float(positions[k].pv(down_m))
                    pv01[i, k] = (up - down) / (2.0 * moved.bp)
                    gamma[i, k] = (up + down - 2.0 * base[k]) / (moved.bp * moved.bp)
        risk[key] = QuoteRisk(key, tuple(curve.labels), curve.years, pv01, gamma, np.ones(nq, dtype=bool), market.asof)
    par01, par_gamma = np.zeros(n), np.zeros(n)
    if n and curves:
        up_m, down_m = _rate_moved(market, curves, BP), _rate_moved(market, curves, -BP)
        up = np.array([float(p.pv(up_m)) for p in positions])
        down = np.array([float(p.pv(down_m)) for p in positions])
        par01, par_gamma = (up - down) / (2.0 * BP), (up + down - 2.0 * base) / (BP * BP)
    cube = next(iter(vols.cubes.values()), None)
    grid = (tuple(cube.expiries), tuple(cube.tenors)) if cube is not None else (GRID_EXPIRIES, GRID_TENORS)
    vegas = np.zeros(n)
    bucket_rows: List[Dict[str, object]] = []
    if vega:
        for k, (trade, position) in enumerate(zip(priced, positions)):
            if trade.type in ("cap", "floor", "swaption"):
                vegas[k], rows = _vega_rows(position, trade, market, vols, grid[0], grid[1])
                bucket_rows += [{"id": trade.id, **r} for r in rows]
                position.details["vega_buckets"] = tuple(rows)  # the RESEARCH risk report reads them
    buckets = pd.DataFrame(bucket_rows, columns=["id", "expiry", "bucket", "vega", "method"])
    return BookRisk(
        asof=market.asof,
        trades=tuple(priced),
        positions=tuple(positions),
        pv=base,
        pv01=par01,
        gamma=par_gamma,
        vega=vegas,
        risk=risk,
        vega_buckets=buckets,
        vega_grid=grid,
        skipped=tuple(skipped),
        currencies=tuple(_currency_of(t, market) for t in priced),
        curve_currency=curve_currency,
        notes=tuple(extrapolation_notes(priced, positions, market, vols)),
    )


# --------------------------------------------------------------------------
# Hedging
# --------------------------------------------------------------------------

HEDGE_TENORS = ("2Y", "5Y", "10Y", "30Y")
ASSUMED_BID_ASK_BP = {"curve": 0.25, "govt": 0.5}  # ASSUMED default bid/ask (bp), not market data; basis: 0.5
ASSUMED_BASIS_BID_ASK_BP = 0.5


@dataclass(frozen=True)
class HedgeInstrument:
    """One unit of a par hedge on a curve: ``position`` for
    ``unit_notional``; a positive multiple receives fixed (swap), receives
    the spread leg (basis swap) or buys the bond. ``quote_dv01`` is its
    value change for 1bp on its own quote (swap and basis annuity, bond
    yield DV01), positive, in currency."""

    curve: str
    label: str
    position: Position
    unit_notional: float
    quote_dv01: float
    par: float
    bid_ask_bp: float = 0.0


def hedge_instruments(
    market: Market,
    benchmarks: Mapping[str, Sequence[str]],
    unit_notional: float = DEFAULT_UNIT_NOTIONAL,
    bid_ask_bp: Optional[Mapping[Tuple[str, str], float]] = None,
) -> List[HedgeInstrument]:
    """Par hedge instruments traded on the valuation date: OIS swaps on the
    discount curve, swaps or tenor basis swaps on a projection curve
    (:func:`~derivx.rates.swaps.benchmark_instruments`), par Treasuries
    issued at settlement on the Treasury curve (``benchmarks["govt"]``, whole
    years). ``bid_ask_bp`` maps (curve, tenor) to the bid/ask in bp of the
    instrument's quote (default: none)."""

    costs = dict(bid_ask_bp or {})
    out: List[HedgeInstrument] = []
    for key, tenors in benchmarks.items():
        tenors = [str(t).strip().upper() for t in tenors]
        if len(set(tenors)) != len(tenors):
            raise ValueError(f"{key} hedge tenors must be distinct: {tenors}")
        if not tenors:
            continue
        if key == GOVT_KEY:
            if market.govt is None:
                raise ValueError("the market has no Treasury curve to hedge with Treasuries")
            conv = market.govt.conventions
            cal = conv.cal if conv is not None else Calendar("USGOV")
            settle = cal.add_business_days(cal.adjust(market.asof, "F"), conv.spot_lag if conv is not None else 1)
            for tenor in tenors:
                years = parse_tenor(tenor).years
                if not float(years).is_integer():
                    raise ValueError(f"Treasury hedge tenors are whole years, got {tenor}")
                bond = benchmark_bond(market.govt, int(years), settle)
                position = bond_position(bond, unit_notional, 0.0, f"Hedge {tenor} UST")
                dv01 = abs(bond.dv01(bond.coupon, settle, unit_notional))
                out.append(HedgeInstrument(key, f"{tenor} UST", position, unit_notional, dv01, bond.coupon,
                                           float(costs.get((key, tenor), 0.0))))  # fmt: skip
            continue
        for (label, position), tenor in zip(benchmark_instruments(market, key, tenors, unit_notional), tenors):
            details = position.details
            unit = VanillaSwap(**details) if "fixed_rate" in details else BasisSwap(**details)
            par = float(details["fixed_rate"] if "fixed_rate" in details else details["spread"])
            out.append(HedgeInstrument(key, label, position, unit_notional, abs(unit.annuity(market)), par,
                                       float(costs.get((key, tenor), 0.0))))  # fmt: skip
    if not out:
        raise ValueError("pick at least one hedge instrument")
    return out


def _lasso(G: np.ndarray, b: np.ndarray, penalty: np.ndarray, x0: np.ndarray, tol: float, sweeps: int) -> np.ndarray:
    """argmin 1/2 x'Gx + b'x + sum_j penalty_j |x_j| by coordinate descent
    (G positive semidefinite; a coordinate with G_jj = 0 stays at 0)."""

    x = x0.copy()
    diag = np.diag(G)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    for _ in range(sweeps):
        biggest = 0.0
        for j in range(x.size):
            if diag[j] <= 0.0:
                x[j] = 0.0
                continue
            z = -(b[j] + G[j] @ x - diag[j] * x[j])
            new = math.copysign(max(abs(z) - penalty[j], 0.0), z) / diag[j]
            biggest = max(biggest, abs(new - x[j]) * diag[j])
            x[j] = new
        if biggest <= tol * scale:
            break
    return x


REDUNDANT = 1e-3  # relative singular value under which a combination of hedges counts as redundant


def least_squares_hedge(target: np.ndarray, hedges: np.ndarray, rcond: float = REDUNDANT) -> np.ndarray:
    """Least-squares hedge multiples ``x`` (min ``||target + hedges @ x||``)
    that trade no redundant combination of hedges: with every hedge's ladder
    scaled to unit length, singular directions under ``rcond`` times the
    largest are left out (truncated SVD). Such a direction is a mix of
    hedges whose ladders almost cancel (a 15Y bond between 10Y and 20Y
    quotes, say): it takes out next to no risk and an exact least-squares
    fit buys it with huge offsetting notionals. Well-conditioned hedges get
    the exact least-squares (minimum-norm) answer."""

    t = np.asarray(target, dtype=float)
    H = np.asarray(hedges, dtype=float)
    scale = np.linalg.norm(H, axis=0)
    live = scale > 0.0
    x = np.zeros(H.shape[1])
    if not live.any():
        return x
    y, *_ = np.linalg.lstsq(H[:, live] / scale[live], -t, rcond=rcond)
    x[live] = y / scale[live]
    return x


def optimise_hedge(
    target: np.ndarray,
    hedges: np.ndarray,
    unit_costs: Optional[np.ndarray] = None,
    move_bp: Optional[float] = None,
    tol: float = 1e-13,
    sweeps: int = 5000,
) -> np.ndarray:
    """Hedge multiples ``x`` minimising ``move_bp * ||target + hedges @ x||_2
    + sum_j unit_costs_j |x_j|`` (``hedges`` is buckets x instruments, in
    currency per bp per unit; ``unit_costs`` in currency per unit traded).
    Without costs (or ``move_bp`` None) it is the least-squares hedge
    without redundant combinations of hedges (:func:`least_squares_hedge`).

    The problem is convex; it is solved through the lasso ``1/2 ||target +
    hedges @ x||^2 + lam * sum_j cost_j |x_j|`` (coordinate descent), whose
    optimality conditions match the original problem's when ``lam =
    ||residual|| / move_bp``: that fixed point is found by root finding."""

    t = np.asarray(target, dtype=float)
    H = np.asarray(hedges, dtype=float)
    if H.ndim != 2 or H.shape[0] != t.shape[0]:
        raise ValueError("hedges must be (buckets x instruments) matching the target ladder")
    m = H.shape[1]
    costs = np.zeros(m) if unit_costs is None else np.asarray(unit_costs, dtype=float)
    if costs.shape != (m,) or np.any(~np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("unit costs must be one non-negative number per hedge instrument")
    if move_bp is not None and not (math.isfinite(move_bp) and move_bp > 0):
        raise ValueError("the move size must be a positive number of bp")
    if not np.any(t):
        return np.zeros(m)
    if move_bp is None or not np.any(costs):
        return least_squares_hedge(t, H)
    G, b = H.T @ H, H.T @ t
    state = {"x": np.zeros(m)}

    def residual_norm(lam: float) -> float:
        state["x"] = _lasso(G, b, lam * costs, state["x"], tol, sweeps)
        return float(np.linalg.norm(t + H @ state["x"]))

    norm_t = float(np.linalg.norm(t))
    hi = norm_t / move_bp

    def g(lam: float) -> float:
        return lam * move_bp - residual_norm(lam)

    lo = hi * 1e-12
    g_lo = g(lo)
    if g_lo >= 0.0:  # the book can be hedged (almost) exactly: the cheapest such hedge
        return state["x"]
    lam = brentq(g, lo, hi, xtol=hi * 1e-14, rtol=1e-14, maxiter=200)
    state["x"] = np.zeros(m)
    residual_norm(lam)
    return state["x"]


def hedge_objective(target: np.ndarray, hedges: np.ndarray, x: np.ndarray, unit_costs: np.ndarray,
                    move_bp: float) -> float:  # fmt: skip
    """``move_bp * ||target + hedges @ x|| + sum_j unit_costs_j |x_j|``."""

    return float(move_bp * np.linalg.norm(np.asarray(target) + np.asarray(hedges) @ x)
                 + np.asarray(unit_costs) @ np.abs(x))  # fmt: skip


@dataclass(frozen=True)
class HedgeRisk:
    """The risk of one currency's curves before and after the hedge, in that
    currency: root sum of squares of the stacked key-rate ladder (per bp),
    the bid/ask crossed (``cost``) and, with the cost penalty, the move size
    ``move_bp`` the hedge was sized against."""

    currency: str
    curves: Tuple[str, ...]
    risk_before: float
    risk_after: float
    cost: float
    move_bp: Optional[float]

    @property
    def risk_removed(self) -> float:
        """Share of the squared ladder the hedge takes out."""

        before = self.risk_before
        return 1.0 - (self.risk_after * self.risk_after) / (before * before) if before > 0 else 1.0

    @property
    def worst_before(self) -> Optional[float]:
        """Largest first-order loss of the book on a curve move of
        ``move_bp`` (Euclidean norm of the bp moves), unhedged."""

        return None if self.move_bp is None else self.move_bp * self.risk_before

    @property
    def worst_after(self) -> Optional[float]:
        return None if self.move_bp is None else self.move_bp * self.risk_after


@dataclass
class HedgeResult:
    """Sized hedges and the residual risk they leave (currency per bp unless
    stated): ``table`` one row per instrument (curve, currency, instrument,
    notional signed as in :class:`HedgeInstrument`, direction, par, bid/ask,
    cost, pv01); ``ladder`` the stacked book, hedge and residual ladders
    (with each curve's currency); ``positions`` the sized hedges;
    ``by_currency`` the risk of each currency's curves (hedged on their
    own). ``risk_before``, ``risk_after``, ``risk_removed``, ``cost`` and
    the ``worst_*`` losses are those of a book in one currency; for a book
    in several they raise ``ValueError`` (amounts in different currencies do
    not add): read :attr:`by_currency`."""

    table: pd.DataFrame
    ladder: pd.DataFrame
    positions: List[Position]
    by_currency: Dict[str, HedgeRisk]
    move_bp: Optional[float]
    multiples: np.ndarray
    unit_ladders: np.ndarray

    @property
    def currencies(self) -> List[str]:
        return list(self.by_currency)

    def _one(self) -> HedgeRisk:
        if len(self.by_currency) != 1:
            raise ValueError(
                f"the book's curves are in {' and '.join(self.by_currency)}: its risk does not add across currencies; "
                "read it per currency (by_currency)"
            )
        return next(iter(self.by_currency.values()))

    @property
    def risk_before(self) -> float:
        return self._one().risk_before

    @property
    def risk_after(self) -> float:
        return self._one().risk_after

    @property
    def risk_removed(self) -> float:
        return self._one().risk_removed

    @property
    def cost(self) -> float:
        return self._one().cost

    @property
    def worst_before(self) -> Optional[float]:
        return self._one().worst_before

    @property
    def worst_after(self) -> Optional[float]:
        return self._one().worst_after


def unit_ladders(
    instruments: Sequence[HedgeInstrument], market: Market, curves: Sequence[str], bumps: Optional[BumpProvider] = None
) -> np.ndarray:
    """Stacked key-rate ladders (every quote of ``curves``) x instruments."""

    bumps = bumps or _default_bumps
    blocks = []
    for key in curves:
        curve = market.get_curve(key)
        block = np.zeros((len(curve.quotes), len(instruments)))
        live = [k for k, h in enumerate(instruments) if (h.curve == GOVT_KEY) == (key == GOVT_KEY)]
        if live:
            moved = bumps(market, key)
            for i, (up_m, down_m) in enumerate(zip(moved.up, moved.down)):
                for k in live:
                    pv = instruments[k].position.pv
                    block[i, k] = (float(pv(up_m)) - float(pv(down_m))) / (2.0 * moved.bp)
        blocks.append(block)
    return np.vstack(blocks) if blocks else np.zeros((0, len(instruments)))


def _sized(instrument: HedgeInstrument, multiple: float) -> Position:
    details = instrument.position.details
    if "bond" in details:
        bond = details["bond"]
        notional = multiple * instrument.unit_notional
        return bond_position(bond, notional, 0.0, f"Hedge {'LONG' if notional > 0 else 'SHORT'} {instrument.label}")
    return _scaled(instrument.position, abs(multiple), multiple > 0)


def hedge_book(
    risk: BookRisk,
    market: Market,
    instruments: Sequence[HedgeInstrument],
    move_bp: Optional[float] = None,
    bumps: Optional[BumpProvider] = None,
    ladders: Optional[np.ndarray] = None,
) -> HedgeResult:
    """Size ``instruments`` against the book's stacked key-rate ladder (see
    :func:`optimise_hedge`), each currency's curves on their own: the
    ladder of the curves of one currency (``risk.curve_currency``) is
    hedged with the instruments on those curves, so no norm or cost adds
    amounts in different currencies (a hedge on one currency's curves has
    no risk on another's, so without costs this is the least-squares hedge
    of the whole stacked ladder). A currency that holds none of the book's
    trades and whose curves carry none of its risk (the US Treasury curve
    under a euro swap book) is left out of ``by_currency``, and hedges on
    its curves stay at zero. The cost of a hedge is half its bid/ask times
    its quote DV01 per unit. ``ladders`` reuses :func:`unit_ladders`."""

    curves = list(risk.risk)
    target = risk.stacked()
    H = unit_ladders(instruments, market, curves, bumps) if ladders is None else np.asarray(ladders, dtype=float)
    if H.shape != (target.size, len(instruments)):
        raise ValueError("the hedge ladders were built on another market")
    known = set(curves)
    stray = [h.label for h in instruments if h.curve not in known]
    if stray:
        raise ValueError(f"hedges on curves the book has no ladder for: {', '.join(stray)}")
    costs = np.array([0.5 * h.bid_ask_bp * h.quote_dv01 for h in instruments])
    row_ccy = np.array([risk.curve_currency[key] for key, qr in risk.risk.items() for _ in qr.labels], dtype=object)
    hedge_ccy = [risk.curve_currency[h.curve] for h in instruments]
    every = ordered(list(risk.curve_currency[k] for k in curves))
    blocks = [c for c in every if c in risk.currencies or np.any(target[row_ccy == c] != 0.0)] or every
    x = np.zeros(len(instruments))
    for ccy in blocks:
        rows_c = np.flatnonzero(row_ccy == ccy)
        cols_c = [j for j, c in enumerate(hedge_ccy) if c == ccy]
        if cols_c and rows_c.size:
            x[cols_c] = optimise_hedge(target[rows_c], H[np.ix_(rows_c, cols_c)], costs[cols_c], move_bp)
    x[np.abs(x) * np.array([h.unit_notional for h in instruments]) < 0.01] = 0.0  # under a cent of notional
    hedge = H @ x
    residual = target + hedge
    penalised = move_bp is not None and bool(np.any(costs))
    by_currency: Dict[str, HedgeRisk] = {}
    for ccy in blocks:
        rows_c = row_ccy == ccy
        cols_c = [j for j, c in enumerate(hedge_ccy) if c == ccy]
        by_currency[ccy] = HedgeRisk(
            ccy, tuple(k for k in curves if risk.curve_currency[k] == ccy), float(np.linalg.norm(target[rows_c])),
            float(np.linalg.norm(residual[rows_c])), float(sum(costs[j] * abs(x[j]) for j in cols_c)),
            float(move_bp) if penalised and any(costs[j] for j in cols_c) else None,
        )  # fmt: skip
    rows, positions = [], []
    for h, multiple, column, cost in zip(instruments, x, H.T, costs):
        if h.curve == GOVT_KEY:
            side = ("Buy", "Sell")
        elif "fixed_rate" in h.position.details:
            side = ("Receive fixed", "Pay fixed")
        else:
            side = ("Receive spread leg", "Pay spread leg")
        rows.append({"curve": h.curve, "currency": risk.curve_currency[h.curve], "instrument": h.label,
                     "notional": float(multiple * h.unit_notional),
                     "direction": side[0] if multiple > 0 else side[1] if multiple < 0 else "-", "par": h.par,
                     "bid_ask_bp": h.bid_ask_bp, "cost": float(cost * abs(multiple)),
                     "pv01": float(multiple * column.sum())})  # fmt: skip
        if multiple != 0.0:
            positions.append(_sized(h, float(multiple)))
    labels = [(key, label, float(y)) for key, qr in risk.risk.items() for label, y in zip(qr.labels, qr.years)]
    ladder = pd.DataFrame([{"curve": k, "currency": risk.curve_currency[k], "quote": q, "years": y, "book": float(a),
                            "hedge": float(b), "residual": float(c)}
                           for (k, q, y), a, b, c in zip(labels, target, hedge, residual)],
                          columns=["curve", "currency", "quote", "years", "book", "hedge", "residual"])  # fmt: skip
    table = pd.DataFrame(rows, columns=["curve", "currency", "instrument", "notional", "direction", "par",
                                        "bid_ask_bp", "cost", "pv01"])  # fmt: skip
    return HedgeResult(
        table=table,
        ladder=ladder,
        positions=positions,
        by_currency=by_currency,
        move_bp=float(move_bp) if penalised else None,
        multiples=x,
        unit_ladders=H,
    )


def default_benchmarks(risk: BookRisk, market: Market, tenors: Sequence[str] = HEDGE_TENORS) -> Dict[str, List[str]]:
    """Hedge tenors per curve the book has risk on (a curve without risk
    gets none)."""

    out: Dict[str, List[str]] = {}
    for key, qr in risk.risk.items():
        if np.any(np.abs(qr.pv01) > NOISE):
            out[key] = list(tenors)
    return out


def assumed_bid_ask(market: Market, benchmarks: Mapping[str, Sequence[str]]) -> Dict[Tuple[str, str], float]:
    """ASSUMED bid/ask (bp) per hedge, not market data: 0.25bp for OIS and
    term swaps, 0.5bp for tenor basis swaps and Treasuries (yield)."""

    out = {}
    for key, tenors in benchmarks.items():
        if key == GOVT_KEY:
            cost = ASSUMED_BID_ASK_BP["govt"]
        elif key != DISCOUNT_KEY and getattr(market.get_curve(key), "basis_index", None):
            cost = ASSUMED_BASIS_BID_ASK_BP
        else:
            cost = ASSUMED_BID_ASK_BP["curve"]
        for tenor in tenors:
            out[(key, str(tenor).strip().upper())] = cost
    return out


__all__ = [
    "BlotterIssue",
    "BookRisk",
    "COLUMNS",
    "HedgeInstrument",
    "HedgeResult",
    "HedgeRisk",
    "TRADE_TYPES",
    "Trade",
    "VolSet",
    "blotter_csv",
    "blotter_frame",
    "book_curves",
    "column_guide",
    "example_blotter",
    "extrapolation_notes",
    "hedge_book",
    "hedge_instruments",
    "optimise_hedge",
    "option_vega_rows",
    "parse_blotter",
    "position_vega_buckets",
    "price_book",
    "read_blotter_csv",
    "summary_table",
    "trade_position",
]
