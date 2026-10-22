"""Validate and normalise user-entered curve quote tables.

Market data arrives in percent (FRED, Treasury), in decimals, or in basis
points, with labels like ``1 Mo``, ``ON`` or ``10y``. :func:`clean_quotes`
turns a table of rows into decimal :class:`CurveQuote` objects and reports
every problem it finds instead of silently building a wrong curve.

Futures rows (``SR1``, ``SR3``, ``FUT``) are quoted as a price (100 - rate,
e.g. 96.25) whatever the table's units, with the contract month as the
tenor (``DEC26``, ``Z26``, ``2026-12``); they become futures rates. ``FRA``
tenors are ``AxB`` (months to start x months to end). ``BASIS`` rows are
spreads in the table's units; the units are read from the rate rows only.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .conventions import parse_tenor
from .curves import INSTRUMENT_KINDS, SPREAD_KINDS, CurveQuote
from .indices import FUTURES_KINDS, contract_label, futures_rate_from_price, parse_contract

_FRA_LABEL = re.compile(r"^\s*(\d{1,3})\s*[Xx\u00d7]\s*(\d{1,3})\s*$")

RATE_RANGE = (-0.05, 0.30)  # accepted after unit conversion
SPREAD_RANGE = (-0.02, 0.02)  # tenor basis spreads: a few bp to tens of bp; 200bp would be a units slip
UNIT_SCALE = {"decimal": 1.0, "percent": 0.01, "bp": 1e-4}


@dataclass(frozen=True)
class Issue:
    level: str  # "error" stops the build; "warning" and "info" do not
    message: str


def detect_units(rates: Iterable[float]) -> str:
    """Guess quote units from magnitudes: decimals (0.045), percent (4.5) or
    basis points (450)."""

    values = np.abs(np.asarray([r for r in rates if np.isfinite(r)], dtype=float))
    if values.size == 0:
        return "decimal"
    median = float(np.median(values))
    if median > 50:
        return "bp"
    if median > 0.5:
        return "percent"
    return "decimal"


AMBIGUOUS_MEDIAN = (0.05, 0.5)  # |rates| with a median in here read as 5%-50% decimals or 0.05%-0.5% percent


def ambiguous_units(rates: Iterable[float]) -> Optional[str]:
    """Why ``auto`` cannot tell the units of ``rates`` apart, or None.

    :func:`detect_units` calls a median of 0.25 decimal (25%), but near-zero
    rates in percent look the same (0.25%), and so do 5-30% decimals; a
    wrong guess moves every rate 100x and still passes the range check.
    Tables whose every rate is that small need their units set. (With a
    larger rate among them, the range check already refuses the guess.)"""

    values = np.abs(np.asarray([r for r in rates if np.isfinite(r)], dtype=float))
    if values.size == 0:
        return None
    median = float(np.median(values))
    lo, hi = AMBIGUOUS_MEDIAN
    if lo < median <= hi and float(values.max()) <= hi:
        return (
            f"rates around {median:g} could be percent ({median:g}%) or decimals ({median:.0%}); "
            "choose the units instead of auto"
        )
    return None


def clean_quotes(
    rows: Iterable[Mapping[str, object]],
    units: str = "auto",
    default_kind: Optional[str] = None,
    kinds: Sequence[str] = INSTRUMENT_KINDS,
    asof: Optional[dt.date] = None,
) -> Tuple[List[CurveQuote], List[Issue], str]:
    """Rows with ``kind`` (or ``instrument``), ``tenor`` and ``rate`` keys ->
    (quotes in decimals, issues, units used).

    ``kinds`` are the instruments the table may hold (the discount curve's
    by default; :data:`derivx.rates.multicurve.PROJECTION_KINDS` for a
    projection curve); ``asof`` resolves one-digit futures years (``Z6``).
    Blank rows are skipped. Unknown kinds or tenors, non-numeric or
    out-of-range rates or prices, duplicate tenors or contracts and (with
    ``auto``) magnitudes whose units cannot be told apart
    (:func:`ambiguous_units`) are errors.
    """

    kinds = tuple(k.upper() for k in kinds)
    issues: List[Issue] = []
    parsed = []
    for n, row in enumerate(rows, start=1):
        tenor_raw = row.get("tenor")
        rate_raw = row.get("rate")
        kind_raw = row.get("kind", row.get("instrument", default_kind))
        if all(v is None or str(v).strip() in ("", "nan", "None") for v in (tenor_raw, rate_raw)):
            continue
        kind = str(kind_raw or "").strip().upper()
        if kind not in kinds:
            issues.append(Issue("error", f"row {n}: instrument {kind_raw!r} is not one of {', '.join(kinds)}"))
            continue
        try:
            tenor = _parse_label(kind, str(tenor_raw), asof)
        except ValueError as exc:
            issues.append(Issue("error", f"row {n}: {exc}"))
            continue
        try:
            rate = float(str(rate_raw).replace("%", "").strip())
        except ValueError:
            issues.append(Issue("error", f"row {n}: {'price' if kind in FUTURES_KINDS else 'rate'} {rate_raw!r} "
                                         "is not a number"))  # fmt: skip
            continue
        if not np.isfinite(rate):
            issues.append(Issue("error", f"row {n}: rate is missing"))
            continue
        parsed.append((n, kind, tenor, rate))

    level_rates = [r for _, k, _, r in parsed if k not in FUTURES_KINDS and k not in SPREAD_KINDS]
    if units == "auto":
        if not level_rates and any(k in SPREAD_KINDS for _, k, _, _ in parsed):
            return (
                [],
                issues + [Issue("error", "a table of basis spreads only: choose its units instead of auto")],
                units,
            )
        reason = ambiguous_units(level_rates)
        if reason:
            return [], issues + [Issue("error", reason)], units
        units = detect_units(level_rates)
        if units != "decimal" and level_rates:
            issues.append(Issue("info", f"Rates read as {units} and converted to decimals"))
    scale = UNIT_SCALE[units]

    quotes: List[CurveQuote] = []
    seen: Dict[tuple, int] = {}  # quote key -> first row
    for n, kind, tenor, rate in parsed:
        if kind in FUTURES_KINDS:
            try:
                value = futures_rate_from_price(rate)
            except ValueError as exc:
                issues.append(Issue("error", f"row {n}: {exc}"))
                continue
            key, label = (kind, tenor), tenor
        else:
            value = rate * scale
            if not RATE_RANGE[0] <= value <= RATE_RANGE[1]:
                issues.append(
                    Issue("error", f"row {n}: {value:.4%} is outside {RATE_RANGE[0]:.0%} to {RATE_RANGE[1]:.0%}")
                )
                continue
            if kind in SPREAD_KINDS and not SPREAD_RANGE[0] <= value <= SPREAD_RANGE[1]:
                issues.append(
                    Issue(
                        "error",
                        f"row {n}: a {value * 1e4:,.1f}bp basis spread is outside {SPREAD_RANGE[0] * 1e4:+.0f} to "
                        f"{SPREAD_RANGE[1] * 1e4:+.0f}bp: spreads are in the table's units ({units}; 8bp is "
                        f"{8e-4 / scale:g})",
                    )
                )
                continue
            if isinstance(tenor, str):  # FRA label
                key, label = ("FRA", tenor), tenor
            else:
                key, label = (tenor.n, tenor.unit, tenor.anchor), str(tenor)
        if key in seen:
            what = "contract" if kind in FUTURES_KINDS else "tenor"
            issues.append(Issue("error", f"row {n}: {what} {label} repeats row {seen[key]}; keep one quote per {what}"))
            continue
        seen[key] = n
        quotes.append(CurveQuote(kind, label, value))
    if not quotes and not any(i.level == "error" for i in issues):
        issues.append(Issue("error", "no quotes entered"))
    return quotes, issues, units


def _parse_label(kind: str, text: str, asof: Optional[dt.date]):
    """Tenor (a :class:`~derivx.rates.conventions.Tenor`), FRA label
    (``"6X12"``) or futures contract label (``"DEC26"``) of a row."""

    if kind in FUTURES_KINDS:
        return contract_label(*parse_contract(text, asof))
    if kind == "FRA":
        match = _FRA_LABEL.match(text)
        if not match or int(match.group(2)) <= int(match.group(1)):
            raise ValueError(f"FRA {text!r}: write months to start x months to end, e.g. 6X12")
        return f"{int(match.group(1))}X{int(match.group(2))}"
    return parse_tenor(text)
