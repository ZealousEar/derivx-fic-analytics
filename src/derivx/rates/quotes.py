"""Validate and normalise user-entered curve quote tables.

Market data arrives in percent (FRED, Treasury), in decimals, or in basis
points, with labels like ``1 Mo``, ``ON`` or ``10y``. :func:`clean_quotes`
turns a table of rows into decimal :class:`CurveQuote` objects and reports
every problem it finds instead of silently building a wrong curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .conventions import parse_tenor
from .curves import INSTRUMENT_KINDS, CurveQuote

RATE_RANGE = (-0.05, 0.30)  # accepted after unit conversion
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


def clean_quotes(
    rows: Iterable[Mapping[str, object]],
    units: str = "auto",
    default_kind: Optional[str] = None,
) -> Tuple[List[CurveQuote], List[Issue], str]:
    """Rows with ``kind`` (or ``instrument``), ``tenor`` and ``rate`` keys ->
    (quotes in decimals, issues, units used).

    Blank rows are skipped. Unknown kinds or tenors, non-numeric or
    out-of-range rates and duplicate tenors are errors.
    """

    issues: List[Issue] = []
    parsed = []
    for n, row in enumerate(rows, start=1):
        tenor_raw = row.get("tenor")
        rate_raw = row.get("rate")
        kind_raw = row.get("kind", row.get("instrument", default_kind))
        if all(v is None or str(v).strip() in ("", "nan", "None") for v in (tenor_raw, rate_raw)):
            continue
        kind = str(kind_raw or "").strip().upper()
        if kind not in INSTRUMENT_KINDS:
            issues.append(
                Issue("error", f"row {n}: instrument {kind_raw!r} is not one of {', '.join(INSTRUMENT_KINDS)}")
            )
            continue
        try:
            tenor = parse_tenor(str(tenor_raw))
        except ValueError as exc:
            issues.append(Issue("error", f"row {n}: {exc}"))
            continue
        try:
            rate = float(str(rate_raw).replace("%", "").strip())
        except ValueError:
            issues.append(Issue("error", f"row {n}: rate {rate_raw!r} is not a number"))
            continue
        if not np.isfinite(rate):
            issues.append(Issue("error", f"row {n}: rate is missing"))
            continue
        parsed.append((n, kind, tenor, rate))

    if units == "auto":
        units = detect_units([r for *_, r in parsed])
        if units != "decimal" and parsed:
            issues.append(Issue("info", f"Rates read as {units} and converted to decimals"))
    scale = UNIT_SCALE[units]

    quotes: List[CurveQuote] = []
    seen = {}
    for n, kind, tenor, rate in parsed:
        value = rate * scale
        if not RATE_RANGE[0] <= value <= RATE_RANGE[1]:
            issues.append(Issue("error", f"row {n}: {value:.4%} is outside {RATE_RANGE[0]:.0%} to {RATE_RANGE[1]:.0%}"))
            continue
        key = (tenor.n, tenor.unit, tenor.anchor)
        if key in seen:
            issues.append(Issue("error", f"row {n}: tenor {tenor} repeats row {seen[key]}; keep one quote per tenor"))
            continue
        seen[key] = n
        quotes.append(CurveQuote(kind, str(tenor), value))
    if not quotes and not any(i.level == "error" for i in issues):
        issues.append(Issue("error", "no quotes entered"))
    return quotes, issues, units
