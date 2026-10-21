"""The original curve API: :func:`build_zero_curve` and the plain
:class:`ZeroCurve` container.

:func:`build_zero_curve` now bootstraps with :func:`derivx.rates.curves.bootstrap`
(every quote reprices) and returns a :class:`derivx.rates.curves.Curve`, which
keeps the old ``ts``/``dfs``/``discount``/``df``/``fwd_rate`` interface.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .curves import Curve, CurveQuote


@dataclass
class ZeroCurve:
    ts: np.ndarray  # times in years
    dfs: np.ndarray  # discount factors

    def discount(self, t: float) -> float:
        t = max(float(t), 0.0)
        return float(np.interp(t, self.ts, self.dfs))

    def df(self, ts: Iterable[float]) -> np.ndarray:
        ts = np.asarray(list(ts), dtype=float)
        return np.interp(ts, self.ts, self.dfs)

    def fwd_rate(self, t1: float, t2: float) -> float:
        t1, t2 = float(t1), float(t2)
        if t2 <= t1:
            return 0.0
        df1 = self.discount(t1)
        df2 = self.discount(t2)
        return float(-np.log(df2 / df1) / (t2 - t1))


CURVE_TIME_DAY_COUNTS = ("ACT/365", "ACT/365F")  # the built curve's time axis
CURVE_COMPOUNDING = ("cont", "continuous")  # what its zero rates are by default
READABLE_COMPOUNDING = ("simple", "annual", "semiannual", "quarterly")  # via Curve.zero_rate
_OVERNIGHT = ("ON", "O/N", "TN", "T/N", "SN", "S/N")
QuoteInput = Union[Mapping[str, float], Iterable[Union["CurveQuote", Mapping[str, object]]]]


def _check_curve_conventions(day_count: Optional[str], compounding: Optional[str]) -> None:
    """``day_count``/``compounding`` describe the curve that is built, not the
    quotes: accept the curve's own (ACT/365F time, continuous zero rates) and
    refuse anything else instead of ignoring it."""

    from .conventions import DAY_COUNTS

    if day_count is not None and str(day_count).strip().upper() not in CURVE_TIME_DAY_COUNTS:
        what = "cannot be applied" if str(day_count).strip().upper() in DAY_COUNTS else "is not a day count"
        raise ValueError(
            f"day_count {day_count!r} {what}: the curve's time axis is ACT/365F (pass 'ACT/365F' or leave it out) "
            f"and each instrument accrues on its conventions preset (USD SOFR OIS and deposits: ACT/360). For other "
            f"instrument day counts bootstrap with derivx.rates.curves.bootstrap and your own CurveConventions "
            f"(day counts: {', '.join(DAY_COUNTS)})."
        )
    if compounding is not None and str(compounding).strip().lower() not in CURVE_COMPOUNDING:
        what = "cannot be applied" if str(compounding).strip().lower() in READABLE_COMPOUNDING else "is unknown"
        raise ValueError(
            f"compounding {compounding!r} {what}: the curve stores discount factors and its zero rates are "
            f"continuously compounded (pass 'cont' or leave it out); read zero rates in another compounding with "
            f"curve.zero_rate(t, compounding=...), one of {', '.join(READABLE_COMPOUNDING)}."
        )


def _typed_quotes(quotes: object) -> Union[Dict[str, float], List[object]]:
    """``quotes`` as a ``{tenor: rate}`` dict or a list of CurveQuote / row
    mappings; ``ValueError`` for anything else."""

    if isinstance(quotes, MappingABC):
        return dict(quotes)
    if isinstance(quotes, (str, bytes)) or not isinstance(quotes, IterableABC):
        raise ValueError(
            "quotes must be a {tenor: rate} mapping, CurveQuote objects (e.g. from "
            "derivx.data.load_rates_quotes_from_csv) or rows with instrument, tenor and rate keys"
        )
    return list(quotes)


def build_zero_curve(
    quotes: QuoteInput,
    day_count: Optional[str] = "ACT/365",
    compounding: Optional[str] = "cont",
    asof: Optional[dt.date] = None,
    conventions: str = "USD SOFR",
    interpolation: str = "loglinear",
    units: str = "auto",
) -> "Curve":
    """Bootstrap a curve on which every quote reprices: a
    :class:`derivx.rates.curves.Curve` (which also offers the original
    ``ts``, ``dfs``, ``discount``, ``df`` and ``fwd_rate``).

    ``quotes`` is one of:

    * a ``{tenor: rate}`` mapping (the original API, which has no instrument
      column): every quote is an OIS par rate, an ``ON``/``TN``/``SN`` quote
      a deposit;
    * quotes that name their instrument, which is honoured:
      :class:`~derivx.rates.curves.CurveQuote` objects (what
      :func:`derivx.data.ingestion.load_rates_quotes_from_csv` returns; rates
      in decimals, futures as futures rates) or rows with ``instrument`` (or
      ``kind``), ``tenor`` and ``rate`` keys. ``IRS`` rows are single-curve
      fixed/float swaps, ``DEPO`` deposits, ``SR3`` futures prices, and so on
      (:data:`derivx.rates.curves.INSTRUMENT_KINDS`); an unknown instrument
      raises.

    Every instrument is priced on the ``conventions`` preset (``USD SOFR``:
    T+2, OIS annual ACT/360; IRS 6M 30/360 fixed vs 3M ACT/360 float).
    ``units`` is the unit of the rates of a mapping or of rows: ``decimal``,
    ``percent``, ``bp`` or ``auto`` (guessed from their size, and refused
    when every rate lies between 0.05 and 0.5, which reads as 5-50% or as
    0.05-0.5%). CurveQuote rates are decimals by definition, so ``units``
    must then be ``auto`` or ``decimal``.

    ``day_count`` and ``compounding`` describe the curve that comes out, not
    the quotes: its time axis is ACT/365F years and its zero rates are
    continuously compounded. Those are the only values accepted (``None``
    also works); anything else raises ``ValueError`` (they used to be
    accepted and ignored). ``asof`` defaults to today.
    """

    from .curves import INSTRUMENT_KINDS, CurveQuote, bootstrap
    from .quotes import UNIT_SCALE, clean_quotes

    _check_curve_conventions(day_count, compounding)
    if units != "auto" and units not in UNIT_SCALE:
        raise ValueError(f"units must be 'auto' or one of {sorted(UNIT_SCALE)}, got {units!r}")
    given = _typed_quotes(quotes)
    asof = asof or dt.date.today()
    if isinstance(given, dict):
        rows: List[Mapping[str, object]] = [
            {"kind": "DEPO" if str(t).strip().upper() in _OVERNIGHT else "OIS", "tenor": t, "rate": r}
            for t, r in given.items()
        ]
    elif given and all(isinstance(q, CurveQuote) for q in given):
        typed = [q for q in given if isinstance(q, CurveQuote)]
        if units not in ("auto", "decimal"):
            raise ValueError(f"CurveQuote rates are decimals; units={units!r} applies to a mapping or rows only")
        unknown = sorted({q.kind for q in typed if q.kind not in INSTRUMENT_KINDS})
        if unknown:
            raise ValueError(f"instruments {unknown} cannot build this curve; expected one of {INSTRUMENT_KINDS}")
        return bootstrap(asof, typed, conventions, interpolation)
    elif all(isinstance(q, MappingABC) for q in given):
        rows = [q for q in given if isinstance(q, MappingABC)]
    else:
        raise ValueError("give CurveQuote objects or rows with instrument, tenor and rate keys, not a mix")
    clean, issues, _ = clean_quotes(rows, units=units, asof=asof)
    errors = [i.message for i in issues if i.level == "error"]
    if errors:
        hint = " (pass units='decimal', 'percent' or 'bp')" if units == "auto" and not clean else ""
        raise ValueError("; ".join(errors) + hint)
    return bootstrap(asof, clean, conventions, interpolation)
