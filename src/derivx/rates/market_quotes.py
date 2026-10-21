"""Turn published curves and fixings into quotes the bootstrapper takes.

* :func:`treasury_quotes` - Treasury par yields by tenor -> BILL quotes (1Y
  and shorter, bond-equivalent yields) and BOND quotes (par coupons).
* :func:`zero_rate_curve` and :func:`ois_par_quotes` - a fitted zero curve
  (the Bank of England's OIS spot curve: continuously compounded rates at
  maturities in years) -> par OIS rates under a currency's OIS conventions,
  priced with the same :class:`~derivx.rates.curves.Instrument` the
  bootstrapper uses, so bootstrapping the result gives back the published
  curve at the quote pillars.
* :func:`fixing_quote` - an overnight fixing (SOFR, ESTR, SONIA) -> an ON
  deposit quote.

All rates are decimals (0.0425 = 4.25%).
"""

from __future__ import annotations

import datetime as dt
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .conventions import parse_tenor
from .curves import Curve, CurveConventions, CurveQuote, Instrument, get_conventions

OIS_PAR_TENORS = (
    "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y",
    "20Y", "25Y",
)  # fmt: skip
MAX_EXTRAPOLATION_YEARS = 0.05  # a par swap may pay a few days past the last published maturity


def treasury_quotes(par_yields: Mapping[str, float]) -> List[CurveQuote]:
    """Treasury par yields ``{tenor: decimal}`` -> curve quotes: tenors of
    one year or less are bills (bond-equivalent yield), longer ones par
    coupon bonds. Missing (NaN) values are skipped."""

    out = []
    for tenor, value in par_yields.items():
        if value is None or not np.isfinite(value):
            continue
        parsed = parse_tenor(str(tenor))
        kind = "BOND" if parsed.years > 1.0 + 1e-9 else "BILL"
        out.append(CurveQuote(kind, str(parsed), float(value)))
    return out


def zero_rate_curve(
    asof: dt.date, years: Sequence[float], zero_cc: Sequence[float], interpolation: str = "linear_zero"
) -> Curve:
    """Discount curve through continuously compounded zero rates.

    ``years`` are maturities from ``asof``, snapped to whole calendar days
    (curve time is ACT/365F); the zero rate is linear in time between them,
    flat before the first and with a flat forward beyond the last. Raises
    ``ValueError`` on mismatched, unsorted, non-finite or duplicate inputs.
    """

    t = np.asarray(years, dtype=float)
    z = np.asarray(zero_cc, dtype=float)
    if t.shape != z.shape or t.ndim != 1 or not t.size:
        raise ValueError("years and zero rates must be equal-length, non-empty sequences")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(z))):
        raise ValueError("years and zero rates must be finite")
    days = np.round(t * 365.0).astype(int)
    if np.any(days <= 0) or np.any(np.diff(days) <= 0):
        raise ValueError("maturities must be positive and increasing by at least one day")
    times = days / 365.0
    dfs = np.exp(-z * times)
    return Curve(asof, np.concatenate([[0.0], times]), np.concatenate([[1.0], dfs]), interpolation)


def ois_par_quotes(
    curve: Curve,
    conventions: Union[str, CurveConventions],
    tenors: Sequence[str] = OIS_PAR_TENORS,
    max_extrapolation: float = MAX_EXTRAPOLATION_YEARS,
) -> List[CurveQuote]:
    """Par OIS rates implied by ``curve`` for each tenor, under the OIS
    conventions of ``conventions`` (spot lag, calendar, annual fixed leg,
    day count, payment lag). Tenors whose last payment falls more than
    ``max_extrapolation`` years past the curve's last pillar are left out
    rather than extrapolated."""

    conv = get_conventions(conventions)
    last = float(curve.ts[-1])
    out = []
    for tenor in tenors:
        inst = Instrument(CurveQuote("OIS", tenor, 0.0), curve.asof, conv)
        if inst._t_pillar > last + max_extrapolation:
            continue
        out.append(CurveQuote("OIS", str(inst.tenor), float(inst.implied(curve))))
    return out


def fixing_quote(rate: float) -> CurveQuote:
    """An overnight fixing as an ON deposit quote (today to the next
    business day, at the curve's deposit day count)."""

    if not np.isfinite(rate):
        raise ValueError("the fixing must be a finite number")
    return CurveQuote("DEPO", "ON", float(rate))


def par_from_zero_rows(
    asof: dt.date,
    rows: Sequence[Tuple[float, float]],
    conventions: Union[str, CurveConventions],
    tenors: Sequence[str] = OIS_PAR_TENORS,
) -> List[CurveQuote]:
    """``[(years, zero_cc), ...]`` on ``asof`` -> par OIS quotes (see
    :func:`zero_rate_curve` and :func:`ois_par_quotes`)."""

    pairs = sorted((float(y), float(z)) for y, z in rows)
    curve = zero_rate_curve(asof, [p[0] for p in pairs], [p[1] for p in pairs])
    return ois_par_quotes(curve, conventions, tenors)


def tenor_label(tenor: str) -> Optional[str]:
    """Canonical tenor label, or None if it does not parse."""

    try:
        return str(parse_tenor(tenor))
    except ValueError:
        return None
