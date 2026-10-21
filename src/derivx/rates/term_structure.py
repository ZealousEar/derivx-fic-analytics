from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


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


def build_zero_curve(
    quotes: Dict[str, float],
    day_count: str = "ACT/365",
    compounding: str = "cont",
    asof: Optional[dt.date] = None,
    conventions: str = "USD SOFR",
    interpolation: str = "loglinear",
):
    """Bootstrap a curve from ``{tenor: OIS par rate}`` quotes.

    Kept for the original API: every quote is treated as an OIS par rate (an
    ``ON``/``TN``/``SN`` quote as a deposit), the curve is bootstrapped so each
    one reprices, and the result is a :class:`derivx.rates.curves.Curve`
    (which also offers ``ts``, ``dfs``, ``discount``, ``df`` and ``fwd_rate``).
    Rates may be decimals, percent or bp. ``day_count`` and ``compounding``
    are ignored: the instrument conventions come from ``conventions``. For
    mixed instruments use :func:`derivx.rates.curves.bootstrap`.
    """

    from .curves import bootstrap
    from .quotes import clean_quotes

    overnight = ("ON", "O/N", "TN", "T/N", "SN", "S/N")
    rows = [
        {"kind": "DEPO" if str(t).strip().upper() in overnight else "OIS", "tenor": t, "rate": r}
        for t, r in quotes.items()
    ]
    clean, issues, _ = clean_quotes(rows)
    errors = [i.message for i in issues if i.level == "error"]
    if errors:
        raise ValueError("; ".join(errors))
    return bootstrap(asof or dt.date.today(), clean, conventions, interpolation)
