from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator


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
        t1 = float(t1); t2 = float(t2)
        if t2 <= t1:
            return 0.0
        df1 = self.discount(t1)
        df2 = self.discount(t2)
        return float(-np.log(df2 / df1) / (t2 - t1))


def build_zero_curve(
    quotes: Dict[str, float],
    day_count: str = "ACT/365",
    compounding: str = "cont",
) -> ZeroCurve:
    """Build a simple zero curve with PCHIP ensuring monotone dfs.

    quotes: mapping tenor->rate (years as floats preferred; supports '1W','1M','2Y',...)
    rates are decimals (0.05 = 5%).
    """

    def tenor_to_years(tenor: str) -> float:
        tenor = tenor.upper().strip()
        if tenor.endswith("W"):
            return float(7 * int(tenor[:-1])) / 365.0
        if tenor.endswith("M"):
            return float(30 * int(tenor[:-1])) / 365.0
        if tenor.endswith("Y"):
            return float(int(tenor[:-1]))
        return float(tenor)

    items = sorted(((tenor_to_years(k), v) for k, v in quotes.items()), key=lambda x: x[0])
    ts = np.array([t for t, _ in items], dtype=float)
    rs = np.array([r for _, r in items], dtype=float)

    # Continuous compounding => df = exp(-r t)
    dfs_raw = np.exp(-rs * ts)
    # Enforce monotonicity via PCHIP smoothing
    interp = PchipInterpolator(ts, dfs_raw, extrapolate=True)
    ts_dense = np.linspace(0.0, float(ts[-1]), num=max(5, len(ts)))
    dfs_dense = interp(ts_dense)
    dfs_dense = np.clip(dfs_dense, 1e-8, 1.0)
    return ZeroCurve(ts=ts_dense, dfs=dfs_dense)


