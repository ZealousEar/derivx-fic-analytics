"""Curve relative value: spreads and butterflies on yields or swap rates,
their rolling statistics and mean-reversion half-lives.

Structures
    A structure is a set of yield weights ``{tenor: w}``; its value is
    ``sum_i w_i y_i`` in bp. Quoted conventions: the 2s10s spread is
    10Y - 2Y (``{"2Y": -1, "10Y": 1}``, positive when the curve slopes up);
    the 2s5s10s fly is 2 x 5Y - 2Y - 10Y (``{"2Y": -1, "5Y": 2, "10Y": -1}``,
    positive when the belly yields more than the wings: a cheap belly). A
    weight is also the DV01 ratio of the trade: being *long* the structure
    (gaining 1 per bp it rises) means holding ``-w_i`` of DV01 in leg i
    (DV01 = P&L per bp of that yield falling), e.g. a 2s10s steepener is
    long 1 of 2Y DV01 and short 1 of 10Y DV01.

Weight schemes (:func:`structure_weights`)
    * ``EQUAL``: spread -1/+1, fly -1/+2/-1 (the quoted structure). As a
      trade it is DV01-neutral, each wing holding half the body's DV01.
    * ``MATURITY``: DV01-neutral fly whose wings are weighted by their
      maturity distance from the body, so it is also neutral to a move
      linear in maturity (a pure curvature trade): the short wing gets
      -2 (T_c - T_b) / (T_c - T_a), the long wing -2 (T_b - T_a) /
      (T_c - T_a), the body +2. A spread is the same as ``EQUAL``.
    * ``PCA``: neutral to PC1 (spread) or to PC1 and PC2 (fly) of a PCA of
      the same history (:func:`derivx.rates.pca.pca_neutral_weights`); long
      leg +1 / body +2.

Rolling statistics (:func:`rolling_stats`)
    Over a trailing calendar window ending at each date (observations dated
    in (t - window, t], window = 365.25 x years days): mean, standard
    deviation (divisor n - 1), z-score = (x_t - mean) / sd and percentile =
    share of the window's observations at or below x_t, in percent.

Mean reversion (:func:`half_life`)
    OLS fit of the AR(1) ``x_{t+1} - x_t = a + b x_t + e`` on consecutive
    observations (an Ornstein-Uhlenbeck process sampled once a business
    day): kappa = -ln(1 + b) per observation and half-life = ln 2 / kappa
    business days; with b >= 0 the series does not mean-revert (half-life
    infinite).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from derivx.rates import pca as pca_mod
from derivx.rates.conventions import parse_tenor

BP = 1e4
PRESETS: Dict[str, tuple] = {
    "2s10s": ("2Y", "10Y"),
    "5s30s": ("5Y", "30Y"),
    "2s5s10s": ("2Y", "5Y", "10Y"),
    "5s10s30s": ("5Y", "10Y", "30Y"),
    "3M10Y": ("3M", "10Y"),
}
WINDOWS: Dict[str, float] = {"1Y": 1.0, "3Y": 3.0, "5Y": 5.0}
WEIGHTINGS = ("EQUAL", "MATURITY", "PCA")
MIN_WINDOW_OBS = 20
DAILY_MEDIAN_GAP = 4  # calendar days: a history whose median spacing is larger is not daily
BUSINESS_DAYS_PER_MONTH = 21


def _sorted(tenors: Sequence[str]) -> list:
    labels = [str(t).strip().upper() for t in tenors]
    if len(set(labels)) != len(labels):
        raise ValueError(f"each tenor once, got {labels}")
    return sorted(labels, key=lambda t: parse_tenor(t).years)


def structure_name(tenors: Sequence[str]) -> str:
    """``2s10s`` / ``2s5s10s`` for whole-year legs, else ``3M/10Y``."""

    legs = _sorted(tenors)
    if all(t.endswith("Y") for t in legs):
        return "".join(f"{t[:-1]}s" for t in legs)
    return "/".join(legs)


def structure_weights(
    tenors: Sequence[str], scheme: str = "EQUAL", pca: Optional[pca_mod.PCAResult] = None
) -> Dict[str, float]:
    """Yield weights of a spread (2 legs) or fly (3 legs) under ``scheme``
    (see the module docstring); ``pca`` is needed for ``PCA``."""

    legs = _sorted(tenors)
    if len(legs) not in (2, 3):
        raise ValueError("a spread has two legs and a fly three")
    scheme = str(scheme).upper()
    if scheme not in WEIGHTINGS:
        raise ValueError(f"weights must be one of {WEIGHTINGS}, got {scheme!r}")
    if scheme == "PCA":
        if pca is None:
            raise ValueError("PCA-neutral weights need a PCA of the same history")
        return pca_mod.pca_neutral_weights(pca, legs)
    if len(legs) == 2:
        return {legs[0]: -1.0, legs[1]: 1.0}
    if scheme == "EQUAL":
        return {legs[0]: -1.0, legs[1]: 2.0, legs[2]: -1.0}
    ta, tb, tc = (parse_tenor(t).years for t in legs)
    return {legs[0]: -2.0 * (tc - tb) / (tc - ta), legs[1]: 2.0, legs[2]: -2.0 * (tb - ta) / (tc - ta)}


def parse_weights(text: str) -> Dict[str, float]:
    """``"2Y:-1, 5Y:2, 10Y:-1"`` -> ``{"2Y": -1.0, "5Y": 2.0, "10Y": -1.0}``
    (a custom structure); ``ValueError`` with the bad item."""

    out: Dict[str, float] = {}
    for item in [p for p in str(text).replace(";", ",").split(",") if p.strip()]:
        tenor, sep, weight = item.partition(":")
        if not sep:
            raise ValueError(f"{item.strip()!r}: write tenor:weight, e.g. 10Y:1")
        label = str(parse_tenor(tenor.strip()))
        try:
            value = float(weight)
        except ValueError:
            raise ValueError(f"{item.strip()!r}: the weight must be a number (separate legs with commas)") from None
        if not math.isfinite(value):
            raise ValueError(f"{item.strip()!r}: the weight must be finite")
        if label in out:
            raise ValueError(f"{label} appears twice")
        out[label] = value
    if len(out) < 2:
        raise ValueError("a custom structure needs at least two tenor:weight legs")
    if all(w == 0.0 for w in out.values()):
        raise ValueError("at least one weight must be non-zero")
    return out


def structure_series(levels: pd.DataFrame, weights: Mapping[str, float], name: str = "") -> pd.Series:
    """``sum_i w_i y_i`` in bp on the dates where every leg is present
    (``levels`` in decimals, one column per tenor)."""

    missing = [t for t in weights if t not in levels.columns]
    if missing:
        raise ValueError(f"tenors {missing} are not in the history (it has {list(levels.columns)})")
    legs = levels[list(weights)].dropna(how="any").sort_index()
    values = legs.to_numpy(dtype=float) @ np.array([float(w) for w in weights.values()]) * BP
    return pd.Series(values, index=legs.index, name=name or structure_name(weights))


def _timed(series: pd.Series) -> pd.Series:
    out = series.astype(float).copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(list(series.index)))
    return out.sort_index()


def window_days(years: float) -> int:
    """Calendar days of a trailing window of ``years`` (365.25 a year)."""

    if not (math.isfinite(years) and years > 0):
        raise ValueError("the window must be a positive number of years")
    return int(round(365.25 * years))


def _window(years: float) -> str:
    return f"{window_days(years)}D"


def trailing(series: pd.Series, years: float) -> pd.Series:
    """The observations of ``series`` (dated by ``datetime.date``) in the
    trailing window (last - window_days(years), last], the window
    :func:`rolling_stats`, :func:`snapshot` and :func:`rank_structures` use."""

    if series.empty:
        return series
    start = series.index.max() - dt.timedelta(days=window_days(years))
    return series[series.index > start]


def rolling_stats(series: pd.Series, years: float, min_obs: int = MIN_WINDOW_OBS) -> pd.DataFrame:
    """Trailing ``years`` statistics at every date (NaN until ``min_obs``
    observations are in the window): value, mean, sd, z, pct, n."""

    s = _timed(series)
    roll = s.rolling(_window(years), min_periods=int(min_obs))
    mean, sd = roll.mean(), roll.std(ddof=1)
    pct = roll.apply(lambda w: 100.0 * float(np.mean(w <= w[-1])), raw=True)
    out = pd.DataFrame(
        {"value": s, "mean": mean, "sd": sd, "z": (s - mean) / sd.where(sd > 0), "pct": pct, "n": roll.count()}
    )
    out.index = [d.date() for d in out.index]
    return out


def sampling(dates: Sequence[dt.date]) -> str:
    """``daily`` if the median spacing is a few calendar days, else
    ``sparse`` (month ends, episodes)."""

    days = sorted(dates)
    if len(days) < 2:
        return "sparse"
    gaps = np.diff([d.toordinal() for d in days])
    return "daily" if float(np.median(gaps)) <= DAILY_MEDIAN_GAP else "sparse"


@dataclass(frozen=True)
class HalfLife:
    """AR(1) / OU fit of a series (see the module docstring)."""

    b: float
    kappa: float  # per business day; nan if not mean-reverting
    days: float  # half-life in business days; inf if not mean-reverting
    long_run_mean: float  # -a / b, in the series' units; nan if not mean-reverting
    n: int

    @property
    def mean_reverting(self) -> bool:
        return math.isfinite(self.days)


def half_life(series: pd.Series, max_gap_days: int = pca_mod.MAX_GAP_DAYS) -> HalfLife:
    """Mean-reversion half-life (business days) of a daily series, from the
    pairs of consecutive observations no more than ``max_gap_days`` apart."""

    s = series.dropna().sort_index()
    dates = list(s.index)
    x = s.to_numpy(dtype=float)
    gaps = np.array([(b - a).days for a, b in zip(dates[:-1], dates[1:])])
    keep = gaps <= max_gap_days
    lag, step = x[:-1][keep], np.diff(x)[keep]
    n = int(keep.sum())
    if n < MIN_WINDOW_OBS:
        raise ValueError(f"only {n} daily changes: a half-life needs at least {MIN_WINDOW_OBS}")
    design = np.column_stack([np.ones(n), lag])
    (a, b), *_ = np.linalg.lstsq(design, step, rcond=None)
    if b <= -1.0:
        raise ValueError(f"the AR(1) slope b = {b:.3f} is <= -1: the series oscillates, it is not an OU process")
    if not b < 0:
        return HalfLife(float(b), float("nan"), float("inf"), float("nan"), n)
    kappa = -math.log1p(b)
    return HalfLife(float(b), kappa, math.log(2.0) / kappa, float(-a / b), n)


def daily_changes(series: pd.Series, max_gap_days: int = pca_mod.MAX_GAP_DAYS) -> pd.Series:
    """Changes between consecutive observations no more than
    ``max_gap_days`` calendar days apart, dated by the later one."""

    s = series.dropna().sort_index()
    dates = list(s.index)
    gaps = np.array([(b - a).days for a, b in zip(dates[:-1], dates[1:])])
    return s.diff().iloc[1:][gaps <= max_gap_days]


def daily_run(series: pd.Series, max_gap_days: int = pca_mod.MAX_GAP_DAYS) -> pd.Series:
    """The most recent unbroken stretch of a series: back from its last
    observation until two consecutive ones are more than ``max_gap_days``
    calendar days apart (so month ends before a daily stretch are left out)."""

    s = series.dropna().sort_index()
    dates = list(s.index)
    i = len(dates) - 1
    while i > 0 and (dates[i] - dates[i - 1]).days <= max_gap_days:
        i -= 1
    return s.iloc[i:]


def snapshot(series: pd.Series, windows: Mapping[str, float] = WINDOWS) -> pd.DataFrame:
    """Latest value against each trailing window: one row per window with
    its first date, observations, sampling, mean, sd, z and percentile."""

    rows = []
    last = series.index.max()
    for label, years in windows.items():
        window = trailing(series, years)
        n = len(window)
        value = float(series.loc[last])
        mean = float(window.mean()) if n else float("nan")
        sd = float(window.std(ddof=1)) if n > 1 else float("nan")
        rows.append(
            {
                "window": label,
                "from": window.index.min() if n else None,
                "n": n,
                "sampling": sampling(list(window.index)),
                "value_bp": value,
                "mean_bp": mean,
                "sd_bp": sd,
                "z": (value - mean) / sd if n >= MIN_WINDOW_OBS and sd > 0 else float("nan"),
                "pct": 100.0 * float(np.mean(window.to_numpy() <= value)) if n >= MIN_WINDOW_OBS else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def long_carry(weights: Mapping[str, float], carry_bp: Mapping[str, float]) -> float:
    """Carry (bp of DV01 over the horizon) of being long a structure:
    ``-sum_i w_i c_i`` with ``c_i`` the carry of a long DV01 (long bond or
    receiver) in leg i; NaN if a leg has no carry."""

    values = [carry_bp.get(t, float("nan")) for t in weights]
    if any(not math.isfinite(v) for v in values):
        return float("nan")
    return -float(sum(w * c for w, c in zip(weights.values(), values)))


def rank_structures(
    levels: pd.DataFrame,
    structures: Mapping[str, Mapping[str, float]],
    window: str = "1Y",
    carry_bp: Optional[Mapping[str, float]] = None,
    horizon_months: Optional[int] = None,
) -> pd.DataFrame:
    """One row per structure: latest value, mean, sd, z and percentile over
    ``window`` (a key of :data:`WINDOWS`), daily-change vol over the same
    window, AR(1) half-life, and (with ``carry_bp`` per tenor and the
    horizon) the long structure's carry + roll-down and carry / vol, where
    vol = daily sd x sqrt(21 x months) business days. Each structure uses
    its own most recent daily stretch (:func:`daily_run`)."""

    if window not in WINDOWS:
        raise ValueError(f"window must be one of {list(WINDOWS)}, got {window!r}")
    rows = []
    for name, weights in structures.items():
        series = daily_run(structure_series(levels, weights, name))
        if len(series) < 2:
            continue
        stats = snapshot(series, {window: WINDOWS[window]}).iloc[0]
        recent = trailing(series, WINDOWS[window])
        changes = daily_changes(recent)
        vol = float(changes.std(ddof=1)) if len(changes) > 1 else float("nan")
        try:
            hl = half_life(recent).days
        except ValueError:
            hl = float("nan")
        row = {
            "structure": name,
            "weights": ", ".join(f"{t} {w:+.3g}" for t, w in weights.items()),
            "value_bp": stats["value_bp"],
            "mean_bp": stats["mean_bp"],
            "sd_bp": stats["sd_bp"],
            "z": stats["z"],
            "pct": stats["pct"],
            "daily_vol_bp": vol,
            "half_life_days": hl,
            "n": int(stats["n"]),
            "sampling": stats["sampling"],
        }
        if carry_bp is not None and horizon_months:
            carry = long_carry(weights, carry_bp)
            horizon_vol = vol * math.sqrt(BUSINESS_DAYS_PER_MONTH * horizon_months)
            row["carry_roll_long_bp"] = carry
            row["carry_to_vol"] = carry / horizon_vol if horizon_vol > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def swap_spreads(
    swap_levels: pd.DataFrame,
    govt_levels: pd.DataFrame,
    window: str = "1Y",
    tenors: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Swap spreads (swap rate minus government yield at the same tenor, in
    bp) on the dates both histories quote, each tenor over its own most
    recent daily stretch: latest value and the trailing ``window``
    statistics. Positive = swaps above government yields."""

    if window not in WINDOWS:
        raise ValueError(f"window must be one of {list(WINDOWS)}, got {window!r}")
    common = [t for t in (tenors or swap_levels.columns) if t in swap_levels.columns and t in govt_levels.columns]
    rows = []
    for tenor in sorted(common, key=lambda t: parse_tenor(t).years):
        pair = pd.concat([swap_levels[tenor], govt_levels[tenor]], axis=1, join="inner").dropna()
        series = daily_run(pd.Series((pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy() * BP, index=pair.index))
        if len(series) < 2:
            continue
        stats = snapshot(series, {window: WINDOWS[window]}).iloc[0].to_dict()
        rows.append({"tenor": tenor, "last_date": series.index[-1], **stats})
    columns = ["tenor", "last_date", "window", "from", "n", "sampling", "value_bp", "mean_bp", "sd_bp", "z", "pct"]
    return pd.DataFrame(rows, columns=columns)
