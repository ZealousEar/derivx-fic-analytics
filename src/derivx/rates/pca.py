"""Principal components of yield-curve moves: level, slope and curvature.

Input
    ``levels``: curve levels by date (an index of ``datetime.date``) and tenor
    (columns such as ``"2Y"``), as decimals (0.0425 = 4.25%), e.g. a
    :meth:`derivx.data.rates_history.HistoryStore.complete` table.

Changes (:func:`curve_changes`)
    The change of each chosen tenor between consecutive observations, dated
    by the later one; dates where a chosen tenor is missing are skipped. A
    change spanning more than ``max_gap_days`` calendar days is dropped
    (and counted), so a history that is monthly in one stretch and daily in
    another never mixes one-month moves into daily statistics.

PCA (:func:`pca_curve_changes`)
    Eigen-decomposition of the sample covariance (divisor N - 1) of the
    analysed matrix: daily changes, or levels with ``use_changes=False``;
    with ``demean=False`` the second moments about zero (divisor N). Units
    are bp (``in_bp=True``) or decimals.

    * ``loadings``: unit-length eigenvectors, one column per factor, the
      largest eigenvalue first. Signs: PC1 sums to a positive number (every
      tenor up: level up); PC2 loads more on the longest than on the shortest
      tenor (positive = steepening); PC3 loads more on the middle of the
      curve than on the average of its two ends (positive = belly up); later
      factors have their largest absolute loading positive.
    * ``eigenvalues``: variance of each factor score (units squared per
      observation); ``explained`` = share of the total variance.
    * ``scores``: ``s_t = (x_t - mean) @ loadings``, so with all factors
      ``x_t = mean + sum_k s_tk L_k`` exactly; a score of +1 moves tenor i by
      ``L_ik`` (bp when ``in_bp``).
    * :meth:`PCAResult.reconstruction_error`: RMS over dates of ``x_t`` minus
      its reconstruction from the first k factors, per tenor and overall.

Hedging and factor risk
    :func:`neutral_weights` solves leg weights on yields whose exposure to
    chosen factors is zero (PCA-neutral spreads and flies). A weight w_i on
    the yield of leg i is the P&L per bp of a position holding w_i of DV01
    (P&L per bp of yield *fall*) in that leg, so the weights are also the
    DV01 ratios to trade. :func:`factor_pv01` is the P&L of positions for the
    curve moving along each loading (a score of +1), from their key-rate
    PV01s (P&L per +1bp of each curve quote) with the loadings mapped to the
    quotes' maturities by linear interpolation in years, flat beyond the
    first and last PCA tenor: the mapping the scenario engine uses for a
    shock profile.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from derivx.rates.conventions import parse_tenor

BP = 1e4  # decimals -> bp
MIN_OBSERVATIONS = 20
DEFAULT_TENORS = ("3M", "1Y", "2Y", "5Y", "10Y", "30Y")
MAX_GAP_DAYS = 5  # calendar days a "daily" change may span (a long weekend plus a holiday)
FACTOR_LABELS = ("LEVEL", "SLOPE", "CURVATURE")


def factor_name(k: int) -> str:
    """``PC1`` for k = 0."""

    return f"PC{k + 1}"


def tenor_years(tenors: Sequence[str]) -> np.ndarray:
    """Tenor labels -> years (months / 12)."""

    return np.array([parse_tenor(str(t)).years for t in tenors], dtype=float)


def tenor_map(to_years: Sequence[float], from_years: Sequence[float]) -> np.ndarray:
    """Matrix ``W`` (len(to) x len(from)) of linear interpolation in years,
    flat beyond the first and last point: ``W @ v`` reads a profile ``v``
    given at ``from_years`` at ``to_years`` (as ``numpy.interp``)."""

    xs = np.asarray(from_years, dtype=float)
    to = np.asarray(to_years, dtype=float)
    if xs.ndim != 1 or xs.size == 0 or not np.all(np.isfinite(xs)) or not np.all(np.isfinite(to)):
        raise ValueError("tenor_map needs finite maturities")
    if np.any(np.diff(xs) <= 0):
        raise ValueError(f"the profile maturities must increase strictly, got {xs.tolist()}")
    out = np.zeros((to.size, xs.size))
    for row, y in enumerate(to):
        if y <= xs[0]:
            out[row, 0] = 1.0
        elif y >= xs[-1]:
            out[row, -1] = 1.0
        else:
            j = int(np.searchsorted(xs, y))
            w = (y - xs[j - 1]) / (xs[j] - xs[j - 1])
            out[row, j - 1], out[row, j] = 1.0 - w, w
    return out


def _select(
    levels: pd.DataFrame,
    tenors: Optional[Sequence[str]],
    start: Optional[dt.date],
    end: Optional[dt.date],
) -> pd.DataFrame:
    """The chosen tenors (shortest first) on dates in [start, end] where all
    of them are present."""

    if levels is None or levels.empty:
        raise ValueError("no curve history to analyse")
    chosen = [str(t) for t in (tenors if tenors else levels.columns)]
    missing = [t for t in chosen if t not in levels.columns]
    if missing:
        raise ValueError(f"tenors {missing} are not in the history (it has {list(levels.columns)})")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"each tenor once, got {chosen}")
    if len(chosen) < 2:
        raise ValueError("pick at least two tenors")
    if start is not None and end is not None and end <= start:
        raise ValueError(f"the end date ({end}) must be after the start date ({start})")
    chosen.sort(key=lambda t: parse_tenor(t).years)
    frame = levels[chosen].sort_index()
    if start is not None:
        frame = frame[frame.index >= start]
    if end is not None:
        frame = frame[frame.index <= end]
    return frame.dropna(how="any").astype(float)


def curve_changes(
    levels: pd.DataFrame,
    tenors: Optional[Sequence[str]] = None,
    max_gap_days: int = MAX_GAP_DAYS,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
) -> Tuple[pd.DataFrame, int]:
    """(changes, dropped): the change of each tenor between consecutive
    complete observations in [start, end] (decimals, dated by the later
    observation), leaving out the ``dropped`` changes that span more than
    ``max_gap_days`` calendar days."""

    if max_gap_days < 1:
        raise ValueError("max_gap_days must be at least 1")
    frame = _select(levels, tenors, start, end)
    if len(frame) < 2:
        return frame.iloc[:0], 0
    dates = list(frame.index)
    gaps = np.array([(b - a).days for a, b in zip(dates[:-1], dates[1:])])
    diffs = frame.diff().iloc[1:]
    keep = gaps <= max_gap_days
    return diffs[keep], int((~keep).sum())


def recent_daily_start(
    levels: pd.DataFrame,
    tenors: Optional[Sequence[str]] = None,
    max_gap_days: int = MAX_GAP_DAYS,
    end: Optional[dt.date] = None,
) -> Optional[dt.date]:
    """First date of the most recent unbroken run of observations (no two
    consecutive ones more than ``max_gap_days`` apart) up to ``end``: where
    a history that is daily only lately (month ends before) turns daily.
    None if there are fewer than two observations."""

    frame = _select(levels, tenors, None, end)
    dates = list(frame.index)
    if len(dates) < 2:
        return None
    i = len(dates) - 1
    while i > 0 and (dates[i] - dates[i - 1]).days <= max_gap_days:
        i -= 1
    return dates[i] if i < len(dates) - 1 else None


def _orient(vectors: np.ndarray) -> np.ndarray:
    """Apply the sign conventions of the module docstring to each column."""

    out = vectors.copy()
    n = out.shape[0]
    middle = out[n // 2] if n % 2 else 0.5 * (out[n // 2 - 1] + out[n // 2])
    for k in range(out.shape[1]):
        v = out[:, k]
        if k == 0:
            s = v.sum()
        elif k == 1:
            s = v[-1] - v[0]
        elif k == 2:
            s = middle[k] - 0.5 * (v[0] + v[-1])
        else:
            s = 0.0
        if abs(s) < 1e-12:
            s = v[int(np.argmax(np.abs(v)))]
        if s < 0:
            out[:, k] = -v
    return out


@dataclass(frozen=True)
class PCAResult:
    """A PCA of curve moves (see the module docstring for conventions)."""

    tenors: Tuple[str, ...]
    years: np.ndarray
    loadings: np.ndarray  # tenors x factors, unit columns
    eigenvalues: np.ndarray  # variance of each score, units^2 per observation
    explained: np.ndarray  # share of total variance
    scores: pd.DataFrame  # date x PC1..PCn
    mean: np.ndarray  # removed per tenor (zeros if not demeaned)
    data: pd.DataFrame  # the analysed matrix, in ``unit``
    unit: str  # "bp" or "decimal"
    use_changes: bool
    demeaned: bool
    dropped: int  # changes left out for spanning more than max_gap_days

    @property
    def n_obs(self) -> int:
        return len(self.data)

    @property
    def start(self) -> dt.date:
        return self.data.index[0]

    @property
    def end(self) -> dt.date:
        return self.data.index[-1]

    @property
    def sigma(self) -> np.ndarray:
        """Standard deviation of each factor score, in ``unit``."""

        return np.sqrt(self.eigenvalues)

    @property
    def sigma_bp(self) -> np.ndarray:
        """Standard deviation of each factor score in bp."""

        return self.sigma * (1.0 if self.unit == "bp" else BP)

    def loadings_frame(self, n_factors: int = 3) -> pd.DataFrame:
        k = min(n_factors, self.loadings.shape[1])
        return pd.DataFrame(
            self.loadings[:, :k], index=pd.Index(self.tenors, name="tenor"), columns=[factor_name(i) for i in range(k)]
        )

    def reconstruction(self, n_factors: int) -> pd.DataFrame:
        """The analysed matrix rebuilt from the first ``n_factors`` factors."""

        k = self._k(n_factors)
        values = self.mean + self.scores.to_numpy()[:, :k] @ self.loadings[:, :k].T
        return pd.DataFrame(values, index=self.data.index, columns=self.data.columns)

    def reconstruction_error(self, n_factors: int) -> pd.Series:
        """RMS over dates of data minus its ``n_factors``-factor
        reconstruction, per tenor and overall (``ALL``), in ``unit``."""

        err = self.data.to_numpy() - self.reconstruction(n_factors).to_numpy()
        rms = np.sqrt(np.mean(err * err, axis=0))
        out = pd.Series(rms, index=list(self.tenors), dtype=float)
        out["ALL"] = float(np.sqrt(np.mean(err * err)))
        return out

    def factor_profile(self, factor: int, n_sigma: float = 1.0, days: int = 1) -> Tuple[Tuple[float, float], ...]:
        """((years, bp), ...): ``n_sigma`` standard deviations of factor
        ``factor`` (0 = PC1) at the PCA tenors, scaled by sqrt(``days``)
        (independent daily moves assumed)."""

        k = self._k(factor + 1) - 1
        if not math.isfinite(n_sigma):
            raise ValueError("the number of standard deviations must be finite")
        if days < 1 or int(days) != days:
            raise ValueError("days must be a whole number >= 1")
        scale = n_sigma * self.sigma_bp[k] * math.sqrt(days)
        return tuple((float(y), float(scale * w)) for y, w in zip(self.years, self.loadings[:, k]))

    def _k(self, n_factors: int) -> int:
        n = self.loadings.shape[1]
        if not 1 <= int(n_factors) <= n:
            raise ValueError(f"pick 1 to {n} factors, got {n_factors}")
        return int(n_factors)


def pca_curve_changes(
    levels: pd.DataFrame,
    tenors: Optional[Sequence[str]] = DEFAULT_TENORS,
    window: Optional[int] = None,
    demean: bool = True,
    in_bp: bool = True,
    use_changes: bool = True,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    max_gap_days: int = MAX_GAP_DAYS,
) -> PCAResult:
    """PCA of a curve's moves (see the module docstring).

    ``window`` keeps the last ``window`` observations of the analysed matrix
    (changes or levels) in [start, end]; None keeps them all. ``ValueError``
    when fewer than max(20, tenors + 1) observations remain.
    """

    if use_changes:
        data, dropped = curve_changes(levels, tenors, max_gap_days, start, end)
    else:
        data, dropped = _select(levels, tenors, start, end), 0
    if window is not None:
        if int(window) != window or window < MIN_OBSERVATIONS:
            raise ValueError(f"the window must be a whole number of at least {MIN_OBSERVATIONS} observations")
        data = data.iloc[-int(window) :]
    n, m = data.shape
    need = max(MIN_OBSERVATIONS, m + 1)
    if n < need:
        what = "daily changes" if use_changes else "curves"
        raise ValueError(
            f"only {n} {what} of {', '.join(data.columns)} in the chosen dates (need {need}); widen the dates or "
            "pick a history with daily data"
        )
    scale = BP if in_bp else 1.0
    x = data.to_numpy(dtype=float) * scale
    mean = x.mean(axis=0) if demean else np.zeros(m)
    centred = x - mean
    cov = centred.T @ centred / ((n - 1) if demean else n)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    tiny = 1e-12 * max(float(eigenvalues[0]), 1e-300)
    if np.any(eigenvalues < -tiny):
        raise ValueError("the covariance matrix is not positive semi-definite")
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)  # rounding of exactly singular directions only
    total = float(eigenvalues.sum())
    if not total > 0:
        raise ValueError("the curve does not move over the chosen dates")
    loadings = _orient(eigenvectors)
    scores = pd.DataFrame(centred @ loadings, index=data.index, columns=[factor_name(k) for k in range(m)])
    analysed = pd.DataFrame(x, index=data.index, columns=list(data.columns))
    return PCAResult(
        tenors=tuple(data.columns),
        years=tenor_years(data.columns),
        loadings=loadings,
        eigenvalues=eigenvalues,
        explained=eigenvalues / total,
        scores=scores,
        mean=mean,
        data=analysed,
        unit="bp" if in_bp else "decimal",
        use_changes=use_changes,
        demeaned=demean,
        dropped=dropped,
    )


@dataclass(frozen=True)
class RollingPCA:
    """Loadings and explained variance of the first factors on rolling
    windows: ``loadings`` has one row per window end and (factor, tenor)
    columns; ``stability`` is |cosine| between each window's loading and
    the whole period's (1 = same shape)."""

    loadings: pd.DataFrame
    explained: pd.DataFrame
    stability: pd.DataFrame
    window: int
    tenors: Tuple[str, ...]


def rolling_loadings(
    levels: pd.DataFrame,
    tenors: Optional[Sequence[str]] = DEFAULT_TENORS,
    window: int = 252,
    step: int = 21,
    n_factors: int = 3,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    max_gap_days: int = MAX_GAP_DAYS,
) -> RollingPCA:
    """PCA (demeaned daily changes in bp) on windows of ``window`` changes
    ending every ``step`` changes, the last window ending on the last change."""

    if int(step) != step or step < 1:
        raise ValueError("the step must be a whole number of observations >= 1")
    full = pca_curve_changes(levels, tenors, None, True, True, True, start, end, max_gap_days)
    k = min(int(n_factors), len(full.tenors))
    if int(window) != window or window < max(MIN_OBSERVATIONS, len(full.tenors) + 1):
        raise ValueError(f"the rolling window must be at least {max(MIN_OBSERVATIONS, len(full.tenors) + 1)} changes")
    if window > full.n_obs:
        raise ValueError(f"the rolling window ({window}) is longer than the {full.n_obs} changes in the chosen dates")
    x = full.data.to_numpy()
    ends = list(range(full.n_obs - 1, int(window) - 2, -int(step)))[::-1]
    rows, explained, stability, index = [], [], [], []
    for e in ends:
        block = x[e - int(window) + 1 : e + 1]
        centred = block - block.mean(axis=0)
        values, vectors = np.linalg.eigh(centred.T @ centred / (len(block) - 1))
        order = np.argsort(values)[::-1]
        values, vectors = values[order], _orient(vectors[:, order])
        rows.append(vectors[:, :k].T.ravel())
        explained.append(values[:k] / values.sum())
        stability.append(np.abs(np.sum(vectors[:, :k] * full.loadings[:, :k], axis=0)))
        index.append(full.data.index[e])
    names = [factor_name(i) for i in range(k)]
    columns = pd.MultiIndex.from_product([names, list(full.tenors)], names=["factor", "tenor"])
    idx = pd.Index(index, name="window_end")
    return RollingPCA(
        loadings=pd.DataFrame(rows, index=idx, columns=columns),
        explained=pd.DataFrame(explained, index=idx, columns=names),
        stability=pd.DataFrame(stability, index=idx, columns=names),
        window=int(window),
        tenors=full.tenors,
    )


def loadings_at(result: PCAResult, tenors: Sequence[str]) -> np.ndarray:
    """Loadings (len(tenors) x factors) at any tenors, interpolated linearly
    in maturity between the PCA tenors (flat beyond)."""

    return tenor_map(tenor_years(tenors), result.years) @ result.loadings


def neutral_weights(
    loadings: np.ndarray, fixed: int, fixed_weight: float = 1.0, factors: Sequence[int] = (0,)
) -> np.ndarray:
    """Weights of legs (rows of ``loadings``, legs x factors) with
    ``weights[fixed] = fixed_weight`` and zero exposure ``sum_i w_i L_ik``
    to each factor k in ``factors``: one free leg per factor."""

    loadings = np.asarray(loadings, dtype=float)
    n_legs = loadings.shape[0]
    factors = [int(k) for k in factors]
    if not 0 <= fixed < n_legs:
        raise ValueError(f"the fixed leg must be one of the {n_legs} legs")
    if any(not 0 <= k < loadings.shape[1] for k in factors) or len(set(factors)) != len(factors):
        raise ValueError(f"factors must be distinct, between 0 and {loadings.shape[1] - 1}")
    free = [i for i in range(n_legs) if i != fixed]
    if len(free) != len(factors):
        raise ValueError(f"{n_legs} legs neutralise exactly {n_legs - 1} factors, not {len(factors)}")
    a = loadings[np.ix_(free, factors)].T
    b = -float(fixed_weight) * loadings[fixed, factors]
    if np.linalg.cond(a) > 1e10:
        raise ValueError("these legs cannot be made neutral to those factors (their loadings are proportional)")
    weights = np.zeros(n_legs)
    weights[fixed] = float(fixed_weight)
    weights[free] = np.linalg.solve(a, b)
    return weights


def pca_neutral_weights(
    result: PCAResult,
    legs: Sequence[str],
    fixed: Optional[str] = None,
    fixed_weight: Optional[float] = None,
) -> Dict[str, float]:
    """PCA-neutral weights on yields for a spread (2 legs: neutral to PC1)
    or a fly (3 legs: neutral to PC1 and PC2). ``fixed`` defaults to the
    longest leg of a spread (weight +1) and the body of a fly (weight +2),
    matching the quoted 2s10s (10Y - 2Y) and 2s5s10s (2 x 5Y - 2Y - 10Y)."""

    legs = sorted([str(t) for t in legs], key=lambda t: parse_tenor(t).years)
    if len(legs) not in (2, 3) or len(set(legs)) != len(legs):
        raise ValueError("a PCA-neutral structure has two (spread) or three (fly) distinct legs")
    if fixed is None:
        fixed = legs[-1] if len(legs) == 2 else legs[1]
    if fixed not in legs:
        raise ValueError(f"the fixed leg {fixed!r} is not one of {legs}")
    if fixed_weight is None:
        fixed_weight = 1.0 if len(legs) == 2 else 2.0
    weights = neutral_weights(loadings_at(result, legs), legs.index(fixed), fixed_weight, range(len(legs) - 1))
    return {t: float(w) for t, w in zip(legs, weights)}


def factor_exposure(result: PCAResult, weights: Dict[str, float], n_factors: int = 3) -> np.ndarray:
    """Exposure ``sum_i w_i L_ik`` of yield weights to the first factors
    (bp of the structure per +1 score)."""

    tenors = list(weights)
    k = min(n_factors, result.loadings.shape[1])
    return np.array([weights[t] for t in tenors]) @ loadings_at(result, tenors)[:, :k]


def factor_pv01(pv01: np.ndarray, quote_years: Sequence[float], result: PCAResult, n_factors: int = 3) -> np.ndarray:
    """P&L for the curve moving along each of the first ``n_factors``
    loadings by a score of +1 (tenor i up ``L_ik`` bp), from key-rate PV01s
    ``pv01`` (quotes, or quotes x positions; P&L per +1bp of each quote) at
    ``quote_years``: factors, or factors x positions."""

    pv01 = np.asarray(pv01, dtype=float)
    years = np.asarray(quote_years, dtype=float)
    if pv01.shape[0] != years.size:
        raise ValueError(f"{pv01.shape[0]} PV01s for {years.size} quote maturities")
    k = min(int(n_factors), result.loadings.shape[1])
    moves = tenor_map(years, result.years) @ result.loadings[:, :k]  # quote bp per +1 score
    return moves.T @ pv01


def explained_table(result: PCAResult) -> pd.DataFrame:
    """One row per factor: eigenvalue, sd (bp), explained and cumulative share
    (percent)."""

    k = len(result.eigenvalues)
    return pd.DataFrame(
        {
            "factor": [factor_name(i) for i in range(k)],
            "name": [FACTOR_LABELS[i] if i < len(FACTOR_LABELS) else "" for i in range(k)],
            f"variance ({result.unit}^2)": result.eigenvalues,
            "sd_bp": result.sigma_bp,
            "explained_pct": 100.0 * result.explained,
            "cumulative_pct": 100.0 * np.cumsum(result.explained),
        }
    )


def summary_rows(result: PCAResult, n_factors: int = 3) -> List[Dict[str, object]]:
    """Loadings by tenor with each tenor's reconstruction error (bp) from
    the first ``n_factors`` factors."""

    k = min(n_factors, result.loadings.shape[1])
    err = result.reconstruction_error(k)
    rows = []
    for i, tenor in enumerate(result.tenors):
        row: Dict[str, object] = {"tenor": tenor, "years": float(result.years[i])}
        row.update({factor_name(j): float(result.loadings[i, j]) for j in range(k)})
        row[f"rms_error_{k}f_{result.unit}"] = float(err[tenor])
        rows.append(row)
    return rows
