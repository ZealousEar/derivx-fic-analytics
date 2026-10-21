"""Realised against implied ATM normal vol of rates options (AL-20).

Implied
    The ATM (offset 0bp) normal vol of each point (option expiry x swap
    tenor) of a :class:`~derivx.validation.vol_history.VolHistory`, in bp
    per year. For caplets the tenor is the accrual period (``3M``), so the
    underlying is the ``E``-into-``3M`` forward rate; flat cap vols are not
    caplet vols (strip them first on RATES > CAP-FLOOR).

Underlying rate of a point (:func:`history_forwards`, :func:`curve_forwards`)
    * the vol history's own ``forward`` column: the point's ATM forward on
      each date. It is a constant-expiry forward (each date quotes the
      forward starting ``E`` after that date), so a daily change also rolls
      one day along the forward curve; that roll is not removed.
    * a curve history of the rates store: on each curve date, the forward
      swap rate starting ``E`` years later for ``T`` years (constant expiry
      too), ``F = (P(E) - P(E + T)) / sum_k a_k P(t_k)``, with fixed
      payments every ``1 / f`` years counted back from ``E + T`` (a short
      first period when ``T`` is not a whole number of periods; one period
      of ``T`` when ``T < 1 / f``) and accruals ``a_k`` in years of the
      tenor labels (no calendar: ``3M`` = 0.25), from the curve's discount
      factors ``P`` (:func:`log_discounts`).

Discount factors from a curve (:class:`CurveBasis`)
    * ``par`` curves (par yields or par swap rates, coupons ``f`` a year):
      a rate at a maturity below ``1 / f`` is a simple money-market rate,
      ``P(t) = 1 / (1 + r t)``; par rates are read at every ``j / f`` by
      linear interpolation in maturity (flat before the first quoted
      maturity) and bootstrapped as par instruments with coupons every
      ``1 / f``: ``P_j = (1 - r_j / f x sum_{i<j} P_i) / (1 + r_j / f)``.
      Between those nodes ``ln P`` is linear in maturity (flat forwards).
    * ``zero`` curves (continuously compounded spot rates):
      ``P(t) = exp(-z(t) t)``, ``z`` linear in maturity, flat before the
      first quoted maturity.

    A point whose swap ends after the curve's last node is not computed
    (no long-end extrapolation; the reason is listed). One whose expiry
    lies before the first quoted maturity is computed with the short end
    extrapolated flat, and its ``note`` says so.

Realised normal vol (:func:`realised_normal_vol`)
    Over a window of ``W`` business days of a calendar, in bp per year:
    ``1e4 x sqrt(days_per_year x sum_i dF_i^2 / sum_i b_i)``, summed over
    the steps between consecutive observations that lie inside the window.
    ``dF_i`` is the step's change in the rate (decimal) and ``b_i`` the
    number of business days it spans, so a step over a holiday or a
    missing day carries its own time. The mean is taken as zero (no drift
    is removed), the usual convention for realised vol and variance swaps.
    Observations dated on a non-business day of the calendar are dropped.
    Steps longer than ``max_gap_days`` calendar days are dropped too: a
    monthly stretch is not daily data. A window is scored only when its
    steps cover at least ``min_coverage`` of its ``W`` business days.

    * ``trailing`` at date ``t``: the ``W`` business days ending on ``t``
      (or on the last business day before it): what had been realised when
      the vol was quoted.
    * ``forward``: the ``W`` business days after it: what was realised
      while a ``W``-day delta-hedged straddle bought at ``t`` was held.

Comparison (:func:`realised_vs_implied`)
    Per point and date (bp per year unless stated): ``implied_bp``,
    ``trailing_bp``, ``forward_bp``, ``spread_bp = implied - trailing``,
    ``ratio = implied / trailing``, ``premium_bp = implied - forward``
    (positive: the option seller's delta-hedged gamma P&L was positive),
    and ``breakeven_daily_bp = implied / sqrt(days_per_year)``, in bp per
    business day. A delta-hedged ATM option's gamma P&L breaks even when
    the rate's RMS daily change equals this number.

    Per point, the summary gives the means, the latest date, and the share
    of dates on which implied was above trailing realised. It also scores
    implied and trailing realised as forecasts of forward realised: the
    RMSE of each (trailing realised is the benchmark),
    ``skill = 1 - RMSE(implied) / RMSE(trailing)``, and a Diebold-Mariano
    statistic on the squared-error differential with an overlap of ``W``
    observations (:func:`derivx.validation.backtest.diebold_mariano`;
    negative favours implied). Consecutive dates share most of their
    windows, so a history much shorter than a few windows says little.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from derivx.data import rates_history as rh
from derivx.rates.conventions import Calendar, parse_tenor
from derivx.rates.pca import tenor_map
from derivx.validation.backtest import diebold_mariano
from derivx.validation.vol_history import VolHistory

BP = 1e4
DAYS_PER_YEAR = 252.0
MAX_GAP_DAYS = 5  # calendar days one "daily" step may span (a long weekend plus a holiday)
MIN_COVERAGE = 0.8  # share of a window's business days its usable steps must cover
WINDOWS = (10, 21, 63, 126, 252)  # business days offered in the lab (2 weeks .. 1 year)
DIRECTIONS = ("trailing", "forward")
PRESENCE = 0.95  # a curve tenor is used when it is quoted on this share of the dates
EPS = 1e-9
CURRENCY_CALENDAR = {"EUR": "TARGET", "GBP": "LONDON", "JPY": "TOKYO", "CHF": "ZURICH", "CAD": "TORONTO",
                     "AUD": "SYDNEY", "NZD": "WELLINGTON"}  # fmt: skip
CALENDARS = ("WEEKENDS", "USGOV", "USD", "TARGET", "LONDON", "TOKYO", "ZURICH", "TORONTO", "SYDNEY", "WELLINGTON")
HISTORY_FORWARDS = "VOL_HISTORY_FORWARDS"  # the vol history's own forward column as the underlying


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RealisedSettings:
    """``window``: business days of each realised-vol window; ``calendar``:
    the business-day calendar the window and the steps are counted on
    (:class:`derivx.rates.conventions.Calendar` names; ``WEEKENDS`` = every
    weekday); ``max_gap_days``: calendar days one step may span;
    ``min_coverage``: share of a window its steps must cover;
    ``days_per_year``: business days a year the variance is annualised by."""

    window: int = 21
    calendar: str = "WEEKENDS"
    max_gap_days: int = MAX_GAP_DAYS
    min_coverage: float = MIN_COVERAGE
    days_per_year: float = DAYS_PER_YEAR

    def __post_init__(self) -> None:
        if int(self.window) < 2:
            raise ValueError(f"the realised-vol window must be at least 2 business days, got {self.window}")
        Calendar(self.calendar)  # ValueError naming the known calendars
        if int(self.max_gap_days) < 1:
            raise ValueError(f"max_gap_days must be at least 1 calendar day, got {self.max_gap_days}")
        if not (math.isfinite(self.min_coverage) and 0.0 < self.min_coverage <= 1.0):
            raise ValueError(f"min_coverage is a share of the window in (0, 1], got {self.min_coverage}")
        if not (math.isfinite(self.days_per_year) and self.days_per_year > 0):
            raise ValueError(f"days_per_year must be a positive number of business days, got {self.days_per_year}")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_calendar(currency: str, kind: str = "") -> str:
    """Business-day calendar of a currency's rates: ``USGOV`` (SIFMA) for
    US Treasuries, ``USD`` (SOFR) for other USD rates, TARGET for EUR,
    London for GBP, ...; ``WEEKENDS`` when unknown."""

    ccy = str(currency).upper()
    if ccy == "USD":
        return "USGOV" if kind == "govt" else "USD"
    return CURRENCY_CALENDAR.get(ccy, "WEEKENDS")


# --------------------------------------------------------------------------
# Realised normal vol
# --------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _holidays(calendar: str, first_year: int, last_year: int) -> np.ndarray:
    days = Calendar(calendar).holidays(dt.date(first_year, 1, 1), dt.date(last_year, 12, 31))
    return np.array(days, dtype="datetime64[D]")


def _days(values: Sequence[object]) -> np.ndarray:
    return np.array([pd.Timestamp(v).date() for v in values], dtype="datetime64[D]")


def _clean(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s.to_numpy(dtype=float))]
    s.index = [pd.Timestamp(d).date() for d in s.index]
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def realised_normal_vol(
    series: pd.Series,
    at: Sequence[dt.date],
    settings: Optional[RealisedSettings] = None,
    direction: str = "trailing",
) -> pd.DataFrame:
    """Realised normal vol of a rate (decimals, indexed by date) over the
    ``trailing`` or ``forward`` window of every date in ``at`` (module
    docs). Returns one row per date of ``at``: ``date``, ``realised_bp``
    (bp per year; NaN where the window is not covered), ``n_steps`` and
    ``coverage`` (share of the window's business days the steps span)."""

    cfg = settings or RealisedSettings()
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    quotes = list(at)
    out = pd.DataFrame(
        {"date": quotes, "realised_bp": np.nan, "n_steps": 0, "coverage": 0.0},
        columns=["date", "realised_bp", "n_steps", "coverage"],
    )
    s = _clean(series)
    if len(s) < 2 or not quotes:
        return out
    obs, q = _days(list(s.index)), _days(quotes)
    years = [int(str(d)[:4]) for d in (min(obs.min(), q.min()), max(obs.max(), q.max()))]
    hol = _holidays(cfg.calendar, years[0] - 1, years[1] + 1)
    business = np.is_busday(obs, holidays=hol)
    obs, values = obs[business], s.to_numpy(dtype=float)[business]
    if obs.size < 2:
        return out
    base = min(obs.min(), q.min())
    pos = np.busday_count(base, obs, holidays=hol)
    start, end = pos[:-1], pos[1:]
    usable = np.diff(obs).astype(int) <= int(cfg.max_gap_days)
    sq = np.concatenate([[0.0], np.cumsum(np.where(usable, np.diff(values) ** 2, 0.0))])
    span = np.concatenate([[0], np.cumsum(np.where(usable, end - start, 0))])
    count = np.concatenate([[0], np.cumsum(usable)])
    # position of the last business day on or before each quote date
    anchor = np.busday_count(base, q + np.timedelta64(1, "D"), holidays=hol) - 1
    w = int(cfg.window)
    lo, hi = (anchor - w, anchor) if direction == "trailing" else (anchor, anchor + w)
    i0 = np.searchsorted(start, lo, side="left")
    i1 = np.maximum(np.searchsorted(end, hi, side="right"), i0)
    var, days, n = sq[i1] - sq[i0], span[i1] - span[i0], count[i1] - count[i0]
    coverage = days / float(w)
    scored = (days > 0) & (coverage >= cfg.min_coverage - EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = np.where(scored, BP * np.sqrt(cfg.days_per_year * var / np.where(days > 0, days, 1)), np.nan)
    out["realised_bp"], out["n_steps"], out["coverage"] = vol, n.astype(int), coverage
    return out


# --------------------------------------------------------------------------
# Forward swap rates from a curve history
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveBasis:
    """How a curve's rates become discount factors (module docs): ``kind``
    ``par`` (``frequency`` coupons a year) or ``zero`` (continuously
    compounded spot rates); ``frequency`` is also the fixed-leg frequency
    of the forward swap rates."""

    kind: str
    frequency: int
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("par", "zero"):
            raise ValueError(f"a curve basis is 'par' or 'zero', got {self.kind!r}")
        if int(self.frequency) not in (1, 2, 4, 12):
            raise ValueError(f"payments a year must be 1, 2, 4 or 12, got {self.frequency}")


def _pillars(levels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    years = np.array([parse_tenor(str(c)).years for c in levels.columns], dtype=float)
    if years.size == 0 or np.any(years <= 0) or np.any(np.diff(years) <= 0):
        raise ValueError(
            f"curve tenors must be positive maturities, shortest first, each once; got {list(levels.columns)}"
        )
    rates = levels.to_numpy(dtype=float)
    if not np.all(np.isfinite(rates)):
        raise ValueError("the curve history has missing rates: pass complete curves (one rate per tenor and date)")
    return years, rates


def log_discounts(levels: pd.DataFrame, basis: CurveBasis, times: Sequence[float]) -> np.ndarray:
    """``ln P(t)`` on every curve date (rows of ``levels``: decimals, one
    column per tenor, shortest first) at ``times`` (years; module docs).
    ``ValueError`` for a time past the curve's last node."""

    pillars, rates = _pillars(levels)
    q = np.asarray(times, dtype=float)
    if q.ndim != 1 or np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("discount times must be finite and non-negative years")
    if basis.kind == "zero":
        last = pillars[-1]
        if q.size and q.max() > last + EPS:
            raise ValueError(f"time {q.max():.4g}y is past the curve's last maturity ({last:.4g}y)")
        z = rates @ tenor_map(q, pillars).T  # linear in maturity, flat before the first pillar
        return -z * q
    f = int(basis.frequency)
    step = 1.0 / f
    money = pillars < step - EPS
    grid = np.arange(1, int(math.floor(pillars[-1] * f + EPS)) + 1) / f
    par = rates @ tenor_map(grid, pillars).T if grid.size else np.zeros((rates.shape[0], 0))
    dfs = np.empty_like(par)
    annuity = np.zeros(rates.shape[0])
    for j in range(grid.size):
        dfs[:, j] = (1.0 - par[:, j] / f * annuity) / (1.0 + par[:, j] / f)
        annuity += dfs[:, j]
    mm = 1.0 / (1.0 + rates[:, money] * pillars[money])
    nodes = np.concatenate([[0.0], pillars[money], grid])
    both = np.concatenate([np.ones((rates.shape[0], 1)), mm, dfs], axis=1)
    if np.any(~np.isfinite(both)) or np.any(both <= 0):
        bad = levels.index[np.any(~np.isfinite(both) | (both <= 0), axis=1)][:3]
        raise ValueError(f"the par curve does not bootstrap to positive discount factors on {list(bad)}")
    if q.size and q.max() > nodes[-1] + EPS:
        raise ValueError(f"time {q.max():.4g}y is past the curve's last node ({nodes[-1]:.4g}y)")
    return np.log(both) @ tenor_map(q, nodes).T


def swap_schedule(expiry: float, tenor: float, frequency: int) -> Tuple[np.ndarray, np.ndarray]:
    """(payment times, accruals) in years of a swap starting at ``expiry``
    for ``tenor`` years with ``frequency`` fixed payments a year, counted
    back from the end (short first period; one period below ``1 /
    frequency``)."""

    if not (expiry >= 0 and tenor > 0):
        raise ValueError(f"a forward swap needs expiry >= 0 and tenor > 0 years, got {expiry} and {tenor}")
    step = 1.0 / int(frequency)
    k = np.arange(int(math.floor(tenor / step + EPS)) + 1)
    times = expiry + tenor - step * k
    times = np.sort(times[times > expiry + EPS])
    return times, np.diff(np.concatenate([[expiry], times]))


def forward_swap_rates(levels: pd.DataFrame, basis: CurveBasis, expiry: float, tenor: float) -> pd.Series:
    """The forward swap rate ``expiry`` years into ``tenor`` years (decimal)
    on every curve date (module docs)."""

    times, accruals = swap_schedule(expiry, tenor, basis.frequency)
    ln = log_discounts(levels, basis, np.concatenate([[expiry], times]))
    p = np.exp(ln)
    annuity = p[:, 1:] @ accruals
    return pd.Series((p[:, 0] - p[:, -1]) / annuity, index=levels.index, name="forward")


def _label(point: Tuple[str, str]) -> str:
    return f"{point[0]}x{point[1]}"


def curve_forwards(
    levels: pd.DataFrame, basis: CurveBasis, points: Sequence[Tuple[str, str]]
) -> Tuple[Dict[str, pd.Series], Dict[str, str], List[Dict[str, str]]]:
    """Forward swap-rate histories of the vol points from a curve history:
    ``({label: series}, {label: note}, skipped)``; a point whose swap ends
    past the curve is skipped with the reason."""

    pillars, _ = _pillars(levels)
    out: Dict[str, pd.Series] = {}
    notes: Dict[str, str] = {}
    skipped: List[Dict[str, str]] = []
    first = str(levels.columns[0])
    for point in points:
        label = _label(point)
        e, t = parse_tenor(point[0]).years, parse_tenor(point[1]).years
        try:
            out[label] = forward_swap_rates(levels, basis, e, t)
        except ValueError as exc:
            skipped.append({"point": label, "reason": f"forward swap rate not computed: {exc}"})
            continue
        if e < pillars[0] - EPS:
            notes[label] = f"expiry {point[0]} is before the curve's first maturity ({first}): short end flat"
    return out, notes, skipped


def history_forwards(history: VolHistory) -> Tuple[Dict[str, pd.Series], List[Dict[str, str]]]:
    """The vol history's own ATM forwards per point (``{label: series}``,
    decimals by date) and the points it gives none for."""

    out, skipped = {}, []
    for point in history.points:
        values = {d: history.forwards[d][point] for d in history.dates if history.has_forwards(d, point)}
        if len(values) < 2:
            skipped.append({"point": _label(point), "reason": "the vol history gives no forwards for this point"})
            continue
        out[_label(point)] = pd.Series(values, dtype=float, name="forward").sort_index()
    return out, skipped


@dataclass(frozen=True)
class StoreUnderlying:
    """A rates-store curve history ready for :func:`curve_forwards`: the
    complete curves, how to read them, what the realised vol is of
    (``what``), whether it is only a ``proxy`` for swap rates, and the
    default business-day calendar."""

    dataset: str
    measure: str
    levels: pd.DataFrame
    basis: CurveBasis
    what: str
    proxy: bool
    calendar: str


def store_underlying(store: rh.HistoryStore, dataset: str) -> StoreUnderlying:
    """How a rates-store dataset is read as the underlying of swaption
    vols: Treasury par yields (semi-annual par bootstrap; a PROXY: no swap
    spread), ICE USD LIBOR swap rates (semi-annual fixed), an uploaded swap
    curve (annual fixed, the OIS convention), the Bank of England OIS spot
    curve or the ECB AAA spot curve (a PROXY: government yields). Tenors
    quoted on fewer than 95% of the dates are left out, and so are the
    dates still missing one of the others."""

    info = rh.dataset_info(dataset)
    if info.kind == "fixings":
        raise ValueError("overnight fixings are not a curve: pick a curve history")
    if dataset == "EUR_AAA_ECB":
        measure, basis = "zero_cc", CurveBasis("zero", 1, "spot rates, cont. comp.")
        what, proxy = "PROXY: ECB AAA euro area government forward rates, not EUR swap (ESTR, EURIBOR) rates", True
    elif info.measure == "zero_cc":
        measure, basis = "zero_cc", CurveBasis("zero", 1, "spot rates, cont. comp.")
        what, proxy = f"forward OIS rates from the {info.title}", False
    elif info.measure == "par_yield":
        measure, basis = "par_yield", CurveBasis("par", 2, "semi-annual par yields, simple below 6M")
        what, proxy = f"PROXY: forward par yields from the {info.title}, not swap rates (no swap-spread vol)", True
    elif dataset == "USD_SWAP_ICE":
        measure, basis = "par_swap", CurveBasis("par", 2, "semi-annual fixed par swap rates")
        what, proxy = "forward ICE USD swap rates vs 3M LIBOR (published to 2016-10-28)", False
    elif info.measure == "par_swap":
        measure, basis = "par_swap", CurveBasis("par", 1, "annual fixed par swap rates, simple below 1Y (OIS)")
        what, proxy = f"forward swap rates from {info.title.lower()} (annual fixed assumed)", False
    else:
        raise ValueError(f"{info.title}: no rule to read {info.measure} rates as discount factors")
    wide = store.complete(dataset, measure)
    if wide.empty:
        raise ValueError(f"no {info.title} {measure.replace('_', ' ')} history in the store")
    present = [c for c in wide.columns if wide[c].notna().mean() >= PRESENCE]
    levels = wide[present].dropna(how="any")
    if levels.shape[1] < 2 or len(levels) < 2:
        raise ValueError(f"{info.title}: fewer than two complete curves on two or more tenors")
    return StoreUnderlying(dataset, measure, levels, basis, what, proxy, default_calendar(info.currency, info.kind))


# --------------------------------------------------------------------------
# Realised against implied
# --------------------------------------------------------------------------


@dataclass
class RealisedImpliedResult:
    """``daily``: one row per point and vol date; ``summary``: one row per
    point; ``skipped``: points or dates not scored, with the reason."""

    settings: RealisedSettings
    underlying: str
    daily: pd.DataFrame
    summary: pd.DataFrame
    skipped: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.daily.empty or not np.isfinite(self.daily["trailing_bp"].to_numpy(dtype=float)).any()


DAILY_COLUMNS = ["date", "point", "rate_pct", "implied_bp", "trailing_bp", "forward_bp", "spread_bp", "ratio",
                 "premium_bp", "breakeven_daily_bp", "trailing_steps", "forward_steps"]  # fmt: skip
SUMMARY_COLUMNS = ["point", "n_dates", "last_date", "implied_last_bp", "trailing_last_bp", "spread_last_bp",
                   "implied_mean_bp", "trailing_mean_bp", "spread_mean_bp", "ratio_mean", "implied_above_share",
                   "n_forward", "forward_mean_bp", "premium_mean_bp", "rmse_implied_bp", "rmse_trailing_bp",
                   "skill", "dm_t", "note"]  # fmt: skip


def implied_atm(history: VolHistory) -> pd.DataFrame:
    """ATM (offset 0) normal vol of every point on every date, bp per year:
    ``date``, ``point``, ``implied_bp``; dates without an ATM quote at a
    point are left out."""

    rows = []
    for d in history.dates:
        grid = history.grids[d]
        for point in history.points:
            offsets, vols = grid.smile(*point)
            atm = vols[offsets == 0.0]
            if atm.size:
                rows.append({"date": d, "point": _label(point), "implied_bp": float(atm[0]) * BP})
    return pd.DataFrame(rows, columns=["date", "point", "implied_bp"])


def _level_on(series: pd.Series, dates: Sequence[dt.date], max_gap_days: int) -> np.ndarray:
    """The rate (decimal) last observed on or before each date, no more
    than ``max_gap_days`` calendar days earlier; NaN otherwise."""

    if series.empty:
        return np.full(len(dates), np.nan)
    obs, q = _days(list(series.index)), _days(dates)
    i = np.searchsorted(obs, q, side="right") - 1
    values = series.to_numpy(dtype=float)[np.maximum(i, 0)]
    fresh = (i >= 0) & ((q - obs[np.maximum(i, 0)]).astype(int) <= max_gap_days)
    return np.where(fresh, values, np.nan)


def _rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e * e))) if e.size else float("nan")


def _point_summary(g: pd.DataFrame, window: int, note: str) -> Dict[str, object]:
    both = g[np.isfinite(g["trailing_bp"])]
    fwd = g[np.isfinite(g["forward_bp"]) & np.isfinite(g["trailing_bp"])]
    last = both.iloc[-1] if len(both) else None
    e_imp = (fwd["implied_bp"] - fwd["forward_bp"]).to_numpy(float)
    e_trl = (fwd["trailing_bp"] - fwd["forward_bp"]).to_numpy(float)
    r_imp, r_trl = _rmse(e_imp), _rmse(e_trl)
    return {
        "point": g["point"].iloc[0],
        "n_dates": int(len(both)),
        "last_date": last["date"] if last is not None else None,
        "implied_last_bp": float(last["implied_bp"]) if last is not None else np.nan,
        "trailing_last_bp": float(last["trailing_bp"]) if last is not None else np.nan,
        "spread_last_bp": float(last["spread_bp"]) if last is not None else np.nan,
        "implied_mean_bp": float(both["implied_bp"].mean()) if len(both) else np.nan,
        "trailing_mean_bp": float(both["trailing_bp"].mean()) if len(both) else np.nan,
        "spread_mean_bp": float(both["spread_bp"].mean()) if len(both) else np.nan,
        "ratio_mean": float(both["ratio"].mean()) if len(both) else np.nan,
        "implied_above_share": float((both["spread_bp"] > 0).mean()) if len(both) else np.nan,
        "n_forward": int(len(fwd)),
        "forward_mean_bp": float(fwd["forward_bp"].mean()) if len(fwd) else np.nan,
        "premium_mean_bp": float(fwd["premium_bp"].mean()) if len(fwd) else np.nan,
        "rmse_implied_bp": r_imp,
        "rmse_trailing_bp": r_trl,
        "skill": 1.0 - r_imp / r_trl if r_trl > 0 else np.nan,
        "dm_t": diebold_mariano(e_imp**2 - e_trl**2, horizon=window) if len(fwd) else np.nan,
        "note": note,
    }


def realised_vs_implied(
    history: VolHistory,
    underlying: Mapping[str, pd.Series],
    settings: Optional[RealisedSettings] = None,
    *,
    what: str = "",
    notes: Optional[Mapping[str, str]] = None,
    skipped: Sequence[Mapping[str, str]] = (),
) -> RealisedImpliedResult:
    """Implied ATM normal vol of every point of ``history`` against the
    realised normal vol of its underlying rate (``underlying``: point
    label ``"1Yx10Y"`` -> decimals by date; from :func:`history_forwards`
    or :func:`curve_forwards`), trailing and forward (module docs).
    ``what`` names the underlying; ``notes`` and ``skipped`` carry the
    caveats and reasons from building it."""

    cfg = settings or RealisedSettings()
    notes = dict(notes or {})
    skip = [dict(s) for s in skipped]
    implied = implied_atm(history)
    for point in history.points:
        if _label(point) not in set(implied["point"]):
            skip.append({"point": _label(point), "reason": "no ATM (0bp offset) quote on any date: nothing to compare"})
    parts = []
    for label, g in implied.groupby("point", sort=False):
        series = underlying.get(label)
        if series is None:
            if not any(s.get("point") == label for s in skip):
                skip.append({"point": label, "reason": "no underlying rate history for this point"})
            continue
        dates = list(g["date"])
        trailing = realised_normal_vol(series, dates, cfg, "trailing")
        forward = realised_normal_vol(series, dates, cfg, "forward")
        s = _clean(series)
        level = _level_on(s, dates, int(cfg.max_gap_days))
        if not np.isfinite(trailing["realised_bp"].to_numpy(float)).any():
            first, last = (s.index[0], s.index[-1]) if len(s) else (None, None)
            skip.append(
                {
                    "point": label,
                    "reason": f"the rate history ({first} .. {last}, {len(s)} observations) covers no "
                    f"{cfg.window}-business-day window ending on a vol date ({dates[0]} .. {dates[-1]}) to "
                    f"{cfg.min_coverage:.0%} with steps of at most {cfg.max_gap_days} calendar days",
                }
            )
        imp = g["implied_bp"].to_numpy(float)
        trl, fwd = trailing["realised_bp"].to_numpy(float), forward["realised_bp"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(trl > 0, imp / trl, np.nan)
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "point": label,
                    "rate_pct": level * 100.0,
                    "implied_bp": imp,
                    "trailing_bp": trl,
                    "forward_bp": fwd,
                    "spread_bp": imp - trl,
                    "ratio": ratio,
                    "premium_bp": imp - fwd,
                    "breakeven_daily_bp": imp / math.sqrt(cfg.days_per_year),
                    "trailing_steps": trailing["n_steps"].to_numpy(int),
                    "forward_steps": forward["n_steps"].to_numpy(int),
                }
            )
        )
    daily = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=DAILY_COLUMNS)
    rows = [_point_summary(g, int(cfg.window), notes.get(p, "")) for p, g in daily.groupby("point", sort=False)]
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return RealisedImpliedResult(cfg, what, daily, summary, pd.DataFrame(skip, columns=["point", "reason"]))
