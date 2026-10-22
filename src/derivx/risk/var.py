"""Historical-simulation and delta-normal value at risk of a rates book.

Scenarios (:func:`build_scenarios`)
    One scenario per historical day: every curve of the market moves its
    quotes by that day's change of the curve's history, in bp at the
    history's tenors, linear in maturity in between and flat beyond (as a
    shock profile; ``BASIS`` spread quotes are held), applied to today's
    curves. The scenario dates are the last ``lookback`` daily changes of the
    US Treasury par curve (FRED H.15 / US Treasury) up to the valuation date,
    within its most recent unbroken daily stretch: a gap of more than 5
    calendar days ends the window, so month-end data or an older episode
    window are never mixed in, and a valuation date early in the daily
    history gets fewer scenarios than asked (``VarScenarios.lookback``
    keeps the number asked for). Another history's change on a
    scenario date runs from its last observation on or before the previous
    scenario date to its last observation on or before that date (as-of
    alignment: a holiday in one market neither drops nor doubles a move); an
    observation more than ``max_stale_days`` older than the date it stands
    for counts as missing, and a missing day moves that curve by 0.

Which history moves which curve (``VarScenarios.sources``)
    The Treasury curve: US Treasury par yields (``real``). FX spots: the ECB
    reference rates (``FX_ECB``), when they move on at least
    ``min_coverage`` of the scenario dates. The swap curve of
    currency c: the user's uploaded swap history (``upload``), else the
    currency's real swap/OIS history (ICE USD swap rates, to 2016-10-28;
    the Bank of England's SONIA OIS curve as derived par OIS rates), else a
    flagged ``proxy`` (government yields of the same currency: swap spreads
    held) or ``cross-currency proxy``; the first that has a move on at least
    ``min_coverage`` of the scenario dates. Projection curves move with their
    currency's swap curve (IBOR-OIS and tenor basis held). Vols are held:
    there is no free vol history.

P&L of each position in each scenario
    Value after the move minus value today, signed to the holder, over one
    day with no carry:

    * :func:`sensitivity_pnl`: ``sum_i PV01_i b_i + 1/2 Gamma_ii b_i^2`` with
      the key-rate PV01s and diagonal gammas of
      :func:`derivx.risk.scenarios.quote_risk` (P&L per +1bp of each quote);
    * :func:`full_revaluation_pnl`: curves rebuilt from the moved quotes and
      every position revalued.

Currencies and FX
    Each position's P&L is in its own currency (swaps and options in the
    swap curve's, Treasuries in USD) and does not move with FX.
    :func:`var_table` with ``currencies`` gives the VaR of each currency's
    positions together (``TOTAL USD``, ``TOTAL EUR``; ``TOTAL`` for a book in
    one currency) and, with a ``reporting`` currency R, of the whole book
    converted (``ALL IN <R>``, :mod:`derivx.risk.reporting`).

    FX is a risk factor of the converted book: the ECB euro reference rates
    of the store (``FX_ECB``, :meth:`HistoryStore.fx_table`) are aligned to
    the same scenario dates as the curves (as-of, the same
    ``max_stale_days``; :class:`FxMoves`). On scenario day i the value in R
    of one unit of currency c moves by ``x_ic = r_iR / r_ic - 1``, with
    ``r_ic`` the ratio of the day's ECB rate of c (units per EUR) to the
    previous scenario day's (``r_EUR = 1``), applied to today's spot S_c.
    The converted P&L of position j (today's value V_j, local P&L p_ij) is
    exact, ``(V_j + p_ij) S_c (1 + x_ic) - V_j S_c``::

        ALL IN R             = sum_j p_ij S_c + (V_j + p_ij) S_c x_ic
        ALL IN R (FX HELD)   = sum_j p_ij S_c              (spots held)
        ALL IN R (FX ONLY)   = sum_j V_j S_c x_ic          (curves held)

    so the FX contribution is visible on its own (the cross term p_ij S_c
    x_ic is in ALL IN R only). The delta-normal check of ALL IN R uses the
    first-order P&L ``sum_j PV01 b S_c + V_j S_c x_ic``: FX returns are risk
    factors with sensitivity V_j S_c. ``fx=None`` (or ``build_scenarios(...,
    fx=False)``) holds the spots, as before.

Risk measures (:func:`var_es`, :func:`delta_normal`), losses as positive
amounts in the positions' currency
    * VaR at confidence a: the k-th largest loss of the N scenarios,
      k = ceil(N (1 - a)); it is negative if even that scenario gains.
    * Expected shortfall: the mean loss over the worst N (1 - a) scenarios,
      the boundary scenario weighted by the fractional part (Acerbi-Tasche),
      so ES >= VaR.
    * Delta-normal: sigma = standard deviation (divisor N - 1) of the
      first-order P&L ``sum_i PV01_i b_i`` over the scenarios, i.e.
      ``sqrt(d' S d)`` for the sample covariance S of the quote moves; VaR =
      z_a sigma and ES = sigma phi(z_a) / (1 - a), the mean daily P&L taken
      as zero.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from derivx.data.rates_history import FX_DATASET, HistoryStore, dataset_info, user_dataset
from derivx.rates.curves import DISCOUNT_KEY, GOVT_KEY, SPREAD_KINDS, Market
from derivx.rates.pca import MAX_GAP_DAYS, tenor_map, tenor_years
from derivx.risk.reporting import CURRENCIES, Reporting, market_pair, ordered, total_label
from derivx.risk.scenarios import GOVT_HISTORY, SWAP_HISTORY, SWAP_PROXIES, QuoteRisk, quoted_levels, shocked_curves
from derivx.risk.sensitivities import Position

CONFIDENCES = (0.95, 0.975, 0.99)
LOOKBACKS = (250, 500, 1000)
ANCHOR = GOVT_HISTORY["USD"]
MIN_COVERAGE = 0.9
MAX_STALE_DAYS = 5
MIN_SCENARIOS = 20
TOTAL = "TOTAL"
FX_HELD = "(FX HELD)"
FX_ONLY = "(FX ONLY)"
FX_KEY = "FX"  # the FX spots' row of the sources table


@dataclass(frozen=True)
class CurveSource:
    """The history behind one curve's moves: ``flag`` is ``real``,
    ``upload``, ``proxy`` or ``cross-currency proxy``; ``coverage`` the share
    of scenario dates with an observed move."""

    key: str
    currency: str
    dataset: str
    title: str
    flag: str
    coverage: float
    note: str


@dataclass(frozen=True)
class FxMoves:
    """Daily moves of the ECB reference rates on the scenario dates:
    ``ratio[c][i]`` is the rate of currency c (units per 1 EUR) on scenario
    day i over its rate on the previous scenario day, as-of aligned like the
    curves (EUR is 1 throughout); a day without a rate within
    ``max_stale_days`` on either side is no move (ratio 1, ``observed``
    False)."""

    ratio: Dict[str, np.ndarray]
    observed: np.ndarray
    title: str = "ECB euro foreign exchange reference rates"

    @property
    def coverage(self) -> float:
        return float(self.observed.mean()) if self.observed.size else 0.0

    def translation(self, currencies: Sequence[str], reporting: str) -> np.ndarray:
        """``x[i, j]``: the relative move on scenario day i of the value in
        ``reporting`` of one unit of ``currencies[j]`` (0 for the reporting
        currency itself)."""

        missing = sorted(c for c in {*currencies, reporting} if c not in self.ratio)
        if missing:
            raise ValueError(f"the FX history has no {', '.join(missing)} rate on the scenario dates")
        base = self.ratio[reporting]
        if not len(currencies):
            return np.zeros((len(base), 0))
        return np.column_stack([base / self.ratio[c] - 1.0 for c in currencies])


@dataclass(frozen=True)
class VarScenarios:
    """Daily quote moves (bp) of each curve of a market: ``bumps[key]`` is
    scenarios x quotes; ``dates[i]`` the day of scenario i and
    ``previous[i]`` the observation it moves from."""

    asof: dt.date
    dates: Tuple[dt.date, ...]
    previous: Tuple[dt.date, ...]
    bumps: Dict[str, np.ndarray]
    labels: Dict[str, Tuple[str, ...]]
    sources: Tuple[CurveSource, ...]
    lookback: int = 0  # daily changes asked for; ``n`` is fewer when the daily history starts later
    fx: Optional[FxMoves] = None  # None: FX spots held
    fx_note: str = ""

    @property
    def n(self) -> int:
        return len(self.dates)

    @property
    def short(self) -> bool:
        """True if the daily history held fewer changes than asked for."""

        return self.n < self.lookback

    def sources_table(self) -> pd.DataFrame:
        rows = [
            {
                "curve": s.key,
                "currency": s.currency,
                "history": s.title,
                "dataset": s.dataset,
                "flag": s.flag,
                "coverage_pct": 100.0 * s.coverage,
                "note": s.note,
            }
            for s in self.sources
        ]
        if self.fx is not None:
            rows.append(
                {
                    "curve": FX_KEY,
                    "currency": ", ".join(c for c in CURRENCIES if c in self.fx.ratio),
                    "history": self.fx.title,
                    "dataset": FX_DATASET,
                    "flag": "real",
                    "coverage_pct": 100.0 * self.fx.coverage,
                    "note": "FX spot moves of the same dates: they move the totals converted to the reporting "
                    "currency (ALL IN), not local P&L",
                }
            )
        return pd.DataFrame(rows, columns=["curve", "currency", "history", "dataset", "flag", "coverage_pct", "note"])


# --------------------------------------------------------------------------
# Histories
# --------------------------------------------------------------------------


def dataset_levels(store: HistoryStore, dataset: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Decimal levels of a dataset on its complete-curve dates in [start,
    end], one column per tenor present on all of them. Zero-curve datasets
    with OIS conventions (the BoE SONIA curve) give their derived par OIS
    rates, the rates the swap curve is quoted in."""

    frame = quoted_levels(store, dataset, start, end)
    return frame if frame.empty else frame.dropna(axis=1, how="any")


def daily_pairs(dates: Sequence[dt.date], max_gap_days: int = MAX_GAP_DAYS) -> List[Tuple[dt.date, dt.date]]:
    """(previous, date) of consecutive observations at most ``max_gap_days``
    calendar days apart."""

    days = sorted(dates)
    return [(a, b) for a, b in zip(days[:-1], days[1:]) if (b - a).days <= max_gap_days]


def recent_daily_pairs(dates: Sequence[dt.date], max_gap_days: int = MAX_GAP_DAYS) -> List[Tuple[dt.date, dt.date]]:
    """(previous, date) of consecutive observations in the most recent
    unbroken stretch: back from the last observation until two consecutive
    ones are more than ``max_gap_days`` calendar days apart."""

    days = sorted(dates)
    i = len(days) - 1
    while i > 0 and (days[i] - days[i - 1]).days <= max_gap_days:
        i -= 1
    return list(zip(days[i:-1], days[i + 1 :]))


def aligned_changes(
    levels: pd.DataFrame,
    previous: Sequence[dt.date],
    dates: Sequence[dt.date],
    max_stale_days: int = MAX_STALE_DAYS,
) -> Tuple[np.ndarray, np.ndarray]:
    """(changes in bp, scenarios x tenors; observed mask): as-of alignment
    of a history to the scenario dates (see the module docstring); rows not
    observed are zero."""

    n = len(dates)
    out = np.zeros((n, levels.shape[1]))
    if levels.empty or not n:
        return out, np.zeros(n, dtype=bool)
    ordinals = np.array([d.toordinal() for d in levels.index])
    values = levels.to_numpy(dtype=float)
    ends = np.array([d.toordinal() for d in dates])
    starts = np.array([d.toordinal() for d in previous])
    i_end = np.searchsorted(ordinals, ends, side="right") - 1
    i_start = np.searchsorted(ordinals, starts, side="right") - 1
    ok = (i_end >= 0) & (i_start >= 0)
    ok &= np.where(ok, ends - ordinals[np.maximum(i_end, 0)], 10**6) <= max_stale_days
    ok &= np.where(ok, starts - ordinals[np.maximum(i_start, 0)], 10**6) <= max_stale_days
    out[ok] = (values[i_end[ok]] - values[i_start[ok]]) * 1e4
    return out, ok


def fx_moves(
    store: HistoryStore,
    previous: Sequence[dt.date],
    dates: Sequence[dt.date],
    max_stale_days: int = MAX_STALE_DAYS,
) -> Optional[FxMoves]:
    """The ECB reference-rate moves of the store on the scenario days
    ``previous[i] -> dates[i]`` (see :class:`FxMoves`); None when the store
    holds no FX history."""

    table = store.fx_table()
    n = len(dates)
    if table.empty or not n:
        return None
    ordinals = np.array([d.toordinal() for d in table.index])
    ends = np.array([d.toordinal() for d in dates])
    starts = np.array([d.toordinal() for d in previous])
    i_end = np.searchsorted(ordinals, ends, side="right") - 1
    i_start = np.searchsorted(ordinals, starts, side="right") - 1
    ok = (i_end >= 0) & (i_start >= 0)
    ok &= np.where(ok, ends - ordinals[np.maximum(i_end, 0)], 10**6) <= max_stale_days
    ok &= np.where(ok, starts - ordinals[np.maximum(i_start, 0)], 10**6) <= max_stale_days
    ratio: Dict[str, np.ndarray] = {}
    for currency in table.columns:
        values = table[currency].to_numpy(dtype=float)
        r = np.ones(n)
        with np.errstate(invalid="ignore", divide="ignore"):
            moved = values[np.maximum(i_end, 0)] / values[np.maximum(i_start, 0)]
        good = ok & np.isfinite(moved) & (moved > 0)
        r[good] = moved[good]
        ratio[str(currency)] = r
    return FxMoves(ratio, ok)


def _candidates(currency: str) -> List[Tuple[str, str]]:
    """(dataset, flag) in order of preference for a currency's swap curve."""

    out = [(user_dataset(currency), "upload")] + [(k, "real") for k in SWAP_HISTORY.get(currency, ())]
    for key in SWAP_PROXIES.get(currency, (ANCHOR,)) + (ANCHOR,):
        flag = "proxy" if dataset_info(key).currency == currency else "cross-currency proxy"
        if key not in [k for k, _ in out]:
            out.append((key, flag))
    return out


def _note(dataset: str, flag: str, currency: str, coverage: float) -> str:
    info = dataset_info(dataset)
    if flag == "upload":
        return f"your uploaded {currency} swap history"
    if flag == "real":
        extra = " (LIBOR swaps standing in for SOFR swaps)" if dataset == "USD_SWAP_ICE" else ""
        return f"{info.title}{extra}"
    if flag == "proxy":
        return f"no {currency} swap history covers the window: {info.title} moves, swap spreads held"
    return (
        f"no {currency} history covers the window: {info.title} moves as a cross-currency proxy; upload "
        f"{currency} swap history for a real one"
    )


def build_scenarios(
    market: Market,
    store: HistoryStore,
    lookback: int = 500,
    end: Optional[dt.date] = None,
    max_stale_days: int = MAX_STALE_DAYS,
    min_coverage: float = MIN_COVERAGE,
    fx: bool = True,
) -> VarScenarios:
    """Quote moves of every curve of ``market`` (see the module docstring)
    for the last ``lookback`` daily changes up to ``end`` (default: the
    valuation date), and the FX moves of the same days (``fx``; held when
    False, or when the store's FX history covers less than
    ``min_coverage`` of the days)."""

    if int(lookback) != lookback or lookback < MIN_SCENARIOS:
        raise ValueError(f"the lookback must be a whole number of at least {MIN_SCENARIOS} days")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    end = end or market.asof
    anchor_dates = [d for d in store.dates(ANCHOR) if d <= end]
    pairs = recent_daily_pairs(anchor_dates)[-int(lookback) :]
    if len(pairs) < MIN_SCENARIOS:
        raise ValueError(
            f"only {len(pairs)} daily US Treasury changes up to {end} in the most recent daily stretch of the "
            f"history store (need {MIN_SCENARIOS}); the bundled history is daily from 2015, the LIVE source adds the "
            "latest days"
        )
    previous = tuple(a for a, _ in pairs)
    dates = tuple(b for _, b in pairs)
    first, last = previous[0] - dt.timedelta(days=max_stale_days), dates[-1]
    cache: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}

    def moves(dataset: str):
        if dataset not in cache:
            levels = dataset_levels(store, dataset, first, last) if store.has(dataset) else pd.DataFrame()
            changes, observed = aligned_changes(levels, previous, dates, max_stale_days)
            cache[dataset] = (levels, changes, observed)
        return cache[dataset]

    currency = market.curve.conventions.currency if market.curve.conventions is not None else "USD"
    swap_choice: Optional[Tuple[str, str, float]] = None
    for dataset, flag in _candidates(currency):
        levels, _, observed = moves(dataset)
        coverage = float(observed.mean()) if len(observed) else 0.0
        if levels.shape[1] >= 1 and coverage >= min_coverage:
            swap_choice = (dataset, flag, coverage)
            break
    if swap_choice is None:  # the anchor always covers its own dates
        raise ValueError("no history covers the scenario dates")

    bumps: Dict[str, np.ndarray] = {}
    labels: Dict[str, Tuple[str, ...]] = {}
    sources: List[CurveSource] = []
    for key in shocked_curves(market):
        curve = market.get_curve(key)
        if not curve.instruments:
            raise ValueError(f"the {key!r} curve has no quotes to move; bootstrap it from quotes")
        if key == GOVT_KEY:
            dataset, flag, ccy = ANCHOR, "real", "USD"
            coverage = float(moves(ANCHOR)[2].mean())
            note = f"{dataset_info(ANCHOR).title}: par yield changes"
        else:
            dataset, flag, coverage = swap_choice
            ccy = currency
            note = _note(dataset, flag, currency, coverage)
            if key != DISCOUNT_KEY:
                flag = "proxy"
                note = f"moves with the {currency} swap curve ({note}): no free {key} history, basis held"
        levels, changes, _ = moves(dataset)
        weights = tenor_map(curve.years, tenor_years(levels.columns))
        spread = np.array([inst.kind in SPREAD_KINDS for inst in curve.instruments])
        quote_moves = changes @ weights.T
        quote_moves[:, spread] = 0.0
        bumps[key] = quote_moves
        labels[key] = tuple(curve.labels)
        sources.append(CurveSource(key, ccy, dataset, dataset_info(dataset).title, flag, coverage, note))
    moves_fx: Optional[FxMoves] = None
    fx_note = "FX spots held (as asked)"
    if fx:
        moves_fx = fx_moves(store, previous, dates, max_stale_days)
        if moves_fx is None:
            fx_note = "FX spots held: the history store has no ECB reference rates"
        elif moves_fx.coverage < min_coverage:
            fx_note = (
                f"FX spots held: the ECB reference rates move on only {100 * moves_fx.coverage:.0f}% of the "
                f"scenario dates (need {100 * min_coverage:.0f}%)"
            )
            moves_fx = None
        else:
            fx_note = (
                f"FX spots move with the ECB reference rates of the same dates ({100 * moves_fx.coverage:.0f}% of "
                "the days observed)"
            )
    return VarScenarios(market.asof, dates, previous, bumps, labels, tuple(sources), int(lookback), moves_fx, fx_note)


# --------------------------------------------------------------------------
# P&L
# --------------------------------------------------------------------------


def sensitivity_pnl(scenarios: VarScenarios, risk: Mapping[str, QuoteRisk], gamma: bool = True) -> np.ndarray:
    """Delta (+ diagonal gamma) P&L, scenarios x positions, from key-rate
    risk of every moved curve on the valuation date."""

    total: Optional[np.ndarray] = None
    for key, bumps in scenarios.bumps.items():
        qr = risk.get(key)
        if qr is None:
            raise ValueError(f"no key-rate risk for the {key!r} curve")
        if qr.pv01.shape[0] != bumps.shape[1] or not qr.computed.all():
            raise ValueError(f"the {key!r} key-rate risk does not cover every quote of the curve")
        if qr.asof != scenarios.asof:
            raise ValueError(f"the {key!r} key-rate risk is dated {qr.asof}, the scenarios {scenarios.asof}")
        part = bumps @ qr.pv01
        if gamma:
            part = part + 0.5 * (bumps * bumps) @ qr.gamma
        total = part if total is None else total + part
    if total is None:
        raise ValueError("the scenarios move no curve")
    return total


def full_revaluation_pnl(
    positions: Sequence[Position],
    market: Market,
    scenarios: VarScenarios,
    progress: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Full-revaluation P&L, scenarios x positions: each day's quote moves
    applied to today's curves, the curves rebuilt, every position revalued
    (vols held). ``progress(i, n)`` is called after each scenario."""

    positions = list(positions)
    base = np.array([float(p.pv(market)) for p in positions])
    out = np.zeros((scenarios.n, len(positions)))
    for i, day in enumerate(scenarios.dates):
        moved = {k: list(b[i]) for k, b in scenarios.bumps.items() if np.any(b[i] != 0.0)}
        try:
            shocked = market.rebuilt(moved) if moved else market
            values = np.array([float(p.pv(shocked)) for p in positions])
        except ValueError as exc:
            raise ValueError(f"the {day} scenario cannot be revalued: {exc}") from exc
        bad = [p.name for p, v in zip(positions, values) if not math.isfinite(v)]
        if bad:
            raise ValueError(f"the {day} scenario gives no finite value for {', '.join(bad)}")
        out[i] = values - base
        if progress is not None:
            progress(i + 1, scenarios.n)
    return out


# --------------------------------------------------------------------------
# Risk measures
# --------------------------------------------------------------------------


def _tail(n: int, confidence: float) -> float:
    if not 0.5 <= confidence < 1.0:
        raise ValueError(f"the confidence must be in [0.5, 1), e.g. 0.99; got {confidence}")
    tail = round(n * (1.0 - confidence), 9)
    if tail < 1.0:
        raise ValueError(
            f"{n} scenarios cannot give a {confidence:.1%} VaR: it needs at least {math.ceil(1 / (1 - confidence))}"
        )
    return tail


def var_es(pnl: Sequence[float], confidence: float) -> Tuple[float, float]:
    """(VaR, expected shortfall) of P&L samples as positive losses (see the
    module docstring)."""

    values = np.asarray(pnl, dtype=float).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("VaR needs finite P&L values")
    tail = _tail(values.size, confidence)
    losses = np.sort(-values)[::-1]
    k = math.ceil(tail)
    whole = math.floor(tail)
    var = float(losses[k - 1])
    es = (float(losses[:whole].sum()) + (tail - whole) * (float(losses[whole]) if whole < losses.size else 0.0)) / tail
    return var, float(es)


@dataclass(frozen=True)
class DeltaNormal:
    sigma: float
    var: float
    es: float


def delta_normal(delta_pnl: Sequence[float], confidence: float) -> DeltaNormal:
    """Delta-normal VaR and ES from first-order scenario P&L (see the
    module docstring)."""

    values = np.asarray(delta_pnl, dtype=float).ravel()
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("delta-normal VaR needs at least two finite P&L values")
    _tail(values.size, confidence)
    sigma = float(values.std(ddof=1))
    z = float(norm.ppf(confidence))
    return DeltaNormal(sigma, z * sigma, sigma * float(norm.pdf(z)) / (1.0 - confidence))


def book_columns(
    pnl: np.ndarray,
    names: Sequence[str],
    currencies: Optional[Sequence[str]] = None,
    reporting: Optional[Reporting] = None,
    fx: Optional[FxMoves] = None,
    base: Optional[Sequence[float]] = None,
    linear: bool = False,
) -> Dict[str, Tuple[np.ndarray, str]]:
    """{label: (P&L per scenario, currency)}: each position, then the book
    per currency (``TOTAL`` or ``TOTAL <CCY>``: P&L summed per scenario over
    that currency's positions) and, when ``reporting`` is given and some
    position is in another currency, ``ALL IN <R>``: every position
    converted at today's spot and, with ``fx`` (and ``base``, today's value
    of each position in its currency), moved by the FX change of the day,
    followed by ``ALL IN <R> (FX HELD)`` and ``ALL IN <R> (FX ONLY)`` (see
    the module docstring). ``linear`` leaves out the cross term p S x (the
    first-order P&L of the delta-normal check). Without ``currencies``
    every position is taken to be in one currency (labelled ``""``)."""

    pnl = np.atleast_2d(np.asarray(pnl, dtype=float))
    ccys = list(currencies) if currencies is not None else [""] * len(names)
    if len(ccys) != len(names) or pnl.shape[1] != len(names):
        raise ValueError(f"{pnl.shape[1]} P&L columns, {len(names)} names and {len(ccys)} currencies")
    out: Dict[str, Tuple[np.ndarray, str]] = {name: (pnl[:, j], ccys[j]) for j, name in enumerate(names)}
    groups = ordered(ccys) if names else [""]
    for ccy in groups:
        picks = [j for j, c in enumerate(ccys) if c == ccy]
        out[total_label(ccy, len(groups) > 1)] = (pnl[:, picks].sum(axis=1), ccy)
    if reporting is not None and currencies is not None and reporting.needs_conversion(ccys):
        rates = reporting.rates(ccys)
        held = (pnl * rates).sum(axis=1)
        if fx is None:
            out[reporting.label] = (held, reporting.currency)
        else:
            if base is None or len(base) != len(names):
                raise ValueError("FX moves need today's value of each position (base)")
            x = fx.translation(ccys, reporting.currency)
            if x.shape[0] != pnl.shape[0]:
                raise ValueError(f"{pnl.shape[0]} scenarios but FX moves for {x.shape[0]}")
            today = np.asarray(base, dtype=float)[None, :]
            fx_only = (today * rates * x).sum(axis=1)
            cross = np.zeros(pnl.shape[0]) if linear else (pnl * rates * x).sum(axis=1)
            out[reporting.label] = (held + fx_only + cross, reporting.currency)
            out[f"{reporting.label} {FX_HELD}"] = (held, reporting.currency)
            out[f"{reporting.label} {FX_ONLY}"] = (fx_only, reporting.currency)
    return out


def var_table(
    pnl: np.ndarray,
    names: Sequence[str],
    confidence: float,
    delta: Optional[np.ndarray] = None,
    currencies: Optional[Sequence[str]] = None,
    reporting: Optional[Reporting] = None,
    fx: Optional[FxMoves] = None,
    base: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """One row per position and a ``TOTAL`` row (the book: P&L summed per
    scenario, so its VaR is not the sum of the rows): VaR, ES, worst loss,
    mean and sd of daily P&L; with ``delta`` (first-order P&L) the
    delta-normal sigma, VaR and ES. With ``currencies`` (one per position)
    the table has a ``currency`` column and a total per currency, and with
    ``reporting`` the whole book converted; with ``fx`` (and ``base``) the
    converted rows move with FX and come with their FX HELD and FX ONLY
    rows (:func:`book_columns`)."""

    columns = book_columns(pnl, names, currencies, reporting, fx, base)
    first = None
    if delta is not None:
        first = book_columns(delta, names, currencies, reporting, fx, base, linear=True)
    rows = []
    for name, (values, ccy) in columns.items():
        var, es = var_es(values, confidence)
        row: Dict[str, object] = {"position": name}
        if currencies is not None:
            row["currency"] = ccy
        row.update(var=var, es=es, worst_loss=float(-values.min()), mean_pnl=float(values.mean()),
                   sd_pnl=float(values.std(ddof=1)))  # fmt: skip
        if first is not None:
            dn = delta_normal(first[name][0], confidence)
            row.update(dn_sigma=dn.sigma, dn_var=dn.var, dn_es=dn.es)
        rows.append(row)
    return pd.DataFrame(rows)


def pair_moves(fx: FxMoves, currencies: Sequence[str], reporting: str) -> Dict[str, np.ndarray]:
    """``{market pair: % move per scenario}`` of the pairs that convert
    ``currencies`` into ``reporting`` (EURUSD +1.0: the euro up 1% in
    dollars)."""

    out: Dict[str, np.ndarray] = {}
    for ccy in ordered([c for c in currencies if c != reporting]):
        pair = market_pair(ccy, reporting)
        x = fx.translation([ccy], reporting)[:, 0]  # the value of ccy in the reporting currency
        out[pair] = 100.0 * (x if pair.startswith(ccy) else 1.0 / (1.0 + x) - 1.0)
    return out


def worst_days(
    scenarios: VarScenarios,
    total_pnl: Sequence[float],
    n: int = 10,
    currencies: Optional[Sequence[str]] = None,
    reporting: Optional[str] = None,
) -> pd.DataFrame:
    """The ``n`` worst scenario days of the book: date, P&L, the largest
    quote move of each curve that day (bp) and, when the scenarios move FX
    and ``currencies`` convert into ``reporting``, each converting pair's
    move that day (%)."""

    total = np.asarray(total_pnl, dtype=float)
    order = np.argsort(total)[: int(n)]
    moves: Dict[str, np.ndarray] = {}
    if scenarios.fx is not None and currencies is not None and reporting is not None:
        moves = pair_moves(scenarios.fx, currencies, reporting)
    rows = []
    for i in order:
        row: Dict[str, object] = {"date": scenarios.dates[i], "from": scenarios.previous[i], "pnl": float(total[i])}
        for key, bumps in scenarios.bumps.items():
            j = int(np.argmax(np.abs(bumps[i]))) if bumps.shape[1] else 0
            row[f"{key} max move (bp)"] = float(bumps[i, j]) if bumps.shape[1] else 0.0
            row[f"{key} at"] = scenarios.labels[key][j] if bumps.shape[1] else ""
        for pair, pct in moves.items():
            row[f"{pair} move (%)"] = float(pct[i])
        rows.append(row)
    return pd.DataFrame(rows)


def risk_report(
    names: Sequence[str],
    risk: Mapping[str, QuoteRisk],
    table: pd.DataFrame,
    vegas: Optional[Sequence[float]] = None,
    vol_models: Optional[Sequence[Optional[str]]] = None,
    currencies: Optional[Sequence[str]] = None,
    vega_buckets: Optional[Sequence[Sequence[Mapping[str, object]]]] = None,
) -> pd.DataFrame:
    """One long table of the book's risk, for export: ``section``
    (``pv01``, ``gamma``, ``vega``, ``vega_bucket``, ``var``), ``position``,
    ``curve``, ``item`` (quote label, bucket or measure), ``value``,
    ``unit`` and ``currency`` (of ``value``; ``currencies`` gives each
    position's). PV01 is P&L per +1bp of one curve quote, gamma per
    (1bp)^2, vega per +1bp normal vol or +1 lognormal vol point (the vol
    shift the position's model takes). ``vega_buckets`` (per position, rows
    of ``expiry``, ``bucket``, ``vega``, ``method`` as
    :func:`derivx.rates.portfolio.position_vega_buckets` gives them) adds
    one ``vega_bucket`` row per bucket, P&L per +1bp normal vol: item
    ``"<option expiry> x <swap tenor>"`` for a swaption, ``"<caplet expiry>
    x CAP <index>"`` for a cap, and a ``TOTAL`` row per currency and bucket
    (``TOTAL <CCY>`` for a book in several). The VaR rows are
    ``var_table``'s columns, in the currency of their row."""

    ccys = list(currencies) if currencies is not None else [""] * len(names)
    if len(ccys) != len(names):
        raise ValueError(f"{len(names)} positions but {len(ccys)} currencies")
    rows: List[Dict[str, object]] = []
    for key, qr in risk.items():
        for i, label in enumerate(qr.labels):
            for j, name in enumerate(names):
                rows.append({"section": "pv01", "position": name, "curve": key, "item": label,
                             "value": float(qr.pv01[i, j]), "unit": "P&L per +1bp", "currency": ccys[j]})  # fmt: skip
                rows.append({"section": "gamma", "position": name, "curve": key, "item": label,
                             "value": float(qr.gamma[i, j]), "unit": "P&L per (1bp)^2",
                             "currency": ccys[j]})  # fmt: skip
    if vegas is not None:
        models = list(vol_models or [None] * len(names))
        for name, vega, model, ccy in zip(names, vegas, models, ccys):
            if model is None:
                continue
            unit = "P&L per +1bp normal vol" if model == "normal" else "P&L per +1 lognormal vol point"
            rows.append({"section": "vega", "position": name, "curve": "", "item": f"{model} vol",
                         "value": float(vega), "unit": unit, "currency": ccy})  # fmt: skip
    if vega_buckets is not None:
        if len(vega_buckets) != len(names):
            raise ValueError(f"{len(names)} positions but vega buckets for {len(vega_buckets)}")
        totals: Dict[Tuple[str, str], float] = {}
        order: List[Tuple[str, str]] = []
        for name, buckets, ccy in zip(names, vega_buckets, ccys):
            for b in buckets:
                item = f"{b['expiry']} x {b['bucket']}"
                rows.append({"section": "vega_bucket", "position": name, "curve": "", "item": item,
                             "value": float(b["vega"]), "unit": f"P&L per +1bp normal vol ({b['method']})",
                             "currency": ccy})  # fmt: skip
                if (ccy, item) not in totals:
                    order.append((ccy, item))
                    totals[(ccy, item)] = 0.0
                totals[(ccy, item)] += float(b["vega"])
        groups = ordered([c for c, _ in order])
        for ccy, item in order:
            rows.append({"section": "vega_bucket", "position": total_label(ccy, len(groups) > 1), "curve": "",
                         "item": item, "value": totals[(ccy, item)], "unit": "P&L per +1bp normal vol",
                         "currency": ccy})  # fmt: skip
    by_position = dict(zip(names, ccys))
    for _, row in table.iterrows():
        ccy = str(row["currency"]) if "currency" in row.index else by_position.get(str(row["position"]), "")
        for column, value in row.items():
            if column in ("position", "currency"):
                continue
            rows.append({"section": "var", "position": row["position"], "curve": "", "item": column,
                         "value": float(value), "unit": "losses positive", "currency": ccy})  # fmt: skip
    return pd.DataFrame(rows, columns=["section", "position", "curve", "item", "value", "unit", "currency"])
