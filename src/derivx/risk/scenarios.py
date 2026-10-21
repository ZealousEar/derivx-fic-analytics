"""Curve, vol and horizon scenarios with a P&L explain.

A :class:`Shock` moves curve *quotes* by a piecewise-linear profile in bp by
tenor (flat beyond its first and last points): quote i of a curve moves by
``profile(curve.instruments[i].tenor.years)`` and the curve is rebuilt with
:meth:`Curve.rebuilt`, so the shocked curve reprices its shocked quotes. It
can also shift vols (normal bp, lognormal vol points) and roll the market
forward ``horizon_days``.

:func:`run_scenario` fully revalues every position and explains the P&L::

    total     = PV(shocked, at horizon) + cash paid in the window - PV(today)
    carry     = PV(today's quotes rebuilt at the horizon) + cash - PV(today)
    delta     = sum_i PV01_i * bp_i                 (key-rate PV01s, both curves)
    gamma     = 1/2 sum_i Gamma_ii * bp_i^2         (diagonal key-rate gammas)
    cross     = second-order P&L along the shock - gamma (cross-gammas)
    vol       = vega * vol shock
    residual  = total - (carry + delta + gamma + cross + vol)

The shock hits at the horizon: the market first rolls forward with today's
quotes (carry and roll-down; fixings and cash in the window follow today's
curve, ``Market.fixings`` is the base curve), then the quotes and vols move.
The Greeks are those of the rolled market, so a trade that matures before
the horizon has no delta and the residual holds only third- and
higher-order terms.

PCA factor shocks (:func:`pca_factor`, :func:`pca_library`) move every
shocked curve by n standard deviations of one principal component (level,
slope, curvature) of a history's daily changes, from
:mod:`derivx.rates.pca`, and record the history used in ``Shock.sources``.

Historical shocks (:func:`historical`, :func:`history_window`) replay
each curve's own history where a free one exists: the Treasury curve moves
by the Treasury's par yield changes, and the swap curve by the swap/OIS
history of its currency (ICE USD swap rates before 2016-10-28, the Bank of
England's OIS curve for GBP, or the user's uploaded history), otherwise by a
flagged proxy (government yield changes, i.e. swap spreads held). FX spots
move with the ECB euro reference rates of the same dates (``Shock.fx``:
EURUSD and GBPUSD, from the store's ``FX_ECB`` history; the nearest rates to
the Treasury curve's start and end dates within the same tolerance), so the
book's converted totals replay the FX move too; ``fx=False`` holds them,
and a window the FX history does not cover holds them with a note. Each
shock records the data behind every curve and the FX spots in
``Shock.sources``.

FX spot shocks (``Shock.fx``, :func:`fx_spot`, :func:`fx_library`) move
EURUSD and GBPUSD by a percentage. They move no curve or vol, so every
position's P&L in its own currency is unchanged; they move the book's
totals converted to a reporting currency
(:func:`~derivx.risk.reporting.add_total_columns`, ``scenario_grid(...,
reporting=...)``: the FX translation of each position's value after the
scenario). Without an FX shock or an FX replay, FX spots are held.

Multi-curve markets: a shock moves the discount curve (``curve``), the
projection curves (``projections``, e.g. EURIBOR 3M/6M) and the Treasury
curve (``govt``) independently, each projection rebuilt on the moved
discount curve. Rate shocks move rate quotes (``rates``) and leave tenor
basis spreads alone; :func:`ibor_ois_basis` moves only the projection
curves' rate quotes (IBOR-OIS basis widens) and :func:`tenor_basis` only
their ``BASIS`` spreads (3s6s). A profile can be given per curve with
``profiles`` selectors ``"<curve key>:<currency>"`` (``"EURIBOR 6M:EUR"``);
a projection curve without its own profile follows its currency's swap
curve profile. The P&L explain splits delta by curve (``delta_by_curve``).

The key-rate gammas reuse the +/-1bp revaluations of the PV01s. On a
bootstrapped curve the cross-gammas are not small (the 10Y swap's annuity
moves with every shorter quote; diagonal gammas alone leave ~3% of a 100bp
parallel move on a 10Y swap unexplained), so ``cross_gamma`` is measured
with one extra pair of revaluations along the shock direction.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from derivx.data.rates_history import (
    EPISODE_TOLERANCE_DAYS,
    FX_DATASET,
    HistoryStore,
    bundled_store,
    dataset_info,
    user_dataset,
)
from derivx.rates import pca as pca_mod
from derivx.rates.conventions import Date, parse_tenor
from derivx.rates.curves import DISCOUNT_KEY, GOVT_KEY, SPREAD_KINDS, Curve, Market
from derivx.rates.indices import CURRENCY_INDICES
from derivx.risk.reporting import SHOCK_PAIRS, Reporting, add_total_columns, ordered, position_currencies
from derivx.risk.sensitivities import Position

NORMAL_VOL_UNIT = 1e-4  # 1bp of normal vol
LOGNORMAL_VOL_UNIT = 0.01  # 1 lognormal vol point
EXPLAIN_COLUMNS = ("delta", "gamma", "cross_gamma", "vol", "carry", "residual")
ROW_COLUMNS = ("position", "kind", "base_pv", "shocked_pv", "cash", "pnl") + EXPLAIN_COLUMNS
TOTAL = "TOTAL"
GOVT_KINDS = ("BILL", "BOND")
Profile = Tuple[Tuple[float, float], ...]

# Named historical windows (nominal dates; each curve uses its nearest
# complete curves within EPISODE_TOLERANCE_DAYS).
EPISODES: Tuple[Tuple[str, dt.date, dt.date], ...] = (
    ("2013 taper tantrum", dt.date(2013, 5, 2), dt.date(2013, 9, 5)),
    ("2020 dash for cash", dt.date(2020, 3, 9), dt.date(2020, 3, 18)),
    ("2022 hiking cycle", dt.date(2022, 1, 3), dt.date(2022, 10, 21)),
    ("Sept 2022 gilt crisis week", dt.date(2022, 9, 21), dt.date(2022, 9, 28)),
    ("2023 SVB", dt.date(2023, 3, 8), dt.date(2023, 3, 15)),
)
CURRENCIES = ("USD", "EUR", "GBP")
GOVT_HISTORY = {"USD": "UST_PAR"}
SWAP_HISTORY = {"USD": ("USD_SWAP_ICE",), "GBP": ("GBP_OIS_BOE",), "EUR": ()}  # real swap/OIS histories
SWAP_PROXIES = {"USD": ("UST_PAR",), "GBP": ("UST_PAR",), "EUR": ("EUR_AAA_ECB", "UST_PAR")}


@dataclass(frozen=True)
class ReplaySource:
    """The data a historical shock uses for one curve: ``selector`` is
    ``"<curve field>:<currency>"`` (``"projections:EUR"`` for the projection
    curves), ``flag`` one of ``"real"`` (the curve's own market),
    ``"upload"`` (the user's history), ``"proxy"`` (another market's moves,
    same currency) or ``"cross-currency proxy"``; ``"fx:EURUSD,GBPUSD"`` is
    the FX spots' (flag ``"real"``, or ``"held"`` when they do not move)."""

    selector: str
    dataset: str
    title: str
    start: str
    end: str
    flag: str
    note: str


# --------------------------------------------------------------------------
# Shocks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Shock:
    """A scenario: quote moves by tenor, vol shifts and a horizon.

    ``profile`` is ``((years, bp), ...)``, linear in between and flat
    outside. ``short_bp``/``long_bp`` (optional) pin the move at the
    shortest/longest quote of each curve being shocked, so a twist reaches
    its end values on any curve. ``curve``/``govt``/``projections`` say
    which of the market's curves move (discount, Treasury, projection;
    ``projections=None`` follows ``curve``, so a swap-curve shock moves the
    projection curves too); ``rates`` and ``spreads`` which of their quotes
    (rate levels, ``BASIS`` spreads). ``normal_vol_bp`` is added to normal vols (bp) and
    ``lognormal_vol_pts`` to lognormal vols (vol points). ``horizon_days``
    rolls the market forward (calendar days, adjusted to a business day).
    ``fx`` is ``((pair, % move), ...)`` of FX spots (EURUSD, GBPUSD): it moves
    only totals converted between currencies (see the module docstring).
    """

    name: str
    profile: Tuple[Tuple[float, float], ...] = ()
    short_bp: Optional[float] = None
    long_bp: Optional[float] = None
    curve: bool = True
    govt: bool = True
    normal_vol_bp: float = 0.0
    lognormal_vol_pts: float = 0.0
    horizon_days: int = 0
    description: str = ""
    profiles: Tuple[Tuple[str, Profile], ...] = ()
    sources: Tuple[ReplaySource, ...] = ()
    projections: Optional[bool] = None
    rates: bool = True
    spreads: bool = False
    fx: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", _clean_profile(self.profile))
        cleaned = tuple((str(sel), _clean_profile(prof)) for sel, prof in self.profiles)
        selectors = [sel for sel, _ in cleaned]
        if len(set(selectors)) != len(selectors) or any(":" not in sel for sel in selectors):
            raise ValueError(f"profiles need distinct '<curve>:<currency>' selectors, got {selectors}")
        object.__setattr__(self, "profiles", cleaned)
        object.__setattr__(self, "sources", tuple(self.sources))
        optional = ("short_bp", "long_bp")
        for label in optional + ("normal_vol_bp", "lognormal_vol_pts"):
            value = getattr(self, label)
            if value is None and label in optional:
                continue
            if value is None or not math.isfinite(value):
                raise ValueError(f"{label} must be a finite number")
            object.__setattr__(self, label, float(value))
        days = self.horizon_days
        if not (math.isfinite(days) and days >= 0 and int(days) == days):
            raise ValueError("horizon_days must be a whole number of days >= 0")
        object.__setattr__(self, "horizon_days", int(days))
        object.__setattr__(self, "fx", _clean_fx(self.fx))

    @property
    def moves_fx(self) -> bool:
        """True if the shock moves an FX spot."""

        return any(pct != 0.0 for _, pct in self.fx)

    @property
    def moves_local(self) -> bool:
        """True if the shock can change a position's value in its own
        currency (quotes, vols or the horizon move)."""

        return bool(self.moves_rates or self.normal_vol_bp or self.lognormal_vol_pts or self.horizon_days)

    def profile_for(self, which: Optional[str] = None, currency: Optional[str] = None) -> Profile:
        """The profile for one curve: ``profiles["<which>:<currency>"]``,
        else ``profiles["<which>:*"]``; a projection curve then falls back
        to the swap curve's (``"curve:<currency>"``, ``"curve:*"``); else
        ``profile``."""

        if which is not None:
            table = dict(self.profiles)
            selectors = [f"{which}:{currency}", f"{which}:*"]
            if which not in (DISCOUNT_KEY, GOVT_KEY):
                selectors += [f"{DISCOUNT_KEY}:{currency}", f"{DISCOUNT_KEY}:*"]
            for selector in selectors:
                if selector in table:
                    return table[selector]
        return self.profile

    def points(
        self,
        span: Optional[Tuple[float, float]] = None,
        which: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(years, bp) of the profile for a curve, with the end anchors
        placed at ``span`` = (shortest, longest) quote years when given."""

        profile = self.profile_for(which, currency)
        xs = [y for y, _ in profile]
        ys = [b for _, b in profile]
        if span is not None:
            lo, hi = span
            if self.short_bp is not None and (not xs or lo < xs[0]):
                xs, ys = [lo] + xs, [self.short_bp] + ys
            if self.long_bp is not None and (not xs or hi > xs[-1]):
                xs, ys = xs + [hi], ys + [self.long_bp]
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def bp_at(
        self,
        years: Union[float, Sequence[float], np.ndarray],
        span: Optional[Tuple[float, float]] = None,
        which: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> np.ndarray:
        """Move in bp at tenors in years (flat beyond the profile)."""

        xs, ys = self.points(span, which, currency)
        years = np.atleast_1d(np.asarray(years, dtype=float))
        if not xs.size:
            return np.zeros_like(years)
        return np.interp(years, xs, ys)

    def quote_bumps(self, curve: Curve, which: Optional[str] = None) -> np.ndarray:
        """bp move of each of ``curve.quotes``. ``which`` (``"curve"``,
        ``"govt"`` or a projection index) picks the per-curve profile; by
        default a projection curve is its index, a curve built only from
        bills and bonds the government curve. Rate quotes move only if
        ``rates``, ``BASIS`` spreads only if ``spreads``."""

        if not curve.instruments:
            raise ValueError("the curve has no quotes to shock; bootstrap it from quotes")
        if which is None:
            index = getattr(curve, "index", None)
            if index is not None:
                which = index.name
            else:
                which = GOVT_KEY if all(inst.kind in GOVT_KINDS for inst in curve.instruments) else DISCOUNT_KEY
        currency = curve.conventions.currency if curve.conventions is not None else None
        years = curve.years
        bumps = self.bp_at(years, (float(years.min()), float(years.max())), which, currency)
        spread = np.array([inst.kind in SPREAD_KINDS for inst in curve.instruments])
        return np.where(spread, bumps if self.spreads else 0.0, bumps if self.rates else 0.0)

    def applies_to(self, which: str) -> bool:
        """Whether the shock moves the curve with key ``which``."""

        if which == DISCOUNT_KEY:
            return self.curve
        if which == GOVT_KEY:
            return self.govt
        return self.moves_projections

    @property
    def moves_projections(self) -> bool:
        return self.curve if self.projections is None else bool(self.projections)

    @property
    def moves_rates(self) -> bool:
        """True if some quote of a curve the shock applies to moves."""

        moves = [b for _, b in self.profile] + [b for b in (self.short_bp, self.long_bp) if b is not None]
        moves += [b for _, profile in self.profiles for _, b in profile]
        curves = self.curve or self.govt or self.moves_projections
        return curves and (self.rates or self.spreads) and any(b != 0.0 for b in moves)


def _clean_fx(fx: object) -> Tuple[Tuple[str, float], ...]:
    """``((pair, % move), ...)`` of :data:`~derivx.risk.reporting.SHOCK_PAIRS`
    (a mapping works too), pairs upper case and distinct, moves above -100%."""

    items = list(fx.items()) if isinstance(fx, Mapping) else list(fx)
    out: Dict[str, float] = {}
    for item in items:
        pair, pct = item
        code = str(pair).strip().upper().replace("/", "")
        if code not in SHOCK_PAIRS:
            raise ValueError(f"FX shock pair {pair!r}: use {' or '.join(SHOCK_PAIRS)} (moves against the dollar)")
        if code in out:
            raise ValueError(f"FX shock moves {code} twice")
        value = float(pct)
        if not (math.isfinite(value) and value > -100.0):
            raise ValueError(f"the {code} move must be a finite % above -100%, got {pct!r}")
        out[code] = value
    return tuple((p, out[p]) for p in SHOCK_PAIRS if p in out)


def _clean_profile(profile: Sequence[Tuple[float, float]]) -> Profile:
    points = tuple(sorted((float(y), float(b)) for y, b in profile))
    years = [y for y, _ in points]
    if any(not math.isfinite(v) for p in points for v in p) or any(y < 0 for y in years):
        raise ValueError("profile points must be finite (years >= 0, bp)")
    if len(set(years)) != len(years):
        raise ValueError(f"profile has two moves at the same tenor: {years}")
    return points


def _shock(name: str, profile, settings: Mapping[str, object], **fields) -> Shock:
    """Builder output; caller ``settings`` (any Shock field) win."""

    return Shock(profile=tuple(profile), **{"name": name, **fields, **settings})


def parallel(bp: float, **settings) -> Shock:
    """Every quote moves by ``bp``. ``settings`` are other :class:`Shock`
    fields (``name``, ``curve``, ``govt``, vols, ``horizon_days``)."""

    return _shock(f"Parallel {bp:+g}bp", [(0.0, bp)], settings, description=f"every quote {bp:+g}bp")


def twist(
    pivot_years: float,
    short_bp: float,
    long_bp: float,
    short_years: Optional[float] = None,
    long_years: Optional[float] = None,
    **settings,
) -> Shock:
    """Zero at ``pivot_years``, linear to ``short_bp`` and ``long_bp``.

    The ends sit at the shortest and longest quote of each shocked curve, or
    at ``short_years``/``long_years`` (flat beyond) when given. A steepener
    is ``twist(5, -x, +x)``.
    """

    if not pivot_years > 0:
        raise ValueError("pivot must be after 0 years")
    if short_years is not None and not short_years < pivot_years:
        raise ValueError("the short end must be before the pivot")
    if long_years is not None and not long_years > pivot_years:
        raise ValueError("the long end must be after the pivot")
    profile = [(pivot_years, 0.0)]
    ends = {}
    if short_years is None:
        ends["short_bp"] = float(short_bp)
    else:
        profile.insert(0, (short_years, short_bp))
    if long_years is None:
        ends["long_bp"] = float(long_bp)
    else:
        profile.append((long_years, long_bp))
    short = "shortest quote" if short_years is None else f"{short_years:g}Y"
    long = "longest quote" if long_years is None else f"{long_years:g}Y"
    return _shock(
        f"Twist {pivot_years:g}Y pivot {short_bp:+g}/{long_bp:+g}bp",
        profile,
        settings,
        description=f"{short} {short_bp:+g}bp, {pivot_years:g}Y 0, {long} {long_bp:+g}bp",
        **ends,
    )


def butterfly(
    wing_bp: float, body_bp: float, body_years: float, wings: Tuple[float, float] = (2.0, 30.0), **settings
) -> Shock:
    """``wing_bp`` at both wings, ``body_bp`` at ``body_years``, linear in
    between and flat beyond the wings."""

    left, right = float(wings[0]), float(wings[1])
    if not left < body_years < right:
        raise ValueError(f"the body ({body_years:g}Y) must lie between the wings ({left:g}Y, {right:g}Y)")
    return _shock(
        f"{left:g}s{body_years:g}s{right:g}s fly {body_bp:+g}/{wing_bp:+g}bp",
        [(left, wing_bp), (body_years, body_bp), (right, wing_bp)],
        settings,
        description=f"{body_years:g}Y {body_bp:+g}bp, {left:g}Y and {right:g}Y {wing_bp:+g}bp",
    )


def _years(tenor: Union[str, float]) -> float:
    return float(tenor) if isinstance(tenor, (int, float)) else parse_tenor(str(tenor)).years


def fx_spot(moves: Mapping[str, float], **settings) -> Shock:
    """FX spots move by ``moves`` (``{"EURUSD": 10}`` = the euro worth 10%
    more dollars), curves and vols unchanged: only totals converted to a
    reporting currency move."""

    fx = _clean_fx(moves)
    if not fx:
        raise ValueError("give at least one FX move (EURUSD or GBPUSD, in %)")
    text = ", ".join(f"{p} {m:+g}%" for p, m in fx)
    return _shock(f"FX {text}", [], settings, description=f"FX spot {text}; curves and vols unchanged", curve=False,
                  govt=False, projections=False, fx=fx)  # fmt: skip


def fx_library(currencies: Sequence[str], reporting: str, pct: float = 10.0) -> List[Shock]:
    """+/-``pct`` % moves of each FX spot that converts ``currencies`` to
    ``reporting`` (EURUSD for euros or a euro reporting currency, GBPUSD for
    pounds); none when every amount is in the reporting currency."""

    if all(c == reporting for c in currencies):
        return []
    pairs = [f"{c}USD" for c in ordered(list(currencies) + [reporting]) if c != "USD"]
    return [fx_spot({pair: sign * pct}) for pair in pairs for sign in (1.0, -1.0)]


def add_fx(shock: Shock, moves: Mapping[str, float]) -> Shock:
    """``shock`` with the FX moves ``moves`` (``{"EURUSD": 2}``, % of each
    spot against the dollar) added to its own: % moves of the same pair add
    (EURUSD +10% and +2% make EURUSD +12%), the other pairs keep their
    moves. The name and description say what was added (``Parallel +10bp
    + FX EURUSD +2%``), so a scenario's label always states its FX moves.
    Zero moves change nothing (``shock`` is returned as it is); a total move
    of -100% or less raises ``ValueError``."""

    extra = tuple((pair, pct) for pair, pct in _clean_fx(moves) if pct != 0.0)
    if not extra:
        return shock
    total = dict(shock.fx)
    for pair, pct in extra:
        total[pair] = total.get(pair, 0.0) + pct
    text = ", ".join(f"{p} {m:+g}%" for p, m in extra)
    note = f"FX {text} added" + (f" (FX moves now {', '.join(f'{p} {m:+g}%' for p, m in _clean_fx(total))})"
                                 if shock.fx else "")  # fmt: skip
    description = f"{shock.description}; {note}" if shock.description else note
    return replace(shock, name=f"{shock.name} + FX {text}", description=description, fx=tuple(total.items()))


def ibor_ois_basis(bp: float, indices: Optional[Sequence[str]] = None, **settings) -> Shock:
    """Projection curves' rate quotes move by ``bp`` (all, or ``indices``
    only) while the discount and Treasury curves stay: the IBOR-OIS basis
    widens. Tenor basis spreads are held, so a curve built from basis swaps
    on a moved index moves with it."""

    fields: Dict[str, object] = dict(curve=False, govt=False, projections=True, rates=True, spreads=False)
    label = "all projection curves" if not indices else ", ".join(indices)
    if indices:
        fields["profiles"] = tuple((f"{str(i).strip().upper()}:*", ((0.0, float(bp)),)) for i in indices)
        profile: List[Tuple[float, float]] = []
    else:
        profile = [(0.0, float(bp))]
    return _shock(f"IBOR-OIS basis {bp:+g}bp", profile, settings, description=f"{label} rate quotes {bp:+g}bp, "
                  "discount curve unchanged", **fields)  # fmt: skip


def tenor_basis(bp: float, indices: Optional[Sequence[str]] = None, **settings) -> Shock:
    """Only tenor basis swap spreads (``BASIS`` quotes, e.g. 3s6s) move by
    ``bp``: the shorter tenor's projection curve moves against the other."""

    fields: Dict[str, object] = dict(curve=False, govt=False, projections=True, rates=False, spreads=True)
    label = "every basis spread" if not indices else ", ".join(indices) + " basis spreads"
    if indices:
        fields["profiles"] = tuple((f"{str(i).strip().upper()}:*", ((0.0, float(bp)),)) for i in indices)
        profile: List[Tuple[float, float]] = []
    else:
        profile = [(0.0, float(bp))]
    return _shock(f"Tenor basis {bp:+g}bp", profile, settings, description=f"{label} {bp:+g}bp", **fields)


def key_rates(moves: Mapping[Union[str, float], float], **settings) -> Shock:
    """Moves at key tenors (``"10Y"`` or years), linear in between and flat
    beyond the first and last: the sum of Ho key-rate tents. To move one
    tenor only, give its neighbours at 0."""

    if not moves:
        raise ValueError("give at least one key-rate move")
    profile = sorted((_years(t), float(bp)) for t, bp in moves.items())
    text = ", ".join(f"{t} {float(bp):+g}" for t, bp in moves.items())
    return _shock(f"Key rates ({text})", profile, settings, description=f"key-rate moves in bp: {text}")


# --------------------------------------------------------------------------
# Historical episodes and windows
# --------------------------------------------------------------------------


def historical_episodes() -> List[str]:
    return [name for name, _, _ in EPISODES]


def _episode(name: str) -> Tuple[dt.date, dt.date]:
    for episode, start, end in EPISODES:
        if episode == name:
            return start, end
    raise ValueError(f"unknown episode {name!r}; expected one of {historical_episodes()}")


def _replay_source(selector: str, move, flag: str, note: str) -> ReplaySource:
    info = dataset_info(move.dataset)
    return ReplaySource(selector, move.dataset, info.title, move.start.isoformat(), move.end.isoformat(), flag, note)


def _swap_move(store: HistoryStore, currency: str, start: dt.date, end: dt.date, tolerance: int):
    """(move, flag, note) for a currency's swap curve: the user's history,
    then real swap/OIS history, then the proxies in order."""

    candidates = [(user_dataset(currency), "upload")]
    candidates += [(key, "real") for key in SWAP_HISTORY.get(currency, ())]
    candidates += [(key, "proxy" if dataset_info(key).currency == currency else "cross-currency proxy")
                   for key in SWAP_PROXIES.get(currency, ("UST_PAR",))]  # fmt: skip
    errors = []
    for key, flag in candidates:
        if not store.has(key):
            continue
        try:
            move = store.curve_move(key, start, end, tolerance)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        info = dataset_info(key)
        if flag == "upload":
            note = f"your uploaded {currency} swap history ({move.what})"
        elif flag == "real":
            note = f"{info.title}: {move.what}"
            if key == "USD_SWAP_ICE":
                note += " (LIBOR swaps: their moves stand in for SOFR swap moves before SOFR swaps traded)"
        elif flag == "proxy":
            note = (
                f"no free {currency} swap history for these dates: {info.title} {move.what} "
                "as a proxy (swap spreads held)"
            )
        else:
            note = (
                f"no {currency} history for these dates: {info.title} {move.what} as a cross-currency proxy; "
                f"upload {currency} swap history for a real replay"
            )
        return move, flag, note
    raise ValueError(f"no {currency} swap-curve or proxy history for {start} to {end}: " + "; ".join(errors))


def replay_fx(
    store: HistoryStore, start: dt.date, end: dt.date, tolerance: int = EPISODE_TOLERANCE_DAYS, scale: float = 1.0
) -> Tuple[Tuple[Tuple[str, float], ...], ReplaySource]:
    """(``Shock.fx`` moves, their source): the % moves of EURUSD and GBPUSD
    between the ECB reference rates nearest ``start`` and ``end`` (within
    ``tolerance`` days), times ``scale``; no moves (flag ``held``, with the
    reason) when the store's FX history does not cover both dates."""

    info = dataset_info(FX_DATASET)
    selector = "fx:" + ",".join(SHOCK_PAIRS)
    try:
        d0 = store.nearest(FX_DATASET, start, tolerance)
        d1 = store.nearest(FX_DATASET, end, tolerance)
        if d1 <= d0:
            raise ValueError(f"the window maps to a single ECB rate date ({d0})")
    except ValueError as exc:
        note = f"FX spots held: {exc}"
        return (), ReplaySource(selector, FX_DATASET, info.title, start.isoformat(), end.isoformat(), "held", note)
    table = store.fx_table()
    a, b = table.loc[d0], table.loc[d1]
    eurusd = 100.0 * (b["USD"] / a["USD"] - 1.0) * scale
    gbpusd = 100.0 * ((b["USD"] / b["GBP"]) / (a["USD"] / a["GBP"]) - 1.0) * scale
    moves = (("EURUSD", float(eurusd)), ("GBPUSD", float(gbpusd)))
    note = (
        f"ECB reference rates {d0} to {d1}: EURUSD {eurusd:+.2f}%, GBPUSD {gbpusd:+.2f}%"
        + (f" (x{scale:g})" if scale != 1.0 else "")
        + "; they move the totals converted to the reporting currency, not local P&L"
    )
    return moves, ReplaySource(selector, FX_DATASET, info.title, d0.isoformat(), d1.isoformat(), "real", note)


def _replay(
    name: str,
    start: dt.date,
    end: dt.date,
    scale: float,
    store: Optional[HistoryStore],
    tolerance: int,
    settings,
    fx: bool = True,
) -> Shock:
    store = store or bundled_store()
    if not math.isfinite(scale):
        raise ValueError("the scale must be a finite number")
    govt = store.curve_move(GOVT_HISTORY["USD"], start, end, tolerance)
    profiles, sources = [], [_replay_source("govt:USD", govt, "real", f"US Treasury {govt.what}")]
    for currency in CURRENCIES:
        move, flag, note = _swap_move(store, currency, start, end, tolerance)
        profiles.append((f"curve:{currency}", tuple((y, b * scale) for y, b in move.profile)))
        sources.append(_replay_source(f"curve:{currency}", move, flag, note))
        indices = CURRENCY_INDICES.get(currency, ())
        if indices:
            sources.append(
                _replay_source(
                    f"projections:{currency}",
                    move,
                    "proxy",
                    f"no free {' / '.join(indices)} swap or basis history: the projection curves move with the "
                    f"{currency} swap curve above ({flag}), so IBOR-OIS and tenor basis are held",
                )
            )
    label = name if scale == 1.0 else f"{name} x{scale:g}"
    fx_moves: Tuple[Tuple[str, float], ...] = ()
    description = f"historical curve changes {start} to {end}, each curve from its own data (see sources)"
    if fx:
        fx_moves, fx_source = replay_fx(store, govt.start, govt.end, tolerance, scale)
        sources.append(fx_source)
        description += f"; {fx_source.note}"
    else:
        description += "; FX spots held (as asked)"
    return _shock(
        label,
        [(y, b * scale) for y, b in govt.profile],
        settings,
        description=description,
        profiles=tuple(profiles),
        sources=tuple(sources),
        fx=fx_moves,
    )


def historical(
    name: str, scale: float = 1.0, store: Optional[HistoryStore] = None, fx: bool = True, **settings
) -> Shock:
    """Replay a named episode (:data:`EPISODES`) times ``scale``: the
    Treasury curve by US Treasury par yield changes, each swap curve by its
    currency's swap/OIS history or a flagged proxy, and (``fx``) EURUSD and
    GBPUSD by the ECB reference-rate moves of the same dates (see
    ``Shock.sources``). ``store`` defaults to the committed history sample
    (no network)."""

    start, end = _episode(name)
    return _replay(name, start, end, float(scale), store, EPISODE_TOLERANCE_DAYS, settings, fx=fx)


def history_window(
    start: dt.date,
    end: dt.date,
    scale: float = 1.0,
    store: Optional[HistoryStore] = None,
    tolerance_days: int = 7,
    fx: bool = True,
    **settings,
) -> Shock:
    """Replay the curve changes between any two dates of the history store
    (nearest complete curves within ``tolerance_days``) and, with ``fx``,
    the FX moves of the same dates, like :func:`historical`."""

    if end <= start:
        raise ValueError(f"the window end ({end}) must be after its start ({start})")
    return _replay(f"History {start} to {end}", start, end, float(scale), store, tolerance_days, settings, fx=fx)


def historical_table(store: Optional[HistoryStore] = None) -> pd.DataFrame:
    """US Treasury moves of every episode: ``episode, start, end,
    tenor_years, change_bp, source`` (start/end are the dates used)."""

    store = store or bundled_store()
    rows = []
    for name, start, end in EPISODES:
        move = store.curve_move(GOVT_HISTORY["USD"], start, end, EPISODE_TOLERANCE_DAYS)
        for years, bp, series in zip(move.years, move.change_bp, move.series):
            rows.append(
                {
                    "episode": name,
                    "start": move.start.isoformat(),
                    "end": move.end.isoformat(),
                    "tenor_years": years,
                    "change_bp": bp,
                    "source": series,
                }
            )
    return pd.DataFrame(rows, columns=["episode", "start", "end", "tenor_years", "change_bp", "source"])


def historical_sources(shock: Shock) -> pd.DataFrame:
    """One row per curve and currency of a historical shock: which data it
    replays, the dates used and whether it is the curve's own market."""

    names = {"govt": "TREASURY", "projections": "PROJECTION", "fx": "FX SPOTS"}
    rows = [
        {
            "curve": names.get(src.selector.split(":", 1)[0], "SWAP"),
            "currency": src.selector.split(":", 1)[1],
            "data": src.title,
            "start": src.start,
            "end": src.end,
            "flag": src.flag,
            "note": src.note,
        }
        for src in shock.sources
    ]
    return pd.DataFrame(rows, columns=["curve", "currency", "data", "start", "end", "flag", "note"])


def quoted_levels(
    store: HistoryStore, dataset: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None
) -> pd.DataFrame:
    """Decimal levels of ``dataset`` on its complete-curve dates in [start,
    end] (one column per tenor, shortest first), in the rates a curve is
    quoted in: a zero-curve dataset with OIS conventions (the Bank of
    England SONIA curve) gives its derived par OIS rates
    (:meth:`HistoryStore.ois_par`), any other its own measure."""

    info = dataset_info(dataset)
    if info.measure == "zero_cc" and info.ois_conventions:
        days = [d for d in store.dates(dataset) if (start is None or d >= start) and (end is None or d <= end)]
        frame = pd.DataFrame({d: store.ois_par(dataset, d) for d in days}).T
    else:
        frame = store.complete(dataset)
        frame = frame[[(start is None or d >= start) and (end is None or d <= end) for d in frame.index]]
    if frame.empty:
        return frame
    frame = frame.sort_index()
    return frame[sorted(frame.columns, key=lambda t: parse_tenor(t).years)].astype(float)


def _pca_flag(dataset: str, selector: str) -> str:
    """How a PCA profile from ``dataset`` relates to the curve ``selector``
    (``"govt:USD"``, ``"curve:EUR"``)."""

    which, currency = selector.split(":", 1)
    if which == "govt":
        own = (GOVT_HISTORY.get(currency),)
    else:
        own = SWAP_HISTORY.get(currency, ()) + (user_dataset(currency),)
    if dataset in own:
        return "real"
    return "proxy" if dataset_info(dataset).currency == currency else "cross-currency proxy"


def pca_factor(
    factor: int,
    n_sigma: float = 2.0,
    store: Optional[HistoryStore] = None,
    dataset: str = GOVT_HISTORY["USD"],
    tenors: Optional[Sequence[str]] = None,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    window: Optional[int] = None,
    days: int = 1,
    **settings,
) -> Shock:
    """``n_sigma`` standard deviations of PCA factor ``factor`` (1 = level,
    2 = slope, 3 = curvature) of a history's daily changes, scaled by
    sqrt(``days``) (independent daily moves assumed): the profile is
    ``n_sigma x sd_k x sqrt(days) x loading_k`` in bp at the PCA tenors
    (linear in between, flat beyond), applied to every curve the shock
    moves. The PCA (:func:`derivx.rates.pca.pca_curve_changes`, demeaned bp
    changes) runs on ``dataset`` from ``start`` (default: where its most
    recent daily stretch begins) to ``end``, keeping the last ``window``
    changes if given; ``tenors`` default to 3M, 1Y, 2Y, 5Y, 10Y, 30Y where
    the history has them. ``Shock.sources`` says which curves the history is
    the curve's own market for."""

    store = store or bundled_store()
    if int(factor) != factor or factor < 1:
        raise ValueError("the factor is 1 (level), 2 (slope), 3 (curvature) or higher")
    if not store.has(dataset):
        raise ValueError(f"the history store has no {dataset} data ({dataset_info(dataset).title})")
    levels = store.complete(dataset)
    if tenors is None:
        tenors = [t for t in pca_mod.DEFAULT_TENORS if t in levels.columns]
        if len(tenors) < 3:
            tenors = list(dataset_info(dataset).core_tenors)
    if start is None:
        start = pca_mod.recent_daily_start(levels, tenors, end=end)
    info = dataset_info(dataset)
    quoted = "par yields" if info.measure.startswith("par") else "rates"
    if info.measure == "zero_cc" and info.ois_conventions:
        # The swap curve is quoted in par OIS rates: analyse those (as the historical replay and the VaR do).
        levels = quoted_levels(store, dataset, start, end)
        quoted = "derived par OIS rates"
    result = pca_mod.pca_curve_changes(levels, tenors, window, start=start, end=end)
    if factor > len(result.tenors):
        raise ValueError(f"{len(result.tenors)} tenors give {len(result.tenors)} factors, not PC{factor}")
    profile = result.factor_profile(int(factor) - 1, float(n_sigma), int(days))
    label = pca_mod.FACTOR_LABELS[factor - 1] if factor <= len(pca_mod.FACTOR_LABELS) else ""
    sd = float(result.sigma_bp[factor - 1])
    span = f"{result.start} to {result.end}"
    shape = ", ".join(f"{t} {b:+.1f}" for t, (_, b) in zip(result.tenors, profile))
    selectors = ["govt:USD"] + [f"curve:{c}" for c in CURRENCIES]
    sources = []
    for selector in selectors:
        flag = _pca_flag(dataset, selector)
        note = f"PC{factor} of {info.title} daily changes in {quoted} ({span}, {result.n_obs} changes)"
        if flag != "real":
            note += f": applied to this curve as a {flag}"
        sources.append(ReplaySource(selector, dataset, info.title, str(result.start), str(result.end), flag, note))
    days_text = "" if days == 1 else f" x sqrt({int(days)} days)"
    return _shock(
        f"PC{factor} {label.lower()} {n_sigma:+g} sd{'' if days == 1 else f' {int(days)}d'}".replace("  ", " "),
        list(profile),
        settings,
        description=(
            f"PC{factor} ({label.lower() or 'factor'}, {100 * result.explained[factor - 1]:.1f}% of the variance) of "
            f"daily {info.title} changes in {quoted} {span} ({result.n_obs} changes): sd {sd:.2f}bp a day, move = "
            f"{n_sigma:+g} sd{days_text} x loading, in bp: {shape}"
        ),
        sources=tuple(sources),
    )


def pca_library(store: Optional[HistoryStore] = None, n_sigma: float = 2.0, **kwargs) -> List[Shock]:
    """+/-``n_sigma`` sd of PC1, PC2 and PC3 (see :func:`pca_factor`)."""

    return [pca_factor(k, sign * n_sigma, store, **kwargs) for k in (1, 2, 3) for sign in (1.0, -1.0)]


def factor_risk(
    market: Market, risk: Mapping[str, QuoteRisk], result: pca_mod.PCAResult, n_factors: int = 3
) -> np.ndarray:
    """P&L (factors x positions) of every curve in ``risk`` moving along each
    of the first ``n_factors`` loadings by a score of +1 (tenor i up
    ``L_ik`` bp): the delta of a :func:`pca_factor` shock of one sd divided
    by the sd. Rate quotes move by the loading read at their maturity
    (:func:`derivx.rates.pca.factor_pv01`); ``BASIS`` spread quotes are held,
    as a shock and the value at risk hold them."""

    total: Optional[np.ndarray] = None
    for which, qr in risk.items():
        curve = market.get_curve(which)
        if len(curve.instruments) != qr.pv01.shape[0]:
            raise ValueError(f"the {which!r} key-rate risk does not match the curve's {len(curve.instruments)} quotes")
        rates = np.array([inst.kind not in SPREAD_KINDS for inst in curve.instruments], dtype=float)
        part = pca_mod.factor_pv01(qr.pv01 * rates[:, None], qr.years, result, n_factors)
        total = part if total is None else total + part
    if total is None:
        raise ValueError("no key-rate risk to combine with the factors")
    return total


def standard_library(store: Optional[HistoryStore] = None, fx: bool = True) -> List[Shock]:
    """Parallel +/-25 and +/-100bp, 2s10s steepener/flattener of 25bp, a
    2s5s10s fly and every historical episode (instantaneous, both curves;
    episodes from ``store``, by default the committed sample, with their FX
    moves unless ``fx`` is False)."""

    shocks = [parallel(bp) for bp in (25.0, -25.0, 100.0, -100.0)]
    shocks += [
        twist(5.0, -12.5, 12.5, short_years=2.0, long_years=10.0, name="2s10s steepener 25bp"),
        twist(5.0, 12.5, -12.5, short_years=2.0, long_years=10.0, name="2s10s flattener 25bp"),
        butterfly(0.0, 25.0, 5.0, wings=(2.0, 10.0), name="2s5s10s fly: 5Y +25bp"),
    ]
    return shocks + [historical(name, store=store, fx=fx) for name in historical_episodes()]


# --------------------------------------------------------------------------
# Shocked markets
# --------------------------------------------------------------------------


def horizon_date(market: Market, days: int) -> Date:
    """``days`` calendar days after the valuation date, moved to the next
    business day of the swap curve's calendar."""

    if days < 0:
        raise ValueError("the horizon must be 0 or more days ahead")
    if days == 0:
        return market.asof
    target = market.asof + dt.timedelta(days=int(days))
    conv = market.curve.conventions
    return conv.cal.adjust(target, "F") if conv is not None else target


def shocked_curves(market: Market) -> Tuple[str, ...]:
    """Curves a shock can move: every key of the market except a projection
    that is the discount curve itself (it moves with it)."""

    return tuple(k for k in market.curve_keys if k in (DISCOUNT_KEY, GOVT_KEY) or not market.projections.is_discount(k))


def _bumps(market: Market, shock: Shock) -> Dict[str, np.ndarray]:
    """bp move of every quote of each curve the shock moves (non-zero only)."""

    out: Dict[str, np.ndarray] = {}
    if not shock.moves_rates:
        return out
    for which in shocked_curves(market):
        if not shock.applies_to(which):
            continue
        curve = market.get_curve(which)
        if not curve.instruments:
            raise ValueError(f"the {which!r} curve has no quotes to shock; bootstrap it from quotes")
        bumps = shock.quote_bumps(curve, which)
        if np.any(bumps != 0.0):
            out[which] = bumps
    return out


def _moved(
    market: Market,
    bumps: Mapping[str, np.ndarray],
    at: Date,
    normal_bp: float = 0.0,
    lognormal_pts: float = 0.0,
) -> Market:
    """Market with curves rebuilt on ``at`` from bumped quotes (projection
    curves rebuilt on the moved discount curve); past fixings stay on
    today's curves."""

    moved = market.rebuilt({k: list(b) for k, b in bumps.items()}, asof=at)
    return moved.replace(
        fixings=market.fixing_curve,
        vol_shift_normal=market.vol_shift_normal + normal_bp * NORMAL_VOL_UNIT,
        vol_shift_lognormal=market.vol_shift_lognormal + lognormal_pts * LOGNORMAL_VOL_UNIT,
    )


def shocked_market(market: Market, shock: Shock, at_horizon: bool = True) -> Market:
    """The market after the shock: quotes moved, curves rebuilt (on the
    horizon date, or today with ``at_horizon=False``) and vols shifted."""

    at = horizon_date(market, shock.horizon_days) if at_horizon else market.asof
    return _moved(market, _bumps(market, shock), at, shock.normal_vol_bp, shock.lognormal_vol_pts)


def carry_market(market: Market, days: int) -> Market:
    """Today's quotes rebuilt on the horizon date (carry and roll-down)."""

    return _moved(market, {}, horizon_date(market, days))


def theta(positions: Sequence[Position], market: Market, days: int = 1) -> np.ndarray:
    """One-day theta of each position: value on :func:`carry_market` (the
    market rolled to ``days`` calendar days ahead, moved to the next business
    day, quotes and vols unchanged) plus cash received, minus value today.
    For an option it is time decay plus the carry and roll-down of its
    forwards and discount factors (and, for vols keyed by time to expiry,
    the roll down the vol curve)."""

    positions = list(positions)
    base = _pvs(positions, market)
    rolled = carry_market(market, days)
    pv, cash = _revalue(positions, market, rolled)
    return pv + cash - base


# --------------------------------------------------------------------------
# Sensitivities
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuoteRisk:
    """Key-rate PV01s and diagonal gammas of positions to one curve's quotes
    on ``asof`` (rows = quotes, columns = positions): P&L per +1bp and per
    (1bp)^2."""

    which: str
    labels: Tuple[str, ...]
    years: np.ndarray
    pv01: np.ndarray
    gamma: np.ndarray
    computed: np.ndarray = field(repr=False)  # quotes actually bumped (others are zero)
    asof: Optional[Date] = None


def _pvs(positions: Sequence[Position], market: Market) -> np.ndarray:
    values = np.array([float(p.pv(market)) for p in positions])
    bad = [p.name for p, v in zip(positions, values) if not math.isfinite(v)]
    if bad:
        raise ValueError(f"no finite value on {market.asof} for {', '.join(bad)}")
    return values


def _unit(n: int, i: int, bp: float) -> np.ndarray:
    out = np.zeros(n)
    out[i] = bp
    return out


def quote_risk(
    positions: Sequence[Position],
    market: Market,
    which: str = "curve",
    indices: Optional[Sequence[int]] = None,
    base: Optional[np.ndarray] = None,
    bp: float = 1.0,
) -> QuoteRisk:
    """PV01 and gamma of each position to each quote of ``market.<which>``
    from one +/-``bp`` pair of rebuilds per quote (only ``indices`` if given;
    the other rows are zero). ``base`` reuses today's PVs."""

    curve = market.get_curve(which) if which in market.curve_keys else None
    if curve is None or not curve.quotes:
        raise ValueError(f"market has no {which!r} curve with quotes to bump")
    if not (math.isfinite(bp) and bp > 0):
        raise ValueError("the bump must be a positive number of bp")
    n = len(curve.quotes)
    base = _pvs(positions, market) if base is None else np.asarray(base, dtype=float)
    pv01, gamma = np.zeros((n, len(positions))), np.zeros((n, len(positions)))
    computed = np.zeros(n, dtype=bool)
    for i in range(n) if indices is None else sorted(set(int(i) for i in indices)):
        up = _pvs(positions, _moved(market, {which: _unit(n, i, bp)}, market.asof))
        down = _pvs(positions, _moved(market, {which: _unit(n, i, -bp)}, market.asof))
        pv01[i] = (up - down) / (2.0 * bp)
        gamma[i] = (up + down - 2.0 * base) / (bp * bp)
        computed[i] = True
    return QuoteRisk(
        which=which,
        labels=tuple(curve.labels),
        years=curve.years,
        pv01=pv01,
        gamma=gamma,
        computed=computed,
        asof=market.asof,
    )


def vegas(positions: Sequence[Position], market: Market) -> np.ndarray:
    """P&L per +1bp of normal vol (``vol_model="normal"``) or +1 vol point of
    lognormal vol (``"lognormal"``) by bumping the matching Market vol shift
    (central difference); zero for positions without a vol model."""

    out = np.zeros(len(positions))
    for k, position in enumerate(positions):
        if position.vol_model == "normal":
            name, unit = "vol_shift_normal", NORMAL_VOL_UNIT
        elif position.vol_model == "lognormal":
            name, unit = "vol_shift_lognormal", LOGNORMAL_VOL_UNIT
        else:
            continue
        level = getattr(market, name)
        up = position.pv(market.replace(**{name: level + unit}))
        down = position.pv(market.replace(**{name: level - unit}))
        out[k] = (up - down) / 2.0
    return out


# --------------------------------------------------------------------------
# Scenario P&L
# --------------------------------------------------------------------------


def _revalue(positions: Sequence[Position], market: Market, moved: Market) -> Tuple[np.ndarray, np.ndarray]:
    """(PV on ``moved``, cash each position pays between the two dates)."""

    pv = _pvs(positions, moved)
    if moved.asof == market.asof:
        return pv, np.zeros(len(positions))
    cash = np.array([float(p.cash(moved, market.asof, moved.asof)) if p.cash else 0.0 for p in positions])
    return pv, cash


def run_scenario(
    positions: Sequence[Position],
    market: Market,
    shock: Shock,
    risk: Optional[Mapping[str, QuoteRisk]] = None,
) -> List[Dict[str, object]]:
    """Full revaluation and P&L explain of each position (see the module
    docstring): one row per position with ``position``, ``kind``,
    ``base_pv``, ``shocked_pv`` (value at the horizon), ``cash`` (paid in the
    window), ``pnl``, the explain columns ``delta``, ``gamma``,
    ``cross_gamma``, ``vol``, ``carry``, ``residual``, ``delta_by_curve``
    ({curve key: delta from that curve's quote moves}, summing to ``delta``)
    and ``currency`` (that of every amount of the row,
    :func:`~derivx.risk.reporting.position_currency`). The FX moves of the
    shock change no local amount (see the module docstring).

    ``risk`` ({curve field: :func:`quote_risk`}) reuses key-rate risk
    precomputed on the market the shock hits (``market`` itself, or
    :func:`carry_market` for a horizon); risk on another date is ignored and
    only the quotes the shock moves are bumped.
    """

    positions = list(positions)
    if not positions:
        return []
    n = len(positions)
    base = _pvs(positions, market)
    if shock.horizon_days:
        rolled = carry_market(market, shock.horizon_days)
        rolled_pv, rolled_cash = _revalue(positions, market, rolled)
        carry = rolled_pv + rolled_cash - base
    else:
        rolled, rolled_pv, carry = market, base, np.zeros(n)
    bumps = _bumps(rolled, shock)

    delta, gamma = np.zeros(n), np.zeros(n)
    delta_by_curve: Dict[str, np.ndarray] = {}
    for which, b in bumps.items():
        moved = np.flatnonzero(b)
        qr = (risk or {}).get(which)
        if qr is None or qr.asof != rolled.asof or qr.pv01.shape != (len(b), n) or not qr.computed[moved].all():
            qr = quote_risk(positions, rolled, which, indices=moved, base=rolled_pv)
        delta_by_curve[which] = b @ qr.pv01
        delta += delta_by_curve[which]
        gamma += 0.5 * (b * b) @ qr.gamma

    cross = np.zeros(n)
    if bumps:
        scale = max(float(np.max(np.abs(b))) for b in bumps.values())
        up = _pvs(positions, _moved(rolled, {w: b / scale for w, b in bumps.items()}, rolled.asof))
        down = _pvs(positions, _moved(rolled, {w: -b / scale for w, b in bumps.items()}, rolled.asof))
        cross = 0.5 * (up + down - 2.0 * rolled_pv) * scale * scale - gamma

    # Vega only where the shock moves the position's vol (it is P&L per unit).
    vol_moves = {"normal": shock.normal_vol_bp, "lognormal": shock.lognormal_vol_pts}
    steps = np.array([vol_moves.get(p.vol_model, 0.0) for p in positions])
    vol = np.zeros(n)
    live = np.flatnonzero(steps)
    if live.size:
        vol[live] = vegas([positions[k] for k in live], rolled) * steps[live]

    shocked = _moved(rolled, bumps, rolled.asof, shock.normal_vol_bp, shock.lognormal_vol_pts)
    shocked_pv, cash = _revalue(positions, market, shocked)
    pnl = shocked_pv + cash - base
    residual = pnl - (carry + delta + gamma + cross + vol)
    columns = (base, shocked_pv, cash, pnl, delta, gamma, cross, vol, carry, residual)
    currencies = position_currencies(positions, market)
    return [
        {
            "position": p.name,
            "kind": p.kind,
            "currency": currencies[k],
            **dict(zip(ROW_COLUMNS[2:], (float(c[k]) for c in columns))),
            "delta_by_curve": {which: float(v[k]) for which, v in delta_by_curve.items()},
        }
        for k, p in enumerate(positions)
    ]


def _unique(names: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


def scenario_grid(
    positions: Sequence[Position],
    market: Market,
    shocks: Sequence[Shock],
    reporting: Optional[Reporting] = None,
) -> pd.DataFrame:
    """Full-revaluation P&L (value at the horizon + cash - value today):
    rows = shocks, columns = positions (each in its currency,
    :func:`~derivx.risk.reporting.position_currency`), then a total per
    currency (``TOTAL``, or ``TOTAL USD``, ``TOTAL EUR``... for a book in
    several) and, with ``reporting`` when some position is in another
    currency, ``ALL IN <R>``: the P&L converted at today's spots plus the FX
    translation of FX shocks. A shock that moves no quote, vol or horizon
    (an FX shock) leaves every local value unchanged, without revaluing."""

    positions = list(positions)
    base = _pvs(positions, market)
    columns = _unique([p.name for p in positions])
    rows = []
    for shock in shocks:
        if not shock.moves_local:
            rows.append(np.zeros(len(positions)))
            continue
        pv, cash = _revalue(positions, market, shocked_market(market, shock))
        rows.append(pv + cash - base)
    table = pd.DataFrame(rows, index=_unique([s.name for s in shocks]), columns=columns, dtype=float)
    if not positions:
        table[TOTAL] = 0.0
    else:
        table = add_total_columns(table, position_currencies(positions, market), reporting, base,
                                  [s.fx for s in shocks])  # fmt: skip
    table.index.name = "scenario"
    return table
