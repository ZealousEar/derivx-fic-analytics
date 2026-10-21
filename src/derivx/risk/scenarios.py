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

The key-rate gammas reuse the +/-1bp revaluations of the PV01s. On a
bootstrapped curve the cross-gammas are not small (the 10Y swap's annuity
moves with every shorter quote; diagonal gammas alone leave ~3% of a 100bp
parallel move on a 10Y swap unexplained), so ``cross_gamma`` is measured
with one extra pair of revaluations along the shock direction.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from derivx.rates.conventions import Date, parse_tenor
from derivx.rates.curves import Curve, Market
from derivx.rates.examples import DATA_DIR
from derivx.risk.sensitivities import CURVE_FIELDS, Position

HISTORICAL_CSV = DATA_DIR / "historical_curve_shocks.csv"
NORMAL_VOL_UNIT = 1e-4  # 1bp of normal vol
LOGNORMAL_VOL_UNIT = 0.01  # 1 lognormal vol point
EXPLAIN_COLUMNS = ("delta", "gamma", "cross_gamma", "vol", "carry", "residual")
ROW_COLUMNS = ("position", "kind", "base_pv", "shocked_pv", "cash", "pnl") + EXPLAIN_COLUMNS
TOTAL = "TOTAL"


# --------------------------------------------------------------------------
# Shocks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Shock:
    """A scenario: quote moves by tenor, vol shifts and a horizon.

    ``profile`` is ``((years, bp), ...)``, linear in between and flat
    outside. ``short_bp``/``long_bp`` (optional) pin the move at the
    shortest/longest quote of each curve being shocked, so a twist reaches
    its end values on any curve. ``curve``/``govt`` say which of the market's
    curves move. ``normal_vol_bp`` is added to normal vols (bp) and
    ``lognormal_vol_pts`` to lognormal vols (vol points). ``horizon_days``
    rolls the market forward (calendar days, adjusted to a business day).
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

    def __post_init__(self) -> None:
        points = tuple(sorted((float(y), float(b)) for y, b in self.profile))
        years = [y for y, _ in points]
        if any(not math.isfinite(v) for p in points for v in p) or any(y < 0 for y in years):
            raise ValueError("profile points must be finite (years >= 0, bp)")
        if len(set(years)) != len(years):
            raise ValueError(f"profile has two moves at the same tenor: {years}")
        object.__setattr__(self, "profile", points)
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

    def points(self, span: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(years, bp) of the profile, with the end anchors placed at
        ``span`` = (shortest, longest) quote years when given."""

        xs = [y for y, _ in self.profile]
        ys = [b for _, b in self.profile]
        if span is not None:
            lo, hi = span
            if self.short_bp is not None and (not xs or lo < xs[0]):
                xs, ys = [lo] + xs, [self.short_bp] + ys
            if self.long_bp is not None and (not xs or hi > xs[-1]):
                xs, ys = xs + [hi], ys + [self.long_bp]
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def bp_at(
        self, years: Union[float, Sequence[float], np.ndarray], span: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Move in bp at tenors in years (flat beyond the profile)."""

        xs, ys = self.points(span)
        years = np.atleast_1d(np.asarray(years, dtype=float))
        if not xs.size:
            return np.zeros_like(years)
        return np.interp(years, xs, ys)

    def quote_bumps(self, curve: Curve) -> np.ndarray:
        """bp move of each of ``curve.quotes``."""

        if not curve.instruments:
            raise ValueError("the curve has no quotes to shock; bootstrap it from quotes")
        years = np.array([inst.tenor.years for inst in curve.instruments])
        return self.bp_at(years, (float(years.min()), float(years.max())))

    @property
    def moves_rates(self) -> bool:
        """True if some quote of a curve the shock applies to moves."""

        moves = [b for _, b in self.profile] + [b for b in (self.short_bp, self.long_bp) if b is not None]
        return (self.curve or self.govt) and any(b != 0.0 for b in moves)


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
# Historical episodes
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _historical() -> pd.DataFrame:
    # round_trip keeps 1/12 years exact, so the 1M move lands on 1M quotes.
    frame = pd.read_csv(HISTORICAL_CSV, float_precision="round_trip")
    missing = {"episode", "start", "end", "tenor_years", "change_bp", "source"} - set(frame.columns)
    if missing:
        raise ValueError(f"{HISTORICAL_CSV.name} is missing columns {sorted(missing)}")
    return frame


def historical_table() -> pd.DataFrame:
    """The shipped episodes: ``episode, start, end, tenor_years, change_bp,
    source`` (US Treasury constant-maturity yield changes from FRED)."""

    return _historical().copy()


def historical_episodes() -> List[str]:
    return list(dict.fromkeys(_historical()["episode"]))


def historical(name: str, scale: float = 1.0, **settings) -> Shock:
    """Replay a historical episode's Treasury yield changes (times
    ``scale``) on the quotes; read from data/examples, no network."""

    frame = _historical()
    rows = frame[frame["episode"] == name]
    if rows.empty:
        raise ValueError(f"unknown episode {name!r}; expected one of {historical_episodes()}")
    start, end = rows["start"].iloc[0], rows["end"].iloc[0]
    label = name if scale == 1.0 else f"{name} x{scale:g}"
    return _shock(
        label,
        zip(rows["tenor_years"], rows["change_bp"] * float(scale)),
        settings,
        description=f"UST constant-maturity yield changes {start} to {end} (FRED)",
    )


def standard_library() -> List[Shock]:
    """Parallel +/-25 and +/-100bp, 2s10s steepener/flattener of 25bp, a
    2s5s10s fly and every historical episode (instantaneous, both curves)."""

    shocks = [parallel(bp) for bp in (25.0, -25.0, 100.0, -100.0)]
    shocks += [
        twist(5.0, -12.5, 12.5, short_years=2.0, long_years=10.0, name="2s10s steepener 25bp"),
        twist(5.0, 12.5, -12.5, short_years=2.0, long_years=10.0, name="2s10s flattener 25bp"),
        butterfly(0.0, 25.0, 5.0, wings=(2.0, 10.0), name="2s5s10s fly: 5Y +25bp"),
    ]
    return shocks + [historical(name) for name in historical_episodes()]


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


def _bumps(market: Market, shock: Shock) -> Dict[str, np.ndarray]:
    """bp move of every quote of each curve the shock moves (non-zero only)."""

    out: Dict[str, np.ndarray] = {}
    if not shock.moves_rates:
        return out
    for which in CURVE_FIELDS:
        curve = getattr(market, which)
        if curve is None or not getattr(shock, which):
            continue
        if not curve.instruments:
            raise ValueError(f"the {which!r} curve has no quotes to shock; bootstrap it from quotes")
        bumps = shock.quote_bumps(curve)
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
    """Market with curves rebuilt on ``at`` from bumped quotes; past fixings
    stay on today's curve."""

    rolled = at != market.asof
    if rolled and market.fixings is not None:
        raise ValueError("cannot roll a market that already has a separate fixings curve")
    changes: Dict[str, object] = {}
    for which in CURVE_FIELDS:
        curve = getattr(market, which)
        if curve is not None and (which in bumps or rolled):
            if not curve.quotes:
                raise ValueError(f"cannot roll the {which!r} curve to {at}: it has no quotes to rebuild from")
            changes[which] = curve.rebuilt(list(bumps.get(which, np.zeros(len(curve.quotes)))), asof=at)
    return market.replace(
        fixings=market.fixing_curve,
        vol_shift_normal=market.vol_shift_normal + normal_bp * NORMAL_VOL_UNIT,
        vol_shift_lognormal=market.vol_shift_lognormal + lognormal_pts * LOGNORMAL_VOL_UNIT,
        **changes,
    )


def shocked_market(market: Market, shock: Shock, at_horizon: bool = True) -> Market:
    """The market after the shock: quotes moved, curves rebuilt (on the
    horizon date, or today with ``at_horizon=False``) and vols shifted."""

    at = horizon_date(market, shock.horizon_days) if at_horizon else market.asof
    return _moved(market, _bumps(market, shock), at, shock.normal_vol_bp, shock.lognormal_vol_pts)


def carry_market(market: Market, days: int) -> Market:
    """Today's quotes rebuilt on the horizon date (carry and roll-down)."""

    return _moved(market, {}, horizon_date(market, days))


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

    curve = getattr(market, which, None) if which in CURVE_FIELDS else None
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
        years=np.array([inst.tenor.years for inst in curve.instruments]),
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
    window), ``pnl`` and the explain columns ``delta``, ``gamma``,
    ``cross_gamma``, ``vol``, ``carry``, ``residual``.

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
    for which, b in bumps.items():
        moved = np.flatnonzero(b)
        qr = (risk or {}).get(which)
        if qr is None or qr.asof != rolled.asof or qr.pv01.shape != (len(b), n) or not qr.computed[moved].all():
            qr = quote_risk(positions, rolled, which, indices=moved, base=rolled_pv)
        delta += b @ qr.pv01
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
    return [
        {"position": p.name, "kind": p.kind, **dict(zip(ROW_COLUMNS[2:], (float(c[k]) for c in columns)))}
        for k, p in enumerate(positions)
    ]


def _unique(names: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


def scenario_grid(positions: Sequence[Position], market: Market, shocks: Sequence[Shock]) -> pd.DataFrame:
    """Full-revaluation P&L (value at the horizon + cash - value today):
    rows = shocks, columns = positions and ``TOTAL``."""

    positions = list(positions)
    base = _pvs(positions, market)
    columns = _unique([p.name for p in positions])
    rows = []
    for shock in shocks:
        pv, cash = _revalue(positions, market, shocked_market(market, shock))
        rows.append(pv + cash - base)
    table = pd.DataFrame(rows, index=_unique([s.name for s in shocks]), columns=columns)
    table[TOTAL] = table.sum(axis=1)
    table.index.name = "scenario"
    return table
