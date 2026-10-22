"""Rate sensitivities by bumping curve quotes and rebuilding.

Every number here is the signed P&L of a move in the market *quotes* (the
way a rates desk sees risk), computed by central differences on full
revaluation:

* ``parallel_pv01`` - P&L for +1bp on every quote of a curve
* ``key_rate_pv01`` - the same, one quote at a time (the risk ladder)
* ``parallel_gamma`` - second difference, P&L per (1bp)^2
* ``rate_pv01_gamma`` - PV01 and gamma for +1bp on every rate quote of the
  discount and projection curves together (basis spreads held)
* ``solve_hedge`` - notionals of hedge instruments that flatten a ladder

A long receiver swap or long bond has negative PV01 (it loses when rates
rise); a long cap has positive PV01.

``which`` names one curve of the market: ``"curve"`` (the discount curve),
``"govt"`` (Treasury) or a projection index such as ``"EURIBOR 6M"``. Bumping
the discount curve rebuilds the projection curves bootstrapped on it with
their own quotes held, so a EURIBOR swap's discount-curve ladder includes
the projection curve's response (market-quote risk, per curve). A bumped
``BASIS`` quote moves a tenor basis spread by 1bp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from derivx.rates.curves import DISCOUNT_KEY, GOVT_KEY, SPREAD_KINDS, Curve, Market

PvFunction = Callable[[Market], float]
CURVE_FIELDS = ("curve", "govt")  # the single-curve keys; projection curves are keyed by index name


@dataclass
class Position:
    """Anything with a value in a :class:`Market`.

    ``pv(market)`` values it on ``market.asof``; ``cash(market, start, end)``
    is the cash it pays the holder in (start, end], used for carry.
    ``vol_model`` (``normal``/``lognormal``) says which vol shift moves it.
    """

    name: str
    pv: PvFunction
    kind: str = ""
    vol_model: Optional[str] = None
    cash: Optional[Callable[[Market, object, object], float]] = None
    details: Dict[str, object] = field(default_factory=dict)


def _curve(market: Market, which: str) -> Curve:
    if which not in market.curve_keys:
        if which == "govt":
            raise ValueError("market has no 'govt' curve")
        raise ValueError(f"which must be one of {list(market.curve_keys)}, got {which!r}")
    return market.get_curve(which)


def bumped_market(market: Market, which: str, bumps_bp) -> Market:
    """Market with one curve rebuilt from bumped quotes (see
    :meth:`Curve.rebuilt` for the forms ``bumps_bp`` can take) and the
    projection curves built on it rebuilt with their quotes held (see
    :meth:`Market.rebuilt`)."""

    _curve(market, which)
    return market.rebuilt({which: bumps_bp})


def parallel_pv01(pv: PvFunction, market: Market, which: str = "curve", bp: float = 1.0) -> float:
    """Signed P&L for a +1bp parallel move in a curve's quotes."""

    up = pv(bumped_market(market, which, bp))
    down = pv(bumped_market(market, which, -bp))
    return (up - down) / (2.0 * bp)


def parallel_gamma(pv: PvFunction, market: Market, which: str = "curve", bp: float = 1.0) -> float:
    """P&L per (1bp)^2 of a parallel move: P&L(dy) ~ PV01*dy + gamma*dy^2/2."""

    base = pv(market)
    up = pv(bumped_market(market, which, bp))
    down = pv(bumped_market(market, which, -bp))
    return (up + down - 2.0 * base) / (bp * bp)


def rate_curves(market: Market) -> Tuple[str, ...]:
    """Curves that price swaps and rate options: the discount curve and every
    projection curve with quotes of its own (not the Treasury curve; a
    projection that is the discount curve moves with it)."""

    return tuple(
        k
        for k in market.curve_keys
        if k != GOVT_KEY and (k == DISCOUNT_KEY or not market.projections.is_discount(k)) and market.get_curve(k).quotes
    )


def rate_moved_market(market: Market, bp: float) -> Market:
    """Every rate quote of every :func:`rate_curves` curve moved by ``bp``
    and the curves rebuilt; tenor basis spreads (``BASIS`` quotes) held."""

    bumps = {}
    for key in rate_curves(market):
        curve = market.get_curve(key)
        bumps[key] = [0.0 if inst.kind in SPREAD_KINDS else bp for inst in curve.instruments]
    if not bumps:
        raise ValueError("the market has no curve with quotes to move")
    return market.rebuilt(bumps)


def rate_pv01_gamma(pv: PvFunction, market: Market, bp: float = 1.0) -> Tuple[float, float]:
    """(PV01, gamma) for a parallel move of every rate quote (see
    :func:`rate_moved_market`): P&L per +1bp and per (1bp)^2, central
    differences on full revaluation, so P&L(x bp) ~ PV01 x + gamma x^2 / 2."""

    base = pv(market)
    up = pv(rate_moved_market(market, bp))
    down = pv(rate_moved_market(market, -bp))
    return (up - down) / (2.0 * bp), (up + down - 2.0 * base) / (bp * bp)


def key_rate_pv01(pv: PvFunction, market: Market, which: str = "curve", bp: float = 1.0) -> List[Dict[str, object]]:
    """P&L for +1bp in each quote of a curve, one at a time.

    Rows carry ``index`` (quote position in ``curve.quotes``), ``label``,
    ``tenor``, ``years`` and ``pv01``. The rows sum to roughly the parallel
    PV01 (exactly, for a position linear in the quotes).
    """

    curve = _curve(market, which)
    rows = []
    for i, (quote, inst) in enumerate(zip(curve.quotes, curve.instruments)):
        up = pv(bumped_market(market, which, {i: bp}))
        down = pv(bumped_market(market, which, {i: -bp}))
        rows.append(
            {
                "index": i,
                "label": f"{quote.tenor} {quote.kind}",
                "tenor": quote.tenor,
                "years": inst.years,
                "pv01": (up - down) / (2.0 * bp),
            }
        )
    return rows


def ladder(rows: Sequence[Dict[str, object]]) -> np.ndarray:
    return np.array([float(r["pv01"]) for r in rows])


def solve_hedge(
    target: np.ndarray,
    hedges: np.ndarray,
    names: Sequence[str],
) -> Dict[str, object]:
    """Notionals (as multiples of each hedge's unit) minimising the residual
    ladder ``target + hedges @ x`` in least squares.

    ``hedges`` is (buckets x instruments): each column is one unit of a hedge
    instrument's key-rate ladder. Returns the multiples, the residual ladder
    and how much of the target's squared risk is removed.
    """

    target = np.asarray(target, dtype=float)
    hedges = np.asarray(hedges, dtype=float)
    if hedges.ndim != 2 or hedges.shape[0] != target.shape[0]:
        raise ValueError("hedges must be (buckets x instruments) matching target")
    x, *_ = np.linalg.lstsq(hedges, -target, rcond=None)
    residual = target + hedges @ x
    base = float(target @ target)
    removed = 1.0 - float(residual @ residual) / base if base > 0 else 1.0
    return {"multiples": dict(zip(names, x.tolist())), "residual": residual, "risk_removed": removed}
