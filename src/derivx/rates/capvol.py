"""Caplet vols stripped from flat cap vols, and caplet smiles.

Brokers quote a cap by one flat vol: the single vol that, applied to every
caplet, gives the cap's price. Caplets of different expiries have different
vols, so pricing any other cap (another maturity, a forward start, another
strike) needs the caplet vols. :func:`strip_caplet_vols` bootstraps them
maturity by maturity so that every quoted cap reprices exactly:

* each quoted cap is priced at its flat vol (the target);
* its caplets that expire before the previous quote's last caplet already
  have vols; the new ones (up to this cap's last caplet expiry, the node)
  are solved together for one unknown:
  - ``constant``: one caplet vol for the new caplets (piecewise constant in
    expiry, flat before the first and after the last node);
  - ``linear``: caplet vol linear in expiry from the previous node's vol to
    the new node's (flat up to the first node and after the last).

Quotes at fixed strikes strip one strike at a time (:func:`strip_surface`),
giving a caplet smile per expiry (:class:`CapletVolSurface`, linear in
strike between quoted strikes, flat outside). ATM quotes strip on each
cap's own ATM strike (the strip's forward swap rate): the result is a curve
of caplet vols at the ATM strikes of the quoted caps, which assumes the
caplet vol at an expiry does not depend on the strike between those ATM
strikes. A cap whose flat vol cannot be met by non-negative caplet vols
(flat vols falling too fast, or quotes out of order) raises with the
maturity named: nothing is clipped.

Caps are :class:`~derivx.rates.capfloor.CapFloor` spot-start caps (first
caplet excluded, as quoted), so stripping uses exactly the caplet schedule,
forwards, discount factors and model of the pricer; QuantLib's
``OptionletStripper1`` gives the same caplet vols when every caplet
maturity is quoted (``tests/test_capvol.py``). Expiries are ACT/365F years
to the fixing; vols in the model's units (normal: rate units, 1bp = 1e-4;
lognormal: decimal).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from derivx.rates.capfloor import MODELS, CapFloor, _model_shift, model_price
from derivx.rates.conventions import parse_tenor
from derivx.rates.curves import Market

STRIP_METHODS = ("constant", "linear")
_EPS = np.finfo(float).eps


# --------------------------------------------------------------------------
# Vol objects a CapFloor can price with
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CapletVolCurve:
    """Caplet vols by option expiry (ACT/365F years to the fixing) for one
    strike (``strike=None``: an ATM strip). ``nodes`` are the last caplet
    expiry of each quoted cap; ``method`` says how vols run between them
    (see the module docstring). ``model``/``shift`` are the pricing model
    the vols belong to; :class:`~derivx.rates.capfloor.CapFloor` checks a
    cap uses the same."""

    nodes: Tuple[float, ...]
    vols: Tuple[float, ...]
    method: str
    model: str
    shift: float = 0.0
    strike: Optional[float] = None

    def __post_init__(self) -> None:
        if self.method not in STRIP_METHODS:
            raise ValueError(f"method must be one of {STRIP_METHODS}, got {self.method!r}")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        nodes, vols = np.asarray(self.nodes, dtype=float), np.asarray(self.vols, dtype=float)
        if nodes.size == 0 or nodes.shape != vols.shape:
            raise ValueError("a caplet vol curve needs one vol per node")
        if np.any(~np.isfinite(nodes)) or np.any(nodes <= 0) or np.any(np.diff(nodes) <= 0):
            raise ValueError("caplet vol nodes must be increasing positive expiries")
        if np.any(~np.isfinite(vols)) or np.any(vols < 0):
            raise ValueError("caplet vols must be non-negative numbers")

    def vol_at(self, expiry) -> np.ndarray:
        """Caplet vol at option expiries (years)."""

        t = np.asarray(expiry, dtype=float)
        nodes, vols = np.asarray(self.nodes), np.asarray(self.vols)
        if self.method == "linear":
            return np.interp(t, nodes, vols)
        idx = np.clip(np.searchsorted(nodes, t, side="left"), 0, nodes.size - 1)
        return vols[idx]

    def caplet_vols(self, expiry, strike: float) -> np.ndarray:
        return self.vol_at(expiry)

    @property
    def lowest(self) -> float:
        return float(min(self.vols))


@dataclass(frozen=True)
class CapletVolSurface:
    """Caplet vols by expiry and strike: one stripped curve per quoted
    strike, linear in strike between them and flat outside."""

    curves: Tuple[CapletVolCurve, ...]

    def __post_init__(self) -> None:
        if not self.curves:
            raise ValueError("a caplet vol surface needs at least one strike")
        strikes = [c.strike for c in self.curves]
        if any(k is None for k in strikes) or np.any(np.diff(np.asarray(strikes, dtype=float)) <= 0):
            raise ValueError("a caplet vol surface needs fixed strikes in increasing order")
        first = self.curves[0]
        if any((c.model, c.shift, c.method) != (first.model, first.shift, first.method) for c in self.curves):
            raise ValueError("every strike of a caplet vol surface must share its model, shift and method")

    @property
    def model(self) -> str:
        return self.curves[0].model

    @property
    def shift(self) -> float:
        return self.curves[0].shift

    @property
    def method(self) -> str:
        return self.curves[0].method

    @property
    def strikes(self) -> np.ndarray:
        return np.array([c.strike for c in self.curves], dtype=float)

    @property
    def nodes(self) -> Tuple[float, ...]:
        return self.curves[0].nodes

    def smile(self, expiry: float) -> Tuple[np.ndarray, np.ndarray]:
        """(strikes, caplet vols) at one expiry."""

        return self.strikes, np.array([float(c.vol_at(expiry)) for c in self.curves])

    def caplet_vols(self, expiry, strike: float) -> np.ndarray:
        t = np.atleast_1d(np.asarray(expiry, dtype=float))
        table = np.vstack([c.vol_at(t) for c in self.curves])  # strikes x expiries
        out = np.array([np.interp(strike, self.strikes, table[:, k]) for k in range(t.size)])
        return out if np.ndim(expiry) else out.reshape(())

    @property
    def lowest(self) -> float:
        return float(min(c.lowest for c in self.curves))


# --------------------------------------------------------------------------
# Stripping
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StripResult:
    """A stripped curve and its repricing table (one row per quoted cap:
    maturity, strike, flat vol, node, target and stripped PV per unit
    notional, error)."""

    curve: CapletVolCurve
    table: pd.DataFrame

    @property
    def max_error(self) -> float:
        """Largest |stripped - target| PV per unit notional."""

        return float(self.table["error"].abs().max())


def _cap(maturity: str, strike: float, vol: float, model: str, shift: float, terms: Mapping[str, object]) -> CapFloor:
    return CapFloor(1.0, "cap", strike, maturity, vol, model, shift, **dict(terms))


def _unit(model: str, F: np.ndarray, K: float, T: np.ndarray, vols: np.ndarray, shift: float) -> np.ndarray:
    """Undiscounted caplet values (intrinsic once fixed)."""

    s = _model_shift(model, shift)
    out = np.empty(F.size)
    for i, (f, t, v) in enumerate(zip(F, T, vols)):
        out[i] = max(f - K, 0.0) if t <= 0 else model_price(model, f, K, t, v, s, True)
    return out


def strip_caplet_vols(
    market: Market,
    maturities: Sequence[str],
    flat_vols: Sequence[float],
    strike: Optional[float] = None,
    method: str = "constant",
    model: str = "bachelier",
    shift: float = 0.0,
    terms: Optional[Mapping[str, object]] = None,
) -> StripResult:
    """Caplet vols that reprice every quoted cap (see the module docstring).

    ``maturities`` are cap tenors (``1Y``, ``18M``, ...) and ``flat_vols``
    their flat vols in the model's units; ``strike`` a decimal rate, or None
    for ATM caps (each at its own ATM strike). ``terms`` are the caps'
    :class:`CapFloor` schedule fields (``index_months``, ``day_count``,
    ``calendar``, ``fixing_lag``, ``projection``; default: 3M ACT/360 on the
    USD calendar, single curve). Raises ``ValueError`` naming the quote that
    cannot be met."""

    if method not in STRIP_METHODS:
        raise ValueError(f"method must be one of {STRIP_METHODS}, got {method!r}")
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    if len(maturities) != len(flat_vols) or not maturities:
        raise ValueError("give one flat vol per cap maturity (at least one)")
    terms = dict(terms or {})
    order = sorted(range(len(maturities)), key=lambda i: parse_tenor(maturities[i]).years)
    quotes = [(str(maturities[i]).strip().upper(), float(flat_vols[i])) for i in order]
    if len({m for m, _ in quotes}) != len(quotes):
        raise ValueError("cap maturities must be distinct")
    for m, v in quotes:
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"the {m} flat vol must be a positive number, got {v!r}")
    nodes: List[float] = []
    vols: List[float] = []
    rows = []
    for m, flat in quotes:
        probe = _cap(m, 0.01 if strike is None else float(strike), flat, model, shift, terms)
        K = probe.atm_strike(market) if strike is None else float(strike)
        cap = _cap(m, K, flat, model, shift, terms)
        strip = cap._strip(market)
        live = strip.live
        if not live.any():
            raise ValueError(f"the {m} cap has no caplet left to expire")
        T, F, scale = strip.expiry, strip.forward, strip.tau * strip.df
        node = float(T[live].max())
        if nodes and node <= nodes[-1] + 1e-12:
            raise ValueError(f"the {m} cap ends on the same caplet as a shorter quoted cap: drop one of them")
        target = float(np.dot(scale, _unit(model, F, K, T, np.full(T.size, flat), shift)))
        prev_node, prev_vol = (nodes[-1], vols[-1]) if nodes else (0.0, None)
        new = T > prev_node + 1e-12
        known = np.zeros(T.size)
        if nodes:
            known[~new] = CapletVolCurve(tuple(nodes), tuple(vols), method, model, shift).vol_at(T[~new])

        def trial(x: float) -> np.ndarray:
            sig = known.copy()
            if method == "constant" or prev_vol is None:
                sig[new] = x
            else:  # linear from the previous node to this one
                w = (T[new] - prev_node) / (node - prev_node)
                sig[new] = prev_vol + w * (x - prev_vol)
            return sig

        def excess(x: float) -> float:
            return float(np.dot(scale, _unit(model, F, K, T, trial(x), shift))) - target

        lowest = excess(0.0)
        if lowest > 16 * _EPS * max(abs(target), 1e-12):
            hint = " (try piecewise-constant caplet vols)" if method == "linear" and prev_vol is not None else ""
            raise ValueError(
                f"the {m} cap at flat vol {flat:.6g} is worth less than its caplets at zero vol beyond the "
                f"{quotes[len(nodes) - 1][0] if nodes else 'first'} quote: the flat vols fall too fast for "
                f"non-negative caplet vols{hint}"
            )
        hi, cap_vol = (0.01, 10.0) if model == "bachelier" else (0.5, 100.0)
        while excess(hi) < 0.0:
            hi *= 2.0
            if hi > cap_vol:
                raise ValueError(f"the {m} cap's flat vol implies a caplet vol above {cap_vol:g}: check the quotes")
        x = 0.0 if lowest >= 0.0 else float(brentq(excess, 0.0, hi, xtol=1e-16, rtol=4 * _EPS, maxiter=500))
        nodes.append(node)
        vols.append(x)
        stripped = target + excess(x)
        rows.append({"maturity": m, "strike": K, "flat_vol": flat, "node": node, "caplets": int(live.sum()),
                     "target_pv": target, "stripped_pv": stripped, "error": stripped - target})  # fmt: skip
    curve = CapletVolCurve(tuple(nodes), tuple(vols), method, model, shift, None if strike is None else float(strike))
    return StripResult(curve, pd.DataFrame(rows))


def strip_surface(
    market: Market,
    maturities: Sequence[str],
    strikes: Sequence[float],
    flat_vols: np.ndarray,
    method: str = "constant",
    model: str = "bachelier",
    shift: float = 0.0,
    terms: Optional[Mapping[str, object]] = None,
) -> Tuple[CapletVolSurface, List[StripResult]]:
    """Strip each strike column of a maturities x strikes flat vol grid
    (NaN cells are left out of that strike's strip)."""

    grid = np.asarray(flat_vols, dtype=float)
    if grid.shape != (len(maturities), len(strikes)):
        raise ValueError(f"flat vols must be maturities x strikes ({len(maturities)} x {len(strikes)})")
    order = np.argsort(np.asarray(strikes, dtype=float))
    results = []
    for j in order:
        col = grid[:, j]
        keep = np.isfinite(col)
        if not keep.any():
            raise ValueError(f"strike {strikes[j]:.3%} has no quotes")
        mats = [m for m, k in zip(maturities, keep) if k]
        try:
            results.append(strip_caplet_vols(market, mats, col[keep], float(strikes[j]), method, model, shift, terms))
        except ValueError as exc:
            raise ValueError(f"strike {strikes[j]:.3%}: {exc}") from None
    node_sets = {r.curve.nodes for r in results}
    if len(node_sets) > 1:
        raise ValueError("every strike needs quotes at the same cap maturities (a full maturity x strike grid)")
    return CapletVolSurface(tuple(r.curve for r in results)), results


# --------------------------------------------------------------------------
# Risk by quote and by expiry
# --------------------------------------------------------------------------


def vega_by_expiry(cf: CapFloor, market: Market, edges: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Caplet vegas (P&L per +1bp normal / +1 vol point, see
    :meth:`CapFloor.caplets`) summed into expiry buckets ``(edges[k-1],
    edges[k]]`` in years (default: one bucket per caplet)."""

    table = cf.caplets(market)
    table = table[~table["fixed"]]
    if edges is None:
        return pd.DataFrame({"fix": table["fix"], "expiry": table["expiry"], "vega": table["vega"]})
    edges = sorted(float(e) for e in edges)
    labels, values = [], []
    lo = 0.0
    for hi in edges + [math.inf]:
        mask = (table["expiry"] > lo) & (table["expiry"] <= hi)
        labels.append(f"{lo:g}-{hi:g}Y" if math.isfinite(hi) else f">{lo:g}Y")
        values.append(float(table.loc[mask, "vega"].sum()))
        lo = hi
    return pd.DataFrame({"bucket": labels, "vega": values})


def quote_vegas(
    cf: CapFloor,
    market: Market,
    maturities: Sequence[str],
    flat_vols: Sequence[float],
    strike: Optional[float] = None,
    method: str = "constant",
    terms: Optional[Mapping[str, object]] = None,
    bump: Optional[float] = None,
) -> pd.DataFrame:
    """P&L of ``cf`` (priced on the caplet vols stripped from the quotes) for
    +1bp normal / +1 vol point on each quoted flat vol, re-stripped. Rows:
    maturity, vega, difference.

    A central difference by default. Next to the edge of what can be
    stripped (flat vols falling about as fast as non-negative caplet vols
    allow, or a vol within ``bump`` of zero) one side of the bump has no
    caplet vols; that quote then takes the one-sided difference of the side
    that strips (``difference`` says which), and only a quote that cannot
    be moved either way raises."""

    unit = 1e-4 if cf.model == "bachelier" else 0.01
    bump = unit if bump is None else bump
    if not math.isfinite(bump) or bump <= 0:
        raise ValueError(f"the vol bump must be a positive number, got {bump!r}")
    flat = np.asarray(flat_vols, dtype=float)

    def value(vols: np.ndarray) -> float:
        curve = strip_caplet_vols(market, maturities, vols, strike, method, cf.model, cf.shift, terms).curve
        return CapFloor(**{**_fields(cf), "vol": curve}).pv(market)

    base = value(flat)
    rows = []
    for i, m in enumerate(maturities):
        moved: Dict[str, float] = {}
        failed: Dict[str, str] = {}
        for side, sign in (("up", 1.0), ("down", -1.0)):
            vols = flat.copy()
            vols[i] += sign * bump
            try:
                moved[side] = value(vols)
            except ValueError as exc:
                failed[side] = str(exc)
        if len(moved) == 2:
            vega, how = (moved["up"] - moved["down"]) / 2.0, "central"
        elif "up" in moved:
            vega, how = moved["up"] - base, "up only"
        elif "down" in moved:
            vega, how = base - moved["down"], "down only"
        else:
            raise ValueError(f"the {m} quote cannot be moved by +/-{bump:g} and re-stripped: {failed['up']}")
        rows.append({"maturity": m, "vega": vega * unit / bump, "difference": how})
    return pd.DataFrame(rows)


def _fields(cf: CapFloor) -> Dict[str, object]:
    return {name: getattr(cf, name) for name in cf.__dataclass_fields__}
