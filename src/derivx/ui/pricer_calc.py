"""Pure calculations behind the EQUITY / FX pricer page (no Streamlit):
input checks, number formatting, the scenario P&L grid, risk-surface
grids, illustrative smiles and the model check.

Units follow :mod:`derivx.core.bs`: spot, strike and premiums in price units
(strike currency per one unit of underlying), T in years (ACT/365F), r, q
and vols as per-annum decimals. Shocks are relative for spot (+0.05 = +5%)
and absolute for vol (+0.01 = +1 vol point). P&L is for the signed
``position`` (units of underlying the options are on; negative = short) and
is in price units of the strike currency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from derivx.core.bs import (
    DAYS_PER_YEAR,
    GREEK_NAMES,
    bs_greeks,
    bs_price,
    desk_greeks,
    finite_difference_greeks,
    no_arbitrage_bounds,
)
from derivx.vol.sabr import hagan_sabr_iv, sabr_alpha_from_atm
from derivx.vol.ssvi import SsviSurface
from derivx.vol.svi import check_butterfly, lee_wing_check

# Relative spot shocks and vol-point shocks of the scenario grid, as
# fractions of the user's maximum shock.
SHOCK_STEPS = (-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0)
VOL_STEPS = (-1.0, -0.4, -0.2, 0.0, 0.2, 0.4, 1.0)
MIN_LOG_HALF_WIDTH = 0.05  # +-5% spot window when sigma sqrt(T) is tiny (expiry, zero vol)
MAX_LOG_HALF_WIDTH = 2.5  # never wider than a factor exp(2.5) ~ 12 either side
SSVI_GAMMA = 0.5  # power-law SSVI exponent (Gatheral-Jacquier); eta (1 + |rho|) <= 2 is then arbitrage free


@dataclass(frozen=True)
class Inputs:
    """One option: spot, strike, years, rates, vol (units of the module docs)."""

    S: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    call: bool

    @property
    def forward(self) -> float:
        return self.S * math.exp((self.r - self.q) * self.T)


def input_errors(S, K, T, r, q, sigma) -> List[str]:
    """Problems that make the inputs unpriceable, as messages that name the
    field and the fix (empty when the inputs are valid)."""

    errors = []
    for label, value in (("SPOT.S", S), ("STRIKE.K", K), ("MATURITY.T", T), ("RATE.R", r), ("Q", q), ("VOL.σ", sigma)):
        if value is None or not np.isfinite(value):
            errors.append(f"{label} is empty or not a number: enter a value.")
    if errors:
        return errors
    if S <= 0:
        errors.append(f"SPOT.S must be greater than 0 (got {S:g}): a price level such as 100 or 1.08.")
    if K <= 0:
        errors.append(f"STRIKE.K must be greater than 0 (got {K:g}).")
    if T < 0:
        errors.append(f"MATURITY.T must be >= 0 years (got {T:g}); 0 is the expiry.")
    if sigma < 0:
        errors.append(f"VOL.σ must be >= 0 (got {sigma:g}); it is a decimal, 0.20 = 20%.")
    return errors


def input_warnings(T, r, q, sigma) -> List[str]:
    """Valid but unusual inputs worth a note (units, expiry, zero vol)."""

    notes = []
    if T == 0:
        notes.append(
            "EXPIRED (T = 0): the premium is the intrinsic value, delta is the payoff's step (1/2 exactly at "
            "the strike) and gamma, vega, theta and every other Greek are 0. Set T > 0 for pre-expiry risk."
        )
    elif T < 1.0 / DAYS_PER_YEAR:
        notes.append(
            f"Less than one day to expiry (T = {T * DAYS_PER_YEAR:.2f} days): at-the-money gamma and theta grow "
            "like 1/sqrt(T); the 1-day theta reprices to expiry."
        )
    if sigma == 0 and T > 0:
        notes.append(
            "VOL.σ = 0: the forward is certain, so the premium is the discounted intrinsic value of the forward "
            "and gamma and vega are 0 (see derivx.core.bs for the limits at the forward)."
        )
    if sigma > 2.0:
        notes.append(f"VOL.σ = {sigma:.0%}: vols are decimals (0.20 = 20%); check the units.")
    for label, value in (("RATE.R", r), ("Q", q)):
        if abs(value) > 0.25:
            notes.append(f"{label} = {value:.0%}: rates are decimals (0.05 = 5%); check the units.")
    return notes


def fmt_sig(x: float, sig: int = 4) -> str:
    """``sig`` significant figures with thousands separators, scientific
    below 1e-4 so a small non-zero value is never shown as 0.0000; exact 0
    (and -0) as '0'."""

    if x is None or not np.isfinite(x):
        return "n/a"
    x = float(x)
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-4:
        return f"{x:.{sig - 1}e}"
    decimals = max(sig - 1 - int(math.floor(math.log10(ax))), 0)
    return f"{x:,.{decimals}f}"


def _sig_text(x: float, sig: int) -> str:
    """``x`` rounded to ``sig`` significant figures, in plain notation."""

    digits = sig - 1 - int(math.floor(math.log10(abs(x))))
    rounded = round(x, digits)
    digits = sig - 1 - int(math.floor(math.log10(abs(rounded))))  # 9.96 -> 10: one fewer decimal
    return f"{rounded:.{max(digits, 0)}f}"


def fmt_compact(x: float, sig: int = 2) -> str:
    """Short label for a chart cell (at most 5 characters for 2 significant
    figures): ``sig`` significant figures with a k / M / B suffix from
    1,000 up (``-50,912`` -> ``'-51k'``, ``7612`` -> ``'7.6k'``), scientific
    below 1e-3."""

    if x is None or not np.isfinite(x):
        return "n/a"
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        return f"{x:.{sig - 1}e}"
    rounded = float(_sig_text(x, sig))
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(rounded) >= scale:
            return _sig_text(rounded / scale, sig) + suffix
    return _sig_text(rounded, sig)


def fmt_fixed(x: float, decimals: int) -> str:
    """Fixed decimals with thousands separators; falls back to
    :func:`fmt_sig` when the value would print as 0."""

    if x is None or not np.isfinite(x):
        return "n/a"
    if x != 0 and abs(x) < 0.5 * 10.0 ** (-decimals):
        return fmt_sig(x)
    return f"{float(x):,.{decimals}f}"


def tenor_text(T: float) -> str:
    """'1.0000Y · 365D' style tenor (ACT/365F days, 2 decimals below a day)."""

    days = T * DAYS_PER_YEAR
    day_text = f"{days:,.0f}D" if abs(days - round(days)) < 1e-6 else f"{days:,.2f}D"
    return f"{T:.4f}Y · {day_text}"


# ---------------------------------------------------------------------------
# Scenario grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioGrid:
    """Full-revaluation P&L ``pnl[i, j]`` for vol shock ``vol_shocks[i]``
    (absolute, decimals) and spot shock ``spot_shocks[j]`` (relative), over
    ``horizon_days`` calendar days, for ``position`` units of underlying."""

    spot_shocks: np.ndarray
    vol_shocks: np.ndarray
    pnl: np.ndarray
    base_value: float
    horizon_days: float
    position: float
    dropped_vol_shocks: Tuple[float, ...]

    def table(self) -> pd.DataFrame:
        cols = [f"{s:+.1%}" for s in self.spot_shocks]
        rows = [f"{v * 100:+.1f} vol pts" for v in self.vol_shocks]
        return pd.DataFrame(self.pnl, index=pd.Index(rows, name="vol shock"), columns=pd.Index(cols, name="spot shock"))


def scenario_grid(
    inp: Inputs,
    position: float,
    max_spot_shock: float = 0.10,
    max_vol_shock: float = 0.05,
    horizon_days: float = 0.0,
) -> ScenarioGrid:
    """P&L of ``position`` long options (one per unit of underlying) when
    spot moves by ``s`` (relative) and vol by ``v`` (absolute), repriced
    after ``horizon_days`` calendar days: ``position * (V(S (1 + s),
    sigma + v, T - h) - V(S, sigma, T))``. Shocks are ``max * SHOCK_STEPS``
    and ``max * VOL_STEPS``; vol shocks that would make the vol negative are
    dropped (listed in ``dropped_vol_shocks``), never floored."""

    if not 0 < max_spot_shock <= 0.9:
        raise ValueError(f"max spot shock must be in (0, 90%], got {max_spot_shock:.2%}")
    if not 0 < max_vol_shock <= 1.0:
        raise ValueError(f"max vol shock must be in (0, 100] vol points, got {max_vol_shock * 100:g}")
    if not np.isfinite(horizon_days) or horizon_days < 0:
        raise ValueError(f"horizon must be >= 0 calendar days, got {horizon_days}")
    spot = max_spot_shock * np.asarray(SHOCK_STEPS)
    vols = max_vol_shock * np.asarray(VOL_STEPS)
    shocked = inp.sigma + vols
    on_zero = np.abs(shocked) <= 1e-12 * max(inp.sigma, 1e-3)  # e.g. 0.02 - 0.05 * 0.4 in floating point
    shocked = np.where(on_zero, 0.0, shocked)
    keep = shocked >= 0
    dropped = tuple(float(v) for v in vols[~keep])
    vols, shocked = vols[keep], shocked[keep]
    T_h = max(inp.T - horizon_days / DAYS_PER_YEAR, 0.0)
    base = float(bs_price(inp.S, inp.K, inp.T, inp.r, inp.sigma, q=inp.q, call=inp.call))
    S_grid, V_grid = np.meshgrid(inp.S * (1.0 + spot), shocked)
    values = bs_price(S_grid, inp.K, T_h, inp.r, V_grid, q=inp.q, call=inp.call)
    return ScenarioGrid(spot, vols, position * (values - base), base, float(horizon_days), float(position), dropped)


# ---------------------------------------------------------------------------
# Risk surfaces
# ---------------------------------------------------------------------------

SURFACE_AXES = ("SPOT × VOL", "SPOT × TIME TO EXPIRY")
SURFACE_Z: Dict[str, Tuple[str, str]] = {
    # label -> (field of desk Greeks, unit)
    "VALUE": ("price", "premium, price units"),
    "DELTA": ("delta", "per 1.00 of spot"),
    "GAMMA": ("gamma_1pct", "change in delta per +1% spot"),
    "VEGA": ("vega_1vol", "per +1 vol pt"),
    "THETA": ("theta_day", "per calendar day"),
}


def spot_window(inp: Inputs, n_sd: float, T_ref: Optional[float] = None) -> Tuple[float, float]:
    """Spot range covering the spot and the strike plus ``n_sd`` standard
    deviations ``sigma sqrt(T_ref)`` of log-spot either side (floored at
    +-5% when sigma sqrt(T) is tiny, capped at a factor exp(2.5))."""

    if n_sd <= 0:
        raise ValueError("the width in standard deviations must be > 0")
    T_ref = inp.T if T_ref is None else T_ref
    half = float(np.clip(n_sd * inp.sigma * math.sqrt(T_ref), MIN_LOG_HALF_WIDTH, MAX_LOG_HALF_WIDTH))
    return min(inp.S, inp.K) * math.exp(-half), max(inp.S, inp.K) * math.exp(half)


def vol_window(sigma: float) -> Tuple[float, float]:
    """Vol axis: sigma -+ max(sigma / 2, 10 vol points), floored at 0."""

    half = max(0.5 * sigma, 0.10)
    return max(sigma - half, 0.0), sigma + half


@dataclass(frozen=True)
class SurfaceGrid:
    """``z[i, j]`` at ``y[i]`` (vol or time to expiry, years) and spot ``x[j]``."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    axes: str
    z_label: str
    z_unit: str
    point: Tuple[float, float, float]  # the current option (spot, y, z)


def surface_grid(inp: Inputs, axes: str, z_label: str, n: int = 25, n_sd: float = 3.0) -> SurfaceGrid:
    """Value or a Greek (desk units, :data:`SURFACE_Z`) of the option over
    spot x vol or spot x time to expiry (0..T). The spot domain is
    :func:`spot_window` (at the full T for the time view)."""

    if axes not in SURFACE_AXES:
        raise ValueError(f"axes must be one of {SURFACE_AXES}")
    if z_label not in SURFACE_Z:
        raise ValueError(f"z must be one of {tuple(SURFACE_Z)}")
    if axes == SURFACE_AXES[1] and inp.T <= 0:
        raise ValueError("the option has expired (T = 0): there is no time to expiry to plot")
    if not 5 <= n <= 200:
        raise ValueError("resolution must be between 5 and 200 points per axis")
    lo, hi = spot_window(inp, n_sd)
    x = np.linspace(lo, hi, n)
    if axes == SURFACE_AXES[0]:
        y = np.linspace(*vol_window(inp.sigma), n)
        Xs, Ys = np.meshgrid(x, y)
        g = bs_greeks(Xs, inp.K, inp.T, inp.r, Ys, q=inp.q, call=inp.call)
        here = bs_greeks(inp.S, inp.K, inp.T, inp.r, inp.sigma, q=inp.q, call=inp.call)
        y_here = inp.sigma
    else:
        y = np.linspace(0.0, inp.T, n)
        Xs, Ys = np.meshgrid(x, y)
        g = bs_greeks(Xs, inp.K, Ys, inp.r, inp.sigma, q=inp.q, call=inp.call)
        here = bs_greeks(inp.S, inp.K, inp.T, inp.r, inp.sigma, q=inp.q, call=inp.call)
        y_here = inp.T
    field, unit = SURFACE_Z[z_label]
    z = g.price if field == "price" else desk_greeks(g, Xs)[field]
    z_here = float(here.price if field == "price" else desk_greeks(here, inp.S)[field])
    return SurfaceGrid(x, y, np.asarray(z, dtype=float), axes, z_label, unit, (inp.S, y_here, z_here))


# ---------------------------------------------------------------------------
# Smiles
# ---------------------------------------------------------------------------


def smile_strikes(F: float, atm_vol: float, T: float, n_sd: float = 3.0, n: int = 121) -> np.ndarray:
    """Strikes ``F exp(z atm_vol sqrt(T))`` for z in [-n_sd, n_sd]: the
    smile's grid in forward moneyness (standard deviations of log-spot)."""

    if not (F > 0 and atm_vol > 0 and T > 0 and n_sd > 0):
        raise ValueError("the smile grid needs F > 0, ATM vol > 0, T > 0 and a positive width")
    half = min(n_sd * atm_vol * math.sqrt(T), MAX_LOG_HALF_WIDTH)
    return F * np.exp(np.linspace(-half, half, n))


@dataclass(frozen=True)
class IllustrativeSmile:
    """A model smile from user-set parameters (not market data)."""

    model: str  # 'SVI' (SSVI slice) or 'SABR'
    F: float
    T: float
    atm_vol: float
    strikes: np.ndarray
    vols: np.ndarray
    params: Dict[str, float]  # model parameters, labelled for display
    fixed: Tuple[str, ...]
    notes: Tuple[str, ...]
    arbitrage_free: Optional[bool]

    def vol_at(self, K: float) -> float:
        return float(np.interp(math.log(K / self.F), np.log(self.strikes / self.F), self.vols))


def illustrative_svi(F: float, T: float, atm_vol: float, rho: float, eta: float, n_sd: float = 3.0):
    """SSVI slice (Gatheral-Jacquier power law, gamma = 1/2) with ATM total
    variance ``theta = atm_vol^2 T``, skew ``rho`` and curvature ``eta``:
    ``w(k) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))``,
    ``phi = eta / (sqrt(theta) sqrt(1 + theta))``. Free of butterfly
    arbitrage when ``eta (1 + |rho|) <= 2``; the raw-SVI parameters of the
    slice are reported (total variance for T)."""

    if not abs(rho) < 1:
        raise ValueError(f"skew rho must be in (-1, 1), got {rho}")
    if not eta > 0:
        raise ValueError(f"curvature eta must be > 0, got {eta}")
    strikes = smile_strikes(F, atm_vol, T, n_sd)
    surf = SsviSurface(expiries=np.array([T]), theta=np.array([atm_vol**2 * T]), rho=rho, eta=eta, gamma=SSVI_GAMMA)
    vols = np.asarray(surf.implied_vol(np.log(strikes / F), T), dtype=float)
    raw = surf.slice_svi(T)
    k = np.log(strikes / F)
    fly = check_butterfly(raw, np.unique(np.r_[k, np.linspace(-3.0, 3.0, 601)]))
    lee = lee_wing_check(raw)
    notes = []
    if eta * (1 + abs(rho)) > 2:
        notes.append(
            f"eta (1 + |rho|) = {eta * (1 + abs(rho)):.2f} > 2: outside the Gatheral-Jacquier no-arbitrage region "
            f"(butterfly check: {'ok' if fly.ok else f'g(k) < 0, min {fly.min_g:.3g} at k = {fly.k_at_min:.3f}'})."
        )
    params = {f"SVI {name}": float(raw[name]) for name in ("a", "b", "rho", "m", "sigma")}
    return IllustrativeSmile(
        "SVI", F, T, atm_vol, strikes, vols, params, (), tuple(notes), bool(fly.ok and lee["lee_ok"])
    )


def illustrative_sabr(F: float, T: float, atm_vol: float, rho: float, nu: float, beta: float = 1.0, n_sd: float = 3.0):
    """Hagan SABR smile with ATM vol ``atm_vol`` at K = F (alpha solved from
    it), skew ``rho`` and vol of vol ``nu``; beta fixed (1 = lognormal
    backbone, the usual FX choice)."""

    strikes = smile_strikes(F, atm_vol, T, n_sd)
    alpha = sabr_alpha_from_atm(F, T, atm_vol, beta, rho, nu)
    vols = np.asarray(hagan_sabr_iv(F, strikes, T, alpha, beta, rho, nu), dtype=float)
    params = {"SABR α": alpha, "SABR β": beta, "SABR ρ": rho, "SABR ν": nu}
    return IllustrativeSmile("SABR", F, T, atm_vol, strikes, vols, params, ("SABR β",), (), None)


# ---------------------------------------------------------------------------
# Model check
# ---------------------------------------------------------------------------

FD_TOLERANCE = {"gamma": 5e-4, "vanna": 5e-4, "volga": 5e-4, "charm": 5e-4}  # second differences; others 2e-5


def model_check(inp: Inputs) -> pd.DataFrame:
    """Analytic Greeks against central finite differences of the pricer
    (raw units), with the relative error and a pass flag; the error is
    measured against ``|fd| + 1e-6 * scale`` so Greeks that are ~0 do not
    fail on rounding. Raises ``ValueError`` at T = 0 or sigma = 0 (kink)."""

    g = bs_greeks(inp.S, inp.K, inp.T, inp.r, inp.sigma, q=inp.q, call=inp.call)
    fd = finite_difference_greeks(inp.S, inp.K, inp.T, inp.r, inp.sigma, q=inp.q, call=inp.call)
    scale = {"delta": 1.0, "gamma": 1.0 / inp.S, "vanna": 1.0, "charm": 1.0}
    rows = []
    for name in GREEK_NAMES:
        a, b = float(getattr(g, name)), float(fd[name])
        err = abs(a - b) / (abs(b) + 1e-6 * scale.get(name, inp.S))
        tol = FD_TOLERANCE.get(name, 2e-5)
        rows.append({"greek": name, "analytic": a, "finite difference": b, "relative error": err, "ok": err <= tol})
    return pd.DataFrame(rows)


def price_bounds(inp: Inputs) -> Tuple[float, float]:
    lo, hi = no_arbitrage_bounds(inp.S, inp.K, inp.T, inp.r, q=inp.q, call=inp.call)
    return float(lo), float(hi)
