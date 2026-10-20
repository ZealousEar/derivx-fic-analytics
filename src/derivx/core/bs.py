"""Black-Scholes-Merton prices and Greeks on spot with a continuous carry
(an equity dividend yield or an FX foreign rate), vectorised.

Conventions and units
---------------------
* ``S`` (spot) and ``K`` (strike) are in price units: the strike currency per
  one unit of the underlying (a share, an index point, one unit of the base
  currency of an FX pair). Both must be > 0.
* ``T`` is the time to expiry in years (ACT/365F when it comes from dates:
  calendar days / 365), ``T >= 0``. ``T = 0`` is the expiry itself.
* ``r`` is the continuously compounded risk-free rate of the strike currency
  and ``q`` the continuously compounded dividend yield (equity) or foreign
  interest rate (FX), per-annum decimals (0.05 = 5%).
* ``sigma`` is the lognormal (Black-Scholes) volatility, a per-annum decimal
  (0.20 = 20%), ``sigma >= 0``.
* ``F = S exp((r - q) T)`` is the forward, ``w = +1`` for a call and ``-1``
  for a put; ``call`` may be one flag or an array (a mixed book).
* Prices are per one unit of underlying, in the strike currency, for a long
  option.

Greeks (:class:`BSGreeks`) are the analytic partial derivatives of that
price, per unit of each input (per 1.00 of spot, of vol and of rate):

=========  ======================================  ==========================
name       definition                              raw unit
=========  ======================================  ==========================
delta      dV/dS                                   per 1.00 of spot
gamma      d2V/dS2                                 per 1.00 of spot, squared
vega       dV/dsigma                               per 1.00 of vol (100 pts)
theta      dV/dt = -dV/dT (calendar time passing)  per year
rho        dV/dr                                   per 1.00 of rate
phi        dV/dq (dividend / foreign-rate rho)     per 1.00 of rate
vanna      d2V/dS dsigma = d delta / d sigma       per 1.00 spot, per 1.00 vol
volga      d2V/dsigma2 (vomma)                     per 1.00 vol, squared
charm      d delta / dt = -d delta / dT            per year
=========  ======================================  ==========================

:func:`desk_greeks` turns them into the usual desk units (P&L per +1% spot,
per +1 vol point, per calendar day, per +1bp) and :func:`bs_theta_1d` gives
the one-day theta by repricing, which differs from ``theta / 365`` close to
expiry (the analytic theta has a ``1 / sqrt(T)`` singularity at the money).

Degenerate inputs
-----------------
Inputs are validated, never clamped: non-finite values, ``S <= 0``,
``K <= 0``, ``T < 0`` or ``sigma < 0`` raise ``ValueError``, as do carries
so large that ``exp(r T)`` or ``exp((r - q) T)`` overflows.

When ``sigma sqrt(T) = 0`` (expiry, or zero vol) the price is the discounted
intrinsic value of the forward, ``max(w (S e^{-qT} - K e^{-rT}), 0)``, and the
Greeks are its limits, with no NaN or inf:

* ``T = 0`` (expired): price ``max(w (S - K), 0)``, delta the payoff's step
  ``w 1{w (S - K) > 0}``; every other Greek is 0 (no time is left to decay,
  and rates and vol no longer matter).
* ``sigma = 0``, ``T > 0``: delta is ``w e^{-qT}`` in the money and 0 out of
  the money; theta, rho, phi and charm are the derivatives of the discounted
  forward intrinsic; gamma, vanna and volga are 0.
* On the forward (``F == K`` to a relative 1e-12), where the payoff's kink
  sits, delta is the midpoint ``w e^{-qT} / 2`` (the sigma -> 0 limit);
  with ``T > 0`` vega ``S e^{-qT} sqrt(T) n(0)`` and vanna ``vega / (2 S)``
  keep their finite sigma -> 0 limits, and gamma and charm, which are
  unbounded there (a Dirac mass), are reported as 0. Use a small positive
  ``T`` or ``sigma`` to see the risk next to the kink.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Tuple

import numpy as np
from scipy.special import ndtr

from .common import as_array, ensure_broadcastable

DAYS_PER_YEAR = 365.0  # ACT/365F: theta per calendar day is theta / 365
_SQRT_2PI = np.sqrt(2.0 * np.pi)
_MAX_EXPONENT = 700.0  # exp(700) ~ 1e304: larger carries overflow a float
_ON_FORWARD = 1e-12  # |ln(F / K)| below this is "on the forward" (a relative 1e-12 is float noise in F)

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho", "phi", "vanna", "volga", "charm")


def _npdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def call_flags(call) -> np.ndarray:
    """Boolean call flags from ``call``: a bool, ``'call'`` / ``'put'``
    (also ``'c'`` / ``'p'``, any case) or an array of either."""

    arr = np.asarray(call)
    if arr.dtype == bool:
        return arr
    if arr.dtype.kind in "USO":
        text = np.char.lower(np.char.strip(arr.astype(str)))
        is_call = np.isin(text, ("call", "c"))
        is_put = np.isin(text, ("put", "p"))
        if not np.all(is_call | is_put):
            bad = sorted(set(np.asarray(arr, dtype=str)[~(is_call | is_put)].ravel().tolist()))
            raise ValueError(f"call must be True/False or 'call'/'put'; got {bad}")
        return is_call
    raise ValueError(f"call must be a bool, 'call'/'put' or an array of them; got dtype {arr.dtype}")


def _check(name: str, x: np.ndarray, ok: np.ndarray, rule: str) -> None:
    if not np.all(ok):
        bad = x[~ok]
        raise ValueError(f"{name} must be {rule}; got {bad.ravel()[0]!r} ({bad.size} invalid value(s))")


def _validated(S, K, T, r, sigma, q) -> Tuple[np.ndarray, ...]:
    """Broadcast float arrays after the checks listed in the module docs."""

    S, K, T, r, sigma, q = ensure_broadcastable(S, K, T, r, sigma, q)
    for name, x in (("S (spot)", S), ("K (strike)", K), ("T (years)", T), ("r", r), ("sigma", sigma), ("q", q)):
        _check(name, x, np.isfinite(x), "finite")
    _check("S (spot)", S, S > 0, "> 0")
    _check("K (strike)", K, K > 0, "> 0")
    _check("T (years to expiry)", T, T >= 0, ">= 0 (0 is the expiry)")
    _check("sigma (vol, decimal)", sigma, sigma >= 0, ">= 0")
    for name, x in (("r T", np.abs(r) * T), ("q T", np.abs(q) * T), ("(r - q) T", np.abs(r - q) * T)):
        _check(name, x, x <= _MAX_EXPONENT, f"<= {_MAX_EXPONENT:g} in absolute value (the discount factor overflows)")
    return S, K, T, r, sigma, q


def forward_price(S, T, r, q=0.0) -> np.ndarray:
    """Forward ``F = S exp((r - q) T)`` (price units)."""

    S, T, r, q = ensure_broadcastable(S, T, r, q)
    return S * np.exp((r - q) * T)


@dataclass(frozen=True)
class BSGreeks:
    """Price and analytic Greeks (raw units: see the module docs) for a long
    option on one unit of underlying. Every field is an array broadcast to
    the inputs' shape (0-d for scalar inputs)."""

    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray
    phi: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray
    charm: np.ndarray
    forward: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def bs_greeks(S, K, T, r, sigma, q=0.0, call=True) -> BSGreeks:
    """Black-Scholes-Merton price and first- and second-order Greeks.

    Parameters are broadcastable arrays in the units of the module docs;
    ``call`` is a flag or an array of flags (see :func:`call_flags`).
    Raises ``ValueError`` on invalid inputs (no clamping)."""

    S, K, T, r, sigma, q = _validated(S, K, T, r, sigma, q)
    S, K, T, r, sigma, q, is_call = np.broadcast_arrays(S, K, T, r, sigma, q, call_flags(call))
    w = np.where(is_call, 1.0, -1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        greeks = _greeks(S, K, T, r, sigma, q, w)
    for name, value in greeks.items():
        if not np.all(np.isfinite(value)):  # pragma: no cover - guarded by _validated
            raise ValueError(f"{name} is not finite for these inputs; check the units (decimals, years)")
    return BSGreeks(
        **{k: np.asarray(v, dtype=float) for k, v in greeks.items()},
        forward=np.asarray(S * np.exp((r - q) * T), dtype=float),
    )


def _greeks(S, K, T, r, sigma, q, w) -> Dict[str, np.ndarray]:
    """Kernel of :func:`bs_greeks` on validated, broadcast arrays."""

    sqrt_t = np.sqrt(T)
    std = sigma * sqrt_t
    live = std > 0
    expired = T == 0
    s = np.where(live, std, 1.0)  # safe denominators; degenerate cells are set from their limits
    sig = np.where(sigma > 0, sigma, 1.0)
    tt = np.where(expired, 1.0, T)

    dq = np.exp(-q * T)
    dr = np.exp(-r * T)
    log_fk = np.log(S / K) + (r - q) * T
    on_fwd = np.abs(log_fk) <= _ON_FORWARD  # F == K to float precision
    d_lim = np.where(on_fwd, 0.0, np.where(log_fk > 0, np.inf, -np.inf))
    d1 = np.where(live, log_fk / s + 0.5 * s, d_lim)
    d2 = np.where(live, d1 - s, d_lim)

    n1 = _npdf(d1)
    dens = n1 > 0  # where n(d1) underflows, every term it multiplies is 0 (avoids 0 * inf)
    nd1 = ndtr(w * d1)
    nd2 = ndtr(w * d2)

    price = np.maximum(w * (S * dq * nd1 - K * dr * nd2), 0.0)  # max: rounding only, the formula is >= 0
    delta = w * dq * nd1
    vega = S * dq * n1 * sqrt_t
    rho = w * K * T * dr * nd2
    phi = -w * S * T * dq * nd1
    gamma = np.where(live, dq * n1 / (S * s), 0.0)
    theta = -S * dq * n1 * sigma / (2.0 * np.where(expired, 1.0, sqrt_t)) - w * r * K * dr * nd2 + w * q * S * dq * nd1
    vanna = np.where(live & dens, -dq * n1 * d2 / sig, np.where(~live & on_fwd, vega / (2.0 * S), 0.0))
    volga = np.where(live & dens, vega * d1 * d2 / sig, 0.0)
    charm_kernel = np.where(live & dens, dq * n1 * (2.0 * (r - q) * T - d2 * s) / (2.0 * tt * s), 0.0)
    charm = np.where(~live & on_fwd, 0.0, w * q * dq * nd1 - charm_kernel)

    if np.any(expired):
        zero = np.zeros_like(price)
        price = np.where(expired, np.maximum(w * (S - K), 0.0), price)
        delta = np.where(expired, np.where(on_fwd, 0.5 * w, np.where(w * (S - K) > 0, w, 0.0)), delta)
        theta, rho, phi, vega, vanna, charm = (
            np.where(expired, zero, x) for x in (theta, rho, phi, vega, vanna, charm)
        )
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "phi": phi,
        "vanna": vanna,
        "volga": volga,
        "charm": charm,
    }


def bs_price(S, K, T, r, sigma, q=0.0, call=True) -> np.ndarray:
    """Black-Scholes-Merton price only (same conventions as
    :func:`bs_greeks`)."""

    return bs_greeks(S, K, T, r, sigma, q=q, call=call).price


def bs_price_greeks(
    S, K, T, r, sigma, call=True, q: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(price, delta, gamma, vega, theta, rho)`` in raw units (vega per
    1.00 of vol, theta per year, rho per 1.00 of rate); kept for older
    callers, see :func:`bs_greeks` for the full set and the conventions."""

    g = bs_greeks(S, K, T, r, sigma, q=q, call=call)
    return g.price, g.delta, g.gamma, g.vega, g.theta, g.rho


def bs_theta_1d(S, K, T, r, sigma, q=0.0, call=True, days: float = 1.0) -> np.ndarray:
    """Theta by repricing: ``V(max(T - days / 365, 0)) - V(T)``, the P&L of a
    long option over ``days`` calendar days with spot, rates and vol
    unchanged (ACT/365F). With fewer than ``days`` left, the option is
    repriced at expiry (intrinsic value)."""

    if not np.isfinite(days) or days < 0:
        raise ValueError(f"days must be finite and >= 0, got {days}")
    T_arr = as_array(T)
    later = np.maximum(T_arr - days / DAYS_PER_YEAR, 0.0)
    return bs_price(S, K, later, r, sigma, q=q, call=call) - bs_price(S, K, T_arr, r, sigma, q=q, call=call)


def no_arbitrage_bounds(S, K, T, r, q=0.0, call=True) -> Tuple[np.ndarray, np.ndarray]:
    """Model-free price bounds ``(lower, upper)`` of a European option:
    ``max(w (S e^{-qT} - K e^{-rT}), 0) <= V`` (the zero-vol value) and
    ``V <= S e^{-qT}`` for a call, ``V <= K e^{-rT}`` for a put (the
    infinite-vol value)."""

    S, K, T, r, sigma, q = _validated(S, K, T, r, 0.0, q)
    is_call = call_flags(call)
    w = np.where(is_call, 1.0, -1.0)
    fwd_s, fwd_k = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = np.maximum(w * (fwd_s - fwd_k), 0.0)
    upper = np.where(is_call, fwd_s, fwd_k)
    return lower, upper


def put_call_parity_residual(call_price, put_price, S, K, T, r, q: float = 0.0) -> np.ndarray:
    """Residual ``C - P - (S e^{-qT} - K e^{-rT})`` of two independently
    obtained prices (e.g. market quotes); zero when they satisfy put-call
    parity. Prices from :func:`bs_greeks` satisfy it by construction, so it
    is not a test of the model."""

    S, K, T, r, q = ensure_broadcastable(S, K, T, r, q)
    return as_array(call_price) - as_array(put_price) - (S * np.exp(-q * T) - K * np.exp(-r * T))


DESK_UNITS: Dict[str, str] = {
    "delta": "per 1.00 of spot (units of underlying)",
    "delta_1pct": "P&L per +1% spot move (price units)",
    "gamma": "change in delta per 1.00 of spot",
    "gamma_1pct": "change in delta per +1% spot move",
    "vega_1vol": "P&L per +1 vol point (price units)",
    "theta_day": "P&L per calendar day, analytic theta / 365 (price units)",
    "theta_1d": "P&L over one calendar day by repricing (price units)",
    "rho_1bp": "P&L per +1bp of r (price units)",
    "rho_1pct": "P&L per +1% of r (price units)",
    "phi_1bp": "P&L per +1bp of q (price units)",
    "vanna_1vol": "change in delta per +1 vol point",
    "volga_1vol": "change in vega (per vol point) per +1 vol point (price units)",
    "charm_day": "change in delta per calendar day",
}


def desk_greeks(g: BSGreeks, S) -> Dict[str, np.ndarray]:
    """Greeks of a long option on one unit of underlying in desk units
    (:data:`DESK_UNITS`): signed as the P&L (or change in delta) for a +1%
    spot move, a +1 vol point (0.01) move, one calendar day passing (1/365)
    and a +1bp (0.0001) move in r or q. ``theta_1d`` needs repricing and is
    not included (see :func:`bs_theta_1d`)."""

    S = as_array(S)
    return {
        "delta": g.delta,
        "delta_1pct": g.delta * S * 0.01,
        "gamma": g.gamma,
        "gamma_1pct": g.gamma * S * 0.01,
        "vega_1vol": g.vega * 0.01,
        "theta_day": g.theta / DAYS_PER_YEAR,
        "rho_1bp": g.rho * 1e-4,
        "rho_1pct": g.rho * 0.01,
        "phi_1bp": g.phi * 1e-4,
        "vanna_1vol": g.vanna * 0.01,
        "volga_1vol": g.volga * 1e-4,
        "charm_day": g.charm / DAYS_PER_YEAR,
    }


def finite_difference_greeks(S, K, T, r, sigma, q=0.0, call=True) -> Dict[str, float]:
    """Central finite differences of :func:`bs_price` for one option (a
    check of the analytic Greeks against the pricer, raw units). Bumps:
    spot 1e-4 S, vol 1e-5 (volga: second differences at 2e-3 and 1e-3 with
    Richardson extrapolation), rates 1e-5, time ``min(1e-4, T / 4)`` years. Needs ``T > 0`` and ``sigma > 0``
    (the price is smooth there)."""

    S, K, T, r, sigma, q = (float(x) for x in (S, K, T, r, sigma, q))
    if T <= 0 or sigma <= 0:
        raise ValueError("finite differences need T > 0 and sigma > 0 (the price has a kink otherwise)")

    def v(s=S, t=T, rr=r, vol=sigma, qq=q) -> float:
        return float(bs_price(s, K, t, rr, vol, q=qq, call=call))

    hs, hv, hv2, hr, ht = 1e-4 * S, 1e-5, min(2e-3, sigma / 2.0), 1e-5, min(1e-4, T / 4.0)
    base = v()

    def dv_ds(**kw) -> float:
        return (v(s=S + hs, **kw) - v(s=S - hs, **kw)) / (2 * hs)

    def d2v_dvol2(h: float) -> float:
        return (v(vol=sigma + h) - 2 * base + v(vol=sigma - h)) / h**2

    return {
        "delta": dv_ds(),
        "gamma": (v(s=S + hs) - 2 * base + v(s=S - hs)) / hs**2,
        "vega": (v(vol=sigma + hv) - v(vol=sigma - hv)) / (2 * hv),
        "theta": -(v(t=T + ht) - v(t=T - ht)) / (2 * ht),
        "rho": (v(rr=r + hr) - v(rr=r - hr)) / (2 * hr),
        "phi": (v(qq=q + hr) - v(qq=q - hr)) / (2 * hr),
        "vanna": (dv_ds(vol=sigma + hv) - dv_ds(vol=sigma - hv)) / (2 * hv),
        "volga": (4.0 * d2v_dvol2(hv2 / 2.0) - d2v_dvol2(hv2)) / 3.0,
        "charm": -(dv_ds(t=T + ht) - dv_ds(t=T - ht)) / (2 * ht),
    }
