"""Garman-Kohlhagen FX option prices, Greeks and quoting conventions.

Conventions and units
---------------------
* ``S`` spot and ``K`` strike in quote-currency (domestic) units per one
  unit of base currency (foreign), both > 0.
* ``rd`` / ``rf`` are the domestic (quote) and foreign (base) interest
  rates, continuously compounded ACT/365F per-annum decimals.
* Two times (:class:`FxMarket`): ``tau_e`` years from the trade date to the
  expiry (the vol time) and ``tau_d`` years from the spot date to the
  delivery date (the rate time), both ACT/365F. With ``DF_d = exp(-rd
  tau_d)``, ``DF_f = exp(-rf tau_d)`` and ``F = S DF_f / DF_d`` (the forward
  to the delivery date)::

      V = DF_d [w F N(w d1) - w K N(w d2)],  d1,2 = (ln(F/K) +- sigma^2 tau_e / 2) / (sigma sqrt(tau_e))

  is the value on the spot date (when the premium is paid) of a long option
  on one unit of base currency, in quote-currency units; ``w = +1`` call
  (on the base currency), ``-1`` put. With ``tau_e = tau_d = T`` this is the
  textbook Garman-Kohlhagen price (QuantLib ``GarmanKohlagenProcess``).

Deltas (Clark 2011, sec. 3.x; Reiswich and Wystup 2010), per one unit of
base-currency notional, :data:`derivx.fx.conventions.DELTA_TYPES`:

=============  =========================================  =============================
type           formula                                    hedge
=============  =========================================  =============================
spot           ``w DF_f N(w d1)``                         spot, premium in quote ccy
forward        ``w N(w d1)``                              forward to delivery
spot_pa        ``w DF_f (K / F) N(w d2)`` = spot - V / S  spot, premium in base ccy
forward_pa     ``w (K / F) N(w d2)``                      forward, premium in base ccy
=============  =========================================  =============================

The premium-adjusted call delta is not monotone in the strike (it rises
from 0, peaks and falls back to 0): :func:`strike_from_delta` returns the
strike to the right of the peak, the market convention (and QuantLib's
``BlackDeltaCalculator``), and raises when the delta is above the peak.

ATM strikes (:data:`ATM_TYPES`): ``ATMF`` K = F; ``SPOT`` K = S; ``DNS``
(delta-neutral straddle) K = F exp(+sigma^2 tau_e / 2) for unadjusted
deltas and K = F exp(-sigma^2 tau_e / 2) for premium-adjusted deltas.

Premium quote styles (Wystup, *FX Options and Structured Products*, 1.x),
from ``V`` in quote currency per one unit of base currency:
``dom_pips = V / pip`` (quote-currency pips per one base unit),
``pct_for = V / S`` (percent of the base-currency notional),
``for_pips = V / (S K) / 0.0001`` (base-currency units, in 0.0001s, per one
unit of quote-currency notional) and ``pct_dom = V / K`` (percent of the
quote-currency notional ``K``).

Greeks (:class:`FxGreeks`) are for a long option on one unit of base
currency, signed as the P&L in quote currency for the stated move: delta
per 1.00 of spot (also per convention, above), gamma as the change in spot
delta per +1% spot, vega per +1 vol point, rho_d / rho_f per +1bp of rd /
rf, theta over one calendar day by repricing (both times shrink by 1/365,
spot, rates and vol unchanged), vanna as the change in spot delta per +1
vol point and volga as the change in vega (per vol point) per +1 vol point.
Inputs are validated, never clamped: ``ValueError`` on invalid values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from derivx.core.bs import bs_greeks
from derivx.fx.conventions import DAYS_PER_YEAR, DELTA_TYPES
from derivx.vol.implied import black_price

ATM_TYPES = ("DNS", "ATMF", "SPOT")
ATM_TYPE_LABELS = {
    "DNS": "ATM DNS (delta-neutral straddle)",
    "ATMF": "ATMF (K = forward)",
    "SPOT": "ATM SPOT (K = spot)",
}
PREMIUM_STYLES = ("dom_pips", "pct_for", "for_pips", "pct_dom")
_MAX_EXPONENT = 700.0
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_RTOL = 4 * np.finfo(float).eps


def _ncdf(x: float) -> float:
    """Standard normal CDF of a float (erfc form: accurate in the left tail)."""

    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _npdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / _SQRT_2PI


def check_delta_type(delta_type: str) -> str:
    if delta_type not in DELTA_TYPES:
        raise ValueError(f"delta type must be one of {DELTA_TYPES}, got {delta_type!r}")
    return delta_type


def is_premium_adjusted(delta_type: str) -> bool:
    return check_delta_type(delta_type).endswith("_pa")


@dataclass(frozen=True)
class FxMarket:
    """Spot, rates and the two times of one FX option expiry (module docs)."""

    S: float
    rd: float
    rf: float
    tau_e: float
    tau_d: float

    def __post_init__(self) -> None:
        for name in ("S", "rd", "rf", "tau_e", "tau_d"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}")
        if self.S <= 0:
            raise ValueError(f"spot must be > 0, got {self.S}")
        if self.tau_e < 0 or self.tau_d < 0:
            raise ValueError(f"times must be >= 0 years (tau_e {self.tau_e}, tau_d {self.tau_d})")
        for name, x in (("rd tau_d", self.rd * self.tau_d), ("rf tau_d", self.rf * self.tau_d)):
            if abs(x) > _MAX_EXPONENT:
                raise ValueError(f"{name} = {x:g} overflows the discount factor: rates are decimals (0.05 = 5%)")

    @property
    def df_d(self) -> float:
        return math.exp(-self.rd * self.tau_d)

    @property
    def df_f(self) -> float:
        return math.exp(-self.rf * self.tau_d)

    @property
    def forward(self) -> float:
        return self.S * self.df_f / self.df_d

    def std(self, sigma) -> np.ndarray:
        return np.asarray(sigma, dtype=float) * np.sqrt(self.tau_e)

    def shifted(self, **changes: float) -> "FxMarket":
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(changes)
        return FxMarket(**values)


def _strikes(K) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    if not np.all(np.isfinite(K)) or np.any(K <= 0):
        raise ValueError(f"strikes must be finite and > 0, got {K.ravel()[:3]}")
    return K


def _vols(sigma) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError(f"vols must be finite and >= 0 (decimals, 0.08 = 8%), got {sigma.ravel()[:3]}")
    return sigma


def _w(call) -> np.ndarray:
    return np.where(np.asarray(call, dtype=bool), 1.0, -1.0)


def _out(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def gk_price(m: FxMarket, K, sigma, call=True):
    """Value on the spot date, quote currency per one unit of base currency
    (module docs). Vectorised over ``K``, ``sigma`` and ``call``."""

    K, sigma = _strikes(K), _vols(sigma)
    return _out(black_price(m.forward, K, m.tau_e, sigma, is_call=np.asarray(call, dtype=bool), df=m.df_d))


def _d1d2(m: FxMarket, K, sigma) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K, sigma = _strikes(K), _vols(sigma)
    std = m.std(sigma)
    if np.any(std <= 0):
        raise ValueError("deltas need sigma > 0 and tau_e > 0 (at expiry the delta is the payoff's step)")
    d1 = (np.log(m.forward / K) + 0.5 * std * std) / std
    return d1, d1 - std, std


def fx_delta(m: FxMarket, K, sigma, call=True, delta_type: str = "spot"):
    """Delta of a long option on one unit of base currency in one of the
    four conventions (module docs). Needs ``sigma sqrt(tau_e) > 0``."""

    check_delta_type(delta_type)
    d1, d2, _ = _d1d2(m, K, sigma)
    w = _w(call)
    if delta_type in ("spot", "forward"):
        delta = w * ndtr(w * d1)
    else:
        delta = w * (np.asarray(K, dtype=float) / m.forward) * ndtr(w * d2)
    return _out(delta * (m.df_f if delta_type.startswith("spot") else 1.0))


def _delta_scale(m: FxMarket, delta_type: str) -> float:
    return m.df_f if delta_type.startswith("spot") else 1.0


def max_pa_call_delta(m: FxMarket, sigma: float, delta_type: str) -> Tuple[float, float]:
    """``(strike, delta)`` at the peak of the premium-adjusted call delta:
    the largest PA call delta any strike reaches (``sigma sqrt(tau_e) N(d2)
    = n(d2)`` there)."""

    if not is_premium_adjusted(delta_type):
        raise ValueError("only premium-adjusted call deltas have a maximum below the discount factor")
    std = float(sigma) * math.sqrt(m.tau_e)
    if not std > 0:
        raise ValueError("needs sigma > 0 and tau_e > 0")
    # h(d2) = std N(d2) - n(d2) is negative for d2 <= -std and increasing beyond: one root, the peak
    d2 = brentq(lambda x: std * _ncdf(x) - math.exp(-0.5 * x * x) / _SQRT_2PI, -std, 40.0, xtol=1e-15, rtol=_RTOL)
    K = m.forward * math.exp(-std * d2 - 0.5 * std * std)
    return K, _delta_scale(m, delta_type) * K / m.forward * _ncdf(d2)


def strike_from_delta(m: FxMarket, delta: float, sigma: float, call: bool = True, delta_type: str = "spot") -> float:
    """Strike whose delta (convention ``delta_type``, signed: puts negative)
    is ``delta`` at vol ``sigma``. Unadjusted deltas invert in closed form;
    premium-adjusted deltas are solved by Brent's method, for calls on the
    branch to the right of the delta's peak (the market convention).
    Raises ``ValueError`` when no strike has that delta."""

    check_delta_type(delta_type)
    delta, sigma = float(delta), float(sigma)
    if not (math.isfinite(sigma) and sigma > 0 and m.tau_e > 0):
        raise ValueError(f"strike from delta needs sigma > 0 and tau_e > 0 (got sigma {sigma}, tau_e {m.tau_e})")
    std = sigma * math.sqrt(m.tau_e)
    w = 1.0 if call else -1.0
    scale = _delta_scale(m, delta_type)
    if not math.isfinite(delta) or w * delta <= 0:
        raise ValueError(f"a {'call' if call else 'put'} delta must be {'> 0' if call else '< 0'}, got {delta}")
    if w * delta >= scale:
        raise ValueError(f"|delta| {abs(delta):.6g} is not below its maximum {scale:.6g} for this convention")
    unadjusted = m.forward * math.exp(-w * std * float(ndtri(w * delta / scale)) + 0.5 * std * std)
    if not is_premium_adjusted(delta_type):
        return unadjusted

    F = m.forward

    def f(log_k: float) -> float:  # scalar premium-adjusted delta minus the target (pure math: fast)
        d2 = (math.log(F) - log_k - 0.5 * std * std) / std
        return w * scale * math.exp(log_k) / F * _ncdf(w * d2) - delta

    hi = math.log(unadjusted)  # the premium-adjusted strike is always below the unadjusted one
    if call:
        k_peak, d_peak = max_pa_call_delta(m, sigma, delta_type)
        if delta > d_peak:
            raise ValueError(
                f"premium-adjusted call delta {delta:.6g} is above its maximum {d_peak:.6g} (reached at K = "
                f"{k_peak:.6g}): no strike has this delta; use a smaller delta or an unadjusted convention"
            )
        lo = math.log(k_peak)
        if delta == d_peak:
            return k_peak
    else:
        lo = hi - 1.0
        while f(lo) < 0:  # |put delta| increases with K: move left until below the target
            lo -= 1.0
            if lo < hi - 200.0:  # pragma: no cover - |delta| -> 0 as K -> 0
                raise ValueError("no strike found for this put delta")
    return math.exp(brentq(f, lo, hi, xtol=1e-15, rtol=_RTOL, maxiter=500))


def atm_strike(m: FxMarket, sigma: float, atm_type: str = "DNS", premium_adjusted: bool = False) -> float:
    """ATM strike of ``atm_type`` (module docs); ``premium_adjusted`` picks
    the DNS strike of premium-adjusted deltas."""

    if atm_type not in ATM_TYPES:
        raise ValueError(f"ATM type must be one of {ATM_TYPES}, got {atm_type!r}")
    if atm_type == "ATMF":
        return m.forward
    if atm_type == "SPOT":
        return m.S
    var = float(_vols(sigma)) ** 2 * m.tau_e
    return float(m.forward * np.exp(-0.5 * var if premium_adjusted else 0.5 * var))


def premium_styles(price, S: float, K, pip: float) -> Dict[str, np.ndarray]:
    """The four premium quote styles of a value ``price`` (quote currency per
    one unit of base currency), module docs: ``dom_pips``, ``pct_for`` (in
    percent), ``for_pips`` (0.0001 units of base currency per one unit of
    quote currency) and ``pct_dom`` (in percent)."""

    price, K = np.asarray(price, dtype=float), _strikes(K)
    return {
        "dom_pips": _out(price / pip),
        "pct_for": _out(100.0 * price / S),
        "for_pips": _out(price / (S * K) / 1e-4),
        "pct_dom": _out(100.0 * price / K),
    }


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FxGreeks:
    """Price and Greeks of a long option on one unit of base currency
    (units in the module docs; amounts in quote currency)."""

    price: float
    delta_spot: float
    delta_forward: float
    delta_spot_pa: float
    delta_forward_pa: float
    gamma: float  # per 1.00 of spot
    gamma_1pct: float  # change in spot delta per +1% spot
    vega_1vol: float
    theta_1d: float
    rho_d_1bp: float
    rho_f_1bp: float
    vanna_1vol: float
    volga_1vol: float
    forward: float

    def delta(self, delta_type: str) -> float:
        check_delta_type(delta_type)
        return {
            "spot": self.delta_spot,
            "forward": self.delta_forward,
            "spot_pa": self.delta_spot_pa,
            "forward_pa": self.delta_forward_pa,
        }[delta_type]

    def as_dict(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _theta_1d(m: FxMarket, K: float, sigma: float, call: bool, price: float) -> float:
    later = m.shifted(tau_e=max(m.tau_e - 1.0 / DAYS_PER_YEAR, 0.0), tau_d=max(m.tau_d - 1.0 / DAYS_PER_YEAR, 0.0))
    return float(gk_price(later, K, sigma, call)) - price


def fx_greeks(m: FxMarket, K: float, sigma: float, call: bool = True) -> FxGreeks:
    """Price and Greeks of one option (module docs). At expiry (``tau_e =
    0``) or zero vol the value is the discounted intrinsic value of the
    forward, the spot delta is its step (``w DF_f`` in the money, half on the
    forward) and gamma, vega, vanna and volga are 0; the rhos are those of
    the discounted intrinsic value."""

    K, sigma = float(_strikes(K)), float(_vols(sigma))
    w = 1.0 if call else -1.0
    price = float(gk_price(m, K, sigma, call))
    F, df_f = m.forward, m.df_f
    if m.tau_e > 0:
        # Garman-Kohlhagen with two times is BSM with T = tau_e and rates
        # scaled by tau_d / tau_e (the same discount factors)
        scale = m.tau_d / m.tau_e
        g = bs_greeks(m.S, K, m.tau_e, m.rd * scale, sigma, q=m.rf * scale, call=call)
        delta, gamma = float(g.delta), float(g.gamma)
        vega, vanna, volga = float(g.vega), float(g.vanna), float(g.volga)
        rho_d, rho_f = float(g.rho) * scale, float(g.phi) * scale
    else:
        itm = w * (F - K) > 0
        on_fwd = abs(np.log(F / K)) <= 1e-12
        step = 0.5 if on_fwd else (1.0 if itm else 0.0)
        delta = w * df_f * step
        gamma = vega = vanna = volga = 0.0
        rho_d = w * K * m.tau_d * m.df_d * step
        rho_f = -w * m.S * m.tau_d * df_f * step
    delta_spot_pa = delta - price / m.S
    return FxGreeks(
        price=price,
        delta_spot=delta,
        delta_forward=delta / df_f,
        delta_spot_pa=delta_spot_pa,
        delta_forward_pa=delta_spot_pa / df_f,
        gamma=gamma,
        gamma_1pct=gamma * m.S * 0.01,
        vega_1vol=vega * 0.01,
        theta_1d=_theta_1d(m, K, sigma, call, price),
        rho_d_1bp=rho_d * 1e-4,
        rho_f_1bp=rho_f * 1e-4,
        vanna_1vol=vanna * 0.01,
        volga_1vol=volga * 1e-4,
        forward=F,
    )


# ---------------------------------------------------------------------------
# Notional, premium amounts and scenario P&L
# ---------------------------------------------------------------------------


def base_notional(notional: float, notional_in_base: bool, K: float) -> float:
    """Notional in base currency: as given, or a quote-currency notional
    divided by the strike (the base amount exchanged at exercise)."""

    notional, K = float(notional), float(K)
    if not np.isfinite(notional):
        raise ValueError("notional must be a finite number")
    if not (np.isfinite(K) and K > 0):
        raise ValueError("strike must be > 0")
    return notional if notional_in_base else notional / K


def premium_amount(price: float, S: float, notional_base: float, in_base: bool) -> float:
    """Premium of ``notional_base`` options: in quote currency
    ``N V``, or in base currency ``N V / S`` (converted at spot)."""

    amount = float(notional_base) * float(price)
    return amount / float(S) if in_base else amount


@dataclass(frozen=True)
class FxScenarioGrid:
    """Full-revaluation P&L (quote currency) of a signed base-currency
    notional: ``pnl[i, j]`` for vol shock ``vol_shocks[i]`` (absolute,
    decimals) and spot shock ``spot_shocks[j]`` (relative), after
    ``horizon_days`` calendar days (both times shrink, spot / rates
    unchanged otherwise)."""

    spot_shocks: np.ndarray
    vol_shocks: np.ndarray
    pnl: np.ndarray
    base_value: float
    horizon_days: float
    notional_base: float
    dropped_vol_shocks: Tuple[float, ...]


def scenario_pnl(
    m: FxMarket,
    K: float,
    sigma: float,
    call: bool,
    notional_base: float,
    spot_shocks,
    vol_shocks,
    horizon_days: float = 0.0,
) -> FxScenarioGrid:
    """P&L of ``notional_base`` long options for each (vol, spot) shock;
    vol shocks that would make the vol negative are dropped (listed), never
    floored."""

    if not np.isfinite(horizon_days) or horizon_days < 0:
        raise ValueError(f"horizon must be >= 0 calendar days, got {horizon_days}")
    spot_shocks = np.asarray(spot_shocks, dtype=float)
    vol_shocks = np.asarray(vol_shocks, dtype=float)
    if np.any(spot_shocks <= -1):
        raise ValueError("spot shocks are relative and must be > -100%")
    shocked = sigma + vol_shocks
    shocked = np.where(np.abs(shocked) <= 1e-12 * max(sigma, 1e-3), 0.0, shocked)
    keep = shocked >= 0
    h = horizon_days / DAYS_PER_YEAR
    later = m.shifted(tau_e=max(m.tau_e - h, 0.0), tau_d=max(m.tau_d - h, 0.0))
    base = float(gk_price(m, K, sigma, call))
    rows = []
    for vol in shocked[keep]:
        row = [float(gk_price(later.shifted(S=m.S * (1.0 + s)), K, vol, call)) for s in spot_shocks]
        rows.append(row)
    pnl = notional_base * (np.asarray(rows, dtype=float) - base)
    dropped = tuple(float(v) for v in vol_shocks[~keep])
    return FxScenarioGrid(spot_shocks, vol_shocks[keep], pnl, base, float(horizon_days), float(notional_base), dropped)
