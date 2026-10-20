"""FX smiles from market quotes: ATM vol, 25-delta and 10-delta risk
reversals (RR) and butterflies (BF) per tenor.

Quotes (vols are per-annum decimals: 0.0750 = 7.50%)
----------------------------------------------------
* ``atm``: the ATM vol, at the ATM strike of the convention (ATM DNS by
  default, :func:`derivx.fx.pricing.atm_strike`).
* ``rr25`` / ``rr10``: ``sigma(X-delta call) - sigma(X-delta put)``, X = 25, 10.
* ``bf25`` / ``bf10``: the butterfly, in one of two conventions
  (:data:`STRANGLE_CONVENTIONS`):

  - ``smile`` (smile strangle): ``(sigma_XC + sigma_XP) / 2 - sigma_ATM``,
    with the vols read off the smile at its own X-delta strikes;
  - ``market`` (market or broker strangle, the interbank quote): a
    strangle whose two strikes have delta +X and -X at the single vol
    ``sigma_ATM + BF`` and is priced at that vol costs the same as the same
    two strikes priced on the smile. The smile-strangle butterfly that
    achieves this differs from the quote and is solved for.

Deltas use :class:`SmileConventions` (spot / forward, premium-adjusted or
not; :mod:`derivx.fx.pricing`).

Construction (Clark 2011, ch. 3.7; Castagna and Mercurio 2007)
-------------------------------------------------------------
1. The ATM strike from ``sigma_ATM`` and the ATM type; every model hits
   ``sigma_ATM`` there exactly.
2. A model (:data:`SMILE_MODELS`):

   - ``SABR``: Hagan's lognormal expansion with beta = 1 *fixed* (the FX
     standard; :mod:`derivx.vol.sabr`). Alpha is solved from the ATM vol at
     the ATM strike; rho and nu are fitted by least squares to the quotes
     themselves: each RR, and each BF as a smile strangle (vols at the
     smile's own delta strikes) or as the market strangle's value error
     divided by its vega. With 25-delta quotes only that is two equations
     for two unknowns, solved exactly; with 10-delta quotes too it is a
     least-squares compromise whose errors are reported.
   - ``VV``: vanna-volga with Castagna-Mercurio's exact weights through
     three pillars ``sigma_25C/P = sigma_ATM + BF_ss +- RR / 2`` at the
     strikes whose delta at those vols is +-25%. With the smile convention
     ``BF_ss`` is the quote; with the market convention ``BF_ss`` is
     iterated (damped Newton) until the smile reprices the market
     strangle. It uses no 10-delta quote; those are only checked.

3. Every quote is checked on the final smile (:attr:`FxSmile.errors`, vol
   decimals): the ATM vol at the ATM strike; RR and smile-strangle BF at the
   smile's own delta strikes (the strike K with ``delta(K, sigma(K)) = X``);
   a market strangle as its value difference divided by its vega. The
   smile-strangle fly implied by the final smile is
   :attr:`FxSmile.smile_strangles`.

Smiles are not market data unless the quotes are: the page labels its
default quotes ILLUSTRATIVE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from derivx.fx.conventions import DELTA_TYPES
from derivx.fx.pricing import (
    ATM_TYPES,
    FxMarket,
    atm_strike,
    gk_price,
    is_premium_adjusted,
    strike_from_delta,
)
from derivx.vol.implied import black_vega, implied_black_vol
from derivx.vol.sabr import sabr_alpha_from_atm, sabr_vol, z_over_x

SMILE_MODELS = ("SABR", "VV")
MODEL_LABELS = {"SABR": "SABR (Hagan lognormal, β = 1 fixed)", "VV": "VANNA-VOLGA (Castagna-Mercurio)"}
STRANGLE_CONVENTIONS = ("market", "smile")
STRANGLE_LABELS = {"market": "MARKET STRANGLE (broker fly)", "smile": "SMILE STRANGLE"}
PILLAR_LABELS = ("10P", "25P", "ATM", "25C", "10C")
SABR_BETA = 1.0
RHO_BOUND = 0.9999
MATCH_TOL = 1e-9  # vol decimals (0.00001 vol bp): a quote is matched exactly when its error is below this
NOTE_TOL = 1e-4  # 1 vol bp: a larger quote error gets a note on the page (every error is shown anyway)
# Calibration budgets, deterministic (counts, not seconds); on market-like quotes the fit needs at most 4
# fixed-point cycles per delta strike and 25 function evaluations per start, so they never bind there.
STALL_CYCLES = 12  # a trial smile's delta strike fails when its fixed-point step has not halved in 12 cycles
FIT_CYCLES = 30  # ... or after 30 cycles
FIT_NFEV = 80  # function evaluations per least-squares start
FIT_BUDGET = 400  # residual evaluations per SABR fit: no further starts once spent


def _delta_grid(deltas: Sequence[int]) -> Tuple[Tuple[str, float, Optional[bool]], ...]:
    """(label, signed delta, call) from the put wing through ATM to the call wing."""

    return tuple(
        [(f"{d}P", -d / 100.0, False) for d in deltas]
        + [("ATM", float("nan"), None)]
        + [(f"{d}C", d / 100.0, True) for d in reversed(deltas)]
    )


SURFACE_DELTAS = _delta_grid((10, 15, 20, 25, 35))  # columns of the tenor x delta surface
SMILE_DELTAS = _delta_grid((5, 10, 15, 20, 25, 30, 35, 40, 45))  # the smile in delta space


@dataclass(frozen=True)
class TenorQuote:
    """Vol quotes of one tenor (decimals). The 10-delta pair is optional,
    but ``rr10`` and ``bf10`` come together."""

    tenor: str
    atm: float
    rr25: float
    bf25: float
    rr10: Optional[float] = None
    bf10: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("atm", "rr25", "bf25"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{self.tenor}: {name.upper()} must be a number")
        if self.atm <= 0:
            raise ValueError(f"{self.tenor}: ATM vol must be > 0 (decimal, 0.075 = 7.5%), got {self.atm}")
        if self.atm > 3.0:
            raise ValueError(f"{self.tenor}: ATM vol {self.atm} is above 300%: vols are decimals (0.075 = 7.5%)")
        ten = (self.rr10, self.bf10)
        if (ten[0] is None) != (ten[1] is None):
            raise ValueError(f"{self.tenor}: give both RR10 and BF10, or neither")
        if ten[0] is not None and not (np.isfinite(ten[0]) and np.isfinite(ten[1])):
            raise ValueError(f"{self.tenor}: RR10 and BF10 must be numbers")

    @property
    def has_10d(self) -> bool:
        return self.rr10 is not None

    @property
    def levels(self) -> Tuple[int, ...]:
        return (25, 10) if self.has_10d else (25,)

    def rr(self, level: int) -> float:
        return float({25: self.rr25, 10: self.rr10}[level])

    def bf(self, level: int) -> float:
        return float({25: self.bf25, 10: self.bf10}[level])


@dataclass(frozen=True)
class SmileConventions:
    """How the quotes are defined: delta type, ATM type, strangle convention."""

    delta_type: str = "spot"
    atm_type: str = "DNS"
    strangle: str = "market"

    def __post_init__(self) -> None:
        if self.delta_type not in DELTA_TYPES:
            raise ValueError(f"delta type must be one of {DELTA_TYPES}")
        if self.atm_type not in ATM_TYPES:
            raise ValueError(f"ATM type must be one of {ATM_TYPES}")
        if self.strangle not in STRANGLE_CONVENTIONS:
            raise ValueError(f"strangle convention must be one of {STRANGLE_CONVENTIONS}")

    @property
    def premium_adjusted(self) -> bool:
        return is_premium_adjusted(self.delta_type)


@dataclass(frozen=True)
class MarketStrangle:
    """The market strangle at delta ``level``: strikes with delta +-X at the
    flat vol ``sigma_ATM + BF``, and its value (call + put, quote currency per
    one unit of base) and vega (per 1.00 of vol) at that vol."""

    level: int
    vol: float
    call_strike: float
    put_strike: float
    value: float
    vega: float

    def smile_value(self, m: FxMarket, vol_fn: Callable) -> float:
        vols = vol_fn(np.array([self.call_strike, self.put_strike]))
        return float(gk_price(m, self.call_strike, vols[0], True) + gk_price(m, self.put_strike, vols[1], False))

    def error(self, m: FxMarket, vol_fn: Callable) -> float:
        """(smile value - quoted value) / vega: the mismatch in vol units."""

        return (self.smile_value(m, vol_fn) - self.value) / self.vega


def market_strangle(m: FxMarket, quote: TenorQuote, level: int, delta_type: str) -> MarketStrangle:
    vol = quote.atm + quote.bf(level)
    if vol <= 0:
        raise ValueError(f"{quote.tenor}: ATM + BF{level} = {vol:.4%} is not a positive vol")
    kc = strike_from_delta(m, level / 100.0, vol, True, delta_type)
    kp = strike_from_delta(m, -level / 100.0, vol, False, delta_type)
    value = float(gk_price(m, kc, vol, True) + gk_price(m, kp, vol, False))
    vega = float(
        black_vega(m.forward, kc, m.tau_e, vol, df=m.df_d) + black_vega(m.forward, kp, m.tau_e, vol, df=m.df_d)
    )
    return MarketStrangle(level, vol, kc, kp, value, vega)


def smile_delta_strikes(
    m: FxMarket,
    vol_fn: Callable,
    deltas: Sequence[float],
    calls: Sequence[bool],
    delta_type: str,
    guess_vols,
    stall_cycles: Optional[int] = None,
    max_cycles: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Strikes ``K_i`` with ``delta(K_i, sigma(K_i)) = deltas[i]`` on the smile
    ``vol_fn`` and their vols, all solved together by the fixed-point
    iteration ``K <- strike_from_delta(delta, sigma(K))`` (one smile
    evaluation per step for every strike; it contracts for any smile whose
    slope in strike is moderate). Raises ``ValueError`` when it does not
    converge in ``max_cycles`` cycles, or the smile has no vol on the way.

    The calibration's inner loop passes a smaller ``max_cycles`` and a
    ``stall_cycles`` (give up as soon as the step has not halved in that
    many cycles): market-like smiles converge in 2-4 cycles, and a smile
    the optimiser merely tries should fail fast rather than spin through
    100 cycles hundreds of times (that stalled the page for a minute on
    extreme quotes)."""

    deltas, calls = list(deltas), list(calls)
    guess = np.broadcast_to(np.asarray(guess_vols, dtype=float), (len(deltas),))

    def g(K: np.ndarray) -> np.ndarray:  # the fixed-point map
        vols = np.asarray(vol_fn(K), dtype=float).reshape(K.shape)
        bad = ~(np.isfinite(vols) & (vols > 0))
        if np.any(bad):
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"the smile has no vol at K = {K[i]:.6g} (on the way to the {abs(deltas[i]):.0%}-delta strike)"
            )
        return np.array([strike_from_delta(m, d, v, c, delta_type) for d, v, c in zip(deltas, vols, calls)])

    def done(new: np.ndarray, old: np.ndarray) -> bool:
        return bool(np.all(np.abs(new - old) <= 1e-14 * old))

    K = np.array([strike_from_delta(m, d, v, c, delta_type) for d, v, c in zip(deltas, guess, calls)])
    best, stalled = math.inf, 0
    for _ in range(max_cycles):  # Aitken-accelerated (Steffensen) fixed point: two map steps per cycle
        K1 = g(K)
        if done(K1, K):
            return K1, np.asarray(vol_fn(K1), dtype=float).reshape(K.shape)
        K2 = g(K1)
        if done(K2, K1):
            return K2, np.asarray(vol_fn(K2), dtype=float).reshape(K.shape)
        if stall_cycles is not None:
            step = float(np.max(np.abs(K2 - K1) / K1))
            best, stalled = (step, 0) if step < 0.5 * best else (best, stalled + 1)
            if stalled >= stall_cycles:
                break
        denom = K2 - 2.0 * K1 + K
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = K - (K1 - K) ** 2 / denom
        sane = np.isfinite(acc) & (acc > 0) & (np.abs(acc - K2) <= 10.0 * np.abs(K2 - K1) + 1e-300)
        K = np.where(sane, acc, K2)
    raise ValueError("no smile delta strike: the fixed point did not converge (smile too steep in strike)")


def smile_delta_strike(
    m: FxMarket, vol_fn: Callable, delta: float, call: bool, delta_type: str, guess_vol: float
) -> float:
    """One strike of :func:`smile_delta_strikes`."""

    K, _ = smile_delta_strikes(m, vol_fn, [delta], [call], delta_type, guess_vol)
    return float(K[0])


def sabr_beta1_vol(F: float, K, T: float, alpha: float, rho: float, nu: float) -> np.ndarray:
    """Hagan lognormal SABR vol with beta = 1, ``alpha z/x(z) [1 + T (rho nu
    alpha / 4 + (2 - 3 rho^2) nu^2 / 24)]``, ``z = nu / alpha ln(F / K)``
    (:func:`derivx.vol.sabr.sabr_vol` with beta = 1, its ``z_over_x``, and no
    input checks: the calibration's inner loop). Callers validate."""

    K = np.asarray(K, dtype=float)
    z = (nu / alpha) * np.log(F / K)
    return alpha * z_over_x(z, rho) * (1.0 + T * (0.25 * rho * nu * alpha + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _safe_vols(fn: Callable, K) -> np.ndarray:
    """``fn(K)`` with NaN wherever the model has no vol (instead of raising)."""

    K = np.atleast_1d(np.asarray(K, dtype=float))
    try:
        return np.asarray(fn(K), dtype=float).reshape(K.shape)
    except ValueError:
        out = np.full(K.shape, np.nan)
        for i, k in enumerate(K.ravel()):
            try:
                out.flat[i] = float(np.asarray(fn(np.array([k])), dtype=float).ravel()[0])
            except ValueError:
                pass
        return out


def _sabr_alpha(F: float, T: float, k_atm: float, vol_atm: float, rho: float, nu: float) -> float:
    """Alpha with ``sigma_SABR(K_ATM) = vol_ATM`` (beta = 1): start from the
    ATM-forward cubic, then rescale (the vol is nearly proportional to
    alpha) until it matches to 1e-15."""

    a = sabr_alpha_from_atm(F, T, vol_atm, SABR_BETA, rho, nu)
    for _ in range(60):
        vol = float(sabr_beta1_vol(F, k_atm, T, a, rho, nu))
        if not (np.isfinite(vol) and vol > 0):
            break
        if abs(vol - vol_atm) <= 1e-15 * vol_atm:
            return a
        a *= vol_atm / vol
    raise ValueError("SABR alpha does not reproduce the ATM vol at the ATM strike")


@dataclass(frozen=True)
class _SabrFit:
    alpha: float
    rho: float
    nu: float
    residuals: np.ndarray  # quote conditions, vol decimals
    at_bound: Tuple[str, ...]
    budget_spent: bool = False  # stopped by FIT_BUDGET / FIT_NFEV before converging


def _quote_residuals(
    m: FxMarket,
    quote: TenorQuote,
    conventions: SmileConventions,
    strangles: Dict[int, "MarketStrangle"],
    fn: Callable,
    levels: Sequence[int],
    guesses: Optional[Dict[Tuple[int, bool], float]] = None,
    stall_cycles: Optional[int] = None,
    max_cycles: int = 100,
) -> Tuple[np.ndarray, Dict[Tuple[int, bool], Tuple[float, float]]]:
    """Errors of the RR and BF quotes of ``levels`` on the smile ``fn`` (vol
    decimals, ``[RR, BF]`` per level) and the smile's own delta strikes and
    vols ``{(level, call): (K, vol)}`` (module docs, step 5);
    ``stall_cycles`` and ``max_cycles`` as in :func:`smile_delta_strikes`."""

    keys = [(lv, call) for lv in levels for call in (True, False)]
    guess = []
    for lv, call in keys:
        side = 0.5 * quote.rr(lv) if call else -0.5 * quote.rr(lv)
        guess.append((guesses or {}).get((lv, call), max(quote.atm + quote.bf(lv) + side, 0.5 * quote.atm)))
    deltas = [(lv if call else -lv) / 100.0 for lv, call in keys]
    K, vols = smile_delta_strikes(
        m, fn, deltas, [c for _, c in keys], conventions.delta_type, guess, stall_cycles, max_cycles
    )
    found = {key: (float(k), float(v)) for key, k, v in zip(keys, K, vols)}
    out = []
    for lv in levels:
        (_, vc), (_, vp) = found[(lv, True)], found[(lv, False)]
        out.append(vc - vp - quote.rr(lv))
        if conventions.strangle == "smile":
            out.append(0.5 * (vc + vp) - quote.atm - quote.bf(lv))
        else:
            out.append(strangles[lv].error(m, fn))
    return np.array(out), found


def _calibrate_sabr(
    m: FxMarket,
    quote: TenorQuote,
    conventions: SmileConventions,
    k_atm: float,
    strangles: Dict[int, "MarketStrangle"],
    start: Optional[Tuple[float, float]] = None,
) -> _SabrFit:
    """SABR (beta = 1) on the quotes: alpha from the ATM vol at the ATM strike
    (exact), (rho, nu) by least squares on the RR and BF conditions, each
    measured at the smile's own delta strikes (and, for market strangles, as
    the strangle's value error / vega). Two conditions for two unknowns with
    25-delta quotes only, so those are matched exactly."""

    F, T = m.forward, m.tau_e
    nu_max = max(5.0, 4.0 / math.sqrt(T))
    n_res = 2 * len(quote.levels)
    guesses: Dict[Tuple[int, bool], float] = {}

    def smile(x: np.ndarray) -> Tuple[float, Callable]:
        alpha = _sabr_alpha(F, T, k_atm, quote.atm, float(x[0]), float(x[1]))
        return alpha, lambda K: sabr_beta1_vol(F, K, T, alpha, float(x[0]), float(x[1]))

    evaluations = [0]

    def resid(x: np.ndarray) -> np.ndarray:
        evaluations[0] += 1
        try:
            _, fn = smile(x)
            r, found = _quote_residuals(
                m, quote, conventions, strangles, fn, quote.levels, guesses, STALL_CYCLES, FIT_CYCLES
            )
        except ValueError:
            return np.full(n_res, 1.0)  # 100 vol points: the expansion broke down or no delta strike
        guesses.update({key: vol for key, (_, vol) in found.items()})  # warm start the next evaluation
        return r

    lb, ub = np.array([-RHO_BOUND, 0.0]), np.array([RHO_BOUND, nu_max])
    skew = float(np.sign(quote.rr25))
    scale = 1.0 / math.sqrt(min(max(T, 1.0 / 52.0), 10.0))
    seeds = [(0.25 * skew, 0.8 * scale), (0.0, 0.3 * scale), (0.6 * skew, 1.6 * scale)]
    if start is not None:
        seeds.insert(0, start)
    best, spent = None, False
    for rho0, nu0 in seeds:
        x0 = np.clip([rho0, min(nu0, 0.9 * nu_max)], lb + 1e-9, ub - 1e-9)
        res = least_squares(
            resid, x0, bounds=(lb, ub), x_scale="jac", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=FIT_NFEV
        )
        if best is None or res.cost < best.cost:
            best = res
        if np.max(np.abs(res.fun)) <= 0.01 * MATCH_TOL or (start is not None and res.success):
            break
        if evaluations[0] >= FIT_BUDGET:
            spent = True
            break
    spent = spent or (best.status == 0)  # status 0: the start ran out of FIT_NFEV evaluations
    alpha, _ = smile(best.x)
    span = ub - lb
    at_bound = tuple(
        name
        for name, x, lo, hi, s in zip(("rho", "nu"), best.x, lb, ub, span)
        if (x - lo <= 1e-4 * s and name == "rho") or hi - x <= 1e-4 * s
    )
    return _SabrFit(alpha, float(best.x[0]), float(best.x[1]), np.asarray(resid(best.x)), at_bound, spent)


def _newton(fn: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, h: float = 1e-5) -> Tuple[np.ndarray, bool]:
    """Damped Newton iteration for ``fn(x) = 0`` (one unknown per market
    strangle, in vol decimals): finite-difference Jacobian with step ``h``
    (0.1 vol bp, well above the inner fit's tolerance), steps halved until
    the residual falls. Returns ``(x, converged)``."""

    x = np.asarray(x0, dtype=float)
    r = fn(x)
    for _ in range(40):
        if np.max(np.abs(r)) <= 0.01 * MATCH_TOL:
            return x, True
        jac = np.empty((x.size, x.size))
        for j in range(x.size):
            step = np.zeros_like(x)
            step[j] = h
            jac[:, j] = (fn(x + step) - r) / h
        try:
            dx = np.linalg.solve(jac, r)
        except np.linalg.LinAlgError:
            return x, False
        t = 1.0
        while t > 1e-4:
            try:
                trial = x - t * dx
                r_trial = fn(trial)
                if np.all(np.isfinite(r_trial)) and np.max(np.abs(r_trial)) < np.max(np.abs(r)):
                    break
            except ValueError:
                pass
            t *= 0.5
        else:
            return x, bool(np.max(np.abs(r)) <= MATCH_TOL)
        x, r = trial, r_trial
    return x, bool(np.max(np.abs(r)) <= MATCH_TOL)


class VannaVolgaSmile:
    """Vanna-volga smile through the pillars 25P, ATM, 25C (Castagna and
    Mercurio 2007): ``C(K) = C_BS(K, sigma_ATM) + sum_i x_i(K) [C_BS(K_i,
    sigma_i) - C_BS(K_i, sigma_ATM)]`` with ``x_i(K) = vega(K) / vega(K_i)
    prod_{j != i} ln(K_j / K) / ln(K_j / K_i)`` (vegas at sigma_ATM), then
    the Black vol of ``C(K)``. ``x_i(K_j) = 1{i = j}``, so the pillars are
    reproduced exactly. Where ``C(K)`` leaves the no-arbitrage bounds (far
    wings) the vol is NaN. Calling it maps strikes to vols."""

    def __init__(self, m: FxMarket, pillars: Dict[str, Tuple[float, float]]):
        (k1, s1), (k2, s2), (k3, s3) = pillars["25P"], pillars["ATM"], pillars["25C"]
        if not k1 < k2 < k3:
            raise ValueError(f"vanna-volga needs K_25P < K_ATM < K_25C, got {k1:.6g}, {k2:.6g}, {k3:.6g}")
        self.m, self.pillars = m, ((k1, s1), (k2, s2), (k3, s3))
        self._std = s2 * math.sqrt(m.tau_e)
        self._c1 = float(gk_price(m, k1, s1, True) - gk_price(m, k1, s2, True))
        self._c3 = float(gk_price(m, k3, s3, True) - gk_price(m, k3, s2, True))

    def _n_d1(self, K):
        d1 = (np.log(self.m.forward / K) + 0.5 * self._std**2) / self._std
        return np.exp(-0.5 * d1 * d1)

    def __call__(self, K) -> np.ndarray:
        K = np.atleast_1d(np.asarray(K, dtype=float))
        if not np.all(np.isfinite(K)) or np.any(K <= 0):
            raise ValueError("strikes must be finite and > 0")
        (k1, s1), (k2, s2), (k3, s3) = self.pillars
        n = self._n_d1(K)
        x1 = n / self._n_d1(k1) * np.log(k2 / K) * np.log(k3 / K) / (math.log(k2 / k1) * math.log(k3 / k1))
        x3 = n / self._n_d1(k3) * np.log(K / k1) * np.log(K / k2) / (math.log(k3 / k1) * math.log(k3 / k2))
        price = np.asarray(gk_price(self.m, K, s2, True), dtype=float) + x1 * self._c1 + x3 * self._c3
        res = implied_black_vol(price, self.m.forward, K, self.m.tau_e, is_call=True, df=self.m.df_d)
        return np.asarray(res.vol, dtype=float).reshape(K.shape)


def vanna_volga_vol_fn(m: FxMarket, pillars: Dict[str, Tuple[float, float]]) -> VannaVolgaSmile:
    return VannaVolgaSmile(m, pillars)


@dataclass(frozen=True)
class SabrSmile:
    """Hagan lognormal SABR smile (beta fixed, :func:`derivx.vol.sabr.sabr_vol`,
    with its input checks); calling it maps strikes to vols."""

    F: float
    T: float
    alpha: float
    beta: float
    rho: float
    nu: float

    def __call__(self, K) -> np.ndarray:
        return np.asarray(sabr_vol(self.F, K, self.T, self.alpha, self.beta, self.rho, self.nu), dtype=float)


@dataclass(frozen=True)
class _Rescaled:
    """``fn(K / scale)``: a smile moved to a spot ``scale`` times larger."""

    fn: Callable
    scale: float

    def __call__(self, K) -> np.ndarray:
        return self.fn(np.asarray(K, dtype=float) / self.scale)


# ---------------------------------------------------------------------------
# The smile of one tenor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmilePoint:
    """One row of the strikes table: the smile's own strike at ``delta`` and
    its vol, with the input pillar (target strike and vol) when there is one."""

    label: str
    delta: float  # signed, in the smile's delta convention (NaN for ATM)
    call: Optional[bool]
    strike: float
    vol: float
    pillar_strike: float = float("nan")
    pillar_vol: float = float("nan")


@dataclass(frozen=True)
class FxSmile:
    """A smile of one tenor in strike space (module docs)."""

    market: FxMarket
    quote: TenorQuote
    conventions: SmileConventions
    model: str
    params: Dict[str, float]
    fixed: Tuple[str, ...]
    atm_strike: float
    pillars: Dict[str, Tuple[float, float]]
    smile_strangles: Dict[int, float]  # smile-strangle fly per delta level, read off the final smile
    market_strangles: Dict[int, MarketStrangle]
    errors: Dict[str, float]  # quote -> repricing error on the final smile, vol decimals
    points: Tuple[SmilePoint, ...]
    fit_rmse: float  # RMS of the errors of the quotes the model was built to match (vol decimals)
    notes: Tuple[str, ...]
    vol_fn: Callable = field(repr=False, compare=False)

    def rescaled(self, S: float) -> "FxSmile":
        """The same smile at spot ``S``. Garman-Kohlhagen is homogeneous of
        degree one in (spot, strike) and both models are scale invariant, so
        strikes, values and vegas scale by ``S / market.S`` while vols, deltas
        and quote errors do not change (smiles are built once at spot 1)."""

        if not (np.isfinite(S) and S > 0):
            raise ValueError(f"spot must be > 0, got {S}")
        lam = float(S) / self.market.S
        if lam == 1.0:
            return self
        strangles = {
            lv: replace(
                ms,
                call_strike=ms.call_strike * lam,
                put_strike=ms.put_strike * lam,
                value=ms.value * lam,
                vega=ms.vega * lam,
            )
            for lv, ms in self.market_strangles.items()
        }
        params = {k: v * lam if k.startswith("K_") else v for k, v in self.params.items()}
        return replace(
            self,
            market=self.market.shifted(S=float(S)),
            params=params,
            atm_strike=self.atm_strike * lam,
            pillars={k: (v[0] * lam, v[1]) for k, v in self.pillars.items()},
            market_strangles=strangles,
            points=tuple(replace(p, strike=p.strike * lam, pillar_strike=p.pillar_strike * lam) for p in self.points),
            vol_fn=_Rescaled(self.vol_fn, lam),
        )

    def vol(self, K):
        """Smile vol(s) at strike(s) ``K`` (NaN where the model has none)."""

        out = _safe_vols(self.vol_fn, K)
        return float(out[0]) if np.ndim(K) == 0 else out

    def strike_at_delta(self, delta: float, call: bool) -> float:
        """The smile's own strike for ``delta`` (signed) in its convention."""

        return smile_delta_strike(self.market, self.vol, delta, call, self.conventions.delta_type, self.quote.atm)

    @property
    def matched(self) -> bool:
        return all(abs(v) <= MATCH_TOL for k, v in self.errors.items() if k in self.used_quotes)

    @property
    def used_quotes(self) -> Tuple[str, ...]:
        levels = self.quote.levels if self.model == "SABR" else (25,)
        return ("ATM",) + tuple(f"{q}{lv}" for lv in levels for q in ("RR", "BF"))


def build_smile(m: FxMarket, quote: TenorQuote, conventions: SmileConventions, model: str = "SABR") -> FxSmile:
    """Smile of one tenor from its quotes (module docs). Raises
    ``ValueError`` at expiry (``tau_e = 0``) or for quotes no smile of the
    model can match (pillar vols <= 0, no strike for a delta)."""

    model = model.upper()
    if model not in SMILE_MODELS:
        raise ValueError(f"smile model must be one of {SMILE_MODELS}, got {model!r}")
    if m.tau_e <= 0:
        raise ValueError("no smile at expiry (tau_e = 0): there is no time value left")
    dtype = conventions.delta_type
    k_atm = atm_strike(m, quote.atm, conventions.atm_type, conventions.premium_adjusted)
    strangles = {lv: market_strangle(m, quote, lv, dtype) for lv in quote.levels}
    notes: List[str] = []
    params: Dict[str, float]
    if model == "VV":
        if quote.has_10d:
            notes.append("Vanna-volga goes through 25P, ATM and 25C only: the 10-delta quotes are checked, not fitted.")

        def pillars_for(bf25: float) -> Dict[str, Tuple[float, float]]:
            vc, vp = quote.atm + bf25 + 0.5 * quote.rr25, quote.atm + bf25 - 0.5 * quote.rr25
            if vc <= 0 or vp <= 0:
                raise ValueError(
                    f"{quote.tenor}: the 25-delta pillar vols are not positive (call {vc:.4%}, put {vp:.4%}): "
                    "RR25 is too large for ATM + BF25"
                )
            return {
                "25P": (strike_from_delta(m, -0.25, vp, False, dtype), vp),
                "ATM": (k_atm, quote.atm),
                "25C": (strike_from_delta(m, 0.25, vc, True, dtype), vc),
            }

        bf25 = quote.bf25
        if conventions.strangle == "market":  # iterate the smile-strangle fly until the market strangle reprices

            def mismatch(x: np.ndarray) -> np.ndarray:
                fn = vanna_volga_vol_fn(m, pillars_for(float(x[0])))
                return np.array([strangles[25].error(m, lambda K: _safe_vols(fn, K))])

            x, converged = _newton(mismatch, np.array([quote.bf25]))
            if not converged:
                notes.append("The market-strangle iteration did not converge: see the BF25 error.")
            bf25 = float(x[0])
        pillars = pillars_for(bf25)
        vol_fn = vanna_volga_vol_fn(m, pillars)
        params = {f"sigma_{lab}": pillars[lab][1] for lab in ("25P", "ATM", "25C")}
        params.update({f"K_{lab}": pillars[lab][0] for lab in ("25P", "ATM", "25C")})
        fixed: Tuple[str, ...] = ()
        used = (25,)
    else:
        start = None
        if quote.has_10d:  # the exact 25-delta fit is the start of the five-quote fit
            q25 = TenorQuote(quote.tenor, quote.atm, quote.rr25, quote.bf25)
            fit25 = _calibrate_sabr(m, q25, conventions, k_atm, {25: strangles[25]})
            start = (fit25.rho, fit25.nu)
        sabr = _calibrate_sabr(m, quote, conventions, k_atm, strangles, start)
        F, T = m.forward, m.tau_e
        vol_fn = SabrSmile(F, T, sabr.alpha, SABR_BETA, sabr.rho, sabr.nu)
        params = {"alpha": sabr.alpha, "beta": SABR_BETA, "rho": sabr.rho, "nu": sabr.nu}
        fixed = ("beta",)
        used = quote.levels
        if sabr.at_bound:
            notes.append(f"SABR {', '.join(sabr.at_bound)} ended at its bound: the quotes push the model to its limit.")
        if sabr.budget_spent and not np.max(np.abs(sabr.residuals)) <= NOTE_TOL:
            notes.append(
                f"The SABR fit stopped at its evaluation budget ({FIT_BUDGET} smiles, {FIT_NFEV} per start) without "
                "converging: no SABR smile is close to these quotes; see the quote errors."
            )

    def safe(K) -> np.ndarray:
        return _safe_vols(vol_fn, K)

    # Re-check every quote on the final smile, at the smile's own delta strikes
    atm_vol = float(safe(k_atm)[0])
    errors = {"ATM": atm_vol - quote.atm}
    points: Dict[str, SmilePoint] = {"ATM": SmilePoint("ATM", float("nan"), None, k_atm, atm_vol, k_atm, quote.atm)}
    bf_ss: Dict[int, float] = {}
    for lv in quote.levels:
        try:
            r, found = _quote_residuals(m, quote, conventions, strangles, safe, (lv,))
            errors[f"RR{lv}"], errors[f"BF{lv}"] = float(r[0]), float(r[1])
            bf_ss[lv] = 0.5 * (found[(lv, True)][1] + found[(lv, False)][1]) - quote.atm
        except ValueError as exc:
            notes.append(f"{lv}-delta quotes cannot be checked: {exc}")
            errors[f"RR{lv}"] = errors[f"BF{lv}"] = float("nan")
            found = {(lv, c): (float("nan"), float("nan")) for c in (True, False)}
        for call in (True, False):
            label = f"{lv}{'C' if call else 'P'}"
            pk, pv = pillars.get(label, (float("nan"), float("nan"))) if model == "VV" else (float("nan"),) * 2
            K, vol = found[(lv, call)]
            points[label] = SmilePoint(label, (lv if call else -lv) / 100.0, call, K, vol, pk, pv)
    if model == "SABR":
        pillars = {lab: (p.strike, p.vol) for lab, p in points.items()}
    used_keys = ("ATM",) + tuple(f"{q}{lv}" for lv in used for q in ("RR", "BF"))
    misfit = np.array([errors[k] for k in used_keys], dtype=float)
    fit_rmse = float(np.sqrt(np.mean(misfit**2))) if np.all(np.isfinite(misfit)) else float("nan")
    unmatched = [k for k in used_keys if not abs(errors[k]) <= NOTE_TOL]
    if unmatched:
        notes.append(
            f"{model} misses {', '.join(unmatched)} by more than {NOTE_TOL * 1e4:g} vol bp (errors in the table): "
            "it has fewer free parameters than quotes, or the quotes are not consistent with it."
        )
    order = [lab for lab in PILLAR_LABELS if lab in points]
    return FxSmile(
        market=m,
        quote=quote,
        conventions=conventions,
        model=model,
        params=params,
        fixed=fixed,
        atm_strike=k_atm,
        pillars=pillars,
        smile_strangles=bf_ss,
        market_strangles=strangles,
        errors=errors,
        points=tuple(points[lab] for lab in order),
        fit_rmse=fit_rmse,
        notes=tuple(notes),
        vol_fn=vol_fn,
    )


def smile_strikes(smile: FxSmile, n_sd: float = 2.5, n: int = 161) -> np.ndarray:
    """Plot grid ``F exp(z sigma_ATM sqrt(tau_e))``, z in [-n_sd, n_sd],
    widened to include every pillar and market-strangle strike."""

    if not n_sd > 0:
        raise ValueError("the smile width must be > 0 standard deviations")
    m = smile.market
    half = min(n_sd * smile.quote.atm * math.sqrt(m.tau_e), 2.5)
    lo, hi = m.forward * math.exp(-half), m.forward * math.exp(half)
    extra = [p.strike for p in smile.points if np.isfinite(p.strike)]
    extra += [s.call_strike for s in smile.market_strangles.values()]
    extra += [s.put_strike for s in smile.market_strangles.values()]
    lo, hi = min([lo] + extra), max([hi] + extra)
    return np.unique(np.r_[np.geomspace(lo, hi, n), extra])


def density_check(smile: FxSmile, strikes: np.ndarray, n: int = 401) -> Tuple[bool, float, float]:
    """Butterfly-arbitrage check over the range of ``strikes``: undiscounted
    call prices on ``n`` log-spaced strikes must be convex in the strike
    (the risk-neutral density, their second difference, is non-negative).
    Returns ``(ok, minimum density, strike at the minimum)``; strikes where
    the smile has no vol are skipped. A tolerance of 1e-7 of the largest
    density absorbs rounding."""

    K = np.asarray(strikes, dtype=float)
    K = np.geomspace(float(np.min(K)), float(np.max(K)), n)
    vols = smile.vol(K)
    keep = np.isfinite(vols) & (vols > 0)
    K, vols = K[keep], vols[keep]
    if K.size < 3:
        return True, float("nan"), float("nan")
    calls = np.asarray(gk_price(smile.market, K, vols, True), dtype=float) / smile.market.df_d
    h1, h2 = np.diff(K)[:-1], np.diff(K)[1:]
    slope = np.diff(calls) / np.diff(K)
    density = 2.0 * np.diff(slope) / (h1 + h2)
    tol = -1e-7 * float(np.max(np.abs(density)))
    i = int(np.argmin(density))
    return bool(density[i] >= tol), float(density[i]), float(K[i + 1])


def smile_by_delta(smile: FxSmile, deltas=SURFACE_DELTAS) -> List[Tuple[str, float, float]]:
    """``(label, strike, vol)`` at each surface delta (the smile's own delta
    strikes, solved together; NaN where no strike has that delta, e.g. a
    premium-adjusted call delta above its maximum)."""

    wings = [(i, d, c) for i, (_, d, c) in enumerate(deltas) if c is not None]
    strikes = np.full(len(deltas), np.nan)
    vols = np.full(len(deltas), np.nan)
    dtype = smile.conventions.delta_type

    def solve(items) -> None:
        K, v = smile_delta_strikes(
            smile.market, smile.vol, [d for _, d, _ in items], [c for _, _, c in items], dtype, smile.quote.atm
        )
        for (i, _, _), k, vol in zip(items, K, v):
            strikes[i], vols[i] = k, vol

    try:
        solve(wings)
    except ValueError:  # solve one by one, so one unreachable delta does not blank the rest
        for item in wings:
            try:
                solve([item])
            except ValueError:
                pass
    for i, (_, _, call) in enumerate(deltas):
        if call is None:
            strikes[i], vols[i] = smile.atm_strike, smile.vol(smile.atm_strike)
    return [(label, float(k), float(v)) for (label, _, _), k, v in zip(deltas, strikes, vols)]


def interpolate_quote(points: Sequence[Tuple[float, TenorQuote]], tau: float, label: str) -> Tuple[TenorQuote, str]:
    """Quotes at ``tau`` years (vol time) from tenor quotes ``(tau_i,
    quote_i)``: ATM total variance ``sigma^2 tau`` linear in tau, RR and BF
    linear in tau; flat beyond the first and last tenors. Returns the quote
    and a note on how it was obtained."""

    pts = sorted(points, key=lambda p: p[0])
    if not pts:
        raise ValueError("no tenor quotes to interpolate")
    taus = np.array([p[0] for p in pts])
    if tau <= taus[0] or tau >= taus[-1]:
        edge = pts[0] if tau <= taus[0] else pts[-1]
        q = edge[1]
        if abs(tau - edge[0]) <= 1e-12:
            return TenorQuote(label, q.atm, q.rr25, q.bf25, q.rr10, q.bf10), f"the {q.tenor} quotes"
        side = "before the first" if tau < taus[0] else "after the last"
        note = f"flat extrapolation of the {q.tenor} quotes ({side} quoted tenor)"
        return TenorQuote(label, q.atm, q.rr25, q.bf25, q.rr10, q.bf10), note
    j = int(np.searchsorted(taus, tau))
    (t1, q1), (t2, q2) = pts[j - 1], pts[j]
    u = (tau - t1) / (t2 - t1)
    w1, w2 = q1.atm**2 * t1, q2.atm**2 * t2
    note = f"interpolated between {q1.tenor} and {q2.tenor} (ATM total variance and RR / BF linear in time)"
    if w2 < w1:
        note += f"; the ATM total variance falls from {q1.tenor} to {q2.tenor} (calendar arbitrage in the quotes)"
    atm = math.sqrt(max(w1 + u * (w2 - w1), 1e-16) / tau)

    def lin(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if a is None or b is None else a + u * (b - a)

    rr10, bf10 = lin(q1.rr10, q2.rr10), lin(q1.bf10, q2.bf10)
    if rr10 is None or bf10 is None:
        rr10 = bf10 = None
    return TenorQuote(label, atm, lin(q1.rr25, q2.rr25), lin(q1.bf25, q2.bf25), rr10, bf10), note
