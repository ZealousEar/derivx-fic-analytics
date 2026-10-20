"""Option prices and implied volatilities: Black-Scholes, Black-76 (optionally
shifted) and Bachelier, vectorised.

Conventions
-----------
* ``F`` is the forward to expiry, ``K`` the strike, both in price units (an
  index level, an FX rate, a decimal rate such as 0.03).
* ``T`` is the time to expiry in years (ACT/365F when it comes from dates).
* ``df`` is the discount factor to the payment date. Prices are
  ``df * undiscounted price`` per unit of underlying notional.
* Lognormal (Black) vols are per-annum decimals: 0.20 is 20%. A ``shift``
  ``s`` prices the lognormal model on ``F + s`` and ``K + s`` (shifted Black).
* Normal (Bachelier) vols are in price units per sqrt(year). For a rate,
  0.0085 is 85 bp normal (``* 1e4`` gives bp).
* Black-Scholes on spot uses ``F = S exp((r - q) T)`` and ``df = exp(-r T)``,
  with continuously compounded ``r`` and dividend yield / foreign rate ``q``.

Implied vols are solved on the out-of-the-money side: an in-the-money price
is first turned into the OTM price at the same strike with put-call parity
(``C - P = df (F - K)``), so deep in-the-money quotes whose time value is
below float precision are reported as ``no_time_value`` instead of being
given a meaningless vol. Every inversion returns a status per quote (see
:data:`STATUS`): failures are ``NaN`` with a reason, never a clamped number.
The solver is a safeguarded Newton iteration on the log of the OTM price,
bracketed by bisection, so it converges for deep OTM prices (1e-30) and for
vols far above 500%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.special import ndtr

_EPS = np.finfo(float).eps
_SQRT_2PI = np.sqrt(2.0 * np.pi)
_MAX_STD = 64.0  # sigma * sqrt(T) cap for the lognormal bracket: the price is then at its bound

STATUS: Dict[str, str] = {
    "ok": "solved",
    "invalid_input": "non-finite input, F + shift <= 0, K + shift <= 0, T < 0 or df <= 0",
    "expired": "T == 0: an expired option has no time value to imply a vol from",
    "below_intrinsic": "price below the zero-vol (intrinsic) value: no vol reproduces it",
    "no_time_value": "time value is zero to float precision: the vol is not identifiable",
    "above_upper_bound": "price at or above the infinite-vol value (F df for a call, K df for a put)",
    "no_convergence": "the solver did not converge",
}


def _npdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _as_float_arrays(*xs) -> Tuple[np.ndarray, ...]:
    return tuple(np.asarray(x, dtype=float) for x in xs)


def _out(x: np.ndarray):
    """Return a Python float for 0-d results, else the array."""

    return float(x) if np.ndim(x) == 0 else x


def _call_flags(is_call, shape) -> np.ndarray:
    flags = np.asarray(is_call)
    if flags.dtype.kind in "USO":
        lowered = np.char.lower(np.char.strip(flags.astype(str)))
        is_c = np.char.startswith(lowered, "c")
        if np.any(~(is_c | np.char.startswith(lowered, "p"))):
            raise ValueError("option types must be 'call'/'put' (or 'C'/'P'), or booleans with True for a call")
        flags = is_c
    return np.broadcast_to(flags.astype(bool), shape)


# ---------------------------------------------------------------------------
# Prices and vegas
# ---------------------------------------------------------------------------


def _black_otm(f: np.ndarray, k: np.ndarray, std: np.ndarray, call: np.ndarray) -> np.ndarray:
    """Undiscounted Black price of a call (``call``) or put with ``std = vol
    sqrt(T) > 0`` (no input checks)."""

    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(f / k)
        d1 = x / std + 0.5 * std
        d2 = d1 - std
        w = np.where(call, 1.0, -1.0)
        return w * (f * ndtr(w * d1) - k * ndtr(w * d2))


def black_price(F, K, T, vol, is_call=True, df=1.0, shift=0.0):
    """Black-76 price ``df * E[(w (F_T - K))^+]`` of a lognormal forward
    (shifted by ``shift``). Vectorised; ``is_call`` may be an array of bools
    or of 'call'/'put' strings. Raises ``ValueError`` on F + shift <= 0,
    K + shift <= 0, T < 0, vol < 0 or non-finite input."""

    F, K, T, vol, df, shift = _as_float_arrays(F, K, T, vol, df, shift)
    f, k = F + shift, K + shift
    shape = np.broadcast_shapes(f.shape, k.shape, T.shape, vol.shape, df.shape)
    call = _call_flags(is_call, shape)
    for name, arr in (("F", F), ("K", K), ("T", T), ("vol", vol), ("df", df), ("shift", shift)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
    if np.any(f <= 0) or np.any(k <= 0):
        raise ValueError("Black needs F + shift > 0 and K + shift > 0: use a shift or the Bachelier model")
    if np.any(T < 0) or np.any(vol < 0) or np.any(df <= 0):
        raise ValueError("Black needs T >= 0, vol >= 0 and df > 0")
    std = np.broadcast_to(vol * np.sqrt(T), shape)
    f, k = np.broadcast_to(f, shape), np.broadcast_to(k, shape)
    intrinsic = np.where(call, np.maximum(f - k, 0.0), np.maximum(k - f, 0.0))
    safe_std = np.where(std > 0, std, 1.0)
    price = np.where(std > 0, _black_otm(f, k, safe_std, call), intrinsic)
    return _out(df * price)


def black_vega(F, K, T, vol, df=1.0, shift=0.0):
    """d(Black price) / d(vol), per unit of vol (multiply by 0.01 for one
    vol point). Same for calls and puts."""

    F, K, T, vol, df, shift = _as_float_arrays(F, K, T, vol, df, shift)
    f, k = F + shift, K + shift
    if np.any(f <= 0) or np.any(k <= 0) or np.any(T < 0) or np.any(vol < 0):
        raise ValueError("Black vega needs F + shift > 0, K + shift > 0, T >= 0 and vol >= 0")
    std = vol * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(f / k) / std + 0.5 * std
        vega = df * f * _npdf(d1) * np.sqrt(T)
    return _out(np.where(std > 0, vega, 0.0))


def _bachelier_h(d: np.ndarray) -> np.ndarray:
    """h(d) = d N(d) + n(d), the normalised Bachelier call price, accurate
    for very negative d (asymptotic series below d = -30)."""

    d = np.asarray(d, dtype=float)
    direct = d * ndtr(d) + _npdf(d)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        i2 = 1.0 / (d * d)
        tail = 1.0 - 9.0 * i2 * (1.0 - 11.0 * i2)
        series = _npdf(d) * i2 * (1.0 - 3.0 * i2 * (1.0 - 5.0 * i2 * (1.0 - 7.0 * i2 * tail)))
    # Direct form loses only log10(d^2) digits; switch before N(d) underflows.
    return np.where(d < -30.0, series, direct)


def bachelier_price(F, K, T, vol_n, is_call=True, df=1.0):
    """Bachelier (normal) price ``df * E[(w (F_T - K))^+]`` with
    ``F_T = F + vol_n W_T``. Works for any sign of F and K. Raises
    ``ValueError`` on T < 0, vol_n < 0, df <= 0 or non-finite input."""

    F, K, T, vol_n, df = _as_float_arrays(F, K, T, vol_n, df)
    shape = np.broadcast_shapes(F.shape, K.shape, T.shape, vol_n.shape, df.shape)
    call = _call_flags(is_call, shape)
    for name, arr in (("F", F), ("K", K), ("T", T), ("vol_n", vol_n), ("df", df)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
    if np.any(T < 0) or np.any(vol_n < 0) or np.any(df <= 0):
        raise ValueError("Bachelier needs T >= 0, vol_n >= 0 and df > 0")
    std = np.broadcast_to(vol_n * np.sqrt(T), shape)
    w = np.where(call, 1.0, -1.0)
    m = w * (F - K)
    safe = np.where(std > 0, std, 1.0)
    price = np.where(std > 0, safe * _bachelier_h(m / safe), np.maximum(m, 0.0))
    return _out(df * price)


def bachelier_vega(F, K, T, vol_n, df=1.0):
    """d(Bachelier price) / d(vol_n), per unit of normal vol (multiply by
    1e-4 for one bp of normal vol)."""

    F, K, T, vol_n, df = _as_float_arrays(F, K, T, vol_n, df)
    if np.any(T < 0) or np.any(vol_n < 0):
        raise ValueError("Bachelier vega needs T >= 0 and vol_n >= 0")
    std = vol_n * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        vega = df * np.sqrt(T) * _npdf((F - K) / std)
    return _out(np.where(std > 0, vega, 0.0))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpliedVolResult:
    """Implied vols with a status code per quote (see :data:`STATUS`).

    ``vol`` is NaN wherever ``status != 'ok'``."""

    vol: np.ndarray
    status: np.ndarray

    @property
    def ok(self) -> np.ndarray:
        return self.status == "ok"

    def reason_counts(self) -> Dict[str, int]:
        """Number of quotes per status."""

        values, counts = np.unique(np.asarray(self.status).ravel(), return_counts=True)
        return {str(v): int(c) for v, c in zip(values, counts)}


def _solve_increasing(
    target: np.ndarray,
    price: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dprice: Callable[[np.ndarray, np.ndarray], np.ndarray],
    s0: np.ndarray,
    hi: np.ndarray,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve ``price(s) = target`` for s in (0, hi] where price is increasing
    with price(0) = 0. ``price(s, idx)`` evaluates the subset ``idx``.
    Safeguarded Newton on ``log price``, falling back to bisection."""

    n = target.size
    lo = np.zeros(n)
    hi = hi.copy()
    s = np.clip(s0, 0.0, hi)
    s = np.where((s <= 0) | ~np.isfinite(s), 0.5 * hi, s)
    converged = np.zeros(n, dtype=bool)
    log_target = np.log(target)
    active = np.arange(n)
    for _ in range(max_iter):
        if active.size == 0:
            break
        sa = s[active]
        p = price(sa, active)
        v = dprice(sa, active)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # a price that rounds to <= 0 is below any positive target
            f = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)) - log_target[active], -np.inf)
            hit = np.abs(f) <= 4 * _EPS
            lo_a = np.where(f < 0, sa, lo[active])
            hi_a = np.where(f > 0, sa, hi[active])
            newton = sa - f * p / v
        bad = ~np.isfinite(newton) | (newton <= lo_a) | (newton >= hi_a)
        mid = np.where(lo_a > 0, np.sqrt(lo_a * hi_a), 0.5 * hi_a)
        s_new = np.where(hit, sa, np.where(bad, mid, newton))
        done = hit | (np.abs(s_new - sa) <= 4 * _EPS * s_new) | (hi_a - lo_a <= 4 * _EPS * hi_a)
        lo[active], hi[active], s[active] = lo_a, hi_a, s_new
        converged[active[done]] = True
        active = active[~done]
    return s, converged


def _prepare(price, F, K, T, df, shift, is_call):
    price, F, K, T, df, shift = _as_float_arrays(price, F, K, T, df, shift)
    shape = np.broadcast_shapes(price.shape, F.shape, K.shape, T.shape, df.shape, shift.shape)
    call = np.array(_call_flags(is_call, shape))
    arrays = [np.array(np.broadcast_to(a, shape), dtype=float).ravel() for a in (price, F, K, T, df, shift)]
    return shape, call.ravel(), arrays


def _otm_target(undisc: np.ndarray, intrinsic_itm: np.ndarray, itm: np.ndarray, scale: np.ndarray):
    """OTM-equivalent price and the status of each quote before solving."""

    otm = np.where(itm, undisc - intrinsic_itm, undisc)
    tol = 64 * _EPS * np.maximum(np.abs(undisc), scale)
    status = np.full(otm.shape, "ok", dtype=object)
    status[otm < -tol] = "below_intrinsic"
    status[(otm >= -tol) & (otm <= np.where(itm, tol, 0.0))] = "no_time_value"
    return otm, status


def implied_black_vol(price, F, K, T, is_call=True, df=1.0, shift=0.0) -> ImpliedVolResult:
    """Black-76 (shifted when ``shift != 0``) implied vols of discounted
    prices. Vectorised over every argument; ``is_call`` may mix calls and
    puts. Quotes outside the no-arbitrage bounds get NaN and a status."""

    shape, call, (p, F, K, T, df, s) = _prepare(price, F, K, T, df, shift, is_call)
    f, k = F + s, K + s
    status = np.full(p.shape, "ok", dtype=object)
    finite = np.isfinite(p) & np.isfinite(f) & np.isfinite(k) & np.isfinite(T) & np.isfinite(df)
    valid = finite & (f > 0) & (k > 0) & (T >= 0) & (df > 0)
    status[~valid] = "invalid_input"
    status[valid & (T == 0)] = "expired"
    live = status == "ok"
    vol = np.full(p.shape, np.nan)
    if np.any(live):
        idx = np.flatnonzero(live)
        fl, kl, Tl = f[idx], k[idx], T[idx]
        undisc = p[idx] / df[idx]
        cl = call[idx]
        itm = np.where(cl, kl < fl, kl > fl)
        otm_call = np.where(itm, ~cl, cl)  # the OTM option we invert
        intrinsic = np.abs(fl - kl)
        target, st = _otm_target(undisc, intrinsic, itm, np.maximum(fl, kl))
        upper = np.where(otm_call, fl, kl)
        st[(st == "ok") & (target >= upper * (1.0 - 4 * _EPS))] = "above_upper_bound"
        solve = st == "ok"
        if np.any(solve):
            j = np.flatnonzero(solve)
            fj, kj, cj, tj = fl[j], kl[j], otm_call[j], target[j]

            def price_fn(sv, a):
                return _black_otm(fj[a], kj[a], sv, cj[a])

            def vega_fn(sv, a):
                d1 = np.log(fj[a] / kj[a]) / sv + 0.5 * sv
                return fj[a] * _npdf(d1)

            x = np.abs(np.log(fj / kj))
            s0 = np.where(x > 1e-8, np.sqrt(2.0 * x), _SQRT_2PI * tj / fj)
            hi = np.maximum(1.0, 2.0 * s0)
            for _ in range(8):  # grow the bracket until it holds the root
                short = price_fn(hi, np.arange(j.size)) < tj
                if not np.any(short):
                    break
                hi = np.where(short, np.minimum(2.0 * hi, _MAX_STD), hi)
            beyond = price_fn(hi, np.arange(j.size)) < tj
            stdv, conv = _solve_increasing(tj, price_fn, vega_fn, s0, hi)
            vj = stdv / np.sqrt(Tl[j])
            sj = np.where(conv, "ok", "no_convergence").astype(object)
            sj[beyond] = "above_upper_bound"
            vj = np.where(sj == "ok", vj, np.nan)
            vol_l = np.full(idx.size, np.nan)
            vol_l[j] = vj
            st[j] = sj
        else:
            vol_l = np.full(idx.size, np.nan)
        vol[idx] = vol_l
        status[idx] = st
    vol = np.where(status == "ok", vol, np.nan)
    return ImpliedVolResult(vol=vol.reshape(shape), status=status.reshape(shape))


def implied_normal_vol(price, F, K, T, is_call=True, df=1.0) -> ImpliedVolResult:
    """Bachelier implied normal vols (price units per sqrt(year)) of
    discounted prices. Works for any sign of F and K."""

    shape, call, (p, F, K, T, df, _) = _prepare(price, F, K, T, df, 0.0, is_call)
    status = np.full(p.shape, "ok", dtype=object)
    finite = np.isfinite(p) & np.isfinite(F) & np.isfinite(K) & np.isfinite(T) & np.isfinite(df)
    valid = finite & (T >= 0) & (df > 0)
    status[~valid] = "invalid_input"
    status[valid & (T == 0)] = "expired"
    vol = np.full(p.shape, np.nan)
    idx = np.flatnonzero(status == "ok")
    if idx.size:
        Fl, Kl, Tl, cl = F[idx], K[idx], T[idx], call[idx]
        undisc = p[idx] / df[idx]
        itm = np.where(cl, Kl < Fl, Kl > Fl)
        intrinsic = np.abs(Fl - Kl)
        target, st = _otm_target(undisc, intrinsic, itm, np.maximum(np.abs(Fl), np.abs(Kl)))
        j = np.flatnonzero(st == "ok")
        if j.size:
            m = -np.abs(Fl[j] - Kl[j])  # OTM: w (F - K) <= 0
            tj = target[j]

            def price_fn(sv, a):
                return sv * _bachelier_h(m[a] / sv)

            def vega_fn(sv, a):
                return _npdf(m[a] / sv)

            s0 = np.where(-m > 0, np.maximum(_SQRT_2PI * tj, -m / 3.0), _SQRT_2PI * tj)
            hi = np.maximum(2.0 * s0, _SQRT_2PI * tj + -m)
            for _ in range(60):
                short = price_fn(hi, np.arange(j.size)) < tj
                if not np.any(short):
                    break
                hi = np.where(short, 2.0 * hi, hi)
            stdv, conv = _solve_increasing(tj, price_fn, vega_fn, s0, hi)
            vj = np.where(conv, stdv / np.sqrt(Tl[j]), np.nan)
            st[j] = np.where(conv, "ok", "no_convergence").astype(object)
            vol_l = np.full(idx.size, np.nan)
            vol_l[j] = vj
        else:
            vol_l = np.full(idx.size, np.nan)
        vol[idx] = vol_l
        status[idx] = st
    vol = np.where(status == "ok", vol, np.nan)
    return ImpliedVolResult(vol=vol.reshape(shape), status=status.reshape(shape))


def implied_vol_bs(price, S, K, T, r, q=0.0, is_call=True) -> ImpliedVolResult:
    """Black-Scholes implied vols on spot ``S`` with continuously compounded
    rate ``r`` and dividend yield / foreign rate ``q`` (see module docs)."""

    S, K, T, r, q = _as_float_arrays(S, K, T, r, q)
    with np.errstate(invalid="ignore", over="ignore"):
        F = S * np.exp((r - q) * T)
        df = np.exp(-r * T)
    F = np.where(S > 0, F, np.nan)
    return implied_black_vol(price, F, K, T, is_call=is_call, df=df)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    call: bool = True,
    q: float = 0.0,
) -> float:
    """Black-Scholes implied vol of one quote; NaN when no vol reproduces
    it (use :func:`implied_vol_bs` for the reason)."""

    return float(implied_vol_bs(price, S, K, T, r, q=q, is_call=call).vol)


def batch_implied_vols(prices, S, Ks, T, r, call=True, q: float = 0.0) -> np.ndarray:
    """Black-Scholes implied vols for a chain; ``call`` may be one flag or
    an array (mixed calls and puts). Invalid quotes are NaN; use
    :func:`implied_vol_bs` for the reasons."""

    return np.asarray(implied_vol_bs(prices, S, Ks, T, r, q=q, is_call=call).vol, dtype=float)


# ---------------------------------------------------------------------------
# Lognormal <-> normal vol conversion
# ---------------------------------------------------------------------------


def lognormal_to_normal_vol(F, K, T, vol, shift=0.0):
    """Normal vol with the same price as Black vol ``vol`` (shifted by
    ``shift``) at (F, K, T): exact price matching. Bachelier is shift
    invariant, so the result applies to the unshifted forward."""

    F, K, T, vol, shift = _as_float_arrays(F, K, T, vol, shift)
    call = (K + shift) >= (F + shift)  # price the OTM side
    price = black_price(F, K, T, vol, is_call=call, shift=shift)
    out = implied_normal_vol(price, F, K, T, is_call=call)
    return _out(out.vol)


def normal_to_lognormal_vol(F, K, T, vol_n, shift=0.0):
    """Black vol (on F + shift, K + shift) with the same price as normal vol
    ``vol_n``: exact price matching. NaN where the Bachelier price is outside
    the shifted-Black bounds (e.g. a strike below -shift)."""

    F, K, T, vol_n, shift = _as_float_arrays(F, K, T, vol_n, shift)
    call = K >= F
    price = bachelier_price(F, K, T, vol_n, is_call=call)
    out = implied_black_vol(price, F, K, T, is_call=call, shift=shift)
    return _out(out.vol)


def hagan_normal_vol_approx(F, K, T, vol, shift=0.0):
    """Hagan's closed-form normal-vol equivalent of a Black vol (Hagan et al.
    2002, 'equivalent Black volatilities'), for display:

    ``vol_n ~ vol sqrt(fk) (1 + l^2/24 + l^4/1920) / (1 + vol^2 T/24 + vol^4 T^2/5760)``

    with ``f = F + shift``, ``k = K + shift`` and ``l = ln(f/k)``. Smooth at
    the money (no 0/0); accurate to a fraction of a bp for usual inputs. Use
    :func:`lognormal_to_normal_vol` where exactness matters."""

    F, K, T, vol, shift = _as_float_arrays(F, K, T, vol, shift)
    f, k = F + shift, K + shift
    if np.any(f <= 0) or np.any(k <= 0):
        raise ValueError("lognormal vols need F + shift > 0 and K + shift > 0")
    lg = np.log(f / k)
    num = np.sqrt(f * k) * (1.0 + lg**2 / 24.0 + lg**4 / 1920.0)
    den = 1.0 + vol**2 * T / 24.0 + vol**4 * T**2 / 5760.0
    return _out(vol * num / den)


def vega_weights(F, K, T, vols, shift=0.0) -> np.ndarray:
    """Black vegas ``(F+s) n(d1) sqrt(T)`` of a smile, scaled to mean 1:
    weights for a vol-space fit that approximate a price-space fit (they
    down-weight far wings whose vols are noisy)."""

    v = np.asarray(black_vega(F, K, T, vols, shift=shift), dtype=float)
    if not np.all(np.isfinite(v)) or np.all(v <= 0):
        raise ValueError("vega weights need finite vols and T > 0")
    v = np.maximum(v, 1e-6 * v.max())
    return v / v.mean()
