"""Smile shape and smile-consistent risk.

* :func:`svi_atm_skew_curvature`: ATM vol, skew ``d sigma / dk`` and
  curvature ``d2 sigma / dk2`` at k = ln(K/F) = 0 of an SVI slice, and
  :func:`smile_term_structure` for a set of calibrated slices.
* :func:`smile_deltas`: Black-76 forward delta under two smile dynamics:
  sticky strike (the vol at a fixed strike does not move when F moves) and
  sticky moneyness (the vol at fixed k = ln(K/F) does not move; to first
  order this is also 'sticky delta'):
  ``delta_sticky_moneyness = delta_sticky_strike - vega * (d sigma/dk) / F``.
* :func:`sabr_deltas`: SABR deltas. Hagan's ``delta = delta_BS + vega
  d sigma/dF`` (alpha held fixed) and Bartlett's (2006) correction, which
  moves alpha with F through the correlation: ``+ vega (d sigma/d alpha)
  rho nu / (F + s)^beta``.

* :func:`svi_parameter_vegas` / :func:`sabr_parameter_vegas`: price
  sensitivities to each smile-model parameter (SVI a, b, rho, m, sigma;
  SABR alpha, rho, nu with beta fixed) at a strike, central differences of
  the Black-76 price at the model vol.
* :func:`svi_bucket_vegas`: vega by quoted strike and by moneyness bucket
  through the calibration: the P&L of an option priced on the fitted SVI
  smile when the market vol quotes of one bucket rise by one vol point and
  the smile is refitted. To first order the refit moves the parameters by
  ``d theta / d q = (J' W J)^-1 J' W`` (the Gauss-Newton normal equations of
  the weighted least-squares fit at its solution; ``J = d sigma_model /
  d theta`` at the quoted strikes, ``W`` the fit weights times the robust
  loss's IRLS weights), so bucket vegas add up across buckets; bounds and
  the arbitrage repair are treated as inactive (first order around the
  fitted parameters).

Units: ``vega`` per 1.00 of vol, deltas per unit of forward notional
(discounted by ``df``); skew per unit of log-moneyness (a skew of -0.1 is
-1 vol point per 10% of log-moneyness), curvature per unit k^2. Parameter
vegas are premium per unit of underlying per +1.00 of the parameter (the
``per_0.01`` columns: per +0.01). Bucket vegas are premium per unit of
underlying per +1 vol point (0.01) on every quote in the bucket. Signed as
P&L of a long option.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from derivx.vol.implied import black_price, black_vega
from derivx.vol.sabr import hagan_sabr_iv
from derivx.vol.svi import SVI_PARAMS, svi_derivatives, svi_iv

SABR_RISK_PARAMS = ("alpha", "rho", "nu")
# Standardised moneyness z = ln(K/F) / (atm_vol sqrt(T)) bucket edges for bucket vegas
Z_EDGES: Tuple[float, ...] = (-np.inf, -2.0, -1.0, -0.25, 0.25, 1.0, 2.0, np.inf)
Z_LABELS: Tuple[str, ...] = ("z < -2", "-2 to -1", "-1 to -0.25", "ATM (|z| <= 0.25)", "0.25 to 1", "1 to 2", "z > 2")


def svi_atm_skew_curvature(params: Mapping, T: float) -> Dict[str, float]:
    """ATM vol, skew and curvature in k of an SVI total-variance slice."""

    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be > 0 years")
    w, w1, w2 = (float(x) for x in svi_derivatives(0.0, params))
    if w <= 0:
        raise ValueError("ATM total variance is zero: skew is undefined")
    vol = np.sqrt(w / T)
    skew = w1 / (2.0 * np.sqrt(w * T))
    curvature = w2 / (2.0 * np.sqrt(w * T)) - w1 * w1 / (4.0 * w * np.sqrt(w * T))
    return {"atm_vol": float(vol), "skew": float(skew), "curvature": float(curvature)}


def smile_term_structure(slices: Iterable) -> pd.DataFrame:
    """ATM vol, skew and curvature by expiry for calibrated slices (objects
    with ``expiry``, ``tau`` and SVI ``params``, e.g. ``CalibratedSlice``).
    ``skew_x_sqrtT`` (= skew sqrt(T)) is the scale-free skew: roughly
    constant across expiries for a stable smile."""

    rows = []
    for s in slices:
        d = svi_atm_skew_curvature(s.params, s.tau)
        rows.append({"expiry": s.expiry, "tau": s.tau, **d, "skew_x_sqrtT": d["skew"] * np.sqrt(s.tau)})
    return pd.DataFrame(rows).sort_values("tau").reset_index(drop=True) if rows else pd.DataFrame(rows)


def _black(F, K, T, vol, df, is_call):
    F, K, T, vol = (np.asarray(x, dtype=float) for x in (F, K, T, vol))
    sd = vol * np.sqrt(T)
    d1 = np.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    call_delta = df * norm.cdf(d1)
    delta = call_delta if is_call else call_delta - df
    vega = df * F * norm.pdf(d1) * np.sqrt(T)
    price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if not is_call:
        price = price - df * (F - K)
    return price, delta, vega


def smile_deltas(F, K, T, vol, dvol_dk, df: float = 1.0, is_call: bool = True) -> Dict[str, np.ndarray]:
    """Sticky-strike and sticky-moneyness forward deltas (see module docs).
    ``dvol_dk`` is the smile slope d sigma / d ln(K/F) at K."""

    F, K, T, vol, dvol_dk = (np.asarray(x, dtype=float) for x in (F, K, T, vol, dvol_dk))
    if np.any(F <= 0) or np.any(K <= 0) or np.any(T <= 0) or np.any(vol <= 0):
        raise ValueError("deltas need F > 0, K > 0, T > 0 and vol > 0")
    _, delta, vega = _black(F, K, T, vol, df, is_call)
    return {"sticky_strike": delta, "sticky_moneyness": delta - vega * dvol_dk / F, "vega": vega}


def sabr_deltas(
    F: float,
    K,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
    df: float = 1.0,
    is_call: bool = True,
) -> Dict[str, np.ndarray]:
    """Black (shifted) delta at the SABR vol, Hagan's SABR delta and
    Bartlett's delta, by central differences of the Hagan vol. Lognormal
    SABR (Black vol of F + shift)."""

    K = np.asarray(K, dtype=float)
    f = F + shift
    vol = np.asarray(hagan_sabr_iv(F, K, T, alpha, beta, rho, nu, shift), dtype=float)
    _, delta, vega = _black(f, K + shift, T, vol, df, is_call)
    hF = 1e-5 * f
    ha = 1e-5 * alpha
    dvol_dF = (
        np.asarray(hagan_sabr_iv(F + hF, K, T, alpha, beta, rho, nu, shift))
        - np.asarray(hagan_sabr_iv(F - hF, K, T, alpha, beta, rho, nu, shift))
    ) / (2 * hF)
    dvol_da = (
        np.asarray(hagan_sabr_iv(F, K, T, alpha + ha, beta, rho, nu, shift))
        - np.asarray(hagan_sabr_iv(F, K, T, alpha - ha, beta, rho, nu, shift))
    ) / (2 * ha)
    hagan = delta + vega * dvol_dF
    bartlett = hagan + vega * dvol_da * rho * nu / f**beta
    return {"black": delta, "hagan": hagan, "bartlett": bartlett, "vol": vol, "vega": vega}


# ---------------------------------------------------------------------------
# Vega by model parameter
# ---------------------------------------------------------------------------


def _check_option(F: float, K: float, T: float, df: float) -> None:
    if not (np.isfinite(F) and F > 0 and np.isfinite(K) and K > 0):
        raise ValueError(f"smile risk needs F > 0 and K > 0, got F={F}, K={K}")
    if not (np.isfinite(T) and T > 0):
        raise ValueError("smile risk needs T > 0 years: at expiry there is no vega")
    if not (np.isfinite(df) and df > 0):
        raise ValueError(f"the discount factor must be > 0, got {df}")


def svi_parameter_vegas(
    K: float, F: float, T: float, params: Mapping, df: float = 1.0, is_call: bool = True
) -> pd.DataFrame:
    """dV/dp for each raw-SVI parameter (total-variance parameters for T)
    of a Black-76 option at strike K priced at the SVI vol (central
    differences, step 1e-6 x max(1, |p|)). Columns: parameter, value,
    dV_dp (premium per +1.00 of p), per_0.01 (per +0.01)."""

    _check_option(F, K, T, df)
    base = {n: float(params[n]) for n in SVI_PARAMS}

    def price(p: Mapping) -> float:
        return float(black_price(F, K, T, svi_iv(K, p, F=F, T=T), is_call, df))

    rows = []
    for name in SVI_PARAMS:
        h = 1e-6 * max(1.0, abs(base[name]))
        up, dn = dict(base), dict(base)
        up[name] += h
        dn[name] -= h
        d = (price(up) - price(dn)) / (2.0 * h)
        rows.append({"parameter": name, "value": base[name], "dV_dp": d, "per_0.01": 0.01 * d})
    return pd.DataFrame(rows)


def sabr_parameter_vegas(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
    df: float = 1.0,
    is_call: bool = True,
) -> pd.DataFrame:
    """dV/dp for SABR alpha, rho and nu (beta and shift fixed) of a
    Black-76 option on ``F + shift`` at the Hagan lognormal vol: alpha-vega,
    rho-vanna and nu-volga in model terms (central differences). Columns as
    :func:`svi_parameter_vegas`."""

    _check_option(F + shift, K + shift, T, df)
    base = {"alpha": float(alpha), "rho": float(rho), "nu": float(nu)}

    def price(p: Mapping) -> float:
        vol = hagan_sabr_iv(F, K, T, p["alpha"], beta, p["rho"], p["nu"], shift)
        return float(black_price(F, K, T, vol, is_call, df, shift))

    rows = []
    for name in SABR_RISK_PARAMS:
        h = 1e-6 * max(1.0, abs(base[name])) if name != "alpha" else 1e-6 * base[name]
        up, dn = dict(base), dict(base)
        up[name] += h
        dn[name] -= h
        if name == "rho":  # stay inside (-1, 1)
            up[name], dn[name] = min(up[name], 0.99999), max(dn[name], -0.99999)
        if name == "nu":
            dn[name] = max(dn[name], 0.0)
        d = (price(up) - price(dn)) / (up[name] - dn[name])
        rows.append({"parameter": name, "value": base[name], "dV_dp": d, "per_0.01": 0.01 * d})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Vega by quote bucket, through the SVI calibration
# ---------------------------------------------------------------------------

_IRLS = {
    "linear": lambda z: np.ones_like(z),
    "soft_l1": lambda z: 1.0 / np.sqrt(1.0 + z),
    "huber": lambda z: np.where(z <= 1.0, 1.0, 1.0 / np.sqrt(np.maximum(z, 1e-300))),
    "cauchy": lambda z: 1.0 / (1.0 + z),
    "arctan": lambda z: 1.0 / (1.0 + z * z),
}


def _svi_vol_jacobian(K: np.ndarray, F: float, T: float, params: Mapping) -> np.ndarray:
    base = {n: float(params[n]) for n in SVI_PARAMS}
    cols = []
    for name in SVI_PARAMS:
        h = 1e-6 * max(1.0, abs(base[name]))
        up, dn = dict(base), dict(base)
        up[name] += h
        dn[name] -= h
        cols.append((np.asarray(svi_iv(K, up, F=F, T=T)) - np.asarray(svi_iv(K, dn, F=F, T=T))) / (2.0 * h))
    return np.column_stack(cols)


def svi_quote_sensitivity(
    strikes: Sequence[float],
    market_vols: Sequence[float],
    F: float,
    T: float,
    params: Mapping,
    weights: Optional[Sequence[float]] = None,
    loss: str = "soft_l1",
    f_scale: float = 0.005,
) -> np.ndarray:
    """``d theta / d q``: (5, n) first-order response of the fitted raw-SVI
    parameters (a, b, rho, m, sigma) to each quoted vol (see the module
    docs). ``weights`` are the fit's weights (default equal); ``loss`` and
    ``f_scale`` its robust loss on the weighted vol residuals."""

    K = np.asarray(strikes, dtype=float)
    q = np.asarray(market_vols, dtype=float)
    if K.shape != q.shape or K.size <= len(SVI_PARAMS):
        raise ValueError(f"need more than {len(SVI_PARAMS)} quotes with one vol per strike")
    if loss not in _IRLS:
        raise ValueError(f"loss must be one of {sorted(_IRLS)}, got {loss!r}")
    w = np.ones(K.size) if weights is None else np.asarray(weights, dtype=float)
    J = _svi_vol_jacobian(K, F, T, params)
    resid = w * (np.asarray(svi_iv(K, params, F=F, T=T)) - q)
    omega = w * w * _IRLS[loss]((resid / f_scale) ** 2)
    A = J.T @ (J * omega[:, None])
    try:
        return np.linalg.solve(A, J.T * omega[None, :])
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"the SVI fit is not identified at these quotes ({exc})") from exc


def svi_bucket_vegas(
    K: float,
    is_call: bool,
    F: float,
    T: float,
    df: float,
    params: Mapping,
    strikes: Sequence[float],
    market_vols: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    loss: str = "soft_l1",
    f_scale: float = 0.005,
    edges: Sequence[float] = Z_EDGES,
    labels: Sequence[str] = Z_LABELS,
) -> Dict[str, object]:
    """Vega by quote of an option (strike ``K``) priced on a fitted SVI
    smile, through the refit (module docs), and summed into standardised
    moneyness buckets ``z = ln(K_i/F) / (atm_vol sqrt(T))``.

    Returns ``{"quotes": DataFrame(strike, k, z, bucket, vega), "buckets":
    DataFrame(bucket, z_from, z_to, n_quotes, vega), "total": sum of the
    quote vegas, "black_vega": the option's Black vega at its smile vol}``;
    vegas per +1 vol point (0.01) on the quotes, premium per unit of
    underlying, signed as long-option P&L. ``total`` is the refitted
    smile's response to a parallel +1 vol point shift of every quote (close
    to ``black_vega`` when the smile can absorb a parallel move)."""

    _check_option(F, K, T, df)
    if len(edges) != len(labels) + 1:
        raise ValueError("need one bucket label per pair of edges")
    Kq = np.asarray(strikes, dtype=float)
    sens = svi_quote_sensitivity(Kq, market_vols, F, T, params, weights, loss, f_scale)
    grad = svi_parameter_vegas(K, F, T, params, df, is_call)["dV_dp"].to_numpy(float)
    vega_q = 0.01 * grad @ sens
    atm = float(svi_iv(F, params, F=F, T=T))
    k = np.log(Kq / F)
    z = k / (atm * np.sqrt(T))
    idx = np.clip(np.searchsorted(np.asarray(edges, dtype=float), z, side="right") - 1, 0, len(labels) - 1)
    quotes = pd.DataFrame({"strike": Kq, "k": k, "z": z, "bucket": [labels[i] for i in idx], "vega": vega_q})
    rows = []
    for i, lab in enumerate(labels):
        sel = idx == i
        rows.append(
            {
                "bucket": lab,
                "z_from": float(edges[i]),
                "z_to": float(edges[i + 1]),
                "n_quotes": int(sel.sum()),
                "vega": float(vega_q[sel].sum()),
            }
        )
    vol_k = float(svi_iv(K, params, F=F, T=T))
    return {
        "quotes": quotes,
        "buckets": pd.DataFrame(rows),
        "total": float(vega_q.sum()),
        "black_vega": 0.01 * float(black_vega(F, K, T, vol_k, df)),
        "vol": vol_k,
    }
