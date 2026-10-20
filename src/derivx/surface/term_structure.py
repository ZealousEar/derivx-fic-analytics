"""Implied-volatility surface from calibrated SVI slices: calendar checks,
time interpolation in total variance, and Black-76 cross-Greeks.

Conventions: one trade date per surface; ``tau`` in ACT/365F years; slices
are raw-SVI total-variance fits in forward log-moneyness ``k = ln(K/F(T))``.
Between expiries total variance is interpolated linearly in T at fixed k
(calendar-arbitrage free when the slices are); before the first expiry and
after the last the implied vol at fixed k is held flat. Forwards are
interpolated log-linearly in T from the spot at T = 0 (a piecewise-constant
carry rate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from derivx.calibration.production import CalibratedSlice, calibrate_chain
from derivx.vol.svi import svi_iv, svi_total_variance


@dataclass
class SurfaceModel:
    """Calibrated slices of one trade date, sorted by ``tau``.

    ``calendar_violations`` counts adjacent expiry pairs whose total
    variance decreases at some k (see ``calendar``, one row per pair)."""

    trade_date: pd.Timestamp
    slices: List[CalibratedSlice] = field(default_factory=list)
    calendar_violations: int = 0
    calendar: pd.DataFrame = field(default_factory=pd.DataFrame)
    spot: Optional[float] = None

    def get_slice(self, expiry: pd.Timestamp) -> CalibratedSlice | None:
        target = pd.Timestamp(expiry).tz_localize(None).normalize()
        for slc in self.slices:
            if slc.expiry == target:
                return slc
        return None

    @property
    def taus(self) -> np.ndarray:
        return np.array([s.tau for s in self.slices])

    def _require(self) -> None:
        if not self.slices:
            raise ValueError("the surface has no calibrated slices")

    def forward(self, T: float) -> float:
        """Forward to ``T`` years, log-linear in T through the slice forwards
        (from the spot at T = 0 when known; flat carry beyond the ends)."""

        self._require()
        taus, fwds = self.taus, np.array([s.forward for s in self.slices])
        if self.spot is not None:
            taus, fwds = np.r_[0.0, taus], np.r_[self.spot, fwds]
        logf = np.log(fwds)
        if taus.size == 1:
            return float(fwds[0])
        if T <= taus[0]:
            slope = (logf[1] - logf[0]) / (taus[1] - taus[0])
            return float(np.exp(logf[0] + slope * (T - taus[0])))
        if T >= taus[-1]:
            slope = (logf[-1] - logf[-2]) / (taus[-1] - taus[-2])
            return float(np.exp(logf[-1] + slope * (T - taus[-1])))
        return float(np.exp(np.interp(T, taus, logf)))

    def total_variance(self, k, T: float) -> np.ndarray:
        """Total implied variance at forward log-moneyness ``k`` and ``T``."""

        self._require()
        if not np.isfinite(T) or T <= 0:
            raise ValueError("T must be > 0 years")
        k = np.asarray(k, dtype=float)
        taus = self.taus
        ws = [np.asarray(svi_total_variance(k, s.params), dtype=float) for s in self.slices]
        if T <= taus[0]:
            return ws[0] * T / taus[0]
        if T >= taus[-1]:
            return ws[-1] * T / taus[-1]
        j = int(np.searchsorted(taus, T))
        t0, t1 = taus[j - 1], taus[j]
        lam = (T - t0) / (t1 - t0)
        return (1.0 - lam) * ws[j - 1] + lam * ws[j]

    def implied_vol(self, K, T: float) -> np.ndarray:
        """Black vol at strike(s) K and any T (see module docs)."""

        K = np.asarray(K, dtype=float)
        if np.any(K <= 0):
            raise ValueError("strikes must be > 0")
        w = self.total_variance(np.log(K / self.forward(T)), T)
        vol = np.sqrt(np.maximum(w, 0.0) / T)
        return float(vol) if np.ndim(vol) == 0 else vol


def _naive_ns_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Date columns as tz-naive datetime64[ns], so frames built with
    different datetime units concatenate (pandas 2.1 mis-sizes mixed-unit
    concatenations)."""

    out = df.copy()
    for col in ("trade_date", "expiry"):
        if col in out.columns:
            dates = pd.to_datetime(out[col])
            if dates.dt.tz is not None:
                dates = dates.dt.tz_convert(None)
            out[col] = dates.astype("datetime64[ns]")
    return out


def calibrate_surface(
    chains_by_maturity: Dict[pd.Timestamp, pd.DataFrame],
    moneyness_band: Optional[Sequence[float]] = None,
    **kwargs,
) -> SurfaceModel:
    """Calibrate SVI to every expiry of one trade date (see
    :func:`derivx.calibration.production.calibrate_chain` for ``kwargs``)
    and check calendar arbitrage between every adjacent pair on the union
    of their quoted k ranges. ``moneyness_band`` is on K/F. Raises
    ``ValueError`` when the chains span several trade dates."""

    if not chains_by_maturity:
        raise ValueError("chains_by_maturity must not be empty")
    chain = pd.concat([_naive_ns_dates(df) for df in chains_by_maturity.values()], ignore_index=True)
    if "trade_date" not in chain.columns and "trade_date" not in kwargs:
        raise ValueError("chains need a 'trade_date' column (or pass trade_date=)")
    dates = (
        pd.to_datetime(chain["trade_date"]).dt.normalize().unique()
        if "trade_date" in chain.columns
        else [pd.Timestamp(kwargs["trade_date"])]
    )
    if len(dates) != 1:
        raise ValueError(f"a surface is for one trade date; got {len(dates)}")
    run = calibrate_chain(chain, moneyness_band=moneyness_band, **kwargs)
    slices = sorted(run.slices, key=lambda s: s.tau)
    violations = int((~run.calendar["calendar_ok"].astype(bool)).sum()) if not run.calendar.empty else 0
    spot = float(slices[0].diagnostics.get("spot")) if slices and slices[0].diagnostics.get("spot") else None
    return SurfaceModel(
        trade_date=pd.Timestamp(dates[0]),
        slices=slices,
        calendar_violations=violations,
        calendar=run.calendar,
        spot=spot,
    )


def black76_cross_greeks(F, K, T, vol, df=1.0) -> Dict[str, np.ndarray]:
    """Black-76 call Greeks on the forward (vectorised).

    Per unit of underlying notional, ``df`` the discount factor to payment:

    * ``delta`` = dV/dF = df N(d1) (put: delta - df)
    * ``vega`` = dV/dsigma = df F n(d1) sqrt(T), per 1.00 of vol (x 0.01 per vol point)
    * ``volga`` = d2V/dsigma2 = vega d1 d2 / sigma
    * ``vanna`` = d2V/dF dsigma = -df n(d1) d2 / sigma (same for puts)
    * ``charm`` = d(delta)/dt as calendar time passes with F, sigma and the
      rate held fixed, per year: ``r df N(d1) + df n(d1) d2 / (2T)`` with
      ``r = -ln(df)/T`` (put: charm - r df)

    with ``d1 = (ln(F/K) + sigma^2 T/2) / (sigma sqrt(T))`` (no rate term:
    F is already a forward) and ``d2 = d1 - sigma sqrt(T)``."""

    F, K, T, vol, df = (np.asarray(x, dtype=float) for x in (F, K, T, vol, df))
    if np.any(F <= 0) or np.any(K <= 0) or np.any(T <= 0) or np.any(vol <= 0) or np.any(df <= 0):
        raise ValueError("cross-Greeks need F > 0, K > 0, T > 0, vol > 0 and df > 0")
    sqrt_t = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    n1 = norm.pdf(d1)
    r = -np.log(df) / T
    vega = df * F * n1 * sqrt_t
    return {
        "delta": df * norm.cdf(d1),
        "vega": vega,
        "volga": vega * d1 * d2 / vol,
        "vanna": -df * n1 * d2 / vol,
        "charm": r * df * norm.cdf(d1) + df * n1 * d2 / (2.0 * T),
        "d1": d1,
        "d2": d2,
    }


def compute_cross_greeks(
    surface: SurfaceModel,
    strikes: Iterable[float],
    rate: float = 0.0,
) -> pd.DataFrame:
    """Vega, volga, vanna, delta and charm of a call at each strike and
    expiry, priced with Black-76 on each slice's forward at the fitted smile
    vol (sticky strike: the smile does not move with F). ``rate`` is the
    continuously compounded discount rate (discounting only; it is not
    added to the forward). Units per :func:`black76_cross_greeks`."""

    rows: List[Dict[str, float]] = []
    strikes = np.asarray(list(strikes), dtype=float)
    for slc in surface.slices:
        if slc.tau <= 0:
            continue
        vols = np.asarray(svi_iv(strikes, slc.params, F=slc.forward, T=slc.tau), dtype=float)
        good = vols > 0
        if not good.any():
            continue
        df = float(np.exp(-rate * slc.tau))
        g = black76_cross_greeks(slc.forward, strikes[good], slc.tau, vols[good], df)
        for i, strike in enumerate(strikes[good]):
            rows.append(
                {
                    "trade_date": surface.trade_date,
                    "expiry": slc.expiry,
                    "tau": slc.tau,
                    "forward": slc.forward,
                    "strike": float(strike),
                    "vol": float(vols[good][i]),
                    "delta": float(g["delta"][i]),
                    "vega": float(g["vega"][i]),
                    "volga": float(g["volga"][i]),
                    "vanna": float(g["vanna"][i]),
                    "charm": float(g["charm"][i]),
                }
            )
    return pd.DataFrame(rows)
