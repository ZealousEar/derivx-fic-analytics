"""Synthetic, labelled option chains for the vol-library tests.

Quotes are generated from a known arbitrage-free SSVI surface (so every
fit has a ground truth): calls and puts at every strike, Black-76 prices on
the carry forward ``S exp((r - q) T)``, a bid/ask around each price, and the
implied vol of the mid. Nothing here is market data.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from derivx.vol.implied import black_price
from derivx.vol.ssvi import SsviSurface

TRADE_DATE = pd.Timestamp("2026-10-19")
SURFACE = SsviSurface(
    expiries=np.array([7, 30, 91, 182, 365]) / 365.0,
    theta=0.18**2 * np.array([7, 30, 91, 182, 365]) / 365.0 * np.array([1.0, 1.03, 1.07, 1.1, 1.15]),
    rho=-0.65,
    eta=1.2,
    gamma=0.4,
)


def true_vol(K, F, T) -> np.ndarray:
    return SURFACE.implied_vol(np.log(np.asarray(K, dtype=float) / F), T)


def synth_chain(
    days: Sequence[int] = (7, 30, 91, 182, 365),
    spot: float = 580.0,
    rate: float = 0.04,
    div_yield: float = 0.012,
    trade_date: pd.Timestamp = TRADE_DATE,
    n_strikes: int = 31,
    width_sd: Sequence[float] = (-4.0, 3.0),
    iv_noise: float = 0.0,
    seed: int = 0,
    strike_step: Optional[float] = None,
) -> pd.DataFrame:
    """Calls and puts on ``n_strikes`` strikes per expiry, spanning
    ``width_sd`` standard deviations (0.18 sqrt(T)) around the forward."""

    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        T = d / 365.0
        F = spot * np.exp((rate - div_yield) * T)
        df = np.exp(-rate * T)
        sd = 0.18 * np.sqrt(max(T, 1 / 365))
        Ks = F * np.exp(np.linspace(width_sd[0] * sd, width_sd[1] * sd, n_strikes))
        if strike_step:
            Ks = np.unique(np.round(Ks / strike_step) * strike_step)
        vols = true_vol(Ks, F, T) if T > 0 else np.full(Ks.size, 0.18)
        for K, vol in zip(Ks, vols):
            for typ in ("call", "put"):
                if T > 0:
                    p = float(black_price(F, K, T, vol, typ == "call", df))
                else:
                    p = max(F - K, 0.0) if typ == "call" else max(K - F, 0.0)
                half = max(0.005, 0.01 * p)
                rows.append(
                    {
                        "trade_date": trade_date,
                        "expiry": trade_date + pd.Timedelta(days=d),
                        "option_type": typ,
                        "strike": float(K),
                        "bid": max(p - half, 0.0),
                        "ask": p + half,
                        "implied_vol": float(vol + iv_noise * rng.standard_normal()) if T > 0 else 1e-5,
                        "underlying_price": spot,
                        "volume": 50,
                        "open_interest": 100,
                        "lastTradeDate": trade_date,
                    }
                )
    out = pd.DataFrame(rows)
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    return out
