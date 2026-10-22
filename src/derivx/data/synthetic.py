"""SYNTHETIC, seeded option-chain history for demonstrating the walk-forward
tab. Nothing here is market data: every row has ``source = 'synthetic'``
and ticker ``SYNTH``.

Generator (one snapshot per trading day): the spot follows a lognormal
random walk; the 30-day ATM vol level follows a mean-reverting log process
whose shocks are correlated with the spot's (spot down, vol up). Each day's
smile is a power-law SSVI surface (``rho``, ``eta``, ``gamma`` fixed) whose
ATM total variance is ``level^2 T`` times a gently upward term structure.
Quotes: calls and puts on up to 33 strikes (multiples of 5) from -4 to
+3 standard deviations around the carry forward ``S exp((r - q) T)``, Black-76 prices of
the smile vol plus i.i.d. vol noise (``noise_bp``), and a bid/ask of
``max(0.01, spread_frac * price)`` around that price. ``implied_vol`` is
the noisy vol. Expiries are the weekly Fridays out to 6 weeks and the
monthly third Fridays out to about 6 months, so short expiries roll off as
the days pass.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from derivx.data.calendar import trading_days
from derivx.vol.implied import black_price
from derivx.vol.ssvi import SsviSurface

DEFAULT_SEED = 20261020
SYNTH_TICKER = "SYNTH"


def _expiries(trade_date: pd.Timestamp) -> list:
    out = set()
    friday = trade_date + pd.Timedelta(days=(4 - trade_date.weekday()) % 7 or 7)
    for i in range(6):
        out.add(friday + pd.Timedelta(weeks=i))
    month = pd.Timestamp(trade_date.year, trade_date.month, 1)
    for i in range(7):
        first = month + pd.DateOffset(months=i)
        third_friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
        if third_friday > trade_date + pd.Timedelta(days=1) and (third_friday - trade_date).days <= 190:
            out.add(pd.Timestamp(third_friday))
    return sorted(d for d in out if (d - trade_date).days >= 1)


def synthetic_history(
    n_days: int = 8,
    *,
    end: str = "2026-10-19",
    seed: int = DEFAULT_SEED,
    spot0: float = 580.0,
    vol0: float = 0.17,
    rate: float = 0.04,
    div_yield: float = 0.012,
    spot_vol: float = 0.15,
    vol_of_vol: float = 0.9,
    spot_vol_corr: float = -0.7,
    noise_bp: float = 15.0,
    spread_frac: float = 0.02,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """``n_days`` daily SYNTHETIC snapshots ending on the trading day
    ``end`` (see the module docs). Deterministic for a given ``seed``."""

    if n_days < 1:
        raise ValueError("n_days must be >= 1")
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end)
    days = trading_days(end_ts - pd.Timedelta(days=3 * n_days + 10), end_ts)[-n_days:]
    dt = 1.0 / 252.0
    spot, log_level = spot0, 0.0
    out: Dict[pd.Timestamp, pd.DataFrame] = {}
    for i, td in enumerate(days):
        if i > 0:
            z1, z2 = rng.standard_normal(2)
            z2 = spot_vol_corr * z1 + np.sqrt(1.0 - spot_vol_corr**2) * z2
            spot *= float(np.exp(-0.5 * spot_vol**2 * dt + spot_vol * np.sqrt(dt) * z1))
            log_level += -2.0 * log_level * dt + vol_of_vol * np.sqrt(dt) * z2
        level = vol0 * float(np.exp(log_level))
        exps = _expiries(td)
        taus = np.array([(e - td).days / 365.0 for e in exps])
        theta = level**2 * taus * (1.0 + 0.15 * np.sqrt(taus))
        surface = SsviSurface(expiries=taus, theta=np.maximum.accumulate(theta), rho=-0.6, eta=1.1, gamma=0.4)
        rows = []
        for exp, T in zip(exps, taus):
            F = spot * np.exp((rate - div_yield) * T)
            sd = level * np.sqrt(T)
            strikes = np.unique(np.round(F * np.exp(np.linspace(-4.0 * sd, 3.0 * sd, 33)) / 5.0) * 5.0)
            if strikes.size < 8:
                continue
            vols = surface.implied_vol(np.log(strikes / F), T)
            for typ in ("call", "put"):
                noisy = np.maximum(vols + noise_bp * 1e-4 * rng.standard_normal(strikes.size), 0.01)
                price = black_price(F, strikes, T, noisy, typ == "call", np.exp(-rate * T))
                half = np.maximum(0.005, 0.5 * spread_frac * price)
                rows.append(
                    pd.DataFrame(
                        {
                            "ticker": SYNTH_TICKER,
                            "trade_date": td,
                            "expiry": exp,
                            "option_type": typ,
                            "strike": strikes,
                            "bid": np.round(np.maximum(price - half, 0.0), 4),
                            "ask": np.round(price + half, 4),
                            "volume": 100.0,
                            "open_interest": 1000.0,
                            "implied_vol": noisy,
                            "underlying_price": round(spot, 4),
                            "last_trade_date": (td + pd.Timedelta(hours=19, minutes=59)).tz_localize("UTC"),
                            "fetched_at_utc": (td + pd.Timedelta(hours=20, minutes=30)).isoformat() + "Z",
                            "source": "synthetic",
                        }
                    )
                )
        frame = pd.concat(rows, ignore_index=True)
        frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
        for col in ("trade_date", "expiry"):
            frame[col] = frame[col].astype("datetime64[ns]")
        out[pd.Timestamp(td)] = frame
    return out
