"""Volatility surface calibration and cross-greek utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from derivx.calibration.production import CalibratedSlice, calibrate_with_constraints
from derivx.vol.svi import svi_iv


@dataclass
class SurfaceModel:
    trade_date: pd.Timestamp
    slices: List[CalibratedSlice] = field(default_factory=list)
    calendar_violations: int = 0

    def get_slice(self, expiry: pd.Timestamp) -> CalibratedSlice | None:
        target = pd.Timestamp(expiry).tz_localize(None)
        for slc in self.slices:
            if slc.expiry == target:
                return slc
        return None


def calibrate_surface(
    chains_by_maturity: Dict[pd.Timestamp, pd.DataFrame],
    moneyness_band: Sequence[float] = (0.85, 1.15),
) -> SurfaceModel:
    """Calibrate SVI for multiple maturities on a given trade date."""

    if not chains_by_maturity:
        raise ValueError("chains_by_maturity must not be empty")
    trade_date = None
    all_slices: List[CalibratedSlice] = []
    for expiry, df in chains_by_maturity.items():
        if trade_date is None:
            trade_date = pd.to_datetime(df["trade_date"].iloc[0])
        results = calibrate_with_constraints(df, moneyness_band=moneyness_band)
        all_slices.extend(results)
    all_slices.sort(key=lambda slc: slc.tau)

    total_variance = []
    for slc in all_slices:
        atm_vol = float(svi_iv(slc.forward, slc.params, F=slc.forward))
        total_variance.append(atm_vol**2 * slc.tau)
    calendar_violations = sum(1 for earlier, later in zip(total_variance, total_variance[1:]) if later < earlier)

    return SurfaceModel(trade_date=trade_date or pd.Timestamp("today"), slices=all_slices, calendar_violations=calendar_violations)


def compute_cross_greeks(
    surface: SurfaceModel,
    strikes: Iterable[float],
    rate: float = 0.0,
) -> pd.DataFrame:
    """Compute basic Vega/Volga/Vanna buckets across the calibrated surface."""

    rows: List[Dict[str, float]] = []
    strikes = list(strikes)
    for slc in surface.slices:
        if slc.tau <= 0:
            continue
        sqrt_tau = np.sqrt(slc.tau)
        vols = np.array([svi_iv(k, slc.params, F=slc.forward) for k in strikes])
        for strike, vol in zip(strikes, vols):
            if vol <= 0:
                continue
            d1 = (np.log(slc.forward / strike) + (rate + 0.5 * vol**2) * slc.tau) / (vol * sqrt_tau)
            d2 = d1 - vol * sqrt_tau
            vega = slc.forward * norm.pdf(d1) * sqrt_tau
            volga = vega * d1 * d2 / vol
            vanna = vega * d2 / (slc.forward * vol)
            rows.append(
                {
                    "trade_date": surface.trade_date,
                    "expiry": slc.expiry,
                    "tau": slc.tau,
                    "strike": strike,
                    "vega": vega,
                    "volga": volga,
                    "vanna": vanna,
                }
            )
    return pd.DataFrame(rows)
