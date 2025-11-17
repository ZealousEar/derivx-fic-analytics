"""Production-oriented calibration utilities for SVI."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from derivx.vol.svi import basic_no_arb_checks, fit_svi_strike_slice, svi_iv


@dataclass
class CalibratedSlice:
    trade_date: pd.Timestamp
    expiry: pd.Timestamp
    forward: float
    tau: float
    params: Dict[str, float]
    rmse: float
    n_points: int
    strikes: np.ndarray
    observed_ivs: np.ndarray
    diagnostics: Dict[str, float] = field(default_factory=dict)
    benchmark: Optional[Dict[str, float]] = None


def _ensure_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None)


def _ensure_moneyness(df_slice: pd.DataFrame) -> pd.Series:
    if "moneyness" in df_slice.columns:
        return df_slice["moneyness"]
    if "underlying_price" not in df_slice.columns:
        raise ValueError("Chain must include 'underlying_price' to compute moneyness")
    return df_slice["strike"] / df_slice["underlying_price"]


def calibrate_with_constraints(
    chain: pd.DataFrame,
    enforce_arbitrage: bool = True,
    moneyness_band: Sequence[float] = (0.8, 1.2),
    min_points: int = 5,
) -> List[CalibratedSlice]:
    """Calibrate SVI to each expiry for the provided option chain."""

    if chain.empty:
        return []
    if "expiry" not in chain.columns:
        raise ValueError("Chain must contain 'expiry' column")

    trade_date = _ensure_timestamp(chain.get("trade_date", pd.Timestamp(dt.date.today())).iloc[0])
    results: List[CalibratedSlice] = []

    for expiry, slice_df in chain.groupby("expiry"):
        expiry_ts = _ensure_timestamp(expiry)
        tau = max((expiry_ts - trade_date) / np.timedelta64(1, "D") / 365.0, 1e-6)
        df_slice = slice_df.copy().sort_values("strike")
        df_slice["moneyness"] = _ensure_moneyness(df_slice)
        mask = (df_slice["moneyness"] >= moneyness_band[0]) & (df_slice["moneyness"] <= moneyness_band[1])
        df_slice = df_slice.loc[mask]
        df_slice = df_slice.dropna(subset=["strike", "implied_vol"])
        if len(df_slice) < min_points:
            continue

        strikes = df_slice["strike"].to_numpy()
        ivs_obs = df_slice["implied_vol"].to_numpy()
        forward = float(df_slice.get("underlying_price", df_slice["strike"]).iloc[0])

        params = fit_svi_strike_slice(strikes, ivs_obs, F=forward)
        ivs_fit = np.array([svi_iv(k, params, F=forward) for k in strikes])
        rmse = float(np.sqrt(np.mean((ivs_fit - ivs_obs) ** 2)))

        logm = np.log(strikes / forward)
        design = np.column_stack([np.ones_like(logm), logm, logm**2])
        condition_number = float(np.linalg.cond(design))

        diagnostics = {
            "condition_number": condition_number,
            "moneyness_min": float(df_slice["moneyness"].min()),
            "moneyness_max": float(df_slice["moneyness"].max()),
            "tau": tau,
        }
        if enforce_arbitrage:
            diagnostics.update(basic_no_arb_checks([params]))

        result = CalibratedSlice(
            trade_date=trade_date,
            expiry=expiry_ts,
            forward=forward,
            tau=tau,
            params=params,
            rmse=rmse,
            n_points=len(df_slice),
            strikes=strikes,
            observed_ivs=ivs_obs,
            diagnostics=diagnostics,
        )
        result.benchmark = benchmark_vs_baseline(result, df_slice)
        results.append(result)
    return results


def benchmark_vs_baseline(calibrated_slice: CalibratedSlice, chain_slice: pd.DataFrame) -> Dict[str, float]:
    """Compare the calibrated SVI smile to a cubic spline baseline."""

    strikes = chain_slice["strike"].to_numpy()
    ivs_obs = chain_slice["implied_vol"].to_numpy()
    order = np.argsort(strikes)
    strikes = strikes[order]
    ivs_obs = ivs_obs[order]

    forward = calibrated_slice.forward
    logm = np.log(strikes / forward)

    try:
        spline = CubicSpline(logm, ivs_obs, bc_type="natural")
        spline_pred = spline(logm)
        rmse_spline = float(np.sqrt(np.mean((spline_pred - ivs_obs) ** 2)))
    except Exception:  # pragma: no cover - spline can fail on duplicate points
        rmse_spline = float("nan")

    svi_pred = np.array([svi_iv(k, calibrated_slice.params, F=forward) for k in strikes])
    rmse_svi = float(np.sqrt(np.mean((svi_pred - ivs_obs) ** 2)))

    improvement = float("nan")
    if np.isfinite(rmse_spline) and rmse_spline > 0:
        improvement = 1.0 - rmse_svi / rmse_spline

    return {
        "rmse_svi": rmse_svi,
        "rmse_spline": rmse_spline,
        "relative_improvement": improvement,
    }
