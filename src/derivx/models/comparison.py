"""Model benchmarking utilities for volatility smiles."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from derivx.vol.sabr import calibrate_sabr_smile, hagan_sabr_iv
from derivx.vol.svi import fit_svi_strike_slice, svi_iv


def _prepare_inputs(chain: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    df = chain.sort_values("strike").dropna(subset=["strike", "implied_vol"])
    if df.empty:
        raise ValueError("chain must contain strike/implied_vol data")
    strikes = df["strike"].to_numpy()
    ivs = df["implied_vol"].to_numpy()
    forward = float(df.get("underlying_price", df["strike"]).iloc[0])
    trade_date = pd.to_datetime(df["trade_date"].iloc[0]) if "trade_date" in df.columns else pd.Timestamp("today")
    expiry = pd.to_datetime(df["expiry"].iloc[0]) if "expiry" in df.columns else trade_date + pd.Timedelta(days=30)
    tau = max((expiry - trade_date) / np.timedelta64(1, "D") / 365.0, 1e-6)
    return strikes, ivs, forward, float(tau)


def compare_models(chain: pd.DataFrame, models: Iterable[str] = ("svi", "sabr", "local_vol")) -> pd.DataFrame:
    """Fit multiple models to the same smile and report in-sample RMSE."""

    strikes, ivs_obs, forward, tau = _prepare_inputs(chain)
    results: List[dict] = []

    if "svi" in models:
        params = fit_svi_strike_slice(strikes, ivs_obs, F=forward)
        ivs_fit = np.array([svi_iv(k, params, F=forward) for k in strikes])
        rmse = float(np.sqrt(np.mean((ivs_fit - ivs_obs) ** 2)))
        results.append({"model": "svi", "rmse": rmse, "n_points": len(strikes)})

    if "sabr" in models:
        params = calibrate_sabr_smile(strikes, ivs_obs, T=tau, F=forward, beta=0.5)
        ivs_fit = np.array([hagan_sabr_iv(forward, k, tau, **params) for k in strikes])
        rmse = float(np.sqrt(np.mean((ivs_fit - ivs_obs) ** 2)))
        results.append({"model": "sabr", "rmse": rmse, "n_points": len(strikes)})

    if "local_vol" in models:
        logm = np.log(strikes / forward)
        try:
            spline = CubicSpline(logm, ivs_obs, bc_type="natural")
            ivs_fit = spline(logm)
            rmse = float(np.sqrt(np.mean((ivs_fit - ivs_obs) ** 2)))
        except Exception:
            rmse = float("nan")
        results.append({"model": "local_vol", "rmse": rmse, "n_points": len(strikes)})

    return pd.DataFrame(results)
