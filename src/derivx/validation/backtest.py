"""Walk-forward validation utilities for calibrated volatility models."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from derivx.calibration.production import calibrate_with_constraints
from derivx.vol.svi import svi_iv


def _slice_chain(chain: pd.DataFrame, expiry: pd.Timestamp, band: Sequence[float]) -> pd.DataFrame:
    df = chain.copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.tz_localize(None)
    mask_expiry = df["expiry"] == pd.Timestamp(expiry).tz_localize(None)
    slice_df = df.loc[mask_expiry].copy()
    if slice_df.empty:
        return slice_df
    if "moneyness" not in slice_df.columns:
        if "underlying_price" not in slice_df.columns:
            raise ValueError("Chain missing 'underlying_price' column")
        slice_df["moneyness"] = slice_df["strike"] / slice_df["underlying_price"]
    mask = (slice_df["moneyness"] >= band[0]) & (slice_df["moneyness"] <= band[1])
    return slice_df.loc[mask].sort_values("strike")


def _prepare_prediction(prev_slice: pd.DataFrame, next_slice: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    merged = pd.merge(
        prev_slice[["strike", "implied_vol"]].rename(columns={"implied_vol": "prev_iv"}),
        next_slice[["strike", "implied_vol", "underlying_price"]],
        on="strike",
        how="inner",
    )
    if merged.empty:
        return merged
    forward_next = float(merged["underlying_price"].iloc[0])
    strikes = merged["strike"].to_numpy()
    merged["predicted_iv"] = [svi_iv(k, params, F=forward_next) for k in strikes]
    return merged


def measure_prediction_error(pred_frame: pd.DataFrame) -> Dict[str, float]:
    if pred_frame.empty:
        return {"rmse": float("nan"), "hit_rate": float("nan"), "n_points": 0}
    diffs = pred_frame["predicted_iv"] - pred_frame["implied_vol"]
    rmse = float(np.sqrt(np.mean(diffs**2)))
    actual_delta = pred_frame["implied_vol"] - pred_frame["prev_iv"]
    predicted_delta = pred_frame["predicted_iv"] - pred_frame["prev_iv"]
    mask = actual_delta != 0
    if mask.any():
        hits = np.sign(actual_delta[mask]) == np.sign(predicted_delta[mask])
        hit_rate = float(hits.mean())
    else:
        hit_rate = float("nan")
    return {"rmse": rmse, "hit_rate": hit_rate, "n_points": int(len(pred_frame))}


def walk_forward_test(
    chains: Mapping[pd.Timestamp, pd.DataFrame],
    window: int = 10,
    moneyness_band: Sequence[float] = (0.85, 1.15),
) -> pd.DataFrame:
    """Run a walk-forward SVI validation across a sequence of daily chains."""

    dates = sorted(chains.keys())
    if len(dates) < window + 2:
        return pd.DataFrame()

    metrics = []
    for idx in range(window, len(dates) - 1):
        trade_date = dates[idx]
        next_date = dates[idx + 1]
        chain_today = chains[trade_date]
        chain_next = chains[next_date]
        slices = calibrate_with_constraints(chain_today, moneyness_band=moneyness_band)
        for cal_slice in slices:
            prev_slice = _slice_chain(chain_today, cal_slice.expiry, moneyness_band)
            next_slice = _slice_chain(chain_next, cal_slice.expiry, moneyness_band)
            if prev_slice.empty or next_slice.empty:
                continue
            pred_frame = _prepare_prediction(prev_slice, next_slice, cal_slice.params)
            if pred_frame.empty:
                continue
            err = measure_prediction_error(pred_frame)
            metrics.append(
                {
                    "trade_date": trade_date,
                    "next_date": next_date,
                    "expiry": cal_slice.expiry,
                    **err,
                }
            )
    return pd.DataFrame(metrics)
