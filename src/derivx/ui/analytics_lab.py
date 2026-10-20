"""ANALYTICS lab: market data, SVI calibration, walk-forward validation and
model comparison on equity option chains."""

from __future__ import annotations

import datetime as dt
from io import BytesIO
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import CubicSpline

from derivx.calibration.production import CalibratedSlice, calibrate_with_constraints
from derivx.data.market_data import apply_liquidity_filters, compute_moneyness, fetch_option_chain
from derivx.models.comparison import compare_models
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.theme import PLOTLY_2D_CONFIG
from derivx.validation.backtest import walk_forward_test
from derivx.vol.sabr import calibrate_sabr_smile, hagan_sabr_iv
from derivx.vol.svi import fit_svi_strike_slice, svi_iv

ANALYTICS_TABS = [
    "Market Data",
    "Production Calibration",
    "Walk-Forward Validation",
    "Model Comparison",
]

state_defaults = {
    "market_chain_raw": pd.DataFrame(),
    "market_chain": pd.DataFrame(),
    "market_chain_meta": {},
    "calibration_results": [],
    "walkforward_chains": {},
    "walkforward_metrics": pd.DataFrame(),
    "walkforward_meta": {},
    "model_comp_metrics": pd.DataFrame(),
    "model_comp_slice": pd.DataFrame(),
    "model_comp_models": ["svi", "sabr", "local_vol"],
    "chain_fetch_nonce": 0,
    "walk_fetch_nonce": 0,
}


def init_state() -> None:
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



@st.cache_data(show_spinner=False)
def _cached_fetch_chain(
    ticker: str,
    trade_date: str,
    expiry_limit: int | None,
    nonce: int,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    trade_dt = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
    meta: Dict[str, object] = {
        "ticker": ticker.upper(),
        "trade_date": trade_date,
        "expiry_limit": expiry_limit,
        "status": "ok",
    }
    try:
        df = fetch_option_chain(ticker, trade_date=trade_dt)
    except Exception as exc:
        meta["status"] = "error"
        meta["error"] = str(exc)
        return pd.DataFrame(), meta
    if df.empty:
        meta["status"] = "empty"
        return df, meta
    if expiry_limit is not None and expiry_limit > 0:
        expiry_order = (
            pd.to_datetime(df["expiry"]).dt.tz_localize(None).sort_values().unique().tolist()
        )
        keep = set(expiry_order[:expiry_limit])
        df = df[pd.to_datetime(df["expiry"]).dt.tz_localize(None).isin(keep)]
    meta["expiry_count"] = int(pd.to_datetime(df["expiry"]).nunique())
    meta["rows"] = int(len(df))
    return df.reset_index(drop=True), meta


def _apply_filters(chain: pd.DataFrame, min_volume: int, max_spread: float | None, min_mid: float) -> pd.DataFrame:
    if chain.empty:
        return chain
    filtered = apply_liquidity_filters(
        chain,
        min_volume=int(min_volume),
        max_spread=float(max_spread) if max_spread is not None else None,
        min_mid=float(min_mid),
    )
    filtered = compute_moneyness(filtered)
    for col in ("expiry", "trade_date"):
        if col in filtered:
            filtered[col] = pd.to_datetime(filtered[col]).dt.tz_localize(None)
    return filtered


def _chain_summary(chain: pd.DataFrame) -> Dict[str, float]:
    if chain.empty:
        return {"quotes": 0, "strikes": 0, "volume": 0, "moneyness_min": np.nan, "moneyness_max": np.nan}
    strikes = chain["strike"].nunique()
    total_volume = float(chain["volume"].fillna(0).sum())
    moneyness_min = float(chain.get("moneyness", pd.Series(dtype=float)).min())
    moneyness_max = float(chain.get("moneyness", pd.Series(dtype=float)).max())
    return {
        "quotes": int(len(chain)),
        "strikes": int(strikes),
        "volume": total_volume,
        "moneyness_min": moneyness_min,
        "moneyness_max": moneyness_max,
    }


def _raw_chain_stats(chain: pd.DataFrame) -> Dict[str, object]:
    if chain.empty:
        return {"quotes": 0, "expiries": 0, "strikes": 0, "last_expiry": None}
    expiries = pd.to_datetime(chain["expiry"]).dt.tz_localize(None)
    return {
        "quotes": int(len(chain)),
        "expiries": int(expiries.nunique()),
        "strikes": int(chain["strike"].nunique()) if "strike" in chain else 0,
        "last_expiry": expiries.max().date() if not expiries.empty else None,
    }


def _infer_regime(chain: pd.DataFrame) -> str:
    if chain.empty:
        return "n/a"
    if "moneyness" not in chain:
        chain = compute_moneyness(chain)
    atm = chain.loc[chain["moneyness"].between(0.95, 1.05)]
    mean_iv = float(atm["implied_vol"].mean()) if not atm.empty else float(chain["implied_vol"].mean())
    if not np.isfinite(mean_iv):
        return "n/a"
    if mean_iv < 0.15:
        return "low"
    if mean_iv < 0.25:
        return "mid"
    return "high"


def _load_chains_from_files(files: Iterable[BytesIO]) -> Dict[pd.Timestamp, pd.DataFrame]:
    dataset: Dict[pd.Timestamp, pd.DataFrame] = {}
    for buffer in files:
        buffer.seek(0)
        df = pd.read_parquet(buffer)
        if df.empty or "trade_date" not in df.columns:
            continue
        trade_date = pd.to_datetime(df["trade_date"].iloc[0]).tz_localize(None).normalize()
        dataset[trade_date] = df
    return dataset


@st.cache_data(show_spinner=False)
def _download_chains_for_range(
    ticker: str,
    _dates: Iterable[pd.Timestamp],
    nonce: int,
) -> tuple[Dict[pd.Timestamp, pd.DataFrame], Dict[str, object]]:
    payload: Dict[pd.Timestamp, pd.DataFrame] = {}
    failures: list[dict] = []
    for trade_date in _dates:
        iso = pd.Timestamp(trade_date).tz_localize(None)
        iso_str = iso.date().isoformat()
        try:
            df = fetch_option_chain(ticker, trade_date=iso.date())
        except Exception as exc:
            failures.append({"date": iso_str, "reason": str(exc)})
            continue
        if df.empty:
            failures.append({"date": iso_str, "reason": "empty"})
            continue
        payload[iso] = df
    summary = {
        "requested": [pd.Timestamp(d).tz_localize(None).date().isoformat() for d in _dates],
        "succeeded": [ts.date().isoformat() for ts in payload.keys()],
        "failed": failures,
    }
    return payload, summary


def _plot_calibration_smile(cal_slice: CalibratedSlice) -> go.Figure:
    strikes = cal_slice.strikes
    forward = cal_slice.forward
    strikes_dense = np.linspace(strikes.min(), strikes.max(), 200)
    fitted = [svi_iv(k, cal_slice.params, F=forward) for k in strikes_dense]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=cal_slice.observed_ivs,
            mode="markers",
            name="Market",
            marker=dict(color="#FFFFFF", size=8, line=dict(color="#000000", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=strikes_dense,
            y=fitted,
            mode="lines",
            name="SVI fit",
            line=dict(color="#FF4444", width=3),
        )
    )
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#FFFFFF"),
        xaxis=dict(title="Strike", gridcolor="#333333"),
        yaxis=dict(title="Implied Vol", gridcolor="#333333"),
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    bound_axes(
        fig,
        xaxis=axis_limits(strikes, floor=0.0),
        yaxis=axis_limits(np.concatenate([cal_slice.observed_ivs, fitted]), floor=0.0),
    )
    return fig


def _plot_walkforward(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics["next_date"],
            y=metrics["rmse"],
            mode="lines+markers",
            name="RMSE",
            line=dict(color="#FF4444", width=3),
            marker=dict(color="#FFFFFF", size=6, line=dict(color="#000000", width=1)),
        )
    )
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#FFFFFF"),
        xaxis=dict(title="Next Date", gridcolor="#333333"),
        yaxis=dict(title="RMSE", gridcolor="#333333"),
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    dates = pd.to_datetime(metrics["next_date"])
    pad = max((dates.max() - dates.min()) * 0.05, pd.Timedelta(days=1))
    bound_axes(
        fig,
        xaxis={"min": (dates.min() - pad).isoformat(), "max": (dates.max() + pad).isoformat()},
        yaxis=axis_limits(metrics["rmse"], floor=0.0),
    )
    return fig


def _plot_model_comparison(slice_df: pd.DataFrame, models: Iterable[str]) -> go.Figure:
    df = slice_df.sort_values("strike")
    strikes = df["strike"].to_numpy()
    forward = float(df.get("underlying_price", df["strike"]).iloc[0])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=df["implied_vol"],
            mode="markers",
            name="Market",
            marker=dict(color="#FFFFFF", size=8, line=dict(color="#000000", width=1)),
        )
    )
    grid = np.linspace(strikes.min(), strikes.max(), 200)
    if "svi" in models:
        params = fit_svi_strike_slice(strikes, df["implied_vol"], F=forward)
        svi_curve = [svi_iv(k, params, F=forward) for k in grid]
        fig.add_trace(
            go.Scatter(x=grid, y=svi_curve, mode="lines", name="SVI", line=dict(width=3, color="#FF4444"))
        )
    if "sabr" in models:
        tau = max(
            (
                (pd.to_datetime(df["expiry"].iloc[0]) - pd.to_datetime(df["trade_date"].iloc[0]))
                / np.timedelta64(1, "D")
                / 365.0
            ),
            1e-6,
        )
        params = calibrate_sabr_smile(strikes, df["implied_vol"], T=tau, F=forward, beta=0.5)
        sabr_curve = [hagan_sabr_iv(forward, k, tau, **params) for k in grid]
        fig.add_trace(
            go.Scatter(x=grid, y=sabr_curve, mode="lines", name="SABR", line=dict(width=3, dash="dash", color="#FFA500"))
        )
    if "local_vol" in models and len(strikes) >= 4:
        logm = np.log(strikes / forward)
        try:
            spline = CubicSpline(logm, df["implied_vol"], bc_type="natural")
            local_curve = spline(np.log(grid / forward))
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=local_curve,
                    mode="lines",
                    name="Local Vol",
                    line=dict(width=3, dash="dot", color="#00FFFF"),
                )
            )
        except Exception:
            pass
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#FFFFFF"),
        xaxis=dict(title="Strike", gridcolor="#333333"),
        yaxis=dict(title="Implied Vol", gridcolor="#333333"),
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    bound_axes(
        fig,
        xaxis=axis_limits(strikes, floor=0.0),
        yaxis=axis_limits(df["implied_vol"], pad=0.25, floor=0.0),
    )
    return fig


def _render_market_data_tab():
    st.markdown("### / MARKET DATA")
    col_upload, col_fetch = st.columns(2)
    with col_upload:
        uploaded = st.file_uploader("Upload chain (.parquet)", type=["parquet"], key="market_upload")
        if uploaded is not None:
            df = pd.read_parquet(BytesIO(uploaded.getvalue()))
            st.session_state["market_chain_raw"] = df
            st.session_state["market_chain_meta"] = {
                "status": "upload",
                "source": uploaded.name,
                "rows": int(len(df)),
                "ticker": uploaded.name.split(".")[0],
            }
            st.success(f"Loaded {len(df):,} rows from {uploaded.name}")
    with col_fetch:
        last_meta = st.session_state.get("market_chain_meta") or {}
        ticker = st.text_input("Ticker", value=last_meta.get("ticker", "SPY")).upper()
        trade_date = st.date_input("Trade date label", value=dt.date.today(), max_value=dt.date.today())
        expiry_limit = st.number_input("Max expiries", min_value=1, max_value=15, value=4)
        st.caption("Yahoo returns the latest chain only. The date above is stored with the rows for reference.")
        if st.button("Pull from Yahoo Finance", width="stretch"):
            st.session_state["chain_fetch_nonce"] += 1
            with st.spinner("Fetching live chain…"):
                df, meta = _cached_fetch_chain(
                    ticker,
                    trade_date.isoformat(),
                    int(expiry_limit),
                    st.session_state["chain_fetch_nonce"],
                )
            meta["source"] = "yfinance"
            st.session_state["market_chain_meta"] = meta
            status = meta.get("status")
            if status == "error":
                st.error(f"Download failed: {meta.get('error', 'unknown error')}")
            elif status == "empty":
                st.warning("Yahoo Finance returned no quotes. Try again later or upload a saved chain.")
            else:
                st.session_state["market_chain_raw"] = df
                st.success(f"Fetched {meta.get('rows', len(df)):,} quotes across {meta.get('expiry_count', 0)} expiries.")

    raw_chain = st.session_state.get("market_chain_raw", pd.DataFrame())
    if raw_chain.empty:
        st.info("Upload a chain or pull from Yahoo Finance to continue.")
        return

    raw_stats = _raw_chain_stats(raw_chain)
    meta = st.session_state.get("market_chain_meta", {})
    status_text = meta.get("status", "n/a")
    st.caption(
        f"Last load ({status_text}): {raw_stats['quotes']:,} quotes across {raw_stats['expiries']} expiries "
        + (f"(max expiry {raw_stats['last_expiry']})" if raw_stats["last_expiry"] else "")
    )

    st.markdown("#### / LIQUIDITY FILTERS")
    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
    min_volume = filt_col1.number_input("Min volume", min_value=0, value=10, step=5)
    min_mid = filt_col2.number_input("Min mid", min_value=0.0, value=0.10, step=0.05)
    max_spread = filt_col3.number_input("Max rel spread", min_value=0.0, value=0.40, step=0.05)
    disable_spread = filt_col4.checkbox("Disable spread cap", value=False)

    filtered = _apply_filters(
        raw_chain,
        min_volume=int(min_volume),
        max_spread=None if disable_spread else float(max_spread),
        min_mid=float(min_mid),
    )
    st.session_state["market_chain"] = filtered

    summary = _chain_summary(filtered)
    if filtered.empty:
        st.warning("Filters removed all quotes. Loosen the thresholds to inspect the data.")
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    met_col1.metric("Quotes", f"{summary['quotes']:,}")
    met_col2.metric("Unique strikes", f"{summary['strikes']:,}")
    met_col3.metric("Total volume", f"{summary['volume']:.0f}")
    met_col4.metric(
        "Moneyness range",
        f"{summary['moneyness_min']:.2f} — {summary['moneyness_max']:.2f}"
        if np.isfinite(summary["moneyness_min"])
        else "n/a",
    )

    st.dataframe(filtered, width="stretch")
    if not filtered.empty:
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv_bytes, file_name="filtered_chain.csv")


def _render_calibration_tab():
    st.markdown("### / PRODUCTION CALIBRATION")
    chain = st.session_state.get("market_chain", pd.DataFrame())
    if chain.empty:
        st.info("Load filtered market data on the first tab to unlock calibration.")
        return

    expiries = sorted(pd.to_datetime(chain["expiry"]).dt.date.unique().tolist())
    sel_exp = st.multiselect(
        "Select expiries",
        options=expiries,
        default=expiries[:2],
        help="Pick the maturity slices to include in the calibration run.",
    )
    band = st.slider("Moneyness band", 0.7, 1.3, (0.85, 1.15), step=0.01)
    enforce = st.checkbox("Enforce no-arbitrage diagnostics", value=True)

    run = st.button("Calibrate slices", width="stretch")
    if run:
        if sel_exp:
            mask = pd.to_datetime(chain["expiry"]).dt.date.isin(sel_exp)
            subset = chain.loc[mask]
        else:
            subset = chain
        with st.spinner("Calibrating SVI slices…"):
            results = calibrate_with_constraints(
                subset,
                enforce_arbitrage=enforce,
                moneyness_band=band,
            )
        st.session_state["calibration_results"] = results
        if not results:
            st.warning("No slices passed the filters. Loosen the moneyness band or min points.")

    results: List[CalibratedSlice] = st.session_state.get("calibration_results", [])
    if not results:
        return

    rows = []
    for slc in results:
        bench = slc.benchmark or {}
        diag = slc.diagnostics or {}
        rows.append(
            {
                "Expiry": slc.expiry.date(),
                "Points": slc.n_points,
                "RMSE": slc.rmse,
                "Cond #": diag.get("condition_number"),
                "No-Arb Viol (b)": diag.get("viol_b"),
                "No-Arb Viol (rho)": diag.get("viol_rho"),
                "SVI vs spline": bench.get("relative_improvement"),
                "RMSE (spline)": bench.get("rmse_spline"),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")

    options = [f"{slc.expiry.date()}" for slc in results]
    selected = st.selectbox("Visualize expiry", options) if options else None
    if selected:
        slc = next(s for s in results if str(s.expiry.date()) == selected)
        st.plotly_chart(_plot_calibration_smile(slc), width="stretch", config=PLOTLY_2D_CONFIG)
        params_df = pd.DataFrame([slc.params])
        st.markdown("#### / CALIBRATED PARAMS")
        st.dataframe(params_df, width="stretch")
        st.markdown("#### / DIAGNOSTICS")
        st.json(slc.diagnostics)
        if slc.benchmark:
            st.markdown("#### / BASELINE COMPARISON")
            st.json(slc.benchmark)


def _render_walkforward_tab():
    st.markdown("### / WALK-FORWARD VALIDATION")
    source = st.radio(
        "Data source",
        ["Upload Parquets", "Download via yfinance"],
        horizontal=True,
        key="wf_source",
    )
    dataset: Dict[pd.Timestamp, pd.DataFrame] = {}
    if source == "Upload Parquets":
        files = st.file_uploader(
            "Upload daily chain files",
            type=["parquet"],
            accept_multiple_files=True,
            key="wf_upload",
        )
        if files:
            buffers = [BytesIO(file.getvalue()) for file in files]
            dataset = _load_chains_from_files(buffers)
            st.session_state["walkforward_meta"] = {
                "status": "upload",
                "succeeded": [d.date().isoformat() for d in dataset.keys()],
                "failed": [],
            }
            st.success(f"Loaded {len(dataset)} trading days from uploads.")
    else:
        ticker = st.text_input("Ticker", value="SPY", key="wf_ticker").upper()
        col_dates = st.columns(2)
        start_date = col_dates[0].date_input("Start", value=dt.date.today() - dt.timedelta(days=7))
        end_date = col_dates[1].date_input("End", value=dt.date.today())
        max_days = (end_date - start_date).days + 1
        if max_days > 20:
            st.warning("Limit the window to <= 20 days to keep the test fast.")
        if st.button("Download history", width="stretch"):
            all_dates = pd.date_range(start_date, end_date, freq="B")  # business days
            st.session_state["walk_fetch_nonce"] += 1
            with st.spinner("Downloading historical chains…"):
                dataset, meta = _download_chains_for_range(
                    ticker,
                    all_dates,
                    st.session_state["walk_fetch_nonce"],
                )
            st.session_state["walkforward_meta"] = meta
            succeeded = meta.get("succeeded", [])
            failed = meta.get("failed", [])
            if dataset:
                st.success(f"Fetched {len(dataset)} daily chains for {ticker} (failed: {len(failed)} days).")
            else:
                if failed:
                    fail_dates = ", ".join(entry["date"] for entry in failed[:5])
                    st.warning(f"No chains fetched. Yahoo Finance failed for: {fail_dates}...")
                else:
                    st.warning("No chains fetched for the requested window.")

    if dataset:
        st.session_state["walkforward_chains"] = dataset

    stored = st.session_state.get("walkforward_chains", {})
    if not stored:
        st.info(
            "Upload Parquet files (e.g., produced via `python tools/download_spy_chains.py --ticker SPY ...`) "
            "or download a short window via Yahoo Finance to continue."
        )
        return

    wf_meta = st.session_state.get("walkforward_meta", {})
    if wf_meta:
        st.caption(
            f"Last load: {len(wf_meta.get('succeeded', []))} days succeeded / {len(wf_meta.get('failed', []))} failed."
        )
        failed = wf_meta.get("failed", [])
        if failed:
            with st.expander("Show failed download dates"):
                st.write(pd.DataFrame(failed))

    filt_col1, filt_col2, filt_col3 = st.columns(3)
    min_volume = filt_col1.number_input("Min volume", min_value=0, value=25, step=5, key="wf_minvol")
    max_spread = filt_col2.number_input("Max rel spread", min_value=0.0, value=0.5, step=0.05, key="wf_maxspread")
    min_mid = filt_col3.number_input("Min mid", min_value=0.0, value=0.10, step=0.05, key="wf_minmid")

    prepared: Dict[pd.Timestamp, pd.DataFrame] = {}
    for trade_date, df in stored.items():
        filtered = _apply_filters(df, int(min_volume), float(max_spread), float(min_mid))
        if not filtered.empty:
            prepared[pd.to_datetime(trade_date).tz_localize(None)] = filtered

    if len(prepared) < 3:
        st.warning("Need at least three prepared days to run walk-forward.")
        return
    st.caption(f"{len(prepared)} of {len(stored)} uploaded/downloaded days survived the liquidity filters.")

    dates_sorted = sorted(prepared.keys())
    # walk_forward_test needs window + 2 days: the window, the day it
    # calibrates on and the next day it is scored against.
    max_window = len(dates_sorted) - 2
    min_window = min(3, max_window)
    if min_window == max_window:
        window = max_window
        st.caption(f"Calibration window: {window} day(s). Load more days to widen it.")
    else:
        window = st.slider(
            "Calibration window (days)",
            min_value=min_window,
            max_value=max_window,
            value=min(10, max_window),
        )
    run = st.button("Run walk-forward test", width="stretch")
    if run:
        with st.spinner("Running walk-forward backtest…"):
            metrics = walk_forward_test(prepared, window=window)
        if metrics.empty:
            st.warning("Walk-forward returned no metrics. Ensure overlapping expiries across days.")
        else:
            regimes = {date: _infer_regime(prepared[date]) for date in prepared}
            metrics["regime"] = metrics["trade_date"].map(regimes)
            st.session_state["walkforward_metrics"] = metrics

    metrics = st.session_state.get("walkforward_metrics", pd.DataFrame())
    if metrics.empty:
        return

    st.plotly_chart(_plot_walkforward(metrics), width="stretch", config=PLOTLY_2D_CONFIG)
    st.dataframe(metrics, width="stretch")

    regime_summary = (
        metrics.dropna(subset=["regime"])
        .groupby("regime")
        .agg(mean_rmse=("rmse", "mean"), mean_hit_rate=("hit_rate", "mean"), observations=("regime", "count"))
    )
    if not regime_summary.empty:
        st.markdown("#### / REGIME BREAKDOWN")
        st.dataframe(regime_summary, width="stretch")


def _render_model_comparison_tab():
    st.markdown("### / MODEL COMPARISON")
    chain = st.session_state.get("market_chain", pd.DataFrame())
    if chain.empty:
        st.info("Load a chain on the Market Data tab to compare models.")
        return

    expiries = sorted(pd.to_datetime(chain["expiry"]).dt.date.unique().tolist())
    expiry = st.selectbox("Expiry", options=expiries)
    models = st.multiselect(
        "Models",
        options=["svi", "sabr", "local_vol"],
        default=["svi", "sabr", "local_vol"],
    )
    if not models:
        st.warning("Select at least one model.")
        return

    run = st.button("Compare models", width="stretch")
    if run:
        slice_df = chain[pd.to_datetime(chain["expiry"]).dt.date == expiry]
        if slice_df.empty:
            st.warning("Selected expiry has no quotes after filtering.")
        else:
            metrics = compare_models(slice_df, models=models)
            st.session_state["model_comp_metrics"] = metrics
            st.session_state["model_comp_slice"] = slice_df
            st.session_state["model_comp_models"] = models

    metrics = st.session_state.get("model_comp_metrics", pd.DataFrame())
    if metrics.empty:
        return

    st.dataframe(metrics, width="stretch")
    slice_df = st.session_state.get("model_comp_slice", pd.DataFrame())
    models = st.session_state.get("model_comp_models", [])
    if not slice_df.empty:
        st.plotly_chart(_plot_model_comparison(slice_df, models), width="stretch", config=PLOTLY_2D_CONFIG)


def render() -> None:
    """Draw the ANALYTICS lab."""

    init_state()
    st.markdown("## / VOL ANALYTICS LAB")
    tab_labels = [f"/ {label.upper()}" for label in ANALYTICS_TABS]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        _render_market_data_tab()
    with tabs[1]:
        _render_calibration_tab()
    with tabs[2]:
        _render_walkforward_tab()
    with tabs[3]:
        _render_model_comparison_tab()

