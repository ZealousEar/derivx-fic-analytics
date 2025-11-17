from __future__ import annotations

import datetime as dt
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline

from derivx.calibration.production import CalibratedSlice, calibrate_with_constraints
from derivx.core.bs import bs_price_greeks, put_call_parity_residual
from derivx.data.market_data import apply_liquidity_filters, compute_moneyness, fetch_option_chain
from derivx.models.comparison import compare_models
from derivx.rates.black_ir import price_cap_floor
from derivx.rates.term_structure import build_zero_curve
from derivx.risk.metrics import scen_shift_curve
from derivx.validation.backtest import walk_forward_test
from derivx.vol.sabr import calibrate_sabr_smile, hagan_sabr_iv
from derivx.vol.svi import fit_svi_strike_slice, svi_iv


st.set_page_config(page_title="DERIVX / FIC ANALYTICS", layout="wide", initial_sidebar_state="expanded")

# Virgil Abloh-inspired: minimalist, industrial, high-contrast
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    
    .stApp {
        background: #000000;
        color: #FFFFFF;
    }
    
    .stSidebar {
        background: #000000 !important;
        border-right: 2px solid #FFFFFF;
    }
    
    .stSidebar [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }
    
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 3px solid #FFFFFF;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        border-bottom: 2px solid #CCCCCC;
        font-size: 1.5rem;
    }
    
    h3 {
        border-bottom: 1px solid #999999;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    .stMetric {
        background: #FFFFFF;
        color: #000000;
        padding: 20px;
        border: 3px solid #000000;
        border-radius: 0;
        box-shadow: 5px 5px 0px #000000;
    }
    
    .stMetric label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    .stButton button {
        background: #000000;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
        border-radius: 0;
        padding: 0.75rem 2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: #FFFFFF;
        color: #000000;
        border: 2px solid #000000;
    }
    
    [data-testid="stHorizontalBlock"] {
        gap: 1.5rem;
    }
    
    /* Industrial brutalist inputs */
    input, select {
        border: 2px solid #FFFFFF !important;
        border-radius: 0 !important;
        background: #000000 !important;
        color: #FFFFFF !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Quote marks / design elements */
    .quote-mark {
        font-size: 3rem;
        font-weight: 700;
        color: #FF4444;
        line-height: 0.8;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# Minimalist mode selector - architectural, no emojis
st.sidebar.markdown("### / NAVIGATION")
mode = st.sidebar.radio(
    "MODE",
    ["[ EQUITY ]", "[ FX ]", "[ RATES ]", "[ ANALYTICS ]"],
    format_func=lambda x: x,
    label_visibility="collapsed"
)
mode = mode.replace("[", "").replace("]", "").strip()  # Extract mode name

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"
DEFAULT_CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.2))
PLOTLY_3D_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": [
        "toImage",
        "lasso3d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
    "modeBarButtonsToAdd": ["orbitRotation", "zoom3d", "pan3d", "resetCameraDefault3d"],
}
PLOTLY_2D_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": [
        "toImage",
        "lasso2d",
        "select2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}
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
        if st.button("Pull from Yahoo Finance", use_container_width=True):
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

    st.dataframe(filtered, use_container_width=True)
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

    run = st.button("Calibrate slices", use_container_width=True)
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    options = [f"{slc.expiry.date()}" for slc in results]
    selected = st.selectbox("Visualize expiry", options) if options else None
    if selected:
        slc = next(s for s in results if str(s.expiry.date()) == selected)
        st.plotly_chart(_plot_calibration_smile(slc), use_container_width=True, config=PLOTLY_2D_CONFIG)
        params_df = pd.DataFrame([slc.params])
        st.markdown("#### / CALIBRATED PARAMS")
        st.dataframe(params_df, use_container_width=True)
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
        if st.button("Download history", use_container_width=True):
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
    window = st.slider(
        "Calibration window (days)",
        min_value=3,
        max_value=max(3, len(dates_sorted) - 1),
        value=min(10, max(3, len(dates_sorted) - 1)),
    )
    run = st.button("Run walk-forward test", use_container_width=True)
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

    st.plotly_chart(_plot_walkforward(metrics), use_container_width=True, config=PLOTLY_2D_CONFIG)
    st.dataframe(metrics, use_container_width=True)

    regime_summary = (
        metrics.dropna(subset=["regime"])
        .groupby("regime")
        .agg(mean_rmse=("rmse", "mean"), mean_hit_rate=("hit_rate", "mean"), observations=("regime", "count"))
    )
    if not regime_summary.empty:
        st.markdown("#### / REGIME BREAKDOWN")
        st.dataframe(regime_summary, use_container_width=True)


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

    run = st.button("Compare models", use_container_width=True)
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

    st.dataframe(metrics, use_container_width=True)
    slice_df = st.session_state.get("model_comp_slice", pd.DataFrame())
    models = st.session_state.get("model_comp_models", [])
    if not slice_df.empty:
        st.plotly_chart(_plot_model_comparison(slice_df, models), use_container_width=True, config=PLOTLY_2D_CONFIG)

if "preset_last" not in st.session_state:
    st.session_state["preset_last"] = "DEFAULT"
    st.session_state["force_camera"] = True


# Header - fixed banner position
st.markdown("""
    <div style='background: #000000; border-top: 4px solid #FF4444; 
    padding: 1.5rem 0 1rem 0; margin: -1rem 0 3rem 0;'>
        <h1 style='margin: 0; font-size: 3rem; border: none !important; letter-spacing: 0.1em; font-weight: 700;'>
            DERIVX
        </h1>
        <div style='margin-top: 0.5rem; display: flex; gap: 1rem; align-items: center;'>
            <span style='font-size: 0.75rem; letter-spacing: 0.2em; font-weight: 600; opacity: 0.8;'>
                FIC DERIVATIVES ANALYTICS
            </span>
            <span style='background: #FF4444; color: #000000; padding: 0.2rem 0.5rem; 
            font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;'>
                v2.0
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

if mode in ("EQUITY", "FX"):
    # Monochrome with strategic red accents (Off-White inspired)
    accent_color = "#FF4444"  # Red accent for key elements
    
    st.markdown(f"## / {mode.upper()} OPTIONS")
    
    # Input parameters in sidebar - minimalist labels
    st.sidebar.markdown(f"### / {mode.upper()} PARAMS")
    
    # Quick presets for dramatic visualization
    preset = st.sidebar.radio("PRESET", ["DEFAULT", "HIGH VOL", "DEEP ITM", "FAR OTM"], horizontal=True, key="preset_choice")
    if preset != st.session_state.get("preset_last", "DEFAULT"):
        st.session_state["preset_last"] = preset
        st.session_state["force_camera"] = True

    if preset == "DEFAULT":
        S_def, K_def, T_def, r_def, q_def, sigma_def = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    elif preset == "HIGH VOL":
        S_def, K_def, T_def, r_def, q_def, sigma_def = 100.0, 100.0, 2.0, 0.05, 0.0, 0.6
    elif preset == "DEEP ITM":
        S_def, K_def, T_def, r_def, q_def, sigma_def = 150.0, 80.0, 1.0, 0.05, 0.0, 0.3
    else:  # FAR OTM
        S_def, K_def, T_def, r_def, q_def, sigma_def = 80.0, 120.0, 0.5, 0.05, 0.0, 0.4
    
    S = st.sidebar.number_input("SPOT.S", value=S_def, key="spot")
    K = st.sidebar.number_input("STRIKE.K", value=K_def, key="strike")
    T = st.sidebar.number_input("MATURITY.T", value=T_def, min_value=0.0, step=0.1, key="maturity")
    r = st.sidebar.number_input("RATE.R", value=r_def, min_value=-0.05, step=0.01, key="rate", format="%.4f")
    q = st.sidebar.number_input("YIELD.Q", value=q_def, step=0.01, key="div", format="%.4f")
    sigma = st.sidebar.number_input("VOL.σ", value=sigma_def, min_value=0.0, step=0.01, key="vol", format="%.4f")
    call = st.sidebar.selectbox("TYPE", ["CALL", "PUT"], key="opttype") == "CALL"

    price, delta, gamma, vega, theta, rho = bs_price_greeks(S, K, T, r, sigma, call=call, q=q)
    
    # Hero metric - brutalist card
    st.markdown(f"""
        <div style='background: #FFFFFF; color: #000000; padding: 3rem 2rem; 
        border: 4px solid #000000; margin-bottom: 2rem; position: relative;'>
            <div style='position: absolute; top: -15px; left: 20px; background: {accent_color}; 
            color: #FFFFFF; padding: 5px 15px; font-weight: 700; font-size: 0.75rem; 
            letter-spacing: 0.1em;'>
                {'CALL' if call else 'PUT'} / PREMIUM
            </div>
            <div style='text-align: center; font-family: "IBM Plex Mono", monospace;'>
                <div style='font-size: 4.5rem; font-weight: 700; line-height: 1;'>
                    ${float(price):.4f}
                </div>
                <div style='margin-top: 1rem; font-size: 0.9rem; letter-spacing: 0.1em; opacity: 0.7;'>
                    STRIKE {K} × TENOR {T}Y
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Greeks - minimalist metric cards
    st.markdown("### / GREEKS")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    greeks_data = [
        ("DELTA", "Δ", float(delta), col1),
        ("GAMMA", "Γ", float(gamma), col2),
        ("VEGA", "ν", float(vega)*0.01, col3),
        ("THETA", "Θ", float(theta)/365.0, col4),
        ("RHO", "ρ", float(rho)*0.01, col5),
    ]
    
    for name, symbol, value, col in greeks_data:
        with col:
            st.metric(f"{name}", f"{value:.4f}")

    # Parity check with explicit values
    c_price, *_ = bs_price_greeks(S, K, T, r, sigma, call=True, q=q)
    p_price, *_ = bs_price_greeks(S, K, T, r, sigma, call=False, q=q)
    lhs = float(c_price - p_price)
    rhs = float(S * np.exp(-q * T) - K * np.exp(-r * T))
    res = lhs - rhs
    parity_status = "[OK]" if abs(res) < 1e-8 else "[!]"
    st.markdown(
        f"""
        <div style='background: #FFFFFF; color: #000000; padding: 1rem; border: 2px solid #000000; margin: 1.5rem 0;'>
            <strong>PUT-CALL PARITY {parity_status}</strong><br/>
            C−P = <code>{lhs:.6f}</code> &nbsp;&nbsp; S·e<sup>−qT</sup>−K·e<sup>−rT</sup> = <code>{rhs:.6f}</code><br/>
            Residual: <code>{abs(res):.3e}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3D Surface Plot - with fixed axes and practical color gradient
    st.markdown("### / PRICE SURFACE")
    gridN = st.slider("RESOLUTION", 15, 50, 25, key="grid")
    
    def _range(center: float, pct: float = 0.35, min_width: float = 20.0) -> list[float]:
        span = max(abs(center) * pct, min_width / 2.0)
        return [max(1e-6, center - span), center + span]

    S_range = _range(S)
    K_range = _range(K)
    S_grid = np.linspace(S_range[0], S_range[1], gridN)
    K_grid = np.linspace(K_range[0], K_range[1], gridN)
    Ss, Ks = np.meshgrid(S_grid, K_grid)
    P, *_ = bs_price_greeks(Ss, Ks, T, r, sigma, call=True, q=q)
    
    price_min = float(np.nanmin(P))
    price_max = float(np.nanmax(P))
    if not np.isfinite(price_min):
        price_min = 0.0
    if not np.isfinite(price_max):
        price_max = max(1e-3, float(price))
    pad = max((price_max - price_min) * 0.1, 1e-3)
    z_range = [max(0.0, price_min - pad), price_max + pad]

    fig_3d = go.Figure(data=[
        go.Surface(
            z=P,
            x=S_grid,
            y=K_grid,
        colorscale=[
                [0.0, '#00113D'],   # deep navy
                [0.1, '#003B82'],   # cobalt
                [0.25, '#0E7BCF'],  # azure
                [0.45, '#35C1FF'],  # cyan
                [0.6, '#7FFFD4'],   # aqua
                [0.75, '#FFE66D'],  # soft yellow
                [0.9, '#FF9E4A'],   # orange
                [1.0, '#E5243B'],   # scarlet
            ],
            hovertemplate="Spot: %{x:.2f}<br>Strike: %{y:.2f}<br>Value: %{z:.4f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color='#FFFFFF', family='IBM Plex Mono', size=10),
            title=dict(text="VALUE", font=dict(color='#FFFFFF', size=12)),
            len=0.7,
            thickness=15,
        ),
        contours={
            "z": {"show": True, "highlightcolor": "#FFFFFF", "project": {"z": True}, "width": 2}
        },
        lighting=dict(ambient=0.9, diffuse=0.7, specular=0.3, roughness=0.5)
        )
    ])

    scene_settings = dict(
        bgcolor='#000000',
        dragmode="turntable",
        xaxis=dict(
            title=dict(text='SPOT (S)', font=dict(color='#FFFFFF')),
            backgroundcolor="#000000",
            gridcolor="#333333",
            showbackground=True,
            range=S_range,
        ),
        yaxis=dict(
            title=dict(text='STRIKE (K)', font=dict(color='#FFFFFF')),
            backgroundcolor="#000000",
            gridcolor="#333333",
            showbackground=True,
            range=K_range,
        ),
        zaxis=dict(
            title=dict(text='VALUE', font=dict(color='#FFFFFF')),
            backgroundcolor="#000000",
            gridcolor="#333333",
            showbackground=True,
            range=z_range,
        ),
    )
    surface_revision = f"{S:.4f}_{K:.4f}_{T:.4f}_{r:.4f}_{sigma:.4f}_{int(call)}"
    if st.session_state.get("force_camera", False):
        scene_settings["camera"] = {
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
            "eye": DEFAULT_CAMERA["eye"],
            "projection": {"type": "orthographic"},
        }

    fig_3d.update_layout(
        scene=scene_settings,
        paper_bgcolor='#000000',
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color='#FFFFFF', family='Helvetica'),
        uirevision=surface_revision,
    )
    fig_3d.update_scenes(camera_projection_type="orthographic")
    if st.session_state.get("force_camera", False):
        st.session_state["force_camera"] = False

    st.plotly_chart(fig_3d, use_container_width=True, key="surface_3d", config=PLOTLY_3D_CONFIG)

    # Volatility Smile - minimalist line chart
    model_name = "SVI" if mode == "EQUITY" else "SABR"
    st.markdown(f"### / {model_name} SMILE")
    Ks = np.linspace(max(1e-6, K - 40), K + 40, 20)
    ivs = np.full_like(Ks, sigma) + 0.01 * np.random.randn(len(Ks))
    
    if mode == "EQUITY":
        params = fit_svi_strike_slice(Ks, ivs, F=S)
        iv_fit = np.array([svi_iv(k, params, F=S) for k in Ks])
    else:
        params = calibrate_sabr_smile(Ks, ivs, T=max(T, 1e-4), F=S, beta=0.5)
        iv_fit = np.array([hagan_sabr_iv(S, k, T, **params) for k in Ks])
    
    fig_smile = go.Figure()
    
    # Market observations
    fig_smile.add_trace(go.Scatter(
        x=Ks, y=ivs,
        mode='markers',
        name='MARKET',
        marker=dict(size=8, color='#FFFFFF', line=dict(width=2, color='#000000'))
    ))
    
    # Model fit
    fig_smile.add_trace(go.Scatter(
        x=Ks, y=iv_fit,
        mode='lines',
        name=f'{model_name}',
        line=dict(color=accent_color, width=3)
    ))
    
    # ATM strike
    fig_smile.add_trace(go.Scatter(
        x=[K], y=[sigma],
        mode='markers',
        name='ATM',
        marker=dict(size=12, color=accent_color, symbol='x', line=dict(width=3, color='#FFFFFF'))
    ))
    
    smile_revision = f"{mode}_{S:.4f}_{K:.4f}_{T:.4f}_{sigma:.4f}_{int(call)}"
    fig_smile.update_layout(
        xaxis_title='STRIKE',
        yaxis_title='IMPLIED VOL',
        hovermode='x unified',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#FFFFFF', family='Helvetica'),
        xaxis=dict(gridcolor='#333333', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#333333', showgrid=True, zeroline=False),
        height=400,
        margin=dict(l=60, r=20, t=30, b=60),
        legend=dict(bgcolor='#000000', bordercolor='#FFFFFF', borderwidth=1, 
                   font=dict(family='IBM Plex Mono', color='#FFFFFF')),
        dragmode="zoom",
        uirevision=smile_revision,
    )
    fig_smile.update_xaxes(autorange=True)
    fig_smile.update_yaxes(autorange=True)
    st.plotly_chart(fig_smile, use_container_width=True, config=PLOTLY_2D_CONFIG)
    
    # Model parameters - always visible below smile
    st.markdown(f"**[ {model_name} CALIBRATED PARAMS ]**")
    param_cols = st.columns(len(params))
    for i, (name, val) in enumerate(params.items()):
        with param_cols[i]:
            st.metric(name.upper(), f"{val:.4f}")

elif mode == "RATES":
    st.markdown("## / INTEREST RATES")
    
    accent_color = "#FF4444"
    
    # Sidebar inputs
    st.sidebar.markdown("### / RATES PARAMS")
    uploaded = st.sidebar.file_uploader("CSV.DATA", type=["csv"])
    if uploaded is not None:
        df_rates = pd.read_csv(uploaded)
    else:
        df_rates = pd.read_csv(DATA_DIR / "rates_quotes.csv")
    quotes = {row["tenor"]: float(row["rate"]) for _, row in df_rates.iterrows()}

    curve = build_zero_curve(quotes)
    
    # Curve visualization - stark monochrome
    st.markdown("### / ZERO CURVE")
    fig_curve = make_subplots(
        rows=1, cols=2,
        subplot_titles=("DISCOUNT FACTORS", "FORWARD RATES"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Discount factors
    fig_curve.add_trace(go.Scatter(
        x=curve.ts, y=curve.dfs,
        mode='lines+markers',
        name='DF',
        line=dict(color='#FFFFFF', width=2),
        marker=dict(size=6, color='#FFFFFF', line=dict(width=1, color='#000000')),
    ), row=1, col=1)
    
    # Forward rates
    fwd_rates = [curve.fwd_rate(max(0, t-0.5), t) for t in curve.ts if t > 0]
    fig_curve.add_trace(go.Scatter(
        x=curve.ts[curve.ts > 0], y=fwd_rates,
        mode='lines+markers',
        name='FWD',
        line=dict(color=accent_color, width=2),
        marker=dict(size=6, color=accent_color, line=dict(width=1, color='#FFFFFF')),
    ), row=1, col=2)
    
    fig_curve.update_xaxes(title_text="TENOR", gridcolor='#333333', row=1, col=1)
    fig_curve.update_xaxes(title_text="TENOR", gridcolor='#333333', row=1, col=2)
    fig_curve.update_yaxes(title_text="DF", gridcolor='#333333', row=1, col=1)
    fig_curve.update_yaxes(title_text="RATE", gridcolor='#333333', row=1, col=2)
    
    fig_curve.update_layout(
        height=350,
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#FFFFFF', family='Helvetica'),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    st.plotly_chart(fig_curve, use_container_width=True, config=PLOTLY_2D_CONFIG)

    # Cap/Floor pricing
    st.markdown("### / CAP PRICING")
    col1, col2 = st.columns(2)
    with col1:
        K = st.number_input("STRIKE.K", value=0.03, step=0.001, format="%.4f", key="cap_strike")
    with col2:
        flat_sigma = st.number_input("VOL.σ", value=0.2, step=0.01, format="%.4f", key="cap_vol")
    
    max_t = float(curve.ts.max())
    tenors = np.linspace(0.5, max(0.5, max_t), num=6)
    vol_curve = type(curve)(ts=curve.ts, dfs=np.full_like(curve.dfs, flat_sigma))
    res = price_cap_floor(curve, K, tenors, vol_curve)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PV", f"{res['PV']:.6f}")
    with col2:
        st.metric("DV01", f"{res['DV01']:.6f}")
    with col3:
        st.metric("PV01", f"{res['PV01']:.6f}")

    # Buckets bar chart - minimalist
    st.markdown("### / BUCKETS")
    bucket_tenors = list(res['buckets'].keys())
    bucket_values = list(res['buckets'].values())
    
    fig_buckets = go.Figure(go.Bar(
        x=[f"{t:.1f}Y" for t in bucket_tenors],
        y=bucket_values,
        text=[f"{v:.5f}" for v in bucket_values],
        textposition='outside',
        marker=dict(color='#FFFFFF', line=dict(color='#000000', width=2)),
    ))
    fig_buckets.update_layout(
        yaxis_title="VALUE",
        xaxis_title="TENOR",
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#FFFFFF', family='IBM Plex Mono'),
        height=300,
        margin=dict(l=50, r=50, t=30, b=50),
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='#333333'),
    )
    st.plotly_chart(fig_buckets, use_container_width=True, config=PLOTLY_2D_CONFIG)

    # Scenario analysis
    st.markdown("### / SCENARIOS")
    col1, col2 = st.columns(2)
    with col1:
        par = st.slider("PARALLEL (BP)", -100, 100, 0, key="parallel")
    with col2:
        tw = st.slider("TWIST (BP)", -100, 100, 0, key="twist")
    
    # Compute scenario PV
    shocked_curve = scen_shift_curve(curve, parallel_bp=par, twist_bp=tw)
    res_shock = price_cap_floor(shocked_curve, K, tenors, vol_curve)
    pnl = res_shock['PV'] - res['PV']
    
    # Scenario P&L - stark display
    st.markdown(f"""
        <div style='background: #FFFFFF; color: #000000; padding: 2rem; border: 3px solid #000000; margin: 1.5rem 0;'>
            <div style='text-align: center; font-family: "IBM Plex Mono", monospace;'>
                <div style='font-size: 0.85rem; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 0.5rem;'>
                    P&L / SCENARIO
                </div>
                <div style='font-size: 3rem; font-weight: 700; color: {"#FF4444" if pnl < 0 else "#000000"}; line-height: 1;'>
                    {"" if pnl < 0 else "+"}{pnl:.6f}
                </div>
                <div style='margin-top: 1rem; font-size: 0.8rem; opacity: 0.7;'>
                    BASE {res['PV']:.6f} → SHOCKED {res_shock['PV']:.6f}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
elif mode == "ANALYTICS":
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
