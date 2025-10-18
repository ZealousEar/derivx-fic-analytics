from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from derivx.core.bs import bs_price_greeks, put_call_parity_residual
from derivx.vol.svi import fit_svi_strike_slice, svi_iv
from derivx.vol.sabr import calibrate_sabr_smile, hagan_sabr_iv
from derivx.rates.term_structure import build_zero_curve
from derivx.rates.black_ir import price_cap_floor
from derivx.risk.metrics import scen_shift_curve


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
    "",
    ["[ EQUITY ]", "[ FX ]", "[ RATES ]"],
    format_func=lambda x: x,
    label_visibility="collapsed"
)
mode = mode.replace("[", "").replace("]", "").strip()  # Extract mode name

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"
DEFAULT_CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.2))
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
        "toImage",
        "lasso2d",
        "select2d",
    ],
    "modeBarButtonsToAdd": ["orbitRotation", "zoom3d", "pan3d", "resetCameraDefault3d"],
}

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
                v1.0
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
    
    # Fixed ranges independent of grid resolution
    S_range = [max(1e-6, S - 50), S + 50]
    K_range = [max(1e-6, K - 50), K + 50]
    S_grid = np.linspace(S_range[0], S_range[1], gridN)
    K_grid = np.linspace(K_range[0], K_range[1], gridN)
    Ss, Ks = np.meshgrid(S_grid, K_grid)
    P, *_ = bs_price_greeks(Ss, Ks, T, r, sigma, call=True, q=q)
    
    # Fixed Z range based on typical price range at current params
    P_center, *_ = bs_price_greeks(S, K, T, r, sigma, call=True, q=q)
    z_range = [0.0, max(1e-6, float(P_center) * 3.0)]

    fig_3d = go.Figure(data=[go.Surface(
        z=P, x=S_grid, y=K_grid,
        colorscale=[
            [0.0, '#0D1B2A'], [0.15, '#1B263B'], [0.3, '#415A77'],
            [0.45, '#778DA9'], [0.55, '#E0E1DD'], [0.65, '#FFB347'],
            [0.75, '#FF8C42'], [0.85, '#FF4444'], [0.95, '#CC0000'], [1.0, '#990000']
        ],
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
    )])

    scene_settings = dict(
        bgcolor='#000000',
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
    if st.session_state.get("force_camera", False):
        scene_settings["camera"] = DEFAULT_CAMERA

    fig_3d.update_layout(
        scene=scene_settings,
        paper_bgcolor='#000000',
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color='#FFFFFF', family='Helvetica'),
        uirevision='constant',
    )
    if st.session_state.get("force_camera", False):
        st.session_state["force_camera"] = False

    st.plotly_chart(fig_3d, use_container_width=True, key="surface_3d", config=PLOTLY_CONFIG)

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
    )
    st.plotly_chart(fig_smile, use_container_width=True, config=PLOTLY_CONFIG)
    
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
    st.plotly_chart(fig_curve, use_container_width=True, config=PLOTLY_CONFIG)

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
    st.plotly_chart(fig_buckets, use_container_width=True, config=PLOTLY_CONFIG)

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


