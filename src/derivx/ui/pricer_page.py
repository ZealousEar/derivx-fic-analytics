"""EQUITY and FX option pricer page: premium, Greeks, price surface and smile."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from derivx.core.bs import bs_price_greeks
from derivx.ui.plot_limits import GUARD_AVAILABLE, axis_limits, bound_axes, bound_scene
from derivx.ui.theme import PLOTLY_2D_CONFIG
from derivx.vol.sabr import calibrate_sabr_smile, hagan_sabr_iv
from derivx.vol.svi import fit_svi_strike_slice, svi_iv

DEFAULT_CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.2))
# Zoom and pan are bounded by derivx.ui.plot_limits. Wheel zoom (pinch or
# Ctrl/Cmd + wheel) and 3D pan need its in-page guard, so both are off without
# it. The 3D zoom tool only moves the camera, which an orthographic view does
# not show, so it is dropped.
ZOOM_HINT = "Pinch or Ctrl/⌘ + scroll to zoom (bounded) · home icon resets the view"
PLOTLY_3D_CONFIG = {
    "displaylogo": False,
    "scrollZoom": GUARD_AVAILABLE,
    "modeBarButtonsToRemove": [
        "toImage",
        "lasso3d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "zoom3d",
    ] + ([] if GUARD_AVAILABLE else ["pan3d"]),
    "modeBarButtonsToAdd": ["orbitRotation", "resetCameraDefault3d"]
    + (["pan3d"] if GUARD_AVAILABLE else []),
}


def render(mode: str) -> None:
    """Draw the EQUITY or FX pricer page."""

    if "preset_last" not in st.session_state:
        st.session_state["preset_last"] = "DEFAULT"
        st.session_state["force_camera"] = True
    # Monochrome with strategic red accents (Off-White inspired)
    accent_color = "#FF4444"  # Red accent for key elements

    st.markdown(f"## / {mode.upper()} OPTIONS")

    # Input parameters in sidebar - minimalist labels
    st.sidebar.markdown(f"### / {mode.upper()} PARAMS")

    # Quick presets for dramatic visualization
    preset = st.sidebar.radio(
        "PRESET", ["DEFAULT", "HIGH VOL", "DEEP ITM", "FAR OTM"], horizontal=True, key="preset_choice"
    )
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
    bound_scene(fig_3d, eye=DEFAULT_CAMERA["eye"])
    if st.session_state.get("force_camera", False):
        st.session_state["force_camera"] = False

    st.plotly_chart(fig_3d, width="stretch", key="surface_3d", config=PLOTLY_3D_CONFIG)
    if GUARD_AVAILABLE:
        st.caption(ZOOM_HINT)

    # Volatility Smile - minimalist line chart
    model_name = "SVI" if mode == "EQUITY" else "SABR"
    st.markdown(f"### / {model_name} SMILE")
    if T <= 0 or S <= 0:
        st.warning("The smile needs MATURITY.T > 0 and SPOT.S > 0 (SVI and SABR are fitted in total variance).")
        return
    Ks = np.linspace(max(1e-6, K - 40), K + 40, 20)
    ivs = np.full_like(Ks, sigma) + 0.01 * np.random.randn(len(Ks))

    if mode == "EQUITY":
        params = fit_svi_strike_slice(Ks, ivs, F=S, T=T)
        iv_fit = np.asarray(svi_iv(Ks, params, F=S, T=T))
    else:
        params = calibrate_sabr_smile(Ks, ivs, T=T, F=S, beta=0.5)
        iv_fit = np.asarray(hagan_sabr_iv(S, Ks, T, **params))

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
    bound_axes(
        fig_smile,
        xaxis=axis_limits(Ks, floor=0.0),
        yaxis=axis_limits(np.concatenate([ivs, iv_fit, [sigma]]), floor=0.0),
    )
    st.plotly_chart(fig_smile, width="stretch", key="smile_2d", config=PLOTLY_2D_CONFIG)
    if GUARD_AVAILABLE:
        st.caption(ZOOM_HINT)

    # Model parameters - always visible below smile
    st.markdown(f"**[ {model_name} CALIBRATED PARAMS ]**")
    param_cols = st.columns(len(params))
    for i, (name, val) in enumerate(params.items()):
        with param_cols[i]:
            st.metric(name.upper(), f"{val:.4f}")

