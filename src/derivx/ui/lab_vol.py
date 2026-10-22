"""ANALYTICS lab, vol research (VM-18): the smile risk of a calibrated
expiry, the arbitrage-free SSVI surface with sigma(K, T) at any expiry and
Dupire local vol, and the daily SVI / SABR parameter history with jumps.

* Smile risk (on the Production Calibration tab): for an option on a
  fitted SVI smile, its smile vol, premium, Black vega, sticky-strike and
  sticky-moneyness deltas, SVI parameter vegas, SABR (beta 1) parameter
  vegas with Hagan and Bartlett deltas, and vega by quote bucket through
  the SVI refit (:mod:`derivx.vol.smile_risk`).
* Vol surface tab: a power-law SSVI surface fitted to every usable expiry
  at once (Gatheral-Jacquier no-arbitrage conditions by construction,
  :mod:`derivx.vol.ssvi`), its arbitrage report, the smile at any expiry
  (total variance linear in T between fitted expiries) and the implied and
  Dupire local vol surfaces.
* Parameter stability (on the Walk-Forward tab):
  :mod:`derivx.validation.stability` on the loaded history.

Units: strikes and premiums per share in the underlying's currency; vols
per-annum decimals (charts in %); vegas per +1 vol point; deltas per unit
of forward (Black-76, discounted); ``k = ln(K/F)``; ``T`` ACT/365F years.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.calibration.production import CalibratedSlice, clean_chain
from derivx.ui.lab_common import downloads, fingerprint, legend, stale_notice, stats
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.state import PERSIST
from derivx.ui.theme import ACCENT, PLOTLY_2D_CONFIG, SEQUENTIAL_COLORS, SERIES, dark_layout
from derivx.validation.manifest import build_manifest
from derivx.validation.stability import PARAMETER_ORDER, parameter_history, parameter_jumps
from derivx.vol.discounting import Discounting
from derivx.vol.implied import black_price, black_vega
from derivx.vol.quotes import apply_std_band
from derivx.vol.sabr import calibrate_sabr_smile
from derivx.vol.smile_risk import (
    sabr_deltas,
    sabr_parameter_vegas,
    smile_deltas,
    svi_atm_skew_curvature,
    svi_bucket_vegas,
    svi_parameter_vegas,
)
from derivx.vol.ssvi import fit_ssvi
from derivx.vol.svi import svi_derivatives, svi_iv

K_SURF = "lab_ssvi_surface"
K_STAB = "lab_param_stability"


# -- smile risk -----------------------------------------------------------------


@st.cache_data(max_entries=32, show_spinner=False)
def smile_risk(
    strikes: Tuple[float, ...],
    vols: Tuple[float, ...],
    weights: Tuple[float, ...],
    loss: str,
    f_scale: float,
    F: float,
    T: float,
    df: float,
    params: Tuple[Tuple[str, float], ...],
    K: float,
    is_call: bool,
) -> Dict[str, Any]:
    """Every smile-risk number of one option on one fitted SVI slice
    (content-keyed: the same quotes, fit and option give the same result in
    any session)."""

    p = dict(params)
    vol = float(svi_iv(K, p, F=F, T=T))
    k = np.log(K / F)
    _, w1, _ = (float(x) for x in svi_derivatives(k, p))
    dvol_dk = w1 / (2.0 * vol * T)
    d = smile_deltas(F, K, T, vol, dvol_dk, df, is_call)
    out: Dict[str, Any] = {
        "vol": vol,
        "price": float(black_price(F, K, T, vol, is_call, df)),
        "vega": 0.01 * float(black_vega(F, K, T, vol, df)),
        "delta_sticky_strike": float(d["sticky_strike"]),
        "delta_sticky_moneyness": float(d["sticky_moneyness"]),
        "shape": svi_atm_skew_curvature(p, T),
        "svi_vegas": svi_parameter_vegas(K, F, T, p, df, is_call),
        "buckets": svi_bucket_vegas(K, is_call, F, T, df, p, strikes, vols, weights, loss, f_scale),
    }
    try:
        sabr = calibrate_sabr_smile(np.asarray(strikes), np.asarray(vols), T=T, F=F, beta=1.0)
    except ValueError as exc:
        out["sabr_error"] = str(exc)
    else:
        sd = sabr_deltas(F, K, T, sabr.alpha, 1.0, sabr.rho, sabr.nu, df=df, is_call=is_call)
        out["sabr"] = {
            "alpha": sabr.alpha,
            "rho": sabr.rho,
            "nu": sabr.nu,
            "rmse_bp": sabr.rmse_bp,
            "delta_hagan": float(sd["hagan"]),
            "delta_bartlett": float(sd["bartlett"]),
        }
        out["sabr_vegas"] = sabr_parameter_vegas(F, K, T, sabr.alpha, 1.0, sabr.rho, sabr.nu, df=df, is_call=is_call)
    return out


def _bucket_chart(buckets: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(meta={"chart": "lab_bucket_vega"})
    v = buckets["vega"].to_numpy(float)
    fig.add_trace(
        go.Bar(
            x=list(range(len(buckets))),
            y=v,
            marker=dict(color=[SERIES[2] if x >= 0 else ACCENT for x in v]),
            text=[f"{x:+.4f}" for x in v],
            textposition="outside",
            hovertemplate="%{customdata}: %{y:+.5f} per vol point<extra></extra>",
            customdata=buckets["bucket"],
            name="bucket vega",
        )
    )
    dark_layout(fig, height=340, xtitle="QUOTE BUCKET (z = ln(K/F) / (ATM vol sqrt T))", ytitle=title, legend=False)
    fig.update_layout(hovermode="closest", margin=dict(l=60, r=20, t=30, b=90))
    fig.update_xaxes(tickvals=list(range(len(buckets))), ticktext=list(buckets["bucket"]))
    bound_axes(
        fig,
        xaxis={"min": -0.5, "max": len(buckets) - 0.5, "min_span": 1.0},
        yaxis=axis_limits(np.r_[v, 0.0], pad=0.25),
    )
    return fig


def smile_risk_section(slc: CalibratedSlice, key: str) -> None:
    """Smile risk of one option on a calibrated slice (see the module docs)."""

    fit, clean = slc.fit, slc.clean
    if fit is None or clean is None:
        return
    st.markdown("#### / SMILE RISK")
    F, T = float(slc.forward), float(slc.tau)
    df = float(clean.discount_factor)
    strikes = np.asarray(fit.strikes, dtype=float)
    default = float(strikes[int(np.argmin(np.abs(strikes - F)))])
    c1, c2 = st.columns(2)
    K = float(
        c1.number_input(
            "Option strike",
            min_value=float(strikes.min()),
            max_value=float(strikes.max()),
            value=default,
            step=float(max(np.median(np.diff(np.sort(strikes))), 0.01)),
            key=f"{key}_K",
            persist_state=PERSIST,
            help="Within the quoted strikes of this expiry (the smile is not extrapolated).",
        )
    )
    typ = c2.radio(
        "Option", ["put", "call"], index=0 if K < F else 1, horizontal=True, key=f"{key}_type", persist_state=PERSIST
    )
    try:
        r = smile_risk(
            tuple(map(float, strikes)),
            tuple(map(float, fit.market_vols)),
            tuple(map(float, fit.weights)),
            fit.loss,
            float(fit.f_scale or 0.005),
            F,
            T,
            df,
            tuple(sorted(fit.params.items())),
            K,
            typ == "call",
        )
    except ValueError as exc:
        st.error(f"Smile risk cannot be computed here: {exc}")
        return
    sabr = r.get("sabr") or {}
    stats(
        [
            ("SMILE VOL", f"{100 * r['vol']:.2f}%"),
            ("PREMIUM", f"{r['price']:,.4f}"),
            ("VEGA / VOL PT", f"{r['vega']:,.4f}"),
            ("DELTA STICKY STRIKE", f"{r['delta_sticky_strike']:+.4f}"),
            ("DELTA STICKY MONEYNESS", f"{r['delta_sticky_moneyness']:+.4f}"),
            ("DELTA SABR BARTLETT", f"{sabr['delta_bartlett']:+.4f}" if sabr else "n/a"),
        ]
    )
    b = r["buckets"]
    st.plotly_chart(
        _bucket_chart(b["buckets"], "VEGA PER +1 VOL PT"),
        width="stretch",
        config=PLOTLY_2D_CONFIG,
        key=f"{key}_bucket_chart",
    )
    st.caption(
        f"Vega by quote bucket: the P&L of this {typ} (strike {K:,.2f}, one share) when every market vol quote in "
        "the bucket rises by one vol point and the SVI smile is refitted (first order, through the least-squares "
        f"fit's normal equations). Sum over buckets {b['total']:+.4f} vs the Black vega at the smile vol "
        f"{b['black_vega']:+.4f}: a parallel move of every quote is almost absorbed by SVI. Deltas per unit of "
        f"forward, discounted (DF {df:.6f}); sticky moneyness keeps the vol at fixed ln(K/F) when F moves (ATM skew "
        f"{r['shape']['skew']:+.4f} per unit k); Bartlett's SABR delta also moves alpha with F through rho."
    )
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**SVI parameter vegas** (total variance for T)")
        st.dataframe(r["svi_vegas"], hide_index=True, width="stretch")
    with p2:
        st.markdown("**SABR parameter vegas** (beta 1, fitted to the same quotes)")
        if "sabr_vegas" in r:
            st.dataframe(r["sabr_vegas"], hide_index=True, width="stretch")
            st.caption(
                f"SABR alpha {sabr['alpha']:.4f}, rho {sabr['rho']:+.3f}, nu {sabr['nu']:.3f}, fit RMSE "
                f"{sabr['rmse_bp']:.1f} vol bp; Hagan delta {sabr['delta_hagan']:+.4f}."
            )
        else:
            st.warning(f"SABR could not be fitted to this slice: {r.get('sabr_error')}")
    st.caption("dV_dp: premium per +1.00 of the parameter; per_0.01: per +0.01. Long-option P&L signs.")
    with st.expander(f"Vega per quote ({len(b['quotes'])} quotes)"):
        st.dataframe(b["quotes"], hide_index=True, width="stretch")


# -- vol surface (SSVI + Dupire) ------------------------------------------------


@st.cache_data(max_entries=8, show_spinner=False)
def fit_surface(
    chain: pd.DataFrame, discount: Discounting, min_days: float, max_days: float, std_band: float, min_points: int
) -> Dict[str, Any]:
    """SSVI on every usable slice of ``chain`` (content-keyed cache)."""

    cleaned, bad = clean_chain(chain, min_days=1.0, discount=discount)
    slices, rows, skipped = [], [], [{"expiry": s.expiry.date(), "reason": s.reason} for s in bad]
    spot = None
    for td, exp, _, clean in cleaned:
        spot = clean.spot if spot is None else spot
        if clean.days < min_days or clean.days > max_days:
            continue
        q = apply_std_band(clean, std_band).quotes if not clean.quotes.empty else clean.quotes
        n_left, n_right = int((q["k"] < 0).sum()), int((q["k"] > 0).sum())
        if len(q) < min_points or n_left < 2 or n_right < 2:
            skipped.append({"expiry": exp.date(), "reason": f"{len(q)} quotes after the band; need {min_points}"})
            continue
        slices.append((clean.tau, q["k"].to_numpy(float), q["implied_vol"].to_numpy(float)))
        rows.append(
            {
                "expiry": exp.date(),
                "days": int(clean.days),
                "tau": clean.tau,
                "forward": clean.forward,
                "quotes": int(len(q)),
            }
        )
    if len(slices) < 2:
        raise ValueError(f"{len(slices)} usable expiries between {min_days:g} and {max_days:g} days: need 2")
    fit = fit_ssvi(slices)
    table = pd.DataFrame(rows)
    table["ssvi_rmse_bp"] = np.asarray(fit.rmse_by_expiry) * 1e4
    table["atm_total_variance"] = fit.surface.theta
    table["atm_vol_pct"] = 100 * np.sqrt(fit.surface.theta / fit.surface.expiries)
    skipped_df = pd.DataFrame(skipped, columns=["expiry", "reason"])
    return {"fit": fit, "table": table, "slices": slices, "skipped": skipped_df, "spot": spot}


def _forward_at(table: pd.DataFrame, spot: Optional[float], T: float) -> float:
    taus, fwds = table["tau"].to_numpy(float), table["forward"].to_numpy(float)
    if spot:
        taus, fwds = np.r_[0.0, taus], np.r_[spot, fwds]
    logf = np.log(fwds)
    if T > taus[-1]:
        return float(np.exp(logf[-1] + (logf[-1] - logf[-2]) / (taus[-1] - taus[-2]) * (T - taus[-1])))
    return float(np.exp(np.interp(T, taus, logf)))


def _smile_at_chart(res: Mapping[str, Any], days: float) -> go.Figure:
    fit, table = res["fit"], res["table"]
    T = days / 365.0
    k = np.linspace(min(s[1].min() for s in res["slices"]), max(s[1].max() for s in res["slices"]), 241)
    vol = 100 * fit.surface.implied_vol(k, T)
    fig = go.Figure()
    fig.update_layout(meta={"chart": "lab_ssvi_smile"})
    taus = table["tau"].to_numpy(float)
    j = int(np.clip(np.searchsorted(taus, T), 1, taus.size - 1))
    lines = [vol]
    for idx, color in ((j - 1, SERIES[2]), (j, SERIES[3])):
        tau_i, k_i, v_i = res["slices"][idx]
        lines.append(100 * v_i)
        fig.add_trace(
            go.Scatter(
                x=k_i,
                y=100 * v_i,
                mode="markers",
                name=f"quotes {int(table['days'].iloc[idx])}d",
                marker=dict(size=6, color=color),
            )
        )
    fig.add_trace(go.Scatter(x=k, y=vol, mode="lines", name=f"SSVI at {days:g}d", line=dict(color=ACCENT, width=3)))
    dark_layout(fig, height=380, xtitle="LOG-MONEYNESS k = ln(K/F(T))", ytitle="IMPLIED VOL (%)")
    fig.update_layout(legend=legend(), margin=dict(l=60, r=20, t=30, b=60))
    bound_axes(fig, xaxis=axis_limits(k), yaxis=axis_limits(np.concatenate(lines), pad=0.1, floor=0.0))
    return fig


def _heatmap(name: str, k: np.ndarray, days: np.ndarray, z: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(meta={"chart": name})
    fig.add_trace(
        go.Heatmap(
            x=k,
            y=days,
            z=z,
            colorscale=SEQUENTIAL_COLORS,
            zmin=float(np.nanpercentile(z, 2)),  # colour range only: the cells keep their values (hover)
            zmax=float(np.nanpercentile(z, 98)),
            colorbar=dict(title=dict(text=title)),
            hovertemplate="k %{x:.3f}, %{y:.0f}d: %{z:.2f}%<extra></extra>",
        )
    )
    dark_layout(fig, height=380, xtitle="LOG-MONEYNESS k = ln(K/F(T))", ytitle="DAYS TO EXPIRY", legend=False)
    fig.update_layout(hovermode="closest")
    dk, dd = float(k[1] - k[0]), float(days[1] - days[0])
    bound_axes(
        fig,
        xaxis={"min": float(k[0] - dk / 2), "max": float(k[-1] + dk / 2), "min_span": 5 * dk},
        yaxis={"min": float(days[0] - dd / 2), "max": float(days[-1] + dd / 2), "min_span": 5 * dd},
    )
    return fig


def render_surface_tab(ctx: Optional[Dict[str, Any]], draw_banner) -> None:
    """The Vol Surface tab (see the module docs); ``draw_banner(prov,
    label=..., kind=...)`` draws the chain's provenance banner."""

    st.markdown("### / VOL SURFACE (SSVI) AND LOCAL VOL")
    if ctx is None:
        st.info("No usable chain: load one on the Market Data tab, or loosen its liquidity filters, to fit a surface.")
        return
    prov = ctx["provenance"]
    draw_banner(prov)
    c1, c2, c3 = st.columns(3)
    max_days = float(
        c1.number_input("Max days to expiry", 10, 2000, 400, step=10, key="lab_surf_maxdays", persist_state=PERSIST)
    )
    std_band = float(c2.slider("Band (sd of ln K/F)", 1.0, 6.0, 3.0, 0.5, key="lab_surf_band", persist_state=PERSIST))
    min_points = int(c3.number_input("Min quotes per expiry", 6, 50, 8, key="lab_surf_minpts", persist_state=PERSIST))
    settings = {
        "min_days": 2.0,
        "max_days": max_days,
        "std_band": std_band,
        "min_points": min_points,
        "model": "power-law SSVI, soft-L1 loss at 50 vol bp",
        "discounting": ctx["discount"].describe(),
    }
    settings_key = fingerprint(settings)
    if st.button("Fit SSVI surface", key="lab_surf_run", width="stretch"):
        try:
            with st.spinner("Fitting one SSVI surface to every usable expiry..."):
                res = fit_surface(ctx["chain"], ctx["discount"], 2.0, max_days, std_band, min_points)
        except ValueError as exc:
            st.session_state[K_SURF] = {"error": str(exc), "data_key": ctx["data_key"], "settings_key": settings_key}
        else:
            st.session_state[K_SURF] = {
                "data_key": ctx["data_key"],
                "settings_key": settings_key,
                "provenance": prov,
                "result": res,
                "manifest": build_manifest(
                    "ssvi_surface",
                    inputs={"filters": ctx["filters"], **settings},
                    data=prov,
                    model={"ssvi": "Gatheral-Jacquier power law", **settings},
                ),
            }
    stored = st.session_state.get(K_SURF)
    if not stored:
        return
    if stored.get("error"):
        if stored.get("data_key") == ctx["data_key"]:
            st.error(f"The SSVI surface could not be fitted: {stored['error']}")
        return
    stale_notice(
        stored,
        ctx["data_key"],
        settings_key,
        "surface",
        lambda: draw_banner(stored["provenance"], label="STALE SURFACE: COMPUTED ON", kind="stale"),
    )
    res = stored["result"]
    fit, table = res["fit"], res["table"]
    surf = fit.surface
    report = surf.arbitrage_report()
    stats(
        [
            ("EXPIRIES", f"{len(table)}"),
            ("QUOTES", f"{fit.n_points:,}"),
            ("RMSE (VOL BP)", f"{fit.rmse_bp:.1f}"),
            ("RHO / ETA / GAMMA", f"{surf.rho:+.3f} / {surf.eta:.3f} / {surf.gamma:.3f}"),
            ("ARBITRAGE-FREE", "yes" if report["butterfly_ok"] and report["calendar_ok"] else "NO"),
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    names = {"u": "eta (at its limit eta (1 + |rho|) = 2)", "rho": "rho", "gamma": "gamma"}
    at_bound = [names.get(k, k) for k, v in fit.at_bound.items() if v]
    st.caption(
        "ssvi_rmse_bp: RMSE of the surface against each expiry's OTM quotes (vol bp); atm_vol_pct = sqrt(theta / T). "
        f"Checks: eta (1 + |rho|) = {report['eta_condition']:.4f} (<= 2), gamma {surf.gamma:.3f} (0-0.5], theta "
        f"monotone {report['theta_monotone']}, min butterfly density g {report['butterfly_min_g']:.4f} (>= 0), "
        f"min calendar dw {report['calendar_min_dw']:.2e} (>= 0)."
        + (f" Parameters at a bound: {', '.join(at_bound)} (the no-arbitrage limits bind)." if at_bound else "")
    )
    if not res["skipped"].empty:
        with st.expander(f"{len(res['skipped'])} expiry(ies) not used, with the reason"):
            st.dataframe(res["skipped"], hide_index=True, width="stretch")
    st.markdown("#### / THE SMILE AT ANY EXPIRY")
    lo_d, hi_d = int(table["days"].min()), int(table["days"].max())
    days = float(
        st.slider(
            "Read the surface at (days to expiry)",
            lo_d,
            hi_d,
            int(np.clip(45, lo_d, hi_d)),
            key="lab_surf_days",
            persist_state=PERSIST,
        )
    )
    T = days / 365.0
    F_T = _forward_at(table, res.get("spot"), T)
    st.plotly_chart(_smile_at_chart(res, days), width="stretch", config=PLOTLY_2D_CONFIG, key="lab_surf_smile")
    atm = float(surf.implied_vol(0.0, T))
    st.caption(
        f"SSVI smile at {days:g} days (T = {T:.4f}y): total variance linear in T between the fitted expiries at "
        f"fixed k, with the quotes of the two expiries around it. ATM vol {100 * atm:.2f}%; forward {F_T:,.2f} "
        "(ln F linear in T between the parity forwards). K = F(T) exp(k)."
    )
    k = np.linspace(-0.4, 0.3, 57)
    grid_days = np.linspace(max(lo_d, 7), max(hi_d, 8), 40)
    iv = np.array([100 * surf.implied_vol(k, d / 365.0) for d in grid_days])
    lv = np.array([100 * surf.local_vol(k, d / 365.0) for d in grid_days])
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Implied vol (%)**")
        st.plotly_chart(
            _heatmap("lab_ssvi_surface", k, grid_days, iv, "IV %"),
            width="stretch",
            config=PLOTLY_2D_CONFIG,
            key="lab_surf_iv",
        )
    with h2:
        st.markdown("**Dupire local vol (%)**")
        st.plotly_chart(
            _heatmap("lab_local_vol", k, grid_days, lv, "LV %"),
            width="stretch",
            config=PLOTLY_2D_CONFIG,
            key="lab_surf_lv",
        )
    st.caption(
        "Dupire local vol from the SSVI surface: sigma_loc^2 = (dw/dT at fixed k) / g(k) (Gatheral 2006, eq. 1.10), "
        "with g the butterfly density factor. theta(T) is piecewise linear, so local vol jumps at the fitted "
        "expiries. Grid k in [-0.4, 0.3], from 7 days; beyond the quoted k-range of an expiry both surfaces are "
        "SSVI extrapolations. Colours span the 2nd-98th percentile of each grid (hover for every value)."
    )
    grid = pd.DataFrame(
        {
            "days": np.repeat(grid_days, k.size),
            "k": np.tile(k, grid_days.size),
            "implied_vol_pct": iv.ravel(),
            "local_vol_pct": lv.ravel(),
        }
    )
    params = pd.DataFrame(
        [
            {
                "rho": surf.rho,
                "eta": surf.eta,
                "gamma": surf.gamma,
                "rmse_bp": fit.rmse_bp,
                **{f"arb_{k_}": v for k_, v in report.items()},
            }
        ]
    )
    downloads(
        "ssvi_surface", stored["manifest"], {"expiries": table, "params": params, "grid": grid}, key="lab_surf_dl"
    )


# -- parameter stability --------------------------------------------------------


def _param_chart(hist: pd.DataFrame, changes: pd.DataFrame, model: str, param: str, name: str) -> go.Figure:
    sub = hist[(hist["model"] == model) & (hist["parameter"] == param)].sort_values("trade_date")
    fig = go.Figure()
    fig.update_layout(meta={"chart": name})
    palette = SERIES[2:] + SERIES[:2]
    for i, (exp, g) in enumerate(sub.groupby("expiry", sort=True)):
        fig.add_trace(
            go.Scatter(
                x=g["trade_date"],
                y=g["value"],
                mode="lines+markers",
                name=str(pd.Timestamp(exp).date()) if not isinstance(exp, str) else exp,
                line=dict(color=palette[i % len(palette)], width=2),
            )
        )
    jumps = changes[(changes["model"] == model) & (changes["parameter"] == param) & changes["jump"]]
    if not jumps.empty:
        fig.add_trace(
            go.Scatter(
                x=jumps["to_date"],
                y=jumps["value_to"],
                mode="markers",
                name="jump",
                marker=dict(size=13, color=ACCENT, symbol="x"),
            )
        )
    dark_layout(fig, height=360, xtitle="TRADE DATE", ytitle=f"{model.upper()} {param}")
    fig.update_layout(legend=legend(), margin=dict(l=60, r=20, t=30, b=60))
    dates = pd.to_datetime(sub["trade_date"])
    pad = max((dates.max() - dates.min()) * 0.05, pd.Timedelta(days=1)) if len(dates) else pd.Timedelta(days=1)
    bound_axes(
        fig,
        xaxis={"min": (dates.min() - pad).isoformat(), "max": (dates.max() + pad).isoformat()},
        yaxis=axis_limits(sub["value"].to_numpy(float), pad=0.1),
    )
    return fig


def stability_results(stored: Mapping[str, Any], key: str, chart_name: str = "lab_param_history") -> None:
    """Summary, chart and jump list of a stored parameter-history run
    (``{"history", "changes", "summary", "skipped", "manifest", "z"}``)."""

    hist, changes, summary = stored["history"], stored["changes"], stored["summary"]
    if hist.empty:
        st.warning("No slice could be fitted (see the reasons below).")
    else:
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption(
            f"Day-to-day changes of each parameter for the same expiry on consecutive snapshots. A jump is a change "
            f"more than {stored['z']:g} robust standard deviations (1.4826 x MAD of the history's changes, floored) "
            "from the median change. Raw SVI parameters trade off against each other (a, b, sigma), so they can "
            "jump while the smile does not: compare with atm_vol, skew and curvature."
        )
        options = [
            (m, p)
            for m in PARAMETER_ORDER
            for p in PARAMETER_ORDER[m]
            if ((hist["model"] == m) & (hist["parameter"] == p)).any()
        ]
        pick = st.selectbox("Parameter", options, format_func=lambda o: f"{o[0].upper()} {o[1]}", key=f"{key}_param")
        if pick is not None:  # None only when no parameter has a history
            model, param = pick
            st.plotly_chart(
                _param_chart(hist, changes, model, param, chart_name),
                width="stretch",
                config=PLOTLY_2D_CONFIG,
                key=f"{key}_chart",
            )
        jumps = changes[changes["jump"]]
        with st.expander(f"{len(jumps)} jump(s) flagged"):
            st.dataframe(jumps, hide_index=True, width="stretch")
    if not stored["skipped"].empty:
        with st.expander(f"{len(stored['skipped'])} slice(s) not fitted, with the reason"):
            st.dataframe(stored["skipped"], hide_index=True, width="stretch")
    downloads(
        f"{key}_parameters",
        stored["manifest"],
        {"history": hist, "changes": changes, "summary": summary, "skipped": stored["skipped"]},
        key=f"{key}_dl",
    )


def stability_section(
    prepared_fn, discount: Discounting, max_days: float, data_key: str, prov: Mapping, draw_banner
) -> None:
    """Daily SVI / SABR parameters and their jumps on the loaded history
    (``prepared_fn()`` returns ``{trade_date: filtered chain}``)."""

    st.markdown("#### / PARAMETER HISTORY AND STABILITY (SVI, SABR)")
    c1, c2, c3 = st.columns(3)
    z = float(c1.slider("Jump threshold (robust sd)", 3.0, 8.0, 4.0, 0.5, key="lab_stab_z", persist_state=PERSIST))
    beta = float(c2.number_input("SABR beta (fixed)", 0.0, 1.0, 1.0, 0.1, key="lab_stab_beta", persist_state=PERSIST))
    warm = c3.checkbox("Warm-start SVI from the previous day", value=True, key="lab_stab_warm", persist_state=PERSIST)
    settings = {
        "z": z,
        "sabr_beta": beta,
        "warm_start": bool(warm),
        "max_days": max_days,
        "std_band": 3.0,
        "min_points": 8,
        "discounting": discount.describe(),
    }
    settings_key = fingerprint(settings)
    if st.button("Fit daily parameters", key="lab_stab_run", width="stretch"):
        try:
            prepared = prepared_fn()
        except ValueError as exc:
            st.error(f"The liquidity filters cannot be applied to this history: {exc}")
            prepared = None
        if prepared is not None:
            with st.spinner(f"Fitting SVI and SABR to every slice of {len(prepared)} snapshots..."):
                hist, skipped = parameter_history(
                    prepared,
                    discount=discount,
                    rate=discount.rate,
                    sabr_beta=beta,
                    max_days=max_days,
                    warm_start=bool(warm),
                )
                changes, summary = parameter_jumps(hist, z=z)
            st.session_state[K_STAB] = {
                "data_key": data_key,
                "settings_key": settings_key,
                "provenance": prov,
                "history": hist,
                "changes": changes,
                "summary": summary,
                "skipped": skipped,
                "z": z,
                "manifest": build_manifest("parameter_stability", inputs=settings, data=prov, model=settings),
            }
    stored = st.session_state.get(K_STAB)
    if not stored:
        return
    stale_notice(
        stored,
        data_key,
        settings_key,
        "parameter history",
        lambda: draw_banner(stored["provenance"], label="STALE PARAMETERS: COMPUTED ON", kind="stale"),
    )
    stability_results(stored, "lab_stab")
