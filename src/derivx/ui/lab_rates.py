"""ANALYTICS lab, rates (AL-20): the lab's walk-forward, regime and
model-comparison tools pointed at dated rates curves and rates-vol smiles.

* Rates Curves tab: a curve history from the rates store (the committed
  US Treasury, Bank of England OIS and ECB AAA histories, every download
  in the cache, or the user's upload), a walk-forward of curve forecasts
  (random walk, Diebold-Li Nelson-Siegel AR(1), PCA AR(1);
  :mod:`derivx.validation.curve_forecast`) with realised-vol regimes, and a
  cross-sectional comparison of Nelson-Siegel, Diebold-Li, Svensson and a
  spline (:mod:`derivx.rates.curve_models`).
* Rates Vol History tab: a dated swaption / cap normal-vol history, the
  SABR smile walk-forward against persistence, the smile-model comparison
  and the daily SABR parameter history with jumps
  (:mod:`derivx.validation.vol_history`), and implied ATM normal vol
  against the realised normal vol of the underlying forward rate, from the
  history's own forwards or a curve history of the store
  (:mod:`derivx.validation.realised_implied`). No free source publishes
  dated rates-vol quotes, so the history is the user's upload or a
  SYNTHETIC demo, labelled as such.

Units: rates decimals in the data, bp in scores; vols in bp of normal vol;
dates are observation dates of the history (horizons in observations).
"""

from __future__ import annotations

import datetime as dt
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.data import rates_history as rh
from derivx.models.comparison import MODEL_LABELS, describe_pairwise, pairwise_wins, summarise_comparison
from derivx.rates.conventions import parse_tenor
from derivx.rates.curve_models import CURVE_MODELS, compare_curve_models, fit_curve_model
from derivx.ui.lab_common import banner, downloads, fingerprint, legend, show_issues, stale_notice, stats
from derivx.ui.lab_vol import stability_results
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.state import PERSIST
from derivx.ui.theme import ACCENT, FOREGROUND, PLOTLY_2D_CONFIG, SERIES, dark_layout
from derivx.validation import realised_implied as ri
from derivx.validation.curve_forecast import PREDICTORS as CURVE_PREDICTORS
from derivx.validation.curve_forecast import CurveForecastSettings, run_curve_walk_forward
from derivx.validation.manifest import build_manifest
from derivx.validation.stability import parameter_jumps
from derivx.validation.vol_history import PREDICTORS as VOL_PREDICTORS
from derivx.validation.vol_history import (
    VOL_MODELS,
    VolHistory,
    compare_vol_models,
    parse_vol_history,
    run_vol_walk_forward,
    sabr_parameter_history,
    synthetic_vol_history,
)

MIN_CURVES = 60  # datasets with fewer complete curves are not offered
K_CURVE_UPLOAD = "lab_rates_upload"
K_CURVE_WF = "lab_rates_wf"
K_CURVE_MC = "lab_rates_mc"
K_VH = "lab_vh_history"
K_VH_WF = "lab_vh_wf"
K_VH_MC = "lab_vh_mc"
K_VH_STAB = "lab_vh_stab"
K_VH_RVI = "lab_vh_rvi"
LABELS = {
    **MODEL_LABELS,
    "ns": "Nelson-Siegel",
    "ns_dl": "Diebold-Li NS",
    "svensson": "Svensson",
    "spline": "spline",
    "normal_sabr": "normal SABR",
    "lognormal_sabr": "shifted-lognormal SABR",
}


# -- data -----------------------------------------------------------------------


@st.cache_resource(show_spinner=False, max_entries=2)
def rates_store(signature: Tuple[Tuple[str, str], ...]) -> rh.HistoryStore:
    """Bundled rates history plus every cached download (no network: the
    RATES page in LIVE mode fills the cache), rebuilt when the cache
    changes (``signature``); shared by the lab's tabs."""

    return rh.build_store([rh.bundled_frame()] + rh.cached_frames(), "bundled snapshot + download cache")


def _active_store() -> rh.HistoryStore:
    base = rates_store(rh.cache_signature())
    up = st.session_state.get(K_CURVE_UPLOAD)
    if not up:
        return base
    memo = st.session_state.setdefault("_lab_rates_store_memo", {})
    key = (base.fingerprint, up["sha"])
    if key not in memo:
        memo.clear()
        memo[key] = rh.build_store([base.frame, up["frame"]], f"{base.label} + upload {up['name']}")
    return memo[key]


def _datasets(store: rh.HistoryStore) -> List[str]:
    out = []
    for key in store.datasets():
        info = rh.dataset_info(key)
        if info.kind == "fixings":
            continue
        if len(store.dates(key)) >= MIN_CURVES:
            out.append(key)
    return sorted(out, key=lambda k: (k != "UST_PAR", k))  # the daily Treasury history first


def _upload_controls() -> None:
    with st.expander("Upload a curve history (date, tenor, rate)"):
        c1, c2 = st.columns(2)
        ccy = c1.selectbox("Currency", ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"], key="lab_rates_up_ccy")
        units = c2.selectbox("Units", ["auto", "percent", "decimal", "bp"], key="lab_rates_up_units")
        f = st.file_uploader(
            "Long (date, tenor, rate) or wide (date + one column per tenor) CSV", type=["csv"], key="lab_rates_up_file"
        )
        if f is not None and st.session_state.get("lab_rates_up_seen") != (f.file_id, ccy, units):
            st.session_state["lab_rates_up_seen"] = (f.file_id, ccy, units)
            try:
                raw = pd.read_csv(io.BytesIO(f.getvalue()))
            except (ValueError, UnicodeDecodeError) as exc:
                st.session_state["lab_rates_up_msg"] = [rh.Issue("error", f"not a CSV table: {exc}")]
            else:
                frame, issues, _ = rh.parse_history_upload(raw, ccy, units, f.name)
                st.session_state["lab_rates_up_msg"] = issues
                if not frame.empty:
                    st.session_state[K_CURVE_UPLOAD] = {
                        "frame": frame,
                        "name": f.name,
                        "sha": fingerprint(f.name, len(f.getvalue()), frame["value_decimal"].sum()),
                    }
        show_issues(st.session_state.get("lab_rates_up_msg") or [], "The curve history was not loaded:")
        if st.session_state.get(K_CURVE_UPLOAD) and st.button("Remove the uploaded history", key="lab_rates_up_rm"):
            st.session_state.pop(K_CURVE_UPLOAD, None)
            st.session_state.pop("lab_rates_up_msg", None)
        st.caption(
            "Uploaded par swap rates join the store as 'Your uploaded <CCY> swap-curve history' (rates read in "
            "the units chosen; auto reads the magnitudes and says so)."
        )


def _curve_controls(store: rh.HistoryStore) -> Optional[Dict[str, Any]]:
    keys = _datasets(store)
    if not keys:
        st.warning(f"No curve history in the store has {MIN_CURVES} complete curves.")
        return None
    c1, c2 = st.columns([2, 3])
    key = c1.selectbox(
        "Curve history", keys, format_func=lambda k: rh.dataset_info(k).title, key="lab_rates_ds", persist_state=PERSIST
    )
    info = rh.dataset_info(key)
    complete = store.complete(key)
    present = [t for t in complete.columns if complete[t].notna().mean() > 0.95]
    core = [t for t in info.core_tenors if t in present and parse_tenor(t).years >= 0.25] or present
    tenors = c2.multiselect("Tenors", present, default=core[:12], key=f"lab_rates_tenors_{key}", persist_state=PERSIST)
    first, last = complete.index[0], complete.index[-1]
    d1, d2 = st.columns(2)
    start = d1.date_input(
        "From",
        value=max(first, last - dt.timedelta(days=3 * 365)),
        min_value=first,
        max_value=last,
        key=f"lab_rates_from_{key}",
        persist_state=PERSIST,
    )
    end = d2.date_input(
        "To", value=last, min_value=first, max_value=last, key=f"lab_rates_to_{key}", persist_state=PERSIST
    )
    if len(tenors) < 4:
        st.warning("Pick at least 4 tenors (the factor models need them).")
        return None
    if end <= start:
        st.warning("The end date must be after the start date.")
        return None
    order = sorted(tenors, key=lambda t: parse_tenor(t).years)
    levels = complete[order]
    levels = levels[(levels.index >= start) & (levels.index <= end)].dropna(how="any")
    origins = sorted({store.origin(key, d) for d in (levels.index[0], levels.index[-1])}) if len(levels) else []
    gaps = np.diff(pd.to_datetime(pd.Index(levels.index)).values).astype("timedelta64[D]").astype(int)
    monthly = int((gaps > 5).sum()) if gaps.size else 0
    parts = [
        info.title,
        f"{info.measure.replace('_', ' ')} ({info.currency})",
        f"source: {info.source}",
        f"{levels.index[0]} .. {levels.index[-1]}" if len(levels) else "no dates",
        f"{len(levels):,} complete curves x {len(order)} tenors",
        "; ".join(origins),
        f"store sha256 {store.fingerprint[:12]}",
    ]
    if monthly:
        parts.append(f"{monthly} step(s) longer than 5 days (monthly stretches are not scored as daily)")
    banner("RATES HISTORY", parts)
    if info.note:
        st.caption(info.note)
    return {
        "dataset": key,
        "info": info,
        "levels": levels,
        "tenors": order,
        "store_sha": store.fingerprint,
        "data_key": fingerprint(store.fingerprint, key, order, str(start), str(end)),
        "provenance": {
            "dataset": key,
            "title": info.title,
            "source": info.source,
            "measure": info.measure,
            "first": str(levels.index[0]) if len(levels) else None,
            "last": str(levels.index[-1]) if len(levels) else None,
            "curves": len(levels),
            "tenors": order,
            "store_sha256": store.fingerprint,
            "origins": origins,
        },
    }


# -- curve walk-forward ------------------------------------------------------------


@st.cache_data(max_entries=8, show_spinner=False)
def _curve_wf(levels: pd.DataFrame, settings: CurveForecastSettings):
    return run_curve_walk_forward(levels, settings)


@st.cache_data(max_entries=8, show_spinner=False)
def _curve_mc(levels: pd.DataFrame, models: Tuple[str, ...]):
    return compare_curve_models(levels, models)


def _daily_chart(daily: pd.DataFrame, predictors, name: str, ytitle: str, x: str = "target") -> go.Figure:
    d = daily.sort_values(x)
    fig = go.Figure()
    fig.update_layout(meta={"chart": name})
    values = []
    for i, p in enumerate(predictors):
        col = f"rmse_bp_{p}"
        if col not in d or not np.isfinite(d[col].to_numpy(float)).any():
            continue
        values.append(d[col].to_numpy(float))
        style = dict(color=FOREGROUND, width=2, dash="dash") if i == 0 else dict(color=SERIES[(i + 1) % 6], width=2)
        fig.add_trace(go.Scatter(x=d[x], y=d[col], mode="lines", name=LABELS.get(p, p), line=style))
    dark_layout(fig, height=360, xtitle="TARGET DATE", ytitle=ytitle)
    fig.update_layout(legend=legend(), margin=dict(l=60, r=20, t=30, b=60))
    dates = pd.to_datetime(d[x])
    pad = max((dates.max() - dates.min()) * 0.03, pd.Timedelta(days=1))
    bound_axes(
        fig,
        xaxis={"min": (dates.min() - pad).isoformat(), "max": (dates.max() + pad).isoformat()},
        yaxis=axis_limits(np.concatenate(values) if values else [0.0, 1.0], pad=0.1, floor=0.0),
    )
    return fig


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["days won (95% CI)"] = [
        "" if not np.isfinite(w) else f"{w:.3f} [{lo:.2f}, {hi:.2f}]"
        for w, lo, hi in zip(out["day_win_rate"], out["day_win_lo"], out["day_win_hi"])
    ]
    keep = [
        c
        for c in (
            "predictor",
            "rmse_bp",
            "mae_bp",
            "benchmark",
            "benchmark_rmse_bp",
            "skill",
            "days won (95% CI)",
            "dm_t",
            "n_points",
            "n_quotes",
            "n_days",
            "description",
        )
        if c in out
    ]
    return out[keep]


def _curve_walkforward(ctx: Dict[str, Any]) -> None:
    st.markdown("#### / WALK-FORWARD: CURVE FORECASTS")
    c1, c2, c3 = st.columns(3)
    horizon = int(c1.number_input("Horizon (observations)", 1, 60, 1, key="lab_rates_h", persist_state=PERSIST))
    window = int(
        c2.number_input(
            "Estimation window (observations)", 30, 2000, 250, step=10, key="lab_rates_w", persist_state=PERSIST
        )
    )
    gap = int(
        c3.number_input(
            "Max days per step",
            1,
            40,
            5,
            key="lab_rates_gap",
            persist_state=PERSIST,
            help="Calendar days one observation step may span (5: a daily history with "
            "holidays). A window crossing a longer step is skipped.",
        )
    )
    try:
        cfg = CurveForecastSettings(horizon=horizon, window=window, max_gap_days=gap)
    except ValueError as exc:
        st.error(str(exc))
        return
    settings_key = fingerprint(cfg.as_dict())
    if st.button("Run curve walk-forward", key="lab_rates_wf_run", width="stretch"):
        try:
            with st.spinner(f"Walk-forward over {len(ctx['levels']):,} curves..."):
                result = _curve_wf(ctx["levels"], cfg)
        except ValueError as exc:
            hint = (
                " Widen the date range or shorten the window; for a monthly history (ECB, Bank of England samples) "
                "set max days per step to about 35."
            )
            st.session_state[K_CURVE_WF] = {"error": str(exc) + hint, "data_key": ctx["data_key"]}
        else:
            st.session_state[K_CURVE_WF] = {
                "data_key": ctx["data_key"],
                "settings_key": settings_key,
                "provenance": ctx["provenance"],
                "result": result,
                "manifest": build_manifest(
                    "curve_walk_forward", inputs=cfg.as_dict(), data=ctx["provenance"], model=dict(CURVE_PREDICTORS)
                ),
            }
    stored = st.session_state.get(K_CURVE_WF)
    if not stored:
        return
    if stored.get("error"):
        if stored["data_key"] == ctx["data_key"]:
            st.error(f"The walk-forward could not run: {stored['error']}")
        return
    stale_notice(
        stored,
        ctx["data_key"],
        settings_key,
        "curve walk-forward",
        lambda: banner(
            "STALE CURVE WALK-FORWARD: COMPUTED ON",
            [stored["provenance"]["title"], f"{stored['provenance']['first']} .. {stored['provenance']['last']}"],
            "stale",
        ),
    )
    result = stored["result"]
    if result.empty:
        st.warning("No origin could be scored. Reasons:")
        st.dataframe(result.skipped.drop_duplicates("reason").head(10), hide_index=True, width="stretch")
        return
    st.dataframe(_format_summary(result.summary), hide_index=True, width="stretch")
    st.caption(
        f"Errors in bp (forecast minus actual) over {len(result.tenors)} tenors, horizon {result.settings.horizon} "
        f"observation(s), models estimated on the {result.settings.window} observations up to each origin only. "
        "skill = 1 - RMSE / RMSE(random walk); days won: target dates with a lower mean squared error than the "
        "random walk (Wilson 95% interval); dm_t: Diebold-Mariano (negative favours the forecast). A random walk "
        "is hard to beat at short horizons."
    )
    st.plotly_chart(
        _daily_chart(result.daily, list(CURVE_PREDICTORS), "lab_curve_wf", "RMSE ACROSS TENORS (BP)"),
        width="stretch",
        config=PLOTLY_2D_CONFIG,
        key="lab_rates_wf_chart",
    )
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**RMSE by tenor (bp)**")
        st.dataframe(result.by_tenor, hide_index=True, width="stretch")
    with t2:
        st.markdown("**By regime (trailing realised vol of the curve level, terciles)**")
        st.dataframe(result.by_regime, hide_index=True, width="stretch")
    if not result.skipped.empty:
        with st.expander(f"{len(result.skipped)} origin(s) not scored, with the reason"):
            st.dataframe(result.skipped, hide_index=True, width="stretch")
    tables = {
        "summary": result.summary,
        "daily": result.daily,
        "by_tenor": result.by_tenor,
        "regimes": result.regimes,
        "by_regime": result.by_regime,
        "errors": result.errors,
        "skipped": result.skipped,
    }
    downloads(
        f"curve_walkforward_{ctx['dataset'].lower().replace(':', '_')}",
        stored["manifest"],
        tables,
        key="lab_rates_wf_dl",
    )


def _curve_chart(levels: pd.DataFrame, day, models) -> go.Figure:
    row = levels.loc[day]
    t = np.array([parse_tenor(c).years for c in levels.columns])
    y = row.to_numpy(float)
    grid = np.linspace(t.min(), t.max(), 200)
    fig = go.Figure()
    fig.update_layout(meta={"chart": "lab_curve_models"})
    fig.add_trace(go.Scatter(x=t, y=100 * y, mode="markers", name="quotes", marker=dict(size=9, color=FOREGROUND)))
    lines = [100 * y]
    for i, m in enumerate(models):
        try:
            fit = fit_curve_model(m, t, y)
        except ValueError:
            continue
        v = 100 * fit.predict(grid)
        lines.append(v)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=v,
                mode="lines",
                name=LABELS.get(m, m),
                line=dict(color=[ACCENT, SERIES[2], SERIES[3], SERIES[4]][i % 4], width=2),
            )
        )
    dark_layout(fig, height=360, xtitle="MATURITY (YEARS)", ytitle="RATE (%)")
    fig.update_layout(legend=legend(), margin=dict(l=60, r=20, t=30, b=60))
    bound_axes(fig, xaxis=axis_limits(grid, floor=0.0), yaxis=axis_limits(np.concatenate(lines), pad=0.1))
    return fig


def _curve_comparison(ctx: Dict[str, Any]) -> None:
    st.markdown("#### / MODEL COMPARISON: CURVE FITS")
    levels = ctx["levels"]
    c1, c2 = st.columns(2)
    models = c1.multiselect(
        "Models",
        list(CURVE_MODELS),
        default=list(CURVE_MODELS),
        format_func=LABELS.get,
        key="lab_rates_models",
        persist_state=PERSIST,
    )
    n_dates = int(
        c2.number_input("Score the most recent dates", 1, 250, 20, key="lab_rates_mc_n", persist_state=PERSIST)
    )
    if not models:
        st.warning("Select at least one model.")
        return
    settings_key = fingerprint(models, n_dates)
    if st.button("Compare curve models", key="lab_rates_mc_run", width="stretch"):
        with st.spinner(f"Fitting {len(models)} model(s) on {min(n_dates, len(levels))} dates (leave-one-out)..."):
            scores, skipped = _curve_mc(levels.tail(n_dates), tuple(models))
        st.session_state[K_CURVE_MC] = {
            "data_key": ctx["data_key"],
            "settings_key": settings_key,
            "provenance": ctx["provenance"],
            "scores": scores,
            "skipped": skipped,
            "models": list(models),
            "manifest": build_manifest(
                "curve_models",
                inputs={"models": models, "dates": n_dates},
                data=ctx["provenance"],
                model=dict(CURVE_MODELS),
            ),
        }
    stored = st.session_state.get(K_CURVE_MC)
    if not stored:
        return
    stale_notice(
        stored,
        ctx["data_key"],
        settings_key,
        "curve model comparison",
        lambda: banner("STALE CURVE MODELS: COMPUTED ON", [stored["provenance"]["title"]], "stale"),
    )
    scores = stored["scores"]
    if scores.empty:
        st.warning("No date could be scored.")
        return
    st.dataframe(summarise_comparison(scores), hide_index=True, width="stretch")
    ms = stored["models"]
    for i, a in enumerate(ms):
        for b in ms[i + 1 :]:
            st.markdown(f"- {describe_pairwise(a, b, pairwise_wins(scores, a, b), LABELS, 'dates', 'bp')}")
    st.caption(
        "rmse_cv_bp: leave-one-out RMSE over the interior tenors (bp), the comparable out-of-sample number; the "
        "spline interpolates the quotes, so its in-sample error is zero by construction. Fitted to the rates as "
        "published (par yields for the Treasury curve)."
    )
    days = sorted(scores["trade_date"].unique())
    day = st.selectbox("Show date", days, index=len(days) - 1, key="lab_rates_mc_day")
    st.plotly_chart(
        _curve_chart(ctx["levels"], day, stored["models"]),
        width="stretch",
        config=PLOTLY_2D_CONFIG,
        key="lab_rates_mc_chart",
    )
    downloads(
        "curve_models",
        stored["manifest"],
        {"scores": scores, "summary": summarise_comparison(scores), "skipped": stored["skipped"]},
        key="lab_rates_mc_dl",
    )


def render_curves_tab() -> None:
    """The Rates Curves tab (see the module docs)."""

    st.markdown("### / RATES CURVES: FORECASTS, REGIMES AND MODELS")
    st.caption(
        "Real curve histories from the rates store: the committed US Treasury par curve (FRED H.15 and the "
        "Treasury, month ends 2008-2014 then daily), Bank of England OIS and ECB AAA curves, plus anything the "
        "RATES page downloaded into the cache (LIVE mode) and your upload."
    )
    _upload_controls()
    ctx = _curve_controls(_active_store())
    if ctx is None:
        return
    _curve_walkforward(ctx)
    _curve_comparison(ctx)


# -- rates vol history ----------------------------------------------------------------


def _vh_banner(hist: VolHistory, label: str = "VOL HISTORY", kind: str = "") -> None:
    fw = sum(1 for d in hist.dates for _ in hist.forwards.get(d, {}))
    parts = [
        f"source: {hist.source}",
        f"{hist.dates[0]} .. {hist.dates[-1]} ({len(hist.dates)} dates)",
        f"{len(hist.points)} points: {', '.join(f'{e}x{t}' for e, t in hist.points[:8])}",
        "forwards given" if fw else "no forwards (normal SABR only)",
    ]
    if hist.synthetic:
        parts.insert(0, "SYNTHETIC: not market data")
    banner(label, parts, kind, synthetic=hist.synthetic)


def _vh_controls() -> Optional[VolHistory]:
    source = st.radio(
        "History",
        ["Upload a dated vol history", "SYNTHETIC demo history"],
        horizontal=True,
        key="lab_vh_source",
        persist_state=PERSIST,
    )
    if source.startswith("Upload"):
        c1, c2 = st.columns(2)
        units = c1.selectbox("Vol units", ["bp", "decimal"], key="lab_vh_units")
        f_units = c2.selectbox("Forward units", ["auto", "decimal", "percent"], key="lab_vh_funits")
        f = st.file_uploader(
            "CSV: date, expiry, tenor, offset_bp, vol (normal vols), optional forward", type=["csv"], key="lab_vh_file"
        )
        if f is not None and st.session_state.get("lab_vh_seen") != (f.file_id, units, f_units):
            st.session_state["lab_vh_seen"] = (f.file_id, units, f_units)
            try:
                raw = pd.read_csv(io.BytesIO(f.getvalue()))
            except (ValueError, UnicodeDecodeError) as exc:
                hist, issues = None, [rh.Issue("error", f"not a CSV table: {exc}")]
            else:
                hist, issues = parse_vol_history(raw, units, f_units, source=f"upload: {f.name}")
            st.session_state["lab_vh_msg"] = issues
            if hist is not None:
                st.session_state[K_VH] = hist
        show_issues(st.session_state.get("lab_vh_msg") or [], "The vol history was not loaded:")
        template = synthetic_vol_history(2, expiries=("1Y",), tenors=("10Y",)).frame().rename(columns={"vol_bp": "vol"})
        st.download_button(
            "Template (.csv; SYNTHETIC values)",
            template.to_csv(index=False).encode(),
            file_name="vol_history_template.csv",
            mime="text/csv",
            key="lab_vh_template",
            on_click="ignore",
            width="stretch",
        )
    else:
        c1, c2 = st.columns(2)
        n = int(c1.slider("Synthetic business days", 5, 60, 20, key="lab_vh_days", persist_state=PERSIST))
        seed = int(c2.number_input("Seed", 0, 10**9, 20261020, key="lab_vh_seed", persist_state=PERSIST))
        st.caption(
            "SYNTHETIC: per point a normal SABR smile (beta 0) whose ATM vol, rho, nu and forward follow random "
            "walks, quotes at -200..+200bp plus 0.5bp noise. Deterministic for a seed. Not market data."
        )
        if st.button("Generate synthetic vol history", key="lab_vh_gen", width="stretch"):
            st.session_state[K_VH] = synthetic_vol_history(n, seed=seed)
    return st.session_state.get(K_VH)


@st.cache_data(max_entries=8, show_spinner=False)
def _vh_wf(hist: VolHistory, horizon: int):
    return run_vol_walk_forward(hist, horizon)


@st.cache_data(max_entries=8, show_spinner=False)
def _vh_mc(hist: VolHistory, day: dt.date, models: Tuple[str, ...]):
    return compare_vol_models(hist, [day], models)


@st.cache_data(max_entries=8, show_spinner=False)
def _vh_params(hist: VolHistory, z: float):
    h, skipped = sabr_parameter_history(hist)
    changes, summary = parameter_jumps(h, z=z)
    return h, changes, summary, skipped


def _vh_key(hist: VolHistory) -> str:
    return fingerprint(hist.source, hist.frame().to_csv(index=False))


def render_vol_tab() -> None:
    """The Rates Vol History tab (see the module docs)."""

    st.markdown("### / RATES VOL HISTORY: SWAPTION / CAP SMILES")
    st.info(
        "No free source publishes dated swaption or cap vol quotes (brokers and data vendors license them): "
        "upload your own daily normal-vol grids (one row per date, expiry, tenor and strike offset from ATM; add "
        "the ATM forward to score sticky-strike forecasts and the shifted-lognormal model), or run the SYNTHETIC "
        "demo to see the mechanics."
    )
    hist = _vh_controls()
    if hist is None:
        return
    _vh_banner(hist)
    data_key = _vh_key(hist)
    prov = {
        "source": hist.source,
        "synthetic": hist.synthetic,
        "dates": [str(hist.dates[0]), str(hist.dates[-1])],
        "points": [f"{e}x{t}" for e, t in hist.points],
        "data_sha256": data_key,
    }
    stats(
        [
            ("DATES", f"{len(hist.dates)}"),
            ("POINTS", f"{len(hist.points)}"),
            ("QUOTES", f"{sum(len(g.rows) for g in hist.grids.values()):,}"),
        ]
    )

    def stale(what: str, stored: Dict[str, Any]):
        return lambda: banner(f"STALE {what}: COMPUTED ON", [stored["provenance"]["source"]], "stale")

    st.markdown("#### / WALK-FORWARD: SABR SMILE FORECASTS")
    horizon = int(st.number_input("Horizon (dates)", 1, 10, 1, key="lab_vh_h", persist_state=PERSIST))
    if st.button("Run vol walk-forward", key="lab_vh_wf_run", width="stretch"):
        with st.spinner(f"Fitting SABR to {len(hist.points)} points on {len(hist.dates)} dates..."):
            result = _vh_wf(hist, horizon)
        st.session_state[K_VH_WF] = {
            "data_key": data_key,
            "settings_key": str(horizon),
            "provenance": prov,
            "result": result,
            "manifest": build_manifest(
                "vol_walk_forward",
                inputs={"horizon": horizon},
                data=prov,
                model={"sabr": "normal, beta 0, shift 3%", **VOL_PREDICTORS},
            ),
        }
    stored = st.session_state.get(K_VH_WF)
    if stored:
        stale_notice(stored, data_key, str(horizon), "vol walk-forward", stale("VOL WALK-FORWARD", stored))
        result = stored["result"]
        if result.empty:
            st.warning("No quote could be scored: the history needs a target date and fitted origin smiles.")
        else:
            st.dataframe(_format_summary(result.summary), hide_index=True, width="stretch")
            st.caption(
                "Errors in bp of normal vol, forecast minus target quote at the same point and offset. Benchmarks: "
                "persistence (the origin quote), and for the sticky-strike SABR forecast the origin quotes read at "
                "the target's absolute strike. Regimes: terciles of the mean ATM normal vol across points."
            )
            st.plotly_chart(
                _daily_chart(result.daily, list(VOL_PREDICTORS), "lab_volhist_wf", "RMSE (BP NORMAL VOL)"),
                width="stretch",
                config=PLOTLY_2D_CONFIG,
                key="lab_vh_wf_chart",
            )
            t1, t2 = st.columns(2)
            t1.dataframe(result.by_point, hide_index=True, width="stretch")
            t2.dataframe(result.by_regime, hide_index=True, width="stretch")
            downloads(
                "vol_walkforward",
                stored["manifest"],
                {
                    "summary": result.summary,
                    "daily": result.daily,
                    "by_point": result.by_point,
                    "by_regime": result.by_regime,
                    "errors": result.errors,
                    "skipped": result.skipped,
                },
                key="lab_vh_wf_dl",
            )
    st.markdown("#### / MODEL COMPARISON: SMILE FITS")
    c1, c2 = st.columns(2)
    day = c1.selectbox("Date", hist.dates, index=len(hist.dates) - 1, key="lab_vh_mc_day")
    models = c2.multiselect(
        "Models", list(VOL_MODELS), default=list(VOL_MODELS), format_func=LABELS.get, key="lab_vh_models"
    )
    mc_key = fingerprint(str(day), models)
    if st.button("Compare smile models", key="lab_vh_mc_run", width="stretch", disabled=not models):
        with st.spinner("Fitting and cross-validating every point..."):
            scores, skipped = _vh_mc(hist, day, tuple(models))
        st.session_state[K_VH_MC] = {
            "data_key": data_key,
            "settings_key": mc_key,
            "provenance": prov,
            "scores": scores,
            "skipped": skipped,
            "manifest": build_manifest(
                "vol_models", inputs={"date": str(day), "models": models}, data=prov, model=dict(VOL_MODELS)
            ),
        }
    stored = st.session_state.get(K_VH_MC)
    if stored:
        stale_notice(stored, data_key, mc_key, "smile model comparison", stale("SMILE MODELS", stored))
        if stored["scores"].empty:
            st.warning("No point could be scored.")
        else:
            st.dataframe(summarise_comparison(stored["scores"]), hide_index=True, width="stretch")
            st.dataframe(stored["scores"], hide_index=True, width="stretch")
            st.caption(
                "One row per point: in-sample RMSE and leave-one-out RMSE over interior offsets (ATM kept, it pins "
                "SABR's alpha), bp of normal vol. lowest on: points where the model has the lowest CV RMSE."
            )
        if not stored["skipped"].empty:
            with st.expander(f"{len(stored['skipped'])} fit(s) skipped, with the reason"):
                st.dataframe(stored["skipped"], hide_index=True, width="stretch")
        downloads(
            "vol_models",
            stored["manifest"],
            {"scores": stored["scores"], "skipped": stored["skipped"]},
            key="lab_vh_mc_dl",
        )
    st.markdown("#### / SABR PARAMETER HISTORY AND STABILITY")
    z = float(st.slider("Jump threshold (robust sd)", 3.0, 8.0, 4.0, 0.5, key="lab_vh_z", persist_state=PERSIST))
    if st.button("Fit daily SABR parameters", key="lab_vh_stab_run", width="stretch"):
        with st.spinner("Fitting SABR on every date and point..."):
            h, changes, summary, skipped = _vh_params(hist, z)
        st.session_state[K_VH_STAB] = {
            "data_key": data_key,
            "settings_key": str(z),
            "provenance": prov,
            "history": h,
            "changes": changes,
            "summary": summary,
            "skipped": skipped,
            "z": z,
            "manifest": build_manifest(
                "vol_parameter_stability", inputs={"z": z}, data=prov, model={"sabr": "normal, beta 0"}
            ),
        }
    stored = st.session_state.get(K_VH_STAB)
    if stored:
        stale_notice(stored, data_key, str(z), "SABR parameter history", stale("SABR PARAMETERS", stored))
        stability_results(stored, "lab_vh_stab", chart_name="lab_volhist_params")
    _realised_vs_implied(hist, data_key, prov)


# -- realised against implied -----------------------------------------------------------


def _rvi_sources(hist: VolHistory, store: rh.HistoryStore) -> List[str]:
    """Underlyings on offer: the vol history's own forwards (when it has
    any), then every curve history of the store (uploads and the daily
    Treasury curve first)."""

    curves = [k for k in store.datasets() if rh.dataset_info(k).kind != "fixings"]
    curves.sort(key=lambda k: (not k.startswith("USER_SWAP:"), k != "UST_PAR", k))
    own = [ri.HISTORY_FORWARDS] if any(hist.forwards.get(d) for d in hist.dates) else []
    return own + curves


def _rvi_label(key: str) -> str:
    if key == ri.HISTORY_FORWARDS:
        return "Forwards in the vol history"
    info = rh.dataset_info(key)
    return f"{info.title} (PROXY: government yields)" if info.kind == "govt" else info.title


@st.cache_data(max_entries=8, show_spinner=False)
def _vh_rvi(
    hist: VolHistory,
    levels: Optional[pd.DataFrame],
    basis: Optional[ri.CurveBasis],
    what: str,
    cfg: ri.RealisedSettings,
) -> ri.RealisedImpliedResult:
    if levels is None:
        underlying, skipped = ri.history_forwards(hist)
        notes: Dict[str, str] = {}
    else:
        underlying, notes, skipped = ri.curve_forwards(levels, basis, hist.points)
    return ri.realised_vs_implied(hist, underlying, cfg, what=what, notes=notes, skipped=skipped)


def _rvi_chart(daily: pd.DataFrame, point: str) -> go.Figure:
    d = daily[daily["point"] == point].sort_values("date")
    fig = go.Figure()
    fig.update_layout(meta={"chart": "lab_volhist_rvi"})
    traces = (
        ("implied_bp", "implied ATM normal vol", dict(color=ACCENT, width=2)),
        ("trailing_bp", "realised, trailing window", dict(color=SERIES[2], width=2)),
        ("forward_bp", "realised, next window", dict(color=SERIES[3], width=2, dash="dash")),
    )
    values = []
    for col, name, line in traces:
        y = d[col].to_numpy(float)
        if np.isfinite(y).any():
            values.append(y[np.isfinite(y)])
            fig.add_trace(go.Scatter(x=d["date"], y=y, mode="lines+markers", name=name, line=line, marker=dict(size=4)))
    dark_layout(fig, height=360, xtitle="VOL QUOTE DATE", ytitle="NORMAL VOL (BP / YEAR)")
    fig.update_layout(legend=legend(), margin=dict(l=60, r=20, t=30, b=60))
    dates = pd.to_datetime(d["date"])
    pad = max((dates.max() - dates.min()) * 0.03, pd.Timedelta(days=1))
    bound_axes(
        fig,
        xaxis={"min": (dates.min() - pad).isoformat(), "max": (dates.max() + pad).isoformat()},
        yaxis=axis_limits(np.concatenate(values) if values else [0.0, 1.0], pad=0.1, floor=0.0),
    )
    return fig


def _rvi_underlying(source: str, hist: VolHistory, store: rh.HistoryStore, window: int):
    """(levels or None, basis or None, what, proxy, default calendar,
    rates provenance) of the chosen underlying; the curve history is cut
    to the vol dates plus two windows either side."""

    if source == ri.HISTORY_FORWARDS:
        what = "the ATM forwards in the vol history (constant expiry)"
        return None, None, what, False, "WEEKENDS", {"underlying": "vol history forwards", "source": hist.source}
    u = ri.store_underlying(store, source)
    pad = dt.timedelta(days=int(window * 7 / 5 * 2) + 30)
    idx = pd.Index(u.levels.index)
    levels = u.levels[(idx >= hist.dates[0] - pad) & (idx <= hist.dates[-1] + pad)]
    info = rh.dataset_info(source)
    if len(levels) < 2:
        raise ValueError(
            f"the store's {info.title} history covers {u.levels.index[0]} .. {u.levels.index[-1]}, not the vol "
            f"dates {hist.dates[0]} .. {hist.dates[-1]}"
        )
    origins = sorted({store.origin(source, d) for d in (levels.index[0], levels.index[-1])}) if len(levels) else []
    prov = {
        "dataset": source,
        "title": info.title,
        "source": info.source,
        "measure": u.measure,
        "tenors": [str(c) for c in levels.columns],
        "first": str(levels.index[0]) if len(levels) else None,
        "last": str(levels.index[-1]) if len(levels) else None,
        "curves": len(levels),
        "store_sha256": store.fingerprint,
        "origins": origins,
        "proxy": u.proxy,
        "basis": u.basis.description,
    }
    return levels, u.basis, u.what, u.proxy, u.calendar, prov


def _format_rvi(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for col in out.columns:
        if col.endswith("_bp") or col in ("ratio_mean", "skill", "dm_t", "implied_above_share"):
            out[col] = out[col].astype(float).round(2 if col.endswith("_bp") else 3)
    return out


def _realised_vs_implied(hist: VolHistory, data_key: str, prov: Dict[str, Any]) -> None:
    st.markdown("#### / REALISED VS IMPLIED: ATM NORMAL VOL")
    st.caption(
        "Each point's ATM normal vol against the realised normal vol of its underlying forward rate: the forwards "
        "in the vol history, or the forward swap rate expiry-into-tenor read off a curve history of the store "
        "(your uploaded swap curve, the Treasury par curve as a proxy, BoE OIS, ECB AAA). Realised = sqrt(252 x "
        "sum of squared changes / business days), zero mean, bp a year, over the window ending on each vol date "
        "(trailing) and the one after it (next)."
    )
    store = _active_store()
    sources = _rvi_sources(hist, store)
    if not sources:
        st.warning("No underlying rate history: the vol history has no forwards and the store holds no curve.")
        return
    c1, c2 = st.columns([3, 2])
    source = c1.selectbox("Realised vol of", sources, format_func=_rvi_label, key="lab_vh_rvi_src")
    window = int(
        c2.selectbox(
            "Window (business days)",
            list(ri.WINDOWS),
            index=list(ri.WINDOWS).index(21),
            format_func=lambda w: f"{w} ({w / 21:.1f} months)" if w < 252 else f"{w} (1 year)",
            key="lab_vh_rvi_w",
            persist_state=PERSIST,
        )
    )
    try:
        levels, basis, what, proxy, default_cal, rates_prov = _rvi_underlying(source, hist, store, window)
    except ValueError as exc:
        st.warning(f"{_rvi_label(source)} cannot be the underlying of this vol history: {exc}")
        return
    d1, d2 = st.columns(2)
    calendar = d1.selectbox(
        "Business-day calendar",
        list(ri.CALENDARS),
        index=list(ri.CALENDARS).index(default_cal),
        key=f"lab_vh_rvi_cal_{source}",
        help="Calendar the window and each step's business days are counted on (WEEKENDS: every weekday).",
    )
    gap = int(
        d2.number_input(
            "Max days per step",
            1,
            40,
            ri.MAX_GAP_DAYS,
            key="lab_vh_rvi_gap",
            persist_state=PERSIST,
            help="Calendar days one observation step may span; a longer step (a monthly stretch) is left out.",
        )
    )
    cfg = ri.RealisedSettings(window=window, calendar=calendar, max_gap_days=gap)
    settings_key = fingerprint(cfg.as_dict())
    rvi_key = fingerprint(data_key, source, store.fingerprint if levels is not None else "")
    if st.button("Compare realised and implied vol", key="lab_vh_rvi_run", width="stretch"):
        with st.spinner(f"Realised vol of {len(hist.points)} point(s) over {window}-business-day windows..."):
            computed = _vh_rvi(hist, levels, basis, what, cfg)
        st.session_state[K_VH_RVI] = {
            "data_key": rvi_key,
            "settings_key": settings_key,
            "provenance": prov,
            "rates": rates_prov,
            "source": source,
            "proxy": proxy,
            "result": computed,
            "manifest": build_manifest(
                "realised_vs_implied",
                inputs={"underlying": source, **cfg.as_dict()},
                data={"vol_history": prov, "rates": rates_prov},
                model={
                    "realised": "1e4 x sqrt(days_per_year x sum dF^2 / sum business days), zero mean",
                    "forward_swap_rate": basis.description if basis is not None else "as given in the vol history",
                    "implied": "ATM (0bp offset) normal vol quote",
                },
            ),
        }
    stored = st.session_state.get(K_VH_RVI)
    if not stored:
        return
    stale_notice(
        stored,
        rvi_key,
        settings_key,
        "realised-vs-implied",
        lambda: banner(
            "STALE REALISED VS IMPLIED: COMPUTED ON",
            [stored["provenance"]["source"], f"realised: {stored['result'].underlying}"],
            "stale",
        ),
    )
    result: ri.RealisedImpliedResult = stored["result"]
    cfg_used = result.settings
    parts = [
        f"implied: ATM normal vols, {stored['provenance']['source']}",
        f"realised: {result.underlying}",
        f"{cfg_used.window}-business-day windows on the {cfg_used.calendar} calendar, steps up to "
        f"{cfg_used.max_gap_days} days, {cfg_used.min_coverage:.0%} covered",
    ]
    rates = stored["rates"]
    if rates.get("dataset"):
        parts.append(f"rates {rates['first']} .. {rates['last']} ({'; '.join(rates['origins'])})")
    if stored["provenance"]["synthetic"]:
        parts.insert(0, "SYNTHETIC implied vols: not market data")
    banner("REALISED VS IMPLIED", parts, synthetic=bool(stored["provenance"]["synthetic"]))
    if stored["proxy"]:
        st.warning(
            "PROXY underlying: government yields stand in for swap rates. Their realised vol leaves out the "
            "swap-spread vol, so a swaption's implied vol is not expected to match it. Upload a swap-curve history "
            "(Rates Curves tab) or vol-history forwards for the like-for-like comparison."
        )
    if result.empty:
        st.warning("No realised vol could be measured at the vol dates (reasons below).")
    else:
        st.dataframe(_format_rvi(result.summary), hide_index=True, width="stretch")
        st.caption(
            "bp of normal vol a year. spread = implied - trailing realised (the window ending on the quote date); "
            "premium = implied - forward realised (the next window; positive: a delta-hedged option seller "
            "earned it); ratio = implied / trailing. Breakeven daily move = implied / sqrt(252), in bp a "
            "business day. skill = 1 - RMSE(implied) / RMSE(trailing) as forecasts of the forward realised vol, "
            "and dm_t is the Diebold-Mariano statistic of that comparison with overlapping windows (negative "
            "favours implied). Consecutive dates share most of a window, so short histories say little. Only "
            "dates whose windows are covered are scored."
        )
        points = list(result.summary["point"])
        point = st.selectbox("Point", points, key="lab_vh_rvi_point")
        if point is not None:
            last = result.daily[(result.daily["point"] == point) & np.isfinite(result.daily["trailing_bp"])]
            if len(last):
                row = last.iloc[-1]
                st.caption(f"{point} on {row['date']}, the latest vol date with a covered trailing window:")
                stats(
                    [
                        ("IMPLIED, BP/YR", f"{row['implied_bp']:.1f}"),
                        ("REALISED, BP/YR", f"{row['trailing_bp']:.1f}"),
                        ("SPREAD, BP/YR", f"{row['spread_bp']:+.1f}"),
                        ("BREAKEVEN, BP/DAY", f"{row['breakeven_daily_bp']:.2f}"),
                    ]
                )
            st.plotly_chart(
                _rvi_chart(result.daily, point), width="stretch", config=PLOTLY_2D_CONFIG, key="lab_vh_rvi_chart"
            )
    if not result.skipped.empty:
        with st.expander(f"{len(result.skipped)} point(s) not compared, with the reason"):
            st.dataframe(result.skipped, hide_index=True, width="stretch")
    downloads(
        "realised_vs_implied",
        stored["manifest"],
        {"summary": result.summary, "daily": result.daily, "skipped": result.skipped},
        key="lab_vh_rvi_dl",
    )
