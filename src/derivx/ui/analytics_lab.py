"""ANALYTICS lab: option-chain data with lineage, production SVI
calibration, walk-forward validation against benchmarks, and smile-model
comparison.

Data and lineage. The Market Data tab holds one *active dataset* (one
snapshot): the bundled real SPY sample (loaded by default), an uploaded file
(CSV, CSV.GZ or parquet, validated by :mod:`derivx.data.chain_schema`) or
the current Yahoo chain. A dataset changes only on an explicit action: a
new upload (tracked by file id, so a still-attached file never replaces a
later pull), a pull, or the sample button. A banner on every tab names the
dataset: source, ticker, session, real fetch time (UTC), rows and a
SHA-256 of its content. Yahoo pulls are shared between sessions only for
the same ticker and request within one 5-minute bucket, never on a
per-session counter, and failed or partial pulls are never cached; the
rows keep their real fetch time, so a cached pull says when it was fetched.

Results (calibration, walk-forward, model comparison) live in this
session's state, each tied to a fingerprint of the data, filters and
settings that produced it: when the active data change, a result is marked
STALE until it is re-run. Each result downloads as a ZIP with a run
manifest (:mod:`derivx.validation.manifest`) and CSV tables.

Units: strikes and premiums in the underlying's currency per share, vols
as per-annum decimals (``*_bp`` columns and chart axes: vol basis points,
1 bp = 0.0001), ``k = ln(K/F)`` on put-call-parity forwards, ``tau`` in
ACT/365F years, horizons in exchange trading days.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.calibration.production import ChainCalibration, calibrate_chain
from derivx.data.calendar import missing_trading_days
from derivx.data.chain_schema import (
    ChainDataset,
    ChainSchemaError,
    chain_to_csv,
    load_chain_file,
    merge_datasets,
)
from derivx.data.market_data import apply_liquidity_filters, fetch_option_chain, hygiene_report
from derivx.data.samples import load_sample_chain, sample_fetch_status
from derivx.data.synthetic import DEFAULT_SEED, SYNTH_TICKER, synthetic_history
from derivx.models.comparison import (
    MODELS,
    ModelComparison,
    compare_models_by_expiry,
    describe_pairwise,
    fit_models,
    pairwise_wins,
    summarise_comparison,
)
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.state import PERSIST
from derivx.ui.theme import ACCENT, FOREGROUND, PLOTLY_2D_CONFIG, SERIES, dark_layout
from derivx.validation.backtest import PREDICTORS, WalkForwardResult, WalkForwardSettings, run_walk_forward
from derivx.validation.manifest import build_manifest, jsonable, manifest_bytes, results_zip
from derivx.vol.quotes import REASONS
from derivx.vol.svi import svi_iv

ANALYTICS_TABS = [
    "Market Data",
    "Production Calibration",
    "Walk-Forward Validation",
    "Model Comparison",
]
UPLOAD_TYPES = ["csv", "gz", "parquet", "pq", "txt"]
PULL_BUCKET_S = 300  # Yahoo pulls are shared between sessions within one 5-minute bucket
MIN_FIT_DAYS = 2.0  # expiries closer than this are not fitted or scored
PRICER_TENORS_DAYS = (7.0, 30.0, 61.0, 91.0, 182.0, 365.0)  # expiries handed to the EQUITY page's MARKET smile
SPOT_DRIFT_WARN = 1e-3  # warn when spot moved more than 0.1% between the first and last expiry response of a fetch

# session-state keys (results are per session; nothing here is shared)
K_DATA = "lab_dataset"
K_LOAD_MSG = "lab_load_message"
K_UPLOAD_SEEN = "lab_upload_seen"
K_UPLOAD_PARSED = "lab_upload_parsed"
K_UPLOAD_DAY = "lab_upload_day_active"
K_CAL = "lab_calibration"
K_WF_DATA = "lab_wf_dataset"
K_WF_SEEN = "lab_wf_upload_seen"
K_WF_MSG = "lab_wf_message"
K_WF = "lab_walkforward"
K_MC = "lab_model_comparison"
K_MC_ALL = "lab_model_comparison_all"

LAB_CSS = """
<style>
.dx-lab-banner { border:1px solid #FFFFFF; border-left:5px solid #35C1FF; background:#000000; color:#FFFFFF;
  padding:0.55rem 0.8rem; margin:0.25rem 0 1rem; font-family:'IBM Plex Mono', monospace; font-size:0.8rem;
  line-height:1.5; overflow-wrap:anywhere; }
.dx-lab-banner b { color:#35C1FF; letter-spacing:0.08em; }
.dx-lab-banner.synthetic { border-left-color:#FFA500; }
.dx-lab-banner.synthetic b { color:#FFA500; }
.dx-lab-banner.stale { border-left-color:#FF4444; }
.dx-lab-banner.stale b { color:#FF4444; }
.dx-lab-stats { display:grid; grid-template-columns:repeat(auto-fill, minmax(9.5rem, 1fr)); gap:0.6rem;
  margin:0.5rem 0 1rem; }
.dx-lab-stat { background:#FFFFFF; color:#000000; border:3px solid #000000; padding:0.6rem 0.8rem; }
.dx-lab-stat span { display:block; font-size:0.7rem; font-weight:700; letter-spacing:0.12em; color:#333333; }
.dx-lab-stat b { display:block; font-size:1.5rem; font-family:'IBM Plex Mono', monospace; white-space:nowrap;
  overflow-x:auto; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color:#FFFFFF !important; }
[data-testid="stMain"] [data-testid="stCaptionContainer"], [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
  color:#C8C8C8 !important; }
[data-testid="stFileUploaderDropzone"] { background:#000000 !important; border:2px dashed #FFFFFF !important;
  border-radius:0 !important; }
[data-testid="stFileUploaderDropzone"] * { color:#E0E0E0 !important; }
[data-testid="stFileUploaderDropzone"] button { background:#000000 !important; color:#FFFFFF !important;
  border:2px solid #FFFFFF !important; border-radius:0 !important; }
[data-testid="stFileUploaderFile"] { background:#000000 !important; border:1px solid #FFFFFF !important; }
[data-testid="stFileUploaderFile"] * { color:#FFFFFF !important; background:transparent !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:#FFFFFF !important; }
.stDownloadButton button { background:#000000; color:#FFFFFF; border:2px solid #FFFFFF; border-radius:0;
  font-weight:700; letter-spacing:0.05em; }
.stDownloadButton button:hover { background:#FFFFFF; color:#000000; }
[data-testid="stMain"] .stButton button { padding:0.6rem 0.9rem; white-space:normal; }
</style>
"""


def init_state() -> None:
    """Load the bundled real sample as the active dataset on first visit."""

    if K_DATA not in st.session_state:
        try:
            ds = load_sample_chain()
        except (OSError, ChainSchemaError) as exc:  # a broken install must not take the page down
            st.session_state[K_DATA] = None
            st.session_state[K_LOAD_MSG] = ("error", f"Bundled sample unavailable: {exc}")
        else:
            st.session_state[K_DATA] = {"dataset": ds, "origin": "sample", "status": sample_fetch_status()}


# -- helpers -------------------------------------------------------------------


def _key(*parts) -> str:
    """Fingerprint of JSON-able parts (data hashes, filters, settings)."""

    blob = json.dumps(jsonable(list(parts)), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _day(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _when(iso: Optional[str]) -> str:
    if not iso:
        return "fetch time not recorded"
    ts = pd.Timestamp(iso)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def _banner(
    prov: Mapping[str, object], label: str = "ACTIVE DATASET", extra: Sequence[str] = (), kind: str = ""
) -> None:
    dates = prov.get("trade_dates") or []
    session = dates[0] if len(dates) == 1 else (f"{dates[0]} .. {dates[-1]} ({len(dates)} sessions)" if dates else "?")
    size = f"{int(prov.get('rows') or 0):,} rows, {prov.get('expiries')} expiries"
    if len(dates) > 1 and prov.get("slices") is not None:
        size += f" ({prov['slices']} session x expiry slices)"
    row_sources = [str(s) for s in prov.get("row_sources") or []]
    labelled = "synthetic" in str(prov.get("source", "")).lower()
    synth_rows = [s for s in row_sources if "synthetic" in s.lower()] and not labelled
    parts = [
        str(prov.get("ticker") or "ticker unknown"),
        f"session {session}",
        f"source: {prov.get('source')}",
        f"fetched {_when(prov['fetched_at_utc'])}" if prov.get("fetched_at_utc") else "fetch time not recorded",
        size,
        f"sha256 {str(prov.get('data_sha256', ''))[:12]}",
        *(["SYNTHETIC ROWS: not market data"] if synth_rows else []),
        *extra,
    ]
    is_synth = labelled or bool(synth_rows)
    css = " ".join(c for c in ("dx-lab-banner", kind, "synthetic" if is_synth else "") if c)
    body = " · ".join(html.escape(p) for p in parts)
    st.markdown(f"<div class='{css}'><b>{html.escape(label)}</b> · {body}</div>", unsafe_allow_html=True)


def _downloads(stem: str, manifest: Mapping[str, object], tables: Mapping[str, pd.DataFrame], key: str) -> None:
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download results (.zip: manifest + CSVs)",
        results_zip(manifest, tables),
        file_name=f"{stem}.zip",
        mime="application/zip",
        key=f"{key}_zip",
        on_click="ignore",
        width="stretch",
    )
    c2.download_button(
        "Download run manifest (.json)",
        manifest_bytes(manifest),
        file_name=f"{stem}_manifest.json",
        mime="application/json",
        key=f"{key}_manifest",
        on_click="ignore",
        width="stretch",
    )


def _stale_notice(stored: Mapping[str, object], data_key: str, settings_key: str, what: str) -> bool:
    """Warn when a stored result no longer matches the data or settings;
    True when the data changed (the result is STALE)."""

    if stored.get("data_key") != data_key:
        _banner(stored["provenance"], label=f"STALE {what.upper()}: COMPUTED ON", kind="stale")
        st.warning(
            f"These {what} results were computed on the dataset above, not on the active one (or with other "
            "filters). Re-run to update them."
        )
        return True
    if stored.get("settings_key") != settings_key:
        st.info(f"Settings changed since this {what} run: press the button to apply them.")
    return False


def _legend(y: float) -> Dict[str, object]:
    """Horizontal legend below the x-axis title, in the foreground colour."""

    return dict(orientation="h", yanchor="top", y=y, x=0.0, font=dict(color=FOREGROUND))


def _stats(items: Sequence[Tuple[str, str]]) -> None:
    """A row of labelled numbers that reflows on narrow screens."""

    cells = "".join(
        f"<div class='dx-lab-stat'><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>" for k, v in items
    )
    st.markdown(f"<div class='dx-lab-stats'>{cells}</div>", unsafe_allow_html=True)


def _manifest(kind: str, prov: Mapping, inputs: Mapping, model: Mapping, seed=None) -> Dict[str, object]:
    return build_manifest(kind, inputs=inputs, data=prov, model=model, seed=seed)


def _filter_breakdown(chain: pd.DataFrame, filters: Mapping[str, object]) -> str:
    """How many quotes fail each liquidity rule (a quote can fail several)."""

    bid = pd.to_numeric(chain["bid"], errors="coerce").fillna(0.0)
    ask = pd.to_numeric(chain["ask"], errors="coerce").fillna(0.0)
    mid = (bid + ask) / 2.0
    parts = []
    if filters["min_volume"] > 0 and "volume" in chain.columns:
        low = int((pd.to_numeric(chain["volume"], errors="coerce").fillna(0) < filters["min_volume"]).sum())
        parts.append(f"volume < {filters['min_volume']} (missing counts as 0): {low:,}")
    parts.append(f"bid <= 0: {int((bid <= 0).sum()):,}")
    parts.append(f"ask <= bid: {int((ask <= bid).sum()):,}")
    parts.append(f"mid < {filters['min_mid']:g}: {int((mid < filters['min_mid']).sum()):,}")
    if filters["max_spread"] is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            wide = ((ask - bid) / mid.replace(0, np.nan)) > filters["max_spread"]
        parts.append(f"relative spread > {filters['max_spread']:g}: {int(wide.sum()):,}")
    return f"{len(chain):,} quotes in; " + "; ".join(parts)


# -- Yahoo pulls -----------------------------------------------------------------


class _NotCached(Exception):
    """A pull that must not be cached (failed or partial)."""

    def __init__(self, chain: pd.DataFrame, status: Dict[str, object]):
        super().__init__(status.get("reason", ""))
        self.chain, self.status = chain, status


@st.cache_data(ttl=2 * PULL_BUCKET_S, max_entries=16, show_spinner=False)
def _pull_shared(ticker: str, max_expiries: int, bucket: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """The current chain, shared by sessions asking for the same ticker and
    expiries within one bucket of ``PULL_BUCKET_S`` seconds (``bucket`` is
    the bucket number, part of the cache key). Only complete pulls are
    cached; the rows carry their real fetch times."""

    result = fetch_option_chain(ticker, max_expiries=max_expiries, min_days=MIN_FIT_DAYS)
    status = result.status.as_dict()
    if result.status.status != "ok":
        raise _NotCached(result.chain, status)
    return result.chain, status


def _pull(ticker: str, max_expiries: int, force: bool) -> Tuple[pd.DataFrame, Dict[str, object], bool]:
    """(chain, status, served_from_cache). ``force`` bypasses the cache."""

    if force:
        result = fetch_option_chain(ticker, max_expiries=max_expiries, min_days=MIN_FIT_DAYS)
        return result.chain, result.status.as_dict(), False
    clicked = pd.Timestamp.now(tz="UTC")
    try:
        chain, status = _pull_shared(ticker, int(max_expiries), int(time.time() // PULL_BUCKET_S))
    except _NotCached as exc:
        return exc.chain, exc.status, False
    finished = pd.Timestamp(status["finished_at_utc"])
    return chain, status, bool(finished < clicked - pd.Timedelta(seconds=5))


# -- dataset activation ----------------------------------------------------------


def _activate(ds: ChainDataset, origin: str, status: Optional[dict] = None, note: str = "") -> None:
    previous = (st.session_state.get(K_DATA) or {}).get("dataset")
    st.session_state[K_DATA] = {"dataset": ds, "origin": origin, "status": status}
    msg = f"Active dataset: {ds.source}, session {', '.join(_day(d) for d in ds.trade_dates)}, {ds.rows:,} rows."
    if previous is not None and previous.fingerprint != ds.fingerprint:
        msg += f" It replaced: {previous.source} (session {', '.join(_day(d) for d in previous.trade_dates)})."
    st.session_state[K_LOAD_MSG] = ("success", (note + " " if note else "") + msg)


def _active() -> Optional[Dict[str, object]]:
    return st.session_state.get(K_DATA)


def _single_day(ds: ChainDataset, day: pd.Timestamp) -> ChainDataset:
    return ChainDataset(
        chains={day: ds.chains[day]},
        source=ds.source,
        ticker=ds.ticker,
        fetched_at_utc=ds.fetched_at_utc,
        warnings=list(ds.warnings),
        renamed=dict(ds.renamed),
        files=list(ds.files),
    )


def _show_problems(title: str, problems: Iterable[str]) -> None:
    items = "\n".join(f"- {p}" for p in problems)
    st.error(f"{title}\n\n{items}")


@st.cache_data(max_entries=8, show_spinner=False)
def _csv_bytes(chain: pd.DataFrame, compress: bool = False) -> bytes:
    """Chain export bytes (content-keyed cache: built once per chain)."""

    data = chain_to_csv(chain)
    return gzip.compress(data, mtime=0) if compress else data


@st.cache_data(max_entries=8, show_spinner=False)
def _hygiene(chain: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Per-expiry hygiene report (content-keyed: the same quotes and rate
    give the same report in any session)."""

    return hygiene_report(chain, rate=rate)


def _load_controls() -> None:
    st.markdown("#### / LOAD DATA")
    c_sample, c_upload, c_pull = st.columns(3)
    with c_sample:
        st.markdown("**Bundled sample**")
        st.caption(
            "One real SPY chain from Yahoo Finance (delayed quotes), fetched 2026-10-20 20:59 UTC after the close, "
            "unfiltered (data/examples). A single snapshot: no history."
        )
        if st.button("Load sample", key="lab_load_sample", width="stretch"):
            _activate(load_sample_chain(), "sample", sample_fetch_status())
    with c_upload:
        st.markdown("**Upload a chain**")
        up = st.file_uploader(
            "CSV, CSV.GZ or parquet in the chain schema",
            type=UPLOAD_TYPES,
            key="lab_upload",
            help="Required columns: trade_date, expiry, option_type, strike, bid, ask, implied_vol (decimal), "
            "underlying_price. Optional: volume (without it, set Min volume to 0), last_trade_date, ticker, "
            "fetched_at_utc and more. Dates as YYYY-MM-DD or YYYYMMDD. Files exported here (and by "
            "tools/download_spy_chains.py) re-upload as is.",
        )
        if up is not None and st.session_state.get(K_UPLOAD_SEEN) != up.file_id:
            st.session_state[K_UPLOAD_SEEN] = up.file_id  # process each file once: no silent re-load on reruns
            st.session_state.pop(K_UPLOAD_PARSED, None)
            try:
                parsed = load_chain_file(up.name, up.getvalue())
            except ChainSchemaError as exc:
                st.session_state[K_LOAD_MSG] = ("problems", (f"{up.name} was not loaded:", exc.problems))
            else:
                st.session_state[K_UPLOAD_PARSED] = (up.file_id, parsed)
                if len(parsed.chains) == 1:
                    _activate(parsed, "upload")
        parsed_entry = st.session_state.get(K_UPLOAD_PARSED)
        if up is not None and parsed_entry and parsed_entry[0] == up.file_id and len(parsed_entry[1].chains) > 1:
            parsed = parsed_entry[1]
            days = parsed.trade_dates
            pick = st.selectbox(
                f"This file holds {len(days)} trade dates: use",
                days,
                index=len(days) - 1,
                format_func=_day,
                key=f"lab_upload_day_{up.file_id}",
                help="This tab analyses one snapshot. The Walk-Forward tab uses every trade date of a file.",
            )
            if st.session_state.get(K_UPLOAD_DAY) != (up.file_id, pick):
                st.session_state[K_UPLOAD_DAY] = (up.file_id, pick)
                _activate(_single_day(parsed, pick), "upload", note=f"Using trade date {_day(pick)} of {up.name}.")
    with c_pull:
        st.markdown("**Yahoo Finance (current chain)**")
        ticker = st.text_input("Ticker", value="SPY", key="lab_ticker", persist_state=PERSIST).strip().upper()
        n_exp = st.number_input(
            "Nearest expiries to request",
            min_value=1,
            max_value=40,
            value=8,
            step=1,
            key="lab_max_exp",
            persist_state=PERSIST,
        )
        force = st.checkbox("Bypass the 5-minute shared cache", value=False, key="lab_force", persist_state=PERSIST)
        st.caption(
            "Yahoo serves only the current (delayed) chain. Rows are stamped with the session the quotes belong "
            "to and their real fetch time; rate limits are retried with backoff. The nearest expiries at least "
            f"{MIN_FIT_DAYS:g} days away are requested (after the close Yahoo still lists the day's expired one)."
        )
        if st.button("Pull current chain", key="lab_pull", width="stretch"):
            try:
                with st.spinner(f"Fetching {ticker} from Yahoo Finance..."):
                    chain, status, cached = _pull(ticker, int(n_exp), bool(force))
            except ValueError as exc:
                st.session_state[K_LOAD_MSG] = ("error", f"Not fetched: {exc}")
            else:
                if status.get("status") in ("ok", "partial") and not chain.empty:
                    ds = ChainDataset(
                        chains={pd.Timestamp(status["quote_date"]): chain},
                        source="Yahoo Finance via yfinance (delayed quotes)",
                        ticker=status["ticker"],
                        fetched_at_utc=status["finished_at_utc"],
                        files=[],
                    )
                    note = "Served from the shared cache. " if cached else ""
                    if status.get("status") == "partial":
                        note += f"PARTIAL: {len(status['failed_expiries'])} expiry request(s) failed. "
                    _activate(ds, "yahoo", status, note=note.strip())
                else:
                    st.session_state[K_LOAD_MSG] = (
                        "error",
                        f"Yahoo pull failed ({status.get('status')}): {status.get('reason')}",
                    )
    msg = st.session_state.get(K_LOAD_MSG)
    if msg:
        level, text = msg
        if level == "problems":
            _show_problems(*text)
        elif level == "error":
            st.error(text)
        else:
            st.success(text)
    active = _active() or {}
    status = active.get("status") or {}
    _fetch_status_panel(status)
    if active.get("origin") == "yahoo" and status.get("failed_expiries"):
        with st.expander(f"{len(status['failed_expiries'])} expiry request(s) failed"):
            st.dataframe(
                pd.DataFrame([{"expiry": k, "reason": v} for k, v in status["failed_expiries"].items()]),
                hide_index=True,
                width="stretch",
            )
    if active.get("dataset") is not None and active["dataset"].warnings:
        with st.expander(f"{len(active['dataset'].warnings)} data note(s)"):
            for w in active["dataset"].warnings:
                st.markdown(f"- {w}")


def _fetch_status_panel(status: Optional[Mapping[str, object]]) -> None:
    """What the fetch of the active dataset did (Yahoo pulls and the
    bundled sample's recorded fetch): requests, retries, the spot and how
    far it moved between the first and last expiry response, and notes."""

    if not isinstance(status, Mapping):
        return
    fs = status.get("fetch_status", status)
    if not isinstance(fs, Mapping) or "status" not in fs:
        return
    drift = fs.get("spot_drift")
    drift_txt = "not measured" if drift is None else f"{float(drift) * 1e4:+.1f} bp of spot"
    spot = fs.get("spot")
    rows = [
        ("status", f"{fs.get('status')}" + (f": {fs.get('reason')}" if fs.get("reason") else "")),
        ("session the quotes belong to", fs.get("quote_date") or "?"),
        ("fetch started / finished (UTC)", f"{fs.get('started_at_utc')} / {fs.get('finished_at_utc')}"),
        (
            "spot (first response) and its time (UTC)",
            f"{float(spot):,.4f} at {fs.get('spot_time_utc') or 'time not reported'}" if spot is not None else "?",
        ),
        ("spot drift, first to last expiry response", drift_txt),
        ("requests / retries / backoff wait", f"{fs.get('requests')} / {fs.get('retries')} / {fs.get('waited_s')} s"),
        (
            "expiries requested / fetched / failed",
            f"{len(fs.get('expiries_requested') or [])} / {len(fs.get('expiries_fetched') or [])} / "
            f"{len(fs.get('failed_expiries') or {})}",
        ),
    ]
    with st.expander(f"Fetch status: {fs.get('status')}, spot drift {drift_txt}"):
        st.dataframe(pd.DataFrame(rows, columns=["field", "value"]), hide_index=True, width="stretch")
        for note in fs.get("notes") or []:
            st.markdown(f"- {note}")
    if drift is not None and abs(float(drift)) > SPOT_DRIFT_WARN:
        st.warning(
            f"Spot moved {float(drift) * 1e4:+.1f} bp between the first and the last expiry response of this fetch: "
            "the expiries were quoted against different spots (each slice keeps the spot of its own response)."
        )


def _publish_to_pricer(ds: ChainDataset, prov: Mapping, filtered: pd.DataFrame, report: pd.DataFrame, filters) -> None:
    """The EQUITY page's MARKET smile reads ``st.session_state['market_chain']``:
    give it the expiries nearest the standard tenors (each one SVI fit there),
    with the dataset's provenance."""

    picks = _nearest(report, PRICER_TENORS_DAYS)
    expiry = pd.to_datetime(filtered["expiry"]).dt.normalize()
    st.session_state["market_chain"] = filtered[expiry.isin(set(picks))].reset_index(drop=True)
    st.session_state["market_chain_meta"] = {
        "ticker": ds.ticker,
        "source": f"{ds.source}; expiries nearest 1W, 1M, 2M, 3M, 6M and 1Y: {', '.join(_day(p) for p in picks)}",
        "trade_date": prov["trade_dates"][0],
        "fetched_at_utc": ds.fetched_at_utc,
        "data_sha256": prov["data_sha256"],
        "filters": dict(filters),
    }


def _render_market_data_tab() -> Optional[Dict[str, object]]:
    """Draw the Market Data tab; returns the context the other tabs use
    (filtered chain, provenance, hygiene report and its fingerprint)."""

    st.markdown("### / MARKET DATA")
    banner_slot = st.container()  # filled after the load controls, so it shows this run's dataset
    _load_controls()
    active = _active()
    if not active or active.get("dataset") is None:
        st.info("Load the bundled sample, upload a chain or pull from Yahoo Finance to continue.")
        return None
    ds: ChainDataset = active["dataset"]
    raw = ds.single()
    prov = ds.provenance()
    with banner_slot:
        _banner(prov)

    st.markdown("#### / LIQUIDITY FILTERS")
    f1, f2, f3, f4, f5 = st.columns(5)
    min_volume = f1.number_input("Min volume", min_value=0, value=10, step=5, key="lab_minvol", persist_state=PERSIST)
    min_mid = f2.number_input("Min mid", min_value=0.0, value=0.10, step=0.05, key="lab_minmid", persist_state=PERSIST)
    max_spread = f3.number_input(
        "Max rel spread", min_value=0.01, value=0.40, step=0.05, key="lab_maxspread", persist_state=PERSIST
    )
    no_cap = f4.checkbox("No spread cap", value=False, key="lab_nocap", persist_state=PERSIST)
    rate_pct = f5.number_input(
        "Rate r (% cc)",
        min_value=-5.0,
        max_value=20.0,
        value=0.0,
        step=0.25,
        key="lab_rate",
        persist_state=PERSIST,
        help="Continuously compounded discount rate used to re-invert vols from bid/ask mids (an input, not market "
        "data). At 0% vols are biased low by about r x T in price terms (tens of vol bp at 1 year for r = 4%).",
    )
    filters = {
        "min_volume": int(min_volume),
        "min_mid": float(min_mid),
        "max_spread": None if no_cap else float(max_spread),
        "rate_cc": float(rate_pct) / 100.0,
    }
    try:
        filtered = apply_liquidity_filters(
            raw, min_volume=filters["min_volume"], max_spread=filters["max_spread"], min_mid=filters["min_mid"]
        )
    except ValueError as exc:
        st.session_state["market_chain"] = raw.iloc[0:0]
        st.error(f"The liquidity filters cannot be applied to this chain: {exc}")
        return None
    data_key = _key(prov["data_sha256"], filters)
    if filtered.empty:
        st.session_state["market_chain"] = filtered
        st.warning(
            "The liquidity filters removed every quote: loosen them to continue. Quotes failing each rule: "
            + _filter_breakdown(raw, filters)
            + "."
        )
        return None
    with st.spinner("Cleaning quotes (hygiene)..."):
        report = _hygiene(filtered, filters["rate_cc"])
    _publish_to_pricer(ds, prov, filtered, report, filters)
    usable = report[(report["days"] >= MIN_FIT_DAYS) & (report["kept"] >= 8)] if not report.empty else report
    spots = pd.to_numeric(raw["underlying_price"], errors="coerce").dropna()
    spot_txt = (
        f"{spots.iloc[0]:,.2f}" if spots.nunique() <= 1 else f"{spots.min():,.2f}-{spots.max():,.2f}"
    )  # one spot per expiry response
    _stats(
        [
            ("QUOTES AFTER FILTERS", f"{len(filtered):,}"),
            ("EXPIRIES", f"{filtered['expiry'].nunique()}"),
            ("OTM QUOTES KEPT", f"{int(report['kept'].sum()) if not report.empty else 0:,}"),
            ("SPOT" if spots.nunique() <= 1 else "SPOT RANGE", spot_txt),
        ]
    )
    st.caption(
        f"{len(raw):,} raw quotes -> {len(filtered):,} after the liquidity filters (calls and puts both kept: "
        "put-call parity needs them) -> hygiene per expiry below (one OTM quote per strike). Spot is Yahoo's "
        "regular-market price from the same response as the quotes. The EQUITY page's MARKET smile gets the "
        "expiries nearest 1W, 1M, 2M, 3M, 6M and 1Y of this filtered chain."
    )
    st.markdown("#### / QUOTE HYGIENE BY EXPIRY")
    st.dataframe(report, hide_index=True, width="stretch")
    st.caption(
        "Removed, by reason: "
        + "; ".join(f"{k}: {v}" for k, v in REASONS.items() if k != "outside_band")
        + ". forward_method: parity = put-call parity on the slice's mids; interpolated = ln F linear in T from the "
        f"other parity forwards; carry = spot (no parity). {len(usable)} expiries have >= {MIN_FIT_DAYS:g} days and "
        ">= 8 OTM quotes."
    )
    with st.expander(f"Filtered quotes ({len(filtered):,} rows)"):
        st.dataframe(filtered, hide_index=True, width="stretch")
    d1, d2, d3 = st.columns(3)
    stem = f"{(ds.ticker or 'chain').lower()}_{prov['trade_dates'][0]}"
    d1.download_button(
        "Filtered chain (.csv)",
        _csv_bytes(filtered),
        file_name=f"{stem}_filtered.csv",
        mime="text/csv",
        key="lab_dl_filtered",
        help="Same schema as the uploads: re-uploadable as is.",
        on_click="ignore",
        width="stretch",
    )
    d2.download_button(
        "Raw chain (.csv.gz)",
        _csv_bytes(raw, compress=True),
        file_name=f"{stem}_raw.csv.gz",
        mime="application/gzip",
        key="lab_dl_raw",
        on_click="ignore",
        width="stretch",
    )
    d3.download_button(
        "Provenance (.json)",
        json.dumps(jsonable({"dataset": prov, "filters": filters, "fetch_status": active.get("status")}), indent=2),
        file_name=f"{stem}_provenance.json",
        mime="application/json",
        key="lab_dl_prov",
        on_click="ignore",
        width="stretch",
    )
    return {
        "chain": filtered,
        "provenance": prov,
        "filters": filters,
        "data_key": data_key,
        "report": report,
        "dataset": ds,
    }


# -- calibration -----------------------------------------------------------------


def _expiry_label(report: pd.DataFrame):
    days = {pd.Timestamp(r.expiry): int(r.days) for r in report.itertuples()}

    def fmt(exp) -> str:
        d = days.get(pd.Timestamp(exp))
        return f"{_day(exp)} ({d}d)" if d is not None else _day(exp)

    return fmt


def _nearest(report: pd.DataFrame, targets: Sequence[float]) -> List[pd.Timestamp]:
    ok = report[(report["days"] >= MIN_FIT_DAYS) & (report["kept"] >= 8)]
    out: List[pd.Timestamp] = []
    for t in targets:
        if ok.empty:
            break
        row = ok.iloc[int(np.argmin(np.abs(ok["days"].to_numpy(float) - t)))]
        exp = pd.Timestamp(row["expiry"])
        if exp not in out:
            out.append(exp)
    return out


def _plot_calibration_smile(slc) -> go.Figure:
    q = slc.clean.quotes if slc.clean is not None else pd.DataFrame({"strike": slc.strikes, "option_type": "smile"})
    q = q[q["strike"].isin(slc.strikes)]
    F, T = slc.forward, slc.tau
    grid = np.linspace(float(np.min(slc.strikes)), float(np.max(slc.strikes)), 241)
    fitted = np.asarray(svi_iv(grid, slc.params, F=F, T=T), dtype=float)
    fig = go.Figure()
    fig.update_layout(meta={"chart": "lab_calibration"})
    obs = dict(zip(slc.strikes, slc.observed_ivs))
    for label, typ, color in (("OTM puts (K < F)", "put", SERIES[2]), ("OTM calls (K > F)", "call", FOREGROUND)):
        sub = q[q["option_type"] == typ]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub["strike"],
                    y=[obs[k] for k in sub["strike"]],
                    mode="markers",
                    name=label,
                    marker=dict(size=7, color=color, line=dict(width=1, color="#000000")),
                )
            )
    other = q[~q["option_type"].isin(["put", "call"])]
    if not other.empty:
        fig.add_trace(
            go.Scatter(
                x=other["strike"],
                y=[obs[k] for k in other["strike"]],
                mode="markers",
                name="call/put averaged at F",
                marker=dict(size=8, color=SERIES[3], symbol="diamond"),
            )
        )
    fig.add_trace(go.Scatter(x=grid, y=fitted, mode="lines", name="SVI fit", line=dict(color=ACCENT, width=3)))
    atm = float(svi_iv(F, slc.params, F=F, T=T))
    fig.add_trace(
        go.Scatter(x=[F], y=[atm], mode="markers", name="ATM-F", marker=dict(size=11, color=SERIES[4], symbol="x"))
    )
    dark_layout(fig, height=400, xtitle="STRIKE", ytitle="IMPLIED VOL (decimal)")
    fig.update_layout(legend=_legend(-0.3), margin=dict(l=60, r=20, t=30, b=120))
    bound_axes(
        fig,
        xaxis=axis_limits(grid, floor=0.0),
        yaxis=axis_limits(np.r_[slc.observed_ivs, fitted], pad=0.1, floor=0.0),
    )
    return fig


def _calibration_tables(result: ChainCalibration) -> Dict[str, pd.DataFrame]:
    metrics = result.table()
    params = pd.DataFrame(
        [{"expiry": s.expiry.date(), "tau": s.tau, "forward": s.forward, **s.params} for s in result.slices]
    )
    skipped = pd.DataFrame([{"expiry": s.expiry.date(), "reason": s.reason} for s in result.skipped])
    return {"metrics": metrics, "params": params, "skipped": skipped, "calendar": result.calendar}


def _render_calibration_tab(ctx: Optional[Dict[str, object]]) -> None:
    st.markdown("### / PRODUCTION CALIBRATION")
    if ctx is None:
        st.info("Load a chain on the Market Data tab to calibrate.")
        return
    prov, report, chain = ctx["provenance"], ctx["report"], ctx["chain"]
    _banner(prov, extra=[f"filters {ctx['filters']}"])
    expiries = [pd.Timestamp(e) for e in report["expiry"]]
    fmt = _expiry_label(report)
    sel = st.multiselect(
        "Expiries",
        options=expiries,
        default=_nearest(report, (30.0, 91.0)),
        format_func=fmt,
        key=f"lab_cal_exp_{prov['data_sha256'][:10]}",
        help="Defaults: the expiries nearest 30 and 91 days with enough quotes (expired and same-day expiries "
        "are never fitted).",
    )
    c1, c2, c3 = st.columns(3)
    std_band = c1.slider(
        "Band (standard deviations of ln K/F)", 1.0, 6.0, 3.0, 0.5, key="lab_cal_std", persist_state=PERSIST
    )
    band = c2.slider("Band K/F", 0.30, 2.00, (0.70, 1.30), 0.01, key="lab_cal_band", persist_state=PERSIST)
    enforce = c3.checkbox(
        "Enforce no-arbitrage",
        value=True,
        key="lab_cal_enforce",
        persist_state=PERSIST,
        help="An arbitrageable free fit (butterfly g(k) < 0 or Lee wings) is replaced by a constrained SVI fit or "
        "an arbitrage-free SSVI slice. Butterfly, Lee and calendar checks are always reported.",
    )
    settings = {
        "expiries": sorted(_day(e) for e in sel),
        "std_band": float(std_band),
        "moneyness_band": [float(band[0]), float(band[1])],
        "enforce_arbitrage": bool(enforce),
        "cv_folds": 5,
        "min_days": MIN_FIT_DAYS,
        "rate_cc": ctx["filters"]["rate_cc"],
    }
    settings_key = _key(settings)
    if st.button("Calibrate selected expiries", key="lab_cal_run", width="stretch", disabled=not sel):
        subset = chain[pd.to_datetime(chain["expiry"]).dt.normalize().isin(set(sel))]
        with st.spinner(f"Calibrating {len(sel)} SVI slice(s)..."):
            result = calibrate_chain(
                subset,
                enforce_arbitrage=enforce,
                std_band=std_band,
                moneyness_band=band,
                rate=settings["rate_cc"],
                min_days=MIN_FIT_DAYS,
                cv_folds=5,
            )
        st.session_state[K_CAL] = {
            "data_key": ctx["data_key"],
            "settings_key": settings_key,
            "provenance": prov,
            "settings": settings,
            "result": result,
            "manifest": _manifest(
                "calibration", prov, {"filters": ctx["filters"], **settings}, {"model": "raw SVI, total variance"}
            ),
        }
    stored = st.session_state.get(K_CAL)
    if not stored:
        if not sel:
            st.info("Select at least one expiry.")
        return
    _stale_notice(stored, ctx["data_key"], settings_key, "calibration")
    result: ChainCalibration = stored["result"]
    tables = _calibration_tables(result)
    if result.skipped:
        with st.expander(f"{len(result.skipped)} slice(s) skipped, with the reason"):
            st.dataframe(tables["skipped"], hide_index=True, width="stretch")
    if not result.slices:
        st.warning("No slice could be fitted (see the reasons above). Loosen the bands or pick other expiries.")
        return
    st.dataframe(tables["metrics"], hide_index=True, width="stretch")
    st.caption(
        "rmse_bp: in-sample RMSE in vol bp; cv_rmse_bp / spline_cv_rmse_bp: 5-fold cross-validated RMSE of SVI and "
        "of a cubic spline of IV in ln K/F at held-out interior strikes (the spline interpolates, so only held-out "
        "errors compare). forward_method: parity / interpolated / carry. Vols re-inverted from bid/ask mids with "
        f"Black-76 at r = {stored['settings']['rate_cc']:.2%} cc. min_g: Gatheral butterfly density minimum (>= 0 "
        "is arbitrage-free); calendar_ok: total variance non-decreasing vs the previous fitted expiry."
    )
    labels = [_day(s.expiry) for s in result.slices]
    pick = st.selectbox("Show expiry", labels, key=f"lab_cal_show_{stored['settings_key'][:10]}")
    slc = result.slices[labels.index(pick)]
    st.plotly_chart(_plot_calibration_smile(slc), width="stretch", config=PLOTLY_2D_CONFIG, key="lab_cal_chart")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### / SVI PARAMETERS (total variance)")
        st.dataframe(
            pd.DataFrame([{"parameter": k, "value": v} for k, v in slc.params.items()]),
            hide_index=True,
            width="stretch",
        )
    with p2:
        st.markdown("#### / DIAGNOSTICS")
        st.json(jsonable(slc.diagnostics), expanded=1)
    stem = f"calibration_{(prov.get('ticker') or 'chain').lower()}_{prov['trade_dates'][0]}"
    _downloads(stem, stored["manifest"], tables, key="lab_cal_dl")


# -- walk-forward ----------------------------------------------------------------

HISTORY_HELP = (
    "A walk-forward test needs **genuine daily snapshots**: one chain per trading day, each saved on its own day. "
    "Yahoo Finance has no history (it serves only the current chain), so the app no longer offers a 'download "
    "history' button: it used to save today's chain under every past date. Build a history by running "
    "`python tools/download_spy_chains.py --ticker SPY --outdir data/market/spy` once per trading day after the "
    "close (it stamps each file with the session and the real fetch time), then upload the files here. The "
    "SYNTHETIC demo shows the mechanics on generated data that is not market data."
)


def _wf_controls() -> None:
    source = st.radio(
        "History",
        ["Upload daily snapshots", "SYNTHETIC demo history"],
        horizontal=True,
        key="lab_wf_source",
        persist_state=PERSIST,
    )
    if source == "Upload daily snapshots":
        files = st.file_uploader(
            "Daily snapshot files (CSV, CSV.GZ or parquet; one trade date per file, or several per file)",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key="lab_wf_upload",
        )
        ids = tuple(f.file_id for f in files or [])
        if files and st.session_state.get(K_WF_SEEN) != ids:
            st.session_state[K_WF_SEEN] = ids
            problems, datasets = [], []
            for f in files:
                try:
                    datasets.append(load_chain_file(f.name, f.getvalue()))
                except ChainSchemaError as exc:
                    problems += [f"{f.name}: {p}" for p in exc.problems]
            if not problems:
                try:
                    merged = merge_datasets(datasets, source=f"upload: {len(files)} file(s)")
                except ChainSchemaError as exc:
                    problems += exc.problems
                else:
                    st.session_state[K_WF_DATA] = {"dataset": merged, "origin": "upload", "seed": None}
                    st.session_state[K_WF_MSG] = (
                        "success",
                        f"Loaded {len(merged.chains)} snapshot(s) from {len(files)} file(s).",
                    )
            if problems:
                st.session_state[K_WF_MSG] = ("problems", ("The files were not loaded:", problems))
    else:
        c1, c2 = st.columns(2)
        n_days = c1.slider("Synthetic trading days", 4, 15, 8, key="lab_wf_synth_days", persist_state=PERSIST)
        seed = c2.number_input(
            "Seed", min_value=0, value=DEFAULT_SEED, step=1, key="lab_wf_seed", persist_state=PERSIST
        )
        st.caption(
            "SYNTHETIC: SSVI smiles with a random-walk spot and vol level, bid/ask around Black-76 prices plus "
            "15 vol bp quote noise (derivx.data.synthetic). Deterministic for a seed. Not market data."
        )
        if st.button("Generate synthetic history", key="lab_wf_synth_go", width="stretch"):
            chains = synthetic_history(int(n_days), seed=int(seed))
            ds = ChainDataset(
                chains=chains,
                source=f"SYNTHETIC demo history (seed {int(seed)}): not market data",
                ticker=SYNTH_TICKER,
                fetched_at_utc=None,
            )
            st.session_state[K_WF_DATA] = {"dataset": ds, "origin": "synthetic", "seed": int(seed)}
            st.session_state[K_WF_MSG] = ("success", f"Generated {len(chains)} SYNTHETIC snapshots (seed {int(seed)}).")
    msg = st.session_state.get(K_WF_MSG)
    if msg:
        level, text = msg
        if level == "problems":
            _show_problems(*text)
        else:
            st.success(text)


def _plot_walkforward(daily: pd.DataFrame) -> go.Figure:
    d = daily.sort_values("target")
    fig = go.Figure()
    fig.update_layout(meta={"chart": "lab_walkforward"})
    styles = {
        "svi_sticky_strike": dict(color=ACCENT, width=3),
        "persistence": dict(color=FOREGROUND, width=2, dash="dash"),
        "svi_sticky_moneyness": dict(color=SERIES[3], width=2),
        "sticky_moneyness": dict(color=SERIES[4], width=2, dash="dot"),
    }
    values = []
    for name in PREDICTORS:
        col = f"rmse_bp_{name}"
        values.append(d[col].to_numpy(float))
        fig.add_trace(go.Scatter(x=d["target"], y=d[col], mode="lines+markers", name=name, line=styles[name]))
    fig.add_trace(
        go.Scatter(
            x=d["target"],
            y=d["noise_bp"],
            mode="lines",
            name="noise floor (RMS half spread)",
            line=dict(color=SERIES[2], width=2, dash="dashdot"),
        )
    )
    values.append(d["noise_bp"].to_numpy(float))
    dark_layout(fig, height=380, xtitle="TARGET DATE", ytitle="RMSE (VOL BP)")
    fig.update_layout(legend=_legend(-0.45), margin=dict(l=60, r=20, t=30, b=170))
    dates = pd.to_datetime(d["target"])
    pad = max((dates.max() - dates.min()) * 0.05, pd.Timedelta(days=1))
    bound_axes(
        fig,
        xaxis={"min": (dates.min() - pad).isoformat(), "max": (dates.max() + pad).isoformat()},
        yaxis=axis_limits(np.concatenate(values), pad=0.1, floor=0.0),
    )
    return fig


def _prepared_history(ds: ChainDataset, filters: Mapping[str, object]) -> Dict[pd.Timestamp, pd.DataFrame]:
    out = {}
    for td, df in ds.chains.items():
        f = apply_liquidity_filters(
            df, min_volume=filters["min_volume"], max_spread=filters["max_spread"], min_mid=filters["min_mid"]
        )
        if not f.empty:
            out[td] = f
    return out


def _render_walkforward_tab() -> None:
    st.markdown("### / WALK-FORWARD VALIDATION")
    st.markdown(HISTORY_HELP)
    _wf_controls()
    entry = st.session_state.get(K_WF_DATA)
    if not entry:
        st.info("Upload daily snapshots or generate the SYNTHETIC demo history to run a walk-forward test.")
        return
    ds: ChainDataset = entry["dataset"]
    prov = ds.provenance()
    if ds.warnings:
        with st.expander(f"{len(ds.warnings)} data note(s) on these files"):
            for w in ds.warnings:
                st.markdown(f"- {w}")
    gaps = missing_trading_days(ds.trade_dates)
    _banner(
        prov,
        label="HISTORY",
        extra=[f"missing trading days: {', '.join(_day(g) for g in gaps[:6])}{'...' if len(gaps) > 6 else ''}"]
        if gaps
        else ["no missing trading days"],
    )
    if len(ds.chains) < 2:
        st.warning(
            f"The history holds {len(ds.chains)} snapshot: a forecast needs at least two trading days. A single "
            "snapshot (like the bundled sample) supports calibration and model comparison, not a walk-forward."
        )
        return
    st.markdown("#### / FILTERS AND SETTINGS")
    f1, f2, f3, f4 = st.columns(4)
    filters = {
        "min_volume": int(
            f1.number_input("Min volume", min_value=0, value=10, step=5, key="lab_wf_minvol", persist_state=PERSIST)
        ),
        "min_mid": float(
            f2.number_input("Min mid", min_value=0.0, value=0.10, step=0.05, key="lab_wf_minmid", persist_state=PERSIST)
        ),
        "max_spread": float(
            f3.number_input(
                "Max rel spread", min_value=0.01, value=0.40, step=0.05, key="lab_wf_maxspread", persist_state=PERSIST
            )
        ),
    }
    rate = (
        float(f4.number_input("Rate r (% cc)", -5.0, 20.0, 0.0, 0.25, key="lab_wf_rate", persist_state=PERSIST)) / 100
    )
    s1, s2, s3, s4 = st.columns(4)
    horizon = int(s1.number_input("Horizon (trading days)", 1, 5, 1, key="lab_wf_horizon", persist_state=PERSIST))
    max_window = max(1, min(10, len(ds.chains) - horizon))
    if max_window > 1:
        window = int(
            s2.slider(
                "Calibration window (snapshots pooled)",
                1,
                max_window,
                1,
                key="lab_wf_window",
                persist_state=PERSIST,
                help="SVI at each origin is fitted to that expiry's quotes pooled over this many snapshots up to the "
                "origin (each at its own k = ln K/F). 1 = the origin day only.",
            )
        )
    else:
        window = 1
        s2.caption("Calibration window: 1 snapshot (load more days to pool).")
    max_days = float(
        s3.number_input("Max days to expiry", 3, 1000, 120, step=10, key="lab_wf_maxdays", persist_state=PERSIST)
    )
    enforce = s4.checkbox("Enforce no-arbitrage", value=True, key="lab_wf_enforce", persist_state=PERSIST)
    settings = WalkForwardSettings(
        horizon=horizon, window=window, rate=rate, max_days=max_days, enforce_arbitrage=enforce, min_days=MIN_FIT_DAYS
    )
    data_key = _key(prov["data_sha256"], filters)
    settings_key = _key(settings.as_dict())
    run = st.button("Run walk-forward test", key="lab_wf_run", width="stretch")
    prepared = None
    if run:
        try:
            prepared = _prepared_history(ds, filters)
        except ValueError as exc:
            st.error(f"The liquidity filters cannot be applied to this history: {exc}")
    if prepared is not None:
        with st.spinner(f"Walk-forward over {len(prepared)} snapshots (one SVI fit per origin and expiry)..."):
            result = run_walk_forward(prepared, settings)
        st.session_state[K_WF] = {
            "data_key": data_key,
            "settings_key": settings_key,
            "provenance": prov,
            "result": result,
            "manifest": _manifest(
                "walk_forward", prov, {"filters": filters}, settings.as_dict(), seed=entry.get("seed")
            ),
        }
    stored = st.session_state.get(K_WF)
    if not stored:
        return
    _stale_notice(stored, data_key, settings_key, "walk-forward")
    result: WalkForwardResult = stored["result"]
    if result.empty:
        st.warning("No forecast could be scored. Reasons:")
        st.dataframe(result.skipped.drop_duplicates("reason").head(20), hide_index=True, width="stretch")
        return
    cfg = result.settings
    st.markdown("#### / SCORES (pooled over every scored quote)")
    st.dataframe(_format_summary(result.summary), hide_index=True, width="stretch")
    st.caption(
        f"Horizon {cfg.horizon} trading day(s); calibration window {cfg.window} snapshot(s). Quotes matched on "
        "(expiry, strike, option type): a call is never scored against a put. svi_sticky_strike and persistence use "
        "origin information only; the sticky-moneyness pair is conditional on the realised target forward. skill = 1 "
        "- RMSE / benchmark RMSE (> 0 beats the benchmark). days won: target days on which the forecast's mean "
        "squared error is below its benchmark's, with a Wilson 95% interval on days. win rate: share of quotes whose "
        "actual move exceeds the target half spread on which the forecast beats its benchmark; its Wilson interval "
        "treats quotes as independent, but quotes of one day move together, so it is too narrow: judge significance "
        "by days won and dm_t. dm_t: Diebold-Mariano statistic on daily squared-error differences, with "
        "autocovariances up to lag horizon - 1 for overlapping multi-day horizons and the Harvey-Leybourne-Newbold "
        "correction (negative favours the forecast; rough with few days). noise_bp: RMS half bid/ask width of the "
        "target quotes, the level below which errors are inside the market."
    )
    st.plotly_chart(_plot_walkforward(result.daily), width="stretch", config=PLOTLY_2D_CONFIG, key="lab_wf_chart")
    st.markdown("#### / REGIMES (30-day constant-maturity ATM vol, terciles of this history)")
    r1, r2 = st.columns(2)
    with r1:
        st.dataframe(
            result.regimes.assign(trade_date=result.regimes["trade_date"].dt.date), hide_index=True, width="stretch"
        )
    with r2:
        st.dataframe(result.by_regime, hide_index=True, width="stretch")
    st.caption(
        f"Buckets are terciles of the history's own 30-day ATM vols (labels use the whole sample); a bucket needs "
        f"{cfg.min_regime_days} origin days to be 'sufficient'. Vol at k = 0 per expiry, total variance linear in T "
        "between the expiries bracketing 30 days."
    )
    with st.expander(f"Per-slice scores ({len(result.slices)} origin x expiry rows)"):
        st.dataframe(result.slices, hide_index=True, width="stretch")
    if not result.skipped.empty:
        counts = result.skipped["stage"].value_counts().to_dict()
        with st.expander(f"Not scored: {len(result.skipped)} ({', '.join(f'{k} {v}' for k, v in counts.items())})"):
            st.dataframe(result.skipped, hide_index=True, width="stretch")
    tables = {
        "summary": result.summary,
        "daily": result.daily,
        "slices": result.slices,
        "quotes": result.quotes,
        "regimes": result.regimes,
        "by_regime": result.by_regime,
        "skipped": result.skipped,
    }
    _downloads(f"walkforward_{(prov.get('ticker') or 'chain').lower()}", stored["manifest"], tables, key="lab_wf_dl")


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["win_rate (95% CI)"] = [
        "" if not np.isfinite(w) else f"{w:.3f} [{lo:.3f}, {hi:.3f}] n={int(n)}"
        for w, lo, hi, n in zip(out["win_rate"], out["win_lo"], out["win_hi"], out["win_n"])
    ]
    out["days won (95% CI)"] = [
        "" if not np.isfinite(w) else f"{int(round(w * n))} of {int(n)} [{lo:.2f}, {hi:.2f}]"
        for w, lo, hi, n in zip(out["day_win_rate"], out["day_win_lo"], out["day_win_hi"], out["day_win_n"])
    ]
    cols = ["predictor", "rmse_bp", "benchmark", "benchmark_rmse_bp", "skill", "days won (95% CI)"]
    cols += ["win_rate (95% CI)", "dm_t", "noise_bp", "n_quotes", "n_days", "description"]
    return out[cols]


# -- model comparison -------------------------------------------------------------


def _plot_model_comparison(comp: ModelComparison, models: Sequence[str]) -> go.Figure:
    q = comp.quotes
    K = q["strike"].to_numpy(float)
    grid = np.linspace(K.min(), K.max(), 241)
    fig = go.Figure()
    fig.update_layout(meta={"chart": "lab_model_comparison"})
    for label, typ, color in (("OTM puts (K < F)", "put", SERIES[2]), ("OTM calls (K > F)", "call", FOREGROUND)):
        sub = q[q["option_type"] == typ]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub["strike"],
                    y=sub["implied_vol"],
                    mode="markers",
                    name=label,
                    marker=dict(size=7, color=color, line=dict(width=1, color="#000000")),
                )
            )
    styles = {
        "svi": dict(name="SVI", line=dict(width=3, color=ACCENT)),
        "sabr": dict(name="SABR", line=dict(width=3, dash="dash", color=SERIES[3])),
        "spline": dict(name="IV spline (ln K/F)", line=dict(width=2, dash="dot", color=SERIES[4])),
    }
    lines = [q["implied_vol"].to_numpy(float)]
    for model in models:
        if model in comp.fits:
            y = np.asarray(comp.predict(model, grid), dtype=float)
            lines.append(y)
            fig.add_trace(go.Scatter(x=grid, y=y, mode="lines", **styles[model]))
    dark_layout(fig, height=400, xtitle="STRIKE", ytitle="IMPLIED VOL (decimal)")
    fig.update_layout(legend=_legend(-0.3), margin=dict(l=60, r=20, t=30, b=120))
    bound_axes(
        fig,
        xaxis=axis_limits(grid, floor=0.0),
        yaxis=axis_limits(np.concatenate(lines), pad=0.1, floor=0.0),
    )
    return fig


def _model_params(comp: ModelComparison) -> pd.DataFrame:
    rows = []
    for name, fit in comp.fits.items():
        if name == "spline":
            continue
        rows += [{"model": name, "parameter": k, "value": float(v)} for k, v in dict(fit.params).items()]
    return pd.DataFrame(rows)


def _render_model_comparison_tab(ctx: Optional[Dict[str, object]]) -> None:
    st.markdown("### / MODEL COMPARISON")
    if ctx is None:
        st.info("Load a chain on the Market Data tab to compare models.")
        return
    prov, report, chain = ctx["provenance"], ctx["report"], ctx["chain"]
    _banner(prov, extra=[f"filters {ctx['filters']}"])
    ok = report[(report["days"] >= MIN_FIT_DAYS) & (report["kept"] >= 8)]
    if ok.empty:
        st.warning(f"No expiry has >= {MIN_FIT_DAYS:g} days and 8 OTM quotes after hygiene.")
        return
    expiries = [pd.Timestamp(e) for e in ok["expiry"]]
    default = _nearest(report, (30.0,))
    fmt = _expiry_label(report)
    c1, c2, c3 = st.columns(3)
    expiry = c1.selectbox(
        "Expiry",
        expiries,
        index=expiries.index(default[0]) if default else 0,
        format_func=fmt,
        key=f"lab_mc_exp_{prov['data_sha256'][:10]}",
        help="Default: the expiry nearest 30 days. Expired, same-day and sparse expiries are not offered.",
    )
    models = c2.multiselect("Models", list(MODELS), default=list(MODELS), key="lab_mc_models", persist_state=PERSIST)
    beta = c3.number_input(
        "SABR beta (fixed)",
        0.0,
        1.0,
        1.0,
        0.1,
        key="lab_mc_beta",
        persist_state=PERSIST,
        help="1 = lognormal (equity index)",
    )
    if not models:
        st.warning("Select at least one model.")
        return
    rate = ctx["filters"]["rate_cc"]
    row = ok[pd.to_datetime(ok["expiry"]) == expiry].iloc[0]
    settings = {
        "expiry": _day(expiry),
        "models": list(models),
        "sabr_beta": float(beta),
        "cv_folds": 5,
        "std_band": 3.0,
        "min_points": 8,
        "rate_cc": rate,
    }
    settings_key = _key(settings)
    if st.button("Compare models", key="lab_mc_run", width="stretch"):
        slice_df = chain[pd.to_datetime(chain["expiry"]).dt.normalize() == expiry]
        try:
            with st.spinner("Fitting and cross-validating..."):
                comp = fit_models(
                    slice_df,
                    models,
                    sabr_beta=float(beta),
                    rate=rate,
                    min_days=MIN_FIT_DAYS,
                    hygiene={"forward": float(row["forward"]), "forward_source": f"{row['forward_method']} (chain)"},
                )
        except ValueError as exc:
            st.session_state[K_MC] = {"error": str(exc), "data_key": ctx["data_key"], "settings_key": settings_key}
        else:
            st.session_state[K_MC] = {
                "data_key": ctx["data_key"],
                "settings_key": settings_key,
                "provenance": prov,
                "comp": comp,
                "models": list(models),
                "forward_method": row["forward_method"],
                "manifest": _manifest("model_comparison", prov, {"filters": ctx["filters"]}, settings),
            }
    stored = st.session_state.get(K_MC)
    if stored and stored.get("error"):
        # an error belongs to the data and settings it was raised on: never shown against other ones
        if stored.get("data_key") == ctx["data_key"] and stored.get("settings_key") == settings_key:
            st.error(f"Model comparison failed for this slice: {stored['error']}")
    elif stored:
        _stale_notice(stored, ctx["data_key"], settings_key, "model comparison")
        comp: ModelComparison = stored["comp"]
        table = comp.table
        st.dataframe(table, hide_index=True, width="stretch")
        failed = table[table["status"].astype(str).str.startswith("failed")]
        for r in failed.itertuples():
            st.warning(f"{r.model.upper()} could not be fitted and is not plotted: {r.status}")
        attrs = table.attrs
        st.caption(
            f"Slice {attrs.get('expiry')} (trade date {attrs.get('trade_date')}, "
            f"tau {attrs.get('tau', float('nan')):.4f}y), "
            f"forward {attrs.get('forward', float('nan')):.4f} ({stored.get('forward_method')}). One OTM quote per "
            "strike (puts below F, calls above: call and put vols are never mixed at a strike), vols re-inverted "
            "from mids, band |k| <= 3 sd. rmse_cv_bp: 5-fold CV at held-out interior strikes, the comparable "
            "out-of-sample number (the spline interpolates, so its in-sample error is 0 by construction)."
        )
        st.plotly_chart(
            _plot_model_comparison(comp, stored["models"]), width="stretch", config=PLOTLY_2D_CONFIG, key="lab_mc_chart"
        )
        tables = {"scores": table.reset_index(drop=True), "params": _model_params(comp), "quotes": comp.quotes}
        _downloads(f"models_{settings['expiry']}", stored["manifest"], tables, key="lab_mc_dl")

    st.markdown("#### / EVERY EXPIRY")
    a1, a2 = st.columns(2)
    max_days = float(
        a1.number_input("Max days to expiry", 3, 1000, 120, step=10, key="lab_mc_all_maxdays", persist_state=PERSIST)
    )
    n_all = int((ok["days"] <= max_days).sum())
    a2.caption(f"{n_all} expiries in range; about 2-3 s each (5-fold CV of every model).")
    all_settings = {**settings, "expiry": "all", "max_days": max_days}
    all_key = _key(all_settings)
    if st.button("Compare models on every expiry", key="lab_mc_all_run", width="stretch"):
        with st.spinner(f"Scoring {len(models)} model(s) on {n_all} expiries..."):
            scores, skipped = compare_models_by_expiry(
                chain, models, rate=rate, min_days=MIN_FIT_DAYS, max_days=max_days, sabr_beta=float(beta)
            )
        st.session_state[K_MC_ALL] = {
            "data_key": ctx["data_key"],
            "settings_key": all_key,
            "provenance": prov,
            "scores": scores,
            "skipped": skipped,
            "models": list(models),
            "manifest": _manifest("model_comparison_all", prov, {"filters": ctx["filters"]}, all_settings),
        }
    stored_all = st.session_state.get(K_MC_ALL)
    if not stored_all:
        return
    _stale_notice(stored_all, ctx["data_key"], all_key, "every-expiry comparison")
    scores = stored_all["scores"]
    if scores.empty:
        st.warning("No expiry could be scored.")
    else:
        st.dataframe(summarise_comparison(scores), hide_index=True, width="stretch")
        ms = stored_all["models"]
        for i, a in enumerate(ms):
            for b in ms[i + 1 :]:
                st.markdown(f"- {describe_pairwise(a, b, pairwise_wins(scores, a, b))}")
        with st.expander(f"Scores per expiry ({scores['expiry'].nunique()} expiries)"):
            pivot = scores.pivot_table(index=["expiry", "days"], columns="model", values="rmse_cv_bp")
            st.dataframe(pivot.reset_index(), hide_index=True, width="stretch")
    if not stored_all["skipped"].empty:
        with st.expander(f"{len(stored_all['skipped'])} expiry(ies) not scored, with the reason"):
            st.dataframe(stored_all["skipped"], hide_index=True, width="stretch")
    _downloads(
        "models_every_expiry",
        stored_all["manifest"],
        {"scores": scores, "summary": summarise_comparison(scores), "skipped": stored_all["skipped"]},
        key="lab_mc_all_dl",
    )


def render() -> None:
    """Draw the ANALYTICS lab."""

    init_state()
    st.markdown(LAB_CSS, unsafe_allow_html=True)
    st.markdown("## / VOL ANALYTICS LAB")
    tabs = st.tabs([f"/ {label.upper()}" for label in ANALYTICS_TABS])
    with tabs[0]:
        ctx = _render_market_data_tab()
    with tabs[1]:
        _render_calibration_tab(ctx)
    with tabs[2]:
        _render_walkforward_tab()
    with tabs[3]:
        _render_model_comparison_tab(ctx)
