"""ANALYTICS lab: option-chain data with lineage, production SVI
calibration with smile risk, an arbitrage-free SSVI surface with Dupire
local vol, walk-forward validation against benchmarks with daily parameter
stability, smile-model comparison, and the same research tools on rates:
curve histories (forecast walk-forward, regimes, curve-model comparison)
and dated swaption / cap vol smiles (:mod:`derivx.ui.lab_vol`,
:mod:`derivx.ui.lab_rates`).

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

Discounting (AL-14): premiums are discounted from the trade date to each
expiry with the US Treasury curve of the trade date (the latest published
on or before it, from the bundled snapshot or the download cache,
bootstrapped by :mod:`derivx.rates.curves`), or, for European-style index
options, with the discount factor put-call parity implies per expiry
(Treasury curve where parity fails its gates), or a flat input rate
(:mod:`derivx.vol.discounting`). The hygiene table shows each expiry's DF,
effective rate and where it came from.

Units: strikes and premiums in the underlying's currency per share, vols
as per-annum decimals (``*_bp`` columns and chart axes: vol basis points,
1 bp = 0.0001), ``k = ln(K/F)`` on put-call-parity forwards, ``tau`` in
ACT/365F years, rates continuously compounded, horizons in exchange
trading days.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.calibration.production import ChainCalibration, calibrate_chain
from derivx.data import rates_history as rh
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
from derivx.data.snapshots import origins_needed, scorable_origins
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
from derivx.ui import lab_history, lab_rates, lab_vol
from derivx.ui.lab_common import banner, downloads, fingerprint, legend, stale_notice, stats
from derivx.ui.plot_limits import axis_limits, bound_axes
from derivx.ui.state import PERSIST
from derivx.ui.theme import ACCENT, FOREGROUND, PLOTLY_2D_CONFIG, SERIES, dark_layout
from derivx.validation.backtest import PREDICTORS, WalkForwardResult, WalkForwardSettings, run_walk_forward
from derivx.validation.manifest import build_manifest, jsonable, results_zip  # noqa: F401 (results_zip: re-export)
from derivx.vol.discounting import Discounting, european_style, treasury_discount
from derivx.vol.quotes import REASONS
from derivx.vol.svi import svi_iv

ANALYTICS_TABS = [
    "Market Data",
    "Production Calibration",
    "Vol Surface",
    "Walk-Forward Validation",
    "Model Comparison",
    "Rates Curves",
    "Rates Vol History",
]
UPLOAD_TYPES = ["csv", "gz", "parquet", "pq", "txt"]
PULL_BUCKET_S = 300  # Yahoo pulls are shared between sessions within one 5-minute bucket
MIN_FIT_DAYS = 2.0  # expiries closer than this are not fitted or scored
PRICER_TENORS_DAYS = (7.0, 30.0, 61.0, 91.0, 182.0, 365.0)  # expiries handed to the EQUITY page's MARKET smile
SPOT_DRIFT_WARN = 1e-3  # warn when spot moved more than 0.1% between the first and last expiry response of a fetch

DISCOUNT_CHOICES: Dict[str, str] = {
    "auto": "Auto: put-call parity for European index options, else the US Treasury curve",
    "curve": "US Treasury curve of the trade date",
    "parity": "Put-call parity per expiry (European options; Treasury curve where it fails)",
    "rate": "Flat rate (input)",
}

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
/* labels, captions, uploaders, expanders and buttons come from the app theme (theme.APP_CSS); long lab button
   labels may wrap */
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


_key = fingerprint
_downloads = downloads
_legend = legend
_stats = stats


def _day(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _when(iso: Optional[str]) -> str:
    if not iso:
        return "fetch time not recorded"
    ts = pd.Timestamp(iso)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def _banner(prov: Mapping[str, Any], label: str = "ACTIVE DATASET", extra: Sequence[str] = (), kind: str = "") -> None:
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
    banner(label, parts, kind, synthetic=labelled or bool(synth_rows))


def _stale_notice(stored: Mapping[str, Any], data_key: str, settings_key: str, what: str) -> bool:
    """Warn when a stored result no longer matches the data or settings;
    True when the data changed (the result is STALE)."""

    return stale_notice(
        stored,
        data_key,
        settings_key,
        what,
        lambda: _banner(stored["provenance"], label=f"STALE {what.upper()}: COMPUTED ON", kind="stale"),
    )


def _manifest(kind: str, prov: Mapping, inputs: Mapping, model: Mapping, seed=None) -> Dict[str, Any]:
    return build_manifest(kind, inputs=inputs, data=prov, model=model, seed=seed)


def _filter_breakdown(chain: pd.DataFrame, filters: Mapping[str, Any]) -> str:
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

    def __init__(self, chain: pd.DataFrame, status: Dict[str, Any]):
        super().__init__(status.get("reason", ""))
        self.chain, self.status = chain, status


@st.cache_data(ttl=2 * PULL_BUCKET_S, max_entries=16, show_spinner=False)
def _pull_shared(ticker: str, max_expiries: int, bucket: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The current chain, shared by sessions asking for the same ticker and
    expiries within one bucket of ``PULL_BUCKET_S`` seconds (``bucket`` is
    the bucket number, part of the cache key). Only complete pulls are
    cached; the rows carry their real fetch times."""

    result = fetch_option_chain(ticker, max_expiries=max_expiries, min_days=MIN_FIT_DAYS)
    status = result.status.as_dict()
    if result.status.status != "ok":
        raise _NotCached(result.chain, status)
    return result.chain, status


def _pull(ticker: str, max_expiries: int, force: bool) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
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


def _active() -> Optional[Dict[str, Any]]:
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


class _ResultStore:
    """A small thread-safe LRU of results keyed by the fingerprint of their
    data and settings, shared by the sessions of this server (see
    :func:`_calibration_store`). Stored results are read, never mutated."""

    def __init__(self, max_entries: int) -> None:
        self._items: "OrderedDict[str, Any]" = OrderedDict()
        self._max = int(max_entries)
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)


@st.cache_resource(show_spinner=False)
def _calibration_store() -> _ResultStore:
    """Calibrations by fingerprint of (data, filters, discounting, settings):
    pressing Calibrate again with inputs already calibrated, in this session
    or another, shows the stored result at once. st.cache_data cannot hold
    these runs because they draw a progress bar while they compute."""

    return _ResultStore(max_entries=16)


def _calibration_key(data_key: str, settings_key: str, discount: Discounting) -> str:
    """Key of a calibration in :func:`_calibration_store`. ``data_key`` and
    ``settings_key`` name the discounting by :meth:`Discounting.describe`
    (mode, curve dates, labels, pillar counts), which two different curves
    of the same day and source share (a store refreshed with revised yields);
    a result shared across sessions must also match the discount factors
    themselves, so the key adds the full repr of ``discount`` (every pillar
    time and DF, exact)."""

    return _key(data_key, settings_key, repr(discount))


@st.cache_data(max_entries=8, show_spinner=False)
def _hygiene(chain: pd.DataFrame, discount: Discounting) -> pd.DataFrame:
    """Per-expiry hygiene report (content-keyed: the same quotes and
    discounting give the same report in any session)."""

    return hygiene_report(chain, discount=discount)


# -- discounting -------------------------------------------------------------------


@st.cache_data(max_entries=32, show_spinner=False)
def _treasury_curves(signature: Tuple[Tuple[str, str], ...], days: Tuple[str, ...]):
    """(trade date, Treasury curve) pairs for ``days`` (ISO dates); dates
    without a curve within a week are left out."""

    store = lab_rates.rates_store(signature)
    out = []
    for day in days:
        curve = treasury_discount(store, pd.Timestamp(day))
        if curve is not None:
            out.append((day, curve))
    return tuple(out)


def _discount_controls(prefix: str, ticker: Optional[str], trade_dates: Sequence) -> Tuple[Discounting, List[str]]:
    """The discounting widgets of a tab (keys ``{prefix}_discount`` and
    ``{prefix}_rate``): returns the :class:`Discounting` and notes to show
    (warnings start with '!')."""

    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox(
        "Discount premiums with",
        list(DISCOUNT_CHOICES),
        format_func=DISCOUNT_CHOICES.get,
        key=f"{prefix}_discount",
        persist_state=PERSIST,
        help="Vols are re-inverted from bid/ask mids with Black-76 on the parity forward and this discount factor "
        "(trade date to expiry). Put-call parity DFs are only valid for European options: SPY, QQQ and single "
        "stocks list American options, whose early-exercise premium biases the parity slope.",
    )
    rate_pct = c2.number_input(
        "Flat / fallback rate r (% cc)",
        min_value=-5.0,
        max_value=20.0,
        value=0.0,
        step=0.25,
        key=f"{prefix}_rate",
        persist_state=PERSIST,
        help="Continuously compounded. Used in 'Flat rate' mode, and for a trade date the rates store has no "
        "Treasury curve for (an input, not market data).",
    )
    rate = float(rate_pct) / 100.0
    notes: List[str] = []
    european = european_style(ticker)
    mode = ("parity" if european else "curve") if choice == "auto" else choice
    if mode == "rate":
        return Discounting.flat(rate), [f"flat rate {rate:.2%} cc (an input, not market data)"]
    if mode == "parity" and not european:
        notes.append(
            f"!{ticker or 'This underlying'} is not a known European-style index: its options are probably "
            "American, and their early-exercise premium biases the put-call-parity discount factor (the gates "
            "fall back to the Treasury curve where the implied rate is more than 100 bp from it)."
        )
    days = tuple(sorted({pd.Timestamp(d).date().isoformat() for d in trade_dates}))
    signature = rh.cache_signature()
    try:
        curves = _treasury_curves(signature, days)
    except ValueError as exc:  # a broken cached curve must not take the page down
        curves = ()
        notes.append(f"!The Treasury curve could not be bootstrapped ({exc}).")
    missing = [d for d in days if d not in {k for k, _ in curves}]
    if missing:
        notes.append(
            f"!No US Treasury curve within a week before {', '.join(missing[:4])}"
            f"{'...' if len(missing) > 4 else ''}: those trade dates are discounted at the flat fallback rate "
            f"{rate:.2%}. Refresh the rates data on the RATES page (LIVE mode) to extend the curve history."
        )
    if not curves and mode == "curve":
        return Discounting.flat(rate), notes
    if curves:
        notes.append(f"{curves[-1][1].label}" + (f" (+{len(curves) - 1} other trade dates)" if len(curves) > 1 else ""))
    return Discounting(mode, rate, curves), notes


def _show_discount_notes(notes: Sequence[str]) -> None:
    for note in notes:
        if note.startswith("!"):
            st.warning(note[1:])
    plain = [n for n in notes if not n.startswith("!")]
    if plain:
        st.caption("Discounting: " + "; ".join(plain) + ".")


def _load_controls() -> None:
    st.markdown("#### / LOAD DATA")
    c_sample, c_upload, c_pull = st.columns(3)
    with c_sample:
        st.markdown("**Bundled sample**")
        st.caption("SPY, Yahoo Finance delayed quotes, fetched 2026-10-20 20:59 UTC after the close, unfiltered.")
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
            help=f"Only expiries at least {MIN_FIT_DAYS:g} days away count (after the close Yahoo still lists the "
            "day's expired one). Yahoo serves only the current, delayed chain.",
        )
        force = st.checkbox("Bypass the 5-minute shared cache", value=False, key="lab_force", persist_state=PERSIST)
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
    st.markdown("**Collected history** (one chain per trading day)")
    lab_history.render_market_data_loader(_activate)
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


def _fetch_status_panel(status: Optional[Mapping[str, Any]]) -> None:
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


def _publish_to_pricer(
    ds: ChainDataset, prov: Mapping, filtered: pd.DataFrame, report: pd.DataFrame, filters, discount: Discounting
) -> None:
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
        "discounting": discount.describe(),
    }
    st.session_state["market_chain_discount"] = discount  # the EQUITY page's MARKET smile discounts the same way


def _render_market_data_tab() -> Optional[Dict[str, Any]]:
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
    f1, f2, f3, f4 = st.columns(4)
    min_volume = f1.number_input("Min volume", min_value=0, value=10, step=5, key="lab_minvol", persist_state=PERSIST)
    min_mid = f2.number_input("Min mid", min_value=0.0, value=0.10, step=0.05, key="lab_minmid", persist_state=PERSIST)
    max_spread = f3.number_input(
        "Max rel spread", min_value=0.01, value=0.40, step=0.05, key="lab_maxspread", persist_state=PERSIST
    )
    no_cap = f4.checkbox("No spread cap", value=False, key="lab_nocap", persist_state=PERSIST)
    filters = {
        "min_volume": int(min_volume),
        "min_mid": float(min_mid),
        "max_spread": None if no_cap else float(max_spread),
    }
    st.markdown("#### / DISCOUNTING")
    discount, disc_notes = _discount_controls("lab", ds.ticker, ds.trade_dates)
    _show_discount_notes(disc_notes)
    try:
        filtered = apply_liquidity_filters(
            raw, min_volume=filters["min_volume"], max_spread=filters["max_spread"], min_mid=filters["min_mid"]
        )
    except ValueError as exc:
        st.session_state["market_chain"] = raw.iloc[0:0]
        st.error(f"The liquidity filters cannot be applied to this chain: {exc}")
        return None
    data_key = _key(prov["data_sha256"], filters, discount.describe())
    if filtered.empty:
        st.session_state["market_chain"] = filtered
        st.warning(
            "The liquidity filters removed every quote: loosen them to continue. Quotes failing each rule: "
            + _filter_breakdown(raw, filters)
            + "."
        )
        return None
    with st.spinner("Cleaning quotes (hygiene)..."):
        report = _hygiene(filtered, discount)
    _publish_to_pricer(ds, prov, filtered, report, filters, discount)
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
        "put-call parity needs them) -> one OTM quote per strike after hygiene. Spot is Yahoo's regular-market "
        "price from the same response as the quotes."
    )
    st.markdown("#### / QUOTE HYGIENE BY EXPIRY")
    st.dataframe(report, hide_index=True, width="stretch")
    st.caption(
        "Removed, by reason: "
        + "; ".join(f"{k}: {v}" for k, v in REASONS.items() if k != "outside_band")
        + ". forward_method: parity = put-call parity on the slice's mids; interpolated = ln F linear in T from the "
        f"other parity forwards; carry = spot (no parity). discount_factor: trade date to expiry; rate_pct: its cc "
        "rate -ln(DF)/tau in %; discount_method: curve (Treasury curve of the trade date), parity (put-call parity "
        "regression, with parity_rate_pct and its standard error parity_se_bp), rate (flat input); "
        f"discount_source says why. {len(usable)} expiries have >= {MIN_FIT_DAYS:g} days and >= 8 OTM quotes."
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
        json.dumps(
            jsonable(
                {
                    "dataset": prov,
                    "filters": filters,
                    "discounting": discount.describe(),
                    "fetch_status": active.get("status"),
                }
            ),
            indent=2,
        ),
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
        "discount": discount,
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
    fig.update_layout(legend=_legend(), margin=dict(l=60, r=20, t=30, b=60))
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


def _render_calibration_tab(ctx: Optional[Dict[str, Any]]) -> None:
    st.markdown("### / PRODUCTION CALIBRATION")
    if ctx is None:
        st.info("No usable chain: load one on the Market Data tab, or loosen its liquidity filters, to calibrate.")
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
        "discounting": ctx["discount"].describe(),
    }
    settings_key = _key(settings)
    if st.button("Calibrate selected expiries", key="lab_cal_run", width="stretch", disabled=not sel):
        cache_key = _calibration_key(ctx["data_key"], settings_key, ctx["discount"])
        store = _calibration_store()
        result = store.get(cache_key)
        if result is None:
            subset = chain[pd.to_datetime(chain["expiry"]).dt.normalize().isin(set(sel))]
            bar = st.progress(0.0, text=f"Calibrating {len(sel)} SVI slice(s)...")

            def show(done: int, total: int, label: str) -> None:
                text = "Calibration done." if done >= total else f"Calibrating {label} ({done + 1} of {total})..."
                bar.progress(min(done / max(total, 1), 1.0), text=text)

            result = calibrate_chain(
                subset,
                enforce_arbitrage=enforce,
                std_band=std_band,
                moneyness_band=band,
                discount=ctx["discount"],
                min_days=MIN_FIT_DAYS,
                cv_folds=5,
                progress=show,
            )
            bar.empty()
            store.put(cache_key, result)
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
    result = stored["result"]  # a ChainCalibration, as stored above
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
        "Black-76 on each expiry's discount factor (rate_pct: its cc rate in %; discount_method: curve, parity or "
        "rate, see the hygiene table). min_g: Gatheral butterfly density minimum (>= 0 is arbitrage-free); "
        "calendar_ok: total variance non-decreasing vs the previous fitted expiry."
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
    lab_vol.smile_risk_section(slc, key=f"lab_risk_{pick}")
    stem = f"calibration_{(prov.get('ticker') or 'chain').lower()}_{prov['trade_dates'][0]}"
    _downloads(stem, stored["manifest"], tables, key="lab_cal_dl")


# -- walk-forward ----------------------------------------------------------------


def _use_collected(ds: ChainDataset, info: Dict[str, Any]) -> None:
    st.session_state[K_WF_DATA] = {"dataset": ds, "origin": "collected", "seed": None, "history": info}
    st.session_state[K_WF_MSG] = (
        "success",
        f"Loaded {len(ds.chains)} collected session(s) of {ds.ticker} from {info.get('source')}.",
    )


def _wf_controls() -> None:
    source = st.radio(
        "History",
        [lab_history.SOURCE_LABEL, "Upload daily snapshots", "SYNTHETIC demo history"],
        horizontal=True,
        key="lab_wf_source",
        persist_state=PERSIST,
        help="One chain per trading day, each saved on its own day. Yahoo serves only the current chain, so a "
        "history is collected (collect-data workflow) or uploaded (tools/download_spy_chains.py); SYNTHETIC is "
        "generated, not market data.",
    )
    if source == lab_history.SOURCE_LABEL:
        lab_history.render_walkforward_source(_use_collected)
    elif source == "Upload daily snapshots":
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
    fig.update_layout(legend=_legend(), margin=dict(l=60, r=20, t=30, b=60))
    dates = pd.to_datetime(d["target"])
    pad = max((dates.max() - dates.min()) * 0.05, pd.Timedelta(days=1))
    bound_axes(
        fig,
        xaxis={"min": (dates.min() - pad).isoformat(), "max": (dates.max() + pad).isoformat()},
        yaxis=axis_limits(np.concatenate(values), pad=0.1, floor=0.0),
    )
    return fig


def _prepared_history(ds: ChainDataset, filters: Mapping[str, Any]) -> Dict[pd.Timestamp, pd.DataFrame]:
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
    _wf_controls()
    entry = st.session_state.get(K_WF_DATA)
    if not entry:
        st.info(
            "Use the collected history once it is long enough, upload daily snapshots, or generate the SYNTHETIC "
            "demo history to run a walk-forward test."
        )
        return
    ds: ChainDataset = entry["dataset"]
    prov = ds.provenance()
    if ds.warnings:
        with st.expander(f"{len(ds.warnings)} data note(s) on these files"):
            for w in ds.warnings:
                st.markdown(f"- {w}")
    gaps = missing_trading_days(ds.trade_dates)
    collected = entry.get("origin") == "collected"
    _banner(
        prov,
        label="COLLECTED HISTORY" if collected else "HISTORY",
        extra=(
            [f"missing trading days: {', '.join(_day(g) for g in gaps[:6])}{'...' if len(gaps) > 6 else ''}"]
            if gaps
            else ["no missing trading days"]
        )
        + (
            [f"from {entry['history'].get('source')}, index sha256 {str(entry['history'].get('index_sha256'))[:12]}"]
            if collected and entry.get("history")
            else []
        ),
    )
    if len(ds.chains) < 2:
        st.warning(
            f"The history holds {len(ds.chains)} snapshot: a forecast needs at least two trading days. A single "
            "snapshot (like the bundled sample) supports calibration and model comparison, not a walk-forward."
        )
        return
    st.markdown("#### / FILTERS AND SETTINGS")
    f1, f2, f3 = st.columns(3)
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
    discount, disc_notes = _discount_controls("lab_wf", ds.ticker, ds.trade_dates)
    _show_discount_notes(disc_notes)
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
        horizon=horizon,
        window=window,
        rate=discount.rate,
        discount=discount,
        max_days=max_days,
        enforce_arbitrage=enforce,
        min_days=MIN_FIT_DAYS,
    )
    data_key = _key(prov["data_sha256"], filters, discount.describe())
    settings_key = _key(settings.as_dict())
    waiting = ""
    if collected:
        have, need = scorable_origins(ds.trade_dates, horizon), origins_needed(settings.min_regime_days)
        if have < need:
            waiting = (
                f"{len(ds.chains)} collected session(s) give {have} origin day(s) with the session {horizon} trading "
                f"day(s) later; the walk-forward on collected history needs {need} (three per vol-regime tercile). "
                "Pick a shorter horizon or wait for more sessions."
            )
            st.warning(waiting)
    run = st.button("Run walk-forward test", key="lab_wf_run", width="stretch", disabled=bool(waiting))
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
                "walk_forward",
                prov,
                {"filters": filters, **({"collected_history": entry["history"]} if entry.get("history") else {})},
                settings.as_dict(),
                seed=entry.get("seed"),
            ),
        }
    stored = st.session_state.get(K_WF)
    if stored:
        _walkforward_results(stored, data_key, settings_key, prov)
    lab_vol.stability_section(lambda: _prepared_history(ds, filters), discount, max_days, data_key, prov, _banner)


def _walkforward_results(stored: Mapping[str, Any], data_key: str, settings_key: str, prov: Mapping) -> None:
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
    with st.expander(f"Per-slice scores and origin SVI parameters ({len(result.slices)} origin x expiry rows)"):
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
    fig.update_layout(legend=_legend(), margin=dict(l=60, r=20, t=30, b=60))
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


def _render_model_comparison_tab(ctx: Optional[Dict[str, Any]]) -> None:
    st.markdown("### / MODEL COMPARISON")
    if ctx is None:
        st.info("No usable chain: load one on the Market Data tab, or loosen its liquidity filters, to compare models.")
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
    discount: Discounting = ctx["discount"]
    row = ok[pd.to_datetime(ok["expiry"]) == expiry].iloc[0]
    settings = {
        "expiry": _day(expiry),
        "models": list(models),
        "sabr_beta": float(beta),
        "cv_folds": 5,
        "std_band": 3.0,
        "min_points": 8,
        "discounting": discount.describe(),
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
                    min_days=MIN_FIT_DAYS,
                    hygiene={
                        "forward": float(row["forward"]),
                        "forward_source": f"{row['forward_method']} (chain)",
                        "discount_factor": float(row["discount_factor"]),
                        "discount_source": f"{row['discount_method']}: {row['discount_source']}",
                    },
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
        comp = stored["comp"]  # a ModelComparison, as stored above
        table = comp.table
        st.dataframe(table, hide_index=True, width="stretch")
        failed = table[table["status"].astype(str).str.startswith("failed")]
        for r in failed.itertuples():
            st.warning(f"{r.model.upper()} could not be fitted and is not plotted: {r.status}")
        attrs = table.attrs
        st.caption(
            f"Slice {attrs.get('expiry')} (trade date {attrs.get('trade_date')}, "
            f"tau {attrs.get('tau', float('nan')):.4f}y), "
            f"forward {attrs.get('forward', float('nan')):.4f} ({stored.get('forward_method')}), "
            f"discount factor {attrs.get('discount_factor', float('nan')):.6f} "
            f"({100 * attrs.get('rate', float('nan')):.3f}% cc). One OTM quote per "
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
                chain, models, discount=discount, min_days=MIN_FIT_DAYS, max_days=max_days, sabr_beta=float(beta)
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
        lab_vol.render_surface_tab(ctx, _banner)
    with tabs[3]:
        _render_walkforward_tab()
    with tabs[4]:
        _render_model_comparison_tab(ctx)
    with tabs[5]:
        lab_rates.render_curves_tab()
    with tabs[6]:
        lab_rates.render_vol_tab()
