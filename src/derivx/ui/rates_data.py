"""Rates page market data: the sidebar DATA SOURCE, the history store the
page prices from, swap-quote prefills with per-row provenance and the
source/as-of badges drawn on curve charts.

Modes:

* ``BUNDLED SNAPSHOT`` - the committed snapshot and history sample, no
  network (the default, and the fallback of the other modes);
* ``LIVE (CACHED)`` - FRED, NY Fed, ECB and BoE downloads with short
  timeouts, kept in the disk cache for an hour; anything that fails falls
  back to the cache or the snapshot with a visible notice. The slow
  Treasury CSV downloads in the background;
* ``UPLOAD`` - the user's swap quotes (and optionally a Treasury par
  curve) on top of the snapshot. Parsed uploads live in session state, so
  they survive switching to another page and back until cleared.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivx.data import rates_history as rh
from derivx.data.rates_history import HistoryStore
from derivx.rates.curves import CONVENTIONS
from derivx.rates.examples import sofr_example_rows

MODES = ("BUNDLED SNAPSHOT", "LIVE (CACHED)", "UPLOAD")
LIVE_MAX_AGE = 3600.0  # seconds a download is reused before it is fetched again
LIVE_TIMEOUT = 5.0  # seconds per request on the page (no retries: the cache and the snapshot are the fallback)
UPLOAD_KEY = "rates_uploaded_quotes"
TREASURY_UPLOAD_KEY = "rates_uploaded_treasury"
STORE_KEY = "rates_store"
BADGE_KEY = "rates_badges"
ILLUSTRATIVE = "ILLUSTRATIVE (not market data)"
SWAP_KINDS = ("DEPO", "OIS", "IRS")
FIXING_LABELS = {
    "SOFR": "SOFR (NY Fed)",
    "EFFR": "EFFR (NY Fed)",
    "OBFR": "OBFR (NY Fed)",
    "ESTR": "ESTR (ECB)",
    "SONIA": "SONIA (Bank of England via FRED)",
}
PROXY_DATASET = {"EUR": "EUR_AAA_ECB"}  # zero curves used as labelled proxies where no OIS curve is free
REAL_DATASET = {"GBP": "GBP_OIS_BOE"}  # real OIS curves


@dataclass
class RatesData:
    """What the sidebar chose: the store to price from and what to tell the
    user about it (``notices`` are ``(level, text)``)."""

    mode: str
    store: HistoryStore
    notices: List[Tuple[str, str]] = field(default_factory=list)
    uploaded: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class QuotePrefill:
    """Swap quotes to start the editor from: ``table`` has ``instrument``,
    ``tenor``, ``rate`` (percent) and ``source`` (per row). ``kind`` is
    ``real``, ``proxy``, ``illustrative`` or ``upload``."""

    table: pd.DataFrame
    label: str
    kind: str
    key: str
    notes: Tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Sticky sidebar state
# --------------------------------------------------------------------------


def sticky(key: str, options: Sequence[object]) -> None:
    """Restore a widget's last value when Streamlit dropped it because the
    widget was not drawn in the previous run (another page was open)."""

    shadow = f"_keep_{key}"
    if key not in st.session_state and shadow in st.session_state and st.session_state[shadow] in list(options):
        st.session_state[key] = st.session_state[shadow]


def remember(key: str) -> None:
    if key in st.session_state:
        st.session_state[f"_keep_{key}"] = st.session_state[key]


# --------------------------------------------------------------------------
# Stores
# --------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def _refresh(day: dt.date) -> rh.LiveRefresh:
    """Live refresh, shared by all sessions for ten minutes (the downloads
    themselves are reused for an hour)."""

    return rh.refresh_live(day, max_age=LIVE_MAX_AGE, timeout=LIVE_TIMEOUT, retries=0)


@st.cache_resource(show_spinner=False, max_entries=4)
def _cached_store(signature: Tuple[Tuple[str, str], ...], label: str) -> HistoryStore:
    """Snapshot plus every cached download (rebuilt when the cache changes)."""

    return rh.build_store([rh.bundled_frame()] + rh.cached_frames(), label)


def _with_frames(base: HistoryStore, frames: List[pd.DataFrame], token: str) -> HistoryStore:
    """``base`` plus extra frames (uploads), memoised per session."""

    if not frames:
        return base
    memo = st.session_state.setdefault("_rates_store_memo", {})
    key = (id(base), token)
    if key not in memo:
        memo.clear()
        memo[key] = rh.build_store([base.frame] + frames, f"{base.label} + {token}")
    return memo[key]


def _live_store() -> Tuple[HistoryStore, List[Tuple[str, str]]]:
    today = dt.datetime.now(dt.timezone.utc).date()
    with st.spinner("Updating live rates data (FRED, NY Fed, ECB, BoE; cached for an hour)..."):
        result = _refresh(today)
    notices = [("warning", n) if "unavailable" in n or "failed" in n else ("info", n) for n in result.notices]
    background = rh.treasury_in_background(today.year, max_age=LIVE_MAX_AGE)
    if background:
        notices.append(("info", background))
    signature = rh.cache_signature()
    store = _cached_store(signature, f"live cache {hashlib.sha1(repr(signature).encode()).hexdigest()[:8]}")
    if result.frames:
        store = _with_frames(store, result.frames, "uncached downloads")
        notices.append(("warning", "The download cache is not writable: live data is kept for this session only."))
    if not any(result.ok.values()):
        notices.insert(0, ("warning", "No live source answered: showing the cache and the bundled snapshot."))
    else:
        answered = sum(result.ok.values())
        notices.insert(
            0,
            ("info", f"LIVE: {answered} of {len(result.ok)} sources up to date (downloads are reused for an hour; "
                     "see 'Market data' below for what each curve comes from)."),
        )  # fmt: skip
    return store, notices


def widget_key(name: str) -> str:
    """Key of an upload widget; it changes when the upload is cleared, so the
    widget comes back empty instead of re-reading the file it still holds."""

    return f"{name}_{st.session_state.get(f'_gen_{name}', 0)}"


def clear_upload(store_key: str, widget: str, item: Optional[str] = None) -> None:
    """Button callback: forget a parsed upload (one ``item`` of a dict
    upload if given) and reset its widget."""

    if item is None:
        st.session_state.pop(store_key, None)
    else:
        st.session_state.get(store_key, {}).pop(item, None)
    st.session_state[f"_gen_{widget}"] = st.session_state.get(f"_gen_{widget}", 0) + 1


def _upload_sidebar() -> Tuple[Optional[Dict[str, object]], List[pd.DataFrame], List[Tuple[str, str]]]:
    """Upload widgets; parsed uploads persist in session state. Returns
    (uploaded quotes, extra store frames, notices)."""

    notices: List[Tuple[str, str]] = []
    file = st.sidebar.file_uploader(
        "SWAP QUOTES (CSV: instrument, tenor, rate)", type=["csv"], key=widget_key("rates_upload")
    )
    if file is not None:
        payload = file.getvalue()
        sha = hashlib.sha256(payload).hexdigest()
        current = st.session_state.get(UPLOAD_KEY)
        if current is None or current.get("sha") != sha:
            table, error = parse_quote_upload(payload)
            if error:
                notices.append(("error", f"{file.name}: {error}"))
            else:
                st.session_state[UPLOAD_KEY] = {"name": file.name, "sha": sha, "table": table}
    treasury = st.sidebar.file_uploader(
        "TREASURY PAR CURVE (optional; Treasury CSV layout)", type=["csv"], key=widget_key("rates_upload_treasury")
    )
    if treasury is not None:
        payload = treasury.getvalue()
        sha = hashlib.sha256(payload).hexdigest()
        current = st.session_state.get(TREASURY_UPLOAD_KEY)
        if current is None or current.get("sha") != sha:
            try:
                frame = rh.treasury_upload(payload.decode("utf-8-sig"), treasury.name)
            except (ValueError, UnicodeDecodeError) as exc:
                notices.append(("error", f"{treasury.name}: {exc}"))
            else:
                st.session_state[TREASURY_UPLOAD_KEY] = {"name": treasury.name, "sha": sha, "frame": frame}
    uploaded = st.session_state.get(UPLOAD_KEY)
    frames = []
    if uploaded is not None:
        st.sidebar.caption(f"Using uploaded file {uploaded['name']} ({len(uploaded['table'])} quotes).")
        st.sidebar.button(
            "CLEAR UPLOADED QUOTES", key="rates_upload_clear", on_click=clear_upload, args=(UPLOAD_KEY, "rates_upload")
        )
    else:
        notices.append(("info", "UPLOAD: no quotes file yet; the table starts from the snapshot prefill."))
    treasury_up = st.session_state.get(TREASURY_UPLOAD_KEY)
    if treasury_up is not None:
        st.sidebar.caption(f"Using uploaded Treasury curve {treasury_up['name']}.")
        st.sidebar.button(
            "CLEAR UPLOADED TREASURY CURVE",
            key="rates_upload_treasury_clear",
            on_click=clear_upload,
            args=(TREASURY_UPLOAD_KEY, "rates_upload_treasury"),
        )
        frames.append(treasury_up["frame"])
    return uploaded, frames, notices


def parse_quote_upload(payload: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Uploaded quotes CSV -> (table with instrument, tenor, rate[, source]),
    or (None, reason). Rates keep the file's units; the page converts them."""

    try:
        raw = pd.read_csv(io.BytesIO(payload))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        return None, f"could not read the file ({exc})"
    raw.columns = [str(col).strip().lower() for col in raw.columns]
    if "kind" in raw.columns and "instrument" not in raw.columns:
        raw = raw.rename(columns={"kind": "instrument"})
    missing = {"instrument", "tenor", "rate"} - set(raw.columns)
    if missing:
        return None, f"missing columns: {', '.join(sorted(missing))}"
    table = raw[["instrument", "tenor", "rate"]].copy()
    table["source"] = raw["source"].astype(str) if "source" in raw.columns else "uploaded"
    return table.reset_index(drop=True), None


def sidebar() -> RatesData:
    """Draw DATA SOURCE (and the upload widgets in UPLOAD mode); returns
    the store to price from. The store is also left in session state for
    the SCENARIOS tab."""

    sticky("rates_source", MODES)
    mode = st.sidebar.radio(
        "DATA SOURCE",
        MODES,
        key="rates_source",
        help=(
            "BUNDLED: the committed snapshot (no network). LIVE: FRED, NY Fed, ECB and BoE downloads, cached for "
            "an hour, falling back to the snapshot. UPLOAD: your quotes CSV (and optional Treasury curve)."
        ),
    )
    remember("rates_source")
    notices: List[Tuple[str, str]] = []
    uploaded = None
    store = rh.bundled_store()
    if mode == MODES[1]:
        store, notices = _live_store()
    elif mode == MODES[2]:
        uploaded, frames, notices = _upload_sidebar()
        if frames:
            store = _with_frames(store, frames, f"upload {str(st.session_state[TREASURY_UPLOAD_KEY]['sha'])[:12]}")
    st.session_state[STORE_KEY] = store
    return RatesData(mode, store, notices, uploaded)


def active_store() -> HistoryStore:
    """The store the CURVE tab priced from (the snapshot if none yet)."""

    return st.session_state.get(STORE_KEY) or rh.bundled_store()


# --------------------------------------------------------------------------
# Quote prefill
# --------------------------------------------------------------------------


def _percent(rate: float) -> float:
    return round(100.0 * rate, 6)


def _fixing_row(store: HistoryStore, currency: str, asof: dt.date) -> Tuple[Optional[dict], Optional[str]]:
    name = rh.CURRENCY_FIXING.get(currency)
    got = store.fixing(name, asof) if name else None
    if got is None:
        return None, None
    day, rate, _ = got
    label = f"{FIXING_LABELS.get(name, name)} fixing for {day}"
    return {"instrument": "DEPO", "tenor": "ON", "rate": _percent(rate), "source": label}, label


def _illustrative_rows() -> List[dict]:
    return [
        {"instrument": r["kind"], "tenor": r["tenor"], "rate": _percent(r["rate"]), "source": ILLUSTRATIVE}
        for r in sofr_example_rows()
    ]


def swap_prefill(store: HistoryStore, conventions: str, asof: dt.date) -> QuotePrefill:
    """Starting quotes for the swap curve on ``asof``: a real OIS curve where
    one is free (GBP: the Bank of England's fitted SONIA curve as par OIS
    rates), a labelled proxy for EUR (par OIS rates derived from the ECB AAA
    government spot curve), otherwise the illustrative SOFR set. The
    currency's overnight fixing, where the store has one, is the ON quote."""

    currency = CONVENTIONS[conventions].currency
    fixing, fixing_label = _fixing_row(store, currency, asof)
    notes: List[str] = []
    rows: List[dict] = []
    kind, label = "illustrative", ""
    dataset = REAL_DATASET.get(currency) or PROXY_DATASET.get(currency)
    day = store.latest_on_or_before(dataset, asof, "zero_cc") if dataset and store.has(dataset) else None
    if day is not None:
        try:
            par = store.ois_par(dataset, day)
        except ValueError as exc:
            par = {}
            notes.append(f"{rh.dataset_info(dataset).title} on {day} could not be converted ({exc})")
        if par:
            info = rh.dataset_info(dataset)
            kind = "real" if dataset in REAL_DATASET.values() else "proxy"
            tag = f"{info.title} {day}, par OIS derived"
            source = tag if kind == "real" else f"PROXY: {tag} (government curve, not OIS quotes)"
            rows = [{"instrument": "OIS", "tenor": t, "rate": _percent(r), "source": source} for t, r in par.items()]
            label = source
            if day != asof:
                notes.append(f"{info.title}: latest curve on or before {asof} is from {day}")
    if not rows:
        if dataset:
            notes.append(
                f"No spot (zero) rates of the {rh.dataset_info(dataset).title} on or before {asof} in the store: "
                "showing the illustrative USD SOFR set instead"
            )
        rows = _illustrative_rows()
        label = "ILLUSTRATIVE quotes (not market data)"
    if fixing is not None:
        rows = [r for r in rows if r["tenor"] not in ("ON", "1D")]
        rows.insert(0, fixing)
        label = f"{label} + {fixing_label}"
    table = pd.DataFrame(rows, columns=["instrument", "tenor", "rate", "source"])
    digest = hashlib.sha1(table.to_csv(index=False).encode()).hexdigest()[:10]
    return QuotePrefill(table, label, kind, f"{conventions}_{digest}", tuple(notes))


def uploaded_prefill(uploaded: Dict[str, object]) -> QuotePrefill:
    table = pd.DataFrame(uploaded["table"]).copy()
    if "source" not in table:
        table["source"] = "uploaded"
    table["source"] = table["source"].replace({"uploaded": f"uploaded: {uploaded['name']}"})
    return QuotePrefill(table, f"uploaded file {uploaded['name']}", "upload", f"upload_{str(uploaded['sha'])[:10]}")


def provenance(prefill: QuotePrefill, edited: pd.DataFrame) -> Tuple[str, int]:
    """(badge label, rows the user changed or added): a row is changed when
    its instrument, tenor or rate differ from the prefill row it started as."""

    base = prefill.table.reset_index(drop=True)
    changed = 0
    for i, row in edited.reset_index(drop=True).iterrows():
        if i >= len(base):
            changed += 1
            continue
        ref = base.iloc[i]
        same_rate = pd.notna(row.get("rate")) and abs(float(row["rate"]) - float(ref["rate"])) < 1e-9
        if str(row.get("instrument")) != str(ref["instrument"]) or str(row.get("tenor")) != str(ref["tenor"]):
            changed += 1
        elif not same_rate:
            changed += 1
    changed += max(0, len(base) - len(edited))
    label = prefill.label if not changed else f"{prefill.label} (+{changed} rows edited by you)"
    return label, changed


# --------------------------------------------------------------------------
# Badges
# --------------------------------------------------------------------------


BADGE_WIDTH = 96  # characters per badge line before it wraps


def badge(fig: go.Figure, source: str, asof: dt.date) -> go.Figure:
    """Stamp ``SOURCE ... | AS OF ...`` in the chart's top-left corner (kept
    when the chart is exported or zoomed), wrapped onto as many lines as it
    needs."""

    lines = textwrap.wrap(f"SOURCE: {source} | AS OF {asof.isoformat()}", BADGE_WIDTH)
    fig.add_annotation(
        text="<br>".join(lines),
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.0,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        align="left",
        font=dict(family="IBM Plex Mono", size=11, color="#FFA500"),
        bgcolor="rgba(0,0,0,0.85)",
        bordercolor="#FFA500",
        borderwidth=1,
    )
    top = fig.layout.margin.t if fig.layout.margin.t is not None else 30
    fig.update_layout(margin=dict(t=max(top, 30 + 16 * len(lines))))
    return fig


def set_badges(**labels: Tuple[str, dt.date]) -> None:
    """Remember each curve's (source, as-of) for charts on other tabs."""

    st.session_state[BADGE_KEY] = dict(labels)


def badge_for(which: str) -> Optional[Tuple[str, dt.date]]:
    return st.session_state.get(BADGE_KEY, {}).get(which)


def show_notices(notices: Sequence[Tuple[str, str]]) -> None:
    for level, text in notices:
        {"error": st.error, "warning": st.warning}.get(level, st.caption)(text)
