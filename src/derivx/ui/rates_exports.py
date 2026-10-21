"""Export buttons shared by the Rates tabs: a ZIP of the tab's tables as CSV
with a run manifest, and the manifest alone (JSON).

The manifest (:class:`derivx.rates.manifest.RunManifest`) records the code
version, package versions, the tab's inputs and model settings, every
curve's quotes with their source and as-of date, and the history store
(label, SHA-256 of its rows, publishers and download URLs)."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import pandas as pd
import streamlit as st

from derivx.data.rates_history import HistoryStore
from derivx.rates.curves import Market
from derivx.rates.manifest import RunManifest, bundle_zip, market_record, run_manifest, sha256_json, store_record
from derivx.ui import rates_data

STORE_MEMO = "_rates_store_record"


def curve_sources() -> Dict[str, str]:
    """{curve key: source label} the CURVE tab stamped on its charts."""

    return {key: label for key, (label, _) in st.session_state.get(rates_data.BADGE_KEY, {}).items()}


def _store_record(store: HistoryStore) -> Dict[str, object]:
    memo = st.session_state.setdefault(STORE_MEMO, {})
    if store.fingerprint not in memo:
        memo.clear()
        memo[store.fingerprint] = store_record(store)
    return memo[store.fingerprint]


def market_data(market: Market, store: Optional[HistoryStore] = None) -> Dict[str, object]:
    """The manifest's ``data`` section for the page's market."""

    record = market_record(market, curve_sources())
    record.pop("data_sha256")
    record["store"] = _store_record(store or rates_data.active_store())
    record["data_sha256"] = sha256_json(record)
    return record


def manifest(kind: str, market: Market, inputs: Mapping[str, object], model: Mapping[str, object],
             store: Optional[HistoryStore] = None) -> RunManifest:  # fmt: skip
    return run_manifest(kind, inputs, market_data(market, store), model)


def export_row(
    kind: str,
    market: Market,
    tables: Mapping[str, pd.DataFrame],
    inputs: Mapping[str, object],
    model: Mapping[str, object],
    key: str,
    store: Optional[HistoryStore] = None,
) -> RunManifest:
    """Two download buttons: every table plus the manifest (ZIP), and the
    manifest (JSON). Returns the manifest."""

    record = manifest(kind, market, inputs, model, store)
    stem = f"{kind.replace(' ', '_')}_{market.asof.isoformat()}"
    c1, c2 = st.columns(2)
    c1.download_button(
        "DOWNLOAD TABLES + MANIFEST (ZIP)", bundle_zip(record, tables), file_name=f"{stem}.zip",
        mime="application/zip", key=f"{key}_zip", on_click="ignore",
        help="Every table of this tab as CSV, with manifest.json: code version, packages, inputs, model settings, "
        "curve quotes and their sources, the history store's hash and download URLs.",
    )  # fmt: skip
    c2.download_button(
        "DOWNLOAD RUN MANIFEST (JSON)", record.to_json(), file_name=f"{stem}_manifest.json", mime="application/json",
        key=f"{key}_manifest", on_click="ignore",
    )  # fmt: skip
    return record
