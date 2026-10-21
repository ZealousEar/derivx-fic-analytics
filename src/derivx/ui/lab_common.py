"""Helpers shared by the ANALYTICS lab's tabs (:mod:`derivx.ui.analytics_lab`,
:mod:`derivx.ui.lab_vol`, :mod:`derivx.ui.lab_rates`): result fingerprints,
the provenance banner, stat rows, legends, STALE notices and the
manifest + CSV downloads. Nothing here holds state."""

from __future__ import annotations

import hashlib
import html
import json
from typing import Callable, Dict, Mapping, Sequence, Tuple

import pandas as pd
import streamlit as st

from derivx.ui.theme import FOREGROUND, LEGEND_BELOW
from derivx.validation.manifest import jsonable, manifest_bytes, results_zip


def fingerprint(*parts) -> str:
    """SHA-256 of JSON-able parts (data hashes, filters, settings)."""

    blob = json.dumps(jsonable(list(parts)), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def banner(label: str, parts: Sequence[str], kind: str = "", synthetic: bool = False) -> None:
    """A provenance banner (``dx-lab-banner``): ``label`` in the accent
    colour, then the parts; ``kind`` 'stale' or '' and an orange edge for
    SYNTHETIC data."""

    css = " ".join(c for c in ("dx-lab-banner", kind, "synthetic" if synthetic else "") if c)
    body = " · ".join(html.escape(str(p)) for p in parts)
    st.markdown(f"<div class='{css}'><b>{html.escape(label)}</b> · {body}</div>", unsafe_allow_html=True)


def downloads(stem: str, manifest: Mapping[str, object], tables: Mapping[str, pd.DataFrame], key: str) -> None:
    """The results ZIP (manifest + one CSV per table) and the manifest alone."""

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


def stale_notice(
    stored: Mapping[str, object], data_key: str, settings_key: str, what: str, draw_banner: Callable[[], None]
) -> bool:
    """Warn when a stored result no longer matches the data or settings;
    True when the data changed (the result is STALE). ``draw_banner``
    draws the banner of the data the result was computed on."""

    if stored.get("data_key") != data_key:
        draw_banner()
        st.warning(
            f"These {what} results were computed on the data above, not on the active one (or with other "
            "filters). Re-run to update them."
        )
        return True
    if stored.get("settings_key") != settings_key:
        st.info(f"Settings changed since this {what} run: press the button to apply them.")
    return False


def legend() -> Dict[str, object]:
    """Horizontal legend at the bottom of the figure, below the x-axis title, in the foreground colour."""

    return dict(**LEGEND_BELOW, font=dict(color=FOREGROUND))


def stats(items: Sequence[Tuple[str, str]]) -> None:
    """A row of labelled numbers that reflows on narrow screens."""

    cells = "".join(
        f"<div class='dx-lab-stat'><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>" for k, v in items
    )
    st.markdown(f"<div class='dx-lab-stats'>{cells}</div>", unsafe_allow_html=True)


def show_issues(issues: Sequence, title: str) -> bool:
    """Errors (as one block), warnings and notes ('info') of a parser's
    ``issues`` (objects with ``level`` and ``message``); True when there
    was an error."""

    errors = [i.message for i in issues if i.level == "error"]
    for issue in issues:
        if issue.level == "warning":
            st.warning(issue.message)
        elif issue.level != "error":
            st.info(issue.message)
    if errors:
        st.error(title + "\n\n" + "\n".join(f"- {e}" for e in errors[:12]))
    return bool(errors)
