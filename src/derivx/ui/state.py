"""Widget state that survives switching between the app's modes.

Streamlit drops a widget's value at the end of every run in which the
widget is not drawn, so switching EQUITY -> RATES -> EQUITY used to reset
every input (UIE-04). Two layers keep the values:

1. Pages that own their widgets pass ``persist_state=PERSIST`` (Streamlit's
   own option, "session": the value is kept while the widget is not drawn).
   The pricer page does this for every input, with keys namespaced per mode
   (``eq_*``, ``fx_*``) so EQUITY and FX keep separate markets.
2. :func:`keep_widget_values`, called by the app shell at the end of every
   run, does the same for keyed widgets of the other pages: it copies the
   value of each keyed widget that was not drawn this run onto its key, so
   it is still there when the page is drawn again. Only value widgets are
   kept (numbers, text, selections, sliders, checkboxes, dates); buttons,
   file uploaders, data editors and chart selections are never written,
   because Streamlit forbids setting those through session state. It reads
   Streamlit's widget registry (not a public API); if that changes, it logs
   a warning once and keeps nothing, and layer 1 still works.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, List, Optional

import streamlit as st

PERSIST = "session"

# Streamlit widget value types whose values may be written back through
# st.session_state (button, download_button and form_submit_button use
# 'trigger_value'; file and camera inputs 'file_uploader_state_value').
_VALUE_TYPES = frozenset(
    {
        "bool_value",
        "int_value",
        "double_value",
        "string_value",
        "int_array_value",
        "double_array_value",
        "string_array_value",
    }
)
_SCALARS = (bool, int, float, str, dt.date, dt.time)
_log = logging.getLogger(__name__)
_warned = False


def is_plain_value(value: Any) -> bool:
    """True for values a value widget can be set to: scalars, dates and
    times, and lists or tuples of them (multiselect, range sliders). Data
    editor states and chart selections are dicts and are excluded."""

    if value is None or isinstance(value, _SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return all(item is None or isinstance(item, _SCALARS) for item in value)
    return False


def _undrawn_widget_keys() -> Optional[List[str]]:
    """User keys of the keyed value widgets registered in an earlier run
    and not drawn in this one, without ``persist_state``; None when
    Streamlit's registry is not readable."""

    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

    ctx = get_script_run_ctx()
    if ctx is None:
        return []
    state = ctx.session_state._state
    shared = getattr(ctx, "shared", None)  # Streamlit >= 1.6x keeps per-run sets here
    drawn = shared.widget_ids_this_run if shared is not None else ctx.widget_ids_this_run
    metadata = state._new_widget_state.widget_metadata
    keys = []
    for widget_id, user_key in dict(state._key_id_mapper.id_key_mapping).items():
        meta = metadata.get(widget_id)
        if meta is None or widget_id in drawn or getattr(meta, "persist_state", None) is not None:
            continue
        if meta.value_type in _VALUE_TYPES:
            keys.append(user_key)
    return keys


def keep_widget_values() -> List[str]:
    """Keep the values of keyed value widgets that this run did not draw
    (see the module docs); returns the keys kept. Call it once, after every
    page has been drawn (the shell does, in a ``finally``)."""

    global _warned
    try:
        keys = _undrawn_widget_keys()
    except (AttributeError, KeyError, TypeError):  # Streamlit internals moved
        if not _warned:
            _log.warning("Widget values will not survive mode switches: Streamlit's widget registry changed")
            _warned = True
        return []
    kept = []
    for key in keys or []:
        try:
            value = st.session_state[key]
        except KeyError:
            continue
        if is_plain_value(value):
            st.session_state[key] = value
            kept.append(key)
    return kept
