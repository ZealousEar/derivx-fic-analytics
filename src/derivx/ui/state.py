"""Widget state that survives switching between the app's modes.

Streamlit drops a widget's value at the end of every run in which the
widget is not drawn, so switching EQUITY -> RATES -> EQUITY used to reset
every input (UIE-04). Three layers keep the values:

1. Pages that own their widgets pass ``persist_state=PERSIST`` (Streamlit's
   own option, "session": the value is kept while the widget is not drawn).
   The pricer page does this for every input, with keys namespaced per mode
   (``eq_*``, ``fx_*``) so EQUITY and FX keep separate markets.
2. :func:`keep_widget_values`, called by the app shell at the end of every
   run, does the same for keyed widgets of the other pages: it copies the
   value of each keyed widget that was not drawn this run onto its key and
   into a kept-values table, and :func:`restore_widget_values`, called by
   the shell at the start of every run, writes the kept values back onto
   their keys before any page draws. The second write is what the browser
   needs: Streamlit tells the frontend to show a widget's value only when
   that value was set in the same run, so a value carried over from an
   earlier run was used by the server while the remounted input showed its
   default (final review: SWAP tenor 7Y priced, 5Y displayed, after
   RATES -> EQUITY -> RATES). A kept value is dropped once its widget is
   drawn again (the live widget is then the source of truth). Only value
   widgets are kept (numbers, text, selections, sliders, checkboxes,
   dates); buttons, file uploaders, data editors and chart selections are
   never written, because Streamlit forbids setting those through session
   state. It reads Streamlit's widget registry (not a public API); if that
   changes, it logs a warning once and keeps nothing, and layer 1 still
   works.
3. Data editors: :func:`data_editor` (a drop-in for ``st.data_editor``)
   keeps the edited table itself and starts a redrawn editor from it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from typing import Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

PERSIST = "session"
KEPT_KEY = "_derivx_kept_widget_values"  # {user key: value} of value widgets not drawn since they were kept
EDITOR_KEY_PREFIX = "_derivx_editor::"  # + editor key: the edited table of a data_editor (see data_editor)

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


def _widget_keys() -> Tuple[List[str], List[str]]:
    """(undrawn, drawn): user keys of the keyed value widgets without
    ``persist_state`` registered in an earlier run and not drawn in this
    one, and user keys of the keyed widgets drawn in this run. Raises
    AttributeError / KeyError / TypeError when Streamlit's registry moved."""

    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

    ctx = get_script_run_ctx()
    if ctx is None:
        return [], []
    state = ctx.session_state._state
    shared = getattr(ctx, "shared", None)  # Streamlit >= 1.6x keeps per-run sets here
    drawn_ids = shared.widget_ids_this_run if shared is not None else getattr(ctx, "widget_ids_this_run")
    metadata = state._new_widget_state.widget_metadata
    undrawn, drawn = [], []
    for widget_id, user_key in dict(state._key_id_mapper.id_key_mapping).items():
        if widget_id in drawn_ids:
            drawn.append(user_key)
            continue
        meta = metadata.get(widget_id)
        if meta is None or getattr(meta, "persist_state", None) is not None:
            continue
        if meta.value_type in _VALUE_TYPES:
            undrawn.append(user_key)
    return undrawn, drawn


def _undrawn_widget_keys() -> Optional[List[str]]:
    """User keys of the keyed value widgets registered in an earlier run
    and not drawn in this one, without ``persist_state``."""

    return _widget_keys()[0]


def _written_this_run(key: str) -> bool:
    """True if ``key`` was already assigned in this run (by a widget
    callback, which runs before the script); False when unknown."""

    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

    ctx = get_script_run_ctx()
    try:
        return bool(ctx is not None and ctx.session_state._state.is_new_state_value(key))
    except (AttributeError, KeyError, TypeError):
        return False


def restore_widget_values() -> List[str]:
    """Write the values :func:`keep_widget_values` kept back onto their keys;
    returns the keys written. Call it once, at the start of every run,
    before any page draws (the shell does): a widget drawn again in this run
    then sends its kept value to the browser instead of mounting at its
    default while the server uses the kept value."""

    kept = st.session_state.get(KEPT_KEY)
    if not isinstance(kept, dict):
        return []
    restored = []
    for key, value in list(kept.items()):
        if _written_this_run(key):  # a callback set it before this run: that value wins
            kept.pop(key, None)
            continue
        try:
            st.session_state[key] = value
        except StreamlitAPIException:  # the key now belongs to a widget that cannot be set
            kept.pop(key, None)
            continue
        restored.append(key)
    return restored


def keep_widget_values() -> List[str]:
    """Keep the values of keyed value widgets that this run did not draw
    (see the module docs); returns the keys kept. Call it once, after every
    page has been drawn (the shell does, in a ``finally``, so a run that a
    page aborts with an exception still keeps the other modes' values)."""

    global _warned
    try:
        keys, drawn = _widget_keys()
    except (AttributeError, KeyError, TypeError):  # Streamlit internals moved
        if not _warned:
            _log.warning("Widget values will not survive mode switches: Streamlit's widget registry changed")
            _warned = True
        return []
    table = st.session_state.get(KEPT_KEY)
    table = dict(table) if isinstance(table, dict) else {}
    for key in drawn:  # drawn again: the live widget holds the value now
        table.pop(key, None)
    kept = []
    for key in keys:
        try:
            value = st.session_state[key]
        except KeyError:
            continue
        if is_plain_value(value):
            st.session_state[key] = value
            table[key] = value
            kept.append(key)
    st.session_state[KEPT_KEY] = table
    return kept


def _frame_signature(data: Any) -> str:
    """Content hash of a table (columns, dtypes, index and values)."""

    try:
        frame = pd.DataFrame(data)
        blob = repr((list(frame.columns), [str(t) for t in frame.dtypes])).encode()
        blob += pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    except (TypeError, ValueError):  # unhashable cells: fall back to the text form
        blob = repr(data).encode()
    return hashlib.sha1(blob).hexdigest()


def data_editor(data: Any, *, key: str, **kwargs: Any) -> pd.DataFrame:
    """``st.data_editor`` whose edits survive runs that do not draw it.

    Streamlit drops an editor's edits with the rest of its widget state when
    a run does not draw it (a mode switch), and its state cannot be written
    back through ``st.session_state`` like a value widget's, so
    :func:`keep_widget_values` cannot keep it. Instead the edited table is
    kept under ``EDITOR_KEY_PREFIX + key`` with a hash of ``data``: when the
    editor is drawn again after a gap (its own state gone) for the same
    ``data``, it starts from the edited table; while it stays drawn it keeps
    being given the table it started from (an editor whose data changes is
    a new widget). New ``data`` (another prefill, a loaded file) starts a
    fresh editor. Returns the edited table, as ``st.data_editor`` does."""

    shadow = EDITOR_KEY_PREFIX + key
    base = _frame_signature(data)
    kept = st.session_state.get(shadow)
    start = data
    if isinstance(kept, dict) and kept.get("base") == base:
        start = kept["start"] if key in st.session_state else kept["edited"]
    edited = st.data_editor(start, key=key, **kwargs)
    st.session_state[shadow] = {"base": base, "start": start, "edited": edited.copy()}
    return edited
