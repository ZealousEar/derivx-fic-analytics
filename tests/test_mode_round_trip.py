"""Every input of every mode survives a trip through the other modes (UIE-04,
B1, B2): non-default values on EQUITY, FX, RATES and ANALYTICS, including
the quote tables, are all still there after switching through every other
mode and back; and a page that raises still lets the other modes keep
theirs, with its error shown."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest

from derivx.ui.state import EDITOR_KEY_PREFIX

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"
MODES = ["[ EQUITY ]", "[ FX ]", "[ RATES ]", "[ ANALYTICS ]"]
VALUE_TYPES = (
    "number_input",
    "slider",
    "select_slider",
    "checkbox",
    "toggle",
    "multiselect",
    "selectbox",
    "radio",
    "text_input",
    "text_area",
    "date_input",
    "time_input",
)
# Choices and texts set by hand (a generic "next option" could switch a page to another instrument or pair,
# which redraws other inputs); every number input, slider, checkbox and multiselect is changed generically.
CHOICES: Dict[str, Dict[str, Any]] = {
    "[ EQUITY ]": {"eq_type": "PUT", "eq_surf_z": "GAMMA", "eq_surf_axes": "SPOT × TIME TO EXPIRY"},
    "[ FX ]": {"fx_model": "VV", "fx_strangle": "smile", "fx_type": "PUT", "fx_side": "SELL"},
    "[ RATES ]": {
        "swap_tenor": "7Y",
        "rates_interp": "monotone_convex",
        "swn_model": "Black (%)",
        "cap_model": "Shifted Black (%)",
    },
    "[ ANALYTICS ]": {
        "lab_ticker": "QQQ",
        "lab_discount": "rate",
        "lab_rates_up_units": "bp",
        "lab_vh_units": "decimal",
    },
}
# Generic changes left out: buttons that load data or reruns that change what the page draws.
SKIP = {"lab_force"}  # bypassing the shared Yahoo cache only matters for a pull, which the test never makes


def _widgets(at: AppTest):
    for kind in VALUE_TYPES:
        for el in getattr(at, kind):
            if el.key and el.key != "app_mode":
                yield kind, el


def _plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_plain(v) for v in value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    return value


def _snapshot(at: AppTest) -> Dict[Tuple[str, str], Any]:
    return {(kind, el.key): _plain(el.value) for kind, el in _widgets(at)}


def _change(at: AppTest, mode: str) -> List[str]:
    """Give the page's inputs non-default values in one batch: every enabled
    number input and numeric slider one step from its value (inside its
    bounds), every checkbox and toggle flipped, every multiselect with two
    or more choices one fewer, and the hand-picked choices and texts."""

    changed: List[str] = []
    for kind, el in _widgets(at):
        if el.key in SKIP or el.disabled:
            continue
        if kind == "number_input" and el.value is not None:
            step = el.step or 1
            new = el.value + step
            if el.max is not None and new > el.max:
                new = el.value - step
            if el.min is not None and new < el.min:
                continue
            el.set_value(type(el.value)(new) if isinstance(el.value, int) else new)
        elif kind == "slider" and isinstance(el.value, (int, float)):
            new = el.value + el.step if el.value + el.step <= el.max else el.value - el.step
            el.set_value(new)
        elif kind == "slider" and isinstance(el.value, (list, tuple)) and isinstance(el.value[0], (int, float)):
            lo, hi = el.value
            el.set_range(lo + el.step, hi) if lo + el.step < hi else el.set_range(lo, hi + el.step)
        elif kind in ("checkbox", "toggle"):
            el.set_value(not el.value)
        elif kind == "multiselect" and len(el.value) > 1:
            el.set_value(list(el.value)[:-1])
        else:
            continue
        changed.append(el.key)
    for key, value in CHOICES[mode].items():
        next(e for _, e in _widgets(at) if e.key == key).set_value(value)
        changed.append(key)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return changed


def _edit_table(at: AppTest, prefix: str, row: int, column: str, value: float) -> str:
    """Edit one cell of the data editor whose key starts with ``prefix``, as
    the browser would (AppTest has no data editor API)."""

    editor = next(d for d in at.dataframe if d.key and d.key.startswith(prefix))
    state = WidgetState(id=editor.proto.id)
    state.string_value = json.dumps({"edited_rows": {str(row): {column: value}}, "added_rows": [], "deleted_rows": []})
    states = at._tree.get_widget_states()
    states.widgets.append(state)
    at._run(states)
    assert not at.exception, [e.value for e in at.exception]
    return editor.key


def _table(at: AppTest, key: str):
    return next(d for d in at.dataframe if d.key == key).value


def test_every_input_of_every_mode_survives_a_trip_through_the_other_modes():
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    before: Dict[str, Dict[Tuple[str, str], Any]] = {}
    for mode in MODES:
        at.radio(key="app_mode").set_value(mode).run()
        assert not at.exception, (mode, [e.value for e in at.exception])
        defaults = _snapshot(at)
        changed = _change(at, mode)
        if mode == "[ RATES ]":
            rates_key = _edit_table(at, "rates_quotes_", 3, "rate", 4.4321)
        if mode == "[ FX ]":
            fx_key = _edit_table(at, "fx_quotes_editor_", 4, "ATM (%)", 9.87)
        before[mode] = _snapshot(at)
        moved = [k for k in changed if any(defaults.get((t, k)) != before[mode].get((t, k)) for t in VALUE_TYPES)]
        assert len(moved) >= 0.8 * len(changed) > 5, (mode, sorted(set(changed) - set(moved)))
    for mode in MODES:  # each mode again, after a visit to every other one
        at.radio(key="app_mode").set_value(mode).run()
        assert not at.exception, (mode, [e.value for e in at.exception])
        after = _snapshot(at)
        lost = {k: (v, after.get(k)) for k, v in before[mode].items() if after.get(k) != v}
        assert not lost, (mode, lost)
        if mode == "[ RATES ]":
            assert 4.4321 in _table(at, rates_key)["rate"].round(4).tolist()  # the edited quote, shown again
            kept = at.session_state[EDITOR_KEY_PREFIX + rates_key]["edited"]
            assert kept["rate"].round(4).tolist().count(4.4321) == 1
        if mode == "[ FX ]":
            assert _table(at, fx_key)["ATM (%)"].iloc[4] == pytest.approx(9.87)


def test_a_page_that_raises_shows_its_error_and_the_other_modes_keep_their_inputs(monkeypatch):
    """B2: keep_widget_values ran as the shell's last statement, so a run that
    a page aborted kept nothing for the modes it did not draw."""

    from derivx.ui import fx_page

    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.radio(key="app_mode").set_value("[ RATES ]").run()
    at.text_input(key="swap_tenor").input("7Y").run()
    real = fx_page.render

    def broken() -> None:
        raise RuntimeError("FX page failed on purpose")

    monkeypatch.setattr(fx_page, "render", broken)
    at.radio(key="app_mode").set_value("[ FX ]").run()
    assert any("FX page failed on purpose" in e.value for e in at.exception)  # shown, not swallowed
    monkeypatch.setattr(fx_page, "render", real)
    at.radio(key="app_mode").set_value("[ RATES ]").run()
    assert not at.exception
    tenor = at.text_input(key="swap_tenor")
    assert tenor.value == "7Y" and tenor.proto.value == "7Y"
