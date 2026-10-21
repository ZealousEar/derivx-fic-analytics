"""Widget state across mode switches (derivx.ui.state, UIE-04)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from derivx.ui.state import KEPT_KEY, is_plain_value

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"


def test_state_helper_keeps_value_widgets_and_skips_buttons_and_uploads():
    """The shell helper keeps keyed widgets of pages that do not set
    persist_state, and never writes buttons, uploaders or data editors."""

    def script():
        import pandas as pd
        import streamlit as st

        from derivx.ui.state import keep_widget_values

        page = st.radio("page", ["A", "B"], key="page")
        if page == "A":
            st.number_input("num", key="a_num", value=1.0)
            st.checkbox("chk", key="a_chk")
            st.multiselect("ms", ["x", "y", "z"], key="a_ms")
            st.button("btn", key="a_btn")
            st.download_button("dl", b"x", key="a_dl")
            st.file_uploader("up", key="a_up")
            st.data_editor(pd.DataFrame({"v": [1, 2]}), key="a_ed")
        st.session_state["kept"] = keep_widget_values()

    at = AppTest.from_function(script, default_timeout=60)
    at.run()
    at.number_input(key="a_num").set_value(5.0).run()
    at.checkbox(key="a_chk").check().run()
    at.multiselect(key="a_ms").set_value(["y", "z"]).run()
    at.radio(key="page").set_value("B").run()
    assert set(at.session_state["kept"]) == {"a_num", "a_chk", "a_ms"}
    at.radio(key="page").set_value("A").run()
    assert not at.exception
    assert at.number_input(key="a_num").value == 5.0
    assert at.checkbox(key="a_chk").value is True
    assert at.multiselect(key="a_ms").value == ["y", "z"]


def test_plain_values_exclude_editor_states_and_selections():
    assert is_plain_value(1.5) and is_plain_value("CALL") and is_plain_value([1.0, 2.0]) and is_plain_value(None)
    assert not is_plain_value({"edited_rows": {}}) and not is_plain_value([{"a": 1}]) and not is_plain_value(object())


def test_remounted_widgets_tell_the_browser_their_kept_value():
    """Final review: the server kept the value of a widget that was not drawn, but the remounted widget's
    proto did not carry it (set_value False), so the browser showed the default while the page computed with
    the kept value. restore_widget_values, run first, makes the remount send it; a value set by a callback
    before the run wins over the kept one."""

    def script():
        import streamlit as st

        from derivx.ui.state import keep_widget_values, restore_widget_values

        restore_widget_values()
        page = st.radio("page", ["A", "B"], key="page")
        st.button("reset", key="reset", on_click=lambda: st.session_state.update(a_txt="RESET"))
        if page == "A":
            st.text_input("txt", "5Y", key="a_txt")
            st.number_input("num", key="a_num", value=1.0)
        keep_widget_values()

    at = AppTest.from_function(script, default_timeout=60)
    at.run()
    at.text_input(key="a_txt").input("7Y").run()
    at.number_input(key="a_num").set_value(4.0).run()
    at.radio(key="page").set_value("B").run()
    assert at.session_state[KEPT_KEY] == {"a_txt": "7Y", "a_num": 4.0}
    at.radio(key="page").set_value("A").run()
    assert not at.exception
    txt, num = at.text_input(key="a_txt"), at.number_input(key="a_num")
    assert txt.value == "7Y" and txt.proto.set_value and txt.proto.value == "7Y"
    assert num.value == 4.0 and num.proto.set_value and num.proto.value == 4.0
    assert at.session_state[KEPT_KEY] == {}  # drawn again: the live widgets hold the values
    at.text_input(key="a_txt").input("10Y").run()  # later edits are not overwritten by the kept value
    assert at.text_input(key="a_txt").value == "10Y"
    at.radio(key="page").set_value("B").run()
    at.button(key="reset").click().run()  # a callback writes the key while it is not drawn
    at.radio(key="page").set_value("A").run()
    assert at.text_input(key="a_txt").value == "RESET"


def test_rates_inputs_survive_a_mode_round_trip_in_the_browser_too():
    """The SWAP tenor typed on RATES is shown again (not just used) after RATES -> EQUITY -> RATES."""

    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    at.text_input(key="swap_tenor").input("7Y").run()
    at.sidebar.radio[0].set_value("[ EQUITY ]").run()
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    assert not at.exception
    tenor = at.text_input(key="swap_tenor")
    assert tenor.value == "7Y" and tenor.proto.set_value and tenor.proto.value == "7Y"


def test_data_editor_edits_survive_runs_that_do_not_draw_it():
    """UIE-04 / B1: Streamlit drops a data editor's edits with its widget state
    when a run does not draw it (and they cannot be written back through
    session state), so a quote edited on RATES was gone after EQUITY -> RATES.
    state.data_editor keeps the edited table and starts the redrawn editor
    from it; new data starts a fresh editor."""

    import json

    from streamlit.proto.WidgetStates_pb2 import WidgetState

    def script():
        import pandas as pd
        import streamlit as st

        from derivx.ui.state import data_editor

        page = st.radio("page", ["A", "B"], key="page")
        base = st.number_input("base", value=1.0, key="base")
        if page == "A":
            out = data_editor(pd.DataFrame({"v": [base, 2.0]}), key="ed", num_rows="dynamic")
            st.write("SUM", float(out["v"].sum()))

    def edit(at, value):
        state = WidgetState(id=at.dataframe[0].proto.id)
        state.string_value = json.dumps({"edited_rows": {"0": {"v": value}}, "added_rows": [], "deleted_rows": []})
        states = at._tree.get_widget_states()
        states.widgets.append(state)
        at._run(states)

    def total(at):
        return next(m.value for m in at.markdown if m.value.startswith("SUM"))

    at = AppTest.from_function(script, default_timeout=60)
    at.run()
    edit(at, 10.0)
    assert total(at) == "SUM `12.0`"
    at.run()  # drawn run after run: the edit stays
    assert total(at) == "SUM `12.0`"
    at.radio(key="page").set_value("B").run()
    at.radio(key="page").set_value("A").run()
    assert not at.exception and total(at) == "SUM `12.0`"  # was SUM 3.0: the edit was lost
    assert at.dataframe[0].value["v"].tolist() == [10.0, 2.0]  # the editor shows it again
    at.number_input(key="base").set_value(5.0).run()  # other data: a fresh editor, no stale edit applied
    assert total(at) == "SUM `7.0`"
