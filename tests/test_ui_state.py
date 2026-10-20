"""Widget state across mode switches (derivx.ui.state, UIE-04)."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from derivx.ui.state import is_plain_value


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
