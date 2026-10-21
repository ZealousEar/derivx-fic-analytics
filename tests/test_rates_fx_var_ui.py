"""FX as a risk factor in the Rates UI: on a euro market with a US
Treasury, reported in USD, RESEARCH > VAR moves the converted book with the
ECB reference rates of each scenario day (FX-held and FX-only rows and
cards, the FX row of the sources table, each worst day's EURUSD move) and
holds FX when asked; SCENARIOS replays an episode's FX moves on the
converted total only, unless HOLD FX SPOTS is ticked."""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

SETUP = """
import streamlit as st
from derivx.data.rates_history import bundled_store
from derivx.rates import curve_io
from derivx.rates.bonds import benchmark_bond, bond_position
from derivx.rates.examples import EXAMPLE_ASOF
from derivx.rates.swaps import VanillaSwap, swap_position
from derivx.ui import rates_research, rates_scenarios

market, _, _ = curve_io.default_market(bundled_store(), EXAMPLE_ASOF, "EUR ESTR")
swap = VanillaSwap.from_conventions(market.curve.conventions, "10Y", 0.025, 1e8, trade_date=market.asof)
settle = market.govt.conventions.cal.add_business_days(market.asof, 1)
positions = [swap_position(swap), bond_position(benchmark_bond(market.govt, 10, settle), 1e7)]
"""
RESEARCH_APP = SETUP + "rates_research.render(market, positions)\n"
SCENARIOS_APP = SETUP + "rates_scenarios.render(market, positions)\n"


def _run(source: str) -> AppTest:
    at = AppTest.from_string(source, default_timeout=600)
    at.run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    return at


def _frame(at: AppTest, column: str) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if column in d.value.columns)


def _var_table(at: AppTest) -> pd.DataFrame:
    """The VaR table (positions, totals; HIST VAR columns), indexed by POSITION."""

    table = next(d.value for d in at.dataframe if any(str(c).startswith("HIST VAR") for c in d.value.columns))
    return table.set_index("POSITION")


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


@pytest.fixture(scope="module")
def research() -> AppTest:
    return _run(RESEARCH_APP)


def test_var_moves_the_converted_book_with_fx(research):
    at = research
    rows = _var_table(at)
    ids = list(rows.index)
    assert ids[-5:] == ["TOTAL USD", "TOTAL EUR", "ALL IN USD", "ALL IN USD (FX HELD)", "ALL IN USD (FX ONLY)"]
    var_col = next(c for c in rows.columns if c.startswith("HIST VAR"))
    assert rows.loc["ALL IN USD (FX ONLY)", var_col] > 0.0
    for label in ("VAR FX HELD (IN $)", "FX-ONLY VAR (IN $)", "FX ADD-ON (IN $)"):
        assert len(label) <= 20 and _metric(at, label)
    assert _metric(at, "HIST VAR 99% (IN $)") == f"{round(rows.loc['ALL IN USD', var_col]) + 0.0:,.0f}"
    sources = _frame(at, "COVERAGE_PCT")
    fx = sources[sources["CURVE"] == "FX SPOTS (ECB REFERENCE RATES)"].iloc[0]
    assert fx["DATASET"] == "FX_ECB" and fx["COVERAGE_PCT"] > 95.0
    worst = _frame(at, "DATE")
    assert "EURUSD MOVE (%)" in worst.columns and worst["EURUSD MOVE (%)"].abs().max() > 0.0
    caption = " ".join(c.value for c in at.caption if c.value.startswith("One-day"))
    assert "FX spots move with the ECB reference rates of the same dates" in caption
    assert "Local P&L and the TOTAL rows do not move with FX" in caption


def test_holding_fx_drops_the_fx_rows_and_keeps_local_var(research):
    moved = _var_table(research)
    at = _run(RESEARCH_APP)
    at.radio(key="rs_var_fx").set_value("HELD").run()
    assert not at.exception and not at.error
    held = _var_table(at)
    assert "ALL IN USD (FX HELD)" not in held.index and "ALL IN USD (FX ONLY)" not in held.index
    var_col = next(c for c in held.columns if c.startswith("HIST VAR"))
    for label in ("TOTAL USD", "TOTAL EUR"):
        assert held.loc[label, var_col] == pytest.approx(moved.loc[label, var_col], rel=1e-12)
    assert held.loc["ALL IN USD", var_col] == pytest.approx(moved.loc["ALL IN USD (FX HELD)", var_col], rel=1e-12)
    assert not [m for m in at.metric if m.label.startswith(("VAR FX HELD", "FX-ONLY VAR"))]
    sources = _frame(at, "COVERAGE_PCT")
    assert "FX SPOTS (ECB REFERENCE RATES)" not in set(sources["CURVE"])
    assert any("FX spots are held at today's" in c.value for c in at.caption)


def test_scenarios_replay_an_episodes_fx_unless_held():
    at = _run(SCENARIOS_APP)
    at.selectbox(key="scn_type").set_value("Historical").run()
    at.selectbox(key="scn_hist").set_value("2022 hiking cycle").run()
    assert not at.exception and not at.error
    sources = next(d.value for d in at.dataframe if list(d.value.columns)[:3] == ["CURVE", "CURRENCY", "DATA"])
    fx = sources.set_index("CURVE").loc["FX SPOTS"]
    assert fx["FLAG"].startswith("REAL") and (fx["START"], fx["END"]) == ("2022-01-03", "2022-10-21")
    assert "EURUSD -14.31%" in fx["NOTE"]
    grid = _frame(at, "ALL IN USD")
    at.checkbox(key="scn_hold_fx").check().run()
    assert not at.exception and not at.error
    held = _frame(at, "ALL IN USD")
    sources = next(d.value for d in at.dataframe if list(d.value.columns)[:3] == ["CURVE", "CURRENCY", "DATA"])
    assert "FX SPOTS" not in set(sources["CURVE"])
    for column in ("TOTAL USD", "TOTAL EUR"):
        assert held[column].iloc[0] == pytest.approx(grid[column].iloc[0], rel=1e-12)  # local P&L unchanged
    assert held["ALL IN USD"].iloc[0] != pytest.approx(grid["ALL IN USD"].iloc[0], rel=1e-6)


def test_an_fx_shock_adds_to_the_replayed_fx_moves():
    at = _run(SCENARIOS_APP)
    at.selectbox(key="scn_type").set_value("Historical").run()
    at.selectbox(key="scn_hist").set_value("2022 hiking cycle").run()
    from derivx.rates.examples import EXAMPLE_ASOF
    from derivx.risk.reporting import default_reporting

    replay = _frame(at, "ALL IN USD")["ALL IN USD"].iloc[0]
    explain = _frame(at, "BASE PV").set_index("POSITION")
    assert explain.loc["ALL IN USD", "FX"] != 0.0  # the explain shows the replay's FX translation
    after = explain.loc["TOTAL EUR", "HORIZON PV"] + explain.loc["TOTAL EUR", "CASH"]  # the euro book after the shock
    at.number_input(key="scn_fx_eurusd").set_value(2.0).run()
    assert not at.exception and not at.error
    assert any("2022 hiking cycle:" in c.value and "FX shock added: EURUSD +2%" in c.value for c in at.caption)
    eurusd = default_reporting(EXAMPLE_ASOF).fx.quote("EURUSD")
    added = _frame(at, "ALL IN USD")["ALL IN USD"].iloc[0] - replay
    assert added == pytest.approx(after * eurusd * 0.02, rel=1e-9)  # the extra 2% on the euro book's value
