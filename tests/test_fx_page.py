"""FX OPTIONS page (AppTest): market conventions on screen, pip precision,
forward points, quote conventions, the editable quote table, expiry,
input errors, the live ECB spot (monkeypatched) and widget state."""

from __future__ import annotations

import dataclasses
import datetime as dt
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _vol_chains import synth_chain
from streamlit.testing.v1 import AppTest

from derivx.fx import market_data
from derivx.fx.conventions import forward_points, fx_dates, get_pair, market_delta_type
from derivx.fx.examples import ILLUSTRATIVE_QUOTES
from derivx.fx.pricing import FxMarket, fx_delta, gk_price
from derivx.fx.smile import SmileConventions, TenorQuote, build_smile
from derivx.fx.smile_risk import quote_vegas
from derivx.ui.fx_page import apply_editor_changes

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"
FIXTURE = Path(__file__).parent / "fixtures" / "ecb_exr_2026-10-20.csv"
TODAY = dt.date.today()
EURUSD_SPOT = 1.1234  # the latest day of the bundled ECB history (2026-10-21)
NOTIONAL = 1_000_000.0


def _fx(**state) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=180)
    # Pin the page's trade date (default: today) to the TODAY the expectations use, so a run that
    # crosses midnight does not compare a 20 October page with 21 October dates.
    state.setdefault("fx_trade_date", TODAY)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    at.sidebar.radio[0].set_value("[ FX ]").run()
    assert not at.exception, at.exception
    return at


def _hero(at: AppTest) -> str:
    return next(m.value for m in at.markdown if "dx-hero-value" in m.value and "<style>" not in m.value)


def _premium(at: AppTest) -> float:
    return float(re.search(r"dx-hero-value'>([^<]+)<", _hero(at)).group(1).replace(",", ""))


def _cell(at: AppTest, label: str) -> str:
    return re.search(rf"<span>{re.escape(label)}</span><b>([^<]+)</b>", _hero(at)).group(1)


def _charts(at: AppTest) -> dict:
    return {
        json.loads(c.proto.spec)["layout"]["meta"]["chart"]: json.loads(c.proto.spec) for c in at.get("plotly_chart")
    }


def _frame(at: AppTest, column: str) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if column in d.value.columns)


def _market(code: str, tenor: str, S: float, rd: float, rf: float):
    d = fx_dates(code, TODAY, tenor=tenor)
    return d, FxMarket(S, rd, rf, d.tau_e, d.tau_d)


def _quote(code: str, tenor: str) -> TenorQuote:
    row = next(r for r in ILLUSTRATIVE_QUOTES[code] if r[0] == tenor)
    return TenorQuote(tenor, *[x / 100 for x in row[1:]])


def test_default_page_prices_the_option_on_its_smile_with_market_conventions():
    """PP-12 / G11: EURUSD 3M 1.1200 call priced at its SABR smile vol, USD
    premium, spot delta, forward points; ILLUSTRATIVE quotes labelled."""

    at = _fx()
    charts = _charts(at)
    assert set(charts) == {"fx_smile", "fx_surface", "fx_scenario"}
    d, m = _market("EURUSD", "3M", EURUSD_SPOT, 0.0375, 0.02)
    smile = build_smile(m, _quote("EURUSD", "3M"), SmileConventions(market_delta_type("EURUSD", d), "DNS", "market"))
    vol = smile.vol(1.12)
    expected = NOTIONAL * gk_price(m, 1.12, vol, True)
    assert _premium(at) == pytest.approx(expected, abs=0.01)
    hero = _hero(at)
    assert "EUR CALL / USD PUT" in hero and " USD</span>" in hero and "$" not in hero
    assert _cell(at, "VOL") == f"{100 * vol:.3f}%"
    assert _cell(at, "FWD POINTS") == f"{forward_points(EURUSD_SPOT, m.forward, 0.0001):+,.2f} pips"
    assert _cell(at, f"FORWARD ({d.delivery.isoformat()})") == f"{m.forward:.5f}"
    assert any("ILLUSTRATIVE</span>" in x.value and "not market" in x.value for x in at.markdown)
    premium = _frame(at, "QUOTE STYLE").set_index("QUOTE STYLE")
    assert premium.loc["USD PIPS  ← premium ccy", "VALUE"] == f"{gk_price(m, 1.12, vol, True) / 1e-4:,.2f}"
    deltas = _frame(at, "DELTA TYPE")
    assert list(deltas["DELTA TYPE"])[0] == "SPOT DELTA  ← hedge"
    # PP-03 / PP-10 / VM-05 (FX side): strikes around the forward, F and the ATM (DNS) strike marked apart
    traces = {t["name"]: t for t in charts["fx_smile"]["data"]}
    xaxis = charts["fx_smile"]["layout"]["xaxis"]
    assert 0.85 * m.forward < xaxis["minallowed"] < smile.atm_strike < xaxis["maxallowed"] < 1.15 * m.forward
    assert traces["F (FORWARD)"]["x"][0] == pytest.approx(m.forward, rel=1e-12)
    pillars = dict(zip(traces["DELTA PILLARS"]["text"], traces["DELTA PILLARS"]["x"]))
    assert pillars["ATM"] == pytest.approx(smile.atm_strike, rel=1e-9) and pillars["ATM"] > m.forward  # DNS > F


def test_pair_precision_and_pip_steps():
    """PP-07 / ui-engineering-V3: 1.0835 is shown as 1.08350, not 1.08;
    steps are one pip; JPY pairs show 3 decimals and step 0.01."""

    at = _fx()
    spot, strike = at.number_input(key="fx_spot_EURUSD"), at.number_input(key="fx_strike_EURUSD")
    assert (spot.proto.format, spot.proto.step, strike.proto.format, strike.proto.step) == (
        "%.5f",
        0.0001,
        "%.5f",
        0.0001,
    )
    spot.set_value(1.0835).run()
    assert _cell(at, "SPOT") == "1.08350"
    at.number_input(key="fx_spot_EURUSD").set_value(1.083456).run()
    assert _cell(at, "SPOT") == "1.083456"  # more decimals than the box: shown as priced
    at.number_input(key="fx_strike_EURUSD").set_value(1.0875).run()
    assert _cell(at, "STRIKE") == "1.08750"
    at.selectbox(key="fx_pair").set_value("USDJPY").run()
    assert not at.exception
    jpy = at.number_input(key="fx_spot_USDJPY")
    assert (jpy.proto.format, jpy.proto.step) == ("%.3f", 0.01)
    assert jpy.value == pytest.approx(round(177.36 / 1.1234, 3))
    assert _cell(at, "SPOT") == f"{jpy.value:.3f}" and " USD</span>" in _hero(at)  # USDJPY premium in USD
    assert "PREMIUM-ADJUSTED" in _frame(at, "DELTA TYPE").iloc[2]["DELTA TYPE"]
    hedge = [t for t in _frame(at, "DELTA TYPE")["DELTA TYPE"] if "hedge" in t]
    assert hedge == ["SPOT DELTA, PREMIUM-ADJUSTED  ← hedge"]


def test_premium_currency_and_notional_currency():
    at = _fx()
    usd = _premium(at)
    at.radio(key="fx_premium_ccy_EURUSD").set_value("EUR").run()
    assert " EUR</span>" in _hero(at)
    assert _premium(at) == pytest.approx(usd / EURUSD_SPOT, abs=0.01)
    assert [t for t in _frame(at, "DELTA TYPE")["DELTA TYPE"] if "hedge" in t] == [
        "SPOT DELTA, PREMIUM-ADJUSTED  ← hedge"
    ]
    assert "% EUR  ← premium ccy" in list(_frame(at, "QUOTE STYLE")["QUOTE STYLE"])
    at.radio(key="fx_premium_ccy_EURUSD").set_value("USD").run()
    at.radio(key="fx_notional_ccy_EURUSD").set_value("USD").run()
    assert _premium(at) == pytest.approx(usd / 1.12, abs=0.01)  # USD 1m at K = 1.12 is EUR 892,857


def test_forward_points_mode_sets_the_forward():
    at = _fx()
    at.radio(key="fx_carry").set_value("FORWARD POINTS").run()
    base = at.session_state["fx_quotes_EURUSD"].copy()
    base.loc[base["TENOR"] == "3M", "FWD PTS (PIPS)"] = 100.0
    at.session_state["fx_quotes_EURUSD"] = base
    at.run()
    assert not at.exception
    d = fx_dates("EURUSD", TODAY, tenor="3M")
    assert _cell(at, f"FORWARD ({d.delivery.isoformat()})") == f"{EURUSD_SPOT + 0.01:.5f}"
    assert _cell(at, "FWD POINTS") == "+100.00 pips"
    assert not [n for n in at.number_input if n.key == "fx_rf_EURUSD"]  # rf is implied, not an input


def test_delta_and_atm_strike_modes_solve_on_the_smile():
    at = _fx()
    at.radio(key="fx_strike_mode").set_value("DELTA").run()
    at.number_input(key="fx_delta").set_value(25.0).run()
    d, m = _market("EURUSD", "3M", EURUSD_SPOT, 0.0375, 0.02)
    smile = build_smile(m, _quote("EURUSD", "3M"), SmileConventions("spot", "DNS", "market"))
    K = float(_cell(at, "STRIKE"))
    # The strike is shown to 0.1 pip (1e-5), so at the shown strike the delta is 25% only to within
    # half a 0.1 pip times d(delta)/dK (up to ~4e-5 here; a fixed 2e-5 failed on some trade dates).
    h = 1e-6
    slope = (
        fx_delta(m, K + h, smile.vol(K + h), True, "spot") - fx_delta(m, K - h, smile.vol(K - h), True, "spot")
    ) / (2 * h)
    assert fx_delta(m, K, smile.vol(K), True, "spot") == pytest.approx(0.25, abs=0.5e-5 * abs(slope) + 1e-7)
    assert K == pytest.approx(smile.strike_at_delta(0.25, True), abs=6e-6)
    at.radio(key="fx_strike_mode").set_value("ATM").run()
    assert float(_cell(at, "STRIKE")) == pytest.approx(smile.atm_strike, abs=6e-6)
    assert _cell(at, "VOL") == "7.500%"  # the ATM quote, at the DNS strike


def test_quote_conventions_and_models_change_the_smile_and_beta_is_fixed():
    """pricer-page-V3: beta is shown as FIXED, alpha as solved; conventions
    and the model change the strikes."""

    at = _fx()
    params = _frame(at, "PARAMETER")
    status = dict(zip(params["PARAMETER"], params["STATUS"]))
    assert status["β (beta)"].startswith("FIXED") and status["α (alpha)"].startswith("solved")
    assert status["ρ (rho)"] == "fitted" and status["ν (vol of vol)"] == "fitted"
    base = _frame(at, "PILLAR").set_index("PILLAR")["STRIKE"]
    at.selectbox(key="fx_delta_conv").set_value("forward_pa").run()
    pa = _frame(at, "PILLAR").set_index("PILLAR")["STRIKE"]
    assert pa["25C"] != base["25C"] and pa["ATM"] != base["ATM"]  # PA DNS strike is below the forward
    assert float(pa["ATM"]) < float(base["ATM"])
    at.selectbox(key="fx_model").set_value("VV").run()
    assert not at.exception
    assert any("checked, not fitted" in w.value for w in at.warning)
    assert "pillar (exact)" in set(_frame(at, "PARAMETER")["STATUS"])
    at.radio(key="fx_space").set_value("DELTA").run()
    smile = _charts(at)["fx_smile"]
    assert smile["layout"]["xaxis"]["ticktext"][0] == "5P" and "ATM" in smile["layout"]["xaxis"]["ticktext"]


def test_edited_quotes_reprice_and_bad_rows_are_reported():
    at = _fx()
    at.radio(key="fx_strike_mode").set_value("ATM").run()
    base = at.session_state["fx_quotes_EURUSD"].copy()
    base.loc[base["TENOR"] == "3M", "ATM (%)"] = 9.0
    base.loc[base["TENOR"] == "1Y", "ATM (%)"] = np.nan
    at.session_state["fx_quotes_EURUSD"] = base
    at.run()
    assert not at.exception
    assert _cell(at, "VOL") == "9.000%"
    assert any("Quote row 1Y: ATM, RR25 and BF25 are required" in e.value for e in at.error)
    assert any("EDITED</span>Your quotes in 3M, 1Y" in x.value for x in at.markdown)  # no longer 'illustrative'
    assert not any("ILLUSTRATIVE</span>" in x.value for x in at.markdown)
    surface = _charts(at)["fx_surface"]
    assert "1Y" not in surface["data"][0]["y"] and len(surface["data"][0]["y"]) == 11
    at.selectbox(key="fx_expiry").set_value("1Y").run()
    assert any("Cannot price this option" in e.value for e in at.error)
    at.radio(key="fx_vol_src").set_value("FLAT").run()
    at.number_input(key="fx_flat_vol").set_value(10.0).run()
    assert _cell(at, "VOL") == "10.000%"  # a flat vol still prices the broken tenor
    at.button(key="fx_reset_quotes").click().run()
    assert not any("Quote row" in e.value for e in at.error)


@pytest.mark.parametrize("atm, ok", [(0.0, False), (-5.0, False), (300.0, True), (300.5, False), (7.5, True)])
def test_atm_quote_bounds_are_stated_exactly(atm, ok):
    """Final review: an ATM of 0 was refused with 'must be between 0 and 300'."""

    from derivx.ui.fx_page import parse_quote_rows

    frame = pd.DataFrame(
        {
            "TENOR": ["3M"],
            "ATM (%)": [atm],
            "RR25 (%)": [-0.2],
            "BF25 (%)": [0.1],
            "RR10 (%)": [None],
            "BF10 (%)": [None],
        }
    )
    (row,) = parse_quote_rows(get_pair("EURUSD"), dt.date(2026, 10, 20), frame)
    if ok:
        assert row.error is None and row.quote is not None
    else:
        assert row.error == f"3M: ATM must be above 0 and at most 300 vol points (7.5 = 7.5%), got {atm:g}"


def test_editor_changes_are_merged_idempotently():
    base = pd.DataFrame({"TENOR": ["1M", "3M"], "ATM (%)": [7.0, 7.5], "RR25 (%)": [-0.2, -0.3]})
    state = {"edited_rows": {"1": {"ATM (%)": 8.0, "TENOR": "9Y"}, "0": {"RR25 (%)": None}}, "added_rows": [{}]}
    once = apply_editor_changes(base, state)
    assert once.loc[1, "ATM (%)"] == 8.0 and once.loc[1, "TENOR"] == "3M" and math.isnan(once.loc[0, "RR25 (%)"])
    pd.testing.assert_frame_equal(apply_editor_changes(once, state), once)
    assert base.loc[1, "ATM (%)"] == 7.5  # the input is not modified


def test_expiry_today_prices_the_discounted_intrinsic_and_draws_no_smile():
    """pricer-page-V1 / PP-06: no smile at 0% vol and no absurd Greeks at
    expiry: an EXPIRED state with the forward's discounted intrinsic."""

    at = _fx()
    at.number_input(key="fx_strike_EURUSD").set_value(1.10).run()
    at.selectbox(key="fx_expiry").set_value("DATE").run()
    at.date_input(key="fx_expiry_date").set_value(TODAY).run()
    assert not at.exception
    assert any("EXPIRES TODAY" in w.value for w in at.warning)
    assert "fx_smile" not in _charts(at) and any("EXPIRED" in i.value for i in at.info)
    d = fx_dates("EURUSD", TODAY, expiry=TODAY)
    m = FxMarket(EURUSD_SPOT, 0.0375, 0.02, 0.0, d.tau_d)
    assert _premium(at) == pytest.approx(NOTIONAL * m.df_d * (m.forward - 1.10), abs=0.01)
    cards = " ".join(x.value for x in at.markdown if "dx-greek-value" in x.value and "<style>" not in x.value)
    values = dict(re.findall(r"dx-greek-name'>([^<]+)</div><div class='dx-greek-value'>([^<]+)<", cards))
    assert values["GAMMA Γ"] == "EUR 0" and values["VEGA ν"] == "USD 0"


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("fx_spot_EURUSD", 0.0, "SPOT must be greater than 0"),
        ("fx_strike_EURUSD", -1.0, "STRIKE must be greater than 0"),  # below min_value: reset to 0
    ],
)
def test_bad_inputs_show_a_friendly_error(key, value, message):
    """PP-05 on the FX page: no traceback, nothing priced."""

    at = _fx()
    at.number_input(key=key).set_value(value).run()
    assert not at.exception
    assert any(message in e.value for e in at.error)
    assert not at.get("plotly_chart")


def test_live_ecb_spot_is_fetched_on_request_with_an_offline_fallback(monkeypatch):
    from derivx.ui import fx_page

    fx_page._live_ecb.clear()
    live = market_data.parse_ecb_csv(FIXTURE.read_text())
    live = dataclasses.replace(live, per_eur={**live.per_eur, "USD": 1.1300})
    calls = []
    monkeypatch.setattr(market_data, "fetch_ecb_rates", lambda *a, **k: calls.append(1) or live)
    at = _fx()
    assert not calls  # nothing is fetched until asked
    assert "bundled snapshot" in " ".join(c.value for c in at.sidebar.caption)
    at.button(key="fx_fetch_ecb").click().run()
    assert at.number_input(key="fx_spot_EURUSD").value == pytest.approx(1.13)
    assert "fetched" in " ".join(c.value for c in at.sidebar.caption) and _cell(at, "SPOT") == "1.13000"

    fx_page._live_ecb.clear()

    def offline(*a, **k):
        raise market_data.FxDataError("ECB Data Portal not reachable: timed out")

    monkeypatch.setattr(market_data, "fetch_ecb_rates", offline)
    at2 = _fx()
    at2.button(key="fx_fetch_ecb").click().run()
    assert not at2.exception
    assert any("not reachable" in w.value and "fallback" in w.value for w in at2.sidebar.warning)
    assert at2.number_input(key="fx_spot_EURUSD").value == pytest.approx(EURUSD_SPOT)
    fx_page._live_ecb.clear()


def test_reruns_are_deterministic_and_no_equity_chain_is_used():
    """PP-01 / VM-14 (FX side): no random 'MARKET' smile; an equity chain in
    the session is never offered as an FX smile."""

    at = _fx(market_chain=synth_chain(days=(30, 91), spot=580.0))
    first = [c.proto.spec for c in at.get("plotly_chart")]
    at.run()
    assert [c.proto.spec for c in at.get("plotly_chart")] == first
    names = [t["name"] for t in _charts(at)["fx_smile"]["data"]]
    assert names[0] == "SABR smile" and not any("MARKET" in n for n in names)
    assert not [r for r in at.radio if r.key and "smile_src" in r.key]


def test_mode_round_trip_keeps_fx_inputs_and_equity_keys_are_separate():
    at = _fx()
    at.number_input(key="fx_spot_EURUSD").set_value(1.0999).run()
    at.selectbox(key="fx_expiry").set_value("1Y").run()
    at.sidebar.radio[0].set_value("[ EQUITY ]").run()
    assert at.number_input(key="eq_spot").value == pytest.approx(100.0)  # not the FX market
    at.sidebar.radio[0].set_value("[ RATES ]").run()
    at.sidebar.radio[0].set_value("[ FX ]").run()
    assert not at.exception
    assert at.number_input(key="fx_spot_EURUSD").value == pytest.approx(1.0999)
    assert at.selectbox(key="fx_expiry").value == "1Y"


def test_smallest_scenario_shocks_render_distinct_labels():
    at = _fx()
    at.number_input(key="fx_scn_spot").set_value(0.1).run()
    at.number_input(key="fx_scn_vol").set_value(0.1).run()
    assert not at.exception
    chart = _charts(at)["fx_scenario"]["data"][0]
    assert len(set(chart["x"])) == len(chart["x"]) == 9 and len(set(chart["y"])) == len(chart["y"])
    assert chart["x"][0] == "-0.10%" and chart["x"][-1] == "+0.10%"


def test_put_sale_and_scenario_signs():
    at = _fx()
    at.radio(key="fx_type").set_value("PUT").run()
    at.radio(key="fx_side").set_value("SELL").run()
    assert "EUR PUT / USD CALL" in _hero(at) and "RECEIVE on the spot date" in _hero(at)
    hedge = next(r for _, r in _frame(at, "DELTA TYPE").iterrows() if "hedge" in r["DELTA TYPE"])
    assert float(hedge["AMOUNT (EUR)"].replace(",", "")) > 0  # short a put: long EUR delta
    z = _charts(at)["fx_scenario"]["data"][0]
    y = z["y"]
    from test_pricer_page import _arr

    pnl = _arr(z["z"])
    assert pnl[y.index("+0"), -1] > 0 > pnl[y.index("+0"), 0]  # spot up: the short put gains; down: it loses


def test_density_warning_for_an_arbitrageable_smile():
    at = _fx()
    base = at.session_state["fx_quotes_EURUSD"].copy()
    columns = ["ATM (%)", "RR25 (%)", "BF25 (%)", "RR10 (%)", "BF10 (%)"]
    base.loc[base["TENOR"] == "2Y", columns] = [10.0, 0.0, 3.0, np.nan, np.nan]  # a 3-vol-point 2Y fly
    at.session_state["fx_quotes_EURUSD"] = base
    at.run()
    at.selectbox(key="fx_expiry").set_value("2Y").run()
    at.selectbox(key="fx_model").set_value("VV").run()
    at.slider(key="fx_smile_sd").set_value(4.0).run()
    assert not at.exception
    assert any("Butterfly arbitrage" in w.value and "vanna-volga" in w.value for w in at.warning)


def test_pair_switch_uses_its_own_calendar_and_spot_lag():
    at = _fx()
    at.selectbox(key="fx_pair").set_value("USDCAD").run()
    assert not at.exception
    d = fx_dates(get_pair("USDCAD"), TODAY, tenor="3M")
    assert f"{d.spot.isoformat()} (T+1)" in _frame(at, "DATE / CONVENTION")["VALUE"].tolist()
    assert d.expiry.isoformat() in _cell(at, "EXPIRY")


def test_expiry_between_tenors_interpolates_the_quotes():
    at = _fx()
    expiry = fx_dates("EURUSD", TODAY, tenor="1M").expiry + dt.timedelta(days=14)
    at.selectbox(key="fx_expiry").set_value("DATE").run()
    at.date_input(key="fx_expiry_date").set_value(expiry).run()
    assert not at.exception
    assert any("interpolated between 1M and 2M" in x.value for x in at.markdown)
    d = fx_dates("EURUSD", TODAY, expiry=expiry)
    m = FxMarket(EURUSD_SPOT, 0.0375, 0.02, d.tau_e, d.tau_d)
    from derivx.fx.smile import interpolate_quote

    pts = [(fx_dates("EURUSD", TODAY, tenor=t).tau_e, _quote("EURUSD", t)) for t in ("1M", "2M")]
    quote, _ = interpolate_quote(pts, d.tau_e, expiry.isoformat())
    smile = build_smile(m, quote, SmileConventions("spot", "DNS", "market"))
    assert _cell(at, "VOL") == f"{100 * smile.vol(1.12):.3f}%"
    assert _cell(at, "EXPIRY") == f"{expiry.isoformat()} · {(expiry - TODAY).days}D"


def test_edited_spot_is_flagged_against_its_reference():
    at = _fx()
    assert not any("EDITED" in c.value for c in at.sidebar.caption)
    at.number_input(key="fx_spot_EURUSD").set_value(1.1300).run()
    assert any(c.value.startswith("EDITED. Reference: 1.12340 = ECB") for c in at.sidebar.caption)


def test_widgets_shared_by_every_pair_keep_their_labels_and_options():
    """A keyed widget whose label or options change with the pair is a new
    widget to the browser, which then shows no selection (seen for the
    OPTION radio): shared widgets stay identical across pairs."""

    at = _fx()

    def shared():
        out = {}
        for w in list(at.radio) + list(at.selectbox) + list(at.number_input) + list(at.slider):
            if w.key and w.key.startswith("fx_") and not w.key.endswith(("_EURUSD", "_USDJPY")):
                out[w.key] = (w.label, tuple(getattr(w, "options", ()) or ()), w.proto.help)
        return out

    before = shared()
    at.selectbox(key="fx_pair").set_value("USDJPY").run()
    after = shared()
    assert before == after and "fx_type" in before
    assert at.radio(key="fx_type").value == "CALL"


POINTS = "FWD PTS (PIPS)"


def _badges(at: AppTest) -> str:
    return " ".join(x.value for x in at.markdown if "dx-badge" in x.value)


def test_missing_forward_points_are_never_replaced_by_a_made_up_rate():
    """Review: with CARRY FROM FORWARD POINTS and no points for the expiry, a
    FLAT vol priced the option on a foreign rate of 0% (the forward came
    from nowhere, and the dates panel called it 'implied from the forward
    points'). Nothing is priced without a forward."""

    at = _fx()
    at.radio(key="fx_carry").set_value("FORWARD POINTS").run()
    at.radio(key="fx_vol_src").set_value("FLAT").run()
    base = at.session_state["fx_quotes_EURUSD"].copy()
    base.loc[base["TENOR"] == "3M", "FWD PTS (PIPS)"] = np.nan
    at.session_state["fx_quotes_EURUSD"] = base
    at.run()
    assert not at.exception
    assert any(
        "Cannot price this option: no forward for this expiry (3M: no forward points)" in e.value for e in at.error
    )
    assert not [m for m in at.markdown if "dx-hero-value" in m.value and "<style>" not in m.value]
    assert "fx_scenario" not in _charts(at)
    # an expiry between tenors with no points anywhere: the same
    base[POINTS] = np.nan
    at.session_state["fx_quotes_EURUSD"] = base
    at.selectbox(key="fx_expiry").set_value("DATE").run()
    assert not at.exception
    assert any("no forward for this expiry" in e.value for e in at.error)


def test_illustrative_carry_is_labelled_until_the_user_enters_it():
    """Review: the default deposit rates (and the forward points computed
    from them) are round numbers, but were shown as plain inputs with no
    ILLUSTRATIVE label; the forward, its points and the premium depend on
    them."""

    at = _fx()
    assert "ILLUSTRATIVE CARRY</span>USD rate 3.75%, EUR rate 2.00%: defaults, not market data" in _badges(at)
    assert any(
        c.value.startswith("ILLUSTRATIVE: the USD and EUR rates are round-number defaults") for c in at.sidebar.caption
    )
    dates = _frame(at, "DATE / CONVENTION").set_index("DATE / CONVENTION")["VALUE"]
    assert "ILLUSTRATIVE defaults: USD rate 3.75%, EUR rate 2.00%" in dates["Rates"]
    at.number_input(key="fx_rd_EURUSD").set_value(3.9).run()
    assert "ILLUSTRATIVE CARRY</span>EUR rate 2.00%:" in _badges(at)
    at.number_input(key="fx_rf_EURUSD").set_value(1.95).run()
    assert "ILLUSTRATIVE CARRY" not in _badges(at)
    assert not any(c.value.startswith("ILLUSTRATIVE") for c in at.sidebar.caption)
    # forward points: the defaults were computed from the default rates
    at.radio(key="fx_carry").set_value("FORWARD POINTS").run()
    assert "ILLUSTRATIVE CARRY</span>forward points (computed from the default rates):" in _badges(at)
    base = at.session_state["fx_quotes_EURUSD"].copy()
    base.loc[base["TENOR"] == "3M", POINTS] = 51.25
    at.session_state["fx_quotes_EURUSD"] = base
    at.run()
    assert "ILLUSTRATIVE CARRY" not in _badges(at) and _cell(at, "FWD POINTS") == "+51.25 pips"
    at.selectbox(key="fx_expiry").set_value("6M").run()  # 6M still has the default points
    assert "forward points (computed from the default rates)" in _badges(at)
    at.button(key="fx_points_from_rates").click().run()  # points from the user's rates: not illustrative
    assert "ILLUSTRATIVE CARRY" not in _badges(at)


def test_trade_and_expiry_dates_on_holidays_are_flagged():
    """Review: an expiry DATE on a Saturday or on Christmas Day priced with
    no word, and the trade-date warning sat inside a collapsed expander."""

    at = _fx()
    at.date_input(key="fx_trade_date").set_value(dt.date(2026, 10, 20)).run()
    at.selectbox(key="fx_expiry").set_value("DATE").run()
    at.date_input(key="fx_expiry_date").set_value(dt.date(2026, 11, 21)).run()  # a Saturday
    assert not at.exception
    msg = [w.value for w in at.warning if w.value.startswith("EXPIRY 2026-11-21 is not a business day")]
    assert msg and "next one 2026-11-23" in msg[0]
    at.date_input(key="fx_expiry_date").set_value(dt.date(2026, 12, 25)).run()  # TARGET and Fedwire holiday
    assert any(w.value.startswith("EXPIRY 2026-12-25 is not a business day") for w in at.warning)
    at.date_input(key="fx_expiry_date").set_value(dt.date(2026, 12, 23)).run()
    assert not any("is not a business day" in w.value for w in at.warning)
    at.date_input(key="fx_trade_date").set_value(dt.date(2026, 10, 24)).run()  # a Saturday
    assert any(w.value.startswith("TRADE DATE 2026-10-24 is not a business day") for w in at.warning)
    for block in at.expander:
        assert not any("is not a business day" in w.value for w in block.warning)  # shown, not tucked away


def test_priced_strike_outside_the_plotted_smile_is_mentioned():
    at = _fx()
    at.number_input(key="fx_strike_EURUSD").set_value(1.40).run()
    assert not at.exception
    assert "K (PRICED)" not in [t["name"] for t in _charts(at)["fx_smile"]["data"]]
    assert any(c.value.startswith("K (PRICED) = 1.40000 is outside the plotted strikes") for c in at.caption)


def test_surface_says_which_tenors_have_no_forward():
    at = _fx()
    at.radio(key="fx_carry").set_value("FORWARD POINTS").run()
    base = at.session_state["fx_quotes_EURUSD"].copy()
    base.loc[base["TENOR"].isin(["9M", "2Y"]), POINTS] = np.nan
    at.session_state["fx_quotes_EURUSD"] = base
    at.run()
    assert not at.exception
    assert _charts(at)["fx_surface"]["data"][0]["y"] == ["ON", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "1Y", "18M"]
    assert any("Left out, no forward (forward points missing or unusable): 9M, 2Y." in c.value for c in at.caption)


def _greek_cards(at: AppTest) -> dict:
    block = next(m.value for m in at.markdown if "class='dx-greeks'" in m.value)
    out = {}
    for card in block.split("<div class='dx-greek'>")[1:]:
        name = html.unescape(re.search(r"dx-greek-name'>([^<]+)<", card).group(1))
        value = html.unescape(re.search(r"dx-greek-value'>([^<]+)<", card).group(1))
        out[name] = (value, [html.unescape(s) for s in re.findall(r"dx-greek-sub'>([^<]+)<", card)])
    return out


@pytest.mark.parametrize("notional", [1.0, 1e8, 1e9, 1e12])
def test_greek_card_values_fit_their_card_at_any_notional(notional):
    """Review: at 1280px a EUR 100m notional's gamma showed as 'EUR 10,336,7'
    (the card cut the value off) and tiny values as 'USD -1.597e-'. Card
    values now fit CARD_VALUE_CHARS (the narrowest card of FX_CSS) and a
    shortened value keeps its exact amount on the line below."""

    from derivx.ui.fx_page import CARD_VALUE_CHARS

    at = _fx()
    at.number_input(key="fx_notional").set_value(notional).run()
    assert not at.exception
    cards = _greek_cards(at)
    assert len(cards) == 8
    for name, (value, subs) in cards.items():
        assert len(value) <= CARD_VALUE_CHARS, (name, value)
        if any(s.startswith("= ") for s in subs):  # shortened: the exact amount is given
            exact = next(s for s in subs if s.startswith("= "))[2:]
            short = float(re.sub(r"[kMBT]$", "", value.split(" ")[1]))
            scale = {"k": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(value[-1], 1.0)
            assert short * scale == pytest.approx(float(exact.split(" ")[1].replace(",", "")), rel=6e-4)
    gamma, subs = cards["GAMMA Γ"]
    if notional >= 1e9:  # EUR 103,367,032 at 1bn: shortened to EUR 103.4M, exact amount below
        assert gamma[-1] in "MB" and subs[0].startswith("= EUR ") and subs[-1].endswith("per 1 EUR")
    elif notional == 1e8:  # EUR 10,336,703 fits as it is
        assert re.fullmatch(r"EUR [\d,]+", gamma) and not subs[0].startswith("= ")


def test_forward_of_an_expiry_between_tenors_says_how_it_was_interpolated():
    """Review: with CARRY FROM FORWARD POINTS an expiry DATE between tenors
    (or after the last one) took its forward from interpolated (or flat
    extrapolated) implied foreign rates without a word on screen."""

    at = _fx()
    at.radio(key="fx_carry").set_value("FORWARD POINTS").run()
    one, two = fx_dates("EURUSD", TODAY, tenor="1M"), fx_dates("EURUSD", TODAY, tenor="2M")
    expiry = one.expiry + dt.timedelta(days=14)
    at.selectbox(key="fx_expiry").set_value("DATE").run()
    at.date_input(key="fx_expiry_date").set_value(expiry).run()
    assert not at.exception
    note = "forward from the implied EUR rates interpolated between 1M and 2M (linear in rate time)"
    assert any(c.value == f"FORWARD: {note}." for c in at.caption)
    rates = _frame(at, "DATE / CONVENTION").set_index("DATE / CONVENTION")["VALUE"]["Rates"]
    assert note in rates
    # the forward: implied EUR rates of the 1M and 2M points, linear in the rate time
    frame = at.session_state["fx_quotes_EURUSD"].set_index("TENOR")[POINTS]

    def implied(dates, pts):
        return 0.0375 - math.log((EURUSD_SPOT + pts * 1e-4) / EURUSD_SPOT) / dates.tau_d

    d = fx_dates("EURUSD", TODAY, expiry=expiry)
    rf = np.interp(d.tau_d, [one.tau_d, two.tau_d], [implied(one, frame["1M"]), implied(two, frame["2M"])])
    assert _cell(at, f"FORWARD ({d.delivery.isoformat()})") == f"{EURUSD_SPOT * math.exp((0.0375 - rf) * d.tau_d):.5f}"
    # after the last tenor: flat
    at.date_input(key="fx_expiry_date").set_value(
        fx_dates("EURUSD", TODAY, tenor="2Y").expiry + dt.timedelta(days=60)
    ).run()
    assert any(
        c.value == "FORWARD: forward from the 2Y implied EUR rate held flat (after the last tenor with forward points)."
        for c in at.caption
    )
    # an expiry today delivers on the spot date: the forward is spot, no points needed and no note
    base = at.session_state["fx_quotes_EURUSD"].copy()
    base[POINTS] = np.nan
    at.session_state["fx_quotes_EURUSD"] = base
    at.date_input(key="fx_expiry_date").set_value(TODAY).run()
    assert not at.exception and not at.error
    assert _premium(at) == pytest.approx(NOTIONAL * (EURUSD_SPOT - 1.12), abs=0.01)
    assert not any(c.value.startswith("FORWARD:") for c in at.caption)
    # rates carry: no such note
    at.date_input(key="fx_expiry_date").set_value(expiry).run()
    at.radio(key="fx_carry").set_value("RATES").run()
    assert not any(c.value.startswith("FORWARD:") for c in at.caption)


def test_greeks_caption_states_what_each_rho_holds_fixed():
    at = _fx()
    text = next(c.value for c in at.caption if c.value.startswith("For the long position"))
    assert "Each rho bumps one deposit rate with spot and the other rate fixed, so the forward moves;" in text
    at.radio(key="fx_carry").set_value("FORWARD POINTS").run()
    text = next(c.value for c in at.caption if c.value.startswith("For the long position"))
    assert "(the foreign rate is the one implied by the forward points)" in text


def test_smile_risk_shows_vega_by_quote_and_by_sabr_parameter():
    """VM-18: the priced option's vega to each ATM / RR / BF quote (smile
    rebuilt from the bumped quotes) and to the SABR parameters."""

    at = _fx()
    table = _frame(at, "POSITION (USD)").set_index("QUOTE")
    assert list(table.index) == ["ATM", "RR25", "BF25", "RR10", "BF10"]
    d, m = _market("EURUSD", "3M", EURUSD_SPOT, 0.0375, 0.02)
    conv = SmileConventions(market_delta_type("EURUSD", d), "DNS", "market")
    direct = quote_vegas(m, _quote("EURUSD", "3M"), conv, "SABR", 1.12, True).set_index("quote")["vega"]
    shown = table["PER 1 EUR (USD)"].astype(float)
    np.testing.assert_allclose(shown.to_numpy(), direct.to_numpy(), atol=2e-6)
    assert shown["ATM"] > 0 and (table["FITTED"] == "yes").all()
    params = _frame(at, "SABR PARAMETER")
    assert list(params["SABR PARAMETER"]) == ["alpha", "rho", "nu"]
    assert any("SMILE RISK" in m_.value for m_ in at.markdown)
