"""Yahoo access and lineage (AL-02, AL-14, AL-15, UIE-13), liquidity
filters and the hygiene report (AL-10). No test touches the network: the
client is :class:`_fake_yahoo.FakeTicker`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _fake_yahoo import Clock, FakeTicker

from derivx.data import calendar
from derivx.data.ingestion import load_equity_chain_from_yf
from derivx.data.market_data import (
    Backoff,
    apply_liquidity_filters,
    classify_error,
    compute_moneyness,
    fetch_option_chain,
    hygiene_report,
)
from derivx.data.samples import load_sample_chain


def _fetch(fake: FakeTicker, waits=None, **kw):
    waits = [] if waits is None else waits
    backoff = Backoff(retries=3, base_delay=1.0, max_delay=8.0, sleep=waits.append, rng=np.random.default_rng(7))
    return fetch_option_chain("spy", backoff=backoff, client_factory=lambda s: fake, now=Clock(), **kw)


def test_rows_carry_the_session_the_real_fetch_time_and_the_spot_of_the_same_response():
    fake = FakeTicker()
    res = _fetch(fake, max_expiries=3)
    st, chain = res.status, res.chain
    assert st.status == "ok" and st.ok and not st.partial and st.quote_date == "2026-10-20"
    assert set(chain["trade_date"]) == {pd.Timestamp("2026-10-20")}  # the session, not the server's date
    assert chain["fetched_at_utc"].str.endswith("Z").all()
    assert chain.groupby("expiry")["fetched_at_utc"].nunique().eq(1).all()  # one fetch time per response
    assert set(chain["underlying_price"]) == {769.05} and st.spot == 769.05 and st.spot_drift == 0.0
    assert st.spot_time_utc == "2026-10-20T20:00:00Z"
    assert set(chain["ticker"]) == {"SPY"} and set(chain["source"]) == {"yahoo"}


def test_a_fetch_after_midnight_utc_keeps_the_new_york_session():
    fake = FakeTicker()
    res = fetch_option_chain(
        "SPY",
        max_expiries=1,
        client_factory=lambda s: fake,
        now=Clock("2026-10-21T02:30:00Z"),
        backoff=Backoff(sleep=lambda s: None),
    )
    assert res.status.quote_date == "2026-10-20"  # 22:30 in New York: still the 20th's session
    assert set(res.chain["trade_date"]) == {pd.Timestamp("2026-10-20")}


def test_a_weekend_fetch_is_labelled_with_the_friday_session():
    fake = FakeTicker(spot_time=int(pd.Timestamp("2026-10-16T20:00:00Z").timestamp()))
    res = fetch_option_chain(
        "SPY",
        max_expiries=1,
        client_factory=lambda s: fake,
        now=Clock("2026-10-18T15:00:00Z"),
        backoff=Backoff(sleep=lambda s: None),
    )
    assert res.status.quote_date == "2026-10-16"
    assert any("belong to the 2026-10-16 session" in n for n in res.status.notes)


def test_without_a_spot_time_the_session_comes_from_the_last_trades():
    res = _fetch(FakeTicker(spot_time=None), max_expiries=2)
    assert res.status.quote_date == "2026-10-20" and res.status.spot_time_utc is None


def test_only_the_requested_expiries_are_downloaded():
    fake = FakeTicker()
    res = _fetch(fake, max_expiries=2)
    assert fake.requests == ["options", "2026-10-20", "2026-10-21"]
    assert res.status.expiries_requested == ["2026-10-20", "2026-10-21"] and res.status.requests == 3
    fake2 = FakeTicker()
    _fetch(fake2, expiries=["2026-11-20"])
    assert fake2.requests == ["options", "2026-11-20"]


def test_expiries_too_close_to_fit_are_not_requested_when_asked():
    """After the close Yahoo still lists the day's expired expiry: a lab pull
    of the nearest N expiries used to spend two of its N requests on the 0-
    and 1-day expiries that no fit can use."""

    fake = FakeTicker()
    res = _fetch(fake, max_expiries=3, min_days=2)  # the clock reads 2026-10-20 17:00 New York
    assert fake.requests == ["options", "2026-10-22", "2026-10-23", "2026-10-26"]
    assert res.status.status == "ok" and res.status.expiries_requested == ["2026-10-22", "2026-10-23", "2026-10-26"]
    assert any("not requested: 2 expiry(ies) fewer than 2 days after 2026-10-20" in n for n in res.status.notes)
    fake = FakeTicker()
    res = _fetch(fake, expiries=["2026-10-20"], min_days=2)
    assert res.status.status == "empty" and fake.requests == ["options"]


def test_unknown_expiries_and_bad_inputs_raise():
    with pytest.raises(ValueError, match="not listed by Yahoo"):
        _fetch(FakeTicker(), expiries=["2030-01-18"])
    with pytest.raises(ValueError, match="ticker"):
        fetch_option_chain("not a ticker!")
    with pytest.raises(ValueError, match="max_expiries"):
        fetch_option_chain("SPY", max_expiries=0)


def test_rate_limits_are_retried_with_jittered_exponential_backoff():
    waits = []
    fake = FakeTicker(failures={"options": 2, "2026-10-21": 1})
    res = _fetch(fake, waits=waits, max_expiries=2, request_pause=0.0)
    assert res.status.status == "ok" and res.status.retries == 3
    assert fake.requests == ["options", "options", "options", "2026-10-20", "2026-10-21", "2026-10-21"]
    # retry i waits min(8, 2^i) * U[0.5, 1)
    assert 0.5 <= waits[0] < 1.0 and 1.0 <= waits[1] < 2.0 and 0.5 <= waits[2] < 1.0
    assert res.status.waited_s == pytest.approx(sum(waits), abs=1e-3)


def test_a_persistent_rate_limit_gives_a_partial_chain_and_stops_requesting():
    fake = FakeTicker(failures={"2026-10-21": -1})
    res = _fetch(fake, max_expiries=4, request_pause=0.0)
    st = res.status
    assert st.status == "partial" and st.partial and st.ok
    assert st.expiries_fetched == ["2026-10-20"]
    assert st.failed_expiries["2026-10-21"].startswith("rate_limited")
    assert st.failed_expiries["2026-10-22"].startswith("not requested")  # no hammering after the limit
    assert "2026-10-22" not in fake.requests and fake.requests.count("2026-10-21") == 4
    assert set(res.chain["expiry"].dt.strftime("%Y-%m-%d")) == {"2026-10-20"}


def test_a_rate_limited_listing_is_reported_not_swallowed():
    res = _fetch(FakeTicker(failures={"options": -1}), max_expiries=2)
    assert res.status.status == "rate_limited" and not res.status.ok and res.chain.empty
    assert "rate-limited" in res.status.reason and "4 attempt" in res.status.reason
    assert res.status.summary().startswith("RATE_LIMITED")


def test_fatal_errors_are_not_retried():
    fake = FakeTicker(fatal={"options": KeyError("optionChain")})
    res = _fetch(fake, max_expiries=2)
    assert res.status.status == "error" and fake.requests == ["options"]
    assert "KeyError" in res.status.reason


def test_error_classification():
    from yfinance.exceptions import YFRateLimitError

    assert classify_error(YFRateLimitError()) == "rate_limited"
    assert classify_error(RuntimeError("HTTP Error 429: Too Many Requests")) == "rate_limited"
    assert classify_error(ConnectionError("reset")) == "transient"
    assert classify_error(TimeoutError()) == "transient"
    assert classify_error(ValueError("Expiration `x` cannot be found")) == "fatal"


def test_past_dates_are_refused_before_any_request(monkeypatch):
    calls = []
    monkeypatch.setattr("derivx.data.ingestion.fetch_option_chain", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="only the current chain"):
        load_equity_chain_from_yf("SPY", "2024-01-02")
    assert calls == []


def test_the_loader_returns_the_chain_or_raises_with_the_status():
    from derivx.data.ingestion import ChainFetchError

    fake = FakeTicker()
    chain = load_equity_chain_from_yf(
        "SPY", max_expiries=2, client_factory=lambda s: fake, backoff=Backoff(sleep=lambda s: None)
    )
    assert chain["expiry"].nunique() == 2
    with pytest.raises(ChainFetchError, match="RATE_LIMITED"):
        load_equity_chain_from_yf(
            "SPY", client_factory=lambda s: FakeTicker(failures={"options": -1}), backoff=Backoff(retries=0)
        )


def test_liquidity_filters_keep_both_option_types():
    data = pd.DataFrame(
        {
            "bid": [1.0, 0.0, 0.5, 0.6],
            "ask": [1.2, 0.5, 1.5, 0.8],
            "volume": [50, 100, 5, 30],
            "strike": [100, 105, 110, 115],
            "underlying_price": [120, 120, 120, 120],
        }
    )
    filtered = apply_liquidity_filters(data, min_volume=10, max_spread=0.2, min_mid=0.5)
    assert filtered["strike"].tolist() == [100]
    chain = load_sample_chain().single()
    f = apply_liquidity_filters(chain, 10, 0.4, 0.1)
    assert set(f["option_type"]) == {"call", "put"}  # parity needs both
    with pytest.raises(ValueError, match="no 'volume' column.*set min volume to 0"):
        apply_liquidity_filters(data.drop(columns=["volume"]))
    with pytest.raises(ValueError, match="missing column"):
        apply_liquidity_filters(data.drop(columns=["bid"]))
    with pytest.raises(ValueError, match="max_spread"):
        apply_liquidity_filters(data, max_spread=0.0)


def test_a_chain_without_volume_is_usable_with_min_volume_zero():
    """volume is optional in the chain schema: the filters must not need it
    when no volume filter is asked for (a file with only the required
    columns used to crash the lab with a traceback)."""

    data = pd.DataFrame({"bid": [1.0, 0.0, 0.5], "ask": [1.2, 0.5, 1.5], "strike": [100, 105, 110]})
    kept = apply_liquidity_filters(data, min_volume=0, max_spread=None, min_mid=0.5)
    assert kept["strike"].tolist() == [100, 110]  # bid > 0, ask > bid and mid >= 0.5; volume not consulted
    with_nan = data.assign(volume=np.nan)
    assert apply_liquidity_filters(with_nan, min_volume=1, max_spread=None, min_mid=0.0).empty  # NaN counts as 0


def test_compute_moneyness_is_spot_moneyness():
    df = pd.DataFrame({"strike": [100, 105], "underlying_price": [120, 120]})
    assert np.allclose(compute_moneyness(df)["moneyness"], [100 / 120, 105 / 120])
    with pytest.raises(ValueError):
        compute_moneyness(pd.DataFrame({"strike": [100]}))


def test_hygiene_report_on_the_real_sample():
    chain = apply_liquidity_filters(load_sample_chain().single(), 10, 0.4, 0.1)
    rep = hygiene_report(chain)
    assert len(rep) == chain["expiry"].nunique() == 31
    same_day = rep[rep["days"] == 0].iloc[0]
    assert same_day["kept"] == 0 and same_day["removed_expired"] == same_day["quotes_in"]
    live = rep[rep["days"] >= 2]
    assert (live["kept"] > 0).all() and (live["removed_itm"] > 0).all()
    assert set(live["forward_method"]) <= {"parity", "interpolated", "carry"}
    assert (live["forward"] > 769.0).all()  # parity forwards above spot (positive carry)
    for exp in live["expiry"][:5]:
        g = chain[pd.to_datetime(chain["expiry"]).dt.date == exp]
        assert int(rep.loc[rep["expiry"] == exp, "kept"].iloc[0]) <= g["strike"].nunique()  # one quote per strike


def test_calendar_rules():
    assert calendar.is_trading_day("2026-10-19") and not calendar.is_trading_day("2026-11-26")  # Thanksgiving
    assert not calendar.is_trading_day("2026-04-03")  # Good Friday
    assert calendar.trading_days_between("2026-10-16", "2026-10-19") == 1  # Fri -> Mon
    assert calendar.trading_days_between("2026-11-25", "2026-11-27") == 1  # over Thanksgiving
    assert calendar.trading_days_between("2026-10-19", "2026-10-16") == -1
    assert calendar.add_trading_days("2026-11-25", 1) == pd.Timestamp("2026-11-27")
    # from a weekend or holiday: the n-th session after it (it used to roll forward first, one session too far)
    assert calendar.add_trading_days("2026-10-24", 1) == pd.Timestamp("2026-10-26")  # Saturday -> Monday
    assert calendar.add_trading_days("2026-11-26", 1) == pd.Timestamp("2026-11-27")  # Thanksgiving -> Friday
    assert calendar.add_trading_days("2026-11-26", 0) == pd.Timestamp("2026-11-27")
    for start in pd.date_range("2026-11-20", "2027-01-10"):
        for n in (1, 2, 5):
            assert calendar.trading_days_between(start, calendar.add_trading_days(start, n)) == n
    assert calendar.session_date("2026-10-21T02:00:00Z") == pd.Timestamp("2026-10-20")
    assert calendar.previous_trading_day("2026-10-18") == pd.Timestamp("2026-10-16")
    assert calendar.missing_trading_days(["2026-10-15", "2026-10-19"]) == [pd.Timestamp("2026-10-16")]
    assert pd.Timestamp("2026-06-19").date() in calendar.holidays()  # Juneteenth
