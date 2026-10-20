"""FX conventions: the FX settlement calendars, spot / expiry / delivery dates
(QuantLib golden values), pair metadata and forwards."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from derivx.fx.conventions import (
    CURRENCY_CALENDARS,
    OPTION_TENORS,
    PAIRS,
    expiry_and_delivery,
    forward_from_points,
    forward_from_rates,
    forward_points,
    fx_dates,
    get_pair,
    implied_foreign_rate,
    market_delta_type,
    spot_date,
)
from derivx.rates.conventions import Calendar

D = dt.date.fromisoformat


def test_fx_calendars_match_quantlib_on_every_day(golden):
    """USFED, TOKYO, ZURICH, TORONTO, SYDNEY, WELLINGTON, AUCKLAND against
    QuantLib's holiday lists, 1990-2080."""

    data = golden("fx")
    start, end = map(D, data["holiday_range"])
    assert set(data["holidays"]) == {"USFED", "TOKYO", "ZURICH", "TORONTO", "SYDNEY", "WELLINGTON", "AUCKLAND"}
    for name, expected in data["holidays"].items():
        assert Calendar(name).holidays(start, end) == [D(x) for x in expected], name


@pytest.mark.parametrize(
    "name, day, holiday",
    [
        ("USFED", "2026-04-03", False),  # Good Friday: Fedwire is open (SOFR's calendar is not)
        ("USFED", "2027-07-05", True),  # Independence Day on a Sunday -> Monday
        ("USFED", "2026-07-03", False),  # ... but a Saturday holiday is not moved to Friday
        ("TOKYO", "2026-09-22", True),  # citizens' holiday between Respect for the Aged Day and the equinox
        ("TOKYO", "2026-01-02", True),  # bank holiday
        ("ZURICH", "2026-05-14", True),  # Ascension Day
        ("TORONTO", "2026-09-30", True),  # National Day for Truth and Reconciliation
        ("SYDNEY", "2026-01-26", True),  # Australia Day
        ("WELLINGTON", "2026-07-10", True),  # Matariki
        ("WELLINGTON", "2026-01-19", True),  # Wellington Anniversary Day
        ("AUCKLAND", "2026-01-26", True),  # Auckland Anniversary Day
    ],
)
def test_calendar_spot_checks(name, day, holiday):
    assert Calendar(name).is_holiday(D(day)) is holiday


def test_joint_calendar_advance_matches_quantlib_joint_calendar(golden):
    """``ql.JointCalendar(base, quote).advance(d, lag, Days)`` for every
    calendar day of 2025-2027 and every pair."""

    data = golden("fx")
    start = D(data["dates_start"])
    for code, offsets in data["joint_advance"].items():
        pair = get_pair(code)
        for i, offset in enumerate(offsets):
            day = start + dt.timedelta(days=i)
            assert pair.calendar.add_business_days(day, pair.spot_lag) == day + dt.timedelta(days=offset), (code, day)


def test_spot_dates_match_the_quantlib_reference(golden):
    """Spot dates by the market rule (USD T+1 exception, USD on spot for
    crosses, USDCAD T+1), every calendar day of 2025-2027."""

    data = golden("fx")
    start = D(data["dates_start"])
    differs = 0
    for code, offsets in data["spot_dates"].items():
        for i, offset in enumerate(offsets):
            day = start + dt.timedelta(days=i)
            assert spot_date(code, day) == day + dt.timedelta(days=offset), (code, day)
            differs += offset != data["joint_advance"][code][i]
    assert differs > 50  # the market rule is not the plain joint-calendar advance


def test_tenor_expiry_and_delivery_dates_match_the_quantlib_reference(golden):
    rows = golden("fx")["tenor_dates"]
    assert {r[2] for r in rows} == set(OPTION_TENORS)
    for code, trade, tenor, expiry, delivery in rows:
        assert expiry_and_delivery(code, D(trade), tenor) == (D(expiry), D(delivery)), (code, trade, tenor)


def test_spot_rules_by_hand():
    # Tuesday 20 Oct 2026: T+2 is Thursday; USDCAD is T+1
    assert spot_date("EURUSD", D("2026-10-20")) == D("2026-10-22")
    assert spot_date("USDCAD", D("2026-10-20")) == D("2026-10-21")
    # Friday before the US Thanksgiving week: no effect; Wednesday 25 Nov 2026 -> T+1 is Thanksgiving (USD
    # holiday), which does not count against EURUSD spot, so spot is Friday 27 Nov
    assert spot_date("EURUSD", D("2026-11-25")) == D("2026-11-27")
    # ... but for a cross a spot date on a USD holiday rolls: EURGBP traded Tue 24 Nov 2026 would settle on
    # Thanksgiving Thursday 26 Nov, so it rolls to Friday 27 Nov
    assert spot_date("EURGBP", D("2026-11-24")) == D("2026-11-27")
    # Good Friday 2026 (3 Apr) and Easter Monday are TARGET holidays: EURUSD on Thu 2 Apr -> Tue 7 Apr + 1
    assert spot_date("EURUSD", D("2026-04-02")) == D("2026-04-08")
    # USDJPY on Fri 2 Jan 2026 (Tokyo bank holiday): T+1 must be a Tokyo day
    assert spot_date("USDJPY", D("2025-12-31")) == D("2026-01-06")


def test_month_tenors_use_the_end_of_month_rule():
    # spot on Fri 27 Feb 2026, the last business day of February -> the 1M delivery is the last business day of
    # March (Tue 31 Mar), and the expiry is two business days before
    expiry, delivery = expiry_and_delivery("EURUSD", D("2026-02-25"), "1M")
    assert spot_date("EURUSD", D("2026-02-25")) == D("2026-02-27")
    assert delivery == D("2026-03-31") and expiry == D("2026-03-27")
    # 1M from spot 22 Oct 2026 delivers 23 Nov (22 Nov is a Sunday), expiring Thu 19 Nov
    assert expiry_and_delivery("EURUSD", D("2026-10-20"), "1M") == (D("2026-11-19"), D("2026-11-23"))
    # ON and weeks: the next business day / n weeks later, delivered at their spot dates
    assert expiry_and_delivery("EURUSD", D("2026-10-23"), "ON") == (D("2026-10-26"), D("2026-10-28"))
    assert expiry_and_delivery("USDCAD", D("2026-10-20"), "1W") == (D("2026-10-27"), D("2026-10-28"))


def test_fx_dates_and_year_fractions():
    dates = fx_dates("EURUSD", D("2026-10-20"), tenor="3M")
    assert dates.tau_e == pytest.approx((dates.expiry - D("2026-10-20")).days / 365.0)
    assert dates.tau_d == pytest.approx((dates.delivery - dates.spot).days / 365.0)
    custom = fx_dates("EURUSD", D("2026-10-20"), expiry=D("2026-12-15"))
    assert custom.delivery == spot_date("EURUSD", D("2026-12-15")) and custom.tenor is None
    today = fx_dates("EURUSD", D("2026-10-20"), expiry=D("2026-10-20"))
    assert today.tau_e == 0.0 and today.delivery == today.spot and today.tau_d == 0.0  # expires today, settles spot
    with pytest.raises(ValueError, match="before the trade date"):
        fx_dates("EURUSD", D("2026-10-20"), expiry=D("2026-10-19"))
    with pytest.raises(ValueError, match="either a tenor or an expiry"):
        fx_dates("EURUSD", D("2026-10-20"))
    with pytest.raises(ValueError, match="not supported"):
        expiry_and_delivery("EURUSD", D("2026-10-20"), "TN")


def test_pairs_have_market_conventions():
    assert set(PAIRS) >= {
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDJPY",
        "USDCHF",
        "USDCAD",
        "EURGBP",
        "EURJPY",
        "EURCHF",
    }
    eurusd, usdjpy, eurgbp, usdcad = (get_pair(c) for c in ("EURUSD", "usd/jpy", "EURGBP", "USDCAD"))
    assert (eurusd.base, eurusd.quote, eurusd.pip, eurusd.decimals) == ("EUR", "USD", 0.0001, 5)
    assert (usdjpy.pip, usdjpy.decimals, usdjpy.fmt(151.2345)) == (0.01, 3, "151.234")
    assert eurusd.fmt(1.0835) == "1.08350"
    assert not eurusd.premium_adjusted and usdjpy.premium_adjusted and eurgbp.premium_adjusted
    assert usdcad.spot_lag == 1 and eurusd.spot_lag == 2
    assert get_pair("NZDUSD").calendar.name == "AUCKLAND+WELLINGTON+USFED"
    assert eurgbp.settlement_calendar.name == "TARGET+LONDON+USFED"
    assert set(CURRENCY_CALENDARS) == {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}
    with pytest.raises(ValueError, match="unknown currency pair"):
        get_pair("EURSEK")


def test_market_delta_type_switches_at_one_year_and_with_premium_currency():
    trade = D("2026-10-20")
    assert market_delta_type("EURUSD", fx_dates("EURUSD", trade, tenor="1Y")) == "spot"
    assert market_delta_type("EURUSD", fx_dates("EURUSD", trade, tenor="18M")) == "forward"
    assert market_delta_type("USDJPY", fx_dates("USDJPY", trade, tenor="3M")) == "spot_pa"
    assert market_delta_type("USDJPY", fx_dates("USDJPY", trade, tenor="2Y")) == "forward_pa"
    # paying a EURUSD premium in EUR makes the hedge delta premium-adjusted
    assert market_delta_type("EURUSD", fx_dates("EURUSD", trade, tenor="3M"), premium_ccy="EUR") == "spot_pa"


def test_forwards_and_points():
    F = forward_from_rates(1.08, 0.043, 0.02, 0.25)
    assert F == pytest.approx(1.08 * 2.718281828459045 ** ((0.043 - 0.02) * 0.25))
    pts = forward_points(1.08, F, 0.0001)
    assert pts == pytest.approx(62.2, abs=0.1)
    assert forward_from_points(1.08, pts, 0.0001) == pytest.approx(F)
    assert implied_foreign_rate(1.08, F, 0.043, 0.25) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="forward of"):
        forward_from_points(1.08, -20000, 0.0001)
    with pytest.raises(ValueError, match="no rate time"):
        implied_foreign_rate(1.08, 1.09, 0.04, 0.0)


def _ql_fx():
    ql = pytest.importorskip("QuantLib")
    import sys
    from pathlib import Path

    tools = str(Path(__file__).resolve().parents[1] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from golden import fx as gfx
    from golden.common import pyd, qd

    return ql, gfx, pyd, qd


def test_random_dates_match_live_quantlib():
    """Random trade dates 2028-2070 (outside the fixture) and every tenor,
    against the QuantLib-calendar reference (skipped without QuantLib)."""

    ql, gfx, pyd, qd = _ql_fx()
    rng = np.random.default_rng(20261020)
    for _ in range(150):
        code = str(rng.choice(sorted(gfx.PAIRS)))
        trade = dt.date(2028, 1, 1) + dt.timedelta(days=int(rng.integers(0, 15000)))
        base, quote, lag = gfx.PAIRS[code]
        joint = ql.JointCalendar(gfx.CCY[base], gfx.CCY[quote])
        assert get_pair(code).calendar.add_business_days(trade, lag) == pyd(joint.advance(qd(trade), lag, ql.Days))
        assert spot_date(code, trade) == pyd(gfx.ql_spot(code, qd(trade))), (code, trade)
        tenor = str(rng.choice(OPTION_TENORS))
        expiry, delivery = gfx.ql_tenor_dates(code, qd(trade), tenor)
        assert expiry_and_delivery(code, trade, tenor) == (pyd(expiry), pyd(delivery)), (code, trade, tenor)
