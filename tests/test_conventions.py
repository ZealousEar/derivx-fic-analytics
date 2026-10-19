import datetime as dt

import pytest

from derivx.rates.conventions import Calendar, easter_sunday, make_schedule, parse_tenor, year_fraction

D = dt.date.fromisoformat


def test_calendars_match_quantlib_on_every_day(golden):
    data = golden("conventions")
    start, end = map(D, data["holiday_range"])
    for name, expected in data["holidays"].items():
        assert Calendar(name).holidays(start, end) == [D(x) for x in expected], name


def test_day_counts_match_quantlib(golden):
    for d1, d2, name, expected in golden("conventions")["day_counts"]:
        assert year_fraction(D(d1), D(d2), name) == pytest.approx(expected, abs=1e-13), (d1, d2, name)


def test_schedules_match_quantlib(golden):
    for case in golden("conventions")["schedules"]:
        dates = make_schedule(
            D(case["effective"]),
            D(case["termination"]),
            case["months"],
            case["calendar"],
            case["bdc"],
            case["termination_bdc"],
            backward=case["backward"],
            eom=case["eom"],
        )
        assert [d.isoformat() for d in dates] == case["dates"], case


def test_advance_matches_quantlib(golden):
    for cal, day, tenor, bdc, eom, expected in golden("conventions")["advances"]:
        assert Calendar(cal).advance(D(day), tenor, bdc, eom) == D(expected), (cal, day, tenor, bdc, eom)


@pytest.mark.parametrize(
    "text, n, unit, anchor",
    [
        ("ON", 1, "D", "today"),
        ("tn", 1, "D", "tomorrow"),
        ("S/N", 1, "D", "spot"),
        ("2D", 2, "D", "spot"),
        ("1W", 1, "W", "spot"),
        ("3m", 3, "M", "spot"),
        ("18M", 18, "M", "spot"),
        ("10y", 10, "Y", "spot"),
        ("1.5Y", 18, "M", "spot"),
        ("1 Mo", 1, "M", "spot"),
        ("30 Yr", 30, "Y", "spot"),
        ("1.5 Month", 6, "W", "spot"),
    ],
)
def test_parse_tenor_market_labels(text, n, unit, anchor):
    tenor = parse_tenor(text)
    assert (tenor.n, tenor.unit, tenor.anchor) == (n, unit, anchor)


@pytest.mark.parametrize("text", ["", "abc", "0M", "-1Y", "1.3M", "5Q"])
def test_parse_tenor_rejects_bad_labels_with_a_message(text):
    with pytest.raises(ValueError, match="enor"):
        parse_tenor(text)


def test_easter_and_us_special_closings():
    assert easter_sunday(2026) == dt.date(2026, 4, 5)
    gov, sofr = Calendar("USGOV"), Calendar("USD")
    assert not gov.is_business_day(dt.date(2012, 10, 30))  # Hurricane Sandy
    assert gov.is_business_day(dt.date(2026, 4, 3))  # Good Friday on payrolls day: early close only
    assert not sofr.is_business_day(dt.date(2026, 4, 3))  # SOFR is not published


def test_joint_calendar_and_icma_needs_reference_period():
    joint = Calendar("USD+LONDON")
    assert not joint.is_business_day(dt.date(2026, 8, 31))  # UK summer bank holiday
    assert not joint.is_business_day(dt.date(2026, 9, 7))  # US Labor Day
    with pytest.raises(ValueError):
        Calendar("MARS")
    with pytest.raises(ValueError, match="ICMA"):
        year_fraction(dt.date(2026, 1, 1), dt.date(2026, 7, 1), "ACT/ACT ICMA")
    assert year_fraction(
        dt.date(2026, 2, 15), dt.date(2026, 8, 15), "ACT/ACT ICMA", dt.date(2026, 2, 15), dt.date(2026, 8, 15), 2
    ) == pytest.approx(0.5)
