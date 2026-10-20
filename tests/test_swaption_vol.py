"""Swaption vol cube: quote parsing, SABR per expiry x tenor against
QuantLib's SABR formulas, variance-time ATM interpolation, arbitrage
diagnostics and vega buckets."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from derivx.rates.examples import example_market
from derivx.rates.swaption_vol import (
    GRID_OFFSETS_BP,
    GridPoint,
    VolGrid,
    butterfly_check,
    calendar_check,
    calibrate_cube,
    calibrate_point,
    grid_points,
    parse_vol_grid,
    smile_vol,
    swaption_on_cube,
    variance_time_atm,
    vega_buckets,
)
from derivx.rates.swaptions import Swaption
from derivx.rates.vol_examples import illustrative_swaption_grid, illustrative_swaption_table
from derivx.vol.sabr import sabr_vol

NOTIONAL = 10_000_000.0


@pytest.fixture(scope="module")
def market():
    return example_market()


@pytest.fixture(scope="module")
def cube(market):
    grid = illustrative_swaption_grid("USD")
    return calibrate_cube(grid, grid_points(grid, market))


# --------------------------------------------------------------------------
# SABR against QuantLib
# --------------------------------------------------------------------------


def test_rates_sabr_smiles_match_quantlib(golden):
    """Normal (beta 0) and shifted-lognormal (beta 0.5) SABR on rates
    parameters, a negative forward included, against
    ql.shiftedSabrVolatility in both vol types."""

    for case in golden("swaptions")["sabr"]:
        F, T, K = case["F"], case["T"], np.array(case["strikes"])
        params = {k: case[k] for k in ("alpha", "beta", "rho", "nu", "shift")}
        normal = sabr_vol(F, K, T, **params, vol_type="normal")
        np.testing.assert_allclose(normal, case["normal"], rtol=1e-12)
        np.testing.assert_allclose(smile_vol(F, K, T, params, "normal"), case["normal"], rtol=1e-12)
        black = smile_vol(F, K, T, params, "lognormal", "lognormal")
        np.testing.assert_allclose(black, case["lognormal"], rtol=1e-12)


@pytest.mark.parametrize("which", [0, 1])
def test_cube_calibration_recovers_quantlib_generated_smiles(golden, which):
    """Quotes QuantLib generates from known SABR parameters at the grid
    offsets (normal vols for a normal fit, shifted Black vols for a
    lognormal fit) are refitted to the same parameters and repriced."""

    case = golden("swaptions")["recovery"][which]
    fit = case["fit"]
    rows = tuple(("2Y", "5Y", o, v) for o, v in zip(case["offsets"], case["vols"]))
    grid = VolGrid(rows, fit, case["shift"] if fit == "lognormal" else 0.0, "QuantLib SABR")
    point = GridPoint("2Y", "5Y", case["T"], 5.0, case["F"], 4.5)
    result = calibrate_point(grid, point, fit, case["beta"], case["shift"])
    assert result.status == "fitted"
    assert result.rho == pytest.approx(case["rho"], abs=1e-6)
    assert result.nu == pytest.approx(case["nu"], rel=1e-6)
    assert result.alpha == pytest.approx(case["alpha"], rel=1e-7)
    assert result.max_abs_bp < 1e-4  # bp of normal vol (the optimiser stops at ~1e-9 in vol)


# --------------------------------------------------------------------------
# Quotes
# --------------------------------------------------------------------------


def test_illustrative_grid_is_labelled_and_round_trips_through_the_wide_table():
    grid = illustrative_swaption_grid("EUR")
    assert grid.source.startswith("ILLUSTRATIVE")
    assert len(grid.rows) == 9 * 6 * 9
    table = illustrative_swaption_table("EUR")
    assert list(table.columns[:2]) == ["expiry", "tenor"] and "ATM" in table.columns
    parsed, issues = parse_vol_grid(table, "normal", "bp", source=grid.source)
    assert not issues
    a, b = parsed.frame(), grid.frame()
    assert a[["expiry", "tenor", "offset_bp"]].equals(b[["expiry", "tenor", "offset_bp"]])
    np.testing.assert_allclose(a["vol"], b["vol"], rtol=1e-14)
    with pytest.raises(ValueError, match="no illustrative vols"):
        illustrative_swaption_grid("JPY")


def test_parse_long_format_and_units():
    long = pd.DataFrame(
        {
            "Expiry": ["1Y"] * 3 + ["2Y"] * 3,
            "Tenor": ["5Y"] * 6,
            "offset_bp": [-50, 0, 50] * 2,
            "vol": [0.0105, 0.0100, 0.0102, 0.0098, 0.0095, 0.0097],
        }
    )
    grid, issues = parse_vol_grid(long, "normal", "decimal")
    assert not issues and grid.expiries == ("1Y", "2Y") and grid.tenors == ("5Y",)
    offsets, vols = grid.smile("2Y", "5Y")
    assert list(offsets) == [-50, 0, 50] and vols[1] == pytest.approx(0.0095)
    lognormal = long.assign(vol=[25.0, 24.0, 24.5, 23.0, 22.0, 22.5])
    grid, issues = parse_vol_grid(lognormal, "lognormal", "percent", shift=0.02)
    assert not issues and grid.vol_type == "lognormal" and grid.shift == 0.02
    assert grid.smile("1Y", "5Y")[1][1] == pytest.approx(0.24)


@pytest.mark.parametrize(
    "table,units,match",
    [
        (pd.DataFrame({"expiry": ["1Y"], "tenor": ["5Y"], "ATM": [0.0095]}), "bp", "outside 1-1000bp"),
        (pd.DataFrame({"expiry": ["1Y"], "tenor": ["5Y"], "ATM": [95.0]}), "percent", "normal vols are in"),
        (pd.DataFrame({"expiry": ["1Y"], "tenor": ["5Y"], "+25": [95.0]}), "bp", "no ATM quote"),
        (pd.DataFrame({"expiry": ["1Y", "2Y"], "tenor": ["5Y", "10Y"], "ATM": [95.0, 90.0]}), "bp", "fill the expiry"),
        (pd.DataFrame({"expiry": ["1Q"], "tenor": ["5Y"], "ATM": [95.0]}), "bp", "row 1"),
        (pd.DataFrame({"expiry": ["2D"], "tenor": ["5Y"], "ATM": [95.0]}), "bp", "weeks, months or years"),
        (pd.DataFrame({"expiry": ["1Y"], "tenor": ["5Y"], "ATM": ["n/a"]}), "bp", "not a number"),
        (pd.DataFrame({"expiry": ["1Y"], "tenor": ["5Y"], "wide": [95.0]}), "bp", "not a strike offset"),
        (pd.DataFrame({"expiry": ["1Y"], "tenor": ["5Y"], "+2000": [95.0]}), "bp", "outside"),
        (pd.DataFrame({"tenor": ["5Y"], "ATM": [95.0]}), "bp", "'expiry' and 'tenor'"),
        (pd.DataFrame({"expiry": ["1Y", "1Y"], "tenor": ["5Y", "5Y"], "offset": [0, 0], "vol": [95.0, 96.0]}), "bp",
         "second quote"),
    ],
)  # fmt: skip
def test_parse_rejects_bad_tables_with_reasons(table, units, match):
    grid, issues = parse_vol_grid(table, "normal", units)
    assert grid is None
    assert any(match in i.message for i in issues if i.level == "error"), [i.message for i in issues]


def test_one_point_cannot_be_quoted_under_two_names(market):
    """12M and 1Y (or 24M and 2Y) are the same expiry or tenor: a second
    label used to add a duplicate cube node, so the ATM vol jumped from one
    quote to the other at 1Y and the first was silently shadowed."""

    rows = [{"expiry": e, "tenor": t, "-50": v + 2.0, "ATM": v, "+50": v + 1.0}
            for e, v in (("6M", 90.0), ("12M", 80.0), ("1Y", 100.0), ("2Y", 95.0)) for t in ("5Y", "10Y")]  # fmt: skip
    grid, issues = parse_vol_grid(pd.DataFrame(rows))
    assert grid is None
    assert any("second quote for 1Yx5Y ATM (12Mx5Y and 1Yx5Y are the same point)" in i.message for i in issues)
    tenors = pd.DataFrame([{"expiry": "1Y", "tenor": t, "ATM": 90.0} for t in ("12M", "1Y", "5Y")])
    assert parse_vol_grid(tenors)[0] is None
    # Labels are canonical, so 12M / 24M quotes become the 1Y / 2Y points the SWAPTION tab selects
    renamed = pd.DataFrame([{"expiry": e, "tenor": t, "-50": 92.0, "ATM": 90.0, "+50": 91.0}
                            for e in ("12M", "24M") for t in ("60M", "10Y")])  # fmt: skip
    grid, issues = parse_vol_grid(renamed)
    assert not issues and grid.expiries == ("1Y", "2Y") and grid.tenors == ("5Y", "10Y")
    with pytest.raises(ValueError, match="12M and 1Y are the same option expiry"):
        VolGrid((("12M", "5Y", 0.0, 0.009), ("1Y", "5Y", 0.0, 0.009)))
    with pytest.raises(ValueError, match="same swap tenor"):
        VolGrid((("1Y", "2Y", 0.0, 0.009), ("1Y", "24M", 0.0, 0.009)))


def test_parse_reads_atm_labels_in_long_tables_and_names_the_table_row():
    long = pd.DataFrame({"expiry": ["1Y"] * 3, "tenor": ["5Y"] * 3, "offset_bp": ["-25bp", "ATM", "ATM+25"],
                         "vol": [90.0, 88.0, 89.0]})  # fmt: skip
    grid, issues = parse_vol_grid(long)
    assert not issues and [r[2] for r in grid.rows] == [-25.0, 0.0, 25.0]
    minus = pd.DataFrame([{"expiry": "1Y", "tenor": "5Y", "\u221225": 90.0, "ATM": 88.0, "+25": 89.0}])
    assert parse_vol_grid(minus)[0].smile("1Y", "5Y")[0].tolist() == [-25.0, 0.0, 25.0]
    # Wide tables: errors cite the row of the user's table, once per row (not once per offset column)
    wide = pd.DataFrame([{"expiry": "1Y", "tenor": "5Y", "-25": 90.0, "ATM": 88.0, "+25": 89.0},
                         {"expiry": "2Y", "tenor": "5Y", "-25": 90.0, "ATM": "abc", "+25": 89.0},
                         {"expiry": "3Q", "tenor": "5Y", "-25": 90.0, "ATM": 88.0, "+25": 89.0}])  # fmt: skip
    grid, issues = parse_vol_grid(wide)
    messages = [i.message for i in issues]
    assert grid is None and len(messages) == 2, messages
    assert "row 2 (2Yx5Y ATM): vol 'abc' is not a number" in messages
    assert sum(m.startswith("row 3: ") for m in messages) == 1
    nan = pd.DataFrame([{"expiry": "1Y", "tenor": "5Y", "ATM": "nan"}])
    assert "vol 'nan' is not a finite number" in parse_vol_grid(nan)[1][0].message


def test_a_point_with_too_few_strikes_gets_a_flat_smile(market):
    table = pd.DataFrame({"expiry": ["1Y", "2Y"], "tenor": ["5Y", "5Y"], "-50": [None, 99.0], "ATM": [95.0, 92.0],
                          "+50": [None, 94.0], "+100": [None, 97.0]})  # fmt: skip
    grid, issues = parse_vol_grid(table, "normal", "bp")
    assert grid is not None and any(i.level == "warning" and "flat" in i.message for i in issues)
    cube = calibrate_cube(grid, grid_points(grid, market))
    flat = cube.point("1Y", "5Y")
    assert flat.status == "flat" and flat.nu == 0.0 and flat.rho == 0.0
    assert cube.point("2Y", "5Y").status == "fitted"


# --------------------------------------------------------------------------
# Cube
# --------------------------------------------------------------------------


def test_illustrative_cube_fits_and_reprices_its_atm_matrix(cube):
    table = cube.fit_table()
    assert (table["status"] == "fitted").all()
    assert table["rmse_bp"].max() < 0.1  # quotes are rounded to 0.1bp
    for f in cube.fits:
        p = f.point
        atm = cube.normal_vol(p.expiry_years, p.tenor_years, p.forward, p.forward)
        assert atm == pytest.approx(f.atm_normal, rel=1e-12)
        np.testing.assert_allclose(f.vol(f.strikes), f.model_normal, rtol=1e-13)
    assert calendar_check(cube) == []
    for f in cube.fits:
        assert butterfly_check(f)["negative_offsets_bp"] == []


def test_lognormal_fit_reports_residuals_in_normal_bp(market):
    grid = illustrative_swaption_grid("USD")
    cube = calibrate_cube(grid, grid_points(grid, market), fit="lognormal")
    assert cube.beta == 0.5 and cube.fit == "lognormal"
    f = cube.point("1Y", "10Y")
    assert f.status == "fitted" and f.max_abs_bp < 2.0  # beta 0.5 against quotes made with beta 0: a model gap
    atm = cube.normal_vol(f.point.expiry_years, 10.0, f.point.forward, f.point.forward)
    assert atm == pytest.approx(f.atm_normal, rel=1e-9)  # ATM still exact, via price matching


def test_atm_interpolation_is_linear_in_total_variance():
    expiries, tenors = np.array([1.0, 2.0, 5.0]), np.array([2.0, 10.0])
    atm = np.array([[0.0100, 0.0090], [0.0110, 0.0095], [0.0100, 0.0088]])
    T = 1.5
    expected = np.sqrt(((0.0100**2 * 1.0) + 0.5 * (0.0110**2 * 2.0 - 0.0100**2 * 1.0)) / T)
    assert variance_time_atm(expiries, tenors, atm, T, 2.0) == pytest.approx(expected, rel=1e-14)
    assert variance_time_atm(expiries, tenors, atm, 0.25, 2.0) == pytest.approx(0.0100)  # flat vol before
    assert variance_time_atm(expiries, tenors, atm, 9.0, 10.0) == pytest.approx(0.0088)  # and after
    mid = variance_time_atm(expiries, tenors, atm, 2.0, 6.0)
    assert mid == pytest.approx(np.sqrt(0.5 * 0.0110**2 + 0.5 * 0.0095**2), rel=1e-14)  # linear in tenor
    assert variance_time_atm(expiries, tenors, atm, 2.0, 30.0) == pytest.approx(0.0095)
    with pytest.raises(ValueError, match="positive"):
        variance_time_atm(expiries, tenors, atm, 0.0, 2.0)


def test_bumping_one_atm_point_moves_only_its_neighbourhood(cube):
    i, j = 3, 3  # 1Y x 10Y
    f = cube.point("1Y", "10Y")
    bumped = cube.bumped(i, j, 1.0)
    p = f.point
    assert bumped.atm_vol(p.expiry_years, 10.0) - cube.atm_vol(p.expiry_years, 10.0) == pytest.approx(1e-4, rel=1e-9)
    other = cube.point("5Y", "2Y").point
    assert bumped.atm_vol(other.expiry_years, 2.0) == cube.atm_vol(other.expiry_years, 2.0)
    assert np.allclose(cube.parallel(1.0).atm_normal - cube.atm_normal, 1e-4)


def test_vega_buckets_sit_on_the_neighbouring_points_and_sum_to_the_vega(cube, market):
    trade = swaption_on_cube(cube, market, NOTIONAL, "4Y", "7Y", offset_bp=25.0)
    buckets = vega_buckets([trade], cube, market)
    nonzero = {(cube.expiries[i], cube.tenors[j]) for i, j in zip(*np.nonzero(buckets))}
    assert nonzero == {("3Y", "5Y"), ("3Y", "10Y"), ("5Y", "5Y"), ("5Y", "10Y")}
    assert buckets.sum() == pytest.approx(trade.greeks(market)["vega_normal_bp"], rel=5e-3)
    on_node = swaption_on_cube(cube, market, NOTIONAL, "2Y", "10Y", offset_bp=0.0)
    node = vega_buckets([on_node], cube, market)
    assert node[4, 3] == pytest.approx(on_node.greeks(market)["vega"], rel=1e-3)
    assert np.count_nonzero(node) == 1
    sold = replace(on_node, notional=-NOTIONAL)
    np.testing.assert_allclose(vega_buckets([sold, on_node], cube, market), 0.0, atol=1e-9)


def test_trades_read_their_vol_from_the_cube_in_any_model(cube, market):
    normal = swaption_on_cube(cube, market, NOTIONAL, "1Y", "5Y", offset_bp=-50.0)
    black = swaption_on_cube(cube, market, NOTIONAL, "1Y", "5Y", offset_bp=-50.0, model="shifted_black", shift=0.02)
    assert normal.model == "bachelier" and black.model == "shifted_black"
    assert black.pv(market) == pytest.approx(normal.pv(market), rel=1e-9)
    f = cube.point("1Y", "5Y")
    assert normal.vol == pytest.approx(float(f.vol(normal.strike)), rel=1e-12)
    with pytest.raises(ValueError, match="below -shift"):
        swaption_on_cube(cube, market, NOTIONAL, "1Y", "5Y", strike=-0.05)


def test_butterfly_and_calendar_diagnostics_flag_bad_inputs(cube):
    f = cube.point("10Y", "30Y")
    bad = replace(f, nu=1.0, rho=0.0, alpha=f.alpha * 0.3)  # a vol of vol far too high for 10Y: wings bend down
    check = butterfly_check(bad)
    assert check["min_density"] < 0 and check["negative_offsets_bp"]
    bumps = np.zeros((len(cube.expiries), len(cube.tenors)))
    bumps[cube.expiries.index("5Y")] = -0.003  # 30bp lower at 5Y: less total variance than at 3Y
    flagged = calendar_check(replace(cube, atm_bumps=bumps))
    assert {r["from"] for r in flagged} == {"3Y"} and {r["to"] for r in flagged} == {"5Y"}
    assert len(flagged) == len(cube.tenors) and all(r["variance_drop_pct"] > 0 for r in flagged)


def test_calibration_inputs_are_validated(market):
    grid = illustrative_swaption_grid("USD")
    points = grid_points(grid, market)
    with pytest.raises(ValueError, match="fit must be"):
        calibrate_cube(grid, points, fit="sabr")
    with pytest.raises(ValueError, match="grid_points"):
        calibrate_cube(grid, points[:-1])
    with pytest.raises(ValueError, match="shift"):
        calibrate_cube(grid, points, shift=-0.01)
    low = replace(points[0], forward=-0.05)
    with pytest.raises(ValueError, match="raise the SABR shift"):
        calibrate_point(grid, low)


def test_grid_points_use_the_swaption_engine_forwards(market):
    grid = illustrative_swaption_grid("USD")
    point = next(p for p in grid_points(grid, market) if (p.expiry, p.tenor) == ("5Y", "10Y"))
    probe = Swaption(1.0, "5Y", "10Y", 0.0)
    assert point.forward == probe.atm_strike(market)
    assert point.expiry_years == probe.inputs(market).expiry_years
    assert point.tenor_years == 10.0 and GRID_OFFSETS_BP[4] == 0.0
