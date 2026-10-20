"""Pure calculations behind the pricer page: input checks, formatting,
scenario grid, risk-surface domain, smile grids and the model check."""

from __future__ import annotations

import math

import numpy as np
import pytest

from derivx.core.bs import bs_greeks, bs_price, desk_greeks
from derivx.ui import pricer_calc as calc
from derivx.vol.sabr import hagan_sabr_iv
from derivx.vol.svi import svi_iv

EQ = calc.Inputs(S=100.0, K=100.0, T=1.0, r=0.05, q=0.01, sigma=0.2, call=True)
FX = calc.Inputs(S=1.08, K=1.08, T=0.25, r=0.04, q=0.025, sigma=0.08, call=True)


def test_input_errors_name_the_field_and_the_fix():
    assert calc.input_errors(100, 100, 1, 0.05, 0.0, 0.2) == []
    assert "SPOT.S must be greater than 0" in calc.input_errors(0, 100, 1, 0.05, 0.0, 0.2)[0]
    assert "STRIKE.K must be greater than 0" in calc.input_errors(100, -5, 1, 0.05, 0.0, 0.2)[0]
    assert "empty" in calc.input_errors(None, 100, 1, 0.05, 0.0, 0.2)[0]
    assert "VOL.σ must be >= 0" in calc.input_errors(100, 100, 1, 0.05, 0.0, -0.1)[0]
    assert any("EXPIRED" in w for w in calc.input_warnings(0.0, 0.05, 0.0, 0.2))
    assert any("Less than one day" in w for w in calc.input_warnings(0.5 / 365, 0.05, 0.0, 0.2))
    assert any("VOL.σ = 0" in w for w in calc.input_warnings(1.0, 0.05, 0.0, 0.0))
    assert any("check the units" in w for w in calc.input_warnings(1.0, 0.3, 0.0, 0.2))
    assert calc.input_warnings(1.0, 0.05, 0.0, 0.2) == []


@pytest.mark.parametrize(
    "x, text",
    [
        (0.0, "0"),
        (-0.0, "0"),
        (0.6368306, "0.6368"),
        (-0.017573, "-0.01757"),
        (12345.678, "12,346"),
        (1.2344e-9, "1.234e-09"),
        (float("nan"), "n/a"),
    ],
)
def test_significant_figure_formatting_never_hides_small_values(x, text):
    """PP-13: 4 significant figures instead of a fixed 4 decimals."""

    assert calc.fmt_sig(x) == text


def test_compact_chart_labels_fit_a_phone_cell():
    cases = {0.0: "0", -536.2: "-540", 12.94: "13", -9.453: "-9.5", 7612.3: "7.6k", -50912.0: "-51k", 1.234e6: "1.2M"}
    for x, text in cases.items():
        assert calc.fmt_compact(x) == text
        assert len(text) <= 5
    assert calc.fmt_compact(1.2e-5) == "1.2e-05"


def test_fixed_formatting_and_tenor_text():
    assert calc.fmt_fixed(999904.87712, 4) == "999,904.8771"  # separators, one token
    assert calc.fmt_fixed(1.2e-9, 4) == "1.200e-09"
    assert calc.tenor_text(1.0) == "1.0000Y · 365D"
    assert calc.fmt_exact(1.083456, 5) == "1.083456" and calc.fmt_exact(1.08, 5) == "1.08000"
    assert calc.fmt_exact(5800.25, 2) == "5,800.25" and calc.fmt_exact(float("nan"), 2) == "n/a"
    assert calc.tenor_text(0.5 / 365) == "0.0014Y · 0.50D"


def test_scenario_grid_is_full_revaluation_for_the_position():
    grid = calc.scenario_grid(EQ, position=-200.0, max_spot_shock=0.10, max_vol_shock=0.05, horizon_days=7)
    j0, i0 = list(grid.spot_shocks).index(0.0), list(grid.vol_shocks).index(0.0)
    base = float(bs_price(100, 100, 1, 0.05, 0.2, q=0.01))
    later = float(bs_price(100, 100, 1 - 7 / 365, 0.05, 0.2, q=0.01))
    assert grid.pnl[i0, j0] == pytest.approx(-200 * (later - base))  # a short position earns the decay
    j, i = list(grid.spot_shocks).index(0.10), list(grid.vol_shocks).index(-0.05)
    assert grid.pnl[i, j] == pytest.approx(-200 * (float(bs_price(110, 100, 1 - 7 / 365, 0.05, 0.15, q=0.01)) - base))
    assert grid.table().shape == (len(calc.VOL_STEPS), len(calc.SHOCK_STEPS))
    assert list(grid.table().columns)[-1] == "+10%"


def test_scenario_grid_drops_negative_vols_and_validates_shocks():
    low = calc.Inputs(100.0, 100.0, 1.0, 0.05, 0.0, 0.02, True)
    grid = calc.scenario_grid(low, 1.0, max_vol_shock=0.05)
    assert grid.dropped_vol_shocks == (-0.05,) and np.all(low.sigma + grid.vol_shocks >= -1e-12)
    assert grid.pnl[list(np.round(grid.vol_shocks, 12)).index(-0.02)].min() < 0  # the zero-vol row is priced
    for kwargs in (dict(max_spot_shock=0.0), dict(max_spot_shock=0.95), dict(max_vol_shock=0.0), dict(horizon_days=-1)):
        with pytest.raises(ValueError):
            calc.scenario_grid(EQ, 1.0, **kwargs)


def test_spot_domain_is_in_standard_deviations_not_a_fixed_box():
    """PP-09: FX levels get an FX-sized window and short tenors are not
    90% flat intrinsic."""

    lo, hi = calc.spot_window(FX, 3.0)
    assert lo == pytest.approx(1.08 * math.exp(-0.12)) and hi == pytest.approx(1.08 * math.exp(0.12))
    week = calc.Inputs(100.0, 100.0, 7 / 365, 0.05, 0.0, 0.2, True)
    grid = calc.surface_grid(week, calc.SURFACE_AXES[0], "VALUE", n=25, n_sd=3.0)
    X = np.meshgrid(grid.x, grid.y)[0]
    time_value = grid.z - np.maximum(X - 100.0, 0.0)
    assert np.mean(time_value > 0.001 * 100.0) > 0.5
    # tiny sigma sqrt(T): +-5% floor; huge: capped at a factor exp(2.5)
    assert calc.spot_window(calc.Inputs(100, 100, 0.0, 0.05, 0.0, 0.2, True), 3.0)[1] == pytest.approx(
        100 * math.exp(0.05)
    )
    assert calc.spot_window(calc.Inputs(100, 100, 30.0, 0.05, 0.0, 2.0, True), 3.0)[1] == pytest.approx(
        100 * math.exp(2.5)
    )
    # strike far from spot stays inside
    lo, hi = calc.spot_window(calc.Inputs(150.0, 80.0, 1.0, 0.05, 0.0, 0.3, True), 3.0)
    assert lo < 80 and hi > 150


def test_surface_grids_honour_put_and_show_greeks_in_desk_units():
    put = calc.Inputs(100.0, 120.0, 2.0, 0.05, 0.0, 0.2, False)
    grid = calc.surface_grid(put, calc.SURFACE_AXES[0], "VALUE", n=11)
    X, Y = np.meshgrid(grid.x, grid.y)
    np.testing.assert_allclose(grid.z, bs_price(X, 120.0, 2.0, 0.05, Y, call=False))
    assert grid.point[2] == pytest.approx(16.5087, abs=1e-4)
    vega = calc.surface_grid(put, calc.SURFACE_AXES[1], "VEGA", n=11)
    X, Y = np.meshgrid(vega.x, vega.y)
    expected = desk_greeks(bs_greeks(X, 120.0, Y, 0.05, 0.2, call=False), X)["vega_1vol"]
    np.testing.assert_allclose(vega.z, expected)
    assert vega.y[0] == 0.0 and vega.y[-1] == 2.0 and np.all(vega.z[0] == 0.0)  # no vega at expiry
    with pytest.raises(ValueError, match="expired"):
        calc.surface_grid(calc.Inputs(100, 100, 0.0, 0.05, 0.0, 0.2, True), calc.SURFACE_AXES[1], "VALUE")


def test_smile_grid_is_in_forward_moneyness():
    """PP-03 / PP-10: strikes F exp(+-n sigma sqrt(T)), centred on the
    forward (not K +- 40 around the strike)."""

    F = FX.forward
    strikes = calc.smile_strikes(F, 0.08, 0.25, n_sd=3.0)
    assert strikes[0] == pytest.approx(F * math.exp(-0.12)) and strikes[-1] == pytest.approx(F * math.exp(0.12))
    assert np.median(strikes) == pytest.approx(F)
    index = calc.smile_strikes(5800.0, 0.18, 1.0, n_sd=3.0)
    assert index[0] < 0.6 * 5800 and index[-1] > 1.6 * 5800  # index wings, not +-0.8%
    with pytest.raises(ValueError):
        calc.smile_strikes(F, 0.08, 0.0)


def test_illustrative_svi_is_deterministic_and_arbitrage_checked():
    smile = calc.illustrative_svi(105.0, 1.0, 0.2, -0.6, 1.0)
    again = calc.illustrative_svi(105.0, 1.0, 0.2, -0.6, 1.0)
    np.testing.assert_array_equal(smile.vols, again.vols)
    raw = {k.replace("SVI ", ""): v for k, v in smile.params.items()}
    assert float(svi_iv(105.0, raw, F=105.0, T=1.0)) == pytest.approx(0.2, rel=1e-12)  # ATM-forward vol
    np.testing.assert_allclose(svi_iv(smile.strikes, raw, F=105.0, T=1.0), smile.vols, rtol=1e-12)
    assert smile.vols[0] > smile.vols[-1]  # negative skew
    assert smile.arbitrage_free and not smile.notes
    wild = calc.illustrative_svi(105.0, 1.0, 0.2, -0.9, 3.0)
    assert wild.notes and "Gatheral-Jacquier" in wild.notes[0]
    with pytest.raises(ValueError):
        calc.illustrative_svi(105.0, 1.0, 0.2, -1.0, 1.0)


def test_illustrative_sabr_reproduces_the_atm_vol():
    smile = calc.illustrative_sabr(FX.forward, 0.25, 0.08, -0.1, 0.6)
    p = smile.params
    atm = float(hagan_sabr_iv(FX.forward, FX.forward, 0.25, p["SABR α"], 1.0, p["SABR ρ"], p["SABR ν"]))
    assert atm == pytest.approx(0.08, rel=1e-10)
    assert smile.fixed == ("SABR β",) and p["SABR β"] == 1.0


def test_model_check_passes_and_refuses_the_kink():
    rng = np.random.default_rng(3)
    for _ in range(25):
        inp = calc.Inputs(
            float(rng.uniform(1, 500)),
            float(rng.uniform(1, 500)),
            float(rng.uniform(0.01, 5)),
            float(rng.uniform(-0.02, 0.08)),
            float(rng.uniform(0.0, 0.05)),
            float(rng.uniform(0.05, 0.9)),
            bool(rng.integers(0, 2)),
        )
        inp = calc.Inputs(
            inp.S, inp.S * float(np.exp(rng.uniform(-0.4, 0.4))), inp.T, inp.r, inp.q, inp.sigma, inp.call
        )
        table = calc.model_check(inp)
        assert bool(table["ok"].all()), table
    with pytest.raises(ValueError, match="kink"):
        calc.model_check(calc.Inputs(100, 100, 0.0, 0.05, 0.0, 0.2, True))
    lo, hi = calc.price_bounds(EQ)
    assert lo < float(bs_price(100, 100, 1, 0.05, 0.2, q=0.01)) < hi


@pytest.mark.parametrize(
    "inp",
    [
        calc.Inputs(100.0, 80.0, 10.0, -0.01, 0.03, 0.01, True),
        calc.Inputs(100.0, 50.0, 0.25, 0.05, 0.0, 0.2, True),
        calc.Inputs(100.0, 50.0, 1 / 365, 0.05, 0.0, 5.0, False),
        calc.Inputs(100.0, 100.0, 50.0, 0.0, 0.0, 0.0001, True),
        calc.Inputs(100.0, 97.0, 0.0001, 0.05, 0.0, 1.0, False),
        calc.Inputs(100.0, 50.0, 50.0, 0.5, -0.5, 0.0001, True),
        calc.Inputs(1.08, 1.2, 7 / 365, 0.04, 0.025, 0.08, True),
    ],
)
def test_model_check_has_no_false_alarms_at_valid_extreme_inputs(inp):
    """The page's model check showed '[!]' for these valid inputs (fixed
    finite-difference bumps: a vol step of sigma / 2, rounding noise above a
    1e-6 floor); a check that cries wolf is no check."""

    table = calc.model_check(inp)
    assert bool(table["ok"].all()), table
    assert {"FD error estimate", "|difference|"} <= set(table.columns)


def test_model_check_flags_a_wrong_greek():
    import dataclasses

    g = bs_greeks(EQ.S, EQ.K, EQ.T, EQ.r, EQ.sigma, q=EQ.q, call=EQ.call)
    assert bool(calc.model_check(EQ, g)["ok"].all())
    for name, factor in (("vega", 1 + 1e-5), ("gamma", 2.0), ("charm", 1 + 1e-4), ("theta", -1.0)):
        wrong = dataclasses.replace(g, **{name: getattr(g, name) * factor})
        table = calc.model_check(EQ, wrong).set_index("greek")
        assert not table.loc[name, "ok"] and table.drop(index=name)["ok"].all(), name


def test_shock_labels_are_distinct_and_exact_for_every_allowed_shock_size():
    """MAX SPOT SHOCK 0.5% used to print -0.05% and -0.10% both as '-0.1%':
    duplicate table columns, and Arrow refused to render the table."""

    for max_shock in np.arange(0.5, 90.01, 0.5):
        for steps in (calc.SHOCK_STEPS, calc.VOL_STEPS):
            values = max_shock * np.asarray(steps)
            labels = calc.shock_labels(values)
            assert len(set(labels)) == len(labels), (max_shock, labels)
            for text, v in zip(labels, values):
                assert float(text) == pytest.approx(v, rel=5e-3, abs=0.0), (max_shock, text, v)
    grid = calc.scenario_grid(EQ, 1.0, max_spot_shock=0.005, max_vol_shock=0.007)
    table = grid.table()
    assert table.columns.is_unique and table.index.is_unique
    assert list(table.columns) == grid.spot_labels and grid.spot_labels[3] == "-0.05%"
    assert grid.vol_labels == ["-0.70", "-0.28", "-0.14", "+0", "+0.14", "+0.28", "+0.70"]
    import pyarrow as pa

    pa.Table.from_pandas(table)  # what st.dataframe does; raised 'Duplicate column names found'


def _breeden_litzenberger(F, T, strikes, alpha, beta, rho, nu, h=1e-4):
    """Density d2C/dK2 of undiscounted Black calls priced at the SABR vols
    (an independent route to the sign of Durrleman's g)."""

    from derivx.vol.implied import black_price

    def call(K):
        return black_price(F, K, T, hagan_sabr_iv(F, K, T, alpha, beta, rho, nu))

    dK = h * strikes
    return (call(strikes + dK) - 2.0 * call(strikes) + call(strikes - dK)) / dK**2


@pytest.mark.parametrize(
    "T, vol, rho, nu, arbitrage_free",
    [(0.25, 0.08, -0.1, 0.6, True), (1.0, 0.1, -0.3, 1.0, True), (5.0, 0.5, -0.5, 2.0, False)],
)
def test_illustrative_sabr_reports_a_negative_density(T, vol, rho, nu, arbitrage_free):
    """The FX SABR smile used to be drawn with no arbitrage check at all
    (the SVI one had one); Hagan's expansion breaks down for large nu^2 T."""

    F = 1.08
    smile = calc.illustrative_sabr(F, T, vol, rho, nu)
    assert smile.arbitrage_free is arbitrage_free
    assert bool(smile.notes) is (not arbitrage_free)
    if not arbitrage_free:
        assert "Butterfly arbitrage" in smile.notes[0] and "ν²T" in smile.notes[0]
    alpha = smile.params["SABR α"]
    k, g = calc.sabr_density_factor(F, T, smile.strikes, alpha, 1.0, rho, nu)
    density = _breeden_litzenberger(F, T, smile.strikes, alpha, 1.0, rho, nu)
    w = T * smile.vols**2
    d2 = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
    expected = np.exp(-0.5 * d2**2) / np.sqrt(2 * np.pi) * g / (smile.strikes * np.sqrt(w))
    np.testing.assert_allclose(density, expected, rtol=2e-3, atol=1e-6 * np.max(np.abs(expected)))


def test_durrleman_g_matches_the_svi_butterfly_density():
    from derivx.vol.svi import butterfly_density, svi_derivatives

    params = {"a": 0.01, "b": 0.12, "rho": -0.5, "m": 0.05, "sigma": 0.2}
    k = np.linspace(-1.5, 1.5, 61)
    np.testing.assert_allclose(calc.durrleman_g(k, *svi_derivatives(k, params)), butterfly_density(k, params))


def test_smile_vol_at_the_strike_is_the_model_not_an_interpolation():
    """The 'at K the smile vol is ...' caption interpolated the 121-point
    grid: 9.196% instead of 9.189% here, a 0.4% premium error."""

    F = 100.0 * math.exp(0.05)
    smile = calc.illustrative_svi(F, 1.0, 0.2, -0.9, 2.0)
    raw = {k.replace("SVI ", ""): v for k, v in smile.params.items()}
    assert smile.vol_at(120.0) == pytest.approx(float(svi_iv(120.0, raw, F=F, T=1.0)), rel=1e-12)
    sabr = calc.illustrative_sabr(FX.forward, 0.25, 0.08, -0.1, 0.6)
    p = sabr.params
    expected = float(hagan_sabr_iv(FX.forward, 1.1, 0.25, p["SABR α"], 1.0, p["SABR ρ"], p["SABR ν"]))
    assert sabr.vol_at(1.1) == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="strike"):
        smile.vol_at(0.0)
