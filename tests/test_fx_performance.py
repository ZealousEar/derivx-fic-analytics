"""Speed of the FX smile builds behind a pair switch without changing the
smiles (B4): the float fast paths of the calibration's inner loop equal the
reference functions, the Newton premium-adjusted strikes equal the Brent
ones, the smiles equal those of the previous implementation to 1e-8 vol,
the smile-risk bumps start from the base smile, and a pair switch of the
whole page takes well under the 2.4-5.3 s it took."""

from __future__ import annotations

import datetime as dt
import math
import time
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import derivx.fx.smile as sm
import derivx.fx.smile_risk as sr
from derivx.fx.conventions import PAIRS, fx_dates, market_delta_type
from derivx.fx.examples import ILLUSTRATIVE_QUOTES, ILLUSTRATIVE_RATES
from derivx.fx.pricing import FxMarket, gk_price, strike_from_delta
from derivx.vol.sabr import sabr_alpha_from_atm, z_over_x

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"
TRADE = dt.date(2026, 10, 20)

# Surface vols (10P, 15P, 20P, 25P, 35P, ATM, 35C, 25C, 20C, 15C, 10C) of the ILLUSTRATIVE quotes, market
# conventions, built at spot 1 by the implementation before B4 (2026-10-21). GBPUSD ON is the worst case: its
# five-quote least-squares SABR fit is flat, and the previous code itself moved by ~1e-9 vol under 1e-15
# relative input perturbations.
PREVIOUS = {
    ("GBPUSD", "ON"): [0.075441324865, 0.074991688359, 0.074660997016, 0.074395977094, 0.073977903812, 0.0735,
                       0.073111556034, 0.072878314906, 0.072766230746, 0.072656534944, 0.072550495837],
    ("EURJPY", "3M"): [0.118629216318, 0.112247173739, 0.107849976109, 0.104524757451, 0.099688132798, 0.095,
                       0.091733497835, 0.090506099732, 0.090265682092, 0.090409800157, 0.091239808501],
    ("EURUSD", "1Y"): [0.088957755053, 0.085973784321, 0.084011701686, 0.082607271735, 0.080767925689, 0.0795,
                       0.079364216731, 0.080070910909, 0.080813394745, 0.081994083859, 0.083975883546],
    ("USDCAD", "2Y"): [0.068283775862, 0.066775115353, 0.065925103932, 0.0654328467, 0.06507656421, 0.0655,
                       0.067300094516, 0.069551511458, 0.071205721401, 0.073473625533, 0.076874247996],
}  # fmt: skip


def _tenor(code: str, tenor: str):
    pair = PAIRS[code]
    row = next(r for r in ILLUSTRATIVE_QUOTES[code] if r[0] == tenor)
    d = fx_dates(pair, TRADE, tenor=tenor)
    m = FxMarket(1.0, ILLUSTRATIVE_RATES[pair.quote], ILLUSTRATIVE_RATES[pair.base], d.tau_e, d.tau_d)
    q = sm.TenorQuote(tenor, *(v / 100 for v in row[1:]))
    return m, q, sm.SmileConventions(market_delta_type(pair, d), "DNS", "market")


def test_float_fast_paths_equal_the_reference_functions():
    rng = np.random.default_rng(4)
    z = np.r_[rng.normal(0, 2, 400), rng.normal(0, 1e-7, 50), 0.0, 1e-6, -1e-6, 60.0, -60.0]
    for rho in (-0.9999, -0.7, 0.0, 0.3, 0.9999):
        ref = z_over_x(z, rho)
        fast = np.array([sm._z_over_x_scalar(float(x), rho) for x in z])
        np.testing.assert_allclose(fast, ref, rtol=2e-15, atol=0)
    assert math.isnan(sm._z_over_x_scalar(math.nan, 0.1))
    K = np.array([0.8, 0.97, 1.0, 1.03, 1.4])
    np.testing.assert_allclose(
        sm.sabr_beta1_vol(1.0, K, 0.5, 0.08, -0.3, 0.9),
        sm.sabr_beta1_vol(1.0, np.tile(K, 3), 0.5, 0.08, -0.3, 0.9)[:5],
        rtol=2e-15,
    )  # noqa: E501
    m = FxMarket(1.08, 0.04, 0.02, 0.7, 0.71)
    for K, vol, call in ((1.0, 0.08, True), (1.2, 0.3, False), (1.08, 0.0, True), (0.9, 1e-9, False)):
        assert sm._black_value(m.forward, K, m.tau_e, vol, call, m.df_d) == pytest.approx(
            float(gk_price(m, K, vol, call)), rel=1e-14, abs=1e-16
        )
    with pytest.raises(ValueError, match="vols must be finite"):
        sm._black_value(m.forward, 1.0, m.tau_e, math.nan, True, m.df_d)
    for T, atm, rho, nu in (
        (0.25, 0.07, -0.3, 0.9),
        (2.0, 0.3, 0.8, 2.5),
        (1 / 365, 0.1, -0.99, 9.0),
        (1.0, 0.1, 0.0, 0.5),
    ):
        assert sm._beta1_alpha_at_forward(1.0, T, atm, rho, nu) == pytest.approx(
            sabr_alpha_from_atm(1.0, T, atm, 1.0, rho, nu), rel=1e-13
        )
    with pytest.raises(ValueError, match="no positive alpha"):
        sm._beta1_alpha_at_forward(1.0, 30.0, 2.0, -0.99, 9.0)


def test_newton_premium_adjusted_strikes_equal_the_brent_ones_or_raise_alike():
    rng = np.random.default_rng(0)
    worst, raised = 0.0, 0
    for _ in range(2000):
        m = FxMarket(
            rng.uniform(0.5, 150),
            rng.uniform(-0.01, 0.08),
            rng.uniform(-0.01, 0.08),
            rng.uniform(1 / 365, 5),
            rng.uniform(1 / 365, 5),
        )  # noqa: E501
        dtype = str(rng.choice(["spot", "forward", "spot_pa", "forward_pa"]))
        deltas = [float(x) for x in rng.choice([0.25, -0.25, 0.10, -0.10, 0.45, -0.45, 0.05, 0.49], 4)]
        calls = [x > 0 for x in deltas]
        vols = rng.uniform(0.02, 1.5, 4)
        refs = []
        for d, v, c in zip(deltas, vols, calls):
            try:
                refs.append(strike_from_delta(m, d, float(v), c, dtype))
            except ValueError:
                refs.append(None)
        if any(r is None for r in refs):
            raised += 1
            with pytest.raises(ValueError):
                sm.strikes_from_deltas(m, deltas, vols, calls, dtype)
            continue
        got = sm.strikes_from_deltas(m, deltas, vols, calls, dtype)
        worst = max(worst, float(np.max(np.abs(got / np.array(refs) - 1.0))))
    assert raised > 100 and worst < 5e-14


@pytest.mark.parametrize("code,tenor", list(PREVIOUS))
def test_smiles_equal_the_previous_implementation_to_1e_8_vol(code, tenor):
    m, q, conv = _tenor(code, tenor)
    smile = sm.build_smile(m, q, conv, "SABR")
    vols = [v for _, _, v in sm.smile_by_delta(smile)]
    np.testing.assert_allclose(vols, PREVIOUS[(code, tenor)], rtol=0, atol=1e-8)


def test_smile_risk_bumps_start_from_the_base_smile(monkeypatch):
    m, q, conv = _tenor("EURJPY", "3M")
    base = sm.build_smile(m, q, conv, "SABR")
    K = base.points[1].strike
    calls = []
    real = sm._quote_residuals

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sm, "_quote_residuals", counting)
    warm = sr.quote_vegas(m, q, conv, "SABR", K, False, base=base)
    n_warm = len(calls)
    calls.clear()
    monkeypatch.setattr(sr, "build_smile", lambda m_, q_, c_, model, start=None: sm.build_smile(m_, q_, c_, model))
    cold = sr.quote_vegas(m, q, conv, "SABR", K, False)
    n_cold = len(calls)
    assert n_warm < 0.6 * n_cold, (n_warm, n_cold)
    atm_vega = abs(cold["vega"].iloc[0])
    np.testing.assert_allclose(warm["vega"], cold["vega"], rtol=0, atol=1e-6 * atm_vega)


def test_an_fx_pair_switch_is_fast():
    """The E3 review measured 2.4-5.3 s per pair switch (12 SABR smiles, the
    surface rows and the quote vegas); now about 1 s here. Bound: 3 s."""

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("[ FX ]").run()
    t0 = time.perf_counter()
    at.selectbox(key="fx_pair").set_value("EURJPY").run()  # premium-adjusted deltas: the slowest kind
    elapsed = time.perf_counter() - t0
    assert not at.exception
    assert elapsed < 3.0, f"pair switch took {elapsed:.2f} s"
