"""Smile vegas by model parameter and by quote bucket, for equity (SVI,
SABR) and FX (ATM / RR / BF quotes) smiles (VM-18), checked against
analytic chain rules and against brute-force bump-and-refit."""

from __future__ import annotations

import numpy as np
import pytest

from derivx.fx.pricing import FxMarket, gk_price
from derivx.fx.smile import SmileConventions, TenorQuote, build_smile
from derivx.fx.smile_risk import quote_vegas
from derivx.fx.smile_risk import sabr_parameter_vegas as fx_sabr_vegas
from derivx.vol.implied import black_price, black_vega
from derivx.vol.sabr import hagan_sabr_iv
from derivx.vol.smile_risk import sabr_parameter_vegas, svi_bucket_vegas, svi_parameter_vegas
from derivx.vol.svi import fit_svi_strike_slice, svi_iv

SVI = {"a": 0.01, "b": 0.1, "rho": -0.6, "m": 0.02, "sigma": 0.15}
F, T, DF = 100.0, 0.5, 0.98


def test_svi_parameter_vegas_follow_the_chain_rule():
    for K, call in ((80.0, False), (100.0, True), (115.0, True)):
        out = svi_parameter_vegas(K, F, T, SVI, DF, call).set_index("parameter")["dV_dp"]
        k = np.log(K / F)
        a, b, rho, m, sig = (SVI[n] for n in ("a", "b", "rho", "m", "sigma"))
        u, R = k - m, np.sqrt((k - m) ** 2 + sig**2)
        vol = float(svi_iv(K, SVI, F=F, T=T))
        vega = float(black_vega(F, K, T, vol, DF))
        dvol_dw = 1.0 / (2.0 * vol * T)
        dw = {"a": 1.0, "b": rho * u + R, "rho": b * u, "m": -b * (rho + u / R), "sigma": b * sig / R}
        for name, d in dw.items():
            assert out[name] == pytest.approx(vega * dvol_dw * d, rel=1e-6, abs=1e-9)


def test_sabr_parameter_vegas_signs_and_alpha_chain_rule():
    alpha, beta, rho, nu = 0.2, 1.0, -0.5, 0.6
    low = sabr_parameter_vegas(F, 85.0, T, alpha, beta, rho, nu, df=DF, is_call=False).set_index("parameter")
    vol = float(hagan_sabr_iv(F, 85.0, T, alpha, beta, rho, nu))
    h = 1e-4
    dvol = hagan_sabr_iv(F, 85.0, T, alpha + h, beta, rho, nu) - hagan_sabr_iv(F, 85.0, T, alpha - h, beta, rho, nu)
    assert low.loc["alpha", "dV_dp"] == pytest.approx(float(black_vega(F, 85.0, T, vol, DF)) * dvol / (2 * h), rel=1e-5)
    assert low.loc["rho", "dV_dp"] < 0  # a less negative correlation lowers low-strike vols
    assert low.loc["nu", "dV_dp"] > 0  # more vol of vol lifts the wings
    assert low.loc["rho", "per_0.01"] == pytest.approx(0.01 * low.loc["rho", "dV_dp"])


def _exact_fit():
    Kq = F * np.exp(np.linspace(-0.5, 0.35, 25))
    fit = fit_svi_strike_slice(Kq, svi_iv(Kq, SVI, F=F, T=T), F=F, T=T)
    assert fit.rmse < 1e-10
    return fit


def test_bucket_vegas_match_a_brute_force_refit_and_add_up():
    fit = _exact_fit()
    K = 95.0
    res = svi_bucket_vegas(
        K, False, F, T, DF, fit.params, fit.strikes, fit.market_vols, fit.weights, fit.loss, fit.f_scale
    )
    base = float(black_price(F, K, T, svi_iv(K, fit.params, F=F, T=T), False, DF))

    def refit_pnl(bumped) -> float:
        f2 = fit_svi_strike_slice(fit.strikes, bumped, F=F, T=T, x0=fit.params)
        return float(black_price(F, K, T, svi_iv(K, f2.params, F=F, T=T), False, DF)) - base

    h = 1e-4
    for i in (2, 12, 20):
        bumped = fit.market_vols.copy()
        bumped[i] += h
        assert res["quotes"]["vega"].iloc[i] == pytest.approx(refit_pnl(bumped) * 0.01 / h, rel=2e-3, abs=1e-6)
    assert res["buckets"]["vega"].sum() == pytest.approx(res["total"], rel=1e-12)
    assert res["buckets"]["n_quotes"].sum() == fit.strikes.size
    assert res["total"] == pytest.approx(refit_pnl(fit.market_vols + h) * 0.01 / h, rel=2e-3)
    assert res["total"] == pytest.approx(res["black_vega"], rel=0.02)  # a parallel move is almost absorbed
    atm = res["buckets"].set_index("bucket")["vega"]
    assert atm.idxmax() in ("ATM (|z| <= 0.25)", "-1 to -0.25")  # a near-the-money put loads on nearby quotes


def test_bucket_vegas_reject_bad_inputs():
    fit = _exact_fit()
    with pytest.raises(ValueError, match="T > 0"):
        svi_bucket_vegas(95.0, False, F, 0.0, DF, fit.params, fit.strikes, fit.market_vols)
    with pytest.raises(ValueError, match="more than 5 quotes"):
        svi_bucket_vegas(95.0, False, F, T, DF, fit.params, fit.strikes[:5], fit.market_vols[:5])


M = FxMarket(1.08, 0.04, 0.025, 0.5, 0.505)
QUOTE = TenorQuote("6M", 0.075, -0.012, 0.003, -0.022, 0.009)


def test_fx_quote_vegas_at_the_atm_strike_are_pure_atm_vega():
    conv = SmileConventions()
    for model in ("SABR", "VV"):
        smile = build_smile(M, QUOTE, conv, model)
        K = smile.atm_strike
        table = quote_vegas(M, QUOTE, conv, model, K, True).set_index("quote")
        bv = 0.01 * float(black_vega(M.forward, K, M.tau_e, QUOTE.atm, df=M.df_d))
        assert table.loc["ATM", "vega"] == pytest.approx(bv, rel=0.02)
        assert (
            abs(table.loc["RR25", "vega"]) < 1e-7 and abs(table.loc["BF25", "vega"]) < 1e-7
        )  # ATM vol is the ATM quote
    vv = quote_vegas(M, QUOTE, conv, "VV", K, True).set_index("quote")
    assert not vv.loc["RR10", "used"] and vv.loc["RR10", "vega"] == 0.0  # vanna-volga only goes through 25-delta


def test_fx_quote_vegas_of_a_25_delta_put_load_on_rr_and_bf():
    conv = SmileConventions(strangle="smile")
    smile = build_smile(M, QUOTE, conv, "VV")
    K = smile.pillars["25P"][0]
    table = quote_vegas(M, QUOTE, conv, "VV", K, False).set_index("quote")
    # 25P pillar vol = ATM + BF25 - RR25 / 2: a +1 point RR is -1/2 point at the put, a +1 point BF is +1 point
    assert table.loc["RR25", "vega"] < 0 < table.loc["BF25", "vega"]
    assert table.loc["RR25", "vega"] / table.loc["BF25", "vega"] == pytest.approx(-0.5, rel=0.1)

    def price(bf25: float) -> float:
        bumped = build_smile(M, TenorQuote("6M", 0.075, -0.012, bf25, -0.022, 0.009), conv, "VV")
        return float(gk_price(M, K, bumped.vol(K), False))

    brute = (price(0.0031) - price(0.0029)) / 0.0002 * 0.01  # an independent central difference, +-1 vol bp
    assert table.loc["BF25", "vega"] == pytest.approx(brute, rel=1e-3)


def test_fx_sabr_parameter_vegas():
    smile = build_smile(M, QUOTE, SmileConventions(), "SABR")
    out = fx_sabr_vegas(smile, smile.atm_strike, True).set_index("parameter")
    vol = smile.vol(smile.atm_strike)
    assert out.loc["alpha", "dV_dp"] > 0 and np.isfinite(out["dV_dp"]).all()
    assert out.loc["alpha", "dV_dp"] == pytest.approx(
        float(black_vega(M.forward, smile.atm_strike, M.tau_e, vol, df=M.df_d)) * vol / smile.params["alpha"], rel=0.05
    )  # ATM vol is about proportional to alpha
    with pytest.raises(ValueError, match="SABR smile"):
        fx_sabr_vegas(build_smile(M, QUOTE, SmileConventions(), "VV"), smile.atm_strike, True)
