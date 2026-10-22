"""Cross-sectional curve models (AL-20): Nelson-Siegel, Diebold-Li,
Svensson and a spline recover curves they can represent exactly, nest as
expected in sample, and are compared out of sample on the real Treasury
history."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from derivx.data import rates_history as rh
from derivx.rates.curve_models import (
    DL_TAU,
    compare_curve_models,
    fit_curve_model,
    fit_ns,
    fit_svensson,
    loo_errors,
    ns_loadings,
    svensson_loadings,
)

T = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def test_nelson_siegel_recovers_its_own_curve():
    beta = np.array([0.045, -0.012, 0.018])
    y = ns_loadings(T, 2.3) @ beta
    fit = fit_ns(T, y)
    assert fit.params["tau"] == pytest.approx(2.3, rel=1e-4)
    np.testing.assert_allclose([fit.params[k] for k in ("b0", "b1", "b2")], beta, atol=1e-7)
    assert fit.rmse_bp < 1e-4
    dl = fit_curve_model("ns_dl", T, ns_loadings(T, DL_TAU) @ beta)
    assert dl.params["tau"] == DL_TAU and dl.rmse_bp < 1e-9
    assert np.nanmax(np.abs(loo_errors("ns", T, y))) < 1e-5  # held-out tenors are predicted exactly


def test_svensson_recovers_its_own_curve_and_nests_nelson_siegel():
    beta = np.array([0.05, -0.02, 0.01, 0.03])
    y = svensson_loadings(T, 0.8, 6.0) @ beta
    fit = fit_svensson(T, y)
    assert fit.rmse_bp < 1e-2
    np.testing.assert_allclose(fit.predict([1.5, 15.0]), svensson_loadings([1.5, 15.0], 0.8, 6.0) @ beta, atol=2e-6)
    assert fit_ns(T, y).rmse_bp > 10 * fit.rmse_bp


def test_models_on_the_treasury_history_are_scored_out_of_sample():
    levels = rh.bundled_store().complete("UST_PAR")
    levels = levels[[c for c in ("3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y")]]
    recent = levels[levels.index >= dt.date(2026, 10, 13)]
    scores, skipped = compare_curve_models(recent)
    assert skipped.empty and set(scores["model"]) == {"ns", "ns_dl", "svensson", "spline"}
    by = scores.pivot_table(index="trade_date", columns="model", values="rmse_bp")
    assert (by["spline"] < 1e-9).all()  # the spline interpolates: only its CV score compares
    assert (by["svensson"] <= by["ns"] + 1e-6).all() and (by["ns"] <= by["ns_dl"] + 1e-6).all()  # nested fits
    assert np.isfinite(scores["rmse_cv_bp"]).all() and (scores["rmse_cv_bp"] > 0).all()
    assert scores["trade_date"].nunique() == len(recent) and (scores["expiry"] == "curve").all()


def test_curve_models_reject_bad_inputs():
    with pytest.raises(ValueError, match="unknown curve model"):
        fit_curve_model("cubic", T, T)
    with pytest.raises(ValueError, match="at least 7"):
        fit_svensson(T[:5], T[:5])
    with pytest.raises(ValueError, match="> 0 years"):
        fit_ns(np.r_[0.0, T[1:]], T)
