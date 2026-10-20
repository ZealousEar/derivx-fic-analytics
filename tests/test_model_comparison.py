"""Smile-model comparison: valid slices only, OTM hygiene, finite in- and
out-of-sample scores for every model, scale-free SABR, and the honest
'spline' baseline that replaced the mislabelled 'local_vol'."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from _vol_chains import synth_chain

from derivx.models.comparison import compare_models, fit_models, select_expiry
from derivx.vol.svi import svi_iv


def _chain(trade_date: dt.date) -> pd.DataFrame:
    expiry = trade_date + dt.timedelta(days=30)
    strikes = np.linspace(88, 112, 13)
    forward = 100.0
    params = {"a": 0.02, "b": 0.25, "rho": -0.1, "m": 0.0, "s": 0.35}
    ivs = svi_iv(strikes, params, F=forward, T=1.0)  # params: total variance at T = 1
    df = pd.DataFrame(
        {
            "trade_date": pd.Timestamp(trade_date),
            "expiry": pd.Timestamp(expiry),
            "strike": strikes,
            "implied_vol": ivs,
            "underlying_price": np.full_like(strikes, forward),
            "option_type": np.where(strikes < forward, "put", "call"),
        }
    )
    return df


def test_compare_models_returns_results():
    chain = _chain(dt.date(2024, 1, 2))
    results = compare_models(chain)
    assert set(results["model"]) == {"svi", "sabr", "spline"}
    assert (results["rmse"] >= 0).all()
    assert np.isfinite(results["rmse_cv"]).all()


def _slice(days=30, **kw):
    chain = synth_chain(days=(0, 7, 30, 91), iv_noise=0.001, seed=3, **kw)
    return chain, chain[pd.to_datetime(chain["expiry"]) == pd.Timestamp("2026-10-19") + pd.Timedelta(days=days)]


def test_every_model_gets_finite_scores_on_a_cleaned_otm_slice():
    _, slc = _slice(30)
    comp = fit_models(slc, rate=0.04, div_yield=0.012, hygiene={"recompute_iv": False})  # noisy quoted vols
    t = comp.table.set_index("model")
    assert np.isfinite(t[["rmse", "rmse_cv", "max_abs_error_bp"]].to_numpy()).all()
    assert (t["status"].str.startswith("ok")).all()
    assert t.loc["spline", "rmse"] == pytest.approx(0.0, abs=1e-12)  # interpolates: in-sample is meaningless
    assert t.loc["spline", "rmse_cv"] > 0
    for model in ("svi", "sabr"):
        assert 5 < t.loc[model, "rmse_bp"] < 30  # 10 vol bp quote noise
    assert comp.quotes["strike"].is_unique  # calls and puts no longer duplicate strikes
    assert comp.table.attrs["forward_source"].startswith("put-call parity")
    assert (comp.quotes.loc[comp.quotes["strike"] < comp.forward, "option_type"] == "put").all()
    grid = np.linspace(comp.quotes["strike"].min(), comp.quotes["strike"].max(), 50)
    for model in ("svi", "sabr", "spline"):
        assert np.isfinite(comp.predict(model, grid)).all()


def test_expired_and_0dte_slices_are_rejected():
    """AL-08: the first (expired) expiry used to be scored with tau = 1e-6."""

    chain, _ = _slice()
    zero = chain[pd.to_datetime(chain["expiry"]) == pd.Timestamp("2026-10-19")]
    with pytest.raises(ValueError, match="expired or too short"):
        compare_models(zero)
    week = chain[pd.to_datetime(chain["expiry"]) == pd.Timestamp("2026-10-26")]
    assert len(compare_models(week)) == 3  # 7 days: valid by default (min_days = 2) ...
    with pytest.raises(ValueError, match="min_days=10"):
        compare_models(week, min_days=10)  # ... and rejected when the caller asks for 10


def test_local_vol_label_is_refused_with_an_explanation():
    """AL-07: 'local_vol' was a cubic spline of implied vol."""

    _, slc = _slice()
    with pytest.raises(ValueError, match="not a local-volatility model"):
        compare_models(slc, models=["svi", "local_vol"])
    with pytest.raises(ValueError, match="unknown model"):
        compare_models(slc, models=["heston"])


def test_a_nan_iv_does_not_break_the_comparison():
    """vol-models-V2: one NaN IV crashed the Model Comparison plot."""

    _, slc = _slice()
    slc = slc.copy()
    slc.iloc[5, slc.columns.get_loc("implied_vol")] = np.nan
    comp = fit_models(slc, hygiene={"recompute_iv": False})
    assert (comp.table["status"].str.startswith("ok")).all()
    assert "iv_missing" in comp.clean.counts


def test_sabr_score_does_not_depend_on_the_price_level():
    """AL-09: with alpha <= 5 and x0 = 0.2, SABR failed at index levels."""

    scores = []
    for spot in (58.0, 580.0, 5800.0):
        _, slc = _slice(30, spot=spot)
        t = compare_models(slc, models=["sabr"], rate=0.04, div_yield=0.012).set_index("model")
        assert "bound" not in t.loc["sabr", "status"]
        scores.append(t.loc["sabr", "rmse_bp"])
    assert max(scores) - min(scores) < 1.0  # vol bp


def test_select_expiry_picks_the_valid_slice_nearest_the_target():
    chain, _ = _slice()
    assert select_expiry(chain, 30) == pd.Timestamp("2026-11-18")
    assert select_expiry(chain, 0) == pd.Timestamp("2026-10-26")  # the 0DTE is never chosen
    with pytest.raises(ValueError, match="no expiry"):
        select_expiry(chain, 30, min_days=400)


def test_too_few_quotes_is_a_clear_error():
    chain = _chain(dt.date(2024, 1, 2)).iloc[:6]
    with pytest.raises(ValueError, match="usable quotes"):
        compare_models(chain)


def test_cv_refits_start_from_training_quotes_only(monkeypatch):
    """Review: the SVI fold refits were warm-started from the full-sample
    fit (which had seen the held-out quotes) while SABR started cold."""

    import derivx.models.comparison as comp_mod

    calls = []
    real = comp_mod.fit_svi_strike_slice

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(comp_mod, "fit_svi_strike_slice", spy)
    _, slc = _slice(30)
    comp_mod.fit_models(slc, models=["svi"], rate=0.04, div_yield=0.012, folds=3)
    assert len(calls) == 4 and all("x0" not in c for c in calls)


def test_every_expiry_is_scored_or_reported_with_a_reason():
    from derivx.models.comparison import compare_models_by_expiry

    chain = synth_chain(days=(0, 1, 30, 91), spot=580.0)
    scores, skipped = compare_models_by_expiry(chain, ("svi", "sabr", "spline"), max_days=60)
    assert sorted(scores["days"].unique()) == [30]
    assert set(scores["model"]) == {"svi", "sabr", "spline"} and scores["rmse_cv_bp"].notna().all()
    reasons = dict(zip(pd.to_datetime(skipped["expiry"]).dt.date.astype(str), skipped["reason"]))
    assert len(reasons) == 3  # same-day, 1-day and beyond max_days: none dropped silently
    assert any("> max_days=60" in r for r in reasons.values())
    assert sum("< min_days" in r for r in reasons.values()) == 2
    assert set(scores["forward_method"]) == {"parity"}


def test_summaries_keep_every_slice_and_word_comparisons_by_sign():
    """critic-1: no one-sided clipping of unfavourable slices, and 'better'
    goes to the model with the lower error."""

    from derivx.models.comparison import describe_pairwise, pairwise_wins, summarise_comparison

    rows = []
    for i, (svi, sabr) in enumerate([(60.0, 30.0), (5000.0, 20.0), (10.0, 40.0)]):
        exp = pd.Timestamp("2026-11-01") + pd.Timedelta(days=7 * i)
        rows += [
            {"trade_date": pd.Timestamp("2026-10-20"), "expiry": exp, "model": "svi", "rmse_cv_bp": svi},
            {"trade_date": pd.Timestamp("2026-10-20"), "expiry": exp, "model": "sabr", "rmse_cv_bp": sabr},
        ]
    scores = pd.DataFrame(rows)
    summary = summarise_comparison(scores).set_index("model")
    assert summary.loc["svi", "slices"] == 3 and summary.loc["svi", "max_rmse_cv_bp"] == 5000.0  # outlier kept
    assert summary.loc["svi", "best_on"] == 1 and summary.loc["sabr", "best_on"] == 2
    res = pairwise_wins(scores, "svi", "sabr")
    assert res == {"slices": 3, "a_better": 1, "b_better": 2, "ties": 0, "median_diff": 30.0}
    text = describe_pairwise("svi", "sabr", res)
    assert text.startswith("SABR has the lower out-of-sample RMSE on 2 of 3 expiries")
    assert "SABR better at the median" in text and "outperforms" not in text
    flipped = describe_pairwise("sabr", "svi", pairwise_wins(scores, "sabr", "svi"))
    assert "SABR has the lower" in flipped and "(median SABR - SVI = -30.0 vol bp; SABR better" in flipped
