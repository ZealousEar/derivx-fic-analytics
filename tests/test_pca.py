"""PCA of curve moves: a numpy reference, the audit's FRED evidence on the
committed sample, sign conventions, reconstruction, rolling loadings,
PCA-neutral weights, factor PV01 and PCA factor shocks."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from derivx.data import rates_history as rh
from derivx.rates import pca
from derivx.rates.examples import EXAMPLE_ASOF, example_market
from derivx.rates.swaps import VanillaSwap, swap_position
from derivx.risk.scenarios import Shock, historical_sources, pca_factor, pca_library, quote_risk, run_scenario

D = dt.date.fromisoformat
AUDIT_TENORS = ("3M", "1Y", "2Y", "5Y", "10Y", "30Y")


def _synthetic(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Levels (decimals) of six tenors driven by level/slope/curvature
    shocks plus noise, on consecutive business days."""

    rng = np.random.default_rng(seed)
    years = pca.tenor_years(AUDIT_TENORS)
    shape = np.column_stack([np.ones(6), years / 30.0 - 0.5, np.exp(-((years - 5.0) ** 2) / 20.0)])
    shocks = rng.normal(size=(n, 3)) * np.array([8.0, 3.0, 1.5]) + rng.normal(size=(n, 1)) * 0.0
    changes_bp = shocks @ shape.T + rng.normal(scale=0.5, size=(n, 6))
    levels = 0.04 + np.cumsum(changes_bp, axis=0) * 1e-4
    days = pd.bdate_range("2020-01-01", periods=n).date
    return pd.DataFrame(levels, index=list(days), columns=list(AUDIT_TENORS))


@pytest.fixture(scope="module")
def ust() -> pd.DataFrame:
    return rh.bundled_store().complete("UST_PAR")


def test_matches_a_numpy_svd_reference():
    levels = _synthetic()
    result = pca.pca_curve_changes(levels, AUDIT_TENORS)
    x = np.diff(levels.to_numpy(), axis=0) * 1e4
    centred = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    eigenvalues = s**2 / (len(x) - 1)
    np.testing.assert_allclose(result.eigenvalues, eigenvalues, rtol=1e-10)
    np.testing.assert_allclose(result.explained, eigenvalues / eigenvalues.sum(), rtol=1e-10)
    for k in range(6):  # same directions up to sign
        assert abs(abs(result.loadings[:, k] @ vt[k]) - 1.0) < 1e-8
    np.testing.assert_allclose(result.loadings.T @ result.loadings, np.eye(6), atol=1e-12)
    # Scores rebuild the data exactly with every factor; with k factors the error is the SVD truncation's.
    assert result.reconstruction_error(6)["ALL"] < 1e-9
    for k in (1, 2, 3):
        truncated = (centred @ vt[:k].T) @ vt[:k]
        rms = np.sqrt(np.mean((centred - truncated) ** 2, axis=0))
        np.testing.assert_allclose(result.reconstruction_error(k)[list(AUDIT_TENORS)].to_numpy(), rms, rtol=1e-8)
    np.testing.assert_allclose(result.scores.std(ddof=1).to_numpy(), np.sqrt(eigenvalues), rtol=1e-10)
    assert result.n_obs == len(levels) - 1 and result.unit == "bp" and result.dropped == 0


def test_units_demeaning_and_levels():
    levels = _synthetic()
    bp = pca.pca_curve_changes(levels, AUDIT_TENORS)
    dec = pca.pca_curve_changes(levels, AUDIT_TENORS, in_bp=False)
    np.testing.assert_allclose(dec.explained, bp.explained, rtol=1e-10)
    np.testing.assert_allclose(dec.eigenvalues * 1e8, bp.eigenvalues, rtol=1e-10)
    np.testing.assert_allclose(dec.sigma_bp, bp.sigma_bp, rtol=1e-10)
    raw = pca.pca_curve_changes(levels, AUDIT_TENORS, demean=False)
    x = np.diff(levels.to_numpy(), axis=0) * 1e4
    moments = np.linalg.eigvalsh(x.T @ x / len(x))[::-1]
    np.testing.assert_allclose(raw.eigenvalues, moments, rtol=1e-10)
    assert not np.any(raw.mean) and raw.demeaned is False
    on_levels = pca.pca_curve_changes(levels, AUDIT_TENORS, use_changes=False)
    assert on_levels.n_obs == len(levels) and on_levels.explained[0] > 0.9  # random-walk levels: one big factor


def test_audit_evidence_on_the_committed_fred_sample(ust):
    """G13: FRED CMT daily changes 3M, 1Y, 2Y, 5Y, 10Y, 30Y, 2015-01-02 to
    2026-10-16 (2949 curves): 77.9 / 12.9 / 5.4 % (96.2% cumulative), PC1
    [0.10, 0.30, 0.46, 0.53, 0.49, 0.42], PC2 [-0.27, -0.52, -0.46, -0.03,
    0.32, 0.58]."""

    result = pca.pca_curve_changes(ust, AUDIT_TENORS, start=D("2015-01-02"), end=D("2026-10-16"))
    assert result.n_obs == 2948 and result.dropped == 0  # 2949 curves, every change one business day
    np.testing.assert_allclose(100 * result.explained[:3], [77.9, 12.9, 5.4], atol=0.05)
    assert 100 * result.explained[:3].sum() == pytest.approx(96.2, abs=0.05)
    np.testing.assert_allclose(result.loadings[:, 0], [0.10, 0.30, 0.46, 0.53, 0.49, 0.42], atol=0.005)
    np.testing.assert_allclose(result.loadings[:, 1], [-0.27, -0.52, -0.46, -0.03, 0.32, 0.58], atol=0.005)


def test_the_store_sample_is_the_raw_fred_and_treasury_data(ust):
    """The same window straight from the committed CSVs (FRED to 2025, the
    Treasury's 2026 CSV after) gives the same numbers: the store adds
    nothing."""

    fred = pd.read_csv(rh.SAMPLE_DIR / "ust_par_fred.csv", index_col=0, parse_dates=True)
    treasury = pd.read_csv(rh.TREASURY_SNAPSHOT, index_col=0)
    treasury.index = pd.to_datetime(treasury.index, format="%m/%d/%Y")
    treasury = treasury.rename(columns={"3 Mo": "DGS3MO", "1 Yr": "DGS1", "2 Yr": "DGS2", "5 Yr": "DGS5",
                                        "10 Yr": "DGS10", "30 Yr": "DGS30"})  # fmt: skip
    ids = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    raw = pd.concat([fred[ids], treasury[ids]]).sort_index().loc["2015-01-02":"2026-10-16"].dropna()
    assert len(raw) == 2949
    x = np.diff(raw.to_numpy(), axis=0) * 100.0  # percent -> bp
    values = np.linalg.eigvalsh(np.cov(x.T))[::-1]
    result = pca.pca_curve_changes(ust, AUDIT_TENORS, start=D("2015-01-02"), end=D("2026-10-16"))
    np.testing.assert_allclose(result.eigenvalues, values, rtol=1e-9)


def test_sign_conventions():
    result = pca.pca_curve_changes(_synthetic(), AUDIT_TENORS)
    pc1, pc2, pc3 = result.loadings[:, 0], result.loadings[:, 1], result.loadings[:, 2]
    assert pc1.sum() > 0 and pc2[-1] > pc2[0]
    assert 0.5 * (pc3[2] + pc3[3]) > 0.5 * (pc3[0] + pc3[-1])
    flipped = pca._orient(-result.loadings)
    np.testing.assert_allclose(flipped, result.loadings)


def test_gaps_windows_and_bad_inputs():
    levels = _synthetic(200)
    gappy = levels.drop(index=levels.index[50:70])  # a four-week hole
    changes, dropped = pca.curve_changes(gappy, AUDIT_TENORS)
    assert dropped == 1 and len(changes) == len(gappy) - 2
    assert pca.pca_curve_changes(gappy, AUDIT_TENORS).dropped == 1
    last = pca.pca_curve_changes(levels, AUDIT_TENORS, window=60)
    assert last.n_obs == 60 and last.end == levels.index[-1]
    # Tenors come back shortest first whatever order they are given in.
    assert pca.pca_curve_changes(levels, ["30Y", "2Y", "3M"]).tenors == ("3M", "2Y", "30Y")
    with pytest.raises(ValueError, match="at least two tenors"):
        pca.pca_curve_changes(levels, ["2Y"])
    with pytest.raises(ValueError, match="not in the history"):
        pca.pca_curve_changes(levels, ["2Y", "4Y"])
    with pytest.raises(ValueError, match="need 20"):
        pca.pca_curve_changes(levels.iloc[:10], AUDIT_TENORS)
    with pytest.raises(ValueError, match="window"):
        pca.pca_curve_changes(levels, AUDIT_TENORS, window=5)
    with pytest.raises(ValueError, match="after the start"):
        pca.pca_curve_changes(levels, AUDIT_TENORS, start=levels.index[-1], end=levels.index[0])
    flat = pd.DataFrame(0.04, index=levels.index, columns=list(AUDIT_TENORS))
    with pytest.raises(ValueError, match="does not move"):
        pca.pca_curve_changes(flat, AUDIT_TENORS)
    with pytest.raises(ValueError, match="1 to 6 factors"):
        last.reconstruction_error(7)


def test_recent_daily_start_skips_month_ends(ust):
    start = pca.recent_daily_start(ust, AUDIT_TENORS)
    assert D("2014-12-01") < start <= D("2015-01-02")  # daily from the last 2014 month end on
    assert pca.recent_daily_start(ust, AUDIT_TENORS, end=D("2013-05-31")) is None  # a lone month end
    eur = rh.bundled_store().complete("EUR_AAA_ECB")
    assert pca.recent_daily_start(eur, ["2Y", "10Y"]) >= D("2026-07-30")


def test_rolling_loadings_match_a_direct_pca_of_each_window():
    levels = _synthetic(400)
    rolling = pca.rolling_loadings(levels, AUDIT_TENORS, window=120, step=40)
    full = pca.pca_curve_changes(levels, AUDIT_TENORS)
    assert rolling.loadings.index[-1] == full.end and rolling.window == 120
    assert list(rolling.loadings.columns.get_level_values(0).unique()) == ["PC1", "PC2", "PC3"]
    last = pca.pca_curve_changes(levels, AUDIT_TENORS, window=120)
    np.testing.assert_allclose(rolling.loadings["PC1"].iloc[-1].to_numpy(), last.loadings[:, 0], atol=1e-10)
    np.testing.assert_allclose(rolling.explained.iloc[-1].to_numpy(), last.explained[:3], atol=1e-10)
    assert (rolling.stability["PC1"] > 0.95).all()  # the level factor keeps its shape
    assert ((rolling.stability >= 0) & (rolling.stability <= 1 + 1e-12)).all().all()
    with pytest.raises(ValueError, match="longer than"):
        pca.rolling_loadings(levels, AUDIT_TENORS, window=1000)


def test_neutral_weights_remove_the_chosen_factors(ust):
    result = pca.pca_curve_changes(ust, AUDIT_TENORS, start=D("2015-01-02"), end=D("2026-10-16"))
    fly = pca.pca_neutral_weights(result, ["2Y", "5Y", "10Y"])
    assert fly["5Y"] == 2.0 and fly["2Y"] < 0 and fly["10Y"] < 0
    np.testing.assert_allclose(pca.factor_exposure(result, fly, 2), [0.0, 0.0], atol=1e-12)
    assert abs(pca.factor_exposure(result, fly, 3)[2]) > 0.1  # a curvature trade
    spread = pca.pca_neutral_weights(result, ["10Y", "2Y"])
    assert spread["10Y"] == 1.0 and pca.factor_exposure(result, spread, 1)[0] == pytest.approx(0.0, abs=1e-12)
    # A leg between the PCA tenors reads the loadings linearly in maturity.
    seven = pca.loadings_at(result, ["7Y"])[0]
    np.testing.assert_allclose(seven, 0.6 * result.loadings[3] + 0.4 * result.loadings[4])
    with pytest.raises(ValueError, match="two .* or three"):
        pca.pca_neutral_weights(result, ["2Y", "5Y", "10Y", "30Y"])
    with pytest.raises(ValueError, match="proportional"):
        pca.neutral_weights(np.array([[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]]), 0, 1.0, (0, 1))


def test_tenor_map_is_numpy_interp():
    xs = np.array([0.25, 1.0, 2.0, 5.0, 10.0, 30.0])
    to = np.array([0.0, 0.25, 0.6, 3.0, 7.5, 30.0, 50.0])
    values = np.array([1.0, -2.0, 3.0, 0.5, 4.0, -1.0])
    np.testing.assert_allclose(pca.tenor_map(to, xs) @ values, np.interp(to, xs, values))
    with pytest.raises(ValueError, match="increase"):
        pca.tenor_map(to, xs[::-1])


def test_factor_pv01_is_the_delta_of_a_shock_along_the_loading():
    market = example_market()
    swap = VanillaSwap.from_conventions(market.curve.conventions, "10Y", 0.0373, 1e7, trade_date=market.asof)
    positions = [swap_position(swap)]
    result = pca.pca_curve_changes(rh.bundled_store().complete("UST_PAR"), AUDIT_TENORS, start=D("2015-01-02"))
    risk = quote_risk(positions, market, "curve")
    per_score = pca.factor_pv01(risk.pv01, risk.years, result, 3)
    for k in range(3):
        profile = tuple(zip(result.years, result.loadings[:, k]))  # +1 score: tenor i up L_ik bp
        shock = Shock(f"PC{k + 1}", profile, govt=False)
        delta = run_scenario(positions, market, shock, {"curve": risk})[0]["delta"]
        assert per_score[k, 0] == pytest.approx(delta, rel=1e-9, abs=1e-9)
    assert per_score[0, 0] < 0  # a receiver loses when the level factor rises


def test_pca_factor_shock_profile_and_sources():
    store = rh.bundled_store()
    shock = pca_factor(1, 2.0, store, end=EXAMPLE_ASOF)
    result = pca.pca_curve_changes(store.complete("UST_PAR"), AUDIT_TENORS,
                                   start=pca.recent_daily_start(store.complete("UST_PAR"), AUDIT_TENORS),
                                   end=EXAMPLE_ASOF)  # fmt: skip
    expected = 2.0 * result.sigma_bp[0] * result.loadings[:, 0]
    np.testing.assert_allclose([b for _, b in shock.profile], expected)
    np.testing.assert_allclose([y for y, _ in shock.profile], result.years)
    assert shock.name == "PC1 level +2 sd" and "77.9% of the variance" in shock.description
    flags = {s.selector: s.flag for s in shock.sources}
    assert flags == {"govt:USD": "real", "curve:USD": "proxy", "curve:EUR": "cross-currency proxy",
                     "curve:GBP": "cross-currency proxy"}  # fmt: skip
    assert set(historical_sources(shock)["curve"]) == {"TREASURY", "SWAP"}
    ten_days = pca_factor(2, -1.0, store, end=EXAMPLE_ASOF, days=10)
    one_day = pca_factor(2, -1.0, store, end=EXAMPLE_ASOF)
    scaled = np.sqrt(10) * np.array([b for _, b in one_day.profile])
    np.testing.assert_allclose([b for _, b in ten_days.profile], scaled)
    window = pca_factor(3, 1.0, store, end=EXAMPLE_ASOF, window=250)
    assert "250 changes" in window.description
    gbp = {s.selector: s.flag for s in pca_factor(1, 1.0, store, "GBP_OIS_BOE").sources}
    assert gbp["curve:GBP"] == "real" and gbp["govt:USD"] == "cross-currency proxy"
    assert [s.name for s in pca_library(store, 1.5, end=EXAMPLE_ASOF)] == [
        "PC1 level +1.5 sd", "PC1 level -1.5 sd", "PC2 slope +1.5 sd", "PC2 slope -1.5 sd",
        "PC3 curvature +1.5 sd", "PC3 curvature -1.5 sd",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="factor"):
        pca_factor(0, 1.0, store)
    with pytest.raises(ValueError, match="no USER_SWAP:USD data"):
        pca_factor(1, 1.0, store, "USER_SWAP:USD")


def test_pca_shock_runs_through_the_scenario_engine():
    market = example_market()
    swap = VanillaSwap.from_conventions(market.curve.conventions, "10Y", 0.0373, 1e7, trade_date=market.asof)
    positions = [swap_position(swap)]
    shock = pca_factor(1, 1.0, end=market.asof)
    row = run_scenario(positions, market, shock)[0]
    risk = quote_risk(positions, market, "curve")
    result = pca.pca_curve_changes(rh.bundled_store().complete("UST_PAR"), AUDIT_TENORS,
                                   start=pca.recent_daily_start(rh.bundled_store().complete("UST_PAR"),
                                                                AUDIT_TENORS), end=market.asof)  # fmt: skip
    expected = pca.factor_pv01(risk.pv01, risk.years, result, 1)[0, 0] * result.sigma_bp[0]
    assert row["delta"] == pytest.approx(expected, rel=1e-9)
    assert abs(row["residual"]) < 1e-3 * abs(row["pnl"])


def _basis_quoted_eur_market():
    """ESTR discount curve, EURIBOR 6M from swaps and EURIBOR 3M quoted as
    3s6s BASIS spreads over the 6M curve (ILLUSTRATIVE quotes)."""

    from derivx.rates.curves import CurveQuote, Market, bootstrap
    from derivx.rates.multicurve import bootstrap_projection, illustrative_quotes

    ois = {"1Y": 0.02, "2Y": 0.021, "5Y": 0.023, "10Y": 0.026, "30Y": 0.028}
    discount = bootstrap(EXAMPLE_ASOF, [CurveQuote("OIS", t, r) for t, r in ois.items()], "EUR ESTR")
    six = bootstrap_projection(EXAMPLE_ASOF, illustrative_quotes(discount, "EURIBOR 6M"), "EURIBOR 6M", discount)
    quotes = illustrative_quotes(discount, "EURIBOR 3M", basis_curve=six)
    three = bootstrap_projection(EXAMPLE_ASOF, quotes, "EURIBOR 3M", discount, basis=six)
    assert any(inst.kind == "BASIS" for inst in three.instruments)
    return Market(curve=discount, projections={"EURIBOR 6M": six, "EURIBOR 3M": three}, govt=example_market().govt)


def test_factor_risk_holds_basis_spreads_like_the_pca_shock():
    """Review finding: the RESEARCH factor risk summed PV01 x loading over
    every quote, BASIS spreads included, so on a 3M curve quoted as 3s6s
    basis the level factor moved the basis too: +1,696 per sd for a 3M
    swap whose PCA level shock loses 43,901. Spread quotes are held, as in
    the shock and the VaR."""

    from derivx.risk.scenarios import factor_risk, shocked_curves

    market = _basis_quoted_eur_market()
    swap = VanillaSwap.from_index("EURIBOR 3M", "10Y", 0.025, 1e7, trade_date=market.asof)
    positions = [swap_position(swap)]
    risk = {k: quote_risk(positions, market, k) for k in shocked_curves(market)}
    store = rh.bundled_store()
    levels = store.complete("UST_PAR")
    result = pca.pca_curve_changes(levels, AUDIT_TENORS, start=pca.recent_daily_start(levels, AUDIT_TENORS),
                                   end=market.asof)  # fmt: skip
    per_score = factor_risk(market, risk, result, 3)
    for k in range(3):
        shock = pca_factor(k + 1, 1.0, store, end=market.asof)
        delta = run_scenario(positions, market, shock, risk)[0]["delta"]
        assert per_score[k, 0] * result.sigma_bp[k] == pytest.approx(delta, rel=1e-9, abs=1e-6)
    assert per_score[0, 0] * result.sigma_bp[0] < -40_000  # receiving fixed vs 3M loses ~44k on a +1 sd level move
    naive = sum(pca.factor_pv01(qr.pv01, qr.years, result, 1) for qr in risk.values())[0, 0]
    assert naive * result.sigma_bp[0] > 0  # what the tab showed before: basis spreads moving with the level
    with pytest.raises(ValueError, match="no key-rate risk"):
        factor_risk(market, {}, result)


def test_pca_shock_on_the_boe_curve_uses_its_par_ois_rates():
    """The swap curve is quoted in par OIS rates: a PCA shock from the BoE
    SONIA history analyses its derived par OIS rates (as the historical
    replay and the VaR do), not its spot rates."""

    from derivx.risk.scenarios import quoted_levels

    store = rh.bundled_store()
    shock = pca_factor(1, 1.0, store, "GBP_OIS_BOE", end=EXAMPLE_ASOF)
    assert "derived par OIS rates" in shock.description
    zero = store.complete("GBP_OIS_BOE")
    tenors = [t for t in pca.DEFAULT_TENORS if t in zero.columns]
    start = pca.recent_daily_start(zero, tenors, end=EXAMPLE_ASOF)
    par = quoted_levels(store, "GBP_OIS_BOE", start, EXAMPLE_ASOF)
    assert par.loc[EXAMPLE_ASOF, "10Y"] == pytest.approx(store.ois_par("GBP_OIS_BOE", EXAMPLE_ASOF)["10Y"])
    expected = pca.pca_curve_changes(par, tenors, start=start, end=EXAMPLE_ASOF)
    np.testing.assert_allclose([b for _, b in shock.profile], expected.sigma_bp[0] * expected.loadings[:, 0])
    on_zero = pca.pca_curve_changes(zero, tenors, start=start, end=EXAMPLE_ASOF)
    assert not np.allclose(expected.sigma_bp[0] * expected.loadings[:, 0], on_zero.sigma_bp[0] * on_zero.loadings[:, 0])
    ust = pca_factor(1, 1.0, store, end=EXAMPLE_ASOF)
    assert "changes in par yields" in ust.description
