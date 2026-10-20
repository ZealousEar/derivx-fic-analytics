"""ANALYTICS lab in AppTest, on the committed real sample and labelled
synthetic data, with Yahoo replaced by a recorded stand-in (no network):
every tab renders, uploads are validated, an attached upload never
replaces a later pull, results go stale when the data change, pulls are
cached only per (ticker, request, 5-minute bucket), reruns are
deterministic, and every result downloads with a manifest (AL-04, AL-08,
AL-12, AL-13, AL-15, UIE-02, UIE-11, UIE-12, UIE-13, critic-4,
ui-engineering-V5, vol-models-V2, G14)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _fake_yahoo import Clock, FakeTicker
from streamlit.testing.v1 import AppTest

from derivx.data import market_data
from derivx.data.chain_schema import REQUIRED_COLUMNS, chain_to_csv, load_chain_file
from derivx.data.samples import load_sample_chain
from derivx.data.synthetic import synthetic_history
from derivx.ui import analytics_lab as lab

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any Yahoo request in these tests goes to the recorded stand-in, and
    the shared pull cache starts empty."""

    calls = []

    def fake_fetch(ticker, **kw):
        calls.append((ticker, kw.get("max_expiries")))
        kw.setdefault("backoff", market_data.Backoff(sleep=lambda s: None))
        kw.setdefault("now", Clock())  # the recorded session's evening: the test does not depend on today's date
        return market_data.fetch_option_chain(ticker, client_factory=lambda s: FakeTicker(s), request_pause=0.0, **kw)

    monkeypatch.setattr(lab, "fetch_option_chain", fake_fetch)
    monkeypatch.setattr("yfinance.Ticker", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    lab._pull_shared.clear()
    yield calls
    lab._pull_shared.clear()


def _lab() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("[ ANALYTICS ]").run()
    assert not at.exception
    return at


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [w.value for w in at.warning] + [s.value for s in at.success]
    return "\n".join(str(p) for p in parts)


def _banners(at: AppTest) -> list:
    return [m.value for m in at.markdown if "dx-lab-banner" in m.value and "<style>" not in m.value]


def _active(at: AppTest):
    return at.session_state[lab.K_DATA]["dataset"]


def _charts(at: AppTest) -> dict:
    return {
        json.loads(c.proto.spec)["layout"]["meta"]["chart"]: json.loads(c.proto.spec) for c in at.get("plotly_chart")
    }


def test_every_tab_renders_on_the_bundled_sample_without_network(no_network):
    at = _lab()
    banners = _banners(at)
    assert len(banners) == 3  # market data, calibration, model comparison
    for b in banners:
        assert "SPY" in b and "session 2026-10-20" in b and "fetched 2026-10-20 20:59:39 UTC" in b
        assert "bundled sample: real SPY chain from Yahoo Finance" in b and "10,040 rows" in b
    text = _text(at)
    assert "genuine daily snapshots" in text and "tools/download_spy_chains.py" in text
    assert not at.error and no_network == []
    report = next(d.value for d in at.dataframe if "removed_itm" in d.value.columns)  # hygiene by expiry
    assert len(report) == 31 and "removed_itm" in report.columns
    # the sample's recorded fetch status is shown, with the spot drift across its expiry responses
    assert any(e.label.startswith("Fetch status: ok, spot drift +0.0 bp of spot") for e in at.expander)
    # the pricer page sees the filtered sample with its provenance
    assert at.session_state["market_chain_meta"]["ticker"] == "SPY"


def test_calibration_defaults_skip_expired_slices_and_export_a_manifest():
    at = _lab()
    sel = next(m for m in at.multiselect if m.label == "Expiries")
    assert [d.date().isoformat() for d in sel.value] == ["2026-11-20", "2027-01-15"]  # nearest 30 / 91 days
    at.button(key="lab_cal_run").click().run()
    assert not at.exception
    stored = at.session_state[lab.K_CAL]
    table = stored["result"].table()
    assert [str(e) for e in table["expiry"]] == ["2026-11-20", "2027-01-15"]
    assert (table["rmse_bp"] < 20).all() and table["butterfly_ok"].all()
    chart = _charts(at)["lab_calibration"]
    names = [t["name"] for t in chart["data"]]
    assert "OTM puts (K < F)" in names and "OTM calls (K > F)" in names and "SVI fit" in names
    m = stored["manifest"]
    sample = load_sample_chain()
    assert m["kind"] == "calibration" and m["data"]["data_sha256"] == sample.fingerprint
    assert m["data"]["fetched_at_utc"].startswith("2026-10-20T20:59:39")
    assert m["inputs"]["filters"] == {"min_volume": 10, "min_mid": 0.1, "max_spread": 0.4, "rate_cc": 0.0}
    zipped = lab.results_zip(m, lab._calibration_tables(stored["result"]))
    with zipfile.ZipFile(io.BytesIO(zipped)) as zf:
        assert {"manifest.json", "params.csv", "metrics.csv", "skipped.csv"} <= set(zf.namelist())
    assert any(b.label.startswith("Download results (.zip") for b in at.get("download_button"))


def test_model_comparison_shows_every_model_and_never_mixes_calls_and_puts():
    at = _lab()
    assert at.selectbox(
        key=next(k for k in at.session_state if str(k).startswith("lab_mc_exp_"))
    ).value == pd.Timestamp("2026-11-20")
    at.button(key="lab_mc_run").click().run()
    assert not at.exception
    comp = at.session_state[lab.K_MC]["comp"]
    assert (
        comp.table["model"].tolist() == ["svi", "sabr", "spline"] and (comp.table["status"].str.startswith("ok")).all()
    )
    q = comp.quotes
    assert q["strike"].is_unique  # one quote per strike
    assert (q.loc[q["k"] < 0, "option_type"] == "put").all() and (q.loc[q["k"] > 0, "option_type"] == "call").all()
    names = [t["name"] for t in _charts(at)["lab_model_comparison"]["data"]]
    assert names == ["OTM puts (K < F)", "OTM calls (K > F)", "SVI", "SABR", "IV spline (ln K/F)"]


def test_a_failed_model_is_reported_not_silently_dropped(monkeypatch):
    import derivx.models.comparison as comparison

    def broken(*a, **k):
        raise ValueError("SABR optimiser diverged")

    monkeypatch.setattr(comparison, "calibrate_sabr_smile", broken)
    at = _lab()
    at.button(key="lab_mc_run").click().run()
    assert not at.exception
    assert any("SABR could not be fitted and is not plotted" in w.value for w in at.warning)
    names = [t["name"] for t in _charts(at)["lab_model_comparison"]["data"]]
    assert "SABR" not in names and "SVI" in names


def test_nan_ivs_do_not_break_the_comparison():
    at = _lab()
    chain = load_sample_chain().single()
    chain.loc[chain.index[::7], "implied_vol"] = np.nan  # quoted vols missing on some rows
    at.file_uploader(key="lab_upload").set_value(("nan_ivs.csv", chain_to_csv(chain), "text/csv")).run()
    assert not at.exception and _active(at).source == "upload: nan_ivs.csv"
    assert load_chain_file("x.csv", chain_to_csv(chain)).warnings  # flagged, not hidden
    at.button(key="lab_mc_run").click().run()
    assert not at.exception and not at.session_state[lab.K_MC].get("error")


def test_bad_uploads_show_every_problem_and_keep_the_active_dataset():
    at = _lab()
    bad = pd.DataFrame({"Strike": ["100", "abc"], "impliedVolatility": [18.0, 20.0], "type": ["call", "spread"]})
    at.file_uploader(key="lab_upload").set_value(("bad.csv", bad.to_csv(index=False).encode(), "text/csv")).run()
    assert not at.exception
    err = "\n".join(e.value for e in at.error)
    assert "bad.csv was not loaded" in err and "missing required column(s)" in err and "trade_date" in err
    assert "Traceback" not in err and "/home/" not in err
    assert _active(at).source.startswith("bundled sample")
    at.file_uploader(key="lab_upload").set_value(
        ("junk.parquet", b"definitely not parquet", "application/octet-stream")
    ).run()
    assert "could not read the file as parquet" in "\n".join(e.value for e in at.error)


def test_an_attached_upload_never_replaces_a_later_pull(no_network):
    at = _lab()
    chain = load_sample_chain().single()
    chain = chain[chain["expiry"] <= "2026-12-31"]
    at.file_uploader(key="lab_upload").set_value(("my_saved_chain.csv", chain_to_csv(chain), "text/csv")).run()
    assert _active(at).source == "upload: my_saved_chain.csv"
    assert "source: upload: my_saved_chain.csv" in _banners(at)[0]  # the banner shows it in the same run
    assert "<b>ACTIVE DATASET</b> · SPY ·" in _banners(at)[0]  # from the file's ticker column, not its name
    assert at.text_input(key="lab_ticker").value == "SPY"  # never the file name
    at.number_input(key="lab_max_exp").set_value(3).run()
    at.button(key="lab_pull").click().run()
    assert not at.exception and no_network == [("SPY", 3)]
    assert _active(at).source.startswith("Yahoo Finance") and _active(at).single()["expiry"].nunique() == 3
    assert "source: Yahoo Finance via yfinance" in _banners(at)[0]
    at.number_input(key="lab_minvol").set_value(15).run()  # any rerun with the file still attached
    at.run()
    assert _active(at).source.startswith("Yahoo Finance")
    assert "It replaced: upload: my_saved_chain.csv" in _text(at)


def test_a_multi_day_upload_asks_which_trade_date_to_use():
    at = _lab()
    a = load_sample_chain().single()
    a = a[a["expiry"] <= "2026-11-30"]
    b = a.assign(trade_date=pd.Timestamp("2026-10-19").as_unit("ns"), underlying_price=766.48)
    data = chain_to_csv(pd.concat([b, a]))
    at.file_uploader(key="lab_upload").set_value(("two_days.csv", data, "text/csv")).run()
    assert not at.exception
    pick = next(s for s in at.selectbox if s.label.startswith("This file holds 2 trade dates"))
    assert pick.value == pd.Timestamp("2026-10-20") and _active(at).trade_dates == [pd.Timestamp("2026-10-20")]
    pick.set_value(pd.Timestamp("2026-10-19")).run()
    assert _active(at).trade_dates == [pd.Timestamp("2026-10-19")]
    assert "Using trade date 2026-10-19 of two_days.csv" in _text(at)


def test_results_are_marked_stale_when_the_data_change():
    at = _lab()
    at.button(key="lab_cal_run").click().run()
    assert not any("STALE" in b for b in _banners(at))
    chain = load_sample_chain().single()
    chain = chain[chain["expiry"] <= "2027-03-31"]
    at.file_uploader(key="lab_upload").set_value(("other.csv", chain_to_csv(chain), "text/csv")).run()
    assert any("STALE CALIBRATION: COMPUTED ON" in b and "bundled sample" in b for b in _banners(at))
    assert any("not on the active one" in w.value for w in at.warning)
    at.button(key="lab_cal_run").click().run()
    assert not any("STALE" in b for b in _banners(at))


def test_pulls_are_shared_only_within_a_bucket_and_failures_are_never_cached(no_network, monkeypatch):
    a = _lab()
    a.number_input(key="lab_max_exp").set_value(2).run()
    a.button(key="lab_pull").click().run()
    b = _lab()
    b.number_input(key="lab_max_exp").set_value(2).run()
    b.button(key="lab_pull").click().run()
    assert no_network == [("SPY", 2)]  # same ticker, request and bucket: one upstream fetch
    assert _active(b).fetched_at_utc == _active(a).fetched_at_utc  # served with its real fetch time
    b.number_input(key="lab_max_exp").set_value(3).run()
    b.button(key="lab_pull").click().run()
    assert no_network[-1] == ("SPY", 3)  # a different request is a different key
    monkeypatch.setattr(lab.time, "time", lambda: 10**10)  # a later bucket
    b.button(key="lab_pull").click().run()
    assert len(no_network) == 3
    # a rate-limited pull is shown with its reason and not cached
    lab._pull_shared.clear()

    def limited(ticker, **kw):
        no_network.append(("limited", ticker))
        return market_data.fetch_option_chain(
            ticker,
            client_factory=lambda s: FakeTicker(failures={"options": -1}),
            backoff=market_data.Backoff(retries=1, sleep=lambda s: None),
            **{k: v for k, v in kw.items() if k != "backoff"},
        )

    monkeypatch.setattr(lab, "fetch_option_chain", limited)
    c = _lab()
    c.button(key="lab_pull").click().run()
    assert any("rate-limited" in e.value for e in c.error) and _active(c).source.startswith("bundled sample")
    c.button(key="lab_pull").click().run()
    assert sum(1 for x in no_network if x[0] == "limited") == 2  # retried upstream, not served from cache


def test_reruns_are_deterministic():
    at = _lab()
    at.button(key="lab_cal_run").click().run()
    at.button(key="lab_mc_run").click().run()
    first = (_charts(at), [d.value.to_json() for d in at.dataframe], _banners(at))
    at.run()
    second = (_charts(at), [d.value.to_json() for d in at.dataframe], _banners(at))
    assert first == second


def test_walkforward_on_synthetic_history_scores_benchmarks_without_zigzag():
    at = _lab()
    at.radio(key="lab_wf_source").set_value("SYNTHETIC demo history").run()
    at.slider(key="lab_wf_synth_days").set_value(4).run()
    at.button(key="lab_wf_synth_go").click().run()
    assert any("HISTORY" in b and "SYNTHETIC demo history (seed 20261020): not market data" in b for b in _banners(at))
    at.number_input(key="lab_wf_maxdays").set_value(40).run()
    at.button(key="lab_wf_run").click().run()
    assert not at.exception
    stored = at.session_state[lab.K_WF]
    s = stored["result"].summary.set_index("predictor")
    assert {"persistence", "sticky_moneyness"} <= set(s.index) and s.loc[
        "svi_sticky_strike", "benchmark"
    ] == "persistence"
    assert stored["manifest"]["seed"] == 20261020 and stored["manifest"]["model"]["horizon"] == 1
    chart = _charts(at)["lab_walkforward"]
    assert len(chart["data"]) == 5  # one line per predictor + the noise floor
    for trace in chart["data"]:
        xs = pd.to_datetime(trace["x"])
        assert xs.is_monotonic_increasing and xs.is_unique  # one point per date: no zig-zag across expiries
    assert "Diebold-Mariano" in _text(at) and "Wilson 95%" in _text(at)


def test_walkforward_uploads_refuse_duplicate_dates_and_explain_single_snapshots():
    at = _lab()
    data = chain_to_csv(load_sample_chain().single().head(400))
    up = at.file_uploader(key="lab_wf_upload")
    up.set_value([("a.csv", data, "text/csv"), ("b.csv", data, "text/csv")]).run()
    assert not at.exception
    assert "trade date 2026-10-20 is in both upload: a.csv and upload: b.csv" in "\n".join(e.value for e in at.error)
    at.file_uploader(key="lab_wf_upload").set_value(("a.csv", data, "text/csv")).run()
    assert any("needs at least two trading days" in w.value for w in at.warning)


def test_every_expiry_comparison_reports_all_models_and_skips_with_reasons():
    at = _lab()
    at.number_input(key="lab_mc_all_maxdays").set_value(10).run()
    at.button(key="lab_mc_all_run").click().run()
    assert not at.exception
    stored = at.session_state[lab.K_MC_ALL]
    scores, skipped = stored["scores"], stored["skipped"]
    assert set(scores["model"]) == {"svi", "sabr", "spline"} and (scores["days"] <= 10).all()
    assert scores["days"].min() >= 2
    assert {"> max_days=10" in r or "< min_days" in r for r in skipped["reason"]} == {True}
    statements = [m.value for m in at.markdown if "out-of-sample RMSE on" in m.value]
    assert len(statements) == 3 and not any("outperforms" in s for s in statements)
    assert stored["manifest"]["model"]["max_days"] == 10


def test_the_equity_page_gets_standard_tenors_of_the_active_chain_with_provenance():
    at = _lab()
    chain = at.session_state["market_chain"]
    days = sorted((pd.to_datetime(chain["expiry"]).dt.normalize() - pd.Timestamp("2026-10-20")).dt.days.unique())
    assert len(days) <= 6 and 31 in days and min(days) >= 2  # nearest 1W, 1M, ... 1Y; never expired slices
    meta = at.session_state["market_chain_meta"]
    assert meta["ticker"] == "SPY" and "expiries nearest 1W, 1M" in meta["source"]
    assert meta["data_sha256"] == load_sample_chain().fingerprint
    at.sidebar.radio[0].set_value("[ EQUITY ]").run()
    at.radio(key="eq_smile_src").set_value("MARKET").run()
    assert not at.exception
    assert len(at.selectbox(key="eq_smile_expiry").options) == len(days)


def test_a_file_with_only_the_required_columns_never_crashes_the_lab():
    """volume is optional in the schema: the default min volume filter used to
    raise a ValueError traceback (with server paths) on such a file."""

    at = _lab()
    chain = load_sample_chain().single()
    chain = chain[chain["expiry"] <= "2026-12-31"][list(REQUIRED_COLUMNS)]
    at.file_uploader(key="lab_upload").set_value(("minimal.csv", chain_to_csv(chain), "text/csv")).run()
    assert not at.exception and _active(at).source == "upload: minimal.csv"
    err = "\n".join(e.value for e in at.error)
    assert "no 'volume' column" in err and "set min volume to 0" in err and "Traceback" not in err
    at.number_input(key="lab_minvol").set_value(0).run()
    assert not at.exception and not at.error
    at.button(key="lab_cal_run").click().run()
    at.button(key="lab_mc_run").click().run()
    assert not at.exception and at.session_state[lab.K_CAL]["result"].slices


def test_walkforward_files_without_volume_explain_the_filter_instead_of_crashing():
    at = _lab()
    files = [
        (f"d{i}.csv", chain_to_csv(df.drop(columns=["volume"])), "text/csv")
        for i, df in enumerate(synthetic_history(3).values())
    ]
    at.file_uploader(key="lab_wf_upload").set_value(files).run()
    assert not at.exception
    assert any("data note(s) on these files" in e.label for e in at.expander)
    at.number_input(key="lab_wf_maxdays").set_value(20).run()
    at.button(key="lab_wf_run").click().run()
    assert not at.exception and lab.K_WF not in at.session_state
    assert any("no 'volume' column" in e.value for e in at.error)
    at.number_input(key="lab_wf_minvol").set_value(0).run()
    at.button(key="lab_wf_run").click().run()
    assert not at.exception and not at.session_state[lab.K_WF]["result"].empty


def test_spot_drift_across_expiry_responses_is_flagged(monkeypatch):
    class Drifting(FakeTicker):
        def option_chain(self, expiry):
            self.spot *= 1.001  # the spot moves 10 bp between consecutive expiry responses
            return super().option_chain(expiry)

    def drifting_fetch(ticker, **kw):
        kw["backoff"] = market_data.Backoff(sleep=lambda s: None)
        return market_data.fetch_option_chain(
            ticker, client_factory=lambda s: Drifting(s), request_pause=0.0, now=Clock(), **kw
        )

    monkeypatch.setattr(lab, "fetch_option_chain", drifting_fetch)
    at = _lab()
    at.number_input(key="lab_max_exp").set_value(3).run()
    at.button(key="lab_pull").click().run()
    assert not at.exception and _active(at).source.startswith("Yahoo Finance")
    assert any(e.label == "Fetch status: ok, spot drift +20.0 bp of spot" for e in at.expander)
    assert any("Spot moved +20.0 bp between the first and the last expiry response" in w.value for w in at.warning)
    assert "SPOT RANGE" in "\n".join(m.value for m in at.markdown)


def test_a_model_comparison_error_is_not_shown_against_other_data(monkeypatch):
    def broken(*a, **k):
        raise ValueError("4 usable quotes after hygiene; need 8")

    monkeypatch.setattr(lab, "fit_models", broken)
    at = _lab()
    at.button(key="lab_mc_run").click().run()
    assert any("Model comparison failed for this slice" in e.value for e in at.error)
    chain = load_sample_chain().single()
    chain = chain[chain["expiry"] <= "2027-03-31"]
    at.file_uploader(key="lab_upload").set_value(("other.csv", chain_to_csv(chain), "text/csv")).run()
    assert not any("Model comparison failed" in e.value for e in at.error)


def test_synthetic_rows_are_labelled_even_under_a_file_name():
    at = _lab()
    day = list(synthetic_history(1).values())[0]
    at.file_uploader(key="lab_upload").set_value(("my_chain.csv", chain_to_csv(day), "text/csv")).run()
    assert not at.exception
    banner = _banners(at)[0]
    assert "source: upload: my_chain.csv" in banner and "SYNTHETIC ROWS: not market data" in banner
    assert "dx-lab-banner synthetic" in banner
