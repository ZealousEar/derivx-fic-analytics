"""Run manifests, export tables and curve-set files: what a result was
produced from is recorded, every table adds up to the numbers it exports,
and a saved curve set rebuilds to the same curves or is refused."""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from derivx.data.rates_history import bundled_store
from derivx.rates import curve_io
from derivx.rates import portfolio as pf
from derivx.rates.curves import bootstrap
from derivx.rates.examples import EXAMPLE_ASOF, example_market
from derivx.rates.exports import book_cashflows, curve_grid, curve_nodes, market_tables, risk_report
from derivx.rates.manifest import RunManifest, bundle_zip, market_record, run_manifest, sha256_json, store_record
from derivx.rates.vol_examples import illustrative_cap_atm, illustrative_swaption_grid


def _book():
    market = example_market()
    trades, _ = pf.parse_blotter(pf.example_blotter("USD"))
    vols, _ = pf.build_vols(market, trades, illustrative_swaption_grid("USD"), illustrative_cap_atm("USD"),
                            hull_white=curve_io.HW_DEFAULT)  # fmt: skip
    return market, pf.price_book(trades, market, vols)


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------


def test_run_manifest_records_code_packages_inputs_data_and_seed():
    market = example_market()
    store = bundled_store()
    when = dt.datetime(2026, 10, 20, 12, 0, tzinfo=dt.timezone.utc)
    record = run_manifest("book risk", {"trades": [{"id": "A", "notional": 1e7}], "asof": market.asof},
                          market_record(market, {"curve": "ILLUSTRATIVE quotes"}, store),
                          {"bump_bp": 1.0, "nan": float("nan")}, created_at=when)  # fmt: skip
    payload = json.loads(record.to_json())
    assert payload["manifest_version"] == 1 and payload["kind"] == "book risk"
    assert payload["created_at_utc"] == "2026-10-20T12:00:00+00:00"
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", payload["code"]["sha"])
    assert {"python", "numpy", "pandas", "scipy", "streamlit"} <= set(payload["packages"])
    assert payload["seed"] is None  # nothing random in the rates stack
    assert payload["model"]["nan"] is None and payload["inputs"]["asof"] == "2026-10-19"
    data = payload["data"]
    assert data["valuation_date"] == "2026-10-19"
    curves = {c["key"]: c for c in data["curves"]}
    assert set(curves) == {"curve", "govt"} and curves["curve"]["source"] == "ILLUSTRATIVE quotes"
    assert curves["curve"]["conventions"] == "USD SOFR" and curves["govt"]["quotes"][0][0] == "BILL"
    assert curves["curve"]["quotes_sha256"] == sha256_json(curves["curve"]["quotes"])
    assert data["store"]["rows_sha256"] == store.fingerprint
    ust = next(d for d in data["store"]["datasets"] if d["dataset"] == "UST_PAR")
    assert any("home.treasury.gov" in u for u in ust["urls"]) and any("fredgraph.csv?id=DGS1MO" in u
                                                                      for u in ust["urls"])  # fmt: skip
    assert ust["origins"] == ["bundled snapshot"] and ust["last"] >= "2026-10-19"
    assert RunManifest.from_dict(payload).to_dict() == payload
    with pytest.raises(ValueError, match="manifest version"):
        RunManifest.from_dict({**payload, "manifest_version": 99})


def test_data_hash_follows_the_quotes():
    market = example_market()
    base = market_record(market)
    quotes = list(market.curve.quotes)
    quotes[3] = replace(quotes[3], rate=quotes[3].rate + 1e-6)
    moved = market.replace(curve=bootstrap(market.asof, quotes, "USD SOFR"))
    other = market_record(moved)
    assert base["data_sha256"] != other["data_sha256"]
    assert market_record(market)["data_sha256"] == base["data_sha256"]  # deterministic


def test_bundle_zip_holds_the_manifest_and_every_table():
    record = run_manifest("curve set", {"x": 1}, created_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    tables = {"a": pd.DataFrame({"x": [1, 2]}), "b": pd.DataFrame({"y": ["u"]})}
    payload = bundle_zip(record, tables)
    assert payload == bundle_zip(record, tables)  # fixed timestamps: same inputs, same bytes
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert zf.namelist() == ["manifest.json", "a.csv", "b.csv"]
        assert json.loads(zf.read("manifest.json"))["inputs"] == {"x": 1}
        assert pd.read_csv(io.BytesIO(zf.read("a.csv")))["x"].tolist() == [1, 2]


def test_store_record_lists_every_dataset():
    store = bundled_store()
    record = store_record(store)
    assert {d["dataset"] for d in record["datasets"]} == set(store.datasets())
    boe = next(d for d in record["datasets"] if d["dataset"] == "GBP_OIS_BOE")
    assert any("bankofengland" in u for u in boe["urls"])


# --------------------------------------------------------------------------
# Export tables
# --------------------------------------------------------------------------


def test_curve_nodes_and_grid_are_the_curve():
    market = example_market()
    curve = market.curve
    nodes = curve_nodes(curve)
    assert len(nodes) == len(curve.quotes) == len(curve.ts) - 1
    assert nodes["df"].to_numpy() == pytest.approx(curve.dfs[1:], rel=1e-15)
    assert list(nodes["pillar"]) == list(curve.pillar_dates)
    t = nodes["years"].to_numpy()
    assert nodes["zero_cc_pct"].to_numpy() == pytest.approx([100 * curve.zero_rate(x) for x in t], rel=1e-12)
    assert nodes["quote"].to_numpy() == pytest.approx([100 * q.rate for q in curve.quotes], rel=1e-12)
    # the forwards between pillars rebuild the discount factors
    dt_ = np.diff(np.concatenate([[0.0], t]))
    assert np.exp(-np.cumsum(nodes["forward_cc_pct"].to_numpy() / 100 * dt_)) == pytest.approx(curve.dfs[1:])
    grid = curve_grid(curve)
    sample = grid.iloc[[0, 50, -1]]
    for row in sample.itertuples():
        assert row.df == pytest.approx(curve.df(row.date), rel=1e-14)
        assert row.fwd_3m_simple_act360_pct == pytest.approx(
            100 * curve.forward_rate(row.date, row.date + dt.timedelta(days=91), "ACT/360"), rel=1e-12)  # fmt: skip
    tables = market_tables(market)
    assert set(tables["curve_nodes"]["curve"]) == {"curve", "govt"}


def test_book_cashflows_add_up_to_the_values():
    market, risk = _book()
    flows = book_cashflows(risk, market)
    for k, trade in enumerate(risk.trades):
        mine = flows[flows["id"] == trade.id]
        if trade.type in ("swap", "bond", "cap", "floor"):
            assert mine["pv"].sum() == pytest.approx(risk.pv[k], rel=1e-10, abs=1e-6), trade.id
        else:
            assert mine.empty  # futures and swaptions have no fixed flows
    bond = flows[flows["id"] == "US6"]
    assert bond["leg"].iloc[-1] == "coupon+principal" and bond["amount"].iloc[-1] > 1e7


def test_risk_report_sections_and_totals():
    _, risk = _book()
    report = risk_report(risk)
    assert set(report["section"]) == {"pv", "pv01", "gamma", "vega", "krd", "krd_gamma", "vega_bucket"}
    total = report[(report["section"] == "pv") & (report["id"] == "TOTAL")]
    assert total["value"].iloc[0] == pytest.approx(float(risk.pv.sum()), rel=1e-12)
    krd = report[report["section"] == "krd"]
    assert krd["value"].sum() == pytest.approx(risk.ladder_totals()["pv01"].sum(), rel=1e-12)
    assert report["unit"].str.len().min() > 0


# --------------------------------------------------------------------------
# Curve sets
# --------------------------------------------------------------------------

QUOTES = """curve,instrument,tenor,rate
,DEPO,1W,3.90
OIS,OIS,1M,3.91
,OIS,3M,3.89
,OIS,1Y,3.69
,OIS,2Y,3.55
,OIS,5Y,3.56
,OIS,10Y,3.73
,OIS,30Y,3.80
TREASURY,BILL,3M,4.27
TREASURY,BOND,2Y,4.77
TREASURY,BOND,10Y,5.15
TREASURY,BOND,30Y,5.48
"""


def test_quotes_file_with_curves_builds_a_curve_set():
    quotes, errors, notes = curve_io.read_quotes(QUOTES.encode(), "USD SOFR", EXAMPLE_ASOF)
    assert errors == [] and set(quotes) == {"curve", "govt"}
    assert any("read as percent" in n for n in notes)
    market = curve_io.build_market(EXAMPLE_ASOF, quotes, "USD SOFR")
    for key in ("curve", "govt"):
        assert max(abs(r["error_bp"]) for r in market.get_curve(key).repricing()) < 1e-8


@pytest.mark.parametrize(
    "text, message",
    [
        ("curve,instrument,tenor,rate\nLIBOR,OIS,1Y,3.5\n", "use blank/OIS (swap curve), TREASURY or a term index"),
        ("instrument,tenor\nOIS,1Y\n", "has no rate column"),
        ("curve,instrument,tenor,rate\nTREASURY,BOND,2Y,4.7\n", "no swap-curve rows"),
        ("curve,instrument,tenor,rate\n,BOND,2Y,4.7\n", "instrument 'BOND' is not one of DEPO, OIS, IRS"),
    ],
)
def test_quotes_file_errors(text, message):
    _, errors, _ = curve_io.read_quotes(text.encode(), "USD SOFR", EXAMPLE_ASOF)
    assert any(message in e for e in errors), errors


def test_eur_quotes_file_with_projection_curves():
    market, _, _ = curve_io.default_market(bundled_store(), EXAMPLE_ASOF, "EUR ESTR")
    rows = []
    for key in ("curve", "EURIBOR 6M", "EURIBOR 3M"):
        for q in market.get_curve(key).quotes:
            value = 100 * (1 - q.rate) if q.kind == "FUT" else 100 * q.rate
            rows.append({"curve": "" if key == "curve" else key, "instrument": q.kind, "tenor": q.tenor, "rate": value})
    payload = pd.DataFrame(rows).to_csv(index=False).encode()
    quotes, errors, _ = curve_io.read_quotes(payload, "EUR ESTR", EXAMPLE_ASOF, "percent")
    assert errors == []
    rebuilt = curve_io.build_market(EXAMPLE_ASOF, quotes, "EUR ESTR")
    for key in ("EURIBOR 6M", "EURIBOR 3M"):
        assert rebuilt.get_curve(key).dfs == pytest.approx(market.get_curve(key).dfs, rel=1e-10)
    assert rebuilt.get_curve("EURIBOR 3M").basis_index == "EURIBOR 6M"
    with pytest.raises(ValueError, match="EUR index: the swap curve is USD SOFR"):
        curve_io.build_market(EXAMPLE_ASOF, {"curve": quotes["curve"], "EURIBOR 6M": quotes["EURIBOR 6M"]},
                              "USD SOFR")  # fmt: skip


@pytest.mark.parametrize("conventions, words", [("USD SOFR", "ILLUSTRATIVE"), ("EUR ESTR", "PROXY"),
                                                ("GBP SONIA", "Bank of England")])  # fmt: skip
def test_default_market_labels_its_sources(conventions, words):
    market, sources, _ = curve_io.default_market(bundled_store(), EXAMPLE_ASOF, conventions)
    assert words in sources["curve"] and "US Treasury par yields" in sources["govt"]
    if conventions == "EUR ESTR":
        assert sources["EURIBOR 6M"].endswith("ILLUSTRATIVE quotes (not market data)")
    assert market.asof == EXAMPLE_ASOF


@pytest.mark.parametrize("conventions", ["USD SOFR", "EUR ESTR"])
def test_curve_set_json_round_trip_and_tamper_check(conventions, tmp_path):
    market, sources, _ = curve_io.default_market(bundled_store(), EXAMPLE_ASOF, conventions, "monotone_convex")
    path = tmp_path / "curve.json"
    curve_io.write_curve_set(path, market, sources, {"kind": "test"})
    loaded, got, manifest = curve_io.read_curve_set(path)
    assert manifest == {"kind": "test"} and got == sources and loaded.curve_keys == market.curve_keys
    for key in market.curve_keys:
        assert np.array_equal(loaded.get_curve(key).dfs, market.get_curve(key).dfs), key
        assert loaded.get_curve(key).interpolation == "monotone_convex"
    payload = json.loads(path.read_text())
    payload["curves"]["curve"]["quotes"][4][2] += 1e-5  # a hand edit
    with pytest.raises(ValueError, match="rebuild discount factors .* away from the saved nodes"):
        curve_io.market_from_json(payload)
    with pytest.raises(ValueError, match="not a derivx.curve_set v1 file"):
        curve_io.market_from_json({"format": "something else"})


def test_a_curve_without_quotes_cannot_be_saved():
    from derivx.rates.curves import Market, flat_curve

    with pytest.raises(ValueError, match="was not bootstrapped from quotes"):
        curve_io.curve_set_json(Market(curve=flat_curve(EXAMPLE_ASOF, 0.03)))
