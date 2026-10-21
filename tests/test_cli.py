"""``derivx`` command line: pull into the cache, offline reads, exports and
errors (recorded fixtures stand in for the network)."""

from __future__ import annotations

import io
import subprocess
import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from derivx import cli
from derivx.data import rates_history as rh
from derivx.data import rates_sources as rs

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "rates"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def online(monkeypatch):
    """Serve recorded fixtures for every source the CLI asks for."""

    routes = {
        "fredgraph.csv": "fred_ust.csv",
        "newyorkfed": "nyfed_sofr_last3.json",
        "data/EST": "ecb_estr.csv",
        "data/YC": "ecb_yc_par.csv",
        "yield-curves": "boe_latest.zip",
        "treasury.gov": "treasury_2026.csv",
    }
    urls = []

    def fake(request, timeout):
        urls.append(request.full_url)
        for needle, name in routes.items():
            if needle in request.full_url:
                return _Response((FIXTURES / name).read_bytes())
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(rs, "_urlopen", fake)
    monkeypatch.setenv("DERIVX_OFFLINE", "0")
    return urls


def test_pull_downloads_caches_and_exports(online, tmp_path, capsys):
    out = tmp_path / "dgs.csv"
    code = cli.main(["pull", "fred", "--ids", "DGS2,DGS10", "--start", "2026-10-13", "--cache", str(tmp_path / "c"),
                     "--out", str(out), "--show", "2"])  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0 and "FRED: 8 rows downloaded" in text and "sha256" in text
    assert "DGS10" in text and "last 5.09 percent (par_yield)" in text
    written = pd.read_csv(out)
    assert list(written.columns) == list(rs.LONG_COLUMNS) and len(written) == 8
    # Offline: the same request from the cache, no network.
    online.clear()
    code = cli.main(["pull", "fred", "--ids", "DGS2,DGS10", "--start", "2026-10-13", "--cache", str(tmp_path / "c"),
                     "--offline", "--out", str(tmp_path / "dgs.parquet")])  # fmt: skip
    assert code == 0 and online == [] and "from the cache" in capsys.readouterr().out
    assert len(pd.read_parquet(tmp_path / "dgs.parquet")) == 8


@pytest.mark.parametrize(
    "argv, expect",
    [
        (["pull", "nyfed", "--rate", "sofr", "--last", "3"], "NY Fed: 3 rows"),
        (["pull", "ecb-estr", "--last", "3"], "ESTR"),
        (["pull", "ecb-yc", "--tenors", "3M,2Y,10Y,30Y", "--last", "2"], "PY_10Y"),
        (["pull", "boe-ois", "--archive", "latest"], "BoE OIS spot 25Y"),
        (["pull", "treasury", "--year", "2026"], "US Treasury: 70 rows"),
    ],
)
def test_pull_every_source(online, tmp_path, capsys, argv, expect):
    assert cli.main(argv + ["--cache", str(tmp_path)]) == 0
    assert expect in capsys.readouterr().out


def test_pull_errors(tmp_path, capsys):
    assert cli.main(["pull", "fred", "--ids", "DGS10", "--offline", "--cache", str(tmp_path)]) == cli.EXIT_SOURCE_ERROR
    assert "offline: no cached FRED data" in capsys.readouterr().err
    assert cli.main(["pull", "nyfed", "--rate", "sofr", "--last", "0", "--cache", str(tmp_path)]) == 2
    assert "last_n must be at least 1" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["pull", "fred", "--ids", "DGS10", "--start", "13/10/2026"])
    assert exit_info.value.code == 2 and "use YYYY-MM-DD" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["pull", "nowhere"])


def test_pull_latest(monkeypatch, tmp_path, capsys):
    result = rh.LiveRefresh(ok={"SOFR (NY Fed)": True, "UK OIS curve (BoE)": False}, notices=["UK OIS curve: down"])
    monkeypatch.setattr(rh, "refresh_live", lambda **kw: result)
    assert cli.main(["pull", "latest", "--cache", str(tmp_path)]) == cli.EXIT_SOURCE_ERROR
    text = capsys.readouterr().out
    assert "ok   SOFR (NY Fed)" in text and "FAIL UK OIS curve (BoE)" in text and "note: UK OIS curve: down" in text


def test_commands_register_themselves_and_run_as_a_module():
    assert "pull" in cli.COMMANDS
    parser = cli.build_parser()
    assert parser.parse_args(["pull", "fred", "--ids", "DGS2"]).ids == ["DGS2"]
    grouped = parser.parse_args(["pull", "fred", "--ids", "ust,DGS10,ice"]).ids
    assert grouped == list(rs.UST_FRED_IDS) + list(rs.ICE_SWAP_FRED_IDS)  # groups expand, duplicates drop
    done = subprocess.run(
        [sys.executable, "-m", "derivx", "pull", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "DERIVX_OFFLINE": "1"},
        timeout=120,
    )
    assert done.returncode == 0 and "boe-ois" in done.stdout and "treasury" in done.stdout
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'derivx = "derivx.cli:main"' in pyproject


def test_units_option_is_validated(capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["pull", "fred", "--ids", "X", "--units", "x=bp,y=percent"])
    assert args.units == {"X": "bp", "Y": "percent"}
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["pull", "fred", "--ids", "X", "--units", "FOO"])  # used to fail with a dict() error message
    assert exit_info.value.code == 2 and "use SERIES=UNIT" in capsys.readouterr().err


# --------------------------------------------------------------------------
# curve / book / risk / scenario (offline: the committed sample and an empty cache)
# --------------------------------------------------------------------------

QUOTES = """curve,instrument,tenor,rate
,DEPO,1W,3.90
,OIS,1M,3.91
,OIS,6M,3.82
,OIS,1Y,3.69
,OIS,2Y,3.55
,OIS,5Y,3.56
,OIS,10Y,3.73
,OIS,30Y,3.80
"""


@pytest.fixture
def book(tmp_path, capsys):
    path = tmp_path / "trades.csv"
    assert cli.main(["book", "example", "--currency", "USD", "--out", str(path)]) == 0
    assert "EXAMPLE USD book (illustrative trades)" in capsys.readouterr().out
    return path


def test_curve_build_round_trips_through_json(tmp_path, capsys):
    from derivx.rates import curve_io

    quotes = tmp_path / "q.csv"
    quotes.write_text(QUOTES)
    out, nodes = tmp_path / "curve.json", tmp_path / "nodes.csv"
    code = cli.main(["curve", "build", "--quotes", str(quotes), "--asof", "2026-10-19", "--interp", "loglinear",
                     "--out", str(out), "--nodes", str(nodes), "--cache", str(tmp_path / "c")])  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0 and "curve set on 2026-10-19" in text and "max repricing error" in text
    assert "US Treasury par yields" in text  # added from the store (default --treasury)
    market, sources, manifest = curve_io.read_curve_set(out)
    assert market.curve_keys == ("curve", "govt") and sources["curve"] == "q.csv"
    assert manifest["kind"] == "curve build" and manifest["inputs"]["quotes"]["sha256"]
    assert manifest["data"]["store"]["rows_sha256"]
    assert [q.tenor for q in market.curve.quotes] == ["1W", "1M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    assert len(pd.read_csv(nodes)) == len(market.curve.quotes) + len(market.govt.quotes)
    assert cli.main(["curve", "show", "--curve", str(out)]) == 0
    assert "rebuilt from its quotes and checked against its nodes" in capsys.readouterr().out
    assert cli.main(["curve", "build", "--quotes", str(quotes), "--asof", "2026-10-19", "--no-treasury", "--out",
                     str(out)]) == 0  # fmt: skip
    assert curve_io.read_curve_set(out)[0].govt is None
    bad = tmp_path / "bad.csv"
    bad.write_text("curve,instrument,tenor,rate\nLIBOR,OIS,1Y,3.5\n")
    assert cli.main(["curve", "build", "--quotes", str(bad), "--asof", "2026-10-19"]) == 2
    assert "use blank/OIS (swap curve), TREASURY or a term index" in capsys.readouterr().err


def test_risk_on_a_curve_file_equals_risk_on_the_date(book, tmp_path, capsys):
    curve = tmp_path / "curve.json"
    assert cli.main(["curve", "build", "--asof", "2026-10-19", "--out", str(curve)]) == 0
    capsys.readouterr()
    reports = []
    for market in (["--curve", str(curve)], ["--asof", "2026-10-19"]):
        report = tmp_path / f"risk{len(reports)}.csv"
        manifest = tmp_path / f"run{len(reports)}.json"
        code = cli.main(["risk", "--trades", str(book), *market, "--illustrative-vols", "--out", str(report),
                         "--ladder", str(tmp_path / "ladder.csv"), "--manifest", str(manifest)])  # fmt: skip
        text = capsys.readouterr().out
        assert code == 0 and "10 trades priced, 0 skipped" in text and "US10" in text
        reports.append(pd.read_csv(report))
        payload = __import__("json").loads(manifest.read_text())
        assert payload["kind"] == "book risk" and payload["inputs"]["trades"]["rows_valid"] == 10
        assert payload["model"]["vol_sources"]["cube:OIS"].startswith("ILLUSTRATIVE")
    assert "curve_file" in __import__("json").loads((tmp_path / "run0.json").read_text())["data"]
    a, b = reports
    assert list(a["section"]) == list(b["section"])
    assert a["value"].to_numpy() == pytest.approx(b["value"].to_numpy(), rel=1e-9, abs=1e-9)
    total_pv = a[(a["section"] == "pv") & (a["id"] == "TOTAL")]["value"].iloc[0]
    assert total_pv == pytest.approx(a[(a["section"] == "pv") & (a["id"] != "TOTAL")]["value"].sum(), rel=1e-12)
    ladder = pd.read_csv(tmp_path / "ladder.csv")
    assert set(ladder["curve"]) == {"curve", "govt"} and set(ladder["id"]) == {f"US{i}" for i in range(1, 11)}


def test_risk_hedge_and_exit_codes(book, tmp_path, capsys):
    hedge = tmp_path / "hedge.csv"
    code = cli.main(["risk", "--trades", str(book), "--asof", "2026-10-19", "--illustrative-vols", "--hedge",
                     "2Y,5Y,10Y,30Y", "--bid-ask", "0.25", "--move", "5", "--hedge-out", str(hedge)])  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0 and "hedge: residual risk" in text and "Receive fixed" in text
    assert set(pd.read_csv(hedge).columns) == {"curve", "currency", "quote", "years", "book", "hedge", "residual"}
    # Without vols the options are skipped with the reason and a hint, and the exit code says so.
    assert cli.main(["risk", "--trades", str(book), "--asof", "2026-10-19"]) == cli.EXIT_TRADES
    err = capsys.readouterr().err
    assert "skipped: US9 (swaption): no swaption vol cube" in err and "--illustrative-vols" in err
    broken = tmp_path / "broken.csv"
    broken.write_text(book.read_text().replace("swap,USD,receive", "swap,USD,sideways", 1))
    assert cli.main(["risk", "--trades", str(broken), "--asof", "2026-10-19", "--illustrative-vols"]) == 3
    assert "direction 'sideways'" in capsys.readouterr().err
    assert cli.main(["risk", "--trades", str(book)]) == 2
    assert "give the market: --curve curve.json or --asof YYYY-MM-DD" in capsys.readouterr().err


def test_scenario_grid_explain_and_manifest(book, tmp_path, capsys):
    grid, explain, manifest = tmp_path / "grid.csv", tmp_path / "explain.csv", tmp_path / "run.json"
    code = cli.main(["scenario", "--trades", str(book), "--asof", "2026-10-19", "--illustrative-vols",
                     "--shock", "parallel:+25", "--shock", "keyrate:2Y=+10,10Y=-5", "--shock", "pca:1:2",
                     "--shock", "historical:2022-hiking-cycle", "--out", str(grid), "--explain", str(explain),
                     "--manifest", str(manifest)])  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0 and "Parallel +25bp" in text and "2022 hiking cycle" in text
    table = pd.read_csv(grid)
    trades = [c for c in table.columns if c.startswith("US")]
    assert len(table) == 4 and len(trades) == 10
    assert table["TOTAL"].to_numpy() == pytest.approx(table[trades].sum(axis=1).to_numpy(), rel=1e-12)
    rows = pd.read_csv(explain)
    assert {"scenario", "position", "currency", "pnl", "delta", "gamma", "residual", "delta_curve"} <= set(rows.columns)
    first = rows[rows["scenario"] == "Parallel +25bp"]
    trades_only = first[first["position"] != "TOTAL"]
    assert set(trades_only["currency"]) == {"USD"} and len(trades_only) == 10
    assert trades_only["pnl"].sum() == pytest.approx(table["TOTAL"].iloc[0], rel=1e-9)
    assert first[first["position"] == "TOTAL"]["pnl"].iloc[0] == pytest.approx(table["TOTAL"].iloc[0], rel=1e-9)
    payload = __import__("json").loads(manifest.read_text())
    assert payload["inputs"]["shocks"][0] == "parallel:+25"
    replay = next(s for s in payload["data"]["scenario_sources"] if s["shock"] == "2022 hiking cycle")
    assert {src["flag"] for src in replay["sources"]} >= {"real"}
    assert cli.main(["scenario", "--trades", str(book), "--asof", "2026-10-19", "--shock", "twist:5"]) == 2
    assert "--shock 'twist:5': missing a field" in capsys.readouterr().err
    assert cli.main(["scenario", "--trades", str(book), "--asof", "2026-10-19", "--shock", "historical:nope"]) == 2
    assert "unknown episode 'nope'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "spec, names",
    [
        ("parallel:-50", ["Parallel -50bp"]),
        ("twist:5:-10:+10", ["Twist 5Y pivot -10/+10bp"]),
        ("fly:5Y:25", ["2s5s10s fly +25/+0bp"]),
        ("fly:7:20:-5:2:30", ["2s7s30s fly +20/-5bp"]),
        ("keyrate:2Y=+10,10Y=-5", ["Key rates (2Y +10, 10Y -5)"]),
        ("pca:2:-1", ["PC2 slope -1 sd"]),
        ("historical:2023-svb:0.5", ["2023 SVB x0.5"]),
        ("window:2022-09-21:2022-09-28", ["History 2022-09-21 to 2022-09-28"]),
    ],
)
def test_shock_specs(spec, names):
    from derivx.rates.examples import example_market

    shocks = cli.parse_shock(spec, example_market(), rh.bundled_store())
    assert [s.name for s in shocks] == names


def test_library_shock_and_book_check(book, tmp_path, capsys):
    from derivx.rates.examples import example_market

    library = cli.parse_shock("library", example_market(), rh.bundled_store())
    assert len(library) == 18 and library[-1].name.startswith("PC3")
    assert cli.main(["book", "check", "--trades", str(book)]) == 0
    assert "10 valid trades" in capsys.readouterr().out
    assert cli.main(["book", "example", "--currency", "EUR"]) == 0
    assert "EURIBOR 3M/EURIBOR 6M" in capsys.readouterr().out


def test_risk_runs_as_a_module(book, tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "derivx", "risk", "--trades", str(book), "--asof", "2026-10-19", "--illustrative-vols"],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(ROOT / "src"),
            "PATH": "/usr/bin:/bin",
            "DERIVX_OFFLINE": "1",
            "DERIVX_CACHE_DIR": str(tmp_path / "cache"),
        },  # fmt: skip
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert "TOTAL" in done.stdout and "key-rate PV01 of the book" in done.stdout


# --------------------------------------------------------------------------
# Review findings (regressions)
# --------------------------------------------------------------------------


def test_missing_and_broken_files_are_errors_not_tracebacks(book, tmp_path, capsys):
    assert cli.main(["risk", "--trades", str(tmp_path / "nope.csv"), "--asof", "2026-10-19"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: No such file or directory") and "nope.csv" in err and "Traceback" not in err
    assert cli.main(["risk", "--trades", str(book), "--curve", str(tmp_path / "nope.json")]) == 2
    assert "nope.json" in capsys.readouterr().err
    cases = {
        "list.json": "[1, 2]",
        "part.json": '{"format": "derivx.curve_set", "version": 1, "asof": "2026-10-19", "curves": {"curve": {}}}',
        "text.json": "not json",
    }
    for name, text in cases.items():
        path = tmp_path / name
        path.write_text(text)
        assert cli.main(["curve", "show", "--curve", str(path)]) == 2, name
        err = capsys.readouterr().err
        assert err.startswith("error: ") and "Traceback" not in err, (name, err)
    assert cli.main(["curve", "show", "--curve", str(tmp_path / "part.json")]) == 2
    assert "not a complete derivx.curve_set v1 file (KeyError: 'quotes')" in capsys.readouterr().err


def test_unreadable_blotter_files_exit_with_the_reason(tmp_path, capsys):
    for name, payload in {"empty.csv": b"", "ids.csv": b"id\nX1\n", "other.csv": b"foo,bar\n1,2\n"}.items():
        path = tmp_path / name
        path.write_bytes(payload)
        assert cli.main(["book", "check", "--trades", str(path)]) == cli.EXIT_TRADES, name
        err = capsys.readouterr().err
        assert f"error: {name} blotter: " in err, err
    assert cli.main(["risk", "--trades", str(tmp_path / "empty.csv"), "--asof", "2026-10-19"]) == cli.EXIT_TRADES
    assert "the file is empty" in capsys.readouterr().err


def test_bid_ask_without_a_move_uses_the_apps_move(book, capsys):
    args = ["risk", "--trades", str(book), "--asof", "2026-10-19", "--illustrative-vols", "--hedge", "2Y,5Y,10Y,30Y"]
    assert cli.main(args + ["--bid-ask", "0.25"]) == 0
    penalised = capsys.readouterr().out
    assert "the cost penalty hedges against 5bp curve moves" in penalised
    assert cli.main(args + ["--bid-ask", "0.25", "--move", "5"]) == 0
    explicit = capsys.readouterr().out
    assert penalised.split("hedge: ")[1] == explicit.split("hedge: ")[1]  # the same hedge as --move 5
    assert cli.main(args + ["--move", "5"]) == 0
    plain = capsys.readouterr().out
    assert "--move only sizes the cost penalty" in plain and "bid/ask cost 0.00" in plain
    assert "e+0" not in explicit.split("hedge: ")[1]  # notionals printed as money, not 1.2e+07


def test_trades_in_another_currency_say_how_to_build_their_market(book, tmp_path, capsys):
    curve = tmp_path / "eur.json"
    assert cli.main(["curve", "build", "--asof", "2026-10-19", "--conventions", "EUR ESTR", "--out", str(curve)]) == 0
    capsys.readouterr()
    assert cli.main(["scenario", "--trades", str(book), "--curve", str(curve), "--shock", "parallel:10"]) == 3
    err = capsys.readouterr().err
    assert "skipped: US1 (swap): a USD swap: the market is EUR ESTR" in err
    assert (
        "(here: --asof with --conventions 'USD SOFR', or --conventions all to price every currency on its own "
        "market, or a curve set built with them)" in err
    )
