"""The ECB FX history (dataset ``FX_ECB``): the EXR parser and fetcher on a
recorded response, the committed sample (every TARGET business day from
2015-01-02, values as published), explicit cross rates, FX spots for any
valuation date from the history store (downloads extend it), the live
refresh request and ``derivx pull ecb-fx``. No network."""

from __future__ import annotations

import datetime as dt
import io
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivx import cli
from derivx.data import rates_history as rh
from derivx.data import rates_sources as rs
from derivx.data.cache import DiskCache
from derivx.rates.conventions import get_calendar
from derivx.risk import reporting as rp

FIXTURE = Path(__file__).parent / "fixtures" / "ecb_exr_2026-10-20.csv"  # a recorded Data Portal response
D = dt.date.fromisoformat


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def ecb(monkeypatch):
    urls = []

    def serve(request, timeout):
        urls.append(request.full_url)
        if "/EXR/" not in request.full_url:
            raise urllib.error.URLError(f"no route for {request.full_url}")
        return _Response(FIXTURE.read_bytes())

    monkeypatch.setattr(rs, "_urlopen", serve)
    return urls


def test_parse_exr_gives_rates_per_euro_in_the_long_format():
    frame = rs.parse_ecb_csv(FIXTURE.read_text())
    assert set(frame["dataset"]) == {"FX_ECB"} and set(frame["measure"]) == {"fx_per_eur"}
    assert set(frame["unit_original"]) == {"fx"} and frame["tenor"].isna().all()
    assert len(frame) == 21 and sorted(set(frame["asof"])) == [D("2026-10-16"), D("2026-10-19"), D("2026-10-20")]
    usd = frame[(frame["series"] == "EXR.D.USD.EUR.SP00.A") & (frame["asof"] == D("2026-10-20"))].iloc[0]
    assert usd["value_original"] == usd["value_decimal"] == 1.1215  # an FX rate is not divided by 100
    store = rh.build_store([frame.assign(origin="test")], "test")
    table = store.fx_table()
    assert list(table.columns) == ["EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
    assert (table["EUR"] == 1.0).all() and table.loc[D("2026-10-20"), "GBP"] == 0.84028
    with pytest.raises(ValueError, match="not positive"):
        rs.parse_ecb_csv("KEY,TIME_PERIOD,OBS_VALUE\nEXR.D.USD.EUR.SP00.A,2026-10-20,0\n")


def test_fetch_builds_the_exr_request_and_caches_it(ecb, tmp_path):
    cache = DiskCache(tmp_path)
    got = rs.ecb_fx(start=D("2026-10-16"), offline=False, cache=cache)
    (url,) = ecb
    assert "/service/data/EXR/D.USD+GBP+JPY+CHF+CAD+AUD+NZD.EUR.SP00.A?" in url and "startPeriod=2026-10-16" in url
    assert len(got.frame) == 21 and got.source == "ECB" and not got.from_cache
    again = rs.ecb_fx(start=D("2026-10-16"), offline=True, cache=cache)
    assert again.from_cache and len(again.frame) == 21 and len(ecb) == 1
    with pytest.raises(ValueError, match="ISO currency codes"):
        rs.ecb_fx(["EUR"], offline=False, cache=cache)
    assert "ECB FX reference rates" in rh.live_requests(D("2026-10-21"))


def test_the_committed_sample_holds_every_target_business_day_since_2015():
    table = rh.bundled_store().fx_table()
    daily = [d for d in table.index if d >= D("2015-01-01")]
    cal = get_calendar("TARGET")
    expected = [d.date() for d in pd.date_range("2015-01-01", "2026-10-21") if cal.is_business_day(d.date())]
    assert daily == expected and len(daily) == 3022
    assert table.index[0] == D("2008-01-31")  # month ends and the episode windows before 2015
    assert D("2013-05-02") in table.index and D("2013-09-05") in table.index  # the 2013 taper tantrum episode
    latest = {"EUR": 1.0, "USD": 1.1234, "GBP": 0.83825, "JPY": 177.36, "CHF": 0.9256, "CAD": 1.5779,
              "AUD": 1.5971, "NZD": 1.9735}  # fmt: skip
    assert table.loc[D("2026-10-21")].to_dict() == latest
    assert rh.bundled_store().coverage(rh.FX_DATASET) == (D("2008-01-31"), D("2026-10-21"))


def test_cross_rates_are_derived_explicitly_through_the_euro():
    table = rh.bundled_store().fx_table()
    gbpusd = rh.fx_cross(table, "GBP", "USD")
    usdjpy = rh.fx_cross(table, "USD", "JPY")
    usdchf = rh.fx_cross(table, "USD", "CHF")
    day = D("2026-10-20")
    assert gbpusd.name == "GBPUSD" and gbpusd[day] == 1.1215 / 0.84028
    assert usdjpy[day] == 177.34 / 1.1215 and usdchf[day] == 0.9237 / 1.1215
    assert rh.fx_cross(table, "EUR", "GBP")[day] == 0.84028
    gbpjpy = rh.fx_cross(table, "GBP", "JPY")
    np.testing.assert_allclose(gbpusd * usdjpy, gbpjpy, rtol=1e-15)  # the triangle closes
    with pytest.raises(ValueError, match="no SEK rate"):
        rh.fx_cross(table, "SEK", "USD")


def test_fx_on_any_valuation_date_and_downloads_extend_the_history():
    store = rh.bundled_store()
    assert store.fx_on(D("2026-10-18"))[0] == D("2026-10-16")  # a Sunday: the Friday before
    assert store.fx_on(D("2007-12-31")) is None
    newer = rs.long_frame([D("2026-10-22")] * 2, "ECB", ["EXR.D.USD.EUR.SP00.A", "EXR.D.GBP.EUR.SP00.A"],
                          [None, None], [1.1300, 0.8400], ["fx", "fx"], ["fx_per_eur"] * 2,
                          ["FX_ECB"] * 2)  # fmt: skip
    partial = rh.build_store([rh.bundled_frame(), newer.assign(origin="downloaded 2026-10-22T15:10:00+00:00")], "x")
    # a day without JPY and CHF is not a complete FX day: spots stay at the last complete one
    assert rp.store_spots(partial, D("2026-10-22")).asof == D("2026-10-21")
    full = rs.long_frame([D("2026-10-22")] * 4, "ECB", [f"EXR.D.{c}.EUR.SP00.A" for c in ("USD", "GBP", "JPY", "CHF")],
                         [None] * 4, [1.1300, 0.8400, 178.0, 0.93], ["fx"] * 4, ["fx_per_eur"] * 4,
                         ["FX_ECB"] * 4)  # fmt: skip
    extended = rh.build_store([rh.bundled_frame(), full.assign(origin="downloaded 2026-10-22T15:10:00+00:00")], "y")
    spots = rp.store_spots(extended, D("2026-10-22"))
    assert spots.asof == D("2026-10-22") and spots.quote("EURUSD") == 1.13
    assert spots.source == "ECB euro reference rates (downloaded 2026-10-22T15:10:00+00:00)"


def test_fx_is_not_a_curve_dataset():
    from derivx.ui import lab_rates, rates_research

    store = rh.bundled_store()
    assert rh.FX_DATASET not in rates_research._datasets(store)
    assert rh.FX_DATASET not in lab_rates._datasets(store)


def test_cli_pull_ecb_fx(ecb, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DERIVX_OFFLINE")
    code = cli.main(["pull", "ecb-fx", "--start", "2026-10-16", "--cache", str(tmp_path), "--show", "2"])
    out = capsys.readouterr().out
    assert code == 0 and "ECB: 21 rows downloaded" in out and "EXR.D.USD.EUR.SP00.A" in out
    assert "last 1.1215 fx (fx_per_eur)" in out
    assert cli.main(["pull", "ecb-fx", "--start", "2026-10-16", "--cache", str(tmp_path), "--offline"]) == 0
    assert "from the cache" in capsys.readouterr().out
    assert cli.main(["pull", "ecb-fx", "--last", "5", "--cache", str(tmp_path / "empty"), "--offline"]) == 1
    assert "offline: no cached ECB data" in capsys.readouterr().err
