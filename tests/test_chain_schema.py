"""Chain schema, upload validation and the bundled real sample (AL-13,
UIE-02, critic-4, ui-engineering-V5, G14 (6))."""

from __future__ import annotations

import gzip
import io

import numpy as np
import pandas as pd
import pytest

from derivx.data.chain_schema import (
    REQUIRED_COLUMNS,
    ChainSchemaError,
    chain_to_csv,
    frames_hash,
    load_chain_file,
    merge_datasets,
    read_chain_file,
    validate_chain,
)
from derivx.data.samples import SAMPLE_CHAIN, load_sample_chain, sample_fetch_status


def _chain(trade_date="2026-10-20", n=6, spot=100.0, **overrides) -> pd.DataFrame:
    strikes = np.linspace(90, 110, n)
    df = pd.DataFrame(
        {
            "trade_date": trade_date,
            "expiry": "2026-11-20",
            "option_type": np.where(strikes < spot, "put", "call"),
            "strike": strikes,
            "bid": 1.0,
            "ask": 1.1,
            "implied_vol": 0.2,
            "underlying_price": spot,
        }
    )
    for k, v in overrides.items():
        df[k] = v
    return df


def _problems(df: pd.DataFrame) -> str:
    with pytest.raises(ChainSchemaError) as info:
        validate_chain(df, "test.csv")
    return str(info.value)


def test_the_bundled_sample_is_one_real_unfiltered_snapshot_with_its_fetch_time():
    ds = load_sample_chain()
    assert SAMPLE_CHAIN.exists() and SAMPLE_CHAIN.stat().st_size < 1_500_000
    assert ds.trade_dates == [pd.Timestamp("2026-10-20")]
    chain = ds.single()
    assert len(chain) == 10_040 and chain["expiry"].nunique() == 31
    assert set(chain["option_type"]) == {"call", "put"}
    assert ds.ticker == "SPY" and ds.fetched_at_utc.startswith("2026-10-20T20:59")
    assert chain["underlying_price"].unique().tolist() == [769.05]
    assert set(chain["source"]) == {"yahoo"}
    status = sample_fetch_status()
    assert status["fetch_status"]["status"] == "ok" and status["fetch_status"]["quote_date"] == "2026-10-20"
    assert status["fetch_status"]["spot_time_utc"] == "2026-10-20T20:00:00Z"  # the 16:00 New York close
    assert "Yahoo" in ds.source and "one snapshot" in ds.source


def test_missing_columns_are_listed_with_units_not_a_keyerror():
    old = pd.DataFrame(
        {"ticker": ["AAPL"], "date": ["2025-10-01"], "option_type": ["call"], "strike": [150], "spot": [172.5]}
    )
    msg = _problems(old)
    for col in ("trade_date", "expiry", "bid", "ask", "implied_vol"):
        assert col in msg
    assert "per-annum decimal" in msg and "Found: ticker, date" in msg


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"strike": ["90", "abc", "100", "105", "110", "115"]}, "strike: 1 non-numeric"),
        ({"strike": [-1.0, 95, 100, 105, 110, 115]}, "strike: 1 value(s) <= 0"),
        ({"underlying_price": 0.0}, "underlying_price"),
        ({"implied_vol": 18.0}, "looks like percent"),
        ({"implied_vol": -0.2}, "negative"),
        ({"option_type": "straddle"}, "not call/put"),
        ({"expiry": "next friday"}, "expiry: 6 value(s) are not dates"),
        ({"bid": -1.0}, "bid: 6 negative"),
        ({"ticker": ["SPY", "QQQ", "SPY", "SPY", "SPY", "SPY"]}, "several underlyings"),
        # numbers that are not YYYYMMDD used to be read as nanoseconds since 1970 (session 1970-01-01)
        ({"trade_date": 46315}, "trade_date: numbers are not dates"),
        ({"expiry": 1.7952e18}, "expiry: numbers are not dates"),
        ({"trade_date": "1900-01-02"}, "trade_date: 6 date(s) outside 1990-01-01 .. 2100-12-31"),
    ],
)
def test_bad_values_raise_a_readable_error(overrides, fragment):
    assert fragment in _problems(_chain(**overrides))


def test_several_spots_in_one_snapshot_are_refused():
    df = _chain()
    df.loc[0, "underlying_price"] = 101.0
    assert "several underlying prices" in _problems(df)


def test_vendor_spellings_are_renamed_and_reported():
    df = _chain().rename(columns={"implied_vol": "impliedVolatility", "expiry": "Expiration", "strike": "Strike"})
    df["openInterest"] = 10
    df["inTheMoney"] = False
    ds = validate_chain(df, "yahoo_like.csv")
    chain = ds.single()
    assert {"implied_vol", "expiry", "strike", "open_interest"} <= set(chain.columns)
    assert ds.renamed["impliedVolatility"] == "implied_vol" and ds.renamed["Expiration"] == "expiry"
    assert any("ignored column(s): inthemoney" in w for w in ds.warnings)
    assert "mid" in chain.columns  # derived from bid/ask when absent


def test_a_multi_day_file_splits_by_trade_date():
    df = pd.concat([_chain("2026-10-19", spot=99.0), _chain("2026-10-20", spot=100.0)])
    ds = validate_chain(df, "two_days.csv")
    assert ds.trade_dates == [pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-20")]
    assert all(len(v) == 6 for v in ds.chains.values())
    with pytest.raises(ChainSchemaError, match="holds 2 trade dates"):
        ds.single()


def test_duplicate_trade_dates_across_files_are_refused_with_both_names():
    a = validate_chain(_chain("2026-10-20"), "monday.csv")
    b = validate_chain(_chain("2026-10-20"), "monday_again.csv")
    with pytest.raises(ChainSchemaError) as info:
        merge_datasets([a, b])
    assert "2026-10-20 is in both monday.csv and monday_again.csv" in str(info.value)
    merged = merge_datasets([a, validate_chain(_chain("2026-10-21"), "tuesday.csv")])
    assert len(merged.chains) == 2 and merged.files == ["monday.csv", "tuesday.csv"]


def test_duplicate_quotes_and_missing_vols_are_warnings_not_silent():
    df = pd.concat([_chain(), _chain().iloc[:2]])
    df.loc[df.index[0], "implied_vol"] = np.nan
    ds = validate_chain(df, "dups.csv")
    assert any("duplicate quote" in w for w in ds.warnings)
    assert any("implied_vol: " in w and "empty" in w for w in ds.warnings)


def test_the_ticker_never_comes_from_the_file_name():
    data = chain_to_csv(_chain())
    ds = load_chain_file("my_saved_chain.csv", data)
    assert ds.ticker is None and ds.source == "upload: my_saved_chain.csv"
    assert load_chain_file("x.csv", data, ticker="spy").ticker == "SPY"


def test_exported_csv_round_trips_to_the_same_quotes():
    ds = load_sample_chain()
    chain = ds.single()
    again = load_chain_file("export.csv", chain_to_csv(chain)).single()
    pd.testing.assert_frame_equal(
        chain[list(REQUIRED_COLUMNS)].reset_index(drop=True), again[list(REQUIRED_COLUMNS)].reset_index(drop=True)
    )
    gz = load_chain_file("export.csv.gz", gzip.compress(chain_to_csv(chain))).single()
    assert len(gz) == len(chain)


def test_parquet_semicolon_and_bad_files():
    df = _chain()
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    assert len(read_chain_file("c.parquet", buf.getvalue())) == 6
    semi = df.to_csv(index=False, sep=";").encode()
    assert list(read_chain_file("c.csv", semi).columns) == list(df.columns)
    with pytest.raises(ChainSchemaError, match="unsupported file type"):
        read_chain_file("c.xlsx", b"PK")
    with pytest.raises(ChainSchemaError, match="could not read the file as parquet"):
        read_chain_file("c.parquet", b"not a parquet file")
    with pytest.raises(ChainSchemaError, match="no rows"):
        validate_chain(pd.DataFrame(columns=list(REQUIRED_COLUMNS)), "empty.csv")


def test_frames_hash_is_content_based():
    a = {pd.Timestamp("2026-10-20"): _chain()}
    b = {pd.Timestamp("2026-10-20"): _chain()}
    assert frames_hash(a) == frames_hash(b)
    b[pd.Timestamp("2026-10-20")].loc[0, "bid"] = 0.99
    assert frames_hash(a) != frames_hash(b)
    assert load_sample_chain().fingerprint == load_sample_chain().fingerprint


def test_a_file_without_volume_loads_with_a_note_on_the_volume_filter():
    ds = validate_chain(_chain(), "no_volume.csv")
    assert any("no volume column" in w and "set min volume to 0" in w for w in ds.warnings)
    ds = validate_chain(_chain(volume=np.nan), "empty_volume.csv")
    assert any("volume is empty on every row" in w for w in ds.warnings)


def test_provenance_counts_distinct_expiries_and_names_row_sources():
    two = pd.concat([_chain("2026-10-19", source="synthetic"), _chain("2026-10-20", source="synthetic")])
    prov = validate_chain(two, "two.csv").provenance()
    assert prov["expiries"] == 1 and prov["slices"] == 2  # one expiry date seen on two sessions
    assert prov["row_sources"] == ["synthetic"]
    assert load_sample_chain().provenance()["row_sources"] == ["yahoo"]


def test_integer_yyyymmdd_dates_are_read_as_dates_not_epoch_nanoseconds():
    """A CSV with 20261020-style dates reaches the validator as int64; it
    used to become 1970-01-01 and every quote looked expired."""

    raw = _chain(trade_date=20261020, expiry=20261120).to_csv(index=False).encode()
    frame = read_chain_file("vendor.csv", raw)
    assert frame["trade_date"].dtype.kind == "i"  # what pandas gives the validator
    ds = validate_chain(frame, "vendor.csv")
    assert ds.trade_dates == [pd.Timestamp("2026-10-20")]
    assert set(ds.single()["expiry"]) == {pd.Timestamp("2026-11-20")}


def test_numeric_last_trade_times_are_dropped_with_a_note_not_read_as_1970():
    ds = validate_chain(_chain(last_trade_date=1_792_526_400), "epoch.csv")
    assert "last_trade_date" not in ds.single().columns
    assert any("last_trade_date holds numbers" in w for w in ds.warnings)
