# data-snapshots

Daily option-chain snapshots for DerivX, one file per trading session, written by the scheduled workflow
`.github/workflows/collect-data.yml` on the default branch (`derivx snapshots collect`). Append-only: a
stored session is never rewritten.

- `options/index.csv`: one row per file (session, ticker, rows, expiries, sha256, bytes, fetched_at, spot,
  status, path).
- `options/<TICKER>/<YYYY>/<TICKER>_<session>.csv.gz`: the quotes in the compact storage schema of
  `derivx.data.snapshots` (expiries 1-730 days, strikes 0.4-1.6 x spot, Yahoo's implied vol rounded to 1e-5;
  bids, asks, strikes, volumes and times as published).
- `options/<TICKER>/<YYYY>/<TICKER>_<session>.json`: the manifest (fetch status with the real fetch time,
  spot and its time, retries, compaction counts, SHA-256, code and package versions).

Read it with `derivx snapshots sync` (or the ANALYTICS lab's Collected history source).

Source: Yahoo Finance delayed quotes via yfinance. Yahoo's terms of use apply (personal and research use);
the files are kept for reproducible research and are not market data for trading.
