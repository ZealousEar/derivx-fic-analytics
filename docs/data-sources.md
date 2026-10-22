# Data sources

Reference page for DerivX (see [the main README](../README.md) and the [docs index](README.md)). It lists every
bundled dataset with its source and terms, says which inputs are real market data and which are not, and
describes the RATES sidebar's data-source and reporting-currency controls and the environment variables that
govern downloads.

Everything bundled under [`data/examples/`](../data/examples/) is offline-ready: the app and the tests never need
the network. Each file below says whether it is market data. What is not market data is labelled on every chart
and table that uses it.

## Bundled data

| Data | Bundled file | Source and licence / terms | Market data? |
|---|---|---|---|
| US Treasury par yield curve, every business day 2015-2025 and month ends 2008-2014 | `rates_history/ust_par_fred.csv` | Federal Reserve H.15 via FRED (fredgraph.csv, no key). US government data, public domain; cite FRED | real |
| US Treasury par yield curve, 2026 to 10/19 (also the lab's discount curve) | `ust_par_yield_curve_2026.csv` | US Treasury daily par yield curve CSV (home.treasury.gov). Public domain | real |
| SOFR, EFFR, OBFR fixings (2026) | `rates_history/fixings.csv` | Federal Reserve Bank of New York Markets API (no key), with attribution | real |
| €STR fixings; euro area AAA government curve (par and spot) | `rates_history/fixings.csv`, `eur_aaa_ecb*.csv` | ECB Data Portal (no key); reuse allowed with "Source: ECB" | real (daily only from 2026-08-19 in the bundle) |
| SONIA; UK OIS fitted spot curve | `rates_history/fixings.csv`, `gbp_ois_boe.csv` | Bank of England (yield-curve archive; SONIA via FRED), with attribution | real (daily only from 2026-08-19 in the bundle) |
| ICE USD swap rates (DSWP*), 2013 episode windows only | `rates_history/usd_swap_ice.csv` | ICE Benchmark Administration, republished by FRED until 2016-10-28. ICE's terms apply; only a few historical rows are bundled for the 2013 replay | real, ends 2016 |
| ECB euro FX reference rates (USD, GBP, JPY, CHF, CAD, AUD, NZD per EUR): every TARGET business day 2015-01-02 to 2026-10-21, month ends and the replay episodes' windows 2008-2014 | `rates_history/fx_ecb.csv` (187 KB) | ECB Data Portal, dataset EXR (`D.<CCY>.EUR.SP00.A`, values as published); reuse allowed with "Source: ECB". Indicative, not dealable | real (reference rates) |
| SPY option chain, one snapshot, 2026-10-20 after the close | `spy_2026-10-20.csv.gz` (+ `.json` fetch status) | Yahoo Finance delayed quotes via yfinance. Yahoo's terms of use apply (personal and research use); kept for reproducibility | real, one day |
| SPY option chains, one per NYSE session, saved daily by `collect-data.yml` | not bundled: branch `data-snapshots` (`options/SPY/<YYYY>/SPY_<session>.csv.gz`), read with `derivx snapshots sync` | Yahoo Finance via yfinance, as above | real, from the first collected session (none yet) |
| USD SOFR OIS swap quotes | `usd_sofr_ois_example.csv`, `rates_quotes.csv` | none: no free SOFR swap feed exists | **ILLUSTRATIVE** |
| EUR €STR swap quotes | derived at run time | built from the ECB AAA government curve | **PROXY** |
| EURIBOR deposits, FRAs, futures, swaps, 3s6s basis | `src/derivx/rates/multicurve.py` | none: no free EURIBOR quotes | **ILLUSTRATIVE** |
| Swaption and cap vol quotes | `src/derivx/rates/vol_examples.py` | none: no free vol quotes | **ILLUSTRATIVE** |
| FX vol quotes (ATM/RR/BF) and FX deposit rates | `src/derivx/fx/examples.py` | none: no free FX vol quotes | **ILLUSTRATIVE** |
| Walk-forward demo history (option chains) | generated, seeded (`derivx.data.synthetic`) | none | **SYNTHETIC** |
| Rates Vol History demo (dated swaption normal vols) | generated, seeded (`derivx.validation.vol_history`) | none | **SYNTHETIC** |

Paths are relative to `data/examples/` unless they start with `src/`. The fetchers and the history store behind
these files are validated in [tests/test_rates_sources.py](../tests/test_rates_sources.py),
[tests/test_rates_history.py](../tests/test_rates_history.py) and
[tests/test_fx_history.py](../tests/test_fx_history.py) (recorded fixtures, no network); see the "Rates market
data" row in [capabilities.md](capabilities.md).

## Rates data source

RATES sidebar, **DATA SOURCE**:

- **BUNDLED SNAPSHOT** (default): the files above, no network.
- **LIVE (CACHED)**: FRED, NY Fed, ECB and BoE with 5 s timeouts, downloads reused for an hour, falling back to
  the cache and then the snapshot with a visible notice; the slow Treasury CSV downloads in the background.
  Downloads go to `data/cache/` (gitignored) with their SHA-256 and fetch time. When a source is down (FRED was
  serving a maintenance page on 2026-10-21), the page names it and keeps working from the cache or the snapshot.
- **UPLOAD**: your own quotes CSV, an optional Treasury-format par curve and (on SCENARIOS) your own swap
  history. Uploads persist across page switches until cleared.

Quote prefills by currency: **GBP SONIA** uses the real BoE OIS curve converted to par OIS rates
(QuantLib-checked); **EUR €STR** is a labelled PROXY from the ECB AAA government curve; **USD SOFR** is a labelled
ILLUSTRATIVE set fixed at 2026-10-19 levels. The real overnight fixing is always the ON quote. Every curve chart
carries a `SOURCE ... | AS OF ...` badge.

Historical scenarios replay each curve's own history where a free one exists (Treasury par yields; ICE swaps
before 2016-10-28; BoE OIS for GBP; an uploaded swap history) and flag a proxy otherwise. After 2016 USD swap
curves move with Treasuries (swap spreads held), and the per-curve source table on SCENARIOS says so.

## FX spots and the reporting currency

FX spots for the reporting currency (RATES sidebar, **REPORTING CURRENCY**, default USD): **ECB REFERENCE RATES**
(the rates of the valuation date from the bundled history, extended by LIVE (CACHED) downloads; a date with no
fixing, such as a weekend or a TARGET holiday, uses the latest day before it, and a date before 2008-01-31 the
first day held, with a warning naming the day used) or **ENTER RATES** (EURUSD and GBPUSD typed in). Cross rates
(GBPUSD, USDJPY, USDCHF) are derived through the euro. The same history makes FX a risk factor: in VaR and in
historical replay, FX spots move by the ECB rates of the same dates (FX IN VAR = HELD on RESEARCH > VAR, HOLD FX
SPOTS on SCENARIOS, or `--hold-fx` hold them). Only the converted totals (`ALL IN <CCY>`) move with FX: each
position's P&L and the per-currency totals do not. SCENARIOS FX shocks (FX SHOCK EURUSD / GBPUSD, the FX +/-10%
rows of the scenario library) still work and add to a replay's FX moves.

## Rebuilding and extending the history

Rebuild or extend the bundled history: `python tools/build_rates_snapshot.py`. Pull more into the cache with
the `derivx pull` commands in [cli-and-deploy.md](cli-and-deploy.md), for example the whole ICE swap history or
the full BoE daily archive (about 12 MB).

## Environment variables

- `DERIVX_CACHE_DIR` (default `data/cache`) and `DERIVX_OFFLINE=1` (cache only, never the network).
- `FRED_API_KEY` (optional, a free key from FRED). Without it, FRED series come from the keyless
  `fredgraph.csv` endpoint. With it, they come from FRED's JSON API, and `derivx pull fred --vintage DATE` can
  fetch ALFRED vintages (data as known on a date), which need the key. Quotes and whitespace around the key are
  ignored, so a key pasted as `"abc..."` works. The key is sent only to FRED and is never cached or logged (URLs
  are stored redacted). Set it in the environment of the process (your shell, `docker run -e FRED_API_KEY`, or the
  host's environment settings); never commit it.

## See also

- [capabilities.md](capabilities.md): the capability matrix with the validation behind each feature.
- [limitations.md](limitations.md): what the proxy, illustrative and synthetic inputs mean for results.
