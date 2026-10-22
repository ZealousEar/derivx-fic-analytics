# DerivX: fixed income and currency derivatives analytics

DerivX is a Python package and a Streamlit app for pricing and risk on rates, FX and equity options, with a
research lab for equity option smiles and for rates curve and vol histories. The rates engine, the FX conventions
and the option pricers are checked against QuantLib 1.44 through committed golden fixtures. Every number on screen
states its units and sign convention, and every input that is not market data is labelled ILLUSTRATIVE, PROXY or
SYNTHETIC.

Hosted demo: https://derivx.streamlit.app/ (Streamlit Community Cloud, deployed from `main`, so it can lag this
README; it may need a minute to wake up).

![RATES > CURVE: the SOFR OIS curve, its quotes and sources, repricing errors and the zero / 3M forward curves](docs/screenshots/rates-curve.png)

---

## Contents

- [What it does](#what-it-does)
- [Capability matrix](#capability-matrix)
- [Data sources](#data-sources)
- [Lab metrics (reproducible)](#lab-metrics-reproducible)
- [Quick start](#quick-start)
- [Command line](#command-line)
- [Tests and validation](#tests-and-validation)
- [Performance](#performance)
- [Deploy](#deploy)
- [Known limitations](#known-limitations)
- [Screenshots](#screenshots)
- [Repository layout](#repository-layout)

---

## What it does

The app has four modes, chosen in the sidebar.

| Mode | What you get |
|---|---|
| **EQUITY** | Black-Scholes-Merton pricer with dividend yield: premium, first- and second-order Greeks in desk units, an implied-vol solver, a spot x vol scenario P&L grid, a risk surface (value or any Greek over spot x vol or spot x time), and a smile: an ILLUSTRATIVE SSVI slice, or an SVI fit to the SPY chain loaded in ANALYTICS, discounted per expiry as in the lab. A model check compares every analytic Greek with a finite difference of the pricer. |
| **FX** | FX options with market conventions for ten G10 pairs: spot and delivery dates on joint settlement calendars, Garman-Kohlhagen with separate vol and rate times, the four delta conventions (spot/forward, premium-adjusted or not), ATM DNS / ATMF / ATM spot, strikes from delta, four premium styles, domestic and foreign rho. Smiles per tenor from ATM, 25D/10D risk reversal and butterfly quotes, fitted with SABR (beta = 1) or vanna-volga, with a butterfly-arbitrage check, and SMILE RISK: the position's vega to each ATM / RR / BF quote (smile rebuilt) and to SABR alpha, rho and nu. Spot defaults to the ECB reference rate. |
| **RATES** | Eight tabs on one market: **CURVE** (OIS and Treasury bootstraps; EUR multi-curve with EURIBOR 3M/6M projection curves), **SWAP**, **SWAPTION**, **CAP-FLOOR** (caps, floors and collars on a term rate or on SOFR / €STR / SONIA compounded in arrears), **BONDS**, **BOOK** (trade blotter), **SCENARIOS** and **RESEARCH** (RV, carry and roll-down, PCA, VaR, and a risk report with vega by bucket). Risk is signed P&L for +1bp on market quotes, with curves rebuilt. The sidebar's **REPORTING CURRENCY** (default USD) adds book totals converted at stated FX spots. |
| **ANALYTICS** | A research lab in seven tabs. Equity options: a real SPY chain (bundled), an upload or a live Yahoo pull, with quote hygiene and per-expiry discounting (MARKET DATA); production SVI calibration with no-arbitrage checks, cross-validation and smile risk (PRODUCTION CALIBRATION); one SSVI surface with Dupire local vol (VOL SURFACE); a walk-forward test against benchmarks, with parameter history and stability (WALK-FORWARD VALIDATION); SVI, SABR and an IV spline compared (MODEL COMPARISON). Rates: curve-forecast walk-forward and curve-model comparison on real curve histories (RATES CURVES); SABR walk-forward, smile models, daily SABR parameters and realised against implied normal vol on dated swaption or caplet vols (RATES VOL HISTORY). Every result downloads as a ZIP with a run manifest. |

Conventions used throughout:

- **Rates risk** (PV01, key-rate ladders, gamma, vega) is the P&L of the position for a +1bp move, signed to the
  holder: a receiver swap has negative PV01. Key-rate buckets bump one curve quote at a time and rebuild the curve
  (central differences); they sum to the parallel PV01.
- **Option Greeks** are for one long option: delta per 1.00 of spot and per +1% spot, gamma as the change in delta
  per +1% spot, vega per vol point, theta per calendar day (plus a one-day repriced theta), rho per +1bp.
- **Units** are in every label: rates in % or bp, vols in % (lognormal) or bp (normal), money in the trade's
  currency.
- **Currencies**: a position's numbers are in its own currency (US Treasuries in USD, everything else in its swap
  curve's currency). Totals are per currency (`TOTAL EUR`, `TOTAL USD`); when a book holds a currency other than
  the reporting one, the whole book is also shown converted (`ALL IN USD`), next to the FX spots and date used.
  Amounts in different currencies are never added unconverted.

---

## Capability matrix

Status: **implemented** (built and validated), **approximate** (works, with a stated simplification),
**planned** (not built). Validation links point at the test that holds the evidence; "golden" means the test
compares against committed QuantLib 1.44 output in `tests/golden/`, and the same tests re-run live QuantLib
checks when QuantLib is installed.

### Rates

| Feature | Status | Validation |
|---|---|---|
| Calendars, schedules, day counts, business-day rules (USD, EUR TARGET, GBP London, FX centres) | implemented | golden: [tests/test_conventions.py](tests/test_conventions.py), [tests/golden/conventions.json](tests/golden/conventions.json) (every date 1990-2080) |
| OIS bootstrap (SOFR, €STR, SONIA): deposits, OIS swaps, SR1/SR3 futures with Hull-White convexity; log-linear DF, linear zero and monotone-convex interpolation | implemented | golden: [tests/test_curves.py](tests/test_curves.py) (QuantLib piecewise curves; every quote reprices, and the CURVE tab shows the error, about 1e-10bp) |
| Library curve API: `load_rates_quotes_from_csv` keeps each row's instrument and units, `build_zero_curve` prices each row as its own instrument and refuses day counts or compounding it cannot apply | implemented | golden: [tests/test_curves.py](tests/test_curves.py) (the bundled `rates_quotes.csv` against QuantLib OIS and swap helpers, within 1e-6bp) |
| US Treasury curve from par yields (bills and par bonds) | implemented | golden: [tests/test_curves.py](tests/test_curves.py), [tests/test_bonds.py](tests/test_bonds.py) |
| Separate discount and projection curves: EUR €STR discounting with EURIBOR 3M/6M projection (deposits, FRAs, futures, IRS, 3s6s basis swaps) | implemented | golden: [tests/test_multicurve.py](tests/test_multicurve.py) ([multicurve.json](tests/golden/multicurve.json)) |
| Vanilla, OIS, forward-start and basis swaps: PV, par rate, annuity, cash flows, PV01, key-rate ladder per curve, hedge notionals | implemented | golden: [tests/test_swaps.py](tests/test_swaps.py) |
| FRAs and rate futures | implemented | golden: [tests/test_multicurve.py](tests/test_multicurve.py) |
| Caps, floors and collars on a term rate (EURIBOR 3M/6M on the multi-curve market; 3M/6M on an OIS curve, flagged as an approximation) or on SOFR, €STR or SONIA compounded in arrears (Lyashenko-Mercurio vol, or an effective vol to the last fixing); Black, shifted Black, Bachelier; flat vol, vol term structure, or caplet vols stripped from cap quotes; gamma, theta, vega by caplet expiry and by quoted cap | implemented | golden: [tests/test_capfloor.py](tests/test_capfloor.py) (compounded caplets equal QuantLib's `OvernightLeg` with `BlackCompoundingOvernightIndexedCouponPricer`, 12 cases), [tests/test_capvol.py](tests/test_capvol.py) (stripping equals QuantLib `OptionletStripper1`) |
| European swaptions (payer, receiver, straddle; Bachelier, Black, shifted Black), implied vols, Greeks | implemented | golden: [tests/test_swaptions.py](tests/test_swaptions.py) (234 cases) |
| Swaption vol cube: SABR per expiry x tenor with alpha from ATM, variance-time ATM interpolation, vega buckets | approximate | [tests/test_swaption_vol.py](tests/test_swaption_vol.py) (smiles equal `ql.shiftedSabrVolatility`; no full SABR parameter cube, Greeks sticky strike) |
| Treasury bonds: clean/dirty, accrued, yield, durations, convexity, DV01, 32nds, Z/G/I/asset-swap spreads | implemented | golden: [tests/test_bonds.py](tests/test_bonds.py) |
| Scenarios: parallel, pivoted twist, butterfly, key-rate, IBOR-OIS and tenor basis, PCA factor, historical episode and any-window replay; vol shocks; FX shocks; P&L explain (delta, gamma, vol, carry, residual) | implemented | [tests/test_scenarios.py](tests/test_scenarios.py), [tests/test_historical_replay.py](tests/test_historical_replay.py) (explain against full revaluation) |
| Historical replay with each curve's own history; per-curve source flags; FX spots move by the ECB reference rates of the episode's dates (HOLD FX SPOTS / `--hold-fx` holds them) | approximate | [tests/test_historical_replay.py](tests/test_historical_replay.py), [tests/test_var_fx.py](tests/test_var_fx.py), [tests/test_rates_fx_var_ui.py](tests/test_rates_fx_var_ui.py) (USD swap curves move with Treasuries after 2016: no free swap history) |
| RV: 2s10s, 5s30s, 2s5s10s and custom spreads/flies, z-scores, percentiles, AR(1) half-life, PCA-neutral weights | implemented | [tests/test_rv.py](tests/test_rv.py) (independent numpy recomputation) |
| Carry and roll-down for bonds and swaps | implemented | [tests/test_carry.py](tests/test_carry.py) (live QuantLib cross-check of bond carry) |
| PCA of daily curve changes, rolling loadings, factor risk | implemented | [tests/test_pca.py](tests/test_pca.py) (77.9 / 12.9 / 5.4% on the committed sample) |
| One-day historical VaR and expected shortfall (delta+gamma or full revaluation), delta-normal cross-check; FX spots as a risk factor of the converted book (ECB reference rates of the same dates; FX-held and FX-only rows, FX in the delta-normal check; FX can be held); a risk report with key-rate PV01, gamma, vega by bucket and VaR | approximate | [tests/test_var.py](tests/test_var.py), [tests/test_var_fx.py](tests/test_var_fx.py), [tests/test_rates_research_ui.py](tests/test_rates_research_ui.py), [tests/test_rates_fx_var_ui.py](tests/test_rates_fx_var_ui.py) (vols held constant; the sensitivity method misses cross-gammas; no FX move on a TARGET holiday) |
| Trade blotter (swaps, basis swaps, FRAs, futures, bonds, caps, floors, swaptions): PV, PV01, ladders, vega matrix, scenarios, VaR, a hedge optimiser that hedges each currency's curves with that currency's instruments | implemented | golden: [tests/test_portfolio.py](tests/test_portfolio.py) ([portfolio.json](tests/golden/portfolio.json)) |
| Multi-currency books: totals per currency, the book converted to USD, EUR or GBP at ECB reference rates or typed FX spots, FX shocks on converted totals, FX moves in VaR and historical replay (UI and CLI) | approximate | [tests/test_reporting.py](tests/test_reporting.py), [tests/test_rates_book_ui.py](tests/test_rates_book_ui.py), [tests/test_cli.py](tests/test_cli.py), [tests/test_var_fx.py](tests/test_var_fx.py) (FX held in curve shocks unless the scenario shocks it; the UI prices one swap-curve currency plus USD Treasuries) |
| Seasoned trades in the blotter (past fixings) | planned | rejected with an explanation today |
| Rates market data: FRED, US Treasury, NY Fed, ECB (curves, €STR, EXR FX reference rates) and BoE fetchers, a dated cache and a history store; BoE spot curves converted to par OIS | implemented | [tests/test_rates_sources.py](tests/test_rates_sources.py), [tests/test_rates_history.py](tests/test_rates_history.py) and [tests/test_fx_history.py](tests/test_fx_history.py) (recorded fixtures, no network); golden: [tests/test_market_quotes.py](tests/test_market_quotes.py) |
| Run manifests and CSV/ZIP exports on every Rates tab; `curve.json` round trip; the `derivx` CLI | implemented | [tests/test_rates_exports.py](tests/test_rates_exports.py), [tests/test_cli.py](tests/test_cli.py) |
| Rates strategy backtester (carry/roll, steepener and PCA signals, DV01-neutral) | planned | - (audit item G18) |
| Meeting-date STIR curve: FOMC / ECB meeting calendar, step forward curve, hike and cut probabilities | planned | - (audit item G19; SR1/SR3 and EURIBOR futures with Hull-White convexity exist as curve inputs and trades) |
| XLSX export | planned | - |

### FX, equity and volatility

| Feature | Status | Validation |
|---|---|---|
| Black-Scholes-Merton price and Greeks (incl. vanna, volga, charm), exact limits at T = 0 and vol = 0 | implemented | golden: [tests/test_bs.py](tests/test_bs.py) (232 QuantLib cases) |
| Implied vol (Black-76, shifted, Bachelier), vectorised, with a reason for every failure | implemented | golden: [tests/test_implied.py](tests/test_implied.py) |
| Per-expiry discounting of option premiums: the US Treasury par curve of the trade date, put-call parity (gated; for European index options), or a flat rate | implemented | [tests/test_discounting.py](tests/test_discounting.py) (exact recovery on European quotes; the American SPY chain fails the parity gates and falls back to the curve) |
| Hagan SABR (lognormal and normal, any beta, shifted), calibration | implemented | golden: [tests/test_sabr.py](tests/test_sabr.py) (matches `ql.sabrVolatility` to 1e-12 relative) |
| SVI in total variance with butterfly, Lee and calendar checks; constrained no-arbitrage fit | implemented | golden: [tests/test_svi.py](tests/test_svi.py), [tests/test_svi_performance.py](tests/test_svi_performance.py) |
| SSVI surface (power law, Gatheral-Jacquier conditions) and Dupire local vol, on the lab's VOL SURFACE tab | approximate | [tests/test_ssvi.py](tests/test_ssvi.py), [tests/test_lab_research_ui.py](tests/test_lab_research_ui.py) (one rho, eta and gamma for every expiry, so it fits each expiry less closely than one SVI per expiry) |
| ECB reference rates for FX spot: a bundled daily history (every TARGET business day 2015-01-02 to 2026-10-21, month ends from 2008), extended by downloads; cross rates derived through the euro | implemented | [tests/test_fx_market_data.py](tests/test_fx_market_data.py), [tests/test_fx_history.py](tests/test_fx_history.py) |
| FX conventions: spot/expiry/delivery dates, joint calendars, forward points | implemented | golden: [tests/test_fx_conventions.py](tests/test_fx_conventions.py) (every day of 2025-2027) |
| FX Garman-Kohlhagen with two times, delta conventions, ATM types, strike from delta | implemented | golden: [tests/test_fx_pricing.py](tests/test_fx_pricing.py) (QuantLib BlackDeltaCalculator) |
| FX smiles from ATM/RR/BF quotes: SABR (beta = 1) or vanna-volga | approximate | [tests/test_fx_smile.py](tests/test_fx_smile.py), [tests/test_fx_performance.py](tests/test_fx_performance.py) (Hagan expansion; SABR cannot fit every RR/BF combination and reports the misfit) |
| FX and equity scenario grids | approximate | [tests/test_pricer_calc.py](tests/test_pricer_calc.py) (sticky strike: the vol shift is flat, the smile is not re-marked) |
| Smile risk: sticky-strike, sticky-moneyness and SABR Bartlett deltas; SVI and SABR parameter vegas; vega by quote bucket through the SVI refit; FX vega to each ATM / RR / BF quote | implemented | [tests/test_smile_risk.py](tests/test_smile_risk.py), [tests/test_smile_vegas.py](tests/test_smile_vegas.py) (bucket vegas match brute-force refits to 0.2%) |

### Analytics lab and app

| Feature | Status | Validation |
|---|---|---|
| Option-chain schema, validated uploads (CSV, CSV.GZ, parquet), lineage and SHA-256 | implemented | [tests/test_chain_schema.py](tests/test_chain_schema.py) |
| Yahoo Finance chains with retries, backoff and a fetch status | implemented | [tests/test_market_data.py](tests/test_market_data.py) (recorded fake, no network) |
| Production SVI calibration with arbitrage repair and k-fold CV against an IV spline; a per-expiry progress bar; identical runs reused across sessions | implemented | [tests/test_calibration_production.py](tests/test_calibration_production.py), [tests/test_analytics_lab.py](tests/test_analytics_lab.py) |
| Walk-forward test against persistence / sticky-moneyness benchmarks, Diebold-Mariano, regimes; SVI and SABR parameter history with robust jump flags | implemented | [tests/test_backtest.py](tests/test_backtest.py), [tests/test_walkforward_tab.py](tests/test_walkforward_tab.py), [tests/test_parameter_stability.py](tests/test_parameter_stability.py) (checked on a seeded SYNTHETIC history; the real history is collected from the merge to `main` on, see [Lab metrics](#lab-metrics-reproducible)) |
| Daily option-chain collection: a scheduled workflow saves each session's chain on the `data-snapshots` branch (compact gzip CSV, a manifest per file, an index with SHA-256); `derivx snapshots sync / list`; the lab's **Collected history** source; `compute_metrics.py --history` | implemented | [tests/test_snapshots.py](tests/test_snapshots.py), [tests/test_collect_workflow.py](tests/test_collect_workflow.py) (the workflow's git steps against a local bare repository), [tests/test_snapshot_sync.py](tests/test_snapshot_sync.py), [tests/test_lab_history_ui.py](tests/test_lab_history_ui.py), [tests/test_compute_metrics_history.py](tests/test_compute_metrics_history.py) (the workflow has not run on GitHub yet) |
| Model comparison (SVI, SABR, IV spline) | implemented | [tests/test_model_comparison.py](tests/test_model_comparison.py), [tests/test_compute_metrics.py](tests/test_compute_metrics.py) |
| Rates curve walk-forward (random walk against Diebold-Li NS AR(1), NS-curve AR(1) and PCA AR(1)) and curve-model comparison (Nelson-Siegel, Diebold-Li, Svensson, spline; leave-one-out) on the bundled Treasury, BoE and ECB histories, the cache or an upload | implemented | [tests/test_curve_forecast.py](tests/test_curve_forecast.py), [tests/test_curve_models.py](tests/test_curve_models.py), [tests/test_lab_research_ui.py](tests/test_lab_research_ui.py) |
| Rates vol history: normal-SABR walk-forward, normal against shifted-lognormal SABR against a spline, daily SABR parameters; realised against implied ATM normal vol | implemented | [tests/test_vol_history.py](tests/test_vol_history.py); golden: [tests/test_realised_implied.py](tests/test_realised_implied.py) (forward swap rates equal QuantLib to 1e-12). No free source publishes dated swaption or cap vols: the tab runs on an upload or a labelled SYNTHETIC demo |
| Every chart zoom-bounded (no zooming to negative tenors or strikes) | implemented | [tests/test_plot_limits.py](tests/test_plot_limits.py) |
| UI smoke tests of every mode and tab (Streamlit AppTest) | implemented | [tests/test_pricer_page.py](tests/test_pricer_page.py), [tests/test_fx_page.py](tests/test_fx_page.py), [tests/test_analytics_lab.py](tests/test_analytics_lab.py), [tests/test_lab_research_ui.py](tests/test_lab_research_ui.py), `tests/test_rates_*_ui.py` |
| Inputs survive a trip through the other modes and a page that raises (the selected tab inside a page is not kept) | implemented | [tests/test_mode_round_trip.py](tests/test_mode_round_trip.py), [tests/test_ui_state.py](tests/test_ui_state.py) |
| Readable dark theme (WCAG AA contrast) | implemented | [tests/test_theme.py](tests/test_theme.py) plus a browser measurement (see [Tests](#tests-and-validation)) |
| Deploy configs (XSRF/CORS on, non-root image, healthcheck) and the keep-alive check | implemented | [tests/test_deploy.py](tests/test_deploy.py), [tests/test_keep_alive.py](tests/test_keep_alive.py) |

---

## Data sources

Everything bundled under `data/examples/` is offline-ready: the app and the tests never need the network. Each
file below says whether it is market data. What is not market data is labelled on every chart and table that
uses it.

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

Rates data source (RATES sidebar, **DATA SOURCE**):

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

FX spots for the reporting currency (RATES sidebar, **REPORTING CURRENCY**, default USD): **ECB REFERENCE RATES**
(the rates of the valuation date from the bundled history, extended by LIVE (CACHED) downloads; a date with no
fixing, such as a weekend or a TARGET holiday, uses the latest day before it, and a date before 2008-01-31 the
first day held, with a warning naming the day used) or **ENTER RATES** (EURUSD and GBPUSD typed in). Cross rates
(GBPUSD, USDJPY, USDCHF) are derived through the euro. The same history makes FX a risk factor: in VaR and in
historical replay, FX spots move by the ECB rates of the same dates (FX IN VAR = HELD on RESEARCH > VAR, HOLD FX
SPOTS on SCENARIOS, or `--hold-fx` hold them). Only the converted totals (`ALL IN <CCY>`) move with FX: each
position's P&L and the per-currency totals do not. SCENARIOS FX shocks (FX SHOCK EURUSD / GBPUSD, the FX +/-10%
rows of the scenario library) still work and add to a replay's FX moves.

Rebuild or extend the bundled history: `python tools/build_rates_snapshot.py`. Pull more into the cache with
the CLI (below), for example the whole ICE swap history or the full BoE daily archive (about 12 MB).

Environment variables:

- `DERIVX_CACHE_DIR` (default `data/cache`) and `DERIVX_OFFLINE=1` (cache only, never the network).
- `FRED_API_KEY` (optional, a free key from FRED). Without it, FRED series come from the keyless
  `fredgraph.csv` endpoint. With it, they come from FRED's JSON API, and `derivx pull fred --vintage DATE` can
  fetch ALFRED vintages (data as known on a date), which need the key. Quotes and whitespace around the key are
  ignored, so a key pasted as `"abc..."` works. The key is sent only to FRED and is never cached or logged (URLs
  are stored redacted). Set it in the environment of the process (your shell, `docker run -e FRED_API_KEY`, or the
  host's environment settings); never commit it.

---

## Lab metrics (reproducible)

Earlier versions of this README quoted SVI/SABR and walk-forward figures (a 71.8% hit rate, 0.052 RMSE, "SVI beats
SABR by 25.3%", "24 trading days") that came from one snapshot saved under 24 dates. They could not be
reproduced and are withdrawn. The numbers below are the only lab metrics this repository claims, and anyone can
regenerate them from the committed sample:

```bash
python tools/compute_metrics.py --outdir docs/metrics/spy_2026-10-20   # about 30 s here; the output is committed
```

Re-run on this revision into a scratch directory, it reproduced `METRICS_SUMMARY.md`, `summary.json`,
`calibration.csv`, `model_scores.csv` and `skipped.csv` byte for byte (`manifest.json` differs only in its run
time).

Data: one real SPY chain (Yahoo, 2026-10-20 after the close, spot 769.05, 10,040 quotes, 31 expiries).
Filters: volume >= 10, mid >= 0.10, relative spread <= 40%, then OTM-only quotes, one per strike, put-call parity
forwards. Premiums are discounted per expiry on the US Treasury par curve of 2026-10-19, the latest published on
or before the trade date (4.10% cc at 2 days to 4.74% at 2.25 years), and vols are re-inverted from mids with
Black-76. 29 of 31 expiries are scored (same-day and 1-day expiries skipped).

| Model | Out-of-sample 5-fold CV RMSE, median over expiries (vol bp) | Mean | Lowest on |
|---|---|---|---|
| IV spline (natural cubic in ln K/F) | 1.7 | 5.8 | 28 of 29 |
| SVI (raw, total variance) | 5.1 | 9.3 | 1 of 29 |
| SABR (beta = 1) | 16.2 | 18.3 | 0 of 29 |

- SVI beats SABR on 25 of 29 expiries; the spline beats SVI on 28 of 29. The spline interpolates dense strikes,
  so it is scored out of sample only and is a strong baseline.
- Production SVI with arbitrage repair: median in-sample RMSE 6.0 vol bp (max 42.0). Butterfly and Lee checks are
  clean on 29 of 29 expiries; the calendar check is clean on 25 of 28 adjacent pairs (fails at 2027-03-19,
  2027-06-30 and 2027-12-17). Calendar arbitrage is reported, not enforced across expiries.
- SPY options are American, and the vols are inverted with European Black-76. At a real discount rate the
  early-exercise premium of out-of-the-money puts, largest at long expiries, reads as extra put vol below the
  forward: a kink no smile fits, which drives the larger long-dated errors. (Earlier metrics used r = 0, which hid
  that premium but biased every vol 30-50 vol bp low at one year.)
- These are single-snapshot fit metrics, not forecasts. Full tables: [docs/metrics/spy_2026-10-20](docs/metrics/spy_2026-10-20/METRICS_SUMMARY.md),
  with a manifest recording the commit, packages, inputs, discount curve and data hash. `--discount
  curve|parity|rate` (default `curve`) and `--rate` choose the discounting.
- SABR parameter recovery from exact quotes: `pytest tests/test_sabr.py -k recovers` (alpha and nu to 1e-5
  relative, rho to 1e-5 absolute, F from 1.08 to 15,800, T from 1W to 5Y).

**Walk-forward results on real data do not exist yet: the history is collected from the merge to `main` on.**
Yahoo serves only the current chain, so a real history has to be saved one session at a time.
`.github/workflows/collect-data.yml` does that:

- **When.** Cron `35 21 * * 1-5` (UTC), which is 17:35 New York in summer (EDT) and 16:35 in winter (EST), after the
  16:15 options close in both, plus a manual run (inputs `tickers`, default `SPY`, and `dry_run`). The collector
  (`derivx snapshots collect`) takes the session that closed 10 minutes to 12 hours before the run. On a weekend,
  an NYSE holiday, during a session or for a session already stored it fetches nothing and the run commits
  nothing. GitHub runs schedules only from the default branch, so collection starts once the workflow is on
  `main`.
- **Where.** The orphan branch `data-snapshots`, created by the first run that saves a session, append-only:
  `options/<TICKER>/<YYYY>/<TICKER>_<session>.csv.gz`, a JSON manifest per file (fetch status, limits, rows dropped
  by each rule, SHA-256) and `options/index.csv` (session, ticker, rows, expiries, sha256, bytes, fetched_at, spot,
  status, path). Commits are made by `github-actions[bot]` and pushed with `GITHUB_TOKEN` (`contents: write`, one
  run at a time). A rejected push is rebuilt on the branch's new tip, up to 3 attempts.
- **Failures.** Every Yahoo request retries with backoff, and the whole chain is fetched up to 3 times (1 and
  then 2 minutes apart); a fetch that raises is retried the same way and the other tickers are still collected. A
  partial chain (the summary names the expiries that failed), or the chain of another session, is never written,
  and a stored session is never rewritten. The job summary says what happened to each ticker. The job fails only
  when a trading day produced nothing for a ticker, so a missed day shows red. A run after midnight New York
  still requests expiries from 1 day after the session, not after the run's own date.
- **Schema and limits.** 14 columns (trade_date, expiry, option_type, strike, bid, ask, implied_vol (Yahoo's, 5
  decimals), underlying_price, volume, last_trade_date, fetched_at_utc, spot_time_utc, ticker, source). Expiries
  are kept from 1 to 730 calendar days and strikes from 0.4 to 1.6 times spot (`--max-days`, `--moneyness`). The
  gzip header has no timestamp, so equal quotes give equal bytes.
- **Sizes, measured on real chains.** The bundled 2026-10-20 chain is 357,609 bytes in the 18-column chain schema,
  181,472 compact with every quote, and 152,576 within the limits (8,352 of 10,040 quotes and 28 of 31 expiries
  kept: 954 quotes outside the expiry window, 734 outside the moneyness band). A dry run of the collector on the
  2026-10-21 session (22:36 UTC) kept 8,102 of 9,543 quotes over 27 expiries in 147,866 bytes. It took 31 Yahoo
  requests with 1 retry, 20.2 s of fetching and 21.5 s for the whole command. That is about 37 MB a year for SPY
  (252 sessions).

Using the history:

```bash
derivx snapshots sync                    # index + files from raw.githubusercontent.com into data/cache/snapshots/
derivx snapshots sync --offline          # the cached copy only (--repo OWNER/NAME, --branch, --tickers)
derivx snapshots list --sessions         # sessions, first, last, missing NYSE trading days (no network)
derivx snapshots list --local ../data-snapshots   # a local checkout (git worktree add ../data-snapshots data-snapshots)
python tools/compute_metrics.py --history --ticker SPY --outdir data/metrics/spy_history
python tools/compute_metrics.py --store ../data-snapshots --ticker SPY --first 2026-11-02 --horizon 1 --outdir ...
```

`sync` checks every file's SHA-256 and its place in the layout (`options/<TICKER>/<YYYY>/<TICKER>_<session>`)
against the index before keeping it, and reuses files it already has (a cached file whose manifest is missing is
downloaded again). A transfer cut short fails only that file, which the next sync completes. The repo
and branch default to `$DERIVX_SNAPSHOT_REPO` / `$DERIVX_SNAPSHOT_BRANCH`, else this repository and
`data-snapshots`. In the lab, WALK-FORWARD VALIDATION's default source is **Collected history**: the synced copy
(a SYNC NOW button fetches it) or, with `$DERIVX_SNAPSHOT_DIR` set, a local checkout of the branch. It shows
coverage and provenance (sessions, first and last, missing trading days, quotes and size, source and sync time,
and what the rows are, read from each file's manifest: Yahoo's delayed quotes, or SYNTHETIC rows named and edged
as such; the run's banner and manifest add the index SHA-256). SESSIONS picks the most recent 10 to 250 sessions
or all of them (default 30): a run takes about 4 s a session on a real SPY chain at the default settings (12
sessions: 44 s), so a year of history would take about 15 minutes in the browser; `compute_metrics.py --history
--first ... --last ...` runs any range offline. Until the chosen sessions give 9 origin days at the chosen
horizon (three per vol-regime tercile; 10 consecutive sessions at 1 day), it says how many there are and how many
more are needed, and the run stays disabled. MARKET DATA loads any collected session as the active dataset, so
PRODUCTION CALIBRATION, VOL SURFACE and MODEL COMPARISON run on it.
`compute_metrics.py --history` (or `--store DIR`) writes the walk-forward summary, daily and regime tables,
`sessions.csv` with every file's SHA-256, `summary.json`, `METRICS_SUMMARY.md` and a manifest. A rerun on the
same sessions reproduces every file but the manifest byte for byte, also from another checkout directory or a
re-synced cache: where and when the store was checked out or synced is recorded in the manifest only. Run on a
store of the two real sessions held here (2026-10-20 bundled and 2026-10-21 from the dry run), it reports
`walk-forward: not run (2 session(s) give 1 origin day(s) ... collect at least 8 more trading day(s))`. A session
more than 7 days after the committed Treasury curves (which end 2026-10-19) is discounted at the flat `--rate`
unless `--with-cache` adds the cached Treasury curves, and the summary names the curve or rate of every trade
date.

`tools/download_spy_chains.py` still saves one chain by hand (`--outdir` in the full schema, or `--store DIR` in
the compact store layout). It refuses past dates, keeps an existing session file unless `--replace`, and exits 0
(saved), 1 (nothing saved) or 3 (partial: with `--store` nothing is written). `data/**/*.parquet` and
`data/metrics/` are gitignored.

Walk-forward method: each origin is scored against the snapshot exactly N NYSE trading days later, matching quotes
on (expiry, strike, option type); the calibration window pools the last W snapshots. Predictors: SVI sticky
strike against persistence, and SVI sticky moneyness against raw sticky moneyness. Scores: pooled RMSE (vol bp),
skill = 1 - RMSE / benchmark, days won with a Wilson 95% interval, a day-clustered Diebold-Mariano t with the
Harvey-Leybourne-Newbold correction, a noise floor (RMS half bid/ask in vol) and regimes (terciles of a 30-day
ATM vol).

---

## Quick start

Python 3.10 to 3.13 (the CI matrix covers all four; Docker, the devcontainer and `.python-version` use 3.11).

```bash
git clone https://github.com/ZealousEar/derivx-fic-analytics.git
cd derivx-fic-analytics
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt          # runtime + the package (editable)
streamlit run src/derivx/ui/streamlit_app.py
```

Open http://localhost:8501. The app starts offline on the bundled data; switch RATES > DATA SOURCE to
LIVE (CACHED) for current FRED, NY Fed, ECB and BoE data.

Development install (tests, the QuantLib oracle, linters) and notebooks:

```bash
pip install -e ".[dev]"                  # pytest, QuantLib 1.44, ruff 0.17.1, mypy 2.4.0, black
pip install -e ".[dev,notebooks]"        # + matplotlib, ipykernel, nbmake
pytest                                   # whole suite, offline
pytest --nbmake notebooks                # execute the three notebooks
ruff check . && ruff format --check .
mypy                                     # src/derivx, with the lenient config in pyproject.toml
```

Dependency policy: `pyproject.toml` holds the supported ranges and `requirements.txt` repeats them
(`tests/test_deploy.py` checks they agree). `requirements-lock.txt` pins the resolution the Docker image and the
CI type check are built from; regenerate it with
`uv pip compile pyproject.toml --python-version 3.11 --python-platform linux -o requirements-lock.txt --upgrade`.
numpy is capped below 2.5 and pandas below 3 until they are supported.

Notebooks (`notebooks/`, executed with outputs committed): 01 SVI on a synthetic slice and then on the real
bundled SPY chain; 02 SABR across expiries with beta = 1; 03 rates: a SOFR OIS curve from illustrative quotes and
the real Treasury curve, a 5Y normal-vol cap with put-call parity, and a PV01 whose sign matches the +/-100bp P&L.

---

## Command line

`pip install -e .` installs a `derivx` command (`python -m derivx` works too). Every command except `pull` works
offline on the bundled data, and `risk` and `scenario` write a run manifest with `--manifest`.

```bash
# market data into the cache (data/cache or $DERIVX_CACHE_DIR)
derivx pull fred --ids DGS2,DGS10 --start 2015-01-01
derivx pull fred --ids ust,ice --start 2000-01-01      # Treasury CMT and the ICE swap history (to 2016-10-28)
derivx pull fred --ids DGS10 --vintage 2020-01-02      # ALFRED: as known on that date (needs $FRED_API_KEY)
derivx pull treasury --year 2026                       # can be slow
derivx pull nyfed --rate sofr --last 30
derivx pull ecb-estr
derivx pull ecb-yc --curve AAA --measure par
derivx pull ecb-fx --start 2026-10-01                  # ECB reference rates per EUR [--currencies USD,GBP --last N]
derivx pull boe-ois --archive daily                    # about 12 MB
derivx pull latest                                     # what LIVE (CACHED) uses (ECB FX included)
#   options: --offline, --out file.csv|parquet, --max-age, --cache, --show N, --timeout

# the daily option-chain history on the data-snapshots branch (see Lab metrics)
derivx snapshots sync [--offline] [--repo OWNER/NAME] [--branch data-snapshots] [--tickers SPY]
derivx snapshots list [--sessions] [--local DIR]
derivx snapshots collect --staging DIR [--store DIR] [--tickers SPY] [--dry-run]   # what collect-data.yml runs
derivx snapshots merge STAGING STORE && derivx snapshots index --store STORE --check

# curves
derivx curve build --asof 2026-10-19 --interp loglinear --out curve.json   # [--quotes q.csv] [--conventions "EUR ESTR"]
derivx curve show --curve curve.json [--nodes nodes.csv --grid grid.csv]   # rebuilds and checks every node (1e-12)

# a trade blotter: example, validation, risk, hedges, scenarios
derivx book example --currency USD > trades.csv
derivx book check --trades trades.csv
derivx risk --trades trades.csv --curve curve.json --hedge 2Y,5Y,10Y,30Y [--bid-ask 0.25 [--move 5]] \
            [--illustrative-vols] [--out risk.csv] [--ladder ladder.csv] [--manifest run.json]
derivx scenario --trades trades.csv --asof 2026-10-19 --illustrative-vols \
            --shock parallel:+25 --shock twist:5Y:-10:+10 --shock pca:1:2 --shock historical:2022-hiking-cycle \
            [--explain explain.csv] [--out grid.csv]

# a book in several currencies (mixed.csv: e.g. the USD, EUR and GBP example books in one file),
# each currency on its own market, with totals converted
derivx risk --trades mixed.csv --asof 2026-10-19 --conventions all --illustrative-vols --report-ccy EUR \
            [--fx EURUSD=1.10,GBPUSD=1.30]
derivx scenario --trades mixed.csv --asof 2026-10-19 --conventions all --illustrative-vols \
            --shock parallel:+25 --shock fx:EURUSD=+10,GBPUSD=-5 [--fx-move EURUSD=+2]
derivx scenario --trades mixed.csv --asof 2026-10-19 --conventions all --illustrative-vols \
            --shock historical:2022-hiking-cycle --shock window:2025-04-01:2025-04-30 [--hold-fx]
```

Every command above was run on this revision. On 2026-10-21 FRED (api.stlouisfed.org and fred.stlouisfed.org)
served a maintenance page, so both `pull fred` lines exited 1 with `FRED: unexpected response` (with and without a
key), the `--vintage` line run without a key exited 2 with `FRED vintages (ALFRED) need a free API key`, and
`pull latest` exited 1 after naming the two FRED series it left to the cache or the snapshot. Every other pull
succeeded (the BoE daily archive in about 11 s). The `scenario` example needs `--illustrative-vols` (or vols) for
its cap and swaptions; without it they are skipped and it exits 3. The commands added with the collection and FX
history were run on this revision too. `pull ecb-fx --start 2026-10-01` downloaded 105 rows (15 days x 7
currencies; EURUSD 1.1234 on 2026-10-21). `snapshots sync` exited 1 with `the index could not be downloaded (HTTP
404 from raw.githubusercontent.com) and nothing is cached`, because the branch does not exist until the workflow
first runs on `main`. `snapshots list --local` on the two-session store printed `SPY: 2 session(s) 2026-10-20 ..
2026-10-21; no missing trading days; 16,454 quotes, 0.30 MB`. On the USD + EUR + GBP example books, the 2022
hiking cycle replay moved EURUSD -14.31% and GBPUSD -17.82%. It gave `ALL IN USD` 1,160,838 with those moves and
1,367,277 with `--hold-fx`, with `TOTAL USD`, `TOTAL EUR` and `TOTAL GBP` identical in both runs (each about 8 s).

Shock specs (`derivx scenario --help`): `parallel:BP`, `twist:PIVOT:SHORT_BP:LONG_BP`,
`fly:BODY:BODY_BP[:WING_BP[:LEFT:RIGHT]]`, `keyrate:2Y=+10,10Y=-5`, `pca:FACTOR[:N_SIGMA[:DAYS]]`,
`historical:NAME[:SCALE]` (episodes: 2013 taper tantrum, 2020 dash for cash, 2022 hiking cycle, Sept 2022 gilt
crisis week, 2023 SVB), `window:START:END[:SCALE]`, `basis:BP`, `tenorbasis:BP`, `fx:EURUSD=+10[,GBPUSD=-5]`
(% moves of FX spots), `library`.

Currencies: every trade's numbers are in its currency, with a `TOTAL <CCY>` per currency. `--report-ccy
{USD,EUR,GBP}` (default USD) adds `ALL IN <CCY>`, the book converted at the ECB reference rates (the bundled
history plus any `pull ecb-fx` in the cache) of the latest day on or before the valuation date (the first day
held, for an earlier date; a note names the day when it is another one) or at `--fx` spots.
`--conventions all` prices each currency's trades on its own market (USD SOFR, EUR ESTR, GBP SONIA; Treasuries on
the US Treasury curve). FX moves change only the converted totals. `historical:` and `window:` replays move spots
by the ECB rates of the episode's dates (`--hold-fx` holds them), `fx:` shocks move spots per scenario, and
`--fx-move` adds its moves to every scenario, on top of a shock's own move of the same pair (an `fx:EURUSD=+10`
shock with `--fx-move EURUSD=+2` moves EURUSD +12%), and is named in each scenario label. The output lists the FX
moves of every scenario that has any. A total move of -100% or less is refused. Spots are held otherwise.

Caps and swaptions in a blotter need vols: a `vol` column per trade, or `--swaption-vols` / `--cap-vols` files, or
`--illustrative-vols` (labelled). Without them those trades are skipped with the reason. Hedges: `--hedge`
solves each currency's curves with that currency's instruments; with `--bid-ask` the optimiser weighs each hedge's
cost against the risk it removes for curve moves of `--move` bp, and it returns no hedge (residual = unhedged risk,
zero cost) when none is worth its bid/ask.

Exit codes: 0 ok; 1 a data source failed (`pull latest` also when any one source failed, after falling back to the
cache or the snapshot); 2 bad arguments, missing files, a malformed curve JSON or a date with no curve data; 3
blotter rows with errors or trades that could not be priced (the rest are).

`curve.json` is the `derivx.curve_set` v1 format: as-of date, conventions, interpolation, convexity settings,
fixings and every curve's quotes. Blotter CSVs accept comma, semicolon, tab or bar delimiters; commas inside
numbers are thousands separators only (`1,000,000`), and a decimal comma is refused with a message.

Other tools: `tools/gen_golden.py` with `tools/golden/` (regenerate the QuantLib fixtures),
`tools/compute_metrics.py` (lab metrics; `--history` on the collected snapshots) and `tools/download_spy_chains.py`
(one chain by hand), `tools/build_rates_snapshot.py` (bundled rates data and the ECB FX history;
`tools/build_historical_shocks.py` is a shim that runs it),
`tools/record_rates_fixtures.py` (parser fixtures), `tools/gen_fx_illustrative_quotes.py` and
`tools/gen_rates_vol_illustrative.py` (print the ILLUSTRATIVE FX and rates vol quotes), `tools/keep_alive/` (the
hosted app's wake-up check).

---

## Tests and validation

Test run on this revision: `pytest -W error::DeprecationWarning -W error::RuntimeWarning` collects 1,540 tests
from 79 files and took about 22 minutes here on 4 cores; run single files while iterating.

| Environment | Result |
|---|---|
| Python 3.11.17 with the dev extra: numpy 1.26.4, pandas 2.1.4, SciPy 1.17.1, Streamlit 1.66.0, Plotly 7.1.0, yfinance 0.2.66, pyarrow 13.0.0 (older than the declared lower bound 14), QuantLib 1.44 (so the live QuantLib checks run); `-W error::DeprecationWarning -W error::RuntimeWarning` | 1,540 passed, 0 skipped, 0 failed |
| `pytest --nbmake notebooks` (Python 3.11.17, numpy 2.4.6, pandas 2.3.3, SciPy 1.17.1; run on the previous revision, and the notebooks use none of the modules changed since) | 3 passed |
| `ruff check .` and `ruff format --check .` (ruff 0.17.1, the CI pin) | clean (219 files formatted) |
| `mypy` (2.4.0, the CI pin, on the local versions above) | clean (no issues in 105 source files) |
| `actionlint` 1.7.12 with shellcheck 0.11.0 on `.github/workflows/*` | clean |

- **QuantLib golden fixtures.** `tools/gen_golden.py` with the modules in `tools/golden/` writes the 15 files
  `tests/golden/<area>.json` from QuantLib 1.44 (calendars, curves, multi-curve, swaps, caps/floors including
  compounded RFR caplets, cap vol stripping, swaptions, bonds, Black-Scholes, Black IR, implied vols, SABR/SVI, FX
  conventions and pricing, the rates market-data conversions, the example trade book, forward swap rates for
  realised vol). The tests read the committed JSON, so the suite needs no QuantLib; with QuantLib installed
  (`pip install -e ".[dev]"`) the same tests also run randomised live QuantLib checks. Regenerate with
  `python tools/gen_golden.py` (all) or `python tools/gen_golden.py swaps capvol` (some).
- **Property and regression tests**: quotes reprice, discount factors positive, cap - floor = swap, collar = cap -
  floor, payer - receiver = forward swap, key-rate buckets sum to PV01, P&L explain against full revaluation,
  put-call parity on forwards, implied-vol round trips, SABR parameter recovery, no-arbitrage checks that fail
  where they should, converted totals against hand conversion.
- **UI tests**: Streamlit AppTest renders every mode and every Rates and Analytics tab with default and edge
  inputs, and takes inputs through a round trip of the other modes; `tests/test_plot_limits.py` asserts every
  chart is zoom-bounded; `tests/test_theme.py` checks the theme's text colours and heatmap scales against WCAG AA.
- **Hermetic**: an autouse fixture blocks network access, points the cache at a temporary directory and removes
  `FRED_API_KEY`; parsers are tested on recorded fixtures under `tests/fixtures/`.
- **Deploy**: `tests/test_deploy.py` checks the Docker, Render, Procfile and devcontainer configs (XSRF and CORS
  on, non-root image, healthcheck), the CI pins and `-W` flags, and starts a live server on the committed config to
  confirm it refuses a token-less upload and sends no wildcard CORS header.

CI (`.github/workflows/ci.yaml`) runs on pushes and pull requests to `main` or `master` and by hand; it has not run
on this revision, which has not been pushed or opened as a pull request. Its jobs: **lint** (ruff 0.17.1 check and
format over the whole repository); **typecheck** (mypy 2.4.0 over `src/derivx`, installed from
`requirements-lock.txt`; a lenient start: the 11 error codes still off and an `attr-defined` baseline of 18 rates,
CLI and rates-UI modules are listed in `[tool.mypy]`); **test**
on Python 3.10, 3.11, 3.12 and 3.13 with the newest versions the ranges allow and the dev extra (live QuantLib
checks), deprecation and runtime warnings as errors; **oldest**: Python 3.10 with every direct dependency at its
lower bound, same flags; **notebooks** with nbmake; and **docker**: builds the test stage (the UI tests, the swap,
cap, bond and scenario tests and a live-server check, run inside the image) and the runtime image, waits for the
healthcheck and checks XSRF and CORS on the running container.

Browser checks (not in CI): on the previous revision every mode, tab and sub-tab was driven in Chromium with
Playwright at 390, 1280 and 1920 px (63 views), plus the timed runs under [Performance](#performance). No page
exceptions, console errors, Streamlit exception blocks, error alerts or horizontal overflow; every chart declares
zoom bounds; no text node below WCAG AA contrast. On this revision the changed views were driven at 1280 and 390
px (12 each). ANALYTICS > WALK-FORWARD VALIDATION was checked with Collected history in four states: not synced,
a real SYNC NOW (the expected 404 message, 1.2 s), two real sessions (held back) and 12 SYNTHETIC sessions (run,
22-23 s). Also checked were MARKET DATA loading a collected session and PRODUCTION CALIBRATION on it, and RATES >
BOOK and RESEARCH > VAR on a euro book reported in USD with FX moving and held. They showed the same absence of
exceptions, console or page errors, error alerts (apart from the intended sync failure) and horizontal overflow.
After the review fixes, 10 views per width were driven again at 1280 and 390 px: 14 SYNTHETIC sessions (labelled
SYNTHETIC with the orange edge), SESSIONS = most recent 10 and its run (13 s), a real collected SPY session
(labelled as Yahoo's delayed quotes, held back, then loaded in MARKET DATA), a server with nothing synced, and
RATES > BOOK and RESEARCH > VAR with FX; none showed an exception, a console or page error, an error alert or
horizontal overflow.

---

## Performance

Measured in headless Chromium at 1280 px on a fresh local server on this revision (bundled data, empty download
cache; seconds, including the websocket round trip and the browser's render; 4 cores, otherwise idle).

| Mode | First visit (cold caches) | Rerun, same inputs | Switching back later |
|---|---|---|---|
| EQUITY | 7.8-7.9 (page load: the script finishes in about 3 s, the rest is the browser loading Plotly and drawing the charts, the 3D surface in software WebGL) | 0.4-0.5 | 0.3-0.4 |
| FX | 1.9-2.2 | 0.5-0.6 | 0.9-2.1 |
| RATES | 8.5-8.7 | 2.1-2.4 | 6.4 |
| ANALYTICS | 3.0-3.6 | 0.4-0.5 | 1.2-1.5 |

Each time runs from the click (or Streamlit's `r` rerun key) until the app is idle: no script running, no stale
element and no loading placeholder for a full second, that second not counted (a chart can still be finishing its
first draw then). Ranges are over two or three fresh servers.

| Action | Seconds |
|---|---|
| FX: switch the currency pair (EUR/JPY, USD/CAD, GBP/USD) | 1.7-1.9 |
| ANALYTICS > PRODUCTION CALIBRATION: the two default expiries | 2.0-2.2 |
| ANALYTICS > PRODUCTION CALIBRATION: all 31 listed expiries (the 29 at least 2 days out are fitted) | 19.2-19.5 |
| the same calibration again (reused from the shared store) | 0.7-0.8 |
| ANALYTICS > VOL SURFACE: fit the SSVI surface | 7.8-8.9 |
| ANALYTICS > WALK-FORWARD VALIDATION: generate the SYNTHETIC history, then run the test | 1.5-1.7, then 11.6-12.0 |
| ANALYTICS > MODEL COMPARISON: compare models on every expiry | 17.5-18.0 |
| ANALYTICS > RATES CURVES: curve walk-forward; compare curve models | 2.0-2.1; 8.0-8.8 |
| RATES > RESEARCH > VAR: full revaluation (500 scenarios) | 11.1-11.5 |
| RATES > DATA SOURCE > LIVE (CACHED), first time, empty cache (FRED was down and failed fast) | 11.1 |
| `python tools/compute_metrics.py` (command line) | 28.6 |

- RATES runs all eight tabs on every rerun (SCENARIOS and RESEARCH need the positions of the others).
- Production calibration shows a per-expiry progress bar. A calibration identical to one already run (same data,
  filters, discount factors and settings) is reused at once from a store of the last 16, shared across sessions.
- Inputs, including edits in the RATES data editors (curve quotes, book, cap-floor, swaption), survive a trip
  through the other modes and a page that raises; the selected tab inside a page does not.

---

## Deploy

All deploy paths read the committed `.streamlit/config.toml`, which keeps XSRF and CORS protection on, sets the
dark theme, and turns off Streamlit's widget-state duplication warning: inputs that survive mode switches are
written back on purpose, and without the setting the server logs a long, harmless stack trace once on the first
mode round trip. Secrets go in `.streamlit/secrets.toml`, which git and `.dockerignore` keep out; the app itself
reads no secrets file, only environment variables (`FRED_API_KEY`, `DERIVX_CACHE_DIR`, `DERIVX_OFFLINE`, and
`DERIVX_GIT_SHA`, which the Docker image sets from its `GIT_SHA` build argument for run manifests).

**Docker**

```bash
docker build -t derivx --build-arg GIT_SHA=$(git rev-parse HEAD) .
docker run --rm -p 8501:8501 derivx          # http://localhost:8501; add -e FRED_API_KEY=... to use the FRED API
docker build --target test .                  # the UI tests, pricer tests and a live-server check, in the image
```

The image runs as a non-root user, is healthy only when `/_stcore/health` answers 200 (the check bypasses any
HTTP(S)_PROXY passed in), takes its port from `-e STREAMLIT_SERVER_PORT=...`, writes downloads to a cache in the
user's home, and shows viewers only the exception type (no tracebacks or server paths). It needs `data/examples/`
and `pyproject.toml` next to `src/`.

**Render**: `render.yaml` builds the Dockerfile with health check path `/_stcore/health`.

**Heroku-style hosts**: the `Procfile` runs
`streamlit run src/derivx/ui/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0` with error details
reduced to the exception type.

**Hugging Face Spaces**: create a Docker Space from this repository and set `app_port: 8501` in the Space's README
front matter.

**Streamlit Community Cloud**: main file `src/derivx/ui/streamlit_app.py`. It installs `requirements.txt`
(including `-e .`) and reads the committed config. Community Cloud controls its own error-detail setting. It pulls
each new commit into the running app without restarting it; `derivx.ui.code_reload` then re-imports the package's
changed modules on the next run (otherwise a new page importing a name from a cached old module fails with
`ImportError`). A commit that changes `requirements.txt` still needs **Manage app → Reboot app** so the new
dependencies are installed. Pick Python 3.10-3.13 in the app's settings.

**Devcontainer / Codespaces**: the app opens in a real browser tab; the editor's preview pane can block the XSRF
cookie that file uploads need.

**Keep-alive**: `.github/workflows/keep-alive.yml` opens the Community Cloud app in headless Chrome every six hours,
wakes it if it is asleep, and fails unless the sidebar heading NAVIGATION renders with no Streamlit exception. It
makes no commits and does not ingest data. Data collection is a separate workflow, `collect-data.yml` (see
[Lab metrics](#lab-metrics-reproducible)), which commits only to the `data-snapshots` branch.

---

## Known limitations

Not built
- G18: no rates strategy backtester (carry/roll, steepener or PCA signals, DV01-neutral sizing).
- G19: no meeting-date STIR curve (no FOMC / ECB meeting calendar, step forward curve or hike/cut probabilities);
  SR1/SR3 and EURIBOR futures are curve inputs and trades only.
- Seasoned trades (past fixings) are rejected; no XLSX export (openpyxl is not a dependency).

Data
- No free SOFR or €STR swap quotes, EURIBOR quotes, swaption/cap vols or FX vol quotes exist. Those defaults are
  ILLUSTRATIVE or PROXY and labelled; paste or upload real quotes to price real trades. The same flat cap quotes
  are stripped as term or compounded caplets, depending on the CAP-FLOOR tab's rate.
- No free dated swaption or cap vol history exists: RATES VOL HISTORY runs on an upload or the SYNTHETIC demo, and
  realised against implied vol needs a vol history's own forwards or an uploaded swap curve for a like-for-like
  USD comparison (the Treasury curve is offered as a labelled PROXY without swap-spread vol).
- USD swap curves in historical replay, VaR and PCA shocks move with Treasury changes after 2016 (swap spreads
  held, flagged). Swap spreads appear only with an uploaded swap history or the ICE series (to 2016-10-28).
- The bundled ECB and BoE histories are daily only from 2026-08-19 (month ends before), so EUR/GBP RV and PCA in
  BUNDLED mode run on about 43 daily changes, with a warning, and the lab's curve walk-forward and realised vol
  need a longer step limit for them. LIVE mode or the CLI pulls more.
- The bundled ECB FX history is daily from 2015-01-02 to 2026-10-21 (month ends and the replay episodes' windows
  2008-2014). A later valuation date converts at 2026-10-21 unless LIVE (CACHED) or `derivx pull ecb-fx` extends
  it, and an earlier one at its first day, each with a warning naming the day used. ECB reference rates are set
  at about 14:15 CET, not at the US close, so a VaR day's FX move and its Treasury move are some hours apart. A
  TARGET holiday (Good Friday, Easter Monday, 1 May, 25-26 December) has no fixing: its FX move is 0, and the next
  scenario day carries both days' moves. A replay takes the ECB rates nearest its start and end dates; a pre-2015
  window with no rate near both ends holds FX, and its source row says so (flag `held`).
- Walk-forward metrics on real data wait for the collected history. Collection starts only when
  `collect-data.yml` is on `main` (GitHub runs schedules from the default branch only), and the walk-forward needs
  9 origin days (10 consecutive sessions at the 1-day horizon), so the first real result comes about two weeks
  after the merge. A session the workflow misses (a GitHub outage, Yahoo failing every retry) cannot be filled in
  later: Yahoo serves only the current chain. It shows as a missing trading day. Only one real SPY snapshot is
  bundled. The bundled Treasury curve ends 2026-10-19; `compute_metrics.py --history` discounts sessions more than
  7 days later at a flat rate unless `--with-cache` adds newer cached curves.
- Stored chains keep expiries of 1 to 730 days and strikes 0.4 to 1.6 times spot (about 15-17% of SPY quotes
  dropped on the two real days measured), and round Yahoo's implied vol to 5 decimals. The branch grows by about
  37 MB a year per ticker in git. That is fine for SPY, but many tickers over many years would need another store
  (release assets or object storage). Yahoo's terms of use apply to the collected files.
- The NYSE calendar models full-day scheduled holidays only.

Models
- Equity options: SPY (and every non-index ticker) lists American options, but vols are inverted with European
  Black-76 (no de-Americanisation); put-call parity discount factors are valid for European options only, so on
  American chains the parity gates fail and the Treasury curve (a stand-in for a risk-free OIS curve) is used.
- Swaption vol cube: SABR per point with alpha from ATM and bilinear rho/nu; no full SABR parameter cube; Greeks
  are sticky strike; the CAP-FLOOR tab strips flat normal-vol cap quotes only.
- Compounded caplets have no lookback, lockout or observation shift; term-rate (3M/6M) caps on an OIS curve are an
  approximation and say so. Collars are a CAP-FLOOR tab and library kind; in the BOOK a collar is a long cap plus a
  short floor.
- VaR holds vols constant and has no carry over the day; the delta+gamma method ignores cross-gammas (full
  revaluation is a button, about 11 s for 500 scenarios). FX moves only the converted totals. The delta-normal
  check takes FX returns as risk factors with sensitivity equal to each position's value in the reporting
  currency, and it leaves out the curve x FX cross term that the historical `ALL IN` row keeps.
- Multi-currency: the UI prices one swap-curve currency plus USD Treasuries; a trade in another currency is
  skipped with a hint to use the CLI's `--conventions all`. FX spots are ECB reference rates (indicative, not
  dealable) or typed in. They move in VaR and in historical replay by the ECB rates of the same dates; otherwise
  they are held unless a scenario shocks them (no FX forward drift, FX carry or cross-currency basis over a
  horizon). With typed-in spots, the VaR still takes its FX moves from the ECB history. The converted VaR is the
  VaR of the converted P&L, not a sum of per-currency VaRs.
- FX and equity scenario grids shift a flat vol (sticky strike) and do not re-mark the smile; FX theta shrinks both
  times by one calendar day. SABR (beta = 1) cannot fit some RR/BF combinations; the misfit is reported. With the
  ATM-at-spot convention on high-carry pairs (JPY, CHF), the least-squares SABR fit can land in one of several
  local minima.
- SSVI uses one rho, eta and gamma for every expiry; Dupire local vol jumps at fitted expiries (theta(T) is
  piecewise linear). Bucket vegas are first order (a linearised SVI refit); FX quote vegas are central differences
  of +/-5 vol bp.
- Curve models are fitted to par yields for the Treasury history (Diebold-Li used zero yields); rates vol
  forwards are constant-expiry, so a daily change also rolls one day along the forward curve.
- Bond roll-down follows the bootstrapped curve between pillars; book bonds are valued as held positions with no
  financing.

Platform
- pandas 3 and numpy 2.5 are not supported yet (capped). mypy runs in CI with a lenient config (11 error codes
  off, an `attr-defined` baseline of 18 modules).
- The GitHub workflows have not run on this revision (see [CI](#tests-and-validation)). `collect-data.yml` has
  never run on GitHub. It passes actionlint 1.7.12 with shellcheck 0.11.0, and its git steps (orphan branch,
  merge, commit, rebuilt push) are tested against a local bare repository. Its step scripts were also run end to
  end here, unchanged, against a local bare origin with real Yahoo pulls of the 2026-10-21 session: a first run
  with two tickers (the orphan branch, one commit), the same session again (no fetch, no commit), Thanksgiving
  and a Saturday (no commit), Yahoo unreachable (red, no commit), a valid and an invalid ticker (the valid one
  committed, the run red) and a dry run (nothing pushed). The `data-snapshots` branch appears with its first run
  on `main`; until then `derivx snapshots sync` and the lab's SYNC NOW report a 404.
- Inputs survive mode switches, but the selected tab inside a page is not kept.
- Yahoo rate-limits shared IPs; lab pulls back off and say so. Yahoo's terms of use govern its data.

---

## Screenshots

Taken with Playwright (Chromium, 1280 px wide; the phone view is 390 px) on this revision. Eight were retaken
(both ANALYTICS views, BOOK, SCENARIOS, and the CURVE, SWAP and RESEARCH views whose sidebar now has REPORTING
CURRENCY); EQUITY, FX, the FX smile, SWAPTION and the phone view came out pixel-identical and were kept.

| | |
|---|---|
| ![EQUITY](docs/screenshots/equity.png) EQUITY: premium, forward and moneyness, Greeks in desk units | ![FX](docs/screenshots/fx.png) FX: EURUSD 3M with market conventions, premium styles, Greeks |
| ![FX smile](docs/screenshots/fx-smile.png) FX smile from ATM/RR/BF quotes (ILLUSTRATIVE quotes) | ![RATES CURVE](docs/screenshots/rates-curve.png) RATES CURVE: quotes, sources, repricing, zero and forward curves |
| ![RATES SWAP](docs/screenshots/rates-swap.png) RATES SWAP: PV, par rate, annuity, PV01, cash flows | ![RATES SWAPTION](docs/screenshots/rates-swaption.png) RATES SWAPTION: SABR smile from the vol cube |
| ![RATES SCENARIOS](docs/screenshots/rates-scenarios.png) RATES SCENARIOS: shocked curve and P&L explain | ![RATES BOOK](docs/screenshots/rates-book.png) RATES BOOK: key-rate ladder by trade and vega by expiry x tenor |
| ![RATES RESEARCH RV](docs/screenshots/rates-research-rv.png) RATES RESEARCH > RV: spreads, z-scores, half-lives | ![RATES RESEARCH PCA](docs/screenshots/rates-research-pca.png) RATES RESEARCH > PCA: loadings and explained variance |
| ![ANALYTICS](docs/screenshots/analytics.png) ANALYTICS: the bundled real SPY chain, filters, discounting | ![ANALYTICS calibration](docs/screenshots/analytics-calibration.png) ANALYTICS: production SVI calibration |
| ![Phone](docs/screenshots/phone-equity.png) Phone (390 px): the sidebar starts collapsed | |

---

## Repository layout

```
src/derivx/
  core/          Black-Scholes-Merton prices, Greeks, limits, finite-difference checks
  vol/           implied vols, SABR, SVI, SSVI / Dupire, quote hygiene, per-expiry discounting, smile risk
  fx/            FX conventions, Garman-Kohlhagen, smiles (SABR, vanna-volga), smile risk, ECB rates,
                 illustrative quotes
  rates/         conventions and calendars, curves, multi-curve, swaps, FRAs and futures, caps/floors and
                 caplet stripping, swaptions and the vol cube, bonds, carry, RV, PCA, curve models, trade book,
                 curve.json, run manifests
  risk/          sensitivities, scenarios and P&L explain, historical replay, VaR / ES, multi-currency reporting
  data/          option-chain schema and Yahoo fetcher, bundled samples, synthetic history, NYSE calendar,
                 rates and ECB FX sources, cache and history store, the daily snapshot store, collector and
                 sync (snapshots.py), a dependency-free .xlsx reader (BoE archives)
  surface/       implied-vol surface from SVI slices (calendar checks, total-variance interpolation)
  calibration/   production SVI calibration;  models/  model comparison
  validation/    walk-forward, parameter stability, curve forecasts, rates vol history, realised vs implied,
                 manifests
  ui/            Streamlit app: streamlit_app.py (shell), pricer_page.py, fx_page.py, rates_*.py,
                 analytics_lab.py and lab_*.py (lab_history.py: the collected history), state.py, theme.py,
                 plot_limits.py
  cli.py         the derivx command line
.github/workflows/  ci.yaml, keep-alive.yml, collect-data.yml (daily chains to the data-snapshots branch)
tests/           pytest suite; golden/ (QuantLib fixtures), fixtures/ (recorded downloads)
tools/           gen_golden.py + golden/, compute_metrics.py, download_spy_chains.py, build_rates_snapshot.py
                 (+ the build_historical_shocks.py shim), record_rates_fixtures.py,
                 gen_fx_illustrative_quotes.py, gen_rates_vol_illustrative.py, keep_alive/
data/examples/   the bundled data (see Data sources); the collected chains live on the data-snapshots branch
docs/            metrics/ (reproducible lab metrics) and screenshots/
notebooks/       01 SVI, 02 SABR, 03 rates (executed)
```

Licence: MIT (see [LICENSE](LICENSE)). Data keep their publishers' terms (see [Data sources](#data-sources)).
