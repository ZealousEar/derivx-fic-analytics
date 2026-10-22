# Capability matrix

Reference page for DerivX (see [the main README](../README.md) and the [docs index](README.md)). Every row names
the feature, its status, and the test that holds the evidence.

Status: **implemented** (built and validated), **approximate** (works, with a stated simplification),
**planned** (not built). Validation links point at the test that holds the evidence; "golden" means the test
compares against committed QuantLib 1.44 output in [`tests/golden/`](../tests/golden/), and the same tests re-run
live QuantLib checks when QuantLib is installed. How the suite is run, and the reproducible lab metrics, are in
[validation.md](validation.md); the simplifications behind every "approximate" row are collected in
[limitations.md](limitations.md).

## Rates

| Feature | Status | Validation |
|---|---|---|
| Calendars, schedules, day counts, business-day rules (USD, EUR TARGET, GBP London, FX centres) | implemented | golden: [tests/test_conventions.py](../tests/test_conventions.py), [tests/golden/conventions.json](../tests/golden/conventions.json) (every date 1990-2080) |
| OIS bootstrap (SOFR, €STR, SONIA): deposits, OIS swaps, SR1/SR3 futures with Hull-White convexity; log-linear DF, linear zero and monotone-convex interpolation | implemented | golden: [tests/test_curves.py](../tests/test_curves.py) (QuantLib piecewise curves; every quote reprices, and the CURVE tab shows the error, about 1e-10bp) |
| Library curve API: `load_rates_quotes_from_csv` keeps each row's instrument and units, `build_zero_curve` prices each row as its own instrument and refuses day counts or compounding it cannot apply | implemented | golden: [tests/test_curves.py](../tests/test_curves.py) (the bundled `rates_quotes.csv` against QuantLib OIS and swap helpers, within 1e-6bp) |
| US Treasury curve from par yields (bills and par bonds) | implemented | golden: [tests/test_curves.py](../tests/test_curves.py), [tests/test_bonds.py](../tests/test_bonds.py) |
| Separate discount and projection curves: EUR €STR discounting with EURIBOR 3M/6M projection (deposits, FRAs, futures, IRS, 3s6s basis swaps) | implemented | golden: [tests/test_multicurve.py](../tests/test_multicurve.py) ([multicurve.json](../tests/golden/multicurve.json)) |
| Vanilla, OIS, forward-start and basis swaps: PV, par rate, annuity, cash flows, PV01, key-rate ladder per curve, hedge notionals | implemented | golden: [tests/test_swaps.py](../tests/test_swaps.py) |
| FRAs and rate futures | implemented | golden: [tests/test_multicurve.py](../tests/test_multicurve.py) |
| Caps, floors and collars on a term rate (EURIBOR 3M/6M on the multi-curve market; 3M/6M on an OIS curve, flagged as an approximation) or on SOFR, €STR or SONIA compounded in arrears (Lyashenko-Mercurio vol, or an effective vol to the last fixing); Black, shifted Black, Bachelier; flat vol, vol term structure, or caplet vols stripped from cap quotes; gamma, theta, vega by caplet expiry and by quoted cap | implemented | golden: [tests/test_capfloor.py](../tests/test_capfloor.py) (compounded caplets equal QuantLib's `OvernightLeg` with `BlackCompoundingOvernightIndexedCouponPricer`, 12 cases), [tests/test_capvol.py](../tests/test_capvol.py) (stripping equals QuantLib `OptionletStripper1`) |
| European swaptions (payer, receiver, straddle; Bachelier, Black, shifted Black), implied vols, Greeks | implemented | golden: [tests/test_swaptions.py](../tests/test_swaptions.py) (234 cases) |
| Swaption vol cube: SABR per expiry x tenor with alpha from ATM, variance-time ATM interpolation, vega buckets | approximate | [tests/test_swaption_vol.py](../tests/test_swaption_vol.py) (smiles equal `ql.shiftedSabrVolatility`; no full SABR parameter cube, Greeks sticky strike) |
| Treasury bonds: clean/dirty, accrued, yield, durations, convexity, DV01, 32nds, Z/G/I/asset-swap spreads | implemented | golden: [tests/test_bonds.py](../tests/test_bonds.py) |
| Scenarios: parallel, pivoted twist, butterfly, key-rate, IBOR-OIS and tenor basis, PCA factor, historical episode and any-window replay; vol shocks; FX shocks; P&L explain (delta, gamma, vol, carry, residual) | implemented | [tests/test_scenarios.py](../tests/test_scenarios.py), [tests/test_historical_replay.py](../tests/test_historical_replay.py) (explain against full revaluation) |
| Historical replay with each curve's own history; per-curve source flags; FX spots move by the ECB reference rates of the episode's dates (HOLD FX SPOTS / `--hold-fx` holds them) | approximate | [tests/test_historical_replay.py](../tests/test_historical_replay.py), [tests/test_var_fx.py](../tests/test_var_fx.py), [tests/test_rates_fx_var_ui.py](../tests/test_rates_fx_var_ui.py) (USD swap curves move with Treasuries after 2016: no free swap history) |
| RV: 2s10s, 5s30s, 2s5s10s and custom spreads/flies, z-scores, percentiles, AR(1) half-life, PCA-neutral weights | implemented | [tests/test_rv.py](../tests/test_rv.py) (independent numpy recomputation) |
| Carry and roll-down for bonds and swaps | implemented | [tests/test_carry.py](../tests/test_carry.py) (live QuantLib cross-check of bond carry) |
| PCA of daily curve changes, rolling loadings, factor risk | implemented | [tests/test_pca.py](../tests/test_pca.py) (77.9 / 12.9 / 5.4% on the committed sample) |
| One-day historical VaR and expected shortfall (delta+gamma or full revaluation), delta-normal cross-check; FX spots as a risk factor of the converted book (ECB reference rates of the same dates; FX-held and FX-only rows, FX in the delta-normal check; FX can be held); a risk report with key-rate PV01, gamma, vega by bucket and VaR | approximate | [tests/test_var.py](../tests/test_var.py), [tests/test_var_fx.py](../tests/test_var_fx.py), [tests/test_rates_research_ui.py](../tests/test_rates_research_ui.py), [tests/test_rates_fx_var_ui.py](../tests/test_rates_fx_var_ui.py) (vols held constant; the sensitivity method misses cross-gammas; no FX move on a TARGET holiday) |
| Trade blotter (swaps, basis swaps, FRAs, futures, bonds, caps, floors, swaptions): PV, PV01, ladders, vega matrix, scenarios, VaR, a hedge optimiser that hedges each currency's curves with that currency's instruments | implemented | golden: [tests/test_portfolio.py](../tests/test_portfolio.py) ([portfolio.json](../tests/golden/portfolio.json)) |
| Multi-currency books: totals per currency, the book converted to USD, EUR or GBP at ECB reference rates or typed FX spots, FX shocks on converted totals, FX moves in VaR and historical replay (UI and CLI) | approximate | [tests/test_reporting.py](../tests/test_reporting.py), [tests/test_rates_book_ui.py](../tests/test_rates_book_ui.py), [tests/test_cli.py](../tests/test_cli.py), [tests/test_var_fx.py](../tests/test_var_fx.py) (FX held in curve shocks unless the scenario shocks it; the UI prices one swap-curve currency plus USD Treasuries) |
| Seasoned trades in the blotter (past fixings) | planned | rejected with an explanation today |
| Rates market data: FRED, US Treasury, NY Fed, ECB (curves, €STR, EXR FX reference rates) and BoE fetchers, a dated cache and a history store; BoE spot curves converted to par OIS | implemented | [tests/test_rates_sources.py](../tests/test_rates_sources.py), [tests/test_rates_history.py](../tests/test_rates_history.py) and [tests/test_fx_history.py](../tests/test_fx_history.py) (recorded fixtures, no network); golden: [tests/test_market_quotes.py](../tests/test_market_quotes.py) |
| Run manifests and CSV/ZIP exports on every Rates tab; `curve.json` round trip; the `derivx` CLI | implemented | [tests/test_rates_exports.py](../tests/test_rates_exports.py), [tests/test_cli.py](../tests/test_cli.py) |
| Rates strategy backtester (carry/roll, steepener and PCA signals, DV01-neutral) | planned | - (audit item G18) |
| Meeting-date STIR curve: FOMC / ECB meeting calendar, step forward curve, hike and cut probabilities | planned | - (audit item G19; SR1/SR3 and EURIBOR futures with Hull-White convexity exist as curve inputs and trades) |
| XLSX export | planned | - |

## FX, equity and volatility

| Feature | Status | Validation |
|---|---|---|
| Black-Scholes-Merton price and Greeks (incl. vanna, volga, charm), exact limits at T = 0 and vol = 0 | implemented | golden: [tests/test_bs.py](../tests/test_bs.py) (232 QuantLib cases) |
| Implied vol (Black-76, shifted, Bachelier), vectorised, with a reason for every failure | implemented | golden: [tests/test_implied.py](../tests/test_implied.py) |
| Per-expiry discounting of option premiums: the US Treasury par curve of the trade date, put-call parity (gated; for European index options), or a flat rate | implemented | [tests/test_discounting.py](../tests/test_discounting.py) (exact recovery on European quotes; the American SPY chain fails the parity gates and falls back to the curve) |
| Hagan SABR (lognormal and normal, any beta, shifted), calibration | implemented | golden: [tests/test_sabr.py](../tests/test_sabr.py) (matches `ql.sabrVolatility` to 1e-12 relative) |
| SVI in total variance with butterfly, Lee and calendar checks; constrained no-arbitrage fit | implemented | golden: [tests/test_svi.py](../tests/test_svi.py), [tests/test_svi_performance.py](../tests/test_svi_performance.py) |
| SSVI surface (power law, Gatheral-Jacquier conditions) and Dupire local vol, on the lab's VOL SURFACE tab | approximate | [tests/test_ssvi.py](../tests/test_ssvi.py), [tests/test_lab_research_ui.py](../tests/test_lab_research_ui.py) (one rho, eta and gamma for every expiry, so it fits each expiry less closely than one SVI per expiry) |
| ECB reference rates for FX spot: a bundled daily history (every TARGET business day 2015-01-02 to 2026-10-21, month ends from 2008), extended by downloads; cross rates derived through the euro | implemented | [tests/test_fx_market_data.py](../tests/test_fx_market_data.py), [tests/test_fx_history.py](../tests/test_fx_history.py) |
| FX conventions: spot/expiry/delivery dates, joint calendars, forward points | implemented | golden: [tests/test_fx_conventions.py](../tests/test_fx_conventions.py) (every day of 2025-2027) |
| FX Garman-Kohlhagen with two times, delta conventions, ATM types, strike from delta | implemented | golden: [tests/test_fx_pricing.py](../tests/test_fx_pricing.py) (QuantLib BlackDeltaCalculator) |
| FX smiles from ATM/RR/BF quotes: SABR (beta = 1) or vanna-volga | approximate | [tests/test_fx_smile.py](../tests/test_fx_smile.py), [tests/test_fx_performance.py](../tests/test_fx_performance.py) (Hagan expansion; SABR cannot fit every RR/BF combination and reports the misfit) |
| FX and equity scenario grids | approximate | [tests/test_pricer_calc.py](../tests/test_pricer_calc.py) (sticky strike: the vol shift is flat, the smile is not re-marked) |
| Smile risk: sticky-strike, sticky-moneyness and SABR Bartlett deltas; SVI and SABR parameter vegas; vega by quote bucket through the SVI refit; FX vega to each ATM / RR / BF quote | implemented | [tests/test_smile_risk.py](../tests/test_smile_risk.py), [tests/test_smile_vegas.py](../tests/test_smile_vegas.py) (bucket vegas match brute-force refits to 0.2%) |

## Analytics lab and app

| Feature | Status | Validation |
|---|---|---|
| Option-chain schema, validated uploads (CSV, CSV.GZ, parquet), lineage and SHA-256 | implemented | [tests/test_chain_schema.py](../tests/test_chain_schema.py) |
| Yahoo Finance chains with retries, backoff and a fetch status | implemented | [tests/test_market_data.py](../tests/test_market_data.py) (recorded fake, no network) |
| Production SVI calibration with arbitrage repair and k-fold CV against an IV spline; a per-expiry progress bar; identical runs reused across sessions | implemented | [tests/test_calibration_production.py](../tests/test_calibration_production.py), [tests/test_analytics_lab.py](../tests/test_analytics_lab.py) |
| Walk-forward test against persistence / sticky-moneyness benchmarks, Diebold-Mariano, regimes; SVI and SABR parameter history with robust jump flags | implemented | [tests/test_backtest.py](../tests/test_backtest.py), [tests/test_walkforward_tab.py](../tests/test_walkforward_tab.py), [tests/test_parameter_stability.py](../tests/test_parameter_stability.py) (checked on a seeded SYNTHETIC history; the real history is collected from the merge to `main` on, see the lab metrics in [validation.md](validation.md)) |
| Daily option-chain collection: a scheduled workflow saves each session's chain on the `data-snapshots` branch (compact gzip CSV, a manifest per file, an index with SHA-256); `derivx snapshots sync / list`; the lab's **Collected history** source; `compute_metrics.py --history` | implemented | [tests/test_snapshots.py](../tests/test_snapshots.py), [tests/test_collect_workflow.py](../tests/test_collect_workflow.py) (the workflow's git steps against a local bare repository), [tests/test_snapshot_sync.py](../tests/test_snapshot_sync.py), [tests/test_lab_history_ui.py](../tests/test_lab_history_ui.py), [tests/test_compute_metrics_history.py](../tests/test_compute_metrics_history.py) (the workflow has not run on GitHub yet) |
| Model comparison (SVI, SABR, IV spline) | implemented | [tests/test_model_comparison.py](../tests/test_model_comparison.py), [tests/test_compute_metrics.py](../tests/test_compute_metrics.py) |
| Rates curve walk-forward (random walk against Diebold-Li NS AR(1), NS-curve AR(1) and PCA AR(1)) and curve-model comparison (Nelson-Siegel, Diebold-Li, Svensson, spline; leave-one-out) on the bundled Treasury, BoE and ECB histories, the cache or an upload | implemented | [tests/test_curve_forecast.py](../tests/test_curve_forecast.py), [tests/test_curve_models.py](../tests/test_curve_models.py), [tests/test_lab_research_ui.py](../tests/test_lab_research_ui.py) |
| Rates vol history: normal-SABR walk-forward, normal against shifted-lognormal SABR against a spline, daily SABR parameters; realised against implied ATM normal vol | implemented | [tests/test_vol_history.py](../tests/test_vol_history.py); golden: [tests/test_realised_implied.py](../tests/test_realised_implied.py) (forward swap rates equal QuantLib to 1e-12). No free source publishes dated swaption or cap vols: the tab runs on an upload or a labelled SYNTHETIC demo |
| Every chart zoom-bounded (no zooming to negative tenors or strikes) | implemented | [tests/test_plot_limits.py](../tests/test_plot_limits.py) |
| UI smoke tests of every mode and tab (Streamlit AppTest) | implemented | [tests/test_pricer_page.py](../tests/test_pricer_page.py), [tests/test_fx_page.py](../tests/test_fx_page.py), [tests/test_analytics_lab.py](../tests/test_analytics_lab.py), [tests/test_lab_research_ui.py](../tests/test_lab_research_ui.py), `tests/test_rates_*_ui.py` |
| Inputs survive a trip through the other modes and a page that raises (the selected tab inside a page is not kept) | implemented | [tests/test_mode_round_trip.py](../tests/test_mode_round_trip.py), [tests/test_ui_state.py](../tests/test_ui_state.py) |
| Readable dark theme (WCAG AA contrast) | implemented | [tests/test_theme.py](../tests/test_theme.py) plus a browser measurement (see [validation.md](validation.md)) |
| Deploy configs (XSRF/CORS on, non-root image, healthcheck) and the keep-alive check | implemented | [tests/test_deploy.py](../tests/test_deploy.py), [tests/test_keep_alive.py](../tests/test_keep_alive.py) |

## See also

- [data-sources.md](data-sources.md): what each bundled file is, its source and terms, and which inputs are illustrative, proxy or synthetic.
- [cli-and-deploy.md](cli-and-deploy.md): the `derivx` command and the deploy paths behind the last rows above.
