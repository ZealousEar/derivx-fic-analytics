# DerivX: fixed income and currency derivatives analytics

DerivX is a Python package and a Streamlit app for pricing and risk on rates, FX and equity options, with a
research lab for equity option smiles. The rates engine, the FX conventions and the option pricers are checked
against QuantLib 1.44 through committed golden fixtures. Every number on screen states its units and sign
convention, and every input that is not market data is labelled ILLUSTRATIVE, PROXY or SYNTHETIC.

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
- [Deploy](#deploy)
- [Known limitations](#known-limitations)
- [Screenshots](#screenshots)
- [Repository layout](#repository-layout)

---

## What it does

The app has four modes, chosen in the sidebar.

| Mode | What you get |
|---|---|
| **EQUITY** | Black-Scholes-Merton pricer with dividend yield: premium, first- and second-order Greeks in desk units, an implied-vol solver, a spot x vol scenario P&L grid, a risk surface (value or any Greek over spot x vol or spot x time), and a smile: an ILLUSTRATIVE SSVI slice, or an SVI fit to the SPY chain loaded in ANALYTICS. A model check compares every analytic Greek with a finite difference of the pricer. |
| **FX** | FX options with market conventions for ten G10 pairs: spot and delivery dates on joint settlement calendars, Garman-Kohlhagen with separate vol and rate times, the four delta conventions (spot/forward, premium-adjusted or not), ATM DNS / ATMF / ATM spot, strikes from delta, four premium styles, domestic and foreign rho. Smiles per tenor from ATM, 25D/10D risk reversal and butterfly quotes, fitted with SABR (beta = 1) or vanna-volga, with a butterfly-arbitrage check. Spot defaults to the ECB reference rate. |
| **RATES** | Eight tabs on one market: **CURVE** (OIS and Treasury bootstraps; EUR multi-curve with EURIBOR 3M/6M projection curves), **SWAP**, **SWAPTION**, **CAP-FLOOR**, **BONDS**, **BOOK** (trade blotter), **SCENARIOS** and **RESEARCH** (RV, carry and roll-down, PCA, VaR). Risk is signed P&L for +1bp on market quotes, with curves rebuilt. |
| **ANALYTICS** | Equity option smile lab: a real SPY chain (bundled) or an upload or a live Yahoo pull, quote hygiene, production SVI calibration with no-arbitrage checks and cross-validation, a walk-forward test against benchmarks, and a model comparison (SVI, SABR, IV spline). Every result downloads as a ZIP with a run manifest. |

Conventions used throughout:

- **Rates risk** (PV01, key-rate ladders, gamma, vega) is the P&L of the position for a +1bp move, signed to the
  holder: a receiver swap has negative PV01. Key-rate buckets bump one curve quote at a time and rebuild the curve
  (central differences); they sum to the parallel PV01.
- **Option Greeks** are for one long option: delta per 1.00 of spot and per +1% spot, gamma as the change in delta
  per +1% spot, vega per vol point, theta per calendar day (plus a one-day repriced theta), rho per +1bp.
- **Units** are in every label: rates in % or bp, vols in % (lognormal) or bp (normal), money in the trade's
  currency.

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
| US Treasury curve from par yields (bills and par bonds) | implemented | golden: [tests/test_curves.py](tests/test_curves.py), [tests/test_bonds.py](tests/test_bonds.py) |
| Separate discount and projection curves: EUR €STR discounting with EURIBOR 3M/6M projection (deposits, FRAs, futures, IRS, 3s6s basis swaps) | implemented | golden: [tests/test_multicurve.py](tests/test_multicurve.py) ([multicurve.json](tests/golden/multicurve.json)) |
| Vanilla, OIS, forward-start and basis swaps: PV, par rate, annuity, cash flows, PV01, key-rate ladder per curve, hedge notionals | implemented | golden: [tests/test_swaps.py](tests/test_swaps.py) |
| FRAs and rate futures | implemented | golden: [tests/test_multicurve.py](tests/test_multicurve.py) |
| Caps and floors: Black, shifted Black, Bachelier; flat vol, vol term structure, or caplet vols stripped from cap quotes; gamma, theta, vega by caplet expiry and by quoted cap | implemented | golden: [tests/test_capfloor.py](tests/test_capfloor.py), [tests/test_capvol.py](tests/test_capvol.py) (stripping equals QuantLib OptionletStripper1) |
| European swaptions (payer, receiver, straddle; Bachelier, Black, shifted Black), implied vols, Greeks | implemented | golden: [tests/test_swaptions.py](tests/test_swaptions.py) (234 cases) |
| Swaption vol cube: SABR per expiry x tenor with alpha from ATM, variance-time ATM interpolation, vega buckets | approximate | [tests/test_swaption_vol.py](tests/test_swaption_vol.py) (smiles equal `ql.shiftedSabrVolatility`; no full SABR parameter cube, Greeks sticky strike) |
| Treasury bonds: clean/dirty, accrued, yield, durations, convexity, DV01, 32nds, Z/G/I/asset-swap spreads | implemented | golden: [tests/test_bonds.py](tests/test_bonds.py) |
| Scenarios: parallel, pivoted twist, butterfly, key-rate, IBOR-OIS and tenor basis, PCA factor, historical episode and any-window replay; vol shocks; P&L explain (delta, gamma, vol, carry, residual) | implemented | [tests/test_scenarios.py](tests/test_scenarios.py), [tests/test_historical_replay.py](tests/test_historical_replay.py) (explain against full revaluation) |
| Historical replay with each curve's own history; per-curve source flags | approximate | [tests/test_historical_replay.py](tests/test_historical_replay.py) (USD swap curves move with Treasuries after 2016: no free swap history) |
| RV: 2s10s, 5s30s, 2s5s10s and custom spreads/flies, z-scores, percentiles, AR(1) half-life, PCA-neutral weights | implemented | [tests/test_rv.py](tests/test_rv.py) (independent numpy recomputation) |
| Carry and roll-down for bonds and swaps | implemented | [tests/test_carry.py](tests/test_carry.py) (live QuantLib cross-check of bond carry) |
| PCA of daily curve changes, rolling loadings, factor risk | implemented | [tests/test_pca.py](tests/test_pca.py) (77.9 / 12.9 / 5.4% on the committed sample) |
| One-day historical VaR and expected shortfall (delta+gamma or full revaluation), delta-normal cross-check | approximate | [tests/test_var.py](tests/test_var.py) (vols held constant; the sensitivity method misses cross-gammas) |
| Trade blotter (swaps, basis swaps, FRAs, futures, bonds, caps, floors, swaptions): PV, PV01, ladders, vega matrix, scenarios, VaR, hedge optimiser | implemented | golden: [tests/test_portfolio.py](tests/test_portfolio.py) ([portfolio.json](tests/golden/portfolio.json)) |
| Seasoned trades in the blotter (past fixings) | planned | rejected with an explanation today |
| Rates market data: FRED, US Treasury, NY Fed, ECB and BoE fetchers, a dated cache and a history store; BoE spot curves converted to par OIS | implemented | [tests/test_rates_sources.py](tests/test_rates_sources.py) and [tests/test_rates_history.py](tests/test_rates_history.py) (recorded fixtures, no network); golden: [tests/test_market_quotes.py](tests/test_market_quotes.py) |
| Run manifests and CSV/ZIP exports on every Rates tab; `curve.json` round trip; the `derivx` CLI | implemented | [tests/test_rates_exports.py](tests/test_rates_exports.py), [tests/test_cli.py](tests/test_cli.py) |
| XLSX export | planned | - |

### FX, equity and volatility

| Feature | Status | Validation |
|---|---|---|
| Black-Scholes-Merton price and Greeks (incl. vanna, volga, charm), exact limits at T = 0 and vol = 0 | implemented | golden: [tests/test_bs.py](tests/test_bs.py) (232 QuantLib cases) |
| Implied vol (Black-76, shifted, Bachelier), vectorised, with a reason for every failure | implemented | golden: [tests/test_implied.py](tests/test_implied.py) |
| Hagan SABR (lognormal and normal, any beta, shifted), calibration | implemented | golden: [tests/test_sabr.py](tests/test_sabr.py) (matches `ql.sabrVolatility` to 1e-12 relative) |
| SVI in total variance with butterfly, Lee and calendar checks; constrained no-arbitrage fit | implemented | golden: [tests/test_svi.py](tests/test_svi.py) |
| SSVI surface and Dupire local vol | implemented | [tests/test_ssvi.py](tests/test_ssvi.py) |
| ECB reference rates for FX spot, with a bundled offline fallback | implemented | [tests/test_fx_market_data.py](tests/test_fx_market_data.py) |
| FX conventions: spot/expiry/delivery dates, joint calendars, forward points | implemented | golden: [tests/test_fx_conventions.py](tests/test_fx_conventions.py) (every day of 2025-2027) |
| FX Garman-Kohlhagen with two times, delta conventions, ATM types, strike from delta | implemented | golden: [tests/test_fx_pricing.py](tests/test_fx_pricing.py) (QuantLib BlackDeltaCalculator) |
| FX smiles from ATM/RR/BF quotes: SABR (beta = 1) or vanna-volga | approximate | [tests/test_fx_smile.py](tests/test_fx_smile.py) (Hagan expansion; SABR cannot fit every RR/BF combination and reports the misfit) |
| FX and equity scenario grids | approximate | [tests/test_pricer_calc.py](tests/test_pricer_calc.py) (sticky strike: the vol shift is flat, the smile is not re-marked) |
| Smile risk: sticky-strike/sticky-moneyness and SABR Bartlett deltas | implemented | [tests/test_smile_risk.py](tests/test_smile_risk.py) |

### Analytics lab and app

| Feature | Status | Validation |
|---|---|---|
| Option-chain schema, validated uploads (CSV, CSV.GZ, parquet), lineage and SHA-256 | implemented | [tests/test_chain_schema.py](tests/test_chain_schema.py) |
| Yahoo Finance chains with retries, backoff and a fetch status | implemented | [tests/test_market_data.py](tests/test_market_data.py) (recorded fake, no network) |
| Production SVI calibration with arbitrage repair and k-fold CV against an IV spline | implemented | [tests/test_calibration_production.py](tests/test_calibration_production.py) |
| Walk-forward test against persistence / sticky-moneyness benchmarks, Diebold-Mariano, regimes | implemented | [tests/test_backtest.py](tests/test_backtest.py), [tests/test_walkforward_tab.py](tests/test_walkforward_tab.py) (checked on a seeded SYNTHETIC history; no real multi-day history is bundled, see [Lab metrics](#lab-metrics-reproducible)) |
| Model comparison (SVI, SABR, IV spline) | implemented | [tests/test_model_comparison.py](tests/test_model_comparison.py), [tests/test_compute_metrics.py](tests/test_compute_metrics.py) |
| Every chart zoom-bounded (no zooming to negative tenors or strikes) | implemented | [tests/test_plot_limits.py](tests/test_plot_limits.py) |
| UI smoke tests of every mode and tab (Streamlit AppTest) | implemented | [tests/test_pricer_page.py](tests/test_pricer_page.py), [tests/test_fx_page.py](tests/test_fx_page.py), [tests/test_analytics_lab.py](tests/test_analytics_lab.py), `tests/test_rates_*_ui.py` |
| Readable dark theme (WCAG AA contrast) | implemented | [tests/test_theme.py](tests/test_theme.py) plus a browser measurement (see [Tests](#tests-and-validation)) |
| Deploy configs (XSRF/CORS on, non-root image, healthcheck) and the keep-alive check | implemented | [tests/test_deploy.py](tests/test_deploy.py), [tests/test_keep_alive.py](tests/test_keep_alive.py) |
| Rates research in the ANALYTICS lab (the lab is equity options only) | planned | rates research lives on RATES > RESEARCH |

---

## Data sources

Everything bundled under `data/examples/` is offline-ready: the app and the tests never need the network. Each
file below says whether it is market data. What is not market data is labelled on every chart and table that
uses it.

| Data | Bundled file | Source and licence / terms | Market data? |
|---|---|---|---|
| US Treasury par yield curve, every business day 2015-2025 and month ends 2008-2014 | `rates_history/ust_par_fred.csv` | Federal Reserve H.15 via FRED (fredgraph.csv, no key). US government data, public domain; cite FRED | real |
| US Treasury par yield curve, 2026 to 10/19 | `ust_par_yield_curve_2026.csv` | US Treasury daily par yield curve CSV (home.treasury.gov). Public domain | real |
| SOFR, EFFR, OBFR fixings (2026) | `rates_history/fixings.csv` | Federal Reserve Bank of New York Markets API (no key), with attribution | real |
| €STR fixings; euro area AAA government curve (par and spot) | `rates_history/fixings.csv`, `eur_aaa_ecb*.csv` | ECB Data Portal (no key); reuse allowed with "Source: ECB" | real (daily only from 2026-08-19 in the bundle) |
| SONIA; UK OIS fitted spot curve | `rates_history/fixings.csv`, `gbp_ois_boe.csv` | Bank of England (yield-curve archive; SONIA via FRED), with attribution | real (daily only from 2026-08-19 in the bundle) |
| ICE USD swap rates (DSWP*), 2013 episode windows only | `rates_history/usd_swap_ice.csv` | ICE Benchmark Administration, republished by FRED until 2016-10-28. ICE's terms apply; only a few historical rows are bundled for the 2013 replay | real, ends 2016 |
| ECB euro FX reference rates, 2026-10-16..20 | `ecb_fx_reference_rates.csv` | ECB Data Portal, dataset EXR. Indicative, not dealable | real (reference rates) |
| SPY option chain, one snapshot, 2026-10-20 after the close | `spy_2026-10-20.csv.gz` (+ `.json` fetch status) | Yahoo Finance delayed quotes via yfinance. Yahoo's terms of use apply (personal and research use); kept for reproducibility | real, one day |
| USD SOFR OIS swap quotes | `usd_sofr_ois_example.csv`, `rates_quotes.csv` | none: no free SOFR swap feed exists | **ILLUSTRATIVE** |
| EUR €STR swap quotes | derived at run time | built from the ECB AAA government curve | **PROXY** |
| EURIBOR deposits, FRAs, futures, swaps, 3s6s basis | `src/derivx/rates/multicurve.py` | none: no free EURIBOR quotes | **ILLUSTRATIVE** |
| Swaption and cap vol quotes | `src/derivx/rates/vol_examples.py` | none: no free vol quotes | **ILLUSTRATIVE** |
| FX vol quotes (ATM/RR/BF) and FX deposit rates | `src/derivx/fx/examples.py` | none: no free FX vol quotes | **ILLUSTRATIVE** |
| Walk-forward demo history | generated, seeded (`derivx.data.synthetic`) | none | **SYNTHETIC** |

Rates data source (RATES sidebar, **DATA SOURCE**):

- **BUNDLED SNAPSHOT** (default): the files above, no network.
- **LIVE (CACHED)**: FRED, NY Fed, ECB and BoE with 5 s timeouts, downloads reused for an hour, falling back to
  the cache and then the snapshot with a visible notice; the slow Treasury CSV downloads in the background.
  Downloads go to `data/cache/` (gitignored) with their SHA-256 and fetch time.
- **UPLOAD**: your own quotes CSV, an optional Treasury-format par curve and (on SCENARIOS) your own swap
  history. Uploads persist across page switches until cleared.

Quote prefills by currency: **GBP SONIA** uses the real BoE OIS curve converted to par OIS rates
(QuantLib-checked); **EUR €STR** is a labelled PROXY from the ECB AAA government curve; **USD SOFR** is a labelled
ILLUSTRATIVE set fixed at 2026-10-19 levels. The real overnight fixing is always the ON quote. Every curve chart
carries a `SOURCE ... | AS OF ...` badge.

Historical scenarios replay each curve's own history where a free one exists (Treasury par yields; ICE swaps
before 2016-10-28; BoE OIS for GBP; an uploaded swap history) and flag a proxy otherwise. After 2016 USD swap
curves move with Treasuries (swap spreads held), and the per-curve source table on SCENARIOS says so.

Rebuild or extend the bundled history: `python tools/build_rates_snapshot.py`. Pull more into the cache with
the CLI (below), for example the whole ICE swap history or the full BoE daily archive (about 12 MB).

Environment variables: `DERIVX_CACHE_DIR` (default `data/cache`), `DERIVX_OFFLINE=1` (cache only),
`FRED_API_KEY` (optional: FRED's JSON API and ALFRED vintages; never cached or logged).

---

## Lab metrics (reproducible)

Earlier versions of this README quoted SVI/SABR and walk-forward figures (a 71.8% hit rate, 0.052 RMSE, "SVI beats
SABR by 25.3%", "24 trading days") that came from one snapshot saved under 24 dates. They could not be
reproduced and are withdrawn. The numbers below are the only lab metrics this repository claims, and anyone can
regenerate them from the committed sample:

```bash
python tools/compute_metrics.py --outdir docs/metrics/spy_2026-10-20   # about 75 s; output is committed
```

Data: one real SPY chain (Yahoo, 2026-10-20 after the close, spot 769.05, 10,040 quotes, 31 expiries).
Filters: volume >= 10, mid >= 0.10, relative spread <= 40%, then OTM-only quotes, one per strike, put-call parity
forwards, vols re-inverted from mids at r = 0. 29 of 31 expiries are scored (same-day and 1-day expiries skipped).

| Model | Out-of-sample 5-fold CV RMSE, median over expiries (vol bp) | Mean | Lowest on |
|---|---|---|---|
| IV spline (natural cubic in ln K/F) | 1.4 | 4.7 | 28 of 29 |
| SVI (raw, total variance) | 4.1 | 6.0 | 1 of 29 |
| SABR (beta = 1) | 15.9 | 17.3 | 0 of 29 |

- SVI beats SABR on 29 of 29 expiries; the spline beats SVI on 28 of 29. The spline interpolates dense strikes,
  so it is scored out of sample only and is a strong baseline.
- Production SVI with arbitrage repair: median in-sample RMSE 4.5 vol bp (max 24.2). Butterfly and Lee checks are
  clean on 29 of 29 expiries; the calendar check is clean on 26 of 28 adjacent pairs (fails at 2027-03-19 and
  2027-12-17). Calendar arbitrage is reported, not enforced across expiries.
- These are single-snapshot fit metrics, not forecasts. Full tables: [docs/metrics/spy_2026-10-20](docs/metrics/spy_2026-10-20/METRICS_SUMMARY.md),
  with a manifest recording the commit, packages, inputs and data hash.
- SABR parameter recovery from exact quotes: `pytest tests/test_sabr.py -k recovers` (alpha and nu to 1e-5
  relative, rho to 1e-5 absolute, F from 1.08 to 15,800, T from 1W to 5Y).

**Walk-forward results on real data do not exist yet.** Yahoo serves only the current chain, so a real history has
to be collected one session at a time. The Walk-Forward tab runs on uploaded daily snapshots or on a labelled,
seeded SYNTHETIC demo. To build a real history, run once per trading day after the close (for example from cron
with `CRON_TZ=America/New_York` and `25 16 * * 1-5`):

```bash
python tools/download_spy_chains.py --ticker SPY --outdir data/market/spy   # one spy_<session>.parquet + .json per day
python tools/compute_metrics.py --chain-dir data/market/spy --outdir data/metrics
```

`download_spy_chains.py` refuses past dates, keeps an existing session file unless `--replace`, and exits 0
(saved), 1 (nothing saved) or 3 (saved but partial). `data/**/*.parquet` and `data/metrics/` are gitignored.

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
pip install -e ".[dev]"                  # pytest, QuantLib 1.44, ruff 0.17.1, black, mypy
pip install -e ".[dev,notebooks]"        # + matplotlib, ipykernel, nbmake
pytest                                   # whole suite, offline
pytest --nbmake notebooks                # execute the three notebooks
ruff check . && ruff format --check .
```

Dependency policy: `pyproject.toml` holds the supported ranges and `requirements.txt` repeats them
(`tests/test_deploy.py` checks they agree). `requirements-lock.txt` pins the resolution the Docker image is built
from; regenerate it with
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
derivx pull treasury --year 2026                       # slow endpoint
derivx pull nyfed --rate sofr --last 30
derivx pull ecb-estr
derivx pull ecb-yc --curve AAA --measure par
derivx pull boe-ois --archive daily                    # about 12 MB
derivx pull latest                                     # what LIVE (CACHED) uses
#   options: --offline, --out file.csv|parquet, --max-age, --cache

# curves
derivx curve build --asof 2026-10-19 --interp loglinear --out curve.json   # [--quotes q.csv] [--conventions "EUR ESTR"]
derivx curve show --curve curve.json [--nodes nodes.csv --grid grid.csv]   # rebuilds and checks every node (1e-12)

# a trade blotter: example, validation, risk, hedges, scenarios
derivx book example --currency USD > trades.csv
derivx book check --trades trades.csv
derivx risk --trades trades.csv --curve curve.json --hedge 2Y,5Y,10Y,30Y [--bid-ask 0.25 [--move 5]] \
            [--illustrative-vols] [--out risk.csv] [--ladder ladder.csv] [--manifest run.json]
derivx scenario --trades trades.csv --asof 2026-10-19 \
            --shock parallel:+25 --shock twist:5Y:-10:+10 --shock pca:1:2 --shock historical:2022-hiking-cycle \
            [--explain explain.csv] [--out grid.csv]
```

Shock specs (`derivx scenario --help`): `parallel:BP`, `twist:PIVOT:SHORT_BP:LONG_BP`,
`fly:BODY:BODY_BP[:WING_BP[:LEFT:RIGHT]]`, `keyrate:2Y=+10,10Y=-5`, `pca:FACTOR[:N_SIGMA[:DAYS]]`,
`historical:NAME[:SCALE]` (episodes: 2013 taper tantrum, 2020 dash for cash, 2022 hiking cycle, Sept 2022 gilt
crisis week, 2023 SVB), `window:START:END[:SCALE]`, `basis:BP`, `tenorbasis:BP`, `library`.

Caps and swaptions in a blotter need vols: a `vol` column per trade, or `--swaption-vols` / `--cap-vols` files, or
`--illustrative-vols` (labelled). Without them those trades are skipped with the reason. Exit codes: 2 for missing
files or a malformed curve JSON, 3 for blotter rows with errors or trades that cannot be priced.

`curve.json` is the `derivx.curve_set` v1 format: as-of date, conventions, interpolation, convexity settings,
fixings and every curve's quotes. Blotter CSVs accept comma, semicolon, tab or bar delimiters; commas inside
numbers are thousands separators only (`1,000,000`), and a decimal comma is refused with a message.

Other tools: `tools/gen_golden.py` (regenerate the QuantLib fixtures), `tools/compute_metrics.py` and
`tools/download_spy_chains.py` (lab metrics and history), `tools/build_rates_snapshot.py` and
`tools/record_rates_fixtures.py` (bundled rates data and parser fixtures).

---

## Tests and validation

Test run on this revision: `pytest` collects 1,269 tests from 59 files (16 minutes here, with three suites
sharing 4 cores; run single files while iterating).

| Environment | Result |
|---|---|
| Python 3.11, the dev extra (QuantLib 1.44 installed, so the live QuantLib checks run) | 1,269 passed |
| the same with `-W error::DeprecationWarning` | 1,269 passed |
| Python 3.10.20 with every direct dependency at its lower bound (numpy 1.26.0, pandas 2.1.0, SciPy 1.11.1, Streamlit 1.59.0, Plotly 5.22.0, pyarrow 14.0.0, yfinance 0.2.66), `-W error::DeprecationWarning` | 1,267 passed, 2 skipped (run from an export without `.git` and without setuptools) |
| `pytest --nbmake notebooks` | 3 passed |

- **QuantLib golden fixtures.** `tools/gen_golden.py` with the modules in `tools/golden/` writes
  `tests/golden/<area>.json` from QuantLib 1.44 (calendars, curves, multi-curve, swaps, caps/floors, cap vol
  stripping, swaptions, bonds, Black-Scholes, implied vols, SABR/SVI, FX conventions and pricing, the rates
  market-data conversions, the example trade book). The tests read the committed JSON, so the suite needs no QuantLib;
  with QuantLib installed (`pip install -e ".[dev]"`) the same tests also run randomised live QuantLib checks.
  Regenerate with `python tools/gen_golden.py` (all) or `python tools/gen_golden.py swaps capvol` (some).
- **Property and regression tests**: quotes reprice, discount factors positive, cap - floor = swap, payer -
  receiver = forward swap, key-rate buckets sum to PV01, P&L explain against full revaluation, put-call parity
  on forwards, implied-vol round trips, SABR parameter recovery, no-arbitrage checks that fail where they should.
- **UI tests**: Streamlit AppTest renders every mode and every Rates and Analytics tab with default and edge
  inputs; `tests/test_plot_limits.py` asserts every chart is zoom-bounded; `tests/test_theme.py` checks the
  theme's text colours and heatmap scales against WCAG AA.
- **Hermetic**: an autouse fixture blocks network access, points the cache at a temporary directory and removes
  `FRED_API_KEY`; parsers are tested on recorded fixtures under `tests/fixtures/`.
- **Deploy**: `tests/test_deploy.py` checks the Docker, Render, Procfile and devcontainer configs (XSRF and CORS
  on, non-root image, healthcheck) and starts a live server on the committed config to confirm it refuses a
  token-less upload and sends no wildcard CORS header.

CI (`.github/workflows/ci.yaml`): ruff check and format over the whole repository; the suite on Python 3.10, 3.11,
3.12 and 3.13 with live QuantLib checks and deprecation warnings as errors; Python 3.10 with every direct
dependency at its lower bound; the notebooks with nbmake; and a Docker job that builds the test and runtime
images, waits for the healthcheck and checks XSRF and CORS on the running container.

Browser checks (not in CI, run on this revision): every mode, tab and sub-tab was driven in Chromium with
Playwright at 390, 1280 and 1920 px, plus the lab's calibration, walk-forward and model-comparison runs and the
EUR (multi-curve) and GBP markets, and the app was run from a copy laid out the way the Dockerfile builds it.
No page exceptions, console errors or Streamlit exception blocks; every chart has declared bounds; a plain wheel
scrolls the page, and ctrl-wheel zooms but never past the bounds; no text node below WCAG AA contrast.

Render times measured in the browser at 1280 px on a fresh local server (bundled data; seconds, including the
websocket round trip):

| Mode | First visit (cold caches) | Rerun, same inputs | Switching back later |
|---|---|---|---|
| EQUITY | 2.5 (page load) | 0.5 | 0.3 |
| FX | 2.5 | 0.5-0.8 | 2.8 |
| RATES | 10.0 | 2.6-2.8 | 6.4 |
| ANALYTICS | 2.7 | 0.4 | 1.6 |

RATES runs all eight tabs on every rerun (SCENARIOS and RESEARCH need the positions of the others); LIVE (CACHED)
adds the downloads (about 17 s the first time, then cached for an hour). In the lab, the walk-forward demo takes
about 16 s and "Compare models on every expiry" about 35 s.

---

## Deploy

All deploy paths read the committed `.streamlit/config.toml`, which keeps XSRF and CORS protection on and sets the
dark theme. Secrets go in `.streamlit/secrets.toml`, which git and `.dockerignore` keep out.

**Docker**

```bash
docker build -t derivx --build-arg GIT_SHA=$(git rev-parse HEAD) .
docker run --rm -p 8501:8501 derivx          # http://localhost:8501
docker build --target test .                  # the UI tests, run inside the image
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
(including `-e .`) and reads the committed config. Community Cloud controls its own error-detail setting.

**Devcontainer / Codespaces**: the app opens in a real browser tab; the editor's preview pane can block the XSRF
cookie that file uploads need.

**Keep-alive**: `.github/workflows/keep-alive.yml` opens the Community Cloud app in headless Chrome every six hours,
wakes it if it is asleep, and fails unless the sidebar heading NAVIGATION renders with no Streamlit exception. It
makes no commits and does not ingest data.

---

## Known limitations

Data
- No free SOFR or €STR swap quotes, EURIBOR quotes, swaption/cap vols or FX vol quotes exist. Those defaults are
  ILLUSTRATIVE or PROXY and labelled; paste or upload real quotes to price real trades.
- USD swap curves in historical replay, VaR and PCA shocks move with Treasury changes after 2016 (swap spreads
  held, flagged). Swap spreads appear only with an uploaded swap history or the ICE series (to 2016-10-28).
- The bundled ECB and BoE histories are daily only from 2026-08-19 (month ends before), so EUR/GBP RV and PCA in
  BUNDLED mode run on about 43 daily changes, with a warning. LIVE mode or the CLI pulls more.
- Walk-forward metrics on real data need daily snapshots collected from now on (see the lab section); only one
  real SPY snapshot is bundled. Vols are re-inverted at a user rate that defaults to 0%.
- The NYSE calendar models full-day scheduled holidays only.

Models
- Swaption vol cube: SABR per point with alpha from ATM and bilinear rho/nu; no full SABR parameter cube; Greeks
  are sticky strike; the CAP-FLOOR tab strips flat normal-vol cap quotes only.
- VaR holds vols constant and has no carry over the day; the delta+gamma method ignores cross-gammas (full
  revaluation is a button, about 10-12 s for 500 scenarios).
- FX and equity scenario grids shift a flat vol (sticky strike) and do not re-mark the smile; FX theta shrinks both
  times by one calendar day. SABR (beta = 1) cannot fit some RR/BF combinations; the misfit is reported.
- Bond roll-down follows the bootstrapped curve between pillars; book bonds are valued as held positions with no
  financing.

Book and exports
- Seasoned trades (past fixings) are rejected; the hedge objective adds key-rate buckets across currencies (mixed
  books are flagged); the risk report carries one vega per position (bucketed vegas are on SWAPTION and
  CAP-FLOOR). No XLSX export (openpyxl is not a dependency).

Platform
- pandas 3 and numpy 2.5 are not supported yet (capped); mypy is not gated in CI.
- The GitHub workflows were checked locally (actionlint, each step run by hand) but this exact revision has not
  run on GitHub.
- Yahoo rate-limits shared IPs; lab pulls back off and say so. Yahoo's terms of use govern its data.

---

## Screenshots

Regenerated from this revision with Playwright (Chromium, 1280 px wide; the phone view is 390 px).

| | |
|---|---|
| ![EQUITY](docs/screenshots/equity.png) EQUITY: premium, forward and moneyness, Greeks in desk units | ![FX](docs/screenshots/fx.png) FX: EURUSD 3M with market conventions, premium styles, Greeks |
| ![FX smile](docs/screenshots/fx-smile.png) FX smile from ATM/RR/BF quotes (ILLUSTRATIVE quotes) | ![RATES CURVE](docs/screenshots/rates-curve.png) RATES CURVE: quotes, sources, repricing, zero and forward curves |
| ![RATES SWAP](docs/screenshots/rates-swap.png) RATES SWAP: PV, par rate, annuity, PV01, cash flows | ![RATES SWAPTION](docs/screenshots/rates-swaption.png) RATES SWAPTION: SABR smile from the vol cube |
| ![RATES SCENARIOS](docs/screenshots/rates-scenarios.png) RATES SCENARIOS: shocked curve and P&L explain | ![RATES BOOK](docs/screenshots/rates-book.png) RATES BOOK: key-rate ladder by trade and vega by expiry x tenor |
| ![RATES RESEARCH RV](docs/screenshots/rates-research-rv.png) RATES RESEARCH > RV: spreads, z-scores, half-lives | ![RATES RESEARCH PCA](docs/screenshots/rates-research-pca.png) RATES RESEARCH > PCA: loadings and explained variance |
| ![ANALYTICS](docs/screenshots/analytics.png) ANALYTICS: the bundled real SPY chain, filters, hygiene | ![ANALYTICS calibration](docs/screenshots/analytics-calibration.png) ANALYTICS: production SVI calibration |
| ![Phone](docs/screenshots/phone-equity.png) Phone (390 px): the sidebar starts collapsed | |

---

## Repository layout

```
src/derivx/
  core/          Black-Scholes-Merton prices, Greeks, limits, finite-difference checks
  vol/           implied vols, SABR, SVI, SSVI / Dupire, quote hygiene, smile risk
  fx/            FX conventions, Garman-Kohlhagen, smiles (SABR, vanna-volga), ECB rates, illustrative quotes
  rates/         conventions and calendars, curves, multi-curve, swaps, FRAs and futures, caps/floors and
                 caplet stripping, swaptions and the vol cube, bonds, carry, RV, PCA, trade book, curve.json,
                 run manifests
  risk/          sensitivities, scenarios and P&L explain, historical replay, VaR / ES
  data/          option-chain schema and Yahoo fetcher, bundled samples, synthetic history, rates sources,
                 cache and history store
  calibration/   production SVI calibration;  models/  model comparison;  validation/  walk-forward, manifests
  ui/            Streamlit app: streamlit_app.py (shell), pricer_page.py, fx_page.py, rates_*.py,
                 analytics_lab.py, theme.py, plot_limits.py
  cli.py         the derivx command line
tests/           pytest suite; golden/ (QuantLib fixtures), fixtures/ (recorded downloads)
tools/           gen_golden.py + golden/, compute_metrics.py, download_spy_chains.py, build_rates_snapshot.py,
                 record_rates_fixtures.py, gen_*_illustrative*.py, keep_alive/
data/examples/   the bundled data (see Data sources)
docs/            metrics/ (reproducible lab metrics) and screenshots/
notebooks/       01 SVI, 02 SABR, 03 rates (executed)
```

Licence: MIT (see [LICENSE](LICENSE)). Data keep their publishers' terms (see [Data sources](#data-sources)).
