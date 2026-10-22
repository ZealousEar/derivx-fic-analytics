```text
  ____  _____ ____  _____     ____  __
 |  _ \| ____|  _ \|_ _\ \   / /\ \/ /
 | | | |  _| | |_) || | \ \ / /  \  /
 | |_| | |___|  _ < | |  \ V /   /  \
 |____/|_____|_| \_\___|  \_/   /_/\\_\

fixed-income, FX and equity derivatives analytics
```

# DerivX

DerivX is a Python package and Streamlit application for pricing, risk, market-data research and model validation across rates, FX and equity options.

It is built for analysts and researchers who need to move from a quote or trade to a reproducible result: every displayed number carries units and sign conventions, while non-market inputs are labelled **ILLUSTRATIVE**, **PROXY** or **SYNTHETIC**.

**[Open the hosted demo](https://derivx.streamlit.app/)** · **[Read the documentation index](docs/README.md)** · **[Install locally](#quick-start)**

![RATES > CURVE: SOFR OIS curve, quotes, sources, repricing errors and zero / 3M forward curves](docs/screenshots/rates-curve.png)

> The hosted demo runs from `main` and may lag this checkout or need a minute to wake up. It starts with bundled data and does not require network access.

## What you can do

| Workspace | Use it for |
|---|---|
| **EQUITY** | Black–Scholes–Merton pricing, Greeks, implied volatility, spot × volatility scenarios, risk surfaces and illustrative or fitted smiles. |
| **FX** | Garman–Kohlhagen pricing, G10 conventions, delta-based strikes, ATM/RR/BF smiles, SABR or vanna-volga fitting, and smile risk. |
| **RATES** | OIS and Treasury curves, EUR multi-curve construction, swaps, swaptions, caps/floors, bonds, books, scenarios, VaR, PCA, relative value and carry/roll-down. |
| **ANALYTICS** | SPY option-chain hygiene, discounting, SVI calibration, SSVI/Dupire, walk-forward validation, model comparison, rates-curve forecasts and rates-vol history. |

### A typical workflow

```text
market data or illustrative quotes
             │
             ▼
     price / calibrate / build curve
             │
             ▼
       sensitivities and scenarios
             │
             ▼
       export, manifest, or report
```

1. Choose a workspace from the sidebar.
2. Start with the bundled snapshot, or select **LIVE (CACHED)** / **UPLOAD** where available.
3. Inspect source, as-of date, units and approximation labels before using a result.
4. Export the calculation or reproduce it through the Python API or `derivx` CLI.

The full feature matrix, validation links and status labels are in **[Capabilities](docs/capabilities.md)**.

## Screenshots

The app is designed to keep the analytical surface dense without hiding provenance or units.

| | |
|---|---|
| ![EQUITY](docs/screenshots/equity.png) **EQUITY** — premium, forward, Greeks and scenario analysis | ![FX](docs/screenshots/fx.png) **FX** — conventions, premium styles and Greeks |
| ![FX smile](docs/screenshots/fx-smile.png) **FX smile** — ATM/RR/BF quotes and fitted smile | ![RATES CURVE](docs/screenshots/rates-curve.png) **RATES CURVE** — quotes, sources, repricing and curves |
| ![RATES SWAP](docs/screenshots/rates-swap.png) **RATES SWAP** — PV, par rate, annuity, PV01 and cash flows | ![RATES SWAPTION](docs/screenshots/rates-swaption.png) **RATES SWAPTION** — SABR smile from the vol cube |
| ![RATES SCENARIOS](docs/screenshots/rates-scenarios.png) **RATES SCENARIOS** — shocked curves and P&L explain | ![RATES BOOK](docs/screenshots/rates-book.png) **RATES BOOK** — key-rate ladder and vega buckets |
| ![RATES RESEARCH RV](docs/screenshots/rates-research-rv.png) **RATES RESEARCH > RV** — spreads, z-scores and half-lives | ![RATES RESEARCH PCA](docs/screenshots/rates-research-pca.png) **RATES RESEARCH > PCA** — loadings and explained variance |
| ![ANALYTICS](docs/screenshots/analytics.png) **ANALYTICS** — bundled SPY chain, filters and discounting | ![ANALYTICS calibration](docs/screenshots/analytics-calibration.png) **ANALYTICS** — production SVI calibration |
| ![Phone](docs/screenshots/phone-equity.png) **Phone** — responsive layout at 390 px | |

## Quick start

Python 3.10–3.13 is supported. The CI matrix covers all four versions; Docker, the devcontainer and `.python-version` use Python 3.11.

```bash
git clone https://github.com/ZealousEar/derivx-fic-analytics.git
cd derivx-fic-analytics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/derivx/ui/streamlit_app.py
```

Open <http://localhost:8501>. The default app is offline-ready on bundled data. For development tools and notebooks:

```bash
pip install -e ".[dev]"            # pytest, QuantLib 1.44, ruff, mypy, black
pip install -e ".[dev,notebooks]"  # add matplotlib, ipykernel and nbmake
pytest
pytest --nbmake notebooks
```

## Python package and CLI

The package is installed editable by `requirements.txt`; `pip install -e .` also installs the `derivx` command. The library is organized around reusable pricing, curve, volatility, market-data, risk and validation modules under `src/derivx/`.

```bash
# Build and inspect a curve
derivx curve build --asof 2026-10-19 --interp loglinear --out curve.json
derivx curve show --curve curve.json

# Create, check and risk an example book
derivx book example --currency USD > trades.csv
derivx book check --trades trades.csv
derivx risk --trades trades.csv --curve curve.json --hedge 2Y,5Y,10Y,30Y \
  --illustrative-vols --out risk.csv --manifest run.json

# Run a scenario
derivx scenario --trades trades.csv --asof 2026-10-19 \
  --illustrative-vols --shock parallel:+25 --shock historical:2022-hiking-cycle
```

Most commands work offline on bundled data. `pull` commands populate the cache from supported public sources; `risk` and `scenario` can write run manifests. See **[CLI and deployment](docs/cli-and-deploy.md)** for the complete command reference, data pulls, Docker, Render, Community Cloud and environment variables.

## Data and reproducibility

Bundled examples are offline-ready. Real sources include FRED, the US Treasury, the NY Fed, ECB and BoE; quote examples without a free public feed are explicitly labelled. SPY data is delayed Yahoo Finance data and remains subject to Yahoo's terms.

- **BUNDLED SNAPSHOT** — deterministic local examples; no network required.
- **LIVE (CACHED)** — fetches public sources, reuses an hourly cache and falls back to the snapshot with a visible notice.
- **UPLOAD** — accepts supported quote and history files for your own market data.

Read **[Data sources](docs/data-sources.md)** before treating an output as market data. It records source terms, bundled paths, cache behavior, environment variables and provenance labels.

The repository's quantitative claims are backed by committed fixtures and tests. The reproducible SPY metrics can be regenerated with:

```bash
python tools/compute_metrics.py --outdir docs/metrics/spy_2026-10-20
```

See **[Validation and metrics](docs/validation.md)** for the test suite, QuantLib golden fixtures, browser checks, performance measurements and lab methodology.

## Conventions that affect interpretation

- Rates risk is signed P&L for a +1 bp move in the market quote; a receiver swap has negative PV01.
- Option vega is per volatility point, theta is per calendar day, rho is per +1 bp, and option delta is stated per 1.00 and per +1% spot.
- Amounts remain in the trade currency. Cross-currency totals are not added until explicitly converted into the reporting currency.
- Reference rates, illustrative quotes and synthetic histories are not dealable market data.

## Scope and limitations

DerivX is a research and analytics platform, not a market-data terminal or execution system. Important limitations include illustrative default swap and volatility quotes, European Black-76 inversion for American SPY options, approximate swaption and cap-vol surfaces, held swap spreads in parts of historical replay, and no rates strategy backtester or XLSX export.

The complete limitation register is in **[Limitations](docs/limitations.md)**. It distinguishes unbuilt features, data availability, model approximations and platform constraints.

## Documentation

- **[Documentation index](docs/README.md)** — map of the reference pages and project assets.
- **[Capabilities](docs/capabilities.md)** — feature matrix with implementation status and validation links.
- **[Data sources](docs/data-sources.md)** — bundled data, source terms, cache modes and provenance.
- **[Validation and metrics](docs/validation.md)** — tests, reproducible metrics, CI, browser checks and performance.
- **[CLI and deployment](docs/cli-and-deploy.md)** — setup, command reference and deployment paths.
- **[Limitations](docs/limitations.md)** — explicit boundaries and known approximations.

## Repository layout

```text
src/derivx/       package: pricing, curves, vol, FX, rates, risk, data, validation and UI
tests/             pytest suite, recorded fixtures and QuantLib golden outputs
tools/             data builders, metric generation, fixture tools and snapshot utilities
data/examples/    bundled sample market data
notebooks/        SVI, SABR and rates examples
docs/              reference documentation, screenshots and reproducible metrics
.github/workflows/ CI, hosted-app keep-alive and daily option-chain collection
```

## Licence

Source code: [MIT](LICENSE). Data remain subject to their publishers' terms; see **[Data sources](docs/data-sources.md)**.
