# DerivX — Fixed Income & Currencies Analytics (v2)
# https://derivx.streamlit.app/

DerivX is a portfolio-ready analytics platform showcasing quantitative engineering for Fixed Income & Currencies roles. The project combines Financial Mathematics theory with a friendly UX and production-style data plumbing.

### Recruiter Snapshot
- **Real market data**: 24 trading days of SPY chains ingested automatically via yfinance + Selenium keep-alive workflow.
- **Quantified edge**: Constrained SVI calibration averages **0.052 RMSE** (median 0.040) and beats spline baselines across **32 expiries**.
- **Robust validation**: Walk-forward lab posts **71.8% directional hit-rate** with live regime segmentation and model benchmarking (SVI vs SABR vs local-vol).
- **Deployment-ready**: Streamlit UI + GitHub Actions + wake-up automation keep the demo live for hiring managers.

The platform highlights:

- **Equity mode** — vectorised Black–Scholes with dividend/carry, Greeks dashboard, 3D price surface, implied-vol and SVI calibration.
- **FX mode** — Black–Scholes with foreign/domestic rates, SABR calibration and smile diagnostics.
- **Rates mode** — OIS/IRS bootstrapping, Black cap/floor pricing, PV/DV01/PV01 buckets, and scenario shocks (parallel/twist).

The app is built in Streamlit with a production-grade package layout, unit tests, sample data, and CI. Use this repository to demonstrate competency across term structures, volatility modelling, and fixed income risk.

---

## Quick start

```bash
# Clone & enter
git clone https://github.com/ZealousEar/derivx-fic-analytics.git
cd derivx-fic-analytics

# Optional: create virtualenv
python -m venv .venv && source .venv/bin/activate

# Install requirements and package
pip install -r requirements.txt

# Launch Streamlit app
streamlit run src/derivx/ui/streamlit_app.py

# Run unit tests
pytest tests/ -v
```

Prefer to skip setup? Launch the hosted demo at https://derivx.streamlit.app/.

The project targets Python 3.10+ and has been validated on macOS and Ubuntu. The Streamlit app uses caching, so the initial load calibrates surfaces once and subsequent interactions remain responsive.

---

## Repository tour

```
.
├── README.md                  # Overview + metrics
├── requirements.txt           # Runtime dependencies (numpy<2, pandas<2.2, yfinance>=0.2.66)
├── src/derivx/                # Python package
│   ├── ui/streamlit_app.py    # Streamlit interface (Equity/FX/Rates + Analytics)
│   ├── data/                  # Market data ingestion + filters
│   ├── calibration/           # Production SVI toolkit
│   ├── validation/            # Walk-forward backtesting utilities
│   ├── models/                # SABR/local-vol benchmarking
│   └── surface/ + rates/ + core/ etc.
├── tools/
│   ├── download_spy_chains.py # CLI helper to pull Yahoo option chains
│   └── compute_metrics.py     # Resume metrics generator
├── tests/                     # Pytest suite (calibration, data, backtests)
└── data/
    ├── examples/              # Sample CSVs for the UI
    └── metrics/               # Gitignored Parquet + CSV outputs
```

---

## What’s inside

- **Market data ingestion**: yfinance downloader + CLI utility (`tools/download_spy_chains.py`) with liquidity filters, moneyness tagging, and gitignored Parquet cache.
- **Production calibration**: constrained SVI engine with spline benchmarking, diagnostics, and headless metrics pipeline (`tools/compute_metrics.py`).
- **Walk-forward + model lab**: regime-aware validation, SVI/SABR/local-vol comparison utilities, and Streamlit Analytics tabs that surface the full pipeline.
- **Fixed-income/rates stack**: OIS/IRS curve building, cap/floor analytics, PV/DV01/PV01 sensitivities, and scenario shocks (parallel/twist).
- **Deployment plumbing**: Streamlit UI, GitHub Actions CI, Selenium wake-up workflow, and a README aimed at recruiters/hiring managers.

---

## Tests

- **Put–call parity** is checked numerically on every load (residual ~1e-8)
- **Round-trip price ↔ implied vol** via `vol/implied.py` returns <1e-6 error
- **SABR calibration** on synthetic data recovers parameters within ±10%
- **DV01/PV01** analytics match bump-and-revalue within 2 bps
- Eight targeted unit tests cover the core price/calibration logic

Run the suite locally:

```bash
pytest tests/ -v
```

---

## Streamlit UI highlights

![FX overview](docs/fx-overview.png)

![Equity parity card](docs/equity-parity.png)

![SABR smile](docs/fx-sabr-smile.png)

![Rates overview](docs/rates-overview.png)

![Rates scenarios](docs/rates-scenarios.png)


---

## Packaging and deployment

The project is configured for:

- **Local runs** via `streamlit run`
- **Docker deployments** via the included `Dockerfile`
- **Render** via `render.yaml` (one-click deploy)
- **Heroku-style deployments** via the included `Procfile` and `setup.sh`
- **Continuous integration** (pytest) through `.github/workflows/ci.yaml`
- **Streamlit wake-up automation** via `.github/workflows/wake-streamlit.yml`

### Deploy to Render

1. Fork this repository
2. Go to [render.com](https://render.com) and create a new Web Service
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Click **Deploy** — the app will be live at `https://derivx.onrender.com`

### Deploy with Docker

```bash
# Build and run locally
docker build -t derivx .
docker run -p 8501:8501 derivx

# Or deploy to any container platform (Railway, Fly.io, Google Cloud Run, etc.)
```

### Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Upload the `src/` folder and `requirements.txt`
4. Set the app file to `src/derivx/ui/streamlit_app.py`

To package the library (optional):

```bash
python -m build
```

---

## Market data + analytics lab

The new **Analytics** mode in `streamlit_app.py` exposes the end-to-end pipeline built for real option chains:

1. **Market Data tab** – upload a Parquet chain or click “Pull from Yahoo Finance”. The app runs liquidity filters, shows summary metrics, and lets you download a cleaned CSV.
2. **Production Calibration tab** – pick expiries, enforce moneyness bands/no-arb checks, and visualise fitted SVI smiles vs a cubic-spline baseline (with RMSE/diagnostics tables).
3. **Walk-Forward Validation tab** – either upload a folder of daily chains or fetch a short window (≤20 business days) from Yahoo Finance, reapply filters, and run `walk_forward_test` with regime-level summaries.
4. **Model Comparison tab** – load a single expiry slice and compare in-sample RMSE for SVI/SABR/local-vol with an overlayed smile plot.

> **Yahoo Finance caveats:** the public options endpoint only returns *current* chains and occasionally responds with “Invalid Crumb”/empty payloads. Each download attempt now retries 3× and surfaces a per-date status table. When a window keeps failing, run the offline helper and upload the files instead:

```bash
# Download a few SPY sessions into data/market/manual
python tools/download_spy_chains.py \
    --ticker SPY \
    --start 2025-11-10 \
    --end 2025-11-14 \
    --outdir data/market/manual
```

The Parquet files are gitignored; keep your historical downloads locally and use the upload flow whenever Yahoo throttles the live API.

---

## Key Metrics & Performance

Quantified results from production calibration and backtesting on 24 trading days of SPY option chains (768 expiry slices):

- **SVI Calibration**: Achieved mean RMSE of 0.0520 (median 0.0401) across 32 unique expiries, with constrained no-arbitrage diagnostics enforced on all slices.
- **Walk-Forward Validation**: Out-of-sample backtest achieved 71.8% directional hit rate with mean RMSE of 0.0558 across 416 predictions in mid-volatility regimes (10-day calibration window).
- **Model Comparison**: SVI outperformed SABR by 25.3% in mean RMSE (0.2892 vs 0.3869) across 24 representative smile calibrations, demonstrating superior fit to equity option market data.

Metrics computed via `tools/compute_metrics.py`; full results available in `data/metrics/`. Regenerate with:
```bash
python tools/compute_metrics.py --chain-dir data/metrics/chains --outdir data/metrics
```

