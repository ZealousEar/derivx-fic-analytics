# Install, command line and deploy

Back to the [README](../README.md). Related pages: [capabilities](capabilities.md),
[data sources](data-sources.md) (what the `pull` commands fetch and the environment variables),
[validation](validation.md) (tests, CI, the collected history), [limitations](limitations.md).

## Contents

- [Quick start](#quick-start)
- [Dependency policy](#dependency-policy)
- [Command line](#command-line)
- [Deploy](#deploy)

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
LIVE (CACHED) for current FRED, NY Fed, ECB and BoE data (see [data sources](data-sources.md)).

The development install (tests, the QuantLib oracle, linters, notebooks) is in
[validation](validation.md#test-suite).

## Dependency policy

`pyproject.toml` holds the supported ranges and `requirements.txt` repeats them
([tests/test_deploy.py](../tests/test_deploy.py) checks they agree). `requirements-lock.txt` pins the resolution
the Docker image and the CI type check are built from; regenerate it with
`uv pip compile pyproject.toml --python-version 3.11 --python-platform linux -o requirements-lock.txt --upgrade`.
numpy is capped below 2.5 and pandas below 3 until they are supported.

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

# the daily option-chain history on the data-snapshots branch (see validation.md)
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

The `snapshots` commands and `tools/compute_metrics.py --history` are described with the collection workflow in
[validation](validation.md#collected-option-chain-history).

### What the commands did on this revision

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

### Shock specs

From `derivx scenario --help`: `parallel:BP`, `twist:PIVOT:SHORT_BP:LONG_BP`,
`fly:BODY:BODY_BP[:WING_BP[:LEFT:RIGHT]]`, `keyrate:2Y=+10,10Y=-5`, `pca:FACTOR[:N_SIGMA[:DAYS]]`,
`historical:NAME[:SCALE]` (episodes: 2013 taper tantrum, 2020 dash for cash, 2022 hiking cycle, Sept 2022 gilt
crisis week, 2023 SVB), `window:START:END[:SCALE]`, `basis:BP`, `tenorbasis:BP`, `fx:EURUSD=+10[,GBPUSD=-5]`
(% moves of FX spots), `library`.

### Currencies

Every trade's numbers are in its currency, with a `TOTAL <CCY>` per currency. `--report-ccy {USD,EUR,GBP}`
(default USD) adds `ALL IN <CCY>`, the book converted at the ECB reference rates (the bundled history plus any
`pull ecb-fx` in the cache) of the latest day on or before the valuation date (the first day held, for an earlier
date; a note names the day when it is another one) or at `--fx` spots. `--conventions all` prices each
currency's trades on its own market (USD SOFR, EUR ESTR, GBP SONIA; Treasuries on the US Treasury curve). FX
moves change only the converted totals. `historical:` and `window:` replays move spots by the ECB rates of the
episode's dates (`--hold-fx` holds them), `fx:` shocks move spots per scenario, and `--fx-move` adds its moves to
every scenario, on top of a shock's own move of the same pair (an `fx:EURUSD=+10` shock with `--fx-move
EURUSD=+2` moves EURUSD +12%), and is named in each scenario label. The output lists the FX moves of every
scenario that has any. A total move of -100% or less is refused. Spots are held otherwise.

### Vols and hedges

Caps and swaptions in a blotter need vols: a `vol` column per trade, or `--swaption-vols` / `--cap-vols` files, or
`--illustrative-vols` (labelled). Without them those trades are skipped with the reason. Hedges: `--hedge`
solves each currency's curves with that currency's instruments; with `--bid-ask` the optimiser weighs each hedge's
cost against the risk it removes for curve moves of `--move` bp, and it returns no hedge (residual = unhedged risk,
zero cost) when none is worth its bid/ask.

### Exit codes

0 ok; 1 a data source failed (`pull latest` also when any one source failed, after falling back to the cache or
the snapshot); 2 bad arguments, missing files, a malformed curve JSON or a date with no curve data; 3 blotter rows
with errors or trades that could not be priced (the rest are).

### File formats

`curve.json` is the `derivx.curve_set` v1 format: as-of date, conventions, interpolation, convexity settings,
fixings and every curve's quotes. Blotter CSVs accept comma, semicolon, tab or bar delimiters; commas inside
numbers are thousands separators only (`1,000,000`), and a decimal comma is refused with a message.

### Other tools

`tools/gen_golden.py` with `tools/golden/` (regenerate the QuantLib fixtures), `tools/compute_metrics.py` (lab
metrics; `--history` on the collected snapshots) and `tools/download_spy_chains.py` (one chain by hand),
`tools/build_rates_snapshot.py` (bundled rates data and the ECB FX history; `tools/build_historical_shocks.py` is
a shim that runs it), `tools/record_rates_fixtures.py` (parser fixtures), `tools/gen_fx_illustrative_quotes.py`
and `tools/gen_rates_vol_illustrative.py` (print the ILLUSTRATIVE FX and rates vol quotes), `tools/keep_alive/`
(the hosted app's wake-up check).

## Deploy

Hosted demo: https://derivx.streamlit.app/ (Streamlit Community Cloud, deployed from `main`, so it can lag the
repository; it may need a minute to wake up).

All deploy paths read the committed `.streamlit/config.toml`, which keeps XSRF and CORS protection on, sets the
dark theme, and turns off Streamlit's widget-state duplication warning: inputs that survive mode switches are
written back on purpose, and without the setting the server logs a long, harmless stack trace once on the first
mode round trip. Secrets go in `.streamlit/secrets.toml`, which git and `.dockerignore` keep out; the app itself
reads no secrets file, only environment variables (`FRED_API_KEY`, `DERIVX_CACHE_DIR`, `DERIVX_OFFLINE`, and
`DERIVX_GIT_SHA`, which the Docker image sets from its `GIT_SHA` build argument for run manifests). The deploy
configs are checked by [tests/test_deploy.py](../tests/test_deploy.py) (see [validation](validation.md)).

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

**Keep-alive**: [.github/workflows/keep-alive.yml](../.github/workflows/keep-alive.yml) opens the Community Cloud
app in headless Chrome every six hours, wakes it if it is asleep, and fails unless the sidebar heading NAVIGATION
renders with no Streamlit exception. It makes no commits and does not ingest data. Data collection is a separate
workflow, `collect-data.yml` (see [validation](validation.md#collected-option-chain-history)), which commits only
to the `data-snapshots` branch.
