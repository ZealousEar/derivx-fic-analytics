# Validation: tests, lab metrics and performance

Back to the [README](../README.md). Related pages: [capabilities](capabilities.md) (which test holds the
evidence for each feature), [data sources](data-sources.md), [CLI and deploy](cli-and-deploy.md),
[limitations](limitations.md).

## Contents

- [Test suite](#test-suite)
- [What the tests check](#what-the-tests-check)
- [CI](#ci)
- [Browser checks](#browser-checks)
- [Lab metrics (reproducible)](#lab-metrics-reproducible)
- [Collected option-chain history](#collected-option-chain-history)
- [Walk-forward method](#walk-forward-method)
- [Performance](#performance)

## Test suite

Development install (tests, the QuantLib oracle, linters) and notebooks:

```bash
pip install -e ".[dev]"                  # pytest, QuantLib 1.44, ruff 0.17.1, mypy 2.4.0, black
pip install -e ".[dev,notebooks]"        # + matplotlib, ipykernel, nbmake
pytest                                   # whole suite, offline
pytest --nbmake notebooks                # execute the three notebooks
ruff check . && ruff format --check .
mypy                                     # src/derivx, with the lenient config in pyproject.toml
```

Test run on this revision: `pytest -W error::DeprecationWarning -W error::RuntimeWarning` collects 1,540 tests
from 79 files and took about 22 minutes here on 4 cores; run single files while iterating.

| Environment | Result |
|---|---|
| Python 3.11.17 with the dev extra: numpy 1.26.4, pandas 2.1.4, SciPy 1.17.1, Streamlit 1.66.0, Plotly 7.1.0, yfinance 0.2.66, pyarrow 13.0.0 (older than the declared lower bound 14), QuantLib 1.44 (so the live QuantLib checks run); `-W error::DeprecationWarning -W error::RuntimeWarning` | 1,540 passed, 0 skipped, 0 failed |
| `pytest --nbmake notebooks` (Python 3.11.17, numpy 2.4.6, pandas 2.3.3, SciPy 1.17.1; run on the previous revision, and the notebooks use none of the modules changed since) | 3 passed |
| `ruff check .` and `ruff format --check .` (ruff 0.17.1, the CI pin) | clean (219 files formatted) |
| `mypy` (2.4.0, the CI pin, on the local versions above) | clean (no issues in 105 source files) |
| `actionlint` 1.7.12 with shellcheck 0.11.0 on `.github/workflows/*` | clean |

Notebooks (`notebooks/`, executed with outputs committed): 01 SVI on a synthetic slice and then on the real
bundled SPY chain; 02 SABR across expiries with beta = 1; 03 rates: a SOFR OIS curve from illustrative quotes and
the real Treasury curve, a 5Y normal-vol cap with put-call parity, and a PV01 whose sign matches the +/-100bp P&L.

## What the tests check

- **QuantLib golden fixtures.** `tools/gen_golden.py` with the modules in `tools/golden/` writes the 15 files
  `tests/golden/<area>.json` from QuantLib 1.44 (calendars, curves, multi-curve, swaps, caps/floors including
  compounded RFR caplets, cap vol stripping, swaptions, bonds, Black-Scholes, Black IR, implied vols, SABR/SVI, FX
  conventions and pricing, the rates market-data conversions, the example trade book, forward swap rates for
  realised vol). The tests read the committed JSON, so the suite needs no QuantLib; with QuantLib installed
  (`pip install -e ".[dev]"`) the same tests also run randomised live QuantLib checks. Regenerate with
  `python tools/gen_golden.py` (all) or `python tools/gen_golden.py swaps capvol` (some). The
  [capability matrix](capabilities.md) links each feature to the test that holds its evidence.
- **Property and regression tests**: quotes reprice, discount factors positive, cap - floor = swap, collar = cap -
  floor, payer - receiver = forward swap, key-rate buckets sum to PV01, P&L explain against full revaluation,
  put-call parity on forwards, implied-vol round trips, SABR parameter recovery, no-arbitrage checks that fail
  where they should, converted totals against hand conversion.
- **UI tests**: Streamlit AppTest renders every mode and every Rates and Analytics tab with default and edge
  inputs, and takes inputs through a round trip of the other modes; [tests/test_plot_limits.py](../tests/test_plot_limits.py)
  asserts every chart is zoom-bounded; [tests/test_theme.py](../tests/test_theme.py) checks the theme's text
  colours and heatmap scales against WCAG AA.
- **Hermetic**: an autouse fixture blocks network access, points the cache at a temporary directory and removes
  `FRED_API_KEY`; parsers are tested on recorded fixtures under `tests/fixtures/`.
- **Deploy**: [tests/test_deploy.py](../tests/test_deploy.py) checks the Docker, Render, Procfile and devcontainer
  configs (XSRF and CORS on, non-root image, healthcheck), the CI pins and `-W` flags, and starts a live server on
  the committed config to confirm it refuses a token-less upload and sends no wildcard CORS header. See
  [CLI and deploy](cli-and-deploy.md#deploy).

## CI

CI ([.github/workflows/ci.yaml](../.github/workflows/ci.yaml)) runs on pushes and pull requests to `main` or
`master` and by hand; it has not run on this revision, which has not been pushed or opened as a pull request. Its
jobs:

- **lint**: ruff 0.17.1 check and format over the whole repository.
- **typecheck**: mypy 2.4.0 over `src/derivx`, installed from `requirements-lock.txt`; a lenient start: the 11
  error codes still off and an `attr-defined` baseline of 18 rates, CLI and rates-UI modules are listed in
  `[tool.mypy]`.
- **test**: Python 3.10, 3.11, 3.12 and 3.13 with the newest versions the ranges allow and the dev extra (live
  QuantLib checks), deprecation and runtime warnings as errors.
- **oldest**: Python 3.10 with every direct dependency at its lower bound, same flags.
- **notebooks**: nbmake.
- **docker**: builds the test stage (the UI tests, the swap, cap, bond and scenario tests and a live-server check,
  run inside the image) and the runtime image, waits for the healthcheck and checks XSRF and CORS on the running
  container.

## Browser checks

Not in CI. On the previous revision every mode, tab and sub-tab was driven in Chromium with Playwright at 390,
1280 and 1920 px (63 views), plus the timed runs under [Performance](#performance). No page exceptions, console
errors, Streamlit exception blocks, error alerts or horizontal overflow; every chart declares zoom bounds; no
text node below WCAG AA contrast.

On this revision the changed views were driven at 1280 and 390 px (12 each). ANALYTICS > WALK-FORWARD VALIDATION
was checked with Collected history in four states: not synced, a real SYNC NOW (the expected 404 message, 1.2 s),
two real sessions (held back) and 12 SYNTHETIC sessions (run, 22-23 s). Also checked were MARKET DATA loading a
collected session and PRODUCTION CALIBRATION on it, and RATES > BOOK and RESEARCH > VAR on a euro book reported
in USD with FX moving and held. They showed the same absence of exceptions, console or page errors, error alerts
(apart from the intended sync failure) and horizontal overflow.

After the review fixes, 10 views per width were driven again at 1280 and 390 px: 14 SYNTHETIC sessions (labelled
SYNTHETIC with the orange edge), SESSIONS = most recent 10 and its run (13 s), a real collected SPY session
(labelled as Yahoo's delayed quotes, held back, then loaded in MARKET DATA), a server with nothing synced, and
RATES > BOOK and RESEARCH > VAR with FX; none showed an exception, a console or page error, an error alert or
horizontal overflow.

## Lab metrics (reproducible)

Earlier versions of the README quoted SVI/SABR and walk-forward figures (a 71.8% hit rate, 0.052 RMSE, "SVI beats
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
- These are single-snapshot fit metrics, not forecasts. Full tables:
  [metrics/spy_2026-10-20/METRICS_SUMMARY.md](metrics/spy_2026-10-20/METRICS_SUMMARY.md), with a
  [manifest](metrics/spy_2026-10-20/manifest.json) recording the commit, packages, inputs, discount curve and
  data hash. `--discount curve|parity|rate` (default `curve`) and `--rate` choose the discounting.
- SABR parameter recovery from exact quotes: `pytest tests/test_sabr.py -k recovers` (alpha and nu to 1e-5
  relative, rho to 1e-5 absolute, F from 1.08 to 15,800, T from 1W to 5Y).

## Collected option-chain history

**Walk-forward results on real data do not exist yet: the history is collected from the merge to `main` on.**
Yahoo serves only the current chain, so a real history has to be saved one session at a time.
[.github/workflows/collect-data.yml](../.github/workflows/collect-data.yml) does that:

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

Using the history (the `derivx snapshots` commands are listed in [CLI and deploy](cli-and-deploy.md#command-line)):

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
`data-snapshots`.

In the lab, WALK-FORWARD VALIDATION's default source is **Collected history**: the synced copy (a SYNC NOW button
fetches it) or, with `$DERIVX_SNAPSHOT_DIR` set, a local checkout of the branch. It shows coverage and provenance
(sessions, first and last, missing trading days, quotes and size, source and sync time, and what the rows are,
read from each file's manifest: Yahoo's delayed quotes, or SYNTHETIC rows named and edged as such; the run's
banner and manifest add the index SHA-256). SESSIONS picks the most recent 10 to 250 sessions or all of them
(default 30): a run takes about 4 s a session on a real SPY chain at the default settings (12 sessions: 44 s), so
a year of history would take about 15 minutes in the browser; `compute_metrics.py --history --first ... --last
...` runs any range offline. Until the chosen sessions give 9 origin days at the chosen horizon (three per
vol-regime tercile; 10 consecutive sessions at 1 day), it says how many there are and how many more are needed,
and the run stays disabled. MARKET DATA loads any collected session as the active dataset, so PRODUCTION
CALIBRATION, VOL SURFACE and MODEL COMPARISON run on it.

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

## Walk-forward method

Each origin is scored against the snapshot exactly N NYSE trading days later, matching quotes on (expiry, strike,
option type); the calibration window pools the last W snapshots. Predictors: SVI sticky strike against
persistence, and SVI sticky moneyness against raw sticky moneyness. Scores: pooled RMSE (vol bp), skill = 1 -
RMSE / benchmark, days won with a Wilson 95% interval, a day-clustered Diebold-Mariano t with the
Harvey-Leybourne-Newbold correction, a noise floor (RMS half bid/ask in vol) and regimes (terciles of a 30-day
ATM vol).

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
