# DerivX documentation

The [README](../README.md) is the short tour. The reference pages here hold the detail: what is built and how
each piece is validated, where the data comes from, what the test and metric runs produced, how to run and
deploy, and what the package does not do.

| Page | What it covers |
|---|---|
| [capabilities.md](capabilities.md) | The capability matrix: every feature of the rates, FX, equity, volatility and lab code with its status (implemented, approximate, planned) and the test that holds the evidence. |
| [data-sources.md](data-sources.md) | Every bundled file with its source and terms, which inputs are real market data and which are ILLUSTRATIVE, PROXY or SYNTHETIC, the RATES data-source modes, FX reference rates, and the environment variables. |
| [validation.md](validation.md) | The test suite and its result on this revision, the QuantLib golden fixtures, CI, browser checks, the reproducible lab metrics, the collected option-chain history and walk-forward method, and measured performance. |
| [cli-and-deploy.md](cli-and-deploy.md) | Quick start, dependency policy, the `derivx` command line (every example run on this revision, shock specs, currencies, exit codes, file formats), and the Docker, Render, Procfile, Hugging Face, Community Cloud and devcontainer deploy paths. |
| [limitations.md](limitations.md) | Known limitations: what is not built, the data gaps, the model simplifications and the platform caveats. |

Also under `docs/`:

- [metrics/spy_2026-10-20/](metrics/spy_2026-10-20/METRICS_SUMMARY.md): the committed output of
  `python tools/compute_metrics.py --outdir docs/metrics/spy_2026-10-20`, with its manifest (commit, packages,
  inputs, discount curve and data hash). How to regenerate it is in [validation.md](validation.md#lab-metrics-reproducible).
- `screenshots/`: the app views shown in the README, taken with Playwright (Chromium, 1280 px wide; the phone
  view is 390 px).

Licence: MIT (see [LICENSE](../LICENSE)). Data keep their publishers' terms (see
[data-sources.md](data-sources.md)).
