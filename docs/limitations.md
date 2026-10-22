# Known limitations

Back to the [README](../README.md). Related pages: [capabilities](capabilities.md) (status of every feature:
implemented, approximate or planned), [data sources](data-sources.md), [validation](validation.md),
[CLI and deploy](cli-and-deploy.md).

## Not built

- G18: no rates strategy backtester (carry/roll, steepener or PCA signals, DV01-neutral sizing).
- G19: no meeting-date STIR curve (no FOMC / ECB meeting calendar, step forward curve or hike/cut probabilities);
  SR1/SR3 and EURIBOR futures are curve inputs and trades only.
- Seasoned trades (past fixings) are rejected; no XLSX export (openpyxl is not a dependency).

## Data

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
- Walk-forward metrics on real data wait for the collected history
  (see [validation](validation.md#collected-option-chain-history)). Collection starts only when
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

## Models

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

## Platform

- pandas 3 and numpy 2.5 are not supported yet (capped). mypy runs in CI with a lenient config (11 error codes
  off, an `attr-defined` baseline of 18 modules).
- The GitHub workflows have not run on this revision (see [CI](validation.md#ci)). `collect-data.yml` has
  never run on GitHub. It passes actionlint 1.7.12 with shellcheck 0.11.0, and its git steps (orphan branch,
  merge, commit, rebuilt push) are tested against a local bare repository. Its step scripts were also run end to
  end here, unchanged, against a local bare origin with real Yahoo pulls of the 2026-10-21 session: a first run
  with two tickers (the orphan branch, one commit), the same session again (no fetch, no commit), Thanksgiving
  and a Saturday (no commit), Yahoo unreachable (red, no commit), a valid and an invalid ticker (the valid one
  committed, the run red) and a dry run (nothing pushed). The `data-snapshots` branch appears with its first run
  on `main`; until then `derivx snapshots sync` and the lab's SYNC NOW report a 404.
- Inputs survive mode switches, but the selected tab inside a page is not kept.
- Yahoo rate-limits shared IPs; lab pulls back off and say so. Yahoo's terms of use govern its data.
