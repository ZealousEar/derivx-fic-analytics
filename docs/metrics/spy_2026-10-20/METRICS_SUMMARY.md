# Smile-model metrics

Data: bundled sample: real SPY chain from Yahoo Finance (delayed quotes), one snapshot; ticker SPY; trade date(s) 2026-10-20; fetched 2026-10-20T20:59:39+00:00; 10,040 rows; sha256 2fa7d599651ed6564caa6a3d4501dd228bffec7b1f30dfbbed660b30be3188b9.
Filters: {'min_volume': 10, 'min_mid': 0.1, 'max_spread': 0.4}. Settings: {'min_days': 2.0, 'max_days': None, 'sabr_beta': 1.0, 'cv_folds': 5, 'std_band_sd': 3.0, 'min_points': 8, 'expiries': 'all'}.
Discounting: US Treasury par curve of 2026-10-19 (US Treasury, bundled snapshot), bootstrapped (trade date to expiry; 1 trade date(s)).

SPY options are American-style; vols are re-inverted from mids with European Black-76. At a real discount rate the early-exercise premium of the out-of-the-money puts (largest at long expiries) reads as extra put vol below the forward, a kink the smiles cannot fit; with r = 0 that premium mostly vanished, but every vol was biased low (30-50 vol bp at one year).

## Model comparison (one row per expiry, out-of-sample k-fold CV RMSE, vol bp)

Slices scored: 29; skipped: 2 (see skipped.csv).

| model | slices | failed | median | mean | max | lowest on |
|---|---|---|---|---|---|---|
| SVI | 29 | 0 | 5.1 | 9.3 | 52.3 | 1 |
| SABR | 29 | 0 | 16.2 | 18.3 | 58.8 | 0 |
| IV spline | 29 | 0 | 1.7 | 5.8 | 71.9 | 28 |

- SVI has the lower out-of-sample RMSE on 25 of 29 expiries (median SVI - SABR = -10.6 vol bp; SVI better at the median).
- IV spline has the lower out-of-sample RMSE on 28 of 29 expiries (median SVI - IV spline = +3.2 vol bp; IV spline better at the median).
- IV spline has the lower out-of-sample RMSE on 28 of 29 expiries (median SABR - IV spline = +14.2 vol bp; IV spline better at the median).

The spline interpolates the quotes, so it is scored out of sample only; with dense strikes it is a strong interpolation baseline. These are single-snapshot fit metrics, not forecasts.

## Production SVI calibration (arbitrage repair on)

- Slices: 29; median in-sample RMSE 6.0 vol bp (max 42.0).
- Butterfly (g(k) >= 0): 29 of 29; Lee wings: 29 of 29; calendar: 25 of 28 adjacent expiry pairs (fails at 2027-03-19, 2027-06-30, 2027-12-17).

## Walk-forward

not run: 1 snapshot(s): a walk-forward test needs daily snapshots one trading day apart. Yahoo has no history; collect one snapshot per trading day with tools/download_spy_chains.py.
