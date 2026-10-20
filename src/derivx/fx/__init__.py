"""FX options with market conventions: currency pairs, spot and delivery
dates, Garman-Kohlhagen pricing with spot/forward and premium-adjusted
deltas, and smiles built from ATM / risk-reversal / butterfly quotes.

* :mod:`derivx.fx.conventions` - pairs, calendars, spot / expiry / delivery
  dates, forwards and forward points.
* :mod:`derivx.fx.pricing` - Garman-Kohlhagen prices and Greeks, the four
  delta conventions, strike from delta, ATM strikes, premium quote styles.
* :mod:`derivx.fx.smile` - smiles from ATM, RR25, BF25, RR10, BF10 quotes
  (market or smile strangle) with SABR (beta = 1) or vanna-volga.
* :mod:`derivx.fx.market_data` - ECB euro reference rates (live, cached,
  with a bundled snapshot as the offline fallback).
"""
