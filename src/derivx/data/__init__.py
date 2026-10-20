"""Data ingestion: equity option chains (yfinance), CSV rates quotes and the
rates market-data layer (:mod:`rates_sources`, :mod:`rates_history`).

The equity helpers are imported on first use, so the rates modules and the
``derivx`` CLI do not pay for importing yfinance.
"""

from importlib import import_module

_LAZY = {
    "load_equity_chain_from_yf": "ingestion",
    "load_rates_quotes_from_csv": "ingestion",
    "fetch_option_chain": "market_data",
    "apply_liquidity_filters": "market_data",
    "compute_moneyness": "market_data",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
