"""Market-data layer.

Equity options (the ANALYTICS lab): live Yahoo option chains with lineage,
the chain schema and upload validation, the bundled real sample, a labelled
synthetic history and the exchange calendar. Rates: CSV quote loaders and the
rates market-data layer (:mod:`rates_sources`, :mod:`rates_history`).

The names below are imported on first use, so the rates modules and the
``derivx`` CLI do not pay for importing yfinance.
"""

from importlib import import_module

_LAZY = {
    "ChainDataset": "chain_schema",
    "ChainSchemaError": "chain_schema",
    "chain_to_csv": "chain_schema",
    "load_chain_file": "chain_schema",
    "merge_datasets": "chain_schema",
    "read_chain_file": "chain_schema",
    "validate_chain": "chain_schema",
    "ChainFetchError": "ingestion",
    "load_equity_chain_from_yf": "ingestion",
    "load_rates_quotes_from_csv": "ingestion",
    "load_curve_quotes_from_csv": "ingestion",
    "Backoff": "market_data",
    "ChainFetch": "market_data",
    "FetchStatus": "market_data",
    "apply_liquidity_filters": "market_data",
    "compute_moneyness": "market_data",
    "fetch_option_chain": "market_data",
    "hygiene_report": "market_data",
    "load_sample_chain": "samples",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
