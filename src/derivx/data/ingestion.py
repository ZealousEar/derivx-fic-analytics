from __future__ import annotations

from typing import Dict, List

import pandas as pd


def load_equity_chain_from_yf(ticker: str, date: str) -> pd.DataFrame:
    """Placeholder: fetch option chain via yfinance (to be implemented)."""

    raise NotImplementedError("yfinance integration deferred; use sample CSVs")


def load_rates_quotes_from_csv(path: str) -> Dict[str, float]:
    """Single-curve quotes CSV (columns ``instrument,tenor,rate``, rates in
    decimals) as ``{tenor: rate}``.

    One quote per tenor: a tenor quoted twice raises ``ValueError``. (This
    loader used to average them, e.g. a 2Y OIS at 4.0% and a 2Y IRS at 5.0%
    became 4.5%, blending two curves' quotes into one.) Quotes for several
    curves go through :func:`load_curve_quotes_from_csv`.
    """

    df = pd.read_csv(path)
    if not {"instrument", "tenor", "rate"}.issubset(set(df.columns)):
        raise ValueError("CSV must contain instrument, tenor, rate columns")
    tenors = df["tenor"].astype(str).str.strip().str.upper()
    repeated = sorted(set(tenors[tenors.duplicated(keep=False)]))
    if repeated:
        raise ValueError(
            f"tenors {repeated} are quoted more than once; keep one quote per tenor (OIS and IRS quotes belong to "
            "different curves: give them a 'curve' column and use load_curve_quotes_from_csv)"
        )
    return {str(k): float(v) for k, v in zip(df["tenor"], df["rate"])}


def load_curve_quotes_from_csv(path: str, units: str = "auto") -> Dict[str, List[object]]:
    """Quotes for a curve set from a CSV with columns ``instrument, tenor,
    rate`` and an optional ``curve`` (the discount curve, e.g. ``EUR ESTR``,
    or a projection index, e.g. ``EURIBOR 6M``; one curve if absent):
    ``{curve: [CurveQuote, ...]}`` in decimals, each curve's rows validated
    with :func:`derivx.rates.quotes.clean_quotes` (its instruments, units,
    tenors, duplicates). ``ValueError`` lists every problem."""

    from derivx.rates.curves import INSTRUMENT_KINDS
    from derivx.rates.indices import INDICES
    from derivx.rates.multicurve import PROJECTION_KINDS
    from derivx.rates.quotes import clean_quotes

    df = pd.read_csv(path)
    if not {"instrument", "tenor", "rate"}.issubset(set(df.columns)):
        raise ValueError("CSV must contain instrument, tenor, rate columns (and optionally curve)")
    names = df["curve"].astype(str).str.strip().str.upper() if "curve" in df.columns else pd.Series(["CURVE"] * len(df))
    out: Dict[str, List[object]] = {}
    errors: List[str] = []
    for name in dict.fromkeys(names):
        rows = df[names.values == name].to_dict("records")
        kinds = PROJECTION_KINDS if name in INDICES else INSTRUMENT_KINDS
        quotes, issues, _ = clean_quotes(rows, units=units, kinds=kinds)
        errors += [f"{name}: {i.message}" for i in issues if i.level == "error"]
        out[name] = quotes
    if errors:
        raise ValueError("; ".join(errors))
    return out
