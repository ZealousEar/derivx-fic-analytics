"""Starting swap-curve quotes for a valuation date, with per-row provenance.

:func:`swap_prefill` picks, for a curve's currency:

* GBP - the Bank of England's fitted SONIA OIS curve, converted to par OIS
  rates (real market data, ``kind="real"``);
* EUR - par OIS rates derived from the ECB AAA government spot curve, a
  labelled PROXY (government yields differ from OIS by the swap spread);
* USD (and any currency without a free curve) - the ILLUSTRATIVE SOFR set
  (not market data: there is no free source of SOFR swap quotes).

The currency's overnight fixing, where the history store has one, is the
``ON`` deposit quote. Rates in the table are percent. This module has no UI
dependency: the Rates page (:mod:`derivx.ui.rates_data`) and the command
line (:mod:`derivx.rates.curve_io`) both start from it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from derivx.data import rates_history as rh
from derivx.data.rates_history import HistoryStore
from derivx.rates.conventions import parse_tenor
from derivx.rates.curves import CONVENTIONS
from derivx.rates.examples import EXAMPLE_ASOF, sofr_example_rows

ILLUSTRATIVE = "ILLUSTRATIVE (not market data)"
FIXING_LABELS = {
    "SOFR": "SOFR (NY Fed)",
    "EFFR": "EFFR (NY Fed)",
    "OBFR": "OBFR (NY Fed)",
    "ESTR": "ESTR (ECB)",
    "SONIA": "SONIA (Bank of England via FRED)",
}
PROXY_DATASET = {"EUR": "EUR_AAA_ECB"}  # zero curves used as labelled proxies where no OIS curve is free
REAL_DATASET = {"GBP": "GBP_OIS_BOE"}  # real OIS curves
SHORT_CURVE_YEARS = 10.0  # a prefilled curve ending before this says that the rest is extrapolated


@dataclass(frozen=True)
class QuotePrefill:
    """Swap quotes to start the editor from: ``table`` has ``instrument``,
    ``tenor``, ``rate`` (percent) and ``source`` (per row). ``kind`` is
    ``real``, ``proxy``, ``illustrative`` or ``upload``."""

    table: pd.DataFrame
    label: str
    kind: str
    key: str
    notes: Tuple[str, ...] = ()


def _percent(rate: float) -> float:
    return round(100.0 * rate, 6)


def _fixing_row(store: HistoryStore, currency: str, asof: dt.date) -> Tuple[Optional[dict], Optional[str]]:
    name = rh.CURRENCY_FIXING.get(currency)
    got = store.fixing(name, asof) if name else None
    if got is None:
        return None, None
    day, rate, _ = got
    label = f"{FIXING_LABELS.get(name, name)} fixing for {day}"
    return {"instrument": "DEPO", "tenor": "ON", "rate": _percent(rate), "source": label}, label


def _illustrative_rows() -> List[dict]:
    return [
        {"instrument": r["kind"], "tenor": r["tenor"], "rate": _percent(r["rate"]), "source": ILLUSTRATIVE}
        for r in sofr_example_rows()
    ]


def swap_prefill(store: HistoryStore, conventions: str, asof: dt.date) -> QuotePrefill:
    """Starting quotes for the swap curve on ``asof`` (see the module
    docstring)."""

    currency = CONVENTIONS[conventions].currency
    fixing, fixing_label = _fixing_row(store, currency, asof)
    notes: List[str] = []
    rows: List[dict] = []
    kind, label = "illustrative", ""
    dataset = REAL_DATASET.get(currency) or PROXY_DATASET.get(currency)
    day = store.latest_on_or_before(dataset, asof, "zero_cc") if dataset and store.has(dataset) else None
    if day is not None:
        try:
            par = store.ois_par(dataset, day)
        except ValueError as exc:
            par = {}
            notes.append(f"{rh.dataset_info(dataset).title} on {day} could not be converted ({exc})")
        if par:
            info = rh.dataset_info(dataset)
            kind = "real" if dataset in REAL_DATASET.values() else "proxy"
            tag = f"{info.title} {day}, par OIS derived"
            source = tag if kind == "real" else f"PROXY: {tag} (government curve, not OIS quotes)"
            rows = [{"instrument": "OIS", "tenor": t, "rate": _percent(r), "source": source} for t, r in par.items()]
            label = source
            if day != asof:
                notes.append(f"{info.title}: latest curve on or before {asof} is from {day}")
            last = list(par)[-1]
            if parse_tenor(last).years < SHORT_CURVE_YEARS:
                notes.append(
                    f"The {info.title} on {day} reaches only {last}: longer rates on this curve are flat-forward "
                    "extrapolations, not market data."
                )
    if not rows:
        if dataset:
            notes.append(
                f"No spot (zero) rates of the {rh.dataset_info(dataset).title} on or before {asof} in the store: "
                "showing the illustrative USD SOFR set instead"
            )
        rows = _illustrative_rows()
        label = "ILLUSTRATIVE quotes (not market data)"
        if asof != EXAMPLE_ASOF:
            notes.append(
                f"The illustrative quotes are fixed at {EXAMPLE_ASOF} levels: they do not follow the valuation date "
                f"{asof}. Paste or upload quotes for that date."
            )
    if fixing is not None:
        rows = [r for r in rows if r["tenor"] not in ("ON", "1D")]
        rows.insert(0, fixing)
        label = f"{label} + {fixing_label}"
    table = pd.DataFrame(rows, columns=["instrument", "tenor", "rate", "source"])
    digest = hashlib.sha1(table.to_csv(index=False).encode()).hexdigest()[:10]
    return QuotePrefill(table, label, kind, f"{conventions}_{digest}", tuple(notes))
