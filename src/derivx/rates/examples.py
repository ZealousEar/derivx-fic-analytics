"""Example market data shipped in ``data/examples``.

* ``usd_sofr_ois_example.csv`` - illustrative SOFR OIS quotes. They are not
  market data: there is no free source of SOFR swap quotes.
* ``ust_par_yield_curve_2026.csv`` - US Treasury daily par yield curve rates
  (home.treasury.gov), percent, one row per business day.

The dated history the Rates page and the historical scenarios use (Treasury,
BoE OIS, ECB, ICE swap curves and fixings) is loaded by
:mod:`derivx.data.rates_history` from ``data/examples/rates_history``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .curves import Curve, Market, bootstrap
from .quotes import clean_quotes

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"
SOFR_EXAMPLE_CSV = DATA_DIR / "usd_sofr_ois_example.csv"
UST_PAR_CSV = DATA_DIR / "ust_par_yield_curve_2026.csv"
EXAMPLE_ASOF = dt.date(2026, 10, 19)  # latest row of the Treasury snapshot


def sofr_example_rows() -> List[Dict[str, object]]:
    df = pd.read_csv(SOFR_EXAMPLE_CSV)
    return [{"kind": r.instrument, "tenor": r.tenor, "rate": float(r.rate)} for r in df.itertuples()]


def ust_par_history() -> pd.DataFrame:
    """Treasury par yields (percent), indexed by date, columns = tenor labels."""

    df = pd.read_csv(UST_PAR_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.date
    return df.set_index("Date").sort_index()


def ust_par_rows(on: Optional[dt.date] = None) -> Tuple[dt.date, List[Dict[str, object]]]:
    """(date, rows) for one day of the Treasury curve: bills (1 Yr and
    shorter) as bond-equivalent yields, longer tenors as par coupon bonds.
    ``on`` defaults to the latest day; a day without data raises."""

    hist = ust_par_history()
    day = on or hist.index[-1]
    if day not in hist.index:
        raise ValueError(f"no Treasury curve for {day}; snapshot covers {hist.index[0]} to {hist.index[-1]}")
    rows = []
    for label, value in hist.loc[day].items():
        if pd.isna(value):
            continue
        years = label.strip().upper().endswith(("YR", "YEAR")) and float(label.split()[0]) > 1
        rows.append({"kind": "BOND" if years else "BILL", "tenor": label, "rate": float(value)})
    return day, rows


def example_curve(interpolation: str = "loglinear", asof: dt.date = EXAMPLE_ASOF) -> Curve:
    quotes, issues, _ = clean_quotes(sofr_example_rows())
    assert not [i for i in issues if i.level == "error"], issues
    return bootstrap(asof, quotes, "USD SOFR", interpolation)


def ust_curve(on: Optional[dt.date] = None, interpolation: str = "loglinear") -> Curve:
    day, rows = ust_par_rows(on)
    quotes, issues, _ = clean_quotes(rows)
    errors = [i.message for i in issues if i.level == "error"]
    if errors:
        raise ValueError("; ".join(errors))
    return bootstrap(day, quotes, "USD UST", interpolation)


def example_market(interpolation: str = "loglinear") -> Market:
    """SOFR example curve and the Treasury curve, both on EXAMPLE_ASOF."""

    return Market(curve=example_curve(interpolation), govt=ust_curve(EXAMPLE_ASOF, interpolation))
