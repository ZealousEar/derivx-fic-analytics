"""Build data/examples/historical_curve_shocks.csv from FRED Treasury yields.

Downloads the H.15 US Treasury constant-maturity yields (percent) from FRED's
keyless CSV endpoint and writes, for each episode, the change in every
tenor's yield in bp between the observations nearest the episode's start and
end dates. Only days on which every tenor has a value are used, and the
``start``/``end`` columns record the dates actually used. The scenario engine
(``derivx.risk.scenarios.historical``) reads the CSV with no network access.

    python tools/build_historical_shocks.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "examples" / "historical_curve_shocks.csv"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={ids}"
TIMEOUT = 30  # seconds per request

SERIES = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}
EPISODES = [
    ("2013 taper tantrum", "2013-05-02", "2013-09-05"),
    ("2020 dash for cash", "2020-03-09", "2020-03-18"),
    ("2022 hiking cycle", "2022-01-03", "2022-10-21"),
    ("Sept 2022 gilt crisis week (UST)", "2022-09-21", "2022-09-28"),
    ("2023 SVB", "2023-03-08", "2023-03-15"),
]


def _fetch(ids: Sequence[str]) -> pd.DataFrame:
    url = FRED_CSV.format(ids=",".join(ids))
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        text = response.read().decode("utf-8")
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = frame.columns[0]  # "observation_date" (older exports: "DATE")
    frame["date"] = pd.to_datetime(frame.pop(date_col)).dt.date
    return frame.set_index("date")[list(ids)]


def download() -> pd.DataFrame:
    """Every series as one frame (percent), indexed by observation date."""

    try:
        return _fetch(list(SERIES))
    except (OSError, ValueError, KeyError) as exc:
        print(f"combined download failed ({exc}); fetching one series at a time", file=sys.stderr)
        return pd.concat([_fetch([sid]) for sid in SERIES], axis=1)


def nearest(days: Iterable[dt.date], target: dt.date) -> dt.date:
    """Closest date to ``target`` (the earlier one on a tie)."""

    return min(days, key=lambda d: (abs((d - target).days), d))


def build(yields: pd.DataFrame) -> pd.DataFrame:
    complete = yields[list(SERIES)].dropna()
    rows: List[dict] = []
    for name, start, end in EPISODES:
        d0 = nearest(complete.index, dt.date.fromisoformat(start))
        d1 = nearest(complete.index, dt.date.fromisoformat(end))
        for sid, years in SERIES.items():
            change_bp = round((complete.at[d1, sid] - complete.at[d0, sid]) * 100.0, 2)
            rows.append(
                {
                    "episode": name,
                    "start": d0.isoformat(),
                    "end": d1.isoformat(),
                    "tenor_years": years,  # unrounded: 1/12 must read back exactly
                    "change_bp": change_bp,
                    "source": f"FRED {sid}",
                }
            )
    return pd.DataFrame(rows)


def main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="CSV to write")
    args = parser.parse_args(argv)
    try:
        yields = download()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"FRED is unreachable ({exc}); nothing written") from exc
    table = build(yields)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    for (name, d0, d1), group in table.groupby(["episode", "start", "end"], sort=False):
        moves = ", ".join(f"{y:g}Y {c:+.0f}" for y, c in zip(group["tenor_years"], group["change_bp"]))
        print(f"{name}: {d0} to {d1}: {moves}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
