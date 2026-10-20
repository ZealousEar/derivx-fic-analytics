"""Record the small raw responses the rates-source parser tests replay.

Writes tests/fixtures/rates/: one trimmed real response per parser (FRED
CSV, Treasury CSV, NY Fed JSON, ECB CSV, BoE xlsx/zip) and
``fixtures.json`` with the URL and recording time of each. The FRED JSON
API fixture needs a key, so it is built in the documented API layout from
the recorded fredgraph.csv values and marked as such.

    python tools/record_rates_fixtures.py
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from derivx.data import rates_sources as rs  # noqa: E402
from derivx.data.cache import utc_now  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "rates"
BOE = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
RECORDINGS = {
    "fred_ust.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    + ",".join(rs.UST_FRED_IDS)
    + "&cosd="
    + ",".join(["2026-10-13"] * 11)
    + "&coed="
    + ",".join(["2026-10-16"] * 11),
    "fred_dswp.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    + ",".join(rs.ICE_SWAP_FRED_IDS)
    + "&cosd="
    + ",".join(["2013-05-01"] * 8)
    + "&coed="
    + ",".join(["2013-05-03"] * 8),
    "fred_fixings.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR,IUDSOIA,ECBESTRVOLWGTTRMDMNRT,"
    "SOFRINDEX&cosd=2026-10-14,2026-10-14,2026-10-14,2026-10-14&coed=2026-10-19,2026-10-19,2026-10-19,2026-10-19",
    "treasury_2026.csv": rs.TREASURY_CSV_URL.format(year=2026),
    "nyfed_sofr_last3.json": "https://markets.newyorkfed.org/api/rates/secured/sofr/last/3.json",
    "nyfed_effr_search.json": "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"
    "?startDate=2026-10-13&endDate=2026-10-16",
    "nyfed_sofrai_last2.json": "https://markets.newyorkfed.org/api/rates/secured/sofrai/last/2.json",
    "ecb_estr.csv": "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata"
    "&lastNObservations=3",
    "ecb_yc_par.csv": "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM."
    "PY_3M+PY_2Y+PY_10Y+PY_30Y+SR_10Y6M?format=csvdata&detail=dataonly&lastNObservations=2",
    "boe_latest.zip": BOE + "latest-yield-curve-data.zip",
    "boe_ois_2009_2015.xlsx": BOE + "oisddata.zip",
}
KEEP_ROWS = 8  # header rows 1-5 plus three curves


def _trim_sheet(xml: str, last_row: int) -> str:
    def keep(match: re.Match) -> str:
        return match.group(0) if int(match.group(1)) <= last_row else ""

    return re.sub(r'<row r="(\d+)"[^>]*?(?:/>|>.*?</row>)', keep, xml, flags=re.S)


def trim_workbook(data: bytes, last_row: int = KEEP_ROWS) -> bytes:
    """Keep rows 1..``last_row`` of every sheet; drop the calc chain."""

    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            if item.filename == "xl/calcChain.xml" or item.filename.startswith("xl/printerSettings"):
                continue
            payload = src.read(item.filename)
            if item.filename.startswith("xl/worksheets/sheet"):
                payload = _trim_sheet(payload.decode("utf-8"), last_row).encode("utf-8")
            elif item.filename in ("[Content_Types].xml", "xl/_rels/workbook.xml.rels"):
                text = payload.decode("utf-8")
                text = re.sub(r"<(Override|Relationship)[^>]*calcChain[^>]*/>", "", text)
                payload = text.encode("utf-8")
            elif item.filename.startswith("xl/worksheets/_rels/"):
                text = re.sub(r"<Relationship[^>]*printerSettings[^>]*/>", "", payload.decode("utf-8"))
                payload = text.encode("utf-8")
            dst.writestr(item.filename, payload)
    return out.getvalue()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, url in RECORDINGS.items():
        timeout = rs.TREASURY_TIMEOUT if "treasury" in url else 60.0
        payload = rs.http_get(url, timeout=timeout)
        note = "raw response"
        if name == "treasury_2026.csv":
            payload = b"\n".join(payload.decode("utf-8-sig").encode().splitlines()[:6]) + b"\n"
            note = "first five days of the response"
        elif name == "boe_latest.zip":
            src = zipfile.ZipFile(io.BytesIO(payload))
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
                for member in ("OIS daily data current month.xlsx", "GLC Nominal daily data current month.xlsx"):
                    dst.writestr(member, trim_workbook(src.read(member)))
            payload = out.getvalue()
            note = "OIS and GLC nominal workbooks only, each trimmed to its first three curves"
        elif name == "boe_ois_2009_2015.xlsx":
            payload = trim_workbook(zipfile.ZipFile(io.BytesIO(payload)).read("OIS daily data_2009 to 2015.xlsx"))
            note = "the 2009-2015 workbook of the daily archive (older sheet layout), trimmed to three curves"
        (OUT / name).write_bytes(payload)
        index[name] = {"url": url, "recorded_at": utc_now().isoformat(timespec="seconds"), "note": note}
        print(f"wrote {name}: {len(payload)} bytes")

    # FRED JSON API layout (needs a key to record): built from the CSV values.
    frame = rs.parse_fred_csv((OUT / "fred_ust.csv").read_text())
    dgs10 = frame[frame["series"] == "DGS10"]
    observations = [
        {"realtime_start": "2026-10-20", "realtime_end": "2026-10-20", "date": d.isoformat(), "value": f"{v:.2f}"}
        for d, v in zip(dgs10["asof"], dgs10["value_original"])
    ]
    observations.append({"realtime_start": "2026-10-20", "realtime_end": "2026-10-20", "date": "2026-10-12",
                         "value": "."})  # fmt: skip
    api = {"realtime_start": "2026-10-20", "realtime_end": "2026-10-20", "observation_start": "2026-10-12",
           "observation_end": "2026-10-16", "units": "lin", "output_type": 1, "file_type": "json",
           "order_by": "observation_date", "sort_order": "asc", "count": len(observations), "offset": 0,
           "limit": 100000, "observations": sorted(observations, key=lambda o: o["date"])}  # fmt: skip
    (OUT / "fred_api_dgs10.json").write_text(json.dumps(api, indent=1) + "\n")
    index["fred_api_dgs10.json"] = {
        "url": rs.FRED_API_URL + "?series_id=DGS10&file_type=json (needs api_key)",
        "recorded_at": utc_now().isoformat(timespec="seconds"),
        "note": "constructed, not recorded: the documented series/observations layout filled with the DGS10 values "
        "of fred_ust.csv plus one '.' (missing) observation; no API key was available",
    }
    (OUT / "fixtures.json").write_text(json.dumps(index, indent=1) + "\n")
    print("wrote fixtures.json")


if __name__ == "__main__":
    main()
