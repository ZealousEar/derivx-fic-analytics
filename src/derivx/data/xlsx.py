"""Minimal, dependency-free reader for the cell values of ``.xlsx`` files.

The Bank of England publishes its yield curves as Office Open XML workbooks.
They are plain numeric grids, so this reader extracts cell values (numbers,
shared and inline strings, booleans, error codes) with the standard library
instead of adding a spreadsheet dependency. Formulas are read as their cached
values; styles, merged cells and formatting are ignored.

Dates stay as Excel serial numbers; convert them with :func:`excel_date`
(1900 date system unless the workbook says ``date1904``).
"""

from __future__ import annotations

import datetime as dt
import io
import re
import zipfile
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from xml.etree import ElementTree as ET

Cell = Union[float, str, bool, None]

_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_REF = re.compile(r"^([A-Z]+)(\d+)$")
_EPOCH_1900 = dt.date(1899, 12, 30)  # exact for serials >= 61 (1900-03-01 on); Excel counts a fake 1900-02-29
_EPOCH_1904 = dt.date(1904, 1, 1)
_BROKEN = (zipfile.BadZipFile, KeyError, ET.ParseError, zlib.error, EOFError)  # unreadable parts


@dataclass(frozen=True)
class Workbook:
    """Sheets as ``{name: rows}``; each row is a list of cells (``None`` for
    blanks), row 1 first. ``date1904`` is the workbook's date system."""

    sheets: Dict[str, List[List[Cell]]]
    date1904: bool = False

    def sheet(self, name: str) -> List[List[Cell]]:
        try:
            return self.sheets[name]
        except KeyError:
            raise ValueError(f"workbook has no sheet {name!r}; sheets: {list(self.sheets)}") from None


def column_index(letters: str) -> int:
    """0-based column of an A1-style column label (``"A"`` -> 0, ``"AA"`` -> 26)."""

    out = 0
    for ch in letters:
        out = out * 26 + (ord(ch) - 64)
    return out - 1


def excel_date(serial: float, date1904: bool = False) -> dt.date:
    """Calendar date of an Excel serial day number."""

    if not isinstance(serial, (int, float)) or isinstance(serial, bool):
        raise ValueError(f"{serial!r} is not an Excel date serial number")
    return (_EPOCH_1904 if date1904 else _EPOCH_1900) + dt.timedelta(days=int(serial))


def _text(node: ET.Element) -> str:
    return "".join(t.text or "" for t in node.iter(f"{_MAIN}t"))


def _shared_strings(archive: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [_text(si) for si in root.findall(f"{_MAIN}si")]


def _sheet_paths(archive: zipfile.ZipFile) -> Dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {r.get("Id"): r.get("Target", "") for r in rels.iter(f"{_PKG_REL}Relationship")}
    out = {}
    for sheet in workbook.iter(f"{_MAIN}sheet"):
        target = targets.get(sheet.get(f"{_REL}id"), "")
        path = target.lstrip("/")
        out[sheet.get("name", "")] = path if path.startswith("xl/") else f"xl/{path}"
    return out


def _cell_value(cell: ET.Element, strings: List[str]) -> Cell:
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        node = cell.find(f"{_MAIN}is")
        return _text(node) if node is not None else None
    value = cell.find(f"{_MAIN}v")
    if value is None or value.text is None:
        return None
    text = value.text
    if kind == "s":
        return strings[int(text)]
    if kind == "b":
        return text == "1"
    if kind in ("e", "str"):
        return text
    try:
        return float(text)
    except ValueError:
        return text


def _read_sheet(data: bytes, strings: List[str]) -> List[List[Cell]]:
    rows: Dict[int, Dict[int, Cell]] = {}
    for _, node in ET.iterparse(io.BytesIO(data)):
        if node.tag != f"{_MAIN}c":
            continue
        match = _REF.match(node.get("r", ""))
        if match:
            rows.setdefault(int(match.group(2)), {})[column_index(match.group(1))] = _cell_value(node, strings)
        node.clear()
    if not rows:
        return []
    out: List[List[Cell]] = []
    for r in range(1, max(rows) + 1):
        cells = rows.get(r, {})
        width = max(cells) + 1 if cells else 0
        out.append([cells.get(c) for c in range(width)])
    return out


def read_workbook(data: bytes, sheets: Optional[List[str]] = None) -> Workbook:
    """Parse an ``.xlsx`` payload; only the named ``sheets`` if given.

    Raises ``ValueError`` if the bytes are not an xlsx workbook, or if a
    sheet is truncated or corrupt.
    """

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        paths = _sheet_paths(archive)
        strings = _shared_strings(archive)
        props = ET.fromstring(archive.read("xl/workbook.xml")).find(f"{_MAIN}workbookPr")
    except _BROKEN as exc:
        raise ValueError(f"not an xlsx workbook ({exc})") from exc
    date1904 = props is not None and props.get("date1904", "0") in ("1", "true")
    wanted = list(paths) if sheets is None else sheets
    missing = [s for s in wanted if s not in paths]
    if missing:
        raise ValueError(f"workbook has no sheet(s) {missing}; sheets: {list(paths)}")
    out = {}
    for name in wanted:
        try:
            out[name] = _read_sheet(archive.read(paths[name]), strings)
        except (*_BROKEN, IndexError) as exc:  # IndexError: a shared-string index past the table
            raise ValueError(f"sheet {name!r} of the workbook is corrupt or truncated ({exc})") from exc
    return Workbook(out, date1904)
