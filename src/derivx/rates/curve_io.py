"""Curve sets from quote files and the history store, saved as JSON.

Quote files (CSV) have ``instrument`` (or ``kind``), ``tenor`` and ``rate``
columns and an optional ``curve`` column saying which curve a row belongs
to: blank, ``OIS`` or ``DISCOUNT`` for the swap (discount) curve, a term
index (``EURIBOR 6M``, ``EURIBOR 3M``) for its projection curve, or
``TREASURY`` for the US Treasury par curve. Rates are percent, decimals or
bp (read from the magnitudes per curve unless ``units`` is given); futures
are prices with the contract as the tenor; ``BASIS`` rows are spreads.

:func:`curve_set_json` writes a market as JSON: per curve its conventions,
interpolation, convexity inputs, overnight fixings and quotes (decimal),
plus the bootstrapped nodes (pillar dates, ACT/365F times, discount factors,
continuously compounded zero rates) and a run manifest.
:func:`market_from_json` rebuilds the curves from the quotes and checks
the rebuilt discount factors against the saved nodes, so a file edited by
hand or written by another version is refused instead of silently pricing
on different curves.

:func:`default_market` builds the market the Rates page starts from on a
date, without the UI: the swap curve from :func:`derivx.rates.prefill.swap_prefill`
(real, proxy or ILLUSTRATIVE quotes, labelled), EURIBOR projection curves
from ILLUSTRATIVE quotes (EUR only) and the US Treasury par curve.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from derivx.data.rates_history import CURRENCY_FIXING, HistoryStore
from derivx.rates.curves import (
    CONVENTIONS,
    DISCOUNT_KEY,
    GOVT_KEY,
    INTERPOLATIONS,
    OIS_FUTURES,
    Curve,
    CurveQuote,
    Market,
    bootstrap,
)
from derivx.rates.indices import CURRENCY_INDICES, INDICES, HullWhite, get_index
from derivx.rates.market_quotes import treasury_quotes
from derivx.rates.multicurve import BASIS_OF, PROJECTION_KINDS, bootstrap_projection, illustrative_quotes
from derivx.rates.prefill import swap_prefill
from derivx.rates.quotes import clean_quotes

FORMAT = "derivx.curve_set"
FORMAT_VERSION = 1
HW_DEFAULT = HullWhite(0.03, 0.0085)  # 3% mean reversion, 85bp normal short-rate vol: model inputs, not market data
DISCOUNT_KINDS = ("DEPO", "OIS", "IRS")
GOVT_KINDS = ("BILL", "BOND")
NODE_TOLERANCE = 1e-12  # largest discount-factor difference accepted between a saved and a rebuilt curve
_DISCOUNT_NAMES = ("", "OIS", "DISCOUNT", "CURVE", "SWAP")
_GOVT_NAMES = ("TREASURY", "GOVT", "UST")


def curve_key_of(label: object) -> str:
    """Curve key of a quote row's ``curve`` cell."""

    text = "" if label is None or (isinstance(label, float) and math.isnan(label)) else str(label).strip().upper()
    text = " ".join(text.split())
    if text in _DISCOUNT_NAMES or text in CONVENTIONS:
        return DISCOUNT_KEY
    if text in _GOVT_NAMES:
        return GOVT_KEY
    if text in INDICES:
        return text
    raise ValueError(f"curve {label!r}: use blank/OIS (swap curve), TREASURY or a term index ({', '.join(INDICES)})")


def read_quotes(
    payload: bytes, conventions: str, asof: dt.date, units: str = "auto"
) -> Tuple[Dict[str, List[CurveQuote]], List[str], List[str]]:
    """Quote CSV -> ({curve key: decimal quotes}, errors, notes)."""

    try:
        raw = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", dtype=str, keep_default_na=False, na_values=[""])
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        return {}, [f"could not read the quotes file ({exc})"], []
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "kind" in raw.columns and "instrument" not in raw.columns:
        raw = raw.rename(columns={"kind": "instrument"})
    missing = {"instrument", "tenor", "rate"} - set(raw.columns)
    if missing:
        return {}, [f"the quotes file has no {', '.join(sorted(missing))} column"], []
    errors: List[str] = []
    groups: Dict[str, List[dict]] = {}
    for n, row in enumerate(raw.to_dict("records"), start=1):
        try:
            key = curve_key_of(row.get("curve"))
        except ValueError as exc:
            errors.append(f"row {n}: {exc}")
            continue
        groups.setdefault(key, []).append(row)
    out: Dict[str, List[CurveQuote]] = {}
    notes: List[str] = []
    for key, rows in groups.items():
        if key == DISCOUNT_KEY:
            kinds = DISCOUNT_KINDS + (OIS_FUTURES if conventions == "USD SOFR" else ())
        elif key == GOVT_KEY:
            kinds = GOVT_KINDS
        else:
            kinds = PROJECTION_KINDS
        quotes, issues, used = clean_quotes(rows, units=units, kinds=kinds, asof=asof)
        name = {DISCOUNT_KEY: "swap curve", GOVT_KEY: "Treasury curve"}.get(key, f"{key} curve")
        errors += [f"{name}: {i.message}" for i in issues if i.level == "error"]
        notes += [f"{name}: {i.message}" for i in issues if i.level != "error"]
        if quotes and not any(i.level == "error" for i in issues):
            out[key] = quotes
    if DISCOUNT_KEY not in out and not errors:
        errors.append("the quotes file has no swap-curve rows (curve blank or OIS)")
    return out, errors, notes


def ois_fixings(store: HistoryStore, currency: str, asof: dt.date) -> Dict[dt.date, float]:
    """Overnight fixings of the four months before ``asof`` (part-fixed
    SOFR futures need them)."""

    name = CURRENCY_FIXING.get(currency)
    return store.fixing_history(name, asof - dt.timedelta(days=125), asof) if name else {}


def build_market(
    asof: dt.date,
    quotes: Mapping[str, Sequence[CurveQuote]],
    conventions: str,
    interpolation: str = "loglinear",
    convexity: Optional[HullWhite] = HW_DEFAULT,
    fixings: Optional[Mapping[dt.date, float]] = None,
) -> Market:
    """Bootstrap the curve set: the swap curve (``quotes["curve"]``), each
    projection curve on it (a curve with ``BASIS`` quotes after the index
    they are against, :data:`~derivx.rates.multicurve.BASIS_OF`) and the
    Treasury curve. ``convexity`` adjusts futures quotes (only used where
    there are futures); ``fixings`` are overnight fixings for part-fixed
    SOFR futures."""

    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation {interpolation!r}: use one of {', '.join(INTERPOLATIONS)}")
    if conventions not in CONVENTIONS or conventions == "USD UST":
        raise ValueError(f"conventions {conventions!r}: use USD SOFR, EUR ESTR or GBP SONIA")
    if DISCOUNT_KEY not in quotes:
        raise ValueError("no swap-curve quotes")
    disc_quotes = list(quotes[DISCOUNT_KEY])
    has_futures = any(q.kind in OIS_FUTURES for q in disc_quotes)
    curve = bootstrap(asof, disc_quotes, conventions, interpolation, convexity=convexity if has_futures else None,
                      fixings=dict(fixings or {}) if has_futures else None)  # fmt: skip
    currency = CONVENTIONS[conventions].currency
    built: Dict[str, Curve] = {}
    pending = [k for k in quotes if k not in (DISCOUNT_KEY, GOVT_KEY)]
    for key in pending:
        if get_index(key).currency != currency:
            raise ValueError(f"{key} is a {get_index(key).currency} index: the swap curve is {conventions}")
    while pending:
        progressed = False
        for key in list(pending):
            rows = list(quotes[key])
            basis_index = BASIS_OF.get(key) if any(q.kind == "BASIS" for q in rows) else None
            if basis_index and basis_index not in built:
                if basis_index not in quotes:
                    raise ValueError(f"the {key} BASIS quotes are against {basis_index}: add its quotes")
                continue
            futures = any(q.kind == "FUT" for q in rows)
            built[key] = bootstrap_projection(asof, rows, key, curve, interpolation,
                                              basis=built.get(basis_index) if basis_index else None,
                                              convexity=convexity if futures else None)  # fmt: skip
            pending.remove(key)
            progressed = True
        if not progressed:
            raise ValueError(f"projection curves wait on each other: {pending}")
    govt = bootstrap(asof, list(quotes[GOVT_KEY]), "USD UST", interpolation) if GOVT_KEY in quotes else None
    return Market(curve=curve, govt=govt, projections=built)


def treasury_curve_quotes(store: HistoryStore, asof: dt.date) -> List[CurveQuote]:
    """The US Treasury par curve on ``asof`` from the store as BILL/BOND
    quotes; ``ValueError`` if the store has no curve that day."""

    return treasury_quotes(store.curve("UST_PAR", asof).to_dict())


def default_market(
    store: HistoryStore,
    asof: dt.date,
    conventions: str = "USD SOFR",
    interpolation: str = "loglinear",
    convexity: HullWhite = HW_DEFAULT,
) -> Tuple[Market, Dict[str, str], List[str]]:
    """The market the Rates page starts from on ``asof`` (see the module
    docstring): (market, {curve key: source label}, notes)."""

    prefill = swap_prefill(store, conventions, asof)
    rows = prefill.table.drop(columns=["source"]).to_dict("records")
    quotes, issues, _ = clean_quotes(rows, units="percent", kinds=DISCOUNT_KINDS, asof=asof)
    errors = [i.message for i in issues if i.level == "error"]
    if errors:
        raise ValueError("the prefilled swap quotes do not build: " + "; ".join(errors))
    by_curve: Dict[str, List[CurveQuote]] = {DISCOUNT_KEY: quotes}
    sources = {DISCOUNT_KEY: f"{conventions} | {prefill.label}"}
    currency = CONVENTIONS[conventions].currency
    indices = CURRENCY_INDICES.get(currency, ())
    if indices:
        prefill_curve = bootstrap(asof, quotes, conventions, "loglinear")
        made: Dict[str, Curve] = {}
        for index in sorted(indices, key=lambda k: k in BASIS_OF):  # 6M before the 3M curve built on it
            basis = made.get(BASIS_OF.get(index, ""))
            q = illustrative_quotes(prefill_curve, index, convexity, basis_curve=basis)
            made[index] = bootstrap_projection(asof, q, index, prefill_curve, "loglinear", basis=basis,
                                               convexity=convexity)  # fmt: skip
            by_curve[index] = q
            sources[index] = f"{index} | ILLUSTRATIVE quotes (not market data)"
    by_curve[GOVT_KEY] = treasury_curve_quotes(store, asof)
    sources[GOVT_KEY] = f"US Treasury par yields ({store.origin('UST_PAR', asof)})"
    market = build_market(asof, by_curve, conventions, interpolation, convexity, ois_fixings(store, currency, asof))
    return market, sources, list(prefill.notes)


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def _nodes(curve: Curve) -> List[Dict[str, object]]:
    pillars = [curve.asof] + list(curve.pillar_dates or [])
    rows = []
    for k, (t, df) in enumerate(zip(curve.ts, curve.dfs)):
        rows.append({
            "pillar": pillars[k].isoformat() if k < len(pillars) else None,
            "t": float(t),
            "df": float(df),
            "zero_cc": float(-math.log(df) / t) if t > 0 else None,
        })  # fmt: skip
    return rows


def _curve_payload(key: str, curve: Curve, source: str) -> Dict[str, object]:
    index = getattr(curve, "index", None)
    return {
        "conventions": curve.conventions.name if curve.conventions is not None else None,
        "index": index.name if index is not None else None,
        "basis_index": getattr(curve, "basis_index", None),
        "interpolation": curve.interpolation,
        "convexity": None if curve.convexity is None else [curve.convexity.mean_reversion, curve.convexity.sigma],
        "fixings": {d.isoformat(): r for d, r in sorted(curve.overnight_fixings.items())},
        "quotes": [[q.kind, q.tenor, q.rate] for q in curve.quotes],
        "nodes": _nodes(curve),
        "source": source,
    }


def curve_set_json(
    market: Market, sources: Optional[Mapping[str, str]] = None, manifest: Optional[Mapping[str, object]] = None
) -> Dict[str, object]:
    """The market's curves as a JSON-ready dict (see the module docstring)."""

    sources = dict(sources or {})
    curves = {}
    for key in market.curve_keys:
        if key not in (DISCOUNT_KEY, GOVT_KEY) and market.projections.is_discount(key):
            continue
        curve = market.get_curve(key)
        if not curve.quotes:
            raise ValueError(f"the {key} curve was not bootstrapped from quotes: it cannot be saved as a curve set")
        curves[key] = _curve_payload(key, curve, sources.get(key, ""))
    return {"format": FORMAT, "version": FORMAT_VERSION, "asof": market.asof.isoformat(), "curves": curves,
            "manifest": dict(manifest or {})}  # fmt: skip


def _check_nodes(key: str, curve: Curve, nodes: Sequence[Mapping[str, object]]) -> None:
    saved = np.array([float(n["df"]) for n in nodes])
    if saved.shape != curve.dfs.shape:
        raise ValueError(f"the {key} curve rebuilds with {curve.dfs.size} nodes, the file has {saved.size}")
    gap = float(np.max(np.abs(saved - curve.dfs)))
    if not gap <= NODE_TOLERANCE:
        raise ValueError(
            f"the {key} curve's quotes rebuild discount factors {gap:.2e} away from the saved nodes: the file was "
            "edited or written by another version; rebuild it with 'derivx curve build'"
        )


def market_from_json(payload: Mapping[str, object]) -> Tuple[Market, Dict[str, str]]:
    """Rebuild a market saved by :func:`curve_set_json` (curves bootstrapped
    again from their quotes and checked against the saved nodes):
    (market, {curve key: source label}). A file that is not such a curve
    set, or is missing parts of one, raises ValueError."""

    if not isinstance(payload, Mapping):
        kind = type(payload).__name__
        raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION} file: the JSON is a {kind}, not an object")
    try:
        return _market_from_json(payload)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(
            f"not a complete {FORMAT} v{FORMAT_VERSION} file ({type(exc).__name__}: {exc}); rebuild it with "
            "'derivx curve build'"
        ) from None


def _market_from_json(payload: Mapping[str, object]) -> Tuple[Market, Dict[str, str]]:
    if payload.get("format") != FORMAT or payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION} file (format {payload.get('format')!r}, version "
                         f"{payload.get('version')!r})")  # fmt: skip
    asof = dt.date.fromisoformat(str(payload["asof"]))
    curves = dict(payload["curves"])
    if DISCOUNT_KEY not in curves:
        raise ValueError("the curve set has no swap (discount) curve")
    sources = {k: str(v.get("source", "")) for k, v in curves.items()}

    def quotes_of(entry: Mapping[str, object]) -> List[CurveQuote]:
        return [CurveQuote(str(k), str(t), float(r)) for k, t, r in entry["quotes"]]

    def convexity_of(entry: Mapping[str, object]) -> Optional[HullWhite]:
        value = entry.get("convexity")
        return None if value is None else HullWhite(float(value[0]), float(value[1]))

    disc = curves[DISCOUNT_KEY]
    fixings = {dt.date.fromisoformat(d): float(r) for d, r in dict(disc.get("fixings") or {}).items()}
    curve = bootstrap(asof, quotes_of(disc), str(disc["conventions"]), str(disc["interpolation"]),
                      convexity=convexity_of(disc), fixings=fixings or None)  # fmt: skip
    _check_nodes(DISCOUNT_KEY, curve, disc["nodes"])
    built: Dict[str, Curve] = {}
    pending = [k for k in curves if k not in (DISCOUNT_KEY, GOVT_KEY)]
    while pending:
        progressed = False
        for key in list(pending):
            entry = curves[key]
            basis_index = entry.get("basis_index")
            if basis_index and basis_index not in built:
                if basis_index not in curves:
                    raise ValueError(f"the {key} curve is built on {basis_index}, which the file does not hold")
                continue
            proj = bootstrap_projection(asof, quotes_of(entry), key, curve, str(entry["interpolation"]),
                                        basis=built.get(basis_index) if basis_index else None,
                                        convexity=convexity_of(entry))  # fmt: skip
            _check_nodes(key, proj, entry["nodes"])
            built[key] = proj
            pending.remove(key)
            progressed = True
        if not progressed:
            raise ValueError(f"projection curves wait on each other: {pending}")
    govt = None
    if GOVT_KEY in curves:
        entry = curves[GOVT_KEY]
        govt = bootstrap(asof, quotes_of(entry), str(entry["conventions"]), str(entry["interpolation"]))
        _check_nodes(GOVT_KEY, govt, entry["nodes"])
    return Market(curve=curve, govt=govt, projections=built), sources


def write_curve_set(path, market: Market, sources: Optional[Mapping[str, str]] = None,
                    manifest: Optional[Mapping[str, object]] = None) -> None:  # fmt: skip
    from derivx.rates.manifest import jsonable

    text = json.dumps(jsonable(curve_set_json(market, sources, manifest)), indent=1, allow_nan=False)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def read_curve_set(path) -> Tuple[Market, Dict[str, str], Dict[str, object]]:
    """(market, sources, manifest) from a curve-set JSON file."""

    with open(path, encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not JSON ({exc})") from None
    market, sources = market_from_json(payload)
    manifest = payload.get("manifest")
    return market, sources, dict(manifest) if isinstance(manifest, Mapping) else {}
