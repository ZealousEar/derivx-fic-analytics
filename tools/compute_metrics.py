"""Reproducible smile-model metrics from option-chain snapshots.

Default input: the bundled real SPY snapshot
``data/examples/spy_2026-10-20.csv.gz`` (one chain from Yahoo Finance,
fetched 2026-10-20 20:59 UTC). Any snapshot files in the chain schema
(:mod:`derivx.data.chain_schema`; CSV, CSV.GZ or parquet, one or several
trade dates) can be passed with ``--chain`` or ``--chain-dir``.

Per snapshot (trade date):

1. Liquidity filters, as the lab's defaults: volume >= 10, mid >= 0.10,
   relative spread <= 40%. Then the :mod:`derivx.vol.quotes` hygiene before
   any fit: expired / too-short expiries, IV placeholders, zero bids,
   crossed and stale quotes and ITM options dropped (one OTM quote per
   strike), put-call-parity forwards, vols re-inverted from bid/ask mids
   discounted per ``--discount``: ``curve`` (default) - the US Treasury
   curve of each trade date from the committed rates snapshot (the latest
   published on or before it, bootstrapped); ``parity`` - the put-call
   parity DF per expiry where it passes its gates (European options only),
   else that curve; ``rate`` - the flat cc rate ``--rate``
   (:mod:`derivx.vol.discounting`). A trade date without a Treasury curve
   within a week uses ``--rate`` and the summary says so.
2. Model comparison per expiry (``derivx.models.comparison.
   compare_models_by_expiry``): SVI (5 params), SABR with fixed beta
   (``--sabr-beta``, 3 params) and a natural cubic spline of IV in ln(K/F),
   each scored in sample and by k-fold cross-validation on interior strikes
   (``rmse_cv_bp``, the comparable out-of-sample number).
3. Production SVI calibration (butterfly / Lee / calendar diagnostics, with
   the arbitrage repair on), unless ``--no-calibration``.
4. Walk-forward (:func:`derivx.validation.backtest.run_walk_forward`) only
   when the input holds snapshots one trading day apart; a single snapshot
   cannot score a forecast and the report says so.

Every slice is reported, including skipped ones with the reason; no score
is clipped, and each comparison is worded by its sign. Outputs in
``--outdir``: ``model_scores.csv``, ``skipped.csv``, ``calibration.csv``,
``summary.json``, ``METRICS_SUMMARY.md`` and ``manifest.json`` (code
version, packages, inputs, data hash). All files but the manifest are
deterministic for a given input and code version.

    python tools/compute_metrics.py --outdir docs/metrics/spy_2026-10-20

History mode (``--history``, or ``--store DIR`` for a local checkout of the
data branch): the walk-forward test on the daily snapshots the
``collect-data`` workflow saves on the ``data-snapshots`` branch
(:mod:`derivx.data.snapshots`), synced into the download cache
(``--repo`` / ``--branch``; ``--offline`` reads only the cache), with the
same liquidity filters and per-trade-date Treasury discounting. It runs
once the sessions hold :func:`derivx.data.snapshots.origins_needed` origin
days (9: three per vol-regime tercile) at ``--horizon``; until then the
summary says how many there are. Outputs: ``walkforward_summary.csv``,
``walkforward_daily.csv``, ``walkforward_by_regime.csv``,
``walkforward_slices.csv``, ``sessions.csv`` (the index rows used, with
each file's SHA-256), ``summary.json``, ``METRICS_SUMMARY.md`` and
``manifest.json``. Everything but the manifest's run time is a function of
the store's files and the code version, so a rerun on the same sessions
reproduces it byte for byte. Sessions after the committed Treasury curves
(2026-10-19) are discounted at the flat ``--rate`` unless ``--with-cache``
adds the Treasury curves in the download cache (``derivx pull fred --ids
ust``); the summary names the curve or rate of every trade date.

    python tools/compute_metrics.py --history --ticker SPY --outdir data/metrics/spy_history
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from derivx.calibration.production import calibrate_chain
from derivx.data import rates_history as rh
from derivx.data.chain_schema import ChainDataset, ChainSchemaError, load_chain_file, merge_datasets
from derivx.data.market_data import apply_liquidity_filters
from derivx.data.samples import SAMPLE_CHAIN, load_sample_chain
from derivx.models.comparison import (
    MODEL_LABELS,
    MODELS,
    compare_models_by_expiry,
    describe_pairwise,
    pairwise_wins,
    summarise_comparison,
)
from derivx.validation.backtest import run_walk_forward
from derivx.validation.manifest import build_manifest, jsonable, manifest_bytes
from derivx.vol.discounting import Discounting, european_style, treasury_discounting

LABEL = MODEL_LABELS


def load_inputs(chains: Sequence[Path], chain_dir: Optional[Path]) -> ChainDataset:
    """The snapshots to score (default: the bundled sample)."""

    paths = list(chains)
    if chain_dir is not None:
        paths += sorted(p for p in chain_dir.iterdir() if p.name.lower().endswith((".csv", ".csv.gz", ".parquet")))
    if not paths:
        return load_sample_chain()
    datasets = []
    for path in paths:
        ds = load_chain_file(path.name, path.read_bytes())
        ds.files = [str(path)]
        datasets.append(ds)
    return merge_datasets(datasets, source=f"{len(paths)} file(s)")


def discounting_for(ds: ChainDataset, mode: str = "curve", rate: float = 0.0, with_cache: bool = False) -> Discounting:
    """The :class:`Discounting` of ``mode`` for the trade dates of ``ds``
    (Treasury curves from the committed rates snapshot, plus those in the
    download cache with ``with_cache``; no network)."""

    if mode == "rate":
        return Discounting.flat(rate)
    store = rh.bundled_store()
    if with_cache:
        store = rh.build_store([rh.bundled_frame()] + rh.cached_frames(), "bundled snapshot + download cache")
    try:
        return treasury_discounting(store, ds.trade_dates, mode=mode, rate=rate)
    except ValueError:  # no curve for any trade date: every slice uses the flat rate (and says so)
        return Discounting("parity" if mode == "parity" else "rate", rate)


def compute(
    ds: ChainDataset,
    *,
    rate: float = 0.0,
    discount: Optional[Discounting] = None,
    min_volume: int = 10,
    min_mid: float = 0.10,
    max_spread: Optional[float] = 0.40,
    min_days: float = 2.0,
    max_days: Optional[float] = None,
    sabr_beta: float = 1.0,
    folds: int = 5,
    calibration: bool = True,
    expiries: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """All tables and the summary for ``ds`` (see the module docs).
    ``discount`` defaults to the Treasury curve of each trade date."""

    discount = discount if discount is not None else discounting_for(ds, "curve", rate)
    scores, skipped, calib = [], [], []
    filtered: Dict[pd.Timestamp, pd.DataFrame] = {}
    for td, chain in sorted(ds.chains.items()):
        f = apply_liquidity_filters(chain, min_volume=min_volume, max_spread=max_spread, min_mid=min_mid)
        if expiries:
            f = f[pd.to_datetime(f["expiry"]).dt.strftime("%Y-%m-%d").isin(set(expiries))]
        filtered[td] = f
        s, k = compare_models_by_expiry(
            f, MODELS, discount=discount, min_days=min_days, max_days=max_days, sabr_beta=sabr_beta, folds=folds
        )
        scores.append(s)
        skipped.append(k)
        if calibration:
            cal = calibrate_chain(f, discount=discount, min_days=min_days, cv_folds=0, enforce_arbitrage=True)
            table = cal.table()
            if max_days is not None and not table.empty:
                table = table[(pd.to_datetime(table["expiry"]) - td).dt.days <= max_days]
            calib.append(table)
    scores_df = (
        pd.concat([s for s in scores if not s.empty], ignore_index=True)
        if any(not s.empty for s in scores)
        else pd.DataFrame()
    )
    # concatenate only non-empty frames (pandas warns about empty ones, and they add nothing)
    skipped_df = (
        pd.concat([k for k in skipped if not k.empty], ignore_index=True)
        if any(not k.empty for k in skipped)
        else pd.DataFrame(columns=["trade_date", "expiry", "reason"])
    )
    calib_df = (
        pd.concat([c for c in calib if not c.empty], ignore_index=True)
        if any(not c.empty for c in calib)
        else pd.DataFrame()
    )
    per_model = summarise_comparison(scores_df)
    pairs = {
        f"{a}_vs_{b}": pairwise_wins(scores_df, a, b)
        for a, b in (("svi", "sabr"), ("svi", "spline"), ("sabr", "spline"))
    }
    summary: Dict[str, object] = {
        "data": ds.provenance(),
        "filters": {"min_volume": min_volume, "min_mid": min_mid, "max_spread": max_spread},
        "settings": {
            "discounting": discount.describe(),
            "min_days": min_days,
            "max_days": max_days,
            "sabr_beta": sabr_beta,
            "cv_folds": folds,
            "std_band_sd": 3.0,
            "min_points": 8,
            "expiries": list(expiries) if expiries else "all",
        },
        "snapshots": len(ds.chains),
        "model_comparison": {
            "metric": "rmse_cv_bp: k-fold CV RMSE on interior strikes, vol bp",
            "slices_scored": int(scores_df[["trade_date", "expiry"]].drop_duplicates().shape[0])
            if not scores_df.empty
            else 0,
            "slices_skipped": int(len(skipped_df)),
            "per_model": per_model.to_dict(orient="records"),
            "pairwise": pairs,
            "statements": [describe_pairwise(*key.split("_vs_"), res) for key, res in pairs.items()],
        },
    }
    if calibration:
        summary["calibration"] = _calibration_summary(calib_df)
    summary["walk_forward"] = _walk_forward(filtered, discount=discount, max_days=max_days)
    return {"scores": scores_df, "skipped": skipped_df, "calibration": calib_df, "summary": summary}


def _calibration_summary(table: pd.DataFrame) -> Dict[str, object]:
    if table.empty:
        return {"slices": 0}
    cal_ok = table["calendar_ok"].dropna()
    return {
        "slices": int(len(table)),
        "median_rmse_bp": float(table["rmse_bp"].median()),
        "max_rmse_bp": float(table["rmse_bp"].max()),
        "butterfly_ok": f"{int(table['butterfly_ok'].sum())} of {len(table)}",
        "lee_ok": f"{int(table['lee_ok'].sum())} of {len(table)}",
        "calendar_ok": f"{int(cal_ok.astype(bool).sum())} of {len(cal_ok)} adjacent expiry pairs",
        "calendar_failures": [str(e) for e in table.loc[table["calendar_ok"] == False, "expiry"]],  # noqa: E712
    }


def _walk_forward(
    filtered: Dict[pd.Timestamp, pd.DataFrame], discount: Discounting, max_days: Optional[float]
) -> Dict[str, object]:
    if len(filtered) < 2:
        return {
            "status": "not run",
            "reason": f"{len(filtered)} snapshot(s): a walk-forward test needs daily snapshots one trading day apart. "
            "Yahoo has no history; collect one snapshot per trading day with tools/download_spy_chains.py.",
        }
    result = run_walk_forward(
        filtered, rate=discount.rate, discount=discount, max_days=max_days if max_days is not None else 120.0
    )
    if result.empty:
        reasons = sorted(set(result.skipped["reason"].astype(str)))[:3]
        return {"status": "no scores", "reason": "; ".join(reasons)}
    return {
        "status": "ok",
        "settings": result.settings.as_dict(),
        "summary": result.summary.drop(columns=["description"]).to_dict(orient="records"),
        "by_regime": result.by_regime.to_dict(orient="records"),
    }


def _discount_text(desc: Dict[str, object]) -> str:
    mode = desc.get("mode", "rate")
    curves = desc.get("curves") or {}
    labels = sorted({str(c.get("label")) for c in curves.values()})
    if mode == "rate" or not curves:
        return f"flat rate {float(desc.get('flat_rate_cc', 0.0)):.2%} cc (an input, not market data)"
    head = "put-call parity per expiry where it passes its gates, else " if mode == "parity" else ""
    return head + "; ".join(labels) + f" (trade date to expiry; {len(curves)} trade date(s))"


def report_markdown(summary: Dict[str, object]) -> str:
    """Human-readable summary (every number from ``summary``)."""

    data = summary["data"]
    mc = summary["model_comparison"]
    lines = [
        "# Smile-model metrics",
        "",
        f"Data: {data['source']}; ticker {data['ticker']}; trade date(s) {', '.join(data['trade_dates'])}; "
        f"fetched {data['fetched_at_utc']}; {data['rows']:,} rows; sha256 {data['data_sha256']}.",
        f"Filters: {summary['filters']}. Settings: "
        f"{ {k: v for k, v in summary['settings'].items() if k != 'discounting'} }.",
        f"Discounting: {_discount_text(summary['settings'].get('discounting') or {})}.",
        *(
            []
            if european_style(data.get("ticker"))
            else [
                "",
                f"{data.get('ticker') or 'These'} options are American-style; vols are re-inverted from mids with "
                "European Black-76. At a real discount rate the early-exercise premium of the out-of-the-money puts "
                "(largest at long expiries) reads as extra put vol below the forward, a kink the smiles cannot fit; "
                "with r = 0 that premium mostly vanished, but every vol was biased low (30-50 vol bp at one year).",
            ]
        ),
        "",
        "## Model comparison (one row per expiry, out-of-sample k-fold CV RMSE, vol bp)",
        "",
        f"Slices scored: {mc['slices_scored']}; skipped: {mc['slices_skipped']} (see skipped.csv).",
        "",
        "| model | slices | failed | median | mean | max | lowest on |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in mc["per_model"]:
        lines.append(
            f"| {LABEL.get(r['model'], r['model'])} | {r['slices']} | {r['failed']} | {r['median_rmse_cv_bp']:.1f} | "
            f"{r['mean_rmse_cv_bp']:.1f} | {r['max_rmse_cv_bp']:.1f} | {r['best_on']} |"
        )
    lines += [""] + [f"- {s}" for s in mc["statements"]]
    scope = (
        "These are single-snapshot fit metrics, not forecasts."
        if summary.get("snapshots", 1) == 1
        else f"These are fit metrics of each expiry of each of the {summary['snapshots']} snapshots, pooled; they "
        "are not forecasts (see the walk-forward section)."
    )
    lines += [
        "",
        "The spline interpolates the quotes, so it is scored out of sample only; with dense strikes it is a strong "
        f"interpolation baseline. {scope}",
    ]
    synthetic = [s for s in data.get("row_sources") or [] if "synthetic" in str(s).lower()]
    if synthetic:
        lines[2:2] = [f"**SYNTHETIC DATA, NOT MARKET DATA**: rows with source {', '.join(map(str, synthetic))}.", ""]
    cal = summary.get("calibration")
    if cal:
        lines += [
            "",
            "## Production SVI calibration (arbitrage repair on)",
            "",
            f"- Slices: {cal['slices']}; median in-sample RMSE {cal.get('median_rmse_bp', float('nan')):.1f} vol bp "
            f"(max {cal.get('max_rmse_bp', float('nan')):.1f}).",
            f"- Butterfly (g(k) >= 0): {cal.get('butterfly_ok')}; Lee wings: {cal.get('lee_ok')}; "
            f"calendar: {cal.get('calendar_ok')}"
            + (f" (fails at {', '.join(cal['calendar_failures'])})" if cal.get("calendar_failures") else "")
            + ".",
        ]
    wf = summary["walk_forward"]
    lines += ["", "## Walk-forward", ""]
    if wf["status"] == "ok":
        lines += [
            "| predictor | RMSE (vol bp) | benchmark | skill | days won | quote win rate | DM | days |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in wf["summary"]:

            def fmt(key: str, spec: str) -> str:
                v = r.get(key)
                return "" if v is None or not np.isfinite(v) else format(v, spec)

            days_won = (
                f"{round(r['day_win_rate'] * r['day_win_n'])} of {r['day_win_n']}"
                if r["benchmark"] and r.get("day_win_n")
                else ""
            )
            lines.append(
                f"| {r['predictor']} | {r['rmse_bp']:.1f} | {r['benchmark']} | {fmt('skill', '+.3f')} | {days_won} | "
                f"{fmt('win_rate', '.3f')} | {fmt('dm_t', '+.2f')} | {r['n_days']} |"
            )
        lines += [
            "",
            "Days won: target days on which the forecast's mean squared error is below its benchmark's. The quote "
            "win rate treats quotes as independent (they are not within a day). DM: Diebold-Mariano statistic on the "
            "daily squared-error differential (negative favours the forecast).",
        ]
    else:
        lines.append(f"{wf['status']}: {wf['reason']}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# History mode: the walk-forward on the collected daily snapshots
# --------------------------------------------------------------------------


def load_history(
    ticker: str,
    store_dir: Optional[Path] = None,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
    offline: bool = False,
    first: Optional[str] = None,
    last: Optional[str] = None,
) -> Tuple[ChainDataset, Dict[str, object], pd.DataFrame]:
    """(sessions of ``ticker``, provenance, their index rows) from a local
    checkout ``store_dir`` or the synced copy of ``repo``@``branch``."""

    from derivx.data.snapshots import SnapshotStore, sha256_hex, sync

    if store_dir is not None:
        store = SnapshotStore(store_dir, f"local checkout {store_dir.name}")
        info: Dict[str, object] = {"source": "local checkout of the data branch"}
    else:
        result = sync(repo, branch, tickers=[ticker], offline=True if offline else None)
        store = result.store
        info = {"source": store.label, "url": result.url, "synced_at": result.synced_at, "notices": result.notices}
        if result.failed:
            info["sync_failed"] = list(result.failed)
    index = store.index()
    rows = index[index["ticker"].astype(str) == ticker.upper()].sort_values("session")
    if first:
        rows = rows[rows["session"].astype(str) >= first]
    if last:
        rows = rows[rows["session"].astype(str) <= last]
    if rows.empty:
        raise ValueError(f"no stored {ticker.upper()} sessions in {store.label} (from {first or '-'} to {last or '-'})")
    ds = store.load(ticker, list(rows["session"].astype(str)))
    info.update(
        {
            "ticker": ticker.upper(),
            "sessions": int(len(rows)),
            "first": str(rows["session"].iloc[0]),
            "last": str(rows["session"].iloc[-1]),
            "sessions_sha256": sha256_hex(rows.to_csv(index=False, lineterminator="\n").encode()),
        }
    )
    return ds, info, rows.reset_index(drop=True)


def compute_history(
    ds: ChainDataset,
    info: Dict[str, object],
    *,
    discount: Discounting,
    min_volume: int = 10,
    min_mid: float = 0.10,
    max_spread: Optional[float] = 0.40,
    horizon: int = 1,
    window: int = 1,
    max_days: float = 120.0,
) -> Dict[str, object]:
    """The walk-forward summary and tables of a collected history (see
    the module docs); ``tables`` is empty when too few origin days exist."""

    from derivx.data.calendar import missing_trading_days
    from derivx.data.snapshots import origins_needed, scorable_origins
    from derivx.validation.backtest import WalkForwardSettings

    filtered = {}
    for td, chain in sorted(ds.chains.items()):
        f = apply_liquidity_filters(chain, min_volume=min_volume, max_spread=max_spread, min_mid=min_mid)
        if not f.empty:
            filtered[td] = f
    settings = WalkForwardSettings(horizon=horizon, window=window, rate=discount.rate, discount=discount,
                                   max_days=max_days)  # fmt: skip
    have, need = scorable_origins(ds.trade_dates, horizon), origins_needed(settings.min_regime_days)
    gaps = [d.date().isoformat() for d in missing_trading_days(ds.trade_dates)]
    summary: Dict[str, object] = {
        "mode": "history",
        "data": ds.provenance(),
        "collection": info,
        "coverage": {
            "sessions": len(ds.chains),
            "missing_trading_days": gaps,
            "origin_days": have,
            "origin_days_needed": need,
        },  # fmt: skip
        "filters": {"min_volume": min_volume, "min_mid": min_mid, "max_spread": max_spread},
        "settings": {**settings.as_dict(), "discounting": discount.describe()},
    }
    tables: Dict[str, pd.DataFrame] = {}
    if have < need:
        summary["walk_forward"] = {
            "status": "not run",
            "reason": f"{len(ds.chains)} session(s) give {have} origin day(s) with the session {horizon} trading "
            f"day(s) later; the walk-forward needs {need} (three per vol-regime tercile): collect at least "
            f"{need - have} more trading day(s)",
        }
        return {"summary": summary, "tables": tables}
    result = run_walk_forward(filtered, settings)
    if result.empty:
        reasons = sorted(set(result.skipped["reason"].astype(str)))[:3]
        summary["walk_forward"] = {"status": "no scores", "reason": "; ".join(reasons)}
        return {"summary": summary, "tables": {"skipped": result.skipped}}
    summary["walk_forward"] = {
        "status": "ok",
        "summary": result.summary.drop(columns=["description"]).to_dict(orient="records"),
        "by_regime": result.by_regime.to_dict(orient="records"),
        "target_days": int(result.daily["target"].nunique()),
    }
    tables = {
        "walkforward_summary": result.summary,
        "walkforward_daily": result.daily,
        "walkforward_by_regime": result.by_regime,
        "walkforward_slices": result.slices,
        "walkforward_skipped": result.skipped,
    }
    return {"summary": summary, "tables": tables}


def history_markdown(summary: Dict[str, object]) -> str:
    """The history mode's METRICS_SUMMARY.md (every number from ``summary``)."""

    data, info, cov = summary["data"], summary["collection"], summary["coverage"]
    synthetic = [s for s in data.get("row_sources") or [] if "synthetic" in str(s).lower()]
    lines = ["# Walk-forward on collected history", ""]
    if synthetic:
        lines += [f"**SYNTHETIC DATA, NOT MARKET DATA**: rows with source {', '.join(map(str, synthetic))}.", ""]
    gaps = cov["missing_trading_days"]
    lines += [
        f"Data: {info.get('source')}; ticker {data['ticker']}; {cov['sessions']} session(s) {info.get('first')} .. "
        f"{info.get('last')} ({len(gaps)} missing trading day(s){': ' + ', '.join(gaps[:8]) if gaps else ''}); "
        f"{data['rows']:,} rows; sha256 {data['data_sha256']}; sessions index sha256 {info.get('sessions_sha256')}.",
        f"Filters: {summary['filters']}. Discounting: {_discount_text(summary['settings'].get('discounting') or {})}.",
        f"Origin days at the {summary['settings']['horizon']}-day horizon: {cov['origin_days']} (needed: "
        f"{cov['origin_days_needed']}).",
        "",
        "## Walk-forward",
        "",
    ]
    wf = summary["walk_forward"]
    if wf["status"] != "ok":
        return "\n".join(lines + [f"{wf['status']}: {wf['reason']}"]) + "\n"
    lines += [
        "| predictor | RMSE (vol bp) | benchmark | skill | days won | quote win rate | DM | days |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in wf["summary"]:

        def fmt(key: str, spec: str) -> str:
            v = r.get(key)
            return "" if v is None or not np.isfinite(v) else format(v, spec)

        days_won = (
            f"{round(r['day_win_rate'] * r['day_win_n'])} of {r['day_win_n']}"
            if r["benchmark"] and r.get("day_win_n")
            else ""
        )
        lines.append(
            f"| {r['predictor']} | {r['rmse_bp']:.1f} | {r['benchmark']} | {fmt('skill', '+.3f')} | {days_won} | "
            f"{fmt('win_rate', '.3f')} | {fmt('dm_t', '+.2f')} | {r['n_days']} |"
        )
    lines += [
        "",
        "Days won: target days on which the forecast's mean squared error is below its benchmark's. The quote win "
        "rate treats quotes as independent (they are not within a day). DM: Diebold-Mariano statistic on the daily "
        "squared-error differential (negative favours the forecast). SPY options are American and vols are inverted "
        "with European Black-76 (see the single-snapshot metrics).",
    ]
    return "\n".join(lines) + "\n"


def write_history(
    result: Dict[str, object], sessions: pd.DataFrame, outdir: Path, manifest: Dict[str, object]
) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in result["tables"].items():
        path = outdir / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
        paths.append(path)
    path = outdir / "sessions.csv"
    sessions.to_csv(path, index=False, lineterminator="\n")
    summary = jsonable(result["summary"])
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (outdir / "METRICS_SUMMARY.md").write_text(history_markdown(summary))
    (outdir / "manifest.json").write_bytes(manifest_bytes(manifest))
    return paths + [path, outdir / "summary.json", outdir / "METRICS_SUMMARY.md", outdir / "manifest.json"]


def main_history(args: argparse.Namespace, argv: Optional[Sequence[str]]) -> int:
    from derivx.data.snapshots import SnapshotError

    try:
        ds, info, sessions = load_history(args.ticker, args.store, args.repo, args.branch, args.offline,
                                          args.first, args.last)  # fmt: skip
    except (SnapshotError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(ds.chains)} collected session(s) of {ds.ticker} ({ds.rows:,} rows) from {info['source']}")
    discount = discounting_for(ds, args.discount, args.rate, with_cache=args.with_cache)
    try:
        result = compute_history(ds, info, discount=discount, min_volume=args.min_volume, min_mid=args.min_mid,
                                 max_spread=args.max_spread, horizon=args.horizon, window=args.window,
                                 max_days=args.max_days if args.max_days is not None else 120.0)  # fmt: skip
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    manifest = build_manifest(
        "metrics_history",
        inputs={"argv": list(argv) if argv is not None else sys.argv[1:], **result["summary"]["filters"]},
        data={**ds.provenance(), "collection": info, "files": sessions[["path", "sha256"]].to_dict(orient="records")},
        model=result["summary"]["settings"],
    )
    for path in write_history(result, sessions, args.outdir, manifest):
        print(f"wrote {path}")
    wf = result["summary"]["walk_forward"]
    print(f"walk-forward: {wf['status']}" + (f" ({wf['reason']})" if wf.get("reason") else ""))
    return 0


def write_outputs(result: Dict[str, object], outdir: Path, manifest: Dict[str, object]) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("scores", "skipped", "calibration"):
        table = result[name]
        path = outdir / ("model_scores.csv" if name == "scores" else f"{name}.csv")
        table.to_csv(path, index=False, float_format="%.6g")
        paths.append(path)
    summary = jsonable(result["summary"])
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (outdir / "METRICS_SUMMARY.md").write_text(report_markdown(summary))
    (outdir / "manifest.json").write_bytes(manifest_bytes(manifest))
    return paths + [outdir / "summary.json", outdir / "METRICS_SUMMARY.md", outdir / "manifest.json"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--chain", type=Path, nargs="*", default=[], help=f"snapshot file(s) (default {SAMPLE_CHAIN.name})"
    )
    parser.add_argument("--chain-dir", type=Path, default=None, help="directory of snapshot files")
    parser.add_argument("--outdir", type=Path, default=Path("data/metrics"), help="output directory")
    parser.add_argument(
        "--discount",
        choices=["curve", "parity", "rate"],
        default="curve",
        help="curve: US Treasury curve of each trade date (default); parity: put-call parity DF per expiry "
        "(European options), else the curve; rate: the flat --rate",
    )
    parser.add_argument("--rate", type=float, default=0.0, help="flat / fallback discount rate, cc decimal")
    parser.add_argument("--min-volume", type=int, default=10)
    parser.add_argument("--min-mid", type=float, default=0.10)
    parser.add_argument("--max-spread", type=float, default=0.40)
    parser.add_argument("--min-days", type=float, default=2.0)
    parser.add_argument("--max-days", type=float, default=None)
    parser.add_argument("--sabr-beta", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-calibration", action="store_true", help="skip the production SVI calibration")
    parser.add_argument("--expiries", nargs="*", help="score only these expiries (YYYY-MM-DD)")
    hist = parser.add_argument_group("history mode (the walk-forward on the collected daily snapshots)")
    hist.add_argument("--history", action="store_true", help="read the synced data-snapshots store")
    hist.add_argument("--store", type=Path, help="a local checkout of the data branch (implies --history)")
    hist.add_argument("--repo", help="OWNER/NAME (default $DERIVX_SNAPSHOT_REPO or ZealousEar/derivx-fic-analytics)")
    hist.add_argument("--branch", help="data branch (default $DERIVX_SNAPSHOT_BRANCH or data-snapshots)")
    hist.add_argument("--offline", action="store_true", help="read only the synced copy in the cache")
    hist.add_argument("--ticker", default="SPY", help="ticker of the collected history (default SPY)")
    hist.add_argument("--first", help="first session to use (YYYY-MM-DD)")
    hist.add_argument("--last", help="last session to use (YYYY-MM-DD)")
    hist.add_argument("--horizon", type=int, default=1, help="trading days from origin to target (default 1)")
    hist.add_argument("--window", type=int, default=1, help="snapshots pooled in each SVI fit (default 1)")
    hist.add_argument("--with-cache", action="store_true",
                      help="discount with the download cache's Treasury curves too (later trade dates)")  # fmt: skip
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.history or args.store is not None:
        return main_history(args, argv)
    try:
        ds = load_inputs(args.chain, args.chain_dir)
    except ChainSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {ds.rows:,} rows, {len(ds.chains)} snapshot(s) from {ds.source}")
    try:
        result = compute(
            ds,
            rate=args.rate,
            discount=discounting_for(ds, args.discount, args.rate),
            min_volume=args.min_volume,
            min_mid=args.min_mid,
            max_spread=args.max_spread,
            min_days=args.min_days,
            max_days=args.max_days,
            sabr_beta=args.sabr_beta,
            folds=args.folds,
            calibration=not args.no_calibration,
            expiries=args.expiries,
        )
    except ValueError as exc:  # e.g. a min volume filter on files without a volume column
        print(f"error: {exc}", file=sys.stderr)
        return 1
    manifest = build_manifest(
        "metrics",
        inputs={"argv": list(argv) if argv is not None else sys.argv[1:], **result["summary"]["filters"]},
        data=ds.provenance(),
        model=result["summary"]["settings"],
    )
    for path in write_outputs(result, args.outdir, manifest):
        print(f"wrote {path}")
    for statement in result["summary"]["model_comparison"]["statements"]:
        print(statement)
    wf = result["summary"]["walk_forward"]
    print(f"walk-forward: {wf['status']}" + (f" ({wf['reason']})" if wf.get("reason") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
