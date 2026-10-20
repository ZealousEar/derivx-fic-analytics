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
   with discount rate ``--rate``.
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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from derivx.calibration.production import calibrate_chain
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


def compute(
    ds: ChainDataset,
    *,
    rate: float = 0.0,
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
    """All tables and the summary for ``ds`` (see the module docs)."""

    scores, skipped, calib = [], [], []
    filtered: Dict[pd.Timestamp, pd.DataFrame] = {}
    for td, chain in sorted(ds.chains.items()):
        f = apply_liquidity_filters(chain, min_volume=min_volume, max_spread=max_spread, min_mid=min_mid)
        if expiries:
            f = f[pd.to_datetime(f["expiry"]).dt.strftime("%Y-%m-%d").isin(set(expiries))]
        filtered[td] = f
        s, k = compare_models_by_expiry(
            f, MODELS, rate=rate, min_days=min_days, max_days=max_days, sabr_beta=sabr_beta, folds=folds
        )
        scores.append(s)
        skipped.append(k)
        if calibration:
            cal = calibrate_chain(f, rate=rate, min_days=min_days, cv_folds=0, enforce_arbitrage=True)
            table = cal.table()
            if max_days is not None and not table.empty:
                table = table[(pd.to_datetime(table["expiry"]) - td).dt.days <= max_days]
            calib.append(table)
    scores_df = (
        pd.concat([s for s in scores if not s.empty], ignore_index=True)
        if any(not s.empty for s in scores)
        else pd.DataFrame()
    )
    skipped_df = pd.concat(skipped, ignore_index=True) if skipped else pd.DataFrame()
    calib_df = pd.concat(calib, ignore_index=True) if calib else pd.DataFrame()
    per_model = summarise_comparison(scores_df)
    pairs = {
        f"{a}_vs_{b}": pairwise_wins(scores_df, a, b)
        for a, b in (("svi", "sabr"), ("svi", "spline"), ("sabr", "spline"))
    }
    summary: Dict[str, object] = {
        "data": ds.provenance(),
        "filters": {"min_volume": min_volume, "min_mid": min_mid, "max_spread": max_spread},
        "settings": {
            "rate_cc": rate,
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
    summary["walk_forward"] = _walk_forward(filtered, rate=rate, max_days=max_days)
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
    filtered: Dict[pd.Timestamp, pd.DataFrame], rate: float, max_days: Optional[float]
) -> Dict[str, object]:
    if len(filtered) < 2:
        return {
            "status": "not run",
            "reason": f"{len(filtered)} snapshot(s): a walk-forward test needs daily snapshots one trading day apart. "
            "Yahoo has no history; collect one snapshot per trading day with tools/download_spy_chains.py.",
        }
    result = run_walk_forward(filtered, rate=rate, max_days=max_days if max_days is not None else 120.0)
    if result.empty:
        reasons = sorted(set(result.skipped["reason"].astype(str)))[:3]
        return {"status": "no scores", "reason": "; ".join(reasons)}
    return {
        "status": "ok",
        "settings": result.settings.as_dict(),
        "summary": result.summary.drop(columns=["description"]).to_dict(orient="records"),
        "by_regime": result.by_regime.to_dict(orient="records"),
    }


def report_markdown(summary: Dict[str, object]) -> str:
    """Human-readable summary (every number from ``summary``)."""

    data = summary["data"]
    mc = summary["model_comparison"]
    lines = [
        "# Smile-model metrics",
        "",
        f"Data: {data['source']}; ticker {data['ticker']}; trade date(s) {', '.join(data['trade_dates'])}; "
        f"fetched {data['fetched_at_utc']}; {data['rows']:,} rows; sha256 {data['data_sha256']}.",
        f"Filters: {summary['filters']}. Settings: {summary['settings']}.",
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
    lines += [
        "",
        "The spline interpolates the quotes, so it is scored out of sample only; with dense strikes it is a strong "
        "interpolation baseline. These are single-snapshot fit metrics, not forecasts.",
    ]
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
        lines += ["| predictor | RMSE (vol bp) | benchmark | skill | win rate | days |", "|---|---|---|---|---|---|"]
        for r in wf["summary"]:
            skill = "" if r["skill"] is None or not np.isfinite(r["skill"]) else f"{r['skill']:+.3f}"
            win = "" if r["win_rate"] is None or not np.isfinite(r["win_rate"]) else f"{r['win_rate']:.3f}"
            lines.append(
                f"| {r['predictor']} | {r['rmse_bp']:.1f} | {r['benchmark']} | {skill} | {win} | {r['n_days']} |"
            )
    else:
        lines.append(f"{wf['status']}: {wf['reason']}")
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--rate", type=float, default=0.0, help="discount rate, cc decimal (default 0)")
    parser.add_argument("--min-volume", type=int, default=10)
    parser.add_argument("--min-mid", type=float, default=0.10)
    parser.add_argument("--max-spread", type=float, default=0.40)
    parser.add_argument("--min-days", type=float, default=2.0)
    parser.add_argument("--max-days", type=float, default=None)
    parser.add_argument("--sabr-beta", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-calibration", action="store_true", help="skip the production SVI calibration")
    parser.add_argument("--expiries", nargs="*", help="score only these expiries (YYYY-MM-DD)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        ds = load_inputs(args.chain, args.chain_dir)
    except ChainSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {ds.rows:,} rows, {len(ds.chains)} snapshot(s) from {ds.source}")
    result = compute(
        ds,
        rate=args.rate,
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
