"""Compute resume-ready metrics from downloaded SPY chains."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from derivx.calibration.production import calibrate_with_constraints
from derivx.models.comparison import compare_models
from derivx.validation.backtest import walk_forward_test


def load_chains_from_dir(chain_dir: Path) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Load all Parquet chains from a directory."""
    dataset: Dict[pd.Timestamp, pd.DataFrame] = {}
    for parquet_file in sorted(chain_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        if df.empty or "trade_date" not in df.columns:
            continue
        trade_date = pd.to_datetime(df["trade_date"].iloc[0]).tz_localize(None).normalize()
        dataset[trade_date] = df
    return dataset


def compute_svi_metrics(chains: Dict[pd.Timestamp, pd.DataFrame], outdir: Path) -> pd.DataFrame:
    """Compute SVI calibration metrics across all chains."""
    rows = []
    for trade_date, chain in chains.items():
        slices = calibrate_with_constraints(chain, enforce_arbitrage=True, moneyness_band=(0.85, 1.15))
        for slc in slices:
            bench = slc.benchmark or {}
            rows.append(
                {
                    "trade_date": trade_date,
                    "expiry": slc.expiry,
                    "rmse": slc.rmse,
                    "n_points": slc.n_points,
                    "rmse_spline": bench.get("rmse_spline"),
                    "relative_improvement": bench.get("relative_improvement"),
                    "tau": slc.tau,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        outfile = outdir / "svi_rmse_summary.csv"
        df.to_csv(outfile, index=False)
        print(f"Saved SVI metrics: {len(df)} slices -> {outfile}")
    return df


def compute_walkforward_metrics(chains: Dict[pd.Timestamp, pd.DataFrame], outdir: Path) -> pd.DataFrame:
    """Run walk-forward backtest and segment by regime."""
    if len(chains) < 12:
        print(f"Skipping walk-forward: need >=12 days, got {len(chains)}")
        return pd.DataFrame()

    metrics = walk_forward_test(chains, window=10, moneyness_band=(0.85, 1.15))
    if metrics.empty:
        print("Walk-forward returned no metrics")
        return pd.DataFrame()

    # Infer regimes from ATM IV
    regimes = {}
    for trade_date, chain in chains.items():
        atm = chain.loc[chain["moneyness"].between(0.95, 1.05)]
        mean_iv = float(atm["implied_vol"].mean()) if not atm.empty else float(chain["implied_vol"].mean())
        if not np.isfinite(mean_iv):
            regimes[trade_date] = "n/a"
        elif mean_iv < 0.15:
            regimes[trade_date] = "low"
        elif mean_iv < 0.25:
            regimes[trade_date] = "mid"
        else:
            regimes[trade_date] = "high"
    metrics["regime"] = metrics["trade_date"].map(regimes)

    # Aggregate by regime
    regime_summary = (
        metrics.dropna(subset=["regime"])
        .groupby("regime")
        .agg(
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_hit_rate=("hit_rate", "mean"),
            observations=("regime", "count"),
        )
        .reset_index()
    )

    outfile = outdir / "walkforward_regimes.csv"
    regime_summary.to_csv(outfile, index=False)
    print(f"Saved walk-forward metrics: {len(regime_summary)} regimes -> {outfile}")

    full_outfile = outdir / "walkforward_full.csv"
    metrics.to_csv(full_outfile, index=False)
    print(f"Saved full walk-forward results: {len(metrics)} rows -> {full_outfile}")

    return regime_summary


def compute_model_comparison_metrics(chains: Dict[pd.Timestamp, pd.DataFrame], outdir: Path) -> pd.DataFrame:
    """Compare SVI/SABR/local-vol on representative slices."""
    rows = []
    for trade_date, chain in chains.items():
        expiries = sorted(pd.to_datetime(chain["expiry"]).dt.date.unique())
        # Use first expiry for each day
        if not expiries:
            continue
        expiry = expiries[0]
        slice_df = chain[pd.to_datetime(chain["expiry"]).dt.date == expiry]
        if slice_df.empty or len(slice_df) < 10:
            continue
        try:
            comp = compare_models(slice_df, models=["svi", "sabr", "local_vol"])
            comp["trade_date"] = trade_date
            comp["expiry"] = pd.Timestamp(expiry)
            rows.append(comp)
        except Exception:
            continue

    if not rows:
        print("No model comparison results")
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    outfile = outdir / "model_comparison.csv"
    df.to_csv(outfile, index=False)
    print(f"Saved model comparison: {len(df)} rows -> {outfile}")
    return df


def generate_summary_report(svi_df: pd.DataFrame, wf_df: pd.DataFrame, comp_df: pd.DataFrame, outdir: Path) -> None:
    """Generate a markdown summary report."""
    lines = ["# Metrics Summary Report\n"]
    lines.append("## SVI Calibration Performance\n")
    if not svi_df.empty:
        mean_rmse = svi_df["rmse"].mean()
        median_rmse = svi_df["rmse"].median()
        improvement = svi_df["relative_improvement"].dropna()
        # Filter out extreme outliers (likely calculation errors)
        improvement = improvement[(improvement > -1) & (improvement < 1)]
        mean_improvement = improvement.mean() * 100 if not improvement.empty else None
        lines.append(f"- **Mean RMSE**: {mean_rmse:.4f}")
        lines.append(f"- **Median RMSE**: {median_rmse:.4f}")
        if mean_improvement is not None:
            lines.append(f"- **Mean improvement vs spline**: {mean_improvement:.1f}%")
        lines.append(f"- **Total slices calibrated**: {len(svi_df)}")
        lines.append(f"- **Unique expiries**: {svi_df['expiry'].nunique()}")
        lines.append(f"- **Unique trading days**: {svi_df['trade_date'].nunique()}")
    else:
        lines.append("- No SVI calibration data available")

    lines.append("\n## Walk-Forward Backtest\n")
    if not wf_df.empty:
        for _, row in wf_df.iterrows():
            regime = row["regime"]
            lines.append(f"### {regime.upper()} Volatility Regime")
            lines.append(f"- Mean RMSE: {row['mean_rmse']:.4f}")
            lines.append(f"- Median RMSE: {row['median_rmse']:.4f}")
            lines.append(f"- Hit Rate: {row['mean_hit_rate']:.2%}")
            lines.append(f"- Observations: {int(row['observations'])}")
    else:
        lines.append("- No walk-forward data available")

    lines.append("\n## Model Comparison\n")
    if not comp_df.empty:
        svi_data = comp_df[comp_df["model"] == "svi"]["rmse"].dropna()
        sabr_data = comp_df[comp_df["model"] == "sabr"]["rmse"].dropna()
        local_data = comp_df[comp_df["model"] == "local_vol"]["rmse"].dropna()
        if not svi_data.empty:
            lines.append(f"- **SVI mean RMSE**: {svi_data.mean():.4f}")
        if not sabr_data.empty:
            lines.append(f"- **SABR mean RMSE**: {sabr_data.mean():.4f}")
            if not svi_data.empty:
                improvement_pct = ((sabr_data.mean() - svi_data.mean()) / sabr_data.mean()) * 100
                lines.append(f"- **SVI outperforms SABR by**: {improvement_pct:.1f}%")
        if not local_data.empty:
            lines.append(f"- **Local Vol mean RMSE**: {local_data.mean():.4f}")
        lines.append(f"- **Total comparisons**: {len(comp_df[comp_df['model'] == 'svi'])}")
    else:
        lines.append("- No model comparison data available")

    report_file = outdir / "METRICS_SUMMARY.md"
    report_file.write_text("\n".join(lines))
    print(f"Saved summary report -> {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-dir", type=Path, default=Path("data/metrics/chains"), help="Directory with Parquet chains")
    parser.add_argument("--outdir", type=Path, default=Path("data/metrics"), help="Output directory for metrics")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loading chains from {args.chain_dir}...")
    chains = load_chains_from_dir(args.chain_dir)
    print(f"Loaded {len(chains)} trading days")

    print("\nComputing SVI calibration metrics...")
    svi_df = compute_svi_metrics(chains, args.outdir)

    print("\nComputing walk-forward metrics...")
    wf_df = compute_walkforward_metrics(chains, args.outdir)

    print("\nComputing model comparison metrics...")
    comp_df = compute_model_comparison_metrics(chains, args.outdir)

    print("\nGenerating summary report...")
    generate_summary_report(svi_df, wf_df, comp_df, args.outdir)

    print("\n✅ Metrics computation complete!")


if __name__ == "__main__":
    main()

