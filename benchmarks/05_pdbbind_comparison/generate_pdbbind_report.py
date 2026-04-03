"""
Generate comparison report for PDBbind CleanSplit benchmark.

Scans results directories for *_training_summary.json files produced by
run_gems_inference.py, train_classical_pdbbind.py, and collect_gnina_pdbbind.py.
Produces a unified comparison table across all methods.

Usage:
    python benchmarks/05_pdbbind_comparison/generate_pdbbind_report.py

    # Specify results directories
    python benchmarks/05_pdbbind_comparison/generate_pdbbind_report.py \
        --results-dir benchmarks/05_pdbbind_comparison/results \
        --output benchmarks/05_pdbbind_comparison/results/pdbbind_comparison_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

REPORT_METRICS = ["rmse", "mae", "r2", "pearson", "spearman", "kendall", "ci", "mse"]


def load_summary(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def collect_summaries(results_dir: Path) -> list[dict]:
    """Recursively find all *_training_summary.json files."""
    summaries = []
    for path in sorted(results_dir.rglob("*_training_summary.json")):
        try:
            summary = load_summary(path)
        except Exception as exc:
            print(f"Warning: could not load {path}: {exc}", file=sys.stderr)
            continue

        # Only include PDBbind CleanSplit results
        if summary.get("dataset") != "pdbbind_cleansplit":
            continue

        summary["_source_file"] = str(path)
        summaries.append(summary)

    return summaries


def parse_row(summary: dict) -> dict:
    """Flatten a training summary into a report row."""
    history = summary.get("training_history", {})

    row = {
        "method": summary.get("model_type", "unknown"),
        "architecture": summary.get("model_architecture", ""),
        "feature_type": summary.get("feature_type", ""),
        "dataset": summary.get("dataset", ""),
        "test_set": summary.get("test_set", ""),
        "n_train": history.get("n_train_samples"),
        "n_test": history.get("n_test_samples"),
        "time_s": round(history.get("training_time", 0), 1),
        "source_file": summary.get("_source_file", ""),
    }

    # Test metrics
    test_metrics = history.get("test_metrics", {})
    for metric in REPORT_METRICS:
        value = test_metrics.get(metric)
        row[f"test_{metric}"] = round(value, 4) if value is not None else None

    # Train metrics (if available)
    train_metrics = history.get("train_metrics") or {}
    for metric in REPORT_METRICS:
        value = train_metrics.get(metric)
        row[f"train_{metric}"] = round(value, 4) if value is not None else None

    return row


def format_val(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_comparison_table(rows: list[dict]) -> None:
    """Print a focused comparison table for the key metrics."""
    if not rows:
        print("No PDBbind CleanSplit results found.")
        return

    # Key columns for the comparison view
    key_cols = ["method", "feature_type", "test_set"]
    metric_cols = [f"test_{m}" for m in ["rmse", "mae", "r2", "pearson", "spearman", "kendall", "ci"]]
    extra_cols = ["n_test", "time_s"]
    all_cols = key_cols + metric_cols + extra_cols

    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
        display_cols = [c for c in all_cols if c in df.columns]
        print(df[display_cols].to_string(index=False))
    else:
        header = "  ".join(f"{c:<20}" for c in all_cols)
        print(header)
        print("-" * len(header))
        for row in rows:
            line = "  ".join(f"{format_val(row.get(c)):<20}" for c in all_cols)
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDBbind CleanSplit benchmark comparison report"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/05_pdbbind_comparison/results",
        help="Directory to scan for *_training_summary.json files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        help="Filter to specific test set (casf2016, casf2013)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show source file paths",
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: results directory not found: {args.results_dir}")
        sys.exit(1)

    summaries = collect_summaries(results_path)
    if not summaries:
        print(f"No PDBbind *_training_summary.json files found in {args.results_dir}")
        print("Run the benchmark scripts first:")
        print("  - run_gems_inference.py")
        print("  - train_classical_pdbbind.py")
        print("  - collect_gnina_pdbbind.py")
        return

    rows = [parse_row(s) for s in summaries]

    # Filter by test set if specified
    if args.test_set:
        rows = [r for r in rows if r["test_set"] == args.test_set]

    print(f"\nFound {len(rows)} result(s) in {args.results_dir}\n")
    print("=" * 80)
    print("  PDBbind CleanSplit Benchmark Comparison")
    print("=" * 80)
    print()
    print_comparison_table(rows)

    if args.verbose:
        print(f"\n{'='*80}")
        print("  Source files")
        print("=" * 80)
        for row in rows:
            print(f"  {row['method']:20s}  {row['source_file']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if PANDAS_AVAILABLE:
            pd.DataFrame(rows).to_csv(output_path, index=False)
        else:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
