"""
Benchmark Report Generation Script for affinity regression runs.

Scans a directory for *_training_summary.json files produced by
train_deeppurpose_wrapper.py and prints a comparison table across models,
similarity thresholds, and splits for the regression track.

Usage:
    python generate_regression_report.py --results-dir ../02_training/trained_models
    python generate_regression_report.py --results-dir ../02_training/trained_models --output regression_report.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


REPORT_METRICS = ["rmse", "mae", "r2", "pearson", "spearman", "kendall", "ci", "mse"]
SPLIT_NAMES = ["train", "val", "test"]


def load_summary(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def collect_summaries(results_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(results_dir.rglob("*_training_summary.json")):
        try:
            summary = load_summary(path)
        except Exception as exc:
            print(f"Warning: could not load {path}: {exc}", file=sys.stderr)
            continue

        if summary.get("task_type") != "regression":
            continue

        summary["_source_file"] = str(path)
        summaries.append(summary)
    return summaries


def parse_row(summary: dict) -> dict:
    history = summary.get("training_history", {})
    row = {
        "model": summary.get("model_type", "unknown"),
        "model_architecture": summary.get("model_architecture", "unknown"),
        "feature_type": summary.get("feature_type", "unknown"),
        "target_transform": summary.get("target_transform", "unknown"),
        "similarity_threshold": summary.get("similarity_threshold", "unknown"),
        "precomputed_split": summary.get("use_precomputed_split", False),
        "source_file": summary.get("_source_file", ""),
    }

    for split in SPLIT_NAMES:
        metrics = history.get(f"{split}_metrics", {})
        for metric in REPORT_METRICS:
            value = metrics.get(metric)
            row[f"{split}_{metric}"] = round(value, 4) if value is not None else None

    row["n_train"] = history.get("n_train_samples")
    row["n_val"] = history.get("n_val_samples")
    row["n_test"] = history.get("n_test_samples")
    row["training_time_s"] = round(history.get("training_time", 0), 1)
    return row


def _format_val(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_plain_table(rows: list[dict], focus_split: str = "test") -> None:
    if not rows:
        print("No regression results found.")
        return

    key_cols = [
        "model",
        "model_architecture",
        "similarity_threshold",
        "target_transform",
        "n_train",
        "n_test",
        "training_time_s",
    ]
    metric_cols = [f"{focus_split}_{metric}" for metric in REPORT_METRICS]
    all_cols = key_cols + metric_cols

    header = "  ".join(f"{column:<22}" for column in all_cols)
    print(header)
    print("-" * len(header))

    for row in rows:
        line = "  ".join(f"{_format_val(row.get(column)):<22}" for column in all_cols)
        print(line)


def print_pandas_table(rows: list[dict], focus_split: str = "test") -> None:
    dataframe = pd.DataFrame(rows)
    key_cols = [
        "model",
        "model_architecture",
        "similarity_threshold",
        "target_transform",
        "n_train",
        "n_test",
        "training_time_s",
    ]
    metric_cols = [
        f"{focus_split}_{metric}" for metric in REPORT_METRICS if f"{focus_split}_{metric}" in dataframe.columns
    ]
    display_cols = [column for column in key_cols + metric_cols if column in dataframe.columns]
    print(dataframe[display_cols].to_string(index=False))


def print_all_splits(rows: list[dict]) -> None:
    for split in SPLIT_NAMES:
        print(f"\n{'=' * 70}")
        print(f"  Split: {split.upper()}")
        print("=" * 70)
        if PANDAS_AVAILABLE:
            print_pandas_table(rows, focus_split=split)
        else:
            print_plain_table(rows, focus_split=split)


def generate_report(results_dir: str, output_csv: str = None, split: str = "all", verbose: bool = False):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = collect_summaries(results_path)
    if not summaries:
        print(f"No regression *_training_summary.json files found in {results_dir}")
        print("Run train_deeppurpose_wrapper.py first to generate results.")
        return []

    rows = [parse_row(summary) for summary in summaries]

    print(f"\nFound {len(rows)} regression training run(s) in {results_dir}\n")

    if split == "all":
        print_all_splits(rows)
    else:
        print(f"\n{'=' * 70}")
        print(f"  Split: {split.upper()}")
        print("=" * 70)
        if PANDAS_AVAILABLE:
            print_pandas_table(rows, focus_split=split)
        else:
            print_plain_table(rows, focus_split=split)

    if verbose:
        print(f"\n{'=' * 70}")
        print("  Source files")
        print("=" * 70)
        for row in rows:
            print(f"  {row['model']:20s}  {row['source_file']}")

    if output_csv:
        if PANDAS_AVAILABLE:
            pd.DataFrame(rows).to_csv(output_csv, index=False)
        else:
            import csv

            with open(output_csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"\nReport saved to {output_csv}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate regression benchmark report from training run summaries")
    parser.add_argument(
        "--results-dir",
        "-d",
        type=str,
        default="../02_training/trained_models",
        help="Directory containing *_training_summary.json files",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional CSV output path")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to display (default: all)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show source file paths")
    args = parser.parse_args()
    generate_report(args.results_dir, args.output, args.split, args.verbose)


if __name__ == "__main__":
    main()