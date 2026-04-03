"""
Visualization for PDBbind CleanSplit benchmark comparison.

Generates:
  1. Scatter plots: predicted vs true pK for each method
  2. Bar chart: all methods compared on each metric
  3. Residual distributions

Usage:
    python benchmarks/05_pdbbind_comparison/plot_pdbbind_comparison.py \
        --results-dir benchmarks/05_pdbbind_comparison/results \
        --output-dir benchmarks/05_pdbbind_comparison/results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "metrics", PROJECT_ROOT / "benchmarks" / "utils" / "metrics.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
summarize_regression = _mod.summarize_regression

METRIC_DISPLAY = {
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "pearson": "Pearson R",
    "spearman": "Spearman ρ",
    "kendall": "Kendall τ",
    "ci": "CI",
}


def load_results(results_dir: Path, test_set: str = "casf2016") -> dict:
    """Load all training summaries with per_complex_predictions."""
    results = {}
    for path in sorted(results_dir.rglob("*_training_summary.json")):
        with open(path) as f:
            summary = json.load(f)

        if summary.get("test_set") != test_set:
            continue
        if "per_complex_predictions" not in summary:
            continue

        method = summary.get("model_type", path.stem)
        preds = summary["per_complex_predictions"]
        y_true = [v["y_true"] for v in preds.values()]
        y_pred = [v["y_pred"] for v in preds.values()]

        results[method] = {
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
            "metrics": summary.get("training_history", {}).get("test_metrics", {}),
            "n": len(y_true),
        }

    return results


def plot_scatter_grid(results: dict, output_path: Path, test_set: str) -> None:
    """Generate scatter plots of predicted vs true pK for each method."""
    n_methods = len(results)
    if n_methods == 0:
        return

    ncols = min(3, n_methods)
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for idx, (method, data) in enumerate(results.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        y_true = data["y_true"]
        y_pred = data["y_pred"]
        metrics = data["metrics"]

        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="none")

        # Diagonal line
        lims = [
            min(y_true.min(), y_pred.min()) - 0.5,
            max(y_true.max(), y_pred.max()) + 0.5,
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("True pK")
        ax.set_ylabel("Predicted pK")

        pearson = metrics.get("pearson", 0)
        rmse = metrics.get("rmse", 0)
        ax.set_title(f"{method}\nR={pearson:.3f}, RMSE={rmse:.2f} (n={data['n']})")
        ax.set_aspect("equal")

    # Hide unused subplots
    for idx in range(n_methods, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"PDBbind CleanSplit — {test_set}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter grid saved to {output_path}")


def plot_metric_bars(results: dict, output_path: Path, test_set: str) -> None:
    """Bar chart comparing all methods on each metric."""
    if not results:
        return

    metrics_to_plot = ["rmse", "mae", "r2", "pearson", "spearman", "kendall", "ci"]
    methods = list(results.keys())

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(3 * len(metrics_to_plot), 5))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        values = []
        for method in methods:
            v = results[method]["metrics"].get(metric, 0)
            values.append(v if v is not None else 0)

        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        bars = ax.bar(range(len(methods)), values, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_title(METRIC_DISPLAY.get(metric, metric))

        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(f"PDBbind CleanSplit — {test_set}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metric bars saved to {output_path}")


def plot_residuals(results: dict, output_path: Path, test_set: str) -> None:
    """Residual distribution histograms."""
    n_methods = len(results)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), squeeze=False)

    for idx, (method, data) in enumerate(results.items()):
        ax = axes[0][idx]
        residuals = data["y_pred"] - data["y_true"]
        ax.hist(residuals, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (pred - true)")
        ax.set_ylabel("Count")
        ax.set_title(f"{method}\nmean={residuals.mean():.2f}, std={residuals.std():.2f}")

    fig.suptitle(f"Residual Distributions — {test_set}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Residuals saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDBbind CleanSplit benchmark visualizations"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/05_pdbbind_comparison/results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: results-dir/figures)",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="casf2016",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir} for {args.test_set}...")
    results = load_results(results_dir, test_set=args.test_set)

    if not results:
        print("No results with per_complex_predictions found.")
        print("Run the benchmark scripts first.")
        return

    print(f"Found {len(results)} methods: {list(results.keys())}\n")

    plot_scatter_grid(results, output_dir / f"scatter_{args.test_set}.png", args.test_set)
    plot_metric_bars(results, output_dir / f"metrics_{args.test_set}.png", args.test_set)
    plot_residuals(results, output_dir / f"residuals_{args.test_set}.png", args.test_set)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
