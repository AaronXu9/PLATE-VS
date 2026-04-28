"""Generate gnina-vs-unidock2 comparison figures.

Reads the comparison CSV and emits four figures into
benchmarks/04_docking/results/figures/:

  1. roc_auc_per_target.png — paired bar chart per target
  2. ef1pct_per_target.png  — paired bar chart per target
  3. ef5pct_per_target.png  — paired bar chart per target
  4. scatter_gnina_vs_unidock2.png — per-target ROC-AUC + EF1% scatter
     with x=y diagonal so wins are above the line.

Run:
    LD_LIBRARY_PATH=/home/aoxu/miniconda3/envs/rdkit_env/lib:$LD_LIBRARY_PATH \\
        /home/aoxu/miniconda3/envs/rdkit_env/bin/python \\
        benchmarks/04_docking/make_comparison_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GNINA_COLOR = "#4C72B0"
UNIDOCK2_COLOR = "#DD8452"


def _paired_bar(ax, df: pd.DataFrame, gcol: str, ucol: str, ylabel: str, title: str):
    targets = df["uniprot"].tolist()
    x = np.arange(len(targets))
    w = 0.4
    ax.bar(x - w / 2, df[gcol].fillna(0), w, label="GNINA", color=GNINA_COLOR)
    ax.bar(x + w / 2, df[ucol].fillna(0), w, label="UniDock2", color=UNIDOCK2_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)


def _scatter(ax, df: pd.DataFrame, gcol: str, ucol: str, label: str, lim: tuple[float, float]):
    g = df[gcol].fillna(0).to_numpy()
    u = df[ucol].fillna(0).to_numpy()
    ax.scatter(g, u, s=70, alpha=0.85, edgecolor="black", linewidth=0.5, color=UNIDOCK2_COLOR)
    for _, row in df.iterrows():
        ax.annotate(
            row["uniprot"],
            (row[gcol] if pd.notna(row[gcol]) else 0, row[ucol] if pd.notna(row[ucol]) else 0),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
            color="#444",
        )
    ax.plot(lim, lim, ls="--", color="grey", alpha=0.6, label="x = y")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"GNINA {label}")
    ax.set_ylabel(f"UniDock2 {label}")
    ax.set_title(f"Per-target {label}: above diagonal = UniDock2 wins")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="/home/aoxu/projects/VLS-Benchmark-Dataset/benchmarks/04_docking/results/comparison_gnina_vs_unidock2.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="/home/aoxu/projects/VLS-Benchmark-Dataset/benchmarks/04_docking/results/figures",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values("uniprot").reset_index(drop=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. ROC-AUC per target
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _paired_bar(ax, df, "gnina_roc_auc", "unidock2_roc_auc",
                ylabel="ROC-AUC", title="Per-target ROC-AUC (15 targets)")
    ax.axhline(0.5, ls=":", color="black", alpha=0.5, label="random")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_auc_per_target.png", dpi=150)
    plt.close(fig)

    # --- 2. EF1% per target
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _paired_bar(ax, df, "gnina_ef1pct", "unidock2_ef1pct",
                ylabel="EF top-1%", title="Per-target enrichment factor at top 1%")
    ax.axhline(1.0, ls=":", color="black", alpha=0.5, label="random")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "ef1pct_per_target.png", dpi=150)
    plt.close(fig)

    # --- 3. EF5% per target
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _paired_bar(ax, df, "gnina_ef5pct", "unidock2_ef5pct",
                ylabel="EF top-5%", title="Per-target enrichment factor at top 5%")
    ax.axhline(1.0, ls=":", color="black", alpha=0.5, label="random")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "ef5pct_per_target.png", dpi=150)
    plt.close(fig)

    # --- 4. Scatter ROC-AUC + EF1% side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    _scatter(axes[0], df, "gnina_roc_auc", "unidock2_roc_auc", "ROC-AUC", lim=(0.3, 1.0))
    ef_max = max(df["gnina_ef1pct"].max(), df["unidock2_ef1pct"].max()) * 1.1
    _scatter(axes[1], df, "gnina_ef1pct", "unidock2_ef1pct", "EF top-1%", lim=(0, ef_max))
    fig.suptitle("GNINA vs UniDock2 — per-target screening performance")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_gnina_vs_unidock2.png", dpi=150)
    plt.close(fig)

    print(f"Wrote 4 figures to {out_dir}/")


if __name__ == "__main__":
    main()
