"""
Parse GNINA docked SDF files, compute classification metrics, and write a
training_summary.json compatible with generate_benchmark_report.py.

Metrics per target and aggregated:
  - ROC-AUC, Average Precision (AUPRC)
  - F1, Accuracy, Precision, Recall, MCC (at Youden's-J optimal threshold)

Usage (from project root):
    conda run -n rdkit_env python3 benchmarks/04_docking/collect_results.py \
        --config benchmarks/04_docking/configs/gnina_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_docked_sdf(sdf_path: str) -> list[dict]:
    """
    Parse a GNINA output SDF. Returns one record per unique molecule
    (best pose = first occurrence) with CNNscore and is_active label.
    """
    records = {}
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        # Keep only the first (best-scored) pose per molecule
        if name in records:
            continue
        try:
            cnn_score = float(mol.GetProp("CNNscore"))
        except (KeyError, ValueError):
            # Fall back to minimizedAffinity (negate: more negative = better)
            try:
                cnn_score = -float(mol.GetProp("minimizedAffinity"))
            except (KeyError, ValueError):
                cnn_score = 0.0

        try:
            is_active = int(mol.GetProp("is_active"))
        except (KeyError, ValueError):
            is_active = -1  # unknown

        records[name] = {"compound_id": name, "cnn_score": cnn_score, "is_active": is_active}

    return [r for r in records.values() if r["is_active"] != -1]


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute full classification metrics for one target."""
    if len(np.unique(y_true)) < 2:
        return {m: None for m in ["roc_auc", "avg_precision", "f1_score", "accuracy", "precision", "recall", "mcc"]}

    roc_auc = float(roc_auc_score(y_true, y_score))
    avg_prec = float(average_precision_score(y_true, y_score))

    # Youden's J threshold: maximizes TPR - FPR
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx])
    y_pred = (y_score >= threshold).astype(int)

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": threshold,
    }


def weighted_aggregate(per_target: dict, weight_key: str = "n_actives") -> dict:
    """Compute weighted average of metrics across targets."""
    metric_keys = ["roc_auc", "avg_precision", "f1_score", "accuracy", "precision", "recall", "mcc"]
    weights = []
    metric_vals = {k: [] for k in metric_keys}

    for uniprot, info in per_target.items():
        w = info.get(weight_key, 1)
        for k in metric_keys:
            v = info.get(k)
            if v is not None:
                metric_vals[k].append(v)
                weights.append(w)
                break  # we only need to check once for weighting

    # Per-metric weighted average
    aggregated = {}
    per_metric_weights = {k: [] for k in metric_keys}
    per_metric_values = {k: [] for k in metric_keys}
    for uniprot, info in per_target.items():
        w = info.get(weight_key, 1)
        for k in metric_keys:
            v = info.get(k)
            if v is not None:
                per_metric_weights[k].append(w)
                per_metric_values[k].append(v)

    for k in metric_keys:
        ws = per_metric_weights[k]
        vs = per_metric_values[k]
        if not vs:
            aggregated[k] = None
        else:
            total_w = sum(ws)
            aggregated[k] = float(sum(w * v for w, v in zip(ws, vs)) / total_w) if total_w > 0 else None

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Collect GNINA docking results and compute metrics")
    parser.add_argument("--config", required=True, help="Path to gnina_config.yaml")
    parser.add_argument("--workdir", default=".", help="Project root directory")
    args = parser.parse_args()

    config = load_config(args.config)
    workdir = Path(args.workdir).resolve()
    output_dir = workdir / config["paths"]["output_dir"]
    docking_dir = workdir / config["paths"]["docking_dir"]

    targets_path = output_dir / "selected_targets.json"
    docking_results_path = output_dir / "docking_results.json"

    if not targets_path.exists():
        print(f"ERROR: {targets_path} not found.")
        sys.exit(1)

    with open(targets_path) as f:
        targets = json.load(f)

    docking_results = {}
    if docking_results_path.exists():
        with open(docking_results_path) as f:
            docking_results = json.load(f)

    print(f"Collecting results for {len(targets)} targets ...\n")

    per_target = {}
    total_actives = 0
    total_decoys = 0
    all_scores = []
    all_labels = []

    for target in targets:
        uniprot = target["uniprot_id"]
        dr = docking_results.get(uniprot, {})

        if dr.get("status") != "ok":
            print(f"[{uniprot}] No docking results (status={dr.get('status', 'missing')}), skipping")
            continue

        out_sdf = docking_dir / f"{uniprot}_docked.sdf"
        if not out_sdf.exists():
            print(f"[{uniprot}] Output SDF not found: {out_sdf}, skipping")
            continue

        records = parse_docked_sdf(str(out_sdf))
        if not records:
            print(f"[{uniprot}] No parseable records in output SDF, skipping")
            continue

        y_true = np.array([r["is_active"] for r in records])
        y_score = np.array([r["cnn_score"] for r in records])

        n_actives = int(y_true.sum())
        n_decoys = int((y_true == 0).sum())

        if n_actives == 0:
            print(f"[{uniprot}] No actives in output, skipping")
            continue

        metrics = compute_metrics(y_true, y_score)
        per_target[uniprot] = {
            "n_actives": n_actives,
            "n_decoys": n_decoys,
            **metrics,
        }

        total_actives += n_actives
        total_decoys += n_decoys
        all_scores.extend(y_score.tolist())
        all_labels.extend(y_true.tolist())

        roc = metrics.get("roc_auc")
        ap = metrics.get("avg_precision")
        print(
            f"[{uniprot}]  n_actives={n_actives:4d}  n_decoys={n_decoys:5d}  "
            f"ROC-AUC={roc:.4f}  AP={ap:.4f}" if roc is not None else
            f"[{uniprot}]  insufficient data for metrics"
        )

    if not per_target:
        print("ERROR: No valid per-target results. Check docking outputs.")
        sys.exit(1)

    # Aggregate metrics
    agg_metrics = weighted_aggregate(per_target, weight_key="n_actives")

    # Also compute global metrics across all targets
    if all_labels:
        y_all = np.array(all_labels)
        s_all = np.array(all_scores)
        global_metrics = compute_metrics(y_all, s_all)
    else:
        global_metrics = {}

    print(f"\n{'='*60}")
    print(f"Aggregated (weighted by n_actives, {len(per_target)} targets):")
    for k in ["roc_auc", "avg_precision", "f1_score", "accuracy", "mcc"]:
        v = agg_metrics.get(k)
        print(f"  {k:20s}: {v:.4f}" if v is not None else f"  {k:20s}: N/A")
    print(f"\nTotal: {total_actives} actives, {total_decoys} decoys across {len(per_target)} targets")

    # Build training_summary.json compatible with generate_benchmark_report.py
    summary = {
        "model_type": "gnina",
        "feature_type": "3d_structure_cnn",
        "similarity_threshold": config["data"]["similarity_threshold"],
        "use_precomputed_split": True,
        "training_history": {
            "train_metrics": None,
            "val_metrics": None,
            "test_metrics": {k: round(v, 6) if v is not None else None for k, v in agg_metrics.items()},
            "n_train_samples": None,
            "n_val_samples": None,
            "n_test_samples": total_actives + total_decoys,
            "training_time": sum(
                docking_results.get(u, {}).get("elapsed_s", 0) for u in per_target
            ),
        },
        "data_config": {
            "similarity_threshold": config["data"]["similarity_threshold"],
            "n_targets": len(per_target),
            "total_actives": total_actives,
            "total_decoys": total_decoys,
        },
        "docking_params": {
            "tool": "gnina",
            "num_modes": config["gnina"]["num_modes"],
            "exhaustiveness": config["gnina"]["exhaustiveness"],
            "cnn_scoring": config["gnina"]["cnn_scoring"],
            "autobox_add": config["gnina"]["autobox_add"],
            "score_used": "CNNscore",
        },
        "per_target_metrics": per_target,
        "global_pooled_metrics": {k: round(v, 6) if v is not None else None for k, v in global_metrics.items()},
    }

    out_path = output_dir / "gnina_training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
