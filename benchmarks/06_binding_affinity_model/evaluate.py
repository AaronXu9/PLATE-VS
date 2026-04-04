"""
Standalone evaluation of trained binding affinity model on CASF-2016.

Usage:
    python benchmarks/06_binding_affinity_model/evaluate.py \
        --checkpoint checkpoints/best_model_f0.pt \
        --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from model.binding_affinity_model import BindingAffinityModel
from model.losses import CompositeLoss
from data.collate import custom_collate
from data.build_dataset import BindingAffinityDataset


def ranking_power(predictions: dict, targets: dict, clusters: dict) -> dict:
    """CASF-2016 ranking power: per-cluster Spearman correlation.

    Args:
        predictions: {pdb_id: predicted_pK}
        targets: {pdb_id: true_pK}
        clusters: {cluster_id: [pdb_ids]}
    """
    spearman_per_cluster = []

    for cluster_id, pdb_ids in clusters.items():
        preds, trues = [], []
        for pid in pdb_ids:
            if pid in predictions and pid in targets:
                preds.append(predictions[pid])
                trues.append(targets[pid])

        if len(preds) < 3:
            continue

        rho, _ = spearmanr(preds, trues)
        if np.isfinite(rho):
            spearman_per_cluster.append(rho)

    if not spearman_per_cluster:
        return {"mean_spearman": float("nan"), "median_spearman": float("nan")}

    return {
        "mean_spearman": float(np.mean(spearman_per_cluster)),
        "median_spearman": float(np.median(spearman_per_cluster)),
        "std_spearman": float(np.std(spearman_per_cluster)),
        "n_clusters": len(spearman_per_cluster),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate binding affinity model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--test-set", default="casf2016")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    config_dir = Path(args.config).resolve().parent
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve paths
    dataset_dir = (config_dir / config["data"]["dataset_dir"]).resolve()
    clusters_path = (config_dir / config["data"]["clusters_path"]).resolve()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    # Load test dataset
    test_path = dataset_dir / "processed" / f"{args.test_set}_data.pt"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found")
        sys.exit(1)

    test_data, test_slices = torch.load(str(test_path), weights_only=False)
    test_dataset = BindingAffinityDataset.__new__(BindingAffinityDataset)
    test_dataset.data, test_dataset.slices = test_data, test_slices
    test_dataset._data_list = None

    test_loader = DataLoader(
        test_dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=0, collate_fn=custom_collate,
    )

    # Build + load model
    mc = config["model"]
    model = BindingAffinityModel(
        esm_dim=mc["esm_dim"], proj_dim=mc["proj_dim"],
        et_layers=mc["et_layers"], et_heads=mc["et_heads"],
        et_rbf=mc["et_rbf"], et_cutoff=mc["et_cutoff"],
        cross_attn_layers=mc["cross_attn_layers"],
        cross_attn_heads=mc["cross_attn_heads"],
        dropout=mc["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # Evaluate
    lc = config["loss"]
    criterion = CompositeLoss(
        lambda_rank=lc["lambda_rank"], huber_delta=lc["huber_delta"],
        rank_margin=lc["rank_margin"], rank_sample_size=lc["rank_sample_size"],
    )

    all_pred, all_target, all_pdb_ids = [], [], []
    pk_max = BindingAffinityDataset.PK_MAX

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            all_pred.append(pred.cpu())
            all_target.append(batch.y.cpu())
            if hasattr(batch, "pdb_id"):
                all_pdb_ids.extend(batch.pdb_id)

    pred_pk = torch.cat(all_pred).numpy().flatten() * pk_max
    true_pk = torch.cat(all_target).numpy().flatten() * pk_max

    # Scoring power
    scoring = {
        "pearson": float(pearsonr(pred_pk, true_pk)[0]),
        "spearman": float(spearmanr(pred_pk, true_pk).correlation),
        "rmse": float(np.sqrt(np.mean((pred_pk - true_pk) ** 2))),
        "r2": float(1 - np.sum((pred_pk - true_pk) ** 2) / np.sum((true_pk - true_pk.mean()) ** 2)),
        "mae": float(np.mean(np.abs(pred_pk - true_pk))),
    }

    print(f"\n{'='*50}")
    print(f"  {args.test_set} Scoring Power ({len(pred_pk)} complexes)")
    print(f"{'='*50}")
    for k, v in scoring.items():
        print(f"  {k:12s}: {v:.4f}")

    # Ranking power (if clusters available)
    ranking = {}
    if clusters_path.exists():
        with open(clusters_path) as f:
            clusters = json.load(f)

        predictions = {pid: float(p) for pid, p in zip(all_pdb_ids, pred_pk)}
        targets = {pid: float(t) for pid, t in zip(all_pdb_ids, true_pk)}

        ranking = ranking_power(predictions, targets, clusters)
        print(f"\n  Ranking Power ({ranking.get('n_clusters', 0)} clusters)")
        for k, v in ranking.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}")

    # Save results
    output_path = args.output or f"results/{args.test_set}_evaluation.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    per_complex = {}
    for pid, yt, yp in zip(all_pdb_ids, true_pk, pred_pk):
        per_complex[pid] = {"y_true": round(float(yt), 4), "y_pred": round(float(yp), 4)}

    result = {
        "test_set": args.test_set,
        "n_complexes": len(pred_pk),
        "scoring_power": scoring,
        "ranking_power": ranking,
        "per_complex_predictions": per_complex,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)
    main()
