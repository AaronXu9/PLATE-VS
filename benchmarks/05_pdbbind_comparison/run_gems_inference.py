"""
Run GEMS model inference on PDBbind CleanSplit test sets.

Loads preprocessed .pt datasets, runs the 5-fold GEMS ensemble,
and computes regression metrics against ground-truth pK values.

Prerequisites:
  - GEMS repo cloned as submodule: external/GEMS/
  - Preprocessed datasets downloaded from Zenodo
  - conda env 'gems' (see benchmarks/envs/env_gems.yml)

Usage:
    conda run -n gems python benchmarks/05_pdbbind_comparison/run_gems_inference.py \
        --config benchmarks/05_pdbbind_comparison/configs/gems_config.yaml

    # Evaluate on CASF-2013 instead
    conda run -n gems python benchmarks/05_pdbbind_comparison/run_gems_inference.py \
        --config benchmarks/05_pdbbind_comparison/configs/gems_config.yaml \
        --test-set casf2013
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Metrics import (works from project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Import metrics via importlib to avoid package name conflicts
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "metrics", PROJECT_ROOT / "benchmarks" / "utils" / "metrics.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
summarize_regression = _mod.summarize_regression


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# GEMS model / dataset helpers
# ---------------------------------------------------------------------------

# Embedding code mapping used by GEMS to identify dataset variants
# Format: {(has_ankh, has_esm2, has_delete_protein): (dataset_id, model_arch)}
EMBEDDING_MAP = {
    (False, False, False): ("00AEPL", "GEMS18e"),   # No embeddings
    (True, False, False):  ("B0AEPL", "GEMS18d"),   # ankh_base only
    (False, True, False):  ("06AEPL", "GEMS18d"),   # esm2_t6 only
    (True, True, False):   ("B6AEPL", "GEMS18d"),   # ankh_base + esm2_t6 (best)
    (True, True, True):    ("B6AE0L", "GEMS18d"),   # ablation: delete protein
}


def detect_model_variant(dataset):
    """Auto-detect GEMS model variant from dataset attributes."""
    prot_embs = getattr(dataset, "protein_embeddings", []) or []
    lig_embs = getattr(dataset, "ligand_embeddings", []) or []
    delete_prot = getattr(dataset, "delete_protein", False) or False

    has_ankh = "ankh_base" in prot_embs
    has_esm2 = "esm2_t6" in prot_embs

    key = (has_ankh, has_esm2, bool(delete_prot))
    if key in EMBEDDING_MAP:
        dataset_id, arch_name = EMBEDDING_MAP[key]
    else:
        # Default: try embeddings variant
        dataset_id, arch_name = "B6AEPL", "GEMS18d"
        print(f"  [WARN] Unknown embedding combo {key}, defaulting to {arch_name}/{dataset_id}")

    # Check if dataset actually has ligand embeddings
    has_lig_emb = hasattr(dataset[0], "lig_emb") and dataset[0].lig_emb is not None
    if not has_lig_emb and arch_name == "GEMS18d":
        # GEMS18d with no lig_emb still uses lig_emb as initial global u
        # but the dataset_id 00AEPL variant of GEMS18d handles this
        pass

    print(f"  Detected: arch={arch_name}, dataset_id={dataset_id}")
    print(f"  Protein embeddings: {prot_embs}")
    print(f"  Ligand embeddings: {lig_embs}")
    print(f"  Delete protein: {delete_prot}")

    return arch_name, dataset_id


def find_state_dicts(model_dir: Path, arch_name: str, dataset_id: str, n_folds: int = 5) -> list[Path]:
    """Find pretrained state dict files for the given architecture and dataset variant."""
    pattern = f"{arch_name}_{dataset_id}_*_f{{fold}}_best_stdict.pt"

    stdict_paths = []
    for fold in range(n_folds):
        fold_pattern = str(model_dir / f"{arch_name}_{dataset_id}_*_f{fold}_best_stdict.pt")
        matches = glob.glob(fold_pattern)
        if not matches:
            raise FileNotFoundError(
                f"No state dict found for fold {fold}: {fold_pattern}"
            )
        stdict_paths.append(Path(matches[0]))

    return stdict_paths


def find_dataset_file(preprocessed_dir: Path, dataset_id: str, test_set: str) -> Path:
    """Find the preprocessed .pt dataset file for the given test set."""
    # Naming convention: {dataset_id}_{split}.pt
    # e.g., B6AEPL_casf2016.pt, B6AEPL_train_cleansplit.pt
    candidate = preprocessed_dir / f"{dataset_id}_{test_set}.pt"
    if candidate.exists():
        return candidate

    # Try searching recursively
    matches = list(preprocessed_dir.rglob(f"*{dataset_id}*{test_set}*.pt"))
    if matches:
        return matches[0]

    # List available files for debugging
    available = list(preprocessed_dir.rglob("*.pt"))
    available_names = [p.name for p in available]
    raise FileNotFoundError(
        f"Dataset file not found for {dataset_id}/{test_set}.\n"
        f"Searched: {candidate}\n"
        f"Available .pt files: {available_names}"
    )


def run_inference(config: dict, test_set_override: str | None = None) -> dict:
    """Run GEMS inference and return results."""
    import torch
    from torch_geometric.loader import DataLoader

    # ---- Config ----
    gems_repo = Path(config["model"]["repo_path"]).resolve()
    model_dir = gems_repo / "model"
    n_folds = config["model"]["n_folds"]
    preprocessed_dir = Path(config["data"]["preprocessed_dir"]).resolve()
    test_set = test_set_override or config["data"]["test_set"]
    batch_size = config["inference"]["batch_size"]
    num_workers = config["inference"]["num_workers"]
    device_str = config["inference"]["device"]

    # Add GEMS repo to path so we can import their model code
    sys.path.insert(0, str(gems_repo))
    from model.GEMS18 import GEMS18d, GEMS18e

    arch_classes = {"GEMS18d": GEMS18d, "GEMS18e": GEMS18e}

    # ---- Device ----
    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # ---- Load a reference dataset to detect variant ----
    # First, try to find any .pt file to detect the variant
    ref_files = list(preprocessed_dir.rglob("*.pt"))
    if not ref_files:
        raise FileNotFoundError(f"No .pt files found in {preprocessed_dir}")

    print(f"\nLoading reference dataset to detect model variant...")
    ref_dataset = torch.load(ref_files[0], map_location="cpu", weights_only=False)
    arch_name, dataset_id = detect_model_variant(ref_dataset)
    del ref_dataset

    # ---- Load test dataset ----
    dataset_path = find_dataset_file(preprocessed_dir, dataset_id, test_set)
    print(f"\nLoading test dataset: {dataset_path.name}")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    print(f"  {len(dataset)} complexes")

    node_feat_dim = dataset[0].x.shape[1]
    edge_feat_dim = dataset[0].edge_attr.shape[1]
    print(f"  Node features: {node_feat_dim}, Edge features: {edge_feat_dim}")

    # ---- Load ensemble models ----
    stdict_paths = find_state_dicts(model_dir, arch_name, dataset_id, n_folds)
    print(f"\nLoading {n_folds}-fold ensemble from {model_dir}")

    ModelClass = arch_classes[arch_name]
    models = []
    for i, path in enumerate(stdict_paths):
        model = ModelClass(
            dropout_prob=0,
            in_channels=node_feat_dim,
            edge_dim=edge_feat_dim,
            conv_dropout_prob=0,
        ).float().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models.append(model)
        print(f"  Loaded fold {i}: {path.name}")

    # ---- Run inference ----
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_true_scaled = []
    y_pred_scaled = []
    complex_ids = []

    print(f"\nRunning inference on {len(dataset)} complexes...")
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = [model(batch).view(-1) for model in models]
            pred = torch.mean(torch.stack(outputs), dim=0)
            y_true_scaled.extend(batch.y.cpu().tolist())
            y_pred_scaled.extend(pred.cpu().tolist())
            complex_ids.extend(batch.id)

    inference_time = time.time() - t0
    print(f"  Inference time: {inference_time:.1f}s")

    # ---- Unscale from [0,1] to pK (pK range = [0, 16]) ----
    # GEMS normalizes pK to [0,1] via: scaled = (pK - 0) / (16 - 0)
    # See GEMS Dataset.py: min_val=0, max_val=16
    PK_MIN, PK_MAX = 0, 16
    y_true = np.array(y_true_scaled) * (PK_MAX - PK_MIN) + PK_MIN
    y_pred = np.array(y_pred_scaled) * (PK_MAX - PK_MIN) + PK_MIN

    # ---- Compute metrics ----
    metrics = summarize_regression(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  GEMS Results on {test_set} ({len(dataset)} complexes)")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # ---- Per-complex predictions ----
    per_complex = {}
    for cid, yt, yp in zip(complex_ids, y_true.tolist(), y_pred.tolist()):
        # Extract PDB ID from complex ID (e.g., "1a1e_L00001_graph" -> "1a1e")
        pdb_id = cid.split("_")[0] if "_" in cid else cid
        per_complex[pdb_id] = {"y_true": round(yt, 4), "y_pred": round(yp, 4)}

    # ---- Build training summary ----
    summary = {
        "task_type": "regression",
        "model_type": "gems_gnn",
        "model_architecture": f"{arch_name}_{dataset_id}_{n_folds}fold_ensemble",
        "feature_type": "3d_structure_gnn_embeddings",
        "dataset": "pdbbind_cleansplit",
        "test_set": test_set,
        "similarity_threshold": "cleansplit",
        "use_precomputed_split": True,
        "training_history": {
            "train_metrics": None,
            "val_metrics": None,
            "test_metrics": {k: round(v, 6) for k, v in metrics.items()},
            "n_train_samples": None,
            "n_val_samples": None,
            "n_test_samples": len(dataset),
            "training_time": round(inference_time, 1),
        },
        "per_complex_predictions": per_complex,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run GEMS inference on PDBbind CleanSplit test set"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmarks/05_pdbbind_comparison/configs/gems_config.yaml",
        help="Path to gems_config.yaml",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        help="Override test set (casf2016, casf2013, casf2016_indep, casf2013_indep)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve paths relative to config file directory
    config_dir = Path(args.config).resolve().parent
    for key in ["repo_path"]:
        config["model"][key] = str((config_dir / config["model"][key]).resolve())
    for key in ["preprocessed_dir", "labels_path", "split_path"]:
        if key in config["data"]:
            config["data"][key] = str((config_dir / config["data"][key]).resolve())

    results_dir = Path(args.output_dir or config["output"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = config_dir / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = run_inference(config, test_set_override=args.test_set)

    test_set = summary["test_set"]
    out_path = results_dir / f"gems_{test_set}_training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
