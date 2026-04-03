"""
Collect GNINA redocking results on PDBbind CleanSplit and compute
regression metrics against ground-truth pK values.

Extracts CNNaffinity from docked SDF files (GNINA's pK predictor),
computes regression metrics, and writes a training_summary.json
compatible with generate_pdbbind_report.py.

Usage:
    conda run -n rdkit_env python benchmarks/05_pdbbind_comparison/collect_gnina_pdbbind.py \
        --config benchmarks/05_pdbbind_comparison/configs/gnina_pdbbind_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

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


def parse_docked_sdf(sdf_path: str) -> dict | None:
    """Parse GNINA docked SDF and extract the best-pose CNNaffinity score.

    GNINA outputs several scores per pose:
      - CNNscore: probability of being a good pose (classification)
      - CNNaffinity: predicted pK (regression, what we want)
      - minimizedAffinity: Vina-like score in kcal/mol

    Returns dict with CNNaffinity for the best pose, or None.
    """
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)

    best_cnn_affinity = None
    best_cnn_score = -1.0

    for mol in supplier:
        if mol is None:
            continue

        try:
            cnn_affinity = float(mol.GetProp("CNNaffinity"))
        except (KeyError, ValueError):
            cnn_affinity = None

        try:
            cnn_score = float(mol.GetProp("CNNscore"))
        except (KeyError, ValueError):
            cnn_score = 0.0

        try:
            min_affinity = float(mol.GetProp("minimizedAffinity"))
        except (KeyError, ValueError):
            min_affinity = None

        # Take the pose with the best CNNscore (most confident pose)
        if cnn_score > best_cnn_score:
            best_cnn_score = cnn_score
            best_cnn_affinity = cnn_affinity
            # Note: minimizedAffinity is in kcal/mol (different units from pK),
            # so we do NOT use it as a fallback for CNNaffinity.

    if best_cnn_affinity is not None:
        return {"cnn_affinity": best_cnn_affinity}
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect GNINA PDBbind redocking results"
    )
    parser.add_argument("--config", required=True, help="Path to gnina_pdbbind_config.yaml")
    parser.add_argument("--test-set", type=str, default=None, help="Override test set")
    args = parser.parse_args()

    config = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    labels_path = (config_dir / config["data"]["labels_path"]).resolve()
    split_path = (config_dir / config["data"]["split_path"]).resolve()
    docking_dir = (config_dir / config["paths"]["docking_dir"]).resolve()
    output_dir = (config_dir / config["paths"]["output_dir"]).resolve()
    test_set = args.test_set or config["data"]["test_set"]

    with open(labels_path) as f:
        labels_dict = json.load(f)
    with open(split_path) as f:
        split_dict = json.load(f)

    test_pdb_ids = [pid.lower() for pid in split_dict.get(test_set, [])]
    print(f"Collecting results for {test_set} ({len(test_pdb_ids)} complexes)")

    # Load docking results status
    results_json = output_dir / "docking_results.json"
    docking_status = {}
    if results_json.exists():
        with open(results_json) as f:
            docking_status = json.load(f)

    y_true_list = []
    y_pred_list = []
    per_complex = {}
    n_parsed = 0
    n_missing = 0
    n_failed = 0

    for pdb_id in test_pdb_ids:
        # Get ground truth pK
        label = labels_dict.get(pdb_id) or labels_dict.get(pdb_id.upper())
        if label is None:
            n_missing += 1
            continue
        pk_true = label.get("log_kd_ki")
        if pk_true is None:
            n_missing += 1
            continue

        # Check docking status
        status = docking_status.get(pdb_id, {})
        if status.get("status") != "ok":
            n_failed += 1
            continue

        # Parse docked SDF
        out_sdf = docking_dir / f"{pdb_id}_docked.sdf"
        if not out_sdf.exists():
            n_failed += 1
            continue

        result = parse_docked_sdf(str(out_sdf))
        if result is None:
            n_failed += 1
            continue

        pk_pred = result["cnn_affinity"]
        y_true_list.append(pk_true)
        y_pred_list.append(pk_pred)
        per_complex[pdb_id] = {
            "y_true": round(pk_true, 4),
            "y_pred": round(pk_pred, 4),
        }
        n_parsed += 1

    if n_parsed == 0:
        print("ERROR: No valid results found. Run run_gnina_pdbbind.py first.")
        sys.exit(1)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    metrics = summarize_regression(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  GNINA Results on {test_set} ({n_parsed} complexes)")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"\n  Parsed: {n_parsed}, Missing labels: {n_missing}, Failed: {n_failed}")

    # Build training summary
    total_elapsed = sum(
        docking_status.get(pid, {}).get("elapsed_s", 0) for pid in per_complex
    )

    summary = {
        "task_type": "regression",
        "model_type": "gnina",
        "model_architecture": "gnina_cnn_rescore",
        "feature_type": "3d_structure_cnn",
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
            "n_test_samples": n_parsed,
            "training_time": round(total_elapsed, 1),
        },
        "docking_params": {
            "tool": "gnina",
            "num_modes": config["gnina"]["num_modes"],
            "exhaustiveness": config["gnina"]["exhaustiveness"],
            "cnn_scoring": config["gnina"]["cnn_scoring"],
            "autobox_add": config["gnina"]["autobox_add"],
            "score_used": "CNNaffinity",
        },
        "per_complex_predictions": per_complex,
    }

    out_path = output_dir / f"gnina_pdbbind_{test_set}_training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
