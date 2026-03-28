"""
Per-target EF/BEDROC for all ML models (RF, GBM, SVM) on the 15 docked targets.

Loads each saved model, restricts the 1D test split to each target's uniprot_id,
runs predict_proba, and computes the same VS metrics as analyze_docking.py.
Only targets that also have GNINA results are included.

Usage:
  conda run -n rdkit_env python3 benchmarks/04_docking/compare_methods_per_target.py \\
      --models-dir benchmarks/02_training/trained_models \\
      --cache-dir  training_data_full/feature_cache \\
      --registry   training_data_full/registry.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "benchmarks/02_training"))

from data.data_loader import DataLoader
from features.featurizer import get_featurizer
from features.combined_featurizer import get_combined_featurizer
from features.feature_cache import FeatureCache
from models.base_trainer import compute_vs_metrics

EF_FRACTIONS  = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
BEDROC_ALPHAS = [20.0, 80.0, 160.0]

DOCKING_METRICS_DIR = ROOT / "benchmarks/04_docking/results/docking_metrics"
OUT_DIR             = ROOT / "benchmarks/04_docking/results/ml_per_target_metrics"


def load_feature_config(models_dir: Path, model_name: str) -> dict:
    with open(models_dir / f"{model_name}_feature_config.json") as f:
        return json.load(f)


def load_training_summary(models_dir: Path, model_name: str) -> dict:
    with open(models_dir / f"{model_name}_training_summary.json") as f:
        return json.load(f)


def build_featurizer(feat_config: dict, protein_mapping_path: Path = None):
    ftype = feat_config.get("type", "morgan_fingerprint")
    if ftype == "combined":
        featurizer = get_combined_featurizer(
            ligand_config=feat_config["ligand_config"],
            protein_config=feat_config["protein_config"],
            concatenation_method=feat_config.get("concatenation_method", "concat"),
        )
        if protein_mapping_path and protein_mapping_path.exists():
            with open(protein_mapping_path) as f:
                mapping = json.load(f)
            featurizer.protein_featurizer.protein_to_idx = mapping
        return featurizer, True
    else:
        return get_featurizer(feat_config), False


def predict_target(model, featurizer, uses_protein,
                   smiles, protein_ids, ligand_cache, scaler=None):
    if uses_protein:
        X, _ = featurizer.featurize(smiles, protein_ids=protein_ids,
                                    show_progress=False,
                                    ligand_cache=ligand_cache)
    else:
        X, _ = featurizer.featurize(smiles, show_progress=False,
                                    cache=ligand_cache)
    if scaler is not None:
        X = scaler.transform(X)
    proba = model.predict_proba(X)[:, 1]
    return proba


def analyze_model(model_name: str, models_dir: Path,
                  registry_path: str, cache_dir: str,
                  target_uniprots: list) -> dict:
    summary     = load_training_summary(models_dir, model_name)
    feat_config = load_feature_config(models_dir, model_name)
    sim_thresh  = summary["data_config"]["similarity_threshold"]

    # Load model
    model = joblib.load(models_dir / f"{model_name}.pkl")

    # Load scaler if SVM
    scaler_path = models_dir / f"{model_name}_scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    # Build featurizer
    protein_mapping_path = models_dir / f"{model_name}_protein_mapping.json"
    featurizer, uses_protein = build_featurizer(feat_config, protein_mapping_path)

    # Feature cache
    ligand_cache = None
    if cache_dir:
        ligand_cfg = feat_config["ligand_config"] if uses_protein else feat_config
        ligand_cache = FeatureCache(cache_dir, ligand_cfg)

    # Load full 1D test split once
    loader = DataLoader(registry_path)
    loader.load_registry()
    test_data = loader.get_training_data(
        similarity_threshold=sim_thresh, include_decoys=True, split="test"
    )
    include_protein = summary["data_config"].get("include_protein_features", False)
    if include_protein:
        all_smiles, all_labels, all_pids = loader.prepare_features_labels(
            test_data, include_protein_info=True
        )
    else:
        all_smiles, all_labels = loader.prepare_features_labels(test_data)
        all_pids = None

    # uniprot_id column for grouping
    uniprot_col = test_data["uniprot_id"].values

    results = {}
    for uniprot in target_uniprots:
        mask = uniprot_col == uniprot
        n = mask.sum()
        if n == 0:
            results[uniprot] = {"status": "not_in_test_split", "n": 0}
            continue

        smiles_t  = [all_smiles[i] for i in np.where(mask)[0]]
        labels_t  = all_labels[mask]
        pids_t    = ([all_pids[i] for i in np.where(mask)[0]]
                     if all_pids is not None else None)

        n_active = int(labels_t.sum())
        if n_active == 0 or n_active == n:
            results[uniprot] = {"status": "no_variance", "n": int(n),
                                "n_active": n_active}
            continue

        try:
            proba = predict_target(model, featurizer, uses_protein,
                                   smiles_t, pids_t, ligand_cache, scaler)
            metrics = compute_vs_metrics(labels_t, proba,
                                         ef_fractions=EF_FRACTIONS,
                                         bedroc_alphas=BEDROC_ALPHAS)
            from sklearn.metrics import roc_auc_score
            metrics["roc_auc"] = round(float(roc_auc_score(labels_t, proba)), 4)
            results[uniprot] = {"status": "ok", "n": int(n),
                                "n_active": n_active, "metrics": metrics}
        except Exception as e:
            results[uniprot] = {"status": f"error: {e}", "n": int(n)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="benchmarks/02_training/trained_models")
    parser.add_argument("--registry",   default="training_data_full/registry.csv")
    parser.add_argument("--cache-dir",  default="training_data_full/feature_cache")
    parser.add_argument("--models", nargs="+",
                        default=["random_forest", "gradient_boosting", "svm"])
    args = parser.parse_args()

    models_dir   = ROOT / args.models_dir
    registry     = str(ROOT / args.registry)
    cache_dir    = str(ROOT / args.cache_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Targets = those with GNINA results
    target_uniprots = sorted([
        f.stem.replace("_docking_metrics", "")
        for f in DOCKING_METRICS_DIR.glob("*_docking_metrics.json")
        if "all_targets" not in f.stem
    ])
    print(f"Targets with GNINA results: {target_uniprots}\n")

    all_model_results = {}

    for model_name in args.models:
        pkl = models_dir / f"{model_name}.pkl"
        if not pkl.exists():
            print(f"Skipping {model_name} — model not found at {pkl}")
            continue
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")
        results = analyze_model(model_name, models_dir, registry, cache_dir,
                                target_uniprots)
        all_model_results[model_name] = results

        # Print per-target summary
        for uniprot, res in results.items():
            if res["status"] == "ok":
                m = res["metrics"]
                print(f"  {uniprot}: n={res['n']} active={res['n_active']}  "
                      f"AUC={m.get('roc_auc','?')}  "
                      f"EF@1%={m.get('ef_1pct','?')}  "
                      f"BEDROC-80={m.get('bedroc_a80','?')}")
            else:
                print(f"  {uniprot}: {res['status']}")

    # Save combined JSON
    out_json = OUT_DIR / "ml_per_target_metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_model_results, f, indent=2)
    print(f"\nSaved → {out_json}")

    # Build comparison DataFrame
    rows = []
    for model_name, model_results in all_model_results.items():
        for uniprot, res in model_results.items():
            if res["status"] == "ok":
                row = {"method": model_name, "uniprot": uniprot,
                       "n": res["n"], "n_active": res["n_active"]}
                row.update(res["metrics"])
                rows.append(row)

    if rows:
        csv_path = OUT_DIR / "ml_per_target_summary.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
