"""
Re-evaluate saved models with virtual-screening metrics (EF, BEDROC).

Loads each trained model from disk, runs predict_proba on the test split,
and computes Enrichment Factor and BEDROC.  Results are saved alongside
the existing training summaries as `{model}_vs_metrics.json`.

Usage
-----
conda run -n rdkit_env python3 evaluate_vs_metrics.py \\
    --models-dir ./trained_models \\
    --registry ../../training_data_full/registry.csv \\
    --cache-dir ../../training_data_full/feature_cache
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader
from features.featurizer import get_featurizer
from features.combined_featurizer import get_combined_featurizer
from features.feature_cache import FeatureCache
from models.base_trainer import compute_vs_metrics


EF_FRACTIONS  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
BEDROC_ALPHAS = [20.0, 80.0, 160.0]


def load_feature_config(models_dir: Path, model_name: str) -> dict:
    path = models_dir / f"{model_name}_feature_config.json"
    with open(path) as f:
        return json.load(f)


def load_training_summary(models_dir: Path, model_name: str) -> dict:
    path = models_dir / f"{model_name}_training_summary.json"
    with open(path) as f:
        return json.load(f)


def build_featurizer(feat_config: dict, protein_mapping_path: Path = None):
    """Reconstruct the featurizer used during training."""
    ftype = feat_config.get("type", "morgan_fingerprint")

    if ftype == "combined":
        featurizer = get_combined_featurizer(
            ligand_config=feat_config["ligand_config"],
            protein_config=feat_config["protein_config"],
            concatenation_method=feat_config.get("concatenation_method", "concat"),
        )
        # Reload protein mapping so embeddings are consistent
        if protein_mapping_path and protein_mapping_path.exists():
            featurizer.load_protein_mapping(str(protein_mapping_path))
        return featurizer, True  # (featurizer, uses_protein)
    else:
        return get_featurizer(feat_config), False


def evaluate_model(model_name: str, models_dir: Path,
                   registry_path: str, cache_dir: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    summary = load_training_summary(models_dir, model_name)
    feat_config = load_feature_config(models_dir, model_name)
    sim_thresh = summary["data_config"]["similarity_threshold"]

    # ── load test data ────────────────────────────────────────────────
    use_2d = summary.get("use_2d_split", False)
    if use_2d:
        reg_path = str(Path(registry_path).parent / "registry_2d_split.csv")
        print(f"[2D split] Using {reg_path}")
    else:
        reg_path = registry_path

    print(f"Loading test split (threshold={sim_thresh})...")
    loader = DataLoader(reg_path)
    loader.load_registry()
    test_data = loader.get_training_data(
        similarity_threshold=sim_thresh, include_decoys=True, split="test",
        protein_partition="test" if use_2d else None,
    )
    include_protein = summary["data_config"].get("include_protein_features", False)

    if include_protein:
        test_smiles, y_test, test_protein_ids = loader.prepare_features_labels(
            test_data, include_protein_info=True
        )
    else:
        test_smiles, y_test = loader.prepare_features_labels(test_data)
        test_protein_ids = None

    print(f"Test set: {len(y_test):,} samples  ({y_test.sum():,} actives)")

    # ── featurize ─────────────────────────────────────────────────────
    protein_mapping_path = models_dir / f"{model_name}_protein_mapping.json"
    featurizer, uses_protein = build_featurizer(feat_config, protein_mapping_path)

    ligand_cache = None
    if cache_dir:
        ligand_cfg = (
            feat_config["ligand_config"] if uses_protein else feat_config
        )
        ligand_cache = FeatureCache(cache_dir, ligand_cfg)
        print(f"Feature cache: {ligand_cache.cache_path.name} "
              f"({ligand_cache.count():,} entries)")

    print("Featurizing test set...")
    if uses_protein:
        X_test, _ = featurizer.featurize(
            test_smiles, protein_ids=test_protein_ids,
            show_progress=True, ligand_cache=ligand_cache,
        )
    else:
        X_test, _ = featurizer.featurize(
            test_smiles, show_progress=True, cache=ligand_cache
        )
    print(f"Feature matrix: {X_test.shape}")

    # ── load model & predict ──────────────────────────────────────────
    model_path = models_dir / f"{model_name}.pkl"
    print(f"Loading model from {model_path} ...")
    model = joblib.load(model_path)

    # SVM needs scaling — reload scaler from pkl (it's bundled via CalibratedClassifierCV)
    # For SVM the saved model IS the CalibratedClassifierCV + scaler is separate
    # Check if a scaler needs to be applied
    scaler_path = models_dir / f"{model_name}_scaler.pkl"
    if scaler_path.exists():
        import joblib as jl
        scaler = jl.load(scaler_path)
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    print("Running predict_proba on test set...")
    y_score = model.predict_proba(X_eval)[:, 1]

    # ── VS metrics ────────────────────────────────────────────────────
    print("Computing EF and BEDROC...")
    vs = compute_vs_metrics(
        y_test, y_score,
        ef_fractions=EF_FRACTIONS,
        bedroc_alphas=BEDROC_ALPHAS,
    )

    print("\nVirtual-Screening Metrics (test set):")
    for k, v in vs.items():
        print(f"  {k:20s}: {v:.4f}")

    # Save
    out_path = models_dir / f"{model_name}_vs_metrics.json"
    result = {
        "model": model_name,
        "similarity_threshold": sim_thresh,
        "n_test": int(len(y_test)),
        "n_actives": int(y_test.sum()),
        "prevalence": float(y_test.mean()),
        "ef_fractions": EF_FRACTIONS,
        "bedroc_alphas": BEDROC_ALPHAS,
        "metrics": vs,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved models with EF and BEDROC metrics."
    )
    parser.add_argument("--models-dir", default="./trained_models")
    parser.add_argument("--registry",   default="../../training_data_full/registry.csv")
    parser.add_argument("--cache-dir",  default="../../training_data_full/feature_cache")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["random_forest", "gradient_boosting", "svm"],
        help="Which models to evaluate",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    all_results = {}

    for model_name in args.models:
        pkl = models_dir / f"{model_name}.pkl"
        if not pkl.exists():
            print(f"Skipping {model_name} — {pkl} not found")
            continue
        result = evaluate_model(
            model_name, models_dir, args.registry, args.cache_dir
        )
        all_results[model_name] = result

    # Combined summary
    print(f"\n{'='*60}")
    print("SUMMARY — Virtual-Screening Metrics (test set)")
    print(f"{'='*60}")
    header = f"{'Model':<22} {'EF@0.5%':>8} {'EF@1%':>7} {'EF@5%':>7} {'BEDROC20':>10} {'BEDROC80':>10}"
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        m = r["metrics"]
        print(f"{name:<22} {m.get('ef_0.5pct', m.get('ef_0p5pct', 0)):>8.3f} "
              f"{m.get('ef_1pct', 0):>7.3f} {m.get('ef_5pct', 0):>7.3f} "
              f"{m.get('bedroc_a20', 0):>10.4f} {m.get('bedroc_a80', 0):>10.4f}")


if __name__ == "__main__":
    main()
