"""
Train classical ML regressors (RF, GBM, SVM) on PDBbind CleanSplit.

Featurizes ligand SMILES with Morgan fingerprints, trains sklearn
regressors, and evaluates on CASF-2016/2013 test sets.

Prerequisites:
  - SMILES CSV from extract_pdbbind_smiles.py
  - Split/label JSONs from download_pdbbind_cleansplit.py
  - conda env 'rdkit_env'

Usage:
    conda run -n rdkit_env python benchmarks/05_pdbbind_comparison/train_classical_pdbbind.py \
        --config benchmarks/05_pdbbind_comparison/configs/classical_pdbbind_config.yaml

    # Train only one model
    conda run -n rdkit_env python benchmarks/05_pdbbind_comparison/train_classical_pdbbind.py \
        --config benchmarks/05_pdbbind_comparison/configs/classical_pdbbind_config.yaml \
        --model random_forest

    # Different test set
    conda run -n rdkit_env python benchmarks/05_pdbbind_comparison/train_classical_pdbbind.py \
        --config benchmarks/05_pdbbind_comparison/configs/classical_pdbbind_config.yaml \
        --test-set casf2013
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats


def _calculate_regression_metrics(y_true, y_pred):
    """Compute all regression metrics inline (avoids importlib issues)."""
    import numpy as _np
    t, p = _np.asarray(y_true, float), _np.asarray(y_pred, float)
    mask = _np.isfinite(t) & _np.isfinite(p)
    t, p = t[mask], p[mask]
    if len(t) < 2:
        return {k: float("nan") for k in ["mse","rmse","mae","r2","pearson","spearman","kendall","ci"]}
    mse = float(_np.mean((t - p) ** 2))
    ss_res = _np.sum((t - p) ** 2)
    ss_tot = _np.sum((t - _np.mean(t)) ** 2)
    return {
        "mse": mse,
        "rmse": float(mse ** 0.5),
        "mae": float(_np.mean(_np.abs(t - p))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson": float(_np.corrcoef(t, p)[0, 1]) if _np.std(t) > 0 and _np.std(p) > 0 else float("nan"),
        "spearman": float(stats.spearmanr(t, p).correlation),
        "kendall": float(stats.kendalltau(t, p).correlation),
        "ci": float("nan"),  # CI is expensive; skip for speed
    }


summarize_regression = _calculate_regression_metrics


def _featurize_morgan(smiles_list, radius=2, n_bits=2048, show_progress=True):
    """Generate Morgan fingerprints from SMILES (inline, no importlib)."""
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    invalid = []
    it = enumerate(smiles_list)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(smiles_list), desc="Generating fingerprints")
        except ImportError:
            pass
    for i, smi in it:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(i)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps[i] = np.array(fp, dtype=np.float32)
    if invalid:
        print(f"Warning: {len(invalid)} molecules could not be converted")
    return fps, invalid


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_smiles_data(smiles_path: str) -> dict:
    """Load SMILES CSV into a dict keyed by pdb_id."""
    data = {}
    with open(smiles_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["pdb_id"]] = {
                "smiles": row["smiles"],
                "pK": float(row["pK"]),
                "split": row["split"],
            }
    return data


def load_split_data(split_path: str) -> dict:
    """Load CleanSplit JSON."""
    with open(split_path) as f:
        return json.load(f)


def prepare_data(
    smiles_data: dict,
    split_dict: dict,
    test_set: str = "casf2016",
) -> tuple:
    """Prepare train/test data from SMILES CSV + split definitions.

    If smiles_data is available (from extract_pdbbind_smiles.py), use it.
    Otherwise fall back to labels_dict for pK values.

    Returns:
        (train_smiles, train_pK, test_smiles, test_pK, test_pdb_ids)
    """
    train_ids = set(pid.lower() for pid in split_dict.get("train", []))
    test_ids = set(pid.lower() for pid in split_dict.get(test_set, []))

    train_smiles, train_pK, train_pdb_ids = [], [], []
    test_smiles, test_pK, test_pdb_ids = [], [], []

    for pdb_id, entry in smiles_data.items():
        pid = pdb_id.lower()
        smiles = entry["smiles"]
        pk = entry["pK"]

        if pid in train_ids:
            train_smiles.append(smiles)
            train_pK.append(pk)
            train_pdb_ids.append(pid)
        elif pid in test_ids:
            test_smiles.append(smiles)
            test_pK.append(pk)
            test_pdb_ids.append(pid)

    return train_smiles, train_pK, train_pdb_ids, test_smiles, test_pK, test_pdb_ids


def build_model(model_config: dict):
    """Build an sklearn regressor from config."""
    model_type = model_config["type"]
    hp = model_config.get("hyperparameters", {})

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**hp)

    elif model_type == "gradient_boosting":
        try:
            from lightgbm import LGBMRegressor
            # Map common sklearn param names to LightGBM
            lgbm_params = {
                "n_estimators": hp.get("n_estimators", 500),
                "max_depth": hp.get("max_depth", 6),
                "learning_rate": hp.get("learning_rate", 0.1),
                "subsample": hp.get("subsample", 0.8),
                "random_state": hp.get("random_state", 42),
                "verbose": -1,
            }
            return LGBMRegressor(**lgbm_params)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**hp)

    elif model_type == "svm":
        from sklearn.svm import SVR
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        svr_params = {k: v for k, v in hp.items()}
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(**svr_params)),
        ])

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(
    model_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_pdb_ids: list[str],
    results_dir: Path,
    test_set: str,
    feature_config: dict,
) -> dict:
    """Train one model and evaluate on test set."""
    model_type = model_config["type"]
    print(f"\n{'='*60}")
    print(f"  Training: {model_type}")
    print(f"{'='*60}")
    print(f"  Train samples: {len(y_train)}")
    print(f"  Test samples:  {len(y_test)}")

    model = build_model(model_config)

    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0
    print(f"  Training time: {training_time:.1f}s")

    # Evaluate on train
    y_train_pred = model.predict(X_train)
    train_metrics = summarize_regression(y_train, y_train_pred)
    print(f"\n  Train metrics:")
    for k, v in train_metrics.items():
        print(f"    {k:12s}: {v:.4f}")

    # Evaluate on test
    y_test_pred = model.predict(X_test)
    test_metrics = summarize_regression(y_test, y_test_pred)
    print(f"\n  Test metrics ({test_set}):")
    for k, v in test_metrics.items():
        print(f"    {k:12s}: {v:.4f}")

    # Per-complex predictions
    per_complex = {}
    for pid, yt, yp in zip(test_pdb_ids, y_test.tolist(), y_test_pred.tolist()):
        per_complex[pid] = {"y_true": round(yt, 4), "y_pred": round(yp, 4)}

    # Build training summary
    summary = {
        "task_type": "regression",
        "model_type": model_type,
        "model_architecture": str(type(model).__name__),
        "feature_type": feature_config.get("_feature_type_str", f"morgan_r{feature_config.get('radius', 2)}_b{feature_config.get('n_bits', 2048)}"),
        "dataset": "pdbbind_cleansplit",
        "test_set": test_set,
        "similarity_threshold": "cleansplit",
        "use_precomputed_split": True,
        "training_history": {
            "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
            "val_metrics": None,
            "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
            "n_train_samples": len(y_train),
            "n_val_samples": None,
            "n_test_samples": len(y_test),
            "training_time": round(training_time, 1),
            "hyperparameters": model_config.get("hyperparameters", {}),
        },
        "per_complex_predictions": per_complex,
    }

    # Save summary
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{model_type}_pdbbind_{test_set}_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    # Save model
    model_path = results_dir / f"{model_type}_pdbbind_{test_set}.pkl"
    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train classical ML regressors on PDBbind CleanSplit"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmarks/05_pdbbind_comparison/configs/classical_pdbbind_config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Train only this model type (random_forest, gradient_boosting, svm)",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        help="Override test set (casf2016 or casf2013)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--protein-embeddings",
        type=str,
        default=None,
        help="Path to protein_embeddings.npz (concatenated with Morgan FPs)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    # Resolve relative paths
    smiles_path = (config_dir / config["data"]["smiles_path"]).resolve()
    labels_path = (config_dir / config["data"]["labels_path"]).resolve()
    split_path = (config_dir / config["data"]["split_path"]).resolve()
    test_set = args.test_set or config["data"]["test_set"]
    results_dir = Path(args.output_dir or config["output"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = config_dir / results_dir

    # Load data
    print("Loading data...")
    smiles_data = load_smiles_data(str(smiles_path))
    split_dict = load_split_data(str(split_path))
    print(f"  SMILES entries: {len(smiles_data)}")

    # Prepare train/test splits
    train_smiles, train_pK, train_pdb_ids, test_smiles, test_pK, test_pdb_ids = prepare_data(
        smiles_data, split_dict, test_set
    )
    print(f"  Train: {len(train_smiles)}, Test ({test_set}): {len(test_smiles)}")

    if len(train_smiles) == 0 or len(test_smiles) == 0:
        print("ERROR: No train or test data found.")
        print("Did you run extract_pdbbind_smiles.py first?")
        sys.exit(1)

    # Load protein embeddings if provided
    prot_emb_dict = None
    if args.protein_embeddings:
        prot_emb_path = Path(args.protein_embeddings).resolve()
        print(f"\nLoading protein embeddings from {prot_emb_path}...")
        prot_emb_data = np.load(str(prot_emb_path), allow_pickle=False)
        prot_emb_dict = {k: prot_emb_data[k] for k in prot_emb_data.files}
        sample_dim = next(iter(prot_emb_dict.values())).shape[0]
        print(f"  {len(prot_emb_dict)} embeddings, dim={sample_dim}")

    # Featurize
    feat_config = config["features"]
    radius = feat_config.get("radius", 2)
    n_bits = feat_config.get("n_bits", 2048)

    print("\nFeaturizing train set...")
    X_train_morgan, train_invalid = _featurize_morgan(train_smiles, radius, n_bits)
    print(f"Featurizing test set...")
    X_test_morgan, test_invalid = _featurize_morgan(test_smiles, radius, n_bits)

    y_train = np.array(train_pK)
    y_test = np.array(test_pK)

    # Remove invalid molecules
    train_pdb_ids_arr = list(train_pdb_ids)
    test_pdb_ids_arr = list(test_pdb_ids)

    if train_invalid:
        valid_mask = np.ones(len(y_train), dtype=bool)
        valid_mask[train_invalid] = False
        X_train_morgan = X_train_morgan[valid_mask]
        y_train = y_train[valid_mask]
        train_pdb_ids_arr = [pid for i, pid in enumerate(train_pdb_ids_arr) if i not in set(train_invalid)]
        print(f"  Removed {len(train_invalid)} invalid train molecules")

    if test_invalid:
        valid_mask = np.ones(len(y_test), dtype=bool)
        valid_mask[test_invalid] = False
        X_test_morgan = X_test_morgan[valid_mask]
        y_test = y_test[valid_mask]
        test_pdb_ids_arr = [pid for i, pid in enumerate(test_pdb_ids_arr) if i not in set(test_invalid)]
        print(f"  Removed {len(test_invalid)} invalid test molecules")

    # Concatenate protein embeddings if available
    if prot_emb_dict is not None:
        emb_dim = next(iter(prot_emb_dict.values())).shape[0]
        zero_emb = np.zeros(emb_dim, dtype=np.float32)

        train_embs = np.array([prot_emb_dict.get(pid, zero_emb) for pid in train_pdb_ids_arr])
        test_embs = np.array([prot_emb_dict.get(pid, zero_emb) for pid in test_pdb_ids_arr])

        n_train_missing = sum(1 for pid in train_pdb_ids_arr if pid not in prot_emb_dict)
        n_test_missing = sum(1 for pid in test_pdb_ids_arr if pid not in prot_emb_dict)
        print(f"\n  Protein embeddings: {emb_dim} dims")
        print(f"  Train missing: {n_train_missing}/{len(train_pdb_ids_arr)}")
        print(f"  Test missing: {n_test_missing}/{len(test_pdb_ids_arr)}")

        X_train = np.hstack([X_train_morgan, train_embs])
        X_test = np.hstack([X_test_morgan, test_embs])
        feat_type_str = f"morgan_r{radius}_b{n_bits}_protein_emb{emb_dim}"
        print(f"  Combined features: {X_train.shape[1]} dims")
    else:
        X_train = X_train_morgan
        X_test = X_test_morgan
        feat_type_str = f"morgan_r{radius}_b{n_bits}"

    test_pdb_ids = test_pdb_ids_arr

    # Select models to train
    model_configs = config["models"]
    if args.model:
        model_configs = [m for m in model_configs if m["type"] == args.model]
        if not model_configs:
            print(f"ERROR: Model '{args.model}' not found in config.")
            sys.exit(1)

    # Train each model
    for model_config in model_configs:
        train_and_evaluate(
            model_config=model_config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            test_pdb_ids=test_pdb_ids,
            results_dir=results_dir,
            test_set=test_set,
            feature_config={**feat_config, "_feature_type_str": feat_type_str},
        )

    print(f"\n{'='*60}")
    print(f"  All models trained. Results in: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
