"""
Evaluate a trained classical model (RF/GB/SVM) on real ChEMBL actives/inactives
using pChEMBL thresholds. Complements eval_real_inactives.py (which does the
dual-encoder).

Usage:
    python benchmarks/07_plate_vs_dl/eval_classical_real_inactives.py \
        --model-dir benchmarks/02_training/trained_models/rf_esm2_hard \
        --registry training_data_full/registry_soft_split_regression.csv \
        --similarity-threshold 0p7 \
        --split test \
        --split-column protein_partition \
        --esm2-path data/plate_vs_protein_embeddings/esm2_embeddings.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, average_precision_score

RDLogger.DisableLog("rdApp.*")


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def load_esm2(path: str) -> dict:
    npz = np.load(path, allow_pickle=False)
    return {k: npz[k].mean(axis=0).astype(np.float32) for k in npz.files}


def load_samples(registry_path, split, split_column, similarity_threshold,
                 active_thresh, inactive_thresh):
    samples = []
    with open(registry_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_active") != "True":
                continue
            row_split = row.get(split_column, row["split"])
            row_thresh = row.get("similarity_threshold", "")
            if row_split != split or row_thresh != similarity_threshold:
                continue

            pchembl_str = row.get("pchembl", "")
            if not pchembl_str:
                continue
            try:
                pchembl = float(pchembl_str)
            except ValueError:
                continue

            if pchembl >= active_thresh:
                label = 1
            elif pchembl < inactive_thresh:
                label = 0
            else:
                continue

            samples.append({
                "uniprot_id": row["uniprot_id"],
                "smiles": row["smiles"],
                "label": label,
            })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--esm2-path", default="data/plate_vs_protein_embeddings/esm2_embeddings.npz")
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-column", default="protein_partition")
    parser.add_argument("--similarity-threshold", default="0p7")
    parser.add_argument("--active-threshold", type=float, default=6.0)
    parser.add_argument("--inactive-threshold", type=float, default=5.0)
    parser.add_argument("--model-name", default="random_forest",
                        help="prefix of .pkl in model-dir")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / f"{args.model_name}.pkl"
    feat_cfg_path = model_dir / f"{args.model_name}_feature_config.json"
    prot_map_path = model_dir / f"{args.model_name}_protein_mapping.json"

    print(f"Loading model from {model_path}...")
    try:
        import joblib
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Determine feature config
    morgan_bits = 2048
    morgan_radius = 2
    protein_type = "esm2_protein"
    if feat_cfg_path.exists():
        with open(feat_cfg_path) as f:
            feat_cfg = json.load(f)
        # feature_config.json structure may differ; parse defensively
        print(f"Feature config: {feat_cfg}")
        lig = feat_cfg.get("ligand", {}) or feat_cfg.get("features", {}).get("ligand", {})
        morgan_bits = lig.get("n_bits", morgan_bits)
        morgan_radius = lig.get("radius", morgan_radius)
        prot = feat_cfg.get("protein", {}) or feat_cfg.get("features", {}).get("protein", {})
        protein_type = prot.get("type", protein_type)
    print(f"Morgan: r={morgan_radius} bits={morgan_bits}, protein type: {protein_type}")

    # Load ESM2 embeddings (only needed if protein type is esm2)
    use_esm2 = protein_type == "esm2_protein"
    use_randid = protein_type == "protein_identifier"
    esm2_embs = {}
    randid_embs = {}
    randid_dim = 32
    if use_esm2:
        print(f"Loading ESM2 embeddings from {args.esm2_path}...")
        esm2_embs = load_esm2(args.esm2_path)
    elif use_randid:
        # Rebuild random embeddings from saved mapping
        if not prot_map_path.exists():
            raise FileNotFoundError(f"Protein mapping not found: {prot_map_path}")
        with open(prot_map_path) as f:
            mapping = json.load(f)
        protein_to_idx = mapping["protein_to_idx"]
        randid_dim = mapping.get("embedding_dim", 32)
        use_onehot = mapping.get("use_onehot", False)
        if use_onehot:
            randid_dim = len(protein_to_idx)
            randid_embs = {p: np.eye(randid_dim, dtype=np.float32)[i]
                           for p, i in protein_to_idx.items()}
        else:
            np.random.seed(42)
            base = np.random.randn(len(protein_to_idx), randid_dim).astype(np.float32)
            base = base / np.linalg.norm(base, axis=1, keepdims=True)
            randid_embs = {p: base[i] for p, i in protein_to_idx.items()}

    # Load samples
    print(f"Loading test samples from {args.registry}...")
    samples = load_samples(
        args.registry, args.split, args.split_column, args.similarity_threshold,
        args.active_threshold, args.inactive_threshold,
    )
    n_a = sum(1 for s in samples if s["label"] == 1)
    n_i = len(samples) - n_a
    n_tgt = len(set(s["uniprot_id"] for s in samples))
    print(f"  {len(samples)} samples ({n_a} active, {n_i} inactive, {n_tgt} targets)")

    # Build features
    print("Building features...")
    X, y, uids = [], [], []
    skipped = 0
    for s in samples:
        fp = morgan_fp(s["smiles"], morgan_radius, morgan_bits)
        if fp is None:
            skipped += 1
            continue

        if use_esm2:
            prot = esm2_embs.get(s["uniprot_id"])
            if prot is None:
                prot = np.zeros(320, dtype=np.float32)
            feat = np.concatenate([fp, prot])
        elif use_randid:
            prot = randid_embs.get(s["uniprot_id"], np.zeros(randid_dim, dtype=np.float32))
            feat = np.concatenate([fp, prot])
        else:
            feat = fp

        X.append(feat)
        y.append(s["label"])
        uids.append(s["uniprot_id"])

    X = np.vstack(X)
    y = np.array(y)
    uids = np.array(uids)
    print(f"  {X.shape[0]} samples, {X.shape[1]}-dim features, {skipped} skipped")

    # Score
    print("Scoring...")
    scores = model.predict_proba(X)[:, 1]

    # Metrics
    roc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)

    per_target = []
    for uid in np.unique(uids):
        mask = uids == uid
        yy = y[mask]
        if len(np.unique(yy)) < 2:
            continue
        per_target.append({
            "target": uid,
            "auc": float(roc_auc_score(yy, scores[mask])),
            "n_active": int(yy.sum()),
            "n_inactive": int(len(yy) - yy.sum()),
        })
    pt_aucs = [t["auc"] for t in per_target]
    pt_mean = float(np.mean(pt_aucs)) if pt_aucs else float("nan")
    pt_std = float(np.std(pt_aucs)) if pt_aucs else float("nan")
    pt_med = float(np.median(pt_aucs)) if pt_aucs else float("nan")

    print(f"\n{'='*60}")
    print(f"  Classical ({args.model_name}, {protein_type}) on real inactives")
    print(f"  active >= {args.active_threshold}, inactive < {args.inactive_threshold}")
    print(f"{'='*60}")
    print(f"  Global ROC-AUC:       {roc:.4f}")
    print(f"  Avg Precision:        {ap:.4f}")
    print(f"  Per-target AUC mean:  {pt_mean:.4f} +/- {pt_std:.4f}")
    print(f"  Per-target AUC median:{pt_med:.4f}")
    print(f"  Targets evaluated:    {len(per_target)}")
    print(f"  Total samples:        {len(y)} ({int(y.sum())} active, {int(len(y)-y.sum())} inactive)")

    per_target.sort(key=lambda t: t["auc"])
    print("  Bottom 5 targets:")
    for t in per_target[:5]:
        print(f"    {t['target']}: AUC={t['auc']:.3f} ({t['n_active']}a/{t['n_inactive']}i)")
    print("  Top 5 targets:")
    for t in per_target[-5:]:
        print(f"    {t['target']}: AUC={t['auc']:.3f} ({t['n_active']}a/{t['n_inactive']}i)")

    # Save
    out = {
        "model_dir": str(model_dir),
        "model_name": args.model_name,
        "protein_type": protein_type,
        "evaluation": "real_inactives",
        "active_threshold": args.active_threshold,
        "inactive_threshold": args.inactive_threshold,
        "split_column": args.split_column,
        "split": args.split,
        "similarity_threshold": args.similarity_threshold,
        "n_samples": int(len(y)),
        "n_targets": len(per_target),
        "global_roc_auc": round(roc, 6),
        "avg_precision": round(ap, 6),
        "per_target_auc_mean": round(pt_mean, 6),
        "per_target_auc_std": round(pt_std, 6),
        "per_target_auc_median": round(pt_med, 6),
        "per_target_detail": per_target,
    }
    out_path = Path(args.output) if args.output else (
        model_dir / f"{args.model_name}_real_inactives_eval.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
