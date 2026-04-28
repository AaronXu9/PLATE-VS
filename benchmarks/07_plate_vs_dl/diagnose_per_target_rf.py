"""
Experiment 1: Per-target Random Forest on Morgan FP only.

For each test protein, train a small RF on its actives + decoys (Morgan FP only,
no protein features), 70/30 train/test split. Records per-target AUC.

Hypothesis: if AUC is high per-target, there's a learnable per-target signal in
Morgan FP that the global classical pipeline can't access (because it has to
condition on the protein, which is encoded as random/coarse features). The
dual-encoder accesses this signal via ESM2 conditioning.

Usage:
    python benchmarks/07_plate_vs_dl/diagnose_per_target_rf.py
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

REGISTRY = "training_data_full/registry_soft_split.csv"
SIM_THRESHOLD = "0p7"
SPLIT_COLUMN = "protein_partition"
TARGET_PARTITION = "test"
MAX_DECOYS_PER_TARGET = 200
MIN_ACTIVES_PER_TARGET = 10
MIN_DECOYS_PER_TARGET = 10
N_TREES = 100
RANDOM_STATE = 42


def morgan(smi: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)


def main():
    print("Loading registry...")
    by_target_active: dict[str, list[str]] = defaultdict(list)
    by_target_decoy: dict[str, list[str]] = defaultdict(list)

    with open(REGISTRY) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uniprot_id"]
            smi = row["smiles"]
            is_active = row["is_active"] == "True"
            if is_active:
                if row.get(SPLIT_COLUMN) != TARGET_PARTITION:
                    continue
                if row.get("similarity_threshold") != SIM_THRESHOLD:
                    continue
                by_target_active[uid].append(smi)
            else:
                by_target_decoy[uid].append(smi)

    targets = sorted(by_target_active.keys() & by_target_decoy.keys())
    print(f"  {len(targets)} targets with both actives and decoys (test partition)")

    results = []
    t_start = time.time()
    for i, uid in enumerate(targets):
        actives = by_target_active[uid]
        decoys = by_target_decoy[uid][:MAX_DECOYS_PER_TARGET]
        if len(actives) < MIN_ACTIVES_PER_TARGET or len(decoys) < MIN_DECOYS_PER_TARGET:
            continue

        # Featurize
        X, y = [], []
        for smi in actives:
            fp = morgan(smi)
            if fp is not None:
                X.append(fp); y.append(1)
        for smi in decoys:
            fp = morgan(smi)
            if fp is not None:
                X.append(fp); y.append(0)

        X = np.vstack(X); y = np.array(y)
        if len(np.unique(y)) < 2 or sum(y == 1) < 5 or sum(y == 0) < 5:
            continue

        # 70/30 stratified split
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE,
            )
        except ValueError:
            continue
        if len(np.unique(y_te)) < 2:
            continue

        rf = RandomForestClassifier(
            n_estimators=N_TREES, n_jobs=-1, class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        rf.fit(X_tr, y_tr)
        scores = rf.predict_proba(X_te)[:, 1]

        auc = float(roc_auc_score(y_te, scores))
        ap = float(average_precision_score(y_te, scores))
        results.append({
            "target": uid,
            "auc": auc,
            "ap": ap,
            "n_active": int(sum(y == 1)),
            "n_decoy": int(sum(y == 0)),
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  {i+1}/{len(targets)} done, {elapsed:.0f}s elapsed; "
                  f"running mean AUC={np.mean([r['auc'] for r in results]):.3f}")

    # Aggregate
    aucs = np.array([r["auc"] for r in results])
    summary = {
        "n_targets_evaluated": len(results),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std()),
        "auc_median": float(np.median(aucs)),
        "auc_p10": float(np.percentile(aucs, 10)),
        "auc_p90": float(np.percentile(aucs, 90)),
        "n_targets_auc_above_0p9": int((aucs > 0.9).sum()),
        "n_targets_auc_above_0p7": int((aucs > 0.7).sum()),
        "n_targets_auc_below_0p6": int((aucs < 0.6).sum()),
    }

    print("\n" + "="*60)
    print("  Per-Target RF (Morgan FP only) on DeepCoy Decoys")
    print("="*60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")

    # Save
    out = {"summary": summary, "per_target": results,
           "config": {
               "registry": REGISTRY,
               "similarity_threshold": SIM_THRESHOLD,
               "split_column": SPLIT_COLUMN,
               "target_partition": TARGET_PARTITION,
               "max_decoys_per_target": MAX_DECOYS_PER_TARGET,
               "n_trees": N_TREES,
           }}
    out_path = Path("benchmarks/07_plate_vs_dl/results/diagnostic_per_target_rf.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
