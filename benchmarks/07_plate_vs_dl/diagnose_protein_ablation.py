"""
Experiment 4: Dual-encoder protein ablation.

Run inference on the soft-split test set with three protein conditions:
  1. Original ESM2 embeddings (baseline 0.999)
  2. All-zeros protein embeddings
  3. Shuffled (each ligand paired with random other protein's embedding)

If the model still hits ~0.999 with zeros/shuffled, the signal is purely
ligand-based (model learned DeepCoy artifacts independent of which protein).
If AUC drops, protein conditioning matters.

Usage:
    python benchmarks/07_plate_vs_dl/diagnose_protein_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BA_DIR = str(PROJECT_ROOT / "benchmarks" / "06_binding_affinity_model")

sys.path.insert(0, str(PROJECT_ROOT / "external" / "GEMS"))
sys.path.insert(0, BA_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from model.vs_model import VirtualScreeningModel  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location("ba_collate", os.path.join(BA_DIR, "data", "collate.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
custom_collate = _mod.custom_collate


REGISTRY = "training_data_full/registry_soft_split.csv"
SIM_THRESHOLD = "0p7"
SPLIT_COLUMN = "protein_partition"
SPLIT = "test"
MAX_DECOYS_PER_TARGET = 100
CHECKPOINT = "benchmarks/07_plate_vs_dl/checkpoints_soft/best_vs_model.pt"
ESM2_PATH = "data/plate_vs_protein_embeddings/esm2_embeddings.npz"
CONFORMERS_PATH = "data/plate_vs_conformers/conformers.pkl"

# Model hyperparameters (match training config)
MODEL_KW = dict(
    esm_dim=320, proj_dim=256, et_layers=4, et_heads=8, et_rbf=64, et_cutoff=10.0,
    cross_attn_layers=3, cross_attn_heads=8, dropout=0.15, ligand_backend="schnet",
)


class AblationDataset(Dataset):
    """Simple dataset that emits a fixed protein_emb tensor per sample.

    The protein_emb is set externally (zeros / shuffled / original).
    """

    def __init__(self, samples, conformers, protein_emb_lookup, max_lig=80, max_pock=80):
        from torch_geometric.data import Data
        self.Data = Data
        self.samples = samples
        self.conformers = conformers
        self.protein_emb_lookup = protein_emb_lookup  # uniprot_id -> ndarray (n_res, 320)
        self.max_lig = max_lig
        self.max_pock = max_pock

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        conf = self.conformers[s["smiles"]]
        z = torch.tensor(conf["z"], dtype=torch.long)[:self.max_lig]
        pos = torch.tensor(conf["pos"], dtype=torch.float32)[:self.max_lig]
        prot = self.protein_emb_lookup[s["assigned_uid"]]
        prot_emb = torch.tensor(prot[:self.max_pock], dtype=torch.float32)
        return self.Data(
            z=z, pos=pos, prot_emb=prot_emb,
            num_lig_atoms=z.shape[0],
            num_pocket_res=prot_emb.shape[0],
            y=torch.tensor([s["label"]], dtype=torch.float32),
            uniprot_id=s["uniprot_id"],
        )


def load_samples():
    """Load test samples (actives by partition + decoys for matching proteins)."""
    target_proteins = set()
    samples = []
    decoy_rows = []
    with open(REGISTRY) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uniprot_id"]
            if row["is_active"] == "True":
                if row.get(SPLIT_COLUMN) != SPLIT or row.get("similarity_threshold") != SIM_THRESHOLD:
                    continue
                target_proteins.add(uid)
                samples.append({"uniprot_id": uid, "smiles": row["smiles"], "label": 1})
            else:
                decoy_rows.append(row)

    decoy_counts = {}
    for row in decoy_rows:
        uid = row["uniprot_id"]
        if uid not in target_proteins:
            continue
        decoy_counts[uid] = decoy_counts.get(uid, 0) + 1
        if decoy_counts[uid] > MAX_DECOYS_PER_TARGET:
            continue
        samples.append({"uniprot_id": uid, "smiles": row["smiles"], "label": 0})

    return samples, target_proteins


def run_inference(model, loader, device):
    all_scores, all_labels, all_uids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(batch.y.squeeze(-1).cpu().numpy().tolist())
    return np.array(all_scores), np.array(all_labels)


def metrics(scores, labels, uids):
    roc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    pt = []
    for uid in np.unique(uids):
        mask = uids == uid
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        pt.append(roc_auc_score(y, scores[mask]))
    pt = np.array(pt) if pt else np.array([np.nan])
    return {
        "roc_auc": roc,
        "avg_precision": ap,
        "per_target_auc_mean": float(pt.mean()),
        "per_target_auc_std": float(pt.std()),
        "n_targets": int(len(pt)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading conformers...")
    with open(CONFORMERS_PATH, "rb") as f:
        conformers = pickle.load(f)

    print("Loading ESM2 embeddings...")
    esm2 = np.load(ESM2_PATH, allow_pickle=False)
    esm2_dict = {k: esm2[k] for k in esm2.files}

    print("Loading samples...")
    raw_samples, target_proteins = load_samples()
    # Filter samples missing conformer or ESM2
    samples = [s for s in raw_samples
               if s["smiles"] in conformers and s["uniprot_id"] in esm2_dict]
    print(f"  {len(samples)} samples ({sum(1 for s in samples if s['label']==1)} active, "
          f"{sum(1 for s in samples if s['label']==0)} decoy, "
          f"{len(set(s['uniprot_id'] for s in samples))} targets)")

    # Load model
    print(f"Loading checkpoint {CHECKPOINT}...")
    model = VirtualScreeningModel(**MODEL_KW).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    uids_list = [s["uniprot_id"] for s in samples]
    uids_arr = np.array(uids_list)
    rng = np.random.default_rng(42)

    results = {}

    # ---- Condition 1: original ESM2 ----
    print("\n[1/3] Original ESM2 embeddings...")
    for s in samples:
        s["assigned_uid"] = s["uniprot_id"]
    ds = AblationDataset(samples, conformers, esm2_dict)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate)
    scores, labels = run_inference(model, loader, device)
    results["original_esm2"] = metrics(scores, labels, uids_arr)
    print(f"  AUC={results['original_esm2']['roc_auc']:.4f}, "
          f"per-target AUC={results['original_esm2']['per_target_auc_mean']:.4f}")

    # ---- Condition 2: zero protein embeddings ----
    print("\n[2/3] All-zero protein embeddings...")
    zero_lookup = {uid: np.zeros_like(emb) for uid, emb in esm2_dict.items()}
    ds = AblationDataset(samples, conformers, zero_lookup)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate)
    scores, labels = run_inference(model, loader, device)
    results["zero_protein"] = metrics(scores, labels, uids_arr)
    print(f"  AUC={results['zero_protein']['roc_auc']:.4f}, "
          f"per-target AUC={results['zero_protein']['per_target_auc_mean']:.4f}")

    # ---- Condition 3: shuffled protein embeddings (each ligand → random OTHER protein) ----
    print("\n[3/3] Shuffled protein embeddings (each sample paired with random other protein)...")
    all_uids = list(esm2_dict.keys())
    for s in samples:
        # Pick a different uniprot_id at random
        candidates = [u for u in all_uids if u != s["uniprot_id"]]
        s["assigned_uid"] = candidates[rng.integers(0, len(candidates))]
    ds = AblationDataset(samples, conformers, esm2_dict)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate)
    scores, labels = run_inference(model, loader, device)
    results["shuffled_protein"] = metrics(scores, labels, uids_arr)
    print(f"  AUC={results['shuffled_protein']['roc_auc']:.4f}, "
          f"per-target AUC={results['shuffled_protein']['per_target_auc_mean']:.4f}")

    # Print comparison
    print("\n" + "="*70)
    print("  Dual-Encoder Protein Ablation Summary (DeepCoy soft-split test)")
    print("="*70)
    print(f"  {'Condition':25s} {'Global AUC':>12s} {'Per-target AUC':>16s}")
    for k in ("original_esm2", "zero_protein", "shuffled_protein"):
        r = results[k]
        print(f"  {k:25s} {r['roc_auc']:12.4f} {r['per_target_auc_mean']:16.4f}")

    # Save
    out_path = Path("benchmarks/07_plate_vs_dl/results/diagnostic_protein_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg in ("--checkpoint", "--esm2", "--conformers") and i + 1 < len(sys.argv):
                sys.argv[i + 1] = str(Path(sys.argv[i + 1]).resolve())
    os.chdir(PROJECT_ROOT)
    main()
