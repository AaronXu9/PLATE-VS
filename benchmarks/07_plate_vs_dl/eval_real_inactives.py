"""
Evaluate trained PLATE-VS model on real ChEMBL actives/inactives (not DeepCoy).

Uses pChEMBL thresholds:
  - Active:   pChEMBL >= 6.0 (IC50 <= 1 μM)
  - Inactive: pChEMBL < 5.0  (IC50 > 10 μM)
  - Excluded: 5.0-6.0 (ambiguous)

Usage:
    python benchmarks/07_plate_vs_dl/eval_real_inactives.py \
        --checkpoint benchmarks/07_plate_vs_dl/checkpoints_soft/best_vs_model.pt \
        --config benchmarks/07_plate_vs_dl/configs/vs_soft_split.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BA_DIR = str(PROJECT_ROOT / "benchmarks" / "06_binding_affinity_model")

sys.path.insert(0, str(PROJECT_ROOT / "external" / "GEMS"))
sys.path.insert(0, BA_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from model.vs_model import VirtualScreeningModel
from evaluation import enrichment_factor, bedroc

import importlib.util
_collate_spec = importlib.util.spec_from_file_location(
    "ba_collate", os.path.join(BA_DIR, "data", "collate.py")
)
_collate_mod = importlib.util.module_from_spec(_collate_spec)
_collate_spec.loader.exec_module(_collate_mod)
custom_collate = _collate_mod.custom_collate


class RealInactiveDataset(torch.utils.data.Dataset):
    """Dataset using pChEMBL thresholds for real active/inactive labels."""

    def __init__(self, registry_path, protein_emb_path, conformer_path,
                 split, split_column, similarity_threshold,
                 active_threshold=6.0, inactive_threshold=5.0,
                 max_ligand_atoms=80, max_pocket_res=80):
        from torch_geometric.data import Data
        self.Data = Data
        self.max_ligand_atoms = max_ligand_atoms
        self.max_pocket_res = max_pocket_res

        # Load conformers
        print(f"  Loading conformers...")
        with open(conformer_path, "rb") as f:
            self.conformers = pickle.load(f)

        # Load protein embeddings
        print(f"  Loading protein embeddings...")
        emb_data = np.load(protein_emb_path, allow_pickle=False)
        self.protein_embs = {k: emb_data[k] for k in emb_data.files}

        # Load and filter registry
        print(f"  Loading registry ({split}/{similarity_threshold}, pChEMBL thresholds)...")
        self.samples = []
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

                # Apply thresholds
                if pchembl >= active_threshold:
                    label = 1
                elif pchembl < inactive_threshold:
                    label = 0
                else:
                    continue  # skip ambiguous

                uid = row["uniprot_id"]
                smi = row["smiles"]
                if uid not in self.protein_embs:
                    continue
                if smi not in self.conformers:
                    continue

                self.samples.append({
                    "uniprot_id": uid,
                    "smiles": smi,
                    "label": label,
                    "pchembl": pchembl,
                })

        n_active = sum(1 for s in self.samples if s["label"] == 1)
        n_inactive = len(self.samples) - n_active
        n_targets = len(set(s["uniprot_id"] for s in self.samples))
        print(f"  {len(self.samples)} samples ({n_active} active, {n_inactive} inactive, {n_targets} targets)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        conf = self.conformers[s["smiles"]]
        z = torch.tensor(conf["z"], dtype=torch.long)[:self.max_ligand_atoms]
        pos = torch.tensor(conf["pos"], dtype=torch.float32)[:self.max_ligand_atoms]
        prot_emb = torch.tensor(
            self.protein_embs[s["uniprot_id"]][:self.max_pocket_res],
            dtype=torch.float32,
        )
        return self.Data(
            z=z, pos=pos, prot_emb=prot_emb,
            num_lig_atoms=z.shape[0],
            num_pocket_res=prot_emb.shape[0],
            y=torch.tensor([s["label"]], dtype=torch.float32),
            uniprot_id=s["uniprot_id"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--active-threshold", type=float, default=6.0)
    parser.add_argument("--inactive-threshold", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    # Load config
    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config_dir = config_path.parent
    for key in ["registry_path", "protein_emb_path", "conformer_path"]:
        if key in config["data"] and config["data"][key]:
            config["data"][key] = str((config_dir / config["data"][key]).resolve())
    # Use regression registry for pChEMBL values (only it has the `pchembl` column).
    # This applies regardless of which split the trained model used.
    reg_path = config["data"]["registry_path"]
    project_root = Path(reg_path).resolve().parent
    pchembl_registry = project_root / "registry_soft_split_regression.csv"
    if not pchembl_registry.exists():
        raise FileNotFoundError(
            f"pchembl registry not found: {pchembl_registry}. "
            "Real-inactive evaluation requires registry_soft_split_regression.csv "
            "(it has the `pchembl` column)."
        )
    config["data"]["registry_path"] = str(pchembl_registry)
    print(f"  Real-inactive eval registry: {pchembl_registry}")

    dc = config["data"]
    mc = config["model"]

    # Load dataset
    print("Loading real inactive evaluation set...")
    dataset = RealInactiveDataset(
        registry_path=dc["registry_path"],
        protein_emb_path=dc["protein_emb_path"],
        conformer_path=dc["conformer_path"],
        split="test",
        split_column=dc.get("split_column", "split"),
        similarity_threshold=dc["similarity_threshold"],
        active_threshold=args.active_threshold,
        inactive_threshold=args.inactive_threshold,
        max_ligand_atoms=dc["max_ligand_atoms"],
        max_pocket_res=dc["max_pocket_residues"],
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0,
                        collate_fn=custom_collate)

    # Load model
    print("Loading model...")
    model = VirtualScreeningModel(
        esm_dim=mc["esm_dim"], proj_dim=mc["proj_dim"],
        et_layers=mc["et_layers"], et_heads=mc["et_heads"],
        et_rbf=mc["et_rbf"], et_cutoff=mc["et_cutoff"],
        cross_attn_layers=mc["cross_attn_layers"],
        cross_attn_heads=mc["cross_attn_heads"],
        dropout=mc["dropout"],
        ligand_backend=mc.get("ligand_backend", "auto"),
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} params")

    # Run inference
    print("Running inference...")
    all_scores = []
    all_labels = []
    all_uids = []

    # Pre-extract uids from dataset samples (batch.uniprot_id may not survive collation)
    sample_uids = [s["uniprot_id"] for s in dataset.samples]
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            logits = model(batch).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            bs = len(scores)
            all_scores.extend(scores.tolist())
            all_labels.extend(batch.y.squeeze(-1).cpu().numpy().tolist())
            all_uids.extend(sample_uids[sample_idx:sample_idx + bs])
            sample_idx += bs

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    uids = np.array(all_uids)

    # Filter NaN/Inf scores. TorchMD-NET ET produces non-finite logits on
    # some inputs even after the dataset-level conformer filter; sklearn's
    # roc_auc_score refuses arrays with NaN. Drop those samples from the
    # metric and record the count.
    n_total = len(scores)
    valid = np.isfinite(scores)
    n_dropped_nan = int(n_total - valid.sum())
    if n_dropped_nan > 0:
        print(f"  WARNING: dropping {n_dropped_nan}/{n_total} samples with non-finite scores "
              f"({n_dropped_nan/n_total*100:.2f}%) before metric computation")
        scores = scores[valid]
        labels = labels[valid]
        uids = uids[valid]

    # Global metrics
    roc_auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    ef_global = enrichment_factor(labels, scores)
    bedroc_global = bedroc(labels, scores)

    # Per-target metrics (AUC + EF at 1% / 5%)
    unique_uids = np.unique(uids)
    per_target = []
    for uid in unique_uids:
        mask = uids == uid
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, scores[mask])
        ef_t = enrichment_factor(y, scores[mask], (1.0, 5.0))
        n_a = int(y.sum())
        n_i = int(len(y) - y.sum())
        per_target.append({
            "target": uid,
            "auc": auc,
            "ef_1pct": ef_t.get("ef_1pct"),
            "ef_5pct": ef_t.get("ef_5pct"),
            "n_active": n_a,
            "n_inactive": n_i,
        })

    pt_aucs = [t["auc"] for t in per_target]
    pt_mean = np.mean(pt_aucs)
    pt_std = np.std(pt_aucs)
    pt_median = np.median(pt_aucs)
    pt_ef1 = [t["ef_1pct"] for t in per_target if t["ef_1pct"] is not None]
    pt_ef5 = [t["ef_5pct"] for t in per_target if t["ef_5pct"] is not None]
    pt_ef1_mean = float(np.mean(pt_ef1)) if pt_ef1 else float("nan")
    pt_ef5_mean = float(np.mean(pt_ef5)) if pt_ef5 else float("nan")

    print(f"\n{'='*60}")
    print(f"  Real Inactive Evaluation (active >= {args.active_threshold}, inactive < {args.inactive_threshold})")
    print(f"{'='*60}")
    print(f"  Global ROC-AUC:       {roc_auc:.4f}")
    print(f"  Avg Precision:        {ap:.4f}")
    print(f"  Global EF1%/5%:       {ef_global.get('ef_1pct', float('nan'))} / {ef_global.get('ef_5pct', float('nan'))}")
    print(f"  Global BEDROC a=20:   {bedroc_global.get('bedroc_a20', float('nan'))}")
    print(f"  Per-target AUC mean:  {pt_mean:.4f} +/- {pt_std:.4f}")
    print(f"  Per-target AUC median:{pt_median:.4f}")
    print(f"  Per-target EF1%/5%:   {pt_ef1_mean:.4f} / {pt_ef5_mean:.4f}")
    print(f"  Targets evaluated:    {len(per_target)}")
    print(f"  Total samples:        {len(scores)} ({int(labels.sum())} active, {int(len(labels)-labels.sum())} inactive)")

    # Show worst and best targets
    per_target.sort(key=lambda x: x["auc"])
    print(f"\n  Bottom 5 targets:")
    for t in per_target[:5]:
        print(f"    {t['target']}: AUC={t['auc']:.3f} ({t['n_active']}a/{t['n_inactive']}i)")
    print(f"  Top 5 targets:")
    for t in per_target[-5:]:
        print(f"    {t['target']}: AUC={t['auc']:.3f} ({t['n_active']}a/{t['n_inactive']}i)")

    # Save results
    results = {
        "evaluation": "real_inactives",
        "active_threshold": args.active_threshold,
        "inactive_threshold": args.inactive_threshold,
        "checkpoint": args.checkpoint,
        "global_roc_auc": round(roc_auc, 6),
        "avg_precision": round(ap, 6),
        **ef_global,
        **bedroc_global,
        "per_target_auc_mean": round(pt_mean, 6),
        "per_target_auc_std": round(pt_std, 6),
        "per_target_auc_median": round(pt_median, 6),
        "per_target_ef_1pct_mean": round(pt_ef1_mean, 6) if pt_ef1 else None,
        "per_target_ef_5pct_mean": round(pt_ef5_mean, 6) if pt_ef5 else None,
        "n_targets": len(per_target),
        "n_samples": int(len(scores)),
        "n_dropped_nan": n_dropped_nan,
        "per_target_detail": per_target,
    }

    out_path = args.output or str(SCRIPT_DIR / "results" / "vs_real_inactives_eval.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    # Resolve config/checkpoint paths before chdir
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg in ("--config", "--checkpoint", "--output") and i + 1 < len(sys.argv):
                sys.argv[i + 1] = str(Path(sys.argv[i + 1]).resolve())
    os.chdir(SCRIPT_DIR)
    main()
