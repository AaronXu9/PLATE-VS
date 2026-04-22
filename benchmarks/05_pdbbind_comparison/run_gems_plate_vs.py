"""
GEMS inference on PLATE-VS dataset (ligand-only, no protein nodes).

Uses the B6AE0L variant (protein-nodes-deleted ablation) with ChemBERTa
embeddings computed from SMILES. Predicted pK values are used as ranking
scores for ROC-AUC evaluation.

Usage:
    python benchmarks/05_pdbbind_comparison/run_gems_plate_vs.py \
        --registry training_data_full/registry_soft_split.csv \
        --conformers data/plate_vs_conformers/conformers.pkl \
        --split-column protein_partition \
        --split test \
        --threshold 0p7 \
        --output benchmarks/05_pdbbind_comparison/results/gems_plate_vs_results.json
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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, add_self_loops

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GEMS_DIR = PROJECT_ROOT / "external" / "GEMS"
sys.path.insert(0, str(GEMS_DIR))

# ── Atom/bond feature constants (from GEMS graph_construction.py) ──────

ALL_ATOMS = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'metal', 'halogen']
HALOGENS = ['F', 'Cl', 'Br', 'I', 'At']
METALS = [
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'As', 'Si', 'Sb', 'Te',
]

SELF_LOOP_ATTR = [0., 1., 0.] + [0.]*4 + [0.]*5 + [0., 0.] + [0.]*6  # 20-dim


def one_of_k(x, allowed):
    return [x == a for a in allowed]


def one_of_k_unk(x, allowed):
    if x not in allowed:
        x = allowed[-1]
    return [x == a for a in allowed]


def get_atom_features(mol, padding_len=20):
    """Extract GEMS-compatible atom features (40-dim + padding)."""
    features = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == 'H':
            continue
        if sym in METALS:
            sym = 'metal'
        elif sym in HALOGENS:
            sym = 'halogen'

        feat = (
            one_of_k(sym, ALL_ATOMS)  # 9
            + [atom.IsInRing()]  # 1
            + one_of_k_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP2D,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED,
            ])  # 8
            + [float(atom.GetFormalCharge())]  # 1
            + [atom.GetIsAromatic()]  # 1
            + [atom.GetMass() / 100]  # 1
            + one_of_k(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 5
            + one_of_k_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 'OTHER'])  # 10
            + one_of_k_unk(str(atom.GetChiralTag()), [
                'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'OTHER',
            ])  # 4
            + [0] * padding_len  # protein residue padding (zeros for ligand atoms)
        )
        features.append(feat)
    return np.array(features, dtype=np.float32)


def get_edge_features(mol, pos):
    """Build edge index and edge attributes from covalent bonds."""
    edge_index = [[], []]
    edge_attr = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0].append(i)
        edge_index[1].append(j)

        length = np.linalg.norm(pos[i] - pos[j])
        feat = (
            one_of_k('covalent', ['covalent', 'self-loop', 'non-covalent'])
            + [length / 10] * 4
            + one_of_k(bond.GetBondTypeAsDouble(), [0., 1.0, 1.5, 2.0, 3.0])
            + [bond.GetIsConjugated()]
            + [bond.IsInRing()]
            + one_of_k(bond.GetStereo(), [
                Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
                Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS,
            ])
        )
        edge_attr.append(feat)

    ei = torch.tensor(edge_index, dtype=torch.long)
    ea = torch.tensor(edge_attr, dtype=torch.float)

    # Make undirected + add self-loops
    sl = torch.tensor([SELF_LOOP_ATTR], dtype=torch.float)
    ei, ea = to_undirected(ei, ea)
    ei, ea = add_self_loops(ei, ea, fill_value=sl)
    return ei, ea


def smiles_to_gems_graph(
    smiles: str,
    conformer: dict | None = None,
    chemberta_emb: np.ndarray | None = None,
    in_channels: int = 60,
) -> Data | None:
    """Convert a SMILES string to a GEMS-compatible PyG Data object.

    For 00AEPL (in_channels=60): no embeddings concatenated, lig_emb=zeros.
    For B6AE0L (in_channels=1148): ChemBERTa(384)+ESM2(320)+ANKH(384) concatenated.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get 3D coordinates — need mol with H for conformer, then remove
    mol = Chem.AddHs(mol)
    if conformer is not None:
        pos = conformer["pos"]
    else:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
    mol = Chem.RemoveHs(mol)
    if mol.GetNumAtoms() == 0:
        return None

    if conformer is None:
        if mol.GetNumConformers() == 0:
            return None
        pos = mol.GetConformer().GetPositions().astype(np.float32)

    # Node features (40 atom + 20 protein padding = 60 base)
    x = get_atom_features(mol, padding_len=20)
    if len(x) == 0:
        return None

    n_atoms = len(x)

    # Pad to in_channels if needed (concatenate embeddings per atom)
    if in_channels > 60:
        extra_dim = in_channels - 60
        if chemberta_emb is not None and extra_dim >= 384:
            # ChemBERTa: same 384-dim vector repeated for each atom
            cb_pad = np.tile(chemberta_emb[:384], (n_atoms, 1))
            # Remaining dims (ESM2=320 + ANKH=384) are zeros for ligand atoms
            rest_pad = np.zeros((n_atoms, extra_dim - 384), dtype=np.float32)
            x = np.concatenate([x, cb_pad, rest_pad], axis=1)
        else:
            x = np.concatenate([x, np.zeros((n_atoms, extra_dim), dtype=np.float32)], axis=1)

    # Edge features
    edge_index, edge_attr = get_edge_features(mol, pos)

    # lig_emb: ChemBERTa global pooled embedding or zeros
    if chemberta_emb is not None:
        lig_emb = torch.tensor(chemberta_emb[:384], dtype=torch.float).unsqueeze(0)
    else:
        lig_emb = torch.zeros(1, 384)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        lig_emb=lig_emb,
    )


# ── ChemBERTa embedding computation ───────────────────────────────────

def compute_chemberta_embeddings(smiles_list: list[str], batch_size: int = 256, device: str = "cuda") -> dict[str, np.ndarray]:
    """Compute ChemBERTa-77M embeddings for a list of SMILES."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading ChemBERTa-77M-MLM...")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device).eval()

    embeddings = {}
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs)
        # Mean pooling over token dimension
        embs = out.last_hidden_state.mean(dim=1).cpu().numpy()
        for smi, emb in zip(batch, embs):
            embeddings[smi] = emb

        if (i + batch_size) % 5000 < batch_size:
            print(f"  {min(i + batch_size, len(smiles_list)):,}/{len(smiles_list):,} embeddings computed")

    return embeddings


# ── GEMS model loading ────────────────────────────────────────────────

def load_gems_ensemble(variant: str = "00AEPL", device: str = "cuda") -> tuple[list, int]:
    """Load 5-fold GEMS ensemble. Returns (models, in_channels)."""
    from model.GEMS18 import GEMS18d

    model_dir = GEMS_DIR / "model"
    models = []
    in_channels = 60  # default

    for fold in range(5):
        pattern = f"GEMS18d_{variant}_*_f{fold}_best_stdict.pt"
        matches = list(model_dir.glob(pattern))
        if not matches:
            print(f"  WARNING: No model found for fold {fold} with pattern {pattern}")
            continue

        state = torch.load(matches[0], map_location=device, weights_only=True)
        in_channels = state["NodeTransform.mlp.0.weight"].shape[1]

        model = GEMS18d(
            dropout_prob=0.2, in_channels=in_channels, edge_dim=20, conv_dropout_prob=0.2,
        )
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)
        print(f"  Loaded fold {fold}: {matches[0].name} (in_channels={in_channels})")

    return models, in_channels


# ── Inference ──────────────────────────────────────────────────────────

def run_inference(models, graphs: list[Data], batch_size: int = 128, device: str = "cuda") -> np.ndarray:
    """Run GEMS ensemble inference, return predicted pK values."""
    all_preds = []

    for i in range(0, len(graphs), batch_size):
        batch_graphs = graphs[i:i + batch_size]
        batch = Batch.from_data_list(batch_graphs).to(device)

        fold_preds = []
        for model in models:
            with torch.no_grad():
                pred = model(batch).squeeze(-1)  # [batch_size]
            fold_preds.append(pred.cpu().numpy())

        # Average across folds, unscale to pK (×16)
        avg_pred = np.mean(fold_preds, axis=0) * 16.0
        all_preds.extend(avg_pred.tolist())

        if (i + batch_size) % 5000 < batch_size:
            print(f"  {min(i + batch_size, len(graphs)):,}/{len(graphs):,} scored")

    return np.array(all_preds)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEMS inference on PLATE-VS")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--conformers", default=None, help="Precomputed conformers .pkl")
    parser.add_argument("--chemberta-cache", default=None, help="Cached ChemBERTa embeddings .pkl")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", default="0p7")
    parser.add_argument("--variant", default="00AEPL", help="GEMS model variant (00AEPL=no embeddings, B6AE0L=ChemBERTa+zeros)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-decoys", type=int, default=100, help="Max decoys per target (None=all)")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    print(f"Device: {device}")

    # Load conformers
    conformers = None
    if args.conformers:
        print(f"Loading conformers from {args.conformers}...")
        with open(args.conformers, "rb") as f:
            conformers = pickle.load(f)
        print(f"  {len(conformers):,} conformers")

    # Load test samples from registry (two-pass: actives first, then decoys for test proteins)
    print(f"Loading registry ({args.split}/{args.threshold}, column={args.split_column})...")
    samples = []
    smiles_set = set()
    target_proteins = set()
    decoy_rows = []
    max_decoys = args.max_decoys

    with open(args.registry) as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_active = row["is_active"] == "True"
            if is_active:
                row_split = row.get(args.split_column, row["split"])
                row_thresh = row.get("similarity_threshold", "")
                if row_split != args.split or row_thresh != args.threshold:
                    continue
                target_proteins.add(row["uniprot_id"])
                samples.append({
                    "smiles": row["smiles"],
                    "uniprot_id": row["uniprot_id"],
                    "is_active": 1,
                })
                smiles_set.add(row["smiles"])
            else:
                decoy_rows.append(row)

    # Add decoys for test proteins
    decoy_counts = {}
    for row in decoy_rows:
        uid = row["uniprot_id"]
        if uid not in target_proteins:
            continue
        decoy_counts[uid] = decoy_counts.get(uid, 0) + 1
        if max_decoys and decoy_counts[uid] > max_decoys:
            continue
        samples.append({
            "smiles": row["smiles"],
            "uniprot_id": uid,
            "is_active": 0,
        })
        smiles_set.add(row["smiles"])

    n_active = sum(1 for s in samples if s["is_active"] == 1)
    n_decoy = len(samples) - n_active
    print(f"  {len(samples):,} test samples ({n_active:,} active, {n_decoy:,} decoy), {len(smiles_set):,} unique SMILES")

    # Load GEMS ensemble first to get in_channels
    print(f"Loading GEMS {args.variant} ensemble...")
    models, in_channels = load_gems_ensemble(args.variant, device=device)
    print(f"  {len(models)} folds loaded, in_channels={in_channels}")

    needs_chemberta = in_channels > 60

    # ChemBERTa embeddings (only if variant needs them)
    chemberta_embs = {}
    if needs_chemberta:
        if args.chemberta_cache and os.path.exists(args.chemberta_cache):
            print(f"Loading cached ChemBERTa embeddings from {args.chemberta_cache}...")
            with open(args.chemberta_cache, "rb") as f:
                chemberta_embs = pickle.load(f)
            missing = [s for s in smiles_set if s not in chemberta_embs]
            if missing:
                print(f"  Computing {len(missing)} missing embeddings...")
                new_embs = compute_chemberta_embeddings(missing, device=device)
                chemberta_embs.update(new_embs)
                with open(args.chemberta_cache, "wb") as f:
                    pickle.dump(chemberta_embs, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            smiles_list = sorted(smiles_set)
            print(f"Computing ChemBERTa embeddings for {len(smiles_list):,} SMILES...")
            chemberta_embs = compute_chemberta_embeddings(smiles_list, device=device)
            if args.chemberta_cache:
                cache_path = Path(args.chemberta_cache)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(chemberta_embs, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"  Cached to {cache_path}")

    # Build graphs
    print("Building GEMS graphs...")
    graphs = []
    labels = []
    uids = []
    n_fail = 0
    for i, s in enumerate(samples):
        smi = s["smiles"]
        conf = conformers.get(smi) if conformers else None
        cb_emb = chemberta_embs.get(smi) if needs_chemberta else None
        g = smiles_to_gems_graph(smi, conf, cb_emb, in_channels=in_channels)
        if g is None:
            n_fail += 1
            continue
        graphs.append(g)
        labels.append(s["is_active"])
        uids.append(s["uniprot_id"])

        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,}/{len(samples):,} processed ({n_fail} failed)")

    print(f"  {len(graphs):,} graphs built, {n_fail} failed")

    # Run inference
    print("Running inference...")
    t0 = time.time()
    preds = run_inference(models, graphs, batch_size=args.batch_size, device=device)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    labels = np.array(labels)

    # Global metrics
    roc_auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else float("nan")
    ap = average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else float("nan")

    # Per-target metrics
    uid_arr = np.array(uids)
    unique_uids = np.unique(uid_arr)
    per_target_aucs = []
    for uid in unique_uids:
        mask = uid_arr == uid
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, preds[mask])
        per_target_aucs.append(auc)

    pt_auc_mean = np.mean(per_target_aucs) if per_target_aucs else float("nan")
    pt_auc_std = np.std(per_target_aucs) if per_target_aucs else float("nan")

    print(f"\n{'='*50}")
    print(f"  GEMS {args.variant} on PLATE-VS ({args.threshold}, {args.split})")
    print(f"{'='*50}")
    print(f"  ROC-AUC (global):     {roc_auc:.4f}")
    print(f"  Avg Precision:        {ap:.4f}")
    print(f"  Per-target AUC mean:  {pt_auc_mean:.4f} +/- {pt_auc_std:.4f}")
    print(f"  Targets evaluated:    {len(per_target_aucs)}")
    print(f"  Samples scored:       {len(graphs)}")

    # Save results
    results = {
        "model": f"GEMS_{args.variant}",
        "dataset": "plate_vs",
        "split": args.split,
        "split_column": args.split_column,
        "similarity_threshold": args.threshold,
        "n_samples": len(graphs),
        "n_failed": n_fail,
        "metrics": {
            "roc_auc": round(roc_auc, 6),
            "avg_precision": round(ap, 6),
            "per_target_roc_auc_mean": round(pt_auc_mean, 6),
            "per_target_roc_auc_std": round(pt_auc_std, 6),
            "n_targets_evaluated": len(per_target_aucs),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
