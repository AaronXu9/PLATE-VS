"""
Experiment 2: Per-target property distribution differences between actives and decoys.

For each test target, compute Cohen's d for ~15 molecular properties between
its actives and decoys. Identifies which properties differ most per-target
(even if globally they match, per-target they may not).

Also runs DeepCoy's own DOE / LADS / Doppelganger metrics per target.

Usage:
    python benchmarks/07_plate_vs_dl/diagnose_per_target_properties.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# Make DeepCoy evaluation utils importable
DEEPCOY_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "DeepCoy" / "evaluation"
sys.path.insert(0, str(DEEPCOY_DIR))
import decoy_utils  # type: ignore

RDLogger.DisableLog("rdApp.*")

REGISTRY = "training_data_full/registry_soft_split.csv"
SIM_THRESHOLD = "0p7"
SPLIT_COLUMN = "protein_partition"
TARGET_PARTITION = "test"
MAX_DECOYS_PER_TARGET = 200
MIN_ACTIVES_PER_TARGET = 10
MIN_DECOYS_PER_TARGET = 10

PROP_NAMES = [
    "MW", "logP", "HBD", "HBA", "rings", "aromatic_rings",
    "rotatable", "TPSA", "n_heavy", "n_unique_elements",
    "frac_C", "frac_N", "frac_O", "frac_S", "frac_halogen",
]


def compute_props(smi: str) -> list[float] | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    n_h = mol.GetNumHeavyAtoms()
    if n_h == 0:
        return None
    halogens = {"F", "Cl", "Br", "I"}
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        Descriptors.TPSA(mol),
        n_h,
        len(set(atoms)),
        atoms.count("C") / n_h,
        atoms.count("N") / n_h,
        atoms.count("O") / n_h,
        atoms.count("S") / n_h,
        sum(atoms.count(h) for h in halogens) / n_h,
    ]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


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
    print(f"  {len(targets)} targets with both actives and decoys")

    # Per-target Cohen's d for each property
    per_target_records = []
    cohens_d_grid = []  # rows: targets, cols: properties
    deepcoy_metrics = []

    t_start = time.time()
    for i, uid in enumerate(targets):
        actives = by_target_active[uid]
        decoys = by_target_decoy[uid][:MAX_DECOYS_PER_TARGET]
        if len(actives) < MIN_ACTIVES_PER_TARGET or len(decoys) < MIN_DECOYS_PER_TARGET:
            continue

        # Compute properties
        a_props = [compute_props(s) for s in actives]
        d_props = [compute_props(s) for s in decoys]
        a_props = np.array([p for p in a_props if p is not None])
        d_props = np.array([p for p in d_props if p is not None])
        if len(a_props) < MIN_ACTIVES_PER_TARGET or len(d_props) < MIN_DECOYS_PER_TARGET:
            continue

        ds = [cohens_d(a_props[:, k], d_props[:, k]) for k in range(len(PROP_NAMES))]
        cohens_d_grid.append(ds)

        # DeepCoy DOE/LADS/DG (use DUDE properties)
        try:
            a_mols = [Chem.MolFromSmiles(s) for s in actives if Chem.MolFromSmiles(s)]
            d_mols = [Chem.MolFromSmiles(s) for s in decoys if Chem.MolFromSmiles(s)]
            if len(a_mols) >= 3 and len(d_mols) >= 3:
                a_feat = np.array([decoy_utils.calc_props_dude(m) for m in a_mols])
                d_feat = np.array([decoy_utils.calc_props_dude(m) for m in d_mols])
                doe = float(decoy_utils.doe_score(a_feat, d_feat))
                lads = float(np.mean(decoy_utils.lads_score_v2(a_mols, d_mols)))
                dg_scores, _ = decoy_utils.dg_score(a_mols, d_mols)
                dg_max = float(max(dg_scores)) if len(dg_scores) else float("nan")
                dg_mean = float(np.mean(dg_scores)) if len(dg_scores) else float("nan")
            else:
                doe = lads = dg_max = dg_mean = float("nan")
        except Exception as e:
            print(f"  WARN {uid}: DeepCoy metrics failed: {e}")
            doe = lads = dg_max = dg_mean = float("nan")

        deepcoy_metrics.append({"target": uid, "doe": doe, "lads": lads,
                                "dg_mean": dg_mean, "dg_max": dg_max})

        per_target_records.append({
            "target": uid,
            "n_active": int(len(a_props)),
            "n_decoy": int(len(d_props)),
            "cohens_d": dict(zip(PROP_NAMES, ds)),
            "doe": doe, "lads": lads, "dg_mean": dg_mean, "dg_max": dg_max,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(targets)} processed ({time.time()-t_start:.0f}s)")

    cohens_d_grid = np.array(cohens_d_grid)  # (n_targets, n_props)

    # Aggregate per property: how often is |d| > 0.5? mean |d|? max |d|?
    per_property_summary = []
    for k, name in enumerate(PROP_NAMES):
        col = cohens_d_grid[:, k]
        col_abs = np.abs(col[~np.isnan(col)])
        per_property_summary.append({
            "property": name,
            "n_targets": int(len(col_abs)),
            "mean_abs_d": float(col_abs.mean()) if len(col_abs) else float("nan"),
            "median_abs_d": float(np.median(col_abs)) if len(col_abs) else float("nan"),
            "max_abs_d": float(col_abs.max()) if len(col_abs) else float("nan"),
            "frac_targets_d_gt_0p5": float((col_abs > 0.5).mean()) if len(col_abs) else float("nan"),
            "frac_targets_d_gt_1p0": float((col_abs > 1.0).mean()) if len(col_abs) else float("nan"),
        })

    # Sort properties by impact (mean |d|)
    per_property_summary.sort(key=lambda r: r["mean_abs_d"], reverse=True)

    # Aggregate DeepCoy metrics
    valid_doe = [m["doe"] for m in deepcoy_metrics if not np.isnan(m["doe"])]
    valid_lads = [m["lads"] for m in deepcoy_metrics if not np.isnan(m["lads"])]
    valid_dgmax = [m["dg_max"] for m in deepcoy_metrics if not np.isnan(m["dg_max"])]
    valid_dgmean = [m["dg_mean"] for m in deepcoy_metrics if not np.isnan(m["dg_mean"])]
    deepcoy_summary = {
        "doe_mean": float(np.mean(valid_doe)) if valid_doe else float("nan"),
        "doe_median": float(np.median(valid_doe)) if valid_doe else float("nan"),
        "lads_mean": float(np.mean(valid_lads)) if valid_lads else float("nan"),
        "dg_max_mean": float(np.mean(valid_dgmax)) if valid_dgmax else float("nan"),
        "dg_mean_mean": float(np.mean(valid_dgmean)) if valid_dgmean else float("nan"),
    }

    # Print
    print("\n" + "="*70)
    print("  Per-target Property Cohen's d (sorted by mean |d|)")
    print("="*70)
    print(f"  {'Property':18s} {'mean |d|':>10s} {'med |d|':>10s} {'max |d|':>10s} {'>0.5':>8s} {'>1.0':>8s}")
    for r in per_property_summary:
        print(f"  {r['property']:18s} {r['mean_abs_d']:10.3f} {r['median_abs_d']:10.3f} "
              f"{r['max_abs_d']:10.3f} {r['frac_targets_d_gt_0p5']*100:7.1f}% "
              f"{r['frac_targets_d_gt_1p0']*100:7.1f}%")

    print(f"\nDeepCoy metrics (lower=better matching):")
    for k, v in deepcoy_summary.items():
        print(f"  {k:25s}: {v:.3f}")

    out = {
        "per_property_summary": per_property_summary,
        "deepcoy_metrics_summary": deepcoy_summary,
        "n_targets": len(per_target_records),
        "per_target_detail": per_target_records,
    }
    out_path = Path("benchmarks/07_plate_vs_dl/results/diagnostic_per_target_properties.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
