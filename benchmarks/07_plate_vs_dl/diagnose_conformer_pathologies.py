"""
Scan the precomputed conformer cache for inputs that would likely make
TorchMD-NET ET produce NaN.

Common culprits:
  1. Two atoms at (nearly) identical coordinates  → 1/r blow-up
  2. NaN / Inf in coordinates
  3. Coordinates with huge magnitudes (suggest broken embed)
  4. Single-atom inputs (already filtered in dataset, but verify)
  5. Atoms with non-standard atomic numbers (Z > 100)

Usage:
    python benchmarks/07_plate_vs_dl/diagnose_conformer_pathologies.py
"""
from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

CACHE = "data/plate_vs_conformers/conformers_full.pkl"
OUT = "benchmarks/07_plate_vs_dl/results/conformer_pathologies.json"

# Cutoffs
MIN_DIST_DANGER = 0.1      # Å; below this, ET 1/r style ops blow up
COORD_MAX_DANGER = 1000.0  # Å; sane molecules are ~30Å in extent
WARN_FIRST_N = 10          # print this many examples per category

def main():
    print(f"Loading {CACHE}...")
    t0 = time.time()
    with open(CACHE, "rb") as f:
        confs = pickle.load(f)
    print(f"  {len(confs):,} entries loaded in {time.time()-t0:.1f}s")

    counts = Counter()
    examples = {k: [] for k in ("nan_inf", "min_dist_danger", "huge_coord",
                                "single_atom", "weird_z", "ok")}
    min_dist_hist = []
    max_coord_hist = []
    n_atoms_hist = []

    t0 = time.time()
    for i, (smi, entry) in enumerate(confs.items()):
        z = entry["z"]
        pos = entry["pos"]
        n_atoms = z.shape[0]
        n_atoms_hist.append(n_atoms)

        # 1. Single atom
        if n_atoms < 2:
            counts["single_atom"] += 1
            if len(examples["single_atom"]) < WARN_FIRST_N:
                examples["single_atom"].append({"smiles": smi[:120], "n_atoms": int(n_atoms)})
            continue

        # 2. NaN / Inf in coordinates
        if not np.isfinite(pos).all():
            counts["nan_inf"] += 1
            if len(examples["nan_inf"]) < WARN_FIRST_N:
                examples["nan_inf"].append({"smiles": smi[:120], "n_atoms": int(n_atoms)})
            continue

        # 3. Huge coordinates
        max_coord = float(np.abs(pos).max())
        max_coord_hist.append(max_coord)
        if max_coord > COORD_MAX_DANGER:
            counts["huge_coord"] += 1
            if len(examples["huge_coord"]) < WARN_FIRST_N:
                examples["huge_coord"].append({
                    "smiles": smi[:120],
                    "n_atoms": int(n_atoms),
                    "max_coord": max_coord,
                })
            continue

        # 4. Min pairwise distance
        # For efficiency at scale, compute on subsampled atoms if very large
        if n_atoms <= 200:
            diff = pos[:, None, :] - pos[None, :, :]  # [n, n, 3]
            d = np.sqrt((diff ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            min_d = float(d.min())
        else:
            # Subsample 200 atoms for speed
            idx = np.random.choice(n_atoms, 200, replace=False)
            sub = pos[idx]
            diff = sub[:, None, :] - sub[None, :, :]
            d = np.sqrt((diff ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            min_d = float(d.min())
        min_dist_hist.append(min_d)
        if min_d < MIN_DIST_DANGER:
            counts["min_dist_danger"] += 1
            if len(examples["min_dist_danger"]) < WARN_FIRST_N:
                examples["min_dist_danger"].append({
                    "smiles": smi[:120],
                    "n_atoms": int(n_atoms),
                    "min_pairwise_dist": min_d,
                })
            continue

        # 5. Weird atomic numbers
        if z.max() > 100:
            counts["weird_z"] += 1
            if len(examples["weird_z"]) < WARN_FIRST_N:
                examples["weird_z"].append({
                    "smiles": smi[:120],
                    "n_atoms": int(n_atoms),
                    "max_z": int(z.max()),
                })
            continue

        counts["ok"] += 1
        if (i + 1) % 100000 == 0:
            print(f"  {i+1:,}/{len(confs):,} scanned in {time.time()-t0:.0f}s; counts: {dict(counts)}")

    elapsed = time.time() - t0
    print(f"\nScan finished in {elapsed/60:.1f} min")
    print(f"  Total: {sum(counts.values()):,}")

    print("\n=== Counts ===")
    for k, v in counts.most_common():
        pct = v / max(1, sum(counts.values())) * 100
        print(f"  {k:25s}: {v:10,d}  ({pct:5.2f}%)")

    print("\n=== Min pairwise distance distribution (Å) ===")
    if min_dist_hist:
        md = np.array(min_dist_hist)
        for q in (1, 5, 50, 95, 99):
            print(f"  p{q:02d}: {np.percentile(md, q):.4f}")
        print(f"  count < 0.5 Å: {(md < 0.5).sum():,} ({(md < 0.5).mean()*100:.2f}%)")
        print(f"  count < 0.1 Å: {(md < 0.1).sum():,} ({(md < 0.1).mean()*100:.2f}%)")
        print(f"  count < 0.01 Å: {(md < 0.01).sum():,} ({(md < 0.01).mean()*100:.2f}%)")

    print("\n=== Max coord magnitude distribution (Å) ===")
    if max_coord_hist:
        mc = np.array(max_coord_hist)
        for q in (50, 95, 99, 99.9):
            print(f"  p{q}: {np.percentile(mc, q):.2f}")

    print("\n=== n_atoms distribution ===")
    if n_atoms_hist:
        na = np.array(n_atoms_hist)
        for q in (1, 50, 99, 99.9):
            print(f"  p{q}: {np.percentile(na, q):.0f}")
        print(f"  max: {na.max()}")

    print("\n=== First few examples per pathology ===")
    for k, items in examples.items():
        if k == "ok" or not items:
            continue
        print(f"\n--- {k} ({counts[k]} total) ---")
        for ex in items[:5]:
            print(f"  {ex}")

    # Save
    out = {
        "cache_path": CACHE,
        "n_total": int(sum(counts.values())),
        "counts": dict(counts),
        "min_dist_percentiles": {
            f"p{q}": float(np.percentile(min_dist_hist, q)) if min_dist_hist else None
            for q in (1, 5, 50, 95, 99)
        },
        "examples": examples,
    }
    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
