"""
Precompute and cache molecular features for the full training dataset.

Run once before training to avoid recomputing expensive features (Morgan
fingerprints, molecular descriptors) on every training run or ablation study.

Storage: HDF5 with gzip compression, packed bits for fingerprints.
  - Morgan FP (r=2, 2048 bits): ~300-500 MB for the full dataset
  - Molecular descriptors:       ~50-80 MB

Usage
-----
conda run -n rdkit_env python3 precompute_features.py \\
    --registry ../../training_data_full/registry.csv \\
    --cache-dir ../../training_data_full/feature_cache

# Custom feature type:
conda run -n rdkit_env python3 precompute_features.py \\
    --feature-type descriptors \\
    --registry ../../training_data_full/registry.csv \\
    --cache-dir ../../training_data_full/feature_cache
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from features.feature_cache import FeatureCache
from features.featurizer import MorganFingerprintFeaturizer, DescriptorFeaturizer


def precompute_morgan(registry_path: str, cache_dir: str,
                      radius: int = 2, n_bits: int = 2048,
                      batch_size: int = 50_000) -> None:
    """Precompute Morgan fingerprints for all unique SMILES in the registry."""
    print(f"\n{'='*60}")
    print(f"Precomputing Morgan fingerprints (r={radius}, {n_bits} bits)")
    print(f"{'='*60}")

    registry = pd.read_csv(registry_path)
    print(f"Registry loaded: {len(registry):,} rows")

    unique_smiles = registry["smiles"].dropna().unique().tolist()
    print(f"Unique SMILES: {len(unique_smiles):,}")

    config = {"type": "morgan_fingerprint", "radius": radius, "n_bits": n_bits}
    cache = FeatureCache(cache_dir, config)

    n_already = cache.count()
    if n_already > 0:
        print(f"Cache already has {n_already:,} entries — checking for missing ones")

    # Find which SMILES are not yet cached
    rows = cache.lookup_rows(unique_smiles)
    missing_mask = rows == -1
    missing_smiles = [s for s, m in zip(unique_smiles, missing_mask) if m]
    print(f"To compute: {len(missing_smiles):,} (skipping {n_already:,} cached)")

    if not missing_smiles:
        print("All SMILES already cached. Nothing to do.")
        return

    featurizer = MorganFingerprintFeaturizer(radius=radius, n_bits=n_bits)

    n_batches = (len(missing_smiles) + batch_size - 1) // batch_size
    t0 = time.time()
    total_invalid = 0

    for batch_i in range(n_batches):
        start = batch_i * batch_size
        end = min(start + batch_size, len(missing_smiles))
        batch = missing_smiles[start:end]

        print(f"\nBatch {batch_i + 1}/{n_batches}  ({start:,}–{end:,})")
        fps, invalid = featurizer._compute_fingerprints(batch, show_progress=True)
        total_invalid += len(invalid)

        # Only store valid entries
        invalid_set = set(invalid)
        valid_idx = [i for i in range(len(batch)) if i not in invalid_set]
        if valid_idx:
            cache.store(
                [batch[i] for i in valid_idx],
                fps[valid_idx],
            )

        elapsed = time.time() - t0
        rate = end / elapsed
        eta = (len(missing_smiles) - end) / rate if rate > 0 else 0
        print(f"  Elapsed: {elapsed:.0f}s | Rate: {rate:.0f} mol/s | ETA: {eta:.0f}s")

    print(f"\nDone. Total invalid SMILES: {total_invalid:,}")
    print(f"Cache size: {cache.count():,} entries")
    print(f"Cache file: {cache.cache_path}  ({cache.cache_path.stat().st_size / 1e6:.1f} MB)")


def precompute_descriptors(registry_path: str, cache_dir: str,
                           batch_size: int = 50_000) -> None:
    """Precompute molecular descriptors for all unique SMILES in the registry."""
    print(f"\n{'='*60}")
    print("Precomputing molecular descriptors")
    print(f"{'='*60}")

    registry = pd.read_csv(registry_path)
    print(f"Registry loaded: {len(registry):,} rows")

    unique_smiles = registry["smiles"].dropna().unique().tolist()
    print(f"Unique SMILES: {len(unique_smiles):,}")

    featurizer = DescriptorFeaturizer()
    config = featurizer.get_config()
    cache = FeatureCache(cache_dir, config)

    n_already = cache.count()
    rows = cache.lookup_rows(unique_smiles)
    missing_smiles = [s for s, m in zip(unique_smiles, rows == -1) if m]
    print(f"To compute: {len(missing_smiles):,} (skipping {n_already:,} cached)")

    if not missing_smiles:
        print("All SMILES already cached. Nothing to do.")
        return

    n_batches = (len(missing_smiles) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_i in range(n_batches):
        start = batch_i * batch_size
        end = min(start + batch_size, len(missing_smiles))
        batch = missing_smiles[start:end]

        print(f"\nBatch {batch_i + 1}/{n_batches}  ({start:,}–{end:,})")
        descs, invalid = featurizer._compute_descriptors(batch, show_progress=True)

        invalid_set = set(invalid)
        valid_idx = [i for i in range(len(batch)) if i not in invalid_set]
        if valid_idx:
            cache.store([batch[i] for i in valid_idx], descs[valid_idx])

        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (len(missing_smiles) - end) / rate if rate > 0 else 0
        print(f"  Elapsed: {elapsed:.0f}s | Rate: {rate:.0f} mol/s | ETA: {eta:.0f}s")

    print(f"\nCache file: {cache.cache_path}  ({cache.cache_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute molecular features and store in cache for reuse across training runs."
    )
    parser.add_argument(
        "--registry",
        default="../../training_data_full/registry.csv",
        help="Path to registry.csv",
    )
    parser.add_argument(
        "--cache-dir",
        default="../../training_data_full/feature_cache",
        help="Directory to store cache files",
    )
    parser.add_argument(
        "--feature-type",
        choices=["morgan", "descriptors", "all"],
        default="all",
        help="Which features to precompute (default: all)",
    )
    parser.add_argument("--radius", type=int, default=2, help="Morgan FP radius")
    parser.add_argument("--n-bits", type=int, default=2048, help="Morgan FP bit count")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Molecules per batch (tune to fit in memory)",
    )
    args = parser.parse_args()

    registry_path = str(Path(args.registry).resolve())
    cache_dir = str(Path(args.cache_dir).resolve())

    print(f"Registry:  {registry_path}")
    print(f"Cache dir: {cache_dir}")

    if args.feature_type in ("morgan", "all"):
        precompute_morgan(
            registry_path, cache_dir,
            radius=args.radius, n_bits=args.n_bits,
            batch_size=args.batch_size,
        )

    if args.feature_type in ("descriptors", "all"):
        precompute_descriptors(registry_path, cache_dir, batch_size=args.batch_size)

    print("\nPrecomputation complete.")


if __name__ == "__main__":
    main()
