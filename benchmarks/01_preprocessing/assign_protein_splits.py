#!/usr/bin/env python3
"""
Assign protein-level train/val/test partitions to the registry using cluster labels.

This creates a 2D split:
  - Axis 1 (ligand): existing 'split' column (train/test) from ligand Tanimoto similarity
  - Axis 2 (protein): new 'protein_partition' column (train/val/test) from sequence clusters

Algorithm:
  - Singletons and pairs (cluster size < min_cluster_size) → always 'train'
  - Clusters of size >= min_cluster_size are assigned as a whole unit to train/val/test
    using a greedy size-balanced approach targeting the given split ratios.

Usage:
    python assign_protein_splits.py \
        --registry ../../training_data_full/registry.csv \
        --cluster-file ../../data/uniprot_bipartite_cluster_labels.csv \
        --cluster-threshold qcov_70 \
        --min-cluster-size 3 \
        --seed 42 \
        --output ../../training_data_full/registry_2d_split.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def assign_protein_partitions(
    cluster_df: pd.DataFrame,
    cluster_col: str,
    min_cluster_size: int = 3,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign each protein a partition label (train/val/test) based on cluster membership.

    Whole clusters are assigned atomically — no cluster is ever split across partitions.
    Small clusters (size < min_cluster_size) are always assigned to train.

    Args:
        cluster_df: DataFrame with columns ['uniprot', cluster_col]
        cluster_col: Column name to use for cluster IDs
        min_cluster_size: Clusters smaller than this go entirely to train
        split_ratios: (train, val, test) target fractions of *splittable* proteins
        seed: Random seed for reproducibility

    Returns:
        DataFrame with original columns plus 'protein_cluster' and 'protein_partition'
    """
    rng = np.random.default_rng(seed)

    result = cluster_df[['uniprot', cluster_col]].copy()
    result = result.rename(columns={cluster_col: 'protein_cluster'})
    result['protein_partition'] = 'train'  # default

    # Identify splittable clusters
    cluster_sizes = result['protein_cluster'].value_counts()
    splittable = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()

    if not splittable:
        print(f"  Warning: No clusters meet min_cluster_size={min_cluster_size}. "
              f"All proteins assigned to train.")
        return result

    # Shuffle cluster order for reproducibility
    splittable = list(rng.permutation(splittable))

    # Greedy assignment: assign each cluster to the partition most under-target
    train_ratio, val_ratio, test_ratio = split_ratios
    total_splittable = sum(cluster_sizes[c] for c in splittable)

    targets = {
        'train': train_ratio * total_splittable,
        'val':   val_ratio   * total_splittable,
        'test':  test_ratio  * total_splittable,
    }
    counts = {'train': 0, 'val': 0, 'test': 0}
    cluster_assignments: Dict[int, str] = {}

    for cluster_id in splittable:
        size = cluster_sizes[cluster_id]
        # Pick partition with largest deficit (target - current)
        deficit = {p: targets[p] - counts[p] for p in ('train', 'val', 'test')}
        chosen = max(deficit, key=deficit.get)
        cluster_assignments[cluster_id] = chosen
        counts[chosen] += size

    # Apply assignments
    for cluster_id, partition in cluster_assignments.items():
        mask = result['protein_cluster'] == cluster_id
        result.loc[mask, 'protein_partition'] = partition

    # Report
    print(f"\n  Protein partition assignment summary ({cluster_col}):")
    for partition in ('train', 'val', 'test'):
        n = (result['protein_partition'] == partition).sum()
        pct = 100 * n / len(result)
        print(f"    {partition}: {n:,} proteins ({pct:.1f}%)")

    n_splittable_clusters = len(splittable)
    val_clusters  = [c for c, p in cluster_assignments.items() if p == 'val']
    test_clusters = [c for c, p in cluster_assignments.items() if p == 'test']
    train_clusters = [c for c, p in cluster_assignments.items() if p == 'train']
    print(f"\n  Clusters assigned — train: {len(train_clusters)}, "
          f"val: {len(val_clusters)}, test: {len(test_clusters)} "
          f"(of {n_splittable_clusters} splittable; "
          f"{len(cluster_sizes) - n_splittable_clusters} small clusters → train)")

    return result


def build_registry_2d_split(
    registry_path: str,
    cluster_path: str,
    cluster_threshold: str = 'qcov_70',
    min_cluster_size: int = 3,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Load registry + cluster labels, assign protein partitions, and return merged DataFrame.

    Args:
        registry_path: Path to existing registry.csv
        cluster_path: Path to uniprot_bipartite_cluster_labels.csv
        cluster_threshold: Which column to use (qcov_50, qcov_70, qcov_95, qcov_100)
        min_cluster_size: Minimum cluster size to be considered for val/test assignment
        split_ratios: (train, val, test) fractions for splittable proteins
        seed: Random seed
        output_path: If given, save result here

    Returns:
        Registry DataFrame with 'protein_cluster' and 'protein_partition' columns added
    """
    cluster_col = f'cluster_bipartite_{cluster_threshold}'

    print(f"Loading registry: {registry_path}")
    registry = pd.read_csv(registry_path)
    print(f"  {len(registry):,} rows, {registry['uniprot_id'].nunique():,} unique proteins")

    print(f"\nLoading cluster labels: {cluster_path}")
    clusters = pd.read_csv(cluster_path)
    if cluster_col not in clusters.columns:
        available = [c for c in clusters.columns if c.startswith('cluster_')]
        raise ValueError(
            f"Column '{cluster_col}' not found. Available: {available}"
        )
    print(f"  {len(clusters):,} proteins, using column: {cluster_col}")

    # Assign partitions
    protein_partitions = assign_protein_partitions(
        clusters,
        cluster_col=cluster_col,
        min_cluster_size=min_cluster_size,
        split_ratios=split_ratios,
        seed=seed,
    )

    # Proteins in registry but missing from cluster file → assign to train
    registry_uniprots = set(registry['uniprot_id'].unique())
    cluster_uniprots = set(protein_partitions['uniprot'])
    missing = registry_uniprots - cluster_uniprots
    if missing:
        print(f"\n  Warning: {len(missing)} registry proteins not in cluster file → assigned to train")
        extra_rows = pd.DataFrame({
            'uniprot': list(missing),
            'protein_cluster': -1,
            'protein_partition': 'train',
        })
        protein_partitions = pd.concat([protein_partitions, extra_rows], ignore_index=True)

    # Decoys are not target-specific (they have no unique protein label for filtering)
    # They are always included in training, so tag them as 'train'
    merged = registry.merge(
        protein_partitions.rename(columns={'uniprot': 'uniprot_id'}),
        on='uniprot_id',
        how='left',
    )

    # Any remaining NaN (e.g. decoys without uniprot match) → train
    merged['protein_partition'] = merged['protein_partition'].fillna('train')
    merged['protein_cluster'] = merged['protein_cluster'].fillna(-1).astype(int)

    # Print 2D split summary
    print("\n  2D split matrix (row=protein_partition, col=ligand split):")
    actives = merged[merged['is_active'] == True]
    pivot = actives.groupby(['protein_partition', 'split']).size().unstack(fill_value=0)
    print(pivot.to_string())

    print("\n  Recommended usage:")
    print("    Train: protein_partition=train AND split=train")
    print("    Val:   protein_partition=val   AND split=test")
    print("    Test:  protein_partition=test  AND split=test")

    if output_path:
        merged.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Assign protein-level train/val/test splits to the benchmark registry.'
    )
    parser.add_argument('--registry', type=str,
                        default='../../training_data_full/registry.csv',
                        help='Path to registry.csv')
    parser.add_argument('--cluster-file', type=str,
                        default='../../data/uniprot_bipartite_cluster_labels.csv',
                        help='Path to uniprot_bipartite_cluster_labels.csv')
    parser.add_argument('--cluster-threshold', type=str, default='qcov_70',
                        choices=['qcov_50', 'qcov_70', 'qcov_95', 'qcov_100'],
                        help='Protein sequence similarity threshold to use for clustering')
    parser.add_argument('--min-cluster-size', type=int, default=3,
                        help='Minimum cluster size to consider for val/test (smaller → train)')
    parser.add_argument('--split-ratios', type=float, nargs=3,
                        default=[0.70, 0.15, 0.15],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Target fractions for splittable proteins (must sum to 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str,
                        default='../../training_data_full/registry_2d_split.csv',
                        help='Output path for updated registry')
    args = parser.parse_args()

    ratios = tuple(args.split_ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        parser.error(f"--split-ratios must sum to 1.0, got {sum(ratios):.3f}")

    build_registry_2d_split(
        registry_path=args.registry,
        cluster_path=args.cluster_file,
        cluster_threshold=args.cluster_threshold,
        min_cluster_size=args.min_cluster_size,
        split_ratios=ratios,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
