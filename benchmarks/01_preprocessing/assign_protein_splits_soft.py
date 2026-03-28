#!/usr/bin/env python3
"""
Soft protein partition: assigns proteins to train/val/test *within* each cluster
(proportional sampling), rather than assigning whole clusters to a single partition.

This creates a 2D split:
  - Axis 1 (ligand): existing 'split' column (train/test) from ligand Tanimoto similarity
  - Axis 2 (protein): new 'protein_partition' column (train/val/test) from intra-cluster split

Algorithm:
  - cluster_id == 0 or size < min_cluster_size → all proteins get partition 'train'
  - otherwise: shuffle proteins within cluster, then assign proportionally to
    train / val / test using max(1, round(n * ratio)) for val and test

Usage:
    python assign_protein_splits_soft.py \
        --registry ../../training_data_full/registry.csv \
        --cluster-file ../../data/uniprot_bipartite_cluster_labels.csv \
        --cluster-threshold qcov_70 \
        --min-cluster-size 3 \
        --seed 42 \
        --output ../../training_data_full/registry_soft_split.csv
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def compute_pchembl(affinity_value_nm) -> float:
    """
    Convert affinity in nM to pChEMBL = -log10(value_nm * 1e-9).

    Returns np.nan for NaN, zero, or negative inputs.

    Examples:
        10 nM  → 8.0
        1000 nM → 6.0
    """
    try:
        val = float(affinity_value_nm)
    except (TypeError, ValueError):
        return np.nan

    if math.isnan(val) or val <= 0:
        return np.nan

    return -math.log10(val * 1e-9)


def assign_soft_partitions(
    cluster_df: pd.DataFrame,
    cluster_col: str,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    min_cluster_size: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign each protein a partition label (train/val/test) by sampling *within* clusters.

    Unlike the hard split (which assigns whole clusters to one partition), this function
    distributes proteins from each cluster proportionally across train/val/test.

    Rules:
      - cluster_id == 0 or cluster size < min_cluster_size → all proteins go to 'train'
      - otherwise: shuffle proteins (seeded), then:
          n_val  = max(1, round(n * split_ratios[1]))
          n_test = max(1, round(n * split_ratios[2]))
          n_train = n - n_val - n_test
          if n_train <= 0: all proteins go to 'train'

    Args:
        cluster_df: DataFrame with columns ['uniprot', cluster_col]
        cluster_col: Column name containing cluster IDs
        split_ratios: (train, val, test) target fractions
        min_cluster_size: Clusters smaller than this (and cluster_id==0) → all train
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns ['uniprot', 'protein_cluster', 'protein_partition']
    """
    rng = np.random.default_rng(seed)

    result = cluster_df[['uniprot', cluster_col]].copy()
    result = result.rename(columns={cluster_col: 'protein_cluster'})
    result['protein_partition'] = 'train'  # default

    _, val_ratio, test_ratio = split_ratios

    groups = result.groupby('protein_cluster')

    for cluster_id, group in groups:
        n = len(group)

        # cluster_id == 0 or too small → all train
        if cluster_id == 0 or n < min_cluster_size:
            continue  # already 'train' by default

        # Shuffle proteins within this cluster
        shuffled_idx = rng.permutation(group.index.to_numpy())

        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test

        if n_train <= 0:
            # Too small after reserving val+test: keep all as train
            continue

        # Assign: first n_train → train, next n_val → val, rest → test
        train_idx = shuffled_idx[:n_train]
        val_idx = shuffled_idx[n_train:n_train + n_val]
        test_idx = shuffled_idx[n_train + n_val:]

        result.loc[val_idx, 'protein_partition'] = 'val'
        result.loc[test_idx, 'protein_partition'] = 'test'
        # train_idx already 'train'

    # Log distribution
    counts = result['protein_partition'].value_counts()
    total = len(result)
    logging.info("Protein partition distribution:")
    for part in ('train', 'val', 'test'):
        n = counts.get(part, 0)
        logging.info("  %-6s %6d  (%.1f%%)", part, n, 100 * n / total)

    return result[['uniprot', 'protein_cluster', 'protein_partition']]


def build_soft_split_registry(
    registry_path: str,
    cluster_file: str,
    cluster_threshold: str = 'qcov_70',
    min_cluster_size: int = 3,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Load registry + cluster labels, assign soft protein partitions, and return merged DataFrame.

    Args:
        registry_path: Path to existing registry.csv
        cluster_file: Path to uniprot_bipartite_cluster_labels.csv
        cluster_threshold: Which column to use (qcov_50, qcov_70, qcov_95, qcov_100)
        min_cluster_size: Minimum cluster size to consider for val/test splits
        split_ratios: (train, val, test) fractions for intra-cluster split
        seed: Random seed
        output_path: If None, saves to registry_path's parent / 'registry_soft_split.csv'

    Returns:
        Registry DataFrame with 'protein_cluster', 'protein_partition', and 'pchembl' added
    """
    cluster_col = f'cluster_bipartite_{cluster_threshold}'

    logging.info("Loading registry: %s", registry_path)
    registry = pd.read_csv(registry_path)
    logging.info("  %d rows, %d unique proteins",
                 len(registry), registry['uniprot_id'].nunique())

    logging.info("Loading cluster labels: %s", cluster_file)
    clusters = pd.read_csv(cluster_file)
    if cluster_col not in clusters.columns:
        available = [c for c in clusters.columns if c.startswith('cluster_')]
        raise ValueError(f"Column '{cluster_col}' not found. Available: {available}")
    logging.info("  %d proteins, using column: %s", len(clusters), cluster_col)

    # Assign soft partitions
    partition_df = assign_soft_partitions(
        clusters[['uniprot', cluster_col]],
        cluster_col=cluster_col,
        split_ratios=split_ratios,
        min_cluster_size=min_cluster_size,
        seed=seed,
    )

    # Handle proteins in registry but missing from cluster file
    registry_uniprots = set(registry['uniprot_id'].unique())
    cluster_uniprots = set(partition_df['uniprot'])
    missing = registry_uniprots - cluster_uniprots
    if missing:
        logging.info("  Warning: %d registry proteins not in cluster file → assigned to train",
                     len(missing))
        extra_rows = pd.DataFrame({
            'uniprot': list(missing),
            'protein_cluster': -1,
            'protein_partition': 'train',
        })
        partition_df = pd.concat([partition_df, extra_rows], ignore_index=True)

    # Merge partition info into registry
    merged = registry.merge(
        partition_df.rename(columns={'uniprot': 'uniprot_id'}),
        on='uniprot_id',
        how='left',
    )

    # Fill any remaining NaN (e.g. decoys without uniprot match)
    merged['protein_partition'] = merged['protein_partition'].fillna('train')
    merged['protein_cluster'] = merged['protein_cluster'].fillna(-1).astype(int)

    # Decoys (source == 'deepcoy') always go to train
    decoy_mask = merged['source'] == 'deepcoy'
    merged.loc[decoy_mask, 'protein_partition'] = 'train'
    merged.loc[decoy_mask, 'protein_cluster'] = -1

    # Compute pChEMBL
    merged['pchembl'] = merged['affinity_value'].apply(compute_pchembl)

    # Log 2D split matrix for actives
    actives = merged[merged['is_active'] == True]
    logging.info("2D split matrix (row=protein_partition, col=ligand split):")
    crosstab = pd.crosstab(actives['protein_partition'], actives['split'])
    for line in crosstab.to_string().split('\n'):
        logging.info("  %s", line)

    if output_path is None:
        output_path = Path(registry_path).parent / 'registry_soft_split.csv'

    merged.to_csv(output_path, index=False)
    logging.info("Saved to: %s", output_path)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Assign soft intra-cluster protein train/val/test splits to the registry.'
    )
    parser.add_argument('--registry', type=str, required=True,
                        help='Path to registry.csv')
    parser.add_argument('--cluster-file', type=str, required=True,
                        help='Path to uniprot_bipartite_cluster_labels.csv')
    parser.add_argument('--cluster-threshold', type=str, default='qcov_70',
                        choices=['qcov_50', 'qcov_70', 'qcov_95', 'qcov_100'],
                        help='Protein sequence similarity threshold to use for clustering')
    parser.add_argument('--min-cluster-size', type=int, default=3,
                        help='Minimum cluster size to consider for val/test (smaller → train)')
    parser.add_argument('--split-ratios', type=float, nargs=3,
                        default=[0.70, 0.15, 0.15],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Target fractions for intra-cluster split (must sum to 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: registry_soft_split.csv next to registry)')
    args = parser.parse_args()

    ratios = tuple(args.split_ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        parser.error(f"--split-ratios must sum to 1.0, got {sum(ratios):.3f}")

    build_soft_split_registry(
        registry_path=args.registry,
        cluster_file=args.cluster_file,
        cluster_threshold=args.cluster_threshold,
        min_cluster_size=args.min_cluster_size,
        split_ratios=ratios,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
