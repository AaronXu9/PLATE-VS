# Soft Protein Partition + Regression Affinity Support

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the all-or-nothing cluster partition strategy with an intra-cluster proportional split, and add a pChEMBL-normalized affinity column to enable regression tasks — stored as a new `registry_soft_split.csv` that is completely independent of the existing `registry_2d_split.csv`.

**Architecture:** A new script `assign_protein_splits_soft.py` generates `registry_soft_split.csv` by sampling proteins *within* each cluster proportionally (70/15/15) rather than assigning whole clusters to a single partition. During generation, a `pchembl` column is computed from the existing `affinity_value` (nM) column via `-log10(value × 1e⁻⁹)`. The `DataLoader.prepare_features_labels()` method gains a `task` parameter so callers can request binary classification labels or continuous pChEMBL regression targets with no API break to existing code.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest; conda env `rdkit_env`

---

## Git Worktree Setup

Work in an isolated worktree so the large generated CSV never touches the main branch until merged.

### Task 0: Create and Enter Git Worktree

**Files:** none (git plumbing only)

- [ ] **Step 0.1: Create the worktree**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset
git worktree add ../VLS-soft-partition -b feature/soft-partition-regression
```

Expected:
```
Preparing worktree (new branch 'feature/soft-partition-regression')
HEAD is now at d7ea227b chore: backup checkpoint...
```

- [ ] **Step 0.2: Verify**

```bash
git worktree list
```

Expected — two lines:
```
/home/aoxu/projects/VLS-Benchmark-Dataset   d7ea227b [feature/ml-benchmarking]
/home/aoxu/projects/VLS-soft-partition      d7ea227b [feature/soft-partition-regression]
```

- [ ] **Step 0.3: All remaining work is in the worktree**

```bash
cd /home/aoxu/projects/VLS-soft-partition
```

---

## Task 1: Soft Partition Core Algorithm

**Files:**
- Create: `benchmarks/01_preprocessing/assign_protein_splits_soft.py`
- Create: `benchmarks/01_preprocessing/tests/__init__.py` (empty)
- Create: `benchmarks/01_preprocessing/tests/test_soft_splits.py`

### Step 1.1: Write the failing tests

```python
# benchmarks/01_preprocessing/tests/test_soft_splits.py
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from assign_protein_splits_soft import assign_soft_partitions, compute_pchembl


def _make_cluster_df():
    """30 proteins across clusters of varying sizes."""
    return pd.DataFrame({
        'uniprot': [f'P{i:05d}' for i in range(30)],
        'cluster_bipartite_qcov_70': (
            [1] * 10 +   # large: 10 → splittable
            [2] * 6  +   # medium: 6 → splittable
            [3] * 3  +   # minimum: 3 → splittable
            [4] * 2  +   # below min: 2 → all train
            [0] * 9      # singletons → all train
        ),
    })


class TestAssignSoftPartitions:
    def test_all_proteins_assigned(self):
        df = _make_cluster_df()
        result = assign_soft_partitions(df, cluster_col='cluster_bipartite_qcov_70')
        assert set(result['uniprot']) == set(df['uniprot'])

    def test_partition_values_valid(self):
        df = _make_cluster_df()
        result = assign_soft_partitions(df, cluster_col='cluster_bipartite_qcov_70')
        assert result['protein_partition'].isin({'train', 'val', 'test'}).all()

    def test_large_cluster_has_all_three_partitions(self):
        """A cluster of 10 must have proteins in train, val, AND test."""
        df = _make_cluster_df()
        result = assign_soft_partitions(df, cluster_col='cluster_bipartite_qcov_70', seed=42)
        large = result[result['protein_cluster'] == 1]
        for part in ('train', 'val', 'test'):
            assert (large['protein_partition'] == part).any(), \
                f"cluster 1 has no proteins in '{part}'"

    def test_below_min_cluster_all_train(self):
        """Cluster of size 2 (< min_cluster_size=3) → all train."""
        df = _make_cluster_df()
        result = assign_soft_partitions(df, 'cluster_bipartite_qcov_70',
                                        min_cluster_size=3, seed=42)
        small = result[result['protein_cluster'] == 4]
        assert (small['protein_partition'] == 'train').all()

    def test_singletons_all_train(self):
        """cluster_id == 0 means singleton → always train."""
        df = _make_cluster_df()
        result = assign_soft_partitions(df, 'cluster_bipartite_qcov_70', seed=42)
        singletons = result[result['protein_cluster'] == 0]
        assert (singletons['protein_partition'] == 'train').all()

    def test_reproducible_with_same_seed(self):
        df = _make_cluster_df()
        r1 = assign_soft_partitions(df, 'cluster_bipartite_qcov_70', seed=42)
        r2 = assign_soft_partitions(df, 'cluster_bipartite_qcov_70', seed=42)
        pd.testing.assert_frame_equal(
            r1.sort_values('uniprot').reset_index(drop=True),
            r2.sort_values('uniprot').reset_index(drop=True),
        )

    def test_different_seeds_differ(self):
        """Different seeds produce different assignments for a large cluster."""
        df = _make_cluster_df()
        r1 = assign_soft_partitions(df, 'cluster_bipartite_qcov_70', seed=42)
        r2 = assign_soft_partitions(df, 'cluster_bipartite_qcov_70', seed=99)
        assert not (r1['protein_partition'].values == r2['protein_partition'].values).all()

    def test_output_columns(self):
        df = _make_cluster_df()
        result = assign_soft_partitions(df, 'cluster_bipartite_qcov_70')
        assert set(result.columns) == {'uniprot', 'protein_cluster', 'protein_partition'}


class TestComputePChEMBL:
    def test_10nm(self):
        """IC50 = 10 nM → pChEMBL = 9 - log10(10) = 8.0"""
        assert abs(compute_pchembl(10.0) - 8.0) < 1e-6

    def test_1000nm(self):
        """IC50 = 1000 nM → pChEMBL = 9 - log10(1000) = 6.0"""
        assert abs(compute_pchembl(1000.0) - 6.0) < 1e-6

    def test_100000nm(self):
        """IC50 = 100000 nM → pChEMBL = 9 - log10(100000) = 4.0"""
        assert abs(compute_pchembl(100000.0) - 4.0) < 1e-6

    def test_nan_returns_nan(self):
        assert np.isnan(compute_pchembl(float('nan')))

    def test_zero_returns_nan(self):
        assert np.isnan(compute_pchembl(0.0))

    def test_negative_returns_nan(self):
        assert np.isnan(compute_pchembl(-5.0))
```

### Step 1.2: Run tests to confirm they fail

```bash
conda run -n rdkit_env pytest benchmarks/01_preprocessing/tests/test_soft_splits.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'assign_protein_splits_soft'`

### Step 1.3: Implement `assign_protein_splits_soft.py`

```python
#!/usr/bin/env python3
"""
Soft protein partition: splits proteins *within* each cluster proportionally
(train/val/test) instead of assigning entire clusters to one partition.

This creates registry_soft_split.csv alongside the existing registry_2d_split.csv.

Key difference from assign_protein_splits.py (hard split):
  Hard:  Cluster A (10 proteins) → ALL go to test  (model sees no similar proteins)
  Soft:  Cluster A (10 proteins) → 7 train, 2 val, 1 test  (model sees related proteins)

Also adds 'pchembl' column: -log10(affinity_value_nM * 1e-9) for regression tasks.

Usage:
    python assign_protein_splits_soft.py \\
        --registry ../../training_data_full/registry.csv \\
        --cluster-file ../../data/uniprot_bipartite_cluster_labels.csv \\
        --output ../../training_data_full/registry_soft_split.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def compute_pchembl(affinity_value_nm) -> float:
    """Convert affinity (nM) to pChEMBL = -log10(value_in_molar).

    pChEMBL = -log10(nM * 1e-9) = 9 - log10(nM)

    Returns np.nan for missing, zero, or negative values.
    """
    if pd.isna(affinity_value_nm) or float(affinity_value_nm) <= 0:
        return np.nan
    return -np.log10(float(affinity_value_nm) * 1e-9)


def assign_soft_partitions(
    cluster_df: pd.DataFrame,
    cluster_col: str = 'cluster_bipartite_qcov_70',
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    min_cluster_size: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Intra-cluster stratified protein split.

    For each cluster with >= min_cluster_size proteins, proteins are
    proportionally distributed across train/val/test by random sampling
    within the cluster. Proteins in small clusters (cluster_id == 0 or
    size < min_cluster_size) go to train.

    Args:
        cluster_df: DataFrame with columns ['uniprot', cluster_col]
        cluster_col: Cluster ID column to use
        split_ratios: (train, val, test) fractions, must sum to 1.0
        min_cluster_size: Clusters below this size → all to train
        seed: Random seed

    Returns:
        DataFrame with columns ['uniprot', 'protein_cluster', 'protein_partition']
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"
    rng = np.random.default_rng(seed)
    records = []

    for cluster_id, group in cluster_df.groupby(cluster_col):
        proteins = list(rng.permutation(group['uniprot'].tolist()))
        n = len(proteins)
        cid = int(cluster_id)

        if cid == 0 or n < min_cluster_size:
            for p in proteins:
                records.append({'uniprot': p, 'protein_cluster': cid,
                                 'protein_partition': 'train'})
            continue

        n_val = max(1, round(n * split_ratios[1]))
        n_test = max(1, round(n * split_ratios[2]))
        n_train = n - n_val - n_test

        if n_train <= 0:
            # Cluster passes min_cluster_size but is too small for 3-way split
            for p in proteins:
                records.append({'uniprot': p, 'protein_cluster': cid,
                                 'protein_partition': 'train'})
            continue

        for p in proteins[:n_train]:
            records.append({'uniprot': p, 'protein_cluster': cid,
                             'protein_partition': 'train'})
        for p in proteins[n_train:n_train + n_val]:
            records.append({'uniprot': p, 'protein_cluster': cid,
                             'protein_partition': 'val'})
        for p in proteins[n_train + n_val:]:
            records.append({'uniprot': p, 'protein_cluster': cid,
                             'protein_partition': 'test'})

    result = pd.DataFrame(records)
    logger.info("Soft partition distribution:\n%s",
                result['protein_partition'].value_counts().to_string())
    return result


def build_soft_split_registry(
    registry_path: str,
    cluster_file: str,
    cluster_threshold: str = 'qcov_70',
    min_cluster_size: int = 3,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    """Merge registry with soft protein partition assignments and pchembl column.

    Reads registry.csv, runs intra-cluster protein split, adds pchembl column
    from affinity_value (nM), and saves registry_soft_split.csv.

    Args:
        registry_path: Path to registry.csv
        cluster_file: Path to uniprot_bipartite_cluster_labels.csv
        cluster_threshold: Which column to use (qcov_50, qcov_70, qcov_95, qcov_100)
        min_cluster_size: Clusters smaller than this → all proteins go to train
        split_ratios: (train, val, test) intra-cluster sampling fractions
        seed: Random seed
        output_path: Output file path; defaults to registry_soft_split.csv next to registry

    Returns:
        Merged DataFrame with columns protein_cluster, protein_partition, pchembl added
    """
    cluster_col = f'cluster_bipartite_{cluster_threshold}'

    logger.info("Loading registry: %s", registry_path)
    registry = pd.read_csv(registry_path)
    logger.info("Registry: %d rows, %d unique proteins",
                len(registry), registry['uniprot_id'].nunique())

    logger.info("Loading cluster labels: %s (column: %s)", cluster_file, cluster_col)
    clusters = pd.read_csv(cluster_file)
    if cluster_col not in clusters.columns:
        available = [c for c in clusters.columns if c.startswith('cluster_')]
        raise ValueError(f"Column '{cluster_col}' not found. Available: {available}")

    logger.info("Assigning soft partitions ...")
    partition_df = assign_soft_partitions(
        clusters[['uniprot', cluster_col]],
        cluster_col=cluster_col,
        split_ratios=split_ratios,
        min_cluster_size=min_cluster_size,
        seed=seed,
    )

    # Handle proteins in registry but missing from cluster file → train
    registry_uniprots = set(registry['uniprot_id'].unique())
    cluster_uniprots = set(partition_df['uniprot'])
    missing = registry_uniprots - cluster_uniprots
    if missing:
        logger.warning("%d registry proteins not in cluster file → assigned to train",
                       len(missing))
        extra = pd.DataFrame({
            'uniprot': list(missing),
            'protein_cluster': -1,
            'protein_partition': 'train',
        })
        partition_df = pd.concat([partition_df, extra], ignore_index=True)

    # Merge into registry
    result = registry.merge(
        partition_df.rename(columns={'uniprot': 'uniprot_id'}),
        on='uniprot_id',
        how='left',
    )
    result['protein_partition'] = result['protein_partition'].fillna('train')
    result['protein_cluster'] = result['protein_cluster'].fillna(-1).astype(int)

    # Decoys are not protein-specific: always train, cluster -1
    decoy_mask = result['source'] == 'deepcoy'
    result.loc[decoy_mask, 'protein_partition'] = 'train'
    result.loc[decoy_mask, 'protein_cluster'] = -1

    # Add pChEMBL column
    logger.info("Computing pChEMBL from affinity_value (nM) ...")
    result['pchembl'] = result['affinity_value'].apply(compute_pchembl)
    n_valid = result['pchembl'].notna().sum()
    logger.info("Valid pChEMBL values: %d / %d (%.1f%%)",
                n_valid, len(result), 100 * n_valid / len(result))

    # Print 2D matrix for verification
    actives = result[result['is_active'] == True]
    matrix = pd.crosstab(actives['protein_partition'], actives['split'],
                          margins=True, margins_name='Total')
    logger.info("2D split matrix (actives only):\n%s", matrix.to_string())
    logger.info("Recommended usage:")
    logger.info("  Train: protein_partition=train AND split=train")
    logger.info("  Val:   protein_partition=val   AND split=test  (novel ligands, related protein)")
    logger.info("  Test:  protein_partition=test  AND split=test  (novel ligands, related protein)")

    if output_path is None:
        output_path = str(Path(registry_path).parent / 'registry_soft_split.csv')

    logger.info("Saving to: %s", output_path)
    result.to_csv(output_path, index=False)
    logger.info("Done. Output shape: %s", result.shape)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Build registry_soft_split.csv with intra-cluster protein split + pchembl'
    )
    parser.add_argument('--registry', default='../../training_data_full/registry.csv')
    parser.add_argument('--cluster-file',
                        default='../../data/uniprot_bipartite_cluster_labels.csv')
    parser.add_argument('--cluster-threshold', default='qcov_70',
                        choices=['qcov_50', 'qcov_70', 'qcov_95', 'qcov_100'])
    parser.add_argument('--min-cluster-size', type=int, default=3)
    parser.add_argument('--split-ratios', nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None)
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
```

### Step 1.4: Run tests — confirm all pass

```bash
conda run -n rdkit_env pytest benchmarks/01_preprocessing/tests/test_soft_splits.py -v
```

Expected:
```
PASSED test_soft_splits.py::TestAssignSoftPartitions::test_all_proteins_assigned
PASSED test_soft_splits.py::TestAssignSoftPartitions::test_partition_values_valid
PASSED test_soft_splits.py::TestAssignSoftPartitions::test_large_cluster_has_all_three_partitions
PASSED test_soft_splits.py::TestAssignSoftPartitions::test_below_min_cluster_all_train
PASSED test_soft_splits.py::TestAssignSoftPartitions::test_singletons_all_train
PASSED test_soft_splits.py::TestAssignSoftPartitions::test_reproducible_with_same_seed
PASSED test_soft_splits.py::TestAssignSoftPartitions::test_different_seeds_differ
PASSED test_soft_splits.py::TestAssignSoftPartitions::test_output_columns
PASSED test_soft_splits.py::TestComputePChEMBL::test_10nm
PASSED test_soft_splits.py::TestComputePChEMBL::test_1000nm
PASSED test_soft_splits.py::TestComputePChEMBL::test_100000nm
PASSED test_soft_splits.py::TestComputePChEMBL::test_nan_returns_nan
PASSED test_soft_splits.py::TestComputePChEMBL::test_zero_returns_nan
PASSED test_soft_splits.py::TestComputePChEMBL::test_negative_returns_nan
14 passed
```

### Step 1.5: Commit

```bash
git add benchmarks/01_preprocessing/assign_protein_splits_soft.py \
        benchmarks/01_preprocessing/tests/__init__.py \
        benchmarks/01_preprocessing/tests/test_soft_splits.py
git commit -m "feat: add intra-cluster soft protein partition + pChEMBL computation"
```

---

## Task 2: Generate `registry_soft_split.csv`

**Files:**
- Generated (not committed): `training_data_full/registry_soft_split.csv`

### Step 2.1: Verify inputs exist

```bash
ls -lh training_data_full/registry.csv data/uniprot_bipartite_cluster_labels.csv
```

Expected: both files present. `registry.csv` should be ~319 MB.

### Step 2.2: Run the script

```bash
conda run -n rdkit_env python benchmarks/01_preprocessing/assign_protein_splits_soft.py \
    --registry training_data_full/registry.csv \
    --cluster-file data/uniprot_bipartite_cluster_labels.csv \
    --cluster-threshold qcov_70 \
    --min-cluster-size 3 \
    --split-ratios 0.70 0.15 0.15 \
    --seed 42 \
    --output training_data_full/registry_soft_split.csv
```

Expected log lines to appear:
```
INFO Soft partition distribution:
train    XXXXX
val      XXXXX
test     XXXXX
INFO Valid pChEMBL values: XXXXX / 2886090
INFO 2D split matrix (actives only):
...
INFO Done. Output shape: (2886090, 19)
```

### Step 2.3: Validate output structure

```bash
conda run -n rdkit_env python -c "
import pandas as pd, numpy as np
df = pd.read_csv('training_data_full/registry_soft_split.csv')
print('Shape:', df.shape)  # expect (2886090, 19)
print('Columns added:', [c for c in df.columns if c not in
    ['sample_id','uniprot_id','pdb_id','compound_id','cif_path','resolution',
     'quality_score','smiles','sdf_path','pkl_path','is_active','affinity_value',
     'affinity_type','similarity_threshold','split','source']])
# expect: ['protein_cluster', 'protein_partition', 'pchembl']

print('protein_partition counts:')
print(df['protein_partition'].value_counts())

print('pchembl describe:')
print(df['pchembl'].describe())

# Spot-check: each uniprot_id has exactly one protein_partition
actives = df[df['is_active'] == True]
multi = actives.groupby('uniprot_id')['protein_partition'].nunique()
multi_count = (multi > 1).sum()
print(f'Proteins with multiple partitions: {multi_count}')  # expect 0
assert multi_count == 0, 'Data integrity failure: protein in multiple partitions'
print('PASS: partition integrity OK')
"
```

Expected:
- Shape `(2886090, 19)`
- `protein_partition` has values train/val/test
- `pchembl` range roughly 4.0–12.0 for valid entries
- `Proteins with multiple partitions: 0`
- `PASS: partition integrity OK`

### Step 2.4: Verify the file is gitignored (training_data_full/ already ignored)

```bash
git check-ignore -v training_data_full/registry_soft_split.csv
```

Expected: prints the .gitignore rule. If nothing is printed, add the line:

```bash
echo "training_data_full/registry_soft_split.csv" >> .gitignore
git add .gitignore
git commit -m "chore: ignore registry_soft_split.csv"
```

---

## Task 3: Regression Support in DataLoader

**Files:**
- Modify: `benchmarks/02_training/data/data_loader.py` — update `prepare_features_labels()`
- Create: `benchmarks/02_training/test_regression_loader.py`

### Step 3.1: Write the failing tests

```python
# benchmarks/02_training/test_regression_loader.py
"""Tests for regression mode in DataLoader.prepare_features_labels()."""
import sys
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.data_loader import DataLoader


def _make_registry_csv(tmp_dir: str, include_pchembl: bool = True) -> str:
    """Build a minimal registry CSV for testing."""
    n = 20
    data = {
        'sample_id': [f'P{i:05d}__0p7_train' for i in range(n)],
        'uniprot_id': [f'P{i:05d}' for i in range(n)],
        'pdb_id': ['1ABC'] * n,
        'compound_id': [f'CMP{i}' for i in range(n)],
        'cif_path': [''] * n,
        'resolution': [2.0] * n,
        'quality_score': [300.0] * n,
        'smiles': ['c1ccccc1'] * n,
        'sdf_path': [''] * n,
        'pkl_path': [''] * n,
        'is_active': [True] * 10 + [False] * 10,
        'affinity_value': [10.0, 100.0, 1000.0, float('nan'), 50.0,
                           200.0, 5.0, 20.0, 500.0, float('nan')] + [float('nan')] * 10,
        'affinity_type': ['IC50'] * 10 + [''] * 10,
        'similarity_threshold': ['0p7'] * n,
        'split': ['train'] * 12 + ['test'] * 8,
        'source': ['chembl'] * 10 + ['deepcoy'] * 10,
        'protein_cluster': [1] * 10 + [-1] * 10,
        'protein_partition': ['train'] * 14 + ['val'] * 3 + ['test'] * 3,
    }
    if include_pchembl:
        # pchembl = 9 - log10(affinity_value): only non-NaN actives get values
        data['pchembl'] = [8.0, 7.0, 6.0, float('nan'), 7.3,
                           6.7, 8.3, 7.7, 6.3, float('nan')] + [float('nan')] * 10
    path = os.path.join(tmp_dir, 'registry_soft_split.csv')
    pd.DataFrame(data).to_csv(path, index=False)
    return path


class TestClassificationModeUnchanged:
    def test_binary_labels_returned(self, tmp_path):
        path = _make_registry_csv(str(tmp_path))
        loader = DataLoader(path)
        loader.load_registry()
        data = loader.get_training_data(split='train')
        smiles, labels = loader.prepare_features_labels(data, task='classification')
        assert set(labels.tolist()).issubset({0, 1})

    def test_default_task_is_classification(self, tmp_path):
        path = _make_registry_csv(str(tmp_path))
        loader = DataLoader(path)
        loader.load_registry()
        data = loader.get_training_data(split='train')
        smiles, labels = loader.prepare_features_labels(data)  # no task arg
        assert set(labels.tolist()).issubset({0, 1})

    def test_include_protein_info_still_works(self, tmp_path):
        path = _make_registry_csv(str(tmp_path))
        loader = DataLoader(path)
        loader.load_registry()
        data = loader.get_training_data(split='train')
        smiles, labels, pids = loader.prepare_features_labels(
            data, task='classification', include_protein_info=True)
        assert len(pids) == len(smiles)


class TestRegressionMode:
    def test_returns_float_targets(self, tmp_path):
        path = _make_registry_csv(str(tmp_path))
        loader = DataLoader(path)
        loader.load_registry()
        data = loader.get_training_data(split='train')
        smiles, labels = loader.prepare_features_labels(data, task='regression')
        assert labels.dtype in (np.float32, np.float64)
        assert not np.isin(labels, [0, 1]).all(), "regression labels must not be binary"

    def test_nan_pchembl_rows_excluded(self, tmp_path):
        path = _make_registry_csv(str(tmp_path))
        loader = DataLoader(path)
        loader.load_registry()
        data = loader.get_training_data(split='train')
        smiles, labels = loader.prepare_features_labels(data, task='regression')
        assert not np.any(np.isnan(labels)), "NaN targets must be filtered out"

    def test_regression_values_match_pchembl_column(self, tmp_path):
        path = _make_registry_csv(str(tmp_path))
        loader = DataLoader(path)
        loader.load_registry()
        data = loader.get_training_data(split='train')
        smiles, labels = loader.prepare_features_labels(data, task='regression')
        # All returned values must be from the known fixture set
        expected = {8.0, 7.0, 6.0, 7.3, 6.7, 8.3, 7.7, 6.3}
        for val in labels:
            assert any(abs(val - e) < 0.01 for e in expected), \
                f"Unexpected pchembl value: {val}"

    def test_include_protein_info_regression(self, tmp_path):
        path = _make_registry_csv(str(tmp_path))
        loader = DataLoader(path)
        loader.load_registry()
        data = loader.get_training_data(split='train')
        smiles, labels, pids = loader.prepare_features_labels(
            data, task='regression', include_protein_info=True)
        assert len(pids) == len(smiles) == len(labels)

    def test_missing_pchembl_column_raises_valueerror(self, tmp_path):
        path = _make_registry_csv(str(tmp_path), include_pchembl=False)
        loader = DataLoader(path)
        loader.load_registry()
        data = loader.get_training_data(split='train')
        with pytest.raises(ValueError, match="pchembl"):
            loader.prepare_features_labels(data, task='regression')

    def test_unknown_task_raises_valueerror(self, tmp_path):
        path = _make_registry_csv(str(tmp_path))
        loader = DataLoader(path)
        loader.load_registry()
        data = loader.get_training_data(split='train')
        with pytest.raises(ValueError, match="task"):
            loader.prepare_features_labels(data, task='invalid_task')
```

### Step 3.2: Run tests — confirm they fail

```bash
conda run -n rdkit_env pytest benchmarks/02_training/test_regression_loader.py -v 2>&1 | head -25
```

Expected: `TypeError: prepare_features_labels() got an unexpected keyword argument 'task'`

### Step 3.3: Update `prepare_features_labels()` in `data/data_loader.py`

Locate lines 146–192 (the `prepare_features_labels` method) and replace with:

```python
def prepare_features_labels(self,
                             data: pd.DataFrame,
                             smiles_column: str = 'smiles',
                             label_column: str = 'is_active',
                             task: str = 'classification',
                             include_protein_info: bool = False,
                             protein_id_column: str = 'uniprot_id') -> tuple:
    """
    Extract SMILES, labels, and optionally protein IDs from the data.

    Args:
        data: DataFrame from get_training_data()
        smiles_column: Column containing SMILES strings
        label_column: Column for binary labels (used only when task='classification')
        task: 'classification' returns binary int labels from label_column;
              'regression' returns float pchembl values (rows with NaN pchembl excluded)
        include_protein_info: Whether to also return protein identifier list
        protein_id_column: Column containing protein UniProt IDs

    Returns:
        (smiles_list, labels_array) or (smiles_list, labels_array, protein_ids_list)
    """
    if smiles_column not in data.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in data")

    if task == 'regression':
        if 'pchembl' not in data.columns:
            raise ValueError(
                "Registry does not have a 'pchembl' column. "
                "Run assign_protein_splits_soft.py to generate registry_soft_split.csv first."
            )
        data = data[data['pchembl'].notna()].copy()
        labels = data['pchembl'].values.astype(np.float32)
        print(f"Regression mode: {len(labels)} samples with valid pChEMBL values")
        print(f"  pChEMBL range: {labels.min():.2f} – {labels.max():.2f} "
              f"(mean {labels.mean():.2f})")
    elif task == 'classification':
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
        labels = data[label_column].astype(int).values
        print(f"Prepared {len(labels)} samples with labels")
        print(f"  Active compounds: {np.sum(labels == 1)}")
        print(f"  Inactive compounds: {np.sum(labels == 0)}")
        print(f"  Class balance: {np.mean(labels):.2%} active")
    else:
        raise ValueError(
            f"Unknown task='{task}'. Choose 'classification' or 'regression'."
        )

    smiles = data[smiles_column].tolist()

    if include_protein_info:
        if protein_id_column not in data.columns:
            raise ValueError(f"Protein ID column '{protein_id_column}' not found in data")
        protein_ids = data[protein_id_column].tolist()
        print(f"  Unique proteins: {len(set(protein_ids))}")
        return smiles, labels, protein_ids

    return smiles, labels
```

### Step 3.4: Run all tests — confirm they pass

```bash
conda run -n rdkit_env pytest \
    benchmarks/01_preprocessing/tests/test_soft_splits.py \
    benchmarks/02_training/test_regression_loader.py \
    -v --tb=short
```

Expected: All tests pass. Verify especially `TestClassificationModeUnchanged` passes — this confirms no regression in existing behavior.

### Step 3.5: Commit

```bash
git add benchmarks/02_training/data/data_loader.py \
        benchmarks/02_training/test_regression_loader.py
git commit -m "feat: add task='regression' to prepare_features_labels for pChEMBL targets"
```

---

## Task 4: Config for Soft Split Training

**Files:**
- Create: `benchmarks/configs/soft_split_config.yaml`

### Step 4.1: Create config

```yaml
# benchmarks/configs/soft_split_config.yaml
# Training config for soft protein partition strategy.
#
# Run (classification):
#   python benchmarks/02_training/train_classical_oddt.py \
#     --config benchmarks/configs/soft_split_config.yaml \
#     --registry training_data_full/registry_soft_split.csv \
#     --use-2d-split
#
# Key difference from classical_config.yaml:
#   - registry_soft_split.csv: proteins split WITHIN clusters (softer generalization test)
#   - pchembl column available for regression (set data.task: regression below)

model_type: "random_forest"

hyperparameters:
  n_estimators: 100
  max_depth: null
  max_features: 'sqrt'
  class_weight: 'balanced'
  random_state: 42
  n_jobs: -1

features:
  type: "combined"
  ligand:
    type: "morgan_fingerprint"
    radius: 2
    n_bits: 2048
  protein:
    type: "protein_identifier"
    embedding_dim: 32

data:
  similarity_threshold: '0p7'
  include_decoys: true
  include_protein_features: true
  use_2d_split: true          # use protein_partition column from registry_soft_split.csv
  task: 'classification'      # change to 'regression' to predict pChEMBL values
  protein_cluster_threshold: qcov_70
  min_cluster_size_for_split: 3
  protein_split_ratios: [0.70, 0.15, 0.15]

metrics:
  - accuracy
  - precision
  - recall
  - f1_score
  - roc_auc
```

### Step 4.2: Commit

```bash
git add benchmarks/configs/soft_split_config.yaml
git commit -m "feat: add soft_split_config.yaml for intra-cluster partition training"
```

---

## Task 5: End-to-End Smoke Test

Validates the full pipeline works with the soft split registry.

### Step 5.1: Quick-test run (1000 samples)

```bash
conda run -n rdkit_env python benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/soft_split_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split \
    --quick-test \
    --test-samples 1000 \
    --output-dir benchmarks/02_training/trained_models_soft_test \
    2>&1 | tail -30
```

Expected — look for these lines (values will vary):
```
Filtered data: ... samples for split='train', threshold='0p7'...
Val ROC-AUC: 0.XX  (should be > 0.5; typically higher than hard-split test AUC)
Test ROC-AUC: 0.XX
```

If the training script doesn't have `--use-2d-split` as a flag, check its argument parser and use the correct flag. The script must pass `protein_partition='train'/'val'/'test'` to `get_training_data()` internally when the registry has a `protein_partition` column.

### Step 5.2: Commit the JSON summary (not the pkl)

```bash
ls benchmarks/02_training/trained_models_soft_test/
git add benchmarks/02_training/trained_models_soft_test/random_forest_training_summary.json \
        benchmarks/02_training/trained_models_soft_test/random_forest_config.json
git commit -m "test: smoke test results for soft-split training (quick-test 1000 samples)"
```

---

## Task 6: Final Check and Push

### Step 6.1: Run full test suite

```bash
conda run -n rdkit_env pytest \
    benchmarks/01_preprocessing/tests/ \
    benchmarks/02_training/test_regression_loader.py \
    -v --tb=short
```

Expected: All tests pass.

### Step 6.2: Push branch

```bash
git push -u origin feature/soft-partition-regression
```

---

## Summary: Hard vs. Soft Split Comparison

| | Hard split (`registry_2d_split.csv`) | Soft split (`registry_soft_split.csv`) |
|---|---|---|
| **Strategy** | Whole clusters → one partition | Proteins sampled within clusters |
| **Train sees cluster** | No (test proteins from unseen clusters) | Yes (70% of every cluster) |
| **Generalization test** | Extremely hard (novel families) | Moderate (held-out proteins from known families) |
| **Use case** | Final generalization benchmark | Day-to-day model development/tuning |
| **pchembl column** | No | Yes (`-log10(nM * 1e-9)`) |
| **Regression support** | No | Yes (`task='regression'`) |

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Soft partition strategy: intra-cluster proportional split in `assign_protein_splits_soft.py`
- ✅ Separate from existing hard split: new `registry_soft_split.csv`, existing `registry_2d_split.csv` untouched
- ✅ Numerical affinity values preserved: `pchembl` column added
- ✅ pChEMBL, Kd, IC50 all handled: `compute_pchembl(nM)` is affinity-type agnostic
- ✅ Regression support: `prepare_features_labels(task='regression')` returns float pChEMBL
- ✅ Git worktree: Task 0 creates `feature/soft-partition-regression` worktree
- ✅ TDD: Tests written before implementation in Tasks 1 and 3

**Placeholder scan:** None found.

**Type consistency:**
- `assign_soft_partitions()` → `pd.DataFrame` with `['uniprot', 'protein_cluster', 'protein_partition']` — used identically in Tasks 1 and 2
- `compute_pchembl(float) → float` — used identically in Tasks 1 and 2
- `prepare_features_labels(data, task='regression')` → `(list, np.ndarray)` — used identically in Tasks 3 and 5
