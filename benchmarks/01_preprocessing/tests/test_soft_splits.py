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
        df = _make_cluster_df()
        result = assign_soft_partitions(df, cluster_col='cluster_bipartite_qcov_70', seed=42)
        large = result[result['protein_cluster'] == 1]
        for part in ('train', 'val', 'test'):
            assert (large['protein_partition'] == part).any(), \
                f"cluster 1 has no proteins in '{part}'"

    def test_below_min_cluster_all_train(self):
        df = _make_cluster_df()
        result = assign_soft_partitions(df, 'cluster_bipartite_qcov_70',
                                        min_cluster_size=3, seed=42)
        small = result[result['protein_cluster'] == 4]
        assert (small['protein_partition'] == 'train').all()

    def test_singletons_all_train(self):
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
        assert abs(compute_pchembl(10.0) - 8.0) < 1e-6

    def test_1000nm(self):
        assert abs(compute_pchembl(1000.0) - 6.0) < 1e-6

    def test_100000nm(self):
        assert abs(compute_pchembl(100000.0) - 4.0) < 1e-6

    def test_nan_returns_nan(self):
        assert np.isnan(compute_pchembl(float('nan')))

    def test_zero_returns_nan(self):
        assert np.isnan(compute_pchembl(0.0))

    def test_negative_returns_nan(self):
        assert np.isnan(compute_pchembl(-5.0))
