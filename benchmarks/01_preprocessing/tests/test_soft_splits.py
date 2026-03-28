import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from assign_protein_splits_soft import assign_soft_partitions, compute_pchembl, build_soft_split_registry


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

    def test_ntrain_zero_fallback_all_train(self):
        """If split ratios leave n_train <= 0, all proteins in that cluster go to train."""
        # Cluster of size 3 with ratios (0.0, 0.5, 0.5) → n_val=2, n_test=2, n_train=-1 → all train
        df = pd.DataFrame({
            'uniprot': ['A', 'B', 'C'],
            'cluster_bipartite_qcov_70': [5, 5, 5],
        })
        result = assign_soft_partitions(df, 'cluster_bipartite_qcov_70',
                                        split_ratios=(0.0, 0.5, 0.5),
                                        min_cluster_size=3, seed=42)
        assert (result['protein_partition'] == 'train').all()

    def test_cluster_at_min_size_is_splittable(self):
        """Cluster of size == min_cluster_size should be split, not forced to train."""
        # Cluster of exactly 3 with min_cluster_size=3 should produce val and/or test
        df = pd.DataFrame({
            'uniprot': ['X', 'Y', 'Z'],
            'cluster_bipartite_qcov_70': [7, 7, 7],
        })
        result = assign_soft_partitions(df, 'cluster_bipartite_qcov_70',
                                        split_ratios=(0.70, 0.15, 0.15),
                                        min_cluster_size=3, seed=42)
        # With n=3: n_val=max(1,round(3*0.15))=1, n_test=max(1,round(3*0.15))=1, n_train=1
        # So result must NOT be all train
        assert not (result['protein_partition'] == 'train').all()


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

    def test_string_input_returns_nan(self):
        """Non-numeric string → np.nan (not a crash)."""
        assert np.isnan(compute_pchembl("not_a_number"))


class TestBuildSoftSplitRegistry:
    def _make_registry_df(self):
        return pd.DataFrame({
            'sample_id': ['A__0p7_train', 'B__0p7_train', 'C__0p7_decoy'],
            'uniprot_id': ['P00001', 'P00002', 'P00001'],
            'pdb_id': ['1ABC', '1ABC', '1ABC'],
            'compound_id': ['CMP1', 'CMP2', 'CMP3'],
            'cif_path': ['', '', ''],
            'resolution': [2.0, 2.0, 2.0],
            'quality_score': [300.0, 300.0, 300.0],
            'smiles': ['c1ccccc1', 'c1ccccc1', 'c1ccccc1'],
            'sdf_path': ['', '', ''],
            'pkl_path': ['', '', ''],
            'is_active': [True, True, False],
            'affinity_value': [10.0, 1000.0, float('nan')],
            'affinity_type': ['IC50', 'IC50', ''],
            'similarity_threshold': ['0p7', '0p7', ''],
            'split': ['train', 'train', 'decoy'],
            'source': ['chembl', 'chembl', 'deepcoy'],
        })

    def _make_cluster_df(self):
        return pd.DataFrame({
            'uniprot': ['P00001', 'P00002'],
            'cluster_bipartite_qcov_70': [1, 1],
        })

    def test_pchembl_column_added(self, tmp_path):
        """Output must have a 'pchembl' column."""
        reg_path = str(tmp_path / 'registry.csv')
        clust_path = str(tmp_path / 'clusters.csv')
        self._make_registry_df().to_csv(reg_path, index=False)
        self._make_cluster_df().to_csv(clust_path, index=False)
        result = build_soft_split_registry(reg_path, clust_path)
        assert 'pchembl' in result.columns

    def test_pchembl_values_correct(self, tmp_path):
        """pchembl for 10 nM active = 8.0, 1000 nM = 6.0, NaN decoy = NaN."""
        reg_path = str(tmp_path / 'registry.csv')
        clust_path = str(tmp_path / 'clusters.csv')
        self._make_registry_df().to_csv(reg_path, index=False)
        self._make_cluster_df().to_csv(clust_path, index=False)
        result = build_soft_split_registry(reg_path, clust_path)
        actives = result[result['is_active'] == True].sort_values('affinity_value').reset_index(drop=True)
        assert abs(actives.loc[0, 'pchembl'] - 8.0) < 0.01   # 10 nM (smallest) → 8.0
        assert abs(actives.loc[1, 'pchembl'] - 6.0) < 0.01   # 1000 nM → 6.0
        assert np.isnan(result[result['source'] == 'deepcoy']['pchembl'].iloc[0])

    def test_decoys_always_train(self, tmp_path):
        """source=='deepcoy' rows must have protein_partition='train' and protein_cluster=-1."""
        reg_path = str(tmp_path / 'registry.csv')
        clust_path = str(tmp_path / 'clusters.csv')
        self._make_registry_df().to_csv(reg_path, index=False)
        self._make_cluster_df().to_csv(clust_path, index=False)
        result = build_soft_split_registry(reg_path, clust_path)
        decoys = result[result['source'] == 'deepcoy']
        assert (decoys['protein_partition'] == 'train').all()
        assert (decoys['protein_cluster'] == -1).all()

    def test_missing_protein_gets_train(self, tmp_path):
        """Protein in registry but not in cluster file → partition='train', cluster=-1."""
        reg = self._make_registry_df()
        reg = pd.concat([reg, pd.DataFrame({
            'sample_id': ['MISSING__0p7_train'],
            'uniprot_id': ['P99999'],  # not in cluster file
            'pdb_id': ['1ABC'], 'compound_id': ['CMP_X'],
            'cif_path': [''], 'resolution': [2.0], 'quality_score': [300.0],
            'smiles': ['c1ccccc1'], 'sdf_path': [''], 'pkl_path': [''],
            'is_active': [True], 'affinity_value': [50.0], 'affinity_type': ['IC50'],
            'similarity_threshold': ['0p7'], 'split': ['train'], 'source': ['chembl'],
        })], ignore_index=True)
        reg_path = str(tmp_path / 'registry.csv')
        clust_path = str(tmp_path / 'clusters.csv')
        reg.to_csv(reg_path, index=False)
        self._make_cluster_df().to_csv(clust_path, index=False)
        result = build_soft_split_registry(reg_path, clust_path)
        missing_row = result[result['uniprot_id'] == 'P99999']
        assert len(missing_row) == 1
        assert missing_row.iloc[0]['protein_partition'] == 'train'
        assert missing_row.iloc[0]['protein_cluster'] == -1

    def test_default_output_path(self, tmp_path):
        """Default output saves to registry_soft_split.csv in same dir as registry."""
        reg_path = str(tmp_path / 'registry.csv')
        clust_path = str(tmp_path / 'clusters.csv')
        self._make_registry_df().to_csv(reg_path, index=False)
        self._make_cluster_df().to_csv(clust_path, index=False)
        build_soft_split_registry(reg_path, clust_path)
        assert (tmp_path / 'registry_soft_split.csv').exists()
