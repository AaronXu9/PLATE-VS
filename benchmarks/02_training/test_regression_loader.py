"""Tests for regression mode in DataLoader.prepare_features_labels()."""
import sys
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.data_loader import DataLoader


def _make_registry_csv(tmp_dir: str, include_pchembl: bool = True) -> str:
    """Minimal registry CSV for testing."""
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
        smiles, labels = loader.prepare_features_labels(data)
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
