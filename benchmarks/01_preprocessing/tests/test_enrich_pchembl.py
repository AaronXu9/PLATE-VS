# benchmarks/01_preprocessing/tests/test_enrich_pchembl.py
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from enrich_pchembl import aggregate_pchembl, enrich_registry, build_protein_pchembl_map


def _make_filtered_activities(rows):
    """Make activities in filtered_chembl_affinity.parquet format."""
    return pd.DataFrame(rows, columns=[
        'source_uniprot_id', 'canonical_smiles', 'pchembl_value',
        'assay_type', 'standard_type'
    ])


def _make_activities(rows):
    return pd.DataFrame(rows, columns=[
        'molecule_chembl_id', 'pchembl_value', 'assay_type', 'standard_type'
    ])


def _make_registry(compound_ids):
    return pd.DataFrame({
        'compound_id': compound_ids,
        'pchembl': [np.nan] * len(compound_ids),
    })


class TestAggregatePChEMBL:
    def test_median_aggregation(self):
        acts = _make_activities([
            ('CHEMBL1', 7.0, 'B', 'Ki'),
            ('CHEMBL1', 8.0, 'B', 'Ki'),
            ('CHEMBL2', 6.0, 'B', 'IC50'),
        ])
        result = aggregate_pchembl(acts)
        assert pytest.approx(result['CHEMBL1'], abs=1e-6) == 7.5
        assert pytest.approx(result['CHEMBL2'], abs=1e-6) == 6.0

    def test_filters_non_binding_assays(self):
        acts = _make_activities([
            ('CHEMBL1', 7.0, 'F', 'Ki'),
            ('CHEMBL2', 6.0, 'B', 'Ki'),
        ])
        result = aggregate_pchembl(acts, assay_types={'B'})
        assert 'CHEMBL1' not in result
        assert 'CHEMBL2' in result

    def test_filters_out_of_range_pchembl(self):
        acts = _make_activities([
            ('CHEMBL1', 3.5, 'B', 'Ki'),
            ('CHEMBL2', 7.0, 'B', 'Ki'),
            ('CHEMBL3', 13.0, 'B', 'Ki'),
        ])
        result = aggregate_pchembl(acts, pchembl_min=4.0, pchembl_max=12.0)
        assert 'CHEMBL1' not in result
        assert 'CHEMBL2' in result
        assert 'CHEMBL3' not in result

    def test_filters_invalid_standard_types(self):
        acts = _make_activities([
            ('CHEMBL1', 7.0, 'B', 'MIC'),
            ('CHEMBL2', 7.0, 'B', 'IC50'),
        ])
        result = aggregate_pchembl(acts)
        assert 'CHEMBL1' not in result
        assert 'CHEMBL2' in result

    def test_empty_activities_returns_empty_series(self):
        acts = _make_activities([])
        result = aggregate_pchembl(acts)
        assert len(result) == 0


class TestEnrichRegistry:
    def test_pchembl_joined_by_compound_id(self):
        registry = _make_registry(['CHEMBL1', 'CHEMBL2', 'CHEMBL3'])
        agg = pd.Series({'CHEMBL1': 7.5, 'CHEMBL2': 6.0})
        result = enrich_registry(registry, agg)
        assert pytest.approx(result.loc[result['compound_id'] == 'CHEMBL1', 'pchembl'].iloc[0]) == 7.5
        assert pytest.approx(result.loc[result['compound_id'] == 'CHEMBL2', 'pchembl'].iloc[0]) == 6.0
        assert np.isnan(result.loc[result['compound_id'] == 'CHEMBL3', 'pchembl'].iloc[0])

    def test_original_non_null_pchembl_overwritten(self):
        registry = pd.DataFrame({'compound_id': ['CHEMBL1'], 'pchembl': [5.0]})
        agg = pd.Series({'CHEMBL1': 8.0})
        result = enrich_registry(registry, agg)
        assert pytest.approx(result['pchembl'].iloc[0]) == 8.0

    def test_row_count_unchanged(self):
        registry = _make_registry(['CHEMBL1'] * 3)
        agg = pd.Series({'CHEMBL1': 7.0})
        result = enrich_registry(registry, agg)
        assert len(result) == 3


class TestBuildProteinPChEMBLMap:
    def test_multiindex_keyed_by_protein_and_smiles(self):
        acts = _make_filtered_activities([
            ('P00000', 'CCO', 7.0, 'B', 'Ki'),
            ('P00001', 'CCO', 5.0, 'B', 'Ki'),   # same SMILES, different protein
        ])
        result = build_protein_pchembl_map(acts)
        assert pytest.approx(result[('P00000', 'CCO')]) == 7.0
        assert pytest.approx(result[('P00001', 'CCO')]) == 5.0

    def test_median_across_multiple_assays(self):
        acts = _make_filtered_activities([
            ('P00000', 'CCO', 6.0, 'B', 'IC50'),
            ('P00000', 'CCO', 8.0, 'B', 'IC50'),
        ])
        result = build_protein_pchembl_map(acts)
        assert pytest.approx(result[('P00000', 'CCO')]) == 7.0

    def test_filters_non_binding_assays(self):
        acts = _make_filtered_activities([
            ('P00000', 'CCO', 7.0, 'F', 'Ki'),   # functional — excluded
            ('P00000', 'CCC', 6.0, 'B', 'Ki'),
        ])
        result = build_protein_pchembl_map(acts, assay_types={'B'})
        assert ('P00000', 'CCO') not in result.index
        assert ('P00000', 'CCC') in result.index

    def test_empty_returns_empty_series(self):
        acts = _make_filtered_activities([])
        result = build_protein_pchembl_map(acts)
        assert len(result) == 0
