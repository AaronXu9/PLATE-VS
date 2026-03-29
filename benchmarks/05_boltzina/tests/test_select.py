# benchmarks/05_boltzina/tests/test_select.py
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from protein_select import select_proteins


def _make_registry(specs):
    """specs: list of (uid, pdb_id, n_actives, pchembl_frac, quality_score)"""
    rows = []
    for uid, pdb_id, n, cov, qscore in specs:
        for i in range(n):
            rows.append({
                'uniprot_id': uid, 'pdb_id': pdb_id,
                'cif_path': f'plate-vs/.../cif_files_raw/{pdb_id}.cif',
                'similarity_threshold': '0p7',
                'protein_partition': 'test',
                'is_active': True,
                'pchembl': 6.5 if i < int(n * cov) else None,
                'quality_score': float(qscore),
            })
    return pd.DataFrame(rows)


def test_filters_min_actives():
    df = _make_registry([
        ('P00001', '1ABC', 60, 0.9, 300),  # passes: 60 actives
        ('P00002', '2DEF', 30, 0.9, 400),  # fails: only 30 actives
    ])
    result = select_proteins(df, n=10, min_actives=50, min_pchembl_coverage=0.5)
    uids = [p['uniprot_id'] for p in result]
    assert 'P00001' in uids
    assert 'P00002' not in uids


def test_filters_pchembl_coverage():
    df = _make_registry([
        ('P00001', '1ABC', 60, 0.9, 300),  # passes: 90% coverage
        ('P00002', '2DEF', 60, 0.3, 400),  # fails: 30% coverage
    ])
    result = select_proteins(df, n=10, min_actives=50, min_pchembl_coverage=0.80)
    uids = [p['uniprot_id'] for p in result]
    assert 'P00001' in uids
    assert 'P00002' not in uids


def test_sorts_by_quality_score():
    df = _make_registry([
        ('P00001', '1ABC', 60, 0.9, 100),
        ('P00002', '2DEF', 60, 0.9, 300),
        ('P00003', '3GHI', 60, 0.9, 200),
    ])
    result = select_proteins(df, n=3)
    assert result[0]['uniprot_id'] == 'P00002'
    assert result[1]['uniprot_id'] == 'P00003'
    assert result[2]['uniprot_id'] == 'P00001'


def test_n_decoys_capped():
    df = _make_registry([('P00001', '1ABC', 100, 0.9, 300)])
    result = select_proteins(df, n=1)
    assert result[0]['n_decoys_to_sample'] == 2500  # 100*50=5000 → capped


def test_n_decoys_not_capped_when_small():
    # 30 actives * 50 ratio = 1500, which is under the 2500 cap
    df = _make_registry([('P00001', '1ABC', 30, 0.9, 300)])
    result = select_proteins(df, n=1, min_actives=30)
    assert result[0]['n_decoys_to_sample'] == 30 * 50  # 1500, not capped


def test_output_keys():
    df = _make_registry([('P00001', '1ABC', 60, 0.9, 300)])
    result = select_proteins(df, n=1)
    assert len(result) == 1
    expected_keys = {'uniprot_id', 'pdb_id', 'cif_path', 'n_actives',
                     'n_decoys_to_sample', 'pchembl_coverage', 'quality_score'}
    assert expected_keys.issubset(result[0].keys())
