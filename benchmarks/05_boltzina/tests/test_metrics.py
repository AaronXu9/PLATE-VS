# benchmarks/05_boltzina/tests/test_metrics.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from metrics import compute_vs_metrics, write_training_summary, parse_boltzina_csv, SCORE_CONFIGS


def test_roc_auc_perfect():
    # Actives score higher → perfect separation
    scores = [1.0, 0.9, 0.8, 0.1, 0.05, 0.02]
    labels = [1,   1,   1,   0,   0,    0  ]
    m = compute_vs_metrics(scores, labels)
    assert m['roc_auc'] == pytest.approx(1.0)


def test_roc_auc_random():
    rng = np.random.default_rng(42)
    scores = rng.random(200).tolist()
    labels = ([1] * 10 + [0] * 190)
    rng.shuffle(labels)
    m = compute_vs_metrics(scores, labels)
    assert 0.3 < m['roc_auc'] < 0.7  # roughly random


def test_ef1pct_perfect():
    # 10 actives in 1000 total; top 1% = 10 slots → all actives ranked first
    n = 1000
    n_actives = 10
    scores = [2.0] * n_actives + [1.0] * (n - n_actives)
    labels = [1] * n_actives + [0] * (n - n_actives)
    m = compute_vs_metrics(scores, labels)
    # Expected EF1% = (10/10) / (10/1000) = 100
    assert m['ef1pct'] == pytest.approx(100.0, rel=0.01)


def test_ef5pct_perfect():
    # 50 actives in 1000 total; top 5% = 50 slots → all actives ranked first
    n = 1000
    n_actives = 50
    scores = [2.0] * n_actives + [1.0] * (n - n_actives)
    labels = [1] * n_actives + [0] * (n - n_actives)
    m = compute_vs_metrics(scores, labels)
    # Expected EF5% = (50/50) / (50/1000) = 20
    assert m['ef5pct'] == pytest.approx(20.0, rel=0.01)


def test_ef5pct_returned():
    scores = [1.0, 0.9, 0.8, 0.1, 0.05, 0.02]
    labels = [1,   1,   1,   0,   0,    0  ]
    m = compute_vs_metrics(scores, labels)
    assert 'ef5pct' in m
    assert m['ef5pct'] is not None


def test_spearman_perfect_correlation():
    scores =  [1.0, 2.0, 3.0, 4.0, 5.0]
    labels =  [1,   1,   1,   1,   1  ]
    pchembl = [4.0, 5.0, 6.0, 7.0, 8.0]  # monotone with scores
    m = compute_vs_metrics(scores, labels, pchembl_values=pchembl)
    assert m['spearman_r'] == pytest.approx(1.0, abs=0.01)


def test_spearman_none_when_too_few_actives():
    scores =  [1.0, 2.0, 0.5]
    labels =  [1,   1,   0  ]
    pchembl = [5.0, 6.0, None]
    m = compute_vs_metrics(scores, labels, pchembl_values=pchembl)
    assert m['spearman_r'] is None  # < 5 actives with pchembl


def test_write_training_summary_flat(tmp_path):
    """Legacy flat metrics format."""
    metrics = {'roc_auc': 0.75, 'ef1pct': 5.2, 'ef5pct': 3.1, 'spearman_r': 0.31}
    out = tmp_path / 'summary.json'
    write_training_summary(metrics, str(out), n_test=1234, elapsed_s=3600.0)

    with open(out) as f:
        doc = json.load(f)

    assert doc['model_type'] == 'boltzina'
    assert doc['feature_type'] == '3d_docking_boltz2'
    assert doc['similarity_threshold'] == '0p7'
    assert doc['use_precomputed_split'] is True
    hist = doc['training_history']
    assert hist['test_metrics']['roc_auc'] == pytest.approx(0.75)
    assert hist['test_metrics']['ef1pct'] == pytest.approx(5.2)
    assert hist['test_metrics']['ef5pct'] == pytest.approx(3.1)
    assert hist['test_metrics']['spearman_r'] == pytest.approx(0.31)
    assert hist['n_test_samples'] == 1234
    assert hist['training_time'] == pytest.approx(3600.0)
    assert 'train_metrics' in hist
    assert 'val_metrics' in hist


def test_write_training_summary_per_score(tmp_path):
    """Per-score prefixed metrics format."""
    metrics = {
        'docking_score_roc_auc': 0.80,
        'docking_score_ef1pct': 10.0,
        'docking_score_ef5pct': 4.0,
        'docking_score_spearman_r': 0.25,
        'affinity_pred_value_roc_auc': 0.65,
        'affinity_pred_value_ef1pct': 3.0,
        'affinity_pred_value_ef5pct': 2.0,
        'affinity_pred_value_spearman_r': 0.40,
        'affinity_probability_binary_roc_auc': 0.60,
        'affinity_probability_binary_ef1pct': 2.5,
        'affinity_probability_binary_ef5pct': 1.8,
        'affinity_probability_binary_spearman_r': None,
    }
    out = tmp_path / 'summary.json'
    write_training_summary(metrics, str(out), n_test=5000, elapsed_s=120.0)

    with open(out) as f:
        doc = json.load(f)

    tm = doc['training_history']['test_metrics']
    # Per-score breakdown
    assert tm['docking_score']['roc_auc'] == pytest.approx(0.80)
    assert tm['affinity_pred_value']['roc_auc'] == pytest.approx(0.65)
    assert tm['affinity_probability_binary']['spearman_r'] is None
    # Backward compat top-level from affinity_pred_value
    assert tm['roc_auc'] == pytest.approx(0.65)
    assert tm['ef5pct'] == pytest.approx(2.0)


def test_parse_boltzina_csv_all_scores(tmp_path):
    """parse_boltzina_csv returns all score columns."""
    csv_path = tmp_path / 'results.csv'
    csv_path.write_text(
        'ligand_name,docking_score,affinity_pred_value,affinity_probability_binary\n'
        '/actives/a1.pdb,-8.5,0.75,0.90\n'
        '/actives/a2.pdb,-7.2,0.60,0.85\n'
        '/decoys/d1.pdb,-6.0,0.30,0.20\n'
        '/decoys/d2.pdb,-5.5,0.25,0.15\n'
    )
    df = parse_boltzina_csv(str(csv_path))
    assert len(df) == 4
    assert df['is_active'].tolist() == [1, 1, 0, 0]
    assert 'docking_score' in df.columns
    assert 'affinity_pred_value' in df.columns
    assert 'affinity_probability_binary' in df.columns
    assert 'boltz_affinity' in df.columns  # backward compat alias
    assert df['docking_score'].iloc[0] == pytest.approx(-8.5)
    assert df['affinity_pred_value'].iloc[0] == pytest.approx(0.75)


def test_score_configs_defined():
    """SCORE_CONFIGS has the expected 3 entries."""
    assert len(SCORE_CONFIGS) == 3
    names = [s[0] for s in SCORE_CONFIGS]
    assert 'docking_score' in names
    assert 'affinity_pred_value' in names
    assert 'affinity_probability_binary' in names
    # docking_score should be negated
    dock_cfg = [s for s in SCORE_CONFIGS if s[0] == 'docking_score'][0]
    assert dock_cfg[2] is True  # negate=True
