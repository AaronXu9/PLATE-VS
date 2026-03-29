# benchmarks/05_boltzina/tests/test_metrics.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from metrics import compute_vs_metrics, write_training_summary


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


def test_write_training_summary_schema(tmp_path):
    metrics = {'roc_auc': 0.75, 'ef1pct': 5.2, 'spearman_r': 0.31}
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
    assert hist['test_metrics']['spearman_r'] == pytest.approx(0.31)
    assert hist['n_test_samples'] == 1234
    assert hist['training_time'] == pytest.approx(3600.0)
    # Required by generate_benchmark_report.py
    assert 'train_metrics' in hist
    assert 'val_metrics' in hist
    assert 'n_train_samples' in hist
    assert 'n_val_samples' in hist
