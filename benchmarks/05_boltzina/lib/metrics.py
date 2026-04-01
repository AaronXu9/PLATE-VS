# benchmarks/05_boltzina/lib/metrics.py
"""VS benchmark metrics and training_summary.json output."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _enrichment_factor(scores, labels, fraction):
    """Compute enrichment factor at a given top-fraction."""
    n_total = len(scores)
    n_top = max(1, int(np.ceil(n_total * fraction)))
    top_idx = np.argsort(scores)[::-1][:n_top]
    hits = int(labels[top_idx].sum())
    n_pos = labels.sum()
    random_rate = n_pos / n_total if n_total > 0 else 0
    return round((hits / n_top) / random_rate, 4) if random_rate > 0 else None


def compute_vs_metrics(scores, labels, pchembl_values=None):
    """Compute ROC-AUC, EF1%, EF5%, and Spearman r for one protein benchmark.

    Args:
        scores: list/array of ranking scores (higher = more active)
        labels: list/array of binary labels (1=active, 0=decoy)
        pchembl_values: list/array of pchembl values (None/NaN for decoys or missing)

    Returns:
        dict with keys: roc_auc, ef1pct, ef5pct, spearman_r (each float or None)
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # ROC-AUC
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        roc_auc = None
    else:
        roc_auc = round(float(roc_auc_score(labels, scores)), 4)

    # Enrichment factors
    ef1pct = _enrichment_factor(scores, labels, 0.01)
    ef5pct = _enrichment_factor(scores, labels, 0.05)

    # Spearman r — affinity correlation for actives with pchembl labels
    spearman_r = None
    if pchembl_values is not None:
        pchembl_arr = np.asarray(
            [float(v) if v is not None else np.nan for v in pchembl_values],
            dtype=float,
        )
        mask = (labels == 1) & ~np.isnan(pchembl_arr)
        if mask.sum() >= 5:
            r, _ = spearmanr(scores[mask], pchembl_arr[mask])
            spearman_r = round(float(r), 4)

    return {'roc_auc': roc_auc, 'ef1pct': ef1pct, 'ef5pct': ef5pct, 'spearman_r': spearman_r}


def parse_boltzina_csv(csv_path, affinity_col=None):
    """Parse boltzina results CSV into a DataFrame with is_active labels and all scores.

    Args:
        csv_path: path to boltzina output CSV
        affinity_col: name of the Boltz-2 affinity column (auto-detected if None)

    Returns:
        DataFrame with columns: ligand_file, is_active, docking_score,
        affinity_pred_value, affinity_probability_binary, boltz_affinity (alias)
    """
    df = pd.read_csv(csv_path)

    # Auto-detect affinity column (first column containing 'affinity' in name)
    if affinity_col is None:
        candidates = [c for c in df.columns if 'affinity' in c.lower()]
        if not candidates:
            raise ValueError(
                f'No affinity column found in {csv_path}. '
                f'Columns: {df.columns.tolist()}. '
                f'Pass affinity_col= explicitly.'
            )
        affinity_col = candidates[0]

    # Identify ligand file column
    file_col = next(
        (c for c in df.columns if 'ligand' in c.lower() or 'file' in c.lower()),
        df.columns[0]
    )

    # Derive is_active from path: actives/ → 1, decoys/ → 0
    out = pd.DataFrame()
    out['ligand_file'] = df[file_col]
    out['is_active'] = df[file_col].str.contains('/actives/').astype(int)

    # Score columns (use NaN if column not present in CSV)
    out['docking_score'] = df['docking_score'].astype(float) if 'docking_score' in df.columns else np.nan
    out['affinity_pred_value'] = df[affinity_col].astype(float)
    out['affinity_probability_binary'] = (
        df['affinity_probability_binary'].astype(float)
        if 'affinity_probability_binary' in df.columns else np.nan
    )
    # Backward compat alias
    out['boltz_affinity'] = out['affinity_pred_value']

    return out


def aggregate_results(per_protein_metrics):
    """Macro-average metrics across proteins, ignoring None values.

    Args:
        per_protein_metrics: list of dicts from compute_vs_metrics().
            Keys can be plain (roc_auc) or prefixed (docking_score_roc_auc).

    Returns:
        dict with macro-averaged values for every key found in the input dicts
    """
    if not per_protein_metrics:
        return {}
    all_keys = {k for m in per_protein_metrics for k in m}
    aggregated = {}
    for key in sorted(all_keys):
        values = [m[key] for m in per_protein_metrics if m.get(key) is not None]
        aggregated[key] = round(float(np.mean(values)), 4) if values else None
    return aggregated


METRIC_KEYS = ('roc_auc', 'ef1pct', 'ef5pct', 'spearman_r')

SCORE_CONFIGS = [
    # (score_name, csv_column, negate_for_ranking)
    ('docking_score', 'docking_score', True),
    ('affinity_pred_value', 'affinity_pred_value', False),
    ('affinity_probability_binary', 'affinity_probability_binary', False),
]


def write_training_summary(metrics, output_path, n_test, elapsed_s):
    """Write boltzina_training_summary.json in generate_benchmark_report.py format.

    Args:
        metrics: dict from aggregate_results(). Supports both flat keys (roc_auc)
            and prefixed keys (affinity_pred_value_roc_auc).
        output_path: where to write the JSON
        n_test: total number of compounds evaluated (actives + decoys)
        elapsed_s: total wall-clock time in seconds
    """
    # Build per-score test_metrics if prefixed keys are present
    test_metrics = {}
    has_prefixed = any(k for k in metrics if '_roc_auc' in k)
    if has_prefixed:
        for score_name, _, _ in SCORE_CONFIGS:
            score_metrics = {}
            for mk in METRIC_KEYS:
                key = f'{score_name}_{mk}'
                score_metrics[mk] = metrics.get(key)
            test_metrics[score_name] = score_metrics
        # Backward compat: top-level keys from affinity_pred_value
        for mk in METRIC_KEYS:
            test_metrics[mk] = metrics.get(f'affinity_pred_value_{mk}')
    else:
        # Legacy flat format
        for mk in METRIC_KEYS:
            test_metrics[mk] = metrics.get(mk)

    doc = {
        'model_type': 'boltzina',
        'feature_type': '3d_docking_boltz2',
        'similarity_threshold': '0p7',
        'use_precomputed_split': True,
        'training_history': {
            'train_metrics': {},
            'val_metrics': {},
            'test_metrics': test_metrics,
            'n_train_samples': 0,
            'n_val_samples': 0,
            'n_test_samples': int(n_test),
            'training_time': round(float(elapsed_s), 1),
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(doc, f, indent=2)
