# benchmarks/05_boltzina/lib/metrics.py
"""VS benchmark metrics and training_summary.json output."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def compute_vs_metrics(scores, labels, pchembl_values=None):
    """Compute ROC-AUC, EF1%, and Spearman r for one protein benchmark.

    Args:
        scores: list/array of boltz_affinity scores (higher = more active)
        labels: list/array of binary labels (1=active, 0=decoy)
        pchembl_values: list/array of pchembl values (None/NaN for decoys or missing)

    Returns:
        dict with keys: roc_auc, ef1pct, spearman_r (each float or None)
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # ROC-AUC
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        roc_auc = None
    else:
        roc_auc = round(float(roc_auc_score(labels, scores)), 4)

    # EF1% — enrichment factor at top 1%
    n_total = len(scores)
    n_top = max(1, int(np.ceil(n_total * 0.01)))
    top_idx = np.argsort(scores)[::-1][:n_top]
    hits = int(labels[top_idx].sum())
    random_rate = n_pos / n_total if n_total > 0 else 0
    ef1pct = round((hits / n_top) / random_rate, 4) if random_rate > 0 else None

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

    return {'roc_auc': roc_auc, 'ef1pct': ef1pct, 'spearman_r': spearman_r}


def parse_boltzina_csv(csv_path, affinity_col=None):
    """Parse boltzina results CSV into a DataFrame with is_active labels.

    Args:
        csv_path: path to boltzina output CSV
        affinity_col: name of the Boltz-2 affinity column (auto-detected if None)

    Returns:
        DataFrame with columns: ligand_file, boltz_affinity, is_active
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
    df['is_active'] = df[file_col].str.contains('/actives/').astype(int)
    df['boltz_affinity'] = df[affinity_col].astype(float)
    df['ligand_file'] = df[file_col]

    return df[['ligand_file', 'boltz_affinity', 'is_active']]


def aggregate_results(per_protein_metrics):
    """Macro-average metrics across proteins, ignoring None values.

    Args:
        per_protein_metrics: list of dicts from compute_vs_metrics()

    Returns:
        dict with macro-averaged roc_auc, ef1pct, spearman_r
    """
    aggregated = {}
    for key in ('roc_auc', 'ef1pct', 'spearman_r'):
        values = [m[key] for m in per_protein_metrics if m.get(key) is not None]
        aggregated[key] = round(float(np.mean(values)), 4) if values else None
    return aggregated


def write_training_summary(metrics, output_path, n_test, elapsed_s):
    """Write boltzina_training_summary.json in generate_benchmark_report.py format.

    Args:
        metrics: dict from aggregate_results() with roc_auc, ef1pct, spearman_r
        output_path: where to write the JSON
        n_test: total number of compounds evaluated (actives + decoys)
        elapsed_s: total wall-clock time in seconds
    """
    doc = {
        'model_type': 'boltzina',
        'feature_type': '3d_docking_boltz2',
        'similarity_threshold': '0p7',
        'use_precomputed_split': True,
        'training_history': {
            'train_metrics': {},
            'val_metrics': {},
            'test_metrics': {
                'roc_auc': metrics.get('roc_auc'),
                'ef1pct': metrics.get('ef1pct'),
                'spearman_r': metrics.get('spearman_r'),
            },
            'n_train_samples': 0,
            'n_val_samples': 0,
            'n_test_samples': int(n_test),
            'training_time': round(float(elapsed_s), 1),
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(doc, f, indent=2)
