"""Standardized metrics for benchmarking."""

from __future__ import annotations

import math

import numpy as np

try:
    from lifelines.utils import concordance_index as _lifelines_concordance_index
except ImportError:
    _lifelines_concordance_index = None

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _prepare_arrays(y_true, y_pred):
    true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)

    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {true_arr.shape}, y_pred has shape {pred_arr.shape}"
        )

    valid_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    return true_arr[valid_mask], pred_arr[valid_mask]


def calculate_mse(y_true, y_pred):
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    if len(true_arr) == 0:
        return float("nan")
    return float(np.mean((true_arr - pred_arr) ** 2))


def calculate_rmse(y_true, y_pred):
    mse = calculate_mse(y_true, y_pred)
    return float(math.sqrt(mse)) if math.isfinite(mse) else float("nan")


def calculate_mae(y_true, y_pred):
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    if len(true_arr) == 0:
        return float("nan")
    return float(np.mean(np.abs(true_arr - pred_arr)))


def calculate_pearson(y_true, y_pred):
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    if len(true_arr) < 2:
        return float("nan")
    if np.std(true_arr) == 0 or np.std(pred_arr) == 0:
        return float("nan")
    return float(np.corrcoef(true_arr, pred_arr)[0, 1])


def calculate_ci(y_true, y_pred):
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    if len(true_arr) < 2:
        return float("nan")

    if _lifelines_concordance_index is not None:
        return float(_lifelines_concordance_index(true_arr, pred_arr))

    concordant = 0.0
    comparable = 0.0
    n_samples = len(true_arr)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if true_arr[i] == true_arr[j]:
                continue

            comparable += 1.0
            true_order = true_arr[i] > true_arr[j]
            pred_diff = pred_arr[i] - pred_arr[j]

            if pred_diff == 0:
                concordant += 0.5
            elif (pred_diff > 0) == true_order:
                concordant += 1.0

    if comparable == 0:
        return float("nan")

    return float(concordant / comparable)


def calculate_r2(y_true, y_pred):
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    if len(true_arr) < 2:
        return float("nan")
    ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
    if ss_tot == 0.0:
        return float("nan")
    ss_res = np.sum((true_arr - pred_arr) ** 2)
    return float(1.0 - ss_res / ss_tot)


def calculate_spearman(y_true, y_pred):
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    if len(true_arr) < 2:
        return float("nan")
    if not _SCIPY_AVAILABLE:
        return float("nan")
    result = _scipy_stats.spearmanr(true_arr, pred_arr)
    return float(result.correlation)


def summarize_regression(y_true, y_pred):
    return {
        "mse":      calculate_mse(y_true, y_pred),
        "rmse":     calculate_rmse(y_true, y_pred),
        "mae":      calculate_mae(y_true, y_pred),
        "r2":       calculate_r2(y_true, y_pred),
        "pearson":  calculate_pearson(y_true, y_pred),
        "spearman": calculate_spearman(y_true, y_pred),
        "ci":       calculate_ci(y_true, y_pred),
    }
