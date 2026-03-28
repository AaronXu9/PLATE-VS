import math
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path to import local metrics module
metrics_path = Path(__file__).parent.parent / "metrics.py"
import importlib.util
spec = importlib.util.spec_from_file_location("metrics", metrics_path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

calculate_r2 = metrics.calculate_r2
calculate_spearman = metrics.calculate_spearman
summarize_regression = metrics.summarize_regression


class TestCalculateR2:
    def test_perfect_prediction(self):
        y = [1.0, 2.0, 3.0, 4.0]
        assert pytest.approx(calculate_r2(y, y), abs=1e-9) == 1.0

    def test_mean_prediction_is_zero(self):
        y = [1.0, 2.0, 3.0, 4.0]
        y_pred = [2.5, 2.5, 2.5, 2.5]
        assert pytest.approx(calculate_r2(y, y_pred), abs=1e-9) == 0.0

    def test_constant_true_returns_nan(self):
        assert math.isnan(calculate_r2([5.0, 5.0, 5.0], [5.0, 5.0, 5.1]))

    def test_known_value(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5,  0.0, 2.0, 8.0])
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        expected = 1 - ss_res / ss_tot
        assert pytest.approx(calculate_r2(y_true, y_pred), abs=1e-6) == expected


class TestCalculateSpearman:
    def test_perfect_rank_correlation(self):
        y = [1.0, 2.0, 3.0, 4.0]
        assert pytest.approx(calculate_spearman(y, y), abs=1e-9) == 1.0

    def test_inverse_rank_correlation(self):
        y = [1.0, 2.0, 3.0, 4.0]
        y_inv = [4.0, 3.0, 2.0, 1.0]
        assert pytest.approx(calculate_spearman(y, y_inv), abs=1e-9) == -1.0

    def test_fewer_than_two_samples_returns_nan(self):
        assert math.isnan(calculate_spearman([5.0], [5.0]))


class TestSummarizeRegressionKeys:
    def test_has_r2_and_spearman(self):
        y = [6.0, 7.0, 8.0, 7.5, 6.5]
        result = summarize_regression(y, y)
        assert 'r2' in result
        assert 'spearman' in result
