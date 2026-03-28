import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rf_regressor import RandomForestRegressorTrainer
from models.gbm_regressor import GBMRegressorTrainer
from models.svm_regressor import SVMRegressorTrainer


def _small_regression_data(n=200, n_features=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (rng.standard_normal(n) + 7.0).astype(np.float32)
    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


_MINIMAL_CONFIG = {
    'hyperparameters': {'n_estimators': 10, 'random_state': 42, 'n_jobs': 1}
}


class TestRandomForestRegressor:
    def test_train_returns_history_with_metrics(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        hist = t.train(X_tr, y_tr, X_v, y_v)
        assert 'train_metrics' in hist
        assert 'val_metrics' in hist
        for key in ('rmse', 'mae', 'r2', 'pearson', 'spearman'):
            assert key in hist['train_metrics'], f"missing {key} in train_metrics"

    def test_predict_shape(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        t.train(X_tr, y_tr)
        preds = t.predict(X_v)
        assert preds.shape == (len(X_v),)

    def test_predict_before_train_raises(self):
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        with pytest.raises(ValueError, match="not trained"):
            t.predict(np.zeros((5, 64)))

    def test_save_load_roundtrip(self, tmp_path):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        t.train(X_tr, y_tr)
        t.save_model(str(tmp_path))
        t2 = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        t2.load_model(str(tmp_path))
        np.testing.assert_array_almost_equal(t.predict(X_v), t2.predict(X_v))

    def test_evaluate_returns_all_regression_keys(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        t.train(X_tr, y_tr)
        metrics = t.evaluate(X_v, y_v)
        for key in ('mse', 'rmse', 'mae', 'r2', 'pearson', 'spearman', 'ci'):
            assert key in metrics, f"missing {key}"


class TestGBMRegressor:
    def test_train_and_evaluate(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = GBMRegressorTrainer({'hyperparameters': {'n_estimators': 10, 'random_state': 42}})
        hist = t.train(X_tr, y_tr, X_v, y_v)
        assert 'val_metrics' in hist
        assert 'rmse' in hist['val_metrics']


class TestSVMRegressor:
    def test_train_and_evaluate(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = SVMRegressorTrainer({'hyperparameters': {'C': 0.1, 'max_iter': 500}})
        hist = t.train(X_tr, y_tr, X_v, y_v)
        assert 'val_metrics' in hist
        assert 'rmse' in hist['val_metrics']
