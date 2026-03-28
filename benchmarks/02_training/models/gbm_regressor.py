"""Gradient Boosting regressor. Backend: hist (sklearn) / xgboost / lightgbm."""
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import xgboost as xgb
    _XGBOOST = True
except ImportError:
    _XGBOOST = False

try:
    import lightgbm as lgb
    _LIGHTGBM = True
except ImportError:
    _LIGHTGBM = False

from .base_regression_trainer import BaseRegressionTrainer


class GBMRegressorTrainer(BaseRegressionTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name='gradient_boosting_regressor')
        self.hyperparameters = config.get('hyperparameters', {})
        self.backend = self._resolve_backend(config.get('backend', 'auto'))
        print(f"GBMRegressorTrainer backend: {self.backend}")

    def _resolve_backend(self, requested: str) -> str:
        if requested == 'auto':
            if _XGBOOST: return 'xgboost'
            if _LIGHTGBM: return 'lightgbm'
            return 'hist'
        if requested == 'xgboost' and not _XGBOOST:
            print("Warning: xgboost not installed, falling back to hist")
            return 'hist'
        if requested == 'lightgbm' and not _LIGHTGBM:
            print("Warning: lightgbm not installed, falling back to hist")
            return 'hist'
        return requested

    def build_model(self):
        hp = self.hyperparameters
        if self.backend == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=hp.get('n_estimators', 300),
                max_depth=hp.get('max_depth', 6),
                learning_rate=hp.get('learning_rate', 0.05),
                subsample=hp.get('subsample', 0.8),
                colsample_bytree=hp.get('colsample_bytree', 0.8),
                random_state=hp.get('random_state', 42),
                n_jobs=hp.get('n_jobs', -1),
                verbosity=0,
            )
        if self.backend == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=hp.get('n_estimators', 300),
                learning_rate=hp.get('learning_rate', 0.05),
                subsample=hp.get('subsample', 0.8),
                colsample_bytree=hp.get('colsample_bytree', 0.8),
                random_state=hp.get('random_state', 42),
                n_jobs=hp.get('n_jobs', -1),
                verbose=-1,
            )
        return HistGradientBoostingRegressor(
            max_iter=hp.get('n_estimators', 300),
            learning_rate=hp.get('learning_rate', 0.05),
            max_depth=hp.get('max_depth', None),
            max_leaf_nodes=hp.get('max_leaf_nodes', 63),
            min_samples_leaf=hp.get('min_samples_leaf', 20),
            l2_regularization=hp.get('l2_regularization', 0.1),
            random_state=hp.get('random_state', 42),
            early_stopping=False,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        print("\n" + "=" * 50)
        print(f"Training GBM Regressor [{self.backend}]")
        print("=" * 50)
        self.model = self.build_model()
        start = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"Trained in {elapsed:.2f}s on {len(X_train)} samples")

        train_metrics = self.evaluate(X_train, y_train)
        print(f"Train  RMSE={train_metrics['rmse']:.3f}  R²={train_metrics['r2']:.3f}")

        self.training_history = {
            'train_metrics': train_metrics,
            'training_time': elapsed,
            'n_train_samples': len(X_train),
            'hyperparameters': self.hyperparameters,
            'backend': self.backend,
        }
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"Val    RMSE={val_metrics['rmse']:.3f}  R²={val_metrics['r2']:.3f}")
            self.training_history['val_metrics'] = val_metrics
            self.training_history['n_val_samples'] = len(X_val)
        return self.training_history
