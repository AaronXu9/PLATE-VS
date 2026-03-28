"""Random Forest regressor for pChEMBL prediction."""
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_regression_trainer import BaseRegressionTrainer


class RandomForestRegressorTrainer(BaseRegressionTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name='random_forest_regressor')
        self.hyperparameters = config.get('hyperparameters', {})

    def build_model(self) -> RandomForestRegressor:
        hp = self.hyperparameters
        model = RandomForestRegressor(
            n_estimators=hp.get('n_estimators', 100),
            max_depth=hp.get('max_depth', None),
            min_samples_split=hp.get('min_samples_split', 2),
            min_samples_leaf=hp.get('min_samples_leaf', 1),
            max_features=hp.get('max_features', 'sqrt'),
            bootstrap=hp.get('bootstrap', True),
            random_state=hp.get('random_state', 42),
            n_jobs=hp.get('n_jobs', -1),
        )
        print(f"RF Regressor: n_estimators={model.n_estimators}, max_depth={model.max_depth}")
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        print("\n" + "=" * 50)
        print("Training Random Forest Regressor")
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
        }
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"Val    RMSE={val_metrics['rmse']:.3f}  R²={val_metrics['r2']:.3f}")
            self.training_history['val_metrics'] = val_metrics
            self.training_history['n_val_samples'] = len(X_val)
        return self.training_history
