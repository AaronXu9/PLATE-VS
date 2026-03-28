"""LinearSVR-based regressor with feature scaling. Sparse-safe."""
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from .base_regression_trainer import BaseRegressionTrainer


class SVMRegressorTrainer(BaseRegressionTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name='svm_regressor')
        self.hyperparameters = config.get('hyperparameters', {})

    def build_model(self) -> Pipeline:
        hp = self.hyperparameters
        svr = LinearSVR(
            C=hp.get('C', 1.0),
            epsilon=hp.get('epsilon', 0.1),
            max_iter=hp.get('max_iter', 2000),
            random_state=hp.get('random_state', 42),
            dual='auto',
        )
        return Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('svr', svr),
        ])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X).astype(np.float32)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        print("\n" + "=" * 50)
        print("Training SVM Regressor (LinearSVR)")
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
