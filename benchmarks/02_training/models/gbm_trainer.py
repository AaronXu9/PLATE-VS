"""
Gradient Boosting Trainer Module.

Uses sklearn's HistGradientBoostingClassifier by default (fast, histogram-based,
equivalent to LightGBM). Optionally falls back to XGBoost or LightGBM if installed.
"""

import numpy as np
from typing import Dict, Any, Optional
import time

from .base_trainer import BaseTrainer

# Try to import faster GBM implementations
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class GBMTrainer(BaseTrainer):
    """
    Gradient Boosting trainer.

    Backend priority (configurable via config['backend']):
      'xgboost'  -> XGBoostClassifier  (requires xgboost)
      'lightgbm' -> LGBMClassifier     (requires lightgbm)
      'hist'     -> HistGradientBoostingClassifier (sklearn, default)
      'auto'     -> picks the first available in the order above
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name='gradient_boosting')
        self.hyperparameters = config.get('hyperparameters', {})
        self.backend = self._resolve_backend(config.get('backend', 'auto'))
        print(f"GBMTrainer using backend: {self.backend}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_backend(self, requested: str) -> str:
        if requested == 'auto':
            if XGBOOST_AVAILABLE:
                return 'xgboost'
            if LIGHTGBM_AVAILABLE:
                return 'lightgbm'
            return 'hist'
        if requested == 'xgboost' and not XGBOOST_AVAILABLE:
            print("Warning: xgboost not installed, falling back to hist backend")
            return 'hist'
        if requested == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            print("Warning: lightgbm not installed, falling back to hist backend")
            return 'hist'
        return requested

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def build_model(self):
        hp = self.hyperparameters

        if self.backend == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=hp.get('n_estimators', 300),
                max_depth=hp.get('max_depth', 6),
                learning_rate=hp.get('learning_rate', 0.05),
                subsample=hp.get('subsample', 0.8),
                colsample_bytree=hp.get('colsample_bytree', 0.8),
                min_child_weight=hp.get('min_child_weight', 3),
                scale_pos_weight=hp.get('scale_pos_weight', None),  # handles imbalance
                random_state=hp.get('random_state', 42),
                n_jobs=hp.get('n_jobs', -1),
                eval_metric='logloss',
                use_label_encoder=False,
            )

        elif self.backend == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=hp.get('n_estimators', 300),
                max_depth=hp.get('max_depth', -1),
                learning_rate=hp.get('learning_rate', 0.05),
                num_leaves=hp.get('num_leaves', 63),
                subsample=hp.get('subsample', 0.8),
                colsample_bytree=hp.get('colsample_bytree', 0.8),
                class_weight=hp.get('class_weight', 'balanced'),
                random_state=hp.get('random_state', 42),
                n_jobs=hp.get('n_jobs', -1),
                verbose=-1,
            )

        else:  # 'hist' — sklearn HistGradientBoostingClassifier
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(
                max_iter=hp.get('n_estimators', 300),
                max_depth=hp.get('max_depth', None),
                learning_rate=hp.get('learning_rate', 0.05),
                max_leaf_nodes=hp.get('max_leaf_nodes', 63),
                min_samples_leaf=hp.get('min_samples_leaf', 20),
                l2_regularization=hp.get('l2_regularization', 0.0),
                class_weight=hp.get('class_weight', 'balanced'),
                random_state=hp.get('random_state', 42),
                verbose=0,
            )

        print(f"Initialized {self.backend} GBM with:")
        print(f"  n_estimators: {hp.get('n_estimators', 300)}")
        print(f"  learning_rate: {hp.get('learning_rate', 0.05)}")
        print(f"  max_depth: {hp.get('max_depth', 'default')}")

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:

        print("\n" + "="*50)
        print(f"Training Gradient Boosting Model ({self.backend})")
        print("="*50)

        self.model = self.build_model()

        print(f"\nTraining on {len(X_train)} samples...")
        start_time = time.time()

        # XGBoost and LightGBM accept eval_set for early stopping
        if self.backend in ('xgboost', 'lightgbm') and X_val is not None:
            early_stopping = self.hyperparameters.get('early_stopping_rounds', 30)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Evaluate
        print("\nEvaluating on training set...")
        train_metrics = self.evaluate(X_train, y_train)
        print("\nTraining Set Metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")

        self.training_history = {
            'train_metrics': train_metrics,
            'training_time': training_time,
            'n_train_samples': len(X_train),
            'hyperparameters': self.hyperparameters,
            'backend': self.backend,
        }

        if X_val is not None and y_val is not None:
            print("\nEvaluating on validation set...")
            val_metrics = self.evaluate(X_val, y_val)
            print("\nValidation Set Metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            self.training_history['val_metrics'] = val_metrics
            self.training_history['n_val_samples'] = len(X_val)

        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50 + "\n")

        return self.training_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        # HistGradientBoostingClassifier supports predict_proba natively
        return self.model.predict_proba(X)
