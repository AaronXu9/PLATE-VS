"""
SVM Trainer Module.

Uses sklearn's LinearSVC for scalability on large datasets (millions of samples).
Probability estimates are obtained via CalibratedClassifierCV.

For small datasets (< 50k samples), you can set config['kernel'] = 'rbf' to
use the full kernelized SVC instead.
"""

import numpy as np
from typing import Dict, Any, Optional
import time

from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from .base_trainer import BaseTrainer


class SVMTrainer(BaseTrainer):
    """
    SVM trainer for binding affinity prediction.

    LinearSVC (default) scales to millions of samples; kernel SVC is available
    for smaller datasets where non-linear decision boundaries matter more.

    Features are auto-scaled (zero mean, unit variance) since SVMs are
    sensitive to feature magnitude.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name='svm')
        self.hyperparameters = config.get('hyperparameters', {})
        self.scaler = StandardScaler(with_mean=False)  # sparse-safe

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def build_model(self):
        hp = self.hyperparameters
        kernel = hp.get('kernel', 'linear')
        C = hp.get('C', 1.0)
        class_weight = hp.get('class_weight', 'balanced')
        random_state = hp.get('random_state', 42)
        max_iter = hp.get('max_iter', 2000)

        if kernel == 'linear':
            base = LinearSVC(
                C=C,
                class_weight=class_weight,
                random_state=random_state,
                max_iter=max_iter,
                dual='auto',
            )
            # Wrap in CalibratedClassifierCV to get predict_proba
            model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
        else:
            # Full kernelized SVC — only practical for < ~50k samples
            model = SVC(
                kernel=kernel,
                C=C,
                gamma=hp.get('gamma', 'scale'),
                class_weight=class_weight,
                random_state=random_state,
                max_iter=max_iter,
                probability=True,
            )

        print(f"Initialized SVM with:")
        print(f"  kernel: {kernel}")
        print(f"  C: {C}")
        print(f"  class_weight: {class_weight}")

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:

        print("\n" + "="*50)
        print("Training SVM Model")
        print("="*50)

        # Scale features — critical for SVM performance
        print(f"\nScaling features (n_features={X_train.shape[1]})...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        self.model = self.build_model()

        print(f"\nTraining on {len(X_train)} samples...")
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Store scaled data for evaluation calls (override predict to scale)
        print("\nEvaluating on training set...")
        train_metrics = self._evaluate_scaled(X_train_scaled, y_train)
        print("\nTraining Set Metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")

        self.training_history = {
            'train_metrics': train_metrics,
            'training_time': training_time,
            'n_train_samples': len(X_train),
            'hyperparameters': self.hyperparameters,
        }

        if X_val is not None and y_val is not None:
            print("\nEvaluating on validation set...")
            val_metrics = self._evaluate_scaled(X_val_scaled, y_val)
            print("\nValidation Set Metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            self.training_history['val_metrics'] = val_metrics
            self.training_history['n_val_samples'] = len(X_val)

        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50 + "\n")

        return self.training_history

    # ------------------------------------------------------------------
    # Override predict/evaluate to apply scaling automatically
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(self.scaler.transform(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return self._evaluate_scaled(self.scaler.transform(X), y)

    # ------------------------------------------------------------------
    # Internal helper — evaluate on pre-scaled data
    # ------------------------------------------------------------------

    def _evaluate_scaled(self, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, matthews_corrcoef
        )
        from .base_trainer import compute_vs_metrics

        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'avg_precision': average_precision_score(y, y_proba),
        }
        metrics.update(compute_vs_metrics(y, y_proba))
        return metrics
