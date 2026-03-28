"""
Base Trainer Class for ML Models.

This module provides an abstract base class for all model trainers,
ensuring a consistent interface across different model types.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)
import joblib
import json
from pathlib import Path


class BaseTrainer(ABC):
    """
    Abstract base class for all model trainers.
    
    This class defines the interface that all model trainers must implement,
    making it easy to add new models while maintaining consistency.
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary containing hyperparameters and settings
            model_name: Name of the model for saving/logging purposes
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.training_history = {}
        
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and return the model instance.
        
        Returns:
            The initialized model object
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training metrics and history
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_name} does not support probability predictions")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on the provided data.

        Args:
            X: Input features
            y: True labels

        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y, y_pred),
        }

        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_proba)
            metrics['avg_precision'] = average_precision_score(y, y_proba)
            metrics.update(compute_vs_metrics(y, y_proba))

        return metrics

    def save_model(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_path = save_path / f"{self.model_name}.pkl"
        joblib.dump(self.model, model_path)
        config_path = save_path / f"{self.model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        history_path = save_path / f"{self.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Model saved to {save_path}")

    def load_model(self, load_dir: str) -> None:
        load_path = Path(load_dir)
        model_path = load_path / f"{self.model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        config_path = load_path / f"{self.model_name}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        print(f"Model loaded from {load_path}")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


def compute_vs_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ef_fractions: List[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
    bedroc_alphas: List[float] = [20.0, 80.0, 160.0],
) -> Dict[str, float]:
    """
    Compute virtual-screening early-recognition metrics.

    Enrichment Factor (EF) at fraction χ:
        EF(χ) = (actives in top χ fraction) / (total actives × χ)
        EF=1  → same as random; EF=1/χ → perfect enrichment.

    BEDROC (Boltzmann-Enhanced Discrimination of ROC):
        Exponentially weights early ranks so that late-found actives
        are discounted. α controls the emphasis window:
            α=20  ≈ top 8% of list weighted
            α=80  ≈ top 2% of list weighted
            α=160 ≈ top 1% of list weighted
        Range: [BEDROC_min, 1], random ≈ 0.5 (depends on prevalence).

    Parameters
    ----------
    y_true   : binary labels (1=active, 0=decoy)
    y_score  : predicted probability of being active (higher = more active)
    ef_fractions  : fractions at which to compute EF
    bedroc_alphas : α values for BEDROC

    Returns
    -------
    dict of metric_name → float
    """
    from rdkit.ML.Scoring import Scoring

    # rdkit expects a list of (score, label) tuples sorted by score descending
    ranked = sorted(zip(y_score, y_true.astype(int)), reverse=True)

    metrics: Dict[str, float] = {}

    # Enrichment Factor
    ef_vals = Scoring.CalcEnrichment(ranked, col=1, fractions=ef_fractions)
    for frac, val in zip(ef_fractions, ef_vals):
        pct = int(frac * 100) if frac >= 0.01 else f"{frac*100:.1f}"
        metrics[f"ef_{pct}pct"] = round(float(val), 4)

    # BEDROC
    for alpha in bedroc_alphas:
        val = Scoring.CalcBEDROC(ranked, col=1, alpha=alpha)
        metrics[f"bedroc_a{int(alpha)}"] = round(float(val), 4)

    return metrics