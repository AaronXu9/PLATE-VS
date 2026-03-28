"""Abstract base class for regression model trainers."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import json
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.metrics import summarize_regression


class BaseRegressionTrainer(ABC):
    def __init__(self, config: Dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.training_history: Dict[str, Any] = {}

    @abstractmethod
    def build_model(self) -> Any: ...

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X).astype(np.float32)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return summarize_regression(y, y_pred)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def save_model(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / f"{self.model_name}.pkl")
        with open(path / f"{self.model_name}_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        with open(path / f"{self.model_name}_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Model saved to {path}")

    def load_model(self, load_dir: str) -> None:
        path = Path(load_dir)
        model_path = path / f"{self.model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)
        print(f"Model loaded from {path}")
