"""
Template for creating a new model trainer.

Copy this file and modify it to add your own model.
Replace 'YourModel' with your model name throughout.
"""

import numpy as np
from typing import Dict, Any, Optional
import time

from .base_trainer import BaseTrainer


class YourModelTrainer(BaseTrainer):
    """
    Trainer for YourModel.
    
    Brief description of what your model does and when to use it.
    
    Example:
        config = {
            'hyperparameters': {
                'param1': value1,
                'param2': value2,
            }
        }
        trainer = YourModelTrainer(config)
        history = trainer.train(X_train, y_train, X_val, y_val)
        trainer.save_model('./output')
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary with hyperparameters
        """
        super().__init__(config, model_name='your_model')
        self.hyperparameters = config.get('hyperparameters', {})
        
    def build_model(self):
        """
        Build and return your model instance.
        
        Returns:
            Initialized model object
        """
        # Extract hyperparameters with defaults
        param1 = self.hyperparameters.get('param1', default_value1)
        param2 = self.hyperparameters.get('param2', default_value2)
        random_state = self.hyperparameters.get('random_state', 42)
        
        # Import your model
        # from sklearn.xxx import YourModelClass
        # or
        # from your_package import YourModel
        
        model = YourModelClass(
            param1=param1,
            param2=param2,
            random_state=random_state,
            # ... other parameters
        )
        
        print(f"Initialized YourModel with:")
        print(f"  param1: {param1}")
        print(f"  param2: {param2}")
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train your model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training metrics and history
        """
        print("\n" + "="*50)
        print("Training YourModel")
        print("="*50)
        
        # Build model
        self.model = self.build_model()
        
        # Train model
        print(f"\nTraining on {len(X_train)} samples...")
        start_time = time.time()
        
        # Basic training
        self.model.fit(X_train, y_train)
        
        # If your model supports validation during training:
        # if X_val is not None:
        #     self.model.fit(X_train, y_train, 
        #                    eval_set=[(X_val, y_val)],
        #                    early_stopping_rounds=10,
        #                    verbose=False)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on training set
        print("\nEvaluating on training set...")
        train_metrics = self.evaluate(X_train, y_train)
        
        print("\nTraining Set Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Store training history
        self.training_history = {
            'train_metrics': train_metrics,
            'training_time': training_time,
            'n_train_samples': len(X_train),
            'hyperparameters': self.hyperparameters
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            print("\nEvaluating on validation set...")
            val_metrics = self.evaluate(X_val, y_val)
            
            print("\nValidation Set Metrics:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            self.training_history['val_metrics'] = val_metrics
            self.training_history['n_val_samples'] = len(X_val)
        
        # Model-specific information (optional)
        # Add any model-specific outputs you want to track
        self.training_history['model_specific'] = {
            # 'example_metric': self.model.some_attribute,
        }
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50 + "\n")
        
        return self.training_history
    
    # Optional: Override these methods if your model has specific behaviors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Override if your model has custom prediction logic.
        """
        return super().predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Override if your model has custom probability prediction logic.
        """
        return super().predict_proba(X)
    
    # Optional: Add model-specific utility methods
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get statistics specific to your model.
        
        Returns:
            Dictionary of model statistics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        stats = {
            'n_features': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None,
            # Add model-specific stats
            # 'example_stat': self.model.some_property,
        }
        
        return stats


# After creating your trainer:
# 1. Add to models/__init__.py:
#    from .your_model_trainer import YourModelTrainer
#    __all__ = [..., 'YourModelTrainer']
#
# 2. Register in train_classical_oddt.py:
#    def get_trainer(model_type, config):
#        ...
#        elif model_type == 'your_model':
#            return YourModelTrainer(config)
#
# 3. Create config file configs/your_model_config.yaml:
#    model_type: "your_model"
#    hyperparameters:
#      param1: value1
#      param2: value2
#
# 4. Test your trainer:
#    python train_classical_oddt.py --config ../configs/your_model_config.yaml
