"""
Random Forest Trainer Module.

This module implements a Random Forest classifier trainer for
predicting binding affinity (active vs inactive).
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time

from .base_trainer import BaseTrainer


class RandomForestTrainer(BaseTrainer):
    """
    Trainer for Random Forest classification models.
    
    Random Forest is an ensemble learning method that operates by constructing
    multiple decision trees and outputting the class that is the mode of the
    classes of the individual trees.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Random Forest trainer.
        
        Args:
            config: Configuration dictionary with hyperparameters
        """
        super().__init__(config, model_name='random_forest')
        self.hyperparameters = config.get('hyperparameters', {})
        
    def build_model(self) -> RandomForestClassifier:
        """
        Build and return a Random Forest classifier.
        
        Returns:
            Initialized RandomForestClassifier
        """
        # Extract hyperparameters with defaults
        n_estimators = self.hyperparameters.get('n_estimators', 100)
        max_depth = self.hyperparameters.get('max_depth', None)
        min_samples_split = self.hyperparameters.get('min_samples_split', 2)
        min_samples_leaf = self.hyperparameters.get('min_samples_leaf', 1)
        max_features = self.hyperparameters.get('max_features', 'sqrt')
        bootstrap = self.hyperparameters.get('bootstrap', True)
        random_state = self.hyperparameters.get('random_state', 42)
        n_jobs = self.hyperparameters.get('n_jobs', -1)
        class_weight = self.hyperparameters.get('class_weight', 'balanced')
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
            verbose=0
        )
        
        print(f"Initialized Random Forest with:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  max_features: {max_features}")
        print(f"  class_weight: {class_weight}")
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training metrics and history
        """
        print("\n" + "="*50)
        print("Training Random Forest Model")
        print("="*50)
        
        # Build model
        self.model = self.build_model()
        
        # Train model
        print(f"\nTraining on {len(X_train)} samples...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
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
        
        # Feature importance
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            top_k = 10
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            print(f"\nTop {top_k} Most Important Features:")
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
            
            self.training_history['feature_importance_stats'] = {
                'mean': float(np.mean(feature_importance)),
                'std': float(np.std(feature_importance)),
                'max': float(np.max(feature_importance)),
                'min': float(np.min(feature_importance))
            }
        
        # Cross-validation if requested
        if self.config.get('cross_validation', False):
            cv_folds = self.config.get('cv_folds', 5)
            print(f"\nPerforming {cv_folds}-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=cv_folds, scoring='roc_auc', n_jobs=-1
            )
            print(f"CV ROC-AUC scores: {cv_scores}")
            print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            self.training_history['cv_scores'] = cv_scores.tolist()
            self.training_history['cv_mean'] = float(cv_scores.mean())
            self.training_history['cv_std'] = float(cv_scores.std())
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50 + "\n")
        
        return self.training_history
    
    def get_tree_depths(self) -> np.ndarray:
        """
        Get the depths of all trees in the forest.
        
        Returns:
            Array of tree depths
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return np.array([tree.get_depth() for tree in self.model.estimators_])
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the trained model.
        
        Returns:
            Dictionary of model statistics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        tree_depths = self.get_tree_depths()
        
        stats = {
            'n_estimators': self.model.n_estimators,
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_,
            'tree_depth_mean': float(np.mean(tree_depths)),
            'tree_depth_std': float(np.std(tree_depths)),
            'tree_depth_max': int(np.max(tree_depths)),
            'tree_depth_min': int(np.min(tree_depths))
        }
        
        if hasattr(self.model, 'oob_score_'):
            stats['oob_score'] = float(self.model.oob_score_)
        
        return stats
