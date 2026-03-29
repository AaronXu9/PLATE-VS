"""
Example script demonstrating how to use the training pipeline.

This script shows various ways to train and evaluate models.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader
from features.featurizer import MorganFingerprintFeaturizer, get_featurizer
from models.rf_trainer import RandomForestTrainer


def example_basic_training():
    """
    Example 1: Basic training with default settings.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Random Forest Training")
    print("="*70 + "\n")
    
    # Load data
    loader = DataLoader('../../training_data_full/registry.csv')
    loader.load_registry()
    
    # Get training data
    train_data = loader.get_training_data(
        similarity_threshold='0p7',
        include_decoys=True,
        split='train'
    )
    
    # Prepare features and labels
    smiles, labels = loader.prepare_features_labels(train_data)
    
    # Generate Morgan fingerprints
    featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
    X, invalid = featurizer.featurize(smiles[:1000])  # Use first 1000 for demo
    y = labels[:1000]
    
    # Split data
    splits = loader.split_data(X, y, val_size=0.2, random_state=42)
    
    # Train Random Forest
    config = {
        'hyperparameters': {
            'n_estimators': 50,  # Reduced for demo
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
    }
    
    trainer = RandomForestTrainer(config)
    history = trainer.train(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val']
    )
    
    # Save model
    trainer.save_model('./example_models/basic_rf')
    
    print("\n✓ Model trained and saved successfully!")


def example_with_config():
    """
    Example 2: Training using configuration file.
    """
    print("\n" + "="*70)
    print("Example 2: Training with Configuration File")
    print("="*70 + "\n")
    
    import yaml
    
    # Load config
    with open('../configs/classical_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded:")
    print(f"  Model: {config['model_type']}")
    print(f"  Features: {config['features']['type']}")
    print(f"  n_estimators: {config['hyperparameters']['n_estimators']}")
    
    # Initialize components
    loader = DataLoader('../../training_data_full/registry.csv')
    loader.load_registry()
    
    # Get data
    train_data = loader.get_training_data(
        similarity_threshold=config['data']['similarity_threshold'],
        include_decoys=config['data']['include_decoys'],
        split='train'
    )
    
    smiles, labels = loader.prepare_features_labels(train_data)
    
    # Use factory function for featurizer
    featurizer = get_featurizer(config['features'])
    X, invalid = featurizer.featurize(smiles[:1000])
    y = labels[:1000]
    
    # Split and train
    splits = loader.split_data(X, y, val_size=0.2, random_state=42)
    
    trainer = RandomForestTrainer(config)
    history = trainer.train(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val']
    )
    
    trainer.save_model('./example_models/config_rf')
    
    print("\n✓ Training with config completed!")


def example_custom_features():
    """
    Example 3: Using different feature types.
    """
    print("\n" + "="*70)
    print("Example 3: Custom Feature Configuration")
    print("="*70 + "\n")
    
    loader = DataLoader('../../training_data_full/registry.csv')
    loader.load_registry()
    
    train_data = loader.get_training_data(
        similarity_threshold='0p7',
        split='train'
    )
    
    smiles, labels = loader.prepare_features_labels(train_data)
    smiles_subset = smiles[:500]
    labels_subset = labels[:500]
    
    # Try different feature configurations
    feature_configs = [
        {'type': 'morgan_fingerprint', 'radius': 2, 'n_bits': 1024},
        {'type': 'morgan_fingerprint', 'radius': 3, 'n_bits': 2048},
    ]
    
    for feat_config in feature_configs:
        print(f"\nTesting: {feat_config}")
        featurizer = get_featurizer(feat_config)
        X, invalid = featurizer.featurize(smiles_subset, show_progress=False)
        print(f"  Feature shape: {X.shape}")
        print(f"  Invalid molecules: {len(invalid)}")


def example_model_evaluation():
    """
    Example 4: Comprehensive model evaluation.
    """
    print("\n" + "="*70)
    print("Example 4: Model Evaluation and Analysis")
    print("="*70 + "\n")
    
    # Load and prepare data
    loader = DataLoader('../../training_data_full/registry.csv')
    loader.load_registry()
    
    train_data = loader.get_training_data(split='train', similarity_threshold='0p7')
    val_data = loader.get_training_data(split='val', similarity_threshold='0p7', include_decoys=False)
    
    train_smiles, y_train = loader.prepare_features_labels(train_data)
    val_smiles, y_val = loader.prepare_features_labels(val_data)
    
    # Featurize
    featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
    X_train, _ = featurizer.featurize(train_smiles[:1000])
    X_val, _ = featurizer.featurize(val_smiles[:200])
    y_train = y_train[:1000]
    y_val = y_val[:200]
    
    # Train
    config = {
        'hyperparameters': {
            'n_estimators': 50,
            'random_state': 42,
            'class_weight': 'balanced'
        },
        'cross_validation': True,
        'cv_folds': 3
    }
    
    trainer = RandomForestTrainer(config)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Get detailed statistics
    print("\n" + "-"*50)
    print("Model Statistics:")
    print("-"*50)
    stats = trainer.get_model_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Feature importance
    importance = trainer.get_feature_importance()
    if importance is not None:
        print(f"\nFeature Importance:")
        print(f"  Top feature importance: {importance.max():.4f}")
        print(f"  Mean importance: {importance.mean():.4f}")
        print(f"  Non-zero features: {(importance > 0).sum()}/{len(importance)}")
    
    print("\n✓ Evaluation completed!")


def main():
    """
    Run all examples.
    """
    print("\n" + "="*70)
    print("Training Pipeline Examples")
    print("="*70)
    
    examples = [
        ("Basic Training", example_basic_training),
        ("Config-based Training", example_with_config),
        ("Custom Features", example_custom_features),
        ("Model Evaluation", example_model_evaluation),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nSelect an example to run (1-4, or 'all'):")
    choice = input("> ").strip().lower()
    
    if choice == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n✗ Error in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        idx = int(choice) - 1
        name, func = examples[idx]
        try:
            func()
        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        print("Invalid choice. Running all examples...")
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n✗ Error in {name}: {e}")


if __name__ == "__main__":
    # For non-interactive execution, run example 3 (fastest)
    try:
        example_custom_features()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("  1. Installed requirements: pip install -r requirements.txt")
        print("  2. Registry file at: ../../training_data_full/registry.csv")
