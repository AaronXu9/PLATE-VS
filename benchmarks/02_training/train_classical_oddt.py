"""
Classical Training Script.
Models: Random Forest, Scikit-learn, ODDT scoring functions.

This script trains classical machine learning models for binding affinity prediction.
Currently supports:
- Random Forest (RF)

Future models to be added:
- Gradient Boosting (GBM)
- Support Vector Machines (SVM)
- ODDT scoring functions
"""

import argparse
import yaml
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import json
import logging
from datetime import datetime

# Optional WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader
from features.featurizer import get_featurizer
from features.combined_featurizer import get_combined_featurizer
from features.feature_cache import FeatureCache
from models.rf_trainer import RandomForestTrainer
from models.gbm_trainer import GBMTrainer
from models.svm_trainer import SVMTrainer


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_trainer(model_type: str, config: dict):
    """
    Factory function to get the appropriate trainer.
    
    Args:
        model_type: Type of model ('random_forest', 'gradient_boosting', etc.)
        config: Configuration dictionary
        
    Returns:
        Initialized trainer object
    """
    if model_type == 'random_forest':
        return RandomForestTrainer(config)
    elif model_type in ('gradient_boosting', 'gbm', 'xgboost', 'lightgbm'):
        return GBMTrainer(config)
    elif model_type == 'svm':
        return SVMTrainer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: ['random_forest', 'gradient_boosting', 'svm']")


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Setup logging to both file and console.
    
    Args:
        output_dir: Directory to save log files
        
    Returns:
        Configured logger
    """
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(output_dir) / f'training_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    return logger


def train_classical(config_path: str,
                    registry_path: str = '../../training_data_full/registry.csv',
                    output_dir: str = './trained_models',
                    use_precomputed_split: bool = False,
                    use_2d_split: bool = False,
                    use_wandb: bool = False,
                    wandb_project: str = 'binding-affinity',
                    quick_test: bool = False,
                    test_samples: int = 1000,
                    cache_dir: Optional[str] = None):
    """
    Main training function for classical models.

    Args:
        config_path: Path to the configuration file
        registry_path: Path to the registry.csv file
        output_dir: Directory to save trained models
        use_precomputed_split: Whether to use the split column from registry (True)
                              or create a new split (False)
        use_2d_split: When True, use a 2D split (protein cluster × ligand similarity).
                      Requires registry_2d_split.csv (produced by assign_protein_splits.py).
                      Train = protein_partition=train × split=train
                      Val   = protein_partition=val   × split=test
                      Test  = protein_partition=test  × split=test
        use_wandb: Whether to log to Weights & Biases
        wandb_project: WandB project name
        quick_test: Whether to run a quick test on small data subset
        test_samples: Number of samples to use in quick test mode
    """
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("="*70)
    logger.info("Classical ML Training Pipeline for Binding Affinity Prediction")
    logger.info("="*70)
    
    if quick_test:
        logger.warning(f"QUICK TEST MODE: Using only {test_samples} samples")
    
    # Load configuration first
    logger.info(f"\n1. Loading configuration from {config_path}")
    config = load_config(config_path)
    logger.info(f"   Model type: {config['model_type']}")
    
    # Initialize WandB if requested
    if use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("WandB not installed. Install with: pip install wandb")
            logger.warning("Continuing without WandB logging...")
            use_wandb = False
        else:
            logger.info("Initializing Weights & Biases...")
            wandb.init(
                project=wandb_project,
                config=config,
                tags=['quick_test'] if quick_test else []
            )
            logger.info(f"WandB run: {wandb.run.name}")
    
    # Resolve 2D split registry path
    if use_2d_split:
        registry_2d = Path(registry_path).parent / 'registry_2d_split.csv'
        if not registry_2d.exists():
            raise FileNotFoundError(
                f"2D split registry not found: {registry_2d}\n"
                f"Run benchmarks/01_preprocessing/assign_protein_splits.py first."
            )
        registry_path = str(registry_2d)
        logger.info(f"  [2D split] Using {registry_path}")

    # Initialize data loader
    logger.info(f"\n2. Loading data from {registry_path}")
    data_loader = DataLoader(registry_path)
    data_loader.load_registry()

    # Quick test mode: limit data size
    if quick_test:
        logger.info(f"Quick test mode: Limiting to {test_samples} samples per split")

    # Get training data configuration
    data_config = config.get('data', {})
    similarity_threshold = data_config.get('similarity_threshold', '0p7')
    include_decoys = data_config.get('include_decoys', True)
    include_protein_features = data_config.get('include_protein_features', False)

    # Override use_2d_split from config if not set via CLI
    if not use_2d_split:
        use_2d_split = data_config.get('use_2d_split', False)

    if use_2d_split:
        logger.info("  [2D split] Train = protein_partition=train × split=train")
        logger.info("  [2D split] Val   = protein_partition=val   × split=test")
        logger.info("  [2D split] Test  = protein_partition=test  × split=test")

    if use_precomputed_split or use_2d_split:
        print("\n3. Loading precomputed splits from registry")

        def _load_split(split_name, protein_partition=None):
            data = data_loader.get_training_data(
                similarity_threshold=similarity_threshold,
                include_decoys=include_decoys,
                split=split_name,
                protein_partition=protein_partition,
            )
            if include_protein_features:
                smiles, labels, pids = data_loader.prepare_features_labels(
                    data, include_protein_info=True
                )
            else:
                smiles, labels = data_loader.prepare_features_labels(data)
                pids = None
            return smiles, labels, pids

        if use_2d_split:
            # 2D split: protein cluster × ligand similarity
            # Train = protein_partition=train × ligand split=train
            # Val   = protein_partition=val   × ligand split=test
            # Test  = protein_partition=test  × ligand split=test
            train_smiles, y_train, train_protein_ids = _load_split('train', 'train')
            val_smiles,   y_val,   val_protein_ids   = _load_split('test',  'val')
            test_smiles,  y_test,  test_protein_ids  = _load_split('test',  'test')
        else:
            # 1D split: load train, create 80/20 train/val from it
            all_train_smiles, all_y_train, all_train_protein_ids = _load_split('train')

            from sklearn.model_selection import train_test_split
            print("\n   Creating train/val split from training data (80/20)...")
            indices = np.arange(len(all_train_smiles))
            train_indices, val_indices = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=all_y_train
            )
            train_smiles = [all_train_smiles[i] for i in train_indices]
            val_smiles   = [all_train_smiles[i] for i in val_indices]
            y_train = all_y_train[train_indices]
            y_val   = all_y_train[val_indices]
            if all_train_protein_ids:
                train_protein_ids = [all_train_protein_ids[i] for i in train_indices]
                val_protein_ids   = [all_train_protein_ids[i] for i in val_indices]
            else:
                train_protein_ids = None
                val_protein_ids   = None

            # Test data loaded later (after training) to keep peak memory low
            test_smiles, y_test, test_protein_ids = _load_split('test')

        print(f"   Train set: {len(train_smiles)} samples")
        print(f"   Val set:   {len(val_smiles)} samples")
        if use_2d_split:
            print(f"   Test set:  {len(test_smiles)} samples (loaded now for 2D split)")
        
        # Quick test: subsample training data
        if quick_test and len(train_smiles) > test_samples:
            logger.info(f"Subsampling training data from {len(train_smiles)} to {test_samples}")
            indices = np.random.choice(len(train_smiles), test_samples, replace=False)
            train_smiles = [train_smiles[i] for i in indices]
            y_train = y_train[indices]
            if train_protein_ids:
                train_protein_ids = [train_protein_ids[i] for i in indices]

        # Quick test: subsample validation data
        if quick_test and len(val_smiles) > test_samples // 5:
            val_cap = test_samples // 5
            logger.info(f"Subsampling validation data from {len(val_smiles)} to {val_cap}")
            indices = np.random.choice(len(val_smiles), val_cap, replace=False)
            val_smiles = [val_smiles[i] for i in indices]
            y_val = y_val[indices]
            if val_protein_ids:
                val_protein_ids = [val_protein_ids[i] for i in indices]

        # Quick test: subsample test data (only when already loaded, i.e. use_2d_split)
        if use_2d_split and quick_test and len(test_smiles) > test_samples // 2:
            test_cap = test_samples // 2
            logger.info(f"Subsampling test data from {len(test_smiles)} to {test_cap} (stratified)")
            from sklearn.model_selection import train_test_split as _tts
            keep_idx, _ = _tts(
                np.arange(len(test_smiles)), train_size=test_cap,
                random_state=42, stratify=y_test
            )
            test_smiles = [test_smiles[i] for i in keep_idx]
            y_test = y_test[keep_idx]
            if test_protein_ids:
                test_protein_ids = [test_protein_ids[i] for i in keep_idx]
        
    else:
        # Create custom train/val split
        print("\n3. Creating custom train/val split")
        all_data = data_loader.get_training_data(
            similarity_threshold=similarity_threshold,
            include_decoys=include_decoys,
            split=None
        )
        
        if include_protein_features:
            all_smiles, all_labels, all_protein_ids = data_loader.prepare_features_labels(
                all_data, include_protein_info=True
            )
        else:
            all_smiles, all_labels = data_loader.prepare_features_labels(all_data)
            all_protein_ids = None
    
    # Initialize featurizer
    print("\n4. Initializing featurizer")
    feature_config = config.get('features', {})
    feature_type = feature_config.get('type', 'morgan_fingerprint')
    
    # Check if we should use combined features
    if include_protein_features and feature_type == 'combined':
        # Use combined ligand + protein features
        ligand_config = feature_config.get('ligand', {
            'type': 'morgan_fingerprint',
            'radius': 2,
            'n_bits': 2048
        })
        protein_config = feature_config.get('protein', {
            'type': 'protein_identifier',
            'embedding_dim': 32
        })
        concatenation_method = feature_config.get('concatenation_method', 'concat')
        
        featurizer = get_combined_featurizer(
            ligand_config=ligand_config,
            protein_config=protein_config,
            concatenation_method=concatenation_method
        )
        print(f"   Featurizer: {featurizer.name}")
        using_combined_features = True
    else:
        # Use ligand-only features
        featurizer = get_featurizer(feature_config)
        print(f"   Featurizer: {featurizer.name}")
        if include_protein_features:
            print("   Warning: include_protein_features=True but feature type is not 'combined'")
            print("   Using ligand-only features. Set features.type='combined' to use protein features.")
        using_combined_features = False
    
    # Build feature cache (if requested)
    ligand_cache = None
    if cache_dir is not None:
        ligand_feat_config = (
            featurizer.ligand_featurizer.get_config()
            if using_combined_features
            else featurizer.get_config()
        )
        ligand_cache = FeatureCache(cache_dir, ligand_feat_config)
        n_cached = ligand_cache.count()
        logger.info(f"   Feature cache: {ligand_cache.cache_path} ({n_cached:,} entries)")

    # Generate features
    # NOTE: Test set is featurized AFTER training to avoid holding all matrices
    # in memory simultaneously (would OOM on large datasets).
    if use_precomputed_split or use_2d_split:
        print("\n5. Generating features for train/val splits")

        if using_combined_features:
            # Fit protein featurizer on all protein IDs (including test, for mapping)
            all_protein_ids = (train_protein_ids or []) + (val_protein_ids or []) + (test_protein_ids or [])
            featurizer.fit_protein_featurizer(all_protein_ids)

            print("   - Training set...")
            X_train, invalid_train = featurizer.featurize(
                train_smiles, protein_ids=train_protein_ids, show_progress=True,
                ligand_cache=ligand_cache,
            )

            print("   - Validation set...")
            X_val, invalid_val = featurizer.featurize(
                val_smiles, protein_ids=val_protein_ids, show_progress=True,
                ligand_cache=ligand_cache,
            )
        else:
            print("   - Training set...")
            X_train, invalid_train = featurizer.featurize(
                train_smiles, show_progress=True, cache=ligand_cache
            )

            print("   - Validation set...")
            X_val, invalid_val = featurizer.featurize(
                val_smiles, show_progress=True, cache=ligand_cache
            )

        print(f"\n   Feature matrix shapes:")
        print(f"   - Train: {X_train.shape}")
        print(f"   - Val:   {X_val.shape}")
        if use_2d_split:
            print(f"   - Test:  (deferred until after training to save memory)")
        else:
            print(f"   - Test:  (deferred until after training to save memory)")
    else:
        print("\n5. Generating features")
        
        if using_combined_features:
            # Fit protein featurizer on all protein IDs
            featurizer.fit_protein_featurizer(all_protein_ids)
            
            X_all, invalid_indices = featurizer.featurize(
                all_smiles, protein_ids=all_protein_ids, show_progress=True
            )
        else:
            X_all, invalid_indices = featurizer.featurize(all_smiles, show_progress=True)
        
        print(f"   Feature matrix shape: {X_all.shape}")
        
        # Split data
        val_size = data_config.get('val_split', 0.2)
        test_size = data_config.get('test_split', 0.0)
        random_state = config.get('hyperparameters', {}).get('random_state', 42)
        
        splits = data_loader.split_data(
            X_all, all_labels,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            stratify=True
        )
        
        X_train = splits['X_train']
        y_train = splits['y_train']
        X_val = splits['X_val']
        y_val = splits['y_val']
        
        if 'X_test' in splits:
            X_test = splits['X_test']
            y_test = splits['y_test']
        else:
            X_test, y_test = None, None
    
    # Initialize trainer
    print("\n6. Initializing trainer")
    model_type = config['model_type']
    trainer = get_trainer(model_type, config)
    
    # Train model
    logger.info("\n7. Training model")
    training_history = trainer.train(X_train, y_train, X_val, y_val)

    # Free train/val feature matrices before featurizing test — avoids OOM on large datasets
    import gc
    del X_train, X_val
    gc.collect()
    logger.info("   Released train/val feature matrices from memory")

    # Log to WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'train_accuracy': training_history['train_metrics']['accuracy'],
            'train_roc_auc': training_history['train_metrics']['roc_auc'],
            'train_f1': training_history['train_metrics']['f1_score'],
            'val_accuracy': training_history.get('val_metrics', {}).get('accuracy', 0),
            'val_roc_auc': training_history.get('val_metrics', {}).get('roc_auc', 0),
            'val_f1': training_history.get('val_metrics', {}).get('f1_score', 0),
        })

    # Featurize test set now (deferred to keep peak memory low)
    # For 2D split, test data was already loaded above but still needs featurization.
    if use_precomputed_split or use_2d_split:
        logger.info("\n8. Featurizing test set")
        if using_combined_features:
            print("   - Test set...")
            X_test, _ = featurizer.featurize(
                test_smiles, protein_ids=test_protein_ids, show_progress=True,
                ligand_cache=ligand_cache,
            )
        else:
            print("   - Test set...")
            X_test, _ = featurizer.featurize(test_smiles, show_progress=True, cache=ligand_cache)
        print(f"   Test feature matrix: {X_test.shape}")

    # Evaluate on test set if available
    if use_precomputed_split or use_2d_split or X_test is not None:
        logger.info("\n9. Evaluating on test set")
        test_metrics = trainer.evaluate(X_test, y_test)
        
        logger.info("\nTest Set Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        training_history['test_metrics'] = test_metrics
        training_history['n_test_samples'] = len(y_test)
        
        # Log to WandB
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'test_accuracy': test_metrics['accuracy'],
                'test_roc_auc': test_metrics['roc_auc'],
                'test_f1': test_metrics['f1_score'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
            })
    
    # Save model
    print(f"\n10. Saving model to {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save protein mapping if using combined features
    if using_combined_features:
        protein_mapping_path = output_path / f"{trainer.model_name}_protein_mapping.json"
        featurizer.save_protein_mapping(str(protein_mapping_path))
    
    trainer.save_model(output_dir)
    
    # Save feature configuration
    feature_config_path = output_path / f"{trainer.model_name}_feature_config.json"
    with open(feature_config_path, 'w') as f:
        json.dump(featurizer.get_config(), f, indent=2)
    
    # Save training summary
    summary = {
        'model_type': model_type,
        'feature_type': featurizer.name,
        'similarity_threshold': similarity_threshold,
        'training_history': training_history,
        'data_config': data_config,
        'use_precomputed_split': use_precomputed_split,
        'use_2d_split': use_2d_split,
        'split_strategy': '2d_protein_cluster_x_ligand_similarity' if use_2d_split else (
            '1d_ligand_similarity_precomputed' if use_precomputed_split else '1d_random'
        ),
    }
    
    summary_path = output_path / f"{trainer.model_name}_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("="*70)
    
    # Finish WandB run
    if use_wandb and WANDB_AVAILABLE:
        # Save model to WandB
        wandb.save(str(Path(output_dir) / f"{trainer.model_name}.pkl"))
        wandb.finish()
        logger.info("WandB run finished")
    
    return trainer


def main():
    """
    Command-line interface for training.
    """
    parser = argparse.ArgumentParser(
        description='Train classical ML models for binding affinity prediction'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/classical_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--registry',
        type=str,
        default='../../training_data_full/registry.csv',
        help='Path to registry.csv file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./trained_models',
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--use-precomputed-split',
        action='store_true',
        help='Use the train/val/test split from registry instead of creating new split'
    )
    parser.add_argument(
        '--use-2d-split',
        action='store_true',
        help='Use 2D split (protein cluster × ligand similarity). '
             'Requires registry_2d_split.csv — run assign_protein_splits.py first.'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Log experiment to Weights & Biases'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='binding-affinity',
        help='WandB project name'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test on small data subset (1000 samples)'
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        default=1000,
        help='Number of samples to use in quick test mode'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Directory for precomputed feature cache (e.g. ../../training_data_full/feature_cache)'
    )

    args = parser.parse_args()

    train_classical(
        config_path=args.config,
        registry_path=args.registry,
        output_dir=args.output,
        use_precomputed_split=args.use_precomputed_split,
        use_2d_split=args.use_2d_split,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        quick_test=args.quick_test,
        test_samples=args.test_samples,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
