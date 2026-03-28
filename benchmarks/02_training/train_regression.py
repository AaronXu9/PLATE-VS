"""
Regression training script for pChEMBL binding affinity prediction.

Mirrors train_classical_oddt.py but:
  - targets pChEMBL (continuous) instead of is_active (binary)
  - excludes decoys (no pChEMBL values)
  - uses regression trainers (RF/GBM/SVM Regressor)
  - reports RMSE/MAE/R²/Pearson/Spearman/CI

Usage:
  conda run -n rdkit_env python train_regression.py \\
      --config ../configs/regression_rf_config.yaml \\
      --registry ../../training_data_full/registry_soft_split_regression.csv \\
      --use-2d-split
"""
import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader
from features.featurizer import get_featurizer
from features.combined_featurizer import get_combined_featurizer
from models.rf_regressor import RandomForestRegressorTrainer
from models.gbm_regressor import GBMRegressorTrainer
from models.svm_regressor import SVMRegressorTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_regression_trainer(model_type: str, config: dict):
    if model_type == 'random_forest':
        return RandomForestRegressorTrainer(config)
    if model_type in ('gradient_boosting', 'gbm', 'xgboost', 'lightgbm'):
        return GBMRegressorTrainer(config)
    if model_type == 'svm':
        return SVMRegressorTrainer(config)
    raise ValueError(
        f"Unknown model_type: {model_type!r}. "
        "Supported: ['random_forest', 'gradient_boosting', 'svm']"
    )


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('regression_training')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(ch)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(Path(output_dir) / f'regression_{ts}.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    return logger


def train_regression(
    config_path: str,
    registry_path: str,
    output_dir: str = './trained_models_regression',
    use_2d_split: bool = False,
    quick_test: bool = False,
    test_samples: int = 500,
    cache_dir: Optional[str] = None,
) -> None:
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("Regression Training Pipeline — pChEMBL Prediction")
    logger.info("=" * 70)

    if quick_test:
        logger.warning(f"QUICK TEST MODE: {test_samples} samples")

    config = load_config(config_path)
    model_type = config['model_type']
    data_config = config.get('data', {})
    similarity_threshold = data_config.get('similarity_threshold', '0p7')
    include_protein_features = data_config.get('include_protein_features', True)

    if not use_2d_split:
        use_2d_split = data_config.get('use_2d_split', False)

    logger.info(f"Model: {model_type} | Threshold: {similarity_threshold} | 2D split: {use_2d_split}")

    logger.info(f"Loading registry: {registry_path}")
    loader = DataLoader(registry_path)
    loader.load_registry()

    def _load(split_name, protein_partition=None):
        data = loader.get_training_data(
            similarity_threshold=similarity_threshold,
            include_decoys=False,
            split=split_name,
            protein_partition=protein_partition,
        )
        if include_protein_features:
            smiles, y, pids = loader.prepare_features_labels(
                data, task='regression', include_protein_info=True
            )
        else:
            smiles, y = loader.prepare_features_labels(data, task='regression')
            pids = None
        return smiles, y, pids

    if use_2d_split:
        train_smiles, y_train, train_pids = _load('train', 'train')
        val_smiles,   y_val,   val_pids   = _load('test',  'val')
        test_smiles,  y_test,  test_pids  = _load('test',  'test')
    else:
        from sklearn.model_selection import train_test_split
        all_smiles, y_all, all_pids = _load('train')
        idx = np.arange(len(all_smiles))
        tr_idx, v_idx = train_test_split(idx, test_size=0.2, random_state=42)
        train_smiles = [all_smiles[i] for i in tr_idx]
        val_smiles   = [all_smiles[i] for i in v_idx]
        y_train, y_val = y_all[tr_idx], y_all[v_idx]
        train_pids = [all_pids[i] for i in tr_idx] if all_pids else None
        val_pids   = [all_pids[i] for i in v_idx]  if all_pids else None
        test_smiles, y_test, test_pids = _load('test')

    logger.info(f"Train: {len(train_smiles)} | Val: {len(val_smiles)} | Test: {len(test_smiles)}")

    if quick_test:
        caps = {'train': test_samples, 'val': test_samples // 5, 'test': test_samples // 2}
        rng = np.random.default_rng(42)
        for split_name in ('train', 'val', 'test'):
            if split_name == 'train':
                smiles_list = train_smiles
            elif split_name == 'val':
                smiles_list = val_smiles
            else:
                smiles_list = test_smiles
            cap = caps[split_name]
            if len(smiles_list) > cap:
                idx = rng.choice(len(smiles_list), cap, replace=False)
                if split_name == 'train':
                    train_smiles = [train_smiles[i] for i in idx]
                    y_train = y_train[idx]
                    if train_pids:
                        train_pids = [train_pids[i] for i in idx]
                elif split_name == 'val':
                    val_smiles = [val_smiles[i] for i in idx]
                    y_val = y_val[idx]
                    if val_pids:
                        val_pids = [val_pids[i] for i in idx]
                else:
                    test_smiles = [test_smiles[i] for i in idx]
                    y_test = y_test[idx]
                    if test_pids:
                        test_pids = [test_pids[i] for i in idx]

    feature_config = config.get('features', {})
    if include_protein_features and feature_config.get('type') == 'combined':
        ligand_config = feature_config.get('ligand', {'type': 'morgan_fingerprint', 'radius': 2, 'n_bits': 2048})
        protein_config = feature_config.get('protein', {'type': 'protein_identifier', 'embedding_dim': 32})
        featurizer = get_combined_featurizer(
            ligand_config=ligand_config,
            protein_config=protein_config,
            concatenation_method=feature_config.get('concatenation_method', 'concat'),
        )
        all_pids_combined = (train_pids or []) + (val_pids or []) + (test_pids or [])
        featurizer.fit_protein_featurizer(all_pids_combined)
        logger.info("Featurizing train/val...")
        X_train, _ = featurizer.featurize(train_smiles, protein_ids=train_pids, show_progress=True)
        X_val,   _ = featurizer.featurize(val_smiles,   protein_ids=val_pids,   show_progress=True)
        using_combined = True
    else:
        featurizer = get_featurizer(feature_config)
        logger.info("Featurizing train/val...")
        X_train, _ = featurizer.featurize(train_smiles, show_progress=True)
        X_val,   _ = featurizer.featurize(val_smiles,   show_progress=True)
        using_combined = False

    logger.info(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

    trainer = get_regression_trainer(model_type, config)
    trainer.train(X_train, y_train, X_val, y_val)

    del X_train, X_val
    gc.collect()

    logger.info("Featurizing test set...")
    if using_combined:
        X_test, _ = featurizer.featurize(test_smiles, protein_ids=test_pids, show_progress=True)
    else:
        X_test, _ = featurizer.featurize(test_smiles, show_progress=True)

    test_metrics = trainer.evaluate(X_test, y_test)
    logger.info("\nTest Set Metrics:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    trainer.training_history['test_metrics'] = test_metrics

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    if using_combined:
        featurizer.save_protein_mapping(str(out / f"{trainer.model_name}_protein_mapping.json"))
    with open(out / f"{trainer.model_name}_feature_config.json", 'w') as f:
        json.dump(featurizer.get_config(), f, indent=2)

    summary = {
        'model_type': model_type,
        'task': 'regression',
        'feature_type': featurizer.name,
        'similarity_threshold': similarity_threshold,
        'training_history': trainer.training_history,
        'data_config': data_config,
        'use_2d_split': use_2d_split,
    }
    with open(out / f"{trainer.model_name}_training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("Regression Training Complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train regression models for pChEMBL prediction')
    parser.add_argument('--config',       required=True)
    parser.add_argument('--registry',     required=True)
    parser.add_argument('--output',       default='./trained_models_regression')
    parser.add_argument('--use-2d-split', action='store_true')
    parser.add_argument('--quick-test',   action='store_true')
    parser.add_argument('--test-samples', type=int, default=500)
    parser.add_argument('--cache-dir',    default=None)
    args = parser.parse_args()
    train_regression(
        config_path=args.config,
        registry_path=args.registry,
        output_dir=args.output,
        use_2d_split=args.use_2d_split,
        quick_test=args.quick_test,
        test_samples=args.test_samples,
        cache_dir=args.cache_dir,
    )


if __name__ == '__main__':
    main()
