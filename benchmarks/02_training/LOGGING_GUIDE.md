# Logging and Experiment Tracking Guide

This guide covers the logging mechanisms and experiment tracking features in the training pipeline.

## 📋 Table of Contents
- [Structured Logging](#structured-logging)
- [WandB Integration](#wandb-integration)
- [Quick Test Mode](#quick-test-mode)
- [Log Files](#log-files)

## 📝 Structured Logging

The pipeline uses Python's built-in `logging` module for structured logging to both console and file.

### Features
- **Console Logging**: Real-time feedback during training
- **File Logging**: Persistent logs saved to `trained_models/training_TIMESTAMP.log`
- **Log Levels**: INFO for general info, DEBUG for detailed debugging, WARNING for issues

### Example Output
```
2026-01-19 10:30:45 - INFO - ======================================================================
2026-01-19 10:30:45 - INFO - Classical ML Training Pipeline for Binding Affinity Prediction
2026-01-19 10:30:45 - INFO - ======================================================================
2026-01-19 10:30:45 - INFO - 
2026-01-19 10:30:45 - INFO - 1. Loading configuration from ../configs/classical_config.yaml
2026-01-19 10:30:45 - INFO -    Model type: random_forest
```

## 🔬 WandB Integration

[Weights & Biases (WandB)](https://wandb.ai) is a powerful experiment tracking tool that logs:
- Training/validation/test metrics
- Model hyperparameters
- System metrics (GPU, CPU usage)
- Model artifacts

### Installation
```bash
pip install wandb
```

### First-Time Setup
```bash
# Login to WandB (creates API key)
wandb login
```

### Usage

#### Basic Training with WandB
```bash
python train_classical_oddt.py \
    --config ../configs/classical_config.yaml \
    --registry ../../training_data_full/registry.csv \
    --use-precomputed-split \
    --wandb \
    --wandb-project my-binding-affinity-project
```

#### Python API
```python
from train_classical_oddt import train_classical

trainer = train_classical(
    config_path='../configs/classical_config.yaml',
    registry_path='../../training_data_full/registry.csv',
    use_precomputed_split=True,
    use_wandb=True,
    wandb_project='my-project'
)
```

### Logged Metrics

#### Training Metrics
- `train_accuracy`: Training set accuracy
- `train_roc_auc`: Training set ROC-AUC score
- `train_f1`: Training set F1 score

#### Validation Metrics
- `val_accuracy`: Validation set accuracy
- `val_roc_auc`: Validation set ROC-AUC score
- `val_f1`: Validation set F1 score

#### Test Metrics (if available)
- `test_accuracy`: Test set accuracy
- `test_roc_auc`: Test set ROC-AUC score
- `test_f1`: Test set F1 score
- `test_precision`: Test set precision
- `test_recall`: Test set recall

### Viewing Results
After training completes:
1. Visit https://wandb.ai
2. Navigate to your project
3. View metrics, charts, and saved models

### Automatic Model Saving
Models are automatically uploaded to WandB:
```python
# Models saved to WandB artifacts
wandb.save('trained_models/random_forest_model.pkl')
```

## ⚡ Quick Test Mode

Quick test mode allows you to verify the pipeline works correctly on a small subset of data before running full training.

### Why Quick Test?
- **Fast iteration**: Test code changes in seconds, not hours
- **Debugging**: Quickly identify issues before full training
- **Resource efficient**: Use minimal compute resources
- **Sanity checks**: Verify pipeline integrity

### Usage

#### Command Line
```bash
# Quick test with default 1000 samples
python train_classical_oddt.py \
    --config ../configs/classical_config.yaml \
    --use-precomputed-split \
    --quick-test

# Custom sample size
python train_classical_oddt.py \
    --config ../configs/classical_config.yaml \
    --use-precomputed-split \
    --quick-test \
    --test-samples 500
```

#### Python API
```python
from train_classical_oddt import train_classical

# Quick test with 500 samples
trainer = train_classical(
    config_path='../configs/classical_config.yaml',
    registry_path='../../training_data_full/registry.csv',
    use_precomputed_split=True,
    quick_test=True,
    test_samples=500
)
```

### What Happens in Quick Test Mode?
1. **Training Set**: Limited to `test_samples` (default 1000)
2. **Validation Set**: Limited to `test_samples // 5` (default 200)
3. **Test Set**: Not subsampled (uses full test set if available)
4. **WandB Tag**: Automatically tagged as `quick_test` in WandB

### Example Output
```
2026-01-19 10:30:45 - WARNING - QUICK TEST MODE: Using only 1000 samples
2026-01-19 10:31:23 - INFO - Subsampling training data from 50000 to 1000
2026-01-19 10:31:24 - INFO - Subsampling validation data from 10000 to 200
```

## 📊 Log Files

### Location
Log files are saved to `OUTPUT_DIR/training_TIMESTAMP.log`

Example: `trained_models/training_20260119_103045.log`

### Content
Log files contain:
- Configuration details
- Data loading information
- Feature generation progress
- Training progress and metrics
- Validation and test results
- Error messages and warnings

### Example Log File
```
2026-01-19 10:30:45,123 - training - INFO - ======================================================================
2026-01-19 10:30:45,123 - training - INFO - Classical ML Training Pipeline for Binding Affinity Prediction
2026-01-19 10:30:45,123 - training - INFO - ======================================================================
2026-01-19 10:30:45,234 - training - INFO - 1. Loading configuration from ../configs/classical_config.yaml
2026-01-19 10:30:45,235 - training - INFO -    Model type: random_forest
2026-01-19 10:30:45,456 - training - INFO - 2. Loading data from ../../training_data_full/registry.csv
2026-01-19 10:30:46,789 - training - INFO - Loaded 75000 rows from registry
...
```

## 🔄 Complete Workflow Examples

### Example 1: Quick Test with WandB
```bash
# Quickly test pipeline and log to WandB
python train_classical_oddt.py \
    --config ../configs/classical_config.yaml \
    --use-precomputed-split \
    --quick-test \
    --wandb \
    --wandb-project binding-affinity-dev
```

### Example 2: Full Training with WandB
```bash
# Full training run with experiment tracking
python train_classical_oddt.py \
    --config ../configs/classical_config.yaml \
    --use-precomputed-split \
    --wandb \
    --wandb-project binding-affinity-production
```

### Example 3: Quick Test without WandB
```bash
# Fast local test without cloud logging
python train_classical_oddt.py \
    --config ../configs/classical_config.yaml \
    --use-precomputed-split \
    --quick-test
```

## 🧪 Automated Testing

Run automated tests to verify the pipeline:

```bash
# Run all tests
python test_training.py

# Run specific test
python test_training.py --test basic    # Basic ligand-only training
python test_training.py --test protein  # Protein-aware training
python test_training.py --test split    # Custom data split
python test_training.py --test save     # Model save/load
```

## 🎯 Best Practices

1. **Always start with quick test**: Verify your setup before full training
   ```bash
   python train_classical_oddt.py --quick-test
   ```

2. **Use WandB for production runs**: Track experiments systematically
   ```bash
   python train_classical_oddt.py --wandb
   ```

3. **Check log files for issues**: Review logs if training fails
   ```bash
   tail -f trained_models/training_*.log
   ```

4. **Run automated tests after changes**: Ensure nothing broke
   ```bash
   python test_training.py
   ```

## 📝 Troubleshooting

### WandB Not Available
If you see:
```
WARNING - WandB not installed. Install with: pip install wandb
WARNING - Continuing without WandB logging...
```

Solution:
```bash
pip install wandb
wandb login
```

### Quick Test Still Too Slow
Reduce sample size:
```bash
python train_classical_oddt.py --quick-test --test-samples 100
```

### Log Files Growing Too Large
Log files are automatically timestamped. Clean old logs:
```bash
rm trained_models/training_2026*.log
```
