# Quick Start Guide

## Setup

### 1. Install Dependencies

```bash
# Navigate to the training directory
cd benchmarks/02_training

# Install required packages
pip install -r requirements.txt
```

Required packages:
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Machine learning models
- `rdkit` - Molecular featurization
- `pyyaml` - Configuration files
- `tqdm` - Progress bars
- `joblib` - Model persistence

### 2. Verify Installation

```bash
python -c "
from data.data_loader import DataLoader
from features.featurizer import MorganFingerprintFeaturizer
from models.rf_trainer import RandomForestTrainer
print('✓ All modules imported successfully!')
"
```

## Training Your First Model

### Option 1: Using the Command Line (Recommended)

Train with the precomputed train/val/test splits:

```bash
python train_classical_oddt.py \
  --config ../configs/classical_config.yaml \
  --registry ../../training_data_full/registry.csv \
  --output ./trained_models/my_first_model \
  --use-precomputed-split
```

### Option 2: Using Python Code

```python
from data.data_loader import DataLoader
from features.featurizer import MorganFingerprintFeaturizer
from models.rf_trainer import RandomForestTrainer

# Load data
loader = DataLoader('../../training_data_full/registry.csv')
train_data = loader.get_training_data(split='train', similarity_threshold='0p7')
smiles, labels = loader.prepare_features_labels(train_data)

# Generate features
featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
X, _ = featurizer.featurize(smiles)

# Split data
splits = loader.split_data(X, labels, val_size=0.2)

# Train model
config = {
    'hyperparameters': {
        'n_estimators': 100,
        'random_state': 42,
        'class_weight': 'balanced'
    }
}

trainer = RandomForestTrainer(config)
history = trainer.train(
    splits['X_train'], splits['y_train'],
    splits['X_val'], splits['y_val']
)

# Save model
trainer.save_model('./my_model')
```

## Configuration

Edit [`configs/classical_config.yaml`](../configs/classical_config.yaml) to customize:

```yaml
model_type: "random_forest"

hyperparameters:
  n_estimators: 100        # Number of trees
  max_depth: null          # Tree depth (null = unlimited)
  class_weight: 'balanced' # Handle class imbalance

features:
  type: "morgan_fingerprint"
  radius: 2
  n_bits: 2048

data:
  similarity_threshold: '0p7'
  include_decoys: true
```

## What Happens During Training

1. **Data Loading**: Reads registry.csv and filters by similarity threshold and split
2. **Feature Generation**: Converts SMILES to molecular fingerprints
3. **Model Training**: Trains Random Forest classifier
4. **Evaluation**: Computes metrics on validation/test sets
5. **Model Saving**: Saves model, config, and training history

## Output Files

After training, you'll find in the output directory:

```
trained_models/
├── random_forest.pkl                    # Trained model
├── random_forest_config.json            # Model hyperparameters
├── random_forest_history.json           # Training metrics
├── random_forest_feature_config.json    # Feature settings
└── random_forest_training_summary.json  # Complete summary
```

## Evaluation Metrics

The pipeline reports:

- **Accuracy**: Overall correct predictions
- **Precision**: Correct positive predictions / all positive predictions
- **Recall**: Correct positive predictions / all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Average Precision**: Area under precision-recall curve
- **MCC**: Matthews correlation coefficient (-1 to 1, 0 = random)

## Next Steps

### Add a New Model

See the [README](README.md#adding-new-models) for instructions on adding new model types (GBM, SVM, etc.)

### Hyperparameter Tuning

Modify the config file and train multiple models:

```bash
# Try different n_estimators
for n in 50 100 200; do
  python train_classical_oddt.py \
    --output ./models/rf_n${n} \
    --config <(cat ../configs/classical_config.yaml | \
              sed "s/n_estimators: 100/n_estimators: $n/")
done
```

### Compare Models

Train multiple configurations and compare metrics in the `*_training_summary.json` files.

## Troubleshooting

**Import Error: No module named 'sklearn'**
```bash
pip install scikit-learn
```

**Import Error: No module named 'rdkit'**
```bash
# Using conda (recommended for RDKit)
conda install -c conda-forge rdkit

# Or using pip
pip install rdkit
```

**Memory Error**
- Reduce `n_bits` in feature config (try 1024 or 512)
- Process fewer samples for testing
- Close other applications

**File Not Found: registry.csv**
- Check the path in `--registry` argument
- Make sure you're running from `benchmarks/02_training/`
- Verify the training_data_full directory exists

## Examples

See [`example_usage.py`](example_usage.py) for more detailed examples:

```bash
python example_usage.py
```

## Support

For issues or questions:
1. Check the main [README](README.md)
2. Review configuration in `configs/classical_config.yaml`
3. Verify data format matches expected structure
