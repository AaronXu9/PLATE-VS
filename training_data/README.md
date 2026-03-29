# Training Data Registry

This directory contains the unified training data registry for the VLS Benchmark Dataset.

## Overview

The registry provides a flexible data pointer system that supports multiple ML model architectures:
- **Random Forest / XGBoost**: Fingerprints + sequence features
- **Graph Neural Networks**: Molecular/protein graphs
- **Transformers**: Sequence-based models (ESM-2, ChemBERTa)
- **3D CNNs**: Voxelized protein-ligand structures

## Data Organization

### Train/Test Split Strategy

The dataset uses **similarity-based splitting** at multiple thresholds:

- **0p3** (30% similarity): Most dissimilar train/test split
- **0p5** (50% similarity): Moderate similarity
- **0p7** (70% similarity): Most similar train/test split

For each threshold:
- **Train set**: Compounds in `{uniprot}_active_{threshold}.sdf`
- **Test set**: ALL active compounds MINUS train set compounds

Example for UniProt O00141 at 0p3 threshold:
```
Train: plate-vs/.../sdf_filtered_by_ligand_similarity/O00141_active_0p3.sdf
Test:  All compounds in O00141_active.sdf NOT in O00141_active_0p3.sdf
```

### Decoy Compounds

Decoy (inactive) compounds are generated using DeepCoy and stored in:
```
plate-vs/.../deepcoy_output/{uniprot}_generated_decoys.txt
```

Decoys are SMILES-only (no 3D structures) and labeled as `is_active=False`.

## Registry Schema

The registry CSV contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | Unique identifier: `{uniprot}_{compound}_{threshold}_{split}` |
| `uniprot_id` | str | UniProt accession (e.g., "O00141") |
| `pdb_id` | str | Representative PDB structure ID |
| `cif_path` | str | Path to protein structure (mmCIF format) |
| `resolution` | float | Structure resolution (Å) |
| `quality_score` | float | Multi-criteria quality score (0-370) |
| `compound_id` | str | Compound identifier |
| `smiles` | str | Canonical SMILES string |
| `sdf_path` | str | Path to 3D ligand structure (SDF) or None |
| `pkl_path` | str | Path to RDKit molecule pickle or None |
| `is_active` | bool | True for actives, False for decoys |
| `affinity_value` | float | Binding affinity (IC50/Ki in nM) if available |
| `affinity_type` | str | Type of affinity measurement (IC50, Ki, etc.) |
| `similarity_threshold` | str | Split threshold ("0p3", "0p5", "0p7") or None for decoys |
| `split` | str | "train", "test", or "decoy" |
| `source` | str | "chembl" or "deepcoy" |

## Usage Example

### 1. Load Registry

```python
import pandas as pd

# Load full registry
registry = pd.read_csv('training_data/registry.csv')

# Filter for specific split
train_0p7 = registry[
    (registry['split'] == 'train') & 
    (registry['similarity_threshold'] == '0p7')
]

# Include decoys in training
train_with_decoys = registry[
    ((registry['split'] == 'train') & 
     (registry['similarity_threshold'] == '0p7')) |
    (registry['split'] == 'decoy')
]
```

### 2. Use with Dataloaders

```python
from dataloaders import FingerprintDataLoader

# For RF/XGBoost models
loader = FingerprintDataLoader(
    registry_csv='training_data/registry.csv',
    split='train',
    similarity_threshold='0p7',
    include_decoys=True,
    fingerprint_type='ecfp4',
    fp_bits=2048
)

# Access samples
sample = loader[0]
# Returns: {'features': ndarray, 'target': float, 'sample_id': str}
```

### 3. Extract Features On-Demand

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# Get sample from registry
row = registry.iloc[0]

# Generate fingerprint
mol = Chem.MolFromSmiles(row['smiles'])
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

# Load 3D structure if available
if row['sdf_path'] and pd.notna(row['sdf_path']):
    suppl = Chem.SDMolSupplier(row['sdf_path'])
    mol_3d = next(suppl)
```

## Dataset Statistics

Example from 3 UniProts (for testing):

```
Total samples: 17,273
  Active compounds: 2,253
  Decoy compounds: 15,020

Split distribution:
  train: 1,934 (11.2%)
  test: 319 (1.8%)
  decoy: 15,020 (87.0%)

Similarity thresholds (actives only):
  0p3: 751 (33.3%)
  0p5: 751 (33.3%)
  0p7: 751 (33.3%)
```

## Building the Registry

To rebuild or expand the registry:

```bash
# Full dataset (830 UniProts)
python benchmarks/01_preprocessing/build_training_registry.py \
    --affinity_dir plate-vs/VLS_benchmark/chembl_affinity \
    --structures_csv benchmarks/01_preprocessing/structure_selection_results/best_structures_per_uniprot.csv \
    --output_dir training_data \
    --include_decoys

# Test on subset
python benchmarks/01_preprocessing/build_training_registry.py \
    --affinity_dir plate-vs/VLS_benchmark/chembl_affinity \
    --structures_csv benchmarks/01_preprocessing/structure_selection_results/best_structures_per_uniprot.csv \
    --output_dir training_data_test \
    --max_uniprots 10 \
    --include_decoys
```

## Available Dataloaders

1. **FingerprintDataLoader** - For RF/XGBoost
   - Outputs: Concatenated protein + ligand fingerprint vectors
   - Configurable fingerprint types (ECFP4, MACCS)
   - Amino acid composition for protein

2. **GraphDataLoader** - For GNNs
   - Outputs: PyTorch Geometric Data objects
   - Molecular graphs with atom/bond features
   - Optional 3D coordinates

3. **SequenceDataLoader** - For Transformers
   - Outputs: Raw protein sequence + SMILES
   - For use with ESM-2, ChemBERTa, etc.

4. **Structure3DDataLoader** - For 3D CNNs (TODO)
   - Outputs: Voxelized protein-ligand complexes
   - Grid-based representations

## Feature Caching

Dataloaders support in-memory caching to speed up training:

```python
loader = FingerprintDataLoader(
    registry_csv='training_data/registry.csv',
    split='train',
    similarity_threshold='0p7',
    cache_features=True  # Enable caching
)
```

## Notes

- **Lazy Loading**: Features are computed on-demand, not pre-computed
- **Memory Efficiency**: Registry is just data pointers (~100 MB for full dataset)
- **Flexibility**: Easy to add new feature extraction methods
- **Reproducibility**: Same registry can be used across different models

## Related Files

- `build_training_registry.py` - Registry builder script
- `dataloaders.py` - PyTorch Dataset implementations
- `protein_references.json` - Protein metadata (structure quality, sequences)
- `registry.csv` - Main data registry
