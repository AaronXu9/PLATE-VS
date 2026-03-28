# VLS Benchmarking Guide

This guide explains how to use the PLATE-VS dataset and the three benchmarking methods
implemented in this repository: classical ML models, deep learning models, and
structure-based molecular docking with GNINA.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Classical ML Benchmark (RF / GBM / SVM)](#4-classical-ml-benchmark)
5. [Deep Learning Benchmark (DeepPurpose / GraphDTA)](#5-deep-learning-benchmark)
6. [Structure-Based Docking Benchmark (GNINA)](#6-structure-based-docking-benchmark)
7. [Generating a Comparison Report](#7-generating-a-comparison-report)
8. [Dataset Schema Reference](#8-dataset-schema-reference)
9. [Adding a New Method](#9-adding-a-new-method)

---

## 1. Dataset Overview

The **PLATE-VS** dataset is a large-scale protein–ligand virtual screening benchmark
derived from the ChEMBL database. It covers **826 UniProt targets** and contains:

| Split | Proteins | Actives | Decoys | Total |
|-------|----------|---------|--------|-------|
| Train | 578 | ~930K | ~1.23M | ~2.16M |
| Val | 124 | ~233K | ~308K | ~541K |
| Test | 124 | ~127K | ~1.96M | ~2.09M |

Decoys are generated with **DeepCoy**, a deep-learning decoy generator that produces
chemically realistic decoys matched to each active's property profile.

### Similarity Thresholds

The dataset provides three **ligand-similarity splits** controlling how "hard" the
benchmark is. All runs in this repo use `0p7` (Tanimoto ≥ 0.7):

| Threshold | Key | Description |
|-----------|-----|-------------|
| 0.3 | `0p3` | Easy — many similar ligands in train |
| 0.5 | `0p5` | Medium |
| 0.7 | `0p7` | Hard — test compounds are dissimilar to training set |

### Data Files

```
training_data_full/
├── registry_2d_split.csv   # Main registry (~978 MB) — all samples with split labels
├── registry.csv            # Original flat registry (~951 MB)
└── protein_references.json # Per-protein metadata (PDB ID, resolution, pocket info)
```

### The 2D Split Strategy

The **2D split** (`registry_2d_split.csv`) applies two independent splitting axes
simultaneously, creating a benchmark that is substantially harder than either axis
alone.

#### Axis 1 — Protein partition (`protein_partition` column)

Proteins are clustered by sequence similarity (70 % query-coverage). Each cluster is
assigned entirely to one of three partitions — **train / val / test protein families
are completely disjoint**. A model trained on train-partition proteins never sees the
sequence space of test-partition proteins during training.

#### Axis 2 — Ligand similarity split (`split` column)

For **every protein, regardless of which protein partition it belongs to**, its known
actives are further split by pairwise Tanimoto fingerprint similarity at the chosen
threshold:

- `split=train` — the "core" actives, forming a chemically coherent cluster.
- `split=test` — actives that are **dissimilar** (Tanimoto < threshold) to all
  `split=train` actives for that protein. Held out from model training.
- `split=decoy` — DeepCoy decoys for this protein.

Concretely, a train-partition protein such as O00141 (0p7 threshold) contributes:
- **457 actives** (`split=train`) → model training set
- **179 actives** (`split=test`) → test set, *even though the protein is a training protein*

A test-partition protein such as O14684 contributes:
- **864 actives** (`split=train`) and **775 actives** (`split=test`) → both go into
  the test set because the protein itself is unseen.

#### What each part of the test set measures

The combined test set therefore probes two distinct failure modes:

| Test sample origin | Protein seen? | Ligand similar to train? | What it tests |
|--------------------|---------------|--------------------------|---------------|
| `protein_partition=train`, `split=test` | Yes | No | Chemical scaffold generalisation |
| `protein_partition=test`, `split=train` | No | n/a | Protein family generalisation |
| `protein_partition=test`, `split=test` | No | No | Both simultaneously — hardest |

#### Why this design matters for method comparison

A model that memorises protein identity (e.g. a random forest with a protein-ID
embedding) will perform well on training proteins with similar ligands, but collapse
when either the protein or the ligand chemistry is novel. This is exactly what the
benchmark reveals: RF/GBM/SVM achieve train ROC-AUC of 0.80–0.86 but test ROC-AUC
of only 0.30–0.43.

Structure-based methods such as GNINA dock directly into the 3D binding pocket and do
not rely on memorising protein identity, so they are expected to generalise more
robustly across both axes — making GNINA results a meaningful upper-bound reference
for the ligand-based ML models.

---

## 2. Environment Setup

All scripts run inside the `rdkit_env` Conda environment.

```bash
# Create the environment (first time only)
conda env create -f benchmarks/envs/env_deep_learning.yml

# Activate
conda activate rdkit_env
```

All commands below assume you are in the **project root**
(`/path/to/VLS-Benchmark-Dataset`) and invoke scripts via `conda run` to avoid
shell activation issues in non-interactive sessions:

```bash
conda run -n rdkit_env python3 benchmarks/...
```

---

## 3. Data Preprocessing

The preprocessing pipeline selects a representative co-crystal structure per
UniProt target and builds the training registry.

```bash
# 1. Select representative structures
conda run -n rdkit_env python3 benchmarks/01_preprocessing/select_representative_structure.py

# 2. Build the protein-level split assignment
conda run -n rdkit_env python3 benchmarks/01_preprocessing/assign_protein_splits.py

# 3. Build the full training registry with 2D split labels
conda run -n rdkit_env python3 benchmarks/01_preprocessing/build_training_registry.py
```

The outputs land in `training_data_full/`. If you are using the pre-built registry
that ships with the repository you can skip this step.

---

## 4. Classical ML Benchmark

Trains **Random Forest**, **Gradient Boosting Machine**, and **SVM** on Morgan
fingerprints concatenated with protein-identifier embeddings.

### Configuration

Edit `benchmarks/configs/classical_config.yaml`:

```yaml
model:
  type: random_forest        # random_forest | gradient_boosting | svm
  n_estimators: 100
  class_weight: balanced

features:
  type: combined             # ligand only: "morgan" | combined: "combined"
  ligand:
    type: morgan
    radius: 2
    n_bits: 2048
  protein:
    type: protein_identifier # learned 32-dim protein embedding

data:
  similarity_threshold: 0p7
  include_decoys: true
```

### Training

```bash
# Random Forest (default config)
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/classical_config.yaml \
    --output-dir benchmarks/02_training/trained_models

# GBM
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/gbm_config.yaml \
    --output-dir benchmarks/02_training/trained_models

# SVM
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/svm_config.yaml \
    --output-dir benchmarks/02_training/trained_models
```

Each run saves to `trained_models/{model_name}_training_summary.json`:

```json
{
  "model_type": "random_forest",
  "feature_type": "combined_morgan_r2_b2048_protein_identifier",
  "similarity_threshold": "0p7",
  "training_history": {
    "train_metrics": { "roc_auc": 0.859, "avg_precision": 0.265, ... },
    "val_metrics":   { "roc_auc": 0.441, ... },
    "test_metrics":  { "roc_auc": 0.304, ... },
    "n_train_samples": 1764226,
    "n_test_samples": 2086662,
    "training_time": 418
  }
}
```

### Current Results (0p7 threshold, combined features)

| Model | Train ROC-AUC | Val ROC-AUC | Test ROC-AUC | Train Time |
|-------|--------------|-------------|--------------|------------|
| Random Forest | 0.859 | 0.441 | 0.304 | 418 s |
| Gradient Boosting | 0.799 | 0.789 | 0.372 | 575 s |
| SVM | 0.703 | 0.696 | 0.431 | 188 s |

The large train→test gap is expected and by design: the 0p7 threshold creates a hard
generalisation split where test compounds are chemically dissimilar from training
compounds, across proteins the model has never seen.

---

## 5. Deep Learning Benchmark

Trains **DeepDTA** (CNN/CNN) or **GraphDTA** on SMILES sequences and protein sequences.

### Configuration

Edit `benchmarks/configs/deep_purpose_config.yaml`:

```yaml
model:
  type: CNN_CNN              # CNN_CNN | GCN_CNN | ...
  cls_hidden_dims: [1024, 1024, 512]

training:
  batch_size: 128
  epochs: 30
  learning_rate: 0.001

data:
  similarity_threshold: 0p7
  only_actives: true         # regression mode on actives only
  require_affinity: true
  target_transform: pIC50
```

### Training

```bash
conda run -n rdkit_env python3 benchmarks/02_training/train_deeppurpose_wrapper.py \
    --config benchmarks/configs/deep_purpose_config.yaml \
    --output-dir benchmarks/02_training/trained_models
```

---

## 6. Structure-Based Docking Benchmark

Uses **GNINA** (CNN-based molecular docking) as a physics-informed reference. GNINA
docks each test compound against the target's co-crystal receptor and scores poses with
a 3D-CNN binding probability (CNNscore).

GPU acceleration is used by default (RTX 4090). The full benchmark runs ~15 representative
targets sequentially to avoid GPU contention.

### Prerequisites

- GNINA binary at `~/projects/PoseBench/forks/GNINA/gnina`
- OpenBabel (`obabel`) in PATH
- Pre-built `registry_2d_split.csv` and `protein_references.json`

### Step-by-Step

#### 6.1 Select Target Proteins

Selects ~15 diverse test-partition proteins with co-crystal structures and ≥ 5 test
actives. Diversity is ensured by binning targets by log(n_actives) and preferring
lowest-resolution (highest quality) structures.

```bash
conda run -n rdkit_env python3 benchmarks/04_docking/select_targets.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml
# Output: benchmarks/04_docking/results/selected_targets.json
```

#### 6.2 Prepare Receptor Structures

For each target: CIF → PDB (obabel), extract co-crystal reference ligand as SDF,
strip ligand + waters from receptor, convert to PDBQT (falls back to clean PDB if
obabel kekulization fails).

```bash
conda run -n rdkit_env python3 benchmarks/04_docking/prepare_structures.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml
# Output: benchmarks/04_docking/results/receptors/{uniprot}.pdbqt
#         benchmarks/04_docking/results/receptors/{uniprot}_ref_ligand.sdf
```

To reprocess only specific targets:

```bash
conda run -n rdkit_env python3 benchmarks/04_docking/prepare_structures.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml \
    --targets Q9Y271 P08912
```

#### 6.3 Prepare Ligand Sets

Actives are loaded from pre-existing 3D SDF files. Decoys are generated as 3D
conformers from SMILES using RDKit ETKDGv3 (parallelised; GNINA refines poses
internally so no MMFF needed).

```bash
conda run -n rdkit_env python3 benchmarks/04_docking/prepare_ligands.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml \
    --max-decoys 500         # cap decoys per target for speed (omit for full set)
    --n-workers 16           # parallel 3D embedding workers
# Output: benchmarks/04_docking/results/ligands/{uniprot}_all_ligands.sdf
```

`is_active` (1/0) is stored as an SDF property on each molecule and propagated
through to GNINA output for downstream metric computation.

#### 6.4 Run Docking

Docks all prepared ligand SDFs against the corresponding receptor. Uses `n_workers=1`
to give GNINA exclusive GPU access (multiple simultaneous GNINA processes contend for
the same GPU and are slower).

```bash
conda run -n rdkit_env python3 benchmarks/04_docking/run_gnina_benchmark.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml
# Output: benchmarks/04_docking/results/docking/{uniprot}_docked.sdf
#         benchmarks/04_docking/results/docking/{uniprot}_gnina.log
```

To dock a subset:

```bash
conda run -n rdkit_env python3 benchmarks/04_docking/run_gnina_benchmark.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml \
    --targets Q9Y271 P08912 --n-workers 1
```

#### 6.5 Collect Results

Parses each docked SDF (best CNNscore pose per compound), reconstructs active/decoy
labels, and computes classification metrics.

```bash
conda run -n rdkit_env python3 benchmarks/04_docking/collect_results.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml
# Output: benchmarks/04_docking/results/gnina_training_summary.json
```

### GNINA Configuration Reference

`benchmarks/04_docking/configs/gnina_config.yaml`:

```yaml
gnina:
  binary: "/path/to/gnina"
  num_modes: 10        # docking poses per compound
  exhaustiveness: 3    # search effort (higher = slower + better)
  cnn_scoring: rescore # CNN rescoring of Vina poses
  autobox_add: 4.0     # Å buffer around reference ligand for pocket box
  cpu: 4               # CPU threads (Vina search; CNN runs on GPU)
  device: 0            # GPU device index (-1 to disable GPU)

parallelism:
  n_workers: 1         # 1 = sequential, full GPU per job (recommended)
```

### Notes on Receptor Preparation

- obabel fails to kekulize some CIF structures (aromatic bond perception error).
  `prepare_structures.py` automatically falls back to using the cleaned PDB directly
  as the GNINA receptor — GNINA accepts PDB natively.
- CIF files containing `*`-element atoms (e.g. some metal ions) crash GNINA's PDB
  parser. These are stripped automatically before passing the receptor to GNINA.

---

## 7. Generating a Comparison Report

Once at least one model and GNINA results exist, generate a unified comparison CSV:

```bash
conda run -n rdkit_env python3 benchmarks/03_analysis/generate_benchmark_report.py \
    --results-dir benchmarks/02_training/trained_models \
    --docking-dir benchmarks/04_docking/results \
    --output benchmarks/03_analysis/report.csv
```

This scans both directories for `*_training_summary.json` files and writes one row per
method to `report.csv` with columns:

```
model, feature_type, similarity_threshold, test_roc_auc, test_avg_precision,
test_f1_score, test_accuracy, test_mcc, n_test, training_time_s, ...
```

Example output:

| model | test_roc_auc | test_avg_precision | test_f1 | n_test |
|-------|--------------|--------------------|---------|--------|
| random_forest | 0.304 | 0.060 | 0.005 | 2,086,662 |
| gradient_boosting | 0.372 | 0.055 | 0.044 | 2,086,662 |
| svm | 0.431 | 0.051 | 0.001 | 2,086,662 |
| gnina | — | — | — | ~15 targets |

---

## 8. Dataset Schema Reference

### `registry_2d_split.csv` columns

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | Unique row identifier |
| `uniprot_id` | str | UniProt accession |
| `pdb_id` | str | PDB code of reference structure |
| `compound_id` | str | ChEMBL compound ID (may be NaN for decoys) |
| `cif_path` | str | Path to mmCIF structure file |
| `resolution` | float | Crystal structure resolution (Å) |
| `quality_score` | float | Structure quality score |
| `smiles` | str | SMILES string; decoys use `"active_smiles decoy_smiles"` format |
| `sdf_path` | str | Path to pre-computed 3D SDF for actives |
| `is_active` | bool | True = active, False = decoy |
| `affinity_value` | float | Binding affinity (nM, for actives) |
| `affinity_type` | str | IC50 / Ki / Kd |
| `similarity_threshold` | str | 0p3 / 0p5 / 0p7 |
| `split` | str | train / test / decoy |
| `protein_partition` | str | train / val / test (protein-level) |
| `protein_cluster` | str | Cluster ID from 70% sequence-coverage clustering |

### `protein_references.json` structure

```json
{
  "O00141": {
    "uniprot_id": "O00141",
    "pdb_id": "2R5T",
    "cif_path": "../../plate-vs/VLS_benchmark/.../2R5T.cif",
    "resolution": 1.9,
    "quality_score": 328.5,
    "method": "X-RAY DIFFRACTION",
    "chosen_ligand": "ANP@A:500(heavy=31)",
    "pocket_residue_count": 28,
    "pocket_completeness": 1.0
  }
}
```

`chosen_ligand` format: `RESIDUE_NAME@CHAIN:RESID(heavy=N_HEAVY_ATOMS)`.
This defines the co-crystal ligand used to set the GNINA autobox.

### Decoy SMILES Format

Decoy SMILES in the registry follow the **DeepCoy** convention:

```
"CC(=O)Nc1ccc(O)cc1 CC(=O)Nc1ccc(N)cc1"
 ^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^
   reference active        decoy SMILES
```

`prepare_ligands.py` extracts the second (space-separated) token as the decoy SMILES.

---

## 9. Adding a New Method

### Adding a new ML model

1. Create `benchmarks/02_training/models/mymodel_trainer.py` following the interface
   in `base_trainer.py`:

   ```python
   from .base_trainer import BaseTrainer

   class MyModelTrainer(BaseTrainer):
       def train(self, X_train, y_train): ...
       def predict_proba(self, X): ...
       def save(self, path): ...
   ```

2. Add a config file at `benchmarks/configs/mymodel_config.yaml` with
   `model.type: my_model`.

3. Register the new trainer in `train_classical_oddt.py`'s trainer factory.

4. Run training and the output `my_model_training_summary.json` will automatically
   appear in `generate_benchmark_report.py` output.

### Adding a new scoring/docking method

Write your scoring pipeline so that it outputs a JSON file matching the
`training_summary.json` schema:

```json
{
  "model_type": "my_method",
  "feature_type": "...",
  "similarity_threshold": "0p7",
  "use_precomputed_split": true,
  "training_history": {
    "train_metrics": null,
    "val_metrics": null,
    "test_metrics": {
      "roc_auc": 0.XX,
      "avg_precision": 0.XX,
      "f1_score": 0.XX,
      "accuracy": 0.XX,
      "precision": 0.XX,
      "recall": 0.XX,
      "mcc": 0.XX
    },
    "n_train_samples": null,
    "n_val_samples": null,
    "n_test_samples": 12345,
    "training_time": 0
  }
}
```

Save it to `benchmarks/04_docking/results/mymethod_training_summary.json` (or any
directory passed to `--docking-dir`) and it will be picked up by
`generate_benchmark_report.py` automatically.
