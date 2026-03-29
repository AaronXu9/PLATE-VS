# Design: GBM/SVM Soft-Split Classification + Branch Merge

**Date:** 2026-03-28
**Branch:** `feature/ml-benchmarking`
**Status:** Approved

---

## Goal

Complete the soft-split classical ML benchmark by training GBM and SVM classifiers on `registry_soft_split.csv`, regenerate the unified benchmark report, and merge the branch to `main`. Deep learning (DeepPurpose, GraphDTA) stubs are out of scope — they continue in a follow-on branch.

---

## Context

### What already exists

| Model | Task | Split | Location | Status |
|---|---|---|---|---|
| RF | Classification | Hard (0p7) | `benchmarks/02_training/trained_models/` | Done |
| GBM | Classification | Hard (0p7) | `benchmarks/02_training/trained_models/` | Done |
| SVM | Classification | Hard (0p7) | `benchmarks/02_training/trained_models/` | Done |
| RF | Classification | Soft | `trained_models/soft_split_classification/` | Done |
| RF | Regression | Soft | `trained_models/regression/` | Done |
| GBM | Regression | Soft | `trained_models/regression/` | Done |
| SVM | Regression | Soft | `trained_models/regression/` | Done |
| GBM | Classification | Soft | `trained_models/soft_split_classification/` | **TODO** |
| SVM | Classification | Soft | `trained_models/soft_split_classification/` | **TODO** |

### Dataset

- Registry: `training_data_full/registry_soft_split.csv` (2,886,090 rows, soft intra-cluster protein split)
- Train set (0p7, protein_partition=train, split=train): ~1,730,902 samples
- Feature cache: `training_data_full/feature_cache/` (Morgan FPs pre-computed from RF run)
- Environment: `rdkit_env` (sklearn, numpy, rdkit — no XGBoost/LightGBM, GBM falls back to HistGradientBoosting)

---

## Architecture

No new code is needed. The existing `train_classical_oddt.py` script handles both models via config files.

```
benchmarks/
  configs/
    gbm_config.yaml       ← model_type: gradient_boosting, backend: auto
    svm_config.yaml       ← model_type: svm, kernel: linear
  02_training/
    train_classical_oddt.py  ← CLI: --config, --registry, --use-2d-split, --output, --cache-dir
  03_analysis/
    generate_benchmark_report.py  ← scans dirs recursively for *_training_summary.json
    report.csv            ← overwritten with all results

trained_models/
  soft_split_classification/
    random_forest_*       ← existing
    gradient_boosting_*   ← new
    svm_*                 ← new
  regression/             ← existing, included in report
```

---

## Execution Plan

### Step 1 — Quick-test gate (both models)

Validate configs before committing to multi-hour full runs. Catches the GBM backend fallback (XGBoost absent → HistGBM) early.

```bash
# From repo root, in rdkit_env
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/gbm_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split \
    --output trained_models/soft_split_classification \
    --cache-dir training_data_full/feature_cache \
    --quick-test

conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/svm_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split \
    --output trained_models/soft_split_classification \
    --cache-dir training_data_full/feature_cache \
    --quick-test
```

Pass criteria: both complete without error, produce `*_training_summary.json` in output dir.

### Step 2 — GBM full training

```bash
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/gbm_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split \
    --output trained_models/soft_split_classification \
    --cache-dir training_data_full/feature_cache
```

Expected: ~10–20 min. Output files: `gradient_boosting_training_summary.json`, `gradient_boosting.pkl`, etc.

### Step 3 — SVM full training

```bash
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/svm_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split \
    --output trained_models/soft_split_classification \
    --cache-dir training_data_full/feature_cache
```

Expected: ~3–10 min (LinearSVC is fast). Output: `svm_training_summary.json`, `svm.pkl`, etc.

### Step 4 — Regenerate unified benchmark report

Scan all three result directories to produce a single consolidated CSV:

```bash
cd benchmarks/03_analysis
conda run -n rdkit_env python3 generate_benchmark_report.py \
    --results-dir ../02_training/trained_models \
    --output report.csv
```

Then manually verify the script also picks up `trained_models/soft_split_classification/` and `trained_models/regression/` (check if `--extra-dirs` flag exists or if script needs a path list). Overwrite `benchmarks/03_analysis/report.csv`.

### Step 5 — Commit results

Stage and commit:
- `trained_models/soft_split_classification/gradient_boosting_*`
- `trained_models/soft_split_classification/svm_*`
- `benchmarks/03_analysis/report.csv`
- This spec doc

Exclude `.pkl` files and large logs (check `.gitignore`).

### Step 6 — PR to main

Open a PR from `feature/ml-benchmarking` → `main` with a summary of all completed benchmarks (hard-split classification, soft-split classification, soft-split regression). Deep learning work is explicitly deferred to a follow-on branch.

---

## Success Criteria

- `trained_models/soft_split_classification/` contains `gradient_boosting_training_summary.json` and `svm_training_summary.json` with non-null test ROC-AUC values
- `benchmarks/03_analysis/report.csv` contains rows for all 9 completed model/task/split combinations
- PR is created and branch is merged to `main`

---

## Out of Scope

- Deep learning models (DeepPurpose, GraphDTA) — stubs exist, not implemented
- Hyperparameter tuning
- Additional similarity thresholds (0p3, 0p5) — only 0p7 for now
- PDBbind comparison script
