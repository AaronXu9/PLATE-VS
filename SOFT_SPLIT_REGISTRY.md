# Soft Protein Partition Registry (`registry_soft_split.csv`)

## Overview

`training_data_full/registry_soft_split.csv` is an extended version of the PLATE-VS training registry that introduces a **soft protein partition** strategy and preserves **numerical affinity values** as pChEMBL scores for regression tasks.

It is a drop-in replacement for `registry_2d_split.csv` when a less stringent generalization test is desired.

`training_data_full/registry_soft_split_regression.csv` extends it further with pChEMBL values and assay metadata joined from `data/filtered_chembl_affinity.parquet`. Use this file for regression training.

---

## Dataset Statistics

| Property | Value |
|---|---|
| Total rows | 2,886,090 |
| Total columns | 19 (soft split) / 22 (regression) |
| Unique proteins | 826 |
| Unique active (protein, compound) pairs | 275,132 |
| Unique active SMILES | 233,466 |
| Active rows | 831,030 (= 275,132 pairs × 3 similarity thresholds) |
| Decoy rows (DeepCoy) | 2,055,060 |
| Similarity thresholds | 0p3, 0p5, 0p7 (277,010 active rows each) |

> **Note on active row count**: each (protein, compound) pair appears three times — once per similarity threshold. The 831,030 figure counts rows, not unique compounds. There are 233,466 unique active molecules and 275,132 unique active (protein, compound) pairs.

### Protein Partition Distribution (actives only)

| Partition | Proteins | Compounds |
|---|---|---|
| train | 631 (76.4%) | 591,828 (71.2%) |
| val | 98 (11.8%) | 113,643 (13.7%) |
| test | 97 (11.7%) | 125,559 (15.1%) |

Decoys are all assigned `protein_partition = 'train'` (shared negative pool for training).

### 2D Split Matrix (actives, 0p7 threshold)

The registry supports a full 2D split: ligand chemical similarity (columns) × protein partition (rows).

|  | ligand: train | ligand: test |
|---|---|---|
| **protein: train** | 441,937 | 149,891 |
| **protein: val** | 85,349 | 28,294 |
| **protein: test** | 93,118 | 32,441 |

The canonical training setup uses `protein=train, ligand=train` as the fit set, `protein=val` for hyperparameter selection, and `protein=test, ligand=test` as the held-out evaluation.

### pChEMBL and Assay Metadata (regression registry)

| Property | Value |
|---|---|
| Enriched rows | 660,321 (22.9% of all rows; 79.4% of active rows) |
| pChEMBL range | 4.00 – 11.00 |
| pChEMBL mean | 6.89 |
| Proteins covered | 812 / 826 |
| affinity_type breakdown | IC50: 524,085 (79.4%) / Ki: 136,236 (20.6%) |
| assay_type_enriched | B (binding) for all enriched rows |
| n_measurements (mean / max) | 1.18 / 103 — 75th percentile = 1 (most compounds have a single measurement) |
| document_year range | 1979 – 2024 (mean: 2014) |

pChEMBL values are populated in `registry_soft_split_regression.csv` by joining against `data/filtered_chembl_affinity.parquet` — the quality-filtered, per-protein ChEMBL activity data (standard_flag=1, no duplicates, within-range). Join key is `(uniprot_id, canonical_smiles)` so each measurement is protein-consistent.

**Consistency guarantee**: `affinity_value` is always back-calculated as `10^(9 − pchembl)` so that `pchembl = 9 − log10(affinity_value)` holds exactly for every row. For compounds with a single measurement this equals the original ChEMBL value; for compounds with multiple measurements it is the geometric mean (appropriate for log-normally distributed affinities).

---

## Construction

### Step 1 — Load the base registry

The full PLATE-VS registry (`training_data_full/registry.csv`, 2,886,090 rows) is used as the starting point. It provides one row per (protein, compound, similarity threshold) combination, with ligand-based train/test splits already assigned via Tanimoto similarity.

### Step 2 — Load protein cluster labels

Protein sequences were clustered using bidirectional BLAST coverage with a 70% query coverage threshold (`cluster_bipartite_qcov_70` column from `data/uniprot_bipartite_cluster_labels.csv`). This yields **281 clusters** across 826 proteins.

Cluster size distribution:
- 199 clusters: 1 protein (singletons)
- 33 clusters: 2 proteins
- 26 clusters: 3–4 proteins
- 15 clusters: 5–9 proteins
- 8 clusters: 10+ proteins (largest: 202 proteins)

### Step 3 — Intra-cluster protein partitioning (the key difference)

**Hard split (original `registry_2d_split.csv`):** Entire protein clusters are assigned atomically to one partition. No protein in the test cluster is sequence-similar to any training protein. This is the strictest generalization test but may be overly pessimistic.

**Soft split (this registry):** Proteins *within each cluster* are randomly sampled into train/val/test in a 70/15/15 ratio. The model sees proteins from every cluster during training, and is evaluated on held-out members of the same clusters.

```
For each cluster with ≥ 3 proteins:
  shuffle proteins (seed=42)
  n_val  = max(1, round(n × 0.15))
  n_test = max(1, round(n × 0.15))
  n_train = n − n_val − n_test
  assign accordingly

Clusters with < 3 proteins, or cluster_id == -1 (unclustered):
  all proteins → train
```

Result: **49 of 281 clusters** have members in multiple partitions — impossible under the hard split. This tests whether a model can generalize *within* a protein family, rather than to entirely novel families.

### Step 4 — Enrich with pChEMBL and assay metadata

Run `enrich_pchembl.py` against `data/filtered_chembl_affinity.parquet`, which contains 333,611 quality-filtered (protein, compound) measurements (standard_flag=1, no duplicates, binding assays only, pChEMBL in [4, 12]).

Join key: `(registry.uniprot_id, canonical_smiles)` ↔ `(activities.source_uniprot_id, activities.canonical_smiles)`. Registry SMILES are canonicalized via RDKit before joining to ensure format consistency. The join is **protein-consistent**: a compound's IC50 against protein A is never assigned to a registry entry for protein B.

For each (protein, compound) pair with multiple measurements, aggregation is:

| Column | Aggregation |
|---|---|
| `pchembl` | median (log-space central tendency) |
| `affinity_value` | `10^(9 − pchembl_median)` — back-calculated, guaranteed consistent |
| `affinity_type` | mode of `standard_type` (IC50, Ki, Kd, …) |
| `assay_type_enriched` | mode of `assay_type` (B/F/A) |
| `document_year` | max (most recent publication) |
| `n_measurements` | count of raw measurements aggregated |

```bash
python benchmarks/01_preprocessing/enrich_pchembl.py \
    --registry training_data_full/registry_soft_split.csv \
    --activities data/filtered_chembl_affinity.parquet \
    --output training_data_full/registry_soft_split_regression.csv
```

### Step 5 — Save

`registry_soft_split.csv` (19 columns) is written by `assign_protein_splits_soft.py`.
`registry_soft_split_regression.csv` (22 columns) adds the enriched assay metadata.

**Scripts:**
```bash
# Step 3: generate soft split registry
python benchmarks/01_preprocessing/assign_protein_splits_soft.py \
    --registry training_data_full/registry.csv \
    --cluster-file data/uniprot_bipartite_cluster_labels.csv \
    --cluster-threshold qcov_70 \
    --min-cluster-size 3 \
    --split-ratios 0.70 0.15 0.15 \
    --seed 42 \
    --output training_data_full/registry_soft_split.csv

# Step 4: enrich with pChEMBL and assay metadata
python benchmarks/01_preprocessing/enrich_pchembl.py \
    --registry training_data_full/registry_soft_split.csv \
    --activities data/filtered_chembl_affinity.parquet \
    --output training_data_full/registry_soft_split_regression.csv
```

---

## Column Schema

### Base registry (`registry_soft_split.csv`, 19 columns)

| Column | Type | Description |
|---|---|---|
| `sample_id` | str | Unique identifier: `{uniprot}_{threshold}_{split}` |
| `uniprot_id` | str | UniProt accession |
| `pdb_id` | str | PDB structure used |
| `compound_id` | str | ChEMBL or DeepCoy compound ID |
| `cif_path` | str | Path to ligand CIF file |
| `resolution` | float | Crystal structure resolution (Å) |
| `quality_score` | float | Structure quality score |
| `smiles` | str | SMILES string. **Note**: for DeepCoy decoy rows this field contains two space-separated SMILES: `{seed_active_smiles} {generated_decoy_smiles}` — the actual decoy molecule is the second part |
| `sdf_path` | str | Path to ligand SDF file |
| `pkl_path` | str | Path to precomputed features |
| `is_active` | bool | True = active (ChEMBL), False = decoy (DeepCoy) |
| `affinity_value` | float | Originally sparse (only 12 rows populated in base registry). Populated in regression registry — see below |
| `affinity_type` | str | Originally sparse. Populated in regression registry — see below |
| `similarity_threshold` | str | Tanimoto threshold: 0p3, 0p5, 0p7 |
| `split` | str | Ligand-based split: train, test, decoy |
| `source` | str | Data source: chembl or deepcoy |
| `protein_cluster` | int | Cluster ID from bidirectional BLAST (−1 = unclustered) |
| `protein_partition` | str | Soft partition assignment: train, val, test |
| `pchembl` | float | −log₁₀(affinity_nM × 10⁻⁹); NaN if no measurement |

### Regression registry additions (`registry_soft_split_regression.csv`, 22 columns)

The following columns are populated for all 660,321 enriched active rows (NaN for decoys and unmatched actives):

| Column | Type | Description |
|---|---|---|
| `affinity_value` | float | Affinity in nM, back-calculated as `10^(9 − pchembl)`. Guaranteed consistent: `pchembl = 9 − log10(affinity_value)` holds exactly |
| `affinity_type` | str | Most frequent measurement type for this (protein, compound): IC50 (79%) or Ki (21%) |
| `assay_type_enriched` | str | Most frequent assay type: B (binding) for all enriched rows |
| `document_year` | float | Most recent publication year for this (protein, compound) pair (range: 1979–2024) |
| `n_measurements` | float | Number of raw ChEMBL measurements aggregated (median=1, max=103) |

---

## How to Use

### Classification training (default)

```bash
python benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/soft_split_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split
```

The `--use-2d-split` flag activates the `protein_partition` column so train/val/test are drawn from the correct protein partitions. Without it, only the ligand-based `split` column is used.

### Regression training (pChEMBL targets)

Use `registry_soft_split_regression.csv` — the enriched file with populated `pchembl`, `affinity_value`, and assay metadata:

```bash
python benchmarks/02_training/train_regression.py \
    --config benchmarks/configs/regression_rf_config.yaml \
    --registry training_data_full/registry_soft_split_regression.csv \
    --use-2d-split
```

Or via the DataLoader directly:

```python
from benchmarks.02_training.data.data_loader import DataLoader

loader = DataLoader('training_data_full/registry_soft_split_regression.csv')
loader.load_registry()

train_data = loader.get_training_data(
    similarity_threshold='0p7',
    split='train',
    protein_partition='train'
)
smiles, targets = loader.prepare_features_labels(train_data, task='regression')
```

### Comparing hard vs. soft split

To compare generalization difficulty between the two strategies, train the same model on both registries and compare test-set metrics:

| Registry | Protein split strategy | Expected test ROC-AUC |
|---|---|---|
| `registry_2d_split.csv` | Hard (cluster-atomic) | Lower (novel protein families) |
| `registry_soft_split.csv` | Soft (intra-cluster) | Higher (seen related proteins) |

A large gap between the two indicates the model relies heavily on protein identity features rather than learning transferable binding patterns.

---

## Files

| File | Location | Notes |
|---|---|---|
| Base registry | `training_data_full/registry_soft_split.csv` | 936 MB, gitignored |
| Regression registry | `training_data_full/registry_soft_split_regression.csv` | ~1 GB, gitignored |
| Soft split script | `benchmarks/01_preprocessing/assign_protein_splits_soft.py` | Reproducible, seeded |
| Soft split tests | `benchmarks/01_preprocessing/tests/test_soft_splits.py` | 22 tests |
| Enrichment script | `benchmarks/01_preprocessing/enrich_pchembl.py` | Protein-consistent join |
| Enrichment tests | `benchmarks/01_preprocessing/tests/test_enrich_pchembl.py` | 19 tests |
| Classification config | `benchmarks/configs/soft_split_config.yaml` | RF classification |
| Regression configs | `benchmarks/configs/regression_{rf,gbm,svm}_config.yaml` | RF/GBM/SVM regression |
| DataLoader | `benchmarks/02_training/data/data_loader.py` | Supports task='regression' |
| Visualization notebook | `notebooks/soft_split_visualization.ipynb` | Cluster/partition/pChEMBL plots |
