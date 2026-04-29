# PLATE-VS

A virtual-screening (VS) benchmark built on the **PLATE-VS** dataset, with reference
implementations for classical ML, deep-learning, and structure-based docking methods.
The benchmark is designed so that every test sample probes a generalisation axis —
either novel protein, novel ligand chemistry, or both — making it substantially harder
than memorisation-friendly benchmarks.

A companion track on **PDBbind CleanSplit** (binding-affinity regression) is included
for cross-dataset comparison against published baselines such as GEMS.

For step-by-step usage, see [`benchmarks/BENCHMARKING_GUIDE.md`](benchmarks/BENCHMARKING_GUIDE.md).
For metric definitions, see [`METRICS.md`](METRICS.md).

## Citing this work

If you use this code, please cite both the paper and the archived release.
Machine-readable metadata for the release is in [`CITATION.cff`](CITATION.cff)
(also published to Zenodo on each tagged release):

> Xu, A.; Hong, Y.; Lam, J. H.; Katritch, V. *PLATE-VS: virtual screening
> benchmark*. v1.0.0. Zenodo. DOI: `10.5281/zenodo.XXXXXXX`.

Ao Xu, Yongchan Hong, and Jordy Homing Lam are joint first authors. Replace
`XXXXXXX` with the concept DOI shown on the Zenodo record once it is minted.

## Reproducing from the Zenodo archive

The Zenodo tarball mirrors the GitHub source archive — it does **not** include
the submodule contents under `external/`. To reproduce results from the
archive:

1. Download and extract the Zenodo tarball, **or** clone with submodules from
   GitHub:
   ```bash
   git clone --recursive https://github.com/AaronXu9/PLATE-VS
   cd PLATE-VS && git checkout v1.0.0
   ```
2. Bulk training data and model checkpoints are not part of the code archive.
   See [`benchmarks/BENCHMARKING_GUIDE.md`](benchmarks/BENCHMARKING_GUIDE.md)
   for download/regeneration steps.

---

## 1. Dataset

### 1.1 PLATE-VS (virtual screening)

PLATE-VS is a large-scale protein–ligand virtual-screening dataset derived from
ChEMBL, covering **826 UniProt targets**. Decoys are generated with **DeepCoy**, a
deep-learning decoy generator that matches each active's physicochemical profile.

| Split | Proteins | Actives | Decoys | Total samples |
|-------|---------:|--------:|-------:|--------------:|
| Train | 578      | ~930 K  | ~1.23 M| ~2.16 M       |
| Val   | 124      | ~233 K  | ~308 K | ~541 K        |
| Test  | 124      | ~127 K  | ~1.96 M| ~2.09 M       |

The dataset ships at three **ligand-similarity thresholds** that control benchmark
difficulty. All results in this repo use `0p7`.

| Threshold | Key  | Description                                        |
|-----------|------|----------------------------------------------------|
| 0.3       | `0p3`| Easy — many similar ligands in train               |
| 0.5       | `0p5`| Medium                                             |
| 0.7       | `0p7`| Hard — test compounds dissimilar to train         |

#### The 2D split

Each sample is split along **two independent axes**, simultaneously:

**Axis 1 — Protein partition (`protein_partition`).** Proteins are clustered by
sequence similarity (70 % query coverage). Each cluster is assigned entirely to one
of `train` / `val` / `test`, so train/val/test protein families are disjoint.

**Axis 2 — Ligand similarity (`split`).** For *every* protein, regardless of its
protein partition, its actives are split by pairwise Tanimoto fingerprint similarity:
`split=train` are the chemically coherent core; `split=test` are actives whose
Tanimoto < threshold to all `split=train` actives. `split=decoy` holds the DeepCoy
decoys.

The combined test set therefore probes three failure modes:

| Test sample origin                       | Protein seen | Ligand similar | Tests                                |
|------------------------------------------|--------------|----------------|--------------------------------------|
| `protein_partition=train`, `split=test`  | yes          | no             | Chemical-scaffold generalisation     |
| `protein_partition=test`, `split=train`  | no           | n/a            | Protein-family generalisation        |
| `protein_partition=test`, `split=test`   | no           | no             | Both at once — hardest               |

A model that memorises protein identity (e.g. a random forest with a protein-ID
embedding) does well on training proteins with similar ligands but collapses when
either axis is novel — exactly what the benchmark reveals.

#### Soft vs. hard splits

The repo ships two flavours of split:

- **Hard split** — 2D split as described above; the canonical PLATE-VS evaluation.
- **Soft split** — protein partition only (no ligand-similarity holdout). Easier;
  used as a sanity check and for diagnosing whether failures are protein-driven or
  chemistry-driven. See [`SOFT_SPLIT_REGISTRY.md`](SOFT_SPLIT_REGISTRY.md).

#### Files

```
training_data_full/
├── registry_2d_split.csv      # main registry, all samples with split labels
├── registry.csv               # original flat registry
└── protein_references.json    # per-protein metadata (PDB id, resolution, pocket)
```

#### Schema (`registry_2d_split.csv`)

| Column                  | Type    | Description                                              |
|-------------------------|---------|----------------------------------------------------------|
| `sample_id`             | str     | Unique row id                                            |
| `uniprot_id`            | str     | UniProt accession                                        |
| `pdb_id`                | str     | PDB code of reference structure                          |
| `compound_id`           | str     | ChEMBL id (NaN for decoys)                               |
| `cif_path`              | str     | Path to mmCIF structure file                             |
| `resolution`            | float   | Crystal resolution (Å)                                   |
| `quality_score`         | float   | Structure quality score                                  |
| `smiles`                | str     | SMILES; for decoys: `"active_smiles decoy_smiles"`       |
| `sdf_path`              | str     | Path to pre-computed 3D SDF (actives)                    |
| `is_active`             | bool    | True = active, False = decoy                             |
| `affinity_value`        | float   | nM (actives only)                                        |
| `affinity_type`         | str     | IC50 / Ki / Kd                                           |
| `similarity_threshold`  | str     | 0p3 / 0p5 / 0p7                                          |
| `split`                 | str     | train / test / decoy                                     |
| `protein_partition`     | str     | train / val / test (protein-level)                       |
| `protein_cluster`       | str     | Cluster id from 70 %-coverage sequence clustering        |

### 1.2 PDBbind CleanSplit (binding-affinity regression)

A second track using the **PDBbind CleanSplit** (CASF-2016 test set) for binding
affinity regression. Used to validate our deep-learning model against published
baselines (GEMS, etc.) on a well-known, structurally-resolved dataset before scaling
to PLATE-VS. See `benchmarks/05_pdbbind_comparison/` and
[`benchmarks/05_pdbbind_comparison/GEMS_INTEGRATION.md`](benchmarks/05_pdbbind_comparison/GEMS_INTEGRATION.md).

---

## 2. How the benchmark works

Each method is wrapped in a small training/inference script that emits a uniform
`*_training_summary.json` artifact. A single report generator scans the artifacts
and produces a comparison CSV — so adding a new method is a matter of writing a
script that emits the right JSON.

### 2.1 Pipeline

```
                     ┌─────────────────────────────────────┐
                     │ 01_preprocessing                    │
                     │   select_representative_structure   │
                     │   assign_protein_splits             │
                     │   build_training_registry           │
                     └────────────────┬────────────────────┘
                                      │ registry_2d_split.csv
                                      │ protein_references.json
          ┌───────────────────────────┼───────────────────────────────────┐
          ▼                           ▼                                   ▼
┌────────────────────┐    ┌────────────────────────┐    ┌────────────────────────────┐
│ 02_training        │    │ 04_docking (GNINA)     │    │ 07_plate_vs_dl             │
│   classical RF/GBM │    │   select_targets       │    │   dual-encoder DL model    │
│   /SVM, DeepPurpose│    │   prepare_structures   │    │   (ESM2 + ligand encoder)  │
│                    │    │   prepare_ligands      │    │                            │
│                    │    │   run_gnina_benchmark  │    │                            │
│                    │    │   collect_results      │    │                            │
└────────┬───────────┘    └───────────┬────────────┘    └─────────────┬──────────────┘
         │                            │                               │
         ▼                            ▼                               ▼
              *_training_summary.json (uniform schema, one per method)
                                      │
                                      ▼
                     ┌──────────────────────────────────┐
                     │ 03_analysis                      │
                     │   generate_benchmark_report.py   │
                     └────────────────┬─────────────────┘
                                      ▼
                                  report.csv
```

### 2.2 Methods implemented

| Family                | Method                | Where                            | Features                                                  |
|-----------------------|-----------------------|----------------------------------|-----------------------------------------------------------|
| Classical ML          | Random Forest, GBM, SVM | `benchmarks/02_training/`      | Morgan fingerprints + protein-identifier embedding        |
| Sequence DL           | DeepDTA / GraphDTA    | `benchmarks/02_training/`        | SMILES + protein sequence (DeepPurpose wrapper)           |
| Structure-based       | GNINA                 | `benchmarks/04_docking/`         | 3D docking + CNN rescoring on co-crystal pocket           |
| Custom dual-encoder DL| PLATE-VS DL (ours)    | `benchmarks/07_plate_vs_dl/`     | ESM2 protein embedding + ligand graph/conformer encoder   |
| PDBbind comparison    | Affinity regression   | `benchmarks/05_pdbbind_comparison/` | CASF-2016 vs. GEMS, classical baselines                  |

### 2.3 Uniform artifact schema

Every method writes a `*_training_summary.json` with this shape so the report
generator can pick it up:

```json
{
  "model_type": "random_forest",
  "feature_type": "combined_morgan_r2_b2048_protein_identifier",
  "similarity_threshold": "0p7",
  "training_history": {
    "train_metrics": { "roc_auc": 0.859, "avg_precision": 0.265 },
    "val_metrics":   { "roc_auc": 0.441 },
    "test_metrics":  { "roc_auc": 0.304 },
    "n_train_samples": 1764226,
    "n_test_samples":  2086662,
    "training_time": 418
  }
}
```

### 2.4 Metrics

Classification: ROC-AUC, average precision (PR-AUC), F1, accuracy, MCC.
Virtual screening: enrichment factor (EF, EFB) and BEDROC.
Regression (PDBbind track): Pearson R, RMSE, MAE.

Full definitions and code pointers in [`METRICS.md`](METRICS.md).

---

## 3. Repository layout

```
PLATE-VS/
├── benchmarks/
│   ├── 01_preprocessing/        # Build registry, assign splits, select structures
│   ├── 02_training/             # Classical ML + DeepPurpose
│   ├── 03_analysis/             # Cross-method report generator
│   ├── 04_docking/              # GNINA structure-based docking benchmark
│   ├── 05_pdbbind_comparison/   # PDBbind CleanSplit / GEMS regression baseline
│   ├── 06_binding_affinity_model/ # Standalone affinity regression model
│   ├── 07_plate_vs_dl/          # Custom dual-encoder DL (ESM2 + ligand encoder)
│   ├── configs/                 # Method-level config YAMLs
│   ├── envs/                    # Conda env files
│   └── BENCHMARKING_GUIDE.md    # Step-by-step usage
├── data/
│   ├── pdbbind_cleansplit/      # PDBbind CleanSplit datasets
│   └── plate_vs_conformers/     # Pre-computed 3D conformers for PLATE-VS ligands
├── training_data_full/          # registry_2d_split.csv, protein_references.json (not bundled)
├── scripts/                     # Helpers
├── METRICS.md                   # Metric definitions
├── SOFT_SPLIT_REGISTRY.md       # Soft-split documentation
└── README.md                    # this file
```

---

## 4. Quickstart

### 4.1 Environment

```bash
conda env create -f benchmarks/envs/env_deep_learning.yml
conda activate rdkit_env
```

All commands in this repo are designed to run from the project root via
`conda run -n rdkit_env python3 ...` so they work in non-interactive sessions
(SLURM, CI).

### 4.2 Run the classical ML baseline

```bash
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/classical_config.yaml \
    --output-dir benchmarks/02_training/trained_models
```

### 4.3 Run the GNINA docking benchmark

```bash
# 1. Pick ~15 representative test-partition targets
conda run -n rdkit_env python3 benchmarks/04_docking/select_targets.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml

# 2. Prepare receptors + ligands
conda run -n rdkit_env python3 benchmarks/04_docking/prepare_structures.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml
conda run -n rdkit_env python3 benchmarks/04_docking/prepare_ligands.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml --max-decoys 500

# 3. Dock and collect
conda run -n rdkit_env python3 benchmarks/04_docking/run_gnina_benchmark.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml
conda run -n rdkit_env python3 benchmarks/04_docking/collect_results.py \
    --config benchmarks/04_docking/configs/gnina_config.yaml
```

### 4.4 Generate the comparison report

```bash
conda run -n rdkit_env python3 benchmarks/03_analysis/generate_benchmark_report.py \
    --results-dir benchmarks/02_training/trained_models \
    --docking-dir benchmarks/04_docking/results \
    --output benchmarks/03_analysis/report.csv
```

---

## 5. Current results

### 5.1 PLATE-VS hard split (0p7, virtual screening)

| Model                       | Train ROC-AUC | Val ROC-AUC | Test ROC-AUC |
|-----------------------------|--------------:|------------:|-------------:|
| Random Forest (Morgan + ID) | 0.859         | 0.441       | 0.304        |
| Gradient Boosting           | 0.799         | 0.789       | 0.372        |
| SVM                         | 0.703         | 0.696       | 0.431        |
| Custom dual-encoder DL      | —             | —           | **0.801**    |

The large train→test gap is **expected and by design**: 0p7 enforces a generalisation
split where test compounds are chemically dissimilar to training compounds, *and*
test proteins come from disjoint sequence-similarity clusters.

### 5.2 PDBbind CleanSplit (CASF-2016, affinity regression)

| Model                  | Pearson R |
|------------------------|----------:|
| GEMS (published)       | 0.815     |
| Custom dual-encoder DL | 0.804     |
| Best classical ML      | 0.722     |

---

## 6. Adding a new method

Write a script that emits a `*_training_summary.json` matching the schema in §2.3
and place it under `benchmarks/02_training/trained_models/` or
`benchmarks/04_docking/results/`. The report generator picks it up automatically.

A worked example for both ML and docking-style methods is in
[`benchmarks/BENCHMARKING_GUIDE.md` §9](benchmarks/BENCHMARKING_GUIDE.md#9-adding-a-new-method).

---

## 7. Further reading

- [`benchmarks/BENCHMARKING_GUIDE.md`](benchmarks/BENCHMARKING_GUIDE.md) — full step-by-step guide
- [`METRICS.md`](METRICS.md) — metric definitions and code pointers
- [`SOFT_SPLIT_REGISTRY.md`](SOFT_SPLIT_REGISTRY.md) — soft-split construction
- [`benchmarks/05_pdbbind_comparison/GEMS_INTEGRATION.md`](benchmarks/05_pdbbind_comparison/GEMS_INTEGRATION.md) — PDBbind / GEMS comparison

---

## License

See [`LICENSE`](LICENSE).
