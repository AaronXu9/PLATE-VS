# Boltzina Benchmark Pipeline

GPU-accelerated molecular docking (Uni-Dock) + neural affinity scoring (Boltz-2) for the PLATE-VS virtual screening benchmark.

## Prerequisites

| Conda env | Purpose |
|-----------|---------|
| `boltzina_env` | Boltz-2 scoring, gemmi, obabel, maxit, pdb-tools |
| `unidock2` | GPU-accelerated Uni-Dock docking |
| `rdkit_env` | Ligand SMILES to 3D PDB, metrics collection |

Environment specs: `benchmarks/envs/boltzina_env.yml`

## Pipeline Stages

All commands run from the worktree root:

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset-boltzina
```

### Stage 01: Select proteins

Pick top-N proteins from the registry by quality score (active count, pChEMBL coverage).

```bash
conda run -n rdkit_env python benchmarks/05_boltzina/01_select_proteins.py \
    --registry /home/aoxu/projects/VLS-Benchmark-Dataset/training_data_full/registry_soft_split_regression.csv \
    --base-dir /home/aoxu/projects/VLS-Benchmark-Dataset \
    --output benchmarks/05_boltzina/results/poc_proteins.json \
    --top-n 10
```

**Output:** `results/poc_proteins.json` — list of proteins with `uniprot_id`, `pdb_id`, `n_actives`, `n_decoys_to_sample`, `quality_score`.

### Stage 02: Boltz predict (GPU)

Generate Boltz-2 predicted protein structures and Vina docking box configs.

```bash
conda run -n boltzina_env python benchmarks/05_boltzina/02_prep_boltz.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --registry /home/aoxu/projects/VLS-Benchmark-Dataset/training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results \
    --base-dir /home/aoxu/projects/VLS-Benchmark-Dataset
```

**Output:** `results/work_dirs/{uid}/` containing:
- `boltz_results_{uid}/predictions/{uid}/` — Boltz-2 predicted structure CIF
- `{uid}.yaml` — Boltz input YAML
- `vina_config.txt` — docking box centered on co-crystal ligand

Resumable: skips proteins with existing `predictions/` directory.

### Stage 03: Prepare ligands (SMILES to 3D PDB)

Convert active and decoy SMILES to 3D PDB files via RDKit ETKDG conformer generation.

```bash
LD_LIBRARY_PATH=/home/aoxu/miniconda3/envs/rdkit_env/lib \
    /home/aoxu/miniconda3/envs/rdkit_env/bin/python \
    benchmarks/05_boltzina/03_prep_ligands.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --registry /home/aoxu/projects/VLS-Benchmark-Dataset/training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results \
    --base-dir /home/aoxu/projects/VLS-Benchmark-Dataset
```

**Output:** `results/ligands/{uid}/actives/*.pdb` and `results/ligands/{uid}/decoys/*.pdb`

Resumable: skips PDB files that already exist. Some SMILES fail (unusual aromatics, conformer generation failures) — these are counted and skipped.

Note: requires `LD_LIBRARY_PATH` workaround for GLIBCXX mismatch when running outside `conda run`.

### Stage 04: Docking + Boltz-2 scoring (GPU)

Dock all ligands against each protein receptor and score with Boltz-2 neural affinity predictor.

```bash
conda run -n boltzina_env python benchmarks/05_boltzina/04_run_boltzina.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --results-dir benchmarks/05_boltzina/results \
    --boltzina-dir external/boltzina \
    --base-dir /home/aoxu/projects/VLS-Benchmark-Dataset \
    --use-unidock \
    --batch-size 4
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--use-unidock` | on | GPU Uni-Dock docking (~3 min/protein) |
| `--no-unidock` | — | Fall back to CPU Vina (~2+ hrs/protein) |
| `--batch-size` | 4 | Boltz-2 scoring batch size (higher = faster, more VRAM) |
| `--num-workers` | 16 | Parallel workers for post-processing and structure prep |
| `--unidock-env` | `unidock2` | Conda env with Uni-Dock |
| `--unidock-batch-size` | 200 | Ligands per Uni-Dock GPU call |

**Pipeline per protein:**
1. Pre-generate `receptor.pdbqt` from experimental CIF (Meeko)
2. Uni-Dock GPU batch docking (or CPU Vina)
3. Parallel post-processing: obabel split, pdb_chain, pdb_merge, maxit CIF conversion
4. Parallel structure preparation: MMCIF parsing to NPZ
5. Batch Boltz-2 affinity scoring on GPU

**Output:** `results/raw_results/{uid}/boltzina_results.csv` with columns:
- `ligand_name` — path to input PDB
- `docking_score` — Uni-Dock/Vina docking score (kcal/mol, lower = better binding)
- `affinity_pred_value` — Boltz-2 predicted affinity (higher = better binding)
- `affinity_probability_binary` — Boltz-2 binary binding probability

Resumable: skips proteins with existing `boltzina_results.csv`.

### Stage 05: Collect metrics

Compute ROC-AUC, enrichment factor (EF1%), and Spearman correlation across all proteins.

```bash
LD_LIBRARY_PATH=/home/aoxu/miniconda3/envs/rdkit_env/lib \
    /home/aoxu/miniconda3/envs/rdkit_env/bin/python \
    benchmarks/05_boltzina/05_collect_results.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --registry /home/aoxu/projects/VLS-Benchmark-Dataset/training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results
```

**Output:**
- `results/boltzina_per_protein_results.csv` — per-protein ROC-AUC, EF1%, Spearman
- `results/boltzina_training_summary.json` — aggregate metrics

## Analyzing Results

### Per-protein metrics

```
uniprot_id,n_actives,n_decoys,roc_auc,ef1pct,spearman_r
Q07820,25,48,0.7208,0.0,
P09211,66,484,0.3966,0.0,
P42262,4,308,0.6031,0.0,
...
```

### Compare with classical ML models

```bash
conda run -n rdkit_env python benchmarks/03_analysis/generate_benchmark_report.py \
    --results-dir benchmarks/02_training/trained_models \
    --extra-dirs benchmarks/05_boltzina/results \
    --output /tmp/report_with_boltzina.csv
```

Produces a unified CSV with rows for RF, GBM, SVM, and boltzina.

### Raw per-ligand scores

Individual docking and affinity scores live in `results/raw_results/{uid}/boltzina_results.csv`. Use `affinity_pred_value` as the primary ranking metric (Boltz-2 neural score). The `docking_score` column contains the Uni-Dock/Vina physics-based score.

## Architecture

```
benchmarks/05_boltzina/
├── 01_select_proteins.py      # Stage 01: protein selection
├── 02_prep_boltz.py           # Stage 02: Boltz predict
├── 03_prep_ligands.py         # Stage 03: SMILES to 3D PDB
├── 04_run_boltzina.py         # Stage 04: docking + scoring
├── 05_collect_results.py      # Stage 05: metrics
├── lib/
│   ├── boltz_prep.py          # CIF parsing, Boltz YAML, receptor PDBQT prep
│   ├── boltzina_runner.py     # Boltzina config + subprocess runner
│   ├── ligands.py             # SMILES to PDB, decoy sampling
│   ├── metrics.py             # ROC-AUC, EF1%, Spearman
│   ├── select_proteins.py     # Protein selection logic
│   └── unidock_docking.py     # Uni-Dock batch docking + parallel post-processing
├── tests/                     # 27 unit tests
└── results/                   # Output directory (gitignored except summaries)
    ├── poc_proteins.json
    ├── work_dirs/{uid}/       # Boltz predict outputs + Vina configs
    ├── ligands/{uid}/         # Active + decoy PDB files
    ├── raw_results/{uid}/     # boltzina_results.csv per protein
    ├── boltzina_per_protein_results.csv
    └── boltzina_training_summary.json
```

## Known Issues

- **P03952, P08254**: first Uni-Dock batch (200 ligands) failed, causing all actives to be lost. Re-run with `--no-unidock` as fallback, or delete `boltzina_results.csv` and retry.
- **Receptor prep failures**: some experimental CIFs have non-standard residues or inter-chain bond artifacts. `prep_receptor_pdbqt()` falls back to chain-A-only extraction.
- **Structure prep bottleneck**: MMCIF parsing (~40s/complex) is the slowest phase even with 16 parallel workers. This is deep inside boltzina's `parse_mmcif()`.
