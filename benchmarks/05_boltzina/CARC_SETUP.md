# Running the Boltzina Pipeline on USC CARC

Instructions for running the Uni-Dock + Boltz-2 affinity scoring pipeline on USC's Center for Advanced Research Computing (CARC) cluster.

## 1. Clone the repo

```bash
cd /project/<PI_username>/<your_username>
git clone --recurse-submodules git@github.com:AaronXu9/PLATE-VS.git
cd PLATE-VS
git checkout feature/boltzina-benchmark
git submodule update --init --recursive
```

## 2. Set up conda environments

CARC uses modules + conda. You need three environments.

### 2a. boltzina_env (Boltz-2 scoring, pdb-tools, maxit)

```bash
module load conda
conda create -n boltzina_env python=3.10 -y
conda activate boltzina_env

# Install boltzina and its dependencies
pip install -e external/boltzina
pip install rdkit gemmi

# Install maxit (PDB/CIF converter, required for post-processing)
conda install -c bioconda maxit -y

# Install pdb-tools (should come with boltzina, verify)
which pdb_chain pdb_merge pdb_tidy
```

### 2b. unidock2 (GPU-accelerated docking)

```bash
conda create -n unidock2 -c conda-forge unidock -y

# Verify
conda run -n unidock2 unidock --version
```

### 2c. rdkit_env (ligand prep + metrics)

```bash
conda create -n rdkit_env python=3.9 -y
conda activate rdkit_env
pip install rdkit numpy pandas scikit-learn
```

## 3. Data setup

The pipeline needs the PLATE-VS dataset (experimental CIFs, registry, DeepCoy decoys):

```bash
# If plate-vs data is not in the repo, symlink or copy it:
ln -s /path/to/plate-vs plate-vs
```

Verify the data path:
```bash
ls plate-vs/VLS_benchmark/zipped_uniprot_raw_cif/uniprot_Q07820/cif_files_raw/
ls plate-vs/VLS_benchmark/chembl_affinity/uniprot_Q07820/deepcoy_output/
```

## 4. Download Boltz-2 model weights

Boltz-2 downloads model weights to `~/.boltz/` on first run. Do this interactively before submitting batch jobs:

```bash
salloc --partition=gpu --gres=gpu:1 --time=00:30:00 --mem=32G
conda activate boltzina_env
python -c "from boltz.main import process_input; print('Boltz cache ready')"
# This downloads ccd.pkl (~200MB) to ~/.boltz/
exit
```

## 5. SLURM job scripts

### 5a. Stage 02 + 03: Artifact generation + ligand prep (CPU only)

```bash
#!/bin/bash
#SBATCH --job-name=boltzina-prep
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep_%j.out

module load conda
cd /project/<PI>/<user>/PLATE-VS

REGISTRY=training_data_full/registry_soft_split_regression.csv
RESULTS=benchmarks/05_boltzina/results
POC=benchmarks/05_boltzina/results/poc_proteins.json
BASE_DIR=$(pwd)

# Stage 01: Select proteins (if poc_proteins.json doesn't exist)
if [ ! -f "$POC" ]; then
    conda run -n rdkit_env python benchmarks/05_boltzina/01_select_proteins.py \
        --registry $REGISTRY --base-dir $BASE_DIR \
        --output $POC --top-n 10
fi

# Stage 02: Generate boltz artifacts (NO GPU needed)
conda run -n boltzina_env python benchmarks/05_boltzina/02_prep_boltz.py \
    --poc-proteins $POC --registry $REGISTRY \
    --results-dir $RESULTS --base-dir $BASE_DIR

# Stage 03: Prepare ligands (SMILES → 3D PDB)
conda run -n rdkit_env python benchmarks/05_boltzina/03_prep_ligands.py \
    --poc-proteins $POC --registry $REGISTRY \
    --results-dir $RESULTS --base-dir $BASE_DIR
```

### 5b. Stage 04: Docking + scoring (GPU required)

Submit one job per protein for parallelism, or one job for all proteins sequentially.

**Option A: One job per protein (recommended for large-scale)**

```bash
#!/bin/bash
#SBATCH --job-name=boltzina-%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/dock_%A_%a.out

module load conda
cd /project/<PI>/<user>/PLATE-VS

RESULTS=benchmarks/05_boltzina/results
BASE_DIR=$(pwd)

# Extract single protein for this array task
python3 -c "
import json
proteins = json.load(open('$RESULTS/poc_proteins.json'))
p = proteins[$SLURM_ARRAY_TASK_ID]
with open('/tmp/protein_${SLURM_ARRAY_TASK_ID}.json', 'w') as f:
    json.dump([p], f)
print(f'Task {$SLURM_ARRAY_TASK_ID}: {p[\"uniprot_id\"]} ({p[\"n_actives\"]} actives)')
"

conda run -n boltzina_env python benchmarks/05_boltzina/04_run_boltzina.py \
    --poc-proteins /tmp/protein_${SLURM_ARRAY_TASK_ID}.json \
    --results-dir $RESULTS \
    --boltzina-dir external/boltzina \
    --base-dir $BASE_DIR \
    --use-unidock \
    --batch-size 4 \
    --num-workers 16
```

**Option B: All proteins sequentially (simpler)**

```bash
#!/bin/bash
#SBATCH --job-name=boltzina-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dock_%j.out

module load conda
cd /project/<PI>/<user>/PLATE-VS

conda run -n boltzina_env python benchmarks/05_boltzina/04_run_boltzina.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --results-dir benchmarks/05_boltzina/results \
    --boltzina-dir external/boltzina \
    --base-dir $(pwd) \
    --use-unidock \
    --batch-size 4 \
    --num-workers 16
```

### 5c. Stage 05: Collect metrics (CPU only)

```bash
#!/bin/bash
#SBATCH --job-name=boltzina-metrics
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/metrics_%j.out

module load conda
cd /project/<PI>/<user>/PLATE-VS

conda run -n rdkit_env python benchmarks/05_boltzina/05_collect_results.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --registry training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results
```

## 6. Submission workflow

```bash
# Create log directory
mkdir -p logs

# 1. Run prep (CPU)
jid1=$(sbatch --parsable scripts/boltzina_prep.sh)

# 2. Run docking + scoring (GPU, waits for prep)
jid2=$(sbatch --parsable --dependency=afterok:$jid1 scripts/boltzina_dock.sh)

# 3. Collect metrics (CPU, waits for docking)
sbatch --dependency=afterok:$jid2 scripts/boltzina_metrics.sh
```

## 7. Expected timing on A100

| Stage | Time (10 proteins, ~17K ligands) |
|-------|----------------------------------|
| Stage 02: Artifacts | ~2 min (CPU) |
| Stage 03: Ligand prep | ~30 min (CPU) |
| Stage 04: Docking + scoring | ~1 hr/protein (A100 GPU) |
| Stage 05: Metrics | <1 min (CPU) |

With SLURM array jobs (10 GPUs parallel): **~1-2 hours total**.

## 8. CARC-specific notes

- **GPU partitions**: Use `gpu` partition with `--gres=gpu:a100:1` (or `v100`, `p100` depending on availability). A100 preferred for Boltz-2 batch scoring.
- **Storage**: Use `/project/` for persistent data, `/scratch1/` for intermediate files. Results in `/project/` survive job completion.
- **Module conflicts**: If you get GLIBCXX errors, try `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` before running.
- **Conda on compute nodes**: Always `module load conda` in SLURM scripts. Login-node conda paths may not transfer.
- **Resumability**: All stages skip completed work. If a job times out, resubmit the same script — it picks up where it left off.
- **Uni-Dock batch failures**: Occasionally one batch of 200 ligands fails (bad PDBQT). The pipeline continues with remaining batches. Check logs for `[warn] Uni-Dock batch X failed`.
