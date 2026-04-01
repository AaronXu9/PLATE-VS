#!/bin/bash
#SBATCH --job-name=boltzina-prep
#SBATCH --account=katritch_223
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/boltzina_prep_%j.out

module load conda
cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

REGISTRY=training_data_full/registry_soft_split_regression.csv
RESULTS=benchmarks/05_boltzina/results
POC=$RESULTS/poc_proteins.json
BASE_DIR=$(pwd)

# Stage 01: Select proteins (run manually before submitting this job)
if [ ! -f "$POC" ]; then
    echo "ERROR: $POC not found. Run stage 01 manually first:"
    echo "  OPENBLAS_NUM_THREADS=1 conda run -n rdkit_env python benchmarks/05_boltzina/01_select_proteins.py \\"
    echo "    --registry $REGISTRY --base-dir \$(pwd) --output $POC --n 10"
    exit 1
fi

# Stage 02: Generate boltz artifacts (CPU only, uses experimental CIF structures)
echo "=== Stage 02: Boltz artifact prep ==="
conda run -n boltzina_env python benchmarks/05_boltzina/02_prep_boltz.py \
    --poc-proteins $POC --registry $REGISTRY \
    --results-dir $RESULTS --base-dir $BASE_DIR

# Stage 03: Prepare ligands (SMILES -> 3D PDB)
echo "=== Stage 03: Ligand preparation ==="
conda run -n rdkit_env python benchmarks/05_boltzina/03_prep_ligands.py \
    --poc-proteins $POC --registry $REGISTRY \
    --results-dir $RESULTS --base-dir $BASE_DIR

echo "=== Prep complete ==="
