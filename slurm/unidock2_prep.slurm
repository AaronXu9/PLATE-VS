#!/bin/bash
#SBATCH --job-name=unidock2_prep
#SBATCH --account=katritch_223
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/unidock2_prep_%j.out
#SBATCH --error=slurm/logs/unidock2_prep_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate /project2/katritch_223/aoxu/conda/envs/unidock2

cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

echo "[$(date)] starting prep on $(hostname)"

# 65 targets total (test/0p7 with >=10 actives). Each prep step writes
# its output JSON; downstream steps read those.
python benchmarks/04_docking/select_targets.py \
    --config benchmarks/04_docking/configs/gnina_full_config.yaml

python benchmarks/04_docking/prepare_structures.py \
    --config benchmarks/04_docking/configs/gnina_full_config.yaml

python benchmarks/04_docking/prepare_ligands.py \
    --config benchmarks/04_docking/configs/gnina_full_config.yaml \
    --n-workers 32

python benchmarks/04_docking/build_unidock2_batches.py \
    --config benchmarks/04_docking/configs/unidock2_config_carc.yaml

echo "[$(date)] PREP COMPLETE."
