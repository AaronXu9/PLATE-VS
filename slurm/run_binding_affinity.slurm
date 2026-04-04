#!/bin/bash
#SBATCH --job-name=binding_affinity
#SBATCH --account=katritch_223
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/binding_affinity_%A_%a.out
#SBATCH --error=slurm/logs/binding_affinity_%A_%a.err

# Usage:
#   Single fold:  sbatch slurm/run_binding_affinity.slurm
#   All 5 folds:  sbatch --array=0-4 slurm/run_binding_affinity.slurm

eval "$(conda shell.bash hook)"
conda activate binding_affinity

cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

FOLD=${SLURM_ARRAY_TASK_ID:-0}
echo "Training fold ${FOLD} on $(hostname), GPU: ${CUDA_VISIBLE_DEVICES}"

python benchmarks/06_binding_affinity_model/train.py \
    --config benchmarks/06_binding_affinity_model/configs/default_config.yaml \
    --fold ${FOLD}

echo "Fold ${FOLD} complete."
