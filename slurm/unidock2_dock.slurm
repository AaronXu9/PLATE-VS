#!/bin/bash
#SBATCH --job-name=unidock2_dock
#SBATCH --account=katritch_223
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/unidock2_dock_%A_%a.out
#SBATCH --error=slurm/logs/unidock2_dock_%A_%a.err

# 65 targets / 3 shards = ~22 targets per shard. At ~7 min/target we expect
# each shard to finish in ~2.5 hours. 12-hour walltime is a safety margin.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate /project2/katritch_223/aoxu/conda/envs/unidock2

cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
N_SHARDS=3

echo "[$(date)] shard ${SHARD_ID}/${N_SHARDS} on $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-unset}"

python benchmarks/04_docking/run_unidock2_benchmark.py \
    --config benchmarks/04_docking/configs/unidock2_config_carc.yaml \
    --shard-id ${SHARD_ID} --n-shards ${N_SHARDS}

echo "[$(date)] shard ${SHARD_ID} DONE."
