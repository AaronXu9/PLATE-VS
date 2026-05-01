#!/bin/bash
#SBATCH --job-name=eval_real_inactives
#SBATCH --account=katritch_223
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/eval_%x_%j.out
#SBATCH --error=slurm/logs/eval_%x_%j.err

# Eval a trained PLATE-VS dual-encoder on real ChEMBL inactives.
#
# Usage:
#   sbatch --job-name=eval_hard_schnet \
#       --export=CKPT=benchmarks/07_plate_vs_dl/checkpoints_hard_schnet/best_vs_model.pt,\
# CONFIG=benchmarks/07_plate_vs_dl/configs/vs_hard_schnet.yaml,\
# RESULT_TAG=hard_schnet \
#       slurm/run_eval_real_inactives.slurm

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate binding_affinity

cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

: "${CKPT:?CKPT env var must be set}"
: "${CONFIG:?CONFIG env var must be set}"
: "${RESULT_TAG:?RESULT_TAG env var must be set}"

mkdir -p slurm/logs benchmarks/07_plate_vs_dl/results

OUT="benchmarks/07_plate_vs_dl/results/${RESULT_TAG}_real_inactives_eval.json"
echo "==== eval ${RESULT_TAG} on $(hostname) (GPU: ${CUDA_VISIBLE_DEVICES:-?}) ===="
echo "  CKPT  : ${CKPT}"
echo "  CONFIG: ${CONFIG}"
echo "  OUT   : ${OUT}"

python -u benchmarks/07_plate_vs_dl/eval_real_inactives.py \
    --checkpoint "${CKPT}" \
    --config "${CONFIG}" \
    --output "${OUT}"

echo "==== done at $(date) ===="
