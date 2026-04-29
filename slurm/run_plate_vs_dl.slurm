#!/bin/bash
#SBATCH --job-name=plate_vs_dl
#SBATCH --account=katritch_223
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/plate_vs_dl_%x_%j.out
#SBATCH --error=slurm/logs/plate_vs_dl_%x_%j.err

# Train PLATE-VS dual-encoder + auto-evaluate on real ChEMBL inactives.
#
# Usage (one variant per submission):
#   sbatch --job-name=hard_et    --export=CONFIG=benchmarks/07_plate_vs_dl/configs/vs_hard_et.yaml,RESULT_TAG=hard_et    slurm/run_plate_vs_dl.slurm
#   sbatch --job-name=hard_schn  --export=CONFIG=benchmarks/07_plate_vs_dl/configs/vs_hard_schnet.yaml,RESULT_TAG=hard_schnet slurm/run_plate_vs_dl.slurm
#   sbatch --job-name=soft_et    --export=CONFIG=benchmarks/07_plate_vs_dl/configs/vs_soft_et.yaml,RESULT_TAG=soft_et    slurm/run_plate_vs_dl.slurm
#   sbatch --job-name=soft_schn  --export=CONFIG=benchmarks/07_plate_vs_dl/configs/vs_soft_schnet.yaml,RESULT_TAG=soft_schnet slurm/run_plate_vs_dl.slurm
#
# Or use the helper: bash slurm/submit_plate_vs_sweep.sh

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate binding_affinity

cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

# Required env vars from --export=
: "${CONFIG:?CONFIG env var must point to YAML config}"
: "${RESULT_TAG:?RESULT_TAG env var must be set (e.g. hard_et)}"

mkdir -p slurm/logs

echo "==== Job ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) on $(hostname) ===="
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  CONFIG     : ${CONFIG}"
echo "  RESULT_TAG : ${RESULT_TAG}"
echo "  Started at : $(date)"

python -c "
import torch, sys
print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}, mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
try:
    from torchmdnet.models.torchmd_et import TorchMD_ET
    print('  torchmd-net: available')
except Exception as e:
    print(f'  torchmd-net: NOT AVAILABLE ({e})')
"

# ---- 1. Train ----
echo ""
echo "==== STEP 1: Training ===="
# -u: unbuffered stdout/stderr so progress shows up live in slurm logs.
python -u benchmarks/07_plate_vs_dl/train_vs.py --config "${CONFIG}"

# ---- 2. Determine the trained checkpoint path (read from config) ----
# train_vs.py does os.chdir(benchmarks/07_plate_vs_dl) before training,
# so checkpoint_dir like "./checkpoints_hard_et" resolves there.
CKPT_DIR_REL=$(python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['output']['checkpoint_dir'])
")
SCRIPT_DIR_ABS="$(pwd)/benchmarks/07_plate_vs_dl"
CKPT_PATH=$(python -c "
import os
print(os.path.abspath(os.path.join('${SCRIPT_DIR_ABS}', '${CKPT_DIR_REL}', 'best_vs_model.pt')))
")
echo "  Checkpoint: ${CKPT_PATH}"

if [ ! -f "${CKPT_PATH}" ]; then
    echo "  WARN: Checkpoint not found, skipping inference"
    exit 0
fi

# ---- 3. Auto-eval: real ChEMBL inactives (pChEMBL >= 6 vs < 5) ----
echo ""
echo "==== STEP 2: Real inactive evaluation ===="
RESULT_OUT="benchmarks/07_plate_vs_dl/results/${RESULT_TAG}_real_inactives_eval.json"
python -u benchmarks/07_plate_vs_dl/eval_real_inactives.py \
    --checkpoint "${CKPT_PATH}" \
    --config "${CONFIG}" \
    --output "${RESULT_OUT}"

# Also save a backup of the standard training summary with the tag
# (train_vs.py writes vs_<threshold>_training_summary.json into ./results
#  relative to benchmarks/07_plate_vs_dl/ — same threshold across all 4 jobs!)
SIM_THRESH=$(python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['data']['similarity_threshold'])
")
TRAIN_SUMMARY="benchmarks/07_plate_vs_dl/results/vs_${SIM_THRESH}_training_summary.json"
TAGGED_SUMMARY="benchmarks/07_plate_vs_dl/results/${RESULT_TAG}_training_summary.json"
if [ -f "${TRAIN_SUMMARY}" ]; then
    cp "${TRAIN_SUMMARY}" "${TAGGED_SUMMARY}"
    echo "  Saved tagged training summary: ${TAGGED_SUMMARY}"
fi

echo ""
echo "==== Job complete at $(date) ===="
