#!/bin/bash
#SBATCH --job-name=boltzina-rescore
#SBATCH --account=katritch_223
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/boltzina_rescore_%A_%a.out

# Scoring-only re-run: skips Uni-Dock docking, only runs Boltz-2 affinity scoring.
# Use when docking completed but scoring failed (e.g. missing CCD cache).

module load conda
cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

RESULTS=benchmarks/05_boltzina/results
POC=$RESULTS/poc_proteins.json

# Extract protein for this task
PROT_UID=$(python3 -c "
import json, sys
proteins = json.load(open('$POC'))
idx = $SLURM_ARRAY_TASK_ID
if idx >= len(proteins):
    sys.exit(0)
print(proteins[idx]['uniprot_id'])
")

if [ -z "$PROT_UID" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: no protein assigned"
    exit 0
fi

RAW_DIR=$RESULTS/raw_results/$PROT_UID

# Skip if results already exist
if [ -f "$RAW_DIR/boltzina_results.csv" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: $PROT_UID already has results, skipping"
    exit 0
fi

# Skip if no docked output exists
if [ ! -d "$RAW_DIR/out" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: $PROT_UID has no docking output, skipping"
    exit 0
fi

echo "Task $SLURM_ARRAY_TASK_ID: Re-scoring $PROT_UID"

CONFIG=$RAW_DIR/config.json

# Verify config exists (was written during original docking run)
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: $CONFIG not found"
    exit 1
fi

conda run -n boltzina_env python external/boltzina/run.py \
    "$CONFIG" \
    --num_workers 16 \
    --vina_cpu 2 \
    --batch_size 4 \
    --skip_docking

echo "=== Re-scoring complete for $PROT_UID (task $SLURM_ARRAY_TASK_ID) ==="
