#!/bin/bash
#SBATCH --job-name=boltzina-dock
#SBATCH --account=katritch_223
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/boltzina_dock_%A_%a.out

module load conda
cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

RESULTS=benchmarks/05_boltzina/results
BASE_DIR=$(pwd)
POC=$RESULTS/poc_proteins.json

# Extract single protein for this array task
PROTEIN_FILE=/tmp/boltzina_protein_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.json
python3 -c "
import json
proteins = json.load(open('$POC'))
if $SLURM_ARRAY_TASK_ID >= len(proteins):
    print(f'Task $SLURM_ARRAY_TASK_ID: no protein (only {len(proteins)} available)')
    exit(0)
p = proteins[$SLURM_ARRAY_TASK_ID]
with open('$PROTEIN_FILE', 'w') as f:
    json.dump([p], f)
print(f'Task $SLURM_ARRAY_TASK_ID: {p[\"uniprot_id\"]} ({p[\"n_actives\"]} actives)')
"

echo "=== Stage 04: Docking + scoring (task $SLURM_ARRAY_TASK_ID) ==="
conda run -n boltzina_env python benchmarks/05_boltzina/04_run_boltzina.py \
    --poc-proteins $PROTEIN_FILE \
    --results-dir $RESULTS \
    --boltzina-dir external/boltzina \
    --base-dir $BASE_DIR \
    --use-unidock \
    --unidock-env unidock2 \
    --batch-size 4 \
    --num-workers 16

rm -f $PROTEIN_FILE
echo "=== Docking complete (task $SLURM_ARRAY_TASK_ID) ==="
