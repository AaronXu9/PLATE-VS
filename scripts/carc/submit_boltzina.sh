#!/bin/bash
# Master submission script — chains all boltzina stages with dependencies
# Usage: bash scripts/carc/submit_boltzina.sh

set -e
cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

mkdir -p logs

echo "Submitting boltzina pipeline..."

# Stage 1-3: Prep (CPU)
jid1=$(sbatch --parsable scripts/carc/boltzina_prep.sh)
echo "  Prep job: $jid1"

# Stage 4: Docking + scoring (GPU array, waits for prep)
jid2=$(sbatch --parsable --dependency=afterok:$jid1 scripts/carc/boltzina_dock.sh)
echo "  Dock job: $jid2 (array 0-9, depends on $jid1)"

# Stage 5: Metrics (CPU, waits for all docking)
jid3=$(sbatch --parsable --dependency=afterok:$jid2 scripts/carc/boltzina_metrics.sh)
echo "  Metrics job: $jid3 (depends on $jid2)"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/boltzina_prep_\${jid1}.out"
