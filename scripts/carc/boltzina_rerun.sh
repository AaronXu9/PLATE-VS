#!/bin/bash
#SBATCH --job-name=boltzina-rerun
#SBATCH --account=katritch_223
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/boltzina_rerun_%j.out

# Re-run Stage 02 (artifact generation) after CCD cache fix.
# Deletes empty manifests, regenerates with proper CCD, verifies.

module load conda
cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

REGISTRY=training_data_full/registry_soft_split_regression.csv
RESULTS=benchmarks/05_boltzina/results
POC=$RESULTS/poc_proteins.json
BASE_DIR=$(pwd)

echo "=== Removing empty manifests ==="
for uid_dir in $RESULTS/work_dirs/*/; do
    uid=$(basename "$uid_dir")
    manifest_dir=$(ls -d "$uid_dir"boltz_results_*/processed 2>/dev/null)
    if [ -n "$manifest_dir" ] && [ -f "$manifest_dir/manifest.json" ]; then
        records=$(python3 -c "import json; print(len(json.load(open('$manifest_dir/manifest.json'))['records']))")
        if [ "$records" = "0" ]; then
            echo "  Removing empty manifest for $uid"
            rm "$manifest_dir/manifest.json"
        else
            echo "  $uid: $records records (keeping)"
        fi
    fi
done

echo "=== Re-running Stage 02: Boltz artifact generation ==="
conda run -n boltzina_env python benchmarks/05_boltzina/02_prep_boltz.py \
    --poc-proteins $POC --registry $REGISTRY \
    --results-dir $RESULTS --base-dir $BASE_DIR

echo "=== Verifying manifests ==="
for uid_dir in $RESULTS/work_dirs/*/; do
    uid=$(basename "$uid_dir")
    manifest_dir=$(ls -d "$uid_dir"boltz_results_*/processed 2>/dev/null)
    if [ -n "$manifest_dir" ] && [ -f "$manifest_dir/manifest.json" ]; then
        records=$(python3 -c "import json; print(len(json.load(open('$manifest_dir/manifest.json'))['records']))")
        echo "  $uid: $records records"
    else
        echo "  $uid: MISSING manifest!"
    fi
done

echo "=== Stage 02 re-run complete ==="
