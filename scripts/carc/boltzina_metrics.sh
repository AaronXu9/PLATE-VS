#!/bin/bash
#SBATCH --job-name=boltzina-metrics
#SBATCH --account=katritch_223
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/boltzina_metrics_%j.out

module load conda
cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset

echo "=== Stage 05: Collecting metrics ==="
conda run -n rdkit_env python benchmarks/05_boltzina/05_collect_results.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --registry training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results

echo "=== Metrics complete ==="
