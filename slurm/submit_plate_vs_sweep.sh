#!/bin/bash
# Submit the 4-job PLATE-VS dual-encoder sweep to CARC.
#
# Usage:
#   bash slurm/submit_plate_vs_sweep.sh           # submit all 4
#   bash slurm/submit_plate_vs_sweep.sh hard_et   # submit just one
#
# Each job: ~6-10h on a CARC GPU; runs train_vs.py, then eval_real_inactives.py.
# Job name format: pvs_<tag> for easy identification in `squeue --me`.

set -euo pipefail

CONFIGS=(
    "hard_et      benchmarks/07_plate_vs_dl/configs/vs_hard_et.yaml"
    "hard_schnet  benchmarks/07_plate_vs_dl/configs/vs_hard_schnet.yaml"
    "soft_et      benchmarks/07_plate_vs_dl/configs/vs_soft_et.yaml"
    "soft_schnet  benchmarks/07_plate_vs_dl/configs/vs_soft_schnet.yaml"
)

FILTER="${1:-}"
SUBMITTED=0

for entry in "${CONFIGS[@]}"; do
    read -r TAG CONFIG <<< "${entry}"
    if [[ -n "${FILTER}" && "${TAG}" != "${FILTER}" ]]; then
        continue
    fi
    JOB_NAME="pvs_${TAG}"
    echo "Submitting ${JOB_NAME}  (config: ${CONFIG})"
    sbatch \
        --job-name="${JOB_NAME}" \
        --export="ALL,CONFIG=${CONFIG},RESULT_TAG=${TAG}" \
        slurm/run_plate_vs_dl.slurm
    SUBMITTED=$((SUBMITTED+1))
done

if [ "${SUBMITTED}" -eq 0 ]; then
    echo "Nothing matched filter '${FILTER}'. Available tags:" >&2
    for entry in "${CONFIGS[@]}"; do
        read -r TAG _ <<< "${entry}"
        echo "  ${TAG}" >&2
    done
    exit 1
fi

echo ""
echo "Submitted ${SUBMITTED} job(s). Monitor with: squeue --me"
echo "Logs: slurm/logs/plate_vs_dl_pvs_<tag>_<jobid>.{out,err}"
