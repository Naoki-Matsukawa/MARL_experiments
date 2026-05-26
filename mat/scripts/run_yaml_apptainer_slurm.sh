#!/bin/bash

#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J smac-yaml-container
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp
#SBATCH -o dump/stdout.%J
#SBATCH -e dump/stderr.%J

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mil/matsukawa/Multi-Agent-Transformer}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${REPO_ROOT}/mat-smac_cuda12.sif}"
CONFIG_PATH="${CONFIG_PATH:-mat/scripts/configs/experiments.yaml}"
RUN_NAME="${1:-${RUN_NAME:-partial_enemy_jitter}}"

cd "${REPO_ROOT}"

echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "CONTAINER_IMAGE: ${CONTAINER_IMAGE}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "RUN_NAME: ${RUN_NAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

apptainer exec --nv \
    --bind "${REPO_ROOT}:/workspace/Multi-Agent-Transformer" \
    --env SC2PATH=/workspace/Multi-Agent-Transformer/3rdparty/StarCraftII \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_MODE="${WANDB_MODE:-online}" \
    "${CONTAINER_IMAGE}" \
    python mat/scripts/run_yaml.py "${CONFIG_PATH}" --run "${RUN_NAME}"
