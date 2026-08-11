#!/bin/bash

#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J smac-yaml-container
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp
#SBATCH -o dump/stdout.%A_%a
#SBATCH -e dump/stderr.%A_%a

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mil/matsukawa/Multi-Agent-Transformer}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${REPO_ROOT}/docker/sif/mat-smac_cuda12.sif}"
CONFIG_PATH="${CONFIG_PATH:-mat/scripts/configs/smac/smac_distillation_partial_enemy_jitter.yaml}"
RUN_NAME="${1:-${RUN_NAME:-}}"
RUN_INDEX="${SLURM_ARRAY_TASK_ID:-${RUN_INDEX:-0}}"

cd "${REPO_ROOT}"

echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "CONTAINER_IMAGE: ${CONTAINER_IMAGE}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "RUN_NAME: ${RUN_NAME:-<all>}"
echo "RUN_INDEX: ${RUN_INDEX}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# --force-python overrides each run's 'python:' field (usually a host venv
# path like ../../.venv-cu121/bin/python, which isn't what we want inside
# the container) with the image's own interpreter.
args=(python mat/scripts/run_yaml.py "${CONFIG_PATH}" --force-python python)
if [[ -n "${RUN_NAME}" ]]; then
    args+=(--run "${RUN_NAME}")
fi
if [[ -n "${RUN_INDEX}" ]]; then
    args+=(--index "${RUN_INDEX}")
fi

apptainer exec --nv \
    --bind "${REPO_ROOT}:/workspace/Multi-Agent-Transformer" \
    --env SC2PATH=/workspace/Multi-Agent-Transformer/3rdparty/StarCraftII \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_MODE="${WANDB_MODE:-online}" \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env JAX_PLATFORMS=cuda \
    --env PYTHONNOUSERSITE=1 \
    "${CONTAINER_IMAGE}" \
    "${args[@]}"
