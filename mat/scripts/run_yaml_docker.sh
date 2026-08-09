#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
IMAGE="${IMAGE:-mat-smac:cuda12}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
CONFIG_PATH="${CONFIG_PATH:-mat/scripts/configs/smac/smac_distillation_partial_enemy_jitter.yaml}"
RUN_NAME="${1:-${RUN_NAME:-}}"

cd "${REPO_ROOT}"

args=(python mat/scripts/run_yaml.py "${CONFIG_PATH}")
if [[ -n "${RUN_NAME}" ]]; then
    args+=(--run "${RUN_NAME}")
fi

docker run --rm --gpus "${DOCKER_GPUS}" --ipc=host \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_MODE="${WANDB_MODE:-online}" \
    -e SC2PATH=/workspace/Multi-Agent-Transformer/3rdparty/StarCraftII \
    -v "${REPO_ROOT}:/workspace/Multi-Agent-Transformer" \
    -w /workspace/Multi-Agent-Transformer \
    "${IMAGE}" \
    "${args[@]}"
