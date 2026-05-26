#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
IMAGE="${IMAGE:-mat-smac:cuda12}"
CONFIG_PATH="${CONFIG_PATH:-mat/scripts/configs/experiments.yaml}"
RUN_NAME="${1:-${RUN_NAME:-partial_enemy_jitter}}"

cd "${REPO_ROOT}"

docker run --rm --gpus all --ipc=host \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_MODE="${WANDB_MODE:-online}" \
    -e SC2PATH=/workspace/Multi-Agent-Transformer/3rdparty/StarCraftII \
    -v "${REPO_ROOT}:/workspace/Multi-Agent-Transformer" \
    -w /workspace/Multi-Agent-Transformer \
    "${IMAGE}" \
    python mat/scripts/run_yaml.py "${CONFIG_PATH}" --run "${RUN_NAME}"
