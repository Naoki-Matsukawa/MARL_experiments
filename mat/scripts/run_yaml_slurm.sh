#!/bin/bash

#SBATCH -p dgx-a100-40g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J smac-yaml
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp
#SBATCH -o dump/stdout.%A_%a
#SBATCH -e dump/stderr.%A_%a

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mil/matsukawa/Multi-Agent-Transformer}"
CONFIG_PATH="${CONFIG_PATH:-mat/scripts/configs/smac/smac_distillation_partial_enemy_jitter.yaml}"
RUN_NAME="${1:-${RUN_NAME:-}}"
RUN_INDEX="${SLURM_ARRAY_TASK_ID:-${RUN_INDEX:-0}}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
else
  PYTHON_CMD=(uv run python)
fi

cd "${REPO_ROOT}"

echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "RUN_NAME: ${RUN_NAME:-<all>}"
echo "RUN_INDEX: ${RUN_INDEX}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PYTHON_CMD: ${PYTHON_CMD[*]}"

"${PYTHON_CMD[@]}" - <<'PY'
import os
import sys

import torch

if not torch.cuda.is_available():
    print(
        "ERROR: Slurm allocated a GPU but torch.cuda.is_available() is False. "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"torch cuda available: {torch.cuda.get_device_name(0)}")
PY

args=("${CONFIG_PATH}")
if [[ -n "${RUN_NAME}" ]]; then
  args+=(--run "${RUN_NAME}")
fi
if [[ -n "${RUN_INDEX}" ]]; then
  args+=(--index "${RUN_INDEX}")
fi

"${PYTHON_CMD[@]}" mat/scripts/run_yaml.py "${args[@]}"
