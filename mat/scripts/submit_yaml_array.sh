#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mil/matsukawa/Multi-Agent-Transformer}"
CONFIG_PATH="${CONFIG_PATH:-mat/scripts/configs/smac_distillation_partial_enemy_jitter.yaml}"
RUN_NAME="${1:-${RUN_NAME:-}}"

cd "${REPO_ROOT}"

slurm_value() {
  local key="$1"
  local default="$2"
  uv run python - "${CONFIG_PATH}" "${key}" "${default}" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]

with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

slurm = data.get("slurm", {})
if not isinstance(slurm, dict):
    slurm = {}
print(slurm.get(key, default))
PY
}

resource_value() {
  local key="$1"
  local default="$2"
  uv run python - "${CONFIG_PATH}" "${key}" "${default}" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]

with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

resources = data.get("resources", {})
if not isinstance(resources, dict):
    resources = {}
print(resources.get(key, default))
PY
}

SLURM_SCRIPT="${SLURM_SCRIPT:-$(slurm_value script mat/scripts/run_yaml_slurm.sh)}"
GPUS_PER_TASK="${GPUS_PER_TASK:-$(slurm_value gpus_per_task "$(resource_value gpus_per_run 1)")}"
NUM_GPUS="${NUM_GPUS:-$(slurm_value num_gpus "$(resource_value num_gpus 1)")}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-$(slurm_value max_concurrent_runs "${NUM_GPUS}")}"

count_args=("${CONFIG_PATH}" --count)
if [[ -n "${RUN_NAME}" ]]; then
  count_args+=(--run "${RUN_NAME}")
fi

run_count="$(uv run python mat/scripts/run_yaml.py "${count_args[@]}")"
if [[ "${run_count}" -le 0 ]]; then
  echo "No expanded runs found for ${CONFIG_PATH} ${RUN_NAME}" >&2
  exit 1
fi

array_end=$((run_count - 1))
if [[ "${MAX_CONCURRENT_RUNS}" -le 0 ]]; then
  echo "MAX_CONCURRENT_RUNS must be positive" >&2
  exit 1
fi

echo "Submitting ${run_count} Slurm array tasks from ${CONFIG_PATH}"
echo "SLURM_SCRIPT=${SLURM_SCRIPT}"
echo "GPUS_PER_TASK=${GPUS_PER_TASK}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "MAX_CONCURRENT_RUNS=${MAX_CONCURRENT_RUNS}"

export CONFIG_PATH
if [[ -n "${RUN_NAME}" ]]; then
  export RUN_NAME
fi

sbatch --array=0-"${array_end}"%"${MAX_CONCURRENT_RUNS}" -G "${GPUS_PER_TASK}" "${SLURM_SCRIPT}"
