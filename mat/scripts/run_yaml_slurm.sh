#!/bin/bash

#SBATCH -p dgx-a100-40g
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -J smac-yaml
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matsukawa@mi.t.u-tokyo.ac.jp
#SBATCH -o dump/stdout.%J
#SBATCH -e dump/stderr.%J

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-mat/scripts/configs/experiments.yaml}"
RUN_NAME="${1:-${RUN_NAME:-partial_enemy_jitter}}"

cd "${REPO_ROOT}"
mkdir -p dump

echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "RUN_NAME: ${RUN_NAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

uv run python mat/scripts/run_yaml.py "${CONFIG_PATH}" --run "${RUN_NAME}"
