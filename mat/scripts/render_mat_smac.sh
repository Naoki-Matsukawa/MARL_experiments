#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mil/matsukawa/Multi-Agent-Transformer}"
CONFIG_PATH="${CONFIG_PATH:-mat/scripts/configs/render_mat_smac.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-cu121/bin/python}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" mat/scripts/run_yaml.py "${CONFIG_PATH}" "$@"
