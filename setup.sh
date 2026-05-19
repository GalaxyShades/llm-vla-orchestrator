#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="envs/environment.yml"
ENV_NAME="llm-vla-orchestrator"
PYTHON_BIN="/home/lem/miniconda3/envs/${ENV_NAME}/bin/python"

need_cmd() { command -v "$1" >/dev/null 2>&1; }

echo "[1/4] Checking prerequisites..."
need_cmd git || { echo "ERROR: git not installed."; exit 1; }
need_cmd conda || { echo "ERROR: conda not installed."; exit 1; }

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: missing ${ENV_FILE}"
  exit 1
fi

echo "[2/4] Creating or updating Conda environment (${ENV_NAME})..."
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  conda env create -f "${ENV_FILE}"
fi

echo "[3/4] Verifying interpreter and key imports..."
conda run -n "${ENV_NAME}" python - <<'PY'
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import chess  # noqa: F401
import scipy  # noqa: F401
import yaml  # noqa: F401
print("Core imports: OK")
PY

echo "[4/4] Setup complete."
echo
echo "Activate environment with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Recommended interpreter:"
echo "  ${PYTHON_BIN}"
