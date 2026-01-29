#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# One-file "edit and run" client launcher.
# - Edit the values below to pick host/port/suite/trials/GPU.
# - This runs the LIBERO simulator and talks to the policy server over WebSocket.

CLIENT_GPU="0"
HOST="127.0.0.1"
PORT="8002"
TASK_SUITE="libero_spatial"   # libero_spatial|libero_object|libero_goal|libero_10
TRIALS="2"

VENV_DIR="examples/libero/.venv"
VIDEO_OUT_ROOT="runs/libero/videos"
VIDEO_OUT_PATH="${VIDEO_OUT_ROOT}/${TASK_SUITE}_$(date +%Y%m%d_%H%M%S)"

echo "=== OpenPI LIBERO Client ==="
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Suite: ${TASK_SUITE}"
echo "Trials: ${TRIALS}"
echo "GPU: ${CLIENT_GPU}"
echo "Video out: ${VIDEO_OUT_PATH}"
echo ""

if [ ! -f "examples/libero/main.py" ]; then
  echo "Error: run from third_party/openpi (missing examples/libero/main.py)"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Error: venv not found: ${VENV_DIR}"
  echo "Create it with: uv venv --python 3.8 ${VENV_DIR}"
  exit 1
fi

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"

CUDA_VISIBLE_DEVICES="${CLIENT_GPU}" python examples/libero/main.py \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE}" \
  --args.num-trials-per-task "${TRIALS}" \
  --args.video-out-path "${VIDEO_OUT_PATH}"
