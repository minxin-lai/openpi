#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

# ===== Config (override via env vars) =====
MODEL_PATH="${MODEL_PATH:-/workspace/laiminxin/models/pi05_libero_pytorch}"
SERVER_GPU="${SERVER_GPU:-5}"
PORT="${PORT:-8002}"

TRACE_OUT_DIR="${TRACE_OUT_DIR:-runs/openpi_pi05_libero_trace_$(date +%Y%m%d_%H%M%S)}"
# Force all outputs under `third_party/openpi/runs/` by normalizing relative paths.
if [[ "${TRACE_OUT_DIR}" != /* && "${TRACE_OUT_DIR}" != runs/* ]]; then
  TRACE_OUT_DIR="runs/${TRACE_OUT_DIR}"
fi
TRACE_DUMP_ATTN="${TRACE_DUMP_ATTN:-true}"
TRACE_ATTN_LAYERS="${TRACE_ATTN_LAYERS:-0,8,16}"   # Comma-separated, empty => last layer
TRACE_SAVE_IMAGES="${TRACE_SAVE_IMAGES:-true}"

echo "=== OpenPI LIBERO Server ==="
echo "Model: ${MODEL_PATH}"
echo "GPU: ${SERVER_GPU}"
echo "Port: ${PORT}"
echo "Trace out: ${TRACE_OUT_DIR}"
echo ""

mkdir -p "${TRACE_OUT_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Error: MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi
if [ ! -f "scripts/serve_policy.py" ]; then
  echo "Error: run from third_party/openpi (missing scripts/serve_policy.py)"
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found in PATH"
  exit 1
fi

EXP_DIR_FROM_REPO_ROOT="third_party/openpi/${TRACE_OUT_DIR}"
echo "After eval (recommended, from repo root):"
echo "  cd \"${REPO_ROOT}/third_party/openpi\" && PYTHONPATH=\"${REPO_ROOT}:${PYTHONPATH:-}\" uv run python -m tracer.plot_routing_overlays --exp_dir \"${TRACE_OUT_DIR}\""
mkdir -p runs
echo "${TRACE_OUT_DIR}" > "runs/_last_openpi_trace_dir.txt"
echo "${EXP_DIR_FROM_REPO_ROOT}" > "runs/_last_openpi_trace_exp_dir_from_repo_root.txt"
echo ""

TRACE_FLAGS=()
if [ "${TRACE_DUMP_ATTN}" = "true" ]; then
  TRACE_FLAGS+=(--trace-dump-attn)
fi
if [ "${TRACE_SAVE_IMAGES}" = "true" ]; then
  TRACE_FLAGS+=(--trace-save-policy-images)
else
  TRACE_FLAGS+=(--no-trace-save-policy-images)
fi

export TRITON_AUTOTUNE=0
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" uv run scripts/serve_policy.py \
  --env LIBERO \
  --port "${PORT}" \
  --trace-out-dir "${TRACE_OUT_DIR}" \
  --trace-attn-layers "${TRACE_ATTN_LAYERS}" \
  "${TRACE_FLAGS[@]}" \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "${MODEL_PATH}"
