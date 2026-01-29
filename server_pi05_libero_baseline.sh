#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Baseline server launcher (NO VLA-OPT: no FiLM, no STE pruning).
# Edit these values to point at the checkpoint you want to evaluate.

CKPT_DIR="checkpoints/pi05_libero_spatial/pi05_baseline/30000"
POLICY_CONFIG="pi05_libero_spatial"

PORT="8002"
GPU="0"

[[ -f "${CKPT_DIR}/model.safetensors" ]] || { echo "Missing ${CKPT_DIR}/model.safetensors" >&2; exit 2; }

# For `pi05_libero_spatial`, norm stats are expected at the dataset repo_id path.
NORM_STATS_PATH="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json"
if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "Missing norm stats: ${NORM_STATS_PATH}" >&2
  echo "Hint: cd third_party/openpi && uv run scripts/compute_norm_stats.py --config-name ${POLICY_CONFIG}" >&2
  exit 2
fi

echo "Serve (baseline): ckpt=${CKPT_DIR} GPU=${GPU} PORT=${PORT}"
echo "Client: HOST=127.0.0.1 PORT=${PORT} TRIALS=1 bash client_libero_eval.sh"

# Tracing (default enabled, first dump only).
OPENPI_ROOT="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TRACE_OUT_DIR="${OPENPI_ROOT}/runs/openpi_pi05_libero_trace_${TS}"
TRACE_ATTN_LAYERS=""   # empty => last layer
TRACE_EVERY_N="1"
TRACE_MAX_DUMPS="1"
echo "Trace: out_dir=${TRACE_OUT_DIR} attn_layers=${TRACE_ATTN_LAYERS:-<last>} every_n=${TRACE_EVERY_N} max_dumps=${TRACE_MAX_DUMPS}"

SERVER_LOG="${SERVER_LOG:-${OPENPI_ROOT}/runs/openpi_pi05_libero_server_baseline_${TS}.log}"
mkdir -p "${OPENPI_ROOT}/runs"
echo "Log: ${SERVER_LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${PORT}" \
  --trace-out-dir "${TRACE_OUT_DIR}" \
  --trace-dump-attn \
  --trace-attn-layers "${TRACE_ATTN_LAYERS}" \
  --trace-every-n "${TRACE_EVERY_N}" \
  --trace-max-dumps "${TRACE_MAX_DUMPS}" \
  policy:checkpoint --policy.config "${POLICY_CONFIG}" --policy.dir "${CKPT_DIR}" 2>&1 | tee "${SERVER_LOG}"
