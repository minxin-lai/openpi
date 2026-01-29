#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# One-file "edit and run" server launcher.
# - Edit the values below to point at the checkpoint you want to evaluate.
# - This starts the LIBERO WebSocket policy server (it does NOT do finetuning).
# - Enables VLA-OPT wrappers: Vision Encoder (VE) FiLM (num_film_blocks=4) + VE STE prune (K=64, stage=gather).
#
# Tracing (default enabled):
# - This script enables tracer dump by default.
# - To reduce noise, it only writes the first dump (typically the first inference of task0/ep0).
# - If you want more dumps, edit TRACE_MAX_DUMPS below.

# Checkpoint directory must contain `model.safetensors`.
# Example (this repo): checkpoints/pi05_libero_spatial/vla_opt_pi05_ve_film_prune/30000
CKPT_DIR="checkpoints/pi05_libero_spatial/vla_opt_pi05_ve_film_prune/30000"

# Must match the config used during training.
POLICY_CONFIG="pi05_libero_spatial"

# Server port and GPU.
PORT="8002"
GPU="7"
[[ -f "${CKPT_DIR}/model.safetensors" ]] || { echo "Missing ${CKPT_DIR}/model.safetensors" >&2; exit 2; }

# OpenPI root, used for placing logs/traces under `third_party/openpi/`.
OPENPI_ROOT="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${SERVER_LOG:-${OPENPI_ROOT}/runs/openpi_pi05_libero_server_${TS}.log}"
mkdir -p "${OPENPI_ROOT}/runs"

# ===== Norm stats check (required) =====
# For `pi05_libero_spatial`, OpenPI loads norm stats from the dataset directory (repo_id) by default.
NORM_STATS_PATH="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json"
if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "Missing norm stats: ${NORM_STATS_PATH}" >&2
  echo "Hint: cd third_party/openpi && uv run scripts/compute_norm_stats.py --config-name ${POLICY_CONFIG}" >&2
  exit 2
fi

echo "Serve: ckpt=${CKPT_DIR} GPU=${GPU} PORT=${PORT}"
echo "Client: HOST=127.0.0.1 PORT=${PORT} TRIALS=1 bash client_libero_eval.sh"
echo "Log: ${SERVER_LOG}"

TRACE_OUT_DIR="${OPENPI_ROOT}/runs/openpi_pi05_libero_trace_${TS}"
TRACE_ATTN_LAYERS=""   # empty => last layer
TRACE_EVERY_N="1"
TRACE_MAX_DUMPS="1"
echo "Trace: out_dir=${TRACE_OUT_DIR} attn_layers=${TRACE_ATTN_LAYERS:-<last>} every_n=${TRACE_EVERY_N} max_dumps=${TRACE_MAX_DUMPS}"

EXTRA_ARGS=(
  --trace-out-dir "${TRACE_OUT_DIR}"
  --trace-dump-attn
  --trace-attn-layers "${TRACE_ATTN_LAYERS}"
  --trace-every-n "${TRACE_EVERY_N}"
  --trace-max-dumps "${TRACE_MAX_DUMPS}"
)

CUDA_VISIBLE_DEVICES="${GPU}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${PORT}" \
  --vla-opt-ve-film --vla-opt-ve-film-num-blocks 4 \
  --vla-opt-ste-prune --vla-opt-ste-prune-k 64 --vla-opt-ste-prune-stage gather --vla-opt-ste-prune-tau 1.0 \
  "${EXTRA_ARGS[@]}" \
  policy:checkpoint --policy.config "${POLICY_CONFIG}" --policy.dir "${CKPT_DIR}" 2>&1 | tee "${SERVER_LOG}"
