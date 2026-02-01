#!/usr/bin/env bash
set -euo pipefail

# LIBERO server launcher (generic; env-overridable)
#
# 这是一个“编辑/导出少量变量然后运行”的脚本。
# 如果你在跑 Pi0.5 Spatial baseline / vla-opt，优先用：
#   - server_pi05_libero_baseline.sh
#   - server_pi05_libero_vla_opt.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${script_dir}"

model_path="${MODEL_PATH:-/workspace/laiminxin/models/pi05_libero_pytorch}"
server_gpu="${SERVER_GPU:-5}"
port="${PORT:-8002}"

trace_out_dir="${TRACE_OUT_DIR:-runs/openpi_pi05_libero_trace_$(date +%Y%m%d_%H%M%S)}"
if [[ "${trace_out_dir}" != /* && "${trace_out_dir}" != runs/* ]]; then
  trace_out_dir="runs/${trace_out_dir}"
fi
trace_dump_attn="${TRACE_DUMP_ATTN:-true}"
trace_attn_layers="${TRACE_ATTN_LAYERS:-0,8,16}" # csv, empty => last layer
trace_save_images="${TRACE_SAVE_IMAGES:-true}"

debug_token="${DEBUG_TOKEN:-false}"
debug_max_infer="${DEBUG_MAX_INFER:-1}"
debug_variant="${DEBUG_VARIANT:-}"
debug_kv_layers="${DEBUG_KV_LAYERS:-ends}"
debug_kv_compare="${DEBUG_KV_COMPARE:-false}"

vla_opt_ve_film="${VLA_OPT_VE_FILM:-false}"
vla_opt_ve_film_num_blocks="${VLA_OPT_VE_FILM_NUM_BLOCKS:-4}"
vla_opt_ste_prune="${VLA_OPT_STE_PRUNE:-false}"
vla_opt_ste_prune_k="${VLA_OPT_STE_PRUNE_K:-64}"
vla_opt_ste_prune_layer="${VLA_OPT_STE_PRUNE_LAYER:-}"
vla_opt_ste_prune_stage="${VLA_OPT_STE_PRUNE_STAGE:-gather}"
vla_opt_ste_prune_tau="${VLA_OPT_STE_PRUNE_TAU:-1.0}"
vla_opt_ste_prune_score_mlp_hidden_dim="${VLA_OPT_STE_PRUNE_SCORE_MLP_HIDDEN_DIM:-}"

echo "=== OpenPI LIBERO Server ==="
echo "model_path: ${model_path}"
echo "gpu: ${server_gpu}"
echo "port: ${port}"
echo "trace_out_dir: ${trace_out_dir}"
echo "vla-opt: ve_film=${vla_opt_ve_film} ste_prune=${vla_opt_ste_prune}"
echo ""

mkdir -p "${trace_out_dir}"

[[ -d "${model_path}" ]] || { echo "Error: MODEL_PATH not found: ${model_path}" >&2; exit 1; }
[[ -f "scripts/serve_policy.py" ]] || { echo "Error: run from third_party/openpi (missing scripts/serve_policy.py)" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "Error: uv not found in PATH" >&2; exit 1; }

mkdir -p runs
echo "${trace_out_dir}" > "runs/_last_openpi_trace_dir.txt"
echo "third_party/openpi/${trace_out_dir}" > "runs/_last_openpi_trace_exp_dir_from_repo_root.txt"
echo ""
echo "After eval (recommended, from repo root):"
echo "  cd \"${repo_root}/third_party/openpi\" && PYTHONPATH=\"${repo_root}:${PYTHONPATH:-}\" uv run python -m tracer.plot_routing_overlays --exp_dir \"${trace_out_dir}\""
echo ""

trace_flags=()
if [[ "${trace_dump_attn}" == "true" ]]; then
  trace_flags+=(--trace-dump-attn)
fi
if [[ "${trace_save_images}" == "true" ]]; then
  trace_flags+=(--trace-save-policy-images)
else
  trace_flags+=(--no-trace-save-policy-images)
fi

vla_opt_flags=()
if [[ "${vla_opt_ve_film}" == "true" ]]; then
  vla_opt_flags+=(--vla-opt-ve-film --vla-opt-ve-film-num-blocks "${vla_opt_ve_film_num_blocks}")
fi
if [[ "${vla_opt_ste_prune}" == "true" ]]; then
  vla_opt_flags+=(
    --vla-opt-ste-prune
    --vla-opt-ste-prune-k "${vla_opt_ste_prune_k}"
    --vla-opt-ste-prune-stage "${vla_opt_ste_prune_stage}"
    --vla-opt-ste-prune-tau "${vla_opt_ste_prune_tau}"
  )
  if [[ -n "${vla_opt_ste_prune_layer}" ]]; then
    vla_opt_flags+=(--vla-opt-ste-prune-layer "${vla_opt_ste_prune_layer}")
  fi
  if [[ -n "${vla_opt_ste_prune_score_mlp_hidden_dim}" ]]; then
    vla_opt_flags+=(--vla-opt-ste-prune-score-mlp-hidden-dim "${vla_opt_ste_prune_score_mlp_hidden_dim}")
  fi
fi

debug_flags=()
if [[ "${debug_token}" == "true" ]]; then
  debug_flags+=(--debug-token --debug-max-infer "${debug_max_infer}" --debug-kv-layers "${debug_kv_layers}")
  if [[ -n "${debug_variant}" ]]; then
    debug_flags+=(--debug-variant "${debug_variant}")
  fi
  if [[ "${debug_kv_compare}" == "true" ]]; then
    debug_flags+=(--debug-kv-compare)
  fi
fi

export TRITON_AUTOTUNE=0
CUDA_VISIBLE_DEVICES="${server_gpu}" uv run scripts/serve_policy.py \
  --env LIBERO \
  --port "${port}" \
  --trace-out-dir "${trace_out_dir}" \
  --trace-attn-layers "${trace_attn_layers}" \
  "${trace_flags[@]}" \
  "${debug_flags[@]}" \
  "${vla_opt_flags[@]}" \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "${model_path}"

