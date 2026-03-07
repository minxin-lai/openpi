#!/usr/bin/env bash
set -euo pipefail

# LIBERO server launcher (generic; env-overridable)
#
# 这是一个“编辑/导出少量变量然后运行”的脚本。
# 如果你在跑 Pi0.5 Spatial baseline / vla-opt，优先用：
#   - server_pi05_libero_baseline.sh
#   - server_pi05_libero_vla_opt.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

model_path="${MODEL_PATH:-/workspace/laiminxin/models/pi05_libero_pytorch}"
server_gpu="${SERVER_GPU:-5}"
port="${PORT:-8002}"

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
echo "vla-opt: ve_film=${vla_opt_ve_film} ste_prune=${vla_opt_ste_prune}"
echo ""

[[ -d "${model_path}" ]] || { echo "Error: MODEL_PATH not found: ${model_path}" >&2; exit 1; }
[[ -f "scripts/serve_policy.py" ]] || { echo "Error: run from third_party/openpi (missing scripts/serve_policy.py)" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "Error: uv not found in PATH" >&2; exit 1; }

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
  "${debug_flags[@]}" \
  "${vla_opt_flags[@]}" \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "${model_path}"
