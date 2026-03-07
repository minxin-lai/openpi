#!/usr/bin/env bash
set -euo pipefail

# Pi0.5 (LIBERO Spatial) Baseline server launcher
#
# 你只需要记住：
#   1) 改“配置区”里 3~5 行（ckpt_dir / gpu / port）
#   2) `bash server_pi05_libero_baseline.sh`
#   3) 另一个终端跑 client：`bash client_libero_eval_baseline.sh`
#
# 这个脚本做什么：
# - 启动 WebSocket policy server（uv run scripts/serve_policy.py）
# - 可选开启 OPENPI_DEBUG（token/KV 形状等）

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash server_pi05_libero_baseline.sh [options...]

Options:
  --ckpt-dir <dir>            must contain model.safetensors
  --policy-config <name>      default: pi05_libero_spatial
  --gpu <id>                  default: 0 (CUDA_VISIBLE_DEVICES)
  --port <port>               default: 8002
  --log <path>                default: runs/openpi_pi05_libero_server_baseline_<ts>.log

Debug (OPENPI_DEBUG):
  --debug-token <bool>        default: false
  --debug-max-infer <n>       default: 1
  --debug-kv-layers <spec>    default: ends (ends|all|0,8,16...)
  --debug-kv-compare <bool>   default: false
EOF
}

ts="$(date +%Y%m%d_%H%M%S)"

# ======================
# 配置区（建议只改这里）
# ======================
policy_config="pi05_libero_spatial"
ckpt_dir="checkpoints/pi05_libero_spatial/pi05_baseline/30000"
gpu="0"
port="8002"

# debug：默认关
debug_token="false"
debug_max_infer="1"
debug_kv_layers="ends"
debug_kv_compare="false"

log_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ckpt-dir) ckpt_dir="${2:?}"; shift 2 ;;
    --policy-config) policy_config="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    --log) log_path="${2:?}"; shift 2 ;;
    --debug-token) debug_token="${2:?}"; shift 2 ;;
    --debug-max-infer) debug_max_infer="${2:?}"; shift 2 ;;
    --debug-kv-layers) debug_kv_layers="${2:?}"; shift 2 ;;
    --debug-kv-compare) debug_kv_compare="${2:?}"; shift 2 ;;
    *) die "Unknown option: $1 (run --help)" ;;
  esac
done

[[ -f "${ckpt_dir}/model.safetensors" ]] || die "Missing ${ckpt_dir}/model.safetensors"
command -v uv >/dev/null 2>&1 || die "Missing 'uv' in PATH"

# pi05_libero_spatial 必需 norm stats（默认路径；如你改了 dataset repo_id，按需改这里）
norm_stats_path="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json"
if [[ "${policy_config}" == "pi05_libero_spatial" && ! -f "${norm_stats_path}" ]]; then
  echo "Missing norm stats: ${norm_stats_path}" >&2
  echo "Hint: cd third_party/openpi && uv run scripts/compute_norm_stats.py --config-name ${policy_config}" >&2
  exit 2
fi

mkdir -p runs

if [[ -z "${log_path}" ]]; then
  log_path="runs/openpi_pi05_libero_server_baseline_${ts}.log"
fi
mkdir -p "$(dirname "${log_path}")"

# baseline：避免 shell 环境里遗留的 VLA-OPT 相关 env 影响
unset VLA_OPT_VE_FILM VLA_OPT_VE_FILM_NUM_BLOCKS
unset VLA_OPT_STE_PRUNE VLA_OPT_STE_PRUNE_K VLA_OPT_STE_PRUNE_STAGE VLA_OPT_STE_PRUNE_TAU
unset VLA_OPT_STE_PRUNE_LAYER VLA_OPT_STE_PRUNE_SCORE_MLP_HIDDEN_DIM

echo "=== OpenPI Server (baseline) ==="
echo "ckpt: ${ckpt_dir}"
echo "policy_config: ${policy_config}"
echo "gpu: ${gpu}"
echo "port: ${port}"
echo "log: ${log_path}"
echo ""
echo "Client (example):"
echo "  HOST=127.0.0.1 PORT=${port} TRIALS=20 bash client_libero_eval_baseline.sh"
echo ""

extra_args=()
if [[ "${debug_token}" == "true" ]]; then
  extra_args+=(
    --debug-token
    --debug-max-infer "${debug_max_infer}"
    --debug-variant "baseline"
    --debug-kv-layers "${debug_kv_layers}"
  )
  if [[ "${debug_kv_compare}" == "true" ]]; then
    extra_args+=(--debug-kv-compare)
  fi
fi

CUDA_VISIBLE_DEVICES="${gpu}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${port}" \
  "${extra_args[@]}" \
  policy:checkpoint --policy.config "${policy_config}" --policy.dir "${ckpt_dir}" 2>&1 | tee "${log_path}"
