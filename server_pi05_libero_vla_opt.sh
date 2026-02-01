#!/usr/bin/env bash
set -euo pipefail

# Pi0.5 (LIBERO Spatial) VLA-OPT server launcher
#
# 你只需要记住：
#   1) 改“配置区”里 3~5 行（ckpt_dir / gpu / port）
#   2) `bash server_pi05_libero_vla_opt.sh`
#   3) 另一个终端跑 client：`bash client_libero_eval_vla_opt.sh`
#
# 重要：VLA-OPT wrapper 的参数必须和训练 ckpt 一致，否则会 key 对不上 / 行为不一致。
# 默认启用：
# - VE FiLM: num_blocks=4
# - VE STE prune: k=64, stage=gather, tau=1.0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash server_pi05_libero_vla_opt.sh [options...]

Options:
  --ckpt-dir <dir>            must contain model.safetensors
  --policy-config <name>      default: pi05_libero_spatial
  --gpu <id>                  default: 0 (CUDA_VISIBLE_DEVICES)
  --port <port>               default: 8003
  --log <path>                default: runs/openpi_pi05_libero_server_vla_opt_<ts>.log

Tracing:
  --trace-llm-attn <bool>     default: true
  --trace-attn-layers <csv>   default: "" (empty => last layer)
  --trace-ve-attn <bool>      default: false
  --trace-ve-attn-layers <csv>default: "" (empty => last layer)
  --trace-every-n <n>         default: 1
  --trace-max-dumps <n>       default: 1
  --trace-out-dir <dir>       default: runs/openpi_pi05_libero_trace_<ts>

Debug (OPENPI_DEBUG):
  --debug-token <bool>        default: false
  --debug-max-infer <n>       default: 1
  --debug-kv-layers <spec>    default: ends (ends|all|0,8,16...)
  --debug-kv-compare <bool>   default: false

VLA-OPT wrappers (must match checkpoint):
  --ve-film-num-blocks <n>    default: 4
  --ste-prune-k <k>           default: 64
  --ste-prune-stage <stage>   default: gather
  --ste-prune-tau <tau>       default: 1.0
EOF
}

ts="$(date +%Y%m%d_%H%M%S)"

# ======================
# 配置区（建议只改这里）
# ======================
policy_config="pi05_libero_spatial"
ckpt_dir="checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/30000"
gpu="0"
port="8003"

# tracer：默认开（只 dump 1 次）
trace_llm_attn="true"
trace_attn_layers=""      # empty => last layer (LLM)
trace_ve_attn="false"
trace_ve_attn_layers=""   # empty => last layer (SigLIP)
trace_every_n="1"
trace_max_dumps="1"
trace_out_dir="runs/openpi_pi05_libero_trace_${ts}"

# debug：默认关
debug_token="false"
debug_max_infer="1"
debug_kv_layers="ends"
debug_kv_compare="false"

# VLA-OPT wrapper 参数（必须和训练一致）
ve_film_num_blocks="4"
ste_prune_k="64"
ste_prune_stage="gather"
ste_prune_tau="1.0"

log_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ckpt-dir) ckpt_dir="${2:?}"; shift 2 ;;
    --policy-config) policy_config="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    --log) log_path="${2:?}"; shift 2 ;;
    --trace-llm-attn) trace_llm_attn="${2:?}"; shift 2 ;;
    --trace-attn-layers) trace_attn_layers="${2:?}"; shift 2 ;;
    --trace-ve-attn) trace_ve_attn="${2:?}"; shift 2 ;;
    --trace-ve-attn-layers) trace_ve_attn_layers="${2:?}"; shift 2 ;;
    --trace-every-n) trace_every_n="${2:?}"; shift 2 ;;
    --trace-max-dumps) trace_max_dumps="${2:?}"; shift 2 ;;
    --trace-out-dir) trace_out_dir="${2:?}"; shift 2 ;;
    --debug-token) debug_token="${2:?}"; shift 2 ;;
    --debug-max-infer) debug_max_infer="${2:?}"; shift 2 ;;
    --debug-kv-layers) debug_kv_layers="${2:?}"; shift 2 ;;
    --debug-kv-compare) debug_kv_compare="${2:?}"; shift 2 ;;
    --ve-film-num-blocks) ve_film_num_blocks="${2:?}"; shift 2 ;;
    --ste-prune-k) ste_prune_k="${2:?}"; shift 2 ;;
    --ste-prune-stage) ste_prune_stage="${2:?}"; shift 2 ;;
    --ste-prune-tau) ste_prune_tau="${2:?}"; shift 2 ;;
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
mkdir -p "$(dirname "${trace_out_dir}")"
echo "${trace_out_dir}" > "runs/_last_openpi_trace_dir.txt" 2>/dev/null || true

if [[ -z "${log_path}" ]]; then
  log_path="runs/openpi_pi05_libero_server_vla_opt_${ts}.log"
fi
mkdir -p "$(dirname "${log_path}")"

echo "=== OpenPI Server (vla_opt) ==="
echo "ckpt: ${ckpt_dir}"
echo "policy_config: ${policy_config}"
echo "gpu: ${gpu}"
echo "port: ${port}"
echo "log: ${log_path}"
echo "trace_out_dir: ${trace_out_dir}"
echo "vla-opt: ve_film_num_blocks=${ve_film_num_blocks} ste_prune_k=${ste_prune_k} stage=${ste_prune_stage} tau=${ste_prune_tau}"
echo ""
echo "Client (example):"
echo "  HOST=127.0.0.1 PORT=${port} TRIALS=20 bash client_libero_eval_vla_opt.sh"
echo ""

extra_args=(
  --trace-out-dir "${trace_out_dir}"
  --trace-every-n "${trace_every_n}"
  --trace-max-dumps "${trace_max_dumps}"
  --trace-attn-layers "${trace_attn_layers}"
)
if [[ "${trace_llm_attn}" == "true" ]]; then
  extra_args+=(--trace-dump-attn)
fi
if [[ "${trace_ve_attn}" == "true" ]]; then
  extra_args+=(--trace-dump-ve-attn --trace-ve-attn-layers "${trace_ve_attn_layers}")
fi
if [[ "${debug_token}" == "true" ]]; then
  extra_args+=(
    --debug-token
    --debug-max-infer "${debug_max_infer}"
    --debug-variant "vla_opt"
    --debug-kv-layers "${debug_kv_layers}"
  )
  if [[ "${debug_kv_compare}" == "true" ]]; then
    extra_args+=(--debug-kv-compare)
  fi
fi

CUDA_VISIBLE_DEVICES="${gpu}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${port}" \
  --vla-opt-ve-film --vla-opt-ve-film-num-blocks "${ve_film_num_blocks}" \
  --vla-opt-ste-prune --vla-opt-ste-prune-k "${ste_prune_k}" --vla-opt-ste-prune-stage "${ste_prune_stage}" --vla-opt-ste-prune-tau "${ste_prune_tau}" \
  "${extra_args[@]}" \
  policy:checkpoint --policy.config "${policy_config}" --policy.dir "${ckpt_dir}" 2>&1 | tee "${log_path}"

