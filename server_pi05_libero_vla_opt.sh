#!/usr/bin/env bash
set -euo pipefail

# Pi0.5 (LIBERO Spatial) VLA-OPT server launcher.

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
  --ve-film-num-blocks <n>    default: 4
  --ste-prune-k <k>           default: 64
  --ste-prune-stage <stage>   default: gather
  --ste-prune-tau <tau>       default: 1.0
  --observe-config <path>     default: configs/observe/infer_light.json
EOF
}

ts="$(date +%Y%m%d_%H%M%S)"

policy_config="pi05_libero_spatial"
ckpt_dir="checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/30000"
gpu="0"
port="8003"

ve_film_num_blocks="4"
ste_prune_k="64"
ste_prune_stage="gather"
ste_prune_tau="1.0"
observe_config="configs/observe/infer_light.json"

log_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ckpt-dir) ckpt_dir="${2:?}"; shift 2 ;;
    --policy-config) policy_config="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    --log) log_path="${2:?}"; shift 2 ;;
    --ve-film-num-blocks) ve_film_num_blocks="${2:?}"; shift 2 ;;
    --ste-prune-k) ste_prune_k="${2:?}"; shift 2 ;;
    --ste-prune-stage) ste_prune_stage="${2:?}"; shift 2 ;;
    --ste-prune-tau) ste_prune_tau="${2:?}"; shift 2 ;;
    --observe-config) observe_config="${2:?}"; shift 2 ;;
    *) die "Unknown option: $1 (run --help)" ;;
  esac
done

[[ -f "${ckpt_dir}/model.safetensors" ]] || die "Missing ${ckpt_dir}/model.safetensors"
command -v uv >/dev/null 2>&1 || die "Missing 'uv' in PATH"

norm_stats_path="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json"
if [[ "${policy_config}" == "pi05_libero_spatial" && ! -f "${norm_stats_path}" ]]; then
  echo "Missing norm stats: ${norm_stats_path}" >&2
  echo "Hint: cd third_party/openpi && uv run scripts/compute_norm_stats.py --config-name ${policy_config}" >&2
  exit 2
fi
if [[ -n "${observe_config}" && ! -f "${observe_config}" ]]; then
  die "Missing observe config: ${observe_config}"
fi

mkdir -p runs

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
echo "prune: blocks=${ve_film_num_blocks} k=${ste_prune_k} stage=${ste_prune_stage} tau=${ste_prune_tau}"
echo "observe_config: ${observe_config}"

extra_args=(--vla-opt-observe-config "${observe_config}")

CUDA_VISIBLE_DEVICES="${gpu}" uv run scripts/serve_policy.py \
  --env LIBERO --port "${port}" \
  --vla-opt-ve-film --vla-opt-ve-film-num-blocks "${ve_film_num_blocks}" \
  --vla-opt-ste-prune --vla-opt-ste-prune-k "${ste_prune_k}" --vla-opt-ste-prune-stage "${ste_prune_stage}" --vla-opt-ste-prune-tau "${ste_prune_tau}" \
  "${extra_args[@]}" \
  policy:checkpoint --policy.config "${policy_config}" --policy.dir "${ckpt_dir}" 2>&1 | tee "${log_path}"
