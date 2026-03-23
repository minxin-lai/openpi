#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash tools/client_libero.sh [options...]

Options:
  --host <ip>         required server host
  --port <port>       required server port
  --suite <name>      required LIBERO suite name
  --trials <n>        required number of trials per task
  --gpu <id>          required CUDA_VISIBLE_DEVICES value
  --run-tag <tag>     required run label used for derived outputs
  --video-out <path>  optional explicit video output path
  --log <path>        optional explicit client log path
  --venv-dir <path>   optional LIBERO venv path, default: examples/libero/.venv
EOF
}

ts="$(date +%Y%m%d_%H%M%S)"
host=""
port=""
suite=""
trials=""
gpu=""
run_tag=""
video_out=""
log_path=""
venv_dir="examples/libero/.venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --host) host="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    --suite) suite="${2:?}"; shift 2 ;;
    --trials) trials="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --run-tag) run_tag="${2:?}"; shift 2 ;;
    --video-out) video_out="${2:?}"; shift 2 ;;
    --log) log_path="${2:?}"; shift 2 ;;
    --venv-dir) venv_dir="${2:?}"; shift 2 ;;
    *) die "Unknown option: $1 (run --help)" ;;
  esac
done

[[ -n "${host}" ]] || die "--host is required"
[[ -n "${port}" ]] || die "--port is required"
[[ -n "${suite}" ]] || die "--suite is required"
[[ -n "${trials}" ]] || die "--trials is required"
[[ -n "${gpu}" ]] || die "--gpu is required"
[[ -n "${run_tag}" ]] || die "--run-tag is required"

cd "${repo_dir}"
[[ -f "examples/libero/main.py" ]] || die "Run from third_party/openpi (missing examples/libero/main.py)"
[[ -d "${venv_dir}" ]] || die "Venv not found: ${venv_dir} (create: uv venv --python 3.8 ${venv_dir})"

if [[ -z "${video_out}" ]]; then
  video_out="runs/libero/videos/${run_tag}/${suite}_${ts}"
fi
if [[ -z "${log_path}" ]]; then
  log_path="runs/libero/logs/${run_tag}/${suite}_${ts}.log"
fi
mkdir -p "$(dirname "${log_path}")"

default_gl_backend="glx"
if [[ -z "${DISPLAY:-}" ]]; then
  default_gl_backend="egl"
fi

echo "=== OpenPI LIBERO Client ==="
echo "host: ${host}"
echo "port: ${port}"
echo "suite: ${suite}"
echo "trials: ${trials}"
echo "gpu: ${gpu}"
echo "run_tag: ${run_tag}"
echo "video_out: ${video_out}"
echo "log: ${log_path}"
echo "venv_dir: ${venv_dir}"
echo "mujoco_gl: ${MUJOCO_GL:-${default_gl_backend}}"
echo "pyopengl_platform: ${PYOPENGL_PLATFORM:-${default_gl_backend}}"
echo

# shellcheck disable=SC1091
source "${venv_dir}/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-${default_gl_backend}}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${default_gl_backend}}"

set +e
CUDA_VISIBLE_DEVICES="${gpu}" python examples/libero/main.py \
  --args.host "${host}" \
  --args.port "${port}" \
  --args.task-suite-name "${suite}" \
  --args.num-trials-per-task "${trials}" \
  --args.video-out-path "${video_out}" 2>&1 | tee "${log_path}"
status=$?
set -e

echo
echo "video_out: ${video_out}"
echo "client_log: ${log_path}"
exit "${status}"
