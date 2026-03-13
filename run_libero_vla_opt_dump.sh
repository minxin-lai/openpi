#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash run_libero_vla_opt_dump.sh [--gauss] [options...]

Options:
  --gauss                    enable gaussian pruning
  --ckpt-dir <dir>           forward to server
  --policy-config <name>     forward to server
  --gpu <id>                 default: 0
  --port <port>              default: 8003
  --run-tag <tag>            default: vla_opt_dump
  --suite <name>             default: libero_spatial
  --host <ip>                default: 127.0.0.1
  --launcher-log <path>      default: runs/<run_tag>/launcher_<ts>.log
  --server-log <path>        default: runs/<run_tag>/server_<ts>.log
  --client-log <path>        default: runs/libero/logs/<run_tag>/<suite>_<ts>.log
  --video-out <path>         default: runs/libero/videos/<run_tag>/<suite>_<ts>
EOF
}

gauss="0"
gpu="0"
port="8003"
run_tag="vla_opt_dump"
suite="libero_spatial"
host="127.0.0.1"
server_log=""
launcher_log=""
client_log=""
video_out=""

server_args=()
client_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gauss) gauss="1"; shift 1 ;;
    --ckpt-dir|--policy-config)
      server_args+=("$1" "${2:?}")
      shift 2
      ;;
    --gpu)
      gpu="${2:?}"
      server_args+=("$1" "$gpu")
      client_args+=("$1" "$gpu")
      shift 2
      ;;
    --port)
      port="${2:?}"
      server_args+=("$1" "$port")
      client_args+=("$1" "$port")
      shift 2
      ;;
    --run-tag)
      run_tag="${2:?}"
      shift 2
      ;;
    --suite)
      suite="${2:?}"
      client_args+=("$1" "$suite")
      shift 2
      ;;
    --host)
      host="${2:?}"
      client_args+=("$1" "$host")
      shift 2
      ;;
    --server-log)
      server_log="${2:?}"
      shift 2
      ;;
    --launcher-log)
      launcher_log="${2:?}"
      shift 2
      ;;
    --client-log)
      client_log="${2:?}"
      client_args+=("$1" "$client_log")
      shift 2
      ;;
    --video-out)
      video_out="${2:?}"
      client_args+=("$1" "$video_out")
      shift 2
      ;;
    *)
      die "Unknown option: $1 (run --help)"
      ;;
  esac
done

ts="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${server_log}" ]]; then
  server_log="runs/${run_tag}/server_${ts}.log"
fi
if [[ -z "${launcher_log}" ]]; then
  launcher_log="runs/${run_tag}/launcher_${ts}.log"
fi

server_args+=(--run-tag "$run_tag" --log "$server_log")
if [[ "${gauss}" == "1" ]]; then
  server_args+=(--ste-prune-gaussian --ste-prune-gaussian-sigma "0.65")
fi

server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$(dirname "${server_log}")"
mkdir -p "$(dirname "${launcher_log}")"

echo "=== One-Click VLA-OPT Dump Run ==="
echo "gauss: ${gauss}"
echo "gpu: ${gpu}"
echo "port: ${port}"
echo "run_tag: ${run_tag}"
echo "launcher_log: ${launcher_log}"
echo "server_log: ${server_log}"
echo ""

bash "${script_dir}/server_pi05_libero_vla_opt.sh" \
  "${server_args[@]}" \
  --observe-config "${repo_root}/configs/observe/infer_debug.json" \
  >>"${launcher_log}" 2>&1 &
server_pid=$!

wait_for_port() {
  local wait_host="$1"
  local wait_port="$2"
  local timeout_s="$3"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if python3 - "$wait_host" "$wait_port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
finally:
    sock.close()
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

if ! wait_for_port "${host}" "${port}" 120; then
  die "Server did not become ready on ${host}:${port}. Check ${server_log}"
fi

observe_dump_dir="$(python3 - "${launcher_log}" <<'PY'
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
value = ""
if log_path.is_file():
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("observe_dump_dir: "):
            value = line.split(": ", 1)[1].strip()
print(value)
PY
)"

if [[ -n "${observe_dump_dir}" ]]; then
  echo "observe_dump_dir: ${observe_dump_dir}"
fi

client_args+=(--run-tag "${run_tag}" --host "${host}" --port "${port}")

set +e
bash "${script_dir}/client_libero_eval_vla_opt_smoke.sh" "${client_args[@]}"
client_status=$?
set -e

cleanup
trap - EXIT INT TERM

render_status=0
stats_status=0
if [[ -n "${observe_dump_dir}" && -d "${observe_dump_dir}" ]]; then
  set +e
  PYTHONPATH="${repo_root}/src" uv run python -m vla_opt.observe.render_png --run-dir "${observe_dump_dir}"
  render_status=$?
  PYTHONPATH="${repo_root}/src" uv run python -m vla_opt.observe.pruning_stats --run-dir "${observe_dump_dir}"
  stats_status=$?
  set -e
fi

if [[ "${client_status}" -ne 0 ]]; then
  exit "${client_status}"
fi
if [[ "${render_status}" -ne 0 ]]; then
  exit "${render_status}"
fi
if [[ "${stats_status}" -ne 0 ]]; then
  exit "${stats_status}"
fi
