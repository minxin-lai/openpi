#!/usr/bin/env bash
set -euo pipefail

# Internal runner for one-click LIBERO eval with observe dump/postprocess.
# Prefer calling:
#   - run_pi05_libero_vla_opt_default.sh
#   - run_pi05_libero_vla_opt_gauss.sh

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

server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local wait_host="$1"
  local wait_port="$2"
  local timeout_s="$3"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      return 2
    fi
    if python3 - "$wait_host" "$wait_port" <<'PY'
import base64
import os
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    if sock.connect_ex((host, port)) != 0:
        raise SystemExit(1)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET / HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("ascii")
    sock.sendall(request)
    response = sock.recv(1024).decode("latin1", errors="ignore")
    raise SystemExit(0 if " 101 " in response or response.startswith("HTTP/1.1 101") else 1)
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

parse_args() {
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
}

prepare_args() {
  ts="$(date +%Y%m%d_%H%M%S)"
  if [[ -z "${server_log}" ]]; then
    server_log="runs/${run_tag}/server_${ts}.log"
  fi
  if [[ -z "${launcher_log}" ]]; then
    launcher_log="runs/${run_tag}/launcher_${ts}.log"
  fi
  mkdir -p "$(dirname "${server_log}")"
  mkdir -p "$(dirname "${launcher_log}")"
  server_args+=(--run-tag "$run_tag" --log "$server_log")
  if [[ "${gauss}" == "1" ]]; then
    server_args+=(--ste-prune-gaussian --ste-prune-gaussian-sigma "0.65")
  fi
  client_args+=(--run-tag "${run_tag}" --host "${host}" --port "${port}")
}

print_summary() {
  echo "=== OpenPI Run (server + client + postprocess) ==="
  echo "gauss: ${gauss}"
  echo "gpu: ${gpu}"
  echo "port: ${port}"
  echo "run_tag: ${run_tag}"
  echo "launcher_log: ${launcher_log}"
  echo "server_log: ${server_log}"
  echo
}

start_server() {
  bash "${script_dir}/server_pi05_libero_vla_opt.sh" \
    "${server_args[@]}" \
    --observe-config "${repo_root}/configs/observe/infer_debug.json" \
    >>"${launcher_log}" 2>&1 &
  server_pid=$!
}

ensure_server_ready() {
  wait_for_port "${host}" "${port}" 120
  wait_status=$?
  if [[ "${wait_status}" -eq 0 ]]; then
    return
  fi
  if [[ "${wait_status}" -eq 2 ]]; then
    die "Server exited before becoming ready on ${host}:${port}. Check ${server_log}"
  fi
  die "Server did not become ready on ${host}:${port}. Check ${server_log}"
}

read_observe_dump_dir() {
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
}

run_client() {
  set +e
  bash "${script_dir}/client_libero_eval_vla_opt_smoke.sh" "${client_args[@]}"
  client_status=$?
  set -e
}

run_postprocess() {
  render_status=0
  stats_status=0
  if [[ -z "${observe_dump_dir}" || ! -d "${observe_dump_dir}" ]]; then
    return
  fi
  set +e
  PYTHONPATH="${repo_root}/src" uv run python -m vla_opt.observe.render_png --run-dir "${observe_dump_dir}"
  render_status=$?
  PYTHONPATH="${repo_root}/src" uv run python -m vla_opt.observe.pruning_stats --run-dir "${observe_dump_dir}"
  stats_status=$?
  set -e
}

parse_args "$@"
prepare_args
print_summary
start_server
ensure_server_ready
read_observe_dump_dir
if [[ -n "${observe_dump_dir}" ]]; then
  echo "observe_dump_dir: ${observe_dump_dir}"
fi
run_client
cleanup
trap - EXIT INT TERM
run_postprocess

if [[ "${client_status}" -ne 0 ]]; then
  exit "${client_status}"
fi
if [[ "${render_status}" -ne 0 ]]; then
  exit "${render_status}"
fi
if [[ "${stats_status}" -ne 0 ]]; then
  exit "${stats_status}"
fi
