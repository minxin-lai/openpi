#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
vla_opt_repo_dir="$(cd "${script_dir}/../../.." && pwd)"

die() { echo "Error: $*" >&2; exit 2; }

extract_ckpt_step() {
  local ckpt_path="$1"
  local step_label=""
  step_label="$(basename "${ckpt_path}")"
  [[ "${step_label}" =~ ^[0-9]+$ ]] || die "Checkpoint dir must end with numeric step, got: ${ckpt_path}"
  printf '%s\n' "${step_label}"
}

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_libero_dump.sh [options...]

Options:
  --ckpt-dir <dir>            required checkpoint directory containing model.safetensors
  --policy-config <name>      required policy config name
  --opt-config <path>         required pruning config
  --gpu <id>                  required CUDA_VISIBLE_DEVICES value
  --port <port>               required server port
  --run-tag <tag>             required run label
  --suite <name>              required LIBERO suite name
  --trials <n>                required number of trials per task
  --host <ip>                 required client host
  --launcher-log <path>       default: runs/<run_tag>/<ts>/server/launcher.log
  --server-log <path>         default: runs/<run_tag>/<ts>/server/server.log
  --client-log <path>         default: runs/<run_tag>/<ts>/client/client.log
  --video-out <path>          default: runs/<run_tag>/<ts>/client/videos
  --observe-config <path>     required observe config
  --observe-output-dir <dir>  optional explicit observe output dir
EOF
}

ckpt_dir=""
policy_config=""
opt_config=""
gpu=""
port=""
run_tag=""
suite=""
trials=""
host=""
launcher_log=""
server_log=""
client_log=""
video_out=""
observe_config=""
observe_output_dir=""

server_pid=""
run_dir=""
observe_dump_dir=""
render_status=0
video_status=0
stats_status=0
client_status=0

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill_server_tree TERM
    wait_for_server_exit 10 || true
    if kill -0 "${server_pid}" >/dev/null 2>&1; then
      kill_server_tree KILL
      wait_for_server_exit 5 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

list_child_pids() {
  local parent_pid="$1"
  ps -o pid= --ppid "${parent_pid}" 2>/dev/null | awk '{print $1}'
}

collect_descendant_pids() {
  local parent_pid="$1"
  local child_pid=""
  for child_pid in $(list_child_pids "${parent_pid}"); do
    collect_descendant_pids "${child_pid}"
    echo "${child_pid}"
  done
}

kill_server_tree() {
  local signal="$1"
  local child_pid=""
  [[ -n "${server_pid}" ]] || return

  for child_pid in $(collect_descendant_pids "${server_pid}"); do
    kill "-${signal}" "${child_pid}" >/dev/null 2>&1 || true
  done
}

wait_for_server_exit() {
  local timeout_s="$1"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      wait "${server_pid}" >/dev/null 2>&1 || true
      return 0
    fi
    sleep 1
  done
  return 1
}

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ckpt-dir) ckpt_dir="${2:?}"; shift 2 ;;
    --policy-config) policy_config="${2:?}"; shift 2 ;;
    --opt-config) opt_config="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    --run-tag) run_tag="${2:?}"; shift 2 ;;
    --suite) suite="${2:?}"; shift 2 ;;
    --trials) trials="${2:?}"; shift 2 ;;
    --host) host="${2:?}"; shift 2 ;;
    --launcher-log) launcher_log="${2:?}"; shift 2 ;;
    --server-log) server_log="${2:?}"; shift 2 ;;
    --client-log) client_log="${2:?}"; shift 2 ;;
    --video-out) video_out="${2:?}"; shift 2 ;;
    --observe-config) observe_config="${2:?}"; shift 2 ;;
    --observe-output-dir) observe_output_dir="${2:?}"; shift 2 ;;
    *) die "Unknown option: $1 (run --help)" ;;
  esac
done

[[ -n "${ckpt_dir}" ]] || die "--ckpt-dir is required"
[[ -n "${policy_config}" ]] || die "--policy-config is required"
[[ -n "${opt_config}" ]] || die "--opt-config is required for dump runs"
[[ -n "${gpu}" ]] || die "--gpu is required"
[[ -n "${port}" ]] || die "--port is required"
[[ -n "${run_tag}" ]] || die "--run-tag is required"
[[ -n "${suite}" ]] || die "--suite is required"
[[ -n "${trials}" ]] || die "--trials is required"
[[ -n "${host}" ]] || die "--host is required"
[[ -n "${observe_config}" ]] || die "--observe-config must not be empty"
[[ -f "${observe_config}" ]] || die "Missing observe config: ${observe_config}"

ts="$(date +%Y%m%d_%H%M%S)"
step_label="$(extract_ckpt_step "${ckpt_dir}")"
run_dir="runs/${run_tag}/viz_${step_label}_${ts}"
launcher_log="${launcher_log:-${run_dir}/server/launcher.log}"
server_log="${server_log:-${run_dir}/server/server.log}"
client_log="${client_log:-${run_dir}/client/client.log}"
video_out="${video_out:-${run_dir}/client/videos}"
observe_output_dir="${observe_output_dir:-${run_dir}/observe}"

mkdir -p "$(dirname "${launcher_log}")" "$(dirname "${server_log}")" "$(dirname "${client_log}")" "${video_out}" "${observe_output_dir}"

echo "=== OpenPI Run (server + client + postprocess) ==="
echo "ckpt: ${ckpt_dir}"
echo "policy_config: ${policy_config}"
echo "opt_config: ${opt_config}"
echo "gpu: ${gpu}"
echo "port: ${port}"
echo "host: ${host}"
echo "suite: ${suite}"
echo "trials: ${trials}"
echo "run_tag: ${run_tag}"
echo "step_label: ${step_label}"
echo "run_dir: ${run_dir}"
echo "launcher_log: ${launcher_log}"
echo "server_log: ${server_log}"
echo "client_log: ${client_log}"
echo "video_out: ${video_out}"
echo "observe_config: ${observe_config}"
echo "observe_output_dir: ${observe_output_dir}"
echo

bash "${script_dir}/serve_pi05_libero.sh" \
  --ckpt-dir "${ckpt_dir}" \
  --policy-config "${policy_config}" \
  --opt-config "${opt_config}" \
  --gpu "${gpu}" \
  --port "${port}" \
  --run-tag "${run_tag}" \
  --log "${server_log}" \
  --observe-config "${observe_config}" \
  --observe-output-dir "${observe_output_dir}" \
  >>"${launcher_log}" 2>&1 &
server_pid=$!

wait_for_port "${host}" "${port}" 120
wait_status=$?
if [[ "${wait_status}" -eq 2 ]]; then
  die "Server exited before becoming ready on ${host}:${port}. Check ${server_log}"
fi
if [[ "${wait_status}" -ne 0 ]]; then
  die "Server did not become ready on ${host}:${port}. Check ${server_log}"
fi

observe_dump_dir="${observe_output_dir}"
if [[ -n "${observe_dump_dir}" ]]; then
  echo "observe_dump_dir: ${observe_dump_dir}"
fi

set +e
bash "${script_dir}/client_libero.sh" \
  --host "${host}" \
  --port "${port}" \
  --suite "${suite}" \
  --trials "${trials}" \
  --gpu "${gpu}" \
  --run-tag "${run_tag}" \
  --video-out "${video_out}" \
  --log "${client_log}"
client_status=$?
set -e

cleanup
trap - EXIT INT TERM

if [[ -d "${observe_dump_dir}" ]]; then
  export PYTHONPATH="${vla_opt_repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
  set +e
  uv run python -m vla_opt.observe.render_png --run-dir "${observe_dump_dir}"
  render_status=$?
  uv run python scripts/render_observe_video.py --run-dir "${observe_dump_dir}" --overlay-kind both
  video_status=$?
  uv run python -m vla_opt.observe.pruning_stats --run-dir "${observe_dump_dir}"
  stats_status=$?
  set -e
fi

if [[ "${client_status}" -ne 0 ]]; then
  exit "${client_status}"
fi
if [[ "${render_status}" -ne 0 ]]; then
  exit "${render_status}"
fi
if [[ "${video_status}" -ne 0 ]]; then
  exit "${video_status}"
fi
if [[ "${stats_status}" -ne 0 ]]; then
  exit "${stats_status}"
fi
