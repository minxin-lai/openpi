#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# One-click LIBERO eval client for the VLA-OPT server.
# Example:
#   HOST=127.0.0.1 PORT=8002 TRIALS=2 CLIENT_GPU=0 bash client_libero_eval_vla_opt.sh

exec bash "${SCRIPT_DIR}/client_libero_eval.sh"
