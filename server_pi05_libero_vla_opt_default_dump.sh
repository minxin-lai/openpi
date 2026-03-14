#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[deprecated] server_pi05_libero_vla_opt_default_dump.sh is a one-click runner, not a server-only launcher."
echo "[deprecated] use run_pi05_libero_vla_opt_default.sh instead."

bash "${script_dir}/run_pi05_libero_vla_opt_default.sh" "$@"
