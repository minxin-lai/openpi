#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/server_pi05_libero_vla_opt.sh" \
  --run-tag vla_opt_gauss \
  --pruning-config "${script_dir}/config/pruning/post_encoder_gauss.yaml" \
  "$@"
