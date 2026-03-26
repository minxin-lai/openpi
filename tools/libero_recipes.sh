#!/usr/bin/env bash

# This file is intentionally a command cookbook.
# Primary use: copy one command block, replace a few variables, and run it
# from third_party/openpi.
# Most of the time you only need to change:
#   - CKPT_DIR
#   - OPT_CONFIG
#   - RUN_TAG
# Sometimes also:
#   - POLICY_CONFIG
#   - SUITE
#
# It is also safe to `source`; recipe execution happens inside subshells.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

recipe_repo_root() {
  local script_path=""
  if [ -n "${BASH_VERSION:-}" ]; then
    script_path="${BASH_SOURCE[0]}"
  elif [ -n "${ZSH_VERSION:-}" ]; then
    script_path="${(%):-%x}"
  else
    script_path="$0"
  fi
  cd "$(dirname "${script_path}")/.." && pwd
}

recipe_run() {
  local repo_dir
  repo_dir="$(recipe_repo_root)"
  (
    set -euo pipefail
    cd "${repo_dir}"
    "$@"
  )
}
# ============================================================================
# Quick Start
# ============================================================================

# Usage:
#   source tools/libero_recipes.sh
#   recipe_run_default
#   recipe_dump_spatial_quick
#   recipe_serve_pruning
#
# Edit the local variables inside a recipe if you need a different
# checkpoint / config / run-tag combination.

# ============================================================================
# Run
# ============================================================================

recipe_run_default() {
  local ckpt_dir="checkpoints/pi05_libero_all_cross_attn_post_gauss/cross_attn_post_t64_gauss/89999"
  local policy_config="pi05_libero_all_cross_attn_post_gauss"
  local opt_config="config/pruning/cross_attn_post_t64_gauss.yaml"
  local run_tag="cross_attn_post_t64_gauss"
  local suite="libero_goal"
  local trials="50"
  local gpu="0"
  local port="8003"

  recipe_run bash tools/run_libero.sh \
    --ckpt-dir "${ckpt_dir}" \
    --policy-config "${policy_config}" \
    --opt-config "${opt_config}" \
    --host 127.0.0.1 \
    --suite "${suite}" \
    --trials "${trials}" \
    --gpu "${gpu}" \
    --port "${port}" \
    --run-tag "${run_tag}"
}

recipe_run_spatial_quick() {
  local ckpt_dir="checkpoints/pi05_libero_spatial/post_t64/29999"
  local policy_config="pi05_libero_spatial"
  local opt_config="config/pruning/post_t64.yaml"
  local run_tag="post_t64"

  recipe_run bash tools/run_libero.sh \
    --ckpt-dir "${ckpt_dir}" \
    --policy-config "${policy_config}" \
    --opt-config "${opt_config}" \
    --host 127.0.0.1 \
    --suite libero_spatial \
    --trials 2 \
    --gpu 0 \
    --port 8003 \
    --run-tag "${run_tag}"
}

# ============================================================================
# Dump
# ============================================================================

recipe_dump_spatial_quick() {
  local ckpt_dir="checkpoints/pi05_libero_spatial/post_t64_gauss/29999"
  local policy_config="pi05_libero_spatial"
  local opt_config="config/pruning/post_t64_gauss.yaml"
  local run_tag="post_t64_gauss"
  local observe_config="/workspace/laiminxin/post_gauss_attn/configs/observe/infer_debug.json"

  recipe_run bash tools/run_libero_dump.sh \
    --ckpt-dir "${ckpt_dir}" \
    --policy-config "${policy_config}" \
    --opt-config "${opt_config}" \
    --observe-config "${observe_config}" \
    --host 127.0.0.1 \
    --suite libero_spatial \
    --trials 1 \
    --gpu 0 \
    --port 8003 \
    --run-tag "${run_tag}"
}

# ============================================================================
# Consensus Diagnose
# ============================================================================

# Two-stage consensus diagnose:
#   source tools/libero_recipes.sh
#   recipe_consensus_diagnose
#
# Stage 1 records real LIBERO policy queries.
# Stage 2 automatically finds the latest policy_records and replays them offline.

# ============================================================================
# Serve
# ============================================================================

recipe_serve_baseline() {
  local ckpt_dir="checkpoints/pi05_libero_spatial/pi05_baseline/29999"
  local policy_config="pi05_libero_spatial"
  local run_tag="baseline"

  recipe_run bash tools/serve_pi05_libero.sh \
    --ckpt-dir "${ckpt_dir}" \
    --policy-config "${policy_config}" \
    --gpu 0 \
    --port 8003 \
    --run-tag "${run_tag}"
}

recipe_serve_pruning() {
  local ckpt_dir="checkpoints/pi05_libero_all_cross_attn_post_gauss/cross_attn_post_t64_gauss/59999"
  local policy_config="pi05_libero_all_cross_attn_post_gauss"
  local opt_config="config/pruning/cross_attn_post_t64_gauss.yaml"
  local run_tag="cross_attn_post_t64_gauss"

  recipe_run bash tools/serve_pi05_libero.sh \
    --ckpt-dir "${ckpt_dir}" \
    --policy-config "${policy_config}" \
    --opt-config "${opt_config}" \
    --gpu 0 \
    --port 8003 \
    --run-tag "${run_tag}"
}

# ============================================================================
# Client
# ============================================================================

recipe_client_baseline() {
  local run_tag="baseline"

  recipe_run bash tools/client_libero.sh \
    --host 127.0.0.1 \
    --port 8003 \
    --suite libero_spatial \
    --trials 20 \
    --gpu 0 \
    --run-tag "${run_tag}"
}

# ============================================================================
# Train
# ============================================================================

recipe_train_baseline() {
  local policy_config="pi05_libero_all_cross_attn_post_gauss"
  local exp_name="baseline"

  recipe_run bash tools/train_pi05_baseline.sh \
    --policy-config "${policy_config}" \
    --exp-name "${exp_name}" \
    --checkpoint-base-dir /workspace/laiminxin/vla-opt/third_party/openpi/checkpoints \
    --gpus 1,2,3,4 \
    --project-name vla-opt \
    --base-ckpt /workspace/laiminxin/models/pi05_base_pytorch \
    --data-repo-id /workspace/laiminxin/datasets/lerobot_datasets/physical-intelligence/libero \
    --num-train-steps 60000
}

recipe_train_lora_jax_aligned() {
  local policy_config="pi05_libero_lora_pytorch"
  local exp_name="lora_jax_aligned"

  recipe_run bash tools/train_pi05_baseline.sh \
    --policy-config "${policy_config}" \
    --exp-name "${exp_name}" \
    --checkpoint-base-dir /workspace/laiminxin/vla-opt/third_party/openpi/checkpoints \
    --gpus 3,4 \
    --project-name vla-opt \
    --base-ckpt /workspace/laiminxin/models/pi05_base_pytorch \
    --data-repo-id /workspace/laiminxin/datasets/lerobot_datasets/physical-intelligence/libero \
    --batch-size 16 \
    --num-train-steps 30000
}

recipe_train_pruning_t64() {
  local policy_config="pi05_libero_all_cross_attn_post_gauss"
  local opt_config="config/pruning/cross_attn_post_t64_gauss.yaml"
  local exp_name="cross_attn_post_t64_gauss"

  recipe_run bash tools/train_pi05_experiment.sh \
    --opt-config "${opt_config}" \
    --policy-config "${policy_config}" \
    --exp-name "${exp_name}" \
    --checkpoint-base-dir /workspace/laiminxin/vla-opt/third_party/openpi/checkpoints \
    --gpus 1,2,3,4 \
    --base-ckpt /workspace/laiminxin/models/pi05_base_pytorch \
    --data-repo-id /workspace/laiminxin/datasets/lerobot_datasets/physical-intelligence/libero \
    --num-train-steps 60000
}

recipe_train_pruning_t128() {
  local policy_config="pi05_libero_all_cross_attn_post_gauss"
  local opt_config="config/pruning/cross_attn_post_t128_gauss.yaml"
  local exp_name="cross_attn_post_t128_gauss"

  recipe_run bash tools/train_pi05_experiment.sh \
    --opt-config "${opt_config}" \
    --policy-config "${policy_config}" \
    --exp-name "${exp_name}" \
    --checkpoint-base-dir /workspace/laiminxin/vla-opt/third_party/openpi/checkpoints \
    --gpus 5,6 \
    --base-ckpt /workspace/laiminxin/models/pi05_base_pytorch \
    --data-repo-id /workspace/laiminxin/datasets/lerobot_datasets/physical-intelligence/libero \
    --num-train-steps 90000
}

# ============================================================================
# Sweep Recipes
# ============================================================================

# Parallel full LIBERO eval on 4 GPUs:
# Usage:
#   1. edit ckpt_dir if needed
#   2. run: source tools/libero_recipes.sh && recipe_parallel_eval_libero_all
recipe_parallel_eval_libero_all() {
  local ckpt_dir="checkpoints/pi05_libero_all_cross_attn_post_gauss/cross_attn_post_t64_gauss/89999"
  local policy_config="pi05_libero_all_cross_attn_post_gauss"
  local opt_config="config/pruning/cross_attn_post_t64_gauss.yaml"
  local host="127.0.0.1"
  local trials="50"
  local repo_dir
  local vla_opt_repo_dir
  repo_dir="$(recipe_repo_root)"
  vla_opt_repo_dir="$(cd "${repo_dir}/../../.." && pwd)"

  (
    set -euo pipefail
    cd "${repo_dir}"
    export PYTHONPATH="${repo_dir}/src:${vla_opt_repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

    bash tools/run_libero.sh \
      --ckpt-dir "${ckpt_dir}" \
      --policy-config "${policy_config}" \
      --opt-config "${opt_config}" \
      --host "${host}" \
      --suite libero_spatial \
      --trials "${trials}" \
      --gpu 3 \
      --port 8001 \
      --run-tag cross_attn_post_step90k_spatial &

    bash tools/run_libero.sh \
      --ckpt-dir "${ckpt_dir}" \
      --policy-config "${policy_config}" \
      --opt-config "${opt_config}" \
      --host "${host}" \
      --suite libero_object \
      --trials "${trials}" \
      --gpu 4 \
      --port 8002 \
      --run-tag cross_attn_post_step90k_object &

    bash tools/run_libero.sh \
      --ckpt-dir "${ckpt_dir}" \
      --policy-config "${policy_config}" \
      --opt-config "${opt_config}" \
      --host "${host}" \
      --suite libero_goal \
      --trials "${trials}" \
      --gpu 3 \
      --port 8003 \
      --run-tag cross_attn_post_step90k_goal &

    bash tools/run_libero.sh \
      --ckpt-dir "${ckpt_dir}" \
      --policy-config "${policy_config}" \
      --opt-config "${opt_config}" \
      --host "${host}" \
      --suite libero_10 \
      --trials "${trials}" \
      --gpu 4 \
      --port 8004 \
      --run-tag cross_attn_post_step90k_libero10 &

    wait
  )
}

# ============================================================================
# Consensus Diagnose
# ============================================================================

# Usage:
#   1. edit the variables in the function if needed
#   2. run: source tools/libero_recipes.sh && recipe_consensus_diagnose
recipe_consensus_diagnose() {
  local ckpt_dir="checkpoints/pi05_libero_all_cross_attn_post_gauss/cross_attn_post_t64_gauss/89999"
  local policy_config="pi05_libero_all_cross_attn_post_gauss"
  local opt_config="config/pruning/cross_attn_post_t64_gauss.yaml"
  local host="127.0.0.1"
  local suite="libero_spatial"
  local trials="1"
  local gpu="0"
  local port="8003"
  local run_tag="consensus_dump"
  local record_limit="10"
  local match_top_m="3"
  local device="cuda:0"
  local torch_compile="0"
  local repo_dir
  local vla_opt_repo_dir
  local step_label
  local latest_run_dir
  local records_dir
  local output_dir
  repo_dir="$(recipe_repo_root)"
  vla_opt_repo_dir="$(cd "${repo_dir}/../../.." && pwd)"
  step_label="$(basename "${ckpt_dir}")"

  echo "== Step 1/2: record real LIBERO policy queries =="
  recipe_run env \
    OPENPI_TORCH_COMPILE="${torch_compile}" \
    bash tools/run_libero_dump.sh \
      --ckpt-dir "${ckpt_dir}" \
      --policy-config "${policy_config}" \
      --opt-config "${opt_config}" \
      --record-only \
      --host "${host}" \
      --suite "${suite}" \
      --trials "${trials}" \
      --gpu "${gpu}" \
      --port "${port}" \
      --run-tag "${run_tag}"

  latest_run_dir="$(find "runs/${run_tag}" -maxdepth 1 -mindepth 1 -type d -name "viz_${step_label}_*" | sort | tail -n 1)"
  [[ -n "${latest_run_dir}" ]] || {
    echo "Error: failed to find latest run under runs/${run_tag}/viz_${step_label}_*" >&2
    return 1
  }

  records_dir="${latest_run_dir}/server/policy_records"
  [[ -d "${records_dir}" ]] || {
    echo "Error: missing policy_records dir: ${records_dir}" >&2
    return 1
  }

  output_dir="${latest_run_dir}/offline_diagnose"

  echo
  echo "== Step 2/2: replay policy_records offline and render consensus PNGs =="
  recipe_run env \
    PYTHONPATH="${vla_opt_repo_dir}/src" \
    MPLCONFIGDIR=/tmp/mpl \
    "${repo_dir}/.venv/bin/python" tools/offline_consensus_diagnose.py \
      --ckpt-dir "${ckpt_dir}" \
      --policy-config "${policy_config}" \
      --opt-config "${opt_config}" \
      --records-dir "${records_dir}" \
      --output-dir "${output_dir}" \
      --record-limit "${record_limit}" \
      --match-top-m "${match_top_m}" \
      --device "${device}"

  echo
  echo "latest_run_dir: ${latest_run_dir}"
  echo "records_dir: ${records_dir}"
  echo "output_dir: ${output_dir}"
  echo "OPENPI_TORCH_COMPILE: ${torch_compile}"
  echo "png_dumps example:"
  echo "${output_dir}/observe/task_*/episode_*/query_*/png_dumps"
}


# ============================================================================
# Serve
# ============================================================================

# Baseline serve:
# bash -lc 'cd /workspace/laiminxin/post_gauss_attn/third_party/openpi && bash tools/serve_pi05_libero.sh \
#   --ckpt-dir checkpoints/pi05_libero_spatial/pi05_baseline/29999 \
#   --policy-config pi05_libero_spatial \
#   --gpu 0 \
#   --port 8003 \
#   --run-tag baseline'

# Pruning serve:
# bash -lc 'cd /workspace/laiminxin/post_gauss_attn/third_party/openpi && bash tools/serve_pi05_libero.sh \
#   --ckpt-dir checkpoints/pi05_libero_all_cross_attn_post_gauss/cross_attn_post_t64_gauss/59999 \
#   --policy-config pi05_libero_all_cross_attn_post_gauss \
#   --opt-config config/pruning/cross_attn_post_t64_gauss.yaml \
#   --gpu 0 \
#   --port 8003 \
#   --run-tag cross_attn_post_t64_gauss'

# ============================================================================
# Client
# ============================================================================

# Client only:
# bash -lc 'cd /workspace/laiminxin/post_gauss_attn/third_party/openpi && bash tools/client_libero.sh \
#   --host 127.0.0.1 \
#   --port 8003 \
#   --suite libero_spatial \
#   --trials 20 \
#   --gpu 0 \
#   --run-tag baseline'
