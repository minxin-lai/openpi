#!/usr/bin/env bash
set -euo pipefail

# This file is intentionally a command cookbook.
# Uncomment one block at a time and run this script, or copy the command you need.



# ============================================================================
# Run
# ============================================================================

# Spatial suite run:
bash tools/run_libero.sh \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t64.yaml \
  --host 127.0.0.1 \
  --suite libero_spatial \
  --trials 2 \
  --gpu 0 \
  --port 8003 \
  --run-tag post_t64

# Full LIBERO run:
# bash tools/run_libero.sh \
#   --ckpt-dir checkpoints/pi05_libero_all_cross_attn_post_gauss/cross_attn_post_t64_gauss/59999 \
#   --policy-config pi05_libero_all_cross_attn_post_gauss \
#   --opt-config config/pruning/cross_attn_post_t64_gauss.yaml \
#   --host 127.0.0.1 \
#   --suite libero_goal \
#   --trials 50 \
#   --gpu 0 \
#   --port 8003 \
#   --run-tag cross_attn_post_t64_gauss

# ============================================================================
# Dump
# ============================================================================

# Dump run with observe:
# bash tools/run_libero_dump.sh \
#   --ckpt-dir checkpoints/pi05_libero_spatial/post_t64_gauss/29999 \
#   --policy-config pi05_libero_spatial \
#   --opt-config config/pruning/post_t64_gauss.yaml \
#   --observe-config /workspace/laiminxin/post_gauss_attn/configs/observe/infer_debug.json \
#   --host 127.0.0.1 \
#   --suite libero_spatial \
#   --trials 1 \
#   --gpu 0 \
#   --port 8003 \
#   --run-tag post_t64_gauss


# ============================================================================
# Train
# ============================================================================

# Full LIBERO baseline train:
# bash tools/train_pi05_baseline.sh \
#   --policy-config pi05_libero_all_cross_attn_post_gauss \
#   --exp-name baseline \
#   --checkpoint-base-dir /workspace/laiminxin/vla-opt/third_party/openpi/checkpoints \
#   --gpus 1,2,3,4 \
#   --project-name vla-opt \
#   --base-ckpt /workspace/laiminxin/models/pi05_base_pytorch \
#   --data-repo-id /workspace/laiminxin/datasets/lerobot_datasets/physical-intelligence/libero \
#   --num-train-steps 60000

# Full LIBERO pruning train:
# bash tools/train_pi05_experiment.sh \
#   --opt-config config/pruning/cross_attn_post_t64_gauss.yaml \
#   --policy-config pi05_libero_all_cross_attn_post_gauss \
#   --exp-name cross_attn_post_t64_gauss \
#   --checkpoint-base-dir /workspace/laiminxin/vla-opt/third_party/openpi/checkpoints \
#   --gpus 1,2,3,4 \
#   --base-ckpt /workspace/laiminxin/models/pi05_base_pytorch \
#   --data-repo-id /workspace/laiminxin/datasets/lerobot_datasets/physical-intelligence/libero \
#   --num-train-steps 60000

# ============================================================================
# Parallel Eval Sweep
# ============================================================================

# Parallel spatial eval sweep:
# Usage:
#   1. copy from `bash <<'SH'` to the ending `SH` and run directly
#   2. or run: source tools/libero_recipes.sh && recipe_parallel_eval_sweep
recipe_parallel_eval_sweep() {
  bash <<'SH'
gpu_arg="1,2,3,4"
IFS=',' read -r -a gpu_ids <<< "${gpu_arg}"
run_specs=(
  "inside_t64|checkpoints/pi05_libero_spatial/inside_t64/59999|config/pruning/inside_t64.yaml"
  "inside_t64_gauss|checkpoints/pi05_libero_spatial/inside_t64/59999|config/pruning/inside_t64_gauss.yaml"
  "inside_t128|checkpoints/pi05_libero_spatial/inside_t128/59999|config/pruning/inside_t128.yaml"
  "inside_t128_gauss|checkpoints/pi05_libero_spatial/inside_t128/59999|config/pruning/inside_t128_gauss.yaml"
  "post_t64|checkpoints/pi05_libero_spatial/post_t64/29999|config/pruning/post_t64.yaml"
  "post_t128|checkpoints/pi05_libero_spatial/post_t128/59999|config/pruning/post_t128.yaml"
  "post_t64_gauss|checkpoints/pi05_libero_spatial/post_t64/29999|config/pruning/post_t64_gauss.yaml"
  "post_t128_gauss|checkpoints/pi05_libero_spatial/post_t128/59999|config/pruning/post_t128_gauss.yaml"
)
queue_size="${#gpu_ids[@]}"
run_gpu_queue() {
  local worker_idx="$1"
  local gpu="$2"
  local spec_idx run_tag ckpt_dir opt_config
  for ((spec_idx=worker_idx; spec_idx<${#run_specs[@]}; spec_idx+=queue_size)); do
    IFS='|' read -r run_tag ckpt_dir opt_config <<< "${run_specs[spec_idx]}"
    bash tools/run_libero.sh \
      --run-tag "${run_tag}" \
      --ckpt-dir "${ckpt_dir}" \
      --policy-config pi05_libero_spatial \
      --opt-config "${opt_config}" \
      --host 127.0.0.1 \
      --suite libero_spatial \
      --trials 50 \
      --port "$((9001 + spec_idx))" \
      --gpu "${gpu}"
  done
}
pids=()
for idx in "${!gpu_ids[@]}"; do
  run_gpu_queue "${idx}" "${gpu_ids[idx]}" &
  pids[idx]=$!
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
SH
}

# ============================================================================
# Parallel Dump Sweep
# ============================================================================

# Parallel spatial dump sweep:
# Usage:
#   1. copy from `bash <<'SH'` to the ending `SH` and run directly
#   2. or run: source tools/libero_recipes.sh && recipe_parallel_dump_sweep
recipe_parallel_dump_sweep() {
  bash <<'SH'
observe_config="/workspace/laiminxin/post_gauss_attn/configs/observe/infer_debug.json"
run_dump() {
  local gpu="$1"
  local run_tag="$2"
  local ckpt_dir="$3"
  local opt_config="$4"
  local port="$5"
  bash tools/run_libero_dump.sh \
    --run-tag "${run_tag}" \
    --ckpt-dir "${ckpt_dir}" \
    --policy-config pi05_libero_spatial \
    --opt-config "${opt_config}" \
    --observe-config "${observe_config}" \
    --host 127.0.0.1 \
    --suite libero_spatial \
    --trials 1 \
    --port "${port}" \
    --gpu "${gpu}"
}
(
  run_dump 1 inside_t64 checkpoints/pi05_libero_spatial/inside_t64/59999 config/pruning/inside_t64.yaml 9001
  run_dump 1 inside_t64_gauss checkpoints/pi05_libero_spatial/inside_t64/59999 config/pruning/inside_t64_gauss.yaml 9002
  run_dump 1 inside_t128 checkpoints/pi05_libero_spatial/inside_t128/59999 config/pruning/inside_t128.yaml 9003
  run_dump 1 inside_t128_gauss checkpoints/pi05_libero_spatial/inside_t128/59999 config/pruning/inside_t128_gauss.yaml 9004
) &
pid_gpu1=$!
(
  run_dump 2 post_t64 checkpoints/pi05_libero_spatial/post_t64/29999 config/pruning/post_t64.yaml 9005
  run_dump 2 post_t128 checkpoints/pi05_libero_spatial/post_t128/59999 config/pruning/post_t128.yaml 9006
  run_dump 2 post_t64_gauss checkpoints/pi05_libero_spatial/post_t64/29999 config/pruning/post_t64_gauss.yaml 9007
  run_dump 2 post_t128_gauss checkpoints/pi05_libero_spatial/post_t128/59999 config/pruning/post_t128_gauss.yaml 9008
) &
pid_gpu2=$!
wait "${pid_gpu1}"
wait "${pid_gpu2}"
SH
}


# ============================================================================
# Serve
# ============================================================================

# Baseline serve:
# bash tools/serve_pi05_libero.sh \
#   --ckpt-dir checkpoints/pi05_libero_spatial/pi05_baseline/29999 \
#   --policy-config pi05_libero_spatial \
#   --gpu 0 \
#   --port 8003 \
#   --run-tag baseline

# Pruning serve:
# bash tools/serve_pi05_libero.sh \
#   --ckpt-dir checkpoints/pi05_libero_all_cross_attn_post_gauss/cross_attn_post_t64_gauss/59999 \
#   --policy-config pi05_libero_all_cross_attn_post_gauss \
#   --opt-config config/pruning/cross_attn_post_t64_gauss.yaml \
#   --gpu 0 \
#   --port 8003 \
#   --run-tag cross_attn_post_t64_gauss

# ============================================================================
# Client
# ============================================================================

# Client only:
# bash tools/client_libero.sh \
#   --host 127.0.0.1 \
#   --port 8003 \
#   --suite libero_spatial \
#   --trials 20 \
#   --gpu 0 \
#   --run-tag baseline
