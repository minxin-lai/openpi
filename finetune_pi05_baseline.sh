#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Baseline finetune (NO VLA-OPT: no FiLM, no STE pruning).
# Edit these values if needed.
CONFIG="pi05_libero_spatial"
EXP="pi05_baseline"

PY=".venv/bin/python"
BASE_CKPT="/workspace/laiminxin/models/pi05_base_pytorch"
DATA_REPO_ID="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial"

LOG_FILE="checkpoints/${CONFIG}/${EXP}/train.log"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "Train (baseline): ${CONFIG}/${EXP} GPUs=4,5 log=${LOG_FILE}"
echo "Ckpt: checkpoints/${CONFIG}/${EXP}/<step>"

export WANDB_API_KEY="wandb_v1_W4H9EVGtleUH60hqT4qIa5sBCI7"

RESUME_FLAG=()
if [[ -d "checkpoints/${CONFIG}/${EXP}" ]] && find "checkpoints/${CONFIG}/${EXP}" -maxdepth 2 -type f -name "model.safetensors" -print -quit | grep -q .; then
  RESUME_FLAG=(--resume)
fi

CUDA_VISIBLE_DEVICES=4,5 "${PY}" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train_pytorch.py "${CONFIG}" \
  --exp-name "${EXP}" "${RESUME_FLAG[@]}" --pytorch-weight-path "${BASE_CKPT}" --data.repo_id "${DATA_REPO_ID}" \
  --batch-size 32 --num-workers 1 --log-interval 100 --save-interval 5000 --pytorch-training-precision bfloat16 --wandb-enabled \
  --log-level INFO 2>&1 | tee "${LOG_FILE}"
