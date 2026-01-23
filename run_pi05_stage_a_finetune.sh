#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="pi05_libero"
EXP_NAME="stage_a_pi05_finetune"
[[ $# -ge 1 ]] && CONFIG_NAME="$1" && shift
[[ $# -ge 1 ]] && EXP_NAME="$1" && shift

PYTHON=".venv/bin/python"
PI05_BASE_PYTORCH_CKPT="/workspace/laiminxin/models/pi05_base_pytorch"
[[ -x "${PYTHON}" ]] || { echo "Missing python: ${PYTHON}" >&2; exit 1; }
[[ -f "${PI05_BASE_PYTORCH_CKPT}/model.safetensors" ]] || { echo "Missing ${PI05_BASE_PYTORCH_CKPT}/model.safetensors" >&2; exit 1; }

LOG_FILE="checkpoints/${CONFIG_NAME}/${EXP_NAME}/train.log"
mkdir -p "$(dirname "${LOG_FILE}")"
echo "Logging to: ${LOG_FILE}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L >/dev/null 2>&1 || { echo "CUDA not available (nvidia-smi failed)"; exit 1; }
fi
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available()=False'; print('CUDA OK:', torch.cuda.get_device_name(0))" >/dev/null

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/train_pytorch.py "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --overwrite \
  --pytorch-weight-path "${PI05_BASE_PYTORCH_CKPT}" \
  --batch-size 32 --num-workers 1 --log-interval 100 --save-interval 5000 \
  --pytorch-training-precision bfloat16 \
  --no-wandb-enabled \
  --stage-a --no-stage-a-freeze-base --stage-a-num-film-blocks 4 \
  --log-level DEBUG \
  --data.repo_id "/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial" \
  "$@" 2>&1 | tee "${LOG_FILE}"
