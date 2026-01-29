#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Usage:
#   bash finetune_pi05_ve_film_prune.sh [config] [exp] [extra train_pytorch flags...]
# Notes:
#   - Single GPU (CUDA_VISIBLE_DEVICES=0; nproc_per_node=1)
#   - BASE_CKPT: init weights (expects model.safetensors)
#   - DATA_REPO_ID: LeRobot dataset path (libero_spatial by default)
#   - You can override any flag by appending it at the end (last one wins).
# Vision Encoder (VE) - FiLM 调制
# - --ve-film：开启在 SigLIP 末端 encoder layer 注入 FiLM（用语言 embedding 生成 γ/β 调制视觉 token）。
# - --ve-film-num-blocks 4：包裹最后 4 个 SigLIP layer 做 FiLM；越大可调制层越多/参数更多。推理加载同一 ckpt 时必须保持一致，否则 state_dict key
#   对不上。
# - --no-ve-film-freeze-base：不冻结 base，全模型一起训（更吃显存/更慢）。若只想训 FiLM 模块用 --ve-film-freeze-base。
# Vision Encoder (VE) - STE 剪枝（Top‑K + STE + 两阶段）
# - --ste-prune：开启可训练剪枝模块（score head + Top‑K + STE）。
# - --ste-prune-k 64：每个 view 保留 64 个 patch tokens；K 越小越快但风险掉性能（Stage‑2 gather 后序列长度直接变 K）。
# - --ste-prune-stage auto：自动两阶段训练：先 mask（不变长，训练稳）→ 再 gather（变短对齐推理，真加速）。
# - --ste-prune-switch-step -1：auto 模式下 -1 表示默认在训练总步数的一半切到 gather（代码里是 num_train_steps//2）。
# - --ste-prune-tau 2.0 --ste-prune-tau-final 0.2：τ 线性退火（2.0 更“软”更稳 → 0.2 更“硬”更像 Top‑K），不设 tau-final 就恒定 τ。
# - --ste-prune-lambda-budget 0.01：预算正则权重，逼近“期望保留数≈K”。
# - --ste-prune-lambda-bin 0.01：二值化正则权重，让 soft 更接近 0/1（减少犹豫态）。
# - --no-ste-prune-freeze-base：不冻结 base，全模型一起训；只训 score head（+FiLM）用 --ste-prune-freeze-base。

CONFIG="pi05_libero_spatial"; EXP="vla_opt_pi05_ve_film_prune"
[[ $# -ge 1 ]] && CONFIG="$1" && shift
[[ $# -ge 1 ]] && EXP="$1" && shift

PY=".venv/bin/python"
BASE_CKPT="/workspace/laiminxin/models/pi05_base_pytorch"
DATA_REPO_ID="/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial"

LOG_FILE="checkpoints/${CONFIG}/${EXP}/train.log"; mkdir -p "$(dirname "${LOG_FILE}")"
OPENPI_ROOT="$(pwd)"
REPO_ROOT="$(cd ../.. && pwd)"
echo "Train: ${CONFIG}/${EXP} GPUs=4,5"
echo "CWD(openpi): ${OPENPI_ROOT}"
echo "RepoRoot(vla-opt): ${REPO_ROOT}"
echo "Log: ${OPENPI_ROOT}/${LOG_FILE}"
echo "Ckpt: checkpoints/${CONFIG}/${EXP}/<step>"
CUDA_VISIBLE_DEVICES=4,5 "${PY}" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train_pytorch.py "${CONFIG}" \
  --exp-name "${EXP}" --resume --pytorch-weight-path "${BASE_CKPT}" --data.repo_id "${DATA_REPO_ID}" \
  --batch-size 32 --num-workers 1 --log-interval 100 --save-interval 5000 --pytorch-training-precision bfloat16 --wandb-enabled \
  --ve-film --ve-film-num-blocks 4 --no-ve-film-freeze-base \
  --ste-prune --ste-prune-k 64 --ste-prune-stage auto --ste-prune-switch-step -1 --ste-prune-tau 2.0 --ste-prune-tau-final 0.2 \
  --ste-prune-lambda-budget 0.01 --ste-prune-lambda-bin 0.01 --no-ste-prune-freeze-base \
  --log-level INFO "$@" 2>&1 | tee "${LOG_FILE}"
