# LightVLA + Pi0.5 推理命令汇总

本文档汇总 PyTorch 微调后的模型推理命令，包含基线和 LightVLA 剪枝对比实验。

---

## 1. Checkpoint 位置汇总

### Base 权重

| 类型 | 路径 |
|------|------|
| JAX 原始权重 | `/workspace/laiminxin/models/pi05_base` |
| PyTorch 转换后权重 | `/workspace/laiminxin/models/pi05_base_pytorch` |

### 自有数据集 (six_object_pick_conveyor)

| 配置 | LightVLA | Checkpoint 路径 | 状态 |
|------|----------|----------------|------|
| `pi05_pick_place_pytorch` | ❌ 基线 | `checkpoints/pi05_pick_place_pytorch/baseline_run1/30000` | ✅ |
| `pi05_pick_place_pytorch_pruning` | ✅ 剪枝 | `checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/30000` | ✅ |

### LIBERO 数据集

| 配置 | LightVLA | Checkpoint 路径 | 状态 |
|------|----------|----------------|------|
| `pi05_libero_pytorch` | ❌ 基线 | `checkpoints/pi05_libero_pytorch/libero_baseline_run1/30000` | ✅ |
| `pi05_libero_pytorch_lightvla` | ✅ 剪枝 | `checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000` | ✅ |

---

## 2. LIBERO 评估命令 (Docker)

### 2.1 基线 (无剪枝)

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

# LIBERO Spatial
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch \
  --policy.dir ./checkpoints/pi05_libero_pytorch/libero_baseline_run1/30000" \
CLIENT_ARGS="--args.task-suite-name libero_spatial" \
docker compose -f examples/libero/compose.yml up --build

# LIBERO Object
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch \
  --policy.dir ./checkpoints/pi05_libero_pytorch/libero_baseline_run1/30000" \
CLIENT_ARGS="--args.task-suite-name libero_object" \
docker compose -f examples/libero/compose.yml up --build

# LIBERO Goal
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch \
  --policy.dir ./checkpoints/pi05_libero_pytorch/libero_baseline_run1/30000" \
CLIENT_ARGS="--args.task-suite-name libero_goal" \
docker compose -f examples/libero/compose.yml up --build

# LIBERO 10
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch \
  --policy.dir ./checkpoints/pi05_libero_pytorch/libero_baseline_run1/30000" \
CLIENT_ARGS="--args.task-suite-name libero_10" \
docker compose -f examples/libero/compose.yml up --build
```

### 2.2 LightVLA 剪枝

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

# LIBERO Spatial
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000" \
CLIENT_ARGS="--args.task-suite-name libero_spatial" \
docker compose -f examples/libero/compose.yml up --build

# LIBERO Object
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000" \
CLIENT_ARGS="--args.task-suite-name libero_object" \
docker compose -f examples/libero/compose.yml up --build

# LIBERO Goal
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000" \
CLIENT_ARGS="--args.task-suite-name libero_goal" \
docker compose -f examples/libero/compose.yml up --build

# LIBERO 10
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000" \
CLIENT_ARGS="--args.task-suite-name libero_10" \
docker compose -f examples/libero/compose.yml up --build
```

---

## 3. 自有数据集推理命令 (serve_policy.py)

### 3.1 基线 (无剪枝)

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

CUDA_VISIBLE_DEVICES=5 uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch/baseline_run1/79999
```

### 3.2 LightVLA 剪枝 (默认自适应)

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

# 默认：使用训练时学到的 router 重要性评分自适应选择 tokens
uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999
```

### 3.3 LightVLA 剪枝 (指定保留比例)

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

# 保留 90% tokens
uv run python scripts/serve_policy.py \
  --port 8000 \
  --token-prune-keep-ratio 0.9 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999

# 保留 75% tokens
uv run python scripts/serve_policy.py \
  --port 8000 \
  --token-prune-keep-ratio 0.75 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999

# 保留 60% tokens
uv run python scripts/serve_policy.py \
  --port 8000 \
  --token-prune-keep-ratio 0.6 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999

# 或者指定固定保留数量 (例如每张图像保留 128 个 patch tokens)
uv run python scripts/serve_policy.py \
  --port 8000 \
  --token-prune-keep-tokens 128 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999
```

> **说明：**  
> - SigLIP-SO400M 对 224×224 图像生成 256 个 patch tokens (16×16 patches)  
> - `token_prune_keep_ratio=0.9` 保留 ~230 tokens，`0.75` 保留 ~192 tokens，`0.6` 保留 ~154 tokens  
> - 推理时会在控制台输出实际的剪枝前后 token 数量

---

## 4. 快速验证剪枝效果

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi
uv run python examples/pi05_lightvla_pruning_demo.py
```

**预期输出：**
```
=== Prefix Token Lengths (eval) ===
no_prune:  tokens=800  pad_true=800
lightvla:  tokens=90   pad_true=90
reduction: 710 tokens (88.75%)
```

---

## 5. 单元测试

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi
uv run python -m pytest src/openpi/models_pytorch/token_pruner_test.py -v
```
