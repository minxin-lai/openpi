# LightVLA vs Baseline 对比实验计划

## Checkpoint 配置

| 数据集 | 配置名 | LightVLA | Checkpoint 路径 |
|--------|--------|----------|----------------|
| 自有数据集 | `pi05_pick_place_pytorch` | ❌ | `checkpoints/pi05_pick_place_pytorch/baseline_run1/30000` |
| 自有数据集 | `pi05_pick_place_pytorch_pruning` | ✅ | `checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/30000` |
| LIBERO | `pi05_libero_pytorch` | ❌ | `checkpoints/pi05_libero_pytorch/libero_baseline_run1/30000` |
| LIBERO | `pi05_libero_pytorch_lightvla` | ✅ | `checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000` |

---

## Token 保留配置

### 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `token_prune_keep_tokens` | int | None | 每路相机保留 K 个 patch tokens |
| `token_prune_keep_ratio` | float | None | 每路相机保留 `ceil(ratio * num_patches)` 个 |

> 若两个都为 None，则使用 implicit selection (argmax 并集)

### 测试保留数量

| Keep Tokens | Keep Ratio | 预期保留数 (256 patches/camera) |
|-------------|------------|-------------------------------|
| None | None | 动态 (implicit) |
| 24 | - | 24/camera |
| 48 | - | 48/camera |
| - | 0.1 | 26/camera |
| - | 0.2 | 52/camera |

---

## 1. 推理延迟测试

### 测试命令 (自有数据集)

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

# Baseline (无剪枝)
CUDA_VISIBLE_DEVICES=6 uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch/baseline_run1/30000

# LightVLA (默认自适应剪枝)
CUDA_VISIBLE_DEVICES=7 uv run python scripts/serve_policy.py \
  --port 8001 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999

# LightVLA (保留 90% tokens)
CUDA_VISIBLE_DEVICES=7 uv run python scripts/serve_policy.py \
  --port 8001 \
  --token-prune-keep-ratio 0.9 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999

# LightVLA (保留 75% tokens)
CUDA_VISIBLE_DEVICES=7 uv run python scripts/serve_policy.py \
  --port 8001 \
  --token-prune-keep-ratio 0.75 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999

# LightVLA (保留 60% tokens)
CUDA_VISIBLE_DEVICES=7 uv run python scripts/serve_policy.py \
  --port 8001 \
  --token-prune-keep-ratio 0.6 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999

# LightVLA (固定保留 128 tokens)
CUDA_VISIBLE_DEVICES=7 uv run python scripts/serve_policy.py \
  --port 8001 \
  --token-prune-keep-tokens 128 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/pi05_pick_place_pytorch_pruning/prune_run1/39999
```

> **说明：**  
> - SigLIP-SO400M 对 224×224 图像生成 256 个 patch tokens (16×16 patches)  
> - 推理时控制台会输出实际的剪枝前后 token 数量  
> - 示例输出：`[Token Pruning] Image 0: before=256, after=192.0, reduced=64.0 (25.0%)`


### 测试命令 (LIBERO)

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

# Baseline (无剪枝)
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch \
  --policy.dir ./checkpoints/pi05_libero_pytorch/libero_baseline_run1/30000" \
CLIENT_ARGS="--args.task-suite-name libero_spatial" \
docker compose -f examples/libero/compose.yml up --build

# LightVLA (默认自适应剪枝)
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000" \
CLIENT_ARGS="--args.task-suite-name libero_spatial" \
docker compose -f examples/libero/compose.yml up --build

# LightVLA (保留 90% tokens)
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000 \
  --policy.config.model.token_prune_keep_ratio 0.9" \
CLIENT_ARGS="--args.task-suite-name libero_spatial" \
docker compose -f examples/libero/compose.yml up --build

# LightVLA (保留 75% tokens)
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000 \
  --policy.config.model.token_prune_keep_ratio 0.75" \
CLIENT_ARGS="--args.task-suite-name libero_spatial" \
docker compose -f examples/libero/compose.yml up --build

# LightVLA (保留 60% tokens)
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000 \
  --policy.config.model.token_prune_keep_ratio 0.6" \
CLIENT_ARGS="--args.task-suite-name libero_spatial" \
docker compose -f examples/libero/compose.yml up --build
```

> **提示：** 将 `libero_spatial` 替换为其他 suite：`libero_object`, `libero_goal`, `libero_10`


### 结果表格 (推理延迟)

| 数据集 | 模型 | Keep Tokens | 延迟 (ms) | GPU 内存 (GB) |
|--------|------|-------------|-----------|--------------|
| 自有 | Baseline | - | | |
| 自有 | LightVLA | implicit | | |
| 自有 | LightVLA | 24 | | |
| 自有 | LightVLA | 48 | | |
| LIBERO | Baseline | - | | |
| LIBERO | LightVLA | implicit | | |
| LIBERO | LightVLA | 24 | | |
| LIBERO | LightVLA | 48 | | |

---

## 2. 精度分析 (LIBERO)

### 测试命令

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

# Baseline
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch \
  --policy.dir ./checkpoints/pi05_libero_pytorch/libero_baseline_run1/30000" \
CLIENT_ARGS="--args.task-suite-name <SUITE>" \
docker compose -f examples/libero/compose.yml up --build

# LightVLA
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000" \
CLIENT_ARGS="--args.task-suite-name <SUITE>" \
docker compose -f examples/libero/compose.yml up --build
```

### 测试选项

| 参数 | 可选值 |
|------|--------|
| `<SUITE>` | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `--args.num-trials-per-task` | 默认 50 |

### 结果表格 (LIBERO 成功率)

| Task Suite | Baseline | LightVLA (implicit) | LightVLA (24) | LightVLA (48) |
|------------|----------|---------------------|---------------|--------------|
| libero_spatial | | | | |
| libero_object | | | | |
| libero_goal | | | | |
| libero_10 | | | | |

---

## 3. Attention Map 可视化

### 测试命令

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

CUDA_VISIBLE_DEVICES=6 uv run python scripts/visualize_attention.py \
  --config pi05_libero_pytorch_lightvla \
  --checkpoint ./checkpoints/pi05_libero_pytorch_lightvla/libero_lightvla_run1/40000 \
  --image_path <IMAGE> \
  --prompt "<PROMPT>" \
  --output_dir ./attention_vis/
```

### 测试选项

| 参数 | 说明 |
|------|------|
| `--image_path` | 输入图像路径 |
| `--prompt` | 任务描述 |
| `--keep_tokens` | 可选，覆盖默认 keep_tokens |

### 输出文件

| 文件名 | 说明 |
|--------|------|
| `attention_heatmap.png` | 重要性热力图叠加 |
| `kept_patches.png` | 被保留的 patch 标记 |

---

## 4. 鲁棒性测试

### 测试命令

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

CUDA_VISIBLE_DEVICES=6 uv run python scripts/robustness_test.py \
  --config <CONFIG> \
  --checkpoint <CHECKPOINT> \
  --perturbations <TYPE>
```

### 扰动选项

| 扰动类型 | 参数 | 可选值 |
|---------|------|--------|
| Gaussian Noise | `--noise_sigma` | 0.01, 0.05, 0.1 |
| Brightness | `--brightness_delta` | -0.3, -0.1, 0.1, 0.3 |
| Occlusion | `--occlude_size` | 32, 64 |
| Motion Blur | `--blur_kernel` | 3, 5, 7 |

### 结果表格 (鲁棒性)

| 扰动 | 参数 | Baseline | LightVLA (implicit) | LightVLA (24) |
|------|------|----------|---------------------|--------------|
| Clean | - | 0 | 0 | 0 |
| Noise | σ=0.05 | | | |
| Brightness | δ=-0.3 | | | |
| Occlusion | size=64 | | | |
| Blur | kernel=5 | | | |
