# 在 OpenPI π0.5（pi05）上集成 LightVLA（PyTorch 侧）

本文档记录将 LightVLA 的"可微视觉 token pruning"正确接入 OpenPI 的 π0.5（`pi05=True`）PyTorch 模型的实现方式，并给出可复现的验证步骤与结果。

> 背景：OpenPI 的 JAX 训练路径已包含"LightVLA-style"裁剪配置项（`Pi0Config.token_pruning_enabled` 等），但 PyTorch 推理模型 `PI0Pytorch` 侧此前没有稳定/可用的 LightVLA 裁剪实现。本次集成将 pruning 放在 **prefix 侧（图像 token）**，从而降低 prefix 长度与 KV cache 大小，进而减少后续 LLM self-attention 计算量。

---

## 0. 集成点与实现（摘要）

**集成点**
- [pi0_pytorch.py](../src/openpi/models_pytorch/pi0_pytorch.py) 的 `PI0Pytorch.embed_prefix()`
- 原因：prefix（图像 tokens + 语言 tokens）在此处生成并拼接；推理阶段 `sample_actions()` 会先对 prefix 计算 KV cache，因此 **缩短 prefix 会同时减少 cache 体积与后续注意力计算量**。

**实现方式**
- 新增 LightVLA 风格 pruner：[token_pruner.py](../src/openpi/models_pytorch/token_pruner.py)
  - `ImageTokenPruner(hidden_size=...)`：对单路相机的 `img_emb:[B,N,D]` 结合 `lang_emb:[B,T,D]` 计算重要性并裁剪
  - 训练态 `train()`：straight-through 可微选择（输出长度不变，便于训练/梯度）
  - 推理态 `eval()`：hard mask 真实删 token（batch>1 时 pad 到 batch 内最大保留数，避免 shape 不一致）

---

## 1. 核心代码位置

### 1.1 TokenPruner 模块

| 文件 | 类/函数 | 行号 | 说明 |
|------|---------|------|------|
| [token_pruner.py](../src/openpi/models_pytorch/token_pruner.py) | `_rms_norm()` | L27-32 | 无参数 RMSNorm，与 LightVLA 完全一致 |
| | `_LightVLACore` | L35-103 | 核心打分与选择逻辑 |
| | `_LightVLACore.compute_importance_score()` | L49-58 | 计算 patch 与 prompt 的相似度分数 |
| | `_LightVLACore.select_hard_mask()` | L60-67 | 推理态 argmax 选择 |
| | `_LightVLACore.select_soft()` | L69-75 | 训练态 straight-through estimator |
| | `ImageTokenPruner` | L106-167 | OpenPI 适配接口，输入分离的 image/lang tokens |
| | `TokenPruner` | L170-254 | 兼容 LightVLA 原始 `[cls, patches, task]` 序列结构 |

### 1.2 PI0Pytorch 集成

| 文件 | 位置 | 行号 | 说明 |
|------|------|------|------|
| [pi0_pytorch.py](../src/openpi/models_pytorch/pi0_pytorch.py) | `__init__` | L91-93 | 读取 `config.token_pruning_enabled` 配置 |
| | `__init__` | L120-124 | 条件创建 `ImageTokenPruner` |
| | `__init__` | L133-141 | 处理 `torch.compile` 与动态形状兼容性 |
| | `embed_prefix` | L233-241 | 配置训练/推理态的噪声强度 |
| | `embed_prefix` | L251-254 | 调用 `token_pruner.prune()` |
| | `set_token_pruner_noise_scale()` | L281-283 | 运行时覆盖噪声强度 |
| | `get_token_pruning_stats()` | L285-294 | 调试/监控接口 |

### 1.3 配置

| 文件 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| (动态读取) | `token_pruning_enabled` | `False` | 是否启用剪枝 |
| (动态读取) | `token_prune_noise_scale` | `0.0` | 训练态噪声强度 |

```python
# 配置示例
@dataclasses.dataclass
class MyPi05Config:
    pi05: bool = True
    token_pruning_enabled: bool = True  # 启用 LightVLA 剪枝
    token_prune_noise_scale: float = 0.5  # 训练时的噪声强度
    ...
```

---

## 2. 算法详解

### 2.1 打分机制（与 LightVLA 一致）

```python
# token_pruner.py L49-58
def compute_importance_score(self, patches, task_tokens):
    patches_n = _rms_norm(patches)      # RMSNorm
    task_n = _rms_norm(task_tokens)     # RMSNorm
    
    # 用 task tokens 作为 context，patches 作为 query
    queries = F.scaled_dot_product_attention(patches_n, task_n, task_n)
    queries = _rms_norm(queries)
    
    # 计算每个 query 与所有 patches 的相似度
    score = (queries @ patches_n.transpose(-2, -1)) * self.scale_factor  # [B, N, N]
    return score
```

### 2.2 推理态选择（真实删除 token）

```python
# token_pruner.py L60-67, L160-163
def select_hard_mask(self, score):
    indices = score.argmax(dim=-1)  # 每行选最重要的 patch
    mask[batch_indices, indices] = True  # 并集成为布尔 mask
    return mask  # [B, N]

# 推理时实际删除
if not self.training:
    hard_mask = self.core.select_hard_mask(score)
    pruned, pruned_mask = self.core._gather_and_pad(image_patches, hard_mask)
```

### 2.3 训练态选择（Straight-Through Estimator）

```python
# token_pruner.py L69-75, L153-158
def select_soft(self, score):
    if self.noise_scale is not None:
        score = score + torch.rand_like(score) * self.noise_scale  # 加噪声探索
    hard = F.one_hot(score.argmax(dim=-1), num_classes=N).to(score.dtype)
    soft = torch.softmax(score, dim=-1)
    return hard + soft - soft.detach()  # ST: 前向用 hard，反向用 soft 梯度

# 训练时不缩短序列
if self.training:
    weights = self.core.select_soft(score)  # [B, N, N]
    pruned = weights @ image_patches  # 加权组合，长度不变
```

---

## 3. 验证方法与结果

### 3.1 单元测试

```bash
cd third_party/openpi
uv run python -m pytest src/openpi/models_pytorch/token_pruner_test.py -v
```

**结果：4/4 passed** ✅
- `test_train_keeps_shape_and_is_differentiable`
- `test_eval_prunes_and_returns_mask`
- `test_eval_shortens_sequence`
- `test_train_keeps_sequence_length`

### 3.2 推理态验证（Demo）

```bash
cd third_party/openpi
uv run python examples/pi05_lightvla_pruning_demo.py
```

**结果：**
```
=== Prefix Token Lengths (eval) ===
no_prune:  tokens=800  pad_true=800
lightvla:  tokens=90   pad_true=90
reduction: 710 tokens (88.75%)

stats: {'enabled': True, 'noise_scale': None, 'last_kept_per_sample': [24]}
```

**解释：**
- baseline: `3 images × 256 patches + 32 lang = 800 tokens`
- LightVLA: `~24 patches (跨3图) + 32 lang = 90 tokens`
- **减少 88.75%** 的 prefix 长度

### 3.3 训练态验证（梯度传播）

```bash
uv run python -c "
import torch, dataclasses
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

@dataclasses.dataclass
class Cfg:
    pi05: bool = True
    paligemma_variant: str = 'dummy'
    action_expert_variant: str = 'dummy'
    dtype: str = 'float32'
    action_dim: int = 32
    action_horizon: int = 50
    token_pruning_enabled: bool = True
    token_prune_noise_scale: float = 0.5

model = PI0Pytorch(Cfg()).train()
images = [torch.randn(2, 3, 224, 224) for _ in range(3)]
img_masks = [torch.ones(2, dtype=torch.bool) for _ in range(3)]
lang_tokens = torch.randint(0, 257152, (2, 32))
lang_masks = torch.ones(2, 32, dtype=torch.bool)

embs, _, _ = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
embs.sum().backward()
print(f'Shape: {embs.shape}, requires_grad: {embs.requires_grad}')
print(f'Stats: {model.get_token_pruning_stats()}')
"
```

**结果：**
```
Shape: torch.Size([2, 800, 64])  # 训练态序列长度不变
requires_grad: True               # 可微
Stats: {'enabled': True, 'noise_scale': 0.5, 'last_kept_per_sample': None}
```

---

## 4. API 使用指南

### 4.1 启用剪枝

```python
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

@dataclasses.dataclass
class MyConfig:
    pi05: bool = True
    token_pruning_enabled: bool = True  # ← 启用
    token_prune_noise_scale: float = 0.5  # 训练态噪声
    ...

model = PI0Pytorch(config)
```

### 4.2 推理

```python
model.eval()  # 必须！推理态才会真实剪枝
with torch.inference_mode():
    actions = model.sample_actions(device, observation)
```

### 4.3 查看剪枝统计

```python
stats = model.get_token_pruning_stats()
# {'enabled': True, 'noise_scale': None, 'last_kept_per_sample': [24]}
```

### 4.4 运行时调整噪声

```python
model.set_token_pruner_noise_scale(0.8)  # 调高探索
model.set_token_pruner_noise_scale(None)  # 恢复配置默认值
```

---

## 5. 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `OPENPI_TORCH_COMPILE` | `1` | 是否使用 `torch.compile` |
| `OPENPI_TORCH_COMPILE_PRUNING` | `0` | 剪枝启用时是否强制编译（需 dynamic shapes） |

---

## 6. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/openpi/models_pytorch/token_pruner.py` | 新增 | LightVLA 剪枝核心 |
| `src/openpi/models_pytorch/token_pruner_test.py` | 新增 | 单元测试 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | 修改 | 集成剪枝调用 |
| `src/openpi/models_pytorch/gemma_config.py` | 新增 | PyTorch-only Gemma 配置 |
| `src/openpi/models_pytorch/gemma_pytorch.py` | 修改 | 修复 vision/text 维度 |
| `src/openpi/conftest.py` | 修改 | pynvml 可选 |
| `examples/pi05_lightvla_pruning_demo.py` | 新增 | 验证 demo |

---

## 7. 参考实现

LightVLA 原始 TokenPruner：
- [third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py:49-136](../../LightVLA/prismatic/extern/hf/modeling_prismatic.py)
- [LightVLA 优化报告](../../LightVLA/docs/lightvla_optimization_report.md)

---

## 8. PyTorch 训练启用剪枝

### 8.1 配置项（训练侧）

PyTorch 训练入口 `scripts/train_pytorch.py` 会把训练配置里的 `model`（`Pi0Config`）传给 `PI0Pytorch`，因此只需要在训练配置中开启：

- `token_pruning_enabled: bool = True`
- `token_prune_noise_scale: float = 0.5`（可选，训练态 straight-through 的探索噪声；推理态会自动禁用）
- （可选，主要影响 `model.eval()` 推理态）控制保留 token 数/比例：
  - `token_prune_keep_tokens: int | None = 24`（每路相机保留 K 个 patch tokens）
  - `token_prune_keep_ratio: float | None = 0.1`（每路相机保留 `ceil(ratio * num_patches)` 个 patch tokens）

### 8.2 使用 `uv run` 启动训练

单卡（示例使用内置的 `debug_pi05` 配置）：

```bash
cd third_party/openpi
uv run python scripts/train_pytorch.py debug_pi05 \
  --exp_name pi05_pruning_train \
  --model.token_pruning_enabled True \
  --model.token_prune_noise_scale 0.5
```

多卡（DDP）：

```bash
cd third_party/openpi
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py debug_pi05 \
  --exp_name pi05_pruning_train \
  --model.token_pruning_enabled True \
  --model.token_prune_noise_scale 0.5
```

### 8.2.1 （常用）先把 JAX checkpoint 转成 PyTorch 权重

`scripts/train_pytorch.py` 训练时如果要加载“基座模型权重”，需要 `pytorch_weight_path`（目录内包含 `model.safetensors`）。
可以用本仓库脚本把 JAX checkpoint 转换成 PyTorch：

```bash
cd third_party/openpi
uv run python examples/convert_jax_model_to_pytorch.py \
  --config_name pi05_aloha \
  --checkpoint_dir /workspace/laiminxin/models/pi05_base \
  --output_path /workspace/laiminxin/models/pi05_base_pytorch \
  --precision bfloat16
```

> [!NOTE]
> `--config_name` 必须是 `src/openpi/training/config.py` 里已有的 `TrainConfig.name`，脚本只用它来读取对应的 `Pi0Config`（例如 `pi05=True` 以及 `paligemma_variant/action_expert_variant`）。

### 8.2.2 使用预配置的训练配置

已在 `config.py` 中添加两个配置：

| 配置名 | 剪枝 | 用途 |
|--------|------|------|
| `pi05_pick_place_pytorch` | ❌ | 基准训练 |
| `pi05_pick_place_pytorch_pruning` | ✅ | 启用 LightVLA 剪枝 |

```python
# config.py 中已有配置（启用剪枝版）
TrainConfig(
    name="pi05_pick_place_pytorch_pruning",
    model=pi0_config.Pi0Config(
        pi05=True,
        token_pruning_enabled=True,      # LightVLA 剪枝已启用
        token_prune_noise_scale=0.5,
    ),
    data=LeRobotAlohaDataConfig(
        repo_id="/workspace/laiminxin/datasets/six_object_pick_conveyor",
        ...
    ),
    pytorch_weight_path="/workspace/laiminxin/models/pi05_base_pytorch",
    ...
)
```

**归一化（必须）**

训练 data loader 会自动做归一化，但要求能加载到 `norm_stats.json`。当前 `pi05_pick_place_pytorch(_pruning)` 配置会从下面位置读取：

- `assets_dir="/workspace/laiminxin/datasets"`
- `asset_id="six_object_pick_conveyor"`
- 读取路径：`/workspace/laiminxin/datasets/six_object_pick_conveyor/norm_stats.json`

如果该文件不存在，先计算一次：

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/compute_norm_stats.py --config-name pi05_pick_place_pytorch_pruning
```

**启动训练命令：**

```bash
cd third_party/openpi


CUDA_VISIBLE_DEVICES=1 uv run python scripts/compute_norm_stats.py --config-name pi05_pick_place_pytorch_pruning

# 不启用剪枝（基准）
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_pytorch.py pi05_pick_place_pytorch \
  --exp_name baseline_run1

# 启用剪枝（LightVLA 默认：noise_scale=1.0 线性衰减，implicit selection）
CUDA_VISIBLE_DEVICES=2 uv run python scripts/train_pytorch.py pi05_pick_place_pytorch_pruning \
  --exp_name prune_run1
```

> [!NOTE]
> 默认参数与 LightVLA 一致：
> - `token_prune_noise_scale=1.0`（会在训练循环中线性衰减到 0）
> - 使用 implicit selection（argmax 并集），不固定保留数量

### 8.3 训练时噪声日程（可选，LightVLA 策略）

LightVLA 原始实现使用**线性衰减噪声**：

```python
# LightVLA finetune.py 约 L1032
noise_scale = 1 - log_step / cfg.max_steps
vla.module.language_model.model.pruner.set_noise_scale(noise_scale)
```

在 OpenPI PyTorch 训练中实现（添加到训练循环，每步更新一次）：

```python
# 在 train_pytorch.py 的训练循环中添加
if hasattr(model, "set_token_pruner_noise_scale"):
    # 线性衰减噪声：前期更随机（探索），后期更确定（收敛）
    denom = max(1, config.num_train_steps - 1)
    noise_scale = max(0.0, 1.0 - (global_step / denom))
    raw_model = model.module if use_ddp else model
    raw_model.set_token_pruner_noise_scale(noise_scale)
```

### 8.4 验证训练已启用剪枝

在训练日志中查找：

```
INFO Enabled LightVLA-style token pruning (hidden_size=2048)
```

或在代码中检查：

```python
print(f"Token pruning enabled: {model.token_pruning_enabled}")
print(f"Pruner stats: {model.get_token_pruning_stats()}")
```

### 8.5 训练态行为说明

| 模式 | 序列长度 | 噪声 | 梯度 |
|------|----------|------|------|
| `model.train()` | 保持不变（800 tokens） | 使用 `noise_scale` | ✅ 通过 straight-through 回传 |
| `model.eval()` | 缩短（~90 tokens） | 禁用 | ❌ 无梯度 |

> [!NOTE]
> 训练时序列长度不变是 LightVLA 设计：通过可微的"软选择"学习剪枝策略，避免训练时的动态形状问题。真正的加速在推理时体现。

---

## 9. LIBERO 微调（对齐官方 pi05_libero）

### 9.1 配置说明

已创建两个**对齐官方 `pi05_libero` JAX 配置**的 PyTorch 训练配置：

| 配置名 | LightVLA 剪枝 | 用途 |
|--------|--------------|------|
| `pi05_libero_pytorch` | ❌ | 基线（全量微调） |
| `pi05_libero_pytorch_lightvla` | ✅ | LightVLA 剪枝版 |

**关键超参数（已对齐官方）：**

| 参数 | 官方 JAX | PyTorch 配置 |
|------|----------|-------------|
| `warmup_steps` | 10,000 | ✅ 10,000 |
| `peak_lr` | 5e-5 | ✅ 5e-5 |
| `decay_steps` | 1,000,000 | ✅ 1,000,000 |
| `decay_lr` | 5e-5 | ✅ 5e-5 |
| `batch_size` | 256 | 64（单卡调整）|
| `num_train_steps` | 30,000 | ✅ 30,000 |

### 9.2 训练流程

```bash
cd third_party/openpi

# 基线训练（无剪枝）
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_pytorch.py pi05_libero_pytorch \
  --exp_name libero_baseline_run1

CUDA_VISIBLE_DEVICES=1,2,4 uv run torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  scripts/train_pytorch.py pi05_libero_pytorch \
  --exp_name libero_baseline_run1 \
  --batch_size 48


# LightVLA 剪枝训练
CUDA_VISIBLE_DEVICES=2 uv run python scripts/train_pytorch.py pi05_libero_pytorch_lightvla \
  --exp_name libero_lightvla_run1


CUDA_VISIBLE_DEVICES=5,6,7 uv run torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  scripts/train_pytorch.py pi05_libero_pytorch_lightvla \
  --exp_name libero_lightvla_run1 \
  --batch_size 48
```

> [!NOTE]
> `norm_stats.json` 已预计算，无需重新生成。

### 9.3 评估

```bash
# 基线评估（libero_spatial）
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch \
  --policy.dir ./checkpoints/pi05_libero_pytorch/baseline_run1/30000" \
CLIENT_ARGS="--args.task-suite-name libero_spatial" \
docker compose -f examples/libero/compose.yml up --build

# LightVLA 评估（libero_spatial）
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_pytorch_lightvla \
  --policy.dir ./checkpoints/pi05_libero_pytorch_lightvla/lightvla_run1/30000" \
CLIENT_ARGS="--args.task-suite-name libero_spatial" \
docker compose -f examples/libero/compose.yml up --build
```

### 9.4 预期对比

| 配置 | LightVLA | libero_spatial 成功率 |
|------|----------|----------------------|
| pi05_libero (JAX 官方) | ❌ | 98.8% |
| pi05_libero_pytorch | ❌ | 待测 |
| pi05_libero_pytorch_lightvla | ✅ | 待测 |

