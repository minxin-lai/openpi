# 在 OpenPI π0.5（pi05）上集成 LightVLA（PyTorch 侧）

本文档记录将 LightVLA 的"可微视觉 token pruning"正确接入 OpenPI 的 π0.5（`pi05=True`）PyTorch 模型的实现方式，并给出可复现的验证步骤与结果。

> 背景:OpenPI 的 JAX 训练路径已包含"LightVLA-style"裁剪配置项(`Pi0Config.token_pruning_enabled` 等),但 PyTorch 推理模型 `PI0Pytorch` 侧此前没有稳定/可用的 LightVLA 裁剪实现。本次集成将 pruning 放在 **prefix 侧(图像 token)**,采用 **跨相机联合剪枝**(所有相机的 patch tokens 拼接后统一剪枝),从而降低 prefix 长度与 KV cache 大小,进而减少后续 LLM self-attention 计算量。

---

## 0. 集成点与实现（摘要）

**集成点**
- [pi0_pytorch.py](../src/openpi/models_pytorch/pi0_pytorch.py) 的 `PI0Pytorch.embed_prefix()` (L223-361)
- 原因:prefix(图像 tokens + 语言 tokens)在此处生成并拼接;**所有相机的图像 patch 在此处跨相机拼接后统一剪枝**;推理阶段 `sample_actions()` 会先对 prefix 计算 KV cache,因此 **缩短 prefix 会同时减少 cache 体积与后续注意力计算量**。

**实现方式(跨相机联合剪枝)**
- 新增 LightVLA 风格 pruner:[token_pruner.py](../src/openpi/models_pytorch/token_pruner.py) (573 行)
  - `ImageTokenPruner(hidden_size=...)`:对 **所有相机拼接后的** patch embeddings `all_img_emb:[B, N_total, D]` 结合 `lang_emb:[B,T,D]` 计算重要性并裁剪
  - **跨相机剪枝**:所有相机的 patches 先 `torch.cat()` 拼接,然后统一打分/选择 (L266-279 in pi0_pytorch.py)
  - 训练态 `train()`:straight-through 可微选择(输出长度不变,便于训练/梯度)
  - 推理态 `eval()`:hard mask 真实删 token(batch>1 时 pad 到 batch 内最大保留数,避免 shape 不一致)
  - **LightVLA-style position_ids**:训练时对齐 noisy selection 索引,推理时保留原始 patch 索引 (L325-359 in pi0_pytorch.py)）

---

## 1. 核心代码位置

### 1.1 TokenPruner 模块

| 文件 | 类/函数 | 行号 | 说明 |
|------|---------|------|------|
| [token_pruner.py](../src/openpi/models_pytorch/token_pruner.py) | `_rms_norm()` | L27-32 | 无参数 RMSNorm,与 LightVLA 完全一致 |
| | `_LightVLACore` | L35-303 | 核心打分与选择逻辑 + 扩展诊断遥测 |
| | `_LightVLACore.set_keep_tokens()` | L69-72 | 设置固定保留 token 数(eval 态生效) |
| | `_LightVLACore.set_keep_ratio()` | L74-81 | 设置保留比例(0,1](eval 态生效) |
| | `_LightVLACore.compute_importance_score()` | L83-164 | 计算 patch 与 prompt 的相似度分数 + 详细诊断 |
| | `_LightVLACore.select_hard_mask()` | L166-189 | 推理态 argmax 选择,支持 implicit/固定预算模式 |
| | `_LightVLACore.select_soft()` | L191-197 | 训练态 straight-through estimator |
| | `ImageTokenPruner` | L306-479 | OpenPI 适配接口,处理跨相机拼接后的 image tokens |
| | `ImageTokenPruner.prune()` | L338-475 | 主剪枝入口,返回 (pruned_patches, pruned_mask) |
| | `TokenPruner` | L482-573 | 兼容 LightVLA 原始 `[cls, patches, task]` 序列结构 |

### 1.2 PI0Pytorch 集成

| 文件 | 位置 | 行号 | 说明 |
|------|------|------|------|
| [pi0_pytorch.py](../src/openpi/models_pytorch/pi0_pytorch.py) | `__init__` | L87-98 | 读取 `config.token_pruning_enabled` 及相关配置 |
| | `__init__` | L125-132 | 条件创建 `ImageTokenPruner`,设置 keep budget |
| | `__init__` | L138-149 | 处理 `torch.compile` 与动态形状兼容性 |
| | `embed_prefix` | L241-249 | 配置训练/推理态的噪声强度 |
| | `embed_prefix` | L266-279 | **跨相机拼接** + 调用 `token_pruner.prune()` |
| | `embed_prefix` | L325-361 | **LightVLA-style position_ids**:训练用 noisy 索引,eval 用原始索引 |
| | `set_token_pruner_noise_scale()` | L363-365 | 运行时覆盖噪声强度 |
| | `set_token_pruner_keep_tokens()` | L367-371 | 运行时设置固定保留数 |
| | `set_token_pruner_keep_ratio()` | L373-377 | 运行时设置保留比例 |
| | `get_token_pruning_stats()` | L379-456 | **扩展诊断接口**,返回详细遥测数据 |

### 1.3 配置

| 文件 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| [pi0_config.py](../src/openpi/models/pi0_config.py) | `token_pruning_enabled` | `False` | 是否启用剪枝 |
| [pi0_config.py](../src/openpi/models/pi0_config.py) | `token_prune_noise_scale` | `1.0` | 训练态噪声强度(建议从 1.0 线性衰减到 0.0) |
| [pi0_config.py](../src/openpi/models/pi0_config.py) | `token_prune_keep_tokens` | `None` | 推理态固定保留 token 数(None=implicit selection) |
| [pi0_config.py](../src/openpi/models/pi0_config.py) | `token_prune_keep_ratio` | `None` | 推理态保留比例(None=implicit selection) |

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

### 2.1 打分机制（改进版，支持 prompt mask）

```python
# token_pruner.py L65-90
def compute_importance_score(
    self,
    patches: torch.Tensor,
    task_tokens: torch.Tensor,
    *,
    task_token_mask: Optional[torch.Tensor] = None,  # 屏蔽 prompt padding
) -> torch.Tensor:
    patches_n = _rms_norm(patches)      # RMSNorm
    task_n = _rms_norm(task_tokens)     # RMSNorm
    
    # 构建 attention mask 屏蔽 padding tokens
    attn_mask = None
    if task_token_mask is not None:
        attn_mask = task_token_mask[:, None, :].expand(B, N, T)  # [B, N, T]
    
    # 用 task tokens 作为 context，patches 作为 query
    queries = F.scaled_dot_product_attention(patches_n, task_n, task_n, attn_mask=attn_mask)
    queries = _rms_norm(queries)
    
    # 计算每个 query 与所有 patches 的相似度
    score = (queries @ patches_n.transpose(-2, -1)) * self.scale_factor  # [B, N, N]
    return score
```

> **与原作差异**：原作 LightVLA 不使用 prompt mask，padding tokens 会参与注意力计算。
> 本实现添加 `task_token_mask` 屏蔽 padding，解决了因 padding 干扰导致「全部剪枝」的问题。

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

**结果：9/9 passed** ✅
- `TestImageTokenPruner::test_train_keeps_shape_and_is_differentiable`
- `TestImageTokenPruner::test_train_gathers_patch_mask_and_records_indices`
- `TestImageTokenPruner::test_train_masks_invalid_patches_in_score`
- `TestImageTokenPruner::test_eval_prunes_and_returns_mask`
- `TestImageTokenPruner::test_eval_fixed_keep_tokens`
- `TestImageTokenPruner::test_eval_records_kept_indices_and_respects_patch_mask`
- `TestImageTokenPruner::test_eval_ignores_masked_task_tokens`
- `TestTokenPruner::test_eval_shortens_sequence`
- `TestTokenPruner::test_train_keeps_sequence_length`

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

embs, _, _, _ = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
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
    token_prune_noise_scale: float = 1.0  # 训练态噪声（会线性衰减到 0）
    token_prune_keep_ratio: float = None  # 推理时保留比例（可选）
    token_prune_keep_tokens: int = None   # 推理时固定保留数（可选）
    ...

model = PI0Pytorch(config)
```

### 4.2 推理

```python
model.eval()  # 必须！推理态才会真实剪枝
with torch.inference_mode():
    actions = model.sample_actions(device, observation)
```

### 4.3 查看剪枝统计(扩展诊断版)

```python
stats = model.get_token_pruning_stats()
# 返回详细的诊断遥测信息(训练时):
# {
#   # === 基础信息 ===
#   'enabled': True,
#   'noise_scale': 0.8,              # 当前噪声强度
#   'keep_tokens': None,              # 固定保留数(None=implicit selection)
#   'keep_ratio': None,               # 固定保留比例(None=implicit selection)
#
#   # === Token 保留统计 ===
#   'last_kept_per_sample': [72],      # 清晰 argmax 选择的 token 数
#   'last_kept_per_sample_noisy': [68], # 加噪声后选择的 token 数
#   'last_kept_ratio_overall_mean': 0.094,  # 跨相机总体保留比例
#   'last_kept_overall_mean': 72.0,    # 跨相机总保留数(平均)
#   'last_kept_overall_min': 68,       # 跨相机总保留数(最小)
#   'last_kept_overall_max': 76,       # 跨相机总保留数(最大)
#
#   # === Task Token 诊断 ===
#   'last_task_valid_len': [32],       # 有效任务 token 长度
#   'last_task_attn_masked_len': [0],  # 被 mask 的 token 数
#
#   # === Argmax Vote 集中度(崩溃检测) ===
#   'last_argmax_union_det': [24],     # 唯一被选中的 patch 数
#   'last_argmax_union_noisy': [22],   # 加噪声后唯一被选中的数
#   'last_argmax_top1_share_det': [0.15],   # 排名第一的 patch 的投票占比
#   'last_argmax_top1_share_noisy': [0.18],
#
#   # === Embedding 多样性(崩溃检测) ===
#   'last_patches_std_mean': [0.42],        # Patch embeddings 标准差
#   'last_task_std_mean': [0.38],           # Task embeddings 标准差
#   'last_queries_before_norm_std_mean': [0.25],  # Queries RMSNorm 前标准差
#   'last_queries_after_norm_std_mean': [0.35],   # Queries RMSNorm 后标准差
#   'last_queries_before_norm_cosine_sim': [0.82], # Queries RMSNorm 前余弦相似度
#   'last_queries_after_norm_cosine_sim': [0.76],  # Queries RMSNorm 后余弦相似度
#
#   # === Score 质量指标 ===
#   'last_score_abs_max_det': [12.5],       # 分数绝对值最大值
#   'last_score_top1_gap_mean_det': [0.8],  # Top1-Top2 gap 平均值
# }
```

**诊断字段说明:**

| 字段组 | 用途 |
|--------|------|
| `argmax_union_*` | 检测选择崩溃:若 union 接近 1-2,说明所有 query 都选同一个 patch |
| `argmax_top1_share_*` | 检测投票集中度:若接近 1.0,说明大部分 query 选同一个 patch |
| `*_std_mean` | 检测 embedding 崩溃:若接近 0,说明所有 token embedding 几乎相同 |
| `*_cosine_sim` | 检测 query 方向一致性:若接近 1.0,说明所有 query 指向同一方向 |
| `score_top1_gap_*` | 检测区分度:若接近 0,说明 top1/top2 patch 分数相近,选择不确定 |

### 4.4 运行时调整噪声

```python
model.set_token_pruner_noise_scale(0.8)  # 调高探索
model.set_token_pruner_noise_scale(None)  # 恢复配置默认值
```

### 4.5 运行时调整 Keep Budget (推理态)

```python
# 固定保留 token 数
model.set_token_pruner_keep_tokens(64)  # 跨相机总共保留 64 个 patch tokens

# 或使用比例
model.set_token_pruner_keep_ratio(0.25)  # 保留 25% 的 patches

# 禁用固定预算,恢复 implicit selection (LightVLA 默认)
model.set_token_pruner_keep_tokens(None)
model.set_token_pruner_keep_ratio(None)
```

### 4.6 serve_policy.py 运行时覆盖（推理）

```bash
# 保留 75% tokens
uv run python scripts/serve_policy.py \
  --port 8001 \
  --token-prune-keep-ratio 0.75 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/.../39999

# 固定保留 128 tokens
uv run python scripts/serve_policy.py \
  --port 8001 \
  --token-prune-keep-tokens 128 \
  policy:checkpoint \
  --policy.config pi05_pick_place_pytorch_pruning \
  --policy.dir ./checkpoints/.../39999
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
| `src/openpi/models_pytorch/token_pruner.py` | 新增 (573 行) | LightVLA 剪枝核心 + 诊断遥测 |
| `src/openpi/models_pytorch/token_pruner_test.py` | 新增 | 单元测试 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | 修改 (698 行) | 集成跨相机剪枝 + position_ids |
| `scripts/train_pytorch.py` | 修改 (729 行) | 集成 noise schedule + Wandb 遥测 |
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
CUDA_VISIBLE_DEVICES=5 uv run python scripts/train_pytorch.py pi05_pick_place_pytorch_pruning \
  --exp_name prune_run1_fix_padding_test --num_train_steps 40000

CUDA_VISIBLE_DEVICES=6,7 uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi05_pick_place_pytorch_pruning \
  --exp_name prune_run1_diagnose_collapse_1  \
  --num_train_steps 40000 \
  --batch_size 16
```

> [!NOTE]
> 默认参数与 LightVLA 一致：
> - `token_prune_noise_scale=1.0`（会在训练循环中线性衰减到 0）
> - 使用 implicit selection（argmax 并集），不固定保留数量

### 8.3 训练时噪声日程（已内置，匹配 LightVLA）

`train_pytorch.py` 已内置线性衰减噪声逻辑，无需手动添加：

```python
# train_pytorch.py L532-539 (已实现)
model_to_configure = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
if getattr(model_to_configure, "token_pruning_enabled", False) and hasattr(
    model_to_configure, "set_token_pruner_noise_scale"
):
    frac = global_step / max(1, int(config.num_train_steps))
    model_to_configure.set_token_pruner_noise_scale(max(0.0, 1.0 - float(frac)))
```

**效果**：
- `global_step=0` → `noise_scale=1.0`（最大探索）
- `global_step=max_steps` → `noise_scale=0.0`（确定选择）

### 8.4 训练时 Telemetry（日志与 Wandb）

训练日志会输出剪枝统计(每 `log_freq` 步):

```
step=1000 loss=0.1234 lr=5.00e-05 prune/kept_ratio_overall_mean=0.0938 prune/kept_tokens_overall_mean=72.0
```

**Wandb 完整指标列表** (`train_pytorch.py` L571-641):

| 指标名 | 说明 |
|--------|------|
| **Token 保留** | |
| `prune/kept_ratio_overall_mean` | 跨相机总体保留比例 |
| `prune/kept_tokens_overall_mean` | 跨相机总保留数(平均) |
| `prune/kept_tokens_overall_min` | 跨相机总保留数(最小) |
| `prune/kept_tokens_overall_max` | 跨相机总保留数(最大) |
| `prune/kept_tokens_overall_mean_noisy` | 加噪声后的保留数(平均) |
| **Task Token  诊断** | |
| `prune/task_valid_len_mean` | 有效任务 token 平均长度 |
| `prune/task_attn_masked_len_mean` | 被 mask 的 token 平均数 |
| **Argmax Vote 集中度(崩溃检测)** | |
| `prune/argmax_union_mean` | 唯一被选patch数(deterministic) |
| `prune/argmax_union_mean_noisy` | 唯一被选patch数(with noise) |
| `prune/argmax_top1_share_mean` | Top1 patch投票占比(det) |
| `prune/argmax_top1_share_mean_noisy` | Top1 patch投票占比(noisy) |
| **Embedding 多样性(崩溃检测)** | |
| `prune/patches_std_mean` | Patch embeddings 标准差 |
| `prune/task_std_mean` | Task embeddings 标准差 |
| `prune/queries_before_norm_std_mean` | Queries RMSNorm前标准差 |
| `prune/queries_after_norm_std_mean` | Queries RMSNorm后标准差 |
| `prune/queries_before_norm_cosine_mean` | Queries RMSNorm前余弦相似度 |
| `prune/queries_after_norm_cosine_mean` | Queries RMSNorm后余弦相似度 |
| **Score 质量** | |
| `prune/score_abs_max_det_mean` | 分数绝对值最大值 |
| `prune/score_top1_gap_det_mean` | Top1-Top2 gap 平均值 |

### 8.5 验证训练已启用剪枝

在训练日志中查找：

```
INFO Enabled LightVLA-style token pruning (hidden_size=2048)
```

或在代码中检查：

```python
print(f"Token pruning enabled: {model.token_pruning_enabled}")
print(f"Pruner stats: {model.get_token_pruning_stats()}")
```

### 8.6 训练态行为说明

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
  --exp_name pi05_libero_lightvla_run1 \
  --batch_size 24
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
