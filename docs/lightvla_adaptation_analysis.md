# LightVLA 适配问题分析报告

## 📋 概述

本报告分析了 LightVLA Token Pruning 机制在 OpenPI 框架中的适配问题（本文 OpenPI 以 **pi05** 设定为准），**核心发现是架构性限制导致 OpenPI 的 pruner 无法访问原版 LightVLA 可用的动作信息**。

> 说明：LIBERO 默认包含 **primary + wrist** 两路图像输入（`num_images_in_input=2`），因此视觉 patch token 数为 `256 * num_images_in_input`。

---

## 🔬 原版 LightVLA 完整数据流（9 步推导）

### Step 1: 数据集构造 input_ids
**位置**: [datasets.py L36-91](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/vla/datasets/datasets.py#L36-91)

```python
lang = "pick up the red cube"
actions = rlds_batch["action"]                     # [T, ACTION_DIM]
current_action = actions[0]
future_actions = actions[1:]
current_action_string = action_tokenizer(current_action)
future_actions_string = ''.join(action_tokenizer(future_actions))
action_chunk_string = current_action_string + future_actions_string
# action token 数量 = ACTION_DIM * NUM_ACTIONS_CHUNK（随平台变化）

conversation = [
    {"from": "human", "value": f"What action should the robot take to {lang}?"},
    {"from": "gpt", "value": action_chunk_string},
]
input_ids = tokenizer(prompt)  # [BOS, prompt_tokens..., action_tokens..., EOS]
```

**输出**: `input_ids = [BOS] + [prompt_tokens] + [ACTION_DIM*NUM_ACTIONS_CHUNK] + [EOS]`

**示例（LIBERO）**：
- `ACTION_DIM=7`、`NUM_ACTIONS_CHUNK=8` → 动作 token 数 `56`
- `seq_len = 1(BOS) + prompt_len + 56 + 1(EOS)`

---

### Step 2: 获取 input_embeddings
**位置**: [modeling_prismatic.py L897](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py#L897)

```python
input_embeddings = self.get_input_embeddings()(input_ids)  # [B, seq_len, hidden]
```

---

### Step 3: 替换动作 embeddings
**位置**: [modeling_prismatic.py L922-943](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py#L922-943)

```python
if noisy_actions is not None:
    # noisy_actions: [B, chunk_len, action_dim] -> [B, chunk_len*action_dim, 1]
    noisy_action_features = noisy_action_projector(noisy_actions.reshape(B, -1).unsqueeze(-1))
    # noisy_action_features: [B, ACTION_DIM*NUM_ACTIONS_CHUNK, 4096]
    input_embeddings = self._replace_input_embeddings(...)  # 替换动作部分
else:
    input_embeddings = input_embeddings * ~all_actions_mask  # 设为 0
```

---

### Step 4: 视觉特征提取
**位置**: [modeling_prismatic.py L760-769](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py#L760-769)

```python
patch_features = self.vision_backbone(pixel_values)           # [B, 256*num_images_in_input, vision_dim]
projected_patch_embeddings = self.projector(patch_features)   # [B, 256*num_images_in_input, 4096]

# 示例（LIBERO）：num_images_in_input=2（primary + wrist）→ num_patches=512
```

---

### Step 5: 构建 multimodal_embeddings
**位置**: [modeling_prismatic.py L796-798](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py#L796-798)

```python
multimodal_embeddings = torch.cat([
    input_embeddings[:, :1, :],      # BOS: [B, 1, 4096]
    projected_patch_embeddings,       # patches: [B, 256*num_images_in_input, 4096]
    input_embeddings[:, 1:, :]        # 语言+动作: [B, seq_len-1, 4096]
], dim=1)
# 结构: [BOS] + [patches × (256*num_images_in_input)] + [语言 + 动作 + EOS] = 1 + num_patches + (seq_len-1)

# 示例（LIBERO）：总长度 = num_patches(512) + seq_len
```

---

### Step 6: 送入 PrunedLlamaModel
**位置**: [modeling_prismatic.py L954-965](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py#L954-965)

```python
language_model_output = self.language_model(inputs_embeds=multimodal_embeddings)
```

---

### Step 7: TokenPruner 剪枝 (LLM 第一层前)
**位置**: [modeling_prismatic.py L198](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py#L198)

```python
# 在 PrunedLlamaModel.forward() 内，第一层前
hidden_states, position_ids, attention_mask = self.pruner(
    hidden_states,  # = multimodal_embeddings [B, 1 + num_patches + (seq_len-1), D]
    position_ids, attention_mask
)
```

---

### Step 8: TokenPruner.forward 内部切分
**位置**: [modeling_prismatic.py L100-136](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py#L100-136)

```python
def forward(self, tokens, position_ids, attention_mask):
    cls_token, patches, task = torch.split(
        tokens, [1, self.num_patches, seq_len - self.num_patches - 1], dim=1
    )
    # cls_token: [B, 1, 4096]     → BOS
    # patches:   [B, 256*num_images_in_input, 4096]   → 视觉 patches
    # task:      [B, seq_len-1, 4096]                 → 语言指令 + 动作 embeddings ⬅️ 关键！
    
    score = self.get_score(patches, task)  # patches attend to task
    # 剪枝选择...
    tokens = torch.cat([cls_token, pruned_patches, task], dim=1)
    return tokens, position_ids, attention_mask
```

---

### Step 9: get_score Cross-Attention
**位置**: [modeling_prismatic.py L70-78](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py#L70-78)

```python
def get_score(self, patches, prompts):
    # patches: [B, 256*num_images_in_input, D], prompts: [B, seq_len-1, D]
    patches = rms_norm(patches)
    prompts = rms_norm(prompts)
    queries = F.scaled_dot_product_attention(patches, prompts, prompts)
    queries = rms_norm(queries)
    score = queries @ patches.transpose(-2, -1) * self.scale_factor
    return score  # [B, num_patches, num_patches]
```

---

### 📊 数据流总览

```
input_ids: [BOS] + [prompt_tokens] + [ACTION_DIM*NUM_ACTIONS_CHUNK] + [EOS]
                          ↓ embedding + 替换
input_embeddings: [BOS_emb, prompt_emb, noisy_action_emb, EOS_emb]
                          ↓ 拼接视觉
multimodal: [BOS] + [patches × (256*num_images_in_input)] + [语言+动作+(EOS)]
                          ↓ TokenPruner.split
            cls[1] + patches[num_patches] + task[seq_len-1]  ← task 包含动作！
                          ↓ get_score(patches, task)
            剪枝后: [BOS] + [kept] + [语言+动作+(EOS)]
```

---

## 🔄 OpenPI (pi0.5) 的数据流

### pi0.5 的 tokenized_prompt 结构
**位置**: [tokenizer.py L22-33](file:///workspace/laiminxin/vla-opt/third_party/openpi/src/openpi/models/tokenizer.py#L22-33)

```python
# Pi0.5 格式：state 被离散化后放入 prompt
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
state_str = " ".join(map(str, discretized_state))
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
tokens = tokenizer.encode(full_prompt, add_bos=True)
```

**实际 tokenized_prompt 结构**（pi05）:
```
"Task: pick up the red cube, State: 128 135 142 ...;\nAction: "
       ↑                            ↑
    语言指令                    离散化 state（长度=state维度，pi05里为32；token 数 >= 32，且取决于 tokenizer 拆分）
```

**示例（pi05 state token 规模）**：
- state 向量维度为 32 → 至少 32 个数字片段
- SentencePiece 会将数字拆分成 1+ 个 token，因此 **state 段 token 数通常 ≥ 32**
- 总 prefix token 数 ≈ `Task:` 字段 + 32 个数字 token + `Action:` 字段（实际以 tokenizer 为准）

### 剪枝时的数据流
**位置**: [pi0_pytorch.py L234-279](file:///workspace/laiminxin/vla-opt/third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py#L234-279)

```python
# embed_prefix()
lang_emb = embed_language_tokens(lang_tokens)  # 包含 Task + State
all_img_emb = embed_image(images)              # 视觉 patches

# 剪枝时
all_img_emb, kept_mask = self.token_pruner.prune(
    all_img_emb,
    lang_emb,  # lang_emb 包含 state，但不含未来动作
    task_token_mask=lang_masks,
    patch_token_mask=all_img_valid,
)
```

### ⚠️ 与原版 LightVLA 的关键差异

| 维度 | 原版 LightVLA | OpenPI pi0.5 |
|------|--------------|--------------|
| **task tokens** | 语言 + **动作 tokens**（`ACTION_DIM * NUM_ACTIONS_CHUNK`；训练时可被 noisy_actions 替换） | 语言 + **当前 state**（pi05：离散化 state 向量） |
| **包含未来轨迹** | ✅ 训练时若提供 noisy_actions | ❌ 无 |
| **包含 state** | ❌ 无 | ✅ 离散化 state |
| **剪枝位置** | LLM 第一层前 | embed_prefix() |

**核心问题仍然存在**：OpenPI 的 pruner 无法"看到"未来要执行的动作序列，只能基于当前状态和语言指令做剪枝。

---

## 📊 开启剪枝下的逐步对比

| Step | 原版 LightVLA | OpenPI pi0.5 |
|------|--------------|--------------|
| **1. 输入构造** | `input_ids = [prompt] + [ACTION_DIM*NUM_ACTIONS_CHUNK]` | `tokenized_prompt = "Task:..., State: 192 102...;"` |
| **2. Embedding** | `input_emb = LLM.embed(input_ids)` | `lang_emb = PaliGemma.embed(tokenized_prompt)` |
| **3. 动作处理** | 训练时若传 `noisy_actions` → 替换；否则置 0 | ❌ **无动作在 prefix 中** |
| **4. 视觉处理** | `patches = VisionBackbone(img)` [256×num_images×D] | `img_emb = SigLIP(img)` [256×num_images×D] |
| **5. 拼接** | `[BOS] + [patches] + [语言+动作+EOS]` | `[img_patches] + [lang+state]` |
| **6. 剪枝输入** | Q=`patches`, K/V=`task(语言+动作)` | Q=`img_emb`, K/V=`lang_emb(含state)` |
| **7. 剪枝计算** | `score = get_score(patches, task)` | `score = compute_importance_score(patches, task)` |
| **8. 剪枝依据** | 语言意图 + （训练时可见 noisy 动作） | 语言意图 + 当前 state |

---

## ⚠️ 关于 "noisy 动作" 的澄清

**训练时（若传入 noisy_actions）**: LightVLA 的动作 embeddings 被替换为 **noisy action features**：
```python
# modeling_prismatic.py L935-938
if noisy_actions is not None:
    noisy_action_features = noisy_action_projector(noisy_actions)
    input_embeddings = self._replace_input_embeddings(..., noisy_action_features)
```

**推理时或未传 noisy_actions**: 动作 embeddings 被设为 **zeros**：
```python
# modeling_prismatic.py L942-943
else:
    input_embeddings = input_embeddings * ~all_actions_mask  # 设为 0
```

### 这意味着什么？

| 阶段 | task tokens 内容 | 剪枝能获得的信息 |
|------|-----------------|-----------------|
| **训练（有 noisy_actions）** | 语言 + **noisy 动作** | 可见动作 token（含噪） |
| **训练（无 noisy_actions）/ 推理** | 语言 + **zeros** | 只有语言意图 |

**仅在 noisy_actions 存在时**，LightVLA 的 pruner 才能获得动作 token 的近似信息。

**OpenPI 完全没有这个信息** - 无论训练还是推理。

---

## 🛠️ 改进方案

| 方案 | 改动位置 | 难度 |
|------|----------|------|
| **1. 添加 Proprio/连续状态** | `embed_prefix()` 拼接 state embedding 到 task tokens | ⭐ |
| **2. Action History** | 数据流 + pi0_pytorch.py | ⭐⭐ |
| **3. 调整超参数** | 配置文件 | ⭐ |
| **4. CogVLA 聚合** | 架构改动 | ⭐⭐⭐ |

### 方案 1 代码示例
```python
if self.token_pruning_enabled and state is not None:
    state_emb = self.state_proj(state)[:, None, :]
    task_tokens = torch.cat([lang_emb, state_emb], dim=1)
else:
    task_tokens = lang_emb

all_img_emb, kept_mask = self.token_pruner.prune(all_img_emb, task_tokens, ...)
```

---

## 📁 相关代码文件

| 模块 | 文件 |
|------|------|
| **LightVLA TokenPruner** | [modeling_prismatic.py](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py) |
| **LightVLA 数据集** | [datasets.py](file:///workspace/laiminxin/vla-opt/third_party/LightVLA/prismatic/vla/datasets/datasets.py) |
| **OpenPI 模型** | [pi0_pytorch.py](file:///workspace/laiminxin/vla-opt/third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py) |
| **OpenPI Pruner** | [token_pruner.py](file:///workspace/laiminxin/vla-opt/third_party/openpi/src/openpi/models_pytorch/token_pruner.py) |
