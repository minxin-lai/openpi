# VLA 架构对比报告

## 1. 概念说明

### LLM Input 的两个层次

| 层次 | 名称 | 说明 |
|------|------|------|
| **Token IDs** | `input_ids` | 离散整数索引，指向词表（如 `[1, 234, 567]`） |
| **Embeddings** | `inputs_embeds` | 连续向量，由 Token ID 转换（如 `[B, seq_len, 4096]`） |

---

## 2. 数据流对比

### OpenVLA

```
input_ids = [BOS] + [prompt] + [真实动作 tokens] + [EOS]
                                ↓ Embedding
            embeddings + vision_patches → Llama → Cross-Entropy Loss
```

### LightVLA（Embedding 替换）

```
input_ids = [BOS] + [prompt] + [占位符 tokens] + [EOS]
                                ↓ Embedding
            input_embeddings[action位置] = noisy_action_features ← 替换！
                                ↓
            TokenPruner(patches, task=prompt+noisy_action) ← 能看到动作
                                ↓
            pruned_multimodal → Llama → 动作预测
```

### OpenPI π0.5（Prefix-Suffix 分离）

```
tokenized_prompt = "Task:..., State: 128 135...;\nAction: "
                                ↓
Prefix: [images] + [lang+state] → PaliGemma → KV cache
                    ↓
        TokenPruner(img, lang) ← ❌ 看不到 action
                                ↓
Suffix: [noisy_actions] → Action Expert（cross-attn to prefix）→ Diffusion
```

---

## 3. 关键差异

| 维度 | OpenVLA | LightVLA | OpenPI π0.5 |
|------|---------|----------|-------------|
| **input_ids 中的动作** | 真实动作 tokens | 占位符 tokens | ❌ 无 |
| **实际 LLM 输入的动作** | token embeddings | noisy action emb（替换） | ❌ 不在 prefix |
| **Pruner 能看到动作？** | N/A | ✅ 能 | ❌ 不能 |

---

## 4. Token Pruning 适配难点

### 根本问题
OpenPI 的 **prefix-suffix 分离**导致 pruning 发生时（prefix 阶段），action 尚未可用（在 suffix）。

### 当前表现
保留 token 数坍塌到 2-3 个（预期 ~78）。

---

## 5. 潜在解决方案

### 方案 A：添加占位符并 Mask

```python
# 在 tokenized_prompt 末尾追加占位符 tokens
# 用 noisy_action_features 替换
# 在 PaliGemma attention 中 mask 掉 action 部分

prefix = [images] + [lang+state] + [noisy_action_emb（masked）]
```

| 优点 | 缺点 |
|------|------|
| 架构一致性 | 可能影响 KV cache |
| 可训练 | attention mask 设计复杂 |

### 方案 B：只给 Pruner 用（推荐）

```python
# embed_prefix() 修改
lang_emb = embed_language_tokens(lang_tokens)
action_emb = action_projector(noisy_actions)  # 新增

# 只给 pruner 用，不送入 PaliGemma
task_for_pruner = concat(lang_emb, action_emb)
pruned_img = pruner.prune(img_emb, task_for_pruner)

# PaliGemma 输入不变
prefix = concat(pruned_img, lang_emb)  # 不含 action_emb
```

| 优点 | 缺点 |
|------|------|
| 改动最小 | pruner 信息与 LLM 不一致 |
| 不影响 KV cache | - |
| 不影响 suffix 处理 | - |

---

## 6. 结论

| 要点 | 说明 |
|------|------|
| **LightVLA 能剪枝** | noisy_action_emb 注入 multimodal sequence |
| **OpenPI 难适配** | action 在 suffix，与 prefix 隔离 |
| **推荐方案** | 方案 B：只给 pruner 拼接 action，不改变 PaliGemma 输入 |
