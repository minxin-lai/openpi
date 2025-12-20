# Pi0.5（`pi05`）模型架构与数据流（Tree 视图）

本文**只依据代码**，按 tree 方式整理 `pi05` 路径下的模块层级、训练/推理前向数据流，以及关键的注意力与条件注入关系。

- 模型实现：`src/openpi/models/pi0.py`（同一个 `Pi0` 类，`config.pi05=True` 走 Pi0.5 分支）
- 配置：`src/openpi/models/pi0_config.py`
- 双-expert Transformer：`src/openpi/models/gemma.py`
- 视觉编码器（SigLIP ViT）：`src/openpi/models/siglip.py`
- tokenizer（Pi0.5 的离散 state prompt 格式）：`src/openpi/models/tokenizer.py`

---

## 1. 模块层级树（Pi0.5 路径）

> Pi0.5 仍然构建同一个 `Pi0` 模型对象，但在 `__init__` 中创建的子层集合不同（见 `src/openpi/models/pi0.py`）。

```text
Pi0 (config.pi05=True)
├─ PaliGemma (nnx.Dict)
│  ├─ img: SigLIP ViT ("So400m/14", pool_type="none", scan=True)
│  │  ├─ stem: Conv patch embedding
│  │  ├─ + 2D positional embedding
│  │  ├─ Transformer Encoder × depth (So400m=27)
│  │  └─ Dense head → 输出每个 patch 的 token（不做 pooling）
│  └─ llm: Gemma.Module(configs=[expert0, expert1], embed_dtype=config.dtype)
│     ├─ expert0: PaliGemma expert（默认 gemma_2b）
│     ├─ expert1: Action expert（默认 gemma_300m）
│     ├─ init(use_adarms=[False, True]) 仅初始化 expert1 的 AdaRMS 分支
│     └─ Block × depth
│        ├─ 对两个 expert 的 token 序列拼接后做同一个 self-attn（共享 head_dim/heads/kv_heads）
│        └─ 逐 expert 的 RMSNorm / MLP（expert1 在 Pi0.5 下可接收 cond 做 AdaRMS + gate）
│     └─ final_norms: RMSNorm × 2（每个 expert 一个）
│        └─ expert1 的 final_norm 同样接收 cond 做 AdaRMS（对输出做最终的时间步调制）
├─ action_in_proj: Linear(action_dim → action_expert.width)
├─ time_mlp_in:  Linear(action_expert.width → action_expert.width)
├─ time_mlp_out: Linear(action_expert.width → action_expert.width)
└─ action_out_proj: Linear(action_expert.width → action_dim)

Pi0.5 路径下“不会创建”的层（因为 __init__ 不走 Pi0 分支）：
└─ state_proj / action_time_mlp_in / action_time_mlp_out
```

---

## 2. 训练前向（`compute_loss`）数据流树（Pi0.5 路径）

入口：`src/openpi/models/pi0.py:compute_loss`。核心特征：**suffix 只有 action tokens**，时间步通过 `adarms_cond` 注入 Transformer。

```text
compute_loss(rng, observation, actions)
├─ observation = preprocess_observation(rng, train=...)
├─ 采样噪声与时间（flow-matching）
│  ├─ noise ~ N(0, 1)
│  ├─ t ~ Beta(1.5, 1) ∈ (0, 1)
│  ├─ x_t = t*noise + (1-t)*actions
│  └─ u_t = noise - actions
├─ prefix_tokens = embed_prefix(observation)
│  ├─ 图像：SigLIP(img) → image_tokens（每个 view 一段 token，token dim = paligemma.width）
│  └─ 文本：Gemma.embed(tokenized_prompt) → text_tokens（token dim = paligemma.width）
│     └─ （若上游 tokenizer 以 Pi0.5 格式构造 prompt，则 state 会以离散 token 形式出现在这里）
├─ suffix_tokens = embed_suffix(observation, x_t, t)   (Pi0.5 分支)
│  ├─ action_tokens = action_in_proj(x_t)             (token dim = action_expert.width)
│  ├─ time_emb = posemb_sincos(t, dim=action_expert.width)
│  ├─ time_emb = swish(time_mlp_out(swish(time_mlp_in(time_emb))))
│  ├─ action_expert_tokens = action_tokens            (不与 time_emb 拼接)
│  └─ adarms_cond = time_emb                          (供 action expert 的 AdaRMSNorm 使用)
├─ mask / positions
│  ├─ prefix_ar_mask：全 False（prefix 内部双向互看）
│  ├─ suffix_ar_mask： [True] + [False]*(H-1)（action 段内部双向互看；同时屏蔽 prefix→suffix）
│  ├─ attn_mask = make_attn_mask(concat(prefix_mask, suffix_mask), concat(prefix_ar_mask, suffix_ar_mask))
│  └─ positions = cumsum(input_mask) - 1
├─ (prefix_out, suffix_out) = PaliGemma.llm([prefix_tokens, suffix_tokens], mask, positions, adarms_cond=[None, adarms_cond])
└─ v_t = action_out_proj(suffix_out[:, -action_horizon:])
   loss = mean((v_t - u_t)^2, dim=action_dim)
```

---

## 3. 推理采样（`sample_actions`）数据流树（Pi0.5 路径）

入口：`src/openpi/models/pi0.py:sample_actions`。核心特征：**prefix 先写 KV cache（只跑 expert0）**，然后 **suffix 迭代读取 KV cache（只跑 expert1，并带 `adarms_cond`）**。

```text
sample_actions(rng, observation, num_steps, noise=None)
├─ observation = preprocess_observation(train=False)
├─ x_1 ~ N(0, 1)（若未提供 noise）
├─ 阶段一：prefix 写 KV cache
│  ├─ prefix_tokens = embed_prefix(observation)
│  ├─ prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
│  ├─ positions = cumsum(prefix_mask) - 1
│  └─ kv_cache = PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
└─ 阶段二：while t 从 1 走到 0（dt = -1/num_steps）
   ├─ suffix_tokens, adarms_cond = embed_suffix(observation, x_t, t)  (Pi0.5 分支)
   ├─ suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)  (suffix 内部)
   ├─ prefix_attn_mask = repeat(prefix_mask, "b p -> b s p")          (suffix → prefix)
   ├─ full_attn_mask = concat(prefix_attn_mask, suffix_attn_mask, axis=-1)
   ├─ positions = sum(prefix_mask) + cumsum(suffix_mask) - 1
   ├─ suffix_out = PaliGemma.llm(
   │    [None, suffix_tokens],
   │    kv_cache=kv_cache,
   │    mask=full_attn_mask,
   │    positions=positions,
   │    adarms_cond=[None, adarms_cond],
   │  )
   ├─ v_t = action_out_proj(suffix_out[:, -action_horizon:])
   └─ x_{t+dt} = x_t + dt * v_t
return x_0
```

---

## 4. 关键关系（注意力 + 条件注入）

### 4.1 Prefix / Suffix 的可见性（由 `make_attn_mask` 的 `ar_mask` 定义）

Pi0.5 路径下：

- prefix 的 `ar_mask` 全为 `False`（图像 tokens + 文本 tokens 全部同组，双向互看）
- suffix 的 `ar_mask` 为 `[True] + [False]*(H-1)`（action 段内部同组，双向互看；且 prefix→suffix 被屏蔽）

因此在一次性前向（训练）或组合 mask（推理）时，得到以下可见性：

```text
prefix tokens: 只能 attend prefix tokens（看不到 suffix）
suffix tokens: 可以 attend prefix tokens + suffix tokens（suffix 内部双向）
```

### 4.2 时间步如何进入 Transformer（AdaRMSNorm）

Pi0.5 路径下，时间步不会与 action tokens 在输入端拼接；而是：

- `time_emb = posemb_sincos(t, dim=action_expert.width)`
- `adarms_cond = swish(time_mlp_out(swish(time_mlp_in(time_emb))))`
- 在调用 `PaliGemma.llm(...)` 时传入 `adarms_cond=[None, adarms_cond]`

在 `src/openpi/models/gemma.py` 的每个 Transformer block 内：

- expert0（prefix stream）对应的 `cond=None` → 普通 RMSNorm
- expert1（action stream）对应的 `cond=adarms_cond` → AdaRMSNorm（Dense 生成 scale/shift/gate），gate 参与残差更新

---

## 5. Pi0.5 的 tokenizer 输入形态（代码侧描述）

模型本身只消费 `Observation.tokenized_prompt`；Pi0.5 的“state 进 prompt”发生在 tokenizer 侧：

- `src/openpi/models/tokenizer.py:PaligemmaTokenizer.tokenize(prompt, state=...)` 会将 `state` 离散化并拼进文本：  
  `full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "`
- `src/openpi/models/pi0_config.py` 默认 `max_token_len = 200 if pi05 else 48`，并设置 `discrete_state_input = pi05`

---

## 6. Pi0.5 vs Pi0：完整差异对比

> 参考：`docs/pi0_model_dataflow_tree.md` 和模型代码 `src/openpi/models/pi0.py`

### 6.1 模型层差异（`__init__`）

| 层名称 | Pi0 | Pi0.5 | 代码位置 |
|--------|-----|-------|----------|
| `state_proj` | ✅ Linear(action_dim → 1024) | ❌ **不存在** | pi0.py:97 |
| `action_time_mlp_in` | ✅ Linear(2048 → 1024) | ❌ **不存在** | pi0.py:98 |
| `action_time_mlp_out` | ✅ Linear(1024 → 1024) | ❌ **不存在** | pi0.py:99 |
| `time_mlp_in` | ❌ 不存在 | ✅ Linear(1024 → 1024) | pi0.py:94 |
| `time_mlp_out` | ❌ 不存在 | ✅ Linear(1024 → 1024) | pi0.py:95 |
| LLM `use_adarms` 参数 | `[False, False]` | `[False, True]` | pi0.py:80 |

### 6.2 状态（State）处理方式

```diff
- Pi0 (pi0.py:152-157):
-   state → state_proj(state) → state_token [b, 1, 1024]
-   state_token 放入 suffix（在 action_tokens 之前）
-   进入 ActionExpert 作为连续向量

+ Pi0.5 (tokenizer.py:24-28):
+   state → np.digitize(state, 256 bins) → "192 85 64 ..."
+   拼接到 prompt: "Task: {task}, State: {state_str};\nAction: "
+   进入 PaliGemma (prefix) 作为离散文本 tokens
+   suffix 中不再有 state token
```

### 6.3 时间（Timestep）注入机制

```diff
- Pi0 (pi0.py:170-177):
-   time_emb = posemb_sincos(t)
-   time_tokens = repeat(time_emb, "b emb -> b s emb", s=action_horizon)
-   action_time_tokens = concat([action_tokens, time_tokens], axis=-1)  # 拼接
-   action_time_tokens = action_time_mlp_out(swish(action_time_mlp_in(...)))
-   adarms_cond = None  # 不使用 AdaRMSNorm

+ Pi0.5 (pi0.py:162-169):
+   time_emb = posemb_sincos(t)
+   time_emb = time_mlp_out(swish(time_mlp_in(time_emb)))
+   action_expert_tokens = action_tokens  # 不拼接时间
+   adarms_cond = time_emb  # 通过 AdaRMSNorm 注入
```

### 6.4 RMSNorm 行为差异（gemma.py:113-131）

| 场景 | `cond` 值 | RMSNorm 行为 | 残差连接 |
|------|-----------|--------------|----------|
| **Pi0** (expert1) | `None` | 标准: `x * (1+scale)` | `x + y` |
| **Pi0.5** (expert1) | `time_emb` | 自适应: `x * (1+scale) + shift` | `x + y * gate` |

> Pi0.5 的 time_emb 在每个 Transformer Block 中**注入 4 次**：pre_attention_norm (scale/shift + gate)、pre_ffw_norm (scale/shift + gate)

### 6.5 Suffix 内容差异

```text
Pi0 suffix:
├── state_token [b, 1, 1024]       ← 连续状态 token
└── action_tokens [b, 50, 1024]    ← 融合了 time 信息

Pi0.5 suffix:
└── action_tokens [b, 50, 1024]    ← 无 time 拼接，无 state token
```

### 6.6 配置参数差异

| 参数 | Pi0 | Pi0.5 | 说明 |
|------|-----|-------|------|
| `pi05` | `False` | `True` | 主开关 |
| `max_token_len` | 48 | **200** | Pi0.5 需要更长的 prompt 容纳离散 state |
| `discrete_state_input` | `False` | `True` | 自动跟随 pi05 |

### 6.7 注意力掩码（ar_mask）差异

```text
Pi0:
├── prefix_ar_mask: 全 False（双向）
├── suffix: [True] (state) + [True] + [False]*49 (actions)
└── state token 开启独立的 attention 组，action tokens 可看 state 但 state 不可看 action

Pi0.5:
├── prefix_ar_mask: 全 False（双向，state 作为文本 token 已在 prefix 中）
├── suffix: [True] + [False]*49 (仅 action tokens)
└── 没有 state token 在 suffix 中
```

---

## 设计理念对比

| 维度 | Pi0 | Pi0.5 |
|------|-----|-------|
| **状态表示** | 连续信号，投影到 action expert 维度 | 离散语言 tokens，利用 LLM 的语言理解能力 |
| **时间注入** | 输入端拼接 + MLP 融合 | AdaRMSNorm 条件调制（类似扩散模型的 conditioning） |
| **模型复杂度** | 稍简单（无 AdaRMS Dense） | 每层额外 Dense(1024→3072) 用于生成 scale/shift/gate |
| **prompt 长度** | 短（48 tokens） | 长（200 tokens，需容纳离散 state 字符串） |