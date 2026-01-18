# LightVLA Pruning 对齐与问题定位（OpenPI π0.5）

本文档用于把当前 OpenPI（`third_party/openpi`）里的 LightVLA pruning 适配实现，与 `third_party/LightVLA` 原实现做对照，并解释当前训练日志中“保留 token 数极低”的现象，给出清晰的 debug 思路。

---

## 1. 现有实现（OpenPI / π0.5 PyTorch）

### 1.1 数据侧：`tokenized_prompt` 的来源与 padding

- 训练数据 transform 使用 `TokenizePrompt(PaligemmaTokenizer(max_token_len))` 产生：
  - `tokenized_prompt: [B, max_token_len]`
  - `tokenized_prompt_mask: [B, max_token_len]`（True=有效 token，False=padding）
- π0.5 的 prompt 可能包含离散 state（`discrete_state_input=True` 时会拼 `Task: ... State: ...; Action:`），否则就是普通文本 + `"\n"`。
- 重要：OpenPI 的 `PaligemmaTokenizer.tokenize()` 会把 padding token id 直接填 `0`，并依赖 `tokenized_prompt_mask` 来区分有效/无效。

### 1.2 模型侧：prefix/suffix 的组成（π0.5 diffusion）

- prefix（LLM 的视觉+语言输入）：
  - image patches：由 `embed_image()` 产生（每张 224×224 对应 256 patch tokens）
  - language tokens：由 `embed_language_tokens(tokenized_prompt)` 产生
  - 两者拼接后作为 prefix 输入 PaliGemma transformer
- suffix（diffusion action expert 输入）：
  - `noisy_actions` 经 `action_in_proj` 得到连续 embedding
  - timestep 经正弦位置编码 + MLP（π0.5 adaRMS cond）进入 expert
- 结论：`tokenized_prompt` 不包含图像 token，也不包含 diffusion action token；它只提供“指令/（可选）离散 state”上下文。

### 1.3 pruning 的集成点与策略（当前版本）

#### 1) 跨相机合并剪枝（LightVLA-style）

- OpenPI 现在采用“跨相机拼接成一个 patch 池，然后剪一次”的路径：
  - 将多路相机的 `img_emb:[B,256,D]` 拼成 `all_img_emb:[B, N_total, D]`（3 路相机时 N_total=768）
  - 同时把每路 `img_mask:[B]` 扩成 per-patch mask 并拼成 `patch_token_mask:[B, N_total]`
  - 对 `all_img_emb` 只调用一次 `ImageTokenPruner.prune(...)`
- 这与 LightVLA “`num_patches *= num_images_in_input`” 的行为一致：把多图 patch 当成同一池子投票/选并集。

#### 2) padding mask 已接入 pruning 打分

- 目前 pruning 的 score 计算会使用 `tokenized_prompt_mask` 作为 `scaled_dot_product_attention(..., attn_mask=...)` 的 mask 来屏蔽 padding。
- 这是为了解决“padding 被当作有效 task token 导致 argmax 并集塌缩到 1”的问题（之前已出现过）。

#### 3) 训练态与推理态的差异（非常关键）

pruner 在 OpenPI 的行为与 LightVLA 一样分两种模式：

- `train()`（训练态）：
  - 不会物理缩短序列长度（输出仍是 `[B, N_total, D]`）
  - 使用 straight-through：`hard(one_hot(argmax)) + soft(softmax)` 形成权重矩阵，输出 `weights @ patches`
  - “加噪声”的地方发生在训练态 argmax 前（用于探索更多 selection）
- `eval()`（推理态）：
  - 会真实删 token：用“每个 query 行的 argmax 的并集”构造 mask，然后 gather patches，缩短 prefix
  - 推理态默认禁用 noise

#### 4) noise schedule（线性衰减）已实现

- 训练循环中，每步会把 pruning noise 设置为线性衰减：`noise_scale = 1 - step/num_train_steps`（与 LightVLA 一致）。
- 训练日志里会同时记录：
  - `prune/kept_tokens_overall_mean`：无噪声（eval-style）并集保留数（更接近推理时会保留多少）
  - `prune/kept_tokens_overall_mean_noisy`：带噪声（训练 forward selection）并集保留数（更接近训练时实际“探索到”的选择规模）

#### 5) position_ids（已按 LightVLA 对齐）

LightVLA 的 HF 实现会在剪枝后把 **被保留 patch 的原始 position_ids** gather 出来继续用（会出现跳号），而不是把剩余 tokens 重新编号成连续的 0..K-1。

OpenPI 当前已对齐该行为：
- prefix 的 patch position_ids：来自“被保留 patch 的原始 index”（在合并 patch 池中的 index）
- prefix 的语言 position_ids：固定从 `N_total` 开始递增（不因剪枝而左移）
- suffix position_ids：固定从“未剪枝的 prefix 总长度（N_total + max_token_len）”开始递增（因此剪枝后会出现跳号）

---

## 2. 原实现（`third_party/LightVLA`，以 `docs/lightvla_optimization_report.md` 为主线）

### 2.1 TokenPruner 的核心逻辑（HF / Prismatic）

- 多模态 token 序列结构（简化）：
  - `[BOS/CLS] + [vision_patches] + [task tokens]`
- score 计算：
  1) RMSNorm patches & task
  2) `queries = attn(patches, task, task)`
  3) `score = queries @ patches^T / sqrt(D)`，形状 `[B, N, N]`
- 推理态选择：对 `score.argmax(-1)` 的结果做并集 mask，真实删 patch tokens，缩短序列。
- 训练态选择：对 score 加噪声，构造 ST 权重，输出 `weights @ patches`（长度不变）；并更新 `position_ids/attention_mask` 与选中的 indices 对齐（这点对 “selection 学习 + rotary 对齐” 很关键）。

### 2.2 noise factor schedule 的目的（论文/报告结论）

论文描述的现象（与你贴的表一致）：

- 完整版（噪声 + 线性衰减）通常能在性能与 token 数之间达到较好平衡；
- 无噪声（w/o noise）会倾向更“保守/集中”的选择，保留 tokens 更少，易丢语义；
- 常量噪声（w/o schedule）会导致长期探索过强，难以收敛到稳定的 pruning 策略，从而保留 tokens 更多。

这三者的关键差异是：训练阶段的“探索-收敛”是否被良好控制。

---

## 3. 现在的问题（基于最新训练日志）

你当前的训练日志（节选）：

```
step=20600 ... 
prune/kept_tokens_overall_mean=2.2963
prune/kept_tokens_overall_mean_noisy=9.2200
prune/kept_ratio_overall_mean=0.0030
```

对 3 张 224×224 图像：
- 总 patch tokens = `3 * 256 = 768`
- `kept_tokens_overall_mean≈2.3` 对应保留比例 ≈ `2.3/768 ≈ 0.3%`
- 这显著低于论文里 “#Tokens ≈ 78”（论文常见配置是 2 图时 N=512；78/512≈15%，即使换成 3 图也不应该到 0.3%）

### 3.1 这行日志“到底代表什么”

- `kept_tokens_overall_mean`：无噪声、eval-style（推理态会更接近它，因为推理禁用 noise）
  - 如果它长期维持在 1～3，意味着推理时会把视觉信息剪到几乎不可用（模型近似“失明”）。
- `kept_tokens_overall_mean_noisy`：带噪声、训练 forward selection 的统计
  - 它更大是正常的，因为当前 step=20600/40000 时 noise_scale≈0.485，仍处于探索阶段。

### 3.2 为什么“无噪声的 kept”仍然这么低值得警惕

对照论文的结论：完整 LightVLA（有 schedule）并不会把 token 压到 1～3；无噪声版本也只是从 78 → 72 这种小幅变化。

因此你现在的 `~2` 更像是“策略坍塌”，而不是论文里正常的“更少一点”。

### 3.3 潜在根因（需要进一步验证）

以下是更可能导致 “eval-style kept 极低” 的差异点/风险点：

1) **训练态 position_ids/attention_mask 是否与 noisy indices 对齐（原版会对齐）**
   - LightVLA 训练态会用 noisy argmax indices 去重排 `position_ids/attention_mask`（HF 里直接 gather）
   - 如果训练态不对齐，可能出现“训练在一种 position 语义下优化，但推理在另一种语义下硬剪枝”，导致剪枝策略学不稳或塌缩

2) **π0.5 diffusion 目标与 OpenVLA action-token 目标不同**
   - LightVLA 原论文/实现是围绕 action token / action head 的 supervised 结构来学习 pruning
   - π0.5 是 diffusion 连续动作监督，剪枝的梯度信号来源与原版不同，可能更弱/更间接，导致策略不容易形成

3) **prompt 多样性不足（任务文本几乎固定）**
   - 如果训练集 prompt 基本不变，pruner 的 task context 几乎恒定，则容易形成“选择极少数 patch 永远够用”的局部最优

4) **统计口径与论文口径不一致**
   - 论文报告的 token 数通常对应推理态真实删 token 后的长度（且可能包含/不包含某些额外 token，如 proprio/timestep 等）
   - 但即使口径略有差异，`2/768` 仍显著异常，需要进一步定位

---

## 4. Debug 的思路（建议按优先级执行）

### 4.1 先确认推理态是否真的“失明”

1) 取当前训练 checkpoint，跑一次 `eval()` 推理（或仅跑 prefix 构建）：
   - 观察 console 里 `[Token Pruning] Visual tokens: before=768, after=...`
2) 对比：
   - 如果推理态 after 也接近 1～3：问题就是“硬剪枝策略塌缩”，必须修。
   - 如果推理态 after 明显更高：说明训练日志口径/实现存在偏差，需要修统计或修状态切换。

### 4.2 分析 argmax 分布是否“极端集中”

对同一 batch：
- 取 `indices = score.argmax(-1)`，统计：
  - unique 数（=并集大小）
  - top-1 patch 的票数占比（是否接近 100%）
- 同时比较 deterministic vs noisy（加噪声前后）：
  - 如果 deterministic 极端集中、noisy 分散，说明“score 本身还没学出可分性”，噪声只是临时扩散
  - 如果 deterministic/noisy 都集中，说明更严重的表征/目标问题

### 4.3 对比论文的三种变体（在本工程中复现实验）

为了与论文结论一致，建议加三组对照（至少在一小段训练上）：
- w/o noise：训练时固定 noise=0
- w/ constant noise：训练时固定 noise=const（例如 0.5）
- full schedule：线性衰减

观测指标：
- `kept_tokens_overall_mean` 在训练后期是否能稳定到一个合理区间（比如几十）
- 与任务成功率/离线指标是否一致变化

### 4.4 检查 task context 是否“有效且多样”

- 检查 `tokenized_prompt_mask` 的有效 token 数分布：
  - 是否绝大多数样本有效 token 很少（例如 5～10）？
  - 是否 prompt 基本恒定（PromptFromTask 只有一个 task）？
- 如果确实恒定/过短：
  - 先用显式预算（keep_ratio/keep_tokens）验证“只要视觉信息足够，任务就能好”
  - 再考虑引入更强的 task context（例如拼入 state、或更丰富 prompt）

### 4.5 检查训练态的“position_ids 对齐”是否真正与 LightVLA 一致

要严格对齐 LightVLA，需要确认训练态也具备以下行为：
- noisy argmax 的 indices 用于更新 patch token 的 position_ids/attention_mask（即使输出长度不变）
- 推理态 keep 的 indices 用于 gather patch token 的 position_ids（已对齐）

如果训练态缺失该对齐，优先补齐后再观察 token 数是否回到合理范围。

### 4.6 追加 telemetry：快速判定“坍塌发生在哪里”

为了把“kept 很低”从现象定位到机制，已在 OpenPI 的 PyTorch 路径里增加 pruning telemetry，并写入训练日志（统一以 `prune/*` 前缀输出）。

#### 4.6.1 新增指标（训练日志中可直接看到）

- `prune/task_valid_len_mean`：batch 内 `tokenized_prompt_mask` 的有效 token 数均值（排除 padding）
- `prune/argmax_union_mean`：deterministic `score.argmax(-1)` 的并集大小均值（= unique 数；近似等同“eval-style 会保留多少”）
- `prune/argmax_top1_share_mean`：deterministic 投票集中度（最热门 patch 获得的票数 / N_queries），越接近 1 越“塌缩”
- `prune/argmax_union_mean_noisy` / `prune/argmax_top1_share_mean_noisy`：同上，但对 “训练时加噪声后的 argmax” 统计
- `prune/patches_std_mean`：`patches_n` 在 token 维度的 std（再对 channel 求均值），衡量 patch embedding 多样性
- `prune/task_std_mean`：`task_n` 在有效 task tokens 维度的 std（再对 channel 求均值），衡量 task embedding 多样性
- `prune/queries_std_mean`：`queries = SDPA(patches_n, task_n, task_n)` 在 token 维度的 std（再对 channel 求均值），衡量 cross-attn 输出的“区分度”
- `prune/score_abs_max_det_mean`：deterministic `score` 的绝对值最大值（均值），用于验证 “scale_factor 太小导致 score 贴近 0” 这类假设
- `prune/score_top1_gap_det_mean`：deterministic 下每行 `top1 - top2` 的平均 gap（再对行求均值），衡量 score 的可分性

#### 4.6.2 基于实际日志的判定（step 100/200/400 一致）

你提供的日志中（以 step=100 为例）：

- `task_valid_len_mean≈87`：task token 并不短，padding/mask 不是主因。
- `argmax_union_mean≈2.54` 且 `argmax_top1_share_mean≈0.75`：deterministic 投票极端集中，确实是“并集塌缩到 2～3 个 patch”的结构性问题。
- `score_abs_max_det_mean≈29`：score 并没有被压到 `[-0.02, 0.02]` 这种量级；因此“scale_factor 太小导致差异被压扁”这个解释不成立。
- 关键异常在 **queries**：
  - `patches_std_mean≈0.209`、`task_std_mean≈0.296`：patch/task embedding 本身并非近似常量。
  - 但 `queries_std_mean≈0.0089`：cross-attn 输出 `queries` 在不同 patch query 上几乎不变（高度同质化）。
  - 同时 `score_top1_gap_det_mean≈0.04`：每行 top1/top2 差极小，导致大量 query 行共享相同的 top 列（并集自然很小）。

结论：当前“kept≈2～3”的主要瓶颈不是 `scale_factor`，而是 `queries = SDPA(...)` 这一步产生了近似常量的 `queries`，导致 `score` 行间同构、投票集中。

#### 4.6.3 下一步应优先做的 ablation（验证 `queries` 同质化的根因）

建议按影响面从小到大做以下对照（每个只跑很短一段就够看趋势）：

1) 去掉/替换 `queries` 的二次 RMSNorm（`queries = _rms_norm(queries)`）：观察 `queries_std_mean`、`argmax_union_mean` 是否显著上升。
2) 调整 score 的温度/噪声相对尺度：让 `score_top1_gap_det_mean` 与 `noise_scale` 在同一量级（否则噪声只是“随机抖动”，不改变结构性偏置）。
3) 引入显式预算（`keep_ratio/keep_tokens`）作为对照：确认任务是否真的需要 >3 个 patch 才能工作，避免“策略塌缩”与“任务本来就简单”混淆。

---

## 结论（当前状态）

- OpenPI 侧已经修复了 “padding mask 缺失导致 1 token” 的致命问题，并实现了跨相机合并剪枝与推理态 position 对齐。
- 已补齐 LightVLA 的关键训练态对齐点：训练态使用 noisy selection indices 对齐 position_ids（并增加了更完整的 pruning telemetry）。
- 但 deterministic kept tokens 仍然异常偏低（~2.5/768），且 telemetry 显示塌缩发生在 `queries = SDPA(...)` 输出同质化（`queries_std_mean` 极低）阶段，而非 `scale_factor` 导致 score 贴近 0。
- 下一步优先做针对 `queries` 同质化的 ablation（见 4.6.3），同时用 `keep_ratio/keep_tokens` 做“任务是否真需要更多视觉 token”的 sanity check。
