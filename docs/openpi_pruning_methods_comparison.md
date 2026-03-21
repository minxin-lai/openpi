# OpenPI 两种剪枝方法对比说明

本文档用于解释当前 OpenPI 代码路径中两种视觉 token 剪枝方法的差别，并补齐它们在训练侧 / 推理侧的完整链路。

实验结果与最新汇总结论统一维护在 [`docs/exp_record.md`](/workspace/laiminxin/post_gauss_attn/third_party/openpi/docs/exp_record.md)，本文档只说明当前实现链路与方法差异，不重复维护结果表。

对比对象：

- encoder 内剪枝：`pruning_inside_encoder`
- encoder 后剪枝：`pruning_after_encoder`

## 1. 一句话总结

- `pruning_inside_encoder` 是在 `SigLIP encoder 中间层` 直接剪枝，后续视觉层继续处理缩短后的 token 序列。
- `pruning_after_encoder` 是在 `完整 SigLIP encoder 之后` 再统一剪枝，视觉编码器本体先完整提特征，后续多模态部分再吃缩短后的 token 序列。

## 2. 当前代码状态说明

当前代码里有两件事需要分开看：

- **配置语义**：两类配置都定义了 `train_stage` 与 `serve_stage`，也都允许 `mask / gather`。
- **明确接线状态**：
  - `serve / eval` 侧已经通过 pruning config + runtime config 明确接通
  - `train_pytorch.py` 也已经接入训练期 pruning runtime
  - 当 `train_stage=auto` 且提供 `train_schedule` 时，训练侧会按进度切换 `stage / k`
  - 如果旧 YAML 没有 `train_schedule`，仍保持现有 `switch_step/半程切换` 行为

因此，下文的“训练侧链路”表示：

- 当前训练入口如何实际驱动这些模块
- 以及当前配置如何定义它们的阶段行为

而“推理侧链路”表示：

- 当前 `load_pruning_config(...) -> to_runtime("serve") -> enable_pi05_pruning_from_runtime_config(...)` 这条明确接通的实际执行路径

## 3. 最核心差别

### 3.1 剪枝位置不同

encoder 内剪枝：

- 剪枝发生在某个 SigLIP encoder layer 内部。
- 当前层输出后，立刻计算 score、执行 Top-K、再做 mask 或 gather。

encoder 后剪枝：

- 剪枝发生在 vision encoder 的末端，也就是 `post_encoder`。
- 完整 encoder 跑完后，先汇总最后几层的分数，再统一做 Top-K 和 mask 或 gather。

### 3.2 后续计算路径不同

encoder 内剪枝：

- 一旦执行真实 gather，后面的 SigLIP layer 就只处理 `K` 个 token。
- 属于 encoder 内部剪枝，更像直接改视觉 backbone 的计算图。

encoder 后剪枝：

- SigLIP encoder 全程仍处理原始 `N` 个 token。
- 真正缩短的是 encoder 输出后的视觉 token 序列。
- 后续进入 PaliGemma / action head 的部分才使用 `K` 个 token。

### 3.3 分数来源不同

encoder 内剪枝：

- 分数来自单个剪枝层的当前特征。
- 当前实现要求 `inside_encoder` 的 `score_num_layers == 1`。

encoder 后剪枝：

- 分数来自最后若干个 FiLM 层的 score。
- 这些 score 会先做平均，再执行 Top-K。
- 当前实现默认支持 `score_num_layers=3`。

### 3.4 FiLM 依赖不同

两条路径当前都不是“无条件剪枝”：

- 当前默认接线都会先启用 `FiLM`
- `pruning_after_encoder` 明确依赖末端 `FiLMedSiglipEncoderLayer`
- `pruning_inside_encoder` 在当前主入口中同样先启用 `FiLM`，再接 inside pruning

所以当前代码中的准确说法是：

- `inside` 和 `after` 的差别在于 **剪枝位置与缩短序列的生效位置**
- 不在于“一个有 FiLM、一个没有 FiLM”

### 3.5 FiLM、`pooled_mlp score head`、`cross_attn score head` 的关系

这三者不是同一层面的东西。

`FiLM`：

- 作用对象是 vision encoder 内部的视觉特征
- 作用位置是视觉 backbone 内部
- 做的是特征调制
- 它回答的是：“视觉特征本身要不要先变得更 instruction-aware”

`pooled_mlp score head`：

- 作用对象是 pruning score
- 作用位置是打分阶段
- 做的是“根据视觉特征 + 文本条件输出每个 patch 的分数”
- 它回答的是：“这些 patch 里谁该留”

`cross_attn score head`：

- 本质上也是 score head
- 它不改剪枝位置，只改“每一层 score 怎么算”
- 与 `pooled_mlp score head` 的区别，是它让每个 patch 逐词看文本，而不是先把整句文本做 mean pooling

因此当前与后续计划中的关系应理解为：

- `FiLM` 和 `score head` 不是互相替代关系
- `pooled_mlp score head` 与 `cross_attn score head` 才是同一层面的两种实现
- 若升级到 `cross_attn score head`，更合理的 v1 仍然是：
  - 保留 `FiLM`
  - 用 `cross_attn score head` 替换当前 `pooled_mlp score head`

### 3.6 为什么有了 score head 还要 FiLM

因为两者承担的职责不同：

- `FiLM`：先把“看什么”注入视觉特征
- `score head`：再根据这些特征决定“留谁”

如果没有 `FiLM`，那么视觉 backbone 更接近先提一套通用视觉特征，最后再由 score head 硬做任务条件打分。
这样并不是不能工作，但 score head 会同时承担两件事：

- 弥补视觉特征不够 task-aware
- 输出最终 pruning score

而保留 `FiLM` 时，score head 的输入已经是更具任务相关性的视觉特征，打分负担更小。

因此在当前路线里，更自然的组合是：

```text
FiLM-adjusted vision features
  ↓
score head
  ↓
gauss / top-k / mask-gather
```

最短概括：

- `FiLM`：改特征
- `score head`：算分数

## 4. 训练侧完整链路

> 这里描述的是当前配置语义和模块执行形态。
> 它回答的是“如果按当前训练配置使用这些模块，链路会怎样工作”。

### 4.1 encoder 内剪枝：训练侧链路

```text
compute_loss
  ↓
preprocess_observation
  ↓
embed_prefix
  ↓
每个 view 进入 SigLIP vision encoder
  ↓
SigLIP 前几层（仍是 N 个 token）
  ↓
末端若干层带 FiLM 调制
  ↓
到指定 prune_layer
  ↓
当前层输出 x
  ↓
Score Head(x, cond_tokens) 生成单层 score
  ↓
可选 Gaussian smoothing
  ↓
按 train_stage 执行：
  - mask: 保持长度 N，只做 STE gating
  - gather: 真实执行 N → K
  ↓
若是 gather，后续 SigLIP 层只处理 K 个 token
  ↓
image embeddings 输出到 prefix
  ↓
与 text prefix / suffix action tokens 一起进入 PaliGemma
  ↓
compute_loss
```

训练侧要点：

- 训练时 inside 的结构特征是“剪枝发生得早”。
- 一旦训练阶段进入真实 gather，后续视觉层就会看到短序列。
- 因此 inside 训练更贴近“真正把视觉 backbone 也改成短序列计算”的路线。

最直白的理解：

- `inside`：先进入带 FiLM 的末端视觉层，在某个层后执行剪枝；如果后面还有 FiLMed 层，那么剪枝后后续视觉层会继续跑，后续 FiLMed 层也会继续做 FiLM。
- 可简写为：`FiLM -> prune -> continue vision encoder`

### 4.2 encoder 后剪枝：训练侧链路

```text
compute_loss
  ↓
preprocess_observation
  ↓
embed_prefix
  ↓
每个 view 进入 SigLIP vision encoder
  ↓
完整 SigLIP encoder 全程处理 N 个 token
  ↓
末端若干层带 FiLM 调制
  ↓
最后 score_num_layers 个层输出 score tap
  ↓
多层 score 平均
  ↓
可选 Gaussian smoothing
  ↓
按 train_stage 执行：
  - mask: 保持长度 N，只做 STE gating
  - gather: 在 encoder 输出后统一执行 N → K
  ↓
缩短后的 image embeddings 输出到 prefix
  ↓
与 text prefix / suffix action tokens 一起进入 PaliGemma
  ↓
compute_loss
```

训练侧要点：

- 训练时 after 的结构特征是“SigLIP 本体仍然完整跑 N 个 token”。
- 即使训练阶段进入真实 gather，缩短序列也只发生在 encoder 输出之后。
- 因此它更像“保守地保持视觉 backbone 稳定，把剪枝影响限制在 encoder 之后”。

最直白的理解：

- `post`：先经过末端 FiLMed 层并收集多层 score，完整 vision encoder 跑完后才统一剪枝。
- 剪枝后已经没有后续 vision encoder 了，直接进入 encoder 后链路。
- 可简写为：`FiLM -> finish vision encoder -> prune -> enter multimodal`

## 5. 推理侧完整链路

> 这里描述的是当前仓内最明确接通的实际执行路径。
> 主入口是通过 pruning config 加载模型，并使用 `to_runtime("serve")` 得到 `stage=serve_stage`。

### 5.1 encoder 内剪枝：推理侧链路

```text
load_pytorch
  ↓
读取 pruning config
  ↓
to_runtime("serve")
  ↓
enable_pi05_pruning_from_runtime_config
  ↓
先启用 FiLM
  ↓
在指定 prune_layer 上挂 VePrunedSiglipEncoderLayer
  ↓
sample_actions / eval
  ↓
embed_prefix
  ↓
每个 view 进入 SigLIP
  ↓
在 prune_layer 内：
  当前层输出 x
  → Score Head(x, cond_tokens)
  → 可选 Gaussian smoothing
  → 按 serve_stage 执行 prune（当前 canonical 配置默认 gather）
  ↓
若为 gather，后续 SigLIP 层只处理 K 个 token
  ↓
输出 image embeddings
  ↓
进入 prefix / multimodal 后续计算
```

推理侧要点：

- `inside` 的 serve 侧收益覆盖更早。
- 后续视觉层和多模态部分都能直接吃到短序列。

一句话记忆：

- `inside`：剪枝后还有 vision encoder。

### 5.2 encoder 后剪枝：推理侧链路

```text
load_pytorch
  ↓
读取 pruning config
  ↓
to_runtime("serve")
  ↓
enable_pi05_pruning_from_runtime_config
  ↓
先启用 FiLM
  ↓
将末端 score_num_layers 个 FiLMed 层替换为 score tap
  ↓
重写 embed_image:
  orig_embed_image(image)
    → 完整 SigLIP encoder
    → 多层 score 累积
  finalize_mean_scores()
    → 可选 Gaussian smoothing
    → 按 serve_stage 执行 prune（当前 canonical 配置默认 gather）
  返回 post-prune image embeddings
  ↓
sample_actions / eval
  ↓
embed_prefix
  ↓
输出 image embeddings
  ↓
进入 prefix / multimodal 后续计算
```

推理侧要点：

- `after` 的 serve 侧是在 encoder 输出后统一剪枝。
- SigLIP 本体没有因为 pruning 变短。
- 主要受益的是 encoder 之后的 prefix / multimodal 部分。

一句话记忆：

- `post`：剪枝后不再有 vision encoder，只剩 encoder 后链路。

## 6. 统一比较

### 6.1 训练侧比较

| 维度 | `pruning_inside_encoder` | `pruning_after_encoder` |
| --- | --- | --- |
| 剪枝发生位置 | SigLIP 中间层内部 | 完整 SigLIP 之后 |
| 分数来源 | 单层当前特征 | 末端多层 score 平均 |
| SigLIP 是否全程处理 `N` | 否，真实 gather 后只处理 `K` | 是，全程仍处理 `N` |
| 真实 gather 的影响范围 | 后续视觉层 + 多模态部分 | 主要是 encoder 后部分 |
| 训练风格 | 更激进，更早改变 backbone 计算图 | 更保守，先保持 backbone 稳定 |
| FiLM 关系 | 当前默认接线中先启用 FiLM | 明确依赖末端 FiLMed 层 |

### 6.2 推理侧比较

| 维度 | `pruning_inside_encoder` | `pruning_after_encoder` |
| --- | --- | --- |
| serve 主执行位置 | `prune_layer` 内 | `embed_image` 返回后 |
| 默认 serve 阶段 | `gather` | `gather` |
| 后续视觉层是否处理短序列 | 是 | 否 |
| encoder 后多模态部分是否处理短序列 | 是 | 是 |
| 观测 / dump 位置 | inside encoder | post encoder |
| 工程形态 | 更侵入视觉 backbone | 更模块化 |

### 6.3 最短对外说明

- encoder 内剪枝：早剪枝，后续视觉层和多模态部分都能受益，但对视觉 backbone 的计算图侵入更强。
- encoder 后剪枝：晚剪枝，视觉 backbone 更稳定，主要减少 encoder 之后的 token 开销，工程上更模块化。

## 7. 当前代码中的对应关系

`pruning_inside_encoder` 对应：

- 当前汇总表中的方案名：`pruning_inside_encoder`
- 当前 canonical 配置：`inside_t64*.yaml` 与 `inside_t128*.yaml`
- `mode`: `inside_encoder`

`pruning_after_encoder` 对应：

- 当前汇总表中的方案名：`pruning_after_encoder`
- 当前 canonical 配置：`post_t64*.yaml` 与 `post_t128*.yaml`
- `mode`: `post_encoder`

`cross_attn_post` 对应：

- 当前新增方法族：`cross_attn_post`
- 当前 canonical 配置：`cross_attn_post_t64_gauss.yaml`
- `mode`: `post_encoder`
- 与 `post_t64_gauss` 的关系：
  - 两者都走 `post_encoder`
  - 两者都保留 `FiLM`
  - 两者都在 `gauss -> top-k -> mask/gather` 这条主链上工作
  - 区别是 `post_t64_gauss` 仍使用 `pooled_mlp score head`
  - `cross_attn_post_t64_gauss` 改为 `cross_attn score head`
  - `cross_attn_post_t64_gauss` 的 canonical 默认是最后 `3` 个 FiLM 层，并聚合最后 `3` 层 score

## 8. 高斯 score 后处理变体

当前两种方法都可以叠加一个 `gauss` 变体：

- 在 Top-K 之前，先对 score map 做高斯平滑。
- 这样选出的 patch 区域通常更连续，更像“成片保留”而不是零散点状保留。

可以把它理解成：

```text
raw scores → Gaussian smoothing → Top-K → mask / gather
```

如果这次只画“两种方法”的主对比，通常不必把高斯分支画进主图。
比较好的做法是单独在角落补一个小注释：

- `gauss` 是剪枝前的 score 后处理变体，可用于 `pruning_inside_encoder` 或 `pruning_after_encoder`

## 9. 画图时建议保留的四个标签

无论你用什么画图软件，建议至少保留这四个标签：

- `Prune Position`
- `Train Path`
- `Serve Path`
- `Who benefits from shorter sequence?`

对应说明：

- encoder 内剪枝：`Prune Position = inside encoder`
- encoder 后剪枝：`Prune Position = after encoder`
- encoder 内剪枝：`shorter sequence benefits later vision layers + LLM`
- encoder 后剪枝：`shorter sequence mainly benefits post-encoder multimodal part`
