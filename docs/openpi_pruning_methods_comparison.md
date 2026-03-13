# OpenPI 两种剪枝方法对比说明

本文档用于解释当前 OpenPI 代码路径中两种视觉 token 剪枝方法的差别，适合直接作为画示意图或写汇报说明的依据。

对比对象：

- 旧方法：`encoder_layer prune`
- 新方法：`post_encoder prune`

## 1. 一句话总结

- 旧方法是在 `SigLIP encoder 中间层` 直接剪枝，后续视觉层继续处理缩短后的 token 序列。
- 新方法是在 `完整 SigLIP encoder 之后` 再统一剪枝，视觉编码器本体先完整提特征，后续多模态部分再吃缩短后的 token 序列。

## 2. 最核心差别

### 2.1 剪枝位置不同

旧方法：

- 剪枝发生在某个 SigLIP encoder layer 内部。
- 当前层输出后，立刻计算 score、执行 Top-K、再做 gather。

新方法：

- 剪枝发生在 vision encoder 的末端，也就是 `post_encoder`。
- 完整 encoder 跑完后，先汇总最后几层的分数，再统一做 Top-K 和 gather。

### 2.2 后续计算路径不同

旧方法：

- 一旦剪枝完成，后面的 SigLIP layer 就只处理 `K` 个 token。
- 属于“早剪枝”，更像直接改视觉 backbone 的计算图。

新方法：

- SigLIP encoder 全程仍处理原始 `N` 个 token。
- 真正缩短的是 encoder 输出后的视觉 token 序列。
- 后续进入 PaliGemma / action head 的部分才使用 `K` 个 token。

### 2.3 分数来源不同

旧方法：

- 分数来自单个剪枝层的当前特征。

新方法：

- 分数来自最后若干个 FiLM 层的 score。
- 这些 score 会先做平均，再执行 Top-K。
- 当前实现默认支持 `score_num_layers=3`。

### 2.4 可解释性和工程形态不同

旧方法：

- 结构更直接。
- 剪枝侵入 encoder 内部。
- 更强调“尽早减少视觉层后续计算量”。

新方法：

- 结构更模块化。
- 更容易在 encoder 输出处统一观测和 dump。
- 当前主入口默认使用这条路径。

## 3. 简单流程图

### 3.1 旧方法：Encoder-Layer Prune

```text
图像
  ↓
Patch Tokens (N)
  ↓
SigLIP 前几层
  ↓
某一层 Encoder 内
  ↓
FiLM + Score Head
  ↓
Top-K / STE 剪枝
  ↓
Gather: N → K
  ↓
SigLIP 后几层（继续处理 K 个 token）
  ↓
PaliGemma / Action Head
```

你在图里可以把“某一层 Encoder 内”这个框标成红色，突出它是“中途剪枝”。

### 3.2 新方法：Post-Encoder Prune

```text
图像
  ↓
Patch Tokens (N)
  ↓
完整 SigLIP Encoder（全程保持 N 个 token）
  ↓
最后几层 FiLM 特征
  ↓
多层 Score 平均
  ↓
Top-K / STE 剪枝
  ↓
Gather: N → K
  ↓
PaliGemma / Action Head
```

你在图里可以把“完整 SigLIP Encoder”画成一个完整大框，再把剪枝模块单独放在它后面，视觉上会非常清楚。

## 4. 一张图的对比画法

如果你只想画一张总图，推荐用下面这个版本：

```text
                OpenPI 两种剪枝方法对比

旧方法
Image → Tokens(N) → SigLIP前层 → [中间层剪枝] → Tokens(K) → SigLIP后层 → LLM

新方法
Image → Tokens(N) → SigLIP全层 → [末端剪枝]   → Tokens(K) → LLM
```

这个版本最适合 PPT。

## 5. 讲解时可以直接说的话

可直接用于口头解释：

> 旧方法是在视觉编码器内部提前剪枝，所以后续视觉层本身也会省计算。
> 新方法是在完整视觉编码之后再剪枝，因此视觉 backbone 更稳定，分数也能来自多层聚合，但减少的主要是 encoder 之后的 token 开销。

再简化一点可以说：

- 旧方法：早剪枝
- 新方法：晚剪枝

## 6. 当前代码中的对应关系

旧方法对应：

- 旧工作树 / 旧目录中的 `encoder_layer prune`
- 典型 checkpoint 描述为 legacy `stage_a_ste`

新方法对应：

- 当前主目录里的默认路径
- 默认 checkpoint 为 `vla_opt_pi05_stage_a_ste_post_encoder_prune`

## 7. 新方法额外的高斯变体

当前新方法还有一个 `gauss` 变体：

- 在 Top-K 之前，先对 score map 做高斯平滑。
- 这样选出的 patch 区域通常更连续，更像“成片保留”而不是零散点状保留。

可以把它理解成：

```text
raw scores → Gaussian smoothing → Top-K → gather
```

如果你这次只画“两种方法”的主对比，通常不必把高斯分支画进主图。
比较好的做法是单独在角落补一个小注释：

- `gauss` 是 `post_encoder prune` 的一个 score 后处理变体。

## 8. 推荐图注

图注可直接使用下面任意一版。

简版：

> 旧方法在视觉编码器内部执行剪枝，新方法在视觉编码器输出后统一执行剪枝。

稍完整版：

> 旧方法属于 encoder-layer pruning，在 SigLIP 中途将 token 从 N 压缩到 K；新方法属于 post-encoder pruning，先完成视觉编码，再基于多层聚合分数统一剪枝。

## 9. 推荐你最终画图时保留的三个标签

无论你用什么画图软件，建议至少保留这三个标签：

- `Prune Position`
- `Tokens: N → K`
- `Who benefits from shorter sequence?`

对应说明：

- 旧方法：`Prune Position = inside encoder`
- 新方法：`Prune Position = after encoder`
- 旧方法：`shorter sequence benefits later vision layers + LLM`
- 新方法：`shorter sequence mainly benefits post-encoder multimodal part`

