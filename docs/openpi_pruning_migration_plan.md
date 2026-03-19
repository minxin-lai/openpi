# OpenPI Pruning Migration Plan

## 1. Background

当前存在两套相关但不等价的剪枝路径。

为避免混淆，先固定父仓和子仓的代码基线：

- 当前主线基线
  - parent repo `/workspace/laiminxin/vla-opt`
    - `branch`: `master`
    - `commit`: `8a4f35707d2c24df4354ab496fc5cb894cc082bf`
  - child repo `/workspace/laiminxin/vla-opt/third_party/openpi`
    - `branch`: `vla-opt`
    - `commit`: `7f2e28ff7f34cc53e86e4cccbbe8fea48be7d7f3`
- legacy inside-encoder 基线
  - parent repo `/workspace/laiminxin/vla-opt`
    - `branch/tag`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`
    - `commit`: `19349f2206bd8e89effc5c8e8f566b7fa5630b80`
  - child repo `/workspace/laiminxin/vla-opt/third_party/openpi`
    - `branch/tag`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`
    - `commit`: `db33a7447afbb62806169fe0b697e0c212fd1eb9`

- `inside-encoder`
  - 历史实现来源：`/workspace/laiminxin/vla-opt-openpi-old`
  - 结果基线也主要来自这个旧 worktree
  - 语义上是旧的 legacy 路径
- `post_encoder`
  - 当前主线实现
  - 已经在 `third_party/openpi` 当前工作分支中使用

迁移目标不是把 `inside-encoder` 简单压缩成“和 `post_encoder` 只有剪枝位置不同”的一个开关。

更准确地说：

- 两者共享一些公共部件
  - VE-FiLM
  - 条件化 score head
  - Top-K + STE
  - `mask/gather` 两阶段
- 但 `inside-encoder` 仍保留独立语义
  - 单层 inside-encoder prune
  - 不依赖 `post_encoder` 的多层 `score_num_layers` 聚合
  - 不走 gaussian smoothing
  - 默认参数和旧训练/推理脚本更接近 legacy worktree

因此，正确的迁移方式是：

- 算法主实现回归父仓 `/workspace/laiminxin/vla-opt`
- `third_party/openpi` 只做训练/推理接入
- pruning 行为统一通过 YAML 配置选择
- 不修改 `src/openpi/training/config.py`

## 2. Architecture And Ownership

### 2.1 Parent Repo Responsibilities

父仓 `/workspace/laiminxin/vla-opt` 负责：

- pruning YAML 解析与校验
- `inside-encoder` legacy 算法实现
- `post_encoder` 当前算法实现
- OpenPI Pi0.5 集成入口
- 运行时模式选择

这部分属于算法与集成核心，不应该下沉到 `third_party/openpi`。

### 2.2 OpenPI Repo Responsibilities

`third_party/openpi` 只负责：

- 训练脚本接收 pruning config 路径
- 推理服务脚本接收 pruning config 路径
- 模型加载时把 pruning config 路径传给父仓 API
- 启动脚本和文档更新

`third_party/openpi` 不应重复实现一套 legacy/post 算法逻辑。

## 3. Target Design

### 3.1 Pruning Config Files

配置文件建议放在：

- `third_party/openpi/config/pruning/legacy_inside_encoder.yaml`
- `third_party/openpi/config/pruning/post_encoder.yaml`
- `third_party/openpi/config/pruning/post_encoder_gauss.yaml`

放在 `openpi` 下的原因：

- 这些配置主要服务 openpi 的训练和推理入口
- 与现有 `docs/`、启动脚本、实验命令放在一起更方便

### 3.2 Runtime Config Loader

父仓提供统一 loader，例如：

- `load_openpi_pruning_config(path) -> OpenPIPruningRuntimeConfig`

这个运行时配置对象负责：

- 解析 YAML
- 校验字段
- 针对 `train` / `serve` 阶段生成 resolved 参数
- 约束 legacy/post 两种模式的合法性

### 3.3 Runtime Enable API

父仓对外只暴露统一入口：

- `enable_ve_film_on_pi05(model, cfg)`
- `enable_openpi_pruning_from_config(model, cfg, phase="train" | "serve")`

`enable_openpi_pruning_from_config(...)` 内部按 `mode` 分流：

- `legacy_inside_encoder -> enable_legacy_inside_encoder_pruning_on_pi05(...)`
- `post_encoder -> enable_ve_pruning_on_pi05(...)`

## 4. YAML Schema

建议 schema 如下：

```yaml
mode: legacy_inside_encoder

ve_film:
  enabled: true
  num_blocks: 4

ste_prune:
  enabled: true
  k: 64
  train_stage: auto
  serve_stage: gather
  train_tau: 2.0
  train_tau_final: 0.2
  serve_tau: 1.0
  switch_step: -1
  prune_layer: null
  score_mlp_hidden_dim: null
  score_num_layers: 1
  lambda_budget: 0.01
  lambda_bin: 0.01
  gaussian:
    enabled: false
    sigma: 0.65
    kernel_size: null
```

### 4.1 Common Fields

- `mode`
- `ve_film.enabled`
- `ve_film.num_blocks`
- `ste_prune.enabled`
- `ste_prune.k`
- `ste_prune.train_stage`
- `ste_prune.serve_stage`
- `ste_prune.train_tau`
- `ste_prune.train_tau_final`
- `ste_prune.serve_tau`
- `ste_prune.switch_step`
- `ste_prune.prune_layer`
- `ste_prune.score_mlp_hidden_dim`
- `ste_prune.score_num_layers`
- `ste_prune.lambda_budget`
- `ste_prune.lambda_bin`
- `ste_prune.gaussian.enabled`
- `ste_prune.gaussian.sigma`
- `ste_prune.gaussian.kernel_size`

### 4.2 Mode Constraints

#### `legacy_inside_encoder`

- `score_num_layers` 必须固定为 `1`
- `gaussian.enabled` 必须为 `false`
- `prune_layer` 可为空，运行时按 first FiLM layer 推断
- 剪枝逻辑走 legacy inside-encoder 路径

#### `post_encoder`

- `score_num_layers >= 1`
- gaussian 可开启
- 剪枝逻辑走当前 post-encoder 路径

## 5. File-Level Implementation Checklist

### 5.1 Parent Repo Changes

#### A. Add YAML Config Loader

新增文件：

- `/workspace/laiminxin/vla-opt/src/vla_opt/pruning/openpi_config.py`

职责：

- 读取 YAML
- dataclass/schema 定义
- 模式合法性校验
- 对外提供 `load_openpi_pruning_config(path)`

建议内容：

- `GaussianConfig`
- `VeFilmConfig`
- `StePruneConfig`
- `OpenPIPruningRuntimeConfig`
- `resolve_for_train()`
- `resolve_for_serve()`

#### B. Extend OpenPI Pi0.5 Integration

修改文件：

- `/workspace/laiminxin/vla-opt/src/vla_opt/integrations/openpi_pi05.py`

需要完成：

- 保留当前 `post_encoder` 逻辑
- 从旧 worktree 回迁 `inside-encoder` 逻辑
- 新增统一入口 API

建议新增/保留的符号：

- `OpenPIVeFilmConfig`
- `OpenPIVePruningConfig`
- `LegacyInsideEncoderPruningConfig`
- `enable_ve_film_on_pi05`
- `enable_ve_pruning_on_pi05`
- `enable_legacy_inside_encoder_pruning_on_pi05`
- `enable_openpi_pruning_from_config`
- `resolve_pi05_vision_encoder_layers`
- `find_pi05_ste_prune_layer_index`

#### C. Parent Repo Tests

新增或更新测试：

- `/workspace/laiminxin/vla-opt/...` 现有测试目录下新增 config 和集成测试

至少覆盖：

- YAML 解析
- legacy/post 两种模式选择
- 不合法配置报错
- legacy 模式禁止 gaussian
- legacy 模式强制 `score_num_layers == 1`

#### D. Parent Repo Dependency

修改文件：

- `/workspace/laiminxin/vla-opt/pyproject.toml`

增加：

- `PyYAML`

### 5.2 OpenPI Repo Changes

#### E. Train Entrypoint

修改文件：

- [train_pytorch.py](/workspace/laiminxin/vla-opt/third_party/openpi/scripts/train_pytorch.py)

改动：

- 增加 `--vla-opt-pruning-config`
- 若设置该参数：
  - 调父仓 `load_openpi_pruning_config`
  - 调父仓 `enable_ve_film_on_pi05`
  - 调父仓 `enable_openpi_pruning_from_config(..., phase="train")`
- 旧 CLI 参数保留一段兼容期，但 YAML 优先

#### F. Serve Entrypoint

修改文件：

- [serve_policy.py](/workspace/laiminxin/vla-opt/third_party/openpi/scripts/serve_policy.py)

改动：

- 增加 `--vla-opt-pruning-config`
- 设置环境变量：
  - `VLA_OPT_PRUNING_CONFIG`
- 不在此处解析 pruning 算法字段

#### G. Model Loading Hook

修改文件：

- [model.py](/workspace/laiminxin/vla-opt/third_party/openpi/src/openpi/models/model.py)

改动：

- 若存在 `VLA_OPT_PRUNING_CONFIG`
  - 调父仓 loader
  - 调父仓统一 enable API
- 若不存在
  - fallback 到旧 env vars 兼容逻辑

#### H. Pruning YAML Files

新增文件：

- `config/pruning/legacy_inside_encoder.yaml`
- `config/pruning/post_encoder.yaml`
- `config/pruning/post_encoder_gauss.yaml`

#### I. Launcher Scripts

修改文件：

- [tools/serve_pi05_libero.sh](/workspace/laiminxin/vla-opt/third_party/openpi/tools/serve_pi05_libero.sh)
- [tools/train_pi05_experiment.sh](/workspace/laiminxin/vla-opt/third_party/openpi/tools/train_pi05_experiment.sh)

改动：

- 增加或内置 `--opt-config`
- 不再硬编码 post-encoder 具体参数集

#### J. Docs

修改文件：

- [pruning_inside_encoder.md](/workspace/laiminxin/vla-opt/third_party/openpi/docs/pruning_inside_encoder.md)
- [pruning_post_encoder.md](/workspace/laiminxin/vla-opt/third_party/openpi/docs/pruning_post_encoder.md)

需要写清楚：

- `inside-encoder` 历史结果来自 `/workspace/laiminxin/vla-opt-openpi-old`
- 当前主线复现方式是使用 `legacy_inside_encoder.yaml`
- `post_encoder` 和 `post_encoder_gauss` 使用对应 YAML

## 6. Recommended Execution Order

建议按下面顺序实施，避免边界不清导致返工。

### Phase 1: Parent Repo Core

1. 在父仓增加 pruning YAML loader
2. 在父仓 `openpi_pi05.py` 中恢复 legacy inside-encoder 路径
3. 补统一入口 API
4. 加最小单元测试

完成标准：

- 父仓已有独立可调用的 legacy/post 两套启用逻辑
- 运行时 config 已能区分两种模式

### Phase 2: OpenPI Wiring

5. 给 `train_pytorch.py` 增加 `--vla-opt-pruning-config`
6. 给 `serve_policy.py` 增加 `--vla-opt-pruning-config`
7. 给 `model.py` 增加 `VLA_OPT_PRUNING_CONFIG` 读取逻辑
8. 新增 YAML 文件

完成标准：

- 训练和推理都可以只传一个 pruning YAML 路径

### Phase 3: Scripts And Docs

9. 修改训练和服务启动脚本
10. 更新 inside/post 文档

完成标准：

- 日常使用不再依赖 legacy worktree
- 文档命令完全基于当前主线

### Phase 4: Validation

11. 旧 inside-encoder checkpoint smoke test
12. 当前 post-encoder checkpoint smoke test
13. 文档命令复核

## 7. Acceptance Criteria

### 7.1 Legacy Inside-Encoder

满足以下条件即视为迁移成功：

- 使用当前主线仓库
- 不切换到 `/workspace/laiminxin/vla-opt-openpi-old`
- 通过 `legacy_inside_encoder.yaml`
- 能成功加载旧 inside-encoder checkpoint
- 能成功跑通一次推理

### 7.2 Current Post-Encoder

满足以下条件即视为不回归：

- 使用 `post_encoder.yaml` 可成功加载并推理
- 使用 `post_encoder_gauss.yaml` 也可成功加载并推理

### 7.3 User Experience

满足以下条件即视为接入完成：

- 训练只需传 `--vla-opt-pruning-config <path>`
- 推理只需传 `--vla-opt-pruning-config <path>`
- 文档不再要求切换 legacy worktree

## 8. Risks

### 8.1 Main Risk

最大的风险是把旧 `inside-encoder` 错误地等同于当前 `point=encoder_layer` 的抽象分支。

这样虽然表面统一了接口，但会导致：

- 行为不再贴近旧 worktree
- 历史结果复现偏移
- 用户误以为两者只差剪枝位置

### 8.2 Integration Risk

如果把算法主实现下沉到 `third_party/openpi`，会出现：

- 代码职责反转
- 父仓和 openpi 两边各有一套 OpenPI 集成
- 后续维护困难

### 8.3 Dependency Risk

如果 YAML 解析依赖未显式加入，会在训练或推理时才失败。

## 9. Rollback Strategy

如果中途发现迁移偏离旧 worktree 语义，优先回退到：

- 保留当前 `post_encoder` 主线不动
- 仅把 legacy inside-encoder 作为独立路径重新接回父仓
- 不做“统一实现”式重构

回滚优先级：

1. 保住当前 `post_encoder`
2. 单独修复 `legacy_inside_encoder`
3. 最后再做 YAML 和脚本层收口

## 10. Final Recommendation

按职责划分，正确实施方式是：

- 父仓 `/workspace/laiminxin/vla-opt` 负责实现和配置解析
- `third_party/openpi` 负责接线和文档
- `inside-encoder` 保留独立 legacy 路径
- `post_encoder` 保持当前主线实现
- 两者通过 YAML 统一选择，而不是通过错误的“同一算法只差位置”来合并

这套方案既能保留旧结果复现能力，也不会污染当前 `post_encoder` 主线。
