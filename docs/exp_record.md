# LIBERO Spatial Pruning 实验总览

本页当前只维护 pruning 实验的统一口径、配置矩阵和文档索引。

旧版“方案名 / 保留率 / 实际运行配置”已确认未对齐，原结果表已清空；在完成按 canonical matrix 的重跑前，本页不再展示跨方案结论性数据。

## 文档索引

- `pruning_inside_encoder`：[`docs/pruning_inside_encoder.md`](/workspace/laiminxin/vla-opt/third_party/openpi/docs/pruning_inside_encoder.md)
- `pruning_after_encoder`：[`docs/pruning_post_encoder.md`](/workspace/laiminxin/vla-opt/third_party/openpi/docs/pruning_post_encoder.md)
- 方法差异说明：[`docs/openpi_pruning_methods_comparison.md`](/workspace/laiminxin/vla-opt/third_party/openpi/docs/openpi_pruning_methods_comparison.md)
- 统一命令手册：[`docs/pruning_runbook.md`](/workspace/laiminxin/vla-opt/third_party/openpi/docs/pruning_runbook.md)

## 统一统计口径

- runtime profile：
  - 默认使用 PyTorch 的 `torch.compile(...)` 默认行为
  - 如需完全关闭编译，显式设置 `OPENPI_TORCH_COMPILE=0`
- 评测命令统一使用 `--suite libero_spatial --trials 50`
- 推理时间口径统一取同一次 client log 里的 `policy_infer_ms`
- 每个表格行只能对应一个真实 run：
  - 一个 `checkpoint`
  - 一个 `opt-config`
  - 一个 `run-tag`
  - 一个 `client log`

## Canonical Matrix

| family | variant | checkpoint | opt-config | keep_tokens_per_view | keep_ratio | gauss |
| --- | --- | --- | --- | ---: | ---: | --- |
| `baseline` | `baseline` | `checkpoints/pi05_libero_spatial/pi05_baseline/29999` | `-` | 256 | 100% | off |
| `pruning_inside_encoder` | `inside_t64` | `checkpoints/pi05_libero_spatial/inside_t64/59999` | `config/pruning/inside_t64.yaml` | 64 | 25% | off |
| `pruning_inside_encoder` | `inside_t64_gauss` | `checkpoints/pi05_libero_spatial/inside_t64/59999` | `config/pruning/inside_t64_gauss.yaml` | 64 | 25% | on |
| `pruning_inside_encoder` | `inside_t128` | `checkpoints/pi05_libero_spatial/inside_t128/59999` | `config/pruning/inside_t128.yaml` | 128 | 50% | off |
| `pruning_inside_encoder` | `inside_t128_gauss` | `checkpoints/pi05_libero_spatial/inside_t128/59999` | `config/pruning/inside_t128_gauss.yaml` | 128 | 50% | on |
| `pruning_after_encoder` | `post_t64` | `checkpoints/pi05_libero_spatial/post_t64/29999` | `config/pruning/post_t64.yaml` | 64 | 25% | off |
| `pruning_after_encoder` | `post_t64_gauss` | `checkpoints/pi05_libero_spatial/post_t64/29999` | `config/pruning/post_t64_gauss.yaml` | 64 | 25% | on |
| `pruning_after_encoder` | `post_t128` | `checkpoints/pi05_libero_spatial/post_t128/59999` | `config/pruning/post_t128.yaml` | 128 | 50% | off |
| `pruning_after_encoder` | `post_t128_gauss` | `checkpoints/pi05_libero_spatial/post_t128/59999` | `config/pruning/post_t128_gauss.yaml` | 128 | 50% | on |

## 当前状态

- 配置文件与命令文档已对齐到上表。
- 历史结果记录保留在旧日志中，但当前不作为跨方案比较依据。
- 后续若恢复结果表，只能从上述 matrix 的重跑日志重新生成。
- 运行产物命名统一以 `variant` 为主键；调用命令时应直接将 canonical `variant` 作为 `--run-tag` 传入。
- 默认目录规则：
  - 普通评测：`runs/<variant>/<step>_<timestamp>/`
  - dump / 可视化：`runs/<variant>/viz_<step>_<timestamp>/`
  - observe video 文件名带 `<variant>_<step>`

## Canonical Run Result Summary

以下成功率按对应 `client.log` 的最终汇总行记录，统一口径为 `--suite libero_spatial --trials 50`，`Total episodes = 500`。多次结果按时间顺序使用 `/` 连接。

| variant | checkpoint step | keep_tokens_per_view | keep_ratio | gauss | success / total | total success rate |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `inside_t64` | 59999 | 64 | 25% | off | `471 / 500` / `470 / 500` | `94.2%` / `94.0%` |
| `inside_t64_gauss` | 59999 | 64 | 25% | on | `465 / 500` / `472 / 500` | `93.0%` / `94.4%` |
| `inside_t128` | 59999 | 128 | 50% | off | `470 / 500` / `471 / 500` | `94.0%` / `94.2%` |
| `inside_t128_gauss` | 59999 | 128 | 50% | on | `470 / 500` / `469 / 500` | `94.0%` / `93.8%` |
| `post_t64` | 29999 | 64 | 25% | off | `470 / 500` / `476 / 500` | `94.0%` / `95.2%` |
| `post_t64_gauss` | 29999 | 64 | 25% | on | `474 / 500` / `486 / 500` | `94.8%` / `97.2%` |
| `post_t128` | 59999 | 128 | 50% | off | `480 / 500` / `482 / 500` | `96.0%` / `96.4%` |
| `post_t128_gauss` | 59999 | 128 | 50% | on | `485 / 500` / `481 / 500` | `97.0%` / `96.2%` |

## 2026-03-21 Parallel Split 重跑结果

以下结果对应本次 `tools/run_libero_parallel_split.sh` 重跑后各 variant 的最新一次普通评测目录，统计口径仍为 `--suite libero_spatial --trials 50`，`Total episodes = 500`。

| variant | latest run dir | success / total | total success rate | policy_infer_ms mean | policy_infer_ms p50 | policy_infer_ms p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `inside_t64` | `runs/inside_t64/59999_20260321_150506` | `474 / 500` | `94.8%` | `103.90` | `99.91` | `104.70` |
| `inside_t64_gauss` | `runs/inside_t64_gauss/59999_20260321_150506` | `473 / 500` | `94.6%` | `106.92` | `103.31` | `116.61` |
| `inside_t128` | `runs/inside_t128/59999_20260321_161530` | `466 / 500` | `93.2%` | `107.53` | `103.69` | `107.06` |
| `inside_t128_gauss` | `runs/inside_t128_gauss/59999_20260321_161644` | `478 / 500` | `95.6%` | `106.15` | `103.01` | `110.35` |
| `post_t64` | `runs/post_t64/29999_20260321_172708` | `480 / 500` | `96.0%` | `132.00` | `125.37` | `142.86` |
| `post_t64_gauss` | `runs/post_t64_gauss/29999_20260321_184253` | `471 / 500` | `94.2%` | `132.40` | `128.48` | `143.05` |
| `post_t128` | `runs/post_t128/59999_20260321_172613` | `486 / 500` | `97.2%` | `136.78` | `131.54` | `142.99` |
| `post_t128_gauss` | `runs/post_t128_gauss/59999_20260321_184318` | `483 / 500` | `96.6%` | `132.88` | `127.97` | `144.69` |

简要观察：

- 本轮最佳结果是 `post_t128`，达到 `486 / 500`，`97.2%`。
- `inside_t128_gauss` 明显优于 `inside_t128`，从 `93.2%` 提升到 `95.6%`。
- 本轮 `post_t64_gauss` 低于 `post_t64`，没有复现旧表里更优的那次结果。
- 时长口径补充为对应 `client.log` 中逐步记录的 `policy_infer_ms` 统计；当前日志里没有统一输出完整 run 的总 wall-clock 时间。

## 2026-03-24 Full LIBERO Baseline 结果

以下结果对应 `pi05_libero_all_cross_attn_post_gauss` baseline checkpoint
`checkpoints/pi05_libero_all_cross_attn_post_gauss/pi05_libero_all_baseline/60000`，
统一评测口径为各 suite 使用 `--trials 50`。本节只汇总成功率，不统计时延。

| suite | run dir | success / total | total success rate |
| --- | --- | ---: | ---: |
| `libero_spatial` | `runs/pi05_libero_all_baseline_step60000_spatial/60000_20260324_110837` | `490 / 500` | `98.0%` |
| `libero_object` | `runs/pi05_libero_all_baseline_step60000_object/60000_20260324_110837` | `487 / 500` | `97.4%` |
| `libero_goal` | `runs/pi05_libero_all_baseline_step60000_goal/60000_20260324_110837` | `476 / 500` | `95.2%` |
| `libero_10` | `runs/pi05_libero_all_baseline_step60000_libero10/60000_20260324_110837` | `467 / 500` | `93.4%` |
| `all 4 suites` | `spatial + object + goal + libero_10` | `1920 / 2000` | `96.0%` |

简要观察：

- 四个 suite 里，本轮最高的是 `libero_spatial`，`490 / 500`，`98.0%`。
- 本轮最低的是 `libero_10`，`467 / 500`，`93.4%`。
- 四套件合并后总成功率为 `96.0%`。



## 2026-03-23 `cross_attn_post` `libero_all` 并行评测

以下结果对应 `recipe_parallel_eval_libero_all()` 的最新一轮普通评测目录，配置固定为：

- checkpoint：`checkpoints/pi05_libero_all_cross_attn_post_gauss/cross_attn_post_t64_gauss/60000`
- policy config：`pi05_libero_all_cross_attn_post_gauss`
- opt config：`config/pruning/cross_attn_post_t64_gauss.yaml`
- trials：`50`

| suite | run dir | success / total | total success rate | policy_infer_ms mean | policy_infer_ms p50 | policy_infer_ms p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `libero_spatial` | `runs/cross_attn_post_step60000_spatial/60000_20260323_162540` | `477 / 500` | `95.4%` | `83.83` | `75.95` | `76.86` |
| `libero_object` | `runs/cross_attn_post_step60000_object/60000_20260323_162540` | `479 / 500` | `95.8%` | `81.51` | `75.40` | `77.93` |
| `libero_goal` | `runs/cross_attn_post_step60000_goal/60000_20260323_162540` | `478 / 500` | `95.6%` | `82.72` | `75.29` | `77.65` |
| `libero_10` | `runs/cross_attn_post_step60000_libero10/60000_20260323_162540` | `416 / 500` | `83.2%` | `75.46` | `74.75` | `76.27` |

简要观察：

- 四个 suite 合计 `1850 / 2000`，总成功率 `92.5%`。
- `spatial` / `object` / `goal` 三个单套件都稳定在 `95%+`。
- `libero_10` 明显更难，当前为 `83.2%`，是整体均值的主要下拉项。
- 四个 suite 的 `policy_infer_ms` p50 都在 `75ms` 左右；`libero_10` 的均值更低，但 query 数明显更多，因此总评测时长最长。


  | suite | baseline | cross_attn_post | 差值 |
  | --- | ---: | ---: | ---: |
  | libero_spatial | 98.0% | 95.4% | -2.6 |
  | libero_object | 97.4% | 95.8% | -1.6 |
  | libero_goal | 95.2% | 95.6% | +0.4 |
  | libero_10 | 93.4% | 83.2% | -10.2 |
  | all 4 suites | 96.0% | 92.5% | -3.5 |

## 2026-03-25 `cross_attn_post` step 89999 最新并行评测

以下结果对应你最新一轮并行评测，配置固定为：

- checkpoint：`checkpoints/pi05_libero_all_cross_attn_post_gauss/cross_attn_post_t64_gauss/89999`
- policy config：`pi05_libero_all_cross_attn_post_gauss`
- opt config：`config/pruning/cross_attn_post_t64_gauss.yaml`
- host：`127.0.0.1`
- trials：`50`

| suite | run dir | success / total | total success rate | policy_infer_ms mean | policy_infer_ms p50 | policy_infer_ms p95 | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `libero_spatial` | `runs/cross_attn_post_step90k_spatial/89999_20260325_141803` | `481 / 500` | `96.2%` | `73.95` | `59.32` | `77.06` | 正常完成 |
| `libero_object` | `runs/cross_attn_post_step90k_object/89999_20260325_141803` | `489 / 500` | `97.8%` | `75.92` | `54.75` | `100.17` | 正常完成 |
| `libero_10` | `runs/cross_attn_post_step90k_libero10/89999_20260325_141803` | `421 / 500` | `84.2%` | `65.80` | `57.98` | `87.47` | 正常完成 |
| `libero_goal` | `runs/cross_attn_post_step90k_goal/89999_20260325_182128` | `476 / 500` | `95.2%` | `61.65` | `59.16` | `60.10` | 正常完成 |

简要观察：

- 四个 suite 合计 `1867 / 2000`，总成功率 `93.35%`。
- 四个 suite 里，`libero_object` 最好，达到 `489 / 500`，`97.8%`。
- `libero_spatial` 也稳定在 `96.2%`，与上一轮 `step60000` 的 `95.4%` 相比提升 `+0.8`。
- `libero_10` 本轮为 `421 / 500`，`84.2%`，相比上一轮 `step60000` 的 `83.2%` 小幅提升 `+1.0`。
- `libero_goal` 最新重跑已恢复正常，达到 `476 / 500`，`95.2%`；`client.log` 与 `server.log` 中未再出现此前的 websocket `1011`、`cuDNN` 或 `CUDA graph` 错误。
