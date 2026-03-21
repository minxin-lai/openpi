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
