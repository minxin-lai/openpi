# OpenPI Pruning Runbook

## Git Labels

current parent repo:

- `repo`: `/workspace/laiminxin/vla-opt`
- `branch`: `master`
- `commit`: `8a4f357`
- `tag_exact`: `none`
- `tag_nearest`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune-20-g8a4f357`

current openpi repo:

- `repo`: `/workspace/laiminxin/vla-opt/third_party/openpi`
- `branch`: `vla-opt`
- `commit`: `7f2e28f`
- `tag_exact`: `none`
- `tag_nearest`: `0312-post-encoder-gauss-2-g7f2e28f`

legacy inside-encoder reference:

- `worktree`: `/workspace/laiminxin/vla-opt-openpi-old/third_party/openpi`
- `branch`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`
- `commit`: `5bd8d12`
- `tag_exact`: `none`
- `tag_nearest`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune-1-g5bd8d12`
- `parent_repo`: `/workspace/laiminxin/vla-opt-openpi-old`
- `parent_branch`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`
- `parent_commit`: `19349f2`
- `parent_tag_nearest`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`

统一入口：

- 分步运行：`bash tools/serve_pi05_libero.sh` + `bash tools/eval_libero.sh`
- 一键运行（不做 dump / 可视化）：`bash tools/run_libero.sh`
- 单次 dump：`bash tools/run_libero_dump.sh`
- 剪枝/实验配置：统一通过 `--opt-config <yaml>`

统一 runtime 口径：

- `OPENPI_TORCH_COMPILE=1`
- `OPENPI_TORCH_COMPILE_MODE=reduce-overhead`
- 不显式设置 `TRITON_AUTOTUNE` / `TORCHINDUCTOR_MAX_AUTOTUNE`

observe / dump 语义：

- `pruning_inside_encoder` 在 trace / tensor dump 中写入 `phase=inside_encoder`
- `pruning_after_encoder` 在 trace / tensor dump 中写入 `phase=post_encoder`
- `tools/run_libero_dump.sh` 使用统一 observe config，同时接受 `inside_encoder` 与 `post_encoder`
- observe 原始输出会按 `task/episode/query` 分层写到 `runs/<variant>/viz_<step>_<timestamp>/observe/`

有效 dump 的最小检查项：

- 每个 query 目录下存在 `trace.jsonl`
- pruning 事件的 `summary.output_tokens` 与 `keep_mask.sum()` 一致
- pruning 事件的 `summary.output_tokens` 与 `len(keep_indices)` 一致
- `patch_grid_hw` 与 token 数量匹配；当前 Pi0.5 LIBERO patch grid 应为 `16 x 16`，即 `input_tokens=256`
- `render_png` / `render_observe_video.py` / `pruning_stats` 的输入都来自上述 observe 目录

Client 默认跑完整 LIBERO 评测：

- `--suite libero_spatial`
- `--trials 50`

一键脚本默认口径：

- `bash tools/run_libero.sh`: `--trials 50`
- `bash tools/run_libero_dump.sh`: `--trials 1`

## Canonical One-Click Matrix

下表专门对应 `docs/exp_record.md` 的 canonical matrix，确保每个 variant 都有一条 `bash tools/run_libero.sh` 入口。

| family | variant | one-click command |
| --- | --- | --- |
| `baseline` | `baseline` | `bash tools/run_libero.sh --run-tag baseline --ckpt-dir checkpoints/pi05_libero_spatial/pi05_baseline/29999 --policy-config pi05_libero_spatial --port 9000 --gpu 0 --trials 50` |
| `pruning_inside_encoder` | `inside_t64` | `bash tools/run_libero.sh --run-tag inside_t64 --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 --policy-config pi05_libero_spatial --opt-config config/pruning/inside_t64.yaml --port 9001 --gpu 0 --trials 50` |
| `pruning_inside_encoder` | `inside_t64_gauss` | `bash tools/run_libero.sh --run-tag inside_t64_gauss --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 --policy-config pi05_libero_spatial --opt-config config/pruning/inside_t64_gauss.yaml --port 9002 --gpu 0 --trials 50` |
| `pruning_inside_encoder` | `inside_t128` | `bash tools/run_libero.sh --run-tag inside_t128 --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 --policy-config pi05_libero_spatial --opt-config config/pruning/inside_t128.yaml --port 9003 --gpu 0 --trials 50` |
| `pruning_inside_encoder` | `inside_t128_gauss` | `bash tools/run_libero.sh --run-tag inside_t128_gauss --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 --policy-config pi05_libero_spatial --opt-config config/pruning/inside_t128_gauss.yaml --port 9004 --gpu 0 --trials 50` |
| `pruning_after_encoder` | `post_t64` | `bash tools/run_libero.sh --run-tag post_t64 --ckpt-dir checkpoints/pi05_libero_spatial/post_t64/29999 --policy-config pi05_libero_spatial --opt-config config/pruning/post_t64.yaml --port 9005 --gpu 0 --trials 50` |
| `pruning_after_encoder` | `post_t64_gauss` | `bash tools/run_libero.sh --run-tag post_t64_gauss --ckpt-dir checkpoints/pi05_libero_spatial/post_t64/29999 --policy-config pi05_libero_spatial --opt-config config/pruning/post_t64_gauss.yaml --port 9007 --gpu 0 --trials 50` |
| `pruning_after_encoder` | `post_t128` | `bash tools/run_libero.sh --run-tag post_t128 --ckpt-dir checkpoints/pi05_libero_spatial/post_t128/59999 --policy-config pi05_libero_spatial --opt-config config/pruning/post_t128.yaml --port 9006 --gpu 0 --trials 50` |
| `pruning_after_encoder` | `post_t128_gauss` | `bash tools/run_libero.sh --run-tag post_t128_gauss --ckpt-dir checkpoints/pi05_libero_spatial/post_t128/59999 --policy-config pi05_libero_spatial --opt-config config/pruning/post_t128_gauss.yaml --port 9008 --gpu 0 --trials 50` |

## Baseline

checkpoint:

- `checkpoints/pi05_libero_spatial/pi05_baseline/29999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag baseline \
  --ckpt-dir checkpoints/pi05_libero_spatial/pi05_baseline/29999 \
  --policy-config pi05_libero_spatial \
  --port 9000 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 9000 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag baseline
```

如需一键运行完整评测但不产出 dump / 可视化：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero.sh \
  --run-tag baseline \
  --ckpt-dir checkpoints/pi05_libero_spatial/pi05_baseline/29999 \
  --policy-config pi05_libero_spatial \
  --port 9000 \
  --gpu 2 \
  --trials 50
```

## Pruning Inside T64

checkpoint:

- `checkpoints/pi05_libero_spatial/inside_t64/59999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag inside_t64 \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t64.yaml \
  --port 9001 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 9001 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag inside_t64
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero_dump.sh \
  --run-tag inside_t64 \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t64.yaml \
  --port 9001 \
  --gpu 0 \
  --trials 1
```

这条命令会自动完成：

- server observe dump
- client eval
- `render_png`
- `render_observe_video.py`
- `pruning_stats`

如需一键运行完整评测但不产出 dump / 可视化：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero.sh \
  --run-tag inside_t64 \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t64.yaml \
  --port 9001 \
  --gpu 3 \
  --trials 50
```

## Pruning Inside T64 Gaussian

gaussian 是推理时通过 `--opt-config` 打开的，checkpoint 仍然用 pruning inside t64 checkpoint。

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag inside_t64_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t64_gauss.yaml \
  --port 9002 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 9002 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag inside_t64_gauss
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero_dump.sh \
  --run-tag inside_t64_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t64_gauss.yaml \
  --port 9002 \
  --gpu 0 \
  --trials 1
```

如需一键运行完整评测但不产出 dump / 可视化：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero.sh \
  --run-tag inside_t64_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t64_gauss.yaml \
  --port 9002 \
  --gpu 4 \
  --trials 50
```

## Pruning Inside T128

checkpoint:

- `checkpoints/pi05_libero_spatial/inside_t128/59999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag inside_t128 \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t128.yaml \
  --port 9003 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 9003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag inside_t128
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero_dump.sh \
  --run-tag inside_t128 \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t128.yaml \
  --port 9003 \
  --gpu 0 \
  --trials 1
```

如需一键运行完整评测但不产出 dump / 可视化：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero.sh \
  --run-tag inside_t128 \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t128.yaml \
  --port 9003 \
  --gpu 6 \
  --trials 50
```

## Pruning Inside T128 Gaussian

gaussian 是推理时通过 `--opt-config` 打开的，checkpoint 仍然用 pruning inside t128 checkpoint。

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag inside_t128_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t128_gauss.yaml \
  --port 9004 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 9004 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag inside_t128_gauss
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero_dump.sh \
  --run-tag inside_t128_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t128_gauss.yaml \
  --port 9004 \
  --gpu 0 \
  --trials 1
```

如需一键运行完整评测但不产出 dump / 可视化：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero.sh \
  --run-tag inside_t128_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t128_gauss.yaml \
  --port 9004 \
  --gpu 4 \
  --trials 50
```

## Pruning After T64

checkpoint:

- `checkpoints/pi05_libero_spatial/post_t64/29999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag post_t64 \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t64.yaml \
  --port 9005 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 9005 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag post_t64
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero_dump.sh \
  --run-tag post_t64 \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t64.yaml \
  --port 9005 \
  --gpu 0 \
  --trials 1
```

如需一键运行完整评测但不产出 dump / 可视化：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero.sh \
  --run-tag post_t64 \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t64.yaml \
  --port 9005 \
  --gpu 4 \
  --trials 50
```

## Pruning After T128

checkpoint:

- `checkpoints/pi05_libero_spatial/post_t128/59999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag post_t128 \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t128.yaml \
  --port 9006 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 9006 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 0 \
  --run-tag post_t128
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero_dump.sh \
  --run-tag post_t128 \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t128.yaml \
  --port 9006 \
  --gpu 0 \
  --trials 1
```

如需一键运行完整评测但不产出 dump / 可视化：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero.sh \
  --run-tag post_t128 \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t128.yaml \
  --port 9006 \
  --gpu 6 \
  --trials 50
```

## Pruning After T64 Gaussian

gaussian 是推理时通过 `--opt-config` 打开的，checkpoint 仍然用 pruning after t64 checkpoint。

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag post_t64_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t64_gauss.yaml \
  --port 9007 \
  --gpu 0
```

对应 client：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 9007 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag post_t64_gauss
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero_dump.sh \
  --run-tag post_t64_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t64_gauss.yaml \
  --port 9007 \
  --gpu 0 \
  --trials 1
```

如需一键运行完整评测但不产出 dump / 可视化：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero.sh \
  --run-tag post_t64_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t64_gauss.yaml \
  --port 9007 \
  --gpu 6 \
  --trials 50
```

## Pruning After T128 Gaussian

gaussian 是推理时通过 `--opt-config` 打开的，checkpoint 仍然用 pruning after t128 checkpoint。

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag post_t128_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t128_gauss.yaml \
  --port 9008 \
  --gpu 1
```

对应 client：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 9008 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag post_t128_gauss
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero_dump.sh \
  --run-tag post_t128_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t128_gauss.yaml \
  --port 9008 \
  --gpu 4 \
  --trials 1
```

如需一键运行完整评测但不产出 dump / 可视化：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/run_libero.sh \
  --run-tag post_t128_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/post_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_t128_gauss.yaml \
  --port 9008 \
  --gpu 6 \
  --trials 50
```

## Re-render Existing Dump

如果只想对已有 `observe_dump_dir` 单独重渲染：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

export OBS_DIR="<observe_dump_dir>"
export PYTHONPATH="/workspace/laiminxin/vla-opt/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "${OBS_DIR}"

uv run python -m vla_opt.observe.render_png \
  --run-dir "${OBS_DIR}"

uv run python scripts/render_observe_video.py \
  --run-dir "${OBS_DIR}" \
  --overlay-kind both

uv run python -m vla_opt.observe.pruning_stats \
  --run-dir "${OBS_DIR}"
```

## Outputs

- `--run-tag` 应直接传 canonical `variant`，例如 `post_t128_gauss`
- `step` 由 `--ckpt-dir` 最后一段推导，例如 `29999` / `59999`
- one-click run root: `runs/<variant>/<step>_<timestamp>/`
- one-click dump root: `runs/<variant>/viz_<step>_<timestamp>/`
- server logs: `runs/<variant>/<step>_<timestamp>/server/`
- client logs: `runs/<variant>/<step>_<timestamp>/client/`
- rollout videos: `runs/<variant>/<step>_<timestamp>/client/videos/`
- observe dump and postprocess outputs: `runs/<variant>/viz_<step>_<timestamp>/observe/`
- observe-only server default dump: `runs/<variant>/observe_<step>_<timestamp>/`
