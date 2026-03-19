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
- 单次 dump：`bash tools/run_libero_dump.sh`
- 剪枝/实验配置：统一通过 `--opt-config <yaml>`

Client 默认跑完整 LIBERO 评测：

- `--suite libero_spatial`
- `--trials 50`

## Pruning Inside T64

checkpoint:

- `checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_inside_encoder_t64/59999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/serve_pi05_libero.sh \
  --run-tag vla_opt_pruning_inside_encoder_t64 \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_inside_encoder_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/legacy_inside_encoder.yaml \
  --port 8003 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag vla_opt_pruning_inside_encoder_t64
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/run_libero_dump.sh \
  --run-tag vla_opt_pruning_inside_encoder_t64_viz \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_inside_encoder_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/legacy_inside_encoder.yaml \
  --port 8003 \
  --gpu 0 \
  --trials 50
```

这条命令会自动完成：

- server observe dump
- client eval
- `render_png`
- `render_observe_video.py`
- `pruning_stats`

## Pruning After T64

checkpoint:

- `checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/serve_pi05_libero.sh \
  --run-tag vla_opt_pruning_after_encoder_t64_29999 \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder.yaml \
  --port 8003 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag vla_opt_pruning_after_encoder_t64_29999
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/run_libero_dump.sh \
  --run-tag vla_opt_pruning_after_encoder_t64_29999_viz \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder.yaml \
  --port 8003 \
  --gpu 0 \
  --trials 50
```

## Pruning After T128

checkpoint:

- `checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t128/59999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/serve_pi05_libero.sh \
  --run-tag vla_opt_pruning_after_encoder_t128_59999 \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder.yaml \
  --port 8003 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 0 \
  --run-tag vla_opt_pruning_after_encoder_t128_59999
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/run_libero_dump.sh \
  --run-tag vla_opt_pruning_after_encoder_t128_59999_viz \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder.yaml \
  --port 8003 \
  --gpu 0 \
  --trials 50
```

## Pruning After T64 Gaussian

gaussian 是推理时通过 `--opt-config` 打开的，checkpoint 仍然用 pruning after t64 checkpoint。

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/serve_pi05_libero.sh \
  --run-tag vla_opt_pruning_after_t64_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder_gauss.yaml \
  --port 8004 \
  --gpu 0
```

对应 client：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8004 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag vla_opt_pruning_after_t64_gauss
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/run_libero_dump.sh \
  --run-tag vla_opt_pruning_after_t64_gauss_viz \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder_gauss.yaml \
  --port 8004 \
  --gpu 0 \
  --trials 50
```

## Pruning After T128 Gaussian

gaussian 是推理时通过 `--opt-config` 打开的，checkpoint 仍然用 pruning after t128 checkpoint。

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/serve_pi05_libero.sh \
  --run-tag vla_opt_pruning_after_t128_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder_gauss.yaml \
  --port 8005 \
  --gpu 1
```

对应 client：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8005 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag vla_opt_pruning_after_t128_gauss
```

可视化推荐直接使用一键 dump runner：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/run_libero_dump.sh \
  --run-tag vla_opt_pruning_after_t128_gauss_viz \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder_gauss.yaml \
  --port 8005 \
  --gpu 0 \
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

- one-click run root: `runs/<run_tag>/<timestamp>/`
- server logs: `runs/<run_tag>/<timestamp>/server/`
- client logs: `runs/<run_tag>/<timestamp>/client/`
- rollout videos: `runs/<run_tag>/<timestamp>/client/videos/`
- observe dump and postprocess outputs: `runs/<run_tag>/<timestamp>/observe/`
