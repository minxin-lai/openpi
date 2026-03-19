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

在两个终端里分别启动 server 和 client。

Client 默认跑完整 LIBERO 评测：

- `--suite libero_spatial`
- `--trials 50`

## Pruning Inside T64

checkpoint:

- `checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_inside_encoder_t64`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash server_pi05_libero_vla_opt.sh \
  --run-tag vla_opt_pruning_inside_encoder_t64 \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_inside_encoder_t64 \
  --policy-config pi05_libero_spatial \
  --pruning-config config/pruning/legacy_inside_encoder.yaml \
  --port 8003 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash client_libero_eval_vla_opt.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag vla_opt_pruning_inside_encoder_t64
```

## Pruning After T64

checkpoint:

- `checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash server_pi05_libero_vla_opt_default.sh \
  --run-tag vla_opt_pruning_after_encoder_t64_29999 \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999 \
  --policy-config pi05_libero_spatial \
  --port 8003 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash client_libero_eval_vla_opt.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag vla_opt_pruning_after_encoder_t64_29999
```

## Pruning After T128

checkpoint:

- `checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t128/59999`

Terminal 1:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash server_pi05_libero_vla_opt_default.sh \
  --run-tag vla_opt_pruning_after_encoder_t128_59999 \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t128/59999 \
  --policy-config pi05_libero_spatial \
  --port 8003 \
  --gpu 0
```

Terminal 2:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash client_libero_eval_vla_opt.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 0 \
  --run-tag vla_opt_pruning_after_encoder_t128_59999
```

## Pruning After T64 Gaussian

gaussian 是推理时通过 server 侧 pruning config 打开的，checkpoint 仍然用 pruning after t64 checkpoint。

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash server_pi05_libero_vla_opt_gauss.sh \
  --run-tag vla_opt_pruning_after_t64_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999 \
  --policy-config pi05_libero_spatial \
  --port 8004 \
  --gpu 0
```

对应 client：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash client_libero_eval_vla_opt.sh \
  --host 127.0.0.1 \
  --port 8004 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag vla_opt_pruning_after_t64_gauss
```

## Pruning After T128 Gaussian

gaussian 是推理时通过 server 侧 pruning config 打开的，checkpoint 仍然用 pruning after t128 checkpoint。

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash server_pi05_libero_vla_opt_gauss.sh \
  --run-tag vla_opt_pruning_after_t128_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t128/59999 \
  --policy-config pi05_libero_spatial \
  --port 8005 \
  --gpu 1
```

对应 client：

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash client_libero_eval_vla_opt.sh \
  --host 127.0.0.1 \
  --port 8005 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag vla_opt_pruning_after_t128_gauss
```

## Outputs

- server log: `runs/<run_tag>/server_<timestamp>.log`
- client log: `runs/libero/logs/<run_tag>/<suite>_<timestamp>.log`
- video: `runs/libero/videos/<run_tag>/<suite>_<timestamp>`
