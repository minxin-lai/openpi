# Pi0.5 Encoder Layer Prune 测试记录

## Git Labels

- `worktree`: `/workspace/laiminxin/vla-opt-openpi-old/third_party/openpi`
- `branch`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`
- `commit`: `5bd8d12`
- `tag_exact`: `none`
- `tag_nearest`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune-1-g5bd8d12`
- `parent_repo`: `/workspace/laiminxin/vla-opt-openpi-old`
- `parent_branch`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`
- `parent_commit`: `19349f2`
- `parent_tag_nearest`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`

## Checkpoint

- `ckpt_dir`: `/workspace/laiminxin/vla-opt/third_party/openpi/checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/59999`
- `policy_config`: `pi05_libero_spatial`
- `ste_prune_k`: `64`

## Current Mainline Command

当前主线不再切 legacy worktree，直接用 pruning YAML。

Server:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/serve_pi05_libero.sh \
  --run-tag vla_opt_legacy_inside \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/legacy_inside_encoder.yaml \
  --port 8003
```

Client:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --trials 1 \
  --run-tag vla_opt_legacy_inside
```

说明：

- `legacy_inside_encoder.yaml` 选择旧的 inside-encoder 语义。
- `59999` 是当前已验证可加载的 legacy checkpoint。

## Case 1: GPU 1, step59999

Server:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/serve_pi05_libero.sh \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/legacy_inside_encoder.yaml \
  --gpu 1 \
  --port 8002
```

Client:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8002 \
  --trials 50 \
  --gpu 1 \
  --run-tag encoder_layer_prune_step59999 \
  &> encoder_layer_prune_step59999.log
```

Outputs:

- `client_log`: `third_party/openpi/encoder_layer_prune_step59999.log`
- `video_out`: `third_party/openpi/runs/libero/videos/libero_spatial_<timestamp>`

## Success Rate Analysis

- `total_success_rate`: `0.952`
- `total_episodes`: `500`
- `total_successes`: `476/500`
- 共有 `10` 个任务，每个任务 `50` 个 trial。

Per-task success rate:

- `49/50 = 0.980`: `pick up the black bowl between the plate and the ramekin and place it on the plate`
- `49/50 = 0.980`: `pick up the black bowl next to the ramekin and place it on the plate`
- `50/50 = 1.000`: `pick up the black bowl from table center and place it on the plate`
- `50/50 = 1.000`: `pick up the black bowl on the cookie box and place it on the plate`
- `48/50 = 0.960`: `pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate`
- `46/50 = 0.920`: `pick up the black bowl on the ramekin and place it on the plate`
- `46/50 = 0.920`: `pick up the black bowl next to the cookie box and place it on the plate`
- `43/50 = 0.860`: `pick up the black bowl on the stove and place it on the plate`
- `49/50 = 0.980`: `pick up the black bowl next to the plate and place it on the plate`
- `46/50 = 0.920`: `pick up the black bowl on the wooden cabinet and place it on the plate`
