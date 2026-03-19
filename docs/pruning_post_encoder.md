# Pi0.5 Prune50 测试记录

## Git Labels

Parent repo (`/workspace/laiminxin/vla-opt`):

- `branch`: `master`
- `commit`: `8a4f35707d2c24df4354ab496fc5cb894cc082bf`
- `tag_exact`: `none`
- `tag_nearest`: ``

OpenPI repo (`/workspace/laiminxin/vla-opt/third_party/openpi`):

- `branch`: `vla-opt`
- `commit`: `7f2e28ff7f34cc53e86e4cccbbe8fea48be7d7f3`
- `tag_exact`: `none`
- `tag_nearest`: ``

## Checkpoint

- `ckpt_dir`: `/workspace/laiminxin/vla-opt/third_party/openpi/checkpoints/pi05_libero_spatial/vla_opt_pi05_post_encoder_128token/60000`
- `policy_config`: `pi05_libero_spatial`
- `ste_prune_k`: `128`

## Experiment Config

- `vision_tokens_per_view`: `256`
- `keep_tokens_per_view`: `128`
- `prune_ratio`: `50%`
- `ve_film_num_blocks`: `4`
- `ste_prune_point`: `post_encoder`
- `ste_prune_stage@serve`: `gather`
- `ste_prune_tau@serve`: `1.0`
- `gauss(off)`: `disabled`
- `gauss(on)`: `sigma=0.65`
- `suite`: `libero_spatial`
- `trials_per_task`: `50`
- `tasks`: `10`

## Case 1: GPU 3, no gauss, t50

Server:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash server_pi05_libero_vla_opt.sh \
  --gpu 3 \
  --port 8003 \
  --run-tag prune50_keep128_nogauss_gpu3
```

Client:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash client_libero_eval_vla_opt.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 3 \
  --run-tag prune50_keep128_nogauss_gpu3 \
  &> post_encoder_prune50_keep128_nogauss_t50.log
```

Outputs:

- `client_log`: `third_party/openpi/post_encoder_prune50_keep128_nogauss_t50.log`
- `video_out`: `third_party/openpi/runs/libero/videos/prune50_keep128_nogauss_gpu3/libero_spatial_20260316_164636`
- `client_artifact_log`: `third_party/openpi/runs/libero/logs/prune50_keep128_nogauss_gpu3/libero_spatial_20260316_164636.log`

## Success Rate Analysis

- `total_success_rate`: `0.96`
- `total_episodes`: `500`
- `total_successes`: `480/500`
- 共有 `10` 个任务，每个任务 `50` 个 trial。

Per-task success rate:

- `50/50 = 1.000`: `pick up the black bowl between the plate and the ramekin and place it on the plate`
- `46/50 = 0.920`: `pick up the black bowl next to the ramekin and place it on the plate`
- `50/50 = 1.000`: `pick up the black bowl from table center and place it on the plate`
- `49/50 = 0.980`: `pick up the black bowl on the cookie box and place it on the plate`
- `49/50 = 0.980`: `pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate`
- `48/50 = 0.960`: `pick up the black bowl on the ramekin and place it on the plate`
- `47/50 = 0.940`: `pick up the black bowl next to the cookie box and place it on the plate`
- `46/50 = 0.920`: `pick up the black bowl on the stove and place it on the plate`
- `48/50 = 0.960`: `pick up the black bowl next to the plate and place it on the plate`
- `47/50 = 0.940`: `pick up the black bowl on the wooden cabinet and place it on the plate`

## Case 2: GPU 4, gauss, t50

Server:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash server_pi05_libero_vla_opt.sh \
  --gpu 4 \
  --port 8004 \
  --run-tag prune50_keep128_gauss_gpu4 \
  --ste-prune-gaussian \
  --ste-prune-gaussian-sigma 0.65
```

Client:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash client_libero_eval_vla_opt.sh \
  --host 127.0.0.1 \
  --port 8004 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 4 \
  --run-tag prune50_keep128_gauss_gpu4 \
  &> post_encoder_prune50_keep128_gauss_t50.log
```

Outputs:

- `client_log`: `third_party/openpi/post_encoder_prune50_keep128_gauss_t50.log`
- `video_out`: `third_party/openpi/runs/libero/videos/prune50_keep128_gauss_gpu4/libero_spatial_20260316_164656`
- `client_artifact_log`: `third_party/openpi/runs/libero/logs/prune50_keep128_gauss_gpu4/libero_spatial_20260316_164656.log`

## Success Rate Analysis

- `total_success_rate`: `0.97`
- `total_episodes`: `500`
- `total_successes`: `485/500`
- 共有 `10` 个任务，每个任务 `50` 个 trial。

Per-task success rate:

- `50/50 = 1.000`: `pick up the black bowl between the plate and the ramekin and place it on the plate`
- `49/50 = 0.980`: `pick up the black bowl next to the ramekin and place it on the plate`
- `50/50 = 1.000`: `pick up the black bowl from table center and place it on the plate`
- `47/50 = 0.940`: `pick up the black bowl on the cookie box and place it on the plate`
- `48/50 = 0.960`: `pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate`
- `47/50 = 0.940`: `pick up the black bowl on the ramekin and place it on the plate`
- `49/50 = 0.980`: `pick up the black bowl next to the cookie box and place it on the plate`
- `46/50 = 0.920`: `pick up the black bowl on the stove and place it on the plate`
- `50/50 = 1.000`: `pick up the black bowl next to the plate and place it on the plate`
- `49/50 = 0.980`: `pick up the black bowl on the wooden cabinet and place it on the plate`
