# OpenPI Pi0.5 LIBERO Spatial Quickstart

只保留当前主入口。旧的 generic / perf / legacy 入口已经删除。

## 0) 一次性准备

```bash
cd third_party/openpi
uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
uv run scripts/compute_norm_stats.py --config-name pi05_libero_spatial
```

确认：

```bash
ls /workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json
```

## 1) baseline 对照

起 server：

```bash
cd third_party/openpi
bash server_pi05_libero_baseline.sh --port 8002 --gpu 0
```

默认行为：
- 开启 `torch.compile`
- 不强制关闭 Inductor/Triton autotune

如需复现旧口径：

```bash
cd third_party/openpi
OPENPI_TORCH_COMPILE=0 TRITON_AUTOTUNE=0 TORCHINDUCTOR_MAX_AUTOTUNE=0 \
  bash server_pi05_libero_baseline.sh --port 8002 --gpu 0
```

对应 client：

```bash
cd third_party/openpi
bash client_libero_eval_baseline.sh --port 8002 --trials 20 --gpu 0
```

## 2) vla-opt server 入口

高斯关，轻量 summary：

```bash
cd third_party/openpi
bash server_pi05_libero_vla_opt_default.sh --port 8003 --gpu 2
```

默认行为与 baseline 相同：开启 `torch.compile`，且不再由脚本强制关闭 autotune。
默认不启用 observe / dump；性能对比只看 client 日志中的 `policy_infer_ms`（来源于 `policy_timing.infer_ms`）。

对应 client：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt_full.sh --port 8003 --gpu 2
```

高斯开，轻量 summary：

```bash
cd third_party/openpi
bash server_pi05_libero_vla_opt_gauss.sh --port 8004 --gpu 1
```

对应 client：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt_full.sh --port 8004 --gpu 1
```

高斯关，trace + dump：

```bash
cd third_party/openpi
bash server_pi05_libero_vla_opt_default_dump.sh --port 8005 --gpu 0
```

这个脚本现在是一键模式：

- 自动后台启动 dump server
- 自动运行 smoke client
- 自动执行 `render_png` 和 `pruning_stats`
- 结束后自动清理 server

高斯开，trace + dump：

```bash
cd third_party/openpi
bash server_pi05_libero_vla_opt_gauss_dump.sh --port 8006 --gpu 0
```

这个脚本同样是一键模式，不需要再手动单独启动 client。

## 3) 其他 client 入口

单个 episode / 快速 smoke：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt_smoke.sh --port 8003 --gpu 0
```

完整成功率评测：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt_full.sh --port 8003 --gpu 0
```

通用 client：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt.sh --port 8003 --suite libero_spatial --trials 20 --gpu 0
```

## 4) dump 在哪里

如果 server 用的是 `*_dump.sh`，启动后会打印：

```text
observe_dump_dir: ...
```

后续离线分析直接用这个目录。

`*_dump.sh` 一键模式会在 client 结束后自动继续执行：

- `python -m vla_opt.observe.render_png --run-dir <observe_dump_dir>`
- `python -m vla_opt.observe.pruning_stats --run-dir <observe_dump_dir>`

合成 overlay 视频：

```bash
cd third_party/openpi
.venv/bin/python scripts/render_observe_video.py \
  --run-dir <observe_dump_dir> \
  --fps 1
```

默认会同时输出 `scores_overlay` 和 `keep_mask_overlay` 两套视频，每个 episode 一个视频，帧内为 `view0 | view1` 并排。

性能对比口径：

- 只看 `policy_timing.infer_ms`
- 默认入口不启用 observe / dump
- `*_dump.sh` 只用于调试，不和默认性能口径混比

不同方案默认目录会自动区分：

- `server_pi05_libero_vla_opt_default.sh` -> `runs/vla_opt_default/...`
- `server_pi05_libero_vla_opt_gauss.sh` -> `runs/vla_opt_gauss/...`
- `client_libero_eval_vla_opt_smoke.sh` -> `runs/libero/videos/vla_opt_smoke/...`
- `client_libero_eval_vla_opt_full.sh` -> `runs/libero/videos/vla_opt_full/...`
