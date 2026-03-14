# OpenPI Pi0.5 LIBERO Spatial Quickstart

## 1) 一次性准备

```bash
cd third_party/openpi
uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
uv run scripts/compute_norm_stats.py --config-name pi05_libero_spatial
```

关键参数：
- `uv sync`: 安装当前仓库依赖
- `uv pip install -e .`: 以 editable 模式安装 `openpi`
- `compute_norm_stats.py`: 生成 LIBERO 推理需要的 `norm_stats.json`

检查：

```bash
ls /workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json
```

## 2) 只启动 server

baseline：

```bash
cd third_party/openpi
bash server_pi05_libero_baseline.sh --gpu 0 --port 8002
```

vla-opt，默认剪枝：

```bash
cd third_party/openpi
bash server_pi05_libero_vla_opt_default.sh --gpu 0 --port 8003
```

vla-opt，高斯平滑剪枝：

```bash
cd third_party/openpi
bash server_pi05_libero_vla_opt_gauss.sh --gpu 0 --port 8004
```

关键参数：
- `--gpu`: 使用哪张 GPU，对应 `CUDA_VISIBLE_DEVICES`
- `--port`: websocket server 监听端口；client 必须连同一个端口

## 3) 连接 server 跑评测

baseline：

```bash
cd third_party/openpi
bash client_libero_eval_baseline.sh --gpu 0 --port 8002 --trials 20
```

vla-opt，完整评测：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt_full.sh --gpu 0 --port 8003
```

vla-opt，快速 smoke：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt_smoke.sh --gpu 0 --port 8003
```

通用 client：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt.sh --gpu 0 --port 8003 --suite libero_spatial --trials 20
```

关键参数：
- `--port`: 要和 server 完全一致
- `--trials`: 每个 task 跑多少个 episode
- `--suite`: 默认 `libero_spatial`

## 4) 一键运行并导出 observe overlay

默认剪枝：

```bash
cd third_party/openpi
bash run_pi05_libero_vla_opt_default.sh --gpu 0 --port 8005 --run-tag debug_default
```

高斯平滑剪枝：

```bash
cd third_party/openpi
bash run_pi05_libero_vla_opt_gauss.sh --gpu 0 --port 8006 --run-tag debug_gauss
```

这两个脚本会自动执行：
- 启动 server
- 运行 smoke client
- 渲染 observe overlay PNG
- 聚合 pruning stats

关键参数：
- `--run-tag`: 本次运行的目录前缀，便于区分不同实验
- `--port`: 必须是空闲端口
- `--gpu`: server 和 client 都会使用这张 GPU

输出位置：
- server 日志：`runs/<run_tag>/server_*.log`
- client 日志：`runs/libero/logs/<run_tag>/...`
- rollout 视频：`runs/libero/videos/<run_tag>/...`
- observe 输出：脚本会打印 `observe_dump_dir: ...`

如需把 overlay PNG 合成视频：

```bash
cd third_party/openpi
.venv/bin/python scripts/render_observe_video.py --run-dir <observe_dump_dir> --fps 1
```
