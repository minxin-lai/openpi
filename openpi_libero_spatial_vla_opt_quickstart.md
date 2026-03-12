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

对应 client：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt_full.sh --port 8005 --gpu 0
```

高斯开，trace + dump：

```bash
cd third_party/openpi
bash server_pi05_libero_vla_opt_gauss_dump.sh --port 8006 --gpu 0
```

对应 client：

```bash
cd third_party/openpi
bash client_libero_eval_vla_opt_full.sh --port 8006 --gpu 0
```

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

不同方案默认目录会自动区分：

- `server_pi05_libero_vla_opt_default.sh` -> `runs/vla_opt_default/...`
- `server_pi05_libero_vla_opt_gauss.sh` -> `runs/vla_opt_gauss/...`
- `client_libero_eval_vla_opt_smoke.sh` -> `runs/libero/videos/vla_opt_smoke/...`
- `client_libero_eval_vla_opt_full.sh` -> `runs/libero/videos/vla_opt_full/...`
