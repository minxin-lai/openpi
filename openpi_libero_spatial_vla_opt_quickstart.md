# OpenPI Pi0.5（LIBERO Spatial）训练 + 评测（VLA-OPT / Baseline）

本文覆盖两条最短闭环：

1) **VLA-OPT 版**：`finetune_pi05_ve_film_prune.sh`（训练）→ `server_pi05_libero_vla_opt.sh`（启 server）→ `client_libero_eval.sh`（评测）
2) **Baseline 版（无 FiLM/STE）**：`finetune_pi05_baseline.sh`（训练）→ `server_pi05_libero_baseline.sh`（启 server）→ `client_libero_eval.sh`（评测）

---

## 这些脚本分别做什么

| 脚本 | 作用 | 什么时候用 |
|---|---|---|
| `third_party/openpi/finetune_pi05_ve_film_prune.sh` | **训练/继续训练**：Pi0.5 + Vision Encoder (VE) FiLM + VE STE pruning（auto 两阶段） | 你要产出新的 `model.safetensors` checkpoint |
| `third_party/openpi/finetune_pi05_baseline.sh` | **训练/继续训练（baseline）**：Pi0.5（不含 FiLM/STE） | 你要做对照实验（baseline） |
| `third_party/openpi/server_pi05_libero_vla_opt.sh` | **推理 server（带 VLA-OPT wrapper）**：加载你训练出的 PyTorch checkpoint，并开启 `--vla-opt-*` | 你要评测/推理你的 VLA-OPT checkpoint |
| `third_party/openpi/server_pi05_libero_baseline.sh` | **推理 server（baseline）**：加载你训练出的 PyTorch checkpoint（不启用 `--vla-opt-*`） | 你要评测/推理 baseline checkpoint |
| `third_party/openpi/server_libero.sh` | **推理 server（通用模板）**：可选 tracing，可选通过 env 打开 `--vla-opt-*` | 你想跑 baseline 或自定义更多 server 参数 |
| `third_party/openpi/client_libero_eval.sh` | **仿真 client（评测端）**：启动 LIBERO 仿真，连到 server，输出视频 | 你要真正跑 episode 看成功率/视频 |

---

## 0) 一次性环境准备（server 侧）

在 `third_party/openpi/` 执行：

```bash
uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

> 上面这步是 PyTorch serve 所必需（transformers patch + openpi 可编辑安装）。

---

## 1) 数据准备（LIBERO Spatial, LeRobot 格式）

本仓库默认使用：

- 数据集目录：`/workspace/laiminxin/datasets/lerobot_datasets/libero_spatial`

确保 norm stats 存在（否则推理会因为归一化统计缺失而报错）：

```bash
cd third_party/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_libero_spatial
ls /workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json
```

---

## 2) 训练（Vision Encoder VE: FiLM + STE，产出 PyTorch checkpoint）

直接跑（脚本内已写死默认 config/exp/路径；需要改就编辑脚本顶部变量）：

```bash
cd third_party/openpi
bash finetune_pi05_ve_film_prune.sh
```

### 2.1 训练日志（只记录剪枝统计，不会 dump tracer `.pt`）

训练脚本会把 stdout/stderr 写到（脚本启动时也会打印绝对路径）：

- `third_party/openpi/checkpoints/${CONFIG}/${EXP}/train.log`

训练阶段 **不会** 生成 tracer 的 `runs/**/dumps/*.pt`（那些是推理 server 开 tracer 才会写的）。  
你关心的剪枝相关信息会以日志字段形式输出（例如 `ste_prune_*`）。

快速筛选剪枝日志：

```bash
cd third_party/openpi
rg -n "ste_prune" "checkpoints/${CONFIG}/${EXP}/train.log"
```

训练产物目录（示例）：

- `third_party/openpi/checkpoints/pi05_libero_spatial/vla_opt_pi05_ve_film_prune/30000/model.safetensors`

评测时 **要把 `CKPT_DIR` 指到某个 step 子目录**（如 `.../30000`）。

---

## 2b) 训练 baseline（不含 FiLM/STE）

```bash
cd third_party/openpi
bash finetune_pi05_baseline.sh
```

---

## 3) 启动推理 server

### 3.1 启动 VLA-OPT server（加载你的 VLA-OPT checkpoint）

编辑 `third_party/openpi/server_pi05_libero_vla_opt.sh` 顶部变量：

- `CKPT_DIR=.../30000`
- `POLICY_CONFIG=pi05_libero_spatial`
- `PORT/GPU`

然后启动：

```bash
cd third_party/openpi
bash server_pi05_libero_vla_opt.sh
```

这一步会：

- 启动 WebSocket policy server（默认端口 `8002`）
- 启用 `--vla-opt-stage-a` + `--vla-opt-ste-prune`（与训练一致）

### 3.1.1 server 日志与可选 tracer

server 日志默认会写到 vla-opt repo 根目录下：

- `vla-opt/runs/openpi_pi05_libero_server_*.log`

你也可以用环境变量改日志路径：

```bash
SERVER_LOG=/tmp/openpi_server.log bash server_pi05_libero_vla_opt.sh
```

推理时 tracer 是 **可选** 的（默认不开，不会写 `.pt` dump）。开启方式：

```bash
TRACE=1 bash server_pi05_libero_vla_opt.sh
```

开启后 trace 默认输出到：

- `vla-opt/runs/openpi_pi05_libero_trace_*/dumps/*.pt`
- `vla-opt/runs/openpi_pi05_libero_trace_*/images/*`

常用可选环境变量：

- `TRACE_OUT_DIR=/abs/path`（不设则默认放到 `vla-opt/runs/openpi_pi05_libero_trace_*`）
- `TRACE_ATTN_LAYERS=0,8,16`（空字符串表示只取 last layer）
- `TRACE_EVERY_N=1`、`TRACE_MAX_DUMPS=200`

---

### 3.2 启动 baseline server（不启用 VLA-OPT）

编辑 `third_party/openpi/server_pi05_libero_baseline.sh` 顶部变量（尤其 `CKPT_DIR`），然后启动：

```bash
cd third_party/openpi
bash server_pi05_libero_baseline.sh
```

---

## 4) 启动仿真评测 client（跑 episode + 导出视频）

### 4.1 一次性：创建 client venv

```bash
cd third_party/openpi
uv venv --python 3.8 examples/libero/.venv
```

依赖安装（更完整步骤见 `third_party/openpi/docs/libero_pytorch_eval.md`）。

### 4.2 每次评测：运行 client

编辑 `third_party/openpi/client_libero_eval.sh` 顶部变量：

- `HOST/PORT`（要与 server 一致）
- `TASK_SUITE=libero_spatial`（或 `libero_object/libero_goal/libero_10`）
- `TRIALS`
- `CLIENT_GPU`

然后运行：

```bash
cd third_party/openpi
bash client_libero_eval.sh
```

视频输出默认在：

- `third_party/openpi/runs/libero/videos/...`

---

## 常见报错（只写最常见的两个）

1) `FileNotFoundError: .../assets/.../norm_stats.json`

- 说明：norm stats 不存在或 policy config 不匹配。
- 对于 `pi05_libero_spatial`：确认存在 `.../libero_spatial/norm_stats.json`，并且 server 用的是 `--policy.config pi05_libero_spatial`。

2) client 连接不上 server

- 检查 `HOST/PORT` 是否一致、server 终端是否仍在运行、端口是否被占用。
