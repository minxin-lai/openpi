# LIBERO (PyTorch) 评测完整流程（含转换 JAX → PyTorch）

本仓库的 `run_libero_server.sh` / `run_libero_eval_client.sh` 走的是 **PyTorch 权重**（目录里有 `model.safetensors`）+ WebSocket server/client 的评测方式。  
如果你直接用 base 权重（例如 `pi05_base`）跑 LIBERO，成功率很可能几乎全是 `False`；LIBERO 评测需要使用 **`pi05_libero` 的 finetuned checkpoint**，并将其转换成 PyTorch 格式后再跑。

## 0. 前置条件

在 `third_party/openpi/` 目录下执行。

1) 安装依赖（server 侧用 uv 环境）：

```bash
uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

2) 确认 transformers 版本（PyTorch 支持要求 4.53.2）：

```bash
uv run python -c "import transformers; print(transformers.__version__)"
```

3) 应用 openpi 的 transformers patch（PyTorch 支持所需）：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

> 注意：这会覆盖 `.venv` 内的 transformers 文件；如果你用 uv hardlink 模式可能影响 uv cache，详见 openpi README 的 WARNING。

## 1) 先下载（缓存）JAX 的 `pi05_libero` checkpoint

openpi 的默认 LIBERO policy 会自动从 `gs://openpi-assets/checkpoints/pi05_libero` 下载并缓存到本地（默认 `~/.cache/openpi`，可用 `OPENPI_DATA_HOME` 自定义）。

最简单的触发下载方式：先用默认 LIBERO server 跑一下（不需要一直跑，看到开始加载即可 `Ctrl+C`）：

```bash
uv run scripts/serve_policy.py --env LIBERO --port 8002
```

缓存目录通常形如：

```bash
ls ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

## 2) 选择要转换的 checkpoint 目录（必须包含 `params/`）

转换脚本要求 `--checkpoint_dir` 指向一个目录，并且该目录下必须存在 `params/`（JAX orbax checkpoint）。

`pi05_libero` 的下载缓存通常是这种结构（没有 step 子目录）：

```text
~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/
  assets/
  params/
```

此时直接用这个目录即可：

```bash
ls ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/params/_METADATA
```

如果你转换的是你自己训练出来的 checkpoint，可能是这种结构（有 step 子目录）：

```text
checkpoints/<config>/<exp>/<step>/
  params/
checkpoints/<config>/<exp>/assets/
```

此时把 `--checkpoint_dir` 指向 `<step>` 目录。

## 3) 转换 JAX → PyTorch（生成 `model.safetensors`）

执行转换（`--config_name` 必须与 checkpoint 对应，这里是 `pi05_libero`）：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --config_name pi05_libero \
  --output_path /workspace/laiminxin/models/pi05_libero_pytorch
```

验证转换产物必须包含：

```bash
ls /workspace/laiminxin/models/pi05_libero_pytorch
# 期望看到：model.safetensors  assets/  config.json
```

> 说明：转换脚本会自动从 `checkpoint_dir/assets/` 或 `checkpoint_dir` 的上一级目录里的 `assets/` 拷贝 norm stats 到输出目录。

## 4) 配置并启动 PyTorch policy server

编辑 `run_libero_server.sh`，把 `MODEL_PATH` 改为上一步输出目录：

```bash
# third_party/openpi/run_libero_server.sh
MODEL_PATH="/workspace/laiminxin/models/pi05_libero_pytorch"
```

然后启动 server：

```bash
bash run_libero_server.sh
```

默认端口在脚本里是 `8002`（`PORT=8002`）。保持该终端窗口运行。

## 5) 启动 LIBERO eval client（仿真端）

`run_libero_eval_client.sh` 会使用 `examples/libero/.venv`（Python 3.8）运行仿真。按脚本提示先创建好 venv：

```bash
uv venv --python 3.8 examples/libero/.venv
```

然后按 `examples/libero/README.md` 安装 client 侧依赖（脚本不会自动装依赖，建议手动执行一次）：

```bash
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
deactivate
```

最后运行评测 client：

```bash
bash run_libero_eval_client.sh
```

你可以在脚本里改这些参数：

- `TASK_SUITE`：`libero_spatial` / `libero_object` / `libero_goal` / `libero_10`
- `TRIALS`：每个任务的 trial 数
- `PORT`：必须与 server 一致

## 6) 常见问题排查

1) **全是 False / 成功率很低**
- 最常见原因：server 用的是 base checkpoint（例如 `pi05_base_pytorch`），不是 `pi05_libero` finetuned。
- 确认 server 的 `MODEL_PATH` 指向的是你转换出的 `pi05_libero_pytorch`，并且目录里有 `model.safetensors` + `assets/`。

2) **server 报 “只支持 PyTorch-loaded policies”**
- 你的 `--policy.dir` 指向的目录没有 `model.safetensors`，导致 openpi 走了 JAX 路径。

3) **下载 checkpoint 失败（网络/SSL 报错）**
- 先确认机器能访问 `gs://openpi-assets`；必要时设置代理/更换网络环境。
- 可尝试先在同机上用 `uv run scripts/serve_policy.py --env LIBERO` 重试触发下载。

4) **Orbax/TensorStore `OUT_OF_RANGE`（读取 params 时字节范围越界）**
- 一般是本地缓存的 checkpoint 文件不完整/已损坏（常见于下载中断）。
- 清理缓存后重试下载：
  ```bash
  rm -rf ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero*
  uv run scripts/serve_policy.py --env LIBERO --port 8002
  ```
