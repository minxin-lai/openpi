# 方案附录: pruning_inside_encoder

本页只维护 `pruning_inside_encoder` 路径的当前可运行配置与命令，不再混合维护历史结果表。

若需要跨方案对照入口，请看 [`docs/exp_record.md`](/workspace/laiminxin/vla-opt/third_party/openpi/docs/exp_record.md)。

## 方法定义

- 剪枝位置：`SigLIP encoder` 中间层
- 配置模式：`legacy_inside_encoder`
- `gauss` 变体：仅在 `Top-K` 之前额外开启 score map 高斯平滑

## 统一 runtime 口径

- `OPENPI_TORCH_COMPILE=1`
- `OPENPI_TORCH_COMPILE_MODE=reduce-overhead`
- 不显式设置 `TRITON_AUTOTUNE` / `TORCHINDUCTOR_MAX_AUTOTUNE`
- 评测命令统一使用 `--suite libero_spatial --trials 50`

## Canonical Matrix

| variant | checkpoint | opt-config | keep_tokens_per_view | keep_ratio | gauss |
| --- | --- | --- | ---: | ---: | --- |
| `inside_t64` | `checkpoints/pi05_libero_spatial/inside_t64/59999` | `config/pruning/inside_t64.yaml` | 64 | 25% | off |
| `inside_t64_gauss` | `checkpoints/pi05_libero_spatial/inside_t64/59999` | `config/pruning/inside_t64_gauss.yaml` | 64 | 25% | on |
| `inside_t128` | `checkpoints/pi05_libero_spatial/inside_t128/59999` | `config/pruning/inside_t128.yaml` | 128 | 50% | off |
| `inside_t128_gauss` | `checkpoints/pi05_libero_spatial/inside_t128/59999` | `config/pruning/inside_t128_gauss.yaml` | 128 | 50% | on |

## Canonical Commands

### `inside_t64`

Server:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag inside_t64 \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t64.yaml \
  --port 8003 \
  --gpu 0
```

Client:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag inside_t64
```

### `inside_t64_gauss`

Server:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag inside_t64_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t64/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t64_gauss.yaml \
  --port 8004 \
  --gpu 0
```

Client:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8004 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag inside_t64_gauss
```

### `inside_t128`

Server:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag inside_t128 \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t128.yaml \
  --port 8005 \
  --gpu 0
```

Client:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8005 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag inside_t128
```

### `inside_t128_gauss`

Server:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_MODE=reduce-overhead \
bash tools/serve_pi05_libero.sh \
  --run-tag inside_t128_gauss \
  --ckpt-dir checkpoints/pi05_libero_spatial/inside_t128/59999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/inside_t128_gauss.yaml \
  --port 8006 \
  --gpu 0
```

Client:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8006 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag inside_t128_gauss
```

## Notes

- 当前 canonical 配置为 `inside_t64*.yaml` 和 `inside_t128*.yaml`；方法模式仍然是 `legacy_inside_encoder`。
- 本页不再把旧 worktree 路径、旧 run-tag 和历史成功率当作当前命令来源。
