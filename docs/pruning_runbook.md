# OpenPI Pruning Runbook

## Source Of Truth

- canonical matrix: [`docs/exp_record.md`](/workspace/laiminxin/post_gauss_attn/third_party/openpi/docs/exp_record.md)
- method comparison: [`docs/openpi_pruning_methods_comparison.md`](/workspace/laiminxin/post_gauss_attn/third_party/openpi/docs/openpi_pruning_methods_comparison.md)
- ready-to-run commands: [`tools/libero_recipes.sh`](/workspace/laiminxin/post_gauss_attn/third_party/openpi/tools/libero_recipes.sh)

## Naming Rule

- `docs/exp_record.md` 里的 `variant` 是唯一运行主键。
- 评测 / 推理时：
  - `--run-tag <variant>`
- 训练时：
  - `--exp-name <variant>`
- run 产物目录统一按 `variant` 组织：
  - 普通评测：`runs/<variant>/<step>_<timestamp>/`
  - dump / 可视化：`runs/<variant>/viz_<step>_<timestamp>/`

## Entrypoints

- server: `bash tools/serve_pi05_libero.sh`
- client: `bash tools/client_libero.sh`
- one-click eval: `bash tools/run_libero.sh`
- one-click dump: `bash tools/run_libero_dump.sh`
- baseline train: `bash tools/train_pi05_baseline.sh`
- pruning train: `bash tools/train_pi05_experiment.sh`

统一约束：

- pruning / experiment 配置统一通过 `--opt-config <yaml>`
- client 默认评测口径应显式传：
  - `--suite libero_spatial`
  - `--trials 50`
- dump 常用口径应显式传：
  - `--trials 1`

## Canonical Matrix

下表直接对齐 [`docs/exp_record.md`](/workspace/laiminxin/post_gauss_attn/third_party/openpi/docs/exp_record.md)。运行时不要自造名字，直接复用 `variant`。

| family | variant | checkpoint | opt-config |
| --- | --- | --- | --- |
| `baseline` | `baseline` | `checkpoints/pi05_libero_spatial/pi05_baseline/29999` | `-` |
| `pruning_inside_encoder` | `inside_t64` | `checkpoints/pi05_libero_spatial/inside_t64/59999` | `config/pruning/inside_t64.yaml` |
| `pruning_inside_encoder` | `inside_t64_gauss` | `checkpoints/pi05_libero_spatial/inside_t64/59999` | `config/pruning/inside_t64_gauss.yaml` |
| `pruning_inside_encoder` | `inside_t128` | `checkpoints/pi05_libero_spatial/inside_t128/59999` | `config/pruning/inside_t128.yaml` |
| `pruning_inside_encoder` | `inside_t128_gauss` | `checkpoints/pi05_libero_spatial/inside_t128/59999` | `config/pruning/inside_t128_gauss.yaml` |
| `pruning_after_encoder` | `post_t64` | `checkpoints/pi05_libero_spatial/post_t64/29999` | `config/pruning/post_t64.yaml` |
| `pruning_after_encoder` | `post_t64_gauss` | `checkpoints/pi05_libero_spatial/post_t64/29999` | `config/pruning/post_t64_gauss.yaml` |
| `pruning_after_encoder` | `post_t128` | `checkpoints/pi05_libero_spatial/post_t128/59999` | `config/pruning/post_t128.yaml` |
| `pruning_after_encoder` | `post_t128_gauss` | `checkpoints/pi05_libero_spatial/post_t128/59999` | `config/pruning/post_t128_gauss.yaml` |

## Templates

baseline one-click eval:

```bash
bash tools/run_libero.sh \
  --run-tag baseline \
  --ckpt-dir checkpoints/pi05_libero_spatial/pi05_baseline/29999 \
  --policy-config pi05_libero_spatial \
  --host 127.0.0.1 \
  --suite libero_spatial \
  --trials 50 \
  --port 9000 \
  --gpu 0
```

pruning one-click eval:

```bash
bash tools/run_libero.sh \
  --run-tag <variant> \
  --ckpt-dir <checkpoint> \
  --policy-config pi05_libero_spatial \
  --opt-config <yaml> \
  --host 127.0.0.1 \
  --suite libero_spatial \
  --trials 50 \
  --port <port> \
  --gpu <gpu>
```

pruning one-click dump:

```bash
bash tools/run_libero_dump.sh \
  --run-tag <variant> \
  --ckpt-dir <checkpoint> \
  --policy-config pi05_libero_spatial \
  --opt-config <yaml> \
  --observe-config /workspace/laiminxin/post_gauss_attn/configs/observe/infer_debug.json \
  --host 127.0.0.1 \
  --suite libero_spatial \
  --trials 1 \
  --port <port> \
  --gpu <gpu>
```

pruning train:

```bash
bash tools/train_pi05_experiment.sh \
  --opt-config <yaml> \
  --policy-config <policy-config> \
  --exp-name <variant> \
  --gpus <gpu-list> \
  --base-ckpt <base-ckpt> \
  --data-repo-id <dataset> \
  --num-train-steps <steps>
```

## Notes

- `cross_attn_post_t64_gauss` 目前仍是单独实验 variant，不在默认 canonical sweep 里；命名规则不变，仍然使用：
  - `--run-tag cross_attn_post_t64_gauss`
  - `--exp-name cross_attn_post_t64_gauss`
- observe / dump 输出目录仍由 `run_libero_dump.sh` 自动派生。
- 如果只是找现成命令，优先直接看 [`tools/libero_recipes.sh`](/workspace/laiminxin/post_gauss_attn/third_party/openpi/tools/libero_recipes.sh)。
