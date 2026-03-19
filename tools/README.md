# Tools

`tools/` is the only supported shell entrypoint directory for this repo.

- Use `serve_pi05_libero.sh` + `eval_libero.sh` for step-by-step debugging.
- Use `run_libero_dump.sh` for a single dump run.
- Use `train_pi05_baseline.sh` / `train_pi05_experiment.sh` for training.

## Scripts

| Script | Purpose | Notes |
| --- | --- | --- |
| `tools/serve_pi05_libero.sh` | Start a LIBERO server for Pi0.5 | Baseline when `--opt-config` is omitted; experiment mode when provided |
| `tools/eval_libero.sh` | Run the LIBERO eval client | Works with either baseline or experiment server |
| `tools/run_libero_dump.sh` | Start server, run eval, render dump outputs, aggregate stats | Requires `--opt-config` |
| `tools/train_pi05_baseline.sh` | Start baseline training | Self-contained baseline training entrypoint |
| `tools/train_pi05_experiment.sh` | Start experiment training | Requires `--opt-config` |

## Examples

### 1. Baseline server + eval

```bash
bash tools/serve_pi05_libero.sh \
  --ckpt-dir checkpoints/pi05_libero_spatial/pi05_baseline/30000 \
  --policy-config pi05_libero_spatial \
  --port 8003 \
  --gpu 0 \
  --run-tag pi05_baseline_eval
```

```bash
bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag pi05_baseline_eval
```

### 2. Experiment server + eval

```bash
bash tools/serve_pi05_libero.sh \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder.yaml \
  --port 8003 \
  --gpu 0 \
  --run-tag pi05_post_encoder_eval
```

```bash
bash tools/eval_libero.sh \
  --host 127.0.0.1 \
  --port 8003 \
  --suite libero_spatial \
  --trials 50 \
  --gpu 1 \
  --run-tag pi05_post_encoder_eval
```

### 3. Single dump run

```bash
bash tools/run_libero_dump.sh \
  --ckpt-dir checkpoints/pi05_libero_spatial/vla_opt_pi05_pruning_after_encoder_t64/29999 \
  --policy-config pi05_libero_spatial \
  --opt-config config/pruning/post_encoder.yaml \
  --port 8003 \
  --gpu 0 \
  --trials 50 \
  --run-tag pi05_post_encoder_dump
```

### 4. Training

```bash
bash tools/train_pi05_baseline.sh
```

```bash
bash tools/train_pi05_experiment.sh \
  --opt-config config/pruning/post_encoder.yaml
```

## Parameters

### Common runtime parameters

- `--ckpt-dir`
  Path to the checkpoint directory. The directory must contain `model.safetensors`.
  Supported by: `serve_pi05_libero.sh`, `run_libero_dump.sh`.

- `--policy-config`
  Base OpenPI policy config name, for example `pi05_libero_spatial`.
  Supported by: `serve_pi05_libero.sh`, `run_libero_dump.sh`.

- `--opt-config`
  Extra experiment/optimization config file. Omit it for baseline behavior.
  Supported by: `serve_pi05_libero.sh`, `run_libero_dump.sh`, `train_pi05_experiment.sh`.

- `--port`
  Server port.
  Supported by: `serve_pi05_libero.sh`, `eval_libero.sh`, `run_libero_dump.sh`.
  Default: `8003`.

- `--host`
  Server host used by the eval client.
  Supported by: `eval_libero.sh`, `run_libero_dump.sh`.
  Default: `127.0.0.1`.

- `--gpu`
  CUDA device id for single-GPU serve/eval workflows.
  Supported by: `serve_pi05_libero.sh`, `eval_libero.sh`, `run_libero_dump.sh`.
  Default: `0`.

- `--run-tag`
  Run label used to group logs, videos, and dump outputs.
  Supported by: `serve_pi05_libero.sh`, `eval_libero.sh`, `run_libero_dump.sh`.
  If omitted, each script falls back to its own default.

### Eval parameters

- `--suite`
  LIBERO task suite name.
  Supported by: `eval_libero.sh`, `run_libero_dump.sh`.
  Default: `libero_spatial`.

- `--trials`
  Number of trials per task.
  Supported by: `eval_libero.sh`, `run_libero_dump.sh`.
  Default: `20`.

### Output parameters

- `--log`
  Log path for a single script.
  Supported by: `serve_pi05_libero.sh`, `eval_libero.sh`.

- `--video-out`
  Eval video output directory.
  Supported by: `eval_libero.sh`, `run_libero_dump.sh`.

- `--observe-output-dir`
  Explicit dump output directory used by the server.
  Supported by: `serve_pi05_libero.sh`, `run_libero_dump.sh`.

- `--launcher-log`
  Log file for the one-click runner itself.
  Supported by: `run_libero_dump.sh`.

- `--server-log`
  Dedicated server log file for the one-click runner.
  Supported by: `run_libero_dump.sh`.

- `--client-log`
  Dedicated client log file for the one-click runner.
  Supported by: `run_libero_dump.sh`.

### Training parameters

- `--opt-config`
  Experiment config file forwarded to training as the extra optimization config.
  Supported by: `train_pi05_experiment.sh`.

## Notes

- `--policy-config` selects the base policy/task config.
- `--opt-config` selects optional experiment behavior on top of the base policy.
- `run_libero_dump.sh` is for experiment runs with dump output; baseline debug runs should use `serve_pi05_libero.sh` + `eval_libero.sh`.
- Historical root-level launcher scripts have been removed.
