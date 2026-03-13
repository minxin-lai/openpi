# Run `vla_opt_pi05_stage_a_ste` In A Legacy Encoder-Layer-Prune Worktree

This note is for the legacy checkpoint:

```text
checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/29999
```

This checkpoint uses the older Pi0.5 VLA-OPT pruning path, not the newer `post_encoder_prune` flow.
Use a separate worktree because it predates later wrapper changes.

## Names To Use

- Legacy tag in mono-repo: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`
- Legacy tag in `third_party/openpi`: `legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`
- Recommended legacy worktree root: `/workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune`

## What This Legacy Setup Means

This legacy checkpoint should be treated as:

- VLA-OPT VE-FiLM enabled
- older STE pruning path
- encoder-layer style pruning integration
- not the current `post_encoder_prune` variant

## Create The Legacy Worktree

Create the mono-repo worktree:

```bash
git -C /workspace/laiminxin/vla-opt worktree add \
  /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune \
  legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune
```

Create the matching `openpi` worktree inside it:

```bash
rmdir /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune/third_party/openpi

git -C /workspace/laiminxin/vla-opt/third_party/openpi worktree add \
  /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune/third_party/openpi \
  legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune
```

If you already created `/workspace/laiminxin/vla-opt-openpi-old`, it is still usable. The path above is the clearer name going forward.

## Prepare The Legacy OpenPI Runtime

Install the `transformers_replace` overlay into that worktree's venv:

```bash
cp -r \
  /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune/third_party/openpi/src/openpi/models_pytorch/transformers_replace/* \
  /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune/third_party/openpi/.venv/lib/python3.11/site-packages/transformers/
```

## Start The Legacy Server

Run from the legacy `openpi` worktree:

```bash
cd /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune/third_party/openpi

CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py \
  --env LIBERO \
  --port 8013 \
  --vla-opt-ve-film \
  --vla-opt-ve-film-num-blocks 4 \
  --vla-opt-ste-prune \
  --vla-opt-ste-prune-k 64 \
  --vla-opt-ste-prune-stage gather \
  --vla-opt-ste-prune-tau 1.0 \
  policy:checkpoint \
  --policy.config pi05_libero_spatial \
  --policy.dir /workspace/laiminxin/vla-opt/third_party/openpi/checkpoints/pi05_libero_spatial/vla_opt_pi05_stage_a_ste/29999
```

Notes:

- Keep `--policy.dir` on one line. Do not split the checkpoint path across lines.
- Change `CUDA_VISIBLE_DEVICES=0` if you want another GPU.
- Change `--port 8013` if that port is occupied.
- Keep the wrapper flags aligned with the checkpoint.

## Test The Legacy Server

Prefer running the client from the same legacy worktree.

The older `client_libero_eval.sh` has hard-coded values, so the more reliable method is to run `examples/libero/main.py` directly.

Create the LIBERO client venv first if needed:

```bash
cd /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune/third_party/openpi
python3.8 -m venv examples/libero/.venv
source examples/libero/.venv/bin/activate
```

Install the minimum client dependencies:

```bash
pip install --upgrade pip setuptools wheel

pip install \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  torch==1.11.0+cu113 \
  torchvision==0.12.0+cu113 \
  torchaudio==0.11.0+cu113

pip install \
  imageio[ffmpeg] \
  numpy==1.22.4 \
  tqdm \
  tyro==0.9.2 \
  PyYAML \
  opencv-python==4.6.0.66 \
  matplotlib==3.5.3 \
  mujoco==3.2.3 \
  robosuite==1.4.1
```

Run the client:

```bash
cd /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune/third_party/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"

CUDA_VISIBLE_DEVICES=0 python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8013 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 \
  --args.video-out-path runs/libero/videos/libero_spatial_$(date +%Y%m%d_%H%M%S)
```

Quick checks:

```bash
lsof -nP -iTCP:8013 -sTCP:LISTEN
pgrep -af "serve_policy.py --env LIBERO --port 8013"
```

## Stop The Legacy Server

```bash
pkill -f "serve_policy.py --env LIBERO --port 8013"
```

## Continue Developing On Current `HEAD`

Your main workspace is still separate:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi
git rev-parse --short HEAD
```

Using a legacy worktree does not switch your main worktree away from current `HEAD`.

## Remove The Legacy Worktree

Stop the legacy server first, then remove both worktrees:

```bash
git -C /workspace/laiminxin/vla-opt/third_party/openpi worktree remove \
  /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune/third_party/openpi

git -C /workspace/laiminxin/vla-opt worktree remove \
  /workspace/laiminxin/vla-opt-legacy-vla_opt_pi05_stage_a_ste_encoder_layer_prune
```
