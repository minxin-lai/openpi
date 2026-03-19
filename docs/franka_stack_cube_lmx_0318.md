# Franka Stack Cube LMX 0318

This note records the final commands for training `pi05_franka_stack_cube_lmx_0318` with PyTorch full fine-tuning on GPUs `0,1`.

## Config

- Config name: `pi05_franka_stack_cube_lmx_0318`
- Dataset: `/workspace/laiminxin/datasets/franka/stack_cube_lmx_0318`
- Base PyTorch weights: `/workspace/laiminxin/models/pi05_base_pytorch`
- Training entrypoint: `scripts/train_pytorch.py`

## Compute Norm Stats

Run from `/workspace/laiminxin/vla-opt/third_party/openpi`:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

UV_CACHE_DIR=/tmp/uv-cache \
uv run scripts/compute_norm_stats.py --config-name pi05_franka_stack_cube_lmx_0318
```

## Train On GPU 0,1

Run from `/workspace/laiminxin/vla-opt/third_party/openpi`:

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi

CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi05_franka_stack_cube_lmx_0318 \
  --exp_name franka_stack_cube_pi05
```

## Checkpoints

Training outputs will be written under:

```bash
./checkpoints/pi05_franka_stack_cube_lmx_0318/franka_stack_cube_pi05
```

## Notes

- This config uses `prompt_from_task=True`, so language comes from the dataset task metadata.
- The Franka data mapping keeps only `head_camera`, `wrist_left_camera`, `state`, `action`, and `prompt`.
- `compute_norm_stats.py` and `train_pytorch.py` already inject the mono-repo `src` path automatically, so no extra `PYTHONPATH` is required.
