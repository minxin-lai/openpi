#!/usr/bin/env python3
"""
Minimal realtime‑vla inference runner.

Loads a converted checkpoint (.pkl produced by third_party/realtime-vla/convert_from_jax.py)
and runs a few inference steps on random inputs to verify the pipeline.

Usage (copy/paste):

  uv run scripts/infer_realtime_vla.py \
    --checkpoint /home/shared_workspace/USER/converted_checkpoint.pkl \
    --num_views 2 \
    --chunk_size 50 \
    --steps 1 \
    --warmup 0

Notes:
- Requires a CUDA-enabled environment. Images/states/noise are created on CUDA as bfloat16.
- Replace the random inputs with your real data when available.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pickle
import torch


def _add_realtime_vla_to_path() -> None:
    here = Path(__file__).resolve().parent
    rt_vla_dir = here.parent / "third_party" / "realtime-vla"
    sys.path.insert(0, str(rt_vla_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime-vla inference with converted checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to converted_checkpoint.pkl")
    parser.add_argument("--num_views", type=int, default=2, help="Number of camera views (1-3)")
    parser.add_argument("--chunk_size", type=int, default=50, help="Trajectory length (50-63 recommended)")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup steps (not timed)")
    parser.add_argument("--steps", type=int, default=1, help="Number of timed inference steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by realtime-vla inference. No GPU detected.")

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Late import after sys.path modification
    _add_realtime_vla_to_path()
    from pi0_infer import Pi0Inference  # type: ignore

    # Load converted weights (PyTorch tensors dict expected)
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)

    num_views = int(args.num_views)
    chunk_size = int(args.chunk_size)
    infer = Pi0Inference(checkpoint, num_views=num_views, chunk_size=chunk_size)

    # Prepare random inputs on CUDA (bfloat16)
    images = torch.rand(num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")  # [0,1]
    state = torch.zeros(32, dtype=torch.bfloat16, device="cuda")
    noise = torch.randn(chunk_size, 32, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(max(0, int(args.warmup))):
        _ = infer.forward(images, state, noise)
        torch.cuda.synchronize()

    # Timed steps
    times_ms: list[float] = []
    for _ in range(max(1, int(args.steps))):
        t0 = time.time()
        _ = infer.forward(images, state, noise)
        torch.cuda.synchronize()
        t1 = time.time()
        times_ms.append((t1 - t0) * 1000.0)

    print(f"views {num_views} chunk_size {chunk_size}")
    print(f"runs {len(times_ms)} median time per inference: {sorted(times_ms)[len(times_ms)//2]:.2f} ms")


if __name__ == "__main__":
    main()

