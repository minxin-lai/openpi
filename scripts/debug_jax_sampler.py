#!/usr/bin/env python3
"""
Debug tool: run a single JAX Pi0 policy forward pass on a saved
observation and dump model_out_normalized (stage=7) to .npz for
alignment with realtime-vla.

Usage (example for pi0_yuanluo_delta):

  uv run scripts/debug_jax_sampler.py \
    --config pi0_yuanluo_delta \
    --ckpt_dir /path/to/llly_usbinsertv2_1028_imgah32delta_ext/29999 \
    --obs_npy /path/to/obs_step0.npy \
    --noise_npy /path/to/noise_step0.npy \
    --output jax_model_out_normalized_step0.npz

The obs_npy file should contain a dict-compatible object with keys
matching the websocket observation, e.g.:
  {
    "observation.state": np.ndarray (27,),
    "observation.images.head_camera": np.ndarray (H,W,3),
    "observation.images.wrist_left_camera": np.ndarray (H,W,3),
    "observation.images.gelsight_left": np.ndarray (H,W,3),
    "prompt": "..."
  }

noise_npy is optional; if provided it should be a float32 array of
shape (T, 32) corresponding to the diffusion noise in action space.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def load_obs(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "obs" in data:
            return dict(data["obs"].item())
        # Fallback: treat all keys as top-level entries
        return {k: data[k] for k in data.files}
    else:
        # Assume .npy with a dict-like object
        obj = np.load(path, allow_pickle=True).item()
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict in {path}, got {type(obj)}")
        return obj


def load_noise(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.load(Path(path), allow_pickle=False)
    return np.asarray(arr, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Train config name, e.g. pi0_yuanluo_delta")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="JAX checkpoint dir (e.g. .../29999)")
    parser.add_argument("--obs_npy", type=str, required=True, help="Path to saved observation (.npy or .npz)")
    parser.add_argument("--noise_npy", type=str, default=None, help="Optional path to saved noise array (.npy)")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path for model_out_normalized")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, args.ckpt_dir, default_prompt=None)
    logging.info("Loaded JAX policy config=%s from %s", args.config, args.ckpt_dir)

    obs = load_obs(Path(args.obs_npy))

    # Prepare noise: use provided file if given, otherwise generate a deterministic one
    # based on the model's action_horizon and action_dim for reproducible comparison.
    if args.noise_npy is not None:
        noise = load_noise(Path(args.noise_npy))
        logging.info("Loaded noise from %s", args.noise_npy)
    else:
        action_horizon = getattr(cfg.model, "action_horizon", 32)
        action_dim = getattr(cfg.model, "action_dim", 32)
        rng = np.random.RandomState(0)
        noise = rng.randn(action_horizon, action_dim).astype(np.float32)
        logging.info(
            "Generated deterministic noise with shape (%d, %d) using seed=0",
            action_horizon,
            action_dim,
        )

    # Monkeypatch _log_alignment_snapshot to capture model_out_normalized.
    captured: dict[str, Any] = {}
    original_log = _policy._log_alignment_snapshot

    def _hook(tag: str, data: dict[str, Any], *, step: int | None = None) -> None:  # type: ignore[override]
        nonlocal captured
        if tag == "jax_model_out_normalized":
            # We only care about the first call (step 0) for this script.
            if not captured:
                captured = {
                    "state": np.asarray(data.get("state")),
                    "actions": np.asarray(data.get("actions")),
                }
        # Always call the original logger to keep normal behaviour.
        original_log(tag, data, step=step)

    _policy._log_alignment_snapshot = _hook  # type: ignore[assignment]

    logging.info("Running JAX policy.infer once for debugging...")
    try:
        _ = policy.infer(obs, noise=noise)
    finally:
        # Restore original logger to avoid side effects.
        _policy._log_alignment_snapshot = original_log  # type: ignore[assignment]

    if not captured:
        logging.error("No jax_model_out_normalized snapshot was captured. "
                      "Ensure OPENPI_ALIGN_DEBUG is not disabling logging.")
        return

    out_path = Path(args.output)
    np.savez_compressed(out_path, state=captured["state"], actions=captured["actions"])
    logging.info("Saved JAX model_out_normalized snapshot to %s", out_path)


if __name__ == "__main__":
    main()
