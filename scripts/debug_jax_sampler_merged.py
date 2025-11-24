#!/usr/bin/env python3
"""
Debug tool: run a single JAX Pi0 policy forward pass using the
**merged JAX checkpoint** (LoRA 已经 merge 进 base 权重)，在保存的观测上跑一遍，
并抓取 model_out_normalized (stage=7) 到 .npz，方便和原始 LoRA / realtime-vla 对齐。

使用方式（针对 pi0_yuanluo_merged）:

  uv run scripts/debug_jax_sampler_merged.py \\
    --config pi0_yuanluo_merged \\
    --ckpt_dir /path/to/llly_usbinsertv2_1028_imgah32delta_ext_merged/29999 \\
    --obs_npy obs_step0.npy \\
    --noise_npy noise_step0.npy \\
    --output jax_merged_model_out_step0.npz

说明:
  - 权重通过 TrainConfig.weight_loader=CheckpointWeightLoader(...) 加载
    （即 pi0_yuanluo_merged 里配置的 merged params 路径），
    本脚本不会直接用 --ckpt_dir/params 做 restore_params。
  - --ckpt_dir 主要用于加载 norm_stats（与原 JAX server 一致，从 checkpoint/assets 读）。
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
        return {k: data[k] for k in data.files}
    else:
        obj = np.load(path, allow_pickle=True).item()
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict in {path}, got {type(obj)}")
        return obj


def load_noise(path: Path | None, *, action_horizon: int, action_dim: int) -> np.ndarray:
    if path is None:
        rng = np.random.RandomState(0)
        return rng.randn(action_horizon, action_dim).astype(np.float32)
    arr = np.load(Path(path), allow_pickle=False)
    return np.asarray(arr, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="pi0_yuanluo_merged",
        help="Train config name for merged JAX checkpoint, e.g. pi0_yuanluo_merged",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Merged JAX checkpoint dir (用于加载 assets/norm_stats，例如 ..._merged/29999)",
    )
    parser.add_argument(
        "--obs_npy",
        type=str,
        required=True,
        help="Path to saved observation (.npy or .npz)",
    )
    parser.add_argument(
        "--noise_npy",
        type=str,
        default=None,
        help="Optional path to saved noise array (.npy). If omitted, uses deterministic noise.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz path for model_out_normalized (state/actions)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = _config.get_config(args.config)
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()

    # 使用与 serve_policy 相同的路径：直接从 ckpt_dir/params 读取 merged 权重，
    # 不再叠加任何 LoRA 模块。
    policy = _policy_config.create_trained_policy(cfg, ckpt_dir, default_prompt=None)
    logging.info("Loaded JAX merged policy config=%s from %s", args.config, ckpt_dir)

    obs = load_obs(Path(args.obs_npy))

    # 噪声：与 debug_jax_sampler 保持一致，优先使用文件，其次用确定性随机噪声。
    action_horizon = getattr(cfg.model, "action_horizon", 32)
    action_dim = getattr(cfg.model, "action_dim", 32)
    noise = load_noise(args.noise_npy, action_horizon=action_horizon, action_dim=action_dim)
    if args.noise_npy is not None:
        logging.info("Loaded noise from %s", args.noise_npy)
    else:
        logging.info(
            "Generated deterministic noise with shape (%d, %d) using seed=0",
            action_horizon,
            action_dim,
        )

    # Monkeypatch _log_alignment_snapshot to捕获 jax_model_out_normalized。
    captured: dict[str, Any] = {}
    original_log = _policy._log_alignment_snapshot

    def _hook(tag: str, data: dict[str, Any], *, step: int | None = None) -> None:  # type: ignore[override]
        nonlocal captured
        if tag == "jax_model_out_normalized":
            if not captured:
                captured = {
                    "state": np.asarray(data.get("state")),
                    "actions": np.asarray(data.get("actions")),
                }
        original_log(tag, data, step=step)

    _policy._log_alignment_snapshot = _hook  # type: ignore[assignment]

    logging.info("Running JAX merged policy.infer once for debugging...")
    try:
        _ = policy.infer(obs, noise=noise)
    finally:
        _policy._log_alignment_snapshot = original_log  # type: ignore[assignment]

    if not captured:
        logging.error(
            "No jax_model_out_normalized snapshot was captured. "
            "Ensure OPENPI_ALIGN_DEBUG is enabled and model logging is active."
        )
        return

    out_path = Path(args.output)
    np.savez_compressed(out_path, state=captured["state"], actions=captured["actions"])
    logging.info("Saved merged JAX model_out_normalized snapshot to %s", out_path)


if __name__ == "__main__":
    main()
