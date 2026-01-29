from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _bootstrap_vla_opt_on_path() -> None:
    # This script lives under `third_party/openpi/scripts/`.
    repo_root = Path(__file__).resolve().parents[3]
    vla_src = repo_root / "src"
    if not vla_src.exists():
        raise FileNotFoundError(f"vla-opt src not found at: {vla_src}")
    if str(vla_src) not in sys.path:
        sys.path.insert(0, str(vla_src))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: load a VLA-OPT wrapped PyTorch checkpoint and run 1 infer.")
    parser.add_argument("--config", type=str, default="debug_pi05", help="OpenPI train config name.")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/debug_pi05/debug_pi05/1",
        help="Checkpoint dir containing model.safetensors.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu/cuda/cuda:0 ...")
    parser.add_argument("--ve-film-num-blocks", type=int, default=2)
    parser.add_argument("--ste-prune-layer", type=int, default=25)
    parser.add_argument("--ste-prune-k", type=int, default=12)
    parser.add_argument("--ste-prune-tau", type=float, default=1.0)
    args = parser.parse_args()

    _bootstrap_vla_opt_on_path()

    # Enable wrappers in OpenPI loader.
    os.environ["VLA_OPT_VE_FILM"] = "1"
    os.environ["VLA_OPT_VE_FILM_NUM_BLOCKS"] = str(int(args.ve_film_num_blocks))
    os.environ["VLA_OPT_STE_PRUNE"] = "1"
    os.environ["VLA_OPT_STE_PRUNE_K"] = str(int(args.ste_prune_k))
    os.environ["VLA_OPT_STE_PRUNE_STAGE"] = "gather"
    os.environ["VLA_OPT_STE_PRUNE_TAU"] = str(float(args.ste_prune_tau))
    os.environ["VLA_OPT_STE_PRUNE_LAYER"] = str(int(args.ste_prune_layer))

    from openpi.policies.policy import Policy
    from openpi.training import config as _config

    train_cfg = _config.get_config(str(args.config))
    ckpt_dir = Path(args.ckpt_dir)
    weight_path = ckpt_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"model.safetensors not found at: {weight_path}")

    model = train_cfg.model.load_pytorch(train_cfg, str(weight_path))
    model = model.to(str(args.device))
    model.eval()

    # Replace sample_actions with a lightweight function so this runs fast on CPU.
    def _smoke_sample_actions(_device: str, observation, **_kwargs):
        stage_a_handle = getattr(model, "_vla_opt_stage_a_handle", None)
        ste_handle = getattr(model, "_vla_opt_ste_prune_handle", None)
        if stage_a_handle is None or ste_handle is None:
            raise RuntimeError("Expected both _vla_opt_stage_a_handle and _vla_opt_ste_prune_handle on model")

        img = observation.images["base_0_rgb"]
        emb = model.paligemma_with_expert.embed_image(img)
        seq_len = int(emb.shape[1])
        expected = int(ste_handle.k)
        if seq_len != expected:
            raise AssertionError(f"Expected embed_image seq_len={expected} after gather, got {seq_len}")

        bsz = int(observation.state.shape[0])
        return torch.zeros(
            (bsz, int(model.config.action_horizon), int(model.config.action_dim)),
            device=observation.state.device,
            dtype=torch.float32,
        )

    model.sample_actions = _smoke_sample_actions  # type: ignore[method-assign]

    policy = Policy(model, is_pytorch=True, pytorch_device=str(args.device))

    h, w = 224, 224
    obs = {
        "image": {
            "base_0_rgb": np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.asarray(True),
            "left_wrist_0_rgb": np.asarray(True),
            "right_wrist_0_rgb": np.asarray(True),
        },
        "state": np.zeros((32,), dtype=np.float32),
        "tokenized_prompt": np.random.randint(0, 64, size=(16,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((16,), dtype=np.bool_),
    }

    out = policy.infer(obs)
    actions = out.get("actions", None)
    assert actions is not None
    assert tuple(np.asarray(actions).shape) == (int(model.config.action_horizon), int(model.config.action_dim))

    print("OK")
    print("ckpt:", str(ckpt_dir))
    print("device:", str(args.device))
    print("ve_film_num_blocks:", int(args.ve_film_num_blocks))
    print("ste_prune_layer:", int(args.ste_prune_layer))
    print("ste_prune_k:", int(args.ste_prune_k))
    print("actions:", tuple(np.asarray(actions).shape), np.asarray(actions).dtype)


if __name__ == "__main__":
    main()

