"""
Minimal demo: LightVLA-style token pruning integrated into OpenPI π0.5 (PyTorch).

Run:
  cd third_party/openpi
  uv run python examples/pi05_lightvla_pruning_demo.py

Notes:
- Uses `dummy` Gemma configs to keep the demo lightweight.
- Demonstrates that `embed_prefix()` returns a shorter prefix in `eval()` when pruning is enabled.
"""

from __future__ import annotations

import dataclasses
import time

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


@dataclasses.dataclass
class TorchPi0Config:
    # Minimal subset required by `PI0Pytorch`.
    pi05: bool = True
    paligemma_variant: str = "dummy"
    action_expert_variant: str = "dummy"
    dtype: str = "float32"
    action_dim: int = 32
    action_horizon: int = 50

    # LightVLA-style pruning knobs (PyTorch path reads these dynamically).
    token_pruning_enabled: bool = True
    token_prune_noise_scale: float = 0.0


def _make_inputs(batch_size: int, device: torch.device):
    # 3 camera images in OpenPI default preprocessing order.
    images = [
        torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float32),
        torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float32),
        torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float32),
    ]
    img_masks = [torch.ones(batch_size, device=device, dtype=torch.bool) for _ in images]

    # Keep prompt short; PaliGemma vocab size is 257152.
    lang_len = 32
    lang_tokens = torch.randint(0, 257152, (batch_size, lang_len), device=device, dtype=torch.long)
    lang_masks = torch.ones(batch_size, lang_len, device=device, dtype=torch.bool)
    return images, img_masks, lang_tokens, lang_masks


def main():
    device = torch.device("cpu")
    batch_size = 1
    images, img_masks, lang_tokens, lang_masks = _make_inputs(batch_size, device)

    # Baseline (no pruning)
    cfg_no = dataclasses.replace(TorchPi0Config(), token_pruning_enabled=False)
    model_no = PI0Pytorch(cfg_no).to(device).eval()

    # Pruning enabled
    cfg_yes = dataclasses.replace(TorchPi0Config(), token_pruning_enabled=True)
    model_yes = PI0Pytorch(cfg_yes).to(device).eval()

    with torch.inference_mode():
        embs_no, pad_no, _ = model_no.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        embs_yes, pad_yes, _ = model_yes.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    print("=== Prefix Token Lengths (eval) ===")
    print(f"no_prune:  tokens={embs_no.shape[1]}  pad_true={int(pad_no.sum().item())}")
    print(f"lightvla:  tokens={embs_yes.shape[1]}  pad_true={int(pad_yes.sum().item())}")
    print(f"reduction: {(embs_no.shape[1] - embs_yes.shape[1])} tokens")
    print(f"stats: {model_yes.get_token_pruning_stats()}")

    # Tiny timing (prefix embedding only; CPU; best-effort)
    iters = 3
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            _ = model_no.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    t1 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            _ = model_yes.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    t2 = time.perf_counter()

    print("=== Timing (embed_prefix, CPU, best-effort) ===")
    print(f"no_prune: {(t1 - t0) / iters * 1000:.1f} ms/iter")
    print(f"lightvla: {(t2 - t1) / iters * 1000:.1f} ms/iter")


if __name__ == "__main__":
    main()
