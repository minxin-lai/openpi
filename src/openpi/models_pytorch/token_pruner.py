"""
token_pruner.py

LightVLA-style differentiable visual token pruning for OpenPI (PyTorch).

Design goals:
- Parameter-free pruning (no additional trainable weights).
- Training-time: differentiable straight-through selection without changing sequence length.
- Inference-time (`eval()`): physically drop visual tokens to reduce prefix length and KV cache size.

This module is intentionally self-contained so it can be plugged into OpenPI's
`PI0Pytorch.embed_prefix()` without touching upstream HuggingFace internals.

Reference implementation: third_party/LightVLA/prismatic/extern/hf/modeling_prismatic.py
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)


class _LightVLACore(nn.Module):
    """Core LightVLA scoring + selection (parameter-free)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.noise_scale: Optional[float] = None
        self.keep_tokens: Optional[int] = None
        self.keep_ratio: Optional[float] = None
        self.scale_factor = 1.0 / math.sqrt(self.hidden_size)

        self.last_kept_per_sample: Optional[torch.Tensor] = None  # shape [B] in eval

    def set_noise_scale(self, noise_scale: Optional[float]) -> None:
        self.noise_scale = noise_scale

    def set_keep_tokens(self, keep_tokens: Optional[int]) -> None:
        if keep_tokens is not None and int(keep_tokens) < 1:
            raise ValueError("`keep_tokens` must be >= 1.")
        self.keep_tokens = None if keep_tokens is None else int(keep_tokens)

    def set_keep_ratio(self, keep_ratio: Optional[float]) -> None:
        if keep_ratio is not None:
            keep_ratio_f = float(keep_ratio)
            if not (0.0 < keep_ratio_f <= 1.0):
                raise ValueError("`keep_ratio` must be in (0, 1].")
            self.keep_ratio = keep_ratio_f
        else:
            self.keep_ratio = None

    def compute_importance_score(self, patches: torch.Tensor, task_tokens: torch.Tensor) -> torch.Tensor:
        patches_n = _rms_norm(patches)
        task_n = _rms_norm(task_tokens)

        # LightVLA: queries = attn(patches, task, task)
        queries = F.scaled_dot_product_attention(patches_n, task_n, task_n)
        queries = _rms_norm(queries)

        # score: [B, N, N]
        return (queries @ patches_n.transpose(-2, -1)) * self.scale_factor

    def select_hard_mask(self, score: torch.Tensor) -> torch.Tensor:
        # score: [B, N, N] -> mask: [B, N]
        bsz, num_patches = score.shape[0], score.shape[1]
        mask = torch.zeros(bsz, num_patches, dtype=torch.bool, device=score.device)
        indices = score.argmax(dim=-1)  # [B, N]
        batch_indices = torch.arange(bsz, device=score.device).unsqueeze(1).expand_as(indices)
        if self.keep_tokens is None and self.keep_ratio is None:
            # LightVLA-style implicit selection: take the union of per-query argmax indices.
            mask[batch_indices, indices] = True
            return mask

        # Convert implicit selection to a fixed budget by keeping the most frequently selected patches.
        # counts: [B, N], counts[b, j] = how many queries chose patch j as argmax.
        counts = F.one_hot(indices, num_classes=num_patches).to(score.dtype).sum(dim=1)
        if self.keep_tokens is not None:
            k = min(self.keep_tokens, num_patches)
        else:
            # keep_ratio in (0, 1]
            k = int(math.ceil(float(self.keep_ratio) * num_patches))  # type: ignore[arg-type]
            k = max(1, min(k, num_patches))

        topk = counts.topk(k=k, dim=-1).indices  # [B, k]
        mask.scatter_(dim=1, index=topk, value=True)
        return mask

    def select_soft(self, score: torch.Tensor) -> torch.Tensor:
        # Straight-through estimator. Returns selection weights: [B, N, N]
        if self.noise_scale is not None and self.noise_scale != 0.0:
            score = score + torch.rand_like(score) * float(self.noise_scale)
        hard = F.one_hot(score.argmax(dim=-1), num_classes=score.shape[-1]).to(dtype=score.dtype)
        soft = torch.softmax(score, dim=-1)
        return hard + soft - soft.detach()

    @staticmethod
    def _gather_and_pad(
        patches: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather variable-length patches per sample and pad to max kept in batch.

        Returns:
          - pruned_patches: [B, Kmax, D]
          - pruned_mask: [B, Kmax] (True for valid, False for padding)
        """
        bsz, _, dim = patches.shape
        kept_counts = mask.sum(dim=-1)
        kmax = int(kept_counts.max().item()) if bsz > 0 else 0
        kmax = max(kmax, 1)

        out = patches.new_zeros((bsz, kmax, dim))
        out_mask = torch.zeros((bsz, kmax), dtype=torch.bool, device=patches.device)

        for i in range(bsz):
            idx = torch.where(mask[i])[0]
            if idx.numel() == 0:
                idx = torch.tensor([0], device=patches.device)
            k = int(min(idx.numel(), kmax))
            out[i, :k] = patches[i, idx[:k]]
            out_mask[i, :k] = True

        return out, out_mask


class ImageTokenPruner(nn.Module):
    """Prunes a single image's patch embeddings using language/task embeddings as context.

    Intended integration point: `PI0Pytorch.embed_prefix()` after image embedding and language embedding.

    Training-time (`train()`):
      - Returns same number of patch tokens as input (differentiable selection).
    Inference-time (`eval()`):
      - Returns fewer patch tokens (padded to max kept in batch if batch>1).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.core = _LightVLACore(hidden_size=hidden_size)

    @property
    def noise_scale(self) -> Optional[float]:
        return self.core.noise_scale

    def set_noise_scale(self, noise_scale: Optional[float]) -> None:
        self.core.set_noise_scale(noise_scale)

    def set_keep_tokens(self, keep_tokens: Optional[int]) -> None:
        self.core.set_keep_tokens(keep_tokens)

    def set_keep_ratio(self, keep_ratio: Optional[float]) -> None:
        self.core.set_keep_ratio(keep_ratio)

    @torch.no_grad()
    def last_kept_per_sample(self) -> Optional[torch.Tensor]:
        return self.core.last_kept_per_sample

    def prune(
        self, image_patches: torch.Tensor, task_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          image_patches: [B, N, D]
          task_tokens: [B, T, D]

        Returns:
          pruned_patches: [B, N, D] (train) or [B, Kmax, D] (eval)
          pruned_token_mask: [B, N] (train, all True) or [B, Kmax] (eval, True for kept)
        """
        if image_patches.ndim != 3 or task_tokens.ndim != 3:
            raise ValueError("Expected `image_patches` and `task_tokens` to be rank-3 tensors [B, S, D].")
        if image_patches.shape[0] != task_tokens.shape[0]:
            raise ValueError("Batch size mismatch between `image_patches` and `task_tokens`.")
        if image_patches.shape[-1] != task_tokens.shape[-1]:
            raise ValueError("Hidden size mismatch between `image_patches` and `task_tokens`.")

        score = self.core.compute_importance_score(image_patches, task_tokens)

        if self.training:
            weights = self.core.select_soft(score)  # [B, N, N]
            pruned = weights @ image_patches
            mask = torch.ones((image_patches.shape[0], image_patches.shape[1]), dtype=torch.bool, device=image_patches.device)
            self.core.last_kept_per_sample = None
            return pruned, mask

        hard_mask = self.core.select_hard_mask(score)  # [B, N]
        pruned, pruned_mask = self.core._gather_and_pad(image_patches, hard_mask)
        self.core.last_kept_per_sample = hard_mask.sum(dim=-1)
        return pruned, pruned_mask

    def forward(self, image_patches: torch.Tensor, task_tokens: torch.Tensor) -> torch.Tensor:
        pruned, _ = self.prune(image_patches, task_tokens)
        return pruned


class TokenPruner(nn.Module):
    """Sequence-level pruner for sequences shaped as: [cls_tokens, patches, task_tokens].

    This mirrors LightVLA's PrunedLlamaModel integration, but is provided as a generic utility.
    OpenPI integration typically uses `ImageTokenPruner` instead.
    """

    def __init__(self, hidden_size: int, num_patches: int) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.core = _LightVLACore(hidden_size=hidden_size)

    @property
    def noise_scale(self) -> Optional[float]:
        return self.core.noise_scale

    def set_noise_scale(self, noise_scale: Optional[float]) -> None:
        self.core.set_noise_scale(noise_scale)

    def set_keep_tokens(self, keep_tokens: Optional[int]) -> None:
        self.core.set_keep_tokens(keep_tokens)

    def set_keep_ratio(self, keep_ratio: Optional[float]) -> None:
        self.core.set_keep_ratio(keep_ratio)

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cls_token_count: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, dim = tokens.shape
        task_token_count = seq_len - cls_token_count - self.num_patches
        if task_token_count < 0:
            raise ValueError(
                f"Invalid sequence split: seq_len={seq_len}, cls_token_count={cls_token_count}, num_patches={self.num_patches}"
            )

        cls_tokens, patches, task_tokens = torch.split(tokens, [cls_token_count, self.num_patches, task_token_count], dim=1)
        cls_pos, patches_pos, task_pos = torch.split(position_ids, [cls_token_count, self.num_patches, task_token_count], dim=1)

        if attention_mask is not None:
            cls_mask, patches_mask, task_mask = torch.split(
                attention_mask, [cls_token_count, self.num_patches, task_token_count], dim=1
            )

        score = self.core.compute_importance_score(patches, task_tokens)

        if self.training:
            weights = self.core.select_soft(score)
            patches_new = weights @ patches

            indices = weights.argmax(dim=-1)  # [B, N]
            batch_indices = torch.arange(bsz, device=tokens.device).unsqueeze(1).expand_as(indices)
            patches_pos_new = patches_pos[batch_indices, indices]
            if attention_mask is not None:
                patches_mask_new = patches_mask[batch_indices, indices]

            tokens_out = torch.cat([cls_tokens, patches_new, task_tokens], dim=1)
            pos_out = torch.cat([cls_pos, patches_pos_new, task_pos], dim=1)
            if attention_mask is None:
                return tokens_out, pos_out, None
            mask_out = torch.cat([cls_mask, patches_mask_new, task_mask], dim=1)
            return tokens_out, pos_out, mask_out

        hard_mask = self.core.select_hard_mask(score)  # [B, N]
        patches_new, patches_keep_mask = self.core._gather_and_pad(patches, hard_mask)

        # Gather position ids and attention masks accordingly (pad with zeros/False).
        bsz, kmax = patches_keep_mask.shape
        patches_pos_new = patches_pos.new_zeros((bsz, kmax))
        if attention_mask is not None:
            patches_mask_new = attention_mask.new_zeros((bsz, kmax))

        for i in range(bsz):
            idx = torch.where(hard_mask[i])[0]
            if idx.numel() == 0:
                idx = torch.tensor([0], device=tokens.device)
            k = int(min(idx.numel(), kmax))
            patches_pos_new[i, :k] = patches_pos[i, idx[:k]]
            if attention_mask is not None:
                patches_mask_new[i, :k] = patches_mask[i, idx[:k]]

        tokens_out = torch.cat([cls_tokens, patches_new, task_tokens], dim=1)
        pos_out = torch.cat([cls_pos, patches_pos_new, task_pos], dim=1)

        if attention_mask is None:
            return tokens_out, pos_out, None
        mask_out = torch.cat([cls_mask, patches_mask_new, task_mask], dim=1)
        return tokens_out, pos_out, mask_out
