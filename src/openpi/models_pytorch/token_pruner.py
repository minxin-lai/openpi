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

        self.last_kept_per_sample: Optional[torch.Tensor] = None  # shape [B] (eval-style deterministic)
        self.last_kept_per_sample_noisy: Optional[torch.Tensor] = None  # shape [B] (train forward selection)
        self.last_kept_indices_padded: Optional[torch.Tensor] = None  # shape [B, Kmax] in eval
        self.last_kept_indices_mask: Optional[torch.Tensor] = None  # shape [B, Kmax] in eval
        self.last_selected_indices_train: Optional[torch.Tensor] = None  # shape [B, N] in train (noisy selection)
        # Debug telemetry: argmax vote concentration + task token length.
        self.last_task_valid_len: Optional[torch.Tensor] = None  # shape [B]
        self.last_task_attn_masked_len: Optional[torch.Tensor] = None  # shape [B]
        self.last_argmax_union_det: Optional[torch.Tensor] = None  # shape [B]
        self.last_argmax_union_noisy: Optional[torch.Tensor] = None  # shape [B]
        self.last_argmax_top1_share_det: Optional[torch.Tensor] = None  # shape [B]
        self.last_argmax_top1_share_noisy: Optional[torch.Tensor] = None  # shape [B]
        # Debug telemetry: diversity of embeddings (std over tokens, then mean over channels).
        self.last_patches_std_mean: Optional[torch.Tensor] = None  # shape [B]
        self.last_task_std_mean: Optional[torch.Tensor] = None  # shape [B]
        self.last_queries_std_mean: Optional[torch.Tensor] = None  # shape [B]
        # Debug telemetry: score magnitude and separability.
        self.last_score_abs_max_det: Optional[torch.Tensor] = None  # shape [B]
        self.last_score_top1_gap_mean_det: Optional[torch.Tensor] = None  # shape [B]

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

    def compute_importance_score(
        self,
        patches: torch.Tensor,
        task_tokens: torch.Tensor,
        *,
        task_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        patches_n = _rms_norm(patches)
        task_n = _rms_norm(task_tokens)

        # LightVLA: queries = attn(patches, task, task)
        attn_mask = None
        if task_token_mask is not None:
            if task_token_mask.ndim != 2:
                raise ValueError("Expected `task_token_mask` to be rank-2 [B, T].")
            if task_token_mask.shape[0] != task_n.shape[0] or task_token_mask.shape[1] != task_n.shape[1]:
                raise ValueError("Shape mismatch between `task_tokens` and `task_token_mask`.")
            # Build an additive attention mask (0 for keep, -inf for mask) to avoid bool semantics ambiguity.
            attn_mask_bool = ~task_token_mask.to(dtype=torch.bool, device=task_n.device)
            # SDPA expects `attn_mask` broadcastable to [B, L, S] for 3D inputs.
            attn_mask = attn_mask_bool[:, None, :].expand(task_n.shape[0], patches_n.shape[1], task_n.shape[1])
            with torch.no_grad():
                valid = ~attn_mask_bool  # [B, T], True for valid task tokens
                self.last_task_valid_len = valid.sum(dim=-1)
                self.last_task_attn_masked_len = attn_mask_bool.sum(dim=-1)
            # Convert to additive mask.
            attn_mask = attn_mask.to(dtype=patches_n.dtype) * torch.finfo(patches_n.dtype).min
        else:
            self.last_task_valid_len = None
            self.last_task_attn_masked_len = None

        queries_raw = F.scaled_dot_product_attention(patches_n, task_n, task_n, attn_mask=attn_mask)

        # Diversity diagnostics: Track queries BEFORE and AFTER RMSNorm to identify collapse point.
        with torch.no_grad():
            self.last_patches_std_mean = patches_n.detach().to(torch.float32).std(dim=1).mean(dim=-1)
            # Queries BEFORE RMSNorm
            queries_raw_f32 = queries_raw.detach().to(torch.float32)
            self.last_queries_before_norm_std_mean = queries_raw_f32.std(dim=1).mean(dim=-1)  # [B] -> scalar
            # Cosine similarity among queries (before norm): check if all queries point in same direction
            # Compute mean pairwise cosine similarity for each sample
            bsz, num_queries, dim = queries_raw_f32.shape
            self.last_queries_before_norm_cosine_sim = torch.zeros(bsz, dtype=torch.float32, device=queries_raw.device)
            for i in range(bsz):
                q = queries_raw_f32[i]  # [N, D]
                q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-8)  # [N, D]
                sim_matrix = q_norm @ q_norm.T  # [N, N]
                # Mean of off-diagonal elements (exclude self-similarity)
                mask = ~torch.eye(num_queries, dtype=torch.bool, device=sim_matrix.device)
                self.last_queries_before_norm_cosine_sim[i] = sim_matrix[mask].mean()

        queries = _rms_norm(queries_raw)

        # Diversity diagnostics: Track queries AFTER RMSNorm.
        with torch.no_grad():
            queries_f32 = queries.detach().to(torch.float32)
            self.last_queries_after_norm_std_mean = queries_f32.std(dim=1).mean(dim=-1)  # [B] -> scalar
            # Cosine similarity among queries (after norm)
            self.last_queries_after_norm_cosine_sim = torch.zeros(bsz, dtype=torch.float32, device=queries.device)
            for i in range(bsz):
                q = queries_f32[i]  # [N, D]
                q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-8)  # [N, D]
                sim_matrix = q_norm @ q_norm.T  # [N, N]
                mask = ~torch.eye(num_queries, dtype=torch.bool, device=sim_matrix.device)
                self.last_queries_after_norm_cosine_sim[i] = sim_matrix[mask].mean()
            if task_token_mask is None:
                self.last_task_std_mean = task_n.detach().to(torch.float32).std(dim=1).mean(dim=-1)
            else:
                bsz = task_n.shape[0]
                out = torch.zeros((bsz,), dtype=torch.float32, device=task_n.device)
                valid = task_token_mask.to(dtype=torch.bool, device=task_n.device)
                task_f = task_n.detach().to(torch.float32)
                for i in range(bsz):
                    idx = torch.where(valid[i])[0]
                    if idx.numel() < 2:
                        out[i] = 0.0
                    else:
                        out[i] = task_f[i, idx].std(dim=0).mean()
                self.last_task_std_mean = out

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

    @staticmethod
    def _gather_and_pad_with_mask(
        patches: torch.Tensor,
        mask: torch.Tensor,
        *,
        extra_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Gather variable-length patches per sample and pad to max kept in batch.

        Optionally gathers `extra_mask` (e.g., per-patch validity) with the same indices.
        """
        bsz, _, dim = patches.shape
        kept_counts = mask.sum(dim=-1)
        kmax = int(kept_counts.max().item()) if bsz > 0 else 0
        kmax = max(kmax, 1)

        out = patches.new_zeros((bsz, kmax, dim))
        out_mask = torch.zeros((bsz, kmax), dtype=torch.bool, device=patches.device)
        out_extra = None
        if extra_mask is not None:
            if extra_mask.shape != mask.shape:
                raise ValueError("Expected `extra_mask` to have the same shape as `mask` ([B, N]).")
            out_extra = torch.zeros((bsz, kmax), dtype=torch.bool, device=patches.device)

        for i in range(bsz):
            idx = torch.where(mask[i])[0]
            if idx.numel() == 0:
                idx = torch.tensor([0], device=patches.device)
            k = int(min(idx.numel(), kmax))
            out[i, :k] = patches[i, idx[:k]]
            out_mask[i, :k] = True
            if out_extra is not None:
                out_extra[i, :k] = extra_mask[i, idx[:k]]  # type: ignore[index]

        return out, out_mask, out_extra

    @staticmethod
    def _gather_and_pad_with_mask_and_indices(
        patches: torch.Tensor,
        mask: torch.Tensor,
        *,
        extra_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Like `_gather_and_pad_with_mask`, but also returns original indices per kept token.

        Returns:
          - pruned_patches: [B, Kmax, D]
          - pruned_mask: [B, Kmax]
          - gathered_extra_mask: [B, Kmax] or None
          - gathered_indices: [B, Kmax] (padding positions set to 0)
        """
        bsz, _, dim = patches.shape
        kept_counts = mask.sum(dim=-1)
        kmax = int(kept_counts.max().item()) if bsz > 0 else 0
        kmax = max(kmax, 1)

        out = patches.new_zeros((bsz, kmax, dim))
        out_mask = torch.zeros((bsz, kmax), dtype=torch.bool, device=patches.device)
        out_indices = torch.zeros((bsz, kmax), dtype=torch.long, device=patches.device)
        out_extra = None
        if extra_mask is not None:
            if extra_mask.shape != mask.shape:
                raise ValueError("Expected `extra_mask` to have the same shape as `mask` ([B, N]).")
            out_extra = torch.zeros((bsz, kmax), dtype=torch.bool, device=patches.device)

        for i in range(bsz):
            idx = torch.where(mask[i])[0]
            if idx.numel() == 0:
                idx = torch.tensor([0], device=patches.device)
            k = int(min(idx.numel(), kmax))
            out[i, :k] = patches[i, idx[:k]]
            out_mask[i, :k] = True
            out_indices[i, :k] = idx[:k].to(dtype=torch.long)
            if out_extra is not None:
                out_extra[i, :k] = extra_mask[i, idx[:k]]  # type: ignore[index]

        return out, out_mask, out_extra, out_indices


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
        self,
        image_patches: torch.Tensor,
        task_tokens: torch.Tensor,
        *,
        task_token_mask: Optional[torch.Tensor] = None,
        patch_token_mask: Optional[torch.Tensor] = None,
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

        score = self.core.compute_importance_score(image_patches, task_tokens, task_token_mask=task_token_mask)
        if patch_token_mask is not None:
            if patch_token_mask.shape[:2] != image_patches.shape[:2]:
                raise ValueError("Expected `patch_token_mask` to have shape [B, N] matching `image_patches`.")
            # Ensure invalid patches cannot be selected as keys/candidates.
            # `score` is [B, N, N] where the last dim indexes candidate patches.
            patch_valid = patch_token_mask.to(dtype=torch.bool, device=score.device)
            # Avoid all-masked rows causing NaNs in softmax; if a sample has no valid patches,
            # allow index 0 as a dummy key (these tokens should still be masked out downstream).
            any_valid = patch_valid.any(dim=-1)
            if (~any_valid).any():
                patch_valid = patch_valid.clone()
                patch_valid[~any_valid, 0] = True
            # Broadcast to [B, N, N] (mask keys/candidates only).
            score = score.masked_fill(~patch_valid[:, None, :], float("-inf"))

        if self.training:
            # Record hard (eval-style) keep counts for telemetry, without affecting gradients/outputs.
            # This is especially useful when implicit selection collapses due to masked/padded task tokens.
            with torch.no_grad():
                score_det = score.detach()
                hard_mask_det = self.core.select_hard_mask(score_det)
                if patch_token_mask is not None:
                    hard_mask_det = hard_mask_det & patch_token_mask.to(dtype=torch.bool, device=hard_mask_det.device)
                self.core.last_kept_per_sample = hard_mask_det.sum(dim=-1)

                # Also track the *noisy* hard selection that training uses for straight-through.
                score_noisy = score_det
                if self.core.noise_scale is not None and float(self.core.noise_scale) != 0.0:
                    score_noisy = score_det + torch.rand_like(score_det) * float(self.core.noise_scale)
                hard_mask_noisy = self.core.select_hard_mask(score_noisy)
                if patch_token_mask is not None:
                    hard_mask_noisy = hard_mask_noisy & patch_token_mask.to(
                        dtype=torch.bool, device=hard_mask_noisy.device
                    )
                self.core.last_kept_per_sample_noisy = hard_mask_noisy.sum(dim=-1)
                self.core.last_kept_indices_padded = None
                self.core.last_kept_indices_mask = None
                self.core.last_selected_indices_train = None
                if task_token_mask is not None:
                    self.core.last_task_valid_len = task_token_mask.to(dtype=torch.bool, device=score.device).sum(dim=-1)
                else:
                    self.core.last_task_valid_len = None

                # Argmax vote concentration diagnostics (independent of keep budget).
                # - union size: number of unique argmax indices across queries
                # - top1 share: fraction of queries voting for the most popular patch
                def _vote_stats(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    bsz, num_queries = idx.shape
                    num_candidates = score_det.shape[-1]
                    counts = torch.zeros((bsz, num_candidates), dtype=torch.int32, device=idx.device)
                    ones = torch.ones_like(idx, dtype=torch.int32)
                    counts.scatter_add_(dim=1, index=idx.to(dtype=torch.long), src=ones)
                    union = (counts > 0).sum(dim=-1).to(dtype=torch.int32)
                    top1_share = counts.max(dim=-1).values.to(dtype=torch.float32) / float(max(1, num_queries))
                    return union, top1_share

                idx_det = score_det.argmax(dim=-1)  # [B, N]
                union_det, top1_det = _vote_stats(idx_det)
                self.core.last_argmax_union_det = union_det
                self.core.last_argmax_top1_share_det = top1_det

                idx_noisy = score_noisy.argmax(dim=-1)  # [B, N]
                union_noisy, top1_noisy = _vote_stats(idx_noisy)
                self.core.last_argmax_union_noisy = union_noisy
                self.core.last_argmax_top1_share_noisy = top1_noisy

                # Score magnitude + top1-top2 gap (helps validate whether scaling/noise are reasonable).
                score_f = score_det.to(dtype=torch.float32)
                finite = torch.isfinite(score_f)
                score_f_safe = score_f.masked_fill(~finite, 0.0)
                self.core.last_score_abs_max_det = score_f_safe.abs().amax(dim=(-2, -1))
                # gap per query row: mean over queries -> [B]
                top2 = score_f.topk(k=2, dim=-1).values  # [B, N, 2]
                gap = (top2[..., 0] - top2[..., 1]).mean(dim=-1)  # [B]
                self.core.last_score_top1_gap_mean_det = gap

            weights = self.core.select_soft(score)  # [B, N, N]
            pruned = weights @ image_patches
            bsz, num_patches = image_patches.shape[:2]
            with torch.no_grad():
                indices = weights.argmax(dim=-1)  # [B, N]
                self.core.last_selected_indices_train = indices.detach()
            if patch_token_mask is not None:
                mask = patch_token_mask.to(dtype=torch.bool, device=image_patches.device)
            else:
                mask = torch.ones((bsz, num_patches), dtype=torch.bool, device=image_patches.device)
            return pruned, mask

        hard_mask = self.core.select_hard_mask(score)  # [B, N]
        if patch_token_mask is not None:
            pruned, pruned_mask, gathered_valid, kept_indices = self.core._gather_and_pad_with_mask_and_indices(
                image_patches,
                hard_mask,
                extra_mask=patch_token_mask.to(dtype=torch.bool, device=image_patches.device),
            )
            assert gathered_valid is not None
            self.core.last_kept_per_sample = (hard_mask & patch_token_mask.to(dtype=torch.bool, device=hard_mask.device)).sum(dim=-1)
            self.core.last_kept_per_sample_noisy = None
            kept_mask = pruned_mask & gathered_valid
            self.core.last_kept_indices_padded = kept_indices
            self.core.last_kept_indices_mask = kept_mask
            self.core.last_selected_indices_train = None
            return pruned, kept_mask

        pruned, pruned_mask, _, kept_indices = self.core._gather_and_pad_with_mask_and_indices(
            image_patches, hard_mask, extra_mask=None
        )
        self.core.last_kept_per_sample = hard_mask.sum(dim=-1)
        self.core.last_kept_per_sample_noisy = None
        self.core.last_kept_indices_padded = kept_indices
        self.core.last_kept_indices_mask = pruned_mask
        self.core.last_selected_indices_train = None
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
