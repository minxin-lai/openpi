import logging
import math
import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch import gemma_config as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.token_pruner import ImageTokenPruner


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.token_pruning_enabled = bool(getattr(config, "token_pruning_enabled", False))
        self.token_prune_noise_scale = float(getattr(config, "token_prune_noise_scale", 0.0))
        self.token_prune_keep_tokens = getattr(config, "token_prune_keep_tokens", None)
        self.token_prune_keep_ratio = getattr(config, "token_prune_keep_ratio", None)
        self._token_pruner_user_noise_scale: float | None = None
        self._token_pruning_last_kept_per_image: list[torch.Tensor] | None = None
        self._token_pruning_last_before_tokens_per_image: list[int] | None = None
        self._last_prefix_unpruned_len: int | None = None

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        # Token pruning (LightVLA-style) integrates into `embed_prefix()` and may introduce data-dependent
        # prefix lengths at inference; TorchInductor compilation can become unstable in this case.
        self.token_pruner: ImageTokenPruner | None = None
        if self.token_pruning_enabled:
            hidden_size = int(self.paligemma_with_expert.paligemma.config.text_config.hidden_size)
            self.token_pruner = ImageTokenPruner(hidden_size=hidden_size)
            # Apply eval-time pruning budget (if configured). Training keeps sequence length unchanged.
            self.token_pruner.set_keep_tokens(self.token_prune_keep_tokens)
            self.token_pruner.set_keep_ratio(self.token_prune_keep_ratio)
            logging.info(f"Enabled LightVLA-style token pruning (hidden_size={hidden_size})")

        def _truthy_env(name: str, default: str = "1") -> bool:
            val = os.getenv(name, default)
            return val is not None and val.lower() not in ("", "0", "false", "no", "off")

        compile_enabled = _truthy_env("OPENPI_TORCH_COMPILE", "1")
        compile_pruning = _truthy_env("OPENPI_TORCH_COMPILE_PRUNING", "0")

        if compile_enabled and (not self.token_pruning_enabled or compile_pruning):
            # Use dynamic shapes when pruning is enabled, otherwise keep default behavior.
            dynamic = True if self.token_pruning_enabled else None
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune", dynamic=dynamic)
        elif compile_enabled and self.token_pruning_enabled and not compile_pruning:
            logging.info(
                "Token pruning is enabled; skipping `torch.compile(sample_actions)` by default. "
                "Set OPENPI_TORCH_COMPILE_PRUNING=1 to force compilation with dynamic shapes."
            )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        # Configure pruning noise: training uses configured noise (or user override), eval disables noise.
        if self.token_pruner is not None:
            if self.training:
                noise_scale = self._token_pruner_user_noise_scale
                if noise_scale is None:
                    noise_scale = self.token_prune_noise_scale
                self.token_pruner.set_noise_scale(noise_scale)
            else:
                self.token_pruner.set_noise_scale(None)

        # Process images and concatenate patches across cameras before pruning (LightVLA-style).
        img_embs: list[torch.Tensor] = []
        img_patch_valid_masks: list[torch.Tensor] = []
        original_tokens_total = 0
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_tokens = img_emb.shape[:2]
            original_tokens_total += int(num_img_tokens)
            img_embs.append(img_emb)
            img_patch_valid_masks.append(img_mask[:, None].expand(bsize, num_img_tokens))

        # Concatenate all image patch embeddings along the token dimension.
        all_img_emb = torch.cat(img_embs, dim=1) if len(img_embs) > 0 else None
        all_img_valid = torch.cat(img_patch_valid_masks, dim=1) if len(img_patch_valid_masks) > 0 else None
        # Unpruned prefix length for LightVLA-style position_ids (counts all patch + prompt tokens, including padding).
        self._last_prefix_unpruned_len = int(original_tokens_total + lang_emb.shape[1])

        if all_img_emb is not None:
            if self.token_pruning_enabled and self.token_pruner is not None:
                all_img_emb, kept_mask = self.token_pruner.prune(
                    all_img_emb,
                    lang_emb,
                    task_token_mask=lang_masks,
                    patch_token_mask=all_img_valid,
                )
                bsize, num_img_embs = all_img_emb.shape[:2]
                pad_masks.append(kept_mask)

                # Log pruning statistics during inference (eval mode)
                if not self.training:
                    kept_per_sample = kept_mask.sum(dim=1)
                    avg_kept = kept_per_sample.float().mean().item()
                    reduction = original_tokens_total - avg_kept
                    reduction_pct = (reduction / original_tokens_total) * 100 if original_tokens_total > 0 else 0
                    logging.info(
                        f"[Token Pruning] Visual tokens: "
                        f"before={original_tokens_total}, after={avg_kept:.1f}, "
                        f"reduced={reduction:.1f} ({reduction_pct:.1f}%)"
                    )
            else:
                pad_masks.append(all_img_valid)

            embs.append(all_img_emb)
            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * all_img_emb.shape[1]

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # Store last batch's token pruning stats (training only).
        if self.training and self.token_pruning_enabled and self.token_pruner is not None:
            if self.token_pruner.core.last_kept_per_sample is not None:
                self._token_pruning_last_kept_per_image = [self.token_pruner.core.last_kept_per_sample.detach()]
                self._token_pruning_last_before_tokens_per_image = [int(original_tokens_total)]
            else:
                self._token_pruning_last_kept_per_image = None
                self._token_pruning_last_before_tokens_per_image = None

        # LightVLA-style positions:
        # - Train: align patch `position_ids` with the (noisy) selection indices used by straight-through selection.
        # - Eval: gather original patch indices for physically kept tokens (may have gaps).
        # - Prompt positions remain offset by the *unpruned* patch count.
        if all_img_emb is None:
            img_pos = torch.zeros((bsize, 0), device=pad_masks.device, dtype=torch.long)
        elif self.token_pruning_enabled and self.token_pruner is not None:
            if not self.training:
                kept_indices = getattr(self.token_pruner.core, "last_kept_indices_padded", None)
                if kept_indices is None:
                    img_pos = (
                        torch.arange(all_img_emb.shape[1], device=pad_masks.device, dtype=torch.long)[None, :]
                        .expand(bsize, -1)
                    )
                else:
                    img_pos = kept_indices.to(device=pad_masks.device, dtype=torch.long)
            else:
                selected_indices = getattr(self.token_pruner.core, "last_selected_indices_train", None)
                if selected_indices is None:
                    img_pos = (
                        torch.arange(all_img_emb.shape[1], device=pad_masks.device, dtype=torch.long)[None, :]
                        .expand(bsize, -1)
                    )
                else:
                    img_pos = selected_indices.to(device=pad_masks.device, dtype=torch.long)
        else:
            img_pos = (
                torch.arange(all_img_emb.shape[1], device=pad_masks.device, dtype=torch.long)[None, :].expand(bsize, -1)
            )

        lang_pos = (
            torch.arange(lang_emb.shape[1], device=pad_masks.device, dtype=torch.long)[None, :].expand(bsize, -1)
            + int(original_tokens_total)
        )
        prefix_position_ids = torch.cat([img_pos, lang_pos], dim=1)

        return embs, pad_masks, att_masks, prefix_position_ids

    def set_token_pruner_noise_scale(self, noise_scale: float | None) -> None:
        """Override pruning noise scale (training only). Use None to revert to config default."""
        self._token_pruner_user_noise_scale = None if noise_scale is None else float(noise_scale)

    def set_token_pruner_keep_tokens(self, keep_tokens: int | None) -> None:
        """Set eval-time keep budget (patch tokens per image). Use None to disable fixed-budget pruning."""
        if self.token_pruner is None:
            return
        self.token_pruner.set_keep_tokens(keep_tokens)

    def set_token_pruner_keep_ratio(self, keep_ratio: float | None) -> None:
        """Set eval-time keep ratio in (0, 1]. Use None to disable fixed-budget pruning."""
        if self.token_pruner is None:
            return
        self.token_pruner.set_keep_ratio(keep_ratio)

    def get_token_pruning_stats(self) -> dict:
        """Best-effort pruning stats for telemetry/debug (PyTorch path)."""
        if not self.token_pruning_enabled or self.token_pruner is None:
            return {"enabled": False}
        kept = self.token_pruner.core.last_kept_per_sample
        kept_noisy = getattr(self.token_pruner.core, "last_kept_per_sample_noisy", None)
        task_valid_len = getattr(self.token_pruner.core, "last_task_valid_len", None)
        task_attn_masked_len = getattr(self.token_pruner.core, "last_task_attn_masked_len", None)
        argmax_union_det = getattr(self.token_pruner.core, "last_argmax_union_det", None)
        argmax_union_noisy = getattr(self.token_pruner.core, "last_argmax_union_noisy", None)
        argmax_top1_det = getattr(self.token_pruner.core, "last_argmax_top1_share_det", None)
        argmax_top1_noisy = getattr(self.token_pruner.core, "last_argmax_top1_share_noisy", None)
        patches_std_mean = getattr(self.token_pruner.core, "last_patches_std_mean", None)
        task_std_mean = getattr(self.token_pruner.core, "last_task_std_mean", None)
        # New diagnostics: queries before/after RMSNorm + cosine similarity
        queries_before_norm_std = getattr(self.token_pruner.core, "last_queries_before_norm_std_mean", None)
        queries_after_norm_std = getattr(self.token_pruner.core, "last_queries_after_norm_std_mean", None)
        queries_before_norm_cosine = getattr(self.token_pruner.core, "last_queries_before_norm_cosine_sim", None)
        queries_after_norm_cosine = getattr(self.token_pruner.core, "last_queries_after_norm_cosine_sim", None)
        score_abs_max_det = getattr(self.token_pruner.core, "last_score_abs_max_det", None)
        score_top1_gap_mean_det = getattr(self.token_pruner.core, "last_score_top1_gap_mean_det", None)
        stats: dict = {
            "enabled": True,
            "noise_scale": self.token_pruner.noise_scale,
            "keep_tokens": getattr(self.token_pruner.core, "keep_tokens", None),
            "keep_ratio": getattr(self.token_pruner.core, "keep_ratio", None),
            "last_kept_per_sample": None if kept is None else kept.detach().cpu().tolist(),
            "last_kept_per_sample_noisy": None if kept_noisy is None else kept_noisy.detach().cpu().tolist(),
            "last_task_valid_len": None if task_valid_len is None else task_valid_len.detach().cpu().tolist(),
            "last_task_attn_masked_len": None
            if task_attn_masked_len is None
            else task_attn_masked_len.detach().cpu().tolist(),
            "last_argmax_union_det": None if argmax_union_det is None else argmax_union_det.detach().cpu().tolist(),
            "last_argmax_union_noisy": None if argmax_union_noisy is None else argmax_union_noisy.detach().cpu().tolist(),
            "last_argmax_top1_share_det": None if argmax_top1_det is None else argmax_top1_det.detach().cpu().tolist(),
            "last_argmax_top1_share_noisy": None if argmax_top1_noisy is None else argmax_top1_noisy.detach().cpu().tolist(),
            "last_patches_std_mean": None if patches_std_mean is None else patches_std_mean.detach().cpu().tolist(),
            "last_task_std_mean": None if task_std_mean is None else task_std_mean.detach().cpu().tolist(),
            # Queries diagnostics: before/after RMSNorm
            "last_queries_before_norm_std_mean": None if queries_before_norm_std is None else queries_before_norm_std.detach().cpu().tolist(),
            "last_queries_after_norm_std_mean": None if queries_after_norm_std is None else queries_after_norm_std.detach().cpu().tolist(),
            "last_queries_before_norm_cosine_sim": None if queries_before_norm_cosine is None else queries_before_norm_cosine.detach().cpu().tolist(),
            "last_queries_after_norm_cosine_sim": None if queries_after_norm_cosine is None else queries_after_norm_cosine.detach().cpu().tolist(),
            "last_score_abs_max_det": None if score_abs_max_det is None else score_abs_max_det.detach().cpu().tolist(),
            "last_score_top1_gap_mean_det": None
            if score_top1_gap_mean_det is None
            else score_top1_gap_mean_det.detach().cpu().tolist(),
        }

        kept_per_image = self._token_pruning_last_kept_per_image
        before_per_image = self._token_pruning_last_before_tokens_per_image
        if kept_per_image is not None and len(kept_per_image) > 0:
            per_image_mean = [float(x.float().mean().item()) for x in kept_per_image]
            per_image_min = [int(x.min().item()) for x in kept_per_image]
            per_image_max = [int(x.max().item()) for x in kept_per_image]
            stats.update(
                {
                    "last_kept_per_image_mean": per_image_mean,
                    "last_kept_per_image_min": per_image_min,
                    "last_kept_per_image_max": per_image_max,
                }
            )
            if before_per_image is not None and len(before_per_image) == len(per_image_mean):
                stats["last_before_tokens_per_image"] = before_per_image
                stats["last_kept_ratio_per_image_mean"] = [
                    (m / b) if b > 0 else 0.0 for m, b in zip(per_image_mean, before_per_image, strict=True)
                ]
                overall_before = sum(before_per_image)
                overall_kept = sum(per_image_mean)
                stats["last_kept_ratio_overall_mean"] = (overall_kept / overall_before) if overall_before > 0 else 0.0

                # Convenience aliases for cross-camera pruning: treat the concatenated image patches as "overall".
                # When pruning is done per-image, this remains the sum over images.
                stats["last_kept_overall_mean"] = float(overall_kept)
                stats["last_kept_overall_min"] = int(sum(per_image_min))
                stats["last_kept_overall_max"] = int(sum(per_image_max))

        return stats

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_position_ids = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        if self._last_prefix_unpruned_len is None:
            raise RuntimeError("Expected `_last_prefix_unpruned_len` to be set by `embed_prefix()`.")
        suffix_len = suffix_pad_masks.shape[1]
        suffix_position_ids = (
            torch.arange(suffix_len, device=pad_masks.device, dtype=torch.long)[None, :].expand(pad_masks.shape[0], -1)
            + int(self._last_prefix_unpruned_len)
        )
        position_ids = torch.cat([prefix_position_ids.to(dtype=torch.long), suffix_position_ids], dim=1)

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_position_ids = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids.to(dtype=torch.long),
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        if self._last_prefix_unpruned_len is None:
            raise RuntimeError("Expected `_last_prefix_unpruned_len` to be set by `embed_prefix()`.")
        suffix_len = suffix_pad_masks.shape[1]
        position_ids = (
            torch.arange(suffix_len, device=suffix_pad_masks.device, dtype=torch.long)[None, :].expand(batch_size, -1)
            + int(self._last_prefix_unpruned_len)
        )

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
