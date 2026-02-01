import json
import logging
import math
import os
import copy
from typing import Any, Iterable

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

logger = logging.getLogger(__name__)

# Debug token/KV introspection (Pi0.5 serve-only)
#
# This module can emit a single-line JSON payload prefixed by "OPENPI_DEBUG " from `PI0Pytorch.sample_actions`.
# The payload is designed to explain:
# - How vision token pruning changes prefix_len/full_len,
# - What past_key_values (KV cache) looks like after the prefix pass,
# - How the suffix model consumes prefix KV and attends over (prefix_len + suffix_len).
#
# Recommended usage:
# - Use `scripts/serve_policy.py --debug-token ...` (preferred) or set env vars directly:
#     OPENPI_DEBUG_TOKEN=1
#     OPENPI_DEBUG_MAX_INFER=1
#     OPENPI_DEBUG_VARIANT=baseline|vla_opt
#     OPENPI_DEBUG_KV_LAYERS=ends|all|0,8,16
#     OPENPI_DEBUG_KV_COMPARE=1   # optional, best-effort, loose tolerance
#
# Notes:
# - For suffix step0, we run an extra forward with `use_cache=True` on a cloned cache object so debug does not mutate the
#   real prefix cache used by the denoise loop.


def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _tensor_bytes(t: Tensor) -> int:
    return int(t.numel() * t.element_size())


def _tensor_info(t: Tensor | None) -> dict[str, Any] | None:
    if t is None or not torch.is_tensor(t):
        return None
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "bytes": _tensor_bytes(t),
    }


def _guess_seq_dim(t: Tensor, expected_seq_len: int | None) -> int | None:
    if expected_seq_len is None:
        return None
    matches = [i for i, s in enumerate(t.shape) if int(s) == int(expected_seq_len)]
    # Prefer a later dim (typical kv: (B, H, S, D) or (B, H, D, S)).
    if matches:
        return int(matches[-1])
    return None


def _iter_past_key_values_layers(past_key_values: Any) -> Iterable[tuple[int, Tensor | None, Tensor | None]]:
    if past_key_values is None:
        return

    # HF Cache-like objects often expose key_cache/value_cache.
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if isinstance(key_cache, (list, tuple)) and isinstance(value_cache, (list, tuple)):
        for i, (k, v) in enumerate(zip(key_cache, value_cache)):
            yield int(i), (k if torch.is_tensor(k) else None), (v if torch.is_tensor(v) else None)
        return

    # Legacy cache: list/tuple of (k, v).
    if isinstance(past_key_values, (list, tuple)):
        for i, layer in enumerate(past_key_values):
            k = v = None
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                if torch.is_tensor(layer[0]):
                    k = layer[0]
                if torch.is_tensor(layer[1]):
                    v = layer[1]
            yield int(i), k, v
        return

    # Fallback: try iterating.
    try:
        for i, layer in enumerate(list(past_key_values)):
            k = v = None
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                if torch.is_tensor(layer[0]):
                    k = layer[0]
                if torch.is_tensor(layer[1]):
                    v = layer[1]
            yield int(i), k, v
    except Exception:
        return


def _clone_cache_for_debug(past_key_values: Any) -> Any:
    """
    Create a best-effort clone of HF cache objects to avoid mutating the original `past_key_values`.

    For `transformers.cache_utils.DynamicCache` (and similar), a shallow clone of the `key_cache`/`value_cache` lists is
    sufficient: the cache update replaces list entries via `torch.cat(...)` rather than in-place editing of tensors.
    """
    if past_key_values is None:
        return None

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list):
        try:
            cloned = type(past_key_values)()
        except Exception:
            cloned = object.__new__(type(past_key_values))
        try:
            cloned.key_cache = list(key_cache)
            cloned.value_cache = list(value_cache)
        except Exception:
            return past_key_values
        # Preserve seen token count for caches that track it (e.g. DynamicCache).
        try:
            if hasattr(past_key_values, "_seen_tokens"):
                cloned._seen_tokens = int(getattr(past_key_values, "_seen_tokens"))  # noqa: SLF001
        except Exception:
            pass
        return cloned

    # Legacy list-of-tuples cache: shallow copy the list so mutations don't affect caller.
    if isinstance(past_key_values, list):
        return list(past_key_values)

    # Generic fallback (may still share internals; debug-only).
    try:
        return copy.copy(past_key_values)
    except Exception:
        return past_key_values


def _parse_debug_kv_layers(num_layers: int) -> list[int]:
    spec = os.environ.get("OPENPI_DEBUG_KV_LAYERS", "ends").strip().lower()
    if spec in {"", "ends", "end"}:
        if num_layers <= 0:
            return []
        return sorted({0, num_layers - 1})
    if spec == "all":
        return list(range(num_layers))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    # Clamp.
    clamped = [i for i in out if 0 <= i < num_layers]
    return sorted(set(clamped))


def _summarize_past_key_values(
    past_key_values: Any,
    *,
    expected_seq_len: int | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": type(past_key_values).__name__ if past_key_values is not None else None,
        "num_layers": None,
        "total_bytes": 0,
        "layers": {},
        "seq_len": None,
        "seq_dim": None,
    }

    layers_list = list(_iter_past_key_values_layers(past_key_values))
    num_layers = len(layers_list)
    summary["num_layers"] = int(num_layers)

    layer_indices = _parse_debug_kv_layers(num_layers)
    total_bytes = 0
    seq_len = None
    seq_dim = None
    for i, k, v in layers_list:
        if torch.is_tensor(k):
            total_bytes += _tensor_bytes(k)
        if torch.is_tensor(v):
            total_bytes += _tensor_bytes(v)

        if i not in layer_indices:
            continue

        layer_d: dict[str, Any] = {"k": _tensor_info(k), "v": _tensor_info(v), "seq_len": None, "seq_dim": None}
        if torch.is_tensor(k) and k.ndim >= 2:
            guessed_dim = _guess_seq_dim(k, expected_seq_len)
            if guessed_dim is None:
                # Best-effort default: kv often uses a sequence dim near the end.
                guessed_dim = int(-2 if k.ndim >= 3 else -1)
            try:
                guessed_len = int(k.shape[guessed_dim])
            except Exception:
                guessed_len = None
            layer_d["seq_dim"] = int(guessed_dim)
            layer_d["seq_len"] = guessed_len
            if seq_len is None:
                seq_len = guessed_len
                seq_dim = int(guessed_dim)

        summary["layers"][str(i)] = layer_d

    summary["total_bytes"] = int(total_bytes)
    summary["seq_len"] = seq_len
    summary["seq_dim"] = seq_dim
    return summary


def _count_parameters(module: nn.Module) -> int:
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except Exception:
        return -1


def _cfg_view(cfg: Any) -> dict[str, Any]:
    fields = [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "_attn_implementation",
    ]
    out: dict[str, Any] = {}
    for f in fields:
        out[f] = getattr(cfg, f, None)
    return out


def _kv_prefix_preserved_check(
    in_pkv: Any,
    out_pkv: Any,
    *,
    prefix_len: int,
    full_len: int,
) -> dict[str, Any]:
    if not _env_flag("OPENPI_DEBUG_KV_COMPARE"):
        return {"enabled": False}
    result: dict[str, Any] = {"enabled": True, "layers": {}}

    in_layers = {i: (k, v) for i, k, v in _iter_past_key_values_layers(in_pkv)}
    out_layers = {i: (k, v) for i, k, v in _iter_past_key_values_layers(out_pkv)}
    num_layers = min(len(in_layers), len(out_layers))
    if num_layers <= 0:
        return result

    check_layers = _parse_debug_kv_layers(num_layers)
    for i in check_layers:
        in_k, in_v = in_layers.get(i, (None, None))
        out_k, out_v = out_layers.get(i, (None, None))
        layer_res: dict[str, Any] = {"ok": None, "max_abs_diff": None}
        if not (torch.is_tensor(in_k) and torch.is_tensor(out_k)):
            result["layers"][str(i)] = layer_res
            continue

        in_seq_dim = _guess_seq_dim(in_k, prefix_len)
        out_seq_dim = _guess_seq_dim(out_k, full_len)
        if in_seq_dim is None or out_seq_dim is None:
            result["layers"][str(i)] = layer_res
            continue

        try:
            # Align shapes by slicing the output cache prefix along the inferred sequence dim.
            out_k_prefix = out_k.movedim(out_seq_dim, -1)[..., :prefix_len]
            in_k_seq = in_k.movedim(in_seq_dim, -1)
            if out_k_prefix.shape != in_k_seq.shape:
                result["layers"][str(i)] = layer_res
                continue

            diff = (out_k_prefix.to(torch.float32) - in_k_seq.to(torch.float32)).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            layer_res["max_abs_diff"] = max_abs
            # bf16 tolerance: loose by design (debug signal, not a unit test).
            layer_res["ok"] = bool(max_abs < 5e-2)
        except Exception:
            pass
        result["layers"][str(i)] = layer_res

    return result


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
        # Keep an eager (non-compiled) handle for tracing/debugging; torch.compile often disables forward hooks.
        self._sample_actions_eager = self.sample_actions
        compile_flag = os.environ.get("OPENPI_TORCH_COMPILE", "1").strip().lower()
        use_compile = compile_flag not in {"0", "false", "no", "n", "off"}
        compile_mode = os.environ.get("OPENPI_TORCH_COMPILE_MODE", "max-autotune").strip()
        if use_compile:
            self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)
            logging.info("torch.compile enabled for PI0Pytorch.sample_actions (mode=%s)", compile_mode)
        else:
            logging.info("torch.compile disabled for PI0Pytorch.sample_actions (OPENPI_TORCH_COMPILE=%s)", compile_flag)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        debug_ctx = getattr(self, "_openpi_debug_ctx", None)
        for view_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
            if isinstance(debug_ctx, dict):
                views = debug_ctx.setdefault("token", {}).setdefault("views", [])
                while len(views) <= view_idx:
                    views.append({})
                views[view_idx]["view_idx"] = int(view_idx)
                ste_handle = getattr(self, "_vla_opt_ste_prune_handle", None)
                if ste_handle is not None:
                    # Reset per-view cache so we can read N(before) after embed_image().
                    try:
                        ste_handle.last_scores = None
                        ste_handle.last_idx = None
                        ste_handle.last_soft_mask = None
                        ste_handle.last_hard_mask = None
                    except Exception:
                        pass

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

            if isinstance(debug_ctx, dict):
                try:
                    views = debug_ctx.setdefault("token", {}).setdefault("views", [])
                    views[view_idx]["img_tokens_after_prune"] = int(num_img_embs)
                    ste_handle = getattr(self, "_vla_opt_ste_prune_handle", None)
                    n_before = None
                    last_scores = getattr(ste_handle, "last_scores", None) if ste_handle is not None else None
                    if torch.is_tensor(last_scores):
                        n_before = int(last_scores.shape[1])
                    views[view_idx]["img_tokens_before_prune_at_prune_layer"] = n_before
                except Exception:
                    pass

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

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

        if isinstance(debug_ctx, dict):
            try:
                views = debug_ctx.setdefault("token", {}).setdefault("views", [])
                after_list = [v.get("img_tokens_after_prune", None) for v in views]
                before_list = [v.get("img_tokens_before_prune_at_prune_layer", None) for v in views]
                debug_ctx["token"]["img_tokens_per_view_after_prune"] = after_list
                debug_ctx["token"]["img_tokens_per_view_before_prune_at_prune_layer"] = before_list
                debug_ctx["token"]["img_tokens_total_after_prune"] = int(
                    sum(int(x) for x in after_list if isinstance(x, int))
                )
                debug_ctx["token"]["img_tokens_total_before_prune_at_prune_layer"] = int(
                    sum(int(x) for x in before_list if isinstance(x, int))
                )
            except Exception:
                pass

        return embs, pad_masks, att_masks

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

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
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
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

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
        debug_enabled = _env_flag("OPENPI_DEBUG_TOKEN")
        do_debug_print = False
        debug_payload: dict[str, Any] | None = None
        if debug_enabled:
            try:
                max_infer = int(os.environ.get("OPENPI_DEBUG_MAX_INFER", "1"))
            except Exception:
                max_infer = 1
            infer_idx = int(getattr(self, "_openpi_debug_infer_idx", 0))
            setattr(self, "_openpi_debug_infer_idx", infer_idx + 1)
            do_debug_print = infer_idx < max_infer
            if do_debug_print:
                debug_payload = {
                    "infer_idx": infer_idx,
                    "variant": os.environ.get("OPENPI_DEBUG_VARIANT", "").strip() or None,
                    "token": {},
                    "model_cfg": {},
                    "kv": {},
                    "attn_mask": {},
                    "vla_opt": {},
                }
                debug_payload["token"]["action_horizon"] = int(getattr(self.config, "action_horizon", -1))
                debug_payload["token"]["num_steps"] = int(num_steps)

        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        if do_debug_print and debug_payload is not None:
            try:
                prefix_model = self.paligemma_with_expert.paligemma.language_model
                suffix_model = self.paligemma_with_expert.gemma_expert.model
                prefix_cfg = getattr(prefix_model, "config", None)
                suffix_cfg = getattr(suffix_model, "config", None)
                prefix_view = _cfg_view(prefix_cfg)
                suffix_view = _cfg_view(suffix_cfg)
                arch_match = {k: (prefix_view.get(k) == suffix_view.get(k)) for k in sorted(set(prefix_view) | set(suffix_view))}
                debug_payload["model_cfg"] = {
                    "prefix": {
                        **prefix_view,
                        "params": _count_parameters(prefix_model),
                    },
                    "suffix": {
                        **suffix_view,
                        "params": _count_parameters(suffix_model),
                    },
                    "arch_match": arch_match,
                }
            except Exception:
                pass

            try:
                ste_handle = getattr(self, "_vla_opt_ste_prune_handle", None)
                debug_payload["vla_opt"]["ste"] = {
                    "enabled": bool(ste_handle is not None),
                    "k": int(getattr(ste_handle, "k", 0)) if ste_handle is not None else None,
                    "tau": float(getattr(ste_handle, "tau", 0.0)) if ste_handle is not None else None,
                    "stage": getattr(ste_handle, "stage", None) if ste_handle is not None else None,
                    "prune_layer_resolved": getattr(self, "_vla_opt_ste_prune_layer_resolved", None),
                    "vision_num_layers": getattr(self, "_vla_opt_ste_prune_num_vision_layers", None),
                }
            except Exception:
                pass
            try:
                stage_a_handle = getattr(self, "_vla_opt_stage_a_handle", None)
                debug_payload["vla_opt"]["film"] = {"enabled": bool(stage_a_handle is not None)}
            except Exception:
                pass

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # Debug context shared with embed_prefix/denoise_step (only when explicitly enabled).
        if do_debug_print and debug_payload is not None:
            setattr(self, "_openpi_debug_ctx", debug_payload)
            try:
                debug_payload["token"]["num_views"] = int(len(images))
            except Exception:
                pass

        try:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            if do_debug_print and debug_payload is not None:
                try:
                    debug_payload["token"]["prefix_embs_shape"] = list(prefix_embs.shape)
                    debug_payload["token"]["prefix_pad_masks_shape"] = list(prefix_pad_masks.shape)
                except Exception:
                    pass
        finally:
            # Always clear the embed_prefix per-view hooks before entering the denoise loop.
            pass

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        if do_debug_print and debug_payload is not None:
            try:
                debug_payload["model_cfg"].setdefault("prefix", {})["_attn_implementation_runtime"] = getattr(
                    self.paligemma_with_expert.paligemma.language_model.config, "_attn_implementation", None
                )
            except Exception:
                pass

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        if do_debug_print and debug_payload is not None:
            try:
                prefix_len = int(prefix_embs.shape[1])
                lang_len = int(lang_tokens.shape[1]) if torch.is_tensor(lang_tokens) and lang_tokens.ndim == 2 else None
                debug_payload["token"]["lang_tokens_len"] = lang_len
                debug_payload["token"]["prefix_len"] = prefix_len
                debug_payload["kv"]["prefix"] = _summarize_past_key_values(past_key_values, expected_seq_len=prefix_len)
                debug_payload["kv"]["prefix"]["python_id"] = int(id(past_key_values))
                debug_payload["kv"]["prefix"]["expected_prefix_len"] = prefix_len
                debug_payload["kv"]["prefix"]["seq_len_matches_prefix"] = bool(
                    debug_payload["kv"]["prefix"].get("seq_len", None) == prefix_len
                )
            except Exception:
                pass

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        if do_debug_print:
            setattr(self, "_openpi_debug_step_idx", 0)
        while time >= -dt / 2:
            if do_debug_print:
                setattr(self, "_openpi_debug_step_idx", int(getattr(self, "_openpi_debug_step_idx", 0)))
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
            if do_debug_print:
                setattr(self, "_openpi_debug_step_idx", int(getattr(self, "_openpi_debug_step_idx", 0)) + 1)

        if do_debug_print and debug_payload is not None:
            try:
                payload = json.dumps(debug_payload, ensure_ascii=False, sort_keys=True)
                logger.info("OPENPI_DEBUG %s", payload)
            except Exception:
                pass
        if do_debug_print:
            try:
                delattr(self, "_openpi_debug_ctx")
            except Exception:
                pass
            try:
                delattr(self, "_openpi_debug_step_idx")
            except Exception:
                pass

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

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        debug_ctx = getattr(self, "_openpi_debug_ctx", None)
        debug_step_idx = getattr(self, "_openpi_debug_step_idx", None)
        capture_step0 = isinstance(debug_ctx, dict) and debug_step_idx == 0 and not debug_ctx.get("_suffix_step0", False)
        if capture_step0:
            try:
                debug_ctx["model_cfg"].setdefault("suffix", {})["_attn_implementation_runtime"] = getattr(
                    self.paligemma_with_expert.gemma_expert.model.config, "_attn_implementation", None
                )
            except Exception:
                pass

        if capture_step0:
            debug_ctx["_suffix_step0"] = True
            try:
                prefix_len = int(prefix_pad_masks.shape[1])
                suffix_len = int(suffix_pad_masks.shape[1])
                full_len = int(prefix_len + suffix_len)
                debug_ctx["token"]["suffix_len"] = suffix_len
                debug_ctx["token"]["full_len"] = full_len
                debug_ctx["token"]["suffix_embs_shape_step0"] = list(suffix_embs.shape)
                debug_ctx["token"]["suffix_pad_masks_shape_step0"] = list(suffix_pad_masks.shape)
                debug_ctx["token"]["position_ids_shape_step0"] = list(position_ids.shape)
                debug_ctx["attn_mask"]["step0"] = {
                    "full_att_2d_masks_4d_shape": list(full_att_2d_masks_4d.shape),
                    "prefix_len": prefix_len,
                    "suffix_len": suffix_len,
                }

                # Run suffix model forward directly so we can capture its past_key_values.
                # IMPORTANT: clone cache to avoid mutating the original prefix cache in-place.
                pkv_debug = _clone_cache_for_debug(past_key_values)
                out = self.paligemma_with_expert.gemma_expert.model.forward(
                    inputs_embeds=suffix_embs,
                    attention_mask=full_att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=pkv_debug,
                    use_cache=True,
                    adarms_cond=adarms_cond,
                )
                suffix_out = out.last_hidden_state
                out_pkv = getattr(out, "past_key_values", None)

                debug_ctx["kv"]["suffix_step0"] = _summarize_past_key_values(out_pkv, expected_seq_len=full_len)
                debug_ctx["kv"]["suffix_step0"]["python_id"] = int(id(out_pkv)) if out_pkv is not None else None
                debug_ctx["kv"]["suffix_step0"]["expected_full_len"] = full_len
                debug_ctx["kv"]["suffix_step0"]["seq_len_matches_full"] = bool(
                    debug_ctx["kv"]["suffix_step0"].get("seq_len", None) == full_len
                )
                debug_ctx["kv"]["suffix_step0"]["input_prefix_cache_python_id"] = int(id(past_key_values))
                debug_ctx["kv"]["suffix_step0"]["input_prefix_seq_len"] = _summarize_past_key_values(
                    past_key_values, expected_seq_len=prefix_len
                ).get("seq_len", None)
                prefix_layers = (
                    debug_ctx.get("kv", {}).get("prefix", {}).get("num_layers", None) if isinstance(debug_ctx, dict) else None
                )
                suffix_layers = debug_ctx["kv"]["suffix_step0"].get("num_layers", None)
                debug_ctx["kv"]["suffix_step0"]["layers_match_prefix"] = (
                    prefix_layers == suffix_layers if prefix_layers is not None and suffix_layers is not None else None
                )
                debug_ctx["kv"]["suffix_step0"]["prefix_preserved"] = _kv_prefix_preserved_check(
                    past_key_values, out_pkv, prefix_len=prefix_len, full_len=full_len
                )
            except Exception:
                # Never fail inference for debug.
                out = None
        else:
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
