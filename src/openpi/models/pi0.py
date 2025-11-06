import logging
import os
import time

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        # Token pruning configuration (LightVLA-style, parameter-free)
        self.token_pruning_enabled = getattr(config, "token_pruning_enabled", False)
        self.token_prune_ratio = getattr(config, "token_prune_ratio", 0.25)
        self.token_prune_tau = getattr(config, "token_prune_tau", 1.0)
        self.token_prune_min_keep = getattr(config, "token_prune_min_keep", 1)
        self.token_prune_noise_scale = getattr(config, "token_prune_noise_scale", 0.0)
        self.token_prune_scoring = getattr(config, "token_prune_scoring", "prompt_attn")
        # Profiling: store last measured pruning overhead (ms) for telemetry
        self._last_prune_overhead_ms: float = 0.0
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        # Image encoder (SigLIP). Keep track of patch size for fast token estimates.
        siglip_variant = "So400m/14"
        self._siglip_patch_size = _siglip.decode_variant(siglip_variant)["patch_size"]  # (ph, pw)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant=siglip_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def fast_prefix_stats(self, obs: _model.Observation) -> dict:
        """Fast, zero-forward token statistics for prefix.

        Computes the original (unpruned) and estimated pruned prefix token lengths using
        only tensor shapes and configured patch/pruning parameters.

        Returns a dict with keys:
          - prefix_orig_len: int
          - prefix_len: int (accounts for pruning if enabled)
          - prefix_keep_ratio: float in [0,1]
        """
        # Image tokens per camera: (H // ph) * (W // pw)
        ph, pw = self._siglip_patch_size
        orig_img_tokens_total = 0
        for name, img in obs.images.items():
            # Expect shape [B, H, W, C]; compute tokens per image for the first (and only) batch element.
            _, H, W, _ = img.shape
            tokens = int((H // ph) * (W // pw))
            orig_img_tokens_total += tokens

        # Text tokens length equals the sequence length dimension (no need to run embed).
        text_tokens = 0
        if obs.tokenized_prompt is not None:
            text_tokens = int(obs.tokenized_prompt.shape[1])

        prefix_orig_len = int(orig_img_tokens_total + text_tokens)

        # Apply pruning estimate per camera deterministically, language tokens are kept.
        if getattr(self, "token_pruning_enabled", False):
            pr = float(getattr(self, "token_prune_ratio", 1.0))
            min_keep = int(getattr(self, "token_prune_min_keep", 1))
            pruned_img_total = 0
            for name, img in obs.images.items():
                _, H, W, _ = img.shape
                s = int((H // ph) * (W // pw))
                k = max(min_keep, int(s * pr))
                pruned_img_total += k
            prefix_len = int(pruned_img_total + text_tokens)
        else:
            prefix_len = int(prefix_orig_len)

        keep = float(prefix_len) / float(prefix_orig_len) if prefix_orig_len > 0 else 0.0
        return {
            "prefix_orig_len": prefix_orig_len,
            "prefix_len": prefix_len,
            "prefix_keep_ratio": keep,
        }

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, rng: at.KeyArrayLike | None = None
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # Determine whether to profile pruning overhead (best-effort, inference-only)
        def _truthy(x: str | None) -> bool:
            return x is not None and x.lower() not in ("", "0", "false", "no", "off")
        profile_overhead = _truthy(os.getenv("OPENPI_INFER_PROFILE")) and _truthy(os.getenv("OPENPI_PROFILE_PRUNE_OVERHEAD"))
        prune_overhead_accum = 0.0
        input_mask = []
        ar_mask: list[bool] = []
        tokens_out = []

        # Collect per-camera image tokens (to optionally prune later)
        image_tokens_list = []
        image_masks_list = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            image_mask = einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            )
            image_tokens_list.append(image_tokens)
            image_masks_list.append(image_mask)

        # Optionally compute prompt embeddings for pruning guidance (do not append yet)
        prompt_embed = None
        prompt_mask = None
        if obs.tokenized_prompt is not None:
            prompt_embed = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            prompt_mask = obs.tokenized_prompt_mask

        # Helpers for pruning
        def _rms_norm(x, eps=1e-6):
            dtype = x.dtype
            var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
            x = x * jax.lax.rsqrt(var + eps)
            return x.astype(dtype)

        def _prune_tokens(patches, mask, prompt, rng_local: at.KeyArrayLike | None):
            # patches: [b, s, d], mask: [b, s], prompt: [b, t, d] | None
            # Returns selected patches [b, k, d] and mask [b, k]
            s = patches.shape[1]
            k = max(self.token_prune_min_keep, int(s * float(self.token_prune_ratio)))

            # Prompt-guided similarity, fallback to L2 norm
            if prompt is not None:
                mode = self.token_prune_scoring
                if mode == "prompt_attn":
                    # LightVLA-style attention scoring: attend from patches to prompt tokens
                    patches_n = _rms_norm(patches)
                    prompts_n = _rms_norm(prompt)
                    d = patches_n.shape[-1]
                    # attn weights over prompt tokens per patch
                    logits = jnp.einsum("bsd,btd->bst", patches_n, prompts_n) / jnp.sqrt(jnp.asarray(d, dtype=patches_n.dtype))
                    attn = jax.nn.softmax(logits, axis=-1)
                    # aggregated prompt features per patch
                    q = jnp.einsum("bst,btd->bsd", attn, prompts_n)
                    q = _rms_norm(q)
                    score = jnp.einsum("bsd,bsd->bs", q, patches_n) / jnp.sqrt(jnp.asarray(d, dtype=patches_n.dtype))
                elif mode == "prompt_mean":
                    prompt_mean = jnp.mean(prompt, axis=1)
                    score = jnp.einsum("bsd,bd->bs", _rms_norm(patches), _rms_norm(prompt_mean))
                else:
                    # Fallback to L2 norm
                    score = jnp.linalg.norm(patches, axis=-1)
            else:
                score = jnp.linalg.norm(patches, axis=-1)

            # Optionally add small noise during training to encourage exploration.
            # Note: embed_prefix does not receive an RNG; we skip true randomness here to keep determinism.
            # Users can emulate noise by data augmentation upstream.
            train_mode = not self.deterministic
            if train_mode and self.token_prune_noise_scale > 0.0:
                # Add (optional) Gumbel noise for exploration during training when RNG is provided.
                # Falls back to deterministic pseudo-noise if RNG is None.
                if rng_local is not None:
                    u = jax.random.uniform(rng_local, shape=score.shape, minval=0.0, maxval=1.0, dtype=jnp.float32)
                    # Standard Gumbel(0, 1)
                    gumbel = -jnp.log(-jnp.log(jnp.clip(u, 1e-6, 1.0 - 1e-6)))
                    score = score + self.token_prune_noise_scale * gumbel
                else:
                    noise = jnp.tanh(score)
                    score = score + self.token_prune_noise_scale * noise

            # Hard Top-K indices (deterministic)
            idx = jnp.argsort(score, axis=1)[:, -k:]
            idx_exp = idx[:, :, None]
            patches_hard = jnp.take_along_axis(patches, idx_exp, axis=1)
            mask_hard = jnp.take_along_axis(mask, idx, axis=1)

            # Straight-through gradient: soften with temperature during training
            if train_mode and self.token_prune_tau > 0:
                w = jax.nn.softmax(score / self.token_prune_tau, axis=-1)  # [b, s]
                # Expected patch as soft mixture; replicate across k slots
                patch_soft = jnp.einsum("bs,bsd->bd", w, patches)
                patches_soft = jnp.repeat(patch_soft[:, None, :], k, axis=1)
                patches_sel = patches_hard + patches_soft - jax.lax.stop_gradient(patches_soft)
            else:
                patches_sel = patches_hard

            return patches_sel, mask_hard

        # Apply pruning per camera if enabled
        if self.token_pruning_enabled:
            # Split RNG per camera stream if provided.
            keys = None
            if rng is not None:
                keys = jax.random.split(rng, len(image_tokens_list))
            for i, (patches, mask) in enumerate(zip(image_tokens_list, image_masks_list, strict=True)):
                if profile_overhead:
                    t_prof = time.monotonic()
                pruned_patches, pruned_mask = _prune_tokens(patches, mask, prompt_embed, None if keys is None else keys[i])
                # Block on device to get accurate timing of pruning compute only
                if profile_overhead:
                    try:
                        _ = pruned_patches.block_until_ready() if hasattr(pruned_patches, "block_until_ready") else pruned_patches
                        _ = pruned_mask.block_until_ready() if hasattr(pruned_mask, "block_until_ready") else pruned_mask
                    except Exception:
                        pass
                    prune_overhead_accum += (time.monotonic() - t_prof)
                tokens_out.append(pruned_patches)
                input_mask.append(pruned_mask)
                ar_mask += [False] * pruned_patches.shape[1]
        else:
            for patches, mask in zip(image_tokens_list, image_masks_list, strict=True):
                tokens_out.append(patches)
                input_mask.append(mask)
                ar_mask += [False] * patches.shape[1]

        # Append language (aka tokenized inputs) without pruning
        if prompt_embed is not None:
            tokens_out.append(prompt_embed)
            input_mask.append(prompt_mask)
            ar_mask += [False] * prompt_embed.shape[1]

        tokens = jnp.concatenate(tokens_out, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        # Store measured prune overhead (ms) for telemetry consumers
        try:
            self._last_prune_overhead_ms = float(prune_overhead_accum * 1000.0) if profile_overhead and self.token_pruning_enabled else 0.0
        except Exception:
            self._last_prune_overhead_ms = 0.0
        return tokens, input_mask, ar_mask

    def pruning_summary(self, obs: _model.Observation, rng: at.KeyArrayLike | None = None) -> dict:
        """
        Returns a summary dict with original vs pruned prefix token lengths.
        Useful for training/inference monitoring of LightVLA-style pruning.

        Keys:
          - original_prefix_len: int
          - pruned_prefix_len: int
          - reduction: float (fractional reduction >= 0)
        """
        original = self.estimate_original_prefix_len(obs)
        tokens, mask, _ = self.embed_prefix(obs, rng=rng)
        pruned = int(tokens.shape[1])
        reduce = float(original - pruned) / float(original) if original > 0 else 0.0
        return {"original_prefix_len": int(original), "pruned_prefix_len": pruned, "reduction": reduce}

    def estimate_original_prefix_len(self, obs: _model.Observation) -> int:
        """Estimate original (unpruned) prefix token length for telemetry.

        This runs the same image/text embedding as embed_prefix but without pruning and only
        returns the total sequence length. Intended for profiling/inference diagnostics.
        """
        total = 0
        # image tokens (sum across cameras)
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            total += int(image_tokens.shape[1])
        # text tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            total += int(tokenized_inputs.shape[1])
        return int(total)

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, prune_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, rng=prune_rng)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
