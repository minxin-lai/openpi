from collections.abc import Sequence
import os
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        profile_enabled = os.getenv("OPENPI_INFER_PROFILE") not in (None, "", "0", "false", "False", "no")

        total_t0 = time.monotonic()

        # Copy inputs
        t0 = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        copy_input_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Input transforms
        t0 = time.monotonic()
        inputs = self._input_transform(inputs)
        input_transform_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Batch & device move
        t0 = time.monotonic()
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        batch_to_device_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = (
                torch.from_numpy(noise).to(self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(noise)
            )
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        t0 = time.monotonic()
        observation = _model.Observation.from_dict(inputs)
        observation_build_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Model sampling (only normal inference timing)
        # IMPORTANT: Avoid extra prefix computations during profiling to prevent large overhead.
        # We intentionally skip any calls to estimate_original_prefix_len/embed_prefix here.

        # Core model forward (measure true device compute time)
        t2 = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        # Ensure device-side work is completed so infer_ms reflects GPU/TPU compute,
        # not deferred into host_copy via implicit synchronization.
        try:
            if self._is_pytorch_model:
                if torch.cuda.is_available() and str(self._pytorch_device).startswith("cuda"):
                    torch.cuda.synchronize(torch.device(self._pytorch_device))
            else:
                # For JAX, block on the primary array(s) produced by the model.
                _ = jax.tree.map(
                    lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                    outputs,
                )
        except Exception:
            # Never let profiling sync interfere with inference
            pass
        model_time = time.monotonic() - t2

        # Host copy
        t0 = time.monotonic()
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        host_copy_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Output transforms (no telemetry timing)
        t0 = time.monotonic()
        outputs = self._output_transform(outputs)
        output_transform_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Timing payload
        if profile_enabled:
            # Measure core total first so any extra lightweight diagnostics can be excluded from totals.
            total_ms = (time.monotonic() - total_t0) * 1000.0
            timing: dict[str, float] = {
                # Backward-compatible overall model call latency
                "infer_ms": model_time * 1000.0,
                # Detailed breakdown
                "copy_input_ms": copy_input_ms,
                "input_transform_ms": input_transform_ms,
                "batch_to_device_ms": batch_to_device_ms,
                "observation_build_ms": observation_build_ms,
                "host_copy_ms": host_copy_ms,
                "output_transform_ms": output_transform_ms,
                "total_ms": total_ms,
            }
            # Attach pruning overhead measured inside embed_prefix (no extra passes)
            try:
                prune_enabled: bool | None = None
                if hasattr(self._model, "token_pruning_enabled"):
                    prune_enabled = bool(getattr(self._model, "token_pruning_enabled"))  # type: ignore[assignment]
                profile_prune = os.getenv("OPENPI_PROFILE_PRUNE_OVERHEAD") not in (None, "", "0", "false", "False", "no")
                if profile_prune and prune_enabled and (not self._is_pytorch_model):
                    prune_overhead_ms = float(getattr(self._model, "_last_prune_overhead_ms", 0.0))
                    timing["prune_overhead_ms"] = prune_overhead_ms
                    timing["infer_wo_prune_ms"] = max(0.0, timing["infer_ms"] - prune_overhead_ms)
            except Exception:
                pass
            # Attach lightweight token statistics without incurring heavy recompute.
            # Compute after total_ms so their (small) overhead is not included in totals.
            try:
                if not self._is_pytorch_model and hasattr(self._model, "fast_prefix_stats"):
                    stats = self._model.fast_prefix_stats(observation)  # type: ignore[attr-defined]
                    # Only counts/ratios, no timing fields added.
                    for k in ("prefix_orig_len", "prefix_len", "prefix_keep_ratio"):
                        if k in stats:
                            timing[k] = float(stats[k])
            except Exception:
                pass
            # Attach pruning diagnostics only when pruning is enabled
            # Requirement: when pruning is OFF, we should not display pruning info.
            try:
                prune_enabled: bool | None = None
                if hasattr(self._model, "token_pruning_enabled"):
                    prune_enabled = bool(getattr(self._model, "token_pruning_enabled"))  # type: ignore[assignment]
                if prune_enabled:
                    timing["prune_enabled"] = True
                    if hasattr(self._model, "token_prune_ratio"):
                        # normalize to float in [0,1], but avoid crashing on unexpected types
                        ratio = float(getattr(self._model, "token_prune_ratio"))  # type: ignore[arg-type]
                        timing["prune_ratio"] = max(0.0, min(1.0, ratio))
            except Exception:
                # Do not let diagnostics break inference
                pass
            outputs["policy_timing"] = timing
        else:
            outputs["policy_timing"] = {"infer_ms": model_time * 1000.0}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        # Persist as a 0-d object array so np.load(..., allow_pickle=True).item() returns the dict
        obj = np.empty((), dtype=object)
        obj[()] = data
        np.save(output_path, obj, allow_pickle=True)
        return results
