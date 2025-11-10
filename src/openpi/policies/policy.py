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
        profile = bool(int(os.environ.get("OPENPI_INFER_PROFILE", "0")))
        stage_times: dict[str, float] = {}

        # Copy + input transforms
        t0 = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        stage_times["input_ms"] = (time.monotonic() - t0) * 1000 if profile else 0.0

        # Batch & device move
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        # Optional: measure prefix embedding length/time (JAX models only)
        prefix_len = None
        if profile and not self._is_pytorch_model:
            try:
                # Run embed_prefix in eval mode to avoid ST path; minimal overhead for telemetry
                model = self._model
                model.eval()
                t1 = time.monotonic()
                prefix_tokens, prefix_mask, _ = model.embed_prefix(observation)
                stage_times["prefix_ms"] = (time.monotonic() - t1) * 1000
                prefix_len = int(prefix_tokens.shape[1])
                # Original length (unpruned) and keep ratio
                t1b = time.monotonic()
                prefix_orig_len = int(model.estimate_original_prefix_len(observation))
                stage_times["prefix_orig_ms"] = (time.monotonic() - t1b) * 1000
                if prefix_orig_len > 0:
                    stage_times["prefix_keep_ratio"] = prefix_len / float(prefix_orig_len)
            except Exception:  # pragma: no cover
                pass

        # Model sampling
        t2 = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - t2

        # Host copy
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Output transforms
        t3 = time.monotonic()
        outputs = self._output_transform(outputs)
        stage_times["output_ms"] = (time.monotonic() - t3) * 1000 if profile else 0.0

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
            # Keep policy timing minimal and generic; do not merge model-internal component timings

            # Attach pruning overhead measured inside embed_prefix (no extra passes)
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                timing["gpu0_mem_mb"] = float(mem.used) / (1024 * 1024)
            except Exception:
                pass
        outputs["policy_timing"] = timing
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

        np.save(output_path, np.asarray(data))
        return results
