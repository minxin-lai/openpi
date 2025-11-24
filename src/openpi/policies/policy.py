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


def _summarize_array(arr: Any, *, max_elems: int = 8) -> dict[str, Any]:
    """Lightweight summary of a NumPy/JAX/Torch array for alignment debugging."""
    try:
        if isinstance(arr, torch.Tensor):
            a = arr.detach().cpu().numpy()
        else:
            a = np.asarray(arr)
    except Exception:
        return {"repr": repr(arr)}

    flat = a.reshape(-1) if a.size else a
    return {
        "shape": tuple(a.shape),
        "dtype": str(a.dtype),
        "min": float(a.min()) if a.size else 0.0,
        "max": float(a.max()) if a.size else 0.0,
        "mean": float(a.mean()) if a.size else 0.0,
        "first": flat[:max_elems].tolist(),
    }


_ALIGN_STAGE_MAP: dict[str, int] = {
    # JAX stages
    "jax_pre_obs": 1,
    "jax_post_yuanluo_inputs": 2,
    "jax_post_normalize": 3,
    "jax_post_resize": 4,
    "jax_post_pad": 5,
    "jax_model_input": 6,
    "jax_model_out_normalized": 7,
    "jax_post_unnormalize": 8,
    "jax_post_absolute_actions": 9,
    "jax_post_output_transform": 10,
    # Realtime stages
    "rt_pre_obs": 1,
    "rt_post_yuanluo_inputs": 2,
    "rt_post_normalize": 3,
    "rt_post_resize": 4,
    "rt_post_pad": 5,
    "rt_model_input": 6,
    "rt_model_out_normalized": 7,
    "rt_post_unnormalize": 8,
    "rt_post_absolute_actions": 9,
    "rt_post_output_transform": 10,
}


def _log_alignment_snapshot(tag: str, data: dict[str, Any], *, step: int | None = None) -> None:
    """Log a compact, multi-line snapshot for JAX/Realtime alignment debugging."""
    if os.getenv("OPENPI_ALIGN_DEBUG") in (None, "", "0", "false", "False", "no"):
        return

    try:
        state = data.get("state", None)
        actions = data.get("actions", None)
        image = data.get("image", None)

        stage_idx = _ALIGN_STAGE_MAP.get(tag)
        meta_parts: list[str] = []
        if step is not None:
            meta_parts.append(f"step={step}")
        if stage_idx is not None:
            meta_parts.append(f"stage={stage_idx}")
        # Strip implementation prefix in the printed tag so that JAX / Realtime
        # logs share the same human-visible stage name.
        printable_tag = tag
        for _prefix in ("jax_", "rt_"):
            if printable_tag.startswith(_prefix):
                printable_tag = printable_tag[len(_prefix) :]
                break

        header = f"[ALIGN] {printable_tag}"
        if meta_parts:
            header += " (" + ", ".join(meta_parts) + ")"

        # Add a leading separator line for readability between stages.
        lines: list[str] = ["", header]

        # State summary (split into multiple short lines)
        if state is not None:
            s = _summarize_array(state, max_elems=8)
            if "shape" in s:
                lines.append("State:")
                lines.append(f"  shape={s['shape']} dtype={s['dtype']}")
                lines.append(f"  range=[{s['min']:.6f},{s['max']:.6f}]")
                lines.append(f"  first={s['first']}")

        # Actions summary (also split for readability)
        if actions is not None:
            tag_is_post_transform = printable_tag == "post_output_transform"
            if tag_is_post_transform:
                try:
                    if isinstance(actions, torch.Tensor):
                        arr = actions.detach().cpu().numpy()
                    else:
                        arr = np.asarray(actions)
                except Exception:
                    arr = None
                if arr is not None:
                    lines.append("Actions:")
                    lines.append(f"  shape={arr.shape} dtype={arr.dtype}")
                    lines.append(f"  values={arr.tolist()}")
                else:
                    lines.append("Actions: <unprintable>")
            else:
                a = _summarize_array(actions, max_elems=8)
                if "shape" in a:
                    lines.append("Actions:")
                    lines.append(f"  shape={a['shape']} dtype={a['dtype']}")
                    lines.append(f"  range=[{a['min']:.6f},{a['max']:.6f}]")
                    lines.append(f"  first={a['first']}")

        # Image summary: include basic stats per camera
        if isinstance(image, dict):
            for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
                if cam in image:
                    s = _summarize_array(image[cam], max_elems=4)
                    if "shape" in s:
                        if "Images:" not in lines:
                            lines.append("Images:")
                        lines.append(f"  {cam}:")
                        lines.append(f"    shape={s['shape']} dtype={s['dtype']}")
                        lines.append(
                            f"    range=[{s['min']:.6f},{s['max']:.6f}] "
                            f"mean={s['mean']:.6f}"
                        )
                        if s.get("first"):
                            lines.append(f"    first={s['first']}")

        # Trailing blank line to visually separate stages in logs.
        lines.append("")

        logging.info("\n".join(lines))

    except Exception:
        # Never let debug logging break inference.
        logging.exception("Failed to log alignment snapshot for tag=%s", tag)


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

    def _print_debug_info(self, tag: str, data: dict):
        """打印调试信息"""
        debug_enabled = os.getenv("OPENPI_DEBUG_TRANSFORMS") not in (None, "", "0", "false", "False", "no")
        if not debug_enabled:
            return

        print(f"\n{'='*70}")
        print(f"[DEBUG JAX] {tag}")
        print(f"{'='*70}")

        def _print_array(name: str, arr):
            if arr is None:
                return
            try:
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().numpy()
                elif hasattr(arr, 'shape'):  # JAX array
                    arr = np.asarray(arr)
                else:
                    arr = np.asarray(arr)

                print(f"  {name}:")
                print(f"    dtype: {arr.dtype}, shape: {arr.shape}")

                # Skip statistics for non-numeric types
                if arr.dtype.kind in ('U', 'S', 'O'):  # Unicode, bytes, object
                    if arr.size <= 3:
                        print(f"    values: {arr.flatten().tolist()}")
                    else:
                        print(f"    first 3: {arr.flatten()[:3].tolist()}")
                elif arr.size > 0:
                    print(f"    min: {arr.min():.6f}, max: {arr.max():.6f}, mean: {arr.mean():.6f}")
                    if arr.size <= 10:
                        print(f"    values: {arr.flatten().tolist()}")
                    else:
                        print(f"    first 8: {arr.flatten()[:8].tolist()}")
            except Exception as e:
                print(f"  {name}: <error printing: {e}>")

        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    _print_array(f"  {k}", v)
            else:
                _print_array(key, value)
        print()

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Debug print: 2. 转换后
        self._print_debug_info("2. 转换后 (输入模型前)", inputs)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
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
