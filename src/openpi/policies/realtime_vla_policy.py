"""Realtime VLA policy implementation using third_party/realtime-vla."""

import os
import time
import numpy as np
import torch

from openpi.policies.policy import _log_alignment_snapshot as _log_align


class RealtimeVLAPolicy:
    """Wrapper for realtime-vla Pi0Inference engine.

    Provides a BasePolicy-compatible interface with dynamic transform chains
    for all supported robots (Aloha, DROID, Yuanluo, Libero, etc).
    """

    def __init__(
        self,
        engine,
        num_views: int,
        chunk_size: int,
        device: str,
        input_transforms,
        output_transforms,
        config_name: str,
    ):
        """Initialize the RealtimeVLA policy.

        Args:
            engine: Pi0Inference engine from third_party/realtime-vla
            num_views: Number of camera views (1-3)
            chunk_size: Trajectory length for diffusion
            device: Device to run on ("cuda" or "cpu")
            input_transforms: List of input transforms (data_transforms + normalize + model_transforms)
            output_transforms: List of output transforms (model_transforms + unnormalize + data_transforms)
            config_name: Training config name for metadata
        """
        self._engine = engine
        self._num_views = num_views
        self._chunk_size = chunk_size
        self._device = device
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms
        self.metadata = {
            "name": f"realtime-vla-{config_name}",
            "num_views": num_views,
            "chunk_size": chunk_size,
            "device": device,
        }
        self._align_step_counter: int = -1

    def _print_debug_info(self, tag: str, data: dict):
        """Print debug information (controlled by OPENPI_DEBUG_TRANSFORMS)."""
        debug_enabled = os.getenv("OPENPI_DEBUG_TRANSFORMS") not in (None, "", "0", "false", "False", "no")
        if not debug_enabled:
            return

        print(f"\n{'='*70}")
        print(f"[DEBUG REALTIME-VLA] {tag}")
        print(f"{'='*70}")

        def _print_array(name: str, arr):
            if arr is None:
                return
            try:
                if isinstance(arr, torch.Tensor):
                    # Fix: Convert BFloat16 to Float32 before numpy conversion
                    if arr.dtype == torch.bfloat16:
                        arr = arr.float()
                    arr = arr.cpu().numpy()
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

    def infer(self, obs: dict) -> dict:
        """Execute inference on the observation.

        Args:
            obs: Observation dictionary with robot-specific keys

        Returns:
            Dictionary with "actions" and "action" keys, optionally with "policy_timing"
        """
        profile_enabled = os.getenv("OPENPI_INFER_PROFILE") not in (None, "", "0", "false", "False", "no")

        total_t0 = time.monotonic()

        # Alignment step index (per-inference call).
        align_step = self._align_step_counter + 1
        self._align_step_counter = align_step

        # Stage 1: Raw observation from websocket
        _state_raw = obs["observation.state"] if "observation.state" in obs else obs.get("state")
        _log_align(
            "rt_pre_obs",
            {
                "state": _state_raw,
                # Images are still nested under observation.images.* at this point.
            },
            step=align_step,
        )

        # Debug print: 1. 转换前
        self._print_debug_info("1. 转换前 (原始观测)", obs)

        t0 = time.monotonic()
        # Apply the complete input transform chain (robot-specific + normalize + model-specific)
        # This matches the JAX inference pipeline for the specified robot config.
        data = dict(obs)
        for transform in self._input_transforms:
            data = transform(data)
            transform_name = transform.__class__.__name__
            if "Inputs" in transform_name:
                _log_align(f"rt_post_{transform_name.lower()}", data, step=align_step)
            elif transform_name == "Normalize":
                _log_align("rt_post_normalize", data, step=align_step)
            elif transform_name == "ResizeImages":
                _log_align("rt_post_resize", data, step=align_step)
            elif "Pad" in transform_name:
                _log_align("rt_post_pad", data, step=align_step)

        # Combined snapshot for compatibility
        _log_align("rt_post_input_transform", data, step=align_step)

        # Extract normalized images/state
        # We keep the state as NumPy for later unnormalization.
        img_dict = data.get("image", {})
        state_np = np.asarray(data.get("state"), dtype=np.float32)

        # Collect images preserving insertion order; pad/truncate to num_views.
        imgs_np = []
        if isinstance(img_dict, dict):
            imgs_np = [np.asarray(v) for v in img_dict.values()]
        elif isinstance(img_dict, (list, tuple)):
            imgs_np = [np.asarray(v) for v in img_dict]
        if len(imgs_np) < self._num_views:
            pad = self._num_views - len(imgs_np)
            zero = np.zeros((224, 224, 3), dtype=np.float32)
            imgs_np.extend([zero] * pad)
        if len(imgs_np) == 0:
            raise RuntimeError("No images found after input transforms")
        imgs_np = np.stack(imgs_np[: self._num_views], axis=0)

        # Mirror JAX Observation.from_dict: uint8 images -> [-1, 1] float.
        # In JAX, this conversion happens inside Observation.from_dict just
        # before the model sees the images. Here we apply the same mapping
        # explicitly before sending tensors to the realtime-vla engine.
        if imgs_np.dtype != np.float32:
            imgs_np = imgs_np.astype(np.float32)
        # If images appear to be in [0, 255], normalize to [-1, 1].
        if imgs_np.max() > 1.0:
            imgs_np = imgs_np / 255.0 * 2.0 - 1.0

        # Move to device and cast to bf16 for the Triton kernels.
        images = torch.from_numpy(imgs_np).to(self._device).to(torch.bfloat16)
        state = torch.from_numpy(state_np).to(self._device).to(torch.bfloat16)

        # Stage 6: Model input (final tensor ready for inference)
        # Use the same keys as the JAX path so alignment logs match.
        _log_align(
            "rt_model_input",
            {
                "state": state_np,
                "image": img_dict,
            },
            step=align_step,
        )

        # Debug print: 2. 转换后
        self._print_debug_info("2. 转换后 (输入模型前)", {
            "images": images,
            "state": state,
        })

        batch_to_device_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Model forward
        t0 = time.monotonic()
        noise = torch.randn(self._chunk_size, 32, dtype=torch.bfloat16, device=self._device)
        actions = self._engine.forward(images, state, noise)
        infer_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Host copy
        t0 = time.monotonic()
        out = actions.float().cpu().numpy()
        host_copy_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Stage 7: Model output in normalized space (before Unnormalize)
        _log_align("rt_model_out_normalized", {"state": state_np, "actions": out}, step=align_step)

        # Debug print: 3. 推理后
        self._print_debug_info("3. 推理后 (模型输出, normalized)", {
            "state": state_np,
            "actions": out,
        })

        # Output transforms: mirror the JAX policy server by first
        # unnormalizing with the same norm_stats, then applying
        # data/output transforms to obtain actions in the dataset space.
        t0 = time.monotonic()
        outputs = {"state": state_np, "actions": out}

        for transform in self._output_transforms:
            outputs = transform(outputs)
            transform_name = transform.__class__.__name__
            if transform_name == "Unnormalize":
                _log_align("rt_post_unnormalize", outputs, step=align_step)
            elif "AbsoluteActions" in transform_name:
                _log_align("rt_post_absolute_actions", outputs, step=align_step)
            elif "Outputs" in transform_name:
                _log_align(f"rt_post_{transform_name.lower()}", outputs, step=align_step)

        _log_align("rt_post_output_transform", outputs, step=align_step)

        out7 = np.asarray(outputs["actions"], dtype=np.float32)
        output_transform_ms = (time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

        # Debug print: 4. 最终输出
        self._print_debug_info("4. 最终输出 (Unnormalize后)", {
            "actions": out7,
        })

        total_ms = (time.monotonic() - total_t0) * 1000.0 if profile_enabled else 0.0

        result = {"actions": out7, "action": out7}
        if profile_enabled:
            result["policy_timing"] = {
                # Minimal set aligned with server pretty-printer
                "copy_input_ms": 0.0,
                "input_transform_ms": 0.0,
                "batch_to_device_ms": batch_to_device_ms,
                "observation_build_ms": 0.0,
                "infer_ms": infer_ms,
                "host_copy_ms": host_copy_ms,
                "output_transform_ms": output_transform_ms,
                "total_ms": total_ms,
            }
        return result
