import dataclasses
import enum
import logging
import os
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


class PruningMode(enum.Enum):
    ON = "on"
    OFF = "off"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class RealtimeVLA:
    """Serve a realtime-vla Pi0Inference checkpoint (.pkl) via the same server.

    This wraps third_party/realtime-vla's Pi0Inference into a BasePolicy-like
    object. Inputs should include Yuanluo-style observation keys or raw images/state.

    To closely match the original JAX pi0_yuanluo_delta inference pipeline, we
    apply the same Yuanluo input/output transforms and dataset normalization
    around the realtime-vla engine:
      - Inputs are repacked, converted via YuanluoInputs, normalized using the
        checkpoint/dataset norm_stats, resized to 224x224, and state is padded
        to the model action_dim.
      - Outputs are unnormalized and passed through YuanluoOutputs so that the
        returned actions match the JAX policy server convention.

    The realtime-vla engine itself always sees normalized images/state; we do
    not modify the model or weights, only the surrounding data transforms.
    """

    # Path to converted_checkpoint.pkl produced by convert_from_jax.py
    checkpoint_pkl: str
    # Optional path to the original JAX checkpoint directory (e.g. .../29999).
    # When provided, we load norm_stats from <jax_checkpoint_dir>/assets/<asset_id>
    # to exactly match the JAX inference normalization. If not provided, we
    # fall back to the dataset assets specified in the train config.
    jax_checkpoint_dir: str | None = None
    # Number of camera views (1-3)
    num_views: int = 2
    # Trajectory length for diffusion
    chunk_size: int = 50
    # Device to run on (must be CUDA for realtime-vla)
    device: str = "cuda"


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy_realtime script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Control LightVLA-style visual token pruning at inference time.
    # If None, use the checkpoint/config default. If set, overrides the runtime behavior.
    pruning: PruningMode | None = None
    # Optional pruning keep ratio (0..1). Only used if pruning is enabled.
    prune_ratio: float | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | RealtimeVLA | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case RealtimeVLA():
            # Lazy imports to avoid heavy deps unless needed
            import sys
            import pickle
            from pathlib import Path as _Path
            import pathlib as _pathlib
            import numpy as _np
            import torch as _torch

            from openpi.policies.policy import _log_alignment_snapshot as _log_align  # type: ignore
            import openpi.transforms as _transforms
            from openpi.training import checkpoints as _checkpoints
            from openpi.training import config as _train_config_mod
            from openpi.policies import yuanluo_policy as _yuanluo_policy

            # Add third_party/realtime-vla to path
            repo_root = _Path(__file__).resolve().parents[1]
            rt_vla_dir = repo_root / "third_party" / "realtime-vla"
            sys.path.insert(0, str(rt_vla_dir))
            from pi0_infer import Pi0Inference  # type: ignore

            ckpt_path = _Path(args.policy.checkpoint_pkl).expanduser().resolve()
            with open(ckpt_path, "rb") as f:
                checkpoint = pickle.load(f)

            if args.policy.device.startswith("cuda") and not _torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but not available for realtime-vla policy")

            # Build Yuanluo config + normalization so that inputs/outputs
            # are transformed in the same way as the JAX pi0_yuanluo_delta
            # policy server.
            yuanluo_config = _train_config_mod.get_config("pi0_yuanluo_realtime")
            data_config = yuanluo_config.data.create(yuanluo_config.assets_dirs, yuanluo_config.model)

            asset_id = data_config.asset_id
            norm_stats = data_config.norm_stats
            use_quantiles = data_config.use_quantile_norm

            # Prefer checkpoint-local norm_stats if JAX checkpoint dir is given,
            # matching create_trained_policy's behavior for JAX inference.
            norm_stats_source = "config"
            if args.policy.jax_checkpoint_dir is not None and asset_id is not None:
                assets_dir = _pathlib.Path(args.policy.jax_checkpoint_dir).expanduser().resolve() / "assets"
                try:
                    loaded_norm_stats = _checkpoints.load_norm_stats(assets_dir, asset_id)
                    norm_stats = loaded_norm_stats
                    norm_stats_source = "jax_checkpoint"
                    logging.info(
                        "✓ Loaded norm_stats for realtime-vla from JAX checkpoint: %s (asset_id=%s)",
                        assets_dir,
                        asset_id,
                    )
                except Exception:
                    logging.exception(
                        "✗ Failed to load norm_stats from JAX checkpoint at %s; "
                        "falling back to config assets. THIS WILL CAUSE OUTPUT MISALIGNMENT!",
                        assets_dir,
                    )
            else:
                if asset_id is not None:
                    logging.warning(
                        "⚠️  --policy.jax_checkpoint_dir NOT specified! Using config norm_stats. "
                        "This may cause large output differences vs JAX inference. "
                        "Please specify --policy.jax_checkpoint_dir to load correct norm_stats."
                    )

            # Log norm_stats summary for verification
            logging.info("=" * 80)
            logging.info("Norm Stats Configuration:")
            logging.info("  Source: %s", norm_stats_source)
            if norm_stats is not None:
                if "state" in norm_stats and norm_stats["state"] is not None:
                    logging.info("  State mean (first 3): %s", norm_stats["state"].mean[:3].tolist())
                    logging.info("  State std (first 3): %s", norm_stats["state"].std[:3].tolist())
                if "actions" in norm_stats and norm_stats["actions"] is not None:
                    logging.info("  Actions mean (first 3): %s", norm_stats["actions"].mean[:3].tolist())
                    logging.info("  Actions std (first 3): %s", norm_stats["actions"].std[:3].tolist())
            else:
                logging.error("  ✗ norm_stats is None! Policy cannot be created.")
            logging.info("=" * 80)

            # Construct the exact Yuanluo input/output transforms we need around
            # the realtime-vla engine. We reuse the same components as the JAX
            # pipeline, but only for observation and action tensors (no prompt
            # tokenization is needed here).
            yuanluo_inputs = _yuanluo_policy.YuanluoInputs(model_type=yuanluo_config.model.model_type)
            yuanluo_outputs = _yuanluo_policy.YuanluoOutputs()
            normalize = _transforms.Normalize(norm_stats, use_quantiles=use_quantiles)
            unnormalize = _transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles)
            resize = _transforms.ResizeImages(224, 224)
            pad_state = _transforms.PadStatesAndActions(yuanluo_config.model.action_dim)

            # Full Yuanluo output pipeline as used in JAX:
            # this may include AbsoluteActions (delta->absolute) followed by YuanluoOutputs.
            data_output_transforms = list(data_config.data_transforms.outputs)

            engine = Pi0Inference(checkpoint, num_views=args.policy.num_views, chunk_size=args.policy.chunk_size)

            class _RealtimeVLAPolicy:
                # Minimal BasePolicy-compatible wrapper
                def __init__(
                    self,
                    engine,
                    num_views: int,
                    chunk_size: int,
                    device: str,
                    yuanluo_inputs,
                    normalize,
                    resize,
                    pad_state,
                    unnormalize,
                    data_output_transforms,
                ):
                    self._engine = engine
                    self._num_views = num_views
                    self._chunk_size = chunk_size
                    self._device = device
                    self._yuanluo_inputs = yuanluo_inputs
                    self._normalize = normalize
                    self._resize = resize
                    self._pad_state = pad_state
                    self._unnormalize = unnormalize
                    self._data_output_transforms = data_output_transforms
                    self.metadata = {
                        "name": "realtime-vla-pi0",
                        "num_views": num_views,
                        "chunk_size": chunk_size,
                        "device": device,
                    }
                    # Alignment step counter (per-inference call)
                    self._align_step_counter: int = -1

                def _to_bf16_cuda_image(self, img: _np.ndarray | _torch.Tensor) -> _torch.Tensor:
                    if isinstance(img, _torch.Tensor):
                        t = img
                    else:
                        t = _torch.from_numpy(_np.array(img))

                    # Accept both CHW and HWC single images and convert to HWC.
                    if t.ndim == 3 and t.shape[0] in (1, 3) and t.shape[1] == 224 and t.shape[2] == 224:
                        # CHW -> HWC
                        t = t.permute(1, 2, 0)

                    # Move to CUDA before further processing.
                    t = t.to(self._device)

                    if t.dtype != _torch.bfloat16:
                        # Mirror JAX Observation.from_dict: uint8 images -> [-1, 1] float.
                        if t.dtype in (_torch.uint8, _torch.int8, _torch.int16, _torch.int32, _torch.int64):
                            t = t.float().div(255.0).mul(2.0).sub(1.0)
                        elif t.dtype in (_torch.float32, _torch.float64):
                            # If images are in [0, 255], normalize to [-1, 1].
                            if t.max() > 1.0:
                                t = t.float().div(255.0).mul(2.0).sub(1.0)
                            # If already in [-1, 1], leave as-is.
                        t = t.to(_torch.bfloat16)
                    return t

                def _gather_images(self, obs: dict) -> _torch.Tensor:
                    # This helper is no longer used as we now mirror the JAX
                    # Yuanluo + Normalize transforms before calling the engine.
                    raise RuntimeError("_gather_images should not be called directly")

                def _print_debug_info(self, tag: str, data: dict):
                    """打印调试信息"""
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
                            if isinstance(arr, _torch.Tensor):
                                # Fix: Convert BFloat16 to Float32 before numpy conversion
                                if arr.dtype == _torch.bfloat16:
                                    arr = arr.float()
                                arr = arr.cpu().numpy()
                            arr = _np.asarray(arr)

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
                    import time as _time

                    profile_enabled = os.getenv("OPENPI_INFER_PROFILE") not in (None, "", "0", "false", "False", "no")

                    total_t0 = _time.monotonic()

                    # Alignment step index (per-inference call).
                    align_step = self._align_step_counter + 1
                    self._align_step_counter = align_step

                    # Stage 1: Raw observation from websocket (Yuanluo-style keys).
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

                    t0 = _time.monotonic()
                    # Reuse the same Yuanluo + normalization pipeline as the JAX
                    # pi0_yuanluo_delta server, but evaluate it in NumPy. This
                    # yields normalized images/state that match the JAX model
                    # inputs before we hand them to the realtime-vla engine.
                    # For inference, the websocket obs is already using the
                    # Yuanluo-style keys (see LeRobotYuanluoDataConfig and
                    # infer_pytorch_pi0.py), so we skip the dataset-only repack
                    # transforms and go straight into YuanluoInputs.

                    # Stage 2: After YuanluoInputs (extract state[:7], remap images)
                    inputs = self._yuanluo_inputs(obs)
                    _log_align("rt_post_yuanluo_inputs", inputs, step=align_step)

                    # Stage 3: After Normalize ((x - mean) / std)
                    inputs = self._normalize(inputs)
                    _log_align("rt_post_normalize", inputs, step=align_step)

                    # Stage 4: After ResizeImages (224x224)
                    inputs = self._resize(inputs)
                    _log_align("rt_post_resize", inputs, step=align_step)

                    # Stage 5: After PadStatesAndActions ([7] -> [32])
                    inputs = self._pad_state(inputs)
                    _log_align("rt_post_pad", inputs, step=align_step)

                    # Combined snapshot for compatibility
                    _log_align("rt_post_input_transform", inputs, step=align_step)

                    # Extract normalized images/state in the same order used by
                    # YuanluoInputs: [base_0_rgb, left_wrist_0_rgb]. We keep the
                    # state as NumPy for later unnormalization.
                    img_dict = inputs["image"]
                    state_np = _np.asarray(inputs["state"], dtype=_np.float32)

                    imgs_np = []
                    for cam_key in ("base_0_rgb", "left_wrist_0_rgb"):
                        if cam_key in img_dict:
                            imgs_np.append(_np.asarray(img_dict[cam_key]))
                    if len(imgs_np) < self._num_views:
                        pad = self._num_views - len(imgs_np)
                        zero = _np.zeros((224, 224, 3), dtype=_np.float32)
                        imgs_np.extend([zero] * pad)
                    imgs_np = _np.stack(imgs_np[: self._num_views], axis=0)

                    # Mirror JAX Observation.from_dict: uint8 images -> [-1, 1] float.
                    # In JAX, this conversion happens inside Observation.from_dict just
                    # before the model sees the images. Here we apply the same mapping
                    # explicitly before sending tensors to the realtime-vla engine.
                    if imgs_np.dtype != _np.float32:
                        imgs_np = imgs_np.astype(_np.float32)
                    # If images appear to be in [0, 255], normalize to [-1, 1].
                    if imgs_np.max() > 1.0:
                        imgs_np = imgs_np / 255.0 * 2.0 - 1.0

                    # Move to device and cast to bf16 for the Triton kernels.
                    images = _torch.from_numpy(imgs_np).to(self._device).to(_torch.bfloat16)
                    state = _torch.from_numpy(state_np).to(self._device).to(_torch.bfloat16)

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

                    batch_to_device_ms = (_time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

                    # Model forward
                    t0 = _time.monotonic()
                    noise = _torch.randn(self._chunk_size, 32, dtype=_torch.bfloat16, device=self._device)
                    actions = self._engine.forward(images, state, noise)
                    infer_ms = (_time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

                    # Host copy
                    t0 = _time.monotonic()
                    out = actions.float().cpu().numpy()
                    host_copy_ms = (_time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

                    # Stage 7: Model output in normalized space (before Unnormalize)
                    _log_align("rt_model_out_normalized", {"state": state_np, "actions": out}, step=align_step)

                    # Debug print: 3. 推理后
                    self._print_debug_info("3. 推理后 (模型输出, normalized)", {
                        "state": state_np,
                        "actions": out,
                    })

                    # Output transforms: mirror the JAX policy server by first
                    # unnormalizing with the same norm_stats, then applying
                    # YuanluoOutputs to obtain 7D actions in the dataset space.
                    t0 = _time.monotonic()
                    outputs = {"state": state_np, "actions": out}

                    # Stage 8: After Unnormalize (x * std + mean)
                    outputs = self._unnormalize(outputs)
                    _log_align("rt_post_unnormalize", outputs, step=align_step)

                    # Stage 9: After each data_output_transform (AbsoluteActions, YuanluoOutputs, etc)
                    for idx, t in enumerate(self._data_output_transforms):
                        outputs = t(outputs)
                        # Log after AbsoluteActions specifically
                        if hasattr(t, '__class__') and 'AbsoluteActions' in t.__class__.__name__:
                            _log_align("rt_post_absolute_actions", outputs, step=align_step)

                    out7 = _np.asarray(outputs["actions"], dtype=_np.float32)
                    output_transform_ms = (_time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

                    # Debug print: 4. 最终输出
                    self._print_debug_info("4. 最终输出 (Unnormalize后)", {
                        "actions": out7,
                    })

                    # Stage 10: Final output after all transforms
                    _log_align("rt_post_output_transform", {"state": state_np, "actions": out7}, step=align_step)

                    total_ms = (_time.monotonic() - total_t0) * 1000.0 if profile_enabled else 0.0

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

            return _RealtimeVLAPolicy(
                engine,
                args.policy.num_views,
                args.policy.chunk_size,
                args.policy.device,
                yuanluo_inputs,
                normalize,
                resize,
                pad_state,
                unnormalize,
                data_output_transforms,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _print_server_config(args: Args, policy: _policy.Policy, hostname: str, local_ip: str) -> None:
    """Print formatted server configuration at startup."""
    print("\n" + "=" * 80)
    print("               OpenPI Realtime Policy Server Starting")
    print("=" * 80)

    # Network configuration
    print("\n[Network Configuration]")
    print(f"  Hostname:        {hostname}")
    print(f"  IP Address:      {local_ip}")
    print(f"  Port:            {args.port}")
    print(f"  Listen Address:  0.0.0.0:{args.port}")

    # Policy configuration
    print("\n[Policy Configuration]")
    if isinstance(args.policy, Checkpoint):
        print(f"  Type:            Checkpoint")
        print(f"  Config:          {args.policy.config}")
        print(f"  Checkpoint Dir:  {args.policy.dir}")
    elif isinstance(args.policy, RealtimeVLA):
        print(f"  Type:            RealtimeVLA (.pkl)")
        print(f"  Checkpoint Pkl:  {args.policy.checkpoint_pkl}")
        print(f"  Num Views:       {args.policy.num_views}")
        print(f"  Chunk Size:      {args.policy.chunk_size}")
        print(f"  Device:          {args.policy.device}")
    else:
        print(f"  Type:            Default ({args.env.value})")

    if args.default_prompt:
        print(f"  Default Prompt:  {args.default_prompt}")
    if args.record:
        print(f"  Recording:       ENABLED (output: policy_records/)")

    # Pruning configuration
    print("\n[Visual Token Pruning]")
    try:
        model = policy._model  # type: ignore[attr-defined]
        if hasattr(model, "token_pruning_enabled"):
            enabled = getattr(model, "token_pruning_enabled")
            ratio = getattr(model, "token_prune_ratio", None)

            if enabled:
                print(f"  Status:          ENABLED ✓")
                if ratio is not None:
                    print(f"  Prune Ratio:     {ratio:.2f} (keeping {ratio*100:.0f}% of tokens)")
            else:
                print(f"  Status:          DISABLED")
                print(f"  Note:            Will show token counts without pruning")
        else:
            print(f"  Status:          Not supported by this model")
    except Exception:
        print(f"  Status:          Unknown (unable to query model)")

    # Profiling configuration
    print("\n[Performance Profiling]")
    profiling_enabled = os.getenv("OPENPI_INFER_PROFILE") not in (None, "", "0", "false", "False", "no")
    if profiling_enabled:
        print(f"  Status:          ENABLED ✓")
        print(f"  Output:          Detailed timing for each inference step")
        print(f"  Metrics:         Server (recv/unpack/infer/pack/send)")
        print(f"                   Policy (transforms/model/host_copy)")
        print(f"                   Tokens (counts and timing)")
    else:
        print(f"  Status:          DISABLED")
        print(f"  Tip:             Set OPENPI_INFER_PROFILE=1 to enable")

    print("\n" + "=" * 80)
    print("  Realtime server is ready and waiting for connections...")
    print("=" * 80 + "\n")


def _apply_pruning_overrides(policy: _policy.Policy, *, mode: PruningMode | None, ratio: float | None) -> None:
    """Apply CLI pruning overrides to the underlying model if supported."""
    if mode is None and ratio is None:
        return
    try:
        model = policy._model  # type: ignore[attr-defined]
        updated = False
        if mode is not None and hasattr(model, "token_pruning_enabled"):
            enabled = True if mode == PruningMode.ON else False
            setattr(model, "token_pruning_enabled", enabled)
            updated = True
        if ratio is not None and hasattr(model, "token_prune_ratio"):
            clamped = max(0.0, min(1.0, float(ratio)))
            setattr(model, "token_prune_ratio", clamped)
            updated = True
        if updated:
            logging.info(
                "[PRUNE] cli override: enabled=%s, ratio=%s",
                getattr(model, "token_pruning_enabled", None),
                getattr(model, "token_prune_ratio", None),
            )
    except Exception:
        # Do not crash the server for override issues.
        logging.exception("Failed to apply CLI pruning overrides")


def main(args: Args) -> None:
    policy = create_policy(args)
    # Apply CLI pruning overrides (takes precedence over env var overrides inside policy_config)
    _apply_pruning_overrides(policy, mode=args.pruning, ratio=args.prune_ratio)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # Print formatted server configuration
    _print_server_config(args, policy, hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
