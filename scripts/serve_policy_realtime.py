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
    """

    # Path to converted_checkpoint.pkl produced by convert_from_jax.py
    checkpoint_pkl: str
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
            import numpy as _np
            import torch as _torch

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

            engine = Pi0Inference(checkpoint, num_views=args.policy.num_views, chunk_size=args.policy.chunk_size)

            class _RealtimeVLAPolicy:
                # Minimal BasePolicy-compatible wrapper
                def __init__(self, engine, num_views: int, chunk_size: int, device: str):
                    self._engine = engine
                    self._num_views = num_views
                    self._chunk_size = chunk_size
                    self._device = device
                    self.metadata = {
                        "name": "realtime-vla-pi0",
                        "num_views": num_views,
                        "chunk_size": chunk_size,
                        "device": device,
                    }

                def _to_bf16_cuda_image(self, img: _np.ndarray | _torch.Tensor) -> _torch.Tensor:
                    if isinstance(img, _torch.Tensor):
                        t = img
                    else:
                        t = _torch.from_numpy(_np.array(img))
                    if t.ndim == 3 and t.shape[0] in (1, 3) and t.shape[1] == 224 and t.shape[2] == 224:
                        # CHW -> HWC
                        t = t.permute(1, 2, 0)
                    # Move to CUDA and cast
                    t = t.to(self._device)
                    if t.dtype != _torch.bfloat16:
                        # Normalize if likely uint8 or float32 in [0,255]
                        if t.dtype in (_torch.uint8, _torch.int8, _torch.int16, _torch.int32, _torch.int64):
                            t = t.float().div(255.0)
                        elif t.dtype in (_torch.float32, _torch.float64) and t.max() > 1.0:
                            t = t.float().div(255.0)
                        t = t.to(_torch.bfloat16)
                    return t

                def _gather_images(self, obs: dict) -> _torch.Tensor:
                    # Try Yuanluo-style keys first
                    keys = [
                        "observation.images.head_camera",
                        "observation.images.wrist_left_camera",
                        "observation.images.gelsight_left",
                    ]
                    imgs = []
                    for k in keys:
                        if k in obs:
                            imgs.append(self._to_bf16_cuda_image(obs[k]))
                    # Fallbacks: a list/array of images
                    if len(imgs) == 0 and "images" in obs:
                        arr = obs["images"]
                        if isinstance(arr, _np.ndarray):
                            if arr.ndim == 4 and arr.shape[-1] == 3:
                                for i in range(min(self._num_views, arr.shape[0])):
                                    imgs.append(self._to_bf16_cuda_image(arr[i]))
                            elif arr.ndim == 4 and arr.shape[1] == 3:
                                for i in range(min(self._num_views, arr.shape[0])):
                                    imgs.append(self._to_bf16_cuda_image(arr[i]))
                        elif isinstance(arr, (list, tuple)):
                            for i in range(min(self._num_views, len(arr))):
                                imgs.append(self._to_bf16_cuda_image(arr[i]))
                    if len(imgs) < self._num_views:
                        # Pad with zeros if fewer views provided
                        pad = self._num_views - len(imgs)
                        zero = _torch.zeros(224, 224, 3, dtype=_torch.bfloat16, device=self._device)
                        imgs.extend([zero] * pad)
                    return _torch.stack(imgs[: self._num_views], dim=0)

                def _get_state(self, obs: dict) -> _torch.Tensor:
                    for k in ("observation.state", "state"):
                        if k in obs:
                            st = obs[k]
                            t = _torch.from_numpy(_np.array(st)) if not isinstance(st, _torch.Tensor) else st
                            t = t.to(self._device)
                            if t.numel() < 32:
                                t = _torch.nn.functional.pad(t.flatten(), (0, 32 - t.numel()))
                            elif t.numel() > 32:
                                t = t.flatten()[:32]
                            return t.to(_torch.bfloat16)
                    return _torch.zeros(32, dtype=_torch.bfloat16, device=self._device)

                def infer(self, obs: dict) -> dict:
                    import time as _time

                    profile_enabled = os.getenv("OPENPI_INFER_PROFILE") not in (None, "", "0", "false", "False", "no")

                    total_t0 = _time.monotonic()

                    t0 = _time.monotonic()
                    images = self._gather_images(obs)
                    state = self._get_state(obs)
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

                    # Output transform (slice to 7 dims to match Yuanluo)
                    t0 = _time.monotonic()
                    out7 = out[:, :7]
                    output_transform_ms = (_time.monotonic() - t0) * 1000.0 if profile_enabled else 0.0

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

            return _RealtimeVLAPolicy(engine, args.policy.num_views, args.policy.chunk_size, args.policy.device)
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

