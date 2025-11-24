import dataclasses
import enum
import logging
import os
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.realtime_vla_policy import RealtimeVLAPolicy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class RealtimeVLA:
    """Load a realtime-vla policy from a converted .pkl checkpoint.

    This wraps third_party/realtime-vla's Pi0Inference into a BasePolicy-like
    object. Uses the OpenPI config system to support all robots (Aloha, DROID,
    Yuanluo, Libero, etc).

    The config parameter specifies which robot-specific transforms to apply,
    matching the same config system used by serve_policy.py for safetensors models.

    The transform pipeline is dynamically loaded from the training config:
      - Inputs are processed via robot-specific transforms (AlohaInputs, DroidInputs,
        YuanluoInputs, etc), normalized using checkpoint/dataset norm_stats, resized
        to 224x224, and padded to the model action_dim.
      - Outputs are unnormalized and passed through robot-specific output transforms
        to match the JAX policy server convention for that robot.

    The realtime-vla engine itself always sees normalized images/state; we do
    not modify the model or weights, only the surrounding data transforms.
    """

    # Training config name (e.g., "pi0_yuanluo_realtime", "pi0_aloha_realtime").
    config: str
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

            # Load training config specified by the user to build transforms dynamically.
            train_config = _train_config_mod.get_config(args.policy.config)
            data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

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

            normalize = _transforms.Normalize(norm_stats, use_quantiles=use_quantiles)
            unnormalize = _transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles)
            input_transforms = [
                *data_config.data_transforms.inputs,
                normalize,
                *data_config.model_transforms.inputs,
            ]
            output_transforms = [
                *data_config.model_transforms.outputs,
                unnormalize,
                *data_config.data_transforms.outputs,
            ]

            engine = Pi0Inference(checkpoint, num_views=args.policy.num_views, chunk_size=args.policy.chunk_size)

            return RealtimeVLAPolicy(
                engine=engine,
                num_views=args.policy.num_views,
                chunk_size=args.policy.chunk_size,
                device=args.policy.device,
                input_transforms=input_transforms,
                output_transforms=output_transforms,
                config_name=train_config.name,
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
        print(f"  Config:          {args.policy.config}")
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


def main(args: Args) -> None:
    policy = create_policy(args)
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
