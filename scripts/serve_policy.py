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
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

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
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


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
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _print_server_config(args: Args, policy: _policy.Policy, hostname: str, local_ip: str) -> None:
    """Print formatted server configuration at startup."""
    print("\n" + "=" * 80)
    print("                    OpenPI Policy Server Starting")
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
    print("  Server is ready and waiting for connections...")
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
