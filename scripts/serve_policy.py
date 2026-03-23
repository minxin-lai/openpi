import dataclasses
import enum
import logging
import os
import socket
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]
_openpi_src = Path(__file__).resolve().parents[1] / "src"
_vla_src = _repo_root / "src"
if _openpi_src.exists() and str(_openpi_src) not in sys.path:
    sys.path.insert(0, str(_openpi_src))
if _vla_src.exists() and str(_vla_src) not in sys.path:
    sys.path.insert(0, str(_vla_src))

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# Usage
#
#   uv run scripts/serve_policy.py --env LIBERO --port 8003 \
#     --vla-opt-pruning-config config/pruning/post_t128.yaml \
#     --vla-opt-observe-config configs/observe/infer_light.json \
#     policy:checkpoint --policy.config pi05_libero_spatial --policy.dir <CKPT_DIR>

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

    # ============================
    # VLA-OPT (Pi0.5 PyTorch wrapper)
    # ============================
    # IMPORTANT: this YAML must match how the checkpoint was trained/saved.
    vla_opt_pruning_config: str | None = None
    vla_opt_observe_config: str | None = None

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
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    _clear_legacy_env()

    if args.vla_opt_pruning_config is not None or args.vla_opt_observe_config is not None:
        # Ensure vla-opt is importable when running inside `third_party/openpi/`.
        vla_src = _repo_root / "src"
        if not vla_src.exists():
            raise FileNotFoundError(f"VLA-OPT enabled but vla-opt src not found at: {vla_src}")
        if str(vla_src) not in sys.path:
            sys.path.insert(0, str(vla_src))

        if args.vla_opt_pruning_config is not None:
            pruning_config = str(args.vla_opt_pruning_config).strip()
            if not pruning_config:
                raise ValueError("--vla-opt-pruning-config must not be empty")
            os.environ["VLA_OPT_PRUNING_CONFIG"] = pruning_config
        if args.vla_opt_observe_config is not None:
            observe_config = str(args.vla_opt_observe_config).strip()
            if not observe_config:
                raise ValueError("--vla-opt-observe-config must not be empty")
            os.environ["VLA_OPT_OBSERVE_CONFIG"] = observe_config

        logging.info(
            "VLA-OPT enabled: pruning_config=%s observe_config=%s",
            str(args.vla_opt_pruning_config),
            str(args.vla_opt_observe_config),
        )

    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


def _clear_legacy_env() -> None:
    for name in tuple(os.environ):
        if name.startswith("VLA_OPT_") or name.startswith("OPENPI_DEBUG_"):
            os.environ.pop(name, None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
