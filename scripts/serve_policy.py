import dataclasses
import enum
import logging
import socket
import sys
from pathlib import Path

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# Allow importing repo-root `tracer/` when this OpenPI copy lives under `third_party/openpi`.
_repo_root = Path(__file__).resolve().parents[3]
if (_repo_root / "tracer").exists() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


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
    # Tracing (server-side, optional)
    # ============================
    # When set, write per-infer dumps to `trace_out_dir/dumps/*.pt` (and images if enabled).
    trace_out_dir: str | None = None
    # Dump reduced expert attention (action tokens -> vision tokens).
    trace_dump_attn: bool = False
    # Comma-separated layer indices, e.g. "0,8,16". Empty means "last layer".
    trace_attn_layers: str = ""
    # Save input images (from client obs) for offline overlays.
    trace_save_policy_images: bool = True
    # Print attention stats to logs.
    trace_print_attn: bool = True
    # Max dumps to write (0 means unlimited).
    trace_max_dumps: int = 200
    # Dump every N inferences (1 means dump every inference).
    trace_every_n: int = 1

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


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    if args.trace_out_dir and args.trace_dump_attn:
        from openpi.serving.traced_policy import PolicyTraceConfig, TracedPolicy

        attn_layers = tuple(int(p.strip()) for p in str(args.trace_attn_layers).split(",") if p.strip())
        trace_cfg = PolicyTraceConfig(
            out_dir=str(args.trace_out_dir),
            dump_attn=bool(args.trace_dump_attn),
            attn_layers=attn_layers,
            save_policy_images=bool(args.trace_save_policy_images),
            print_attn=bool(args.trace_print_attn),
            max_dumps=int(args.trace_max_dumps),
            every_n=int(args.trace_every_n),
        )
        logging.info("Tracing enabled: out_dir=%s attn_layers=%s max_dumps=%s every_n=%s", trace_cfg.out_dir, attn_layers or ("last",), trace_cfg.max_dumps, trace_cfg.every_n)
        policy = TracedPolicy(policy, trace_cfg=trace_cfg)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
