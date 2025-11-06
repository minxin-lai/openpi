"""Replay LeRobot dataset through policy server for evaluation.

This script loads observations from a LeRobot dataset and sends them to a
running policy server via WebSocket to get predicted actions. It compares
the predictions with ground truth actions and reports detailed performance
statistics with beautiful hierarchical profiling output.

For detailed profiling usage and examples, see: scripts/PROFILING_USAGE.md

Quick Start:
    # Terminal 1: Start policy server with profiling
    OPENPI_INFER_PROFILE=1 uv run scripts/serve_policy.py \\
        --pruning OFF \\
        policy:checkpoint \\
        --policy.config=pi05_pick_place_inference \\
        --policy.dir=/path/to/checkpoint

    # Terminal 2: Replay dataset with per-step profiling
    uv run scripts/replay_dataset.py \\
        --dataset_path=/home/shared_workspace/shiweikai/local/without_any \\
        --num_episodes=10 \\
        --warmup_steps=2 \\
        --host=localhost \\
        --port=8000 \\
        --log_per_step

Features:
    - Hierarchical profiling output with color-coded timing breakdown
    - Client/Server/Policy timing at multiple granularities
    - Token statistics (original count → processed count)
    - Pruning diagnostics (only shown when --pruning ON)
    - Beautiful summary tables with performance metrics
    - Copy-pastable command templates printed at startup
"""

import dataclasses
import json
import logging
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import rich
import tqdm
import tyro

try:
    import polars as pl
    from PIL import Image
    import io
except ImportError as e:
    print(f"ERROR: Required package not found: {e}")
    print("  pip install polars pyarrow pillow")
    exit(1)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments for dataset replay."""

    # Path to the LeRobot dataset directory
    dataset_path: str = "/home/shared_workspace/shiweikai/local/without_any"
    # Host to connect to the policy server
    host: str = "localhost"
    # Port to connect to the policy server
    port: int = 8000
    # API key for the server (if required)
    api_key: str | None = None
    # Number of episodes to replay (None = all episodes)
    num_episodes: int | None = 10
    # Starting episode index
    start_episode: int = 0
    # Output path for results JSON
    output_path: pathlib.Path | None = None
    # Skip first N steps per episode (warmup)
    warmup_steps: int = 0
    # If true, log per-step timings and keys returned by server
    log_per_step: bool = False


class ReplayStats:
    """Tracks statistics during dataset replay."""

    def __init__(self):
        self.episode_stats = []
        self.all_action_errors = []
        self.all_latencies = []

    def add_step(self, episode_idx: int, step_idx: int, action_mse: float, latency_ms: float):
        """Record statistics for a single step."""
        self.all_action_errors.append(action_mse)
        self.all_latencies.append(latency_ms)

    def finish_episode(self, episode_idx: int, num_steps: int, avg_mse: float, avg_latency: float):
        """Record episode-level statistics."""
        self.episode_stats.append({
            "episode_idx": episode_idx,
            "num_steps": num_steps,
            "avg_action_mse": avg_mse,
            "avg_latency_ms": avg_latency,
        })

    def print_summary(self):
        """Print overall summary statistics with enhanced formatting."""
        from rich.panel import Panel

        num_episodes = len(self.episode_stats)
        total_steps = sum(ep["num_steps"] for ep in self.episode_stats)

        # Performance metrics table
        perf_table = rich.table.Table(
            title="[bold cyan]Performance Metrics[/bold cyan]",
            show_header=True,
            header_style="bold white on blue",
            border_style="cyan",
            show_lines=True,
        )
        perf_table.add_column("Metric", style="bold cyan", justify="left", no_wrap=True)
        perf_table.add_column("Value", style="yellow", justify="right")

        perf_table.add_row("Episodes Evaluated", f"[green]{num_episodes}[/green]")
        perf_table.add_row("Total Steps", f"[green]{total_steps}[/green]")

        # Action error metrics
        perf_table.add_section()
        def safe_mean(a):
            return float(np.mean(a)) if len(a) > 0 else float("nan")

        def safe_std(a):
            return float(np.std(a)) if len(a) > 0 else float("nan")

        def safe_min(a):
            return float(np.min(a)) if len(a) > 0 else float("nan")

        def safe_max(a):
            return float(np.max(a)) if len(a) > 0 else float("nan")

        def safe_pct(a, p):
            return float(np.percentile(a, p)) if len(a) > 0 else float("nan")

        # Action error metrics
        perf_table.add_section()
        perf_table.add_row(
            "[bold]Action MSE (Mean)[/bold]",
            f"[yellow]{safe_mean(self.all_action_errors):.6f}[/yellow]" if len(self.all_action_errors) > 0 else "-"
        )
        perf_table.add_row(
            "Action MSE (Std)",
            f"{safe_std(self.all_action_errors):.6f}" if len(self.all_action_errors) > 0 else "-"
        )
        perf_table.add_row(
            "Action MSE (Min)",
            f"[green]{safe_min(self.all_action_errors):.6f}[/green]" if len(self.all_action_errors) > 0 else "-"
        )
        perf_table.add_row(
            "Action MSE (Max)",
            f"[red]{safe_max(self.all_action_errors):.6f}[/red]" if len(self.all_action_errors) > 0 else "-"
        )

        # Latency metrics
        perf_table.add_section()
        perf_table.add_row(
            "[bold]Latency (Mean)[/bold]",
            f"[yellow]{safe_mean(self.all_latencies):.2f} ms[/yellow]" if len(self.all_latencies) > 0 else "-"
        )
        perf_table.add_row(
            "Latency (Std)",
            f"{safe_std(self.all_latencies):.2f} ms" if len(self.all_latencies) > 0 else "-"
        )
        perf_table.add_row(
            "Latency (P50)",
            f"{safe_pct(self.all_latencies, 50):.2f} ms" if len(self.all_latencies) > 0 else "-"
        )
        perf_table.add_row(
            "Latency (P95)",
            f"[red]{safe_pct(self.all_latencies, 95):.2f} ms[/red]" if len(self.all_latencies) > 0 else "-"
        )
        perf_table.add_row(
            "Latency (Min)",
            f"[green]{safe_min(self.all_latencies):.2f} ms[/green]" if len(self.all_latencies) > 0 else "-"
        )
        perf_table.add_row(
            "Latency (Max)",
            f"[red]{safe_max(self.all_latencies):.2f} ms[/red]" if len(self.all_latencies) > 0 else "-"
        )

        console = rich.console.Console()
        console.print("\n")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]                     Evaluation Summary                              [/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════════[/bold cyan]")
        console.print(perf_table)
        console.print("\n[dim]Evaluation completed successfully![/dim]\n")

    def save_to_json(self, path: pathlib.Path):
        """Save statistics to JSON file."""
        results = {
            "summary": {
                "num_episodes": len(self.episode_stats),
                "total_steps": sum(ep["num_steps"] for ep in self.episode_stats),
                "avg_action_mse": float(np.mean(self.all_action_errors)),
                "action_mse_std": float(np.std(self.all_action_errors)),
                "avg_latency_ms": float(np.mean(self.all_latencies)),
                "latency_p50_ms": float(np.percentile(self.all_latencies, 50)),
                "latency_p95_ms": float(np.percentile(self.all_latencies, 95)),
            },
            "episodes": self.episode_stats,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {path}")

def _format_step_timing(
    episode_idx: int,
    step_idx: int,
    *,
    client_latency_ms: float | None,
    server_t: dict,
    policy_t: dict,
    mse: float,
) -> str:
    """Pretty hierarchical formatting for per-step timing with rich colors."""
    from rich.tree import Tree
    from rich.console import Console
    from io import StringIO

    def safe_float(x) -> float:
        if x is None or np.isnan(x):
            return 0.0
        return float(x)

    def pct(part: float, total: float) -> str:
        if total <= 0:
            return "0.0%"
        return f"{(part / total * 100):5.1f}%"

    # Helper percent functions for different baselines
    def pct_client(x: float) -> str:
        return pct(x, client_ms)

    def pct_policy(x: float) -> str:
        return pct(x, p_total)

    # Extract values
    client_ms = safe_float(client_latency_ms)
    s_recv = safe_float(server_t.get("recv_ms"))
    s_unpack = safe_float(server_t.get("unpack_ms"))
    s_infer = safe_float(server_t.get("infer_ms"))
    s_pack = safe_float(server_t.get("pack_ms"))
    s_send = safe_float(server_t.get("send_ms"))
    s_total = safe_float(server_t.get("total_ms"))

    p_copy = safe_float(policy_t.get("copy_input_ms"))
    p_inp = safe_float(policy_t.get("input_transform_ms"))
    p_bdev = safe_float(policy_t.get("batch_to_device_ms"))
    p_obs = safe_float(policy_t.get("observation_build_ms"))
    p_infer = safe_float(policy_t.get("infer_ms"))
    p_hcopy = safe_float(policy_t.get("host_copy_ms"))
    p_out = safe_float(policy_t.get("output_transform_ms"))
    p_total = safe_float(policy_t.get("total_ms"))

    # Build hierarchical tree
    tree = Tree(
        f"[bold cyan]Episode {episode_idx} Step {step_idx}[/bold cyan] | "
        f"[yellow]MSE: {mse:.6f}[/yellow]",
        guide_style="dim"
    )

    # Client latency (root level)
    client_node = tree.add(f"[bold blue]Client E2E[/bold blue]: {client_ms:7.2f} ms (100%)")

    # Server breakdown
    if s_total > 0:
        # Show pure server timings only (no comparison to client, no percentages)
        server_node = client_node.add(
            f"[bold green]Server Total[/bold green]: {s_total:7.2f} ms"
        )
        server_node.add(f"[green]recv[/green]:   {s_recv:7.2f} ms")
        server_node.add(f"[green]unpack[/green]: {s_unpack:7.2f} ms")
        server_node.add(f"[green]infer[/green]:  {s_infer:7.2f} ms")
        server_node.add(f"[green]pack[/green]:   {s_pack:7.2f} ms")
        server_node.add(f"[green]send[/green]:   {s_send:7.2f} ms")

    # Policy breakdown
    # Extract token statistics first (needed for proper accounting)
    p_pref_len = policy_t.get("prefix_len")
    p_pref_orig_len = policy_t.get("prefix_orig_len")
    p_keep = policy_t.get("prefix_keep_ratio")
    p_pref_ms = safe_float(policy_t.get("prefix_ms"))
    p_prefo_ms = safe_float(policy_t.get("prefix_orig_ms"))

    if p_total > 0:
        policy_node = client_node.add(
            f"[bold yellow]Policy Total[/bold yellow]: {p_total:7.2f} ms ({pct_client(p_total)})"
        )
        policy_node.add(f"[yellow]copy_input[/yellow]:          {p_copy:7.2f} ms ({pct_policy(p_copy)})")
        policy_node.add(f"[yellow]input_transform[/yellow]:     {p_inp:7.2f} ms ({pct_policy(p_inp)})")
        policy_node.add(f"[yellow]batch_to_device[/yellow]:     {p_bdev:7.2f} ms ({pct_policy(p_bdev)})")
        policy_node.add(f"[yellow]observation_build[/yellow]:   {p_obs:7.2f} ms ({pct_policy(p_obs)})")

        # Add token profiling time as sub-items (these are profiling overhead, not inference)
        if p_prefo_ms > 0 or p_pref_ms > 0:
            policy_node.add(f"[dim yellow]├─ token_profiling:[/dim yellow]")
            if p_prefo_ms > 0:
                policy_node.add(f"[dim yellow]│  ├─ calc_orig_len[/dim yellow]:  {p_prefo_ms:7.2f} ms ({pct_policy(p_prefo_ms)})")
            if p_pref_ms > 0:
                policy_node.add(f"[dim yellow]│  └─ embed_prefix[/dim yellow]:   {p_pref_ms:7.2f} ms ({pct_policy(p_pref_ms)})")

        policy_node.add(f"[yellow]model_infer[/yellow]:         {p_infer:7.2f} ms ({pct_policy(p_infer)})")
        policy_node.add(f"[yellow]host_copy[/yellow]:           {p_hcopy:7.2f} ms ({pct_policy(p_hcopy)})")
        policy_node.add(f"[yellow]output_transform[/yellow]:    {p_out:7.2f} ms ({pct_policy(p_out)})")

        # Show accounting summary
        accounted = p_copy + p_inp + p_bdev + p_obs + p_prefo_ms + p_pref_ms + p_infer + p_hcopy + p_out
        unaccounted = p_total - accounted
        if abs(unaccounted) > 0.5:  # More than 0.5ms unaccounted
            policy_node.add(f"[dim yellow]unaccounted[/dim yellow]:         {unaccounted:7.2f} ms ({pct(unaccounted, client_ms)})")

    # Token statistics summary (always show if available) — counts only, no timing
    if p_pref_len is not None and p_pref_orig_len is not None:
        token_info = (
            f"[bold magenta]Tokens[/bold magenta]: "
            f"{int(p_pref_orig_len):4d} → {int(p_pref_len):4d} "
            f"({float(p_keep or 0)*100:5.1f}% kept)"
        )
        client_node.add(token_info)

    # Pruning info (only when enabled)
    if policy_t.get("prune_enabled"):
        r = policy_t.get("prune_ratio", 0.0)
        prune_info = f"[bold red]Pruning[/bold red]: ENABLED | ratio={r:.2f}"
        client_node.add(prune_info)

    # Render to string
    console = Console(file=StringIO(), width=150, legacy_windows=False)
    console.print(tree)
    return console.file.getvalue().rstrip()


class SimpleDataset:
    """Simple dataset loader for LeRobot v2.1 format."""

    def __init__(self, dataset_path: str):
        self.dataset_path = pathlib.Path(dataset_path)

        # Load metadata
        with open(self.dataset_path / "meta" / "info.json") as f:
            self.info = json.load(f)

        # Load episodes metadata
        self.episodes = []
        with open(self.dataset_path / "meta" / "episodes.jsonl") as f:
            for line in f:
                episode = json.loads(line)
                self.episodes.append(episode)

        logger.info(f"Dataset loaded: {len(self.episodes)} episodes")
        logger.info(f"Robot type: {self.info.get('robot_type', 'unknown')}")

    def get_episode_data(self, episode_idx: int) -> pl.DataFrame:
        """Load all frames for a given episode."""
        episode_chunk = episode_idx // self.info.get("chunks_size", 1000)
        parquet_path = (
            self.dataset_path / "data" / f"chunk-{episode_chunk:03d}" /
            f"episode_{episode_idx:06d}.parquet"
        )

        if not parquet_path.exists():
            raise FileNotFoundError(f"Episode file not found: {parquet_path}")

        return pl.read_parquet(parquet_path)


def load_dataset(dataset_path: str) -> SimpleDataset:
    """Load LeRobot dataset from path."""
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = SimpleDataset(dataset_path)
    return dataset


def polars_row_to_observation(row: dict) -> dict:
    """Convert Polars DataFrame row to policy observation format.

    The policy expects observations in this format:
    {
        "state": (state_dim,) array,
        "images": {
            "cam_high": (3, H, W) uint8,
            "cam_left_wrist": (3, H, W) uint8,
            "cam_right_wrist": (3, H, W) uint8,
            ...
        },
        "prompt": str (optional),
    }
    """
    observation = {}

    # Extract state
    if "observation.state" in row:
        state = row["observation.state"]
        # Convert to numpy array if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        observation["state"] = state

    # Extract images
    images = {}
    for key in row.keys():
        if key.startswith("observation.images."):
            # Remove "observation.images." prefix
            img_key = key.replace("observation.images.", "")
            img_data = row[key]

            # Images may be stored as PNG bytes (either raw bytes or a dict with 'bytes' key)
            if isinstance(img_data, (bytes, bytearray)):
                img_pil = Image.open(io.BytesIO(img_data))
                # Convert to numpy array (H, W, C)
                img_array = np.array(img_pil)
                # Convert to CHW format
                if img_array.ndim == 3:
                    img_array = np.transpose(img_array, (2, 0, 1))
                images[img_key] = img_array.astype(np.uint8)
            elif isinstance(img_data, dict) and "bytes" in img_data:
                # Decode PNG bytes
                img_bytes = img_data["bytes"]
                img_pil = Image.open(io.BytesIO(img_bytes))
                # Convert to numpy array (H, W, C)
                img_array = np.array(img_pil)
                # Convert to CHW format
                if img_array.ndim == 3:
                    img_array = np.transpose(img_array, (2, 0, 1))
                images[img_key] = img_array.astype(np.uint8)
            elif isinstance(img_data, np.ndarray):
                # Already a numpy array
                images[img_key] = img_data.astype(np.uint8)
            else:
                logger.warning(f"Unexpected image format for {img_key}: {type(img_data)}")

    if images:
        observation["images"] = images

    # Add default prompt
    observation["prompt"] = "Dual-arm pick-and-place task"

    return observation


def extract_ground_truth_action(row: dict) -> np.ndarray:
    """Extract ground truth action from dataset row."""
    if "action" in row:
        action = row["action"]
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        return action
    raise ValueError("No action found in dataset row")


def compute_action_mse(pred_action: np.ndarray, gt_action: np.ndarray) -> float:
    """Compute mean squared error between predicted and ground truth actions."""
    return float(np.mean((pred_action - gt_action) ** 2))


def replay_episode(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    dataset: SimpleDataset,
    episode_idx: int,
    warmup_steps: int = 0,
    *,
    log_per_step: bool = False,
    stats: ReplayStats | None = None,
) -> tuple[int, float, float]:
    """Replay a single episode through the policy.

    Returns:
        (num_steps, avg_mse, avg_latency_ms)
    """
    # Load episode data
    episode_df = dataset.get_episode_data(episode_idx)
    num_steps = len(episode_df)

    errors = []
    latencies = []

    for step_idx in range(num_steps):
        # Get row as dict
        row = episode_df.row(step_idx, named=True)

        # Convert to observation format
        observation = polars_row_to_observation(row)
        ground_truth_action = extract_ground_truth_action(row)

        # Run inference
        start_time = time.time()
        result = policy.infer(observation)
        latency_ms = 1000 * (time.time() - start_time)

        # Extract predicted action
        # The policy returns "actions" (plural), not "action"
        if "actions" in result:
            predicted_action = result["actions"]
        elif "action" in result:
            predicted_action = result["action"]
        else:
            raise KeyError(f"No 'action' or 'actions' key in policy result. Available keys: {list(result.keys())}")

        # Compute error
        mse = compute_action_mse(predicted_action, ground_truth_action)

        # Skip warmup steps from statistics
        if step_idx >= warmup_steps:
            errors.append(mse)
            latencies.append(latency_ms)
            if stats is not None:
                stats.add_step(episode_idx, step_idx, mse, latency_ms)

        if log_per_step:
            server_t = result.get("server_timing", {})
            policy_t = result.get("policy_timing", {})
            pretty = _format_step_timing(
                episode_idx,
                step_idx,
                client_latency_ms=latency_ms,
                server_t=server_t,
                policy_t=policy_t,
                mse=mse,
            )
            logger.info(pretty)

    avg_mse = np.mean(errors) if errors else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0

    return num_steps, avg_mse, avg_latency


def _print_execution_commands(args: Args) -> None:
    """Print formatted, copy-pastable execution commands for both server and client."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    console = Console()

    # Generate server commands for both pruning modes
    server_cmd_no_prune = (
        "# Server command (No Pruning - shows token counts only)\n"
        "OPENPI_INFER_PROFILE=1 uv run scripts/serve_policy.py \\\n"
        "  --pruning OFF \\\n"
        "  policy:checkpoint \\\n"
        "  --policy.config=pi05_pick_place_inference \\\n"
        "  --policy.dir=/path/to/checkpoint \\\n"
        "  &> server_no_pruned.log"
    )

    server_cmd_pruned = (
        "\n# Server command (Pruning Enabled - 75%)\n"
        "OPENPI_INFER_PROFILE=1 uv run scripts/serve_policy.py \\\n"
        "  --pruning ON \\\n"
        "  --prune-ratio 0.75 \\\n"
        "  policy:checkpoint \\\n"
        "  --policy.config=pi05_without_any_lora_pruned_75pct \\\n"
        "  --policy.dir=/path/to/pruned_checkpoint \\\n"
        "  &> server_pruned.log"
    )

    # Generate client command with current args
    client_cmd = (
        f"# Client command (current configuration)\n"
        f"uv run scripts/replay_dataset.py \\\n"
        f"  --dataset_path={args.dataset_path} \\\n"
        f"  --num_episodes={args.num_episodes or 'ALL'} \\\n"
        f"  --warmup_steps={args.warmup_steps} \\\n"
        f"  --host={args.host} \\\n"
        f"  --port={args.port}"
    )
    if args.log_per_step:
        client_cmd += " \\\n  --log_per_step"
    if args.output_path:
        client_cmd += f" \\\n  --output_path={args.output_path}"
    client_cmd += " \\\n  &> client.log"

    # Print commands in panels
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]            Execution Commands (Copy & Paste Ready)                [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════[/bold cyan]\n")

    console.print(Panel(
        Syntax(server_cmd_no_prune, "bash", theme="monokai", line_numbers=False),
        title="[green]Terminal 1: Server (No Pruning)[/green]",
        border_style="green"
    ))

    console.print(Panel(
        Syntax(server_cmd_pruned, "bash", theme="monokai", line_numbers=False),
        title="[yellow]Terminal 1: Server (With Pruning)[/yellow]",
        border_style="yellow"
    ))

    console.print(Panel(
        Syntax(client_cmd, "bash", theme="monokai", line_numbers=False),
        title="[blue]Terminal 2: Client (Dataset Replay)[/blue]",
        border_style="blue"
    ))

    console.print("[dim]Note: Start the server (Terminal 1) first, then run the client (Terminal 2)[/dim]\n")


def main(args: Args) -> None:
    """Main function for dataset replay."""
    # Print execution commands at the start
    _print_execution_commands(args)

    # Load dataset
    dataset = load_dataset(args.dataset_path)

    # Connect to policy server
    logger.info(f"Connecting to policy server at {args.host}:{args.port}...")
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Determine episodes to replay
    total_episodes = len(dataset.episodes)
    end_episode = args.start_episode + args.num_episodes if args.num_episodes else total_episodes
    end_episode = min(end_episode, total_episodes)
    episodes_to_replay = range(args.start_episode, end_episode)

    logger.info(f"Replaying episodes {args.start_episode} to {end_episode-1} ({len(episodes_to_replay)} total)")

    # Replay episodes
    stats = ReplayStats()

    for episode_idx in tqdm.tqdm(episodes_to_replay, desc="Replaying episodes"):
        num_steps, avg_mse, avg_latency = replay_episode(
            policy,
            dataset,
            episode_idx,
            warmup_steps=args.warmup_steps,
            log_per_step=args.log_per_step,
            stats=stats,
        )

        stats.finish_episode(episode_idx, num_steps, avg_mse, avg_latency)

        logger.info(
            f"Episode {episode_idx}: {num_steps} steps, "
            f"MSE={avg_mse:.6f}, Latency={avg_latency:.2f}ms"
        )

    # Print summary
    stats.print_summary()

    # Save results if requested
    if args.output_path:
        stats.save_to_json(args.output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(tyro.cli(Args))
