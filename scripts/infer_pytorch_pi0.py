"""
Minimal inference + profiling script for PyTorch pi0.

Supports exactly three modes described in task/pi0_pytorch_profiling.md:
  1) Baseline (fastest, torch.compile)
  3) Baseline + Trace (export Chrome trace)
  4) Component + Trace (enable component ranges + export Chrome trace)

Usage examples (copy-paste ready):

# Mode 1: Baseline (fastest, with torch.compile)
uv run scripts/infer_pytorch_pi0.py \
  --checkpoint_dir /home/laiminxin/pi0_base_pytorch \
  --config_name pi0_droid \
  --num_steps 20 \
  --warmup_steps 5

# Mode 3: Baseline + Trace (export Chrome trace, low overhead)
uv run scripts/infer_pytorch_pi0.py \
  --checkpoint_dir /home/laiminxin/pi0_base_pytorch \
  --config_name pi0_droid \
  --num_steps 20 \
  --warmup_steps 5 \
  --enable_profiler \
  --profiler-num-steps 3 \
  --profiler-output ./pi0_profiler_trace.json

# Mode 4: Component + Trace (named component ranges + Chrome trace)
uv run scripts/infer_pytorch_pi0.py \
  --checkpoint_dir /home/laiminxin/pi0_base_pytorch \
  --config_name pi0_droid \
  --num_steps 20 \
  --warmup_steps 5 \
  --enable-component-profile \
  --enable_profiler \
  --profiler-num-steps 3 \
  --profiler-output ./pi0_profiler_trace.json
"""

import dataclasses
import logging
import pathlib
import time
from typing import Any

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.shared import profiling
from openpi.training import config as _config

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Path to PyTorch checkpoint directory
    checkpoint_dir: pathlib.Path
    # Config name (e.g., pi0_droid, pi0_aloha_sim, pi0_aloha)
    config_name: str
    # Number of inference steps to run
    num_steps: int = 50
    # Number of warmup steps (not included in statistics)
    warmup_steps: int = 5
    # Device to use (e.g., "cuda", "cuda:0", "cpu")
    device: str | None = None
    # Enable PyTorch profiler for exporting Chrome trace
    enable_profiler: bool = False
    # Enable model-internal component profiling (record_function ranges)
    enable_component_profile: bool = False
    # Path to save profiler trace (only if enable_profiler=True)
    profiler_output: pathlib.Path | None = None
    # Start profiling at this step (default: after warmup)
    profiler_start_step: int | None = None
    # Number of steps to profile (default: 5)
    profiler_num_steps: int = 5
    # Prompt to use for inference
    prompt: str = "pick up the object"

def create_dummy_observation_droid(prompt: str) -> dict[str, Any]:
    """Create a dummy DROID observation."""
    return {
        "observation/exterior_image_1_left": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7).astype(np.float32),
        "observation/gripper_position": np.random.rand(1).astype(np.float32),
        "prompt": prompt,
    }


def create_dummy_observation_aloha(prompt: str) -> dict[str, Any]:
    """Create a dummy ALOHA observation."""
    return {
        "state": np.random.rand(14).astype(np.float32),
        "images": {
            "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": prompt,
    }


def create_dummy_observation_yuanluo(prompt: str) -> dict[str, Any]:
    """Create a dummy Yuanluo observation matching yuanluo_policy expectations."""
    return {
        # YuanluoInputs expects CHW or HWC; if CHW it will transpose to HWC.
        "observation.images.head_camera": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        "observation.images.wrist_left_camera": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        "observation.images.gelsight_left": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        # YuanluoInputs slices [:7]
        "observation.state": np.random.rand(7).astype(np.float32),
        "prompt": prompt,
    }


def get_observation_fn(config_name: str):
    """Get the appropriate observation function based on config name."""
    name = config_name.lower()
    if "droid" in name:
        return create_dummy_observation_droid
    elif "aloha" in name:
        return create_dummy_observation_aloha
    elif "yuanluo" in name:
        return create_dummy_observation_yuanluo
    else:
        logger.warning(f"Unknown config name: {config_name}, defaulting to DROID observation")
        return create_dummy_observation_droid


def check_pytorch_model(checkpoint_dir: pathlib.Path) -> bool:
    """Check if the checkpoint is a PyTorch model."""
    weight_path = checkpoint_dir / "model.safetensors"
    return weight_path.exists()
def main(args: Args) -> None:
    """Main function."""
    import os

    # Enable detailed profiling in policy.infer()
    os.environ["OPENPI_INFER_PROFILE"] = "1"
    # Enable component-level profiling inside PyTorch model only when explicitly requested
    if args.enable_component_profile:
        os.environ["OPENPI_COMPONENT_PROFILE"] = "1"
    else:
        os.environ.pop("OPENPI_COMPONENT_PROFILE", None)

    # Check if checkpoint is PyTorch
    if not check_pytorch_model(args.checkpoint_dir):
        print(
            f"Error: {args.checkpoint_dir} does not contain a PyTorch model (model.safetensors not found)"
        )
        print("Please convert the JAX model to PyTorch first using convert_jax_model_to_pytorch.py")
        return

    print(f"Loading PyTorch model from {args.checkpoint_dir}...")

    # Load config and policy
    config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        config,
        str(args.checkpoint_dir),
        pytorch_device=args.device,
    )

    # Determine actual device used
    try:
        import torch

        if args.device:
            device_str = args.device
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device_str = args.device or "cpu"

    # Get observation function
    obs_fn = get_observation_fn(args.config_name)

    print(f"Running {args.warmup_steps} warmup steps...")

    # Warmup
    for i in range(args.warmup_steps):
        obs = obs_fn(args.prompt)
        _ = policy.infer(obs)
        if i == 0:
            print("✓ First inference completed successfully")

    print(f"Running {args.num_steps} inference steps for profiling...")

    # Performance recorder (only overall inference time)
    recorder = profiling.PerformanceRecorder()

    # Profiler context (optional)
    profiler_ctx = None
    profiler_start = 0
    profiler_end = 0

    if args.enable_profiler:
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile, schedule

            # Determine profiling range
            profiler_start = args.profiler_start_step if args.profiler_start_step is not None else args.warmup_steps
            profiler_end = profiler_start + args.profiler_num_steps

            output_path = args.profiler_output or pathlib.Path("./pytorch_profiler_trace.json")
            print("PyTorch profiler enabled")
            print(f"  Profiling steps {profiler_start} to {profiler_end-1} ({args.profiler_num_steps} steps)")
            print(f"  Trace will be saved to {output_path}")

            # Custom schedule: only profile specific steps
            def custom_schedule(step: int):
                if step < profiler_start:
                    return torch.profiler.ProfilerAction.NONE
                elif step < profiler_end - 1:
                    return torch.profiler.ProfilerAction.RECORD
                elif step == profiler_end - 1:
                    return torch.profiler.ProfilerAction.RECORD_AND_SAVE
                else:
                    return torch.profiler.ProfilerAction.NONE

            profiler_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=custom_schedule,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
            )
            profiler_ctx.__enter__()
        except ImportError:
            print("Warning: PyTorch not available, profiler disabled")

    # Run inference
    try:
        for step in range(args.num_steps):
            obs = obs_fn(args.prompt)

            # Measure total inference time
            start_time = time.time()
            result = policy.infer(obs)
            end_time = time.time()

            inference_time_ms = (end_time - start_time) * 1000
            recorder.record("total_inference", inference_time_ms)

            # Extract timing info from result if available
            # Step the profiler
            if profiler_ctx is not None:
                profiler_ctx.step()

            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{args.num_steps} - {inference_time_ms:.2f} ms")

    finally:
        if profiler_ctx is not None:
            profiler_ctx.__exit__(None, None, None)
            output_path = args.profiler_output or pathlib.Path("./pytorch_profiler_trace.json")
            profiler_ctx.export_chrome_trace(str(output_path))
            print(f"✓ Profiler trace saved to {output_path}")
            print(f"  Collected data for steps {profiler_start} to {profiler_end-1}")

    # Print summary (overall inference time)
    recorder.print_summary(title="PyTorch pi0 Inference Performance (overall)")
    print("✓ Inference profiling completed!\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(tyro.cli(Args))
