"""
Performance profiling utilities for OpenPI models.

This module provides tools for collecting and analyzing performance metrics
during model inference, including timing statistics and memory usage.

Usage examples (executed via the runner script that uses this module):

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

For details, see task/pi0_pytorch_profiling.md.
"""

import logging
from typing import Any

import numpy as np
import rich
import rich.console
import rich.table

logger = logging.getLogger(__name__)


class PerformanceRecorder:
    """Records and analyzes performance metrics for model inference.

    This class collects timing measurements and provides statistical analysis
    including mean, standard deviation, and various percentiles.

    Example:
        recorder = PerformanceRecorder()
        recorder.record("inference", 15.3)
        recorder.record("inference", 16.1)
        recorder.print_summary()
    """

    def __init__(self) -> None:
        """Initialize the performance recorder."""
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement.

        Args:
            key: Name of the metric being recorded
            time_ms: Time measurement in milliseconds
        """
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistical summary for a timing key.

        Args:
            key: Name of the metric to analyze

        Returns:
            Dictionary containing statistical measures (count, mean, std, min, max, percentiles)

        Raises:
            KeyError: If the key has not been recorded
        """
        if key not in self._timings:
            raise KeyError(f"No timing data recorded for key: {key}")

        times = self._timings[key]
        return {
            "count": len(times),
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "p25": float(np.percentile(times, 25)),
            "p50": float(np.percentile(times, 50)),
            "p75": float(np.percentile(times, 75)),
            "p90": float(np.percentile(times, 90)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all recorded metrics.

        Returns:
            Dictionary mapping metric names to their statistics
        """
        return {key: self.get_stats(key) for key in self._timings.keys()}

    def print_summary(
        self,
        title: str = "Performance Statistics",
        sort_by: str | None = None,
    ) -> None:
        """Print a formatted table of performance statistics.

        Args:
            title: Title to display at the top of the table
            sort_by: Optional key to sort by (e.g., "mean", "p95"). If None, sorts by metric name.
        """
        if not self._timings:
            logger.warning("No timing data to display")
            return

        console = rich.console.Console()

        table = rich.table.Table(
            title=f"[bold blue]{title}[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
        )

        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)
        table.add_column("Count", style="white", justify="right")
        table.add_column("Mean", style="yellow", justify="right")
        table.add_column("Std", style="yellow", justify="right")
        table.add_column("Min", style="green", justify="right")
        table.add_column("Max", style="red", justify="right")
        table.add_column("P50", style="magenta", justify="right")
        table.add_column("P95", style="magenta", justify="right")
        table.add_column("P99", style="magenta", justify="right")

        # Get all stats and optionally sort
        all_stats = [(key, self.get_stats(key)) for key in self._timings.keys()]
        if sort_by and sort_by != "name":
            try:
                all_stats.sort(key=lambda x: x[1][sort_by], reverse=True)
            except KeyError:
                logger.warning(f"Invalid sort key: {sort_by}, sorting by name instead")
                all_stats.sort(key=lambda x: x[0])
        else:
            all_stats.sort(key=lambda x: x[0])

        for key, stats in all_stats:
            table.add_row(
                key,
                f"{int(stats['count'])}",
                f"{stats['mean']:.2f} ms",
                f"{stats['std']:.2f} ms",
                f"{stats['min']:.2f} ms",
                f"{stats['max']:.2f} ms",
                f"{stats['p50']:.2f} ms",
                f"{stats['p95']:.2f} ms",
                f"{stats['p99']:.2f} ms",
            )

        console.print("\n")
        console.print(table)
        console.print("\n")

    def reset(self) -> None:
        """Clear all recorded timing data."""
        self._timings.clear()

    def get_raw_timings(self, key: str) -> list[float]:
        """Get raw timing measurements for a key.

        Args:
            key: Name of the metric

        Returns:
            List of recorded timing values in milliseconds

        Raises:
            KeyError: If the key has not been recorded
        """
        if key not in self._timings:
            raise KeyError(f"No timing data recorded for key: {key}")
        return self._timings[key].copy()

    def keys(self) -> list[str]:
        """Get list of all recorded metric names.

        Returns:
            List of metric names that have been recorded
        """
        return list(self._timings.keys())


"""
Note: GPU memory trackers and model-parameter summaries were removed to keep
profiling minimal and pluggable. Only a generic timing recorder is retained.
"""
