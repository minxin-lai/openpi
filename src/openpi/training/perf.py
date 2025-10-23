import contextlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _get_process_rss_mb() -> float:
    """Returns current process RSS in MB. Fallbacks to /proc if psutil is unavailable."""
    try:
        import psutil  # type: ignore

        p = psutil.Process()
        return float(p.memory_info().rss) / (1024 * 1024)
    except Exception:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                parts = f.read().split()
                pages = int(parts[1]) if len(parts) > 1 else 0
            # get pagesize in bytes
            import resource  # type: ignore

            page_size = resource.getpagesize()
            return float(pages * page_size) / (1024 * 1024)
        except Exception:
            return 0.0


@dataclass
class PerformanceProfiler:
    """Lightweight profiler for training.

    Tracks per-stage durations, approximate throughput, and RSS memory.
    Designed to minimize code changes and overhead.
    """

    enabled: bool = False
    batch_size: int = 0
    log_to_wandb: bool = True
    stage_times: Dict[str, List[float]] = field(default_factory=dict)
    step_times: List[float] = field(default_factory=list)
    last_step_wall_start: Optional[float] = None
    step_count: int = 0
    latest_prefix_len: Optional[int] = None
    latest_prefix_orig_len: Optional[int] = None
    # Attempt to log GPU 0 memory if NVML is available.
    _nvml_inited: bool = field(default=False, init=False)

    def next_step(self) -> None:
        if not self.enabled:
            return
        self.step_count += 1
        self.last_step_wall_start = time.perf_counter()

    def end_step(self) -> None:
        if not self.enabled:
            return
        if self.last_step_wall_start is not None:
            dt = time.perf_counter() - self.last_step_wall_start
            self.step_times.append(dt)
            self.last_step_wall_start = None

    @contextlib.contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.stage_times.setdefault(name, []).append(dt)

    def snapshot_memory_mb(self) -> float:
        if not self.enabled:
            return 0.0
        return _get_process_rss_mb()

    def set_prefix_len(self, n_tokens: int) -> None:
        if not self.enabled:
            return
        self.latest_prefix_len = n_tokens

    def set_prefix_orig_len(self, n_tokens: int) -> None:
        if not self.enabled:
            return
        self.latest_prefix_orig_len = n_tokens

    def flush(self, step: int) -> Dict[str, float]:
        """Returns dict of metrics and clears accumulators."""
        if not self.enabled:
            return {}
        metrics: Dict[str, float] = {}
        if self.step_times:
            mean_step = sum(self.step_times) / len(self.step_times)
            metrics["perf/step_time_s"] = mean_step
            if self.batch_size:
                metrics["perf/examples_per_s"] = self.batch_size / mean_step
        for k, vs in self.stage_times.items():
            if vs:
                metrics[f"perf/{k}_s"] = sum(vs) / len(vs)
        rss = self.snapshot_memory_mb()
        if rss > 0:
            metrics["perf/rss_mb"] = rss
        # Optional: GPU memory via NVML
        try:
            import pynvml  # type: ignore

            if not self._nvml_inited:
                pynvml.nvmlInit()
                self._nvml_inited = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["perf/gpu0_mem_mb"] = float(mem.used) / (1024 * 1024)
        except Exception:
            pass
        if self.latest_prefix_len is not None:
            metrics["perf/prefix_len"] = float(self.latest_prefix_len)
        if self.latest_prefix_orig_len is not None:
            metrics["perf/prefix_orig_len"] = float(self.latest_prefix_orig_len)
        if self.latest_prefix_len is not None and self.latest_prefix_orig_len is not None and self.latest_prefix_orig_len > 0:
            metrics["perf/prefix_keep_ratio"] = float(self.latest_prefix_len) / float(self.latest_prefix_orig_len)

        # reset
        self.stage_times.clear()
        self.step_times.clear()
        self.latest_prefix_len = None
        self.latest_prefix_orig_len = None
        return metrics
