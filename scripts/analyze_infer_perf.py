"""Aggregate and compare inference profiling metrics from policy_records.

Usage examples:

1) Single run (prints averages after warmup):
   uv run scripts/analyze_infer_perf.py policy_records --warmup 20

2) Compare two runs (rename dirs after each run):
   mv policy_records policy_records_pruned
   # run the second server with --record, then client, then:
   mv policy_records policy_records_noprune
   uv run scripts/analyze_infer_perf.py policy_records_pruned policy_records_noprune --warmup 20
"""

from __future__ import annotations

import dataclasses
import glob
import os
from typing import Any, Dict, List

import numpy as np
import tyro


def _load_steps(record_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(record_dir, "step_*.npy")))
    steps: List[Dict[str, Any]] = []
    for f in files:
        try:
            arr = np.load(f, allow_pickle=True)
            data = arr.item() if isinstance(arr, np.ndarray) else arr
            assert isinstance(data, dict)
            steps.append(data)
        except Exception:
            # Skip unreadable files silently
            continue
    return steps


def _mean_numeric(values: List[Any]) -> float | None:
    xs: List[float] = []
    for v in values:
        try:
            xs.append(float(v))
        except Exception:
            pass
    if not xs:
        return None
    return float(sum(xs) / len(xs))


@dataclasses.dataclass
class Args:
    # One or more policy_records directories to summarize/compare
    record_dirs: List[str]
    # Number of warmup steps to discard from the beginning
    warmup: int = 20
    # Optionally limit the number of steps after warmup
    limit: int | None = None


def summarize_dir(record_dir: str, *, warmup: int, limit: int | None) -> Dict[str, float | None]:
    steps = _load_steps(record_dir)
    if not steps:
        return {"count": 0}
    data = steps[warmup : (None if limit is None else warmup + limit)]
    count = len(data)

    def col(key: str) -> List[Any]:
        return [d.get(f"outputs/policy_timing/{key}") for d in data]

    keys = [
        "infer_ms",
        "input_ms",
        "output_ms",
        "prefix_ms",
        "prefix_orig_ms",
        "prefix_len",
        "prefix_keep_ratio",
        "gpu0_mem_mb",
    ]

    out: Dict[str, float | None] = {"count": float(count)}
    for k in keys:
        out[f"mean/{k}"] = _mean_numeric(col(k))
    return out


def main(args: Args) -> None:
    rows = []
    labels = []
    for rd in args.record_dirs:
        rows.append(summarize_dir(rd, warmup=args.warmup, limit=args.limit))
        labels.append(os.path.basename(os.path.abspath(rd)))

    # Print simple aligned output
    all_keys = sorted({k for row in rows for k in row.keys()})
    print("dir\t" + "\t".join(all_keys))
    for lbl, row in zip(labels, rows, strict=True):
        vals = [row.get(k) for k in all_keys]
        def fmt(v: Any) -> str:
            if v is None:
                return "-"
            try:
                return f"{float(v):.4f}"
            except Exception:
                return str(v)
        print(lbl + "\t" + "\t".join(fmt(v) for v in vals))


if __name__ == "__main__":
    tyro.cli(main)

