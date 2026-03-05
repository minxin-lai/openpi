#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _as_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s in ("", "nan", "NaN", "None"):
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _fmt(x: float, *, digits: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{float(x):.{digits}f}"


def _read_grid(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for r in reader:
            out: dict[str, Any] = dict(r)
            for k in list(out.keys()):
                if k in (
                    "threshold",
                    "margin",
                    "count_dumps",
                    "shared_ratio_a_mean",
                    "shared_ratio_b_mean",
                    "unique_ratio_cls_a_mean",
                    "unique_ratio_cls_b_mean",
                    "tokens_thr_rate_a_mean",
                    "tokens_margin_rate_a_mean",
                    "tokens_mutual_rate_a_mean",
                    "shared_ratio_a_p10",
                    "shared_ratio_a_p50",
                    "shared_ratio_a_p90",
                    "unique_ratio_cls_a_p10",
                    "unique_ratio_cls_a_p50",
                    "unique_ratio_cls_a_p90",
                ):
                    out[k] = _as_float(out.get(k))
            rows.append(out)
    return rows


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _suggest_ranges(meta: dict[str, Any]) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    best = meta.get("global_best") if isinstance(meta, dict) else None
    delta = meta.get("global_delta") if isinstance(meta, dict) else None

    thr_rng = None
    if isinstance(best, dict):
        p10 = _as_float(best.get("p10"))
        p90 = _as_float(best.get("p90"))
        if not (math.isnan(p10) or math.isnan(p90)):
            lo = max(-1.0, p10 - 0.01)
            hi = min(1.0, p90 + 0.01)
            thr_rng = (float(lo), float(hi))

    m_rng = None
    if isinstance(delta, dict):
        p90 = _as_float(delta.get("p90"))
        if not math.isnan(p90):
            m_rng = (0.0, float(max(0.0, p90 * 1.5)))

    return thr_rng, m_rng


def _suggest_step(width: float, *, base: float) -> float:
    """
    Pick a human-friendly step size given a typical width.
    This is only for printing copy/paste hints (not used in computation).
    """
    width = float(abs(width))
    if width <= 0:
        return float(base)
    # Aim for ~20-30 points.
    raw = width / 25.0
    # Round up to a multiple of base.
    n = math.ceil(raw / float(base))
    return float(max(base, n * float(base)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect consensus sweep grid.csv/meta.json and print a compact report.")
    ap.add_argument("--grid", default="", help="Path to grid.csv (preferred).")
    ap.add_argument("--meta", default="", help="Optional path to meta.json (for suggested scan ranges).")
    ap.add_argument("--trace", default="", help="Trace dir under third_party/openpi, e.g. runs/<trace>.")
    ap.add_argument("--tag", default="", help="Sweep tag (folder under <trace>/plots/consensus_sweep/).")
    ap.add_argument("--margin", type=float, default=0.0, help="Which margin slice to print as a curve (default 0.0).")
    ap.add_argument("--top", type=int, default=8, help="Show top-N configs (default 8).")
    args = ap.parse_args(argv)

    if str(args.grid).strip():
        grid_path = Path(str(args.grid)).expanduser()
        trace_dir = None
    else:
        if not (str(args.trace).strip() and str(args.tag).strip()):
            ap.error("Either --grid or (--trace and --tag) is required.")
        trace_dir = Path(str(args.trace)).expanduser()
        grid_path = trace_dir / "plots" / "consensus_sweep" / str(args.tag) / "grid.csv"

    if not grid_path.exists():
        raise FileNotFoundError(str(grid_path))

    if str(args.meta).strip():
        meta_path = Path(str(args.meta)).expanduser()
    else:
        if trace_dir is not None:
            meta_path = grid_path.parent / "meta.json"
        else:
            meta_path = grid_path.parent / "meta.json"

    meta = _read_meta(meta_path) if meta_path.exists() else {}
    rows = _read_grid(grid_path)
    if not rows:
        raise RuntimeError(f"Empty grid: {str(grid_path)}")

    margins = sorted({float(r["margin"]) for r in rows if not math.isnan(float(r.get("margin", float("nan"))))})
    thresholds = sorted({float(r["threshold"]) for r in rows if not math.isnan(float(r.get("threshold", float("nan"))))})
    used_dumps = meta.get("used_dumps")
    pair = meta.get("pair")

    print(f"grid: {grid_path}")
    if meta_path.exists():
        print(f"meta: {meta_path}")
    if used_dumps is not None:
        print(f"used_dumps: {used_dumps}")
    if isinstance(pair, dict):
        a_name = pair.get("a_name", "viewA")
        b_name = pair.get("b_name", "viewB")
        print(f"pair: {pair.get('a')}({a_name}) vs {pair.get('b')}({b_name})")
    print(f"rows: {len(rows)}  thresholds: {len(thresholds)}  margins: {len(margins)}")

    thr_rng, m_rng = _suggest_ranges(meta)
    if thr_rng is not None or m_rng is not None:
        print("suggested_scan_ranges:")
        if thr_rng is not None:
            print(f"  thresholds ~ [{thr_rng[0]:.3f}, {thr_rng[1]:.3f}]")
        if m_rng is not None:
            print(f"  margins    ~ [{m_rng[0]:.6f}, {m_rng[1]:.6f}]")
        if thr_rng is not None and m_rng is not None:
            thr_step = _suggest_step(thr_rng[1] - thr_rng[0], base=0.001)
            m_step = _suggest_step(m_rng[1] - m_rng[0], base=0.0005)
            print("  range_syntax: 'start:stop:step' (inclusive-ish), e.g. 0.93:0.98:0.002")
            print(f"  suggested_steps: thresholds step≈{thr_step:.3f}, margins step≈{m_step:.6f}")

    # Global extrema for sanity.
    shared_min = min(float(r["shared_ratio_a_mean"]) for r in rows)
    shared_max = max(float(r["shared_ratio_a_mean"]) for r in rows)
    print(f"shared_ratio_a_mean: min={_fmt(shared_min)} max={_fmt(shared_max)}")

    # Print a threshold curve for one margin slice.
    m_sel = float(args.margin)
    slice_rows = [r for r in rows if abs(float(r["margin"]) - m_sel) < 1e-12]
    if not slice_rows:
        print(f"no_rows_for_margin: {m_sel}")
    else:
        slice_rows = sorted(slice_rows, key=lambda r: float(r["threshold"]))
        print(f"\ncurve: margin={m_sel:g}  (threshold -> shared / thr_rate / mutual_rate / margin_rate)")
        step = max(1, len(slice_rows) // 12)
        for r in slice_rows[::step]:
            print(
                f'{float(r["threshold"]):.3f}\t'
                f'shared={_fmt(float(r["shared_ratio_a_mean"]), digits=6)}\t'
                f'thr_rate={_fmt(float(r["tokens_thr_rate_a_mean"]), digits=3)}\t'
                f'mutual={_fmt(float(r["tokens_mutual_rate_a_mean"]), digits=3)}\t'
                f'margin_rate={_fmt(float(r["tokens_margin_rate_a_mean"]), digits=3)}'
            )

    # Top configs (by shared ratio).
    top_n = max(1, int(args.top))
    top = sorted(rows, key=lambda r: float(r["shared_ratio_a_mean"]), reverse=True)[:top_n]
    print(f"\ntop_by_shared (n={len(top)}):")
    for r in top:
        print(
            f'thr={float(r["threshold"]):.3f} m={float(r["margin"]):.6f} '
            f'shared={_fmt(float(r["shared_ratio_a_mean"]), digits=6)} '
            f'unique_cls={_fmt(float(r["unique_ratio_cls_a_mean"]), digits=6)} '
            f'thr_rate={_fmt(float(r["tokens_thr_rate_a_mean"]), digits=3)} '
            f'margin_rate={_fmt(float(r["tokens_margin_rate_a_mean"]), digits=3)} '
            f'mutual={_fmt(float(r["tokens_mutual_rate_a_mean"]), digits=3)}'
        )

    # Convenience: print a suggested viz command for the best row.
    best = top[0]
    print("\nviz_hint:")
    thr = float(best["threshold"])
    m = float(best["margin"])
    if trace_dir is None:
        print(f'  bash viz_trace_overlays.sh "<TRACE_DIR>" --plots-subdir "plots_thr{thr:.3f}_m{m:.6f}" --consensus-threshold {thr:.3f} --consensus-margin {m:.6f}')
    else:
        print(
            f'  bash viz_trace_overlays.sh "{trace_dir}" '
            f'--plots-subdir "plots_thr{thr:.3f}_m{m:.6f}" '
            f'--consensus-threshold {thr:.3f} --consensus-margin {m:.6f}'
        )

    if thr_rng is not None and m_rng is not None and trace_dir is not None:
        a_idx = 0
        b_idx = 1
        if isinstance(pair, dict):
            try:
                a_idx = int(pair.get("a", a_idx))
                b_idx = int(pair.get("b", b_idx))
            except Exception:
                a_idx, b_idx = 0, 1
        thr_step = _suggest_step(thr_rng[1] - thr_rng[0], base=0.001)
        m_step = _suggest_step(m_rng[1] - m_rng[0], base=0.0005)
        print("\nsweep_hint:")
        print(
            "  .venv/bin/python ../../tools/view_consensus/sweep_threshold_margin.py "
            f'--input "{trace_dir}/dumps" --pair {a_idx},{b_idx} '
            f'--thresholds {thr_rng[0]:.3f}:{thr_rng[1]:.3f}:{thr_step:.3f} '
            f'--margins {m_rng[0]:.6f}:{m_rng[1]:.6f}:{m_step:.6f} '
            '--max-dumps 200 --plots --tag sweep_tight'
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
