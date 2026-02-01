#!/usr/bin/env python3
from __future__ import annotations

"""
Generate a Markdown report (with plots) from a debug_kv run directory.

This script reads:
  - runs/<run>/baseline/server.log  (expects one "OPENPI_DEBUG {json}" line)
  - runs/<run>/vla_opt/server.log   (expects one "OPENPI_DEBUG {json}" line)
  - runs/<run>/*/timing.parquet     (client/server/policy timing columns)

It writes:
  - runs/<run>/report.md
  - runs/<run>/plots/*.png

Example:
  cd third_party/openpi
  uv run python scripts/generate_debug_kv_report.py \
    --run-dir runs/debug_kv_pi05_libero_20260131_211413 \
    # (default outputs)
    #   runs/debug_kv_pi05_libero_20260131_211413/report.md
    #   runs/debug_kv_pi05_libero_20260131_211413/plots/*.png
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import tyro


@dataclass(frozen=True)
class Args:
    run_dir: Path
    # Optional overrides. By default, the report is written under `run_dir/`.
    out_md: Path | None = None
    out_plot_dir: Path | None = None


def _read_last_openpi_debug_json(server_log: Path) -> dict[str, Any]:
    txt = server_log.read_text(errors="ignore")
    matches = re.findall(r"OPENPI_DEBUG\s+(\{.*\})", txt)
    if not matches:
        raise FileNotFoundError(f"No OPENPI_DEBUG JSON found in: {server_log}")
    return json.loads(matches[-1])


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        if part not in cur:
            return default
        cur = cur[part]
    return cur


def _fmt_int(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return str(int(v))
    except Exception:
        return str(v)


def _fmt_float(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _fmt_mb(v_bytes: Any) -> str:
    if v_bytes is None:
        return "—"
    try:
        mb = float(v_bytes) / (1024.0 * 1024.0)
        return f"{mb:.2f} MB"
    except Exception:
        return str(v_bytes)


def _ratio(a: Any, b: Any) -> Optional[float]:
    try:
        af = float(a)
        bf = float(b)
        if bf == 0:
            return None
        return af / bf
    except Exception:
        return None


def _save_bar(
    *,
    out_path: Path,
    title: str,
    labels: list[str],
    baseline: list[float],
    vla_opt: list[float],
    ylabel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.bar([i - width / 2 for i in x], baseline, width=width, label="baseline")
    ax.bar([i + width / 2 for i in x], vla_opt, width=width, label="vla_opt")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_stacked_prefix_composition(
    *,
    out_path: Path,
    baseline_vis: float,
    baseline_lang: float,
    vla_vis: float,
    vla_lang: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    labels = ["baseline", "vla_opt"]
    vis = [baseline_vis, vla_vis]
    lang = [baseline_lang, vla_lang]
    ax.bar(labels, vis, label="vision tokens")
    ax.bar(labels, lang, bottom=vis, label="lang tokens")
    ax.set_ylabel("tokens")
    ax.set_title("Prefix composition (vision + lang)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _polars_stats_one_row(df: pl.DataFrame) -> dict[str, Any]:
    # In this debug run, timing.parquet typically has 1 row; still handle N>1.
    out: dict[str, Any] = {"n": df.height, "cols": df.columns, "stats": {}}
    for col in df.columns:
        if df[col].dtype not in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            continue
        s = df[col].cast(pl.Float64)
        out["stats"][col] = {
            "mean": float(s.mean()),
            "p50": float(s.quantile(0.5, "nearest")),
            "p90": float(s.quantile(0.9, "nearest")),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return out


def main(args: Args) -> None:
    run_dir = args.run_dir
    baseline_log = run_dir / "baseline" / "server.log"
    vla_log = run_dir / "vla_opt" / "server.log"
    baseline_timing = run_dir / "baseline" / "timing.parquet"
    vla_timing = run_dir / "vla_opt" / "timing.parquet"

    out_md = args.out_md if args.out_md is not None else (run_dir / "report.md")
    out_plot_dir = args.out_plot_dir if args.out_plot_dir is not None else (run_dir / "plots")

    b = _read_last_openpi_debug_json(baseline_log)
    v = _read_last_openpi_debug_json(vla_log)

    # Derived fields.
    b_prefix = _get(b, "token.prefix_len")
    v_prefix = _get(v, "token.prefix_len")
    b_full = _get(b, "token.full_len")
    v_full = _get(v, "token.full_len")

    # Timing.
    b_df = pl.read_parquet(baseline_timing) if baseline_timing.exists() else pl.DataFrame()
    v_df = pl.read_parquet(vla_timing) if vla_timing.exists() else pl.DataFrame()
    b_t = _polars_stats_one_row(b_df) if b_df.height else None
    v_t = _polars_stats_one_row(v_df) if v_df.height else None

    # Plots.
    plot_dir = out_plot_dir
    plot_token_bars = plot_dir / "token_lengths.png"
    plot_kv_bars = plot_dir / "kv_cache_bytes.png"
    plot_timing_bars = plot_dir / "timing_ms.png"
    plot_prefix_stack = plot_dir / "prefix_composition.png"

    _save_bar(
        out_path=plot_token_bars,
        title="Token lengths (debug sample)",
        labels=["img_total", "lang_len", "prefix_len", "suffix_len", "full_len"],
        baseline=[
            float(_get(b, "token.img_tokens_total_after_prune") or 0),
            float(_get(b, "token.lang_tokens_len") or 0),
            float(_get(b, "token.prefix_len") or 0),
            float(_get(b, "token.suffix_len") or 0),
            float(_get(b, "token.full_len") or 0),
        ],
        vla_opt=[
            float(_get(v, "token.img_tokens_total_after_prune") or 0),
            float(_get(v, "token.lang_tokens_len") or 0),
            float(_get(v, "token.prefix_len") or 0),
            float(_get(v, "token.suffix_len") or 0),
            float(_get(v, "token.full_len") or 0),
        ],
        ylabel="tokens",
    )

    _save_stacked_prefix_composition(
        out_path=plot_prefix_stack,
        baseline_vis=float(_get(b, "token.img_tokens_total_after_prune") or 0),
        baseline_lang=float(_get(b, "token.lang_tokens_len") or 0),
        vla_vis=float(_get(v, "token.img_tokens_total_after_prune") or 0),
        vla_lang=float(_get(v, "token.lang_tokens_len") or 0),
    )

    _save_bar(
        out_path=plot_kv_bars,
        title="KV cache size (bytes)",
        labels=["kv.prefix.total_bytes", "kv.suffix_step0.total_bytes"],
        baseline=[
            float(_get(b, "kv.prefix.total_bytes") or 0),
            float(_get(b, "kv.suffix_step0.total_bytes") or 0),
        ],
        vla_opt=[
            float(_get(v, "kv.prefix.total_bytes") or 0),
            float(_get(v, "kv.suffix_step0.total_bytes") or 0),
        ],
        ylabel="bytes",
    )

    if b_df.height and v_df.height:
        _save_bar(
            out_path=plot_timing_bars,
            title="Timing (ms) from timing.parquet",
            labels=["client_infer_ms", "server_infer_ms", "policy_infer_ms"],
            baseline=[
                float(b_df["client_infer_ms"][0]),
                float(b_df["server_infer_ms"][0]),
                float(b_df["policy_infer_ms"][0]),
            ],
            vla_opt=[
                float(v_df["client_infer_ms"][0]),
                float(v_df["server_infer_ms"][0]),
                float(v_df["policy_infer_ms"][0]),
            ],
            ylabel="ms",
        )

    # Report markdown.
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = run_dir.name

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(out_md.parent))
        except Exception:
            return str(p)

    md: list[str] = []
    md.append(f"# Debug KV Report: {run_name}")
    md.append("")
    md.append(f"- Generated: {now}")
    md.append(f"- Run dir: `{run_dir}`")
    md.append(f"- Baseline log: `{baseline_log}`")
    md.append(f"- VLA-OPT log: `{vla_log}`")
    md.append(f"- Baseline timing: `{baseline_timing}`")
    md.append(f"- VLA-OPT timing: `{vla_timing}`")
    md.append("")

    md.append("## Config Summary")
    md.append("")
    md.append("| Variant | ve_film | ve_pruning | k | stage | tau | prune_layer_resolved | vision_num_layers |")
    md.append("|---|---:|---:|---:|---|---:|---:|---:|")
    md.append(
        "| baseline | false | false | — | — | — | — | — |"
    )
    md.append(
        "| vla_opt | true | true | "
        + _fmt_int(_get(v, "vla_opt.ste.k"))
        + " | "
        + str(_get(v, "vla_opt.ste.stage") or "—")
        + " | "
        + _fmt_float(_get(v, "vla_opt.ste.tau"), 3)
        + " | "
        + _fmt_int(_get(v, "vla_opt.ste.prune_layer_resolved"))
        + " | "
        + _fmt_int(_get(v, "vla_opt.ste.vision_num_layers"))
        + " |"
    )
    md.append("")

    md.append("## Key Findings")
    md.append("")
    md.append(
        f"- Vision tokens reduced from `{_fmt_int(_get(b,'token.img_tokens_total_after_prune'))}` to `{_fmt_int(_get(v,'token.img_tokens_total_after_prune'))}`."
    )
    md.append(
        f"- Prefix length reduced from `{_fmt_int(b_prefix)}` to `{_fmt_int(v_prefix)}` (ratio `{_fmt_float(_ratio(b_prefix, v_prefix), 3)}x`)."
    )
    md.append(
        f"- Suffix attention K/V length (`full_len`) reduced from `{_fmt_int(b_full)}` to `{_fmt_int(v_full)}` (ratio `{_fmt_float(_ratio(b_full, v_full), 3)}x`)."
    )
    md.append(
        f"- Prefix KV cache bytes reduced from `{_fmt_mb(_get(b,'kv.prefix.total_bytes'))}` to `{_fmt_mb(_get(v,'kv.prefix.total_bytes'))}`."
    )
    md.append("")

    md.append("## Figures")
    md.append("")
    md.append(f"![token lengths]({rel(plot_token_bars)})")
    md.append("")
    md.append(f"![prefix composition]({rel(plot_prefix_stack)})")
    md.append("")
    md.append(f"![kv bytes]({rel(plot_kv_bars)})")
    md.append("")
    if plot_timing_bars.exists():
        md.append(f"![timing ms]({rel(plot_timing_bars)})")
        md.append("")

    md.append("## Numeric Tables")
    md.append("")
    md.append("### Token & KV Summary (from OPENPI_DEBUG)")
    md.append("")
    md.append("| Metric | baseline | vla_opt | ratio (baseline/vla_opt) |")
    md.append("|---|---:|---:|---:|")
    for metric, bp, vp in [
        ("img_tokens_total_after_prune", _get(b, "token.img_tokens_total_after_prune"), _get(v, "token.img_tokens_total_after_prune")),
        ("lang_tokens_len", _get(b, "token.lang_tokens_len"), _get(v, "token.lang_tokens_len")),
        ("prefix_len", b_prefix, v_prefix),
        ("suffix_len", _get(b, "token.suffix_len"), _get(v, "token.suffix_len")),
        ("full_len", b_full, v_full),
        ("kv.prefix.total_bytes", _get(b, "kv.prefix.total_bytes"), _get(v, "kv.prefix.total_bytes")),
        ("kv.suffix_step0.total_bytes", _get(b, "kv.suffix_step0.total_bytes"), _get(v, "kv.suffix_step0.total_bytes")),
    ]:
        r = _ratio(bp, vp)
        if "bytes" in metric:
            b_disp = _fmt_mb(bp)
            v_disp = _fmt_mb(vp)
        else:
            b_disp = _fmt_int(bp)
            v_disp = _fmt_int(vp)
        md.append(f"| `{metric}` | {b_disp} | {v_disp} | {_fmt_float(r, 3) if r is not None else '—'} |")
    md.append("")

    md.append("### Attention Mask Shape (step0)")
    md.append("")
    md.append("| Variant | full_att_2d_masks_4d_shape |")
    md.append("|---|---|")
    md.append(f"| baseline | `{_get(b,'attn_mask.step0.full_att_2d_masks_4d_shape')}` |")
    md.append(f"| vla_opt | `{_get(v,'attn_mask.step0.full_att_2d_masks_4d_shape')}` |")
    md.append("")

    md.append("### Model Config (prefix vs suffix; from OPENPI_DEBUG)")
    md.append("")
    md.append("| Component | hidden_size | num_layers | num_heads | num_kv_heads | head_dim | params |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    md.append(
        "| prefix | "
        + _fmt_int(_get(b, "model_cfg.prefix.hidden_size"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.prefix.num_hidden_layers"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.prefix.num_attention_heads"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.prefix.num_key_value_heads"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.prefix.head_dim"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.prefix.params"))
        + " |"
    )
    md.append(
        "| suffix | "
        + _fmt_int(_get(b, "model_cfg.suffix.hidden_size"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.suffix.num_hidden_layers"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.suffix.num_attention_heads"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.suffix.num_key_value_heads"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.suffix.head_dim"))
        + " | "
        + _fmt_int(_get(b, "model_cfg.suffix.params"))
        + " |"
    )
    md.append("")

    md.append("### Timing Summary (from timing.parquet)")
    md.append("")
    if b_t is None or v_t is None:
        md.append("- Missing timing.parquet for baseline or vla_opt.")
    else:
        md.append(f"- Note: this debug run recorded `N={b_t['n']}` baseline samples and `N={v_t['n']}` vla_opt samples.")
        md.append("")
        md.append("| Metric | baseline (mean) | vla_opt (mean) | ratio (baseline/vla_opt) |")
        md.append("|---|---:|---:|---:|")
        for k in ["client_infer_ms", "server_infer_ms", "policy_infer_ms", "server_prev_total_ms"]:
            b_mean = b_t["stats"].get(k, {}).get("mean", None)
            v_mean = v_t["stats"].get(k, {}).get("mean", None)
            md.append(
                f"| `{k}` | {_fmt_float(b_mean,3)} | {_fmt_float(v_mean,3)} | {_fmt_float(_ratio(b_mean,v_mean),3) if _ratio(b_mean,v_mean) is not None else '—'} |"
            )
        md.append("")

    md.append("## Appendix: Raw OPENPI_DEBUG JSON")
    md.append("")
    md.append("### baseline")
    md.append("")
    md.append("```json")
    md.append(json.dumps(b, ensure_ascii=False, indent=2, sort_keys=True))
    md.append("```")
    md.append("")
    md.append("### vla_opt")
    md.append("")
    md.append("```json")
    md.append(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True))
    md.append("```")
    md.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote report: {out_md}")
    print(f"Wrote plots:  {plot_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
