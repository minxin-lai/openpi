#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _iter_pt_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(str(path))
    return sorted([p for p in path.rglob("*.pt") if p.is_file()])


def _get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _shape(x: Any) -> str:
    try:
        return str(tuple(x.shape))
    except Exception:
        return "None"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sanity-check dumps for Stage2 consensus inputs.")
    ap.add_argument("--input", required=True, help="Path to dumps/ dir or a single .pt file")
    ap.add_argument("--pair", default="0,1", help="View pair by index, e.g. '0,1'")
    ap.add_argument("--max", type=int, default=3, help="Max number of dumps to print (default 3)")
    args = ap.parse_args(argv)

    if not _torch_available():
        raise SystemExit("torch not available; run under the openpi .venv")
    import torch

    input_path = Path(str(args.input)).expanduser()
    files = _iter_pt_files(input_path)
    if not files:
        raise FileNotFoundError(f"No .pt files under {str(input_path)}")

    pair = [p.strip() for p in str(args.pair).split(",") if p.strip()]
    if len(pair) != 2 or not all(x.lstrip("-").isdigit() for x in pair):
        raise SystemExit(f"--pair must be like '0,1' (indices), got: {args.pair!r}")
    a, b = int(pair[0]), int(pair[1])

    n = min(int(args.max), len(files))
    print(f"input: {str(input_path)}")
    print(f"found_dumps: {len(files)} (showing {n})")
    print(f"pair: {a},{b}")

    for i, p in enumerate(files[:n]):
        d = torch.load(str(p), map_location="cpu")
        if not isinstance(d, dict):
            print(f"\n[{i}] dump: {str(p)}  ERROR: not a dict")
            continue

        meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        view_names = meta.get("view_names") if isinstance(meta.get("view_names"), list) else None
        routing = d.get("routing") if isinstance(d.get("routing"), dict) else {}

        keep_tokens = routing.get("keep_tokens")
        keep_scores = routing.get("keep_scores")
        keep_indices = routing.get("keep_indices")

        print(f"\n[{i}] dump: {str(p)}")
        print(f"  meta.view_names: {view_names}")
        print(f"  routing.keep_tokens: {type(keep_tokens).__name__} shape={_shape(keep_tokens)}")
        print(f"  routing.keep_scores: {type(keep_scores).__name__} shape={_shape(keep_scores)}")
        print(f"  routing.keep_indices: {type(keep_indices).__name__} shape={_shape(keep_indices)}")

        if not (torch.is_tensor(keep_tokens) and torch.is_tensor(keep_scores) and torch.is_tensor(keep_indices)):
            print("  ERROR: missing keep_* tensors; consensus cannot run.")
            continue

        if keep_tokens.ndim != 4 or keep_scores.ndim != 3 or keep_indices.ndim != 3:
            print("  ERROR: wrong dims; expected keep_tokens[B,V,K,D], keep_scores[B,V,K], keep_indices[B,V,K].")
            continue

        bsz, v, k, d_ = (int(keep_tokens.shape[0]), int(keep_tokens.shape[1]), int(keep_tokens.shape[2]), int(keep_tokens.shape[3]))
        print(f"  shapes: B={bsz} V={v} K={k} D={d_}")
        if a < 0 or b < 0 or a >= v or b >= v:
            print(f"  ERROR: pair ({a},{b}) out of range for V={v}")
            continue

        # Quick sanity: do the two views select the exact same patch indices?
        idx_a = keep_indices[0, a].to(dtype=torch.long)
        idx_b = keep_indices[0, b].to(dtype=torch.long)
        same_pos = int((idx_a == idx_b).sum().item())
        # overlap as sets
        set_overlap = int(len(set(idx_a.tolist()).intersection(set(idx_b.tolist()))))
        print(f"  keep_indices[0,{a}] vs keep_indices[0,{b}]: same_position={same_pos}/{k} set_overlap={set_overlap}/{k}")

        # Optional debug fields (best-effort)
        patch_hw = _get(d, "routing", "patch_grid_hw") or _get(d, "align", "patch_grid_hw")
        image_paths = _get(d, "align", "image_paths")
        if patch_hw is not None:
            print(f"  patch_grid_hw: {patch_hw}")
        if image_paths is not None:
            try:
                print(f"  align.image_paths: {list(image_paths)[:min(3, len(image_paths))]}")
            except Exception:
                print("  align.image_paths: <unprintable>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

