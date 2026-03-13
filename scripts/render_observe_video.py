from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw


@dataclass(frozen=True)
class FrameRecord:
    task_id: int
    task_slug: str
    episode_idx: int
    query_idx: int
    view_idx: int
    overlay_kind: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render observe overlay PNGs into side-by-side videos.")
    parser.add_argument("--run-dir", required=True, help="Observe run directory containing png_dumps/render_index.jsonl")
    parser.add_argument("--fps", type=int, default=1, help="Output video FPS")
    parser.add_argument(
        "--overlay-kind",
        choices=("scores_overlay", "keep_mask_overlay", "both"),
        default="both",
        help="Which overlay to render into video",
    )
    parser.add_argument("--view-a", type=int, default=0, help="Left-side view index")
    parser.add_argument("--view-b", type=int, default=1, help="Right-side view index")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory, default: <run-dir>/video_dumps",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "video_dumps"
    overlay_kinds = ["scores_overlay", "keep_mask_overlay"] if args.overlay_kind == "both" else [args.overlay_kind]

    records = load_records(run_dir, overlay_kinds)
    outputs = render_videos(
        records,
        output_dir=output_dir,
        fps=int(args.fps),
        view_a=int(args.view_a),
        view_b=int(args.view_b),
    )
    for output_path in outputs:
        print(output_path)
    return 0


def load_records(run_dir: Path, overlay_kinds: list[str]) -> list[FrameRecord]:
    index_paths = sorted(run_dir.rglob("png_dumps/render_index.jsonl"))
    if not index_paths:
        raise FileNotFoundError(f"No png_dumps/render_index.jsonl found under: {run_dir}")

    records: list[FrameRecord] = []
    for index_path in index_paths:
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                context = data.get("context", {})
                if not isinstance(context, dict):
                    continue
                task_id = context.get("task_id")
                task_slug = context.get("task_slug")
                episode_idx = context.get("episode_idx")
                query_idx = context.get("query_idx")
                if None in {task_id, task_slug, episode_idx, query_idx}:
                    continue
                for overlay_kind in overlay_kinds:
                    path_key = f"{overlay_kind}_path"
                    overlay_path = data.get(path_key)
                    if not overlay_path:
                        continue
                    view_idx = parse_view_idx(str(overlay_path))
                    records.append(
                        FrameRecord(
                            task_id=int(task_id),
                            task_slug=str(task_slug),
                            episode_idx=int(episode_idx),
                            query_idx=int(query_idx),
                            view_idx=view_idx,
                            overlay_kind=overlay_kind,
                            image_path=Path(str(overlay_path)),
                        )
                    )
    return records


def parse_view_idx(path_text: str) -> int:
    match = re.search(r"_view(\d+)_", Path(path_text).name)
    if match is None:
        raise ValueError(f"Cannot parse view index from: {path_text}")
    return int(match.group(1))


def render_videos(
    records: list[FrameRecord],
    *,
    output_dir: Path,
    fps: int,
    view_a: int,
    view_b: int,
) -> list[Path]:
    groups: dict[tuple[str, int, str], dict[int, dict[int, Path]]] = defaultdict(lambda: defaultdict(dict))
    task_ids: dict[tuple[str, int, str], int] = {}

    for record in records:
        group_key = (record.task_slug, record.episode_idx, record.overlay_kind)
        task_ids[group_key] = record.task_id
        groups[group_key][record.query_idx][record.view_idx] = record.image_path

    outputs: list[Path] = []
    for (task_slug, episode_idx, overlay_kind), query_map in sorted(groups.items()):
        frames: list[np.ndarray] = []
        task_id = task_ids[(task_slug, episode_idx, overlay_kind)]
        for query_idx in sorted(query_map):
            view_map = query_map[query_idx]
            left_path = view_map.get(view_a)
            right_path = view_map.get(view_b)
            if left_path is None or right_path is None:
                continue
            frames.append(build_frame(left_path, right_path, task_slug, episode_idx, query_idx, overlay_kind, view_a, view_b))

        if not frames:
            continue

        task_dir = output_dir / overlay_kind / f"task_{task_id:02d}_{task_slug}"
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / f"episode_{episode_idx:03d}_view{view_a}_view{view_b}.mp4"
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        outputs.append(output_path)

    if not outputs:
        raise FileNotFoundError("No episode videos were rendered; check that both requested views exist in render_index.jsonl")
    return outputs


def build_frame(
    left_path: Path,
    right_path: Path,
    task_slug: str,
    episode_idx: int,
    query_idx: int,
    overlay_kind: str,
    view_a: int,
    view_b: int,
) -> np.ndarray:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    target_h = max(left.height, right.height)
    left = resize_to_height(left, target_h)
    right = resize_to_height(right, target_h)

    title = f"{overlay_kind} | task={task_slug} | episode={episode_idx:03d} | query={query_idx:03d} | view{view_a} | view{view_b}"
    title_h = 28
    gap = 8
    canvas = Image.new("RGB", (left.width + right.width + gap, target_h + title_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(255, 255, 255))
    canvas.paste(left, (0, title_h))
    canvas.paste(right, (left.width + gap, title_h))
    return np.asarray(pad_to_macro_block(canvas, block_size=16))


def resize_to_height(image: Image.Image, target_h: int) -> Image.Image:
    if image.height == target_h:
        return image
    target_w = round(image.width * (target_h / image.height))
    return image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)


def pad_to_macro_block(image: Image.Image, *, block_size: int) -> Image.Image:
    target_w = ((image.width + block_size - 1) // block_size) * block_size
    target_h = ((image.height + block_size - 1) // block_size) * block_size
    if target_w == image.width and target_h == image.height:
        return image
    canvas = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    canvas.paste(image, (0, 0))
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
