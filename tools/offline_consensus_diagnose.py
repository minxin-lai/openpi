from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OPENPI_SRC = Path(__file__).resolve().parents[1] / "src"
_VLA_SRC = _REPO_ROOT / "src"
if _OPENPI_SRC.exists() and str(_OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_SRC))
if _VLA_SRC.exists() and str(_VLA_SRC) not in sys.path:
    sys.path.insert(0, str(_VLA_SRC))

import numpy as np
import torch

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from vla_opt.adapter import StageInput
from vla_opt.adapters import get_adapter_class
import vla_opt.adapters.openpi_pytorch  # noqa: F401
from vla_opt.observe.consensus_stats import aggregate_run_dir
from vla_opt.observe.render_consensus_png import render_run_dir
from vla_opt.pipeline import VLAOptPipeline
from vla_opt.pruning import load_pruning_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay policy_records into the Stage II consensus diagnostics pipeline for wrapped OpenPI PI0/PI0.5 PyTorch policies."
    )
    parser.add_argument("--ckpt-dir", required=True, help="Checkpoint directory containing model.safetensors")
    parser.add_argument(
        "--policy-config",
        required=True,
        help="OpenPI PI0/PI0.5 policy config name for wrapped PyTorch replay.",
    )
    parser.add_argument("--opt-config", required=True, help="Pruning YAML used to derive Stage I/II runtime config")
    parser.add_argument("--records-dir", required=True, help="Directory containing policy record .npy files")
    parser.add_argument("--output-dir", required=True, help="Output directory for observe dumps, stats, and PNGs")
    parser.add_argument("--record-limit", type=int, default=None, help="Optional max number of records to replay")
    parser.add_argument("--match-top-m", type=int, default=3, help="Top-m matches to keep per source patch")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda:0 or cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records_dir = Path(args.records_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    observe_dir = output_dir / "observe"

    if not records_dir.is_dir():
        raise FileNotFoundError(f"records_dir not found: {records_dir}")

    _clear_vla_opt_env()
    os.environ["VLA_OPT_PRUNING_CONFIG"] = str(Path(args.opt_config).expanduser().resolve())
    train_cfg = _config.get_config(args.policy_config)
    _validate_supported_policy_config(train_cfg)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.ckpt_dir,
        pytorch_device=args.device,
    )
    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError(
            f"offline_consensus_diagnose only supports wrapped OpenPI PI0/PI0.5 PyTorch checkpoints: {args.policy_config}"
        )

    model = policy._model
    runtime_cfg = load_pruning_config(args.opt_config).to_runtime("serve")
    adapter = _build_openpi_adapter(train_cfg=train_cfg, model=model)
    pipeline = VLAOptPipeline(config=_build_pipeline_config(runtime_cfg, observe_dir=observe_dir, match_top_m=int(args.match_top_m)))

    record_paths = sorted(records_dir.rglob("*.npy"), key=_record_sort_key)
    if not record_paths:
        raise FileNotFoundError(f"No .npy policy records found under: {records_dir}")
    if args.record_limit is not None:
        if int(args.record_limit) <= 0:
            raise ValueError("--record-limit must be > 0 when provided")
        record_paths = record_paths[: int(args.record_limit)]

    for record_path in record_paths:
        raw_inputs, trace_context = _load_record_inputs(record_path)
        transformed_inputs = _apply_policy_input_transform(policy, raw_inputs)
        observation = _to_observation(transformed_inputs, device=str(policy._pytorch_device))
        batch = {"observation": observation}
        with torch.no_grad():
            instr_emb = adapter.encode_text(batch)
            tokens_per_view, align = _encode_wrapped_openpi_vision(adapter, batch=batch, cond_tokens=instr_emb)
            pipeline(StageInput(tokens_per_view=tokens_per_view, instr_emb=instr_emb, align=align, meta=trace_context))

    aggregate_run_dir(observe_dir)
    render_run_dir(observe_dir)
    browse_dir, browse_queries = _build_browse_dir(output_dir=output_dir, observe_dir=observe_dir)
    print(f"records_replayed: {len(record_paths)}")
    print(f"observe_dir: {observe_dir}")
    print(f"browse_dir: {browse_dir}")
    print(f"browse_queries: {browse_queries}")
    return 0


def _clear_vla_opt_env() -> None:
    for name in tuple(os.environ):
        if name.startswith("VLA_OPT_") or name.startswith("OPENPI_DEBUG_"):
            os.environ.pop(name, None)


def _build_pipeline_config(runtime_cfg: Any, *, observe_dir: Path, match_top_m: int) -> dict[str, Any]:
    importance_impl = "cross_attn" if str(runtime_cfg.score_head_type) == "cross_attn" else "pairwise"
    return {
        "observe": {
            "enabled": True,
            "live": False,
            "trace": True,
            "debug_dump": True,
            "output_dir": str(observe_dir),
            "modules": {
                "pruning": {
                    "enabled": True,
                    "phases": ["post_encoder"],
                },
                "consensus": {
                    "enabled": True,
                    "phases": ["stage2_probe"],
                },
            },
        },
        "stage_a": {
            "enabled": True,
            "modulation": "none",
            "importance_impl": importance_impl,
        },
        "vision_pruning": {
            "enabled": True,
            "fg_select_mode": "topk",
            "fg_k": int(runtime_cfg.k),
            "gaussian_smooth_enabled": bool(runtime_cfg.gaussian_enabled),
            "gaussian_smooth_sigma": float(runtime_cfg.gaussian_sigma),
            "gaussian_smooth_kernel_size": runtime_cfg.gaussian_kernel_size,
            "bg_enabled": False,
        },
        "consensus_probe": {
            "enabled": True,
            "match_top_m": int(match_top_m),
        },
    }


def _validate_supported_policy_config(train_cfg: Any) -> None:
    if isinstance(getattr(train_cfg, "model", None), _pi0_config.Pi0Config):
        return
    model_name = type(getattr(train_cfg, "model", None)).__name__
    raise ValueError(
        "offline_consensus_diagnose only supports wrapped OpenPI PI0/PI0.5 PyTorch policies; "
        f"got model config {model_name} for {getattr(train_cfg, 'name', '<unknown>')}. "
        "PI0_FAST and JAX-only configs are not supported."
    )


def _build_openpi_adapter(*, train_cfg: Any, model: Any) -> Any:
    adapter_name = _resolve_openpi_adapter_name(train_cfg)
    adapter_cls = get_adapter_class(adapter_name)
    return adapter_cls(model, enable_film=False)


def _resolve_openpi_adapter_name(train_cfg: Any) -> str:
    model_cfg = getattr(train_cfg, "model", None)
    if isinstance(model_cfg, _pi0_config.Pi0Config):
        return "openpi_pytorch"
    raise ValueError(
        "offline_consensus_diagnose only supports wrapped OpenPI PI0/PI0.5 PyTorch policies; "
        f"unsupported model config: {type(model_cfg).__name__}"
    )


def _record_sort_key(path: Path) -> tuple[int, str, int, int, int, str]:
    trace_context = _load_record_trace_context(path)
    task_id = _int_or_default(trace_context.get("task_id"), sys.maxsize)
    task_slug = str(trace_context.get("task_slug", ""))
    episode_idx = _int_or_default(trace_context.get("episode_idx"), sys.maxsize)
    query_idx = _int_or_default(trace_context.get("query_idx"), sys.maxsize)
    step_idx = _step_index_from_path(path)
    return (task_id, task_slug, episode_idx, query_idx, step_idx, str(path))


def _step_index_from_path(path: Path) -> int:
    stem = path.stem
    if stem.startswith("step_"):
        suffix = stem[len("step_") :]
        if suffix.isdigit():
            return int(suffix)
    return sys.maxsize


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_record_trace_context(record_path: Path) -> dict[str, Any]:
    payload = np.load(record_path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError(f"Record payload must be a dict: {record_path}")

    trace_prefix = "inputs/__vla_opt_trace__/"
    trace_context = {
        str(key)[len(trace_prefix) :]: value
        for key, value in payload.items()
        if str(key).startswith(trace_prefix)
    }
    if not trace_context:
        raise ValueError(f"Record is missing __vla_opt_trace__: {record_path}")
    return trace_context


def _build_browse_dir(*, output_dir: Path, observe_dir: Path) -> tuple[Path, int]:
    browse_dir = output_dir
    browse_dir.mkdir(parents=True, exist_ok=True)
    for task_dir in browse_dir.glob("task_*"):
        if task_dir.is_dir():
            shutil.rmtree(task_dir)

    records = _collect_render_records(observe_dir)
    query_count = 0
    for record in records:
        context = record.get("context", {})
        task_id = int(context.get("task_id", 0))
        task_slug = _slugify(str(context.get("task_slug", "task")))
        episode_idx = int(context.get("episode_idx", 0))
        query_idx = int(context.get("query_idx", 0))

        task_dir = browse_dir / f"task_{task_id:02d}_{task_slug}"
        task_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"e{episode_idx:03d}_q{query_idx:03d}"

        primary_outputs = record.get("primary_outputs", {})
        for view_idx, payload in sorted(primary_outputs.get("views", {}).items(), key=lambda item: int(item[0])):
            src = Path(str(payload["consensus_mask_overlay_path"]))
            dst = task_dir / f"{prefix}_view{int(view_idx)}_consensus_mask_overlay.png"
            _link_or_copy_file(src, dst)

        for pair_key, payload in sorted(primary_outputs.get("pairs", {}).items()):
            src = Path(str(payload["anchor_correspondence_path"]))
            dst = task_dir / f"{prefix}_pair_{pair_key.replace('-', '_')}_anchor_correspondence.png"
            _link_or_copy_file(src, dst)

        meta = {
            "task_id": task_id,
            "task_slug": str(context.get("task_slug", "task")),
            "episode_idx": episode_idx,
            "query_idx": query_idx,
            "tensor_dump_path": record.get("tensor_dump_path"),
            "source_trace_dir": str(_source_trace_dir_from_record(record)),
        }
        (task_dir / f"{prefix}_query_meta.json").write_text(
            json.dumps(meta, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        query_count += 1

    return browse_dir, query_count


def _collect_render_records(observe_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index_path in observe_dir.rglob("consensus_render_index.jsonl"):
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid render index record: {index_path}")
                record["_index_path"] = str(index_path)
                records.append(record)
    records.sort(key=_render_record_sort_key)
    return records


def _render_record_sort_key(record: dict[str, Any]) -> tuple[int, str, int, int]:
    context = record.get("context", {})
    task_id = int(context.get("task_id", 0))
    task_slug = str(context.get("task_slug", ""))
    episode_idx = int(context.get("episode_idx", 0))
    query_idx = int(context.get("query_idx", 0))
    return (task_id, task_slug, episode_idx, query_idx)


def _source_trace_dir_from_record(record: dict[str, Any]) -> Path:
    index_path = Path(str(record.get("_index_path", "")))
    if index_path.name != "consensus_render_index.jsonl":
        return index_path
    return index_path.parent.parent


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing source artifact: {src}")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _slugify(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "task"


def _load_record_inputs(record_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = np.load(record_path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError(f"Record payload must be a dict: {record_path}")

    input_items = {str(key)[len("inputs/") :]: value for key, value in payload.items() if str(key).startswith("inputs/")}
    if not input_items:
        raise ValueError(f"Record does not contain inputs/* keys: {record_path}")

    trace_prefix = "__vla_opt_trace__/"
    trace_context = {
        key[len(trace_prefix) :]: value for key, value in input_items.items() if key.startswith(trace_prefix)
    }
    if not trace_context:
        raise ValueError(f"Record is missing __vla_opt_trace__: {record_path}")
    flat_inputs = {key: value for key, value in input_items.items() if not key.startswith(trace_prefix)}
    return flat_inputs, trace_context


def _apply_policy_input_transform(policy: Any, raw_inputs: dict[str, Any]) -> dict[str, Any]:
    inputs = copy.deepcopy(raw_inputs)
    transformed = policy._input_transform(inputs)
    if not isinstance(transformed, dict):
        raise ValueError("policy input transform must return a dict")
    return transformed


def _to_observation(data: dict[str, Any], *, device: str) -> _model.Observation:
    batched = {}
    for key, value in data.items():
        if isinstance(value, dict):
            batched[key] = {sub_key: _to_torch_leaf(sub_value, device=device) for sub_key, sub_value in value.items()}
        else:
            batched[key] = _to_torch_leaf(value, device=device)
    return _model.Observation.from_dict(batched)


def _to_torch_leaf(value: Any, *, device: str) -> torch.Tensor:
    array = np.asarray(value)
    tensor = torch.from_numpy(array).to(device)
    return tensor[None, ...]


def _encode_wrapped_openpi_vision(adapter: Any, *, batch: dict[str, Any], cond_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    observation = batch["observation"]
    images, img_masks, _lang_tokens, lang_masks, _state = adapter._preprocess(observation, train=False)

    stage_a_handle = getattr(adapter.model, "_vla_opt_stage_a_handle", None)
    ste_handle = getattr(adapter.model, "_vla_opt_ste_prune_handle", None)
    paligemma_with_expert = getattr(adapter.model, "paligemma_with_expert", None)
    orig_embed_image = getattr(paligemma_with_expert, "_vla_opt_orig_embed_image", None) if paligemma_with_expert is not None else None
    if stage_a_handle is None or ste_handle is None or not callable(orig_embed_image):
        raise RuntimeError(
            "wrapped OpenPI model is missing _vla_opt_stage_a_handle, _vla_opt_ste_prune_handle, or _vla_opt_orig_embed_image"
        )

    toks: list[torch.Tensor] = []
    cls_toks: list[torch.Tensor] = []
    has_any_cls = False
    stage_a_handle.set_condition(cond_tokens, cond_mask=lang_masks)
    ste_handle.set_condition(cond_tokens, cond_mask=lang_masks)
    try:
        for img in images:
            t = orig_embed_image(img)
            grid_hw, has_cls = adapter._infer_grid_and_cls(int(t.shape[1]))
            has_any_cls = has_any_cls or has_cls
            if has_cls:
                cls_toks.append(t[:, :1, :])
                t = t[:, 1:, :]
            toks.append(t)
    finally:
        ste_handle.clear_condition()
        stage_a_handle.clear_condition()

    tokens_per_view = torch.stack(toks, dim=1)
    n = int(tokens_per_view.shape[2])
    grid_hw = adapter._infer_patch_grid_hw(n)

    align: dict[str, Any] = {
        "patch_grid_hw": grid_hw,
        "image_keys": list(getattr(observation, "images").keys()),
        "images_per_view": torch.stack(images, dim=1),
        "image_masks_per_view": torch.stack(img_masks, dim=1),
        "vision_has_cls": bool(has_any_cls),
    }
    if has_any_cls:
        align["vision_cls_tokens_per_view"] = torch.stack(cls_toks, dim=1)
    return tokens_per_view, align


if __name__ == "__main__":
    raise SystemExit(main())
