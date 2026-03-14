from __future__ import annotations

from typing import Any

from vla_opt.observe.openpi import build_openpi_pruning_tensors


def build_openpi_vision_pruning_tensors(
    *,
    model: Any,
    input_tokens: int | None,
    scores: Any,
    smoothed_scores: Any | None,
    keep_indices: Any,
    keep_mask: Any,
    image: Any | None,
) -> dict[str, Any] | None:
    patch_grid_hw = _resolve_patch_grid_hw(model=model, image=image, input_tokens=input_tokens)
    return build_openpi_pruning_tensors(
        input_tokens=input_tokens,
        patch_grid_hw=patch_grid_hw,
        scores=scores,
        smoothed_scores=smoothed_scores,
        keep_indices=keep_indices,
        keep_mask=keep_mask,
        image=image,
    )


def _resolve_patch_grid_hw(*, model: Any, image: Any, input_tokens: int | None) -> tuple[int, int] | None:
    if image is None or input_tokens is None or input_tokens <= 0:
        return None

    image_shape = getattr(image, "shape", None)
    if image_shape is None or len(image_shape) != 4:
        return None
    image_hw = (int(image_shape[-2]), int(image_shape[-1]))

    patch_size = _resolve_patch_size(model)
    if patch_size is None:
        return None

    grid_hw = (image_hw[0] // patch_size[0], image_hw[1] // patch_size[1])
    grid_tokens = int(grid_hw[0] * grid_hw[1])
    if grid_tokens == int(input_tokens):
        return grid_hw
    if grid_tokens + 1 == int(input_tokens):
        return grid_hw
    return None


def _resolve_patch_size(model: Any) -> tuple[int, int] | None:
    pge = getattr(model, "paligemma_with_expert", None)
    paligemma = getattr(pge, "paligemma", None)
    pgm = getattr(paligemma, "model", None)
    vision_tower = getattr(pgm, "vision_tower", None)
    config = getattr(vision_tower, "config", None)
    patch_size = getattr(config, "patch_size", None)
    return _normalize_hw(patch_size)


def _normalize_hw(value: Any) -> tuple[int, int] | None:
    if isinstance(value, int):
        if value <= 0:
            return None
        return (int(value), int(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        h, w = (int(value[0]), int(value[1]))
        if h <= 0 or w <= 0:
            return None
        return (h, w)
    return None
