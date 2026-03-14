from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.observe_adapter import build_openpi_vision_pruning_tensors


def test_build_openpi_vision_pruning_tensors_uses_model_patch_grid() -> None:
    model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(
                model=SimpleNamespace(
                    vision_tower=SimpleNamespace(config=SimpleNamespace(patch_size=14))
                )
            )
        )
    )

    tensors = build_openpi_vision_pruning_tensors(
        model=model,
        input_tokens=256,
        scores=torch.full((1, 256), 0.1),
        smoothed_scores=torch.full((1, 256), 0.2),
        keep_indices=torch.tensor([[0, 1, 2, 3]]),
        keep_mask=torch.tensor([[True, False] * 128]),
        image=torch.zeros((1, 3, 224, 224)),
    )

    assert tensors is not None
    assert tuple(tensors["patch_grid_hw"]) == (16, 16)
    assert tensors["display_scores"].tolist() == pytest.approx([0.2] * 256)
    assert tensors["raw_scores"].tolist() == pytest.approx([0.1] * 256)


def test_build_openpi_vision_pruning_tensors_falls_back_when_grid_mismatches() -> None:
    model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(
                model=SimpleNamespace(
                    vision_tower=SimpleNamespace(config=SimpleNamespace(patch_size=16))
                )
            )
        )
    )

    tensors = build_openpi_vision_pruning_tensors(
        model=model,
        input_tokens=4,
        scores=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        smoothed_scores=None,
        keep_indices=torch.tensor([[0, 2]]),
        keep_mask=torch.tensor([[True, False, True, False]]),
        image=torch.zeros((1, 3, 32, 64)),
    )

    assert tensors is not None
    assert tuple(tensors["patch_grid_hw"]) == (2, 2)
