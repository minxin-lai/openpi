import sys
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn

_MONOREPO_SRC = Path(__file__).resolve().parents[5] / "src"
if str(_MONOREPO_SRC) not in sys.path:
    sys.path.insert(0, str(_MONOREPO_SRC))

from openpi.models_pytorch.lora_pytorch import apply_lora_to_pi0_pytorch
from openpi.models_pytorch.lora_pytorch import LoRATrainingConfig
from openpi.models_pytorch.lora_pytorch import LoRALinear
from openpi.models_pytorch.lora_pytorch import freeze_for_lora_training
from openpi.models_pytorch.lora_pytorch import validate_lora_weight_load_result
from openpi.models.pi0_config import Pi0Config
import openpi.models.model as model_lib
from openpi.training.config import get_config


class _DummyLoRAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = nn.Linear(2, 2)
        self.multi_modal_projector = nn.Linear(2, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.language_backbone = nn.Linear(2, 2)
        self.register_parameter("lora_a", nn.Parameter(torch.zeros(1, 1)))


class _DummyAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
        self.gate_proj = nn.Linear(8, 8)


class _DummyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _DummyAttentionBlock()


class _DummyPaliGemmaModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _DummyLanguageModel()
        self.vision_tower = _DummyLanguageModel()
        self.multi_modal_projector = nn.Linear(8, 8)


class _DummyGemmaExpertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _DummyLanguageModel()


class _DummyFullLoRAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.model = _DummyPaliGemmaModel()
        self.paligemma_with_expert.gemma_expert = _DummyGemmaExpertModel()
        self.action_in_proj = nn.Linear(8, 8)


class _DummyLoadTrainConfig:
    def __init__(self, *, lora_enabled: bool) -> None:
        self.model = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
        self.lora_config = LoRATrainingConfig(enabled=lora_enabled)


def test_validate_lora_weight_load_result_accepts_lora_continuation_checkpoint() -> None:
    assert validate_lora_weight_load_result([], []) is True


def test_validate_lora_weight_load_result_accepts_base_checkpoint() -> None:
    assert validate_lora_weight_load_result(
        [
            "paligemma_with_expert.paligemma.language_model.layers.0.self_attn.q_proj.lora_a",
            "paligemma_with_expert.paligemma.language_model.layers.0.self_attn.q_proj.lora_b",
        ],
        [],
    ) is False


def test_validate_lora_weight_load_result_rejects_unexpected_keys() -> None:
    with pytest.raises(ValueError, match="Unexpected checkpoint keys"):
        validate_lora_weight_load_result(
            ["paligemma_with_expert.paligemma.language_model.layers.0.self_attn.q_proj.lora_a"],
            ["unexpected.weight"],
        )


def test_validate_lora_weight_load_result_rejects_missing_non_lora_keys() -> None:
    with pytest.raises(ValueError, match="Missing non-LoRA checkpoint keys"):
        validate_lora_weight_load_result(
            ["paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight"],
            [],
        )


def test_freeze_for_lora_training_keeps_projector_trainable_with_vision_encoder() -> None:
    model = _DummyLoRAModel()

    freeze_for_lora_training(
        model,
        LoRATrainingConfig(enabled=True, train_vision_encoder=True, train_non_lora_layers=False),
    )

    assert model.vision_tower.weight.requires_grad is True
    assert model.multi_modal_projector.weight.requires_grad is True
    assert model.language_backbone.weight.requires_grad is False


def test_freeze_for_lora_training_freezes_projector_without_vision_encoder() -> None:
    model = _DummyLoRAModel()

    freeze_for_lora_training(
        model,
        LoRATrainingConfig(enabled=True, train_vision_encoder=False, train_non_lora_layers=False),
    )

    assert model.vision_tower.weight.requires_grad is False
    assert model.multi_modal_projector.weight.requires_grad is False
    assert model.lora_a.requires_grad is True


def test_apply_lora_to_pi0_pytorch_uses_separate_paligemma_and_expert_configs() -> None:
    model = _DummyFullLoRAModel()

    apply_lora_to_pi0_pytorch(
        model,
        LoRATrainingConfig(
            enabled=True,
            paligemma_attn_rank=16,
            paligemma_ffn_rank=12,
            paligemma_attn_alpha=16.0,
            paligemma_ffn_alpha=12.0,
            expert_attn_rank=32,
            expert_ffn_rank=24,
            expert_attn_alpha=32.0,
            expert_ffn_alpha=24.0,
            train_vision_encoder=True,
            train_non_lora_layers=False,
        ),
    )

    paligemma_q = model.paligemma_with_expert.paligemma.model.language_model.block.q_proj
    paligemma_gate = model.paligemma_with_expert.paligemma.model.language_model.block.gate_proj
    expert_q = model.paligemma_with_expert.gemma_expert.model.block.q_proj
    expert_gate = model.paligemma_with_expert.gemma_expert.model.block.gate_proj

    assert isinstance(paligemma_q, LoRALinear)
    assert isinstance(paligemma_gate, LoRALinear)
    assert isinstance(expert_q, LoRALinear)
    assert isinstance(expert_gate, LoRALinear)
    assert paligemma_q.lora_a.shape[0] == 16
    assert paligemma_gate.lora_a.shape[0] == 12
    assert expert_q.lora_a.shape[0] == 32
    assert expert_gate.lora_a.shape[0] == 24
    assert isinstance(model.paligemma_with_expert.paligemma.model.vision_tower.block.q_proj, nn.Linear)
    assert isinstance(model.paligemma_with_expert.paligemma.model.multi_modal_projector, nn.Linear)


def test_lora_linear_initializes_both_adapter_matrices_like_jax() -> None:
    layer = LoRALinear(8, 6, lora_config=model_lib.lora_pytorch.LoRAConfig(rank=4))

    assert layer.lora_a.shape == (4, 8)
    assert layer.lora_b.shape == (6, 4)
    assert torch.count_nonzero(layer.lora_a).item() > 0
    assert torch.count_nonzero(layer.lora_b).item() > 0


def test_load_pytorch_uses_non_strict_load_for_lora_base_checkpoints() -> None:
    train_config = _DummyLoadTrainConfig(lora_enabled=True)
    model = nn.Module()

    with (
        mock.patch.object(model_lib.pi0_pytorch, "PI0Pytorch", return_value=model) as pi0_ctor,
        mock.patch.object(model_lib.lora_pytorch, "apply_lora_to_pi0_pytorch") as apply_lora,
        mock.patch.object(
            model_lib.safetensors.torch,
            "load_model",
            return_value=(["adapter.lora_a", "adapter.lora_b"], []),
        ) as load_model,
        mock.patch.object(model_lib.lora_pytorch, "validate_lora_weight_load_result", return_value=False) as validate,
    ):
        loaded = train_config.model.load_pytorch(train_config, "/tmp/model.safetensors")

    assert loaded is model
    pi0_ctor.assert_called_once_with(config=train_config.model)
    apply_lora.assert_called_once_with(model, train_config.lora_config)
    load_model.assert_called_once_with(model, "/tmp/model.safetensors", strict=False)
    validate.assert_called_once_with(["adapter.lora_a", "adapter.lora_b"], [])


def test_load_pytorch_keeps_strict_load_without_lora() -> None:
    train_config = _DummyLoadTrainConfig(lora_enabled=False)
    model = nn.Module()

    with (
        mock.patch.object(model_lib.pi0_pytorch, "PI0Pytorch", return_value=model) as pi0_ctor,
        mock.patch.object(model_lib.lora_pytorch, "apply_lora_to_pi0_pytorch") as apply_lora,
        mock.patch.object(model_lib.safetensors.torch, "load_model", return_value=None) as load_model,
        mock.patch.object(model_lib.lora_pytorch, "validate_lora_weight_load_result") as validate,
    ):
        loaded = train_config.model.load_pytorch(train_config, "/tmp/model.safetensors")

    assert loaded is model
    pi0_ctor.assert_called_once_with(config=train_config.model)
    apply_lora.assert_not_called()
    load_model.assert_called_once_with(model, "/tmp/model.safetensors")
    validate.assert_not_called()


def test_pi05_libero_lora_pytorch_defaults_match_jax_lora_ranks() -> None:
    config = get_config("pi05_libero_lora_pytorch")
    lora_config = config.lora_config

    assert lora_config is not None
    assert lora_config.paligemma_attn_rank == 16
    assert lora_config.paligemma_ffn_rank == 16
    assert lora_config.paligemma_attn_alpha == 16.0
    assert lora_config.paligemma_ffn_alpha == 16.0
    assert lora_config.expert_attn_rank == 32
    assert lora_config.expert_ffn_rank == 32
    assert lora_config.expert_attn_alpha == 32.0
    assert lora_config.expert_ffn_alpha == 32.0
