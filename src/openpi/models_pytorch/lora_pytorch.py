"""PyTorch LoRA support for PI0/PI0.5 training."""

import logging
import math
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


logger = logging.getLogger(__name__)
_VISION_PARAM_MARKERS = ("vision_tower", "vision_model", "multi_modal_projector")


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter."""

    rank: int
    alpha: float = 1.0
    rslora: bool = False
    dropout: float = 0.0

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


@dataclass
class LoRATrainingConfig:
    """Training-time LoRA settings for the PyTorch PI0/PI0.5 path."""

    enabled: bool = False
    paligemma_attn_rank: int = 16
    paligemma_ffn_rank: int = 16
    paligemma_attn_alpha: float = 16.0
    paligemma_ffn_alpha: float = 16.0
    expert_attn_rank: int = 16
    expert_ffn_rank: int = 16
    expert_attn_alpha: float = 16.0
    expert_ffn_alpha: float = 16.0
    use_rslora: bool = False
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    apply_to: Literal["all", "paligemma_only", "expert_only", "paligemma_attn", "expert_attn"] = "all"
    train_vision_encoder: bool = True
    train_non_lora_layers: bool = True
    trainable_modules: list[str] = field(
        default_factory=lambda: [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
    )

    def get_lora_configs(self) -> dict[str, dict[str, LoRAConfig]]:
        return {
            "paligemma": {
                "attn": LoRAConfig(
                    rank=self.paligemma_attn_rank,
                    alpha=self.paligemma_attn_alpha,
                    rslora=self.use_rslora,
                    dropout=self.dropout,
                ),
                "ffn": LoRAConfig(
                    rank=self.paligemma_ffn_rank,
                    alpha=self.paligemma_ffn_alpha,
                    rslora=self.use_rslora,
                    dropout=self.dropout,
                ),
            },
            "expert": {
                "attn": LoRAConfig(
                    rank=self.expert_attn_rank,
                    alpha=self.expert_attn_alpha,
                    rslora=self.use_rslora,
                    dropout=self.dropout,
                ),
                "ffn": LoRAConfig(
                    rank=self.expert_ffn_rank,
                    alpha=self.expert_ffn_alpha,
                    rslora=self.use_rslora,
                    dropout=self.dropout,
                ),
            },
        }


class LoRALinear(nn.Module):
    """Linear layer with additive LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        lora_config: LoRAConfig,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_config = lora_config

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.lora_a = nn.Parameter(torch.empty(lora_config.rank, in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.empty(out_features, lora_config.rank, device=device, dtype=dtype))
        self.lora_dropout = nn.Dropout(p=float(lora_config.dropout)) if lora_config.dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.lora_a, std=0.01)
        nn.init.normal_(self.lora_b, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_a), self.lora_b)
        return out + lora_out * self.lora_config.scaling_value


def apply_lora_to_linear(linear: nn.Linear, lora_config: LoRAConfig) -> LoRALinear:
    """Replace an ``nn.Linear`` with a LoRA-augmented equivalent."""

    lora_linear = LoRALinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        lora_config=lora_config,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    lora_linear.weight.data.copy_(linear.weight.data)
    if linear.bias is not None and lora_linear.bias is not None:
        lora_linear.bias.data.copy_(linear.bias.data)
    return lora_linear


def _is_lora_adapter_parameter(name: str) -> bool:
    return name == "lora_a" or name == "lora_b" or name.endswith((".lora_a", ".lora_b"))


def _is_vision_side_parameter(name: str) -> bool:
    return any(marker in name for marker in _VISION_PARAM_MARKERS)


def validate_lora_weight_load_result(missing: list[str], unexpected: list[str]) -> bool:
    """Validate a LoRA weight load and report whether adapters came from the checkpoint."""

    non_lora_missing = [name for name in missing if not _is_lora_adapter_parameter(name)]
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys in LoRA load: {unexpected}")
    if non_lora_missing:
        raise ValueError(f"Missing non-LoRA checkpoint keys in LoRA load: {non_lora_missing}")
    return len(missing) == 0


def freeze_for_lora_training(model: nn.Module, lora_config: LoRATrainingConfig) -> tuple[int, int]:
    """Freeze parameters according to the LoRA training policy."""

    trainable_modules = set(lora_config.trainable_modules) if lora_config.train_non_lora_layers else set()
    frozen_count = 0
    trainable_count = 0
    vision_trainable = 0

    for name, param in model.named_parameters():
        is_lora_param = "lora_" in name
        is_vision_param = _is_vision_side_parameter(name)
        is_trainable_module = any(module_name in name for module_name in trainable_modules)

        should_train = False
        if is_lora_param:
            should_train = True
        elif is_vision_param:
            should_train = bool(lora_config.train_vision_encoder)
            if should_train:
                vision_trainable += int(param.numel())
        elif is_trainable_module:
            should_train = True

        param.requires_grad = should_train
        if should_train:
            trainable_count += int(param.numel())
        else:
            frozen_count += int(param.numel())

    logger.info("LoRA training: trainable=%s frozen=%s", trainable_count, frozen_count)
    if lora_config.train_vision_encoder:
        logger.info("Vision encoder is trainable under LoRA: params=%s", vision_trainable)
    else:
        logger.info("Vision encoder is frozen under LoRA")
    return frozen_count, trainable_count


def apply_lora_to_pi0_pytorch(model: nn.Module, lora_config: LoRATrainingConfig) -> tuple[int, int]:
    """Apply LoRA adapters to a ``PI0Pytorch`` model."""

    if not lora_config.enabled:
        total_params = sum(int(p.numel()) for p in model.parameters())
        logger.info("LoRA disabled; leaving model unchanged")
        return 0, total_params

    lora_configs = lora_config.get_lora_configs()
    target_modules = list(lora_config.target_modules)
    apply_to_paligemma = lora_config.apply_to in {"all", "paligemma_only", "paligemma_attn"}
    apply_to_expert = lora_config.apply_to in {"all", "expert_only", "expert_attn"}
    if lora_config.apply_to in {"paligemma_attn", "expert_attn"}:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    attn_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    ffn_modules = {"gate_proj", "up_proj", "down_proj"}
    applied = 0

    def get_lora_group(full_name: str, module_name: str) -> str | None:
        if module_name not in target_modules:
            return None
        if "vision_tower" in full_name or "vision_model" in full_name:
            return None

        is_paligemma = "paligemma" in full_name and "language_model" in full_name
        is_expert = "gemma_expert" in full_name
        if is_paligemma and apply_to_paligemma:
            return "paligemma"
        if is_expert and apply_to_expert:
            return "expert"
        return None

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        module_name = name.split(".")[-1]
        lora_group = get_lora_group(name, module_name)
        if lora_group is None:
            continue

        if module_name in attn_modules:
            lora_cfg = lora_configs[lora_group]["attn"]
        elif module_name in ffn_modules:
            lora_cfg = lora_configs[lora_group]["ffn"]
        else:
            continue

        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            attr_name = parts[1]
        else:
            parent = model
            attr_name = name
        setattr(parent, attr_name, apply_lora_to_linear(module, lora_cfg))
        applied += 1

    logger.info("Applied LoRA to %s linear layers", applied)
    return freeze_for_lora_training(model, lora_config)
