"""
Unit tests for LightVLA-style token pruning (PyTorch).

Run:
  `python -m pytest src/openpi/models_pytorch/token_pruner_test.py -v`
"""

import pytest
import torch

from openpi.models_pytorch.token_pruner import ImageTokenPruner, TokenPruner


class TestImageTokenPruner:
    @pytest.fixture
    def pruner(self) -> ImageTokenPruner:
        return ImageTokenPruner(hidden_size=128)

    def test_train_keeps_shape_and_is_differentiable(self, pruner: ImageTokenPruner) -> None:
        pruner.train()
        pruner.set_noise_scale(0.1)

        b, n, t, d = 2, 64, 16, 128
        patches = torch.randn(b, n, d, requires_grad=True)
        task = torch.randn(b, t, d)

        out, out_mask = pruner.prune(patches, task)
        assert out.shape == (b, n, d)
        assert out_mask.shape == (b, n)
        assert out_mask.dtype == torch.bool
        assert out_mask.all()

        out.sum().backward()
        assert patches.grad is not None

    def test_eval_prunes_and_returns_mask(self, pruner: ImageTokenPruner) -> None:
        pruner.eval()
        pruner.set_noise_scale(None)

        b, n, t, d = 2, 64, 16, 128
        patches = torch.randn(b, n, d)
        task = torch.randn(b, t, d)

        out, out_mask = pruner.prune(patches, task)
        assert out.shape[0] == b and out.shape[2] == d
        assert out.shape[1] <= n
        assert out_mask.shape == (b, out.shape[1])
        assert out_mask.dtype == torch.bool
        assert out_mask.sum(dim=-1).min().item() >= 1

    def test_eval_fixed_keep_tokens(self, pruner: ImageTokenPruner) -> None:
        pruner.eval()
        pruner.set_noise_scale(None)
        pruner.set_keep_tokens(8)

        b, n, t, d = 2, 64, 16, 128
        patches = torch.randn(b, n, d)
        task = torch.randn(b, t, d)

        out, out_mask = pruner.prune(patches, task)
        assert out.shape == (b, 8, d)
        assert out_mask.shape == (b, 8)
        assert out_mask.sum(dim=-1).min().item() >= 1


class TestTokenPruner:
    @pytest.fixture
    def pruner(self) -> TokenPruner:
        return TokenPruner(hidden_size=128, num_patches=64)

    def test_eval_shortens_sequence(self, pruner: TokenPruner) -> None:
        pruner.eval()
        b, cls, n, t, d = 2, 1, 64, 16, 128
        seq_len = cls + n + t

        tokens = torch.randn(b, seq_len, d)
        pos = torch.arange(seq_len).unsqueeze(0).expand(b, -1)
        mask = torch.ones(b, seq_len, dtype=torch.long)

        out_tokens, out_pos, out_mask = pruner(tokens, pos, mask, cls_token_count=cls)
        assert out_tokens.shape[0] == b and out_tokens.shape[2] == d
        assert out_tokens.shape[1] <= seq_len
        assert out_pos.shape == (b, out_tokens.shape[1])
        assert out_mask is not None
        assert out_mask.shape == (b, out_tokens.shape[1])

    def test_train_keeps_sequence_length(self, pruner: TokenPruner) -> None:
        pruner.train()
        pruner.set_noise_scale(0.1)
        b, cls, n, t, d = 2, 1, 64, 16, 128
        seq_len = cls + n + t

        tokens = torch.randn(b, seq_len, d, requires_grad=True)
        pos = torch.arange(seq_len).unsqueeze(0).expand(b, -1)
        mask = torch.ones(b, seq_len, dtype=torch.long)

        out_tokens, out_pos, out_mask = pruner(tokens, pos, mask, cls_token_count=cls)
        assert out_tokens.shape == (b, seq_len, d)
        assert out_pos.shape == (b, seq_len)
        assert out_mask is not None and out_mask.shape == (b, seq_len)
        out_tokens.sum().backward()
        assert tokens.grad is not None
