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
        assert pruner.core.last_kept_per_sample is not None

        out.sum().backward()
        assert patches.grad is not None
        assert pruner.core.last_selected_indices_train is not None

    def test_train_gathers_patch_mask_and_records_indices(self, pruner: ImageTokenPruner) -> None:
        pruner.train()
        pruner.set_noise_scale(None)

        b, n, t, d = 2, 4, 3, 8
        patches = torch.randn(b, n, d, requires_grad=True)
        task = torch.randn(b, t, d)

        # Force selection to always pick index 2 for every query (no noise).
        forced_score = torch.zeros(b, n, n, dtype=patches.dtype)
        forced_score[:, :, 2] = 10.0

        pruner.core.compute_importance_score = lambda *args, **kwargs: forced_score  # type: ignore[method-assign]

        patch_mask = torch.zeros(b, n, dtype=torch.bool)
        patch_mask[:, 2] = True

        out, out_mask = pruner.prune(patches, task, patch_token_mask=patch_mask)
        assert out.shape == (b, n, d)
        assert out_mask.shape == (b, n)
        assert torch.equal(out_mask, patch_mask)

        indices = pruner.core.last_selected_indices_train
        assert indices is not None
        assert indices.shape == (b, n)
        assert torch.equal(indices, torch.full((b, n), 2, dtype=torch.long))

    def test_train_masks_invalid_patches_in_score(self, pruner: ImageTokenPruner) -> None:
        pruner.train()
        pruner.set_noise_scale(None)

        b, n, t, d = 2, 4, 3, 8
        patches = torch.randn(b, n, d)
        task = torch.randn(b, t, d)

        # Make an invalid patch (index 1) have the highest score, but only index 2 is valid.
        forced_score = torch.zeros(b, n, n, dtype=patches.dtype)
        forced_score[:, :, 1] = 10.0
        forced_score[:, :, 2] = 1.0
        pruner.core.compute_importance_score = lambda *args, **kwargs: forced_score  # type: ignore[method-assign]

        patch_mask = torch.zeros(b, n, dtype=torch.bool)
        patch_mask[:, 2] = True

        out, out_mask = pruner.prune(patches, task, patch_token_mask=patch_mask)
        assert out.shape == (b, n, d)
        assert torch.equal(out_mask, patch_mask)

        indices = pruner.core.last_selected_indices_train
        assert indices is not None
        assert torch.equal(indices, torch.full((b, n), 2, dtype=torch.long))

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

    def test_eval_records_kept_indices_and_respects_patch_mask(self, pruner: ImageTokenPruner) -> None:
        pruner.eval()
        pruner.set_noise_scale(None)

        b, n, t, d = 2, 64, 16, 128
        patches = torch.randn(b, n, d)
        task = torch.randn(b, t, d)

        # Disallow keeping patches in the second half.
        patch_mask = torch.zeros(b, n, dtype=torch.bool)
        patch_mask[:, : (n // 2)] = True

        out, out_mask = pruner.prune(patches, task, patch_token_mask=patch_mask)
        assert out.shape[0] == b and out.shape[2] == d
        assert out_mask.shape == (b, out.shape[1])
        assert out_mask.sum(dim=-1).min().item() >= 1

        kept_idx = pruner.core.last_kept_indices_padded
        kept_idx_mask = pruner.core.last_kept_indices_mask
        assert kept_idx is not None and kept_idx_mask is not None
        assert kept_idx.shape == out_mask.shape
        assert torch.equal(kept_idx_mask, out_mask)

        # All kept indices should point to the allowed half.
        for i in range(b):
            idx = kept_idx[i][out_mask[i]].cpu()
            assert (idx < (n // 2)).all()

    def test_eval_ignores_masked_task_tokens(self, pruner: ImageTokenPruner) -> None:
        pruner.eval()
        pruner.set_noise_scale(None)

        b, n, t, d = 2, 64, 32, 128
        valid_t = 8
        patches = torch.randn(b, n, d)
        task = torch.randn(b, t, d)
        task_mask = torch.zeros(b, t, dtype=torch.bool)
        task_mask[:, :valid_t] = True

        # Change the masked-out tail aggressively; output should not change when the mask is applied.
        task_corrupted = task.clone()
        task_corrupted[:, valid_t:] = torch.randn_like(task_corrupted[:, valid_t:]) * 1000.0

        out1, mask1 = pruner.prune(patches, task, task_token_mask=task_mask)
        out2, mask2 = pruner.prune(patches, task_corrupted, task_token_mask=task_mask)

        assert torch.allclose(out1, out2, atol=1e-5, rtol=0.0)
        assert torch.equal(mask1, mask2)


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
