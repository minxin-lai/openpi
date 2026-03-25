import dataclasses
import pathlib
from types import SimpleNamespace

import pytest
import torch
import wandb

from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as _config
from vla_opt.pruning.config import TrainScheduleEntry, resolve_train_schedule_entry

from . import train_pytorch


class _FakeRun:
    def __init__(self, run_id: str, *, entity: str = "mxlai-ustc", project: str = "vla-opt"):
        self.id = run_id
        self.entity = entity
        self.project = project
        self.url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def _make_config(tmp_path: pathlib.Path):
    ckpt_base = tmp_path / "checkpoints"
    config = dataclasses.replace(
        _config._CONFIGS_DICT["debug"],  # noqa: SLF001
        checkpoint_base_dir=str(ckpt_base),
        exp_name="wandb_resume_test",
        entity_name="mxlai-ustc",
        project_name="vla-opt",
    )
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return config


def test_init_wandb_resume_existing_run(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    config = _make_config(tmp_path)
    (config.checkpoint_dir / "wandb_id.txt").write_text("existing-run")

    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        train_pytorch.wandb.run = _FakeRun("existing-run")

    monkeypatch.setattr(train_pytorch.wandb, "init", fake_init)

    train_pytorch.init_wandb(config, resuming=True, enabled=True)

    assert calls == [{"id": "existing-run", "resume": "must", "entity": "mxlai-ustc", "project": "vla-opt"}]
    assert (config.checkpoint_dir / "wandb_id.txt").read_text() == "existing-run"


def test_init_wandb_resume_falls_back_to_new_run(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    config = _make_config(tmp_path)
    (config.checkpoint_dir / "wandb_id.txt").write_text("old-run")

    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        if kwargs.get("resume") == "must":
            raise wandb.errors.UsageError("resume target not found")
        train_pytorch.wandb.run = _FakeRun("new-run")

    monkeypatch.setattr(train_pytorch.wandb, "init", fake_init)

    train_pytorch.init_wandb(config, resuming=True, enabled=True)

    assert calls[0] == {"id": "old-run", "resume": "must", "entity": "mxlai-ustc", "project": "vla-opt"}
    assert calls[1]["name"] == "wandb_resume_test"
    assert calls[1]["entity"] == "mxlai-ustc"
    assert calls[1]["project"] == "vla-opt"
    assert calls[1]["config"]["project_name"] == "vla-opt"
    assert (config.checkpoint_dir / "wandb_id.txt").read_text() == "new-run"


def test_mean_prompt_tokens_effective_uses_mask():
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, False, False, False],
        ]
    )

    value = train_pytorch._mean_prompt_tokens_effective(mask)

    assert value == pytest.approx(2.0)


def test_resolve_train_schedule_entry_uses_absolute_step():
    schedule = (
        TrainScheduleEntry(progress=0, stage="mask", keep_ratio=1.0),
        TrainScheduleEntry(progress=20000, stage="gather", keep_ratio=0.75),
        TrainScheduleEntry(progress=30000, stage="gather", keep_ratio=0.50),
        TrainScheduleEntry(progress=40000, stage="gather", keep_ratio=0.25),
    )

    assert resolve_train_schedule_entry(schedule, step=19999).keep_ratio == pytest.approx(1.0)
    assert resolve_train_schedule_entry(schedule, step=20000).keep_ratio == pytest.approx(0.75)
    assert resolve_train_schedule_entry(schedule, step=30000).keep_ratio == pytest.approx(0.50)
    assert resolve_train_schedule_entry(schedule, step=60000).keep_ratio == pytest.approx(0.25)


def test_maybe_log_prefix_summary_is_throttled_and_deduped(monkeypatch: pytest.MonkeyPatch):
    model = SimpleNamespace(
        _vla_opt_prefix_log_state={
            "global_step": 100,
            "is_main": True,
            "log_interval": 100,
            "prompt_tokens_effective": 14.0,
        }
    )
    messages = []
    monkeypatch.setattr(pi0_pytorch.logger, "info", lambda msg: messages.append(msg))

    pi0_pytorch._maybe_log_prefix_summary(
        model,
        view_token_pairs=[(256, 64), (256, 64), (256, 64)],
        lang_slots=200,
        total_prefix_tokens=392,
    )
    pi0_pytorch._maybe_log_prefix_summary(
        model,
        view_token_pairs=[(256, 64), (256, 64), (256, 64)],
        lang_slots=200,
        total_prefix_tokens=392,
    )

    assert messages == [
        "[vla-opt] prefix_tokens step=100 views=[256->64,256->64,256->64] prompt_tokens=14.00 lang_slots=200 total_prefix_tokens=392"
    ]


def test_maybe_log_prefix_summary_skips_non_main_rank(monkeypatch: pytest.MonkeyPatch):
    model = SimpleNamespace(
        _vla_opt_prefix_log_state={
            "global_step": 100,
            "is_main": False,
            "log_interval": 100,
            "prompt_tokens_effective": 14.0,
        }
    )
    messages = []
    monkeypatch.setattr(pi0_pytorch.logger, "info", lambda msg: messages.append(msg))

    pi0_pytorch._maybe_log_prefix_summary(
        model,
        view_token_pairs=[(256, 64), (256, 64), (256, 64)],
        lang_slots=200,
        total_prefix_tokens=392,
    )

    assert messages == []
