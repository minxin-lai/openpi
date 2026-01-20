from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import numpy as np
import torch
from openpi_client import base_policy as _base_policy
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy as _policy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyTraceConfig:
    out_dir: str
    dump_attn: bool = False
    attn_layers: Tuple[int, ...] = ()
    save_policy_images: bool = True
    print_attn: bool = True
    max_dumps: int = 200
    every_n: int = 1


def _pil_images_from_obs(obs: Dict[str, Any]) -> List[Image.Image]:
    out: List[Image.Image] = []
    for key in ("observation/image", "observation/wrist_image"):
        arr = obs.get(key, None)
        if arr is None:
            continue
        try:
            a = np.asarray(arr)
            if a.ndim != 3 or a.shape[-1] not in (1, 3, 4):
                continue
            if a.dtype != np.uint8:
                a = a.astype(np.uint8, copy=False)
            out.append(Image.fromarray(a))
        except Exception:
            continue
    return out


class TracedPolicy(_base_policy.BasePolicy):
    """
    Wrap an OpenPI policy to enable server-side tracing (Pi0.5 / PyTorch).

    This re-implements the PyTorch branch of `openpi.policies.policy.Policy.infer` so we can:
      - run `sample_actions` un-compiled (torch.compile often disables hooks),
      - wrap the sampling call with `OpenPIPytorchTracer`,
      - write tracer dumps + optional images.
    """

    def __init__(self, policy: _policy.Policy, *, trace_cfg: PolicyTraceConfig) -> None:
        self._policy = policy
        self._cfg = trace_cfg

        if not getattr(self._policy, "_is_pytorch_model", False):
            raise ValueError("TracedPolicy currently only supports PyTorch-loaded OpenPI policies.")

        try:
            from tracer.adapters.openpi_pytorch import OpenPIPytorchTraceConfig, OpenPIPytorchTracer
            from tracer.run_writer import TraceWriter
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Tracing requested but `tracer` package cannot be imported. "
                "Ensure you are running inside the vla-opt repo (so `tracer/` is on PYTHONPATH)."
            ) from e

        self._OpenPIPytorchTracer = OpenPIPytorchTracer
        self._OpenPIPytorchTraceConfig = OpenPIPytorchTraceConfig
        run_root = Path(self._cfg.out_dir)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "dumps").mkdir(parents=True, exist_ok=True)
        (run_root / "images").mkdir(parents=True, exist_ok=True)
        (run_root / "report").mkdir(parents=True, exist_ok=True)
        logger.info("trace out_dir initialized: %s", str(run_root.resolve()))

        self._writer = TraceWriter(run_root, always_dump=True)

        self._infer_idx = 0

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        obs = dict(obs)
        trace_meta = obs.pop("__trace_meta__", None)

        idx = int(self._infer_idx)
        self._infer_idx += 1

        should_trace = bool(self._cfg.dump_attn) and bool(self._cfg.out_dir)
        if should_trace and int(self._cfg.every_n) > 1 and (idx % int(self._cfg.every_n) != 0):
            should_trace = False
        if should_trace and int(self._cfg.max_dumps) > 0 and int(self._writer.dump_count) >= int(self._cfg.max_dumps):
            should_trace = False

        sample = None
        tracer = None
        if should_trace:
            meta_d = trace_meta if isinstance(trace_meta, dict) else {}
            task_id = meta_d.get("task_id", None)
            episode_idx = meta_d.get("episode_idx", None)
            step_idx = meta_d.get("step_idx", None)
            query_idx = meta_d.get("query_idx", idx)
            layout = "task_ep" if isinstance(task_id, int) and isinstance(episode_idx, int) else "flat"

            sample = self._writer.new_sample(
                task_id=task_id if isinstance(task_id, int) else None,
                episode_idx=episode_idx if isinstance(episode_idx, int) else None,
                step_idx=step_idx if isinstance(step_idx, int) else None,
                query_idx=query_idx if isinstance(query_idx, int) else None,
                layout=layout,
            )
            tracer_cfg = self._OpenPIPytorchTraceConfig(
                dump_attn=True,
                attn_layers=tuple(int(x) for x in (self._cfg.attn_layers or ())),
                save_policy_images=bool(self._cfg.save_policy_images),
                print_attn=bool(self._cfg.print_attn),
            )
            tracer = self._OpenPIPytorchTracer(getattr(self._policy, "_model"), config=tracer_cfg)

        t0 = time.monotonic()
        if tracer is not None:
            tracer.__enter__()
        try:
            outputs = self._infer_pytorch(obs, noise=noise, use_eager_sample_actions=bool(tracer is not None))
        finally:
            if tracer is not None:
                tracer.__exit__(None, None, None)
        infer_s = time.monotonic() - t0

        if tracer is not None and sample is not None:
            if not getattr(tracer, "attn_task_to_vis_by_layer", None):
                logger.warning(
                    "trace captured no attention (will skip dump): infer_idx=%s out_dir=%s",
                    idx,
                    str(self._cfg.out_dir),
                )
            try:
                logger.info("trace debug: infer_idx=%s %s", idx, getattr(tracer, "debug_summary", {}))
            except Exception:
                pass

            pil_images = _pil_images_from_obs(obs) if bool(self._cfg.save_policy_images) else None
            meta_d = trace_meta if isinstance(trace_meta, dict) else {}
            meta = {
                "sample_id": sample.sample_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "infer_idx": idx,
                "prompt": obs.get("prompt", None),
                "task_suite": meta_d.get("task_suite", None),
                "task_id": meta_d.get("task_id", None),
                "episode_idx": meta_d.get("episode_idx", None),
                "step_idx": meta_d.get("step_idx", None),
                "query_idx": meta_d.get("query_idx", None),
            }
            try:
                dump = tracer.build_dump(
                    meta=meta,
                    policy_images=pil_images,
                    images_dir=str(sample.images_dir),
                    dump_path=str(sample.dump_path),
                    perf={"infer_latency_s": float(infer_s)},
                )
                if dump is not None:
                    self._writer.write_dump(dump, dump_path=sample.dump_path)
            except Exception:
                logger.exception("trace dump failed (will continue): dump_path=%s", str(sample.dump_path))

            if bool(self._cfg.print_attn) and getattr(tracer, "attn_task_to_vis_by_layer", None):
                for layer, vec in tracer.attn_task_to_vis_by_layer.items():
                    try:
                        v = torch.as_tensor(vec).flatten()
                        topv, topi = torch.topk(v, k=min(8, int(v.numel())))
                        logger.info(
                            "trace attn layer=%s n=%d min=%.6g max=%.6g mean=%.6g top_idx=%s top_val=%s dump=%s",
                            str(layer),
                            int(v.numel()),
                            float(v.min().item()),
                            float(v.max().item()),
                            float(v.mean().item()),
                            topi.tolist(),
                            [float(x) for x in topv.tolist()],
                            str(sample.dump_path),
                        )
                    except Exception:
                        continue

        return outputs

    def _infer_pytorch(self, obs: dict, *, noise: np.ndarray | None, use_eager_sample_actions: bool) -> dict:
        policy = self._policy

        # Copy and apply input transforms (same as Policy.infer).
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = policy._input_transform(inputs)  # noqa: SLF001

        pytorch_device = getattr(policy, "_pytorch_device", "cpu")
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(pytorch_device)[None, ...], inputs)

        sample_kwargs = dict(getattr(policy, "_sample_kwargs", {}) or {})
        if noise is not None:
            n = torch.from_numpy(np.asarray(noise)).to(pytorch_device)
            if n.ndim == 2:
                n = n[None, ...]
            sample_kwargs["noise"] = n

        observation = _model.Observation.from_dict(inputs)

        model = getattr(policy, "_model")  # noqa: SLF001
        sample_actions = None
        if use_eager_sample_actions:
            sample_actions = getattr(model, "_sample_actions_eager", None)
        if not callable(sample_actions):
            sample_actions = getattr(model, "sample_actions")

        start_time = time.monotonic()
        actions = sample_actions(pytorch_device, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        outputs = policy._output_transform(outputs)  # noqa: SLF001
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs
