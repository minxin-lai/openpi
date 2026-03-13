# LIBERO Spatial 实验记录

## 统计口径

- 默认配置：开启 `torch.compile`，不强制关闭 `TRITON_AUTOTUNE` / `TORCHINDUCTOR_MAX_AUTOTUNE`
- 推理时间口径：`policy_timing.infer_ms`
- 旧口径复现：
  `OPENPI_TORCH_COMPILE=0 TRITON_AUTOTUNE=0 TORCHINDUCTOR_MAX_AUTOTUNE=0`

## 结果

| 方案 | 成功率 | 关闭优化 `infer_ms` | 默认配置 `infer_ms` |
| --- | ---: | ---: | ---: |
| baseline | 97.4% | 315.0 | 71.0 |
| Encoder-Layer-Prune (30000 step) | 92.4% | 297.0 | 58.5 |
| post encoder pruning | 94.0% | 307.0 | 59.2 |
| post encoder pruning with gauss | 95.0% | 306.0 | 59.0 |

## 备注

- `Encoder-Layer-Prune` 对应旧 worktree：
  `/workspace/laiminxin/vla-opt-openpi-old/third_party/openpi`
- 跨方案比较时，只比较同一列里的 `infer_ms`
