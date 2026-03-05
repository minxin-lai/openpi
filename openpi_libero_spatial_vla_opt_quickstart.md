# OpenPI Pi0.5（LIBERO Spatial）Quickstart（VLA-OPT / Baseline）

这份文档只保留“可复制命令”。参数细节/解释请看各脚本顶部注释（和可选的 `--help`）。

## 0) 一次性环境准备（server 侧）

```bash
uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## 1) 数据与 norm stats（必需）

```bash
cd third_party/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_libero_spatial
ls /workspace/laiminxin/datasets/lerobot_datasets/libero_spatial/norm_stats.json
```

## 2) 训练（可选）

- baseline：`bash finetune_pi05_baseline.sh`
- vla-opt：`bash finetune_pi05_ve_film_prune.sh`

## 3) 推理 server（最常用）

- baseline：`bash server_pi05_libero_baseline.sh`
- vla-opt：`bash server_pi05_libero_vla_opt.sh`

## 4) LIBERO 评测 client

- baseline：`bash client_libero_eval_baseline.sh`
- vla-opt：`bash client_libero_eval_vla_opt.sh`

## 5) 纯推理 perf 对比（baseline vs vla-opt）

```bash
cd third_party/openpi
bash compare_pi05_libero_perf.sh
```

## 6) 我该用哪个脚本？

- **要 dump + 可视化（生成 `*.pt`）**：启动 `server_pi05_libero_{baseline,vla_opt}.sh`（tracer dump），然后 `bash viz_trace_overlays.sh --last`（画图到 `<trace_dir>/plots/`）
- **要性能对比**：`bash compare_pi05_libero_perf.sh`（输出 `timing.parquet` + `nvidia_smi.csv`）
- **要分析/定位（token/KV）**：`bash debug_pi05_libero_kv.sh`（配合 `scripts/generate_debug_kv_report.py`）

## 6) Tracer：dump + plot overlays（LLM + Vision Encoder）

```bash
cd third_party/openpi
bash viz_trace_overlays.sh --last
```

## 7) Debug token/KV（OPENPI_DEBUG）+ 自动报告（写入 runs/）

```bash
cd third_party/openpi
bash debug_pi05_libero_kv.sh
```

```bash
uv run python scripts/generate_debug_kv_report.py --run-dir runs/debug_kv_pi05_libero_<...>
```

---
## 8) Tracer：dump + plot overlays（LLM + Vision Encoder）

• # 1) 对最新一次 trace 画原有 overlays + Stage2(shared/unique) 彩色 overlays（默认开启）
  cd third_party/openpi
  bash viz_trace_overlays.sh --last

  # 2) 指定 trace 目录
  bash viz_trace_overlays.sh runs/openpi_pi05_libero_trace_YYYYMMDD_HHMMSS

  # 3) 关闭 Stage2(shared/unique) 彩色 overlays，只画原有 tracer overlays
  bash viz_trace_overlays.sh --last --no-consensus-viz

  # 4) 自定义 Stage2 参数（默认 pair=0,1 threshold=0.4 margin=0 max=0）
  bash viz_trace_overlays.sh --last --consensus-pair 0,1 --consensus-threshold 0.4 --consensus-margin 0.0
  --consensus-max 50

  # Stage2 consensus 文档（设计/实现/中间输出查看/调参）
  # docs/consensus_stage2.md

  # 5) 原 tracer.plot_routing_overlays 的额外参数仍然支持（放在后面即可）
  bash viz_trace_overlays.sh --last --heatmap_scale fixed --vmin 0 --vmax 0.003 --alpha 0.75 --cmap inferno

如果你发现仓库里有多个脚本/文档重复：以脚本顶部注释 + `--help` + 本 quickstart 为准。
