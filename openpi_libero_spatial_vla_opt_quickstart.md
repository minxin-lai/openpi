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
- vla-opt：

```bash
cd third_party/openpi
bash server_pi05_libero_vla_opt.sh --observe-config configs/observe/infer_light.json
```

## 4) LIBERO 评测 client

- baseline：`bash client_libero_eval_baseline.sh`
- vla-opt：`bash client_libero_eval_vla_opt.sh`

## 5) 纯推理 perf 对比（baseline vs vla-opt）

```bash
cd third_party/openpi
bash compare_pi05_libero_perf.sh
```

## 6) 我该用哪个脚本？

- **要性能对比**：`bash compare_pi05_libero_perf.sh`（输出 `timing.parquet` + `nvidia_smi.csv`）

如果你发现仓库里有多个脚本/文档重复：以脚本顶部注释 + `--help` + 本 quickstart 为准。
