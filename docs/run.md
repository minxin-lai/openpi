# Run

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi
bash server_pi05_libero_vla_opt.sh \
  --observe-config /workspace/laiminxin/vla-opt/configs/observe/infer_debug.json
```

记下 server 打印的 `observe_dump_dir`。

```bash
cd /workspace/laiminxin/vla-opt/third_party/openpi
bash client_libero_eval_vla_opt.sh
```

渲染整轮 overlay / heatmap：

```bash
PYTHONPATH=/workspace/laiminxin/vla-opt/src \
/workspace/laiminxin/vla-opt/third_party/openpi/.venv/bin/python \
-m vla_opt.observe.render_png \
--run-dir <observe_dump_dir>
```

聚合 pruning stats：

```bash
PYTHONPATH=/workspace/laiminxin/vla-opt/src \
/workspace/laiminxin/vla-opt/third_party/openpi/.venv/bin/python \
-m vla_opt.observe.pruning_stats \
--run-dir <observe_dump_dir>
```
