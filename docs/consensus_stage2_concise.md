# Stage2 Consensus（跨视角 Top‑K 一致性）简明 README

本文件提供 Stage2 consensus 的最小可复现流程：从定义/实现入口，到验证输入、扫参、可视化与排查。

> 运行位置：`third_party/openpi` 目录下。
>
> 约定变量：
> ```bash
> TRACE="runs/<openpi_trace_dir>"
> ```

---

## 1. 定义与输出（设计）

目标：比较两个 view（如 `0,1`）各自保留下来的 **Top‑K 视觉 token**，判断哪些 token 在跨 view 上可稳定匹配：
- **Shared**：跨 view 可匹配且置信度足够高
- **Unique**：跨 view 找不到足够稳定/足够相似的匹配

主要输出：
- **彩色叠加图**：shared=绿，unique=红
- **可汇总数值**：`shared_ratio_*`、`unique_ratio_*` 等（JSON/CSV）
- **扫参结果**：`grid.csv` + 直方图（用于选 `threshold/margin`）

---

## 2. 输入数据（dump schema）

Stage2 consensus 计算依赖 tracer dumps 中的以下字段（缺一不可）：
- `routing.keep_tokens`: `Tensor[B,V,K,D]`（Top‑K token embedding）
- `routing.keep_scores`: `Tensor[B,V,K]`（Top‑K score）
- `routing.keep_indices`: `Tensor[B,V,K]`（原始 patch index）

可视化建议具备：
- `align.image_paths`（每个 view 的原图路径）
- `routing.patch_grid_hw` 或 `align.patch_grid_hw`（patch 网格大小）

---

## 3. 算法要点（实现逻辑）

对 view A/B 的 Top‑K tokens：
1) 相似度矩阵：`S[i,j] = cosine(a_i, b_j)`（`S` 为 `[B,K,K]`）
2) **MNN**（Mutual Nearest Neighbor，互为最近邻）
   - `j* = argmax_j S[i,j]`（对每个 A token `i`，找 B 里最像的 `j*`）
   - `i* = argmax_i S[i,j]`（对每个 B token `j`，找 A 里最像的 `i*`）
   - 满足 MNN：`i` 的最近邻是 `j*`，且 `j*` 的最近邻反过来也是 `i`
3) **threshold**：`best_sim = max_j S[i,j] >= threshold`（要求“像得够像”）
4) **margin（可选）**：`delta = top1 - top2 >= margin`（要求“像得够明确”，过滤歧义匹配）

最终 shared（逻辑为 AND）：`MNN ∩ (best_sim>=threshold) ∩ (delta>=margin)`

直观关系：
- **MNN** 是结构性约束（匹配是否稳定）；`threshold/margin` 是质量约束（相似度是否足够、是否足够明确）
- `threshold/margin` 只能让 shared 变少，不会让 shared 超过 MNN 的上限；当 `mutual_rate` 很低时，调阈值通常只能更严格/更干净

---

## 4. 参数含义（怎么调）

| 参数 | 作用 | 典型现象 |
|---|---|---|
| `threshold` | best_sim 门槛 | 低→thr_rate≈1，shared 不随 threshold 变化；高→shared 下降 |
| `margin` | 匹配明确度门槛 | 过大→margin_rate≈0，shared 接近 0；合适→过滤模糊匹配 |
| `pair` | 比较的 view | pair 不对会导致结果不可解释 |

重要提示：shared 的上限通常受 `MNN` 比例（`mutual_rate`）限制；当 `mutual_rate` 很低时，调 `threshold/margin` 只能“更严格/更干净”，难以显著提高 shared。

---

## 5. 从零到一：验证 + debug 工作流（可一键执行）

以下命令默认在 `third_party/openpi` 下执行。

### 5.1 检查 dumps 是否可算 consensus（第一步）

```bash
cd third_party/openpi
TRACE="runs/openpi_pi05_libero_trace_20260203_215106"
bash scripts/check_consensus_inputs.sh --input "${TRACE}/dumps" --pair 0,1 --max 3
```

关注输出：
- `keep_*` 的 shape 是否为 `[B,V,K,D]/[B,V,K]/[B,V,K]`
- `meta.view_names`、`patch_grid_hw`、`align.image_paths`
- `keep_indices` 在 view0/view1 是否完全相同（若相同，shared 可能天然偏高）

### 5.2 直接出图（先建立直觉）

```bash
cd third_party/openpi
TRACE="runs/openpi_pi05_libero_trace_20260203_215106"
bash viz_trace_overlays.sh "${TRACE}" \
  --plots-subdir "plots_consensus_default" \
  --consensus-pair 0,1 --consensus-threshold 0.4 --consensus-margin 0.0 \
  --consensus-max 200
```

输出目录：
- `runs/.../plots_consensus_default/consensus/shared_unique_color/`

图说明（`runs/.../<plots-subdir>/consensus/`）：
- `similarity/*.png`：相似度矩阵 `S[i,j]=cos(a_i,b_j)`（batch‑0）
  - 横轴：view B 的 token index `j`（`0..K-1`）；纵轴：view A 的 token index `i`（`0..K-1`）
  - 颜色：cosine similarity（范围固定 `[-1,1]`）
  - 黑点：最终 shared 匹配对 `(i,j)`（已应用 MNN + threshold + margin）
  - 判读口诀（快速判断 MNN 多/少）：
    - **点很多且靠近对角线**：两 view Top‑K embedding 排序/语义结构相似，MNN 通常较多（shared 上限更高）
    - **点很少但底色整体很强**：整体相似度高但互为最近邻少，常见于“很多 token 都很像 → 最近邻竞争激烈”（delta 往往很小）
    - **点很少且底色接近中性**：跨 view 对齐差或内容差异大，MNN/threshold 都可能偏低
    - **点离对角线成团**：存在稳定重排（A 的某段更像 B 的另一段），MNN 可能不低但对应关系会错位
- `topk_r/*.png`：Top‑K score（`routing.keep_scores`）按 `keep_indices` scatter 回 patch 网格并叠加到原图
  - 颜色：热力图使用 `heatmap_scale="minmax"`（每张图按自身 min/max 归一化，适合看“哪里更热”，不适合跨图比绝对值）
- `shared_mask/*.png` / `unique_mask/*.png`：shared/unique 的二值 mask 叠加到原图
- `shared_unique_color/*.png`：shared=绿、unique=红（patch 级别 NEAREST 放大叠加），用于直观看共识/分歧区域

### 5.3 批量导出数值（JSON/CSV）并汇总

```bash
cd third_party/openpi
TRACE="runs/openpi_pi05_libero_trace_20260203_215106"
OUT="${TRACE}/stage2_consensus_thr0.4_m0.0"
.venv/bin/python ../../tools/view_consensus/compute_consensus.py \
  --input "${TRACE}/dumps" --pair 0,1 --threshold 0.4 --margin 0.0 --out-dir "${OUT}"
.venv/bin/python ../../tools/view_consensus/summarize_consensus.py \
  --input "${OUT}" --plots
```

输出：
- `${OUT}/summary/summary.csv`
- `${OUT}/summary/plots/*`

图说明（`${OUT}/summary/plots/`，统计单位通常是“每个 dump 一条记录”，使用 batch‑0）：
- `hist_shared_ratio_a.png` / `hist_shared_ratio_b.png`
  - 横轴：`shared_ratio_*`（范围 `[0,1]`）；纵轴：dump 计数
  - 定义：`shared_ratio_a = shared_count_a / K`
- `hist_unique_ratio_cls_a.png` / `hist_unique_ratio_cls_b.png`
  - 横轴：`unique_ratio_cls_*`（范围 `[0,1]`）；纵轴：dump 计数
  - 定义（score mass，按是否 shared 硬切分）：`unique_ratio_cls = (total_mass - shared_mass) / total_mass`
- `hist_unique_ratio_c_a.png` / `hist_unique_ratio_c_b.png`
  - 横轴：`unique_ratio_c_*`（范围 `[0,1]`）；纵轴：dump 计数
  - 定义（连续共识，按 best_sim 加权）：`unique_mass=sum(score*(1-best_sim))`，`consensus_mass=sum(score*best_sim)`，`unique_ratio_c=unique_mass/(unique_mass+consensus_mass)`
- `scatter_unique_ratio_cls_*_vs_step.png`
  - 横轴：`step_idx`；纵轴：`unique_ratio_cls_*`

### 5.4 先粗扫一次（得到 best_sim/delta 的分布直方图）

一句话目的：先摸清 `best_sim`（用于 threshold）和 `delta`（用于 margin）的数值量级，避免下一步扫到“全通过/全拒绝”的无效区间。

```bash
cd third_party/openpi
TRACE="runs/openpi_pi05_libero_trace_20260203_215106"
.venv/bin/python ../../tools/view_consensus/sweep_threshold_margin.py \
  --input "${TRACE}/dumps" --pair 0,1 \
  --thresholds 0.2:0.8:0.05 --margins 0.0,0.05,0.1 \
  --max-dumps 200 --plots --tag sweep_coarse
```

参数写法说明（`--thresholds/--margins`）：
- 支持两种格式：
  - CSV：`0.2,0.4,0.6`
  - 区间：`start:stop:step`（从 `start` 开始每次加 `step`，直到超过 `stop` 为止；实现包含少量浮点容差，`stop` 通常会被包含）
- `--thresholds`：cosine 相似度门槛（对 `best_sim=max_j S[i,j]` 生效）
- `--margins`：匹配明确度门槛（对 `delta=top1-top2` 生效）

直方图位置：
- `${TRACE}/plots/consensus_sweep/sweep_coarse/plots/best_sim_hist.png`
- `${TRACE}/plots/consensus_sweep/sweep_coarse/plots/delta_hist.png`

直方图含义：
- `best_sim_hist`：`best_sim=max_j S[i,j]` 的分布
  - 横轴：`best_sim`；纵轴：`count`（样本数；每个样本对应“某个 dump 的某个 A‑side Top‑K token”，约为 `used_dumps * B * K`）
  - 红竖线：本次 sweep 扫描的 `threshold`；用于选择 `threshold` 的有效区间（让 `thr_rate` 既不是 0 也不是 1）
- `delta_hist`：`delta=top1-top2` 的分布
  - 横轴：`delta`；纵轴：`count`（同上）
  - 红竖线：本次 sweep 扫描的 `margin`；用于选择 `margin` 的有效区间（让 `margin_rate` 既不是 0 也不是 1）

粗扫是否“扫对”的快速判断：
- `thr_rate` 在这轮里几乎都是 `1.000`：threshold 区间太低（这一轮看不出 threshold 的影响）
- `margin_rate` 在这轮里几乎都是 `0.000`：margin 区间太高（这一轮 shared 会塌到接近 0）

### 5.5 用脚本读 grid/meta，输出建议扫参区间 + top 配置

一句话目的：把粗扫结果翻译成“下一轮 tight sweep 应该扫的区间”，并快速判断这轮扫参是否有效。

```bash
cd third_party/openpi
TRACE="runs/openpi_pi05_libero_trace_20260203_215106"
bash scripts/inspect_consensus_sweep.sh --trace "${TRACE}" --tag sweep_coarse --margin 0.0 --top 10
```

如何读输出（优先级从高到低）：
- `suggested_scan_ranges`：基于直方图分位数推导出的建议 `threshold/margin` 区间
- `thr_rate / margin_rate / mutual`：分别对应 `(best_sim>=thr)/(delta>=m)/MNN` 的通过率
  - `thr_rate≈1`：threshold 太低（几乎全通过）；`thr_rate≈0`：threshold 太高（几乎全拒绝）
  - `margin_rate≈1`：margin 太低（几乎不起作用）；`margin_rate≈0`：margin 太高（几乎全拒绝）
- `curve:`：固定一个 margin 时，threshold 增大后 `thr_rate`/`shared` 是否开始下降（是否“起作用”）
- `shared≈mutual`：这轮里 threshold/margin 基本没筛选，shared 被 MNN 上限卡住（需要按 `suggested_scan_ranges` 换区间）
- `sweep_hint`：一条可直接复制执行的 tight sweep 命令（已带 `start:stop:step`）

### 5.6 按建议区间再扫一轮（这一轮才会看到 consensus 随参数变化）

一句话目的：tight sweep 才用于“选参数”。目标是让 `thr_rate` 和 `margin_rate` 都落在中间段（既不是 0 也不是 1），从而观察 shared/unique 随参数变化。

示例（范围需按 5.5 的输出调整）：
```bash
cd third_party/openpi
TRACE="runs/openpi_pi05_libero_trace_20260203_215106"
.venv/bin/python ../../tools/view_consensus/sweep_threshold_margin.py \
  --input "${TRACE}/dumps" --pair 0,1 \
  --thresholds 0.93:0.98:0.002 \
  --margins 0.0:0.01:0.001 \
  --max-dumps 200 --plots --tag sweep_tight
```

区间示例解释：
- `--thresholds 0.926:0.971:0.002` 表示扫 `0.926, 0.928, 0.930, ...`（直到接近/达到 `0.971`）
- `--margins 0.0:0.011:0.001` 表示扫 `0.000, 0.001, 0.002, ...`（直到接近/达到 `0.011`）

再用 inspect 看不同 margin 切片下的 top 配置：
```bash
cd third_party/openpi
TRACE="runs/openpi_pi05_libero_trace_20260203_215106"
bash scripts/inspect_consensus_sweep.sh --trace "${TRACE}" --tag sweep_tight --margin 0.0 --top 10
bash scripts/inspect_consensus_sweep.sh --trace "${TRACE}" --tag sweep_tight --margin 0.002 --top 10
bash scripts/inspect_consensus_sweep.sh --trace "${TRACE}" --tag sweep_tight --margin 0.005 --top 10
```

从 tight sweep 里选参数（大白话规则）：
- 想让 threshold 真正在筛：选能让 `thr_rate` 明显小于 `1.000`、但又不接近 `0.000` 的那段 threshold（`curve:` 里能看到）
- margin 用来“去掉模棱两可的匹配”：从小的 margin 开始加，直到 `margin_rate` 开始下降、shared 也更“干净”
- 实际对比建议挑 2~3 组参数：一组 shared 偏多、一组 shared 偏少（便于对比空间位置差异）

### 5.7 选 2~3 组参数做可视化对比（验证数值 ⇔ 空间位置）

一句话目的：sweep/grid 回答“比例/趋势”，可视化回答“共识/分歧落在图像哪里”，用于确认结论是否合理。

```bash
cd third_party/openpi
TRACE="runs/openpi_pi05_libero_trace_20260203_215106"
bash viz_trace_overlays.sh "${TRACE}" \
  --plots-subdir "plots_thr0.950_m0.002" \
  --consensus-pair 0,1 --consensus-threshold 0.950 --consensus-margin 0.002 --consensus-max 200
```

对比方式：每组参数用不同 `--plots-subdir`，直接切换查看 `shared_unique_color/` 即可。

---

## 6. 常见现象快速定位（用 sweep 指标判断）

- `thr_rate≈1` 且 shared 不变：`threshold` 太低（区间不在 best_sim 分布附近）
- `margin_rate≈0` 且 shared≈0：`margin` 太大（区间不在 delta 分布附近）
- `shared` 上限贴着 `mutual`：瓶颈在 `MNN` 本身比例偏低，`threshold/margin` 只能更严格/更干净

---

## 7. 深入分析：为什么调参无效？(The MNN Bottleneck)

如果你发现 **`shared` 数量极低且不随参数降低而增加**（例如卡在 `mutual_rate ≈ 0.05`），这不是参数没调好，而是 **MNN (Mutual Nearest Neighbor)** 算法在当前数据下的**硬约束**。

### 原理
Stage 2 Consensus 的核心漏斗是：
`Input Tokens` (100%) → **`MNN` (互为最近邻)** → `Threshold/Margin` (质量过滤) → `Shared`

- **MNN 是必选项**：只有当 A 的 Top-1 是 B，且 B 的 Top-1 也是 A 时，才有可能成为 Shared。
- **参数只能做减法**：`threshold` 和 `margin` 只能从 MNN 筛选出的集合里进一步剔除不自信的匹配，**永远无法找回被 MNN 淘汰的匹配**。

### 为什么 MNN 会低？(不一定是坏事，但无法通过调参解决)
1.  **物理重叠少**：两个 View 确实没看同一个地方。
2.  **特征畸变**：View 变化导致特征空间距离改变，A 的“最像”是 B，但 B 的“最像”可能是 C（导致 MNN 破裂）。
3.  **算法过于严格**：MNN 假设了严格的一一对应关系。在某些视角差异大或覆盖率低的场景下，MNN 可能**不太合适**（过于保守），导致明明是同一个物体但因为特征稍有偏移就没法匹配。

### 怎么解决？(Potential Solutions)

针对这种“局部纹理重复导致 MNN 匹配率低”的问题，常规的工程解法有：

#### 1. 注入位置编码 (Add Positional Encoding)
- **原理**：Token Embedding 本身是不带位置信息的。如果把 Patch 的 (x,y) 坐标编码加进去，那么“左边的桌角”和“右边的桌角”特征就会大不相同。
- **代价**：需要修改 Model 或在 Consensus 计算前手动 concat 位置特征（会破坏纯视觉相似度，但能强行区分不同位置的同类纹理）。

#### 2. 放宽约束：从“双向奔赴”改为“单向强匹配”
- **方法**：放弃 MNN，只算单向 `High Confidence Match`。
- **逻辑**：只要 `S[i,j] > 0.98`（阈值设极高），即使 `j` 的最爱不是 `i`，我们也认。
- **适用**：适合召回率（Shared 数量）比准确率更重要的场景。

#### 3. 引入空间一致性 (Spatial Consistency / RANSAC)
- **原理**：真正的 Shared Patch 不应该是一个个孤立的点，而应该是“成团”的。
- **方法**：增加一步过滤——如果 `(i, j)` 是匹配的，那它们的邻居 `(i+1, j+1)` 也大概率是匹配的。用这种几何约束来捞回那些被 MNN 误杀的边缘点。

#### 4. 甚至...不做任何事
- **反直觉**：如果你的下游任务（如 VLA 动作预测）并不强依赖这些 Shared Token，或者 Shared Token 少反而代表去掉了噪声，那么 **Low Consensus 也许就是正确的表现**。不要为了刷高指标而强行匹配。
