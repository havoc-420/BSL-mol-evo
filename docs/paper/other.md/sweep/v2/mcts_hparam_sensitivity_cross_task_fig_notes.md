# OFO-guided MCTS 超参数试验配套作图说明与图注文案（v2）

## 1. 使用定位

本文件服务于 `v2` 版超参数分析的出图与图注撰写，目标是让**正文跨任务结论**、**附录单任务机制解释**与现有 `csv` 表格保持一致。与 `v1` 不同，`v2` 的核心落点不再是单一 `LUMO(D)` 任务，而是：

- **正文**：用 `LUMO(U)` 与 `HOMO(D)` 的已完成真值评估结果展示跨任务规律；
- **附录**：用 `LUMO(D)` 展示更细的搜索强度与 trade-off 机制；
- **结论层**：强调“存在稳健工作区间”，而不是“存在全任务统一最优点”。

当前建议采用：**正文 1 张主图 + 附录 1 张机制图 + 可选 1 张结论图**。

## 2. 作图总原则

### 2.1 横轴

- 每个子图只展示一个 sweep 维度；
- 横轴保持参数自然顺序，不做人为重排；
- `num_simulations` 建议使用离散点连线图；
- `exploration_weight`、`max_branching`、`max_depth`、`pruning_patience` 使用普通折线图即可。

### 2.2 纵轴

建议统一保留三类信息：

- 主性能轴：`average_improvement`；
- 稳定性轴：`improved_percentage`；
- 结构/多样性轴：`intdiv_avg` 或 `morgan_similarity_avg`。

其中需要明确：

- 对 `increase` 任务，`average_improvement` **越大越好**；
- 对 `decrease` 任务，`average_improvement` **越负越好**；
- 若同图中同时包含 `LUMO(U)` 与 `HOMO(D)`，图注必须显式说明两类任务的判定方向不同。

### 2.3 视觉规范

- `average_improvement` 使用深色主线；
- `improved_percentage` 使用浅色或虚线；
- `intdiv_avg` / `morgan_similarity_avg` 使用第三种颜色；
- 所有图尽量保持相同图例顺序；
- 若正文图同时画两项任务，建议用颜色区分任务、用线型区分指标，避免图例过多。

## 3. 正文主图建议

### 图 1：跨任务主超参数敏感性摘要图

建议正文主图做成 `3` 个子图：

- **(a) `num_simulations`**：展示预算增加对 `LUMO(U)` 更有帮助，而 `HOMO(D)` 更早进入平台区；
- **(b) `exploration_weight`**：展示探索强度的任务依赖性，即 `LUMO(U)` 更偏好更强探索，而 `HOMO(D)` 更偏好较保守探索；
- **(c) `max_branching`**：展示较窄分支可能更激进，但 `20` 作为默认值更稳健。

### 图 1 推荐画法

- 每个子图放两条主性能曲线：`LUMO(U)` 与 `HOMO(D)`；
- 若版面允许，可再叠加成功率曲线；
- 若版面较紧，正文只保留 `average_improvement`，把成功率移到附录表或补充材料；
- 图中不必同时放 `LUMO(D)`，否则会削弱“正文跨任务摘要”这一主线。

### 正文主图图注（英文，可直接使用）

**Figure X. Cross-task hyperparameter sensitivity of OFO-guided MCTS.** We compare the effects of search budget (`num_simulations`), exploration coefficient (`exploration_weight`), and branching width (`max_branching`) using true-evaluated results on `LUMO(U)` and `HOMO(D)`. The results show that OFO-guided MCTS is robust within a reasonable hyperparameter range, while the best operating point remains task-dependent. Increasing the search budget consistently improves the upward `LUMO` task, whereas the `HOMO` decrease task reaches a near-saturated regime at a lower budget. The exploration coefficient exhibits a similar task dependence: stronger exploration benefits `LUMO(U)` in terms of optimization strength, while `HOMO(D)` prefers a smaller coefficient for both improvement and success rate. Narrower branching can be more aggressive on some tasks, but a moderate width provides a more stable cross-task default.

## 4. 附录补充图建议

### 图 S1：`LUMO(D)` 机制型敏感性结果

建议附录放一张 `3` 子图或 `2 + 1` 结构图：

- **(a) `exploration_weight`**：突出“更强探索显著增强改善，但压缩多样性与结构保持”；
- **(b) `max_depth`**：突出“更深路径几乎持续增益，但带来更明显结构漂移”；
- **(c) `pruning_patience`**：突出“该参数影响较温和，属于次级敏感因素”。

若版面不足，可把 `pruning_patience` 单独降为表格，只保留 `exploration_weight` 与 `max_depth` 两个主子图。

### 图 S1 图注（英文，可直接使用）

**Figure S1. Additional single-task sensitivity analysis on the `LUMO` decrease task.** The `LUMO(D)` results provide a detailed view of the search trade-offs behind the cross-task summary. Increasing the exploration coefficient substantially strengthens the optimization objective, but also reduces diversity and structural similarity, indicating a more aggressive search regime. Increasing `max_depth` brings a near-monotonic gain in the objective, suggesting that longer edit trajectories are beneficial, although this gain is accompanied by stronger structural drift. In contrast, varying `pruning_patience` changes the final performance only moderately, implying that pruning strength is a secondary factor compared with budget, exploration, and search depth.

### 图 S2：平衡型默认配置的证据汇总图（可选）

如果需要一张更偏“结论型”的图，可基于 `table_main_key_evidence.csv` 做一个证据摘要图：

- 按 `num_simulations`、`exploration_weight`、`max_branching` 三个维度列出跨任务观察；
- 每一维只强调“最稳健默认值”与“更激进但不更稳健的备选值”；
- 用该图支撑正文中的默认配置选择，而不是再重复所有数值细节。

### 图 S2 图注（英文，可直接使用）

**Figure S2. Evidence summary for the balanced default configuration.** The selected default setting is motivated by cross-task consistency rather than the single best point on one metric. A higher simulation budget remains useful, a moderate exploration coefficient avoids over-specializing to one task, and a branching width of `20` provides a safer balance between optimization strength and stability. These observations support describing the hyperparameter behavior in terms of a stable operating range rather than a universal optimum.

## 5. 中文图注备选

### 正文图中文图注

**图 X  OFO-guided MCTS 的跨任务超参数敏感性结果。** 我们基于已完成真值评估的 `LUMO(U)` 与 `HOMO(D)` 结果，比较了搜索预算、探索系数与分支宽度的影响。结果表明，该方法在合理参数范围内整体稳健，但最优工作点具有任务依赖性：更高预算与更强探索更有利于 `LUMO(U)`，而 `HOMO(D)` 在较保守设置下已达到接近饱和的性能；较窄分支宽度在部分任务上更激进，但中等宽度作为跨任务默认值更稳妥。

### 附录图中文图注

**图 S1  `LUMO(D)` 任务上的补充敏感性分析。** `LUMO(D)` 的结果更清楚地揭示了搜索强度与结果性质之间的权衡：增大 `exploration_weight` 能显著增强改善值，但会压缩多样性和结构保持；增大 `max_depth` 基本持续提升优化幅度，但也会带来更明显的结构漂移；相比之下，`pruning_patience` 的影响整体较温和，说明其属于次级敏感因素。

## 6. 与当前表格 CSV 的对应关系

当前建议的图表—数据映射如下：

- 正文图 1 主要对应：
  - `table_3_1_lumo_up_num_simulations.csv`
  - `table_3_1_homo_down_num_simulations.csv`
  - `table_3_2_lumo_up_exploration_weight.csv`
  - `table_3_2_homo_down_exploration_weight.csv`
  - `table_3_3_lumo_up_max_branching.csv`
  - `table_3_3_homo_down_max_branching.csv`
- 附录图 S1 主要对应：
  - `table_4_1_lumo_down_exploration_weight.csv`
  - `table_4_2_lumo_down_max_depth.csv`
  - `table_4_3_lumo_down_pruning_patience.csv`
- 可选图 S2 对应：
  - `table_main_key_evidence.csv`

如果后续重跑 `generate_table_csvs.py`，上述映射应继续保持不变。

## 7. 版面落地建议

- 正文若篇幅有限，只保留 `Figure X`；
- 附录至少保留 `Figure S1`；
- 若需要更清楚解释默认配置来源，再加入 `Figure S2`；
- 正文避免重新展开 `LUMO(D)` 的细节曲线，以免冲淡跨任务主线；
- 图中数值口径应始终与 `table_manifest.csv` 和对应 `csv` 文件一致。