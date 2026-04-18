# OFO-guided MCTS 超参数试验配套作图说明与图注文案

## 1. 使用定位

本文件用于给 `LUMO(D)` 超参数敏感性部分配套出图，目标是让图形与正文/附录叙事直接对齐，减少后续反复改图注的成本。当前建议采用：**正文 1 张主图 + 附录 2 张补充图** 的结构。

## 2. 作图总原则

### 2.1 横轴

- 每张图只展示一个 sweep 维度；
- 横轴保持参数自然顺序，不做重排；
- `num_simulations` 建议使用离散点连线图，必要时可在视觉上采用近似对数间隔；
- 其余参数使用普通折线图即可。

### 2.2 纵轴

建议统一保留三类纵轴信息：

- 主性能轴：`average_improvement`（越负越好，图注中需明确说明）；
- 辅助性能轴：`success_rate`；
- 结构/多样性轴：`IntDiv` 或 `Morgan`。

若版面紧张，正文图优先保留：

- `average_improvement`
- `success_rate`
- `IntDiv`

### 2.3 视觉规范

- `average_improvement` 用深色主线；
- `success_rate` 用浅色或虚线；
- `IntDiv` 用第三种颜色，避免和成功率混淆；
- 所有图统一保留相同图例顺序；
- `LUMO(D)` 任务需在图注中明确说明“`average_improvement` 越负表示改善越强”。

## 3. 正文主图建议

### 图 1：主超参数敏感性摘要图

建议将正文主图做成 `3` 个子图：

- **(a) `num_simulations`**：突出“预算增加带来更强优化，但收益递减且多样性下降”；
- **(b) `exploration_weight`**：突出“探索增强提升性能，但显著压缩多样性”；
- **(c) `max_branching`**：突出“有限宽度已足够，过宽展开无明显收益”。

### 正文主图图注（可直接使用）

**Figure X. Hyperparameter sensitivity of OFO-guided MCTS on the `LUMO` decrease task.** We vary the search budget (`num_simulations`), exploration coefficient (`exploration_weight`), and branching width (`max_branching`) while keeping the remaining settings fixed. More simulations consistently improve the optimization objective, but the gain becomes less pronounced at high budgets and is accompanied by reduced internal diversity. Increasing the exploration coefficient further strengthens both improvement and success rate, yet also causes a clear drop in diversity, indicating a direct exploration–diversity trade-off. In contrast, expanding the branching width beyond a moderate range does not improve performance, suggesting that the OFO prior is sufficiently informative to support effective search under limited-width tree expansion.

## 4. 附录补充图建议

### 图 S1：搜索深度与剪枝耐心值

建议附录单独放一张 `2` 子图：

- **(a) `max_depth`**：突出“更深路径几乎单调提升改善值，但逐步远离起始结构并压缩多样性”；
- **(b) `pruning_patience`**：突出“该参数影响相对温和，属于次级敏感因素”。

### 图 S1 图注（可直接使用）

**Figure S1. Additional sensitivity results for search depth and pruning patience.** Increasing `max_depth` leads to a near-monotonic improvement in the optimization objective, indicating that longer editing trajectories are beneficial for molecular property optimization. However, deeper search also reduces internal diversity and structural similarity to the starting molecules, suggesting that the additional gain comes at the cost of more aggressive structural drift. By comparison, varying `pruning_patience` changes the final results only moderately, implying that pruning strength is a secondary factor relative to search budget, exploration, and branching width.

### 图 S2：平衡型默认配置的证据汇总图

如果需要额外补一张更偏“结论型”的附录图，可用条形图或雷达图展示推荐配置 `800 / 2.0 / 10 / 10 / 3` 相对于各 sweep 极端取值的相对位置。该图不必追求信息量最大，而是用于支撑“为何选择这组默认配置”。

### 图 S2 图注（可直接使用）

**Figure S2. Empirical motivation for the balanced default configuration.** The selected setting (`num_simulations = 800`, `exploration_weight = 2.0`, `max_branching = 10`, `max_depth = 10`, and `pruning_patience = 3`) provides a practical trade-off between optimization strength, success rate, molecular quality, and diversity. Rather than maximizing a single metric, this configuration remains close to the best-performing points while avoiding the excessive diversity loss or computational cost associated with more extreme settings.

## 5. 中文图注备选

### 正文图中文图注

**图 X  OFO-guided MCTS 在 `LUMO` 下降任务上的主超参数敏感性结果。** 在保持其余设置不变的条件下，我们分别考察了搜索预算、探索系数与分支宽度对结果的影响。结果表明，增加 `num_simulations` 能够持续提升优化幅度，但边际收益逐渐减弱，并伴随内部多样性下降；增大 `exploration_weight` 虽可进一步提高改善值与成功率，但也会明显压缩多样性；相比之下，`max_branching` 超过中等范围后并未继续带来性能提升，说明 `OFO` 提供的候选排序已经足以支撑有限宽度搜索。

### 附录图中文图注

**图 S1  搜索深度与剪枝耐心值的补充敏感性结果。** 更大的 `max_depth` 几乎单调提升优化幅度，说明更长编辑路径有利于释放多步优化潜力；但与此同时，多样性与结构相似性持续下降，表明性能提升伴随着更强的结构偏移。相比之下，`pruning_patience` 的影响整体较为温和，说明剪枝强度属于次级敏感因素，而非决定性性能来源。

## 6. 版面落地建议

- 正文若篇幅有限，只保留 `Figure X`；
- 附录至少补 `Figure S1`；
- 若审稿或答辩阶段需要解释默认参数来源，再加入 `Figure S2`；
- 图中不要重新引入 `logP` 相关 sweep，以免偏离当前超参数主线。

## 7. 若后续补 `LUMO(U)` / `HOMO(D)` 的图文扩展建议

如果后续按最小可行集补齐 `LUMO(U)` 与 `HOMO(D)`，建议图文扩展遵循以下原则：

- 保留当前 `LUMO(D)` 图文作为**附录中的完整单任务案例**；
- 在正文中新增一张跨任务摘要图，只保留 `num_simulations`、`exploration_weight`、`max_branching` 三组主文级 sweep；
- 正文叙述强调“趋势在不同任务/方向上是否一致”，而不是重复逐项汇报所有绝对数值；
- 若 `LUMO(U)` / `HOMO(D)` 只完成最小取值集，则正文中以趋势一致性为主，不强行给出新的全局最优点。
