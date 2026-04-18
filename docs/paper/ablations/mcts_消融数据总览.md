# MCTS 侧消融数据总览

## 1. 目的

本文档用于从“实验资产管理”的角度整理当前已经产生的 MCTS 侧消融数据，回答三个问题：

1. **已经跑了什么？**
2. **哪些结果已经能用于论文？**
3. **哪些结果还缺最后一步，不能直接当终稿？**

## 2. 当前实验矩阵

### 2.1 已完成真值评估的实验

| 层级 | 任务 | 变体 | seed | 起始分子数 | 真值评估 | 输出根目录 |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke_lumo` | `lumo_up` | `full / wo_prior / wo_leaf_value / random_topb` | `42` | `5` | 已完成 | `mol_evo/output/paper/ablations/20260413_112535/` |
| `smoke_homo` | `homo_down` | `full / wo_prior / wo_leaf_value / random_topb` | `42` | `5` | 已完成 | `mol_evo/output/paper/ablations/20260413_113451/` |
| `pilot` | `lumo_up / homo_down` | `full / wo_prior / wo_leaf_value / random_topb` | `42` | `20` | 已完成 | `mol_evo/output/paper/ablations/20260413_1536_pilot_mcts/` |

### 2.2 `official` 本轮的实际结果

| 层级 | 任务 | 变体 | seeds | 起始分子数 | 当前状态 | 输出根目录 |
| --- | --- | --- | --- | --- | --- | --- |
| `official` | `lumo_up / homo_down` | `full / wo_prior / wo_leaf_value` | `42 / 43 / 44` | `50` | 搜索已结束；`15 / 18 success`，`3 / 18 search_failed`；未做真值评估 | `mol_evo/output/paper/ablations/20260414_0115_official_mcts/` |

补充说明：

- 失败项全部来自 `homo_down / wo_leaf_value / seed42,43,44`；
- 三个失败 run 都在同一个起始分子 `CC#CC(C)(C)C` 上复现了同一错误：`'NoneType' object has no attribute 'is_expanded'`；
- 本轮 `official` 启动参数中 `RUN_EVAL=0`，因此目录下**没有** `true_eval_summary.tsv` 与 `true_eval_summary_aggregated.tsv`。

## 3. 关键原始文件索引

### 3.1 smoke

- `mol_evo/output/paper/ablations/20260413_112535/true_eval_summary_aggregated.tsv`
  - `LUMO(U)` smoke 真值聚合汇总
- `mol_evo/output/paper/ablations/20260413_113451/true_eval_summary_aggregated.tsv`
  - `HOMO(D)` smoke 真值聚合汇总

### 3.2 pilot

- `mol_evo/output/paper/ablations/20260413_1536_pilot_mcts/true_eval_summary.tsv`
  - 每个 `task × variant × seed` 的论文口径真值结果
- `mol_evo/output/paper/ablations/20260413_1536_pilot_mcts/true_eval_summary_aggregated.tsv`
  - 当前最重要的聚合结果表
- `mol_evo/output/paper/ablations/20260413_1536_pilot_mcts/eval_summary.tsv`
  - 评估阶段逐 run 状态表，已确认 `8 / 8 success`

### 3.3 official

- `mol_evo/output/paper/ablations/20260414_0115_official_mcts/run_summary.tsv`
  - `18` 个 run units 的最终状态表；本轮最关键的 official 索引文件
- `mol_evo/output/paper/ablations/20260414_0115_official_mcts/run_manifest.txt`
  - 记录 profile、任务矩阵、搜索预算与 `RUN_EVAL=0` 等启动参数
- `mol_evo/output/paper/ablations/20260414_0115_official_mcts/tmux_search.log`
  - 官方搜索总日志；可见 `homo_down / wo_leaf_value` 的失败收尾信息
- 各成功 run 下的 `search/batch_results.json`
  - 已有 `15` 份搜索层产物
- 本轮未生成：
  - `true_eval_summary.tsv`
  - `true_eval_summary_aggregated.tsv`
  - `best_results_*.csv`
  - `statistics_summary_*.json`

## 4. 当前可直接引用的结果范围

### 4.1 可以直接用于论文定量论证的

- `pilot` 的 `true_eval_summary_aggregated.tsv`
- `pilot` 的 `true_eval_summary.tsv`
- `smoke` 的两个 `true_eval_summary_aggregated.tsv`

这些结果已经足够支持：

- `wo_leaf_value` 是明显负面对照；
- `wo_prior` 与 `full` 接近；
- `random_topb` 效率显著更差，效果优势不稳定。

### 4.2 可以作为支持性证据引用的

- `official` 的 `run_summary.tsv`
- `official` 的失败日志（`homo_down / wo_leaf_value / seed42,43,44`）

这些搜索层证据可以支持：

- `wo_leaf_value` 在 `official` 预算与 `homo_down` 任务上暴露出**复现性的稳定性问题**；
- `wo_prior` 在本轮 `official` 中没有出现系统性失败（`6 / 6 success`）；
- `full` 在本轮 `official` 中也保持了完整通过（`6 / 6 success`）。

### 4.3 还不适合作为最终主表定稿的

- 本轮 `official` 的任何数值性终表

原因是：

- 本轮没有真值评估；
- `wo_leaf_value` 的 `homo_down` 三个 seed 都失败了，矩阵本身不完整；
- 因此目前拿不到跨 `3` 个 seed 的可比真值聚合均值和标准差。

## 5. 当前状态快照（截至 2026-04-15 13:04）

### 5.1 `official` 到底完成了什么

根据 `run_summary.tsv`，本轮 `official` 的最终状态是：

- `lumo_up / full`：`3 / 3 success`
- `lumo_up / wo_prior`：`3 / 3 success`
- `lumo_up / wo_leaf_value`：`3 / 3 success`
- `homo_down / full`：`3 / 3 success`
- `homo_down / wo_prior`：`3 / 3 success`
- `homo_down / wo_leaf_value`：`0 / 3 success`，全部 `search_failed`

这意味着：

- `official` 已经结束，不再是“进行中”；
- 但它也没有形成可直接替换 `pilot` 的多 seed 真值终表；
- 现阶段它主要提供的是**搜索层的稳定性证据**，而不是最终论文数值。

### 5.2 `official` 失败模式是什么

三条失败 run 具有高度一致性：

- **任务/变体一致**：都发生在 `homo_down / wo_leaf_value`
- **seed 一致性**：`42 / 43 / 44` 全部复现
- **起始分子一致**：`CC#CC(C)(C)C`
- **错误类型一致**：`AttributeError: 'NoneType' object has no attribute 'is_expanded'`

从论文解读角度，这至少说明：

- 去掉 `leaf value` 后，系统不仅在 `pilot` 真值结果上更差；
- 在更正式预算下，还额外暴露出**搜索过程稳定性不足**的问题。

### 5.3 为什么 `pilot` 仍然是主数值依据

因为当前 MCTS 侧真正完整闭环的数据仍然是：

- `smoke`：用于早期趋势一致性；
- `pilot`：用于当前主结论的真值量化；
- `official`：仅提供搜索层补充证据，尚不能替代 `pilot` 真值表。

## 6. 当前阶段建议引用顺序

如果后面继续补实验、补图表，建议优先按下面顺序引用数据：

1. **主判断**：优先用 `pilot` 的 `true_eval_summary_aggregated.tsv`
2. **趋势一致性补充**：引用 `smoke` 两个任务的聚合表
3. **稳定性补充**：引用 `official` 的 `run_summary.tsv` 与失败日志，说明 `wo_leaf_value` 在 `homo_down` 上出现 `3 / 3` 复现失败
4. **最终定稿替换**：只有在后续补齐失败 run 并完成 `official` 真值评估后，才考虑用新的多 seed 真值结果替换 `pilot`

## 7. 一句话总结

当前已经整理出的最有价值资产不是“official 给出了最终主表”，而是：**`pilot` 已经把 `leaf value` 的关键性定量坐实，而 `official` 又进一步暴露出 `wo_leaf_value` 在 `homo_down` 上的复现性稳定性问题。**
