# MCTS 侧消融数据总览

## 1. 目的

本文档用于从“实验资产管理”的角度整理当前已经产生的 MCTS 侧消融数据，回答三个问题：

1. **已经跑了什么？**
2. **哪些结果已经能用于论文？**
3. **哪些结果还在路上？**

## 2. 当前实验矩阵

### 2.1 已完成真值评估的实验

| 层级 | 任务 | 变体 | seed | 起始分子数 | 真值评估 | 输出根目录 |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke_lumo` | `lumo_up` | `full / wo_prior / wo_leaf_value / random_topb` | `42` | `5` | 已完成 | `mol_evo/output/paper/ablations/20260413_112535/` |
| `smoke_homo` | `homo_down` | `full / wo_prior / wo_leaf_value / random_topb` | `42` | `5` | 已完成 | `mol_evo/output/paper/ablations/20260413_113451/` |
| `pilot` | `lumo_up / homo_down` | `full / wo_prior / wo_leaf_value / random_topb` | `42` | `20` | 已完成 | `mol_evo/output/paper/ablations/20260413_1536_pilot_mcts/` |

### 2.2 正在运行的实验

| 层级 | 任务 | 变体 | seeds | 起始分子数 | 当前状态 | 输出根目录 |
| --- | --- | --- | --- | --- | --- | --- |
| `official` | `lumo_up / homo_down` | `full / wo_prior / wo_leaf_value` | `42 / 43 / 44` | `50` | 搜索进行中（截至 2026-04-14 13:26，`6 / 18` run units 完成） | `mol_evo/output/paper/ablations/20260414_0115_official_mcts/` |

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
  - 当前 official 搜索进度表
- 当前尚未生成：
  - `true_eval_summary.tsv`
  - `true_eval_summary_aggregated.tsv`
  - 原因：真值评估尚未开始

## 4. 当前可直接引用的结果范围

### 4.1 可以直接用于论文内部讨论的

- `pilot` 的 `true_eval_summary_aggregated.tsv`
- `pilot` 的 `true_eval_summary.tsv`
- `smoke` 的两个 `true_eval_summary_aggregated.tsv`

这些结果已经足够支持：

- `wo_leaf_value` 是明显负面对照；
- `wo_prior` 与 `full` 接近；
- `random_topb` 效率显著更差，效果优势不稳定。

### 4.2 还不建议当作最终主表定稿的

- `official` 的当前搜索结果

原因是：

- 目前只有搜索完成状态，没有真值评估汇总；
- 还没有跨 `3` 个 seed 的聚合均值和标准差；
- 现阶段更适合作为“进度信息”，而不是“论文最终数值”。

## 5. 当前进度快照（截至 2026-04-14 13:26）

### 5.1 official 已完成部分

根据 `run_summary.tsv`，当前已完成的 `6` 个 run units 都来自：

- `lumo_up / full / seed42,43,44`
- `lumo_up / wo_prior / seed42,43,44`

也就是说：

- `lumo_up` 的前两个变体搜索已经完成；
- `wo_leaf_value` 以及全部 `homo_down` 还在继续；
- 当前还没有任何 official 真值聚合表可供论文引用。

### 5.2 为什么现在仍然值得整理文档

因为当前 MCTS 侧已经具备了**写作前整理**所需的最小证据闭环：

- `smoke` 给出早期趋势；
- `pilot` 给出更可靠的真值结论；
- `official` 已经明确了最终主文矩阵，不再是开放式探索。

这意味着，虽然最终主表还不能定稿，但**论文叙述框架已经可以先写出来**。

## 6. 当前阶段建议引用顺序

如果后面继续补实验、补图表，建议优先按下面顺序引用数据：

1. **主判断**：优先用 `pilot` 的 `true_eval_summary_aggregated.tsv`
2. **趋势一致性补充**：引用 `smoke` 两个任务的聚合表
3. **最终定稿替换**：等 `official` 真值评估完成后，用 `official` 的多 seed 聚合值替换 `pilot` 中用于主表的数值

## 7. 一句话总结

当前已经整理出的最有价值资产不是“所有实验都跑完了”，而是：**MCTS 侧已经有足够清晰的 `pilot` 真值证据，可以先把论文里的论证框架和写作素材搭起来。**
