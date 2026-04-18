# 消融实验整理目录

## 1. 目录定位

本目录用于集中整理当前论文相关的**消融实验数据、阶段性结论与写作素材**。

当前优先整理的是 **MCTS 侧消融**。目前 `smoke` 与 `pilot` 已完成真值评估；`official` 这一轮也已结束，但只完成了搜索层，且在 `homo_down / wo_leaf_value` 上出现了 `3 / 3` 的系统性失败，因此最终论文数值仍应以 `pilot` 的真值结果为主。

如果后续 OFO 侧消融（如 `abs_target`、`wo_opfeat`、`gcn2d`）开始推进，也建议继续沿用本目录结构补充新文档。

## 2. 文件说明

- `mcts_消融数据总览.md`
  - 面向"查数据 / 查产物 / 查当前进度"。
  - 汇总 `smoke`、`pilot`、`official` 三个层级的实验矩阵、输出路径、完成状态与当前可用证据。

- `mcts_消融结果解读.md`
  - 面向"看结果 / 下判断"。
  - 以 `full` 为基线，整理 `wo_prior`、`wo_leaf_value`、`random_topb` 在 `LUMO(U)` 与 `HOMO(D)` 上的关键数值、差值和解释，并把 `official` 搜索层的稳定性信息纳入解读。

- `mcts_消融论文写作草稿.md`
  - 面向"写论文"。
  - 提供主文可用论点、推荐表述、应避免的过度结论，以及正文 / 附录如何分工的建议。

- `mcts_消融主文表格草稿.md`
  - 面向"搭主表"。
  - 当前仍以 `pilot` 真值数值固定正文主表结构；只有在后续补齐 `official` 的失败 run 并完成真值评估后，才适合替换成多 seed 终版。

- `mcts_消融图注与段落草稿.md`
  - 面向"直接落文"。
  - 整理主文 / 附录可复用的 caption、结果段、讨论段与汇报短句。

- `主实验BFS-vs-MCTS参考数据.md`
  - 面向"跨层级对比"。
  - 从主实验表 5-2（`docs/paper/data/full.txt`）提取 OFO-BFS 与 OFO-MCTS 在 `LUMO(U)` / `HOMO(D)` 上的关键指标，用于在消融表中提供 BFS 参考行，支撑"去掉 leaf value 后 MCTS 甚至不如 BFS"的叙事。

## 3. 当前总判断（截至 2026-04-15 13:04）

- **已经完成真值评估并可用于写作的部分**：
  - `smoke`：`LUMO(U)`、`HOMO(D)`，4 个变体
  - `pilot`：`LUMO(U)`、`HOMO(D)`，4 个变体

- **`official` 这轮的实际结果**：
  - 搜索层已结束，共 `18` 个 run units，其中 **`15` 个 success，`3` 个 search_failed**；
  - 失败项全部集中在 `homo_down / wo_leaf_value / seed42,43,44`；
  - 本轮未生成 `true_eval_summary.tsv` 与 `true_eval_summary_aggregated.tsv`，因为启动参数为 `RUN_EVAL=0`，且失败 run 也阻断了完整终表的形成。

- **当前最稳的结论**：
  - `wo_leaf_value` 已经可以明确判定为**显著更差**，并且在 `official / homo_down` 上暴露出**稳定性问题**；
  - `wo_prior` 当前仅表现为**轻微退化或近似持平**，但没有出现系统性失败；
  - `random_topb` 当前更像是**效率明显更差、效果收益不稳定**的变体。

## 4. 建议使用方式

如果你现在是从论文视角继续推进，建议按下面顺序使用这些文档：

1. 先看 `mcts_消融数据总览.md`，确认哪些数据已经可以引用；
2. 再看 `mcts_消融结果解读.md`，判断哪些结论已经足够稳定；
3. 然后使用 `mcts_消融主文表格草稿.md`，快速确定正文主表结构；
4. 再从 `mcts_消融图注与段落草稿.md` 中抽取 caption 和结果段；
5. 最后用 `mcts_消融论文写作草稿.md` 统一整理正文 / 附录 / rebuttal 的写法。
