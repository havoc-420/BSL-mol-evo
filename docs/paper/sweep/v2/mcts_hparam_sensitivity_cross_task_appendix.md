# OFO-guided MCTS 超参数敏感性补充实验（附录长版，v2）

## 1. 章节定位

本稿是 `v2` 版超参数分析的附录长版，目标是把当前已经具备真值评估支撑的结果重新整理成**适合论文落地**的结构。与 `v1` 相比，`v2` 的主要变化是：

- 主文不再只依赖 `LUMO(D)` 单任务；
- 主文级结论优先建立在 `LUMO(U)` 与 `HOMO(D)` 的真值评估 sweep 上；
- `LUMO(D)` 主要作为**扩展单任务案例**保留，用于解释更细的搜索机制。

因此，`v2` 的建议叙事是：

- **正文**：强调三组主 sweep 在跨任务上的总体规律与任务依赖性；
- **附录**：用 `LUMO(D)` 给出更完整的趋势、trade-off 以及默认参数的经验动机。

## 2. 数据来源与口径说明

### 2.1 数据来源

本稿基于以下已汇总 CSV：

- `output/evo-mo/超参数实验/part-1/part1_hparam_summary.csv`
- `output/evo-mo/超参数实验/part-2/part2_hparam_summary.csv`
- `output/evo-mo/超参数实验/all_hparam_summary.csv`

这些记录均来自**已完成真值评估**的 `batch_optimization_*` 目录，而不是仅基于搜索阶段 `json` 的预测值。

### 2.2 当前适合写入论文的实验边界

当前 `v2` 最适合写入论文的结果分为两层：

- **跨任务主文级 sweep**：`LUMO(U)` 与 `HOMO(D)` 上的 `num_simulations`、`exploration_weight`、`max_branching`；
- **单任务附录级扩展**：`LUMO(D)` 上的 `exploration_weight`、`max_depth`、`pruning_patience`，以及有限的 `max_branching` 补充结果。

需要特别说明的是：当前 `v2` 汇总表里，`LUMO(D)` 的预算 sweep 与较完整分支宽度 sweep 并未以同口径真值评估全部补齐，因此不建议在 `v2` 中继续把它们写成跨任务定量主证据。

### 2.3 指标解释

- 对 `direction = increase` 的任务，`average_improvement` **越大越好**；
- 对 `direction = decrease` 的任务，`average_improvement` **越负越好**；
- `improved_percentage` 用作成功率；
- `average_drug_likeness` 反映分子质量；
- `intdiv_avg` 反映内部多样性；
- `morgan_similarity_avg` 越高通常表示与起始分子更接近。

从论文角度，本节最重要的是区分两类问题：

1. **优化强度**是否变强；
2. 这种增强是否伴随**稳定性、多样性或结构保持**的代价。

## 3. 跨任务主文级 sweep

### 3.1 搜索预算：`num_simulations`

#### `LUMO(U)`

| `num_simulations` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200 | 1.1933 | 85.57 | 0.4681 | 0.3745 | 0.1946 |
| 400 | 1.2449 | 85.69 | 0.4746 | 0.3789 | 0.1857 |
| 800 | 1.3681 | 83.13 | 0.4790 | 0.3778 | 0.1526 |

#### `HOMO(D)`

| `num_simulations` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200 | -1.5795 | 94.14 | 0.3876 | 0.5085 | 0.1577 |
| 400 | -1.5593 | 93.26 | 0.3873 | 0.5140 | 0.1661 |
| 800 | -1.5211 | 93.25 | 0.3888 | 0.5519 | 0.1697 |

这组结果首先说明，预算效应并不是简单的“越大越好”。在 `LUMO(U)` 上，增加预算能够持续增强平均改善值，说明该任务对更强搜索更敏感；但在 `HOMO(D)` 上，`200` 已经取得了最强改善值和最高成功率，进一步增加预算主要体现在质量和多样性的轻微改善，而非目标指标继续增强。

因此，正文更稳妥的写法应是：**中高预算是必要的，但额外预算收益具有任务依赖性；`200 ~ 800` 已经构成当前方法的有效工作区间。** 这比“预算越大越好”的口径更符合现有数据。

### 3.2 探索系数：`exploration_weight`

#### `LUMO(U)`

| `exploration_weight` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1.2027 | 85.15 | 0.4718 | 0.3787 | 0.1971 |
| 1.4 | 1.2158 | 88.64 | 0.4727 | 0.4026 | 0.1818 |
| 2.0 | 1.3681 | 83.13 | 0.4790 | 0.3778 | 0.1526 |

#### `HOMO(D)`

| `exploration_weight` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 | -1.5899 | 94.14 | 0.3879 | 0.5056 | 0.1598 |
| 1.4 | -1.5630 | 92.66 | 0.3887 | 0.5153 | 0.1649 |
| 2.0 | -1.5211 | 93.25 | 0.3888 | 0.5519 | 0.1697 |

探索系数是当前 `v2` 中最能体现**任务依赖性**的一组 sweep。在 `LUMO(U)` 上，更强探索能带来更高的平均改善值，表明该任务更受益于搜索对潜在高收益轨迹的主动扩展；但在 `HOMO(D)` 上，`1.0` 反而给出最强改善与最高成功率，说明该任务并不需要过度激进的探索。

因此，正文层面更合理的结论不是“探索越强越好”，而是：**探索系数决定搜索激进程度，其最优点取决于任务；从跨任务角度看，`1.4 ~ 2.0` 更适合作为稳健工作带，而 `1.0` 可视为对部分下降任务更友好的保守设置。**

### 3.3 分支宽度：`max_branching`

#### `LUMO(U)`

| `max_branching` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8  | 1.5173 | 72.21 | 0.4865 | 0.4932 | 0.1299 |
| 20 | 1.3681 | 83.13 | 0.4790 | 0.3778 | 0.1526 |

#### `HOMO(D)`

| `max_branching` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8  | -1.4885 | 90.97 | 0.3937 | 0.5848 | 0.1875 |
| 20 | -1.5211 | 93.25 | 0.3888 | 0.5519 | 0.1697 |

分支宽度并未表现出统一单调趋势。`LUMO(U)` 上，`8` 给出了更强平均改善与更高多样性，但成功率明显下降；`HOMO(D)` 上，`20` 在改善值和成功率上都更优。也就是说，较窄宽度更像是一种**高风险高收益**的设置，而不是普适默认值。

因此，这组结果最适合支持的论文判断是：**分支宽度主要改变搜索的激进程度与稳定覆盖之间的权衡；若正文需要单一默认配置，`20` 比更窄宽度更稳妥。**

## 4. `LUMO(D)` 的附录级单任务证据

### 4.1 探索系数：更强探索显著增强改善，但代价清晰

| `exploration_weight` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | -3.3785 | 89.88 | 0.3893 | 0.4362 | 0.0919 |
| 1.0 | -3.5700 | 89.19 | 0.3902 | 0.4372 | 0.0879 |
| 2.0 | -3.9946 | 93.32 | 0.3895 | 0.2832 | 0.0781 |
| 3.0 | -4.1629 | 93.06 | 0.3824 | 0.3208 | 0.0712 |

`LUMO(D)` 给出的附录级证据非常明确：更强探索可以显著提高优化强度，但同时会压缩多样性并降低结构保持水平。这一现象解释了为何 `v2` 正文不应把探索系数写成全任务统一最优，而应写成**任务驱动的工作区间选择**。

### 4.2 搜索深度：更深路径持续增益，但伴随结构漂移

| `max_depth` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2  | -2.1164 | 93.00 | 0.3896 | 0.7220 | 0.2190 |
| 4  | -2.8944 | 91.51 | 0.3862 | 0.6937 | 0.1314 |
| 6  | -3.2780 | 92.37 | 0.3891 | 0.6615 | 0.1026 |
| 8  | -3.4683 | 89.50 | 0.3897 | 0.6038 | 0.0949 |
| 10 | -3.5076 | 90.64 | 0.3918 | 0.5164 | 0.0920 |
| 12 | -3.5700 | 89.19 | 0.3902 | 0.4372 | 0.0879 |

这组结果几乎是 `LUMO(D)` 中最规整的一组：随着深度增加，平均改善值持续增强，而 `IntDiv` 与 `Morgan` 相似度持续下降。这说明更深路径确实能释放多步编辑的累计收益，但代价是结果更远离起始结构、搜索也更集中于少数高收益轨迹。

从论文角度，更稳妥的写法是：**更深搜索整体有利于提升改善值，但 `10` 附近已是较好的折中点，继续增加深度的收益开始趋缓。**

### 4.3 剪枝耐心值：次级敏感因素

| `pruning_patience` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | -3.2158 | 91.30 | 0.3965 | 0.5100 | 0.1001 |
| 2 | -3.4967 | 91.01 | 0.3934 | 0.4390 | 0.0874 |
| 3 | -3.5700 | 89.19 | 0.3902 | 0.4372 | 0.0879 |
| 4 | -3.6167 | 91.70 | 0.3903 | 0.4363 | 0.0859 |

与前几组 sweep 相比，`pruning_patience` 的影响明显更温和。更大的耐心值略有助于提升改善值，但各设置之间差距有限，且 `1` 在 `QED` 和结构保持上反而更好。因此，`pruning_patience` 更适合作为**稳定性调节项**，而不是决定性性能来源。

### 4.4 `LUMO(D)` 的分支宽度补充结果

| `max_branching` | 平均改善值 | 成功率 (%) | QED | IntDiv | Morgan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 | -3.5700 | 89.19 | 0.3902 | 0.4372 | 0.0879 |
| 30 | -3.5613 | 90.23 | 0.3941 | 0.4291 | 0.0868 |

当前 `v2` 中 `LUMO(D)` 的真值评估分支宽度结果只足以支持一个较弱但有用的结论：**从 `20` 扩大到 `30` 并未带来明确性能收益。** 因此，它更适合在附录里作为补充证据，而不应在正文中承担跨任务核心论点。

## 5. 综合讨论

综合 `v2` 当前全部真值评估结果，可以把超参数结论概括为以下三层：

- **第一层：方法稳定性。** `OFO-guided MCTS` 并不依赖某个极端脆弱的参数点；在当前实验覆盖的范围内，方法整体可用且趋势可解释。
- **第二层：任务依赖性。** 不同任务对搜索强度的偏好不同，因此正文更应强调“稳健工作区间”，而不是“单一全局最优配置”。
- **第三层：结构化 trade-off。** 更高预算、更强探索和更深路径往往意味着更强改善，但也更可能带来成功率波动、多样性收缩或更大的结构偏移。

这意味着论文中的超参数部分最适合写成：**方法对搜索参数不脆弱，但最优工作点与任务目标有关；因此我们选择一组跨任务平衡型默认设置，而不是追逐单项指标上的极端最优。**

## 6. `v2` 支持的推荐配置

### 6.1 跨任务平衡型默认配置

建议正文优先采用：

- `num_simulations = 800`
- `exploration_weight = 1.4`
- `max_branching = 20`
- `max_depth = 10`
- `pruning_patience = 3`

### 6.2 性能优先型备选配置

如果更希望强调“更强搜索强度”，可在附录或答辩材料中补充：

- `num_simulations = 800`
- `exploration_weight = 2.0`
- `max_branching = 20`
- `max_depth = 10`
- `pruning_patience = 3`

其区别在于：`1.4` 更偏稳健与成功率平衡，`2.0` 更偏向对部分任务追求更强性质改善。

## 7. 可直接写入论文的附录段落

We further revisited the hyperparameter sensitivity of OFO-guided MCTS using only runs with completed true evaluation. The updated results suggest that the method is robust within a reasonable hyperparameter range, while the best operating point remains task-dependent rather than globally fixed. On `LUMO(U)`, increasing the search budget from `200` to `800` improves the average gain from `1.1933` to `1.3681`, indicating that the upward optimization task benefits from stronger search. In contrast, on `HOMO(D)`, the same increase does not further improve the objective, suggesting that this task reaches a useful operating regime at a lower budget. A similar pattern is observed for the exploration coefficient: stronger exploration is beneficial for `LUMO(U)` in terms of optimization strength, whereas `HOMO(D)` prefers a smaller coefficient for both improvement and success rate. The branching-width sweep further shows that narrower search can be more aggressive but less stable, while a moderate width provides a safer default across tasks. Additional `LUMO(D)` results confirm that deeper search improves the optimization objective at the cost of reduced diversity and structural similarity, whereas pruning patience has a comparatively mild effect. Overall, these observations support using a balanced default configuration and describing the hyperparameter findings in terms of stable working ranges rather than a single universal optimum.
