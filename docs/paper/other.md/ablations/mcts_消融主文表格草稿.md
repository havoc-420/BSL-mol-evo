# MCTS 侧消融主文表格草稿

## 1. 使用目的

本文档用于提前搭好论文正文里最可能出现的 **主表结构**。当前 `official` 还没跑完，因此这里采用的是：

- **结构先定好**；
- **当前先用 `pilot` 数值占位**；
- **等 `official` 多 seed 真值评估完成后，再把数值替换为最终版。**

## 2. 主文最推荐保留的表格

当前最推荐的正文主表只保留三个变体：

- `full`
- `wo_prior`
- `wo_leaf_value`

原因是：

- `full`：完整方法，作为基线；
- `wo_prior`：对应“policy prior 是否必要”；
- `wo_leaf_value`：对应“leaf value 是否必要”，且是当前最强负面对照；
- `random_topb` 当前更适合作为附录或补充材料，而不是正文核心表格。

## 3. 主表草稿（当前可用 `pilot` 版）

### 3.1 版本 A：正文最紧凑写法

| Task | Variant | Avg. Improvement | Success Rate | Runtime ↓ | Expanded Nodes ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| `HOMO(D)` | `full` | `2.0041` | `1.00` | `2.31` | `7.60` |
| `HOMO(D)` | `wo_prior` | `1.9007` | `1.00` | `1.97` | `5.95` |
| `HOMO(D)` | `wo_leaf_value` | `0.4896` | `0.95` | `204.35` | `188.25` |
| `LUMO(U)` | `full` | `1.1452` | `1.00` | `11.49` | `12.45` |
| `LUMO(U)` | `wo_prior` | `1.0953` | `1.00` | `9.79` | `10.85` |
| `LUMO(U)` | `wo_leaf_value` | `1.2969` | `1.00` | `240.14` | `207.95` |

这张表最适合正文篇幅紧张时使用。它能够直接支撑两个判断：

1. `wo_leaf_value` 在 `HOMO(D)` 上明显退化；
2. 即便在 `LUMO(U)` 上 improvement 未下降，其成本也极端上升，因此整体仍不可接受。

### 3.2 版本 B：正文更强调“相对 full 的变化”

| Task | Variant | Δ Improvement vs. `full` | Runtime Ratio | Node Ratio | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `HOMO(D)` | `wo_prior` | `-0.1034` | `0.85x` | `0.78x` | 轻微退化，但更省 |
| `HOMO(D)` | `wo_leaf_value` | `-1.5145` | `88.4x` | `24.8x` | **显著更差** |
| `LUMO(U)` | `wo_prior` | `-0.0499` | `0.85x` | `0.87x` | 轻微退化，但更省 |
| `LUMO(U)` | `wo_leaf_value` | `+0.1517` | `20.9x` | `16.7x` | improvement 略高，但整体不可接受 |

这张表更适合在正文讨论中突出“为什么 `wo_leaf_value` 是关键负面对照”。

## 4. 推荐 caption 草稿

### 4.1 中文 caption

**表 X：MCTS 侧关键组件消融的真值评估结果。** 我们在 `LUMO(U)` 与 `HOMO(D)` 上比较完整方法、移除 policy prior，以及移除 leaf-value 估计后的搜索表现。结果表明，移除 leaf value 会显著恶化整体性能：在 `HOMO(D)` 上平均改善大幅下降，同时搜索时间与展开节点数急剧上升。相比之下，移除 prior 仅带来较小幅度退化，说明其贡献更偏向温和的搜索引导。

### 4.2 英文 caption

**Table X: Truth-evaluated ablations of key MCTS components.** We compare the full method against variants without the policy prior and without the leaf-value estimate on `LUMO(U)` and `HOMO(D)`. Removing the leaf-value signal leads to the most severe degradation: on `HOMO(D)`, the average improvement drops substantially, while the runtime and the number of expanded nodes increase dramatically. In contrast, removing the prior causes only mild degradation, suggesting that its contribution is weaker and mainly acts as a soft guidance signal.

## 5. 等 `official` 跑完后怎么替换

### 5.1 需要替换的字段

把当前表中的 `pilot` 单 seed 数值替换成：

- `3` 个 seed 的均值；
- 如篇幅允许，再补标准差或 `±`；
- 若需要更规范，也可以把 success rate 写成百分比。

### 5.2 建议替换策略

- **正文主表**：使用 `official` 的多 seed 均值；
- **正文叙述中的量化句子**：同步替换为多 seed 数值；
- **如果 `wo_prior` 最终差异仍然很小**：可以把它保留在主表，但弱化文字力度；
- **如果 `wo_prior` 在 official 中完全持平**：可考虑把它降到附录，而正文只保留 `full` 与 `wo_leaf_value` 的最强对照。

## 6. 主文写作时的推荐强调点

### 6.1 应该强调

- `leaf value` 对**效果 + 效率**双重重要；
- `wo_leaf_value` 在 `HOMO(D)` 上的巨大 effect size；
- `wo_prior` 不是主要性能来源；
- 当前完整框架至少明显优于一个关键消融版本。

### 6.2 不要强调过头

- 不要说“所有消融都同等严重”；
- 不要把 `LUMO(U)` 上 `wo_leaf_value` 的单项 improvement 提升直接写成“更好”；
- 不要在 `official` 完成前把 `pilot` 的单 seed 数值写成终版主表。

## 7. 一句话总结

如果今天就要往主文里落表，最稳的做法是：**先用 `pilot` 把表格结构和论证逻辑搭好，等 `official` 跑完后只替换数值，不改叙述骨架。**
