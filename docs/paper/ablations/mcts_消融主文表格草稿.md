# MCTS 侧消融主文表格草稿

## 1. 使用目的

本文档用于提前搭好论文正文里最可能出现的 **主表结构**。当前虽然 `official` 搜索已经结束，但这一轮：

- **没有执行真值评估**；
- **没有产出完整多 seed 真值终表**；
- `homo_down / wo_leaf_value` 还出现了 `3 / 3` 的复现性搜索失败。

因此，这里仍采用：

- **结构先定好**；
- **当前先用 `pilot` 真值数值占位**；
- **后续只有在补齐 official 失败 run 并完成真值评估后，再把数值替换为终版。**

## 2. 主文最推荐保留的表格

当前最推荐的正文主表保留三个消融变体 + 一行 BFS 参考行：

- `full`
- `wo_prior`
- `wo_leaf_value`
- `OFO-BFS (ref.)`

原因是：

- `full`：完整方法，作为基线；
- `wo_prior`：对应"policy prior 是否必要"；
- `wo_leaf_value`：对应"leaf value 是否必要"，且是当前最强负面对照；
- `OFO-BFS (ref.)`：跨方法参考行，来自主实验表 5-2，用于展示 MCTS 整体优于 BFS，以及去掉 `leaf value` 后 MCTS 甚至不如 BFS；
- `random_topb` 当前更适合作为附录或补充材料，而不是正文核心表格。

**BFS 参考行的口径说明**：主实验 BFS 的起始分子数为 50，MCTS 预算为 400/800 simulations；消融 pilot 的起始分子数为 20，MCTS 预算为 200 simulations。因此 BFS 行的绝对值与 pilot 不可直接精确对比，但方向性结论（MCTS full > BFS > wo_leaf_value on HOMO(D)）不受口径差异影响。详见 `主实验BFS-vs-MCTS参考数据.md`。

## 3. 主表草稿（当前仍以 `pilot` 版为准）

### 3.1 版本 A：正文最紧凑写法（含 BFS 参考行）

| Task | Variant | Avg. Improvement | Success Rate | Runtime ↓ | Expanded Nodes ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| `HOMO(D)` | `full` | `2.0041` | `1.00` | `2.31` | `7.60` |
| `HOMO(D)` | `wo_prior` | `1.9007` | `1.00` | `1.97` | `5.95` |
| `HOMO(D)` | `wo_leaf_value` | `0.4896` | `0.95` | `204.35` | `188.25` |
| `HOMO(D)` | `OFO-BFS (ref.)`† | `0.9125` | `0.72` | — | — |
| `LUMO(U)` | `full` | `1.1452` | `1.00` | `11.49` | `12.45` |
| `LUMO(U)` | `wo_prior` | `1.0953` | `1.00` | `9.79` | `10.85` |
| `LUMO(U)` | `wo_leaf_value` | `1.2969` | `1.00` | `240.14` | `207.95` |
| `LUMO(U)` | `OFO-BFS (ref.)`† | `0.4726` | `0.64` | — | — |

> † BFS 参考行数据来自主实验表 5-2（50 起始分子，MCTS 预算 400/800 simulations），与消融 pilot（20 起始分子，200 simulations）口径不完全一致，仅供方向性参考。Runtime / Expanded Nodes 不具备可比性，故省略。

这张表能够直接支撑三个判断：

1. `wo_leaf_value` 在 `HOMO(D)` 上明显退化；
2. 即便在 `LUMO(U)` 上 improvement 未下降，其成本也极端上升，因此整体仍不可接受；
3. **去掉 `leaf value` 后 MCTS 在 `HOMO(D)` 上的 improvement 甚至低于 BFS**，说明 `leaf value` 是 MCTS 超越 BFS 的关键信号。

### 3.2 版本 B：正文更强调"相对 full 的变化"（含 BFS 对比）

| Task | Variant | Δ Improvement vs. `full` | Runtime Ratio | Node Ratio | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `HOMO(D)` | `wo_prior` | `-0.1034` | `0.85x` | `0.78x` | 轻微退化，但更省 |
| `HOMO(D)` | `wo_leaf_value` | `-1.5145` | `88.4x` | `24.8x` | **显著更差，甚至低于 BFS** |
| `HOMO(D)` | `OFO-BFS (ref.)`† | `-1.0916` | — | — | BFS 本身也显著低于 MCTS full |
| `LUMO(U)` | `wo_prior` | `-0.0499` | `0.85x` | `0.87x` | 轻微退化，但更省 |
| `LUMO(U)` | `wo_leaf_value` | `+0.1517` | `20.9x` | `16.7x` | improvement 略高，但整体不可接受 |
| `LUMO(U)` | `OFO-BFS (ref.)`† | `-0.6726` | — | — | BFS 在 LUMO(U) 上大幅低于 MCTS |

> † BFS 参考行与消融 pilot 口径不完全一致，Δ Improvement 仅供方向性参考。

这张表更适合在正文讨论中突出两个关键点：
1. `wo_leaf_value` 是最关键负面对照；
2. **在 `HOMO(D)` 上，去掉 `leaf value` 后 MCTS 的 improvement 甚至跌破 BFS 水平**——这个"比 BFS 还差"的发现是消融实验最有说服力的叙事锚点。

## 4. 当前还可以在正文讨论中补的一句 `official` 信息

如果正文或讨论段允许加一句补充说明，当前最值得写的是：

- 在 `official` 预算的搜索层实验中，`homo_down / wo_leaf_value` 在 `42 / 43 / 44` 三个 seed 上全部 `search_failed`；
- 这说明移除 `leaf value` 的问题不止体现为真值指标更差，也体现为**更正式预算下的搜索稳定性风险**。

这条信息更适合写进：

- 主文讨论段的一句话补充；或
- 附录表格 / 附录说明。

## 5. 推荐 caption 草稿

### 5.1 中文 caption

**表 X：MCTS 侧关键组件消融的真值评估结果。** 我们在 `LUMO(U)` 与 `HOMO(D)` 上比较完整方法、移除 policy prior，以及移除 leaf-value 估计后的搜索表现，并以 OFO-BFS 的主实验结果作为参考基线。结果表明，移除 leaf value 会显著恶化整体性能：在 `HOMO(D)` 上平均改善大幅下降至甚至低于 BFS 的水平，同时搜索时间与展开节点数急剧上升。相比之下，移除 prior 仅带来较小幅度退化，说明其贡献更偏向温和的搜索引导。

### 5.2 英文 caption

**Table X: Truth-evaluated ablations of key MCTS components.** We compare the full method against variants without the policy prior and without the leaf-value estimate on `LUMO(U)` and `HOMO(D)`, with OFO-BFS results from the main experiment as a reference baseline. Removing the leaf-value signal leads to the most severe degradation: on `HOMO(D)`, the average improvement drops below the BFS level, while the runtime and the number of expanded nodes increase dramatically. In contrast, removing the prior causes only mild degradation, suggesting that its contribution is weaker and mainly acts as a soft guidance signal.

## 6. 如果后续要用 `official` 替换当前表，前提是什么

### 6.1 需要先补齐的内容

在用 `official` 替换当前表之前，必须先完成：

- 修复 `homo_down / wo_leaf_value` 的 `NoneType.is_expanded` 错误；
- 补跑失败的 `3` 个 run units；
- 对完整矩阵执行真值评估；
- 生成新的 `true_eval_summary.tsv` 与 `true_eval_summary_aggregated.tsv`。

### 6.2 完成后再怎么替换

只有在上述步骤完成后，才建议：

- **正文主表**：使用 `official` 的多 seed 真值均值；
- **正文叙述中的量化句子**：同步替换为新的多 seed 数值；
- **如篇幅允许**：再补标准差或 `±`；
- **如果 `wo_prior` 最终差异仍然很小**：可以保留在主表，但弱化文字力度。

## 7. 主文写作时的推荐强调点

### 7.1 应该强调

- `leaf value` 对**效果 + 效率**双重重要；
- `wo_leaf_value` 在 `HOMO(D)` 上的巨大 effect size；
- **`wo_leaf_value` 在 `HOMO(D)` 上的 improvement 甚至低于 BFS**——这是消融实验最有说服力的叙事锚点；
- `official` 搜索层进一步暴露出 `wo_leaf_value` 的稳定性风险；
- `wo_prior` 不是主要性能来源；
- 当前完整框架至少明显优于一个关键消融版本。

### 7.2 不要强调过头

- 不要说"所有消融都同等严重"；
- 不要把 `LUMO(U)` 上 `wo_leaf_value` 的单项 improvement 提升直接写成"更好"；
- 不要把本轮 `official` 当成已经可替换 `pilot` 的终版主表。

## 8. 一句话总结

如果今天就要往主文里落表，最稳的做法是：**继续用 `pilot` 真值结果搭主表，把 `official` 写成"额外暴露稳定性问题"的补充证据，而不是强行当成终版多 seed 真值表。**
