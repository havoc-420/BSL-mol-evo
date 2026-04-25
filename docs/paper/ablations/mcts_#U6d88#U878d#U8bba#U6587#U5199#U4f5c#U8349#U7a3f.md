# MCTS 侧消融论文写作草稿

## 1. 本文档的角色

本文档不是简单记实验，而是从“论文落地写作”的角度，回答下面几个问题：

- 主文里最值得讲什么？
- 哪些结论现在就能写？
- 哪些内容更适合放附录？
- 如果后续要形成最终多 seed 真值主表，还差哪一步？

当前状态需要先讲清楚：

- `smoke` 与 `pilot` 已完成真值评估；
- 本轮 `official` 已结束，但只完成了搜索，没有接真值评估；
- 同时 `homo_down / wo_leaf_value` 在 `42 / 43 / 44` 三个 seed 上全部 `search_failed`。

因此，现在论文里最稳的做法是：**以 `pilot` 真值结果作为主数值依据，以 `official` 搜索层失败作为稳定性补充证据。**

## 2. 当前最适合写进主文的论点

### 2.1 主论点 A：`leaf value` 是关键组件

这是当前最成熟、最有说服力的主文论点。

原因是：

- 它同时影响**效果**和**效率**；
- 在 `HOMO(D)` 上表现出非常大的 effect size；
- 这一趋势已经被 `smoke` 和 `pilot` 两轮真值结果共同支持；
- 本轮 `official` 搜索层又进一步显示，`homo_down / wo_leaf_value` 会出现 `3 / 3` 的复现性失败。

推荐主文句式：

> To understand which components are essential in the proposed search framework, we performed truth-evaluated ablations on the key MCTS modules. Among them, removing the leaf-value estimate leads to the most severe degradation. On `HOMO(D)`, the average improvement drops from `2.0041` to `0.4896`, while the average runtime and expanded nodes increase by `88.4x` and `24.8x`, respectively. These results indicate that the leaf-value signal is not merely an auxiliary heuristic, but a key component for both search quality and search efficiency. In an additional official-budget search-only run, the `homo_down / wo_leaf_value` setting further fails reproducibly on all three seeds, suggesting an extra stability risk after removing the leaf-value signal.

### 2.2 主论点 B：`prior` 的贡献是温和但一致的

这个论点可以写，但不宜写得过重。

更合适的角度是：

- `prior` 有帮助；
- 但它不是最主要的性能来源；
- 它更像提升搜索起步质量与稳定性的辅助信号。

推荐主文句式：

> In contrast, removing the policy prior only causes mild degradation across the current tasks, suggesting that the prior mainly provides a softer guidance signal, whereas the leaf-value estimate contributes more directly to the final search effectiveness.

### 2.3 主论点 C：expansion ranking 的贡献目前更偏效率侧

对于 `random_topb`，更稳妥的写法不是“它显著降低最终效果”，而是：

- 随机化 ranking 会让搜索变慢、展开更多节点；
- 对最终改善值的影响方向在不同任务 / 不同规模下不完全一致。

推荐主文句式：

> Replacing the OFO-guided expansion ranking with random top-b selection consistently increases search cost, although its impact on the final improvement is currently less stable across tasks. This suggests that the ranking signal is particularly important for search efficiency, while its quality contribution may be more task-dependent.

## 3. 当前适合放附录的内容

### 3.1 `random_topb` 的全量表格

因为它现在的证据强度不如 `wo_leaf_value`，更适合作为：

- 主文简要提及；
- 附录放完整数值与更多解释。

### 3.2 smoke 结果

`smoke` 的价值主要是：

- 展示趋势不是 pilot 才突然出现；
- 说明我们在放大样本数前就已经观察到相同方向。

但由于样本数仅 `5`，更适合作为附录或内部备稿，而不是主文主表。

### 3.3 `official` 的失败日志与状态表

本轮 `official` 虽未给出真值终表，但它的 `run_summary.tsv` 与失败日志很适合作为：

- rebuttal 时的稳定性补充证据；
- 附录中对 `wo_leaf_value` 风险的补充说明；
- 内部汇报时解释“为什么目前主表仍使用 pilot 真值数值”。

## 4. 主文与附录的建议分工

### 4.1 主文

主文建议只保留最重要的矩阵：

- tasks：`LUMO(U)`、`HOMO(D)`
- variants：`full`、`wo_prior`、`wo_leaf_value`
- 指标：`paper_avg_improvement`、`paper_success_rate`、`avg_runtime`、`avg_expanded_nodes`

这里的逻辑是：

- `full`：完整方法
- `wo_prior`：回答“先验是否必要”
- `wo_leaf_value`：回答“价值估计是否必要”

这三行已经足够支撑主文关于 MCTS 侧必要性的核心论证。

### 4.2 附录

附录建议放：

- `random_topb`
- smoke 全部表格
- 本轮 `official` 的 `run_summary.tsv` 与 `wo_leaf_value` 失败说明
- 若后续补齐 official 真值评估，再决定是否追加更细粒度的多 seed 表格

## 5. 现在就可以写的中文段落

### 5.1 主文结果段（中文）

为验证 `OFO-guided MCTS` 中各搜索信号的必要性，我们在 `LUMO(U)` 与 `HOMO(D)` 两个主任务上进行了真值消融。结果表明，`leaf value` 是当前框架中最关键的组成部分。以 `HOMO(D)` 为例，移除 `leaf value` 后，平均改善从 `2.0041` 大幅下降至 `0.4896`，同时平均运行时间与展开节点数分别增至完整模型的 `88.4` 倍和 `24.8` 倍。这说明 `leaf value` 不仅影响最终优化质量，而且直接决定搜索过程的效率。相比之下，移除 `prior` 仅带来较温和的退化，表明其作用更偏向辅助性的搜索引导；而将 expansion ranking 随机化则主要导致搜索成本上升，其对最终改善值的影响目前仍表现出一定任务依赖性。进一步地，在 `official` 预算的搜索层实验中，`homo_down / wo_leaf_value` 还出现了 `3 / 3` seeds 的复现失败，说明移除 `leaf value` 还会带来额外的稳定性风险。

### 5.2 讨论段（中文）

这些结果说明，当前框架的有效性并不是由所有搜索模块平均贡献形成的。相反，`leaf value` 构成了最主要的性能支点，而 `prior` 与 expansion ranking 更像是在此基础上进一步改善搜索稳定性与效率的辅助信号。换言之，当前方法的核心优势首先来自“是否能够对叶节点质量进行有效估计”，而不是单纯依赖更激进或更随机的搜索扩展策略。

## 6. 如果后续要形成最终主表，还需要补什么

当前离“official 多 seed 真值终表”还差三步：

1. 修复 `homo_down / wo_leaf_value` 在 `CC#CC(C)(C)C` 上触发的 `NoneType.is_expanded` 错误；
2. 补跑失败的 `3` 个 `official` run units；
3. 对完整矩阵重新执行真值评估，生成新的 `true_eval_summary.tsv` 与 `true_eval_summary_aggregated.tsv`。

只有补齐这三步之后，才适合把当前正文中的 `pilot` 数值替换为新的 official 多 seed 真值数值。

## 7. 当前应避免的写法

- 不要写“所有消融都显著变差”；
- 不要写“random top-b 一定会显著破坏最终优化质量”；
- 不要写“wo_prior 与 full 的差异已经被完全坐实”；
- 不要把本轮 `official` 写成“已完成的多 seed 真值主表”；
- 不要忽略 `wo_leaf_value` 在 `official / homo_down` 上出现的 `3 / 3` 复现失败。

## 8. 当前最推荐的叙述策略

如果现在就开始组织 paper，我建议采用下面的顺序：

1. 先在主文中把 `leaf value` 作为最关键消融结论写清楚；
2. 再把 `prior` 写成“有帮助但贡献较温和”；
3. 把 `random_topb` 作为效率侧补充放进附录或次要段落；
4. 用本轮 `official` 的失败信息补上一句“稳定性风险”；
5. 只有在后续补齐 official 真值评估后，再把正文数值替换成多 seed 终版。

## 9. 一句话总结

从写论文的角度看，当前最值得抓住的不是“把所有消融都展开”，而是：**围绕 `leaf value` 这个最强负面对照，把效果、效率和稳定性三条证据链一起写稳；至于最终多 seed 真值主表，仍需后续补齐。**
