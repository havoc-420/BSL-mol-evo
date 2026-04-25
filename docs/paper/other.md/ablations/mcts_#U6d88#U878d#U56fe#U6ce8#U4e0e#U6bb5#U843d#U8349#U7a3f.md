# MCTS 侧消融图注与段落草稿

## 1. 使用定位

本文档用于整理两类可以直接复用到论文里的素材：

- **图注 / caption 草稿**
- **结果段 / 讨论段草稿**

这样后续无论是写主文、附录还是回 rebuttal，都能直接复制并微调。

## 2. 主文结果段草稿

### 2.1 中文版

我们进一步在 `LUMO(U)` 与 `HOMO(D)` 上对 `OFO-guided MCTS` 的关键搜索组件进行了真值消融。结果显示，`leaf value` 是当前框架中最关键的搜索信号。去掉该项后，`HOMO(D)` 上的平均改善由 `2.0041` 显著下降至 `0.4896`，同时平均运行时间与展开节点数分别上升至完整方法的 `88.4` 倍与 `24.8` 倍。这表明 `leaf value` 不仅影响最终优化质量，也直接决定搜索过程的效率。相比之下，去掉 `prior` 仅带来较小幅度退化，说明其贡献更偏向温和的搜索引导；而将 expansion ranking 随机化则主要带来显著更高的搜索成本，其对最终改善值的影响目前仍表现出一定任务依赖性。

### 2.2 英文版

We further conduct truth-evaluated ablations of key search components in `OFO-guided MCTS` on `LUMO(U)` and `HOMO(D)`. The results show that the leaf-value signal is the most critical component in the current framework. Removing it causes the average improvement on `HOMO(D)` to drop sharply from `2.0041` to `0.4896`, while increasing the average runtime and the number of expanded nodes by `88.4x` and `24.8x`, respectively. This indicates that the leaf-value estimate is essential not only for the final optimization quality but also for search efficiency. By contrast, removing the prior causes only mild degradation, suggesting that its contribution mainly acts as a softer guidance signal. Replacing the expansion ranking with random top-b selection mainly increases search cost, while its influence on the final improvement is currently more task-dependent.

## 3. 讨论段草稿

### 3.1 中文版

这些结果说明，当前框架的有效性并不是由所有模块平均贡献形成的。相反，`leaf value` 构成了最主要的性能支点，而 `prior` 与 expansion ranking 更像是在此基础上进一步改善搜索稳定性与效率的辅助信号。换言之，当前方法的核心优势首先来自“是否能够对叶节点质量进行有效估计”，而不是单纯依赖更激进或更随机的搜索扩展策略。

### 3.2 英文版

These results suggest that the effectiveness of the current framework is not supported equally by all search modules. Instead, the leaf-value signal serves as the primary performance anchor, whereas the prior and the expansion ranking behave more like auxiliary signals that further improve search stability and efficiency. In other words, the core advantage of the current method lies first in whether it can reliably estimate leaf quality, rather than merely relying on more aggressive or more random expansion strategies.

## 4. 图注草稿

### 4.1 主文表格 caption

**中文**：

表 X：`OFO-guided MCTS` 在 `LUMO(U)` 与 `HOMO(D)` 上的关键组件真值消融结果。与完整方法相比，移除 `leaf value` 会显著降低 `HOMO(D)` 上的优化质量，并同时带来数量级更高的搜索成本；移除 `prior` 仅造成较温和的退化。

**English**:

Table X: Truth-evaluated ablations of key `OFO-guided MCTS` components on `LUMO(U)` and `HOMO(D)`. Compared with the full method, removing the leaf-value signal substantially degrades optimization quality on `HOMO(D)` and incurs orders-of-magnitude higher search cost, while removing the prior leads to only mild degradation.

### 4.2 附录表格 caption（含 `random_topb`）

**中文**：

表 Y：MCTS 侧扩展消融结果。除主文中的 `wo_prior` 与 `wo_leaf_value` 外，我们还报告随机 `top-b` expansion ranking 的结果。该变体在当前证据下主要表现为搜索效率下降，而其对最终优化质量的影响仍具有一定任务依赖性。

**English**:

Table Y: Extended MCTS ablation results. In addition to `wo_prior` and `wo_leaf_value`, we also report the variant with random top-b expansion ranking. Under the current evidence, this variant mainly reduces search efficiency, while its impact on final optimization quality remains task-dependent.

## 5. rebuttal / 汇报场景可复用短句

### 5.1 中文短句

- 当前最强的消融证据来自 `wo_leaf_value`，它同时破坏了效果和效率。
- `wo_prior` 的退化方向是一致的，但效应量明显小于 `wo_leaf_value`。
- `random_topb` 更适合作为效率侧消融，而不是最强效果侧消融。
- 因此，当前框架并不是“任意组件都可替换”，而是至少存在一个关键搜索信号不可或缺。

### 5.2 English short sentences

- The strongest ablation evidence currently comes from `wo_leaf_value`, which harms both effectiveness and efficiency.
- The degradation caused by removing the prior is directionally consistent but much smaller than that of removing the leaf-value signal.
- `random_topb` is better interpreted as an efficiency-side ablation rather than the strongest quality-side ablation.
- Therefore, the current framework is not arbitrarily modular; at least one key search signal is indispensable.

## 6. 等 official 完成后的更新位置

等 `official` 真值评估完成后，本文件最需要更新的地方只有两类：

1. 把结果段中的数值改成多 seed 均值；
2. 判断 `wo_prior` 最终是继续保留在主文，还是退到附录。

其余大部分叙述逻辑大概率不需要推翻。

## 7. 一句话总结

当前最值得提前写下来的不是更多原始数字，而是：**围绕 `wo_leaf_value` 这个最强负面对照，把正文结果段、讨论段和 caption 先写成型。**
