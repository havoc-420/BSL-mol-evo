# Part 2 — MCTS 内部消融：OFO 组件贡献

## 1. 目标

拆解 MCTS 内部各 OFO 组件的贡献，支撑论文机制层面的论述：
1. `leaf value` 是最关键的搜索信号（效果 + 效率 + 稳定性三重证据）
2. `prior` 提供温和正向引导但非决定性
3. `expansion ranking` 主要影响效率

## 2. 数据来源

- **效果与效率数据**：pilot 真值评估（20 起始分子，200 simulations）
- **稳定性数据**：official 搜索层（50 起始分子，400 simulations，3 seeds）
- **BFS 参考线**：主实验表 5-2（仅作为参考虚线，不作为正式消融条目）

注意：pilot 与 main 实验口径不同，BFS 仅作方向性参考。

## 2.5 变体含义说明

图中四种变体分别对应 MCTS 内部 OFO 组件的不同消融配置：

| 变体 | 含义 | 移除的组件 |
| --- | --- | --- |
| **full** | 完整 OFO-MCTS，所有组件均启用（基线） | 无 |
| **w/o prior** | 移除先验（prior）引导 | 移除 MCTS 选择阶段的先验概率（`P(s,a)`），即不使用策略网络为子节点提供先验偏好，搜索完全依赖价值信号 |
| **w/o leaf value** | 移除叶节点价值（leaf value）评估 | 移除 MCTS 叶节点的价值估计（`V(s)`），即不对未展开节点做价值预测，搜索失去最重要的评估信号 |
| **random top-k** | 将扩展排序（expansion ranking）随机化 | 保留其他所有组件，但将 OFO 的扩展候选排序替换为随机排序，即不使用优化目标引导的片段排序来选择 top-k 候选进行扩展 |

简而言之：
- **full**：完整系统；
- **w/o prior**：测试"策略先验"的贡献——搜索不靠先验指引方向，仅靠回传价值；
- **w/o leaf value**：测试"叶节点价值估计"的贡献——搜索无法在展开前评估节点质量，必须大量展开才能获取信号；
- **random top-k**：测试"OFO 扩展排序"的贡献——不按优化目标排序候选片段，而是随机选取 top-k 进行扩展。

## 3. 图形设计

### 双面板（统一 variant 颜色编码）

- **Panel A — Pilot Truth Improvement（效果）**
  - 数据源：pilot `paper_avg_improvement_mean`
  - 类型：grouped bar（x=task, hue=variant）
  - 变体：full / w/o prior / w/o leaf value / random top-b
  - BFS 参考线：浅灰虚线标注 HOMO(D)=0.9125 / LUMO(U)=0.4726
  - 作用：直接展示 `wo_leaf_value` 在 HOMO(D) 上的崩溃性退化

- **Panel B — Efficiency: Runtime & Expanded Nodes Fold vs Full（效率）**
  - 数据源：pilot `avg_runtime_mean` 和 `avg_expanded_nodes_mean` 相对 full 的倍率
  - 类型：grouped bar，log scale Y 轴
  - 变体同上
  - 作用：展示 `wo_leaf_value` 和 `random_topb` 的搜索成本数量级膨胀
  - 设计：log scale 避免 88x / 20x 压扁其他柱子

### 设计要点
- 双面板共享 variant 颜色映射，视觉一致性
- Panel A 中 BFS 参考线不占据正式 bar 位，仅作虚线标注
- Panel B 使用 log scale，数值标注在柱顶
- 稳定性证据（official 搜索层 0/3 seed 失败）暂不在图中展示，留待正文文字说明

## 4. 关键数据

### Panel A — Improvement

| Task | Variant | Avg. Improvement | Success Rate |
| --- | --- | ---: | ---: |
| HOMO(D) | full | 2.0041 | 1.00 |
| HOMO(D) | w/o prior | 1.9007 | 1.00 |
| HOMO(D) | w/o leaf value | 0.4896 | 0.95 |
| HOMO(D) | random top-b | 1.6479 | 0.95 |
| LUMO(U) | full | 1.1452 | 1.00 |
| LUMO(U) | w/o prior | 1.0953 | 1.00 |
| LUMO(U) | w/o leaf value | 1.2969 | 1.00 |
| LUMO(U) | random top-b | 1.1130 | 1.00 |

### Panel B — Efficiency Fold vs Full

| Task | Variant | Runtime Fold | Nodes Fold |
| --- | --- | ---: | ---: |
| HOMO(D) | w/o prior | 0.85x | 0.78x |
| HOMO(D) | w/o leaf value | **88.4x** | **24.8x** |
| HOMO(D) | random top-b | 1.57x | 1.16x |
| LUMO(U) | w/o prior | 0.85x | 0.87x |
| LUMO(U) | w/o leaf value | **20.9x** | **16.7x** |
| LUMO(U) | random top-b | 3.00x | 2.43x |

### Panel C — Official Stability

| Task | Variant | Seeds Success |
| --- | --- | ---: |
| HOMO(D) | full | 3/3 |
| HOMO(D) | w/o prior | 3/3 |
| HOMO(D) | w/o leaf value | **0/3** |
| HOMO(D) | random top-b | N/A |
| LUMO(U) | full | 3/3 |
| LUMO(U) | w/o prior | 3/3 |
| LUMO(U) | w/o leaf value | 3/3 |
| LUMO(U) | random top-k | N/A |

## 5. 输出物

- `part2_mcts_ablation.png`
- `part2_mcts_ablation.pdf`
- `part2_data_improvement.csv`
- `part2_data_efficiency.csv`
- `part2_data_stability.csv`（备用，当前未入图）
- `make_part2_mcts_ablation.py`

## 6. 图注建议

> **Figure X.** Ablation of OFO-MCTS components across optimization quality and search efficiency. (A) Truth-evaluated pilot improvement; the dashed gray line marks the OFO-BFS reference level from the main experiment. Removing the leaf-value signal causes catastrophic degradation on HOMO(D). (B) Runtime and expanded-node folds relative to the full configuration (log scale). The `w/o leaf value` variant incurs 20–88× higher cost.
