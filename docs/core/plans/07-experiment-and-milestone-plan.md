# 07：实验设计与里程碑计划

> 目标：为整条新路线提供统一实验问题、对比设置、验收口径和阶段里程碑，避免各模块各自推进但难以形成最终结论。

---

## 1. 研究主问题

建议统一收敛到以下几个主问题：

### Q1：动作空间升级是否真的有效？

比较：

- primitive actions
- fragment actions
- template-enhanced actions

### Q2：`OFO-frag` 是否优于当前 OFO？

比较：

- 旧 `edge_attr`
- 新 `fragment_op` `edge_attr`
- 不同分子主干（GCN / FragNet）

### Q3：planner 升级是否优于当前 BFS/MCTS？

比较：

- BFS baseline
- fragment + BFS
- fragment + beam
- fragment + best-first

### Q4：路径级模型是否真正带来长程收益？

比较：

- 单步分数累加
- step + heuristic
- step + value
- step + value + policy

---

## 2. 最小实验矩阵

建议至少覆盖以下组合：

| 编号 | 动作空间 | 单步模型 | Planner | 目标 |
|------|----------|----------|---------|------|
| E1 | primitive | OFO-old | BFS | 基线 |
| E2 | fragment | OFO-old（弱 rerank） | BFS | 先验证动作空间 |
| E3 | fragment | OFO-frag | BFS | 验证模型升级 |
| E4 | fragment | OFO-frag | beam | 验证 planner 升级 |
| E5 | fragment | OFO-frag + value | best-first / A* | 验证长程规划 |

---

## 3. 指标建议

### 3.1 任务结果指标

- Top-1 改善值
- Top-k 最优改善值
- 达到阈值改善的样本比例
- 平均最优路径长度

### 3.2 搜索效率指标

- 平均展开节点数
- 单分子平均运行时间
- 单位预算下的有效候选数

### 3.3 化学合理性指标

- 合法分子率
- scaffold 保持率
- SA 分布
- `logP` / 其他约束满足率
- 候选多样性

### 3.4 模型指标

- step-level 回归误差
- path/value 回归误差
- rank correlation
- calibration / uncertainty（若有）

---

## 4. 阶段性里程碑

### Milestone 1：baseline 可对比

完成标志：

- baseline benchmark 集固定
- baseline 指标可重复输出

### Milestone 2：片段动作可运行

完成标志：

- `fragment_op` schema 稳定
- 能生成一批合法 fragment 候选
- 相比 primitive 候选更像真实化学优化动作

### Milestone 3：`OFO-frag` 单步有效

完成标志：

- 训练正常收敛
- 验证集上能区分高低质量 fragment transition

### Milestone 4：planner 有收益

完成标志：

- `beam` 或 `best-first` 在固定预算下优于 BFS baseline

### Milestone 5：长程建模有收益

完成标志：

- path/value 至少在一部分 benchmark 上保留了更多长期有利路径

---

## 5. 消融实验建议

建议做的消融包括：

- 去掉 `fragment_op` 中的 `anchor` 信息
- 去掉 `scaffold_preserving` 约束
- `fragment fingerprint` vs learned fragment encoder
- BFS vs beam vs best-first
- only step score vs step + value

---

## 6. 推荐输出产物

- `benchmark_results.csv`
- `ablation_results.csv`
- `planner_budget_analysis.csv`
- `case_studies.md`
- `milestone_summary.md`

---

## 7. 风险与注意事项

- 不要在数据集口径变化时直接比较模型结论
- 不要只看最终最优值，不看预算和化学合理性
- 不要忽略失败案例分析
- 不要让实验矩阵过大到无法完成

---

## 8. 退出条件

这条路线最终应回答的不是“模块都做了没有”，而是：

> **fragment-level action + OFO-frag + planner + value/path 是否比当前 primitive baseline 显著更强，且更符合真实分子优化语义。**

当这个问题能够被一组完整实验较清楚地回答时，这组计划的主体目标就算完成。
