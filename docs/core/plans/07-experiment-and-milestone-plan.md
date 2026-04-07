# 07：实验设计与里程碑计划（重构收口版）

> 目标：为重构后的整条路线提供统一的研究问题、实验矩阵、里程碑和验收口径，避免各阶段各自推进却无法形成全局结论。

> 本文件不重新定义动作协议、数据主干、模型结构或 planner 机制；它只负责把 `01-06` 的产物放进同一套比较框架里，回答“这条新路线是否真的更强、更稳、更可解释”。

---

## 1. 统一研究主问题

建议把后续实验收敛到四个主问题。

### Q1：双层动作协议是否真的比旧 primitive 表示更有价值？

比较重点：

- primitive-only
- semantic mixed（fragment + atomic fallback）
- strict fragment-only

### Q2：`OFO-frag-step` 是否优于旧 OFO？

比较重点：

- 旧 primitive edge
- semantic-step edge
- strict fragment edge

### Q3：semantic planner 是否优于当前 BFS/MCTS 基线？

比较重点：

- primitive + BFS
- semantic + BFS
- semantic + beam
- semantic + best-first

### Q4：semantic path / value / policy 是否真正带来长程收益？

比较重点：

- step-only 累加
- step + heuristic
- step + value
- step + value + policy

---

## 2. 本文件的边界

### 2.1 本文件负责什么

本文件只负责：

- 统一研究问题
- 统一实验矩阵
- 统一里程碑定义
- 统一验收证据与输出物

### 2.2 本文件不负责什么

本文件**不**重新负责：

- 动作 schema 的设计
- 数据主干的实现
- 单步模型结构细节
- planner / value / policy 的工程接口设计

如果这些内容需要改，应回到 `02-06` 对应文件。

---

## 3. 推荐的最小实验矩阵

建议至少覆盖以下组合：

| 编号 | 动作层 | 单步模型 | Planner | 长程信号 | 目的 |
|------|--------|----------|---------|----------|------|
| E1 | primitive | OFO-old | BFS | 无 | 旧 baseline |
| E2 | semantic mixed | OFO-old rerank / 弱接入 | BFS | 无 | 验证动作协议本身 |
| E3 | semantic mixed | OFO-frag-step | BFS | 无 | 验证单步模型升级 |
| E4 | strict fragment | OFO-frag-step | BFS / beam | 无 | 验证纯 fragment 子集 |
| E5 | semantic mixed | OFO-frag-step | best-first | value | 验证长程规划 |

说明：

- `semantic mixed` 往往更接近真实工程入口
- `strict fragment` 更适合回答“纯片段语义是否值得”
- `E1-E3` 主要覆盖 `01-04`
- `E4-E5` 开始覆盖 `05-06`

---

## 4. 指标建议

### 4.1 效果指标

- Top-1 改善值
- Top-k 最优改善值
- 达到阈值改善的样本比例
- 平均最优路径长度

### 4.2 搜索效率指标

- 平均展开节点数
- 单分子平均运行时间
- 单位预算有效候选数
- 重复扩展比例

### 4.3 化学合理性指标

- 合法分子率
- scaffold 保持率
- 约束满足率
- 候选多样性
- SA / `logP` 等分布

### 4.4 模型指标

- step-level 回归误差
- path/value 回归误差
- rank correlation
- fallback / fragment 子集分层指标

---

## 5. 分层统计原则

重构后，实验不应只给一个总平均数。

建议至少按以下切片分层统计：

- `semantic_level = fragment` vs `atomic_fallback`
- `annotation_status = resolved` vs `approximate`
- `QM9 bootstrap` vs 后续 fragment-rich 数据源
- 不同路径长度区间
- 不同预算区间

这样才能判断：

- 提升到底来自哪里
- 退化到底发生在哪一层
- 收益是否只是预算放大造成的错觉

---

## 6. 里程碑与上游阶段的对应关系

### Milestone 1：baseline 可比较

对应阶段：`01`

完成标志：

- benchmark 子集固定
- baseline 指标和评估链固定

### Milestone 2：双层动作协议与数据主干可运行

对应阶段：`02-03`

完成标志：

- `primitive_trace + semantic_step + fragment_op` 协议稳定
- fallback 规则明确
- 能导出一批合法 `semantic_pairs / semantic_paths`

### Milestone 3：单步 semantic-step 模型有效

对应阶段：`04`

完成标志：

- `OFO-frag-step` 稳定收敛
- 至少一类 semantic 子集上优于旧 OFO

### Milestone 4：planner 内核有明确收益

对应阶段：`05`

完成标志：

- semantic planner 在固定预算下优于 baseline BFS
- replay 可以解释动作与路径
- 不依赖 learned value / policy 也能稳定工作

### Milestone 5：长程建模有收益

对应阶段：`06`

完成标志：

- value / policy 至少在部分 benchmark 上提升长程路径保留能力
- learned long-range signals 的收益可以与 planner 内核收益区分开

---

## 7. 推荐消融实验

建议优先做：

- 去掉 `fragment_op` 中的 anchor 信息
- 去掉 primitive trace 特征，只保留 fragment 语义
- mixed semantic 训练 vs strict fragment 训练
- BFS vs beam vs best-first
- step-only vs step + value
- 无 learned planner vs + value bonus vs + policy prior

这些消融直接对应重构后的核心设计选择。

---

## 8. 推荐输出产物

建议统一沉淀：

- `benchmark_results.csv`
- `ablation_results.csv`
- `planner_budget_analysis.csv`
- `semantic_level_breakdown.csv`
- `case_studies.md`
- `milestone_summary.md`

---

## 9. 结果解释纪律

后续写结论时，必须避免：

- 数据口径变化却直接比较模型结论
- 只看最终最优值，不看预算和合法性
- 忽略 fallback 样本对整体结果的影响
- 用 QM9 的结果替代对长期主数据源的结论
- 把 `05` 的 planner 内核收益和 `06` 的 learned signal 收益混为一谈

---

## 10. 退出条件

这条路线最终应回答的不是“模块都写完没有”，而是：

> **基于双层动作协议的 `semantic step + OFO-frag-step + planner + value/path`，是否比当前 primitive baseline 更强、更稳、更可解释。**

当这个问题能被一组完整实验较清楚地回答时，本计划目录的主体目标才算完成。
