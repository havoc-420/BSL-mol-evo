# P2：把搜索结果变成更强的学习信号
<!-- last-updated: 2026-04-12 -->

本文档对应长期更有潜力的升级方向：
不是只靠在线 policy gradient 硬学，而是把 `A* / MCTS / OFO` 搜索本身产生的结构化结果，转化成 `policy / value` 更强、更稳的监督信号。

---

## 为什么这条路线重要

当前系统的真正强项并不是单独的 RL，而是：

- 能生成候选动作集合；
- 能结合 OFO 估值做 expensive 评估；
- 已经有 `A* / MCTS` 搜索器；
- 已经能离线导出树和 transition。

这意味着更自然的问题其实不是：

> “怎么让 policy 一步步自己蒙出最优动作？”

而是：

> **“怎么让搜索先给出更好的动作分布 / 价值判断，再让 policy/value 去拟合这个更强 target？”**

这更接近 AlphaZero 式的 `search-improved policy/value learning`，也更符合当前项目的工程结构。

---

## 目标

建立一条搜索引导学习路线：

- **policy** 学习搜索改进后的动作分布；
- **value** 学习状态未来可达收益；
- 在线 RL 变成补充信号，而不是唯一信号；
- 最终减少高方差 Monte Carlo 更新对结果的支配。

---

## 三种可落地的信号来源

### 1. 搜索访问分布（首选）

如果使用 `MCTS`，最自然的 target 是：

- 每个 action 的 visit count
- 或归一化后的 visit distribution

policy 直接学习这组 improved distribution。

### 2. 搜索后验排序

即便不是完整 MCTS，也可以从 `A* / astar_demo` 导出：

- 候选动作排序；
- top-k 集合；
- sampled action 与最终优秀路径的重合情况。

这可以支持 ranking / listwise 学习。

### 3. 最终可达收益

value 学习的不应只是 immediate delta，而应尽量靠近：

- 从当前状态出发，搜索预算内能达到的最佳收益；
- 或 rollout / subtree 的聚合收益统计。

---

## 推荐任务拆解

### Task 1：扩展数据导出协议

现有 transition 导出更偏向 BC / RL demo 训练。

下一步建议额外导出：

- `candidate_actions`
- `search_rank`
- `search_score`
- `visit_count`（如果有 MCTS）
- `best_descendant_gain`
- `subtree_stats`

这一步决定后面能不能做 search-guided learning。

### Task 2：先做 policy imitation，而不是直接 online RL

优先训练：

- policy 拟合搜索改进分布；
- 或学习 top-k / pairwise preference。

目标是先把 prefilter 变强，再看是否需要更多在线 RL 微调。

### Task 3：让 value 学“未来最好能到哪”

建议 value target 不再只基于 step return，而应更贴近：

- `best reachable improvement`
- `discounted subtree return`
- 或固定预算下的 `best-of-search` 统计

这样 value 才更像真正的 `h(s)`。

### Task 4：把 policy/value 明确接回搜索器

接回时建议边界清晰：

- **policy**：负责 prefilter / proposal rerank / prior bonus
- **value**：负责 heuristic / frontier priority
- **搜索器**：仍负责 tree expansion、预算控制和去重

不要把所有逻辑重新揉成一个黑箱 RL agent。

---

## 与纯在线 RL 的关系

这条路线不是完全取代 PPO，而是给 PPO 一个更好的上游。

推荐关系：

1. **先有强的 search-guided BC / imitation**；
2. **再用 PPO 做小步在线微调**；
3. **不要指望在线 RL 单独完成所有 credit assignment。**

---

## 备选：bandit / ranking 作为低风险落地版

如果短期目标只是提升 `top_n_prefilter` 与 OFO 预算利用率，
那么可以先做一个更轻量的版本：

- contextual bandit
- pairwise ranking
- listwise ranking

这种路线的优势：

- 工程实现简单；
- 不依赖完整 on-policy rollout；
- 更容易直接对齐“候选排序质量”这个业务目标。

适合作为 P2 的低风险子路线。

---

## 代码与文档落点建议

可能涉及：

- `mol_evo/dataset/export_rl_demo_transitions.py`
- `mol_evo/core/data/rl_demo_processing.py`
- `mol_evo/core/models/astar_rl/policy_network.py`
- `mol_evo/core/models/astar_rl/value_network.py`
- `mol_evo/core/molecular_evolution_expansion.py`
- `mol_evo/docs/arl/experiments.md`

如后续形成更稳定的数据协议，可再单独拆出：

- 搜索监督数据导出脚本
- ranking / bandit 训练脚本
- search-guided evaluation 脚本

---

## 评估方式

### policy 侧

- top-k 命中率
- sampled action 被搜索优选动作覆盖的比例
- prefilter 后保留下来的高价值候选比例

### value 侧

- 与 `best reachable gain` 的相关性
- 接入搜索后对 frontier 质量的提升
- 在固定预算下的最终 holdout 指标改善

### 系统侧

- OFO 调用效率
- `top1_median / trimmed_mean`
- `win_rate_vs_bc`
- 不同 seeds 下是否更稳

---

## 完成标准

P2 不要求一开始就全面替代在线 RL，但至少应达到：

- 可以导出搜索引导监督数据；
- policy 或 value 至少一侧能在离线指标上显示正信号；
- 接入搜索后在固定预算下有可复现收益；
- 能判断“继续走 PPO 微调”还是“转向 ranking / bandit”更划算。

---

## 一句话结论

> **如果长期仍以搜索为核心，那么最有前途的不是继续强化高方差 REINFORCE，而是把搜索本身产出的改进策略与未来价值，转化成 policy/value 的主要学习信号。**
