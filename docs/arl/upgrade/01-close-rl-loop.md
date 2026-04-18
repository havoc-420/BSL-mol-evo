# P0：先修成真正可学的 RL 闭环
<!-- last-updated: 2026-04-12 -->

本文档对应当前 `astar_demo` 路线的**第一优先级任务**：
在不大改搜索框架的前提下，把在线 RL 从“看起来像 RL”修成“真正能学到 policy 决策”的闭环。

---

## 目标

解决三个一致性问题：

- **谁做动作决策**
- **谁拿到 reward**
- **谁被 policy gradient 更新**

当前如果这三者不一致，那么后续即便换成 PPO，也只是更稳定地学错东西。

---

## 当前问题摘要

基于现有实现，当前风险主要在：

- `molecular_evolution_expansion.py` 中在线 RL 轨迹收集更像是把 `top_scored[0]` 当作“被选动作”；
- 存在固定 `selected_idx = 0` 的模式，policy 没有真正采样动作；
- reward 更贴近搜索排序后的 top1，而不是 policy 真正选中的动作；
- 最终 trajectory 里的 `log_prob`、`reward`、`next_state` 之间可能并非同一动作链路。

这会直接导致：

- 探索是假的；
- advantage 学习信号偏掉；
- `episode_return` 与 holdout 指标关系更不稳定。

---

## 任务拆解

### Task 1：明确“policy 决策点”

先把在线训练里真正由 policy 决策的时刻定义清楚。

建议规则：

- policy 只在**候选动作集合已经生成完成**之后做决策；
- 决策对象应是这组候选里的某一个 action index；
- 搜索过程中的其它排序逻辑要么变成 prior，要么退到 logging，不应继续冒充被更新的 action。

### Task 2：把 `selected_idx = 0` 改成真实采样

建议最小版本：

- 从 policy logits 生成 categorical 分布；
- 训练时按分布采样，评估时可保留 greedy / top1；
- 支持温度、entropy 正则或 epsilon-greedy 作为探索控制。

建议预留参数：

- `sample_mode`: `sample / greedy / epsilon_greedy`
- `temperature`
- `epsilon`

### Task 3：reward 必须与 sampled action 对齐

这是最关键的地方。

要求：

- 当前 step 的 reward 必须来源于 **policy 实际选中的 action**；
- `next_state` 也必须是该 action 执行后的结果；
- 不允许继续用搜索排序后的 top1 奖励回头训练另一个 action 的 log-prob。

### Task 4：重新定义 trajectory 协议

建议让 `TrajectoryStep` 明确包含：

- `state_vec`
- `candidate_action_vecs`
- `selected_idx`
- `selected_log_prob`
- `reward`
- `done`
- `next_state_vec`（可选）
- `metadata`（如 parent smiles、operation details、search depth）

### Task 5：把探索和搜索影响拆开记录

为了后续分析，建议同时记录：

- policy 分布熵；
- sampled action 的原始 rank；
- `top_scored[0]` 与 sampled action 是否相同；
- sampled action 带来的 property change；
- 搜索 bonus（如 value bonus / policy bonus）对最终选择的影响。

---

## 代码落点建议

优先涉及以下文件：

- `mol_evo/core/molecular_evolution_expansion.py`
- `mol_evo/core/models/astar_rl/rl_trainer.py`
- `mol_evo/core/models/astar_rl/policy_network.py`
- `mol_evo/core/models/astar_rl/reward.py`
- `mol_evo/train_astar_rl_demo.py`

建议原则：

- **先最小修改**，不要在 P0 阶段顺手把算法、网络结构、日志系统全重写；
- 先把 decision / reward / update 三者对齐，再考虑 PPO。

---

## 验证方案

### 单元级验证

至少补以下测试：

- 采样模式下 `selected_idx` 不再固定为 `0`；
- `selected_log_prob` 与 sampled action 对应；
- reward 来源于 sampled action，而不是排序第一名；
- greedy 模式下行为与旧版评估逻辑兼容。

### 运行级验证

建议跑两组最小实验：

- **训练 smoke test**：看 trajectory 是否正常收集、episode 是否能完成；
- **固定 holdout 对照**：比较修正前后 `episode_return` 与 holdout 指标的相关性是否改善。

---

## 完成标准

只有满足下面这些条件，P0 才算完成：

- policy 真实参与动作采样；
- reward、next state、log_prob 三者严格对齐；
- trajectory 可被独立检查和复现；
- 修正后训练能稳定运行，不引入新的搜索崩溃；
- holdout 指标至少不显著劣化。

---

## 为什么 P0 必须先做

> **因为 PPO、GAE、甚至 search-guided learning，都是建立在“当前这一步到底是谁做出的决策”这个问题已经说清楚的基础上。**

如果这个问题没修完，后续所有更复杂的方法都很容易变成“更复杂、更稳定、但仍然学偏”的版本。
