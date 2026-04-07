# 05：Planner 升级计划（重构收口版）

> 目标：在单步模型升级为 semantic-step scorer 后，把当前偏实验性质的层式搜索，升级为真正消费 **semantic actions** 的 planner，同时保留 primitive replay 能力。

> 本文件只解决 **搜索内核、frontier 管理、约束、去重、replay 与接口抽象**；所有 **learned long-range signals**（如 value / policy / RL）统一后移到 `06`。

---

## 1. 本阶段先明确 planner 的新角色

planner 不再只是：

- 按层扩展 primitive 邻居
- 再按单步分数做粗排序

新 planner 应该负责：

- 管理 semantic action frontier
- 控制预算、去重、多样性与约束
- 记录可回放的 transition
- 为后续 value / policy / A* 预留稳定接入面

---

## 2. 本文件的边界

### 2.1 本阶段负责什么

本阶段只负责把“搜索器”本身做对：

- action proposal / apply / replay 协议
- frontier 数据结构
- beam / best-first 等搜索策略
- pruning、dominance、budget 管理
- semantic path 与 primitive replay 的同步记录

### 2.2 本阶段不负责什么

本阶段**不**负责：

- 训练 value 模型
- 训练 policy / prior
- 设计 RL 目标与采样闭环
- 把 learned heuristic 写进 planner 计划主体

这些统一放在 `06`。

---

## 3. planner 与动作协议的关系

本阶段默认 `02-04` 已经成立，planner 应直接消费：

- `semantic_step` 风格动作对象
- 或与之同构的在线候选动作对象

关键要求：

- **训练时看到的动作语义**
- **规划时扩展的动作语义**
- **replay 时记录的动作语义**

尽量保持同构。

同时还要保留：

- `primitive_trace`
- `primitive_span`
- 或可回放的执行细节

这样 planner 才不会只剩“高层标签”，却无法解释动作是如何执行出来的。

---

## 4. 推荐的 planner 抽象

推荐把 planner 的核心抽象成三部分：

### 4.1 `ActionProposal`

描述一个候选 semantic action，建议至少包含：

- `semantic_level`
- `fragment_op` 或 atomic fallback 描述
- `expected_constraints`
- `provenance`
- 可选的 `primitive_template` / `primitive_trace_hint`

### 4.2 `TransitionResult`

描述 action 被应用后的结果，建议至少包含：

- `state_from`
- `state_to`
- `semantic_step`
- `primitive_trace`
- `step_score`
- `constraint_penalty`
- `is_valid`

### 4.3 `PlannerState`

描述搜索树中的一个节点，建议至少包含：

- 当前分子状态
- 历史 semantic path
- 历史 primitive replay
- 累计收益
- 预算消耗
- 去重 key / scaffold key

---

## 5. 推荐推进顺序

### Step 1：先做 `beam search`

原因：

- 工程复杂度最低
- 易于和当前层式搜索对照
- 便于观察 semantic action 质量本身

### Step 2：再做 `best-first`

原因：

- 更适合把预算集中在高价值 frontier
- 更适合验证 frontier / pruning / replay 抽象是否足够稳

### Step 3：最后做轻量 `A*` 壳层

前提：

- 状态去重与 budget 控制足够稳定
- frontier score 接口已经标准化
- 即使暂时没有 learned `h(s)`，也能先留 heuristic 插槽

这里强调的是：**先把 A* 所需接口做出来，而不是在本阶段就把 learned heuristic 训练完。**

---

## 6. 推荐评分函数

本阶段的 planner score 应尽量保持“无 learned 长程信号”的可解释版本。

第一版建议拆成：

- `g(s)`：历史累计收益
- `q(s, a)`：单步 semantic-step score
- `c(s, a)`：约束惩罚
- `d(s)`：多样性或新颖性奖励

第一版可先用：

> `planner_score = λ1 * cumulative_gain + λ2 * step_gain - λ3 * constraint_penalty + λ4 * diversity_bonus`

到 `06` 再引入：

- learned value bonus
- learned heuristic `h(s)`
- policy prior

---

## 7. 本阶段任务拆解

### Task 1：抽象统一 planner 接口

建议新增抽象接口，例如：

- `propose_actions(state)`
- `apply_action(state, action)`
- `score_transition(state, action, result)`
- `push_frontier(node)`
- `select_frontier()`
- `should_prune(node)`

### Task 2：实现 `beam search`

需要明确：

- beam width
- 每节点最大扩展数
- 每层预算
- 去重规则
- 多样性规则

### Task 3：实现 `best-first`

需要明确：

- priority queue 结构
- frontier 更新逻辑
- 节点重访 / dominance 判定
- budget 截止条件

### Task 4：接入约束与 replay

建议尽早纳入：

- SA / `logP` / scaffold 等约束
- semantic path replay
- primitive trace replay
- 失败候选日志

### Task 5：预留 `06` 的接入点

需要先把下面这些接口留出来，但不在本阶段把 learned 模块做完：

- value bonus hook
- heuristic score hook
- action prior hook

### Task 6：与 baseline 做公平对比

至少比较：

- primitive + BFS
- semantic + BFS
- semantic + beam
- semantic + best-first

---

## 8. 与 `06` 的交接关系

当以下条件成立时，就说明 `05` 已经足够完成，可以把后续复杂度交给 `06`：

- planner 接口稳定
- replay 稳定
- frontier / pruning / budget 机制稳定
- semantic planner 已经能在**不依赖 learned value/policy** 的情况下跑出可信结果

从这一步开始，`06` 才负责把 learned long-range signals 接进来。

---

## 9. 代码落点建议

优先新增：

- `core/planners/base.py`
- `core/planners/beam.py`
- `core/planners/best_first.py`
- `core/planners/score_utils.py`
- `core/planners/replay.py`

必要时改造：

- `core/evolution_optimizer.py`
- 现有搜索扩展入口

---

## 10. 验收标准

完成本阶段时，应满足：

- 至少一条 semantic planner 路线可稳定运行
- planner 消费的动作对象与训练侧协议清晰一致
- replay 同时保留 semantic path 与 primitive trace
- 相比 baseline BFS，至少在部分预算口径上更有效
- 即便不引入 learned value / policy，planner 内核也已足够稳定

---

## 11. 一句话结论

> **新 planner 的关键不是把 BFS 换个名字，而是先把搜索真正围绕 semantic action 做对，并把 replay、frontier 和接口层收口好；learned long-range signals 再统一交给 `06`。**
