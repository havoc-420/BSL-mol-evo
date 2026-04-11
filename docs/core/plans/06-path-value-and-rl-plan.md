# 06：Semantic Path / Value / Policy / RL 计划（重构收口版）

> 目标：在单步 semantic-step scorer 与新 planner 内核稳定后，引入真正支持长程规划的 path/value/policy 路线，而不是继续假设“单步分数累加就等于长程收益”。

> 本文件只解决 **learned long-range signals**：`semantic path` 数据、value、policy、heuristic 与 RL；**不重新设计 planner 内核、frontier 机制与 replay 抽象**，这些统一视为 `05` 已提供的基础设施。

---

## 1. 本阶段的出发点

当前系统最大的长程短板不是搜索器名字，而是：

> **单步 delta predictor 不等于未来价值估计。**

同时，新路线下的 path 也不能再简单理解成 primitive op 序列。

更合理的定义应是：

- **事实层**：primitive path
- **语义层**：semantic path

其中 semantic path 的每条边是一个 `semantic_step`，并引用其对应的 `primitive_span / primitive_trace`。

---

## 2. 本文件的边界

### 2.1 本阶段负责什么

本阶段负责：

- 构造 `semantic_paths` 数据
- 训练 path / value 模型
- 训练 policy / prior
- 为 `05` 已经存在的 planner 接入 learned heuristic / bonus
- 在 offline 路线稳定后再讨论 RL

### 2.2 本阶段不负责什么

本阶段**不**重新负责：

- planner 的 frontier 抽象
- beam / best-first 的主体实现
- pruning / dominance / budget 的主逻辑
- replay 协议本身的重新设计

这些都应复用 `05` 的结果。

---

## 3. 新的 path 观

### 3.1 semantic path 的最小抽象

推荐把路径表示成：

- `node_smiles_list`
- `semantic_steps`
- `primitive_replay`
- `step_targets`
- `path_target`

也就是说：

- 节点仍然是分子状态
- 边不再只是一条 primitive op
- 边是一个 `semantic_step`
- 每条 semantic edge 可以挂一段 primitive trace

### 3.2 为什么这很重要

这样做能同时保住：

- path 的长程规划语义
- fragment 级动作语义
- primitive 级可执行与可审计能力

---

## 4. 本阶段目标

- 构造 `semantic_paths` 训练数据
- 训练 path / value 模型
- 训练轻量 policy / prior
- 为 planner 提供更像样的 `h(s)` 与动作优先级
- 只在前面都稳定后再讨论 RL

---

## 5. 与 `05` 的交接前提

只有在下面这些条件成立后，`06` 才应进入主线：

- planner 接口稳定
- frontier / pruning / replay 已稳定
- semantic planner 已能在不依赖 learned long-range signals 的情况下稳定运行
- `semantic_pairs / semantic_paths / planner_replay` 的导出链已经可用

也就是说：

> **`06` 不是替代 `05`，而是建立在 `05` 之上的 learned augmentation 层。**

---

## 6. 推荐推进顺序

### Step 1：先做 `semantic path` 数据

先把 path 表示、mask、replay 和 target 组织清楚。

### Step 2：再做 value / path model

优先训练：

- `path_target` 预测
- 或 `V(s)` 未来可达收益预测

### Step 3：再做 policy / prior

让模型学习：

- 哪些 semantic actions 值得优先扩展
- 哪些候选动作大概率低价值

### Step 4：把 learned signals 接入 `05`

优先接入：

- `best-first + value bonus`
- `A*` 的 learned / semi-learned `h(s)`
- action prior 作为 proposal rerank 或 pruning 辅助

### Step 5：最后再做 RL

RL 只在以下前提下进入主线：

- semantic action 协议稳定
- planner 稳定
- value / policy 已显示出正收益

---

## 7. 推荐的数据协议

### 7.1 `semantic_paths_*`

建议至少包含：

- `path_id`
- `node_smiles_list`
- `semantic_steps`
- `primitive_replay`
- `target_property`
- `step_targets`
- `path_target`
- `valid_step_mask`
- `meta`

### 7.2 `step_targets` 的组织建议

当一个 semantic step 覆盖多个 primitive 步时，建议：

- `step_target` 对应 semantic span 的聚合收益
- `primitive_step_targets` 可保留在 replay 或 meta 中

这样可以同时支持：

- semantic-step 监督
- primitive replay 分析

---

## 8. 可选模型方向

### Route A：semantic path 累计收益模型

输入：

- `start_smiles`
- `semantic_step` 序列
- 可选的中间状态表示

输出：

- `path_target`
- 可选每步收益分配

### Route B：状态价值模型 `V(s)`

输入：

- 当前状态 `s`
- 目标属性与约束条件

输出：

- 从当前状态出发的预期剩余收益

### Route C：动作先验 `π(a|s)`

输入：

- 当前状态
- 候选 semantic actions

输出：

- 候选动作优先级分布

---

## 9. 本阶段任务拆解

### Task 1：整理 semantic path 数据

需要补齐：

- path 长度分布
- 有效 step mask
- semantic step 与 primitive replay 的对齐关系
- planner 轨迹回放导出

### Task 2：训练 value / path 模型

建议第一版目标：

- 预测 `path_target`
- 或预测当前状态未来可达收益

并与“单步分数累加”做对比。

### Task 3：把 value 接入 planner

优先做：

- `best-first + value bonus`
- `A*` 的轻量 `h(s)`

### Task 4：训练 policy / prior

目标不是直接端到端生成，而是：

- 降低无效扩展比例
- 缩小候选动作集合
- 提高 frontier 质量

### Task 5：最后再做 RL

只在 offline 路线稳定后进行。

---

## 10. 代码落点建议

优先复用：

- `core/data/path_processing.py`
- 当前路径模型相关代码
- `core/planners/*` 中由 `05` 提供的稳定接口

建议新增：

- `core/data/semantic_path_processing.py`
- `core/models/value/*`
- `core/models/policy/*`
- `core/planners/heuristics.py`

---

## 11. 验收标准

完成本阶段时，应满足：

- 至少一版 `semantic_path` 数据可训练可验证
- planner 能调用 value 作为辅助分数
- 与纯单步分数累加相比，长程保留能力至少在部分任务上更强
- policy / prior 至少能减少一部分低价值扩展
- `06` 的新增复杂度主要体现为 learned long-range signals，而不是重新改写 planner 内核

---

## 12. 一句话结论

> **path 这条线不应在 fragment 时代消失；它应该升级成“以 semantic step 为边、以 primitive replay 为底”的长程规划层，并作为 `05` 之上的 learned augmentation 层进入系统。**

---

## 13. 与当前 primitive `A* RL Demo` 的衔接状态（2026-04-11）

### 13.1 当前已存在的可运行链

仓库里当前真实跑通的 RL 链，不是本文件目标态的 `semantic path/value/policy`，而是 `docs/arl` 中维护的 primitive-action demo：

- `BFS / MCTS` 搜索树 JSON
- `export_rl_demo_transitions.py` 导出 transition
- `train_bc_pretrain.py` 训练 `PolicyNet / ValueNet`
- `train_astar_rl_demo.py` 进行在线 RL
- `eval_astar_rl_holdout.py` 做固定 holdout 评估

### 13.2 这条链对 `06` 的现实意义

- **它证明了 `policy / value / RL` 这条工程闭环已经可以 operational 地跑通**，因此 `06` 不是从零开始。
- **但它还不是 `06` 的目标实现**：当前 state/action 仍是 primitive demo 编码，不是 `semantic_path`、也不是 `semantic_step` 语义层上的长程建模。
- **它更像 `06` 的过渡验证台**：先用现有 primitive demo 验证 “更好的离线基石数据 -> 更强 BC -> 更稳 RL” 是否成立，再决定 semantic 路线里的 long-range signals 该如何接入。

### 13.3 当前观测到的直接约束

- 导出桥虽已修复，但当前 `MCTS` 在 `num_simulations=200` 下导出的树仍偏稀，导致 `A1-mcts-main` 只有 `1089` 条 transition。
- 这说明当前问题**不只是有没有 value / policy**，还包括 **长程基石数据本身是否足够厚、是否能覆盖更多有效路径**。
- 因此，眼下正在推进的 `mcts-expand-rl` 更适合作为 `06` 的前置证据：先验证厚 `MCTS` 数据源能否提供更强 bootstrap，再决定 semantic `path/value` 训练该如何构造目标和比较口径。

### 13.4 当前边界纪律

- 不要把当前 primitive `A* RL Demo` 的短期实验结果，直接等同于 `semantic path/value/policy` 的最终结论。
- 但也不要忽略它：它提供了 `06` 未来落地时最现实的**评估外壳、holdout 口径和 checkpoint 选优经验**。
- 换句话说：**`docs/arl` 是当前可运行的证据台，`06` 是下一阶段要接管这套证据台的语义长程版本。**
