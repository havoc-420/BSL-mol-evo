# 05：Planner 升级计划（`beam / best-first / A*`）

> 目标：在动作空间质量提升后，把当前“层内统一排序 + 前 50% 截断”的近似树搜，升级为真正面向高质量候选的规划器。

---

## 1. 本阶段核心判断

当前 baseline 的主要问题之一，不只是动作太 primitive，还包括：

- 层内统一排序过于粗糙
- 单步分数容易压制长程有利路径
- BFS/MCTS 的当前实现更偏实验平台，而不是强规划器

因此在 `fragment_op + OFO-frag` 初步成立后，应优先升级 planner。

---

## 2. 本阶段目标

- 引入 `beam search` 或 `best-first search`
- 定义新的候选状态评分函数
- 控制预算、去重与多样性
- 为未来 `A*` 和 value model 预留接口

边界说明：

- 本文件聚焦 **planner 机制本身**：frontier 管理、budget、去重、扩展顺序、score 组合
- `value / policy / heuristic` 的**训练方案与数据组织**放到 `06-path-value-and-rl-plan.md`
- 这里只定义它们如何接入 planner，不重复展开训练路线

---

## 3. 推荐推进顺序

### Step 1：先做 `beam search`

原因：

- 结构简单
- 易于与当前层式搜索对比
- 易于控制 beam 宽度与预算

### Step 2：再做 `best-first`

原因：

- 能更灵活地把预算投给高价值节点
- 更适合插入复合评分函数

### Step 3：最后做轻量 `A*`

前提是已有可用 `h(s)` 或 value proxy。

---

## 4. 新评分函数设计

建议统一定义：

- `g(s)`: 历史累计收益
- `q(s, a)`: `OFO-frag` 预测的一步收益
- `c(s, a)`: 约束惩罚
- `u(s, a)`: 不确定性惩罚（可选）
- `d(s)`: 多样性或新颖性奖励

第一版可以先用：

> `score = λ1 * accumulated_delta + λ2 * step_delta - λ3 * constraint_penalty + λ4 * diversity_bonus`

后续再扩展到 value / heuristic。

---

## 5. 本阶段任务拆解

### Task 1：抽象统一 planner 接口

建议新增统一接口，例如：

- `Planner.expand()`
- `Planner.select_frontier()`
- `Planner.score_candidate()`
- `Planner.should_prune()`

### Task 2：实现 `beam search`

需要明确：

- beam 宽度
- 每节点最大扩展数
- 每层总预算
- 去重规则
- 多样性过滤规则

### Task 3：实现 `best-first`

需要明确：

- priority queue 结构
- 状态去重策略
- budget 截止条件
- 扩展优先级解释

### Task 4：加约束与辅助分数

建议尽早纳入：

- SA penalty
- `logP` 约束
- scaffold 保持约束
- 结构相似性约束
- 候选多样性约束

### Task 5：和 baseline 对比

至少比较：

- primitive + BFS
- fragment + BFS
- fragment + beam
- fragment + best-first

---

## 6. 代码落点建议

优先改造：

- `core/evolution_optimizer.py`
- `core/molecular_evolution_expansion.py`

更推荐新增：

- `core/planners/base.py`
- `core/planners/beam.py`
- `core/planners/best_first.py`
- `core/planners/score_utils.py`

---

## 7. 验收标准

完成本阶段时，应满足：

- `beam` 或 `best-first` 至少一条线可稳定运行
- planner 与 `OFO-frag` 接口清晰
- 相比 baseline BFS，至少在部分 benchmark 上显示出更优的预算利用率或 Top-k 质量
- 结果日志能解释为什么某些节点被优先扩展

---

## 8. 风险与注意事项

- 不要在动作空间未升级前就过度投入 planner
- 不要过早引入复杂 A* 细节
- 不要让 score 函数混入太多未经验证的项
- 不要忽略去重与多样性，否则 beam 很容易塌缩
