# 06：路径级 / Value / Policy 计划

> 目标：在单步 `OFO-frag` 与新 planner 稳定后，引入真正支持长程规划的路径级建模，而不是继续依赖“单步分数累加”。

---

## 1. 本阶段核心判断

当前系统的根本短板之一是：

> **单步 delta predictor 不等于未来价值估计。**

因此，一旦 `fragment_op` 和 planner 基本稳定，就应该进入：

- path model
- value model
- policy model

这也是后续认真做 `A* + RL` 的前提。

---

## 2. 本阶段目标

- 构造可用于多步训练的路径数据
- 训练 `V(s)` 或 `path_target` 模型
- 训练轻量 policy / prior model
- 为 planner 提供更像样的 `h(s)` 或扩展优先级

边界说明：

- 本文件默认 `beam / best-first` 等基础 planner 结构已由 `05-planner-upgrade-plan.md` 提供
- 这里重点解决 **value / path / policy 的建模与接入**，而不是重复设计搜索器骨架
- 若需要讨论 `A*` 的状态队列、budget、去重等基础机制，回到 `05`

---

## 3. 推荐推进顺序

### Step 1：先做 path/value

优先于 RL，因为：

- 更贴近当前已有的 `v0.3` 路径思路
- 更容易利用离线轨迹数据
- 更适合作为 planner 的直接辅助信号

### Step 2：再做 policy

让模型学习：

- 当前状态优先扩哪些动作
- 哪些动作大概率无效

### Step 3：最后再做 RL

把 RL 放在：

- 利用已有轨迹离线学习
- 微调 policy/value
- 减少在线搜索浪费

而不是一开始就做端到端生成。

---

## 4. 可选模型方向

### Route A：路径累计收益预测

输入：

- `start_smiles`
- `fragment_op` 序列
- 中间状态（可选）

输出：

- `path_target`
- 每步 `step_target`

### Route B：状态价值模型 `V(s)`

输入：

- 当前分子状态 `s`
- 可选约束条件 / target property

输出：

- 从当前状态出发的预期剩余收益

### Route C：动作先验 `π(a|s)`

输入：

- 当前状态
- 候选动作列表

输出：

- 候选动作优先级分布

---

## 5. 任务拆解

### Task 1：整理多步训练数据

复用或扩展：

- `core/data/path_processing.py`

需要补齐：

- 路径长度分布
- 动作序列有效性掩码
- 中间状态缺失时的策略
- planner 轨迹回放数据导出

### Task 2：训练 value/path 模型

建议第一版目标：

- 预测 `path_target`
- 或预测当前状态的未来可达收益

并与“单步分数累加”做对比。

### Task 3：把 value 接入 planner

优先做：

- `best-first + value bonus`
- `A*` 中的轻量 `h(s)`

### Task 4：训练 policy / prior

目标不是直接生成最终分子，而是：

- 缩小候选动作集合
- 提高扩展优先级质量
- 降低无效分支比例

### Task 5：最后引入 RL

仅在以下前提下进行：

- planner 已稳定
- path/value 已有正收益
- 动作空间已基本固定

---

## 6. 建议代码落点

优先复用：

- `core/data/path_processing.py`
- `core/models/v1/*`

建议新增：

- `core/models/value/*`
- `core/models/policy/*`
- `core/planners/heuristics.py`

---

## 7. 验收标准

完成本阶段时，应满足：

- 至少一版 path/value 模型可训练、可验证
- planner 能调用 value 作为辅助分数
- 与纯单步评分相比，在部分任务上显示出更好的长程路径保留能力
- policy / prior 至少能减少一部分低价值扩展

---

## 8. 风险与注意事项

- 不要在没有稳定轨迹数据时急着上 RL
- 不要把 value 和 step delta 混为一谈
- 不要在 planner 仍不稳定时做过多 path/value 结论
- 不要忽略路径数据里的噪声与中间状态异常
