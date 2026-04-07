# 01：Baseline 平台计划（重构收口版）

> 目标：把当前 `batch_optimizer.py + EvolutionTreeOptimizer + OFO + BFS/MCTS` 收口为**可重复、可比较、可归档**的 baseline 平台，同时固定它的 benchmark、I/O 契约、结果目录与评估链，作为后续所有新路线的统一对照平台。

---

## 1. 为什么这一阶段必须先做

如果 baseline 没有被真正固定，后续任何收益都无法判断到底来自：

- 动作空间变化
- 数据口径变化
- 模型结构变化
- planner 变化
- 还是实验预算变化

因此这一阶段的重点不是继续调旧方法，而是：

> **把旧方法收口成可信的比较对象。**

---

## 2. 本阶段只做什么，不做什么

### 2.1 本阶段要做的事

- 固定 baseline benchmark 子集
- 固定 baseline 的最小推荐参数
- 固定输入输出与结果目录约定
- 固定评估脚本主入口与指标口径
- 形成后续所有实验共用的 baseline 平台契约

### 2.2 本阶段不做的事

- 不继续大规模调 BFS / MCTS 超参数
- 不把 fragment、新 planner、value、RL 混入 baseline 文档
- 不因为“看起来更强”就修改 baseline 的可比性

---

## 3. baseline 的角色定义

当前 baseline 应明确被视为：

- **历史方法的正式对照组**
- **后续新方法的统一评测入口**
- **结果目录与评估脚本契约来源**

而不应再被视为：

- 项目最终路线
- 未来主要研发投入方向
- 与 fragment 新路线并列竞争的长期主线

---

## 4. 文档边界与交付物

### 4.1 本阶段文档边界

本计划统一覆盖原来分散在“baseline 计划”和“I/O 规范”中的内容：

- baseline 角色与退出条件
- benchmark 与推荐运行口径
- 输入输出契约
- 结果目录与评估链契约

单独保留的只有：

- `01a-baseline-workflow-guide.md`：操作手册

### 4.2 工程交付物

- 一组固定的 benchmark 子集
- 一份 baseline 推荐运行模板
- 一套统一评估链主入口
- 一套可追溯的结果目录结构
- 一套可供后续路线兼容的 I/O 契约

### 4.3 结论交付物

最终应形成一页可复用的 baseline 结论，至少回答：

- 当前系统对哪些样本有效
- 当前系统的主要失败模式是什么
- primitive 动作空间的主要短板是什么
- 后续新路线至少要在哪些指标上超过 baseline

---

## 5. 主要任务拆解

### Task 1：冻结 baseline 入口

聚焦：

- `scripts/batch_optimizer.py`
- `core/evolution_optimizer.py`
- `core/molecular_evolution_expansion.py`

需要收口：

- 推荐 CLI 参数模板
- `search_mode` 的正式对比口径
- 默认预算与约束参数
- 模型权重、配置文件、数据文件的对应关系

### Task 2：固定 benchmark 子集

当前建议继续保留已有的 QM9 评测子集作为 baseline 对照入口，但要明确：

- baseline benchmark 是对照平台，不代表 fragment 主训练源
- benchmark 子集应稳定、可回放、可复用
- 后续所有新方法优先在同一子集上对比

### Task 3：定义统一指标

至少应固定三类指标：

- **效果指标**：Top-1 / Top-k 改善值、达到阈值改善比例
- **效率指标**：运行时间、展开节点数、单位预算收益
- **合理性指标**：合法分子率、唯一率、约束满足率、多样性

### Task 4：固定输入契约

对于 `batch_optimizer.py`，正式运行至少固定：

- 输入 CSV：`--input-csv`
- 模型权重：`--model-path`
- 模型目录：`--model-dir`
- 动作配置：`--config-file`

输入 CSV 最低字段：

- `smiles`
- 与 `--target-property` 同名的真实属性列

例如：

- 优化 `lumo` 时，CSV 至少有 `smiles, lumo`
- 优化 `homo` 时，CSV 至少有 `smiles, homo`

每次正式运行都应记录：

- `target_property`
- `direction`
- `optimization_mode`
- `max_depth`
- `max_branching`
- `pruning_patience`
- `logp_min / logp_max / logp_patience`
- `topK`
- `batch_size`
- `start_index / end_index`
- `search_mode`
- `num_simulations / exploration_weight`（若为 MCTS）

### Task 5：固定输出目录与关键产物

MO baseline 输出目录默认形如：

- `mol_evo/output/evo-mo/batch_optimization_<timestamp>/`

至少应保留：

- `batch_optimization_main.log`
- `batch_optimization_total.log`
- `batch_results.json`
- 每个输入分子的优化树 JSON
- 每个输入分子的 Top-k CSV

`batch_results.json` 应满足：

- 顶层能按输入样本回查
- 每个样本能关联原始输入信息
- 每个样本能关联优化结果、Top-k 结果和运行时间
- 后续新方法尽量保留同等聚合语义

Top-k CSV 应被视为：

- 单个起始分子的候选摘要表
- 后续评估链的主要输入之一

### Task 6：固定评估链契约

第一阶段评估主入口：

- `utils/evaluate_batch_mo.py`

职责：

- 消费 Top-k CSV
- 回查真实性质与辅助指标
- 输出逐分子评估结果 CSV
- 保存评估配置快照

第二阶段评估主入口：

- `utils/evaluate_csv_results.py`

职责：

- 消费逐分子评估结果 CSV
- 输出统计摘要、最佳结果表与可视化

关键要求不是字段名逐字一致，而是：

> **同一份结果能同时支持人工排查、脚本统计和跨方法对比。**

### Task 7：形成 baseline 结论页

这一页结论不是实验笔记，而是后续所有阶段的共同参照。

---

## 6. 与 `01a` 的关系

`01a-baseline-workflow-guide.md` 只回答：

- baseline 怎么跑
- baseline 怎么验
- baseline 结果怎么保存

而本文件回答：

- baseline 为什么先固定
- baseline 平台的输入输出契约是什么
- baseline 平台以什么口径做公平比较

---

## 7. 对后续阶段的接口要求

后续 `fragment_op`、`semantic_step`、新 planner、path/value 路线在工程上应尽量保持：

- 输入 CSV 兼容
- 结果目录可映射
- Top-k 摘要表可统一评估
- 聚合结果可统一比较

这样新路线才不是“另起一个实验宇宙”，而是真正站在 baseline 对照平台上演进。

---

## 8. 验收标准

完成本阶段时，应满足：

- 同一 benchmark 子集可重复运行
- baseline 输出目录结构稳定
- 评估链路主入口明确
- 输入输出契约固定
- 指标口径固定
- 后续新方法可以直接与 baseline 做公平对比

---

## 9. 风险与注意事项

- 不要把 baseline 写成“顺手还在演化的旧系统”
- 不要一边改输入输出，一边又拿旧结果做比较
- 不要用不一致的预算或约束条件制造伪提升
- 不要在 baseline 尚未固定前开始大规模新旧路线结论比较

---

## 10. 退出条件

当以下条件成立时，本阶段即可退出：

- 团队内部对 baseline 的角色有共识
- benchmark、结果目录和评估链都已固定
- 输入输出契约足够支撑后续新方法兼容接入
- baseline 结论页已足够支撑后续阶段对比
