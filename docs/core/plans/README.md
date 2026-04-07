# 分步实施计划总览

> 本目录用于承接 `mol-ofo` 项目从“primitive graph edit + OFO 单步打分 + BFS/MCTS”向“fragment-level action + OFO-frag + planning + value/path model”演进的逐步计划。
>
> 建议结合以下文档一起阅读：
> - `../batch-optimizer-framework.md`
> - `../ofo-model-framework.md`
> - `../mo-direction-recommendation.md`

---

## 1. 目录目标

这组计划不追求一次性重构，而是强调：

1. **先保留当前 baseline 能力**
2. **再升级动作空间**
3. **再升级 OFO 的动作表示**
4. **再升级搜索器**
5. **最后进入路径级 / value 级建模**

也就是说，推进顺序应遵循：

> **baseline 固化 → fragment_op 定义 → 数据构造 → OFO-frag → planner → path/value → 实验闭环**

---

## 2. 文件顺序与依赖关系

| 编号 | 文件 | 主要目标 | 是否前置依赖 |
|------|------|----------|--------------|
| 01 | `01-baseline-and-benchmark-plan.md` | 固化当前 baseline、评测与日志 | 无 |
| 02 | `02-fragment-op-schema-plan.md` | 定义 `fragment_op` schema 与动作库 | 01 |
| 03 | `03-fragment-dataset-bootstrap-plan.md` | 固定 QM9 canonical 数据主线与导出分层 | 01, 02 |
| 03a | `03a-qm9-frag-pair-build-plan.md` | 细化 `OFO-frag` 所需单步 `frag pair` 的构造方案 | 01, 02, 03 |
| 04 | `04-ofo-frag-model-plan.md` | 升级 OFO 为片段动作条件打分器 | 02, 03, 03a |
| 05 | `05-planner-upgrade-plan.md` | 以 `beam / best-first` 取代粗粒度层截断 | 01, 04 |
| 06 | `06-path-value-and-rl-plan.md` | 引入路径级 / value / policy 建模 | 04, 05 |
| 07 | `07-experiment-and-milestone-plan.md` | 组织对比实验、里程碑与验收 | 贯穿全程 |

---

## 3. 进度同步看板

> 说明：本清单用于同步 `plans/0x` 各阶段的推进状态。只有**明确已落地的交付物**才勾选；如果只是“已讨论 / 已写计划”，仍保持未完成。

### Phase A：先站稳已有系统

对应文件：

- `01-baseline-and-benchmark-plan.md`
- `01a-baseline-workflow-guide.md`
- `01b-baseline-io-spec.md`

进度清单：

- [x] 已补齐正式 baseline 操作手册（`01a-baseline-workflow-guide.md`）
- [x] 已补齐 baseline 输入输出与结果结构规范（`01b-baseline-io-spec.md`）
- [x] 已固定当前默认 baseline benchmark 子集（`qm9_test_molecules.csv` 前 50 个 case）
- [ ] 固定 baseline 指标定义与统计口径
- [ ] 产出一组可重复的 baseline 运行结果目录
- [ ] 形成 baseline 结论页（适用范围、失败模式、后续对比目标）

### Phase B：先解决“动作是什么”

对应文件：

- `02-fragment-op-schema-plan.md`
- `03-fragment-dataset-bootstrap-plan.md`
- `03a-qm9-frag-pair-build-plan.md`

进度清单：

- [x] 明确 `fragment_op` 正式 schema 主字段
- [x] 明确 `scaffold_preserving` 的第一版判定规则
- [x] 形成 `fragment_op` 小词表 / 动作词表 v0
- [x] 输出 `fragment_op.schema.json` 与样例动作文件
- [x] 补齐动作合法性校验逻辑
- [x] 建立 `fragment_op` 数据构造脚本原型
- [x] 固定 QM9 `frag pair` 候选筛选规则、样本协议与导出字段（`03a-qm9-frag-pair-build-plan.md`）
- [ ] 生成第一版 `canonical_pairs_raw.jsonl`
- [ ] 生成第一版 `canonical_pairs_labeled.jsonl`
- [ ] 生成第一版 `fragment_pairs_{train,valid,test}.jsonl`
- [ ] 生成第一版 `fragment_paths_train.jsonl` 与 `fragment_dataset_stats.json`

### Phase C：再解决“模型怎么学这个动作”

对应文件：

- `04-ofo-frag-model-plan.md`

进度清单：

- [ ] 新增 `prepare_fragment_op_features()`
- [ ] 定义 `fragment_op` 相关特征维度与配置项
- [ ] 跑通兼容版 `edge_attr` / `EdgeFeatureExtractor` 升级
- [ ] 增加至少一版 `OFO-frag` 模型变体
- [ ] 跑通第一版片段动作单步训练
- [ ] 产出训练日志、验证结果与样例分析

### Phase D：最后解决“怎么规划”

对应文件：

- `05-planner-upgrade-plan.md`
- `06-path-value-and-rl-plan.md`

进度清单：

- [ ] 抽象统一 planner 接口
- [ ] 实现 `beam search` 原型
- [ ] 实现 `best-first search` 原型
- [ ] 接入约束惩罚、多样性与预算控制
- [ ] 完成 `fragment + BFS / beam / best-first` 对比
- [ ] 整理多步训练数据并跑通 path/value 原型
- [ ] 将 value / heuristic 接入 planner
- [ ] 训练 policy / prior 原型
- [ ] 在有稳定收益前不进入 RL 主实验

### Phase E：实验闭环

对应文件：

- `07-experiment-and-milestone-plan.md`

进度清单：

- [ ] 固定最小实验矩阵（`E1` 到 `E5`）
- [ ] 固定统一评测指标与预算口径
- [ ] 输出 `benchmark_results.csv`
- [ ] 输出 `ablation_results.csv`
- [ ] 输出 `planner_budget_analysis.csv`
- [ ] 输出 `case_studies.md`
- [ ] 输出 `milestone_summary.md`
- [ ] 给出“新路线是否显著优于 primitive baseline”的阶段性结论

---

## 4. 文档边界与阅读方式

为避免不同计划文件互相复述，建议按下面的边界理解和维护：

- **`01`**：回答“为什么要先固化 baseline、验收标准是什么”
- **`01a`**：回答“baseline 具体怎么跑”
- **`01b`**：回答“baseline 输入输出长什么样、结果如何对齐”
- **`02`**：只负责定义 `fragment_op` 的概念边界、schema、词表与校验规则
- **`03`**：负责 QM9 canonical 数据主线、property labels、path normalization 与训练视图导出分层
- **`03a`**：只负责把 `03` 进一步细化为 `OFO-frag` 可直接消费的单步 `frag pair` 构造计划
- **`04`**：只负责 `OFO-frag` 的特征编码与模型改造，默认训练入口已由 `03a` 提供
- **`05`**：只负责 planner 机制与搜索预算分配，不展开 value / policy 训练细节
- **`06`**：只负责 path / value / policy 与 heuristic，不重复 `beam / best-first` 的基础接口设计
- **`07`**：只负责跨阶段实验矩阵、指标、里程碑与结论汇总

建议阅读顺序：

1. 先看 `README.md` 把握阶段依赖与当前进度
2. 进入某阶段前，先读该阶段的主计划文件
3. 若是 Phase 1，再根据需要跳到 `01a` 或 `01b`
4. 做实验对比时，统一回到 `07-experiment-and-milestone-plan.md`

这样可以把“总览 / 执行手册 / 规范 / 实验设计”四类信息拆开，减少维护时的重复修改。

---

## 5. 建议的落地原则

- **先做兼容式增量改造**：不要一开始推倒 `batch_optimizer.py` / `EvolutionTreeOptimizer`
- **先做弱版本再做强版本**：先上 `fragment fingerprint`，再考虑 learned fragment encoder
- **先做可解释动作，再做复杂策略**：先把 `fragment_op` 定义清楚，再引入 RL
- **先把路径数据跑通，再谈 value**：没有稳定多步样本，value model 很容易空转
- **每一阶段都要有可验收产物**：schema、数据集、模型 checkpoint、planner 结果、对比表格都要落盘

---

## 6. 本目录的使用方式

建议每完成一个计划文件里的核心交付物，就：

1. 更新相应实现或实验脚本
2. 在该计划文件中补充“已完成 / 待验证 / 风险”
3. 将关键结论回写到 `../mo-direction-recommendation.md` 或独立实验文档

这样这组计划文件就不只是路线图，而会逐渐变成真正的项目推进记录。
