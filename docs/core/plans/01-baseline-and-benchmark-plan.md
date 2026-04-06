# 01：Baseline 固化与评测基线计划

> 目标：把当前 `batch_optimizer.py + EvolutionTreeOptimizer + OFO + BFS/MCTS` 体系明确降级为 baseline，但把它固定成后续所有新方法的对照平台。

---

## 1. 为什么先做这个

如果没有稳定 baseline，后面引入 `fragment_op`、`OFO-frag`、`beam search` 后，结果很难判断到底是：

- 动作空间变好了
- 搜索器变好了
- 模型变好了
- 还是实验口径变了

因此第一阶段的工作重点不是“继续调当前方法”，而是：

> **把当前方法固定成可信的比较对象。**

---

## 2. 本阶段目标

### 2.1 功能目标

- 固定一套 baseline 数据输入格式
- 固定一套 baseline 输出格式
- 固定一组主要评测指标
- 固定一组推荐超参数设置
- 确保同一输入分子在同一配置下结果可复现实验趋势

### 2.2 工程目标

- 明确 `batch_optimizer.py` 的输入约定
- 明确 `EvolutionTreeOptimizer.predict_batch()` 的行为约定
- 明确 BFS / MCTS 的日志与结果落盘位置
- 为后续新 planner 保留兼容接口

---

## 3. 建议交付物

当前已补齐的正式文档：

- `01a-baseline-workflow-guide.md`：正式 baseline 操作手册
- `01b-baseline-io-spec.md`：输入输出与结果结构规范

当前已存在的评估模块基础：

- `utils/evaluate_batch_mo.py`：消费 `batch_optimizer.py` 产出的 `*_topK.csv`，生成 `batch_evaluation_results.csv`
- `utils/evaluate_csv_results.py`：对评估 CSV 做汇总统计、最佳结果导出与可视化
- `utils/calculate_homo_lumo.py`、`utils/HamDiv/*`、`utils/core/molecular_graph.py`：提供真值性质、多样性、Morgan/GED 等评估能力

当前已固定的默认 benchmark 子集：

- `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`：当前默认评测源文件
- `--start-index 0 --end-index 50`：默认使用前 50 个 case 作为 Phase 1 baseline 对照子集

后续仍建议补齐：

- `baseline_metrics.md`：指标定义与统计口径
- 一组可重复 baseline 运行结果目录
- 若后续需要发布版或分层 benchmark，再单独导出 `baseline_benchmark_set.csv`

也就是说，**Phase 1 的“评估脚本基础”和“默认 benchmark 子集”已经有了，但“指标口径正式化、结果归档标准化”仍未完成。**

---

## 4. 主要任务拆解

### Task 1：冻结 baseline 入口与配置

聚焦文件：

- `scripts/batch_optimizer.py`
- `core/evolution_optimizer.py`
- `core/molecular_evolution_expansion.py`

需要完成：

- 固定推荐 CLI 参数模板
- 固定 `search_mode` 的对比口径（至少 BFS / MCTS）
- 固定 `max_depth / max_branching / pruning_patience / logP range`
- 确认预测值是否已经反标准化

### Task 2：建立 benchmark 分子集

当前已经可以先固定使用：

- `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`
- 通过 `--start-index 0 --end-index 50` 取默认前 50 个 case

这已经足够作为 Phase 1 的当前 baseline 对照子集。

如果后续要把 benchmark 做成更正式的发布物，再进一步构建三类输入：

- **容易优化样本**：当前系统能明显改善
- **中等难度样本**：当前系统偶尔有效
- **困难样本**：当前系统容易卡住或搜索爆炸

每类建议保留 20~50 个分子，形成一个更稳定、可解释的分层对照集。

### Task 3：定义统一指标

建议至少统计：

- Top-1 改善值
- Top-k 最优改善值
- 有效候选率
- 合法分子率
- 唯一分子率
- 平均搜索节点数
- 平均运行时间
- `logP` / SA / 多样性 等约束统计

### Task 4：补齐结果产物规范

统一要求：

- 每次运行都保存配置快照
- 每个输入分子都保存优化树 JSON
- 每个输入分子都保存 Top-k CSV
- 批量层面保存聚合 JSON
- 为后续 planner 新增字段预留扩展位

### Task 5：写出 baseline 结论页

内容建议包括：

- 当前系统适合什么类型分子
- 当前系统主要失败模式是什么
- primitive action 的典型问题是什么
- 后续新方法应以哪些指标超过 baseline

---

## 5. 验收标准

完成本阶段时，应满足：

- 能稳定跑完一套固定 benchmark 集
- 能输出统一格式的结果目录
- 能在一页表格里汇总 baseline 指标
- 后续任何新方法都可以直接和这套 baseline 结果做对比

---

## 6. 风险与注意事项

- 不要继续在这一阶段大量调 BFS/MCTS 超参数
- 不要把 baseline 改成和原方法不可比较
- 不要把实验波动误当成方法提升
- 不要在没有固定 benchmark 的情况下开始新方法对比

---

## 7. 退出条件

当以下条件全部满足时，可以进入下一阶段：

- baseline 评测集已经固定
- baseline 指标统计脚本可复用
- baseline 结果目录结构稳定
- 团队内部对“当前方法的上限和短板”已有共识
