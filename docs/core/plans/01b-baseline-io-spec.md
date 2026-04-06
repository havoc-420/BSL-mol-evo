# 01B：Baseline 输入输出与结果结构规范

> 本文档用于补齐 Phase 1 的正式规范：baseline 到底吃什么输入、产出什么结果、日志放在哪里、后续实验该如何对齐。

---

## 1. 文档范围

本规范主要覆盖：

- `train_v0.py`
- `predict_v0.py`
- `scripts/batch_optimizer.py`
- `utils/evaluate_batch_mo.py`
- `utils/evaluate_csv_results.py`
- `utils/calculate_homo_lumo.py`

其中最关键的是：

> **`batch_optimizer.py` 的输入 CSV、输出目录和结果文件结构，以及 `utils` 评估链如何消费这些产物。**

---

## 2. v0 训练阶段输入输出

### 2.1 输入

`train_v0.py` 的核心输入包括：

- `--data-file`：pair 数据文件
- `--target-property`：训练目标属性
- `--model-type` 或 `--config-file`
- `--max-pairs`、`--epochs`、`--batch-size`、`--learning-rate`、`--seed`

### 2.2 输出

训练目录位于：

- `mol_evo/output/v0/<model_type>/train-<timestamp>-<target>-<max_pairs>-<epochs>/`

建议视为 baseline 关键产物的文件包括：

- `last.pth`
- 训练日志
- 训练配置快照
- 可能的最佳模型与评估记录

### 2.3 约定

- baseline 对比时，应尽量固定 `seed`
- baseline 对比时，应记录 `data-file` 与 `config-file` 的对应关系
- checkpoint 对比时，应避免混用不同数据口径训练出的模型

---

## 3. v0 预测阶段输入输出

### 3.1 单条预测输入

需要提供：

- `--model-path`
- `--model-dir`
- `--smiles-from`
- `--smiles-to`
- `--atom-symbol`
- `--operation-type`

### 3.2 批量预测输入

可提供：

- `--json-file`
- `--indices-file`
- `--use-test-indices`
- `--num-samples`
- `--sample-method`
- `--config-file`

### 3.3 用途

这部分更适合做：

- 单步模型 sanity check
- checkpoint 快速验证
- 若干样本的误差检查

而不适合作为 MO 结果对比本身。

---

## 4. MO baseline 输入规范

### 4.1 主入口

`batch_optimizer.py`

### 4.2 必要输入文件

必须提供：

- 输入 CSV：通过 `--input-csv`
- 模型权重：通过 `--model-path`
- 模型目录：通过 `--model-dir`
- 动作配置：通过 `--config-file`

### 4.3 输入 CSV 最低字段要求

至少应包含：

- `smiles`
- 与 `--target-property` 同名的属性列

例如：

- 若 `--target-property lumo`，则 CSV 至少有：`smiles`, `lumo`
- 若 `--target-property homo`，则 CSV 至少有：`smiles`, `homo`

当前 Phase 1 默认约定：

- `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv` 作为默认评测源文件
- 通过 `--start-index 0 --end-index 50` 选择前 50 个 case 作为当前 baseline 对照子集

### 4.4 常用运行参数

建议记录以下参数作为 baseline 配置快照的一部分：

- `target_property`
- `optimization_mode`
- `max_depth`
- `max_branching`
- `direction`
- `pruning_patience`
- `logp_min`
- `logp_max`
- `logp_patience`
- `topK`
- `batch_size`
- `start_index`
- `end_index`
- `search_mode`
- `num_simulations`（若为 MCTS）
- `exploration_weight`（若为 MCTS）

---

## 5. MO baseline 输出目录规范

### 5.1 输出目录位置

默认由 `create_output_dir()` 生成：

- `mol_evo/output/evo-mo/batch_optimization_<timestamp>/`

### 5.2 目录内关键文件

至少包括：

- `batch_optimization_main.log`
- `batch_optimization_total.log`
- `batch_results.json`
- 每个输入分子的优化树 JSON
- 每个输入分子的 Top-k CSV

### 5.3 单分子文件命名规则

当前实现中，每个输入分子会生成：

- `<smiles_prefix>_<run_id>.json`
- `<smiles_prefix>_<run_id>_topK.csv`

其中：

- `smiles_prefix` 来自前 20 个字符，并对 `/` 做替换
- `run_id` 为 UUID 前 8 位

这意味着：

- 文件名适合机器生成与批量保存
- 但不适合直接作为稳定样本 ID
- 若后续做 benchmark 对齐，建议额外维护输入样本索引表

---

## 6. `batch_results.json` 结构约定

批量结果 JSON 是整个 baseline 最重要的聚合产物。

### 6.1 顶层结构

顶层是一个以输入 `smiles` 为 key 的字典。

### 6.2 每个分子条目建议关注的字段

- `original_data`
  - 原始 CSV 行内容
- `optimization_result`
  - `status`
  - `smiles`
  - `initial_property`
  - `optimized_result`
  - `topk_results`
  - `runtime`

### 6.3 含义

- `optimized_result`：从单分子 JSON 回读的优化树结果
- `topk_results`：`get_topK_results()` 的结构化结果
- `runtime`：单分子处理耗时（秒）

---

## 7. 日志文件规范

### 7.1 `batch_optimization_main.log`

作用：

- 记录整次批量运行的配置
- 记录开始/结束时间
- 记录总体成功/失败统计

应视为：

- **运行配置快照**
- **批量实验摘要日志**

### 7.2 `batch_optimization_total.log`

作用：

- 记录每个分子的处理进度
- 记录单分子状态与耗时
- 记录 Top-k 数量等中间信息

应视为：

- **逐分子处理日志**

---

## 8. Top-k CSV 的角色

Top-k CSV 是后续评估脚本的主要输入之一。

建议把它理解为：

- 单个起始分子的候选摘要表

通常会包含：

- 起始分子
- 起始属性值
- 若干候选分子
- 候选预测值 / 排名信息

由于不同实现版本可能有列名细节差异，建议后续在评测脚本里统一读取接口，而不要在多个地方手写列名假设。

---

## 9. 评估脚本约定

当前更推荐的 baseline 评估链路为：

- `utils/evaluate_batch_mo.py`
- `utils/evaluate_csv_results.py`

### 9.1 第一阶段评估：`utils/evaluate_batch_mo.py`

职责：

- 从 `*_topK.csv` 中抽取起始分子与候选分子
- 计算真实属性或辅助属性
- 生成逐分子评估结果 CSV
- 保存评估配置快照

输入：

- `mol_evo/output/evo-mo/<batch_run_dir>/` 目录

输出：

- `.evaluation_results_<target>_<direction>/<eval_run_dir>/evaluation_config.json`
- `.evaluation_results_<target>_<direction>/<eval_run_dir>/batch_evaluation_results.csv`

### 9.2 第二阶段评估：`utils/evaluate_csv_results.py`

职责：

- 消费 `batch_evaluation_results.csv`
- 生成统计摘要、最佳结果表和可视化

输出通常包括：

- `statistics_report_<timestamp>.txt`
- `statistics_summary_<timestamp>.json`
- `best_results_<timestamp>.csv`
- `evaluation_plots_<target>_<timestamp>.png`
- `evaluation_results_<target>_<timestamp>.json`

### 9.3 当前限制

这条评估链已经能服务 baseline，但仍有几个 Phase 1 尚未完全收口的问题：

- 环境依赖尚未完全统一到 `mol-opt-evo`
- `gap` 的第二阶段汇总入口还不完整
- `有效候选率 / 合法分子率 / 唯一率 / SA / runtime / node count` 等指标还没有全部进入统一摘要
- `mol_evo/evaluate_batch_mo.py` 仍保留在仓库中，但更适合作为旧版/简化版脚本，而不是正式主入口

---

## 10. Phase 1 推荐保存的最小产物集

每一次正式 baseline 运行，建议至少保留：

- 输入 CSV 副本或版本号
- 模型 checkpoint 路径
- 配置 YAML 路径
- 完整 CLI 命令
- 输出目录
- `batch_results.json`
- `batch_optimization_main.log`
- 一组 Top-k CSV
- 汇总评估结果

---

## 11. 对后续阶段的接口要求

这份规范不仅服务 Phase 1，也应该为后续 `fragment_op` / planner 升级预留兼容性。

因此建议后续所有新方法尽量保持：

- 输入 CSV 结构兼容
- `batch_results.json` 顶层结构兼容
- Top-k CSV 可被统一评估脚本消费
- 日志目录结构尽量兼容

这样后续新方法才能真正和当前 baseline 做公平对比。
