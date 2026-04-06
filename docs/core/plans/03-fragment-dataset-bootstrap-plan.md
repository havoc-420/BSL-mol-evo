# 03：片段动作数据构造与 Bootstrap 计划

> 目标：从现有 pair/path 数据中构造 `fragment_op` 样本，产出可以训练 `OFO-frag` 的数据集，而不是停留在概念层。

---

## 1. 为什么这是关键阶段

`fragment_op` 如果只有 schema 没有数据，就无法训练真正的片段级转移打分器。当前项目已经有：

- `(smiles_from, smiles_to)` pair 样本
- `operations` 列表式路径数据
- 部分路径级处理逻辑

因此最合理的策略不是重新开始采数据，而是：

> **基于现有 pair/path 数据，做一轮片段动作重解释与 bootstrap。**

---

## 2. 本阶段目标

- 把现有样本重新映射为 `fragment_op`
- 产出第一版 `fragment_op` 监督数据
- 统计样本覆盖率、动作分布、失败率
- 为后续 `OFO-frag` 训练提供正式数据入口

---

## 3. 数据来源优先级

### Source A：现有 pair 数据

适合构造：

- 单步 `from -> to`
- 单步属性变化标签 `delta`
- 单步 fragment replacement / attach 样本

### Source B：现有 path 数据

适合构造：

- 多步 `start -> ... -> end`
- `step_targets`
- `path_target`
- 后续 value/path 模型轨迹样本

### Source C：规则自动生成数据

适合作为 bootstrap 弱标签：

- BRICS attach 候选
- R-group swap 候选
- scaffold 保持型 grow/prune 候选

---

## 4. 推荐任务拆解

### Task 1：pair 对齐与局部差异识别

建议做：

- Murcko scaffold 对齐
- MCS 对齐
- R-group decomposition
- 局部替换片段识别

输出：

- `diff_type`
- `anchor`
- `fragment_in`
- `fragment_out`
- 是否 `scaffold_preserving`

### Task 2：把 pair 转成 `fragment_op`

对于能解释成单步片段动作的样本，输出：

- `from_smiles`
- `to_smiles`
- `fragment_op`
- `target_property`
- `step_delta`

### Task 3：处理无法直接解释的样本

建议分成三类：

- **可近似**：拆成一条主片段动作 + 若干忽略扰动
- **复杂样本**：标记为 `complex_multi_change`
- **不可用样本**：直接丢弃

### Task 4：从 path 数据中抽多步样本

利用已有路径数据结构，生成：

- 每步 `fragment_op`
- 每步 `step_target`
- 整体 `path_target`
- 动作序列长度

### Task 5：生成训练所需统计信息

需要统计：

- 每类 `op_type` 数量
- scaffold 保持比例
- 各动作的 size/ring/hetero delta 分布
- 数据解释成功率
- 被丢弃样本比例

---

## 5. 推荐输出格式

建议统一生成：

- `fragment_pairs_train.jsonl`
- `fragment_pairs_valid.jsonl`
- `fragment_pairs_test.jsonl`
- `fragment_paths_train.jsonl`
- `fragment_dataset_stats.json`
- `fragment_action_config.yaml`

每条单步样本至少包含：

- `smiles_from`
- `smiles_to`
- `fragment_op`
- `target_property`
- `step_target`
- `meta`

---

## 6. 建议依赖代码位置

优先复用或改造：

- `dataset/extract_evolution_pairs.py`
- `core/data/path_processing.py`
- `core/data/processing.py`
- `core/utils/fragnet_data/*`

必要时新增：

- `dataset/build_fragment_op_dataset.py`
- `dataset/fragment_alignment_utils.py`

---

## 7. 验收标准

完成本阶段时，应满足：

- 能稳定生成一版 `fragment_op` 单步数据集
- 能稳定生成一版多步路径数据集
- 数据解释覆盖率可统计、可复查
- 至少能产出 1 个小规模可训练集用于原型验证

---

## 8. 风险与注意事项

- 不要假设所有 pair 都能解释成干净的单步片段动作
- 不要把复杂多处改动强行贴成单一 `fragment_op`
- 不要忽略数据质量报告
- 不要只生成训练集，不生成验证/测试集
