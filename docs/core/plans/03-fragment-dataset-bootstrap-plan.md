# 03：基于 QM9 的新训练数据构造计划

> 目标：围绕 `mol_evo/dataset/data` 下的 QM9 数据，构建**适配新架构的 canonical 训练数据主线**；`fragment_alignment_utils.py` / `build_fragment_op_dataset.py` 仅保留为 bootstrap / fallback，而不再作为主数据生产方案。
>
> 如果当前关心的是“**新的 `OFO-frag` 单步训练 pair 具体怎么从 QM9 build 出来**”，请与 `03a-qm9-frag-pair-build-plan.md` 一起阅读；`03` 负责总纲，`03a` 负责单步 pair 的实施细化。

---

## 1. 先修正旧版本计划里的方向偏差

旧版本把本阶段描述成：

> 基于现有 pair/path 数据，做一轮片段动作重解释与 bootstrap。

这个表述适合 Phase B 的“原型验证”，但**不适合新架构的数据主线**。

因为新架构要解决的不是：

- 能不能把若干 pair 事后解释成 `fragment_op`

而是：

- 如何从 QM9 原始数据稳定地产生 **canonical pair/path 数据骨架**
- 如何在此基础上补齐属性标签、路径标签、fragment annotation
- 如何把这些主事实数据导出成训练视图、planner 视图、回放视图

所以新的主线应当改成：

> **QM9 raw CSV → canonical pair → property delta labels → canonical path normalization → fragment annotation sidecar → train/val/test export**

---

## 2. 本阶段的正确目标

本阶段要达成的不是“做出一份 fragment bootstrap 数据”，而是：

- 固定 QM9 原始数据入口
- 固定 canonical pair 数据格式
- 固定 canonical path 数据格式
- 固定属性标签补齐流程
- 固定 `fragment_op` 标注层与主骨架的关系
- 固定训练/验证/测试切分与导出规范

最终要让后续 `OFO-frag`、planner、path/value 模型都围绕同一条数据主链工作。

---

## 3. 数据主线总览

### Stage A：QM9 原始数据层

主数据源位于：

- `dataset/data/qm9_smiles_all_atoms.csv`
- `dataset/data/qm9_smiles_heavy_*_atoms.csv`

这些文件提供：

- `index`
- `smiles`
- `num_atoms`
- QM9 属性列（如 `homo / lumo / gap / ...`）

这一层是所有训练数据的唯一原始事实源。

### Stage B：canonical primitive pair 层

基于现有 `MoleculeEvolverAnalysis` / `analyze_evolution_operation_dict(...)`，从 QM9 生成：

- `smiles_from`
- `smiles_to`
- `operations`
- 可选的 `from_index / to_index`
- 可选的 `step` / `distance`

这一层的核心作用是：

- 先固定“分子之间的 canonical 编辑关系”
- 保留现有系统最成熟的 primitive 语义

### Stage C：属性标签层

在 canonical pair 上补齐：

- `target_property`
- `step_target`
- 多属性 `*_change`
- 多属性 `*_change_pct`

这一步是从结构变换样本走向可监督训练样本的关键桥梁。

### Stage D：canonical path 层

将 pair 或多步轨迹规范化成统一 path 协议：

- `path_id`
- `node_smiles_list`
- `operations`
- `start_smiles`
- `end_smiles`
- `target_property`
- `path_target`
- `step_targets`

这里的重点不是“再做一层 MCS 对齐”，而是：

- 修复 path 格式不一致
- 让路径数据可被 `core/data/path_processing.py` 稳定消费

### Stage E：fragment annotation sidecar 层

在 canonical pair/path 之上，为每一个 `operations[i]` 增加：

- `fragment_op`
- 或 `operation_annotations[i]`
- 或统一写回兼容版 `operations[i]`

这是**标注层**，不是主骨架。

### Stage F：训练视图导出层

从 canonical 主链导出多个视图：

- **pair 视图**：给单步 `OFO-frag`
- **path 视图**：给 path/value 模型
- **planner replay 视图**：给策略学习与轨迹分析
- **action library 视图**：给候选动作库与统计

---

## 4. 为什么不再把 MCS bootstrap 当主方案

`fragment_alignment_utils.py` / `build_fragment_op_dataset.py` 的问题不在于“不能用”，而在于它们更像：

- 事后解释器
- 弱标注器
- 局部对照工具

它们不适合作为主方案，主要因为：

### 4.1 主语义在 `operations`，不在 MCS 对齐

当前项目最稳定、最可复用的“动作事实”来自：

- `MoleculeEvolverAnalysis`
- `analyze_evolution_operation_dict(...)`
- `operations` 序列

如果主线反过来变成“先 MCS、后猜动作”，会把 canonical 语义降级成旁证。

### 4.2 解释质量天然分层

这类脚本往往会产生：

- `ok`
- `approximate`
- `complex`
- `invalid`

它适合提供：

- 置信度
- fallback 标注
- 调试案例

但不适合直接充当唯一主训练集生成器。

### 4.3 不利于 path 数据收口

path 的本质是：

- 有序节点序列
- 有序动作序列
- 路径级/步骤级标签

MCS 重解释更擅长 pair 层局部差异分析，不擅长定义 canonical path 协议。

### 4.4 容易放大数据切分问题

如果只是对记录随机切分，会造成：

- 相近分子泄露到不同 split
- 相同 scaffold 在 train/test 之间过度重叠
- 评估过于乐观

新主线必须把 split 设计前置，而不是在 bootstrap 输出后随机拆分。

---

## 5. 推荐的新脚本分层

不要求当前一次性全部实现，但后续脚本职责应按下面拆分。

### 5.1 `dataset/build_canonical_qm9_pairs.py`

职责：

- 从 `qm9_smiles_all_atoms.csv` 或 `qm9_smiles_heavy_*_atoms.csv` 读取 QM9
- 调用现有 evolver / operation analysis 逻辑
- 生成 canonical pair 数据
- 输出统一的 pair JSON/JSONL

可优先复用/收口现有：

- `dataset/extract_evolution_pairs.py`
- `dataset/extract_evolution_pairs_v1.py`

### 5.2 `dataset/annotate_pair_property_deltas.py`

职责：

- 从 canonical pair 中补齐 QM9 属性变化标签
- 输出 `*_change` / `*_change_pct`
- 固定 `target_property` / `step_target` 的组织方式

可优先复用/收口现有：

- `dataset/calculate_property_changes.py`

### 5.3 `dataset/normalize_canonical_paths.py`

职责：

- 把 pair 或多步轨迹转换成统一 path 协议
- 统一输出 `node_smiles_list / operations / path_target / step_targets`
- 修复当前 path 生产与 path 消费格式不一致问题

这是新主线必须新增或重构的关键脚本。

### 5.4 `dataset/annotate_fragment_ops.py`

职责：

- 基于 canonical `operations` 和前后分子结构
- 为每个 step 生成 `fragment_op` 标注
- 写入 `operations[i]` 或 `operation_annotations[i]`
- 对无法精确解释的样本标注 `resolved / approximate / unresolved`

这里可以按优先级使用：

1. canonical op + 局部规则映射
2. 规则动作库辅助补齐
3. `fragment_alignment_utils.py` 作为 fallback

### 5.5 `dataset/export_training_views.py`

职责：

- 从 canonical + fragment annotation 主链导出训练视图
- 输出 pair/path/planner replay/actionlib 等不同视图
- 统一 train/valid/test split
- 产出统计文件与配置文件

### 5.6 `dataset/build_split_manifest.py`

职责：

- 生成 molecule-level / scaffold-level split 清单
- 保证各训练视图共用同一套 split manifest
- 避免 record-level random split 带来的数据泄露

---

## 6. 推荐的数据协议

### 6.1 Canonical pair 协议

建议第一版 pair 样本至少包含：

- `pair_id`
- `smiles_from`
- `smiles_to`
- `from_index`
- `to_index`
- `operations`
- `num_steps`
- `target_property`
- `step_target`
- `property_changes`
- `meta`

其中 `meta` 建议至少记录：

- `source_dataset = qm9`
- `generation_version`
- `annotation_status`
- `split_key`

### 6.2 Canonical path 协议

建议第一版 path 样本至少包含：

- `path_id`
- `node_smiles_list`
- `node_indices`
- `operations`
- `start_smiles`
- `end_smiles`
- `target_property`
- `path_target`
- `step_targets`
- `meta`

### 6.3 Fragment annotation 协议

推荐两种兼容写法，二选一即可：

- **方式 A**：直接把 `operations[i]` 升级成 fragment-aware dict
- **方式 B**：保留原 `operations[i]`，新增 `operation_annotations[i]`

如果以落地成本优先，推荐：

- 第一版保留 primitive 字段
- 同时新增 fragment 字段
- 逐步过渡到完整 `fragment_op`

---

## 7. 推荐的数据构造流程

### Step 1：固定 QM9 输入清单

明确使用：

- `qm9_smiles_all_atoms.csv` 作为全量入口
- `qm9_smiles_heavy_*_atoms.csv` 作为按重原子桶的辅助输入

### Step 2：生成 canonical pairs

按 step 或编辑距离约束生成：

- 单步 pair
- 多步 pair
- 带 `operations` 的 canonical 样本

### Step 3：补齐属性标签

对每个 pair 补齐：

- 所有可用 QM9 属性变化
- 训练关心的主目标属性标签
- 统一命名与归一化前原值

### Step 4：规范化 path 数据

把 path 层统一到当前 `path_processing.py` 期望的格式，重点解决：

- `nodes / edges` 风格输出
- `node_smiles_list / operations / step_targets` 风格消费

之间的断层。

### Step 5：做 fragment annotation

这一步是对 canonical step 的语义增强，而不是重新定义主事实数据：

- 能精确映射则标为 `resolved`
- 只能近似映射则标为 `approximate`
- 无法映射则标为 `unresolved`

### Step 6：做 split manifest

优先采用：

- molecule-level split
- 或 scaffold-level split

不推荐继续使用：

- 纯 record-level random split

### Step 7：导出训练视图

至少导出：

- `canonical_pairs_{train,valid,test}.jsonl`
- `canonical_paths_{train,valid,test}.jsonl`
- `fragment_pairs_{train,valid,test}.jsonl`
- `fragment_paths_{train,valid,test}.jsonl`
- `planner_replay_{train,valid,test}.jsonl`
- `split_manifest.json`
- `dataset_stats.json`
- `action_config.yaml`

其中 `fragment_*` 是导出视图，不是唯一主事实源。

---

## 8. 与现有代码的关系

### 8.1 可以直接复用/收口的部分

优先复用：

- `dataset/extract_smiles.py`
- `dataset/dev-tools/merge_qm9_files.py`
- `dataset/extract_evolution_pairs.py`
- `dataset/extract_evolution_pairs_v1.py`
- `dataset/calculate_property_changes.py`
- `dataset/extract_operation_config.py`
- `core/data/path_processing.py`

### 8.2 当前需要重点修复的问题

需要重点处理：

- `pair_to_path.py` 的输出格式与 `path_processing.py` 不一致
- split 仍偏向 record random split
- fragment 标注层还没有和 canonical `operations` 真正打通

### 8.3 降级为 bootstrap / fallback 的部分

以下脚本保留，但不再作为主生产线：

- `dataset/fragment_alignment_utils.py`
- `dataset/build_fragment_op_dataset.py`

它们更适合：

- 原型验证
- 对照实验
- 覆盖率分析
- fallback 标注

---

## 9. 验收标准

完成本阶段时，应满足：

- QM9 原始数据入口固定
- canonical pair 构造脚本固定
- property delta 补齐流程固定
- canonical path 协议固定并能被 `path_processing.py` 消费
- `fragment_op` 被明确为 annotation sidecar / operation upgrade，而非平行主协议
- split manifest 采用 molecule/scaffold 级策略
- 至少能稳定导出一版 pair/path/fragment/planner replay 训练视图

---

## 10. 风险与注意事项

- 不要把 bootstrap 输出误当成主事实数据
- 不要让 fragment 视图脱离 canonical pair/path 骨架单独演化
- 不要在 path 协议尚未收口前急着堆更多下游模型
- 不要继续沿用随机 record split 作为默认方案
- 不要把 `fragment_alignment_utils.py` 的高覆盖率误解为“主方案已经成立”

---

## 11. 本阶段一句话结论

> **新架构的数据集构造主线，应建立在 QM9 canonical pair/path 数据骨架之上；`fragment_op` 是主骨架上的结构化动作标注层，而 `fragment_alignment_utils.py` / `build_fragment_op_dataset.py` 只应保留为 bootstrap / fallback。**
