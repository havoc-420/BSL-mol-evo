# 02：双层动作协议与 `fragment_op` Schema 计划（重构版）

> 目标：把项目中的“动作”从单层 primitive dict，重构为**事实层 primitive trace + 语义层 semantic step / fragment_op** 的统一协议，让离线数据、单步模型、planner 和 replay 共用同一套动作对象。

---

## 1. 为什么这一阶段必须先重写

当前系统里最稳定的动作事实来自：

- `primitive operations`
- 可重放的 `evo-path`
- `MoleculeEvolverAnalysis` 的构建序列

而后续新路线真正需要的是：

- 可表达局部片段语义的 `fragment_op`
- 可供模型和 planner 共用的动作对象
- 不会丢失可执行轨迹与可审计性的表示层

因此新协议不能简单理解为：

- 把 `operation + atom + position` 改名成 `fragment_op`

而应理解为：

> **保留 primitive trace 作为事实层，在其上建立 semantic step 作为训练和规划层。**

---

## 2. 新协议的核心抽象

### 2.1 `primitive_trace`

`primitive_trace` 是事实层。

它回答的是：

- 这一段变化具体由哪些原子级步骤组成
- 这些步骤能否回放
- 哪些 RDKit / MCS / anchor 信息能稳定对应回去

它不负责提供高层化学语义。

### 2.2 `semantic_step`

`semantic_step` 是训练与规划层。

它回答的是：

- 这段 primitive 变化在语义上算什么动作
- 应该被模型当成一个 step 还是多个 step
- planner 在扩展节点时该把它当成什么候选动作

### 2.3 `fragment_op`

`fragment_op` 是 `semantic_step` 的一种高语义载荷。

也就是说：

- 所有 `fragment_op` 都属于 `semantic_step`
- 但不是所有 `semantic_step` 都必须成功解释成 `fragment_op`

当无法稳定压缩成片段动作时，应允许：

- `semantic_level = atomic_fallback`
- `fragment_op = null`

---

## 3. 推荐的数据对象关系

推荐把动作协议组织成三层：

### 3.1 Pair / Path 事实骨架

保留：

- `smiles_from`
- `smiles_to`
- `primitive_ops`
- `node_smiles_list`
- `path_target`
- `step_targets`

### 3.2 语义标注层

新增：

- `semantic_steps`
- `annotation_status`
- `annotation_confidence`
- `provenance`

### 3.3 训练 / planner 导出层

从同一骨架导出：

- `semantic_pairs_*`
- `fragment_pairs_*`
- `semantic_paths_*`
- planner replay / action library 视图

这样能保证：

> **事实源只有一套，视图可以有多套。**

---

## 4. 推荐的 `semantic_step` 协议

建议每个 `semantic_step` 至少包含：

- `semantic_step_id`
- `semantic_level`：`fragment | atomic_fallback`
- `primitive_span`：`[start_idx, end_idx]`
- `primitive_ops`：可选，便于导出与调试
- `fragment_op`：对象或 `null`
- `annotation_status`：`resolved | approximate | unresolved`
- `annotation_confidence`
- `provenance`

推荐语义：

- `primitive_span` 指向它覆盖的 primitive 子序列
- `fragment_op` 只在 `semantic_level = fragment` 时存在
- `atomic_fallback` 表示这一步仍保留为语义 step，但不能干净压成片段动作

---

## 5. 推荐的 `fragment_op` schema

建议 `fragment_op` 至少包含：

- `op_type`
- `anchor`
- `fragment`
- `leaving_group`
- `connection`
- `constraints`
- `provenance`

### 5.1 `op_type`

第一版建议小词表：

- `attach_fragment`
- `replace_substituent`
- `grow_r_group`
- `delete_fragment`
- `bioisostere_swap`

### 5.2 `anchor`

建议包含：

- `anchor_atom_indices`
- `anchor_frag_idx`
- `anchor_env_type`
- `anchor_site_label`（可选）

### 5.3 `fragment`

建议包含：

- `fragment_id`
- `fragment_smiles`
- `attachment_points`
- `fragment_size`
- `fragment_source`

### 5.4 `leaving_group`

建议显式存在，为对象或 `null`。

若存在，建议包含：

- `matched_atom_indices`
- `leaving_fragment_smiles`
- `leaving_group_size`

### 5.5 `connection`

建议包含：

- `bond_type`
- `attachment_mapping`
- `topology_change`

### 5.6 `constraints`

建议包含：

- `scaffold_preserving`
- `rgroup_only`
- `ring_change`
- `charge_change`
- `valence_safe`

### 5.7 `provenance`

建议包含：

- `source`
- `rule_id`
- `template_id`
- `confidence`
- `actionlib_version`

---

## 6. 推荐的标注与 fallback 纪律

### 6.1 可以升级为 fragment 的情况

满足以下条件时，优先输出 `semantic_level = fragment`：

- 改动局限于一个局部连通区域
- anchor 可以稳定确定
- primitive span 连续
- 能明确判断是 attach / replace / grow / delete 一类动作

### 6.2 只能近似解释的情况

保留：

- `annotation_status = approximate`
- `annotation_confidence < 1.0`

并允许仍输出 `fragment_op`。

### 6.3 无法干净压缩的情况

不要硬写 `fragment_op`。

应输出：

- `semantic_level = atomic_fallback`
- `fragment_op = null`
- `annotation_status = unresolved`

这不是失败，而是协议允许的正常回退路径。

---

## 7. 与下游的关系

### 7.1 对数据构造

`03` 应围绕下面的关系组织：

- `primitive_pairs / primitive_paths` 是事实源
- `semantic_steps` 是标注层
- `fragment_pairs / semantic_paths` 是导出视图

### 7.2 对单步模型

`04` 不应直接吃原始 primitive dict，而应吃：

- `semantic_step`
- 或其中的 `fragment_op + primitive_trace`

### 7.3 对 planner

`05` 的候选动作对象应与 `semantic_step` 同构，保证：

- 离线训练看到的动作
- 在线规划扩展的动作
- replay 中记录的动作

尽量是同一协议。

---

## 8. 推荐交付物

本阶段建议交付：

- `fragment_op.schema.json`
- `fragment_op.examples.json`
- `semantic_step` 协议说明
- `annotation_status` / `provenance.source` 词表
- 下游特征提取字段清单

---

## 9. 验收标准

完成本阶段时，应满足：

- primitive 事实层与 semantic 语义层边界清晰
- `fragment_op` 不再被写成 primitive 的替身
- 无法片段化的样本有正式 fallback 路径
- 数据、模型、planner 至少能共享同一套动作对象定义

---

## 10. 一句话结论

> **新路线的关键不是“把 atom-op 变成 fragment-op”，而是“把 primitive trace 保留下来，再在其上建立 semantic step / fragment_op 作为统一的语义动作层”。**
