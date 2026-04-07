# 02：`fragment_op` Schema 与数据协议收口计划

> 目标：把 `fragment_op` 定义成**可嵌入现有 canonical pair/path 数据骨架**的统一动作对象，而不是另起一条只服务于 bootstrap 脚本的平行协议。

---

## 1. 本阶段先纠正一个关键判断

对于新架构，训练数据主线**不应**是：

- 先拿任意 `(smiles_from, smiles_to)`
- 再用 `MCS` / scaffold 对齐做一次事后“重解释”
- 最后把结果直接当成主训练数据

那条路线更适合：

- bootstrap
- 弱标注
- fallback
- 调试与可视化分析

而新的主线应该是：

> **QM9 原始 CSV → canonical primitive pair/path → property labels → path normalization → `fragment_op` annotation sidecar → 训练视图导出**

也就是说，`fragment_op` 首先是**现有 `operations[i]` 的升级协议**，其次才是一个独立的 schema 文件。

---

## 2. 本阶段核心问题

这一阶段需要先回答四个问题：

- `fragment_op` 应该挂在什么数据骨架上
- 它和当前 `operations` / `node_smiles_list` / `path_target` 的关系是什么
- 它如何同时被训练、planner、日志导出复用
- 哪些来源属于 canonical annotation，哪些只属于 bootstrap/fallback

如果这些问题不先收口，后续很容易出现：

- schema 能单独存在，但不能稳定落进现有数据管线
- pair/path/planner 各用一套动作表示
- `fragment_op` 被写成“只适合离线重解释”的对象
- 在线候选生成与离线训练监督不一致

---

## 3. 新架构下的正确定位

### 3.1 `fragment_op` 的本质

`fragment_op` 不是一个“更大的 label”，而是：

> **一个带锚点、片段身份、连接语义、约束语义、来源信息的条件动作对象。**

### 3.2 它应该出现在哪里

第一优先级不是单独保存为 `fragment_pairs_*.jsonl`，而是先让它进入 canonical 样本中的：

- `pair['operations'][i]`
- `path['operations'][i]`

即把原本轻量的 primitive op：

- `operation`
- `atom`
- `position`

逐步升级成结构化 `fragment_op dict`，同时保留必要的 backward-compatible 字段。

### 3.3 它与现有数据格式的关系

推荐遵循下面的兼容原则：

- **pair 层**：仍然保留 `smiles_from / smiles_to / operations`
- **path 层**：仍然保留 `node_smiles_list / operations / path_target / step_targets`
- **fragment 层**：作为 `operations[i]` 的结构化升级，或与其一一对应的 `operation_annotations[i]`

换句话说：

> **canonical 数据骨架不换，动作对象升级。**

---

## 4. Schema 设计原则

### 4.1 先服务数据主链，再服务单独样例文件

schema 必须先满足：

- `dataset` 构造脚本可稳定写入
- `core/data/*` 可稳定读取
- planner 可复用相同字段做候选动作表达
- 日志和评估可序列化导出

而不是先追求“看起来完整”的大而全字段集合。

### 4.2 先保证可兼容升级

第一版 schema 应允许下面两种样本同时存在：

- primitive-only `operations`
- fragment-aware `operations`

从而支持渐进迁移，而不是要求所有历史样本一次性重做。

### 4.3 强制区分动作语义与来源语义

`fragment_op` 至少要区分：

- **动作是什么**：`op_type`
- **改在哪儿**：`anchor`
- **加/换/删了什么**：`fragment` / `leaving_group`
- **怎么连**：`connection`
- **受什么限制**：`constraints`
- **怎么得到的**：`provenance`

### 4.4 `leaving_group` 建议作为显式字段保留

即便某些动作没有 leaving group，也建议显式写成：

- `"leaving_group": null`

而不是有时存在、有时缺省。这样更适合：

- schema 校验
- 训练前字段标准化
- planner/日志统一解析

---

## 5. 推荐 schema（vNext）

建议 `fragment_op` 至少包含：

- `op_type`
- `anchor`
- `fragment`
- `leaving_group`
- `connection`
- `constraints`
- `provenance`

### 5.1 `op_type`

第一版建议仍然保持小词表：

- `attach_fragment`
- `replace_substituent`
- `grow_r_group`
- `bioisostere_swap`
- `delete_fragment`

### 5.2 `anchor`

建议包含：

- `anchor_atom_indices`
- `anchor_frag_idx`
- `anchor_env_type`
- `anchor_position_encoding`
- `anchor_site_label`（可选，例如 para/meta/terminal/ring-edge）

### 5.3 `fragment`

建议包含：

- `fragment_id`
- `fragment_smiles`
- `attachment_points`
- `fragment_size`
- `fragment_source`

### 5.4 `leaving_group`

建议为对象或 `null`。

如果存在，建议包含：

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

建议至少包含：

- `source`
- `template_id`
- `rule_id`
- `confidence`
- `actionlib_version`

其中 `source` 建议能区分：

- `canonical_rule`
- `canonical_annotated`
- `bootstrap_mcs`
- `manual_template`

这样后续训练时才能方便做：

- 仅用高置信样本
- 分来源 ablation
- bootstrap 样本降权

---

## 6. 与 canonical pair/path 的嵌入方式

### 6.1 Pair 样本协议

第一版 pair 样本建议保持：

- `smiles_from`
- `smiles_to`
- `operations`
- `target_property`
- `step_target`
- `meta`

其中 `operations` 中的每个元素，允许是：

- 旧 primitive dict
- 新 `fragment_op` 兼容 dict

### 6.2 Path 样本协议

第一版 path 样本建议保持：

- `path_id`
- `node_smiles_list`
- `operations`
- `start_smiles`
- `end_smiles`
- `target_property`
- `path_target`
- `step_targets`

这样就能直接兼容当前 `core/data/path_processing.py` 的消费方式。

### 6.3 不建议另起一条平行主协议

不建议把主训练入口改成只认：

- `fragment_pairs_train.jsonl`
- `fragment_paths_train.jsonl`

而与当前 pair/path 协议脱钩。

更推荐：

- 先把 canonical pair/path 规范化
- 再导出 fragment 训练视图
- 视图只是导出结果，不是主事实源

---

## 7. 动作来源设计（按主次关系重排）

### Route A：canonical annotation（主路线）

从 QM9 上稳定生成的 primitive `operations` 出发，做结构化标注：

- 利用已有 `operations` 序列作为主语义
- 利用 `smiles_from / smiles_to` 与局部编辑位点补齐片段信息
- 让每一步操作都能映射到 `fragment_op` 或标记为 `unresolved`

这是新架构最重要的来源。

### Route B：规则动作库（并行补充）

可从规则库补充动作原型：

- BRICS attach
- 高频 R-group grow
- scaffold-preserving substituent swap

这一路更适合：

- 动作库初始化
- planner 候选枚举
- 冷启动覆盖不足场景

### Route C：MCS/bootstrap（降级为 fallback）

`fragment_alignment_utils.py` 一类基于 MCS 的重解释脚本应降级为：

- bootstrap
- 弱标注
- 纠错对照
- 无法从 canonical op 直接补齐时的 fallback

它**不应**再被写成主训练数据生产线。

---

## 8. 推荐任务拆解

### Task 1：收口 canonical 样本上的动作嵌入协议

明确：

- `operations[i]` 是否直接升级为 `fragment_op`
- 还是保留 `primitive_op`，并新增 `operation_annotations[i]`

推荐第一版优先采用：

- `operations[i]` 保留可兼容字段
- 再附带 fragment 结构字段

### Task 2：补齐 schema 与运行时字段一致性

需要收口：

- `leaving_group` 是否始终显式存在
- `anchor_position_encoding` 的序列化格式
- `attachment_mapping` 的数据类型
- `topology_change` 的枚举集合
- `provenance.source` 的受控词表

### Task 3：给 `core/data/*` 留出读取接口

需要明确：

- `prepare_edge_features()` 如何识别旧样本
- `prepare_fragment_op_features()` 如何读取新字段
- pair/path 混合数据如何做 fallback

### Task 4：给 planner 留出同构动作协议

离线训练和在线候选生成最好使用同构动作对象，避免：

- 训练看到的是 `fragment_op`
- planner 生成的是另一套 `action dict`

### Task 5：定义 unresolved / approximate 的处理策略

对于无法精确映射成 `fragment_op` 的步骤，需要明确写法，例如：

- `meta.annotation_status = resolved | approximate | unresolved`
- `provenance.source = bootstrap_mcs`
- `provenance.confidence < 1.0`

---

## 9. 推荐交付物

本阶段推荐交付物应包括：

- `docs/core/schemas/fragment_op.schema.json`：正式 schema
- `docs/core/schemas/fragment_op.examples.json`：覆盖主要 `op_type` 的样例
- `fragment_op` 与 canonical pair/path 的嵌入协议说明
- `prepare_fragment_op_features()` 所需字段清单
- `provenance.source` 与 `annotation_status` 词表说明

---

## 10. 验收标准

完成本阶段时，应满足：

- `fragment_op` 能稳定嵌入现有 pair/path 骨架
- 同一个动作对象可以被数据管线、模型管线、planner 管线共用
- `provenance` 能区分 canonical annotation 与 bootstrap/fallback
- schema 不再假设“先做 MCS 重解释再训练”
- 后续 `03` 中的数据构造脚本可以直接围绕该协议实现

---

## 11. 风险与注意事项

- 不要把 schema 设计成只适合离线重解释脚本
- 不要为了兼容旧格式而放弃 `fragment_op` 的结构语义
- 不要把 `fragment_op` 写成与 pair/path 主协议平行的孤立对象
- 不要默认所有样本都能被精确解释成 clean fragment action
- 不要把 bootstrap 来源和 canonical 来源混在一起训练却不留痕迹
