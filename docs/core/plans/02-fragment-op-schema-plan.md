# 02：`fragment_op` Schema 与动作库计划

> 目标：定义片段级动作到底是什么，形成统一 schema、动作词表、约束规则和动作来源机制。

---

## 1. 本阶段核心问题

当前系统默认动作是 primitive graph edit，例如：

- `add_atom`
- `replace_atom`
- `delete_atom`
- `add_bond`
- `change_bond`

而新的主线要求转向：

> **chemically meaningful fragment-level actions**

因此本阶段要先解决：

- 什么叫一个合法的 `fragment_op`
- 动作类型有哪些
- 动作需要记录哪些字段
- 动作库从哪里来
- 哪些动作属于 `scaffold_preserving`

---

## 2. 本阶段目标

### 2.1 概念目标

明确区分三层动作：

- **primitive actions**：原子/键级基础动作
- **fragment actions**：片段级宏动作
- **template actions**：局部环境条件下的模板替换动作

### 2.2 数据目标

形成统一的 `fragment_op` 字段定义，使其可被：

- 数据构造脚本使用
- `prepare_fragment_op_features()` 使用
- planner 候选生成器使用
- 实验日志与 JSON 导出使用

---

## 3. 推荐 schema

建议 `fragment_op` 至少包含：

- `op_type`
- `anchor`
- `fragment`
- `leaving_group`
- `connection`
- `constraints`
- `provenance`

### 3.1 建议字段

#### `op_type`

候选集合建议先从小词表开始：

- `attach_fragment`
- `replace_substituent`
- `grow_r_group`
- `bioisostere_swap`
- `delete_fragment`

#### `anchor`

建议包含：

- `anchor_atom_indices`
- `anchor_frag_idx`
- `anchor_env_type`
- `anchor_position_encoding`

#### `fragment`

建议包含：

- `fragment_id`
- `fragment_smiles`
- `attachment_points`
- `fragment_size`
- `fragment_source`

#### `leaving_group`

如果是替换类动作，建议包含：

- `matched_atom_indices`
- `leaving_fragment_smiles`
- `leaving_group_size`

#### `connection`

建议包含：

- `bond_type`
- `attachment_mapping`
- `topology_change`

#### `constraints`

建议包含：

- `scaffold_preserving`
- `rgroup_only`
- `ring_change`
- `charge_change`
- `valence_safe`

#### `provenance`

建议包含：

- `source`
- `template_id`
- `rule_id`
- `confidence`

---

## 4. 动作来源设计

### Route A：规则生成

优先从以下规则启动：

- BRICS 片段 attach
- R-group 替换
- 简化的 side-chain grow / prune

### Route B：数据抽取

从已有 `(mol_from, mol_to)` 或路径数据中抽取：

- MCS 对齐后的局部替换
- matched molecular pair
- 高频 edit motif

### Route C：模板沉淀

当数据积累后，再把高频局部替换沉淀为模板动作：

- scaffold 位点模板
- 属性导向模板
- 局部环境条件模板

---

## 5. 本阶段任务拆解

### Task 1：定义 `scaffold_preserving`

建议第一版使用：

- Murcko scaffold 是否保持不变

后续再视情况增加更宽松定义。

### Task 2：形成动作词表

动作词表不宜过大，建议第一版严格控制在 4~8 类核心动作。

### Task 3：输出 schema 示例文件

建议编写：

- `fragment_op.schema.json`
- `fragment_op.examples.json`

### Task 4：补齐校验逻辑

需要对动作进行：

- 字段完整性校验
- 锚点合法性校验
- 连接合法性校验
- scaffold 约束校验

### Task 5：动作库版本管理

建议为动作库打版本，例如：

- `actionlib_v0`: BRICS + 规则型片段
- `actionlib_v1`: 加入数据驱动模板

---

## 6. 推荐交付物

- `fragment_op` 正式 schema 文档
- 一组样例动作 JSON
- 一版动作词表与说明
- 一版动作库来源清单
- 一版动作合法性校验规则

---

## 7. 验收标准

完成本阶段时，应满足：

- 任意一个候选动作都能序列化成统一的 `fragment_op`
- 同一个动作可以被数据管线、模型管线、planner 管线共用
- `scaffold_preserving` 的标注规则明确
- 动作词表与字段定义在团队内部不再反复变化

---

## 8. 风险与注意事项

- 不要一开始就做过大的动作词表
- 不要让 schema 只适合训练数据、不适合 planner 在线生成
- 不要把 `fragment_op` 退化成“多了一个 `add_fragment` 标签”
- 不要把 scaffold 定义写得过于模糊
