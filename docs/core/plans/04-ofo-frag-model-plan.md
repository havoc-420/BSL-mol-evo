# 04：`OFO-frag` 模型升级计划

> 目标：在保留 `from/to` 双分子输入框架的前提下，把当前 OFO 从 primitive-op scorer 升级为 fragment-level transition scorer。

---

## 1. 本阶段核心判断

当前 OFO 的核心抽象并不是生成器，而是：

> **给定 `from_mol`、`to_mol` 和动作信息，预测单步属性变化。**

因此升级成 `OFO-frag` 的关键，不是推翻 OFO，而是：

- 保留 `from_data`
- 保留 `to_data`
- 升级 `edge_attr`
- 升级 `EdgeFeatureExtractor`

---

## 2. 本阶段目标

- 新增 `prepare_fragment_op_features()`
- 将 `fragment_op` 转成新的 `edge_attr`
- 让当前模型工厂可支持 `OFO-frag` 版本
- 跑通第一版片段动作单步训练

### 2.1 数据前置条件

本阶段默认训练入口已经由 `03a-qm9-frag-pair-build-plan.md` 固定，优先消费：

- `fragment_pairs_train.jsonl`
- `fragment_pairs_valid.jsonl`
- `fragment_pairs_test.jsonl`

单条样本至少应稳定提供：

- `smiles_from`
- `smiles_to`
- `fragment_op`
- `target_property`
- `step_target`
- `annotation_status`

也就是说，`04` 默认不再从 bootstrap 风格的离线重解释脚本直接取训练入口，而是消费 `03/03a` 导出的 canonical `frag pair` 视图。

---

## 3. 推荐表示策略

### 3.1 `edge_attr` 不再是简单 one-hot

推荐表示：

- `op_type_emb`
- `anchor_context_features`
- `fragment_features`
- `connection_features`
- `constraint_features`

### 3.2 编码阶段建议

#### Stage 1：轻量版

- `op_type`: embedding 或 one-hot
- `anchor`: position / local flags
- `fragment`: Morgan fingerprint 或 BRICS fingerprint
- `constraints`: size/ring/hetero delta

#### Stage 2：增强版

- `anchor_context`: 局部环境 embedding
- `fragment`: learned fragment embedding
- `leaving_group`: 可选编码

#### Stage 3：强版本

- 使用 `FragNet` 或局部 GNN 编码 fragment
- 用学习式模块替代大部分手工特征

---

## 4. 任务拆解

### Task 1：新增片段动作特征准备函数

建议新增：

- `prepare_fragment_op_features(row, ...)`

职责：

- 解析 `fragment_op`
- 生成固定维度的 `edge_attr`
- 与现有 `prepare_edge_features()` 并行存在

### Task 2：定义新配置与维度

需要明确：

- `fragment_op_types`
- `fragment_feature_dim`
- `anchor_feature_dim`
- `constraint_feature_dim`
- 最终 `edge_feature_dim`

### Task 3：升级 edge encoder

可选路线：

- **兼容路线**：继续用现有 `LinearEdgeFeatureExtractor`
- **增强路线**：新增 `FragmentEdgeFeatureExtractor`

建议先走兼容路线，先验证数据与特征是否有效。

### Task 4：增加模型变体

建议先做 2 个版本：

- `gcn_linear_linear_frag`
- `frag_linear_linear_frag`

这样可分别观察：

- 普通图编码器 + fragment edge 是否足够
- FragNet 主干 + fragment edge 是否更适配

### Task 5：训练与日志

建议输出：

- 训练配置 YAML
- edge feature 维度说明
- 训练日志
- 验证集结果
- 若干样本可解释性分析

---

## 5. 代码落点建议

优先修改或扩展：

- `core/data/processing.py`
- `core/data/unified_processing.py`
- `core/data/path_processing.py`
- `core/models/v0/edge_feature_extractors/*`
- `core/models/v0/*`

必要时新增：

- `core/data/fragment_processing.py`
- `core/models/v0/edge_feature_extractors/fragment.py`

---

## 6. 验收标准

完成本阶段时，应满足：

- 能使用 `fragment_op` 数据跑通单步训练
- 能在验证集得到稳定 loss 曲线
- 新模型输入输出接口与旧 OFO 尽量兼容
- 至少有一版 `OFO-frag` 能与当前 OFO 做公平对比

---

## 7. 风险与注意事项

- 不要一开始就把 fragment encoder 做得太重
- 不要让新模型只能吃新数据、无法兼容旧实验体系
- 不要把结构差异完全寄希望于 `from/to` 图自学，动作语义必须显式输入
- 不要在没有验证数据质量前盲目调模型结构
