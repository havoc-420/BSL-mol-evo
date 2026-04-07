# 04：`OFO-frag-step` 单步模型升级计划（重构版）

> 目标：在保留 `from/to` 双分子输入框架的前提下，把当前 OFO 从 **primitive edge scorer** 升级为 **semantic-step scorer**，并允许 fragment 语义与 atomic fallback 共存。

---

## 1. 本阶段的核心判断

当前 OFO 的真正抽象不是生成器，而是：

> **给定 `from_mol`、`to_mol` 和一步动作表示，预测单步属性变化。**

因此新模型升级的关键不在于“推翻 OFO”，而在于：

- 保留 `from_data`
- 保留 `to_data`
- 把旧 `edge_attr` 升级成 `semantic_step` 表示
- 允许 fragment 与 atomic fallback 两种语义层级并存

---

## 2. 新训练入口的默认假设

本阶段默认训练数据来自 `03` 的导出视图，优先消费：

- `semantic_pairs_train.jsonl`
- `semantic_pairs_valid.jsonl`
- `semantic_pairs_test.jsonl`

其中纯 fragment 训练实验可再过滤为：

- `fragment_pairs_train.jsonl`
- `fragment_pairs_valid.jsonl`
- `fragment_pairs_test.jsonl`

也就是说：

- **默认主入口是 `semantic_pair`**
- **`fragment_pair` 是高置信子集实验入口**

---

## 3. 推荐的模型对象

### 3.1 输入对象

单条样本至少应提供：

- `smiles_from`
- `smiles_to`
- `semantic_step`
- `primitive_ops`
- `semantic_level`
- `target_property`
- `step_target`

### 3.2 输出对象

第一版仍保持与 OFO 接近：

- 单步属性变化预测值
- 可选置信度/不确定性

---

## 4. 推荐表示策略

### 4.1 `edge_attr` 不再直接等于 primitive one-hot

推荐把 edge 表示拆成两路：

- **semantic 路**：`semantic_step` / `fragment_op` 语义特征
- **trace 路**：`primitive_trace` 的压缩特征

也就是说，新的动作输入不应只是：

- `operation_type`
- `atom`
- `position`

而应更接近：

> **`edge_attr = semantic_features + primitive_trace_features + context_features`**

### 4.2 建议的特征组成

第一版建议包括：

- `semantic_level`
- `op_type`
- `anchor features`
- `fragment features`
- `constraint features`
- `primitive_trace summary`
- `span length`
- `fallback flags`

---

## 5. 编码阶段建议

### Stage 1：可运行版本

先支持：

- `semantic_level` embedding
- `op_type` one-hot / embedding
- `anchor` 位置或局部标记
- `fragment` fingerprint
- `constraints`（size / ring / hetero delta）
- `primitive_trace` 的轻量统计特征

目标：先证明 semantic-step 比旧 primitive edge 更有信息量。

### Stage 2：增强版本

再加入：

- anchor local context encoder
- leaving group 表征
- 更强的 primitive trace encoder
- fragment learned embedding

### Stage 3：强版本

最后再考虑：

- 局部 GNN / FragNet 编码 fragment
- 学习式 trace encoder
- uncertainty / calibration 支持

---

## 6. fallback 兼容原则

这是本阶段最重要的工程原则之一。

### 6.1 不要求所有样本都有 fragment_op

若 `semantic_level = atomic_fallback`，模型仍应能前向。

### 6.2 不要求立刻废弃旧 edge 流程

推荐保留：

- 旧 `prepare_edge_features()`
- 新 `prepare_semantic_step_features()` / `prepare_fragment_op_features()`

并允许短期并行存在。

### 6.3 不要求所有实验都用纯 fragment 子集

建议同时做：

- `semantic_pair` 混合训练
- `fragment_pair` 严格子集训练

比较哪条路线更稳。

---

## 7. 任务拆解

### Task 1：新增语义动作特征准备函数

建议新增：

- `prepare_semantic_step_features()`
- 或在其内部调用 `prepare_fragment_op_features()`

职责：

- 读取 `semantic_step`
- 生成固定维度 edge 表示
- 对 `atomic_fallback` 做兼容处理

### Task 2：定义新配置项

需要明确：

- `semantic_levels`
- `fragment_op_types`
- `fragment_feature_dim`
- `trace_feature_dim`
- `constraint_feature_dim`
- 最终 `edge_feature_dim`

### Task 3：升级 edge encoder

建议先走兼容路线：

- 保留旧 OFO 主干
- 先替换 edge 输入
- 仅在必要时新增 `SemanticStepEdgeExtractor`

### Task 4：增加模型变体

建议至少做两类：

- 普通分子编码器 + semantic-step edge
- 更强分子编码器 + semantic-step edge

### Task 5：训练与日志

建议每次训练记录：

- 使用的是 `semantic_pairs` 还是 `fragment_pairs`
- fallback 样本占比
- edge 特征维度与组成
- 不同 `semantic_level` 的验证表现

---

## 8. 代码落点建议

优先修改或扩展：

- `core/data/processing.py`
- `core/data/unified_processing.py`
- `core/data/path_processing.py`
- `core/models/*/edge_feature_extractors/*`
- 单步训练入口相关脚本

必要时新增：

- `core/data/semantic_step_processing.py`
- `core/models/.../semantic_step_edge_extractor.py`

---

## 9. 验收标准

完成本阶段时，应满足：

- `semantic_pairs` 可直接训练
- `atomic_fallback` 不会导致训练链断裂
- 至少一版 semantic-step scorer 能稳定收敛
- 至少能与旧 OFO 做一次公平单步对比
- 能解释 fragment 子集和 mixed semantic 子集的效果差异

---

## 10. 一句话结论

> **`OFO-frag-step` 的正确升级方式，不是把 edge 从 primitive 换成一个更大的 one-hot，而是让模型学会对“带 primitive trace 的 semantic step”做单步打分。**
