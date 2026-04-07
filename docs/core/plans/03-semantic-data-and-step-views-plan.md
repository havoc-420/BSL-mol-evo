# 03：数据主干与单步视图计划（重构收口版）

> 目标：把 `mol-ofo` 的数据路线收口成一条统一主干：**primitive facts → semantic annotation → split/labels → training/planner views**，并把单步 `semantic_pair / fragment_pair` 明确纳入同一份数据计划，而不再拆成第二份几乎同任务的文档。

> 这里仍允许先用 `QM9` 跑通链路，但必须明确：`QM9` 只承担 bootstrap / debug / ablation 角色，不承担长期 fragment 主训练源角色。

---

## 1. 本阶段先固定一个总原则

新数据主线不应再是：

- 先拿一批 pair/path
- 再离线做一轮 MCS 重解释
- 最后把 bootstrap 输出当成主训练数据

更合理的主线应是：

> **原始分子表 → canonical primitive facts → semantic annotation → property / split / export views**

也就是说：

- **主事实源**来自 primitive pair/path
- **semantic 标注层**建立在事实源之上
- **训练、planner、replay** 都只是导出视图
- **单步训练视图**也只是导出层的一部分，不是另一套事实源

---

## 2. 本阶段真正要回答的问题

这一阶段不只是“做数据”，而是要回答：

- 哪些文件是主事实源
- 哪些文件只是导出视图
- primitive path 与 semantic step 如何共存
- 单步模型到底吃什么最小样本
- 哪些样本可以进入 `fragment_pairs`
- 哪些样本只能保留在 `semantic_pairs` 里做 fallback
- QM9 在整个体系里处于什么位置
- 后续如何无痛切换到更 fragment-rich 的数据源

---

## 3. 数据主干总览

推荐把数据组织成五层。

### Layer A：Molecule manifest

记录每个分子的稳定入口信息，例如：

- `mol_id`
- `source_dataset`
- `smiles`
- `num_atoms`
- `scaffold_key`
- 真实属性列

作用：

- 统一原始分子入口
- 统一 split key
- 统一属性回查与数据源切换

### Layer B：Primitive facts

这是真实的结构变化事实层，建议产出：

- `primitive_pairs_*.jsonl`
- `primitive_paths_*.jsonl`

单条记录至少包含：

- `smiles_from`
- `smiles_to`
- `primitive_ops`
- `num_steps`
- `meta`

对 path 再补充：

- `node_smiles_list`
- `path_target`
- `step_targets`

### Layer C：Semantic annotation

在 primitive facts 之上新增：

- `semantic_steps`
- `annotation_status`
- `annotation_confidence`
- `provenance`

这里不是替换 primitive，而是标注 primitive。

### Layer D：Split + labels

对 facts / annotation 补齐：

- 目标属性标签
- `step_target` / `path_target`
- `split_manifest`
- scaffold / molecule 级切分信息

### Layer E：Export views

从同一主干导出：

- `semantic_pairs_*`
- `fragment_pairs_*`
- `semantic_paths_*`
- planner replay / action library 视图

---

## 4. 推荐的数据文件分层

### 4.1 主事实源

推荐作为 source of truth 的文件：

- `molecule_manifest.jsonl`
- `primitive_pairs_raw.jsonl`
- `primitive_pairs_labeled.jsonl`
- `primitive_paths_raw.jsonl`
- `primitive_paths_labeled.jsonl`
- `semantic_pairs_annotated.jsonl`
- `semantic_paths_annotated.jsonl`
- `split_manifest.json`

### 4.2 导出视图

推荐作为训练 / planner 入口视图的文件：

- `semantic_pairs_{train,valid,test}.jsonl`
- `fragment_pairs_{train,valid,test}.jsonl`
- `semantic_paths_{train,valid,test}.jsonl`
- `planner_replay_{train,valid,test}.jsonl`
- `action_library_stats.json`

关键原则：

> **导出视图可以丢弃部分样本，但主事实源不应丢。**

---

## 5. 单步训练视图如何纳入同一主干

### 5.1 模型真正该学的对象

新单步模型学的不是“任意两个分子之间谁更好”，而是：

> **一个 semantic step 在给定 `from_mol` 下会带来怎样的单步属性变化。**

因此单步样本的最小单元应是：

- 一个 `from` 状态
- 一个 `to` 状态
- 一个 `semantic_step`
- 一个 `step_target`

### 5.2 `semantic_pairs_*`

这是更宽的单步训练视图，包含：

- `fragment` 语义 step
- `atomic_fallback` 语义 step

适合：

- 覆盖率分析
- mixed semantic 训练
- 过渡期兼容实验

### 5.3 `fragment_pairs_*`

这是更严格的单步训练视图，只包含：

- `semantic_level = fragment`
- `annotation_status in {resolved, high_conf_approximate}`

适合：

- 纯 `OFO-frag-step` 训练
- 严格片段语义实验

关键区别是：

> **`fragment_pairs` 是 `semantic_pairs` 的高置信子集，而不是新的主事实源。**

### 5.4 推荐的单步样本字段

建议单条 `semantic_pairs_*.jsonl` 至少包含：

- `pair_id`
- `smiles_from`
- `smiles_to`
- `primitive_ops`
- `semantic_step`
- `semantic_level`
- `annotation_status`
- `annotation_confidence`
- `target_property`
- `step_target`
- `property_changes`
- `split_key`
- `meta`

其中：

- `primitive_ops`：保留事实层回溯能力
- `semantic_step`：模型的主动作输入
- `semantic_level`：区分 fragment 与 atomic fallback
- `annotation_status`：控制训练过滤策略

对 `fragment_pairs_*.jsonl`，可直接沿用同一协议，只是过滤出 fragment 子集。

---

## 6. 从 primitive facts 到单步样本的推荐流程

### Stage 0：准备 primitive facts

输入来自本计划主数据干线：

- `primitive_pairs_*`
- 或 `primitive_paths_*`

### Stage 1：做 primitive span segmentation

这是本计划最关键的一步。

需要把原始 primitive 操作序列切成若干连续 span，每个 span 对应一个 semantic step 候选。

推荐原则：

- 优先保持 span 连续
- 优先保证单 anchor
- 优先保证局部连通变化
- 无法 clean segmentation 时允许保留较小 atomic span

### Stage 2：为每个 span 做 semantic annotation

对每个 span 判断：

- 是否可解释为 fragment-level action
- 若可解释，是哪类 `fragment_op`
- 若不可解释，是否保留 atomic fallback

### Stage 3：补齐 `step_target`

若原始标签已经是单步级，直接对齐。

若 span 覆盖多个 primitive 步，则推荐：

- 对应 `step_target` 由该 span 覆盖的 primitive changes 聚合而来
- path 级场景下优先用 span 内目标的求和或同义聚合

### Stage 4：导出 `semantic_pairs`

为每个 semantic step 输出一条单步训练样本。

### Stage 5：筛出 `fragment_pairs`

只保留高置信 fragment 样本，作为严格片段训练入口。

### Stage 6：同步导出 path / planner 视图

确保单步视图和长程视图来自同一事实主干：

- `semantic_paths_*`
- `planner_replay_*`
- `action_library_stats.json`

---

## 7. 推荐的过滤纪律

### 7.1 可以进入 `fragment_pairs` 的样本

建议满足：

- `semantic_level = fragment`
- `annotation_status = resolved`
- 或 `annotation_status = approximate` 且 `annotation_confidence` 足够高
- 局部变化单区域、单 anchor
- `step_target` 可稳定定义

### 7.2 只进入 `semantic_pairs` 的样本

建议包括：

- `atomic_fallback`
- `unresolved`
- 仍有训练分析价值但不足以当纯 fragment 样本的记录

### 7.3 暂不进入训练的样本

例如：

- 多区域同时变化
- 多 anchor 纠缠
- 无法稳定切 span
- 标签难以对齐的复杂 path

这些样本可先进入分析集或错误案例集。

---

## 8. `QM9` 在这条主线里的角色

### 8.1 `QM9` 适合做什么

- 跑通 primitive pair/path 构造
- 验证 semantic annotation 协议
- 跑通 `semantic_pairs / fragment_pairs / semantic_paths` 导出链路
- 验证特征提取、训练入口和日志链路
- 做 smoke test / ablation / regression test

### 8.2 `QM9` 不适合做什么

- 不适合作为长期 fragment 主训练源
- 不适合作为最终 planner 评估的唯一来源
- 不适合作为 fragment action 多样性结论的主要依据

### 8.3 迁移原则

只要后续新数据源仍能产出同样的：

- molecule manifest
- primitive facts
- semantic annotation
- export views

那么模型和 planner 就不应依赖“它是不是 QM9”。

---

## 9. 推荐脚本分层

### 9.1 `build_molecule_manifest.py`

职责：

- 读取原始分子表
- 生成统一的 molecule manifest
- 提供 `mol_id`、`scaffold_key`、属性回查入口

### 9.2 `build_canonical_pairs.py`

职责：

- 基于现有 evolver / diff 逻辑生成 primitive pairs
- 输出 `primitive_pairs_raw.jsonl`

### 9.3 `annotate_property_deltas.py`

职责：

- 对 pair/path 补齐属性变化标签
- 输出 labeled 版本

### 9.4 `normalize_primitive_paths.py`

职责：

- 统一 primitive path 协议
- 修复 path 构造与 path 消费之间的格式断层

### 9.5 `annotate_semantic_steps.py`

职责：

- 从 primitive facts 生成 `semantic_steps`
- 从 primitive ops/path 中切分 primitive span
- 输出 `fragment_op` 或 atomic fallback
- 补齐 annotation status / confidence / provenance

### 9.6 `build_split_manifest.py`

职责：

- 生成 molecule/scaffold 级 split
- 让所有导出视图共用同一份 split manifest

### 9.7 `export_training_views.py`

职责：

- 导出 `semantic_pairs / fragment_pairs / semantic_paths / planner_replay`
- 汇总统计信息

如需拆分脚本，也建议围绕同一条导出链，而不是再额外维护另一套“单步构造计划”。

---

## 10. 对当前仓库的直接指导

### 10.1 可以继续复用的部分

优先复用：

- 现有 evolver / primitive diff 逻辑
- 现有属性变化计算逻辑
- 现有 path 消费侧核心字段

### 10.2 当前需要重点修复的部分

重点修复：

- pair/path 协议不一致
- semantic annotation 尚未成为正式中间层
- split 仍过于接近 record random split

### 10.3 降级为 bootstrap / fallback 的部分

以下逻辑应明确降级为：原型验证、fallback 标注、覆盖率分析工具，而不是主生产线：

- MCS 重解释工具链
- 仅从 pair 事后推断 fragment 语义的脚本

---

## 11. 与 `04` 的接口要求

`04` 默认应把 `semantic_pair` 视为单步训练入口。

也就是说，单步模型优先消费：

- `semantic_step`
- `primitive_ops`
- `step_target`

而不是再回退到历史的 `operation + atom + position` 轻量三元组。

---

## 12. 验收标准

完成本阶段时，应满足：

- source of truth 文件与 export views 边界清晰
- primitive facts 能稳定构造
- semantic annotation 成为正式中间层
- 能从 primitive facts 稳定切出 `semantic_step`
- 能导出一版 `semantic_pairs_{train,valid,test}.jsonl`
- 能进一步筛出 `fragment_pairs_{train,valid,test}.jsonl`
- split manifest 统一且可复用
- QM9 只被写作 bootstrap，而不是主训练源
- 数据主干可以在不改协议的情况下切换主数据源
- fallback 样本不会被强行伪装成 fragment 样本

---

## 13. 一句话结论

> **新数据主线的核心不是“做出一份 fragment 数据”，而是“围绕 primitive facts 搭好可迁移的数据主干，并在同一条主干上导出 `semantic_pairs / fragment_pairs / semantic_paths` 等训练与规划视图”。**
