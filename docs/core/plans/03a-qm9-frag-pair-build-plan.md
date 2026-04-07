# 03a：QM9 `frag pair` 构造实施计划

> 目标：回答一个更具体的问题——**新的 `OFO-frag` 训练数据 pair，应该如何从 `mol_evo/dataset/data` 下的 QM9 数据稳定构造出来。**
>
> 本文默认沿用 `03-fragment-dataset-bootstrap-plan.md` 已经收口的主线：
>
> **QM9 raw CSV → canonical local pair → property delta labels → fragment annotation → split manifest → `fragment_pairs_{train,valid,test}.jsonl`**

---

## 1. 本文解决什么问题

`03` 已经回答了“新架构的数据主线应该是什么”。

但对于 `OFO-frag` 的第一版落地，还需要更具体地回答：

- 从 QM9 中到底挑哪些分子对
- 哪些 pair 才能算作适合 `frag` 模型的训练样本
- 样本字段应该长什么样
- 训练/验证/测试如何切分
- 怎样避免把“任意相似分子对”误当成“可执行局部动作样本”

所以 `03a` 的定位是：

> **把 `03` 里的数据主线，进一步细化为“单步 `frag pair` 训练集”的可执行构造方案。**

---

## 2. 核心判断：不要从 QM9 做任意两两配对

新的 `frag` 模型学习的不是：

- 任意两个 QM9 分子谁更优
- 任意相似分子之间的 ranking
- 事后用 `MCS` 硬解释出的伪动作

它学习的应该是：

> **一个局部可执行的结构编辑，从 `smiles_from` 走到 `smiles_to` 时，会带来怎样的单步属性变化。**

因此训练 pair 必须满足两个条件：

1. **结构上是局部可达的**
2. **语义上能对应一个单个 `fragment_op` 或高置信近似 `fragment_op`**

所以正确主线不是：

- `QM9 all-pairs → MCS 重解释 → 直接训练`

而是：

- `QM9 → canonical local pairs → property labels → fragment annotation → frag pair export`

---

## 3. 目标样本的最小定义

### 3.1 模型真正需要的样本语义

对 `OFO-frag` 而言，单条样本本质上是：

- 一个 `from` 分子
- 一个 `to` 分子
- 一个局部片段动作
- 一个单步属性变化标签

所以训练 pair 的核心不是“配对本身”，而是：

> **pair 是否能稳定承载一个局部 fragment-level transition。**

### 3.2 建议的最终样本字段

建议第一版 `fragment_pairs_*.jsonl` 中每条记录至少包含：

- `pair_id`
- `from_index`
- `to_index`
- `smiles_from`
- `smiles_to`
- `num_steps`
- `primitive_operations`
- `fragment_op`
- `annotation_status`
- `target_property`
- `step_target`
- `property_changes`
- `similarity`
- `scaffold_preserving`
- `split_key`
- `meta`

其中：

- `primitive_operations`：保留 canonical `operations`，便于回溯和兼容旧流程
- `fragment_op`：给 `OFO-frag` 的主要动作输入
- `annotation_status`：建议使用 `resolved | approximate | unresolved`
- `property_changes`：保留宽表标签，后续可展开成单属性任务视图

### 3.3 第一版推荐的任务视图

推荐把同一个结构样本展开成 **按目标属性拆开的训练视图**。

例如同一条结构 pair，可以展开成：

- `target_property = homo`, `step_target = homo_change`
- `target_property = lumo`, `step_target = lumo_change`
- `target_property = gap`, `step_target = gap_change`

这样更适合当前 `OFO` / `OFO-frag` 的单目标训练方式。

---

## 4. QM9 到 `frag pair` 的推荐构造流程

## 4.1 Stage 0：固定 QM9 输入源与索引

统一使用：

- `dataset/data/qm9_smiles_all_atoms.csv`：全量主入口
- `dataset/data/qm9_smiles_heavy_*_atoms.csv`：按重原子桶的辅助索引

这一阶段建议先生成一个只读 manifest，至少包含：

- `index`
- `smiles`
- `num_atoms`
- `scaffold_key`（可后算）
- 常用 QM9 属性列

作用：

- 后续 candidate 检索不用反复扫全表
- split 可以复用同一套 molecule/scaffold key
- 属性回查逻辑可以统一

## 4.2 Stage 1：先生成 canonical local pair candidates

### 候选生成原则

第一版不做全量笛卡尔积，而是只为每个 `from` 分子构造一小批**局部邻接候选**：

- 同重原子数桶内候选
- 相邻重原子数桶内候选（如 `k-1 / k / k+1`）
- 可按 scaffold、子结构或快速指纹做预筛

然后对候选执行已有的：

- `MoleculeEvolverAnalysis`
- `analyze_evolution_operation_dict(...)`

只保留能产出 canonical `operations` 的样本。

### 第一版建议过滤条件

建议初版先收紧，做高精度子集：

- `1 <= num_steps <= 3`
- `similarity >= 0.55`（可后续调参）
- 变化区域尽量局部
- 优先保留 `scaffold_preserving = true` 的样本
- 大片段替换、复杂多区域变化先排除

### 这一层的输出

建议输出：

- `canonical_pairs_raw.jsonl`

单条记录建议包含：

- `pair_id`
- `from_index`
- `to_index`
- `smiles_from`
- `smiles_to`
- `operations`
- `num_steps`
- `similarity`
- `candidate_meta`

## 4.3 Stage 2：补齐 QM9 属性变化标签

在 `canonical_pairs_raw.jsonl` 上补齐属性监督，复用/收口现有：

- `dataset/calculate_property_changes.py`

至少补齐：

- `homo_change`
- `lumo_change`
- `gap_change`
- 可选的其他 QM9 属性变化

建议同时写入：

- `property_changes`
- `property_changes_pct`
- `target_property`
- `step_target`

这里推荐分成两层产物：

1. **宽表层**：一条 pair 带所有属性变化
2. **任务展开层**：按 `target_property` 拆成多条单目标样本

推荐第一版训练优先消费第二层。

## 4.4 Stage 3：为 canonical pair 标注 `fragment_op`

这一步的重点是：

> **把 canonical `operations` 升级为 fragment-aware 动作语义，而不是抛弃 canonical pair 重新造一套 pair。**

建议标注优先级：

1. **canonical primitive op → fragment rule mapping**
2. **规则动作库补齐**
3. **`fragment_alignment_utils.py` fallback**

并给每条 pair 输出：

- `fragment_op`
- `annotation_status`
- `annotation_confidence`
- `provenance`

第一版训练建议只保留：

- `resolved`
- 高置信 `approximate`

`unresolved` 先进入分析集，不进训练集。

## 4.5 Stage 4：构造 split manifest

这里不要再做 record-level random split。

建议默认使用：

- `molecule-level split`
- 或 `scaffold-level split`

并要求：

- 同一个分子不要跨 train/valid/test 泄露
- 同一 scaffold 尽量只出现在一个 split
- 所有导出视图共用同一份 `split_manifest.json`

## 4.6 Stage 5：导出 `fragment_pairs_{train,valid,test}.jsonl`

最终给 `OFO-frag` 的数据建议只导出：

- 结构上局部可达
- 语义上能对应单个 `fragment_op`
- 标签上有可靠 QM9 属性变化
- 切分上满足 molecule/scaffold 约束

这一步才真正产生模型训练输入。

---

## 5. 推荐的候选筛选策略

## 5.1 基础候选池

对于每个 `from` 分子，候选 `to` 分子建议来自：

- 相同 `num_atoms`
- `num_atoms ± 1`
- 必要时 `num_atoms ± 2`

这样做的目的不是限制化学空间，而是优先保留：

- 更像局部片段插入/删除/替换
- 更容易对应短步 `operations`

## 5.2 结构预筛

进入 `MoleculeEvolverAnalysis` 之前，建议先做廉价过滤：

- SMILES 有效性
- 去重与规范化
- 指纹相似度下限
- 可选的 scaffold 是否一致

否则 pair 数量会爆炸，而且大部分 pair 对 `frag` 学习没有意义。

## 5.3 动作过滤

运行出 canonical `operations` 后，再按动作语义过滤：

- `num_steps <= 3`
- 优先单步或近单步
- 尽量只保留单区域变化
- 排除明显需要多段解释的复杂样本

## 5.4 片段大小过滤

第一版建议限制：

- `fragment_size <= 8`
- `leaving_group_size <= 8`

这样更有利于先学到稳定的局部模式。

---

## 6. 推荐的数据产物分层

建议不要一步直接导出最终训练集，而是保留中间层，方便回溯。

### 6.1 原始候选层

- `canonical_pairs_raw.jsonl`

### 6.2 带属性标签层

- `canonical_pairs_labeled.jsonl`

### 6.3 带 fragment 标注层

- `canonical_pairs_fragment_annotated.jsonl`

### 6.4 训练导出层

- `fragment_pairs_train.jsonl`
- `fragment_pairs_valid.jsonl`
- `fragment_pairs_test.jsonl`

### 6.5 分析与配置层

- `split_manifest.json`
- `dataset_stats.json`
- `fragment_action_config.yaml`
- `annotation_coverage_report.json`

---

## 7. 推荐脚本拆分

当前落地状态：

- 已新增 `dataset/build_canonical_qm9_pairs.py` 首版脚本骨架，负责 `QM9 -> canonical_pairs_raw.jsonl` 这一层的候选筛选、primitive `operations` 提取与 JSONL 导出
- `annotate_pair_property_deltas.py`、`annotate_fragment_ops.py`、`build_split_manifest.py`、`export_fragment_pairs.py` 仍属于后续实现项

## 7.1 `dataset/build_canonical_qm9_pairs.py`

职责：

- 读取 QM9 CSV
- 建立局部候选池
- 调用 evolver / operation analysis
- 输出 `canonical_pairs_raw.jsonl`

优先复用：

- `dataset/extract_evolution_pairs.py`
- `dataset/extract_evolution_pairs_v1.py`

## 7.2 `dataset/annotate_pair_property_deltas.py`

职责：

- 给 canonical pairs 补齐 QM9 属性变化
- 输出 `canonical_pairs_labeled.jsonl`

优先复用：

- `dataset/calculate_property_changes.py`

## 7.3 `dataset/annotate_fragment_ops.py`

职责：

- 基于 canonical `operations` 与前后分子生成 `fragment_op`
- 写入 `annotation_status` / `provenance`
- 输出 `canonical_pairs_fragment_annotated.jsonl`

## 7.4 `dataset/build_split_manifest.py`

职责：

- 生成 molecule/scaffold 级 split
- 产出统一 `split_manifest.json`

## 7.5 `dataset/export_fragment_pairs.py`

职责：

- 按 `target_property` 展开样本
- 过滤 `unresolved`
- 导出 train/valid/test 三个 split 的最终 `fragment_pairs_*.jsonl`

---

## 8. 第一版默认策略建议

如果目标是**先尽快训通第一版 `OFO-frag`**，建议默认策略收紧为：

- 只做 `homo / lumo / gap`
- 只保留 `num_steps <= 3`
- 优先 `scaffold_preserving`
- 只保留 `resolved + 高置信 approximate`
- 限制片段规模
- 优先高相似局部变化样本

这会让数据量小一些，但更容易先判断：

- `fragment_op` 是否真的比 primitive op 更有信息量
- 新模型是否在单步预测上有稳定增益

---

## 9. 正确性验证

构造完成后，至少要检查四类问题：

### 9.1 结构正确性

- `smiles_from` / `smiles_to` 可解析
- `fragment_op` 字段满足 schema
- `annotation_status` 与 `provenance` 一致

### 9.2 标签正确性

- `step_target` 是否与 QM9 原始属性差一致
- 宽表层与任务展开层是否一致
- 正负样本比例是否异常失衡

### 9.3 数据泄露

- 同一分子是否跨 split
- 同一 scaffold 是否严重跨 split
- 同一 pair 的正反向样本是否泄露到不同 split

### 9.4 训练可用性

- `04` 所需字段是否完整
- `prepare_fragment_op_features()` 是否能稳定消费
- 至少能抽样检查若干 `from → fragment_op → to` 是否化学上可解释

---

## 10. 与现有计划文件的关系

- `02`：负责 `fragment_op` schema、来源词表、嵌入协议
- `03`：负责 QM9 canonical 数据主线与脚本分层总纲
- `03a`：负责把 `OFO-frag` 需要的**单步 frag pair** 具体做出来
- `04`：消费 `03a` 产出的 `fragment_pairs_*.jsonl` 做模型训练

所以 `03a` 不是替代 `03`，而是：

> **把“数据主线”进一步细化成“模型训练入口”。**

---

## 11. 本阶段验收标准

完成本计划时，应至少满足：

- 能从 QM9 稳定导出一版 `canonical_pairs_raw.jsonl`
- 能补齐 `homo / lumo / gap` 的 `step_target`
- 能为大部分训练样本标出 `fragment_op`
- 能导出 `fragment_pairs_{train,valid,test}.jsonl`
- 数据切分满足 molecule/scaffold 级约束
- `04` 可直接以该数据为入口启动第一版 `OFO-frag` 训练

---

## 12. 一句话结论

> **新的 `frag` 训练 pair，不应从 QM9 做任意两两配对再事后重解释；而应先从 QM9 构造局部可达的 canonical pair，再补齐属性变化与 `fragment_op` 标注，最终留下“一个局部 fragment action 对应一次单步属性变化”的训练样本。**
