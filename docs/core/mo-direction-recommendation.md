# MO 方向调整建议（基于现有 `batch-optimizer` 与 `OFO` 框架）

> 本文用于总结当前 `mol-ofo` 项目在 MO（Molecular Optimization）任务上的既有落地、瓶颈判断，以及后续更值得投入的新方向。
>
> 建议结合以下文档一起阅读：
> - `batch-optimizer-framework.md`
> - `ofo-model-framework.md`

---

### 1. 当前系统的本质定位

从现有实现看，项目中的 MO 主链路本质上是一个**搜索驱动的编辑式分子优化系统**，而不是端到端分子生成系统。

其核心流程为：

1. 从起始分子出发，枚举一批允许的分子编辑操作
2. 应用操作得到候选下一步分子
3. 使用 `OFO` 模型预测该步操作带来的属性变化
4. 使用 `BFS` / `MCTS` 对候选分支进行保留、剪枝和继续展开
5. 最终输出一棵带预测属性值的进化树，并提取 TopK 候选

因此，当前系统可以概括为：

> **Primitive graph edit + single-step OFO scoring + tree search**

这条路线的优点是工程上容易闭环、便于做批量实验；但它的上限也比较明显。

---

### 2. `OFO` 模型的正确定位

结合 `ofo-model-framework.md`，`OFO` 当前最准确的定位不是“优化器”或“生成器”，而是：

> **分子转移打分器（transition scorer）/ 单步属性变化预测器**

#### 2.1 它在做什么

`OFO` 的输入是：

- 起始分子 `from_smiles`
- 目标分子 `to_smiles`
- 操作信息 `edge features`

输出是：

- 一步属性变化 `property delta`

这意味着：

- `OFO` 能回答“如果从当前分子走到这个候选分子，属性大概会怎么变”
- 但它**不能直接回答**“下一步应该生成什么候选”
- 它也**不能单独承担**长程规划或完整路径决策

#### 2.2 对研究方向的启示

所以当前 `OFO` 更适合作为：

- **一步候选打分器**
- **搜索中的局部 evaluator**
- **reward shaping / reranking 模块**

而不适合作为：

- 完整的分子生成模型
- 直接决定动作空间的模型
- 真正意义上的 A* 启发函数终态替代品

因为 A* 需要的是“未来剩余代价估计”，而 `OFO v0` 当前主要学到的是“一步变化收益”。

---

### 3. 为什么原来的 MO 落地效果有限

原有 `batch_optimizer` 路线之所以容易“能跑通，但效果一般”，根本原因不只是实现细节，而是当前问题表述本身存在上限。

#### 3.1 动作空间过于原始

当前搜索主要依赖 `_get_possible_operations()` 枚举 primitive 操作，例如：

- `add_atom`
- `replace_atom`
- `form_double_bond`
- `form_triple_bond`
- `form_ring`
- 对应的一些 reverse/remove 操作

这类动作的特点是：

- 化学可执行性弱依赖后验过滤
- 组合空间极大
- 缺乏 medicinal chemistry 语义
- 很难自然表达真实优化中常见的取代基替换、片段嫁接、骨架保持式修改

因此系统虽然在“图编辑”意义上可扩展，但在“化学优化”意义上并不够强。

#### 3.2 搜索策略偏局部、偏粗糙

当前 BFS 风格搜索的核心做法是：

1. 当前层为每个节点生成全部合法候选
2. 用 `OFO` 批量预测单步 `property_change`
3. 汇总整层候选并统一排序
4. 直接取前 50% 进入下一层

这意味着系统隐含地假设：

> **单步变化更优的候选，更可能属于全局更优路径**

这个假设在分子优化里往往不成立。很多真实有效路径需要：

- 先接受一两步中性甚至轻微负收益修改
- 再在后续几步中获得更大提升

而当前层内 top-50% 截断，会天然压制这类“长程有利、短程不占优”的路径。

#### 3.3 缺少真正的路径价值建模

当前系统主要维护：

- `property_change`
- `accumulated_change`
- `property_value`

本质上仍然是把**一步打分不断累加**。

但这并不等价于真正的 value function，因为它没有显式回答：

- 当前分子是否还具备继续优化的潜力
- 哪些状态虽暂时不优，但更接近高质量终点
- 哪类编辑路径更可能在未来产生跃迁式收益

换句话说，当前系统是**局部打分 + 累积求和**，而不是**状态价值评估 + 规划**。

#### 3.4 化学约束仍偏弱

虽然系统里已经有：

- 分子合法性校验
- `logP` 范围约束
- 停滞剪枝

但这些更多属于**后验过滤**，而不是动作生成时的强先验。

所以系统常见问题仍然包括：

- 候选很多，但真正有意义的少
- 合法但不“像药化优化”的编辑较多
- 搜索预算被大量浪费在低价值局部扰动上

---

### 4. 对当前路线的总体判断

基于以上分析，可以给出现阶段的明确判断：

> **当前 `MO` 主链路适合作为 baseline / 实验平台继续保留，但不适合作为未来研究主线继续重投入。**

更具体地说：

#### 4.1 应该保留的部分

- `batch_optimizer` 作为批量实验 orchestration layer
- `EvolutionTreeOptimizer` 作为搜索控制器接口
- `OFO` 作为单步候选打分器
- 当前 BFS / MCTS 作为 baseline 方法

#### 4.2 不建议继续重投入的部分

- 继续围绕 `top-50%` 截断策略做小修小补
- 继续仅靠 primitive edit 扩动作空间
- 仅通过调 `max_depth / max_branching / pruning_patience` 期待系统质变
- 仅把 `BFS` 换成 `MCTS` 期待根本提升

这些改法可能带来局部收益，但很难改变系统的能力上限。

---

### 5. 更值得探索的新方向

如果要从当前基础上继续推进，我更推荐转向下面这条主路线：

> **数据驱动的分子编辑动作空间 + OFO 打分 + 规划式搜索**

这是对现有系统最自然、也最有希望带来实质提升的升级方向。

---

### 6. 推荐的新主线：编辑规划而非继续暴力树搜

#### 6.1 核心思想

不要再把重点放在：

- “如何把现有树搜索得更深”
- “如何在 primitive 动作上做更复杂的剪枝”

而应转向：

- “如何让系统一开始就提出更像真实化学优化的动作”
- “如何让模型学会区分值得扩展的局部结构改动”
- “如何用规划器利用这些更高质量动作”

也就是说，研究重点应从：

> **search over primitive edits**

转向：

> **planning over chemically meaningful edits**

---

### 7. 动作空间应如何升级

这是后续最核心的研究点。

#### 7.1 Version A：保留最小 primitive action 集作为底座

保留当前最稳定、最容易做掩码的一组基本动作，例如：

- 加原子
- 替换原子
- 删末端原子 / 可逆边
- 改键级
- 成环 / 断指定类型键

用途：

- 作为基础可达性保证
- 用于和历史系统兼容
- 作为对照组 baseline

#### 7.2 Version B：引入片段级编辑

在 primitive action 之上，尽快引入片段/取代基层面的操作，例如：

- `attach_fragment(anchor, frag_id)`
- `replace_substituent(match, frag_id)`
- `grow_r_group(anchor, frag_id)`
- `bioisostere_swap(pattern_id)`

片段来源可优先考虑：

- BRICS 分解高频片段
- 数据集中高频取代基
- 从已有分子对 / 路径数据中抽取的局部模板

这一步非常关键，因为真实分子优化常见的不是“单个原子替换”，而是：

- 侧链生长 / 缩短
- 基团替换
- 局部异构等排体替换
- scaffold-preserving 的定点修饰

#### 7.3 Version C：引入数据驱动模板动作

如果后续能从 `(mol_from, mol_to)` 或 path 数据中自动抽模板，则建议升级为：

- 局部环境条件下的替换模板
- scaffold 上特定位点的常见修饰模板
- 针对目标属性改善频繁出现的 edit motif

此时动作不再只是：

- “在第 7 个原子加一个 N”

而会变成：

- “在芳环 para 位接入吸电子基”
- “把某类侧链替换为更疏水的等排体”
- “保 scaffold 前提下做局部杂原子替换”

这会显著提高：

- 化学合理性
- 搜索效率
- 路径可解释性
- 论文价值

---

### 8. `OFO` 后续应如何使用

后续不建议把 `OFO` 继续理解成“完整优化器”，而应将其明确降维成一个中间组件。

建议角色如下：

#### 8.1 近期角色

- **一步候选打分器**：评估 `s -> s'` 的属性变化
- **候选重排序器**：对动作生成器产出的候选做 rerank
- **搜索中的 expansion scorer**：为 `beam / best-first / MCTS` 提供局部分数

#### 8.2 中期角色

- 作为 reward shaping 的一部分
- 与 uncertainty / SA / diversity 等约束一起组成综合评分
- 为更高层 value model 提供训练轨迹或中间监督信号

#### 8.3 不建议直接期待的角色

- 单独充当全局 value function
- 单独定义动作空间
- 直接替代完整规划器

---

### 9. 搜索器应该怎么升级

#### 9.1 近期不建议继续主攻 BFS / MCTS 参数微调

当前 `BFS` / `MCTS` 可以作为 baseline 保留，但不建议继续把主要精力放在：

- 调层数
- 调分支数
- 调 patience
- 调 logP 约束
- 调 MCTS 超参数

因为这些更多是在救火，而不是在换引擎。

#### 9.2 更推荐的近期替代：Beam / Best-First

在动作空间升级后，更推荐先使用：

- `beam search`
- `best-first search`
- 轻量级 `A*`

原因：

- 更适合接 top-k 高质量动作
- 更容易分析预算与收益关系
- 比当前层内前 50% 截断更可控
- 更适合后续插入 value / heuristic 模型

#### 9.3 A* 应该在什么阶段引入

A* 是值得做的，但前提是先有：

- 更合理的候选动作空间
- 更干净的状态表示
- 一个像样的 `h(s)` 或未来价值估计

否则 A* 只是在 primitive 枚举空间上换一个更复杂的调度器，收益有限。

#### 9.4 RL 应该放在什么时候做

RL 不适合一开始就承担“直接生成分子”的职责。

更合适的角色是：

- 学 `policy`：在当前状态下优先提出哪些动作
- 学 `value`：从当前状态出发还能有多少未来收益
- 从搜索轨迹中离线学习，减少在线探索浪费

所以顺序应当是：

1. 先做动作空间升级
2. 再做搜索器升级
3. 最后再把 RL 接进来学 policy / value

---

### 10. `v0.3` / 路径模型的意义

当前 `v0.3` 路线往路径级预测演进，是一个非常正确的信号。

它说明项目已经意识到：

> **单步 delta predictor 不足以支撑真正的长程规划，必须进入路径级或 value 级建模。**

因此，中期很值得探索：

- 从 `start + operations + path_target` 学路径级累积收益
- 单独训练 `V(s)` 预测剩余可达收益
- 把路径模型作为 A* / best-first 的未来价值来源

这会比单纯继续堆单步 rerank 更有潜力。

---

### 11. 推荐的主路线结论

综合来看，后续最值得投入的主线应当明确为：

> **面向 OFO surrogate 的受约束分子编辑规划**

具体来说：

- **上游**：重做动作空间
- **中游**：保留 `OFO` 做单步打分
- **下游**：升级为真正的规划式搜索
- **中长期**：加入路径级 / value 级学习

这比继续围绕原始 `MO` 的 BFS/MCTS 系统做修补，更可能形成真正有研究价值和结果增益的新方向。

---

### 12. 具体落地建议（推荐顺序）

#### Phase 1：把当前系统降级为 baseline 平台

目标：保留现有能力，但不再当主路线。

建议：

- 保留 `batch_optimizer.py` 作为批量实验入口
- 保留 `EvolutionTreeOptimizer` 作为统一控制器
- 保留现有 `BFS / MCTS` 作为 baseline
- 用现有结果做后续新方法对照

#### Phase 2：实现新的 `ActionGenerator`

目标：从 primitive graph edit 升级到 chemically meaningful edits。

建议模块化拆分为：

- `ActionGenerator`
- `ActionMask`
- `ActionLibrary`（fragment/template）
- `CandidateDiversityFilter`

#### Phase 3：保留 `OFO` 做 scoring，替换搜索策略

目标：验证“动作空间升级”本身是否带来提升。

建议：

- 先用 `beam / best-first` 替代当前层内 top-50% 策略
- 引入多样性约束、SA 约束、结构相似性约束
- 比较 primitive-only 与 fragment/template action 的差异

#### Phase 4：加入路径级 / value 模型

目标：支持真正的长程规划。

建议：

- 以 `v0.3` 为基础探索路径累计预测
- 或独立训练 `V(s)` / heuristic model
- 在此基础上再认真做 `A* + RL`

---

### 13. `fragment_op` 的定义与编码建议

如果后续主线转向片段级 MO，那么最关键的问题不是“把 `add_atom` 改名成 `add_fragment`”，而是要**显式定义片段宏动作的数据结构**，并将其编码为 OFO 可消费的条件信息。

#### 13.1 推荐的动作定义原则

建议把一个 `fragment_op` 定义为：

> **在给定分子状态下，对某个锚点/局部子结构执行一次受约束的片段级替换、嫁接、生长或删除。**

这个定义需要同时包含：

- **动作类型**：这是哪一类宏动作
- **作用位置**：改动发生在什么局部环境 / 哪个锚点
- **片段身份**：引入、删除或替换的片段是谁
- **连接方式**：如何与母体分子连接
- **约束语义**：是否保持 scaffold、是否是 R-group 操作、是否属于某类模板替换

#### 13.2 推荐的 `fragment_op` schema

建议统一使用结构化字典，而不是继续沿用当前只包含 `atom / operation / position` 的轻量格式。

可采用类似下面的字段组织：

- `op_type`：`attach_fragment` / `replace_substituent` / `grow_r_group` / `bioisostere_swap` / `delete_fragment`
- `anchor`：锚点定义
  - `anchor_atom_indices`
  - `anchor_frag_idx`
  - `anchor_env_type`
- `fragment`：片段定义
  - `fragment_id`
  - `fragment_smiles`
  - `attachment_points`
  - `fragment_size`
- `leaving_group`：被替换或被删除的局部子结构（可选）
  - `leaving_fragment_smiles`
  - `matched_atom_indices`
- `connection`：连接方式
  - `bond_type`
  - `attachment_mapping`
  - `topology_change`
- `constraints`：动作约束
  - `scaffold_preserving`
  - `rgroup_only`
  - `ring_change`
  - `charge_change`
- `provenance`：动作来源
  - `source = brics | mmp | template | rule`
  - `template_id`

#### 13.3 不建议的定义方式

以下几种方式不建议作为主方案：

- 只保留一个 `fragment_id`，其余全靠 `from_smiles / to_smiles` 自己学
- 直接把整段 `fragment_smiles` 做 one-hot
- 继续复用当前 `operation + atom + position` 三元组，外加一个 `add_fragment` 标签就结束

原因是：

- 片段空间远大于原子空间，纯 ID / one-hot 很快失控
- 同一个片段在不同锚点和局部环境下，收益可能完全不同
- 不显式建模连接与约束，模型会把很多关键语义都丢给 `from/to` 图差分，训练会很吃力

#### 13.4 推荐的编码方式：分层编码，而不是单块 one-hot

推荐把 `fragment_op` 编码成四部分，再拼接成最终 `edge_attr`：

1. **动作类型编码**
   - `op_type embedding`
   - 建议用小词表 embedding，而不是 one-hot

2. **锚点与局部环境编码**
   - 锚点原子 / 片段索引
   - 锚点邻域的局部图表示
   - scaffold / ring / aromatic / degree / hetero 环境标签
   - 可继续复用当前 Laplacian PE 思路，但应升级为局部环境编码，而不只是单点位置编号

3. **片段身份编码**
   - 不建议对 `fragment_id` 做巨型 one-hot
   - 建议对 `fragment_smiles` 生成 learned embedding
   - 可优先使用 Morgan fingerprint、BRICS 片段指纹，或复用 `FragNet` 得到片段向量

4. **连接与约束编码**
   - `bond_type`
   - `attachment_mapping`
   - `scaffold_preserving`
   - `size_delta / ring_delta / hetero_delta / aromatic_delta`
   - 这些更适合作为显式数值或离散 embedding 输入

因此，一个更合理的总表示应为：

> **`edge_attr = [op_type_emb ; anchor_context_emb ; fragment_emb ; connection_features ; constraint_features]`**

#### 13.5 我更推荐的工程落地：显式拆成两路

从工程上，我更推荐把当前单一 `edge_attr` 思路升级成两路：

- **轻量动作元信息**
  - `op_type`
  - `bond_type`
  - `scaffold_preserving`
  - `size_delta`
  - `ring_delta`
- **片段内容表征**
  - `fragment_smiles` 对应的 fingerprint / graph embedding
  - `anchor local context` 对应的局部环境 embedding

再在 `EdgeFeatureExtractor` 或 fusion 层里进行拼接融合。

这样做的好处是：

- 保持 schema 清晰
- 允许后续替换片段 encoder
- 不需要把所有语义都硬塞进一个扁平 one-hot 向量

#### 13.6 与当前项目最兼容的过渡实现

考虑到当前项目已经有：

- `prepare_edge_features()` 这条边特征通路
- `FragNet` 相关片段结构特征工具
- `operations` 列表式样本格式

最务实的过渡方案是：

1. 先把 `operations[i]` 的元素从原来的轻量 dict 升级为 `fragment_op dict`
2. 在数据预处理里新增 `prepare_fragment_op_features()`
3. 第一版先输出：
   - `op_type one-hot / embedding`
   - `anchor position encoding`
   - `fragment fingerprint`
   - `size / ring / aromaticity delta`
4. 暂时不改动 from/to 分子编码主干，只替换 edge encoder 输入
5. 跑通后，再把 `fragment fingerprint` 升级为 learned fragment encoder

#### 13.7 推荐的三阶段编码路线

为了避免一开始把系统做得过重，我建议分三阶段推进：

- **Stage 1：可运行版本**
  - `op_type`
  - `anchor position`
  - `fragment fingerprint`
  - `size / ring / hetero delta`
  - 目标：先验证片段动作空间本身是否带来明显增益

- **Stage 2：语义增强版本**
  - 加入 `anchor local context`
  - 加入 `scaffold_preserving / rgroup_only / template_source`
  - 加入 leaving group 表征
  - 目标：提高对不同局部环境下同一片段动作的辨别能力

- **Stage 3：学习式动作编码版本**
  - 片段 encoder 从 fingerprint 升级为图网络 / FragNet encoder
  - 锚点环境也升级为局部子图 encoder
  - 目标：让 `OFO-frag` 真正成为片段级 transition scorer，而不是规则特征拼接器

#### 13.8 一个最关键的判断

从方法上讲，`fragment_op` 不是“一个标签”，而是：

> **一个带位置、带片段身份、带连接语义、带约束上下文的条件动作对象。**

只有这样定义，`OFO` 才能从“atom-op 单步打分器”自然升级成“fragment-level transition scorer”。

---

### 14. 一句话总结

> **原来的 MO 落地应被视作“可运行 baseline”，而不是未来主路线；真正值得投入的新方向，是“数据驱动的分子编辑动作空间 + OFO 单步打分 + 规划式搜索 + 路径价值建模”。**
