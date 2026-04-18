# OFO-MCTS 消融试验执行计划

## 1. 目的

本计划用于补齐论文中 OFO-guided MCTS 的**模块消融实验**，重点回答以下问题：

1. 预测 Δy（增量）而非绝对性质 y 本身，对搜索效果是否确有必要；
2. OFO 中的双分支几何编码、显式操作特征、3D 几何信息各自贡献了什么；
3. MCTS 内部 prior、leaf value、expansion ranking 三重作用是否各自不可或缺；
4. 剪枝与约束机制是否影响了最终结果的可信度。

本计划只关注**结构消融 / 模块消融**，不与超参数敏感性混在一起。超参数试验见 `ofo-mcts-超参数试验执行计划.md`。

---

## 2. 消融原则

- 固定同一个 OFO checkpoint，消融仅改变推理/搜索侧的配置或输入；
- 消融对比以**全模型（Full）**为基线，每个变体只移除或替换一个模块；
- 优先在**最小可行任务集**（LUMO(U)、HOMO(D)）上跑完，再扩展到标准集；
- 同一变体在 OFO 侧与 MCTS 侧分别评估：OFO 侧看 MAE / Rank Loss，MCTS 侧看改善值 / 成功率 / IntDiv；
- 主文只保留最有解释力的 5–7 项，其余放附录。

---

## 3. 推荐基线设置

消融实验的统一基线与超参数计划保持一致：

- OFO checkpoint：`visnet_linear_linear`（最佳基座）
- `search_mode = mcts`
- `num_simulations = 800`
- `exploration_weight = 2.0`
- `max_branching = 20`
- `max_depth = 10`
- `pruning_patience = 3`
- `logp_range = [-0.5, 6.0]`
- `logp_patience = 5`
- `topK = 20`

说明：所有消融变体均在此基线上做**单因素替换**，除非该消融本身需要调整搜索参数（如 w/o expansion ranking 需要全展开）。

---

## 4. 任务选择策略

与超参数计划一致，分三档推进：

### 4.1 最小可行集

- `LUMO(U)`：MCTS 收益最明显的任务，最能体现 prior / leaf value 的作用；
- `HOMO(D)`：BFS vs MCTS 已有显著差异的任务。

### 4.2 标准集

在最小可行集基础上增加：

- `HOMO(U)`：观察成功率提升与多样性下降的 trade-off 是否稳定。

### 4.3 完整集

- `LUMO(D)`：OFO 本身已较强的任务，可证明"并非所有任务都同样依赖搜索"。

---

## 5. OFO 侧消融

OFO 侧消融的目标是验证**单步响应建模的各设计选择是否必要**。每项消融都需要在两个层面评估：

- **单步预测层面**：在同一测试集上比较 MAE、Rank Loss；
- **多步优化层面**：将消融后的模型接入 MCTS，比较改善值、成功率、IntDiv。

### 5.1 Δy 增量 vs 绝对性质 y（必做 ★★★）

#### 消融方式

将监督目标从 Δy 替换为终态绝对性质 y^{to}，即模型直接预测编辑后分子的性质值，而非编辑导致的变化量。训练协议（backbone、操作编码器、融合头、数据划分）完全不变，仅替换回归目标。

#### 需要回答的问题

- 增量建模是否比绝对值建模更适合做候选排序？
- 在搜索场景下，Δy 的局部敏感性是否真正转化为更好的搜索引导？

#### 评估指标

| 层面 | 指标 |
|------|------|
| 单步预测 | MAE、Rank Loss |
| 多步优化 | 平均改善值、成功率、IntDiv |

#### 预期结论

- Δy 在 Rank Loss 上应优于绝对值，因为它天然聚焦于编辑带来的差异排序；
- 在多步优化中，Δy 的局部响应信号应更稳定，因为绝对值预测会混入分子本身的基线差异。

#### 实现注意

- 需要重新训练一个 `visnet_linear_linear` 的绝对值版本（监督目标改为 y^{to}）；
- 推理时，候选排序改为对预测终态值排序，而非对预测增量排序；
- MCTS 中的 prior 构造与 leaf value 累积方式需要相应调整。

#### 推荐呈现形式

- 主文一张表：Full (Δy) vs Abs (y)，分单步与多步两个板块；
- 正文一段讨论：说明增量建模的必要性。

---

### 5.2 w/o operation feature（必做 ★★★）

#### 消融方式

移除操作特征向量 e，模型输入仅保留编辑前后分子对，即：

$$\hat{\Delta y} = g_{\omega}([h_{\mathrm{from}}; h_{\mathrm{to}}])$$

#### 需要回答的问题

- 显式编辑条件是否对响应预测有贡献，还是仅靠分子结构变化就足够？

#### 评估指标

| 层面 | 指标 |
|------|------|
| 单步预测 | MAE、Rank Loss |
| 多步优化 | 平均改善值、成功率、IntDiv |

#### 预期结论

- 去掉操作特征后 Rank Loss 上升，说明操作语义对区分不同编辑类型的性质效应有帮助；
- 多步优化中，缺少操作条件可能导致候选排序变模糊。

#### 实现注意

- 推理时将操作特征向量置零或跳过拼接即可，无需重新训练（推荐重新训练以公平比较）。

---

### 5.3 w/o 3D geometry → 2D only（必做 ★★★）

#### 消融方式

将分子编码器从 VisNet（3D 几何感知）替换为 GCN（仅 2D 拓扑），保持其余结构不变。

#### 需要回答的问题

- 量子化学性质变化是否对三维几何信息敏感？

#### 评估指标

| 层面 | 指标 |
|------|------|
| 单步预测 | MAE、Rank Loss |
| 多步优化 | 平均改善值、成功率、IntDiv |

#### 预期结论

- 2D 编码在 MAE 和 Rank Loss 上均明显劣于 3D，支撑"几何感知建模是必要的"；
- 多步优化中，2D 模型的搜索引导质量下降。

#### 实现注意

- 使用已训练好的 `gcn_linear_linear` 作为消融变体即可（表 5-1 已有数据）；
- 需要将其接入 MCTS 跑多步优化。

#### 推荐呈现形式

- 可直接复用表 5-1 的基座筛选数据作为单步层面的证据，再补 MCTS 层面的结果。

---

### 5.4 双分支 vs 单分支（建议做 ★★）

#### 消融方式

将双分支独立编码改为单分支差分编码：

$$h_{\Delta} = E_{\theta}(G_{\mathrm{to}}) - E_{\theta}(G_{\mathrm{from}})$$

或仅使用终态分子表示：

$$\hat{\Delta y} = g_{\omega}([E_{\theta}(G_{\mathrm{to}}); h_e])$$

#### 需要回答的问题

- 双分支是否比单分支更适合建模性质增量？

#### 评估指标

| 层面 | 指标 |
|------|------|
| 单步预测 | MAE、Rank Loss |
| 多步优化 | 平均改善值、成功率、IntDiv |

#### 预期结论

- 双分支保留了 from/to 的独立结构信息，比差分或单侧表示更适合捕捉编辑导致的局部变化。

#### 实现注意

- 需要重新训练单分支版本，保证对比公平；
- 若时间不够，可只做单步预测层面的对比。

---

### 5.5 from+to 分子对 vs to-only（可选 ★）

#### 消融方式

去掉编辑前分子 M^{from} 的输入，模型仅接收编辑后分子与操作特征：

$$\hat{\Delta y} = g_{\omega}([h_{\mathrm{to}}; h_e])$$

#### 需要回答的问题

- 编辑前分子的结构信息是否对响应预测有独立贡献？

#### 预期结论

- to-only 变体本质退化成静态性质预测 + 操作条件，缺乏"转移"语义，性能应明显下降。

#### 实现注意

- 需要重新训练；
- 优先级低，若 5.4 已证明双分支有效，此项可省略。

---

## 6. MCTS 侧消融

MCTS 侧消融的目标是验证**OFO 信号在搜索中的三重作用（expansion score / prior / leaf value）是否各自不可或缺**，以及**约束与剪枝是否影响了结果可信度**。

所有 MCTS 侧消融均使用同一个 OFO checkpoint，仅改变搜索侧逻辑。

### 6.1 w/o prior（必做 ★★★）

#### 消融方式

将 PUCT 中的先验概率 P(n') 替换为均匀分布：

$$P_i = \frac{1}{|\mathcal{C}_t|}$$

即不再用 OFO 分数构造 prior，所有候选在搜索初期享有相同引导。

#### 需要回答的问题

- OFO 的作用是否仅限于叶节点打分，还是在搜索方向引导上也有贡献？

#### 评估指标

- 平均改善值
- 成功率
- IntDiv
- 平均运行时间

#### 预期结论

- 去掉 prior 后，搜索需要更多模拟才能聚焦到高价值区域，导致同样预算下改善值下降；
- 若 prior 无贡献，说明 OFO 只是一个 leaf scorer；若 prior 有贡献，说明 OFO 同时充当了搜索引导器。

#### 实现注意

- 代码改动小：在 prior 构造步骤中将 softmax 输出替换为均匀分布即可；
- expansion score 和 leaf value 保持不变。

#### 推荐呈现形式

- 主文一张表：Full vs w/o prior vs w/o leaf value vs w/o expansion ranking；
- 可合并为一张 MCTS 侧消融总表。

---

### 6.2 w/o leaf value（必做 ★★★）

#### 消融方式

不再使用累计局部响应 G(n) 作为叶节点价值，改为：

- 方案 A：叶节点价值恒为零（即回传不依赖 OFO 信号，完全靠 prior + 访问统计）；
- 方案 B：叶节点价值仅使用当前步的 δ（不累计，仅看单步增益）。

推荐方案 A，因为它更干净地隔离了 leaf value 的贡献。

#### 需要回答的问题

- 路径级累计信号对回传是否重要？OFO 不只是"候选筛选器"？

#### 评估指标

- 平均改善值
- 成功率
- IntDiv

#### 预期结论

- 去掉 leaf value 后，Q(n) 完全靠先验驱动，无法根据实际搜索结果修正方向；
- 成功率与改善值应明显下降，尤其在长路径任务上。

#### 实现注意

- 方案 A：在回传步骤中将 V=0；
- 方案 B：在回传步骤中 V = δ_current（仅当前步，不累计）。

---

### 6.3 w/o expansion ranking / 随机候选（必做 ★★★）

#### 消融方式

在扩展阶段，不再按 OFO 分数排序后截取 TopB，而是：

- **随机 TopB**：从全部合法候选中随机选取 B 个；
- **全展开**：不做截断，展开全部合法候选（设 max_branching = K_t）。

#### 需要回答的问题

- OFO 的前置排序是否真的节省了搜索预算并提高了搜索质量？

#### 评估指标

- 平均改善值
- 成功率
- IntDiv
- 平均运行时间
- 展开节点数

#### 预期结论

- 随机 TopB：改善值下降，说明排序有效；
- 全展开：改善值可能略高但运行时间暴增，说明截断是效率-性能权衡下的合理选择。

#### 实现注意

- 随机 TopB：在扩展排序步骤中替换为随机采样；
- 全展开：设 max_branching 为一个极大值（如 999），或在代码中跳过截断逻辑。

#### 推荐呈现形式

- 与 6.1、6.2 合并为 MCTS 侧消融总表；
- 若全展开时间过长，可只在 LUMO(U) 上跑一组对比。

---

### 6.4 w/o pruning（建议做 ★★）

#### 消融方式

去掉无改善耐心剪枝（pruning_patience → ∞），仅保留深度约束与 logP 约束。

#### 需要回答的问题

- 剪枝是否只是节省时间，还是也影响了搜索质量？

#### 评估指标

- 平均改善值
- 成功率
- IntDiv
- 平均运行时间
- 展开节点数

#### 预期结论

- 去掉剪枝后运行时间增加，但改善值未必提升（低价值路径被保留但不带来收益）；
- 可以支撑"剪枝主要是效率工具，而非人为制造性能提升"。

---

### 6.5 w/o logP constraint（建议做 ★★）

#### 消融方式

去掉 logP 可行域约束，搜索不再限制分子的 logP 范围。

#### 需要回答的问题

- logP 约束是否排除了真正有价值的分子，还是主要防止搜索偏离药化可行域？

#### 评估指标

- 平均改善值
- 成功率
- IntDiv
- 优化后分子平均 logP
- 优化后分子平均 QED

#### 预期结论

- 去掉约束后改善值可能略升，但优化后分子的 logP 分布可能偏离合理范围；
- 支撑"当前约束属于启发式近似，后续可替换为更精细的药化约束"。

#### 推荐呈现形式

- 附录优先；
- 若结果很说明问题（如无约束下 logP 飙升），可在主文 discussion 中引用。

---

### 6.6 BFS vs MCTS（已有，保留）

这组对比已在表 5-2 中通过 OFO-BFS / OFO-MCTS 结果呈现，不需要额外跑。它在消融体系中的定位是"搜索机制本身的消融"，证明搜索策略的贡献不等于局部评分器的贡献。

---

## 7. 消融总表结构建议

主文建议合并为以下结构：

### 7.1 OFO 侧消融表

| 变体 | MAE ↓ | Rank Loss ↓ | 平均改善值 ↑ | 成功率 ↑ | IntDiv |
|------|-------|-------------|-------------|---------|--------|
| Full (Δy, dual-branch, op feat, 3D) | | | | | |
| Abs target (y^{to}) | | | | | |
| w/o operation feature | | | | | |
| w/o 3D → GCN | | | | | |
| w/o dual-branch → single | | | | | |

### 7.2 MCTS 侧消融表

| 变体 | 平均改善值 ↑ | 成功率 ↑ | IntDiv | 运行时间 |
|------|-------------|---------|--------|---------|
| Full | | | | |
| w/o prior (uniform) | | | | |
| w/o leaf value (V=0) | | | | |
| w/o expansion ranking (random) | | | | |
| BFS | | | | |

### 7.3 附录级消融表

| 变体 | 平均改善值 ↑ | 成功率 ↑ | IntDiv | 运行时间 |
|------|-------------|---------|--------|---------|
| Full | | | | |
| w/o pruning | | | | |
| w/o logP constraint | | | | |
| w/o expansion ranking (full expand) | | | | |

---

## 8. 执行顺序

建议严格按以下顺序推进：

### 阶段 A：OFO 侧消融（单步预测层面）

- [ ] 训练 Abs target (y^{to}) 版本，评估 MAE / Rank Loss；
- [ ] 训练 w/o operation feature 版本，评估 MAE / Rank Loss；
- [ ] 整理 GCN vs VisNet 的已有数据（表 5-1），补齐 Rank Loss；
- [ ] （可选）训练单分支版本，评估 MAE / Rank Loss。

### 阶段 B：OFO 侧消融（多步优化层面）

- [ ] 将 Abs target 模型接入 MCTS，在 LUMO(U) + HOMO(D) 上跑优化；
- [ ] 将 w/o op feat 模型接入 MCTS，在 LUMO(U) + HOMO(D) 上跑优化；
- [ ] 将 GCN 模型接入 MCTS，在 LUMO(U) + HOMO(D) 上跑优化。

### 阶段 C：MCTS 侧消融

- [ ] w/o prior：替换先验为均匀分布，在 LUMO(U) + HOMO(D) 上跑；
- [ ] w/o leaf value：设置 V=0，在 LUMO(U) + HOMO(D) 上跑；
- [ ] w/o expansion ranking：替换为随机 TopB，在 LUMO(U) + HOMO(D) 上跑；
- [ ] （可选）全展开对比，在 LUMO(U) 上跑一组。

### 阶段 D：附录级消融

- [ ] w/o pruning：在 LUMO(U) 上跑；
- [ ] w/o logP constraint：在 LUMO(U) 上跑。

### 阶段 E：结果整理

- [ ] 汇总 OFO 侧消融表与 MCTS 侧消融表；
- [ ] 为每组消融撰写结论段落；
- [ ] 将可直接写入 5-experiments.md 的文字整理好。

### 8.1 统一运行单元与重复策略

为避免“不同任务、不同种子、不同样本切片”混在一起，建议把一次正式实验定义为：

- **1 个 run unit = 1 个 task × 1 个 variant × 1 个 seed**；
- **同一 task 的所有 variant 必须使用同一批起始分子**，即固定 `input_csv + start_index + end_index`；
- **同一 task 的 Full 基线必须和所有 ablation 在同一轮复跑一次**，不要直接复用很久以前的旧结果；
- **主文结果建议至少 3 个 seed**：`42 / 43 / 44`；
- **预筛阶段**可只用 `seed=42`，但进入主文表格前必须补齐多 seed 均值与标准差。

建议采用三层执行节奏。**这里以当前脚本实现为准**（详见 `ofo-mcts-消融试验-pilot执行计划.md`）：

- **Smoke**：`5` 个起始分子，`1` 个 seed，仅检查命令、输出目录和指标链路是否打通；
- **Pilot**：`20` 个起始分子，`1` 个 seed，用于确认趋势是否符合预期；
- **Official**：`50` 个起始分子，`3` 个 seed，用于论文主表；若后续需要扩到 `100`，应显式覆盖 `START_INDEX/END_INDEX` 或同步修改脚本默认值。

### 8.2 目录、命名与归档规范

建议为消融单独建立统一输出根目录，而不是散落在普通 `batch_optimization_*` 目录中：

```bash
REPO=/home/ubuntu/mol_opt/mol-ofo
ABL_ROOT=$REPO/mol_evo/output/paper/ablations
RUN_TAG=$(date +%Y%m%d)
```

每个 run unit 推荐使用如下层级：

```text
$ABL_ROOT/$RUN_TAG/
  ofo_side/
    lumo_up/abs_target/seed42/
    lumo_up/wo_opfeat/seed42/
  mcts_side/
    lumo_up/wo_prior/seed42/
    homo_down/wo_leaf_value/seed42/
```

每个 `seed` 目录下至少保存以下路径或软链接：

- **`train_dir.txt`**：OFO 训练输出目录；
- **`predict_dir.txt`**：`predict_v0_testset.py` 的评估输出目录；
- **`search_dir.txt`**：`batch_optimizer.py` 的输出目录；
- **`eval_dir.txt`**：`utils/evaluate_batch_mo.py` 输出目录；
- **`csv_eval_dir.txt`**：`utils/evaluate_csv_results.py` 输出目录；
- **`manifest.json`**：记录 task、variant、seed、checkpoint、样本切片、关键参数。

建议固定以下 slug，避免后处理时名称不统一：

- **OFO 侧**：`full`、`abs_target`、`wo_opfeat`、`gcn2d`、`single_branch`
- **MCTS 侧**：`full`、`wo_prior`、`wo_leaf_value`、`random_topb`、`full_expand`、`wo_pruning`、`wo_logp`

### 8.3 OFO 侧执行模板

#### Step A：训练 / 准备 checkpoint

需要重新训练的变体：

- `abs_target`
- `wo_opfeat`
- `single_branch`（若执行）

可直接复用已有 checkpoint 的变体：

- `full`：`visnet_linear_linear`
- `gcn2d`：`gcn_linear_linear`

训练命令建议固定成如下模板：

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 python mol_evo/train_v0.py \
  --data-file "$TRAIN_DATA" \
  --target-property "$TARGET_PROPERTY" \
  --model-type "$MODEL_TYPE" \
  --max-pairs 120000 \
  --epochs 200 \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LR" \
  --seed "$SEED"
```

执行约束：

- **Abs target**：只改监督目标，从 `Δy` 改为 `y^{to}`；
- **w/o op feat**：推荐重新训练，而不是只在推理时把操作向量置零；
- **single-branch**：若时间不足，可只做单步预测，不强行接 MCTS。

#### Step B：单步预测评估

所有 OFO 侧变体都应在同一测试集上调用：

```bash
cd "$REPO"
python mol_evo/predict_v0_testset.py \
  --model-path "$MODEL_PATH" \
  --model-dir "$MODEL_DIR" \
  --data-file "$TEST_DATA" \
  --seed "$SEED" \
  --batch-size 64
```

记录来源：

- `full_dataset_prediction_results.json`
- 取其中的 `metrics.mae` 与 `metrics.rank_loss`

#### Step C：接入 MCTS 做多步优化

```bash
cd "$REPO"
python -m mol_evo.scripts.batch_optimizer \
  --input-csv "$INPUT_CSV" \
  --model-path "$MODEL_PATH" \
  --model-dir "$MODEL_DIR" \
  --config-file "$CONFIG_FILE" \
  --target-property "$TARGET_PROP" \
  --optimization-mode sub \
  --search-mode mcts \
  --num-simulations 800 \
  --exploration-weight 2.0 \
  --direction "$DIRECTION" \
  --max-depth 10 \
  --max-branching 20 \
  --pruning-patience 3 \
  --logp-min -0.5 \
  --logp-max 6 \
  --logp-patience 5 \
  --topK 20 \
  --start-index "$START_INDEX" \
  --end-index "$END_INDEX"
```

完成搜索后，统一调用两级评估：

```bash
cd /home/ubuntu/mol_opt
python utils/evaluate_batch_mo.py \
  --target-prop "$TARGET_PROP" \
  --direction "$DIRECTION" \
  --item-size 20 \
  --result-dir "$SEARCH_DIR"

python utils/evaluate_csv_results.py \
  --csv-file "$BATCH_EVAL_CSV" \
  --target-prop "$TARGET_PROP" \
  --direction "$DIRECTION" \
  --max-opt-molecules 20
```

### 8.4 MCTS 侧执行设计与最小代码接口

`MCTS` 侧消融里，`w/o pruning` 与 `w/o logP constraint` 已经可以直接用现有参数表达；其余 `prior / leaf value / expansion ranking` 三项，建议显式做成 CLI 开关，而不是靠临时改代码反复手工 patch。

建议新增三个搜索侧参数：

```text
--mcts-prior-mode {softmax,uniform}
--mcts-value-mode {accumulated,zero,step}
--mcts-expansion-mode {topk,random_topk,full}
```

推荐变体与参数映射如下：

| variant | prior_mode | value_mode | expansion_mode | 其他参数 |
|------|------------|------------|----------------|---------|
| Full | `softmax` | `accumulated` | `topk` | 基线参数 |
| w/o prior | `uniform` | `accumulated` | `topk` | 其余不变 |
| w/o leaf value | `softmax` | `zero` | `topk` | 其余不变 |
| w/o expansion ranking | `softmax` | `accumulated` | `random_topk` | 其余不变 |
| full expand | `softmax` | `accumulated` | `full` | 仅建议在 `LUMO(U)` 上跑 |

其中：

- **`w/o pruning`**：直接设置 `--pruning-patience 0`；
- **`w/o logP constraint`**：直接设置 `--logp-patience 0`，并将 `logp-min/max` 设为宽松占位值；
- **`w/o prior`** 对应 `generate_expansion_tree_mcts()` 中 prior 归一化处；
- **`w/o leaf value`** 对应 `_evaluate_leaf()`；
- **`w/o expansion ranking`** 对应候选排序截断 `sorted_cands[:max_branching]` 这一段。

如果暂时不改 CLI，也至少要保证三件事：

- **每个变体都有单独 commit 或 patch 文件**；
- **每次运行前把变体名写入 `manifest.json`**；
- **Full 必须在同一代码版本上复跑一次**，避免拿不同代码版本的旧结果做对照。

### 8.5 论文口径的统计规则

为避免 `TopK` 多个候选把同一个起始分子重复计算，建议论文主表统一采用“**每个起始分子只取 best-of-topK**”的口径：

- **`avg_improvement`**：从 `best_results_*.csv` 中读取每个起始分子的最佳结果；若 `direction=decrease`，则记 `paper_improvement = -improvement`，若 `direction=increase`，则 `paper_improvement = improvement`；最后对所有起始分子求均值；
- **`success_rate`**：`paper_improvement > 0` 的起始分子占比；
- **`IntDiv`**：读取 `statistics_summary_*.json` 中的 `intdiv_best`；
- **`Morgan_sim`**：优先读取 `best_results_*.csv` 中 `morgan_similarity` 列的均值；
- **`runtime`**：优先从 `batch_results.json` 中逐分子 `runtime` 求均值；若缺失，则用主日志总耗时除以起始分子数；
- **`expanded_nodes`**：从每个搜索树 JSON 的 `mcts_stats.unique_states_expanded` 求平均。

单步预测侧的统计口径固定为：

- **`MAE`**：`full_dataset_prediction_results.json -> metrics.mae`
- **`Rank Loss`**：`full_dataset_prediction_results.json -> metrics.rank_loss`

多 seed 汇总建议：

- **主文表格**：填 `mean ± std`；
- **附录表格**：保留每个 seed 的明细；
- **若某个 run 缺 `batch_results.json / batch_evaluation_results.csv / statistics_summary_*.json` 三者之一，则该 run 视为无效，不并入汇总。**

### 8.6 最小可执行矩阵（按当前进展更新）

截至 `2026-04-14`，MCTS 侧 `pilot` 已完成，当前建议的执行顺序更新为：

1. **OFO 侧 - `LUMO(U)`**
   - `full`
   - `abs_target`
   - `wo_opfeat`
   - `gcn2d`
2. **MCTS 侧 - `official` / `LUMO(U)` + `HOMO(D)`**
   - `full`
   - `wo_prior`
   - `wo_leaf_value`
   - `random_topb` 已完成 `pilot`，但当前未表现出优于 `full` 的稳定价值，先不进入主文 `official` 主矩阵
3. **附录级**
   - `wo_pruning`
   - `wo_logp`
   - `full_expand`（仅 `LUMO(U)`）

这样做的好处是：

- **先证明 OFO 信号本身是必要的**；
- **再用 `official` 多 seed 把最关键的 MCTS 结论压实**；
- **把 `random_topb` 收敛到 `pilot` 证据层，避免主文级算力被低优先级变体占用**；
- **最后再处理 pruning / constraint 这种“可信度与效率”类补充论证。**

---

## 9. 结果记录模板

### 9.1 OFO 侧消融

建议把**论文主表**和**执行台账**分开记录。

#### 主表字段

| task | variant | MAE | Rank Loss | avg_improvement | success_rate | IntDiv | Morgan_sim | notes |
|------|---------|-----|-----------|----------------|-------------|--------|-----------|-------|
| LUMO(U) | Full | | | | | | | |
| LUMO(U) | Abs target | | | | | | | |
| LUMO(U) | w/o op feat | | | | | | | |
| LUMO(U) | w/o 3D (GCN) | | | | | | | |

#### 执行台账字段

| task | variant | seed | model_dir | predict_json | best_results_csv | stats_json | MAE | Rank Loss | avg_improvement | success_rate | IntDiv | notes |
|------|---------|------|-----------|--------------|------------------|------------|-----|-----------|----------------|-------------|--------|-------|
| LUMO(U) | Full | 42 | | | | | | | | | | |
| LUMO(U) | Abs target | 42 | | | | | | | | | | |

### 9.2 MCTS 侧消融

#### 主表字段

| task | variant | avg_improvement | success_rate | IntDiv | Morgan_sim | runtime | expanded_nodes | notes |
|------|---------|----------------|-------------|--------|-----------|---------|---------------|-------|
| LUMO(U) | Full | | | | | | | |
| LUMO(U) | w/o prior | | | | | | | |
| LUMO(U) | w/o leaf value | | | | | | | |
| LUMO(U) | w/o expansion ranking | | | | | | | |

#### 执行台账字段

| task | variant | seed | search_dir | batch_json | batch_eval_csv | stats_json | best_results_csv | avg_improvement | success_rate | IntDiv | runtime | expanded_nodes | notes |
|------|---------|------|-----------|-----------|----------------|------------|------------------|----------------|-------------|--------|---------|---------------|-------|
| LUMO(U) | Full | 42 | | | | | | | | | | | |
| LUMO(U) | w/o prior | 42 | | | | | | | | | | | |
| LUMO(U) | w/o leaf value | 42 | | | | | | | | | | | |

---

## 10. 论文写作时的预期结论模板

### 10.1 OFO 侧

1. **增量建模的必要性**：将监督目标从 Δy 替换为绝对性质 y^{to} 后，模型在 Rank Loss 上显著上升，且多步优化中成功率下降。这表明增量建模并非仅仅是回归目标的简单替换，而是天然更适合为搜索提供候选排序信号。
2. **显式操作特征的贡献**：去掉操作特征后，模型丧失了对不同编辑语义的条件区分能力，导致 Rank Loss 上升与候选排序变模糊。
3. **3D 几何信息的必要性**：从 VisNet 退化为 GCN 后，单步预测与多步优化的性能均明显下降，证实量子化学性质变化对局部三维结构敏感。
4. **双分支编码的优势**：双分支结构保留了编辑前后分子的独立结构信息，比单分支差分或单侧编码更适合捕捉编辑导致的局部性质变化。

### 10.2 MCTS 侧

1. **Prior 的引导作用**：去掉 OFO prior 后，搜索在同等预算下无法有效聚焦到高价值分支，说明 OFO 不只是叶节点打分器，同时也是搜索方向的引导器。
2. **Leaf value 的回传贡献**：去掉叶节点价值后，回传完全依赖先验与访问统计，无法根据实际搜索结果修正方向，导致成功率下降。
3. **Expansion ranking 的筛选效率**：随机候选截断导致改善值下降，而全展开虽可能带来边际收益但运行时间暴增，说明 OFO 的前置排序在搜索效率与质量之间实现了有效平衡。
4. **BFS vs MCTS**：在固定局部评分器的前提下，MCTS 通过层次化决策进一步提升了搜索效率，但这种提升伴随着探索范围收缩与多样性下降。

---

## 11. 最小可执行版本（推荐先跑）

如果当前时间或算力有限，建议先完成下面这套最小版本：

### OFO 侧（3 项）

- **Δy vs y^{to}**：训练绝对值版本 + 接入 MCTS 跑 LUMO(U)
- **w/o operation feature**：训练无操作特征版本 + 接入 MCTS 跑 LUMO(U)
- **w/o 3D (GCN)**：已有基座数据，补跑 MCTS 优化

### MCTS 侧（当前主文优先 3 项）

- **w/o prior**：均匀先验，跑 `LUMO(U)` + `HOMO(D)`，并在 `official` 中补齐 `42 / 43 / 44` 三个 seed
- **w/o leaf value**：`V=0`，跑 `LUMO(U)` + `HOMO(D)`，作为当前最关键的负面对照
- **w/o expansion ranking**：随机 `TopB` 已完成 `pilot`，当前不进入主文 `official` 主矩阵；若后续篇幅允许，可作为附加或附录结果保留

这 6 项结果已经足以支撑主文中"增量建模必要""操作条件必要""几何感知必要""OFO 三重作用各自不可或缺"的核心论点；其中 MCTS 侧当前最优先压实的是 `w/o prior` 与 `w/o leaf value`。

---

## 12. 建议文件产出

执行本计划后，建议同步产出以下材料：

- 两张消融总表（OFO 侧 + MCTS 侧），Markdown 格式；
- 一张附录级消融表（pruning / logP constraint / full expand）；
- 一段可直接并入 `5-experiments.md` 的 Ablation study 小节，包含 5.4.1 OFO-side ablations 与 5.4.2 Search-side ablations；
- 一段可并入 Discussion 的消融结论总结。

总体上，这组消融试验的目标不是"展示所有变体"，而是**用最少的实验证明 OFO 的每个核心设计选择都是必要的，以及 MCTS 中 OFO 的三重作用各自不可或缺**。
