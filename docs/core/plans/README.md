# 分步实施计划总览（重构整理版）

> 本目录用于承接 `mol-ofo` 从 **primitive action baseline** 演进到 **双层动作表示（primitive execution trace + semantic fragment step）**、`OFO-frag-step`、新 planner 与 path/value 模型的整体路线。
>
> 这一轮整理后的总纲只保留一条主线：
>
> **先固定 baseline 平台 → 再定义双层动作协议 → 再搭数据主干与单步视图 → 再做单步模型 → 再做 planner → 最后做 path/value 与实验闭环**

---

## 1. 先统一三个核心共识

### 1.1 `atom / primitive path` 不是历史包袱，而是事实层

当前项目里最稳定、最可回放、最可审计的结构变化事实，仍然来自：

- `MoleculeEvolverAnalysis`
- canonical primitive `operations`
- 可逐步回放的 `evo-path`

因此后续路线**不能直接抛弃 primitive path**。

### 1.2 `fragment_op` 不是替代事实，而是语义层

新路线里的 `fragment_op` 不应被理解成“把 atom-op 全部改名成 fragment-op”，而应被理解成：

> **建立在 primitive execution trace 之上的语义压缩动作。**

也就是说：

- **底层**：`primitive_ops` / `primitive_path`
- **上层**：`semantic_step` / `fragment_op`

两层并存，而不是二选一。

### 1.3 `QM9` 只保留为 bootstrap

`QM9` 能帮助我们：

- 跑通 canonical pair/path 构造
- 跑通标注、导出、训练与日志链路
- 做 smoke test、debug、ablation

但它**不应**被当作长期的 fragment 主训练源。

---

## 2. 重构后的路线图

建议按下面的顺序推进：

> **baseline 平台 → 双层动作协议 → 数据主干与单步视图 → 单步模型 → planner → path/value → 实验闭环**

这里的关键变化是：

- 从“先想 fragment 再想 path”改成“先承认 primitive path 是事实层，再在其上抽 semantic step”
- 从“pair/path/planner 各自定义动作”改成“离线训练、在线候选、planner replay 共用一套动作协议”
- 从“QM9 frag 数据集构造”改成“数据主干先跑通，训练主源后续可切换”
- 从“数据计划 + 单步 pair 计划分开写”改成“单步视图作为数据导出层的一部分统一维护”

---

## 3. 文件顺序与职责边界

| 编号 | 文件 | 回答的问题 | 前置依赖 |
|------|------|------------|----------|
| 01 | `01-baseline-platform-plan.md` | baseline 为什么先固定、benchmark / I/O / 评估链以什么口径收口 | 无 |
| 01a | `01a-baseline-workflow-guide.md` | baseline 具体怎么跑 | 01 |
| 02 | `02-fragment-op-schema-plan.md` | 双层动作协议是什么，`fragment_op` 与 `primitive_trace` 如何共存 | 01 |
| 03 | `03-semantic-data-and-step-views-plan.md` | 数据主干如何围绕 primitive facts + semantic annotation 组织，并导出 `semantic pair / fragment pair / semantic path` | 01, 02 |
| 04 | `04-ofo-frag-model-plan.md` | 单步模型如何从 primitive edge scorer 升级为 semantic-step scorer | 02, 03 |
| 05 | `05-planner-upgrade-plan.md` | planner 如何消费 semantic actions 并保留可回放性 | 04 |
| 06 | `06-path-value-and-rl-plan.md` | semantic path、value、policy 与 RL 如何接上 | 04, 05 |
| 07 | `07-experiment-and-milestone-plan.md` | 如何做统一实验矩阵、里程碑与结论闭环 | 贯穿全程 |

---

## 4. 文档边界（避免重复定义）

- **`01`**：固定 baseline 平台的角色、benchmark、I/O 契约和评估链
- **`01a`**：只保留正式 baseline 操作手册
- **`02`**：只定义双层动作协议，不讨论训练细节
- **`03`**：统一定义数据主干、标注中间层与导出视图，包括单步 `semantic pair / fragment pair`
- **`04`**：只讨论单步模型，不把 planner/path 细节混进来
- **`05`**：只讨论 planner 内核、frontier、约束、去重与 replay
- **`06`**：只讨论 semantic path、value、policy、heuristic 与 RL，不重新设计 planner 内核
- **`07`**：只保留实验问题、对比矩阵、验收口径与里程碑

---

## 5. 当前建议的推进顺序

### Phase A：站稳 baseline 平台

先完成：

- benchmark 子集固定
- baseline 指标口径固定
- 输入输出与评估链固定

### Phase B：定义动作协议

先完成：

- `primitive_trace` 与 `semantic_step` 的关系
- `fragment_op` schema
- fallback / unresolved / approximate 策略

### Phase C：搭数据主干与单步视图

先完成：

- `primitive_pairs / primitive_paths`
- `semantic annotation`
- `split_manifest`
- `semantic_pairs / fragment_pairs / semantic_paths` 导出

### Phase D：升级单步模型

先完成：

- `prepare_semantic_step_features()` / `prepare_fragment_op_features()`
- semantic-step scorer
- primitive fallback 通路

### Phase E：升级 planner 与 path/value

先完成：

- planner 新接口
- semantic action replay
- path/value 接入

### Phase F：实验闭环

先完成：

- 最小实验矩阵
- 预算/效果对比
- 失败案例与迁移结论

---

## 6. 阶段-脚本-产物映射表

> 下面这张表的目的，是把 `plans` 从“概念路线图”再往前推一步，变成一份可直接指导实现排期的导航表。
>
> 其中：
>
>- **现有入口**：仓库里已经存在、可以复用或改造的脚本 / 模块
>- **计划新增**：当前计划里明确建议新增的入口或模块
>- **阶段产物**：完成该阶段时，应该能落盘或稳定存在的结果

| 阶段 | 主要任务 | 现有入口 / 可复用模块 | 计划新增 / 建议新增 | 阶段产物 |
|------|----------|----------------------|---------------------|----------|
| Phase A / `01` | 固定 baseline 平台 | `mol_evo/train_v0.py`、`mol_evo/predict_v0.py`、`mol_evo/scripts/batch_optimizer.py`、`utils/evaluate_batch_mo.py`、`utils/evaluate_csv_results.py` | 无强制新增，优先先固化运行模板与评估口径 | 固定 benchmark 子集、推荐 CLI 模板、稳定输出目录、baseline 结论页 |
| Phase A / `01a` | baseline 操作手册 | 同上 | 无 | 一份可复跑的 baseline workflow |
| Phase B / `02` | 双层动作协议定义 | `mol_evo/core/evolver.py`、`mol_evo/core/molecule_rebuilder.py`、`mol_evo/dataset/build_canonical_qm9_pairs.py`、`mol_evo/dataset/build_fragment_op_dataset.py` | `fragment_op.schema.json`、`fragment_op.examples.json`、协议词表说明 | `primitive_trace / semantic_step / fragment_op` 协议说明、fallback 纪律 |
| Phase C / `03` | 数据主干与单步视图 | `mol_evo/dataset/extract_evolution_pairs.py`、`mol_evo/dataset/calculate_property_changes.py`、`mol_evo/dataset/extract_operation_config.py`、`mol_evo/core/data/pair_to_path.py`、`mol_evo/core/data/path_processing.py` | `build_molecule_manifest.py`、`build_canonical_pairs.py`、`annotate_property_deltas.py`、`normalize_primitive_paths.py`、`annotate_semantic_steps.py`、`build_split_manifest.py`、`export_training_views.py` | `molecule_manifest.jsonl`、`primitive_pairs_*`、`primitive_paths_*`、`semantic_pairs_*`、`fragment_pairs_*`、`semantic_paths_*`、`split_manifest.json` |
| Phase D / `04` | 单步 semantic-step 模型 | `mol_evo/core/data/processing.py`、`mol_evo/core/data/unified_processing.py`、`mol_evo/core/data/path_processing.py`、现有 `core/models/*`、`mol_evo/train_v0.py` | `core/data/semantic_step_processing.py`、`core/models/.../semantic_step_edge_extractor.py`、`prepare_semantic_step_features()`、`prepare_fragment_op_features()` | 可训练的 `semantic_pairs` 入口、semantic-step scorer checkpoint、训练日志与对比结果 |
| Phase E / `05` | planner 内核与 replay | `mol_evo/core/evolution_optimizer.py`、`mol_evo/core/molecular_evolution_expansion.py`、`mol_evo/scripts/batch_optimizer.py`、`mol_evo/tests/test_astar.py`、`mol_evo/tests/test_astar_mol.py` | `core/planners/base.py`、`core/planners/beam.py`、`core/planners/best_first.py`、`core/planners/score_utils.py`、`core/planners/replay.py` | semantic planner 原型、稳定 replay、预算/约束/去重机制、与 baseline 的公平对比 |
| Phase E / `06` | long-range signals：path / value / policy / RL | `mol_evo/train_v0_3_path.py`、`mol_evo/core/data/path_processing.py`、`mol_evo/core/data/pair_to_path.py`、当前路径模型代码 | `core/data/semantic_path_processing.py`、`core/models/value/*`、`core/models/policy/*`、`core/planners/heuristics.py` | `semantic_paths_*`、value / policy checkpoint、planner heuristic 接口接入、长程收益对比 |
| Phase F / `07` | 实验矩阵与里程碑闭环 | `utils/evaluate_batch_mo.py`、`utils/evaluate_csv_results.py`、前面各阶段日志与结果目录 | 统一汇总脚本（如有需要再补） | `benchmark_results.csv`、`ablation_results.csv`、`planner_budget_analysis.csv`、`semantic_level_breakdown.csv`、`milestone_summary.md` |

---

## 7. 当前落地进度（截至 2026-04-08）

> 这里记录的是**代码仓库里的真实落地状态**，不是目标态。标记口径分为：
>
>- **已落地**：仓库里已有对应脚本 / schema / 接口
>- **待实跑验证**：入口与产物定义已齐，但还需要在目标服务器上用真实数据跑 smoke / bootstrap
>- **未完成**：仍停留在计划层，或只有零散前置模块

### 7.1 `02` 双层动作协议

- **状态**：**已落地（协议产物已补齐）**
- **已完成**：
  - `docs/core/schemas/fragment_op.schema.json`
  - `docs/core/schemas/fragment_op.examples.json`
  - `docs/core/schemas/semantic_step.schema.json`
  - `docs/core/schemas/semantic_step.examples.json`
  - `dataset/fragment_alignment_utils.py` 中的 `build_semantic_step()`
- **说明**：`semantic_step / fragment_op / primitive_ops` 三层关系已经能在代码里被同构表达，不再只存在于计划文档中。

### 7.2 `03` 数据主干与单步视图

- **状态**：**已落地入口，待实跑验证**
- **已完成**：
  - `dataset/build_fragment_op_dataset.py`
    - 能直接输出 `semantic_pairs_* / fragment_pairs_* / semantic_paths_*`
    - 能输出 `fragment_dataset_stats.json` 和 `fragment_action_config.yaml`
  - `dataset/annotate_semantic_steps.py`
    - 负责 annotation 阶段，输出 `semantic_pairs_annotated.jsonl` / `semantic_paths_annotated.jsonl`
  - `dataset/export_training_views.py`
    - 负责 train/valid/test 视图导出
- **当前约束**：
  - 当前输入协议以 **JSON / JSONL** 为主；还**没有**把现有评估 CSV 直接接成正式输入协议
  - `build_split_manifest.py`、`molecule_manifest.jsonl`、`primitive_pairs_* / primitive_paths_*` 的统一规范化链路还没补完
- **结论**：从“脚本能力”上看，**build 数据集的主入口已经接上**；从“工程验收”上看，还差一次服务器上的真实样本运行来把状态从“待实跑验证”升级成“已验证”。

### 7.3 `04` 单步 semantic-step 模型入口

- **状态**：**最小入口已落地**
- **已完成**：
  - `core/data/processing.py` 中新增 `prepare_semantic_step_features()`
  - `core/data/unified_processing.py` 中增加同名包装入口
- **未完成**：
  - 还没有把 `semantic_pairs_*` 正式接进完整训练主链
  - `prepare_fragment_op_features()`、semantic-step scorer 训练入口与 checkpoint 产物仍待补

### 7.4 `05 / 06 / 07`

- **状态**：**文档边界已收口，核心实现未开始**
- **说明**：当前不建议提前展开，优先把 `02 / 03 / 04` 的最短闭环真正跑通。

---

## 8. 服务器验证 CLI（建议按这个顺序）

> 下列命令假设你已经在仓库根目录，并使用 `conda activate mol-opt-evo` 进入环境。
>
> 目的不是一上来跑完整数据，而是先做一次 **smoke run**，确认 annotation / export / all-in-one build 三条链路都能落盘。

### 8.1 准备最小输入样本

如果你手头已经有正式的 pair/path `JSON / JSONL`，可以直接跳过这一步。否则可以先把现有评估 CSV 抽成一个最小 smoke 输入：

```bash
REPO=/path/to/mol-ofo && cd "$REPO" && eval "$(conda shell.zsh hook)" && conda activate mol-opt-evo && mkdir -p mol_evo/dataset/tests/smoke_inputs && python -c "import csv,json,pathlib; src=pathlib.Path('mol_evo/dataset/eval-data/20251205_131636/qm9_test_ab_pairs_gap_pairs.csv'); rows=[]; f=src.open(); reader=csv.DictReader(f); [rows.append(r) for _,r in zip(range(6), reader)]; f.close(); pairs=[]; paths=[]; [pairs.append({'pair_id': f'gap_smoke_pair_{i}', 'smiles_from': r['A_smiles'], 'smiles_to': r['B_smiles'], 'target_property': 'gap', 'step_target': float(r['improvement'])}) or paths.append({'path_id': f'gap_smoke_path_{i}', 'node_smiles_list': [r['A_smiles'], r['B_smiles']], 'step_targets': [float(r['improvement'])], 'path_target': float(r['improvement']), 'target_property': 'gap'}) for i,r in enumerate(rows)]; pathlib.Path('mol_evo/dataset/tests/smoke_inputs/gap_pairs_smoke.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in pairs), encoding='utf-8'); pathlib.Path('mol_evo/dataset/tests/smoke_inputs/gap_paths_smoke.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in paths), encoding='utf-8')"
```

### 8.2 跑 annotation 阶段

```bash
cd /Users/havoc420/Documents/Projects/whu/mol-ofo && eval "$(conda shell.zsh hook)" && conda activate mol-opt-evo && python mol_evo/dataset/annotate_semantic_steps.py --pairs_input mol_evo/dataset/tests/smoke_inputs/gap_pairs_smoke.jsonl --paths_input mol_evo/dataset/tests/smoke_inputs/gap_paths_smoke.jsonl --output_dir mol_evo/dataset/tests/smoke_outputs/annotated --max_workers 1
```

**预期产物**：

- `semantic_pairs_annotated.jsonl`
- `semantic_paths_annotated.jsonl`
- `semantic_annotation_stats.json`

### 8.3 跑 training views 导出阶段

```bash
cd /Users/havoc420/Documents/Projects/whu/mol-ofo && eval "$(conda shell.zsh hook)" && conda activate mol-opt-evo && python mol_evo/dataset/export_training_views.py --annotated_pairs_input mol_evo/dataset/tests/smoke_outputs/annotated/semantic_pairs_annotated.jsonl --annotated_paths_input mol_evo/dataset/tests/smoke_outputs/annotated/semantic_paths_annotated.jsonl --output_dir mol_evo/dataset/tests/smoke_outputs/views
```

**预期产物**：

- `semantic_pairs_train.jsonl` / `valid` / `test`
- `fragment_pairs_train.jsonl` / `valid` / `test`
- `semantic_paths_train.jsonl` / `valid` / `test`
- `training_view_stats.json`

### 8.4 跑一体化 build 入口

```bash
REPO=/path/to/mol-ofo && cd "$REPO" && eval "$(conda shell.zsh hook)" && conda activate mol-opt-evo && python mol_evo/dataset/build_fragment_op_dataset.py --pairs_input mol_evo/dataset/tests/smoke_inputs/gap_pairs_smoke.jsonl --paths_input mol_evo/dataset/tests/smoke_inputs/gap_paths_smoke.jsonl --output_dir mol_evo/dataset/tests/smoke_outputs/bootstrap --max_workers 1
```

**预期产物**：

- `semantic_pairs_train.jsonl` / `valid` / `test`
- `fragment_pairs_train.jsonl` / `valid` / `test`
- `semantic_paths_train.jsonl` / `valid` / `test`
- `fragment_dataset_stats.json`
- `fragment_action_config.yaml`

### 8.5 最低通过标准

如果满足下面 4 条，就可以把 `03` 的状态从“待实跑验证”更新为“已验证”：

1. 三个脚本都能正常退出，不报 Python / RDKit / import 错误
2. `annotated`、`views`、`bootstrap` 三个输出目录都能成功落盘
3. `semantic_pairs_*` 与 `semantic_paths_*` 文件非空
4. `fragment_dataset_stats.json` / `training_view_stats.json` 中 split size 合理，且 `fragment_pairs_*` 至少不是全空

---

## 9. 推荐实现顺序（按工程依赖）

如果从“先做最少闭环”来排，建议顺序是：

1. **先跑稳 `01 / 01a`**：把 baseline 平台和评估链钉住
2. **再做 `02`**：把动作对象定义清楚，避免后面返工
3. **然后做 `03`**：让 `semantic_pairs / fragment_pairs / semantic_paths` 真正能导出并完成服务器验线
4. **再做 `04`**：基于 `semantic_pair` 启动第一版单步训练
5. **接着做 `05`**：把 semantic planner 内核与 replay 跑起来
6. **最后做 `06`**：把 value / policy / heuristic 作为增强层接入
7. **全过程统一回到 `07`**：保证每一步都有可比实验，而不是只堆模块

---

## 10. 计划目录的维护原则

- **先承认事实层，再设计语义层**：不能把 `fragment_op` 写成脱离 `primitive_path` 的空中楼阁
- **先统一协议，再写下游模型**：pair/path/planner/actionlib 尽量共用同构动作对象
- **先留 fallback，再追求纯 fragment**：不能 fragmentize 的样本允许保留 atomic fallback
- **先以 QM9 验线，再迁移主数据源**：不要再把 QM9 包装成长期主训练源
- **每阶段必须有落盘产物**：schema、manifest、数据导出、checkpoint、日志、对比表必须可追溯
- **按任务边界收口，而不是按历史文件名拆分**：避免一个工程任务拆成多份高度重叠的计划
- **README 只做总纲与导航，不吞掉各阶段细节**：具体规则与实现约束应回到对应阶段文件维护

---

## 11. 推荐阅读顺序

1. 先看 `README.md`，确认总纲、阶段边界和“阶段-脚本-产物”映射
2. 再看 `01 / 01a`，对齐 baseline 平台口径
3. 再看 `02 / 03`，对齐动作协议、数据主干和单步视图
4. 做模型时看 `04`
5. 做搜索时看 `05`
6. 做长程建模时看 `06`
7. 做阶段结论时统一回到 `07`

---

## 12. 一句话总纲

> **新路线不是“用 fragment 替代 atom path”，而是“保留 primitive path 作为事实层，并在其上建立 semantic fragment step 作为训练、规划和价值建模的统一语义层”。**
