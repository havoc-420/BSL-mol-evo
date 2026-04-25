# OFO-MCTS 消融试验 Pilot 执行计划

## 1. 文档目标

本文档是 `ofo-mcts-消融试验执行计划.md` 的 **Pilot 专项落地版**，用于承接已完成的 `smoke` 结果，并给出下一阶段可直接执行的 `pilot` 级实验方案。

它重点回答三件事：

1. 当前这轮 `smoke` 已经给了我们什么信号；
2. `pilot` 在当前代码实现里到底指什么规模；
3. 下一步应该如何以最小成本把 `smoke` 的迹象升级为可决策证据。

> 说明：本文件优先面向 **当前最紧迫的一轮 MCTS 侧消融 pilot**。`OFO` 侧消融需要先补训练与 checkpoint 准备，不纳入本轮直接执行范围。

---

## 2. 与主计划的关系

与主计划的关系如下：

- `ofo-mcts-消融试验执行计划.md`：负责解释**为什么要做这些消融**、整体结构如何组织；
- **本文档**：负责定义**下一轮 pilot 具体怎么跑**，包括任务范围、实验矩阵、命令模板、评估产物、验收标准和进入 `official` 的判据。

如果两份文档存在口径冲突，**以当前脚本实现和本文档为准**。

---

## 3. 当前状态总结（基于已完成 smoke）

### 3.1 已完成内容

当前已经完成一轮 `MCTS` 侧 `smoke`：

- **tasks**：`lumo_up`、`homo_down`
- **variants**：`full`、`wo_prior`、`wo_leaf_value`、`random_topb`
- **seed**：`42`
- **真值评估**：已跑通 `evaluate_batch_mo.py`、`evaluate_csv_results.py`
- **论文口径汇总**：已可自动生成 `true_eval_summary.tsv` 与 `true_eval_summary_aggregated.tsv`

### 3.2 当前 smoke 的论文口径结果

以下数值来自已生成的 `true_eval_summary_aggregated.tsv`。

#### `LUMO(U)`

| variant | paper_avg_improvement | success_rate | IntDiv | avg_runtime | avg_expanded_nodes |
|--------|------------------------|--------------|--------|-------------|--------------------|
| `full` | 1.620682 | 1.000000 | 0.916881 | 4.474966 | 7.600000 |
| `wo_prior` | 1.485394 | 1.000000 | 0.857715 | 4.079797 | 6.000000 |
| `wo_leaf_value` | 1.435364 | 1.000000 | 0.931254 | 21.619826 | 40.800000 |
| `random_topb` | 1.867929 | 1.000000 | 0.896864 | 10.001655 | 17.200000 |

#### `HOMO(D)`

| variant | paper_avg_improvement | success_rate | IntDiv | avg_runtime | avg_expanded_nodes |
|--------|------------------------|--------------|--------|-------------|--------------------|
| `full` | 1.825361 | 1.000000 | 0.848845 | 1.538823 | 4.400000 |
| `wo_prior` | 1.825361 | 1.000000 | 0.848845 | 1.549716 | 4.400000 |
| `wo_leaf_value` | 0.775134 | 1.000000 | 0.868815 | 12.702796 | 39.000000 |
| `random_topb` | 1.689350 | 1.000000 | 0.924889 | 2.035209 | 6.200000 |

### 3.3 从 smoke 得到的直接结论

- **`wo_leaf_value` 已经出现稳定退化信号**：
  - 两个任务上的 `paper_avg_improvement` 都低于 `full`；
  - 同时 `avg_runtime` 与 `avg_expanded_nodes` 大幅升高；
  - 这说明 `leaf value` 很可能既影响效果，也影响效率。

- **`wo_prior` 证据不足**：
  - `LUMO(U)` 上略弱于 `full`；
  - `HOMO(D)` 上与 `full` 几乎完全重合；
  - 目前还不足以支撑“prior 是否必要”的稳定结论。

- **`random_topb` 仍是波动项**：
  - `LUMO(U)` 上看起来比 `full` 更高；
  - `HOMO(D)` 上则弱于 `full`；
  - 同时运行时间与展开节点显著更多，说明它很可能是“样本较小 + 随机性”的混合信号。

因此，`pilot` 的主要作用不是重新证明链路能跑通，而是：

- 验证 `wo_leaf_value` 的退化是否在更大样本上仍成立；
- 判断 `wo_prior` 的差异究竟是真差异还是 `smoke` 噪声；
- 判断 `random_topb` 是否值得进入 `official` 主表。

---

## 4. Pilot 的定义（按当前代码实现）

### 4.1 当前脚本中的 profile 定义

按 `run_mcts_ablations.sh` 当前实现：

- **`smoke`**
  - `START_INDEX=0`
  - `END_INDEX=5`
  - `NUM_SIMULATIONS=60`
  - `MAX_DEPTH=6`
  - `MAX_BRANCHING=12`
  - `ITEM_SIZE=10`

- **`pilot`**
  - `START_INDEX=0`
  - `END_INDEX=20`
  - `NUM_SIMULATIONS=400`
  - `MAX_DEPTH=10`
  - `MAX_BRANCHING=20`
  - `ITEM_SIZE=20`

- **`official`**
  - `START_INDEX=0`
  - `END_INDEX=50`
  - `NUM_SIMULATIONS=800`
  - `MAX_DEPTH=10`
  - `MAX_BRANCHING=20`
  - `ITEM_SIZE=20`

### 4.2 这意味着什么

`pilot` 在当前项目里不是“50 个起始分子的小正式实验”，而是：

- **20 个起始分子**；
- **1 个 seed**；
- **搜索预算明显高于 smoke**，但仍低于 official；
- **评估口径已经与正式实验对齐**，都使用 `item_size=20`。

换句话说，`pilot` 的本质是：

> **在接近正式设置的搜索参数下，用中等规模样本验证结论是否稳定。**

---

## 5. 本轮 pilot 的范围与边界

### 5.1 本轮只做什么

本轮 `pilot` 只做 **MCTS 侧消融**，矩阵固定为：

- **tasks**：`lumo_up`、`homo_down`
- **variants**：`full`、`wo_prior`、`wo_leaf_value`、`random_topb`
- **seeds**：`42`
- **profile**：`pilot`

### 5.2 本轮暂时不做什么

以下内容暂不进入本轮 `pilot`：

- `OFO` 侧变体：`abs_target`、`wo_opfeat`、`gcn2d`、`single_branch`
- 附录级变体：`full_expand`、`wo_pruning`、`wo_logp`
- 多 seed 重复：`43`、`44`
- 标准集 / 完整集任务扩展：`homo_up`、`lumo_down`

### 5.3 为什么这样收敛范围

这是当前**性价比最高**的选择，因为：

- `smoke` 已经把最核心的 4 个 `MCTS` 变体跑过一遍；
- 这些变体最接近论文主文的核心论点；
- 继续扩大任务或 seed 前，先把这 4 个变体的趋势坐实更划算。

---

## 6. Pilot 实验矩阵

### 6.1 运行单元定义

统一采用：

- **1 个 run unit = 1 个 task × 1 个 variant × 1 个 seed**

因此本轮共有：

- `2` 个 task
- `4` 个 variant
- `1` 个 seed
- 合计 **8 个 run unit**

### 6.2 具体矩阵

| task | variant | seed | 目标 |
|------|---------|------|------|
| `lumo_up` | `full` | `42` | 基线 |
| `lumo_up` | `wo_prior` | `42` | 验证 prior 是否稳定贡献 |
| `lumo_up` | `wo_leaf_value` | `42` | 验证 leaf value 退化是否复现 |
| `lumo_up` | `random_topb` | `42` | 验证随机 TopB 的波动是否收敛 |
| `homo_down` | `full` | `42` | 基线 |
| `homo_down` | `wo_prior` | `42` | 验证 prior 是否稳定贡献 |
| `homo_down` | `wo_leaf_value` | `42` | 验证 leaf value 退化是否复现 |
| `homo_down` | `random_topb` | `42` | 验证随机 TopB 的波动是否收敛 |

---

## 7. 变体与代码参数映射

当前 `run_mcts_ablations.sh` 已经把核心 `MCTS` 侧消融映射成显式参数，不需要再手工 patch：

| variant | `mcts_prior_mode` | `mcts_value_mode` | `mcts_expansion_mode` | 其他 |
|--------|--------------------|-------------------|------------------------|------|
| `full` | `softmax` | `accumulated` | `topk` | 基线 |
| `wo_prior` | `uniform` | `accumulated` | `topk` | 其余不变 |
| `wo_leaf_value` | `softmax` | `zero` | `topk` | 其余不变 |
| `random_topb` | `softmax` | `accumulated` | `random_topk` | `mcts_random_seed=seed` |

这意味着本轮 `pilot` 的重点在于**采集稳定结果**，而不是再做额外代码改造。

---

## 8. 推荐执行流程

### 8.1 原则

推荐继续沿用“**搜索与真值评估分离**”的两阶段流程：

- **搜索阶段**：用 `mol-ofo` 环境跑 `run_mcts_ablations.sh`
- **真值评估阶段**：用 `mol-tdc` 环境跑 `run_eval_mcts_ablations.sh`

这样更稳妥，原因是：

- 搜索依赖主项目模型与推理代码；
- 真值评估依赖 `tdc / ase / python_tsp` 等评估侧依赖；
- 两边环境职责更清晰，出问题时也更容易定位。

### 8.2 搜索阶段命令

建议输出目录固定到一个独立 run root，例如：

```bash
REPO=/home/ubuntu/mol_opt/mol-ofo
OUT_ROOT=$REPO/mol_evo/output/paper/ablations/$(date +%Y%m%d_%H%M%S)_pilot_mcts
```

启动命令：

```bash
cd "$REPO"
CONDA_ENV=mol-ofo \
PROFILE=pilot \
TASKS=lumo_up,homo_down \
VARIANTS=full,wo_prior,wo_leaf_value,random_topb \
SEEDS=42 \
RUN_EVAL=0 \
RUN_SUMMARY=0 \
OUTPUT_ROOT="$OUT_ROOT" \
bash mol_evo/scripts/run_mcts_ablations.sh
```

### 8.3 搜索阶段 tmux 模板

```bash
SESSION=mcts-pilot-search-$(date +%m%d-%H%M)
REPO=/home/ubuntu/mol_opt/mol-ofo
OUT_ROOT=$REPO/mol_evo/output/paper/ablations/$(date +%Y%m%d_%H%M%S)_pilot_mcts

tmux new-session -d -s "$SESSION" \
  "cd $REPO && CONDA_ENV=mol-ofo PROFILE=pilot TASKS=lumo_up,homo_down VARIANTS=full,wo_prior,wo_leaf_value,random_topb SEEDS=42 RUN_EVAL=0 RUN_SUMMARY=0 OUTPUT_ROOT=$OUT_ROOT bash mol_evo/scripts/run_mcts_ablations.sh 2>&1 | tee $OUT_ROOT/tmux_search.log"
```

查看进度：

```bash
tmux attach -t "$SESSION"
```

### 8.4 真值评估阶段命令

搜索结束后，直接对该 `run_root` 执行：

```bash
cd "$REPO"
CONDA_ENV=mol-tdc \
RUN_ROOTS="$OUT_ROOT" \
RUN_CSV_EVAL=1 \
RUN_SUMMARY=1 \
FORCE=0 \
bash mol_evo/scripts/run_eval_mcts_ablations.sh
```

### 8.5 真值评估阶段 tmux 模板

```bash
EVAL_SESSION=mcts-pilot-eval-$(date +%m%d-%H%M)
REPO=/home/ubuntu/mol_opt/mol-ofo
OUT_ROOT=/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/<替换成搜索输出目录>

tmux new-session -d -s "$EVAL_SESSION" \
  "cd $REPO && CONDA_ENV=mol-tdc RUN_ROOTS=$OUT_ROOT RUN_CSV_EVAL=1 RUN_SUMMARY=1 FORCE=0 bash mol_evo/scripts/run_eval_mcts_ablations.sh 2>&1 | tee $OUT_ROOT/tmux_eval.log"
```

---

## 9. 预期产物

本轮 `pilot` 完成后，每个 `run_root` 下至少应出现以下关键文件：

### 9.1 搜索侧

- `run_manifest.txt`
- `run_summary.tsv`
- `run_commands.tsv`
- `task/variant/seed42/search/batch_results.json`
- `task/variant/seed42/search.log`
- `task/variant/seed42/manifest.json`

### 9.2 评估侧

- `eval_summary.tsv`
- `true_eval_summary.tsv`
- `true_eval_summary_aggregated.tsv`
- `task/variant/seed42/evaluate_batch.log`
- `task/variant/seed42/evaluate_csv.log`
- `best_results_*.csv`
- `statistics_summary_*.json`

### 9.3 用于论文判断的核心字段

以后续判断为主，优先看：

- `paper_avg_improvement`
- `paper_success_rate`
- `morgan_similarity_best_mean`
- `intdiv_best`
- `avg_runtime`
- `avg_expanded_nodes`

其中：

- `paper_avg_improvement` 与 `paper_success_rate` 来自 **best-of-topK** 论文口径；
- `avg_runtime` 与 `avg_expanded_nodes` 用来判断“效果提升是不是靠明显更高代价换来的”。

---

## 10. Pilot 的验收标准

### 10.1 运行层验收

以下条件全部满足，才视为 `pilot` 运行成功：

- `8/8` 个 run unit 均为 `success`；
- `eval_summary.tsv` 中所有条目状态正常；
- `true_eval_summary.tsv` 与 `true_eval_summary_aggregated.tsv` 成功生成；
- 不存在缺失 `batch_results.json / best_results_*.csv / statistics_summary_*.json` 的 run。

### 10.2 结论层验收

`pilot` 的目标不是“所有变体都拉开大差距”，而是至少回答下面三个问题中的两个：

1. **`wo_leaf_value` 是否在两个任务上都稳定弱于 `full`**？
2. **`wo_prior` 是否在更大样本上仍接近 `full`，还是开始出现稳定差异**？
3. **`random_topb` 的提升/退化是否在两个任务上方向一致，还是继续呈现随机波动**？

如果这三个问题仍全部无法回答，就说明当前 `pilot` 规模不够，或者需要引入第二个 seed。

---

## 11. 进入 official 前的决策规则

建议按下面规则决定 `official` 矩阵：

### 11.1 必进 official

满足任一条件的变体，建议保留到 `official`：

- 在两个任务上都表现出**稳定退化或稳定提升**；
- 在一个任务上效果接近，但在效率指标上显著更差；
- 能直接支撑论文核心论点。

按当前 `smoke` 信号，**`wo_leaf_value` 基本属于必进 official 候选**。

### 11.2 待观察

- `wo_prior`
- `random_topb`

这两项的去留应由 `pilot` 决定：

- 若趋势稳定，则保留进入 `official`；
- 若仍几乎重合或高波动，则可以考虑只保留一项，避免主文表过长。

### 11.3 暂不进入本轮 official

以下项建议继续放在后续或附录级：

- `full_expand`
- `wo_pruning`
- `wo_logp`
- `OFO` 侧全部新 checkpoint 变体

---

## 12. 风险与应对

### 12.1 风险一：文档口径与代码口径不一致

当前已有过一次偏差：旧文档把 `pilot` 写成了 `50` 个起始分子，但脚本实际是 `20`。

**应对**：

- 本轮执行一律以 `run_manifest.txt` 为最终记录；
- 写论文时引用数值前，先核对 `PROFILE / START_INDEX / END_INDEX / NUM_SIMULATIONS`。

### 12.2 风险二：评估口径被 `improvement` 符号误导

当前评估链里，底层 `improvement` 字段的符号与论文口径相反。

**应对**：

- 主表与内部汇报统一使用 `true_eval_summary.tsv` 与 `true_eval_summary_aggregated.tsv`；
- 不再手工直接看原始 `best_results_*.csv` 里的 `improvement` 号数做判断。

### 12.3 风险三：`random_topb` 结果受随机性影响过大

**应对**：

- 先看 `pilot` 单 seed 是否比 `smoke` 更稳定；
- 若仍不稳定，则优先在 `official` 阶段给它补 `seed=43/44`，而不是提前扩大更多任务。

---

## 13. 已完成执行记录（2026-04-14）

### 13.1 执行状态

- [x] 确认 `mol-ofo` 环境可运行搜索
- [x] 确认 `mol-tdc` 环境可运行评估
- [x] 确认 `run_mcts_ablations.sh` 与 `run_eval_mcts_ablations.sh` 为当前最新版
- [x] 确认输出目录未与旧 run 混淆
- [x] 启动 `pilot` 搜索 tmux 并完成 `8/8` 个 run unit
- [x] 启动 `pilot` 真值评估 tmux 并生成 `eval_summary.tsv`
- [x] 生成 `true_eval_summary.tsv` 与 `true_eval_summary_aggregated.tsv`
- [x] 检查关键产物路径，未发现缺失 `best_results_*.csv / statistics_summary_*.json` 的 run
- [x] 完成 `full / wo_prior / wo_leaf_value / random_topb` 的结果对比
- [x] 给出 `official` 保留变体名单

### 13.2 本轮 pilot 的产物与验收结果

- **run root**：`mol_evo/output/paper/ablations/20260413_1536_pilot_mcts`
- **搜索汇总**：`run_summary.tsv`
- **评估汇总**：`eval_summary.tsv`
- **论文口径汇总**：`true_eval_summary.tsv`、`true_eval_summary_aggregated.tsv`
- **运行层验收**：通过。`eval_summary.tsv` 中 `8/8` 个 run unit 均为 `success`。

补充说明：`lumo_up / wo_leaf_value / seed42` 在 `true_eval_summary.tsv` 中记录为 `start_count=19`。当前搜索与评估产物均已完整生成，说明并非 run 失败，而是有 `1` 个起始分子未进入最终 `best_results` 汇总；这不影响本轮对整体趋势的判断，但在写正式主文时应避免把该项单独解释为“效果提升”。

### 13.3 本轮 pilot 的实际结论

1. **`wo_leaf_value`**：已经可以判定为明显更差的消融。
   - `homo_down` 上，`paper_avg_improvement` 从 `2.004136` 降到 `0.489601`；
   - 同时 `avg_runtime` 从 `2.311089` 飙升到 `204.348653`，`avg_expanded_nodes` 从 `7.6` 升到 `188.25`；
   - `lumo_up` 上虽然 improvement 略高于 `full`，但成本从 `11.490089 / 12.45` 飙到 `240.144709 / 207.95`，因此从效果-成本权衡看仍显著劣于 `full`。

2. **`wo_prior`**：在两个任务上都与 `full` 接近。
   - `homo_down` 上较 `full` 略低（`2.004136 → 1.900705`），但 runtime 更低；
   - `lumo_up` 上也仅略低（`1.145172 → 1.095274`），且 runtime / expanded nodes 更省；
   - 这说明 prior 的贡献目前弱于 `leaf value`，但仍值得保留到 `official` 进一步确认。

3. **`random_topb`**：当前没有表现出值得进入主文 `official` 的稳定优势。
   - 两个任务上都没有超过 `full`；
   - 同时运行时间与展开节点明显更高；
   - 因此更适合作为已完成的 `pilot` 证据保留，而不是继续占用主表级算力预算。

4. **进入 official 的建议名单**：
   - **保留**：`full`、`wo_prior`、`wo_leaf_value`
   - **暂不进入本轮 official 主矩阵**：`random_topb`

---

## 14. 下一步执行决策

基于本轮 `pilot`，下一步直接进入 **`official` 级 MCTS 侧消融**：

- **tasks**：`lumo_up`、`homo_down`
- **variants**：`full`、`wo_prior`、`wo_leaf_value`
- **seeds**：`42`、`43`、`44`
- **profile**：`official`

这样做的原因是：

- `wo_leaf_value` 已经足够证明“去掉 leaf value 会显著破坏效果与效率”，必须补齐多 seed 形成主文表；
- `wo_prior` 虽然差异较小，但它直接对应“prior 是否必要”这一核心论点，值得用 `official` 级多 seed 做最终确认；
- `random_topb` 在 `pilot` 中未显示出正向价值，继续投入正式算力的优先级较低。

---

## 15. 一句话总结

> **`pilot` 已经完成了它的任务：我们已经能确认 `leaf value` 是关键组件，并据此把下一步 `official` 矩阵收敛到 `full / wo_prior / wo_leaf_value`。**
