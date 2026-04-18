# A* RL Demo — 分阶段实验得分台账
<!-- last-updated: 2026-04-11 -->

本文档专门记录 **不同阶段 / 不同 run 的实验得分数据**，用于把“结论描述”和“原始分数台账”分开。
后续新增实验时，优先把核心分数先补到这里，再回填 `experiments.md` 中的结论段。

---

## 记录规则

- **按阶段记录**：优先对应 `Step 4 / Step 5 / Step 6 / 方案 A A0~A3`。
- **优先写主指标**：至少记录 `top1_mean`、`top1_median`、`nonempty_topk`、必要时补 `trimmed_mean / win_rate / delta_median / trimmed_delta_mean`。
- **保留 run 来源**：每条记录都带 run 目录、JSON 或 TSV 来源，便于回查。
- **结论与分数分离**：这里以分数表为主，不展开长篇分析。

---

## 0. 2026-04-11 状态刷新（进行中）

### 0.1 修复与当前口径总表

| 项目 | 当前结论 | 影响 |
|------|----------|------|
| 离线导出修复 | `export_rl_demo_transitions.py` 已修复把 `property_change` 错读成 `predicted_change` 的问题 | 早期 BFS / A1 BC transition 中大量 `property_change=0.0` 的问题已定位并修复 |
| 影响范围 | `BFS15`、`A1-small`、`A1-main` 的旧版 BC 数据都受影响 | 这些 run 的旧分数仅保留为历史归档 |
| 当前正式口径 | 最新判断统一以 `fixed` 重导 / 重训 / 重评估结果为准 | `A0 fixed` 作为当前默认参考线 |
| fixed eval JSON 说明 | fixed 任务原始 eval JSON 顶层是按分子索引的结果字典，不是 `summary` | 正式台账请以 `*_summary.json` 或手工汇总后的 summary 为准 |
| 当前主推进方向 | 继续做更厚的 `MCTS -> BC -> RL` 数据链 | 因为修复后 BFS 扩规模优势不再明显，MCTS 更值得继续放大 |

### 0.2 当前主方法效果总表

| 方法类别 | run_name / 来源 | 搜索基石 / 数据源 | 评估集 | 样本规模 | top1_mean | top1_median | trimmed_mean | 当前判断 |
|----------|------------------|-------------------|--------|----------|-----------|-------------|--------------|----------|
| Step 4 BC-only | `lumo_bc_bfs15_depth3/bc_20260409_145233` | `15` 棵 BFS 树导出的 BC 冷启动 | `5` 分子 | `9090` transitions | `2.0896` | `—` | `—` | 早期小样本基线 |
| Step 5 RL-after-train | `lumo_rl_bfs15_depth3_run100/rl_20260409_161937` | `BFS15 -> BC -> RL(100 ep)` | `5` 分子 | `100` episodes | `1.0347` | `—` | `—` | 仅作早期参考 |
| Step 6 最优 BC | `lumo_eval_grid50_parallel_20260409_194105/summary.tsv` | `bc + budget=200 + prefilter=50` | `50` 分子 | 网格复核 | `3.3620` | `3.2222` | `—` | 历史最优 BC 参考线 |
| Step 6 最优 RL 均值 | `lumo_eval_grid50_parallel_20260409_194105/summary.tsv` | `rl + budget=50 + prefilter=50` | `50` 分子 | 网格复核 | `7.4856` | `3.0376` | `—` | 高均值，但稳健性不足 |
| Plan A A0 fixed | `a0_fixed_bc_holdout50_budget200_pref50` | fixed `bc + budget=200 + prefilter=50` | holdout `50` 分子 | `A0` 基线 | `3.0514` | `3.1228` | `2.9509` | 当前 fixed 正式参考线 |
| Plan A A1-small fixed | `a1_small_fixed_bc_holdout50_budget200_pref50_vs_a0_fixed` | `50` 棵 BFS 树，fixed 重导 | holdout `50` 分子 | `27351` transitions | `3.0438` | `2.9572` | `3.0420` | 总体接近 A0，稳健指标略强 |
| Plan A A1-main fixed | `a1_main_fixed_bc_holdout50_budget200_pref50_vs_a0_fixed` | `100` 棵 BFS 树，fixed 重导 | holdout `50` 分子 | `53269` transitions | `2.9425` | `2.9894` | `2.9669` | 不再重现旧口径里的明显优势 |
| Plan A A1-mcts-main | `a1_mcts_main_bc_holdout50_budget200_pref50_vs_a0_fixed` | `MCTS` 小规模树导出 | holdout `50` 分子 | `1089` transitions | `3.0971` | `3.0167` | `2.9934` | 小样本下已接近 `a0_fixed` 持平 |

### 0.3 当前配对稳健性总表

| 对比 | 评估口径 | win_rate | delta_median | trimmed_delta_mean | 备注 |
|------|----------|----------|--------------|--------------------|------|
| `rl 50/50` vs `bc 50/50` | Step 6 网格复核 | `12 / 50` 分子更优 | `-0.3116` | `-0.2753` | RL 高均值主要由少量 outlier 拉动 |
| `a1_small_fixed` vs `a0_fixed` | Plan A fixed holdout | `54%` | `+0.0100` | `+0.1221` | 中位数差很小，但 trimmed 指标略优 |
| `a1_main_fixed` vs `a0_fixed` | Plan A fixed holdout | `54%` | `≈0` | `+0.0457` | 比 A0 略好，但优势不大 |
| `a1_mcts_main` vs `a0_fixed` | Plan A fixed holdout | `50%` | `-0.0190` | `+0.0538` | 在仅 `1089` 条样本下已接近持平 |

### 0.4 当前任务进度表

| run_name | 状态 | 进度 / 规模 | 当前主指标 | 备注 |
|----------|------|-------------|------------|------|
| `a0_fixed_bc_holdout50_budget200_pref50` | 已完成 | holdout `50` 分子 | `top1_mean=3.0514`，`top1_median=3.1228`，`trimmed_mean=2.9509` | 当前 fixed 参考线 |
| `a1_small_fixed_bc_holdout50_budget200_pref50_vs_a0_fixed` | 已完成 | `27351` 条 fixed transition | `top1_mean=3.0438`，`top1_median=2.9572`，`trimmed_mean=3.0420`，`win_rate=54%` | 稳健指标略优于 A0 |
| `a1_main_fixed_bc_holdout50_budget200_pref50_vs_a0_fixed` | 已完成 | `53269` 条 fixed transition | `top1_mean=2.9425`，`top1_median=2.9894`，`trimmed_mean=2.9669`，`win_rate=54%` | 与 A0 基本持平 |
| `a1_mcts_main_bc_holdout50_budget200_pref50_vs_a0_fixed` | 已完成 | `1089` 条 transition（`1089 / 1089` 非零） | `top1_mean=3.0971`，`top1_median=3.0167`，`trimmed_mean=2.9934`，`win_rate=50%` | 小样本 MCTS 已能对齐 fixed A0 |
| `plan_a_mcts_expand_rl_seed11` | 进行中 | `tmux:mcts-expand-rl`，约 `376 / 983`（`38.3%`） | `TBD` | 执行 `MCTS 扩规模 -> BC -> RL(300 ep)` 全流程 |

---

## 1. 跨阶段总览

### 1.1 按阶段主结果总表

| 阶段 | 日期 | run / 结果源 | 评估规模 | 主要设置 | 关键分数 | 当前备注 |
|------|------|--------------|----------|----------|----------|----------|
| Step 4：BC-only baseline | 2026-04-09 | `lumo_bc_bfs15_depth3/bc_20260409_145233` + `lumo_bc_eval_5.json` | `5` 分子 | BC 冷启动后 `astar_demo` | `top1_mean=2.0896` | 早期小样本基线 |
| Step 5：RL 训练后 `5` 分子评估 | 2026-04-09 | `lumo_rl_bfs15_depth3_run100/rl_20260409_161937` + `lumo_rl_eval_5_after_run100.json` | `5` 分子 | RL `100` episodes 后评估 | `top1_mean=1.0347` | 仅作早期参考 |
| Step 6：`50` 分子网格复核（最佳 BC） | 2026-04-09 | `lumo_eval_grid50_parallel_20260409_194105/summary.tsv` | `50` 分子 | `bc + budget=200 + prefilter=50` | `top1_mean=3.3620`，`top1_median=3.2222` | 历史最优 BC 参考线 |
| Step 6：`50` 分子网格复核（最佳 RL 均值） | 2026-04-09 | `lumo_eval_grid50_parallel_20260409_194105/summary.tsv` | `50` 分子 | `rl + budget=50 + prefilter=50` | `top1_mean=7.4856`，`top1_median=3.0376` | 高均值，但稳健性不足 |
| Plan A：A0 fixed | 2026-04-11 | `a0_fixed_bc_holdout50_budget200_pref50` | holdout `50` 分子 | fixed `bc + budget=200 + prefilter=50` | `top1_mean=3.0514`，`top1_median=3.1228`，`trimmed_mean=2.9509` | 当前正式参考线 |
| Plan A：A1-small fixed | 2026-04-11 | `a1_small_fixed_bc_holdout50_budget200_pref50_vs_a0_fixed` | holdout `50` 分子 | `50` 棵 BFS 树，fixed 重导 | `top1_mean=3.0438`，`top1_median=2.9572`，`trimmed_mean=3.0420` | 接近 A0，trimmed 略优 |
| Plan A：A1-main fixed | 2026-04-11 | `a1_main_fixed_bc_holdout50_budget200_pref50_vs_a0_fixed` | holdout `50` 分子 | `100` 棵 BFS 树，fixed 重导 | `top1_mean=2.9425`，`top1_median=2.9894`，`trimmed_mean=2.9669` | 与 A0 基本持平 |
| Plan A：A1-mcts-main | 2026-04-11 | `a1_mcts_main_bc_holdout50_budget200_pref50_vs_a0_fixed` | holdout `50` 分子 | 小规模 `MCTS -> BC` | `top1_mean=3.0971`，`top1_median=3.0167`，`trimmed_mean=2.9934` | 小样本下已接近 A0 |

### 1.2 按方法最佳结果总表

| 方法 | 当前最佳 run / 配置 | 评估集 | top1_mean | top1_median | trimmed_mean | 稳健性备注 |
|------|---------------------|--------|-----------|-------------|--------------|------------|
| 早期 BC-only | `lumo_bc_bfs15_depth3/bc_20260409_145233` | `5` 分子 | `2.0896` | `—` | `—` | 仅作流程检查 |
| 早期 RL-after-train | `lumo_rl_bfs15_depth3_run100/rl_20260409_161937` | `5` 分子 | `1.0347` | `—` | `—` | 小样本下弱于 BC |
| 历史最优 BC | `bc + budget=200 + prefilter=50` | `50` 分子 | `3.3620` | `3.2222` | `—` | 历史基线 |
| 历史最优 RL 均值 | `rl + budget=50 + prefilter=50` | `50` 分子 | `7.4856` | `3.0376` | `-0.2753`（配对 trimmed） | 高均值但不稳 |
| 当前 fixed 基线 | `a0_fixed_bc_holdout50_budget200_pref50` | holdout `50` 分子 | `3.0514` | `3.1228` | `2.9509` | 当前正式对照组 |
| 当前 MCTS 候选 | `a1_mcts_main_bc_holdout50_budget200_pref50_vs_a0_fixed` | holdout `50` 分子 | `3.0971` | `3.0167` | `2.9934` | 样本还薄，但已接近持平 |

---

## 2. Step 4 — BC-only baseline

### 2.1 BC 数据与评估总表

| 项目 | 数值 |
|------|------|
| BFS 树数 | `15` |
| 导出样本数 | `9090` |
| 最大深度 | `3` |
| 最佳 BC run | `lumo_bc_bfs15_depth3/bc_20260409_145233` |
| 评估分子数 | `5` |
| 非空 `topK` 分子数 | `5 / 5` |
| `top1_mean` | **`2.0896`** |
| 单分子 top1 改善 | `0.9892`, `2.4505`, `2.3665`, `2.4696`, `2.1723` |

### 2.2 数据来源表

| 类型 | 路径 / 来源 |
|------|-------------|
| 结果说明 | `experiments.md` 中 `Step 4.4` |
| BC 最优权重目录 | `mol_evo/output/astar_rl/lumo_bc_bfs15_depth3/bc_20260409_145233` |

---

## 3. Step 5 — 在线 RL 训练与训练后评估

### 3.1 RL 训练统计（100 episodes）

| 指标 | 数值 |
|------|------|
| run 目录 | `lumo_rl_bfs15_depth3_run100/rl_20260409_161937` |
| episodes | `100 / 100` |
| `return_mean` | **`13.6251`** |
| `return_max` | **`170.9716`** |
| `return_min` | `-10.4454` |
| `steps_mean` | **`26.82`** |
| `steps_max` | `50` |
| `steps_nonzero` | **`100 / 100`** |

### 3.2 RL 训练后 `5` 分子评估

| 指标 | 数值 |
|------|------|
| 非空 `topK` 分子数 | `3 / 5` |
| `top1_mean` | **`1.0347`** |
| 单分子 top1 改善 | `0.0000`, `0.0000`, `1.1830`, `1.1719`, `2.8188` |

### 3.3 与 Step 4 对照表

| 对比项 | BC-only | RL-after-train |
|--------|---------|----------------|
| 样本数 | `5` | `5` |
| 非空 `topK` | `5 / 5` | `3 / 5` |
| `top1_mean` | `2.0896` | `1.0347` |

> 注：这一阶段分数只保留为 **早期小样本检查**，不再作为是否继续推进 RL 的主依据。

---

## 4. Step 6 — 50 分子网格复核

### 4.1 全组合得分表

| method | budget | prefilter | nonempty topK | top1 mean | top1 median | actual expansions mean | ofo calls mean |
|--------|--------|-----------|---------------|-----------|-------------|------------------------|----------------|
| bc | 50 | 20 | `35 / 50` | `1.6826` | `1.5217` | `36.76` | `44.26` |
| rl | 50 | 20 | `50 / 50` | `2.2473` | `2.3492` | `256.58` | `367.22` |
| bc | 50 | 50 | `50 / 50` | `3.3464` | `3.1834` | `341.02` | `583.60` |
| rl | 50 | 50 | `50 / 50` | **`7.4856`** | `3.0376` | `341.86` | `567.58` |
| bc | 100 | 20 | `50 / 50` | `1.6633` | `1.5657` | `367.42` | `564.84` |
| rl | 100 | 20 | `45 / 50` | `2.2687` | `1.3959` | `208.16` | `333.44` |
| bc | 100 | 50 | `50 / 50` | `2.8019` | `2.7170` | `479.60` | `1093.78` |
| rl | 100 | 50 | `50 / 50` | `2.9246` | `2.9101` | `499.04` | `1326.82` |
| bc | 200 | 20 | `47 / 50` | `2.0748` | `2.1244` | `167.16` | `226.06` |
| rl | 200 | 20 | `45 / 50` | `1.8982` | `1.7691` | `114.48` | `163.86` |
| bc | 200 | 50 | `50 / 50` | `3.3620` | **`3.2222`** | `486.26` | `1026.88` |
| rl | 200 | 50 | `50 / 50` | `3.1404` | `2.7801` | `401.12` | `905.58` |

### 4.2 当前阶段最佳配置总表

| 类别 | 配置 | 分数 | 说明 |
|------|------|------|------|
| 最优 BC 均值 | `bc + budget=200 + prefilter=50` | `top1_mean=3.3620` | 历史最优 BC 均值 |
| 最优 BC 中位数 | `bc + budget=200 + prefilter=50` | `top1_median=3.2222` | 历史最优 BC 中位数 |
| 最优 RL 均值 | `rl + budget=50 + prefilter=50` | `top1_mean=7.4856` | 均值最高，但不稳健 |
| 最优 RL 中位数 | `rl + budget=50 + prefilter=50` | `top1_median=3.0376` | 仍低于最优 BC 中位数 |
| 默认 prefilter 候选 | `prefilter=50` | BC / RL 均优于 `20` | 当前稳定结论 |

### 4.3 稳健性补充指标表

| 对比对象 | 指标 | 数值 |
|----------|------|------|
| `rl 50/50` vs `bc 50/50` | RL 更优分子数 | `12 / 50` |
| `rl 50/50` vs `bc 50/50` | RL 更差分子数 | `38 / 50` |
| `rl 50/50` vs `bc 50/50` | `delta_median` | `-0.3116` |
| `rl 50/50` vs `bc 50/50` | `trimmed_mean` | `-0.2753` |
| `rl 50/50` | 超大 improvement 样本 | `3` 个（其中 `2` 个 `>50`） |

### 4.4 数据来源表

| 类型 | 路径 |
|------|------|
| 汇总表 | `mol_evo/output/astar_rl/lumo_eval_grid50_parallel_20260409_194105/summary.tsv` |
| 分子级 JSON | 同目录下各 `bc_budget*_pref*.json` / `rl_budget*_pref*.json` |

---

## 5. 方案 A 台账

### 5.1 A0 — fixed 基线总表

| run_name | holdout 分子数 | BC baseline 配置 | top1_mean | top1_median | trimmed_mean | 备注 |
|----------|----------------|------------------|-----------|-------------|--------------|------|
| `a0_fixed_bc_holdout50_budget200_pref50` | `50` | `bc + budget=200 + prefilter=50` | `3.0514` | `3.1228` | `2.9509` | 当前 fixed A0 参考线 |

### 5.2 A1 / MCTS 当前正式结果总表（fixed 口径）

| run_name | 搜索基石 | 树数 / 数据源 | 深度 | transitions | top1_mean | top1_median | trimmed_mean | vs `a0_fixed` win_rate | delta_median | trimmed_delta_mean | 当前判断 |
|----------|----------|---------------|------|-------------|-----------|-------------|--------------|------------------------|--------------|--------------------|----------|
| `a1_small_fixed_bc_holdout50_budget200_pref50_vs_a0_fixed` | BFS | `50` 棵 fixed BFS 树 | `3` | `27351` | `3.0438` | `2.9572` | `3.0420` | `54%` | `+0.0100` | `+0.1221` | 均值略低于 A0，但稳健指标略优 |
| `a1_main_fixed_bc_holdout50_budget200_pref50_vs_a0_fixed` | BFS | `100` 棵 fixed BFS 树 | `3` | `53269` | `2.9425` | `2.9894` | `2.9669` | `54%` | `≈0` | `+0.0457` | 不再重现旧口径里的明显优势 |
| `a1_mcts_main_bc_holdout50_budget200_pref50_vs_a0_fixed` | MCTS | 小规模 MCTS 树 | `—` | `1089` | `3.0971` | `3.0167` | `2.9934` | `50%` | `-0.0190` | `+0.0538` | 数据偏少，但已接近 `a0_fixed` 持平 |

### 5.3 修复前归档总表（仅保留历史，不作当前主判断）

| run_name | 搜索基石 | 树数 | 深度 | transitions | top1_mean | top1_median | trimmed_mean | vs 旧 A0 BC200/50 win_rate | trimmed_delta_mean | 备注 |
|----------|----------|------|------|-------------|-----------|-------------|--------------|----------------------------|--------------------|------|
| `a1_small_bc_holdout50_budget200_pref50_vs_a0_bc200_pref50` | BFS | `50` | `3` | `27351` | `3.2498` | `3.2552` | `3.1784` | `44%` | `-0.1194` | `2026-04-10 11:43` 完成 |
| `a1_main_bc_holdout50_budget200_pref50_vs_a0_bc200_pref50` | BFS | `100` | `3` | `53269` | `4.2094` | `3.3734` | `3.3053` | `48%` | `-0.0250` | `2026-04-11 01:34` 完成 |

### 5.4 方案 A 产物与数据切分表

| 类别 | 路径 / 内容 | 备注 |
|------|-------------|------|
| holdout CSV | `mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_holdout50_start0.csv` | 当前固定 holdout |
| train pool CSV | `mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_train_pool_excluding_holdout.csv` | 训练池 |
| split metadata | `mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_split_metadata.json` | 切分说明 |
| A0 summary 目录 | `mol_evo/output/astar_rl/plan_a_a0` | 含 A0 各 summary |
| A0 summary 文件 | `a0_bc_best_budget200_pref50_summary.json/tsv` | 当前最佳 BC |
| A0 summary 文件 | `a0_bc_match_budget50_pref50_summary.json/tsv` | 与 RL 对齐配置 |
| A0 summary 文件 | `a0_rl_best_mean_budget50_pref50_vs_bc50_pref50_summary.json/tsv` | 历史 RL 最佳均值对照 |
| 当前切分规模 | 源 CSV `1033` 个分子；holdout `50`；训练池 `983` | 当前固定口径 |

### 5.5 A2 / A3 预留表

| 阶段 | run_name | seed | episodes | best_ckpt | holdout top1_mean | holdout top1_median | trimmed_mean | win_rate_vs_bc | 备注 |
|------|----------|------|----------|-----------|-------------------|---------------------|--------------|----------------|------|
| A2 — RL 第一轮放大 | `TBD` | `TBD` | `300` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | 待补 |
| A3 — RL 第二轮放大 | `TBD` | `TBD` | `1000` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | 待补 |

---

## 6. 更新建议

| 顺序 | 建议动作 | 目的 |
|------|----------|------|
| 1 | 先把 `summary.tsv` / `json` 中的核心指标补到本文件 | 保持台账先完整 |
| 2 | 再在 `experiments.md` 中写阶段性结论 | 让结论和原始分数分离 |
| 3 | 如果影响后续执行节奏，再同步 `plan_scaleup_a.md` 的默认设置或门槛 | 保持执行计划与分数口径一致 |
