# A* RL Demo — 实验推进手册
<!-- last-updated: 2026-04-11 -->

本文档面向实验推进阶段，给出**从零到有效 RL 结果**的完整实验路线、
对比基准设计、消融方向和预期结果解读。

---

## 实验路线总览

```
Step 0  环境 smoke check
Step 1  BFS baseline 建立（收集树数据 + 记录指标）
Step 2  MCTS baseline 建立
Step 3  astar_demo (random init) — 验证流程
Step 4  BC 冷启动 + astar_demo (BC only)
Step 5  BC + 在线 RL — 完整 astar_demo
Step 6  消融实验 / 超参扫描
Step 7  论文级结果整理
```

---

## 当前进度刷新（2026-04-11 18:16）

### 本轮新增事实

- **离线导出 bug 已修复**：`export_rl_demo_transitions.py` 之前把树里的 `property_change` 错读成 `predicted_change`，导致早期 BFS / A1 BC transition 中的 `property_change` 大量变成 `0.0`；目前已修复并兼容旧字段。
- **动作编码桥已补齐**：导出侧已统一 `operation / details`，下游 `encode_action()` 也已兼容 `dict / str / None` 与 `atom_idx / atom2_idx / bond_idx` 等参数来源。
- **fixed BFS 三条重跑已全部完成**：
  - `a0_fixed`：`top1_mean=3.0514`，`top1_median=3.1228`，`trimmed_mean=2.9509`
  - `a1_small_fixed`：`top1_mean=3.0438`，`top1_median=2.9572`，`trimmed_mean=3.0420`
  - `a1_main_fixed`：`top1_mean=2.9425`，`top1_median=2.9894`，`trimmed_mean=2.9669`
- **fixed A0 口径下的新比较已经生成**：
  - `a1_small_fixed vs a0_fixed`：`win_rate=54%`，`delta_median=+0.0100`，`trimmed_delta_mean=+0.1221`
  - `a1_main_fixed vs a0_fixed`：`win_rate=54%`，`delta_median≈0`，`trimmed_delta_mean=+0.0457`
  - `a1_mcts_main vs a0_fixed`：`win_rate=50%`，`delta_median=-0.0190`，`trimmed_delta_mean=+0.0538`
- **MCTS 小规模基石链已完成且不再明显弱于 fixed BFS**：虽然 `A1-mcts-main` 仍只有 `1089` 条 transition，但在 fixed 口径下已接近 `a0_fixed` 持平。

### 当前正在运行的任务

- **`mcts-expand-rl`**：仍在 `tmux` 中运行，当前搜索进度约 `376 / 983`（`38.3%`），执行 `MCTS 扩规模 -> BC -> holdout -> RL(300 episodes) -> holdout` 的完整新链路。

### 当前判断

- **修复后，BFS 扩规模结论变得更保守**：`A1-small fixed` 和 `A1-main fixed` 都没有形成对 `a0_fixed` 的压倒性优势，只能算“分布略有变化、稳健指标小幅改善”。
- **MCTS 的信号反而变得更值得继续追**：在仅 `1089` 条 transition 的情况下，`a1_mcts_main` 已经和 `a0_fixed` 大致持平，说明问题更像是“树还不够厚”，而不是“MCTS 本身不适合做 RL 基石”。
- **因此当前最合理的推进方向不再是继续堆 BFS 树**，而是把更厚的 `MCTS` 数据源跑完，再看 `MCTS-BC / MCTS-RL` 是否能拉开差距。

---

## 历史快照（2026-04-09）

### 已完成

- **Step 0 ~ Step 3**：已完成，`astar_demo` 的基础搜索链路、测试与 smoke check 可用。
- **Step 4（BC 冷启动）**：已完成。
  - BFS 树来源：`mol_evo/output/evo-mo/batch_optimization_20260409_125846`
  - 导出样本：`mol_evo/dataset/rl_demo/lumo_bfs15_depth3_bc_transitions_20260409.json`
  - 样本量：**9090** 条 transition（来自 15 棵 `lumo` BFS 树，`max-depth=3`）
  - BC 最优权重：`mol_evo/output/astar_rl/lumo_bc_bfs15_depth3/bc_20260409_145233`
- **Step 5（BC + 在线 RL）**：主训练与训练后评估均已跑通。
  - 在线 RL run：`mol_evo/output/astar_rl/lumo_rl_bfs15_depth3_run100/rl_20260409_161937`
  - 训练统计：`return_mean=13.6251`，`return_max=170.9716`，`steps_mean=26.82`
  - 轨迹收集状态：**100/100 episode 的 `episode_steps` 均 > 0**
- **Step 6（P0 + P1）**：`50` 分子网格评估已完成。
  - 结果目录：`mol_evo/output/astar_rl/lumo_eval_grid50_parallel_20260409_194105`
  - 覆盖组合：`BC / RL × open_set_budget(50/100/200) × top_n_prefilter(20/50)`

### 当前结论

- **工程链路已闭环**：`BFS 树 → BC 样本 → BC 训练 → BC-only 评估 → 在线 RL → RL 后评估 → 50 分子网格复核` 全部已跑通。
- **搜索参数结论已经明确**：`top_n_prefilter=50` 对 BC 和 RL 都明显优于 `20`，当前不建议继续把 `20` 当默认值。
- **RL 已经展现出超过 BC 的潜力**：最佳均值配置为 `rl + budget=50 + prefilter=50`，`top1_mean=7.4856`。
- **但当前还不能下“RL 已稳定领先”的结论**：同配置下分子级配对比较里，RL 仅 **12 / 50** 个分子优于 BC、**38 / 50** 个落后；`delta_median=-0.3116`，去掉头尾各 3 个样本后的 trimmed mean 也为 **-0.2753**。这说明 `7.4856` 的高均值主要由 **3 个超大 improvement outlier** 拉动。
- **阶段结论**：下一阶段重点应转向 **outlier 复核、checkpoint 选择策略、扩大 BC 冷启动数据**，而不是盲目继续加 episode。

---

## 实验结果归档（2026-04-10）

### 归档范围

- **BC 冷启动 run**：`mol_evo/output/astar_rl/lumo_bc_bfs15_depth3/bc_20260409_145233`
- **在线 RL run**：`mol_evo/output/astar_rl/lumo_rl_bfs15_depth3_run100/rl_20260409_161937`
- **50 分子网格复核**：`mol_evo/output/astar_rl/lumo_eval_grid50_parallel_20260409_194105`

### 冻结结论

- **5 分子后评估不再作为主判断**：`BC-only` 在 `5` 分子上 `top1_mean=2.0896`，`RL-after-train` 在同口径下为 `1.0347`；但这一定性已被 `50` 分子网格复核覆盖，不再单独作为 go / no-go 依据。
- **当前最稳的搜索参数结论**：`top_n_prefilter=50` 对 BC 和 RL 都明显优于 `20`，后续扩规模实验默认优先固定为 `50`。
- **当前最优 BC 参考线**：`bc + budget=200 + prefilter=50`，`top1_mean=3.3620`，`top1_median=3.2222`。
- **当前最强 RL 候选配置**：`rl + budget=50 + prefilter=50`，`top1_mean=7.4856`，但仅 **12 / 50** 个分子优于同配置 BC，`trimmed mean=-0.2753`，因此现阶段只能说 RL 出现了**高均值潜力**，还不能说已经**稳定领先**。
- **后续放大训练的前提**：任何更大规模 RL 训练，都应以 `50` 分子级 holdout 指标和稳健统计为主，不再只看 `episode_return`。

### 下一步文档入口

- **分阶段实验得分台账**：见 `stage_scores.md`
- **低风险扩规模计划**：见 `plan_scaleup_a.md`

---

## Step 0 — 环境 smoke check

验证核心依赖可用、模块可导入：

```bash
# 检查 torch / rdkit
python -c "import torch; from rdkit import Chem; print(torch.__version__)"

# 检查 astar_rl 包（不依赖 e3nn / torch_geometric）
python -c "
import sys; sys.path.insert(0, '.')
from mol_evo.core.models.astar_rl import PolicyNet, ValueNet, RLTrainer
from mol_evo.core.data.rl_demo_processing import STATE_DIM, ACTION_DIM
print('OK  STATE_DIM=%d  ACTION_DIM=%d' % (STATE_DIM, ACTION_DIM))
"

# 跑内置 smoke test（无需模型权重）
python mol_evo/tests/test_rl_training.py
python mol_evo/tests/test_astar_demo_search_mode.py
```

期望全部输出 `OK`，如有 skip 但无 ERROR 也可继续。

---

## Step 1 — BFS Baseline

### 运行

```bash
python -m mol_evo.scripts.batch_optimizer \
  --input-csv   mol_evo/dataset/eval-data/qm9_test_molecules.csv \
  --model-path  $OFO_MODEL \
  --model-dir   $OFO_MODEL_DIR \
  --config-file $OFO_CONFIG \
  --search-mode bfs \
  --direction   decrease \
  --max-depth   4 \
  --max-branching 8 \
  --topK        20 \
  --start-index 0 --end-index 100
```

### 记录指标（每次实验均需记录）

| 指标 | 来源字段 | 说明 |
|------|----------|------|
| `topK_best_improvement` | `topK_results.topK_results[0].property_change` | TopK 中最大属性改善 |
| `topK_mean_improvement` | TopK 列表均值 | TopK 整体改善水平 |
| `topK_diversity` | Tanimoto 1 - similarity | TopK 分子多样性 |
| `expanded_nodes` | `astar_stats.expanded_nodes` | 展开节点数（仅 astar_demo） |
| `ofo_calls` | `astar_stats.ofo_scored_candidates` | OFO 调用次数（仅 astar_demo） |
| `time_per_mol` | batch log `耗时` 字段 | 秒/分子 |

---

## Step 2 — MCTS Baseline

```bash
python -m mol_evo.scripts.batch_optimizer \
  ... \
  --search-mode      mcts \
  --num-simulations  200 \
  --exploration-weight 1.4
```

---

## Step 3 — astar_demo (Random Init)

验证 astar_demo 搜索流程正确，无 policy/value 引导时的随机搜索质量：

```bash
python -m mol_evo.scripts.batch_optimizer \
  ... \
  --search-mode    astar_demo \
  --rl-eval \
  --open-set-budget 200 \
  --top-n-prefilter 20
```

> 此时 policy/value 随机初始化，预期效果不如 BFS，只用于确认流程无误。

---

## Step 4 — BC 冷启动 + astar_demo

### 4.1 收集 BC 数据

建议用 BFS baseline 已跑出的树 JSON 目录直接复用（节省时间）。

```bash
python mol_evo/dataset/export_rl_demo_transitions.py \
  --input-dir  mol_evo/output/evo-mo/batch_optimization_<bfs_timestamp> \
  --output-json mol_evo/dataset/rl_demo/bc_transitions.json \
  --direction  decrease \
  --max-depth  4
```

检查导出数量：

```bash
python -c "
import json
with open('mol_evo/dataset/rl_demo/bc_transitions.json') as f:
    data = json.load(f)
print('BC samples:', len(data))
"
```

> 经验值：100 个分子 × depth 4 × branching 8 ≈ 2000~5000 条样本。
> 如果样本 < 500，建议先多跑几批 BFS 再导出。

### 4.2 BC 训练

```bash
python mol_evo/train_bc_pretrain.py \
  --data-json  mol_evo/dataset/rl_demo/bc_transitions.json \
  --output-dir mol_evo/output/astar_rl/bc \
  --direction  decrease \
  --epochs     100 \
  --batch-size 64
```

训练过程中关注：
- `policy_loss` 下降趋势（in-batch cross-entropy，理论下界 ≈ log(1)=0）
- `value_loss` 下降趋势（MSE，目标为 future_best_gain）
- 早停是否触发（`--patience 20`）

### 4.3 BC-only 评估

```bash
python -m mol_evo.scripts.batch_optimizer \
  ... \
  --search-mode astar_demo --rl-eval \
  --policy-path mol_evo/output/astar_rl/bc/policy_best.pth \
  --value-path  mol_evo/output/astar_rl/bc/value_best.pth
```

与 BFS / MCTS baseline 横向对比 topK_best_improvement 和 ofo_calls。

### 4.4 本轮结果（2026-04-09）

- 使用 `15` 棵 BFS 树（`max-depth=3`）导出得到 **9090** 条 BC 样本。
- `BCDataset.encode_action()` 已兼容字符串 `operation`，因此旧数据可以直接训练，无需重导。
- 本轮最佳 BC 权重目录：`mol_evo/output/astar_rl/lumo_bc_bfs15_depth3/bc_20260409_145233`
- 当前 5 分子 BC-only baseline：

| 指标 | 数值 |
|------|------|
| 非空 `topK` 分子数 | 5 / 5 |
| top1 改善均值 | **2.0896** |
| 单分子 top1 改善 | `0.9892`, `2.4505`, `2.3665`, `2.4696`, `2.1723` |

> 注：这里的“改善”统一按 `lumo decrease` 口径计算，即 `initial_property_value - top1.property_value`。

---

## Step 5 — BC + 在线 RL

```bash
python mol_evo/train_astar_rl_demo.py \
  --input-csv   mol_evo/dataset/eval-data/qm9_test_molecules.csv \
  --model-path  $OFO_MODEL \
  --model-dir   $OFO_MODEL_DIR \
  --config-file $OFO_CONFIG \
  --policy-path mol_evo/output/astar_rl/bc/policy_best.pth \
  --value-path  mol_evo/output/astar_rl/bc/value_best.pth \
  --output-dir  mol_evo/output/astar_rl/rl \
  --num-episodes 300 \
  --checkpoint-every 50
```

训练过程监控（每 10 episode 打一行日志）：

```
Episode   10/300 | return=0.1234 p_loss=0.012345 v_loss=0.003456 steps=12 t=2.3s
```

关键信号：
- `episode_return` 整体上升趋势 → 策略在改善
- `policy_loss` 数量级 0.01~0.1，过大说明熵正则不够；接近 0 说明 policy 坍缩
- `episode_steps` 接近 `max_depth × branching / budget`，过低说明搜索提前终止

训练完成后评估：

```bash
python -m mol_evo.scripts.batch_optimizer \
  ... \
  --search-mode astar_demo --rl-eval \
  --policy-path mol_evo/output/astar_rl/rl/rl_<timestamp>/policy_best.pth \
  --value-path  mol_evo/output/astar_rl/rl/rl_<timestamp>/value_best.pth
```

### 5.1 本轮结果（2026-04-09）

- 正式 run：`mol_evo/output/astar_rl/lumo_rl_bfs15_depth3_run100/rl_20260409_161937`
- 主训练已完成：**100 / 100 episodes**
- 训练统计：

| 指标 | 数值 |
|------|------|
| `return_mean` | **13.6251** |
| `return_max` | **170.9716** |
| `return_min` | `-10.4454` |
| `steps_mean` | **26.82** |
| `steps_max` | `50` |
| `steps_nonzero` | **100 / 100** |

- 后评估结果：`mol_evo/output/astar_rl/lumo_rl_eval_5_after_run100.json`
- 当前 5 分子 RL 评估：

| 指标 | 数值 |
|------|------|
| 非空 `topK` 分子数 | 3 / 5 |
| top1 改善均值 | **1.0347** |
| 单分子 top1 改善 | `0.0000`, `0.0000`, `1.1830`, `1.1719`, `2.8188` |

### 5.2 当前判断

- **正向结论**：在线 RL 已经不是“空跑搜索”，而是真正采到了轨迹并发生了参数更新。
- **负向结论**：在当前小规模评估上，RL 结果**仍弱于** BC-only baseline（`1.0347 < 2.0896`）。
- **最可能的瓶颈**：
  1. `open_set_budget=50` 偏小，搜索空间过浅；
  2. BC 数据仅来自 15 棵树，冷启动覆盖不足；
  3. 当前仅评估 5 个分子，结论噪声较大；
  4. 训练 checkpoint 以 episode return 为主，未直接按离线评估指标选优。

---

## Step 6 — 消融实验

### 6.0 本轮结果：P0 + P1 已完成（50 分子网格）

- 结果目录：`mol_evo/output/astar_rl/lumo_eval_grid50_parallel_20260409_194105`
- 覆盖组合：`BC / RL × open_set_budget(50/100/200) × top_n_prefilter(20/50)`
- 评估口径：`lumo decrease`，改善定义为 `initial_property_value - top1.property_value`

| method | budget | prefilter | nonempty topK | top1 mean | top1 median | actual expansions mean | ofo calls mean |
|--------|--------|-----------|---------------|-----------|-------------|------------------------|----------------|
| bc | 50 | 20 | 35 / 50 | 1.6826 | 1.5217 | 36.76 | 44.26 |
| rl | 50 | 20 | 50 / 50 | 2.2473 | 2.3492 | 256.58 | 367.22 |
| bc | 50 | 50 | 50 / 50 | 3.3464 | 3.1834 | 341.02 | 583.60 |
| rl | 50 | 50 | 50 / 50 | **7.4856** | 3.0376 | 341.86 | 567.58 |
| bc | 100 | 20 | 50 / 50 | 1.6633 | 1.5657 | 367.42 | 564.84 |
| rl | 100 | 20 | 45 / 50 | 2.2687 | 1.3959 | 208.16 | 333.44 |
| bc | 100 | 50 | 50 / 50 | 2.8019 | 2.7170 | 479.60 | 1093.78 |
| rl | 100 | 50 | 50 / 50 | 2.9246 | 2.9101 | 499.04 | 1326.82 |
| bc | 200 | 20 | 47 / 50 | 2.0748 | 2.1244 | 167.16 | 226.06 |
| rl | 200 | 20 | 45 / 50 | 1.8982 | 1.7691 | 114.48 | 163.86 |
| bc | 200 | 50 | 50 / 50 | 3.3620 | **3.2222** | 486.26 | 1026.88 |
| rl | 200 | 50 | 50 / 50 | 3.1404 | 2.7801 | 401.12 | 905.58 |

### 6.1 关键观察

- **`top_n_prefilter=50` 明显优于 `20`**：这个结论对 BC 和 RL 都成立，是当前最稳定的搜索参数发现。
- **按均值看，RL 最优配置已经出现**：`rl 50/50` 的 `top1_mean=7.4856`，显著高于当前所有 BC 组合。
- **但按稳健性看，RL 还没有稳定胜出**：
  - 与同配置 `bc 50/50` 做分子级配对时，RL 仅 **12 / 50** 个分子更好，**38 / 50** 个分子更差；
  - `delta_median=-0.3116`，说明多数样本上 RL 并未占优；
  - 去掉头尾各 3 个样本后的 `trimmed mean = -0.2753`，也说明 RL 的均值优势主要由少数极大值驱动。
- **当前最优 BC 配置**：`bc 200/50`，`top1_mean=3.3620`，`top1_median=3.2222`。
- **当前最优 RL 配置**：`rl 50/50`，但该结果包含 3 个超大 improvement 样本（`>20`，其中 2 个 `>50`），需要进一步复核这些分子是否属于真实有效提升。

### 6.2 阶段结论

- **P0 结论**：50 分子评估已经足够说明，原先 5 分子的“RL 不如 BC”结论过早；在更大样本上，RL 至少已经表现出**局部强优势**。
- **P1 结论**：继续调搜索时，应优先固定 `top_n_prefilter=50`；`open_set_budget` 并不是越大越好，当前最佳 RL 反而出现在 `budget=50`。
- **尚未完成的判断**：`rl 50/50` 的大均值，到底是“找到了少量极优样本”还是“少量异常值/评估口径问题”，目前还不能直接下结论。

### 6.3 下一步优化方案（更新后优先级）

#### P2：先做 outlier 复核，而不是立刻继续训练

- **目标**：确认 `rl 50/50` 的 3 个超大 improvement 是否真实可信。
- **优先检查分子**：`COCC(C)(C)O`、`C#CC(C)(C)CC`、`CC#CC(C)(C)C`。
- **检查内容**：
  - top1 候选的 `property_value` 是否异常极端；
  - 候选是否满足当前分子合法性 / 图构建约束；
  - 与 BC 同配置结果逐分子对照，确认是否为真实“RL-only 命中”。

#### P3：扩大 BC 冷启动数据，降低 RL 起点噪声

- **建议规模**：把 BFS 树从当前 `15` 棵扩大到 `50 ~ 100` 棵。
- **建议动作**：
  - 继续收集 `lumo` BFS 树；
  - 重新导出 `depth=3` / `depth=4` 两版 BC 数据；
  - 先比较更强 BC baseline，再决定 RL 是否基于更强起点重训。

#### P4：把 checkpoint 选择从训练回报切到离线评估

- **当前问题**：现在主要按 `episode_return` 保存 best checkpoint，这不一定和 50 分子离线指标一致。
- **建议**：训练期间每隔固定 episode 跑一个小型 holdout eval（例如 10 分子），按 `top1_mean / top1_median` 选 best checkpoint，而不是只看 return。

#### P5：在更强 BC 起点上重做在线 RL 扫描

在 P2 / P3 / P4 完成后，再考虑做以下训练扫描：

| 变量 | 建议值 |
|------|--------|
| `num_episodes` | `100 / 300 / 500` |
| `lr_policy` | `1e-4 / 5e-5` |
| `lr_value` | `1e-3 / 5e-4` |
| `entropy_coef` | `0.01 / 0.02 / 0.05` |
| `seed` | 至少 `3` 个 |

> 现在的重点已经从“链路能不能跑通”切换为：**先确认 RL 的大提升是否真实稳健，再决定是否继续放大训练规模。**

---

### 6.4 细分消融模板

#### 6.4.1 PolicyNet 预筛的影响

固定 `open_set_budget=200`，改变 `top_n_prefilter`：

| top_n_prefilter | 预期效果 |
|-----------------|---------|
| 全量（不预筛） | OFO 调用最多，效果接近纯 A* |
| 20（默认） | 平衡点 |
| 5 | OFO 调用少，但可能漏掉好候选 |

```bash
for N in 5 10 20 50; do
  python -m mol_evo.scripts.batch_optimizer \
    ... --search-mode astar_demo --rl-eval \
    --top-n-prefilter $N --open-set-budget 200
done
```

### 6.2 open_set_budget 的影响

固定 `top_n_prefilter=20`，改变 `open_set_budget`：

| budget | 预期效果 |
|--------|---------|
| 50 | 快但浅，与 BFS depth=2 差不多 |
| 100 | 中等 |
| 200（默认） | 接近 BFS depth=4 的覆盖面 |
| 500 | 深度搜索，时间增加明显 |

### 6.3 BC 数据量的影响

控制 `--max-trees` 参数，用不同数量的树训练 BC：

```bash
for N in 10 50 100 200; do
  python mol_evo/dataset/export_rl_demo_transitions.py \
    --input-dir ... --max-trees $N \
    --output-json mol_evo/dataset/rl_demo/bc_N${N}.json
  python mol_evo/train_bc_pretrain.py \
    --data-json mol_evo/dataset/rl_demo/bc_N${N}.json \
    --output-dir mol_evo/output/astar_rl/bc_N${N}
done
```

### 6.4 有无 ValueNet h_score

修改调用时不传 `value_net`：

```bash
# 无 ValueNet（只有 g_score + policy_bonus）
python -m mol_evo.scripts.batch_optimizer \
  ... --search-mode astar_demo --rl-eval \
  --policy-path mol_evo/output/astar_rl/bc/policy_best.pth
  # 不传 --value-path
```

---

## 实验结果记录模板

```markdown
### 实验 ID: exp-<date>-<desc>

**配置**
- search_mode:      astar_demo
- open_set_budget:  200
- top_n_prefilter:  20
- policy_path:      mol_evo/output/astar_rl/bc/policy_best.pth
- num_molecules:    100

**指标**（100 分子均值）

| 方法 | topK_best ↑ | topK_mean ↑ | diversity ↑ | ofo_calls ↓ | time(s/mol) ↓ |
|------|------------|------------|------------|------------|---------------|
| BFS  |            |            |            | —          |               |
| MCTS |            |            |            | —          |               |
| astar_demo (BC) |  |           |            |            |               |
| astar_demo (RL) |  |           |            |            |               |

**观察**
- ...
```

---

## 预期结论（假说）

1. **astar_demo (BC)** ofo_calls 应 < MCTS（通过预筛减少调用），topK 质量 ≥ random init
2. **astar_demo (RL)** topK_best 经 200+ episode 后应超过 BC-only，说明在线 RL 有增益
3. **diversity** astar_demo 路径感知去重应不低于 BFS
4. **time** 由于 PolicyNet 推理开销，astar_demo 比 BFS 慢约 10~30%（MLP 前向可忽略，主要是 Python 层循环）

---

## 失败排查

| 现象 | 可能原因 | 排查方法 |
|------|----------|----------|
| `episode_return` 不上升或下降 | lr 过高 / entropy_coef 太低导致 policy 坍缩 | 降低 `--lr-policy`，提高 `--entropy-coef` |
| `policy_loss` 为 NaN | 梯度爆炸 | 检查 reward 数值范围；rl_trainer.py 有梯度 clip（`max_norm=1.0`）是否生效 |
| `value_loss` 不收敛 | BC 数据不足 / lr 太小 | 增加 BC 数据量；提高 `--lr-value` |
| astar_demo topK 比 BFS 差 | open_set_budget 太小 | 增大到 `200`；或确认 BC 训练收敛 |
| 搜索提前结束（expanded_nodes 很小） | 所有候选被 path-aware dedup 过滤 | 减小 `--max-depth` 或换分子 |
| BC 训练在 DataLoader 阶段崩溃（`operation.get` 报错） | 离线样本里 `operation` 是字符串而非字典 | 确认 `encode_action()` 已兼容 `dict / str / None`；旧样本无需重导 |
| 在线 RL 日志显示 `steps=0`，但搜索本身能跑完 | policy/value 在 CUDA，状态/动作编码张量仍在 CPU，异常被搜索层 `try/except` 吞掉 | 检查 `generate_expansion_tree_astar_demo()` 中 state/action/value 张量是否显式 `.to(device)`，并确认 `rl_history.json` 的 `episode_steps` 非零 |
| ImportError: e3nn | 主 models __init__ 有副作用 | 改用 `from mol_evo.core.models.astar_rl import ...` 直接导入，绕过主 `__init__` |
