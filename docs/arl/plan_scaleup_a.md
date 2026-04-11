# A* RL Demo — 方案 A：低风险扩规模计划
<!-- last-updated: 2026-04-10 -->

本文档给出一版**不改核心算法、以工程稳健性为主**的扩规模计划。
目标不是立刻把 RL 改成全新框架，而是在当前 `BC → astar_demo → REINFORCE` 链路上，
先把**数据规模、训练时长、评估密度、seed 可靠性**做起来。

---

## 0. 当前结果冻结（作为方案 A 的起点）

### 当前已完成 run

- **BC 冷启动**：`mol_evo/output/astar_rl/lumo_bc_bfs15_depth3/bc_20260409_145233`
- **在线 RL（100 episodes）**：`mol_evo/output/astar_rl/lumo_rl_bfs15_depth3_run100/rl_20260409_161937`
- **50 分子网格复核**：`mol_evo/output/astar_rl/lumo_eval_grid50_parallel_20260409_194105`

### 当前冻结结论

- **5 分子结果仅作早期参考**：`BC-only top1_mean=2.0896`，`RL-after-train top1_mean=1.0347`，但该判断已被 50 分子网格复核覆盖。
- **稳定搜索结论**：`top_n_prefilter=50` 明显优于 `20`，方案 A 中默认优先固定为 `50`。
- **当前最优 BC 参考线**：`bc + budget=200 + prefilter=50`，`top1_mean=3.3620`，`top1_median=3.2222`。
- **当前最强 RL 候选**：`rl + budget=50 + prefilter=50`，`top1_mean=7.4856`，但仅 **12 / 50** 分子优于同配置 BC，`trimmed mean=-0.2753`，说明其均值优势仍可能主要来自少数大样本。

### 方案 A 的核心判断

- **可以放大训练规模**，但不应只把 `num_episodes` 机械拉大。
- **先做低风险扩规模**：更强 BC 起点、更大训练分子池、更多 episodes、固定 holdout eval、多 seed 复核。
- **best checkpoint 不再只看 `episode_return`**，而应以固定 holdout 的离线指标作为主标准。

---

## 1. 目标与边界

### 目标

- **目标 1**：确认在更强 BC 起点下，RL 是否能在 `50` 分子 holdout 上稳定超过 BC。
- **目标 2**：验证扩大 episodes 后，RL 的提升是否仍然只来自少数 outlier。
- **目标 3**：建立一套后续可复用的训练/评估节奏，便于继续扩大规模。

### 非目标

- **不在方案 A 中改搜索算法本身**：不改 `astar_demo` 的基本逻辑，不引入 PPO / actor-learner / replay buffer。
- **不在方案 A 中重定义 action**：仍沿用当前 `PolicyNet + ValueNet + OFO` 的搜索指导结构。
- **不把 5 分子评估作为主决策标准**。

---

## 2. 固定默认设置

方案 A 默认先固定一组更稳的设置，避免同时扫描太多变量。

| 项目 | 方案 A 默认值 | 说明 |
|------|---------------|------|
| `target_property` | `lumo` | 与当前结果保持一致 |
| `direction` | `decrease` | 与现有实验口径一致 |
| `max_depth` | `3` | 与当前已跑通 RL run 对齐 |
| `max_branching` | `8` | 与当前 astar_demo 默认实验口径一致 |
| `top_n_prefilter` | **`50`** | 来自 50 分子网格的稳定结论 |
| `open_set_budget` | **`50`** | 当前 RL 最佳均值配置所在 budget |
| `checkpoint_every` | `25` | 提高快照密度，便于选优 |
| `seed` | 至少 `3` 个 | 建议如 `11 / 22 / 33` |

> 备注：方案 A 优先验证“更大训练规模是否让 `rl 50/50` 变得更稳”，而不是同时把搜索空间也扩得更大。

---

## 3. 实验矩阵

### A0：准备固定 holdout 集与指标口径

#### 目标

- 固定一批**不参与在线训练选样逻辑**的 holdout 分子，用于 checkpoint 选优。
- 统一后续所有 run 的评价口径。

#### 建议设置

- **holdout 分子数**：`50`
- **训练分子池**：`500 ~ 2000`
- **主指标**：
  - `top1_mean`
  - `top1_median`
  - `win_rate_vs_bc`
  - `trimmed_mean`（去头尾各 10% 或固定各 3 个样本）
  - `nonempty_topk`

#### 交付物

- 固定的 holdout CSV
- BC 对应 holdout baseline 结果 JSON / TSV
- RL 各 checkpoint 对应 holdout 结果表

#### 当前执行进度（2026-04-10）

- **固定 holdout 已生成**：`mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_holdout50_start0.csv`
- **训练池已生成**：`mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_train_pool_excluding_holdout.csv`
- **切分元数据已落盘**：`mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_split_metadata.json`
- **当前 A0 baseline summary 已生成**：位于 `mol_evo/output/astar_rl/plan_a_a0`
  - `a0_bc_best_budget200_pref50_summary.json/tsv`
  - `a0_bc_match_budget50_pref50_summary.json/tsv`
  - `a0_rl_best_mean_budget50_pref50_vs_bc50_pref50_summary.json/tsv`
- **当前固定口径**：holdout `50` 分子、train pool `983` 分子、trim 规则为“两端各去掉 3 个样本”

### A1：扩大 BC 冷启动数据

#### 目标

把 BC 起点从当前 **15 棵树 / 9090 条样本** 扩大到更强版本，降低 RL 起点噪声。

#### 建议数据规模

| 版本 | BFS 树数 | 深度 | 用途 |
|------|----------|------|------|
| `BC-A1-small` | `50` | `3` | 第一轮扩大版，快速验证 |
| `BC-A1-main` | `100` | `3` | 方案 A 主版本 |
| `BC-A1-depth4` | `50` | `4` | 可选扩展，用于比较更深数据是否有帮助 |

#### 成功标准

- `BC-only` 在固定 holdout 上**至少不弱于**当前最优 BC 参考线；
- 若 `BC-A1-main` 的 holdout 中位数无法超过当前 BC 参考线，则暂停继续放大 RL。

#### 当前执行进度（2026-04-11 01:34）

- **A1-small 已完成**：`train_pool(983) → BFS 50 trees → 导出 transitions → BC 训练 → holdout eval`
- **A1-main 已完成**：`train_pool(983) → BFS 100 trees → 导出 transitions → BC 训练 → holdout eval`
- **A1-main 启动脚本**：`mol_evo/output/astar_rl/start_plan_a_a1_main_tmux.sh`
- **训练池来源**：`mol_evo/dataset/eval-data/plan_a_a0/lumo_plan_a_train_pool_excluding_holdout.csv`
- **A1-small 样本数 / 结果**：`27351` 条，`top1_mean=3.2498`，`top1_median=3.2552`，`trimmed_mean=3.1784`
- **A1-main 样本数 / 结果**：`53269` 条，`top1_mean=4.2094`，`top1_median=3.3734`，`trimmed_mean=3.3053`
- **A1-main BC 权重目录**：`mol_evo/output/astar_rl/lumo_plan_a_a1_main_bc/bc_20260411_000941`
- **A1-main holdout 评估目录**：`mol_evo/output/astar_rl/plan_a_a1_main_eval`
- **A1-main 训练停止方式**：设置 `epochs=100`，但因 `patience=20` 的 early stop，实际在 `epoch 51` 停止；最佳验证损失位于 `epoch 31`
- **相对 A0 最优 BC 基线**（`bc_budget200_pref50`）：`win_rate=48%`，`delta_mean=+0.8474`，`delta_median≈0`，`trimmed_delta_mean=-0.0250`
- **阶段判断**：`A1-main` 明显优于 `A1-small`，且 raw holdout 汇总指标已略高于 A0；但配对稳健性仍然接近持平，因此更适合视为“**达到可进入 A2 的门槛边缘，但不是特别强的压倒性胜出**”

### A2：在线 RL 第一轮放大（300 episodes）

#### 目标

先做一轮**低风险增量放大**，看更长训练是否带来更稳的 RL 提升。

#### 实验设置

| 项目 | 值 |
|------|----|
| 起点权重 | `BC-A1-main` 最优权重 |
| `num_episodes` | `300` |
| `seed` | `3` 个 |
| `top_n_prefilter` | `50` |
| `open_set_budget` | `50` |
| `checkpoint_every` | `25` |
| holdout eval 频率 | 每 `25` episodes |

#### 观察重点

- RL 的 holdout `top1_median` 是否开始稳定超过 BC；
- `win_rate_vs_bc` 是否从当前明显偏低，提升到至少接近或超过 `50%`；
- `trimmed_mean` 是否由负转正。

#### 阶段门槛

满足以下任一条件，可进入 A3：

- `3` 个 seed 中至少 `2` 个在 holdout 上同时满足：
  - `top1_median > BC baseline`
  - `trimmed_mean > 0`
- 或者 RL 在 holdout 上虽然均值优势不大，但 `win_rate_vs_bc ≥ 50%` 且中位数不劣于 BC。

### A3：在线 RL 第二轮放大（1000 episodes）

#### 目标

验证第一轮观察到的 RL 改善，是否在更长训练中持续存在。

#### 实验设置

| 项目 | 值 |
|------|----|
| 起点权重 | A2 中各 seed 的最佳 checkpoint |
| `num_episodes` | `1000` |
| `seed` | 延续 A2 的 `3` 个 |
| 搜索参数 | 仍固定 `budget=50`、`prefilter=50` |
| holdout eval 频率 | 每 `25` 或 `50` episodes |

#### 成功标准

- 至少 `2 / 3` seeds 在 holdout 上稳定优于 BC；
- RL 的 `trimmed_mean` 与 `top1_median` 均不再主要依赖少数 outlier；
- 最佳 checkpoint 在相邻两次 holdout eval 上表现波动可控，而不是单点尖峰。

---

## 4. Checkpoint 选择规则

方案 A 明确要求：**不再只按 `episode_return` 选 best checkpoint。**

### 建议优先级

1. **主标准**：holdout `top1_median`
2. **并列打破**：holdout `trimmed_mean`
3. **辅助指标**：`win_rate_vs_bc`
4. **监控但不单独决策**：训练期 `episode_return`

### 不建议的做法

- 只因某个 checkpoint 的 `episode_return` 很高就保存为“最终最好”；
- 只看 `top1_mean`，忽略分布偏斜和 outlier 影响。

---

## 5. 记录模板

每个 run 至少记录以下内容：

| 字段 | 示例 |
|------|------|
| `run_name` | `rl_a2_seed11_ep300` |
| `bc_init` | `BC-A1-main/policy_best.pth` |
| `seed` | `11` |
| `episodes` | `300` |
| `budget/prefilter` | `50 / 50` |
| `best_ckpt_by_holdout` | `ep0250` |
| `holdout_top1_mean` | `...` |
| `holdout_top1_median` | `...` |
| `holdout_trimmed_mean` | `...` |
| `win_rate_vs_bc` | `...` |
| `notes` | `是否存在明显 outlier` |

建议每轮训练结束后，统一输出：

- 一个汇总 `summary.tsv`
- 一个 run 级简报 markdown
- 一次 `docs/arl/experiments.md` 的结果回填

---

## 6. 风险与止损条件

### 主要风险

- **风险 1**：RL 继续放大后，均值仍主要依赖少数超大样本；
- **风险 2**：更强 BC 起点本身已足够强，RL 难以再提供稳定增益；
- **风险 3**：只放大 episode，会放大训练噪声而不是实际策略收益。

### 止损条件

若出现以下任一情况，方案 A 暂停，转向机制升级而不是继续堆训练：

- `A2 + A3` 后，`3` 个 seeds 中超过 `2` 个仍无法在 holdout `top1_median` 上超过 BC；
- RL 的 `trimmed_mean` 长期为负，但 `top1_mean` 仍偶发冲高；
- 最佳 checkpoint 在不同 holdout 切片之间极不稳定。

---

## 7. 预期交付物

方案 A 完成后，应至少得到：

- **一套更强 BC 基线**（50 / 100 棵树版本）
- **两轮 RL 扩规模结果**（300 episodes 与 1000 episodes）
- **多 seed 汇总表**
- **固定 holdout 选优机制**
- **明确结论**：
  - RL 是否已在稳健指标上超过 BC；
  - 若没有，下一步该转向 `方案 B`（批量更新 / reward 稳健化）还是更改 action 定义。

---

## 8. 建议执行顺序

- **第 1 周**：完成 A0 + A1，拿到更强 BC baseline。
- **第 2 周**：完成 A2（300 episodes × 3 seeds），形成第一轮结论。
- **第 3 周**：仅在 A2 有正向信号时进入 A3（1000 episodes × 3 seeds）。
- **第 4 周**：汇总结果，回填 `experiments.md`，决定是否进入后续机制升级。

> 方案 A 的核心不是“先把训练跑得很大”，而是：**先在当前框架下，把更大规模训练做成一套能稳定比较、稳定选优、稳定复现实验结论的流程。**
