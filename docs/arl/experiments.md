# A* RL Demo — 实验推进手册
<!-- last-updated: 2026-04-09 -->

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

---

## Step 6 — 消融实验

### 6.1 PolicyNet 预筛的影响

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
| ImportError: e3nn | 主 models __init__ 有副作用 | 改用 `from mol_evo.core.models.astar_rl import ...` 直接导入，绕过主 `__init__` |
