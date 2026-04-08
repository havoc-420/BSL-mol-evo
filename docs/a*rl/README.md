# A\* RL Demo：分子优化 RL 方向实验文档
<!-- last-updated: 2026-04-08 -->

## 概述

本目录记录基于现有 BFS/MCTS OFO 搜索框架的 **RL 方向分子优化 demo（astar_demo）** 的设计、训练流程、运行方法与评测口径。

Demo 定位：**最小可验证实验**，不替换现有 BFS/MCTS 生产路径，仅在 `astar_demo` 分支下验证 RL 路线是否能提升搜索效率与 TopK 质量。

---

## 目录结构

```
mol_evo/
├── core/
│   ├── models/astar_rl/
│   │   ├── __init__.py               # 包导出
│   │   ├── policy_network.py         # PolicyNet（候选动作排序模型）
│   │   ├── value_network.py          # ValueNet（未来收益估计模型）
│   │   ├── reward.py                 # 统一奖励函数 + RewardConfig
│   │   └── rl_trainer.py             # REINFORCE / PPO 在线训练器
│   └── data/
│       └── rl_demo_processing.py     # 状态/动作编码 + BC Dataset/DataLoader
├── dataset/
│   └── export_rl_demo_transitions.py # BFS/MCTS 树 → BC 样本导出
├── train_bc_pretrain.py              # Phase 1: BC 冷启动训练入口
├── train_astar_rl_demo.py            # Phase 2: 在线 RL 训练入口
└── scripts/
    └── batch_optimizer.py            # 复用现有批量入口（已新增 astar_demo 相关参数）
```

---

## 两阶段训练流程

### Phase 1：离线 BC 冷启动（Behavioral Cloning）

**目的**：给 PolicyNet / ValueNet 提供合理初始参数，避免 `astar_demo` 搜索完全随机展开。

**数据来源**：现有 BFS/MCTS 搜索树的真实轨迹（每条边 = 一个训练样本）。

```bash
# 1. 用现有 BFS 跑一批实验，保存搜索树 JSON
python -m mol_evo.scripts.batch_optimizer \
  --input-csv mol_evo/dataset/eval-data/qm9_test_molecules.csv \
  --search-mode bfs --max-depth 3 --max-branching 8

# 2. 从搜索树导出 BC 样本
python -m mol_evo.dataset.export_rl_demo_transitions \
  --input-dir mol_evo/output/evo-mo/batch_optimization_XXX \
  --output-json mol_evo/dataset/rl_demo/bc_transitions.json \
  --direction decrease --max-depth 3

# 3. BC 预训练
python mol_evo/train_bc_pretrain.py \
  --data-json mol_evo/dataset/rl_demo/bc_transitions.json \
  --output-dir mol_evo/output/astar_rl/bc \
  --direction decrease --epochs 50 --batch-size 64
```

### Phase 2：在线 RL 微调（REINFORCE）

**目的**：让 policy/value 在 `astar_demo` 搜索过程中持续进化，超越 BFS/MCTS 基线。

```bash
# 在线 RL 训练（每个 episode = 一次完整搜索）
python mol_evo/train_astar_rl_demo.py \
  --input-csv mol_evo/dataset/eval-data/qm9_test_molecules.csv \
  --model-path /path/to/ofo_model.pth \
  --model-dir /path/to/ofo_model_dir \
  --config-file /path/to/config.yaml \
  --policy-path mol_evo/output/astar_rl/bc/policy_best.pth \
  --value-path mol_evo/output/astar_rl/bc/value_best.pth \
  --output-dir mol_evo/output/astar_rl/rl \
  --num-episodes 200 --direction decrease

# 纯评估（加载固定权重，不更新）
python -m mol_evo.scripts.batch_optimizer \
  --search-mode astar_demo \
  --rl-eval \
  --policy-path mol_evo/output/astar_rl/rl/rl_ckpt_ep000200.pth \
  ...
```

---

## 搜索流程（astar_demo）

```
1. _get_possible_operations()  →  全量合法候选操作
2. PolicyNet.top_k_actions()   →  top-N 预筛（减少 OFO 调用次数）
3. predict_batch()             →  OFO 批量打分（property_change）
4. ValueNet.estimate()         →  未来收益 h_score
5. f_score = g_score + h_score + policy_bonus
6. heapq 优先队列按 f_score 展开
7. 收集 TrajectoryStep → RLTrainer（在线训练）或忽略（纯评估）
```

### 节点额外字段

在现有 `nodes` 字段基础上，`astar_demo` 新增：

| 字段 | 说明 |
|------|------|
| `g_score` | 从初始节点出发的累计已实现收益 |
| `h_score` | ValueNet 估计的未来收益（启发值） |
| `f_score` | `g_score + h_score`（优先队列排序依据）|
| `policy_score` | PolicyNet 对此节点的优先级分数 |
| `stagnation_count` | 连续无改善步数 |
| `logp_violation_count` | 连续 logP 超限步数 |

### astar_stats 字段

每次搜索附加 `astar_stats` 到树结构 JSON：

| 字段 | 说明 |
|------|------|
| `expanded_nodes` | 实际展开的节点数 |
| `open_set_peak` | open set 队列峰值大小 |
| `policy_prefilter_size` | policy 预筛后的候选数 |
| `ofo_scored_candidates` | 送入 OFO 打分的候选数 |
| `actual_expansions` | 实际加入树的节点数 |

---

## 奖励函数（RewardConfig 默认值）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `property_weight` | 1.0 | 属性改善主信号权重 |
| `logp_penalty` | 0.5 | logP 越界固定惩罚 |
| `stagnation_penalty` | 0.3 | 停滞惩罚（超过 patience 后） |
| `stagnation_patience` | 2 | 与现有 BFS `pruning_patience` 对齐 |
| `diversity_weight` | 0.1 | 多样性 bonus 权重 |
| `episode_weight` | 0.5 | episode 全局奖励权重 |

---

## 评测口径（BFS vs MCTS vs astar_demo）

使用 `mol_evo/evaluate_batch_mo.py` 或自定义脚本对比：

| 指标 | 说明 |
|------|------|
| TopK best improvement | TopK 中属性改善最大值 |
| TopK diversity | TopK SMILES 的 Tanimoto 多样性 |
| expanded_nodes | 实际展开节点数（搜索预算利用率） |
| OFO calls | 送入 OFO 模型的总调用次数 |
| 耗时（秒/分子） | 端到端运行时间 |

推荐运行顺序：
1. BFS baseline（现有 `--search-mode bfs`）
2. MCTS baseline（现有 `--search-mode mcts`）
3. astar_demo (BC only, `--rl-eval`)
4. astar_demo (BC + RL, `--rl-eval` after online training)

---

## 关键技术决策

- **不覆盖 BFS/MCTS**：`astar_demo` 作为独立分支，旧代码行为不变
- **OFO 只做单步 scorer**：延续现有 `predict_batch()` 用法，不当全局 heuristic
- **路径感知去重**：不用简单 `closed_set(smiles)`，而是允许同一分子在不同路径下被访问（同 MCTS 策略）
- **REINFORCE 先行**：PPO 作为 `PPORLTrainer` 子类预留接口，demo 阶段不强制使用
- **日志复用**：沿用 `batch_optimization_main.log` + `batch_optimization_total.log`，新增 `astar_stats` 字段

---

## 已知限制

1. **PolicyNet / ValueNet 均为轻量 MLP**：状态仅用 50-bit Morgan 指纹表示，未利用图结构，BC 后的初始质量有限
2. **离线 BC 数据有限**：bootstrap 质量依赖跑了多少 BFS/MCTS 树，数据越多效果越好
3. **在线 RL 每 episode 一次搜索**：episode 较短（depth≤4），梯度信号稀疏，需要足够 episode 数才能收敛
4. **RDKit logP 计算开销**：每步打分增加约 0.5ms，在候选数量大时有可见耗时
5. **PPO 未完整实现**：`PPORLTrainer` 当前 fallback 到 REINFORCE

---

## 参考

- 现有主链路：`mol_evo/core/evolution_optimizer.py` + `molecular_evolution_expansion.py`
- 数据构建参考：`mol_evo/core/data/path_processing.py`
- 训练脚本参考：`mol_evo/train_v0_3_path.py`
- 设计文档：`mol_evo/docs/ideas/astar-rl-molecular-expansion.md`（如有）
