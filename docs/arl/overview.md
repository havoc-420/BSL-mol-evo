# A* RL Demo — 系统概览
<!-- last-updated: 2026-04-09 -->

## 定位

在现有 BFS / MCTS 搜索框架之上叠加一条**轻量 RL 分支**（`astar_demo`），
通过 A* 优先队列 + PolicyNet / ValueNet 引导搜索，验证 RL 路线能否在相同预算下
产出更优的 TopK 分子。

- 不替换、不修改 BFS / MCTS 生产路径
- 采用"先 BC 冷启动，再在线 REINFORCE 微调"两阶段训练策略
- 所有输出与现有 `save_optimized_tree / get_topK_results` 接口完全兼容

---

## 模块地图

```
mol_evo/
├── core/
│   ├── models/astar_rl/                    ← RL 网络与训练器
│   │   ├── __init__.py                     # 包导出（PolicyNet/ValueNet/RLTrainer/RewardConfig）
│   │   ├── policy_network.py               # PolicyNet：候选动作排序 MLP
│   │   ├── value_network.py                # ValueNet：未来收益估计 MLP
│   │   ├── reward.py                       # RewardConfig + 奖励函数
│   │   └── rl_trainer.py                   # TrajectoryStep + RLTrainer（REINFORCE）
│   ├── data/
│   │   └── rl_demo_processing.py           # 状态/动作编码 + BCDataset + DataLoader
│   ├── molecular_evolution_expansion.py    # 新增 generate_expansion_tree_astar_demo()
│   └── evolution_optimizer.py             # 新增 astar_demo 路由分支
├── dataset/
│   └── export_rl_demo_transitions.py       # BFS/MCTS 树 JSON → BC 样本导出
├── train_bc_pretrain.py                    # Phase 1：BC 冷启动
├── train_astar_rl_demo.py                  # Phase 2：在线 RL 训练
├── scripts/
│   └── batch_optimizer.py                  # 批量评估入口（已扩展 astar_demo 参数）
└── tests/
    ├── test_astar_demo_search_mode.py      # 搜索模式集成测试
    └── test_rl_training.py                 # RLTrainer / BCDataset / Reward 单元测试
```

---

## 三条搜索模式对比

| 维度 | BFS | MCTS | astar_demo |
|------|-----|------|------------|
| 展开顺序 | 层序宽度优先 | UCT 模拟得分 | f_score 优先（A*） |
| 候选预筛 | 无 | 无 | PolicyNet top-k |
| 启发函数 | 无 | rollout 估值 | ValueNet h_score |
| 在线更新 | 无 | 无 | REINFORCE |
| 主要超参 | `max_depth / max_branching` | `num_simulations / exploration_weight` | `open_set_budget / top_n_prefilter` |

---

## 数据流总览

```
BFS/MCTS 搜索树 JSON
        │
        ▼
export_rl_demo_transitions.py   ──→  bc_transitions.json
        │
        ▼
train_bc_pretrain.py            ──→  policy_best.pth / value_best.pth
        │
        ▼
train_astar_rl_demo.py          ──→  policy_last.pth / value_last.pth
   (每 episode = 一次 astar_demo 搜索，内部调用 RLTrainer.collect_step + end_episode)
        │
        ▼
batch_optimizer.py --rl-eval    ──→  TopK 评估结果 JSON / CSV
```

---

## f_score 公式

```
f_score = g_score + h_score + policy_bonus

g_score     = 已实现的属性累计改善（方向归一化，越大越好）
h_score     = ValueNet.estimate(next_state)  （可选，None 时 = 0）
policy_bonus = logit × 0.1                  （PolicyNet 排序分的轻量影响）
```

---

## 关键技术决策

| 问题 | 选择 | 原因 |
|------|------|------|
| 状态表示 | 50-bit Morgan 指纹 + 7 个标量 → pad to 64 | 轻量可微，与现有 rdkit 依赖对齐 |
| 动作表示 | op_type one-hot(10) + position/fg_size/ofo_change → pad to 16 | 对应现有操作类型数量 |
| BC 策略损失 | in-batch softmax cross-entropy（batch 内负样本） | 无显式排序标注时的次优近似 |
| 去重策略 | `(smiles, parent_id)` 路径感知 | 允许同分子出现在不同路径（同 MCTS） |
| 训练算法 | REINFORCE + ValueNet baseline + 熵正则 | 实现简单，PPORLTrainer 接口已预留 |
| end_episode 时机 | 由 `generate_expansion_tree_astar_demo` 内部调用 | 训练脚本只读 `rl_trainer.history[-1]` |
