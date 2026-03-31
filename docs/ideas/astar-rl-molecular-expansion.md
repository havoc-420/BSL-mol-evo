# A* + RL 分子扩展搜索优化方案

## 1. 背景与动机

### 1.1 现状分析

当前分子优化系统（MO 任务）采用进化树搜索策略，核心架构如下：

```
EvolutionTreeOptimizer (协调器)
  ├── GNN 预测器 (v0 VisNet, 预测单步 Δproperty)
  └── MolecularEvolutionExpansion (搜索引擎)
        ├── BFS + Beam Search (默认, top-50% 过滤)
        └── MCTS + PUCT (可选)
```

**现有搜索策略的局限**：

| 问题 | BFS (默认) | MCTS (可选) |
|------|-----------|------------|
| **搜索效率** | 无启发式引导，暴力扩展所有节点 | 依赖随机模拟，需要大量 rollout |
| **操作空间** | 每步操作数 = `N_atoms × |atom_types| + N_pairs × |bond_types|`，按 top-50% 粗暴截断 | 扩展时仍需全量生成候选，开销大 |
| **可解释性** | 无评分机制，仅靠排序截断 | Q 值来自累积回传，语义不明确 |
| **收敛保证** | 不保证最优 | 不保证最优 |

### 1.2 引入 A* + RL 的目标

1. **提升搜索效率**：通过 RL 学到的启发函数 h(s) 引导搜索方向，优先扩展最有潜力的路径
2. **减少搜索空间**：通过 PolicyNet 候选过滤，替代全量操作扩展
3. **保证解的质量**：当 h(s) admissible 时，A* 保证找到最优解
4. **与现有模型兼容**：v0 预测器冻结作为 reward shaping 的一部分，v0.3 链式模型可直接利用 HeuristicNet 的中间步先验

---

## 2. 问题建模

### 2.1 MDP 建模

将分子扩展过程建模为马尔可夫决策过程 (MDP)：

| MDP 元素 | 对应含义 | 现有代码映射 |
|----------|---------|-------------|
| **State s** | 当前分子 SMILES + 累积属性变化 | `node["smiles"]`, `node["accumulated_change"]` |
| **Action a** | 执行一个化学操作 | `_get_possible_operations()` 返回的操作列表 |
| **Transition T(s,a)→s'** | 应用操作生成新分子 | `_apply_operation()` |
| **Reward r(s,a,s')** | 属性改善 + 约束满足 | 基于现有 `property_change` + `logP_in_range` |
| **Goal** | 累积属性变化满足目标，logP 在范围内 | `optimization_mode` 配置 |

### 2.2 A* 搜索建模

在 MDP 基础上引入 A* 搜索：

$$f(s) = g(s) + h(s)$$

- **g(s)**：从初始状态到 s 的实际代价，定义为 $g(s) = -\text{accumulated\_change}$（最小化代价 = 最大化属性改善）
- **h(s)**：从 s 到目标的估计剩余代价，由 RL 学到的 HeuristicNetwork 预测
- **admissible 条件**：$h(s) \leq h^*(s)$，保证 A* 找到最优解

### 2.3 RL 目标

同时训练两个网络：

- **PolicyNet π(a|s)**：给定状态，输出操作选择的概率分布，用于候选过滤（替代全量扩展）
- **HeuristicNet h(s)**：估计从当前状态到目标的剩余代价，用于 A* 的启发估计

---

## 3. 系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                   A* Search Engine                       │
│                                                          │
│  f(s) = g(s) + h(s)                                      │
│  g(s) = -accumulated_change                              │
│  h(s) = HeuristicNet(s, goal)    ← RL 学出的启发函数      │
│                                                          │
│  节点扩展:                                                │
│    candidates = PolicyNet(s)     ← RL 学出的策略筛选       │
│    candidates += model_predict() ← v0 预测补充            │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│                 RL Training Pipeline                      │
│                                                          │
│  Actor:  PolicyNet(s) → π(a|s)   输出 top-K 操作         │
│  Critic: HeuristicNet(s,g) → h    输出剩余代价估计        │
│                                                          │
│  Reward: Δproperty + logP bonus + validity bonus          │
│  Method: PPO                                              │
└──────────────────────────────────────────────────────────┘
```

### 3.2 与现有系统的集成点

| 修改位置 | 改动内容 | 说明 |
|----------|---------|------|
| `scripts/batch_optimizer.py` | 新增 `--search-mode astar` 选项 | 传入 `heuristic_fn` 和 `policy_fn` |
| `core/search/evolution_optimizer.py` | `optimize_evolution_tree()` 新增 A* 分支 | 加载 RL 模型权重 |
| `core/search/molecular_evolution_expansion.py` | 新增 `generate_expansion_tree_astar()` 方法 | 实现 A* 搜索逻辑 |
| 新增 `core/models/astar_rl/` | 放置 RL 模型定义 | `OperationPolicyNetwork` + `HeuristicNetwork` |
| 新增 `train/train_astar_rl.py` | RL 训练脚本 | 从搜索结果离线学习 |
| 新增 `scripts/batch_optimizer_astar.py` | 带 A* 搜索的批量优化脚本 | 封装完整 A* + RL 流程 |

---

## 4. A* 搜索实现

### 4.1 核心算法

在 `MolecularEvolutionExpansion` 类中新增 `generate_expansion_tree_astar()` 方法：

```python
import heapq

def generate_expansion_tree_astar(
    self,
    max_depth: int,
    max_branching: int,
    predictor,           # v0 预测器（冻结）
    heuristic_fn,        # HeuristicNetwork
    policy_fn,           # OperationPolicyNetwork
    optimization_direction: str,
    pruning_patience: int,
    logp_range: tuple,
    logp_patience: int,
    initial_property_value: float,
    optimization_mode: str,
):
    """
    A* 搜索: f(s) = g(s) + h(s)

    g(s): 从初始状态到 s 的实际代价 = -accumulated_change
    h(s): 从 s 到目标的估计剩余代价 = HeuristicNet(s, goal)
    """
    root_g = 0.0
    root_h = heuristic_fn(self.initial_smiles, 0.0, initial_property_value)
    root_f = root_g + root_h

    counter = 0
    open_set = [(root_f, counter, self.initial_smiles, 0, 0.0, [])]
    closed_set = set()

    while open_set:
        f, _, current_smiles, depth, accumulated, path = heapq.heappop(open_set)

        if current_smiles in closed_set:
            continue
        closed_set.add(current_smiles)

        if depth >= max_depth:
            continue

        mol = Chem.MolFromSmiles(current_smiles)
        if mol is None:
            continue

        # 获取所有候选操作
        possible_ops = self._get_possible_operations(mol)

        # PolicyNet 候选过滤: 只取 top-max_branching 个最有潜力的操作
        scored_ops = policy_fn(current_smiles, possible_ops)
        selected_ops = scored_ops[:max_branching]

        for op, policy_score in selected_ops:
            new_mol = self._apply_operation(mol, op["type"], **op.get("params", {}))
            if new_mol is None or not self.validate_molecule(new_mol):
                continue

            new_smiles = Chem.MolToSmiles(new_mol)
            if new_smiles in closed_set:
                continue

            # v0 模型预测属性变化
            property_change = predictor.predict_property_change(
                current_smiles, new_smiles, op
            )
            new_accumulated = accumulated + property_change

            # 计算 g, h, f
            new_g = -new_accumulated
            new_h = heuristic_fn(new_smiles, new_accumulated, initial_property_value)
            new_f = new_g + new_h

            counter += 1
            heapq.heappush(open_set, (
                new_f, counter, new_smiles, depth + 1, new_accumulated,
                path + [{"smiles": new_smiles, "op": op, "change": property_change}]
            ))
```

### 4.2 关键设计决策

1. **g(s) 的定义**：使用 `-accumulated_change`，这样最小化代价等价于最大化属性改善
2. **h(s) 的 admissible 保证**：HeuristicNet 输出经过 `ReLU`，保证 $h(s) \geq 0$；训练时用保守目标 $h^*(s)$ 作为监督信号
3. **PolicyNet 候选过滤**：替代全量 `_get_possible_operations()` 扩展，将每步操作数从数百级降低到 `max_branching`（通常 5-20）
4. **与 v0 的协作**：v0 预测器冻结，仅用于预测 `property_change`，不参与 RL 训练

---

## 5. RL 模型设计

### 5.1 Policy Network（操作选择策略）

```python
class OperationPolicyNetwork(nn.Module):
    """
    给定状态 s，输出操作选择策略 π(a|s)
    用于候选过滤：从所有可能的操作中筛选 top-K
    """

    def __init__(self, node_feature_dim=11, hidden_dim=256, num_op_types=17):
        super().__init__()
        # 共享 v0 的 VisNet 特征提取器（预训练，可微调）
        self.molecule_encoder = VisNetMoleculeFeatureExtractor(
            node_feature_dim, [128, 256, 256]
        )
        # 操作打分头
        self.op_scoring_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_op_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mol_graph_data, candidate_op_features):
        """
        Args:
            mol_graph_data: 当前分子图数据
            candidate_op_features: (K, num_op_types) 候选操作的 one-hot 编码
        Returns:
            scores: (K,) 每个候选操作的得分
        """
        mol_features = self.molecule_encoder(mol_graph_data)  # (hidden_dim * 2,)
        K = len(candidate_op_features)
        mol_expanded = mol_features.unsqueeze(0).expand(K, -1)
        combined = torch.cat([mol_expanded, candidate_op_features], dim=-1)
        scores = self.op_scoring_head(combined).squeeze(-1)
        return scores
```

**操作特征编码**：

```python
def encode_operation(op: dict, num_op_types: int = 17) -> torch.Tensor:
    """将操作编码为 one-hot 向量"""
    op_type_idx = OP_TYPE_TO_IDX[op["type"]]  # add_atom, replace_atom, ...
    feature = torch.zeros(num_op_types)
    feature[op_type_idx] = 1.0
    # 可选: 加入原子类型信息、位置信息等
    return feature
```

### 5.2 Heuristic Network（剩余代价估计）

```python
class HeuristicNetwork(nn.Module):
    """
    估计从当前状态到目标的剩余代价 h(s)
    需满足 admissible: h(s) <= h*(s)
    """

    def __init__(self, node_feature_dim=11, hidden_dim=256):
        super().__init__()
        self.molecule_encoder = VisNetMoleculeFeatureExtractor(
            node_feature_dim, [128, 256, 256]
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),  # +2: accumulated_change, target
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mol_graph_data, accumulated_change, target_value):
        """
        估计 h(s) = 剩余需要改善的量
        输出经过 ReLU 保证 h >= 0 (admissible)
        """
        mol_features = self.molecule_encoder(mol_graph_data)
        state_input = torch.cat([
            mol_features,
            torch.tensor([accumulated_change, target_value])
        ])
        h = self.value_head(state_input)
        return torch.relu(h)  # 保证非负
```

**admissible 的训练保证**：

训练时，$h^*(s)$ 定义为从状态 $s$ 到目标的最优路径的实际剩余代价（从搜索结果中回溯计算）。HeuristicNet 的输出经过 `ReLU` 保证 $h(s) \geq 0$，同时在损失函数中加入保守性惩罚：

$$\mathcal{L}_{\text{conservative}} = \max(0, h(s) - h^*(s))^2$$

这鼓励网络 underestimate，满足 admissible 条件。

---

## 6. RL 训练流程

### 6.1 整体训练流程

```
Phase 1: 离线数据收集        Phase 2: RL 训练           Phase 3: 在线微调
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  BFS/MCTS 搜索   │      │  PPO 训练        │      │  A* 搜索         │
│  收集 (s,a,r,s') │ ──→  │  PolicyNet       │ ──→  │  收集新数据       │
│  计算优势函数     │      │  HeuristicNet    │      │  Fine-tune       │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

### 6.2 离线数据收集

从现有 BFS/MCTS 搜索结果中提取训练数据：

```python
def collect_training_data_from_search_results(search_results, optimization_direction):
    """
    从进化树搜索结果中提取 (s, a, r, s') 转移

    Returns:
        transitions: list of dict, 每条记录包含状态、动作、奖励、下一状态、实际剩余代价
    """
    transitions = []

    for tree in search_results:
        # 找到最终目标值，用于计算 h*(s)
        target_value = tree.get("target_property_value")

        for edge in tree["edges"]:
            parent = tree["nodes"][edge["from"]]
            child = tree["nodes"][edge["to"]]

            # Reward 设计（详见 6.3 节）
            reward = compute_reward(child, optimization_direction, logp_range)

            # 从搜索结果回溯计算实际剩余代价 h*(s)
            actual_remaining = compute_actual_remaining_cost(tree, edge["to"], target_value)

            transitions.append({
                "smiles_from": parent["smiles"],
                "smiles_to": child["smiles"],
                "operation": edge["operation"],
                "property_change": child["property_change"],
                "accumulated_from": parent["accumulated_change"],
                "accumulated_to": child["accumulated_change"],
                "reward": reward,
                "actual_remaining_cost": actual_remaining,
                "depth": child["depth"],
            })

    return transitions
```

### 6.3 Reward 设计

```python
def compute_reward(child_node, optimization_direction, logp_range):
    """
    多目标 Reward = 属性改善 + 约束满足 + 有效性

    Args:
        child_node: 进化树子节点
        optimization_direction: "increase" 或 "decrease"
        logp_range: (min_logp, max_logp)

    Returns:
        total_reward: float
    """
    # 1. 属性改善 reward（主目标）
    delta = child_node["property_change"]
    if optimization_direction == "decrease":
        reward_property = -delta   # 下降方向: 负变化 = 正 reward
    else:
        reward_property = delta

    # 2. logP 约束惩罚
    reward_logp = 0.0
    logp = child_node.get("logP")
    if logp is not None:
        if logp < logp_range[0]:
            reward_logp = -0.5 * (logp_range[0] - logp)
        elif logp > logp_range[1]:
            reward_logp = -0.5 * (logp - logp_range[1])

    # 3. 分子有效性 bonus
    mol = Chem.MolFromSmiles(child_node["smiles"])
    reward_validity = 0.1 if mol is not None else -1.0

    total_reward = reward_property + reward_logp + reward_validity
    return total_reward
```

### 6.4 PPO 训练循环

```python
def train_astar_rl(transitions, config):
    """PPO 训练 PolicyNet + HeuristicNet"""

    policy_net = OperationPolicyNetwork(node_feature_dim=11).to(device)
    heuristic_net = HeuristicNetwork(node_feature_dim=11).to(device)
    v0_predictor = load_pretrained_v0_model()  # 冻结，仅用于 reward shaping

    # 可选: 共享 encoder 参数初始化
    policy_net.molecule_encoder.load_state_dict(
        v0_predictor.molecule_encoder.state_dict(), strict=False
    )

    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(heuristic_net.parameters()),
        lr=config.learning_rate
    )

    for epoch in range(config.num_epochs):
        for batch in DataLoader(transitions, batch_size=config.batch_size, shuffle=True):

            # === Policy Loss (Actor) ===
            mol_features = [smiles_to_graph_data(t["smiles_from"]) for t in batch]
            op_features = torch.stack([encode_operation(t["operation"]) for t in batch])

            policy_scores = policy_net(mol_features, op_features)  # (B,)
            action_probs = F.softmax(policy_scores, dim=0)

            # 优势函数: A(s,a) = R(s,a) - V(s)
            rewards = torch.tensor([t["reward"] for t in batch])
            values = heuristic_net(mol_features, accumulated, target)
            advantages = rewards - values.detach()

            # PPO clip loss
            ratio = action_probs / (old_action_probs + 1e-8)
            clipped = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

            # === Value Loss (Critic / Heuristic) ===
            actual_remaining = torch.tensor([t["actual_remaining_cost"] for t in batch])
            predicted_h = heuristic_net(mol_features, accumulated, target)
            value_loss = F.mse_loss(predicted_h, actual_remaining)

            # === Conservative Penalty (保证 admissible) ===
            conservative_loss = F.relu(predicted_h - actual_remaining).pow(2).mean()

            # === Entropy Bonus (鼓励探索) ===
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()

            # === Total Loss ===
            loss = (
                policy_loss
                + 0.5 * value_loss
                + 0.1 * conservative_loss
                - 0.01 * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(heuristic_net.parameters()),
                max_norm=1.0
            )
            optimizer.step()
```

### 6.5 在线微调（Feedback Loop）

A* 搜索产生新数据后，进一步 fine-tune RL 模型：

```python
def online_finetune_loop(config):
    """
    A* 搜索 → 收集数据 → Fine-tune → 循环
    """
    policy_net, heuristic_net = load_trained_models()

    for iteration in range(config.num_iterations):
        # 1. 用当前 RL 模型运行 A* 搜索
        search_results = run_astar_search(
            policy_fn=policy_net,
            heuristic_fn=heuristic_net,
            predictor=v0_predictor,
        )

        # 2. 收集新的 transitions
        new_transitions = collect_training_data_from_search_results(search_results)

        # 3. Fine-tune（小学习率）
        finetune(policy_net, heuristic_net, new_transitions, lr=config.finetune_lr)

        # 4. 评估搜索质量提升
        metrics = evaluate_search_quality(search_results)
        log_metrics(metrics, iteration)
```

---

## 7. 分阶段实施路线

| 阶段 | 内容 | 关键产出 | 预计周期 |
|------|------|---------|---------|
| **Phase 1** | 离线数据收集 + RL 模型实现 | 训练数据集、PolicyNet、HeuristicNet | 1-2 周 |
| **Phase 2** | A* 搜索实现 + 集成 | `generate_expansion_tree_astar()`、CLI 入口 | 1 周 |
| **Phase 3** | 离线评估 + 超参调优 | 与 BFS/MCTS 的对比实验报告 | 1 周 |
| **Phase 4** | 在线微调 + Feedback Loop | 在线微调脚本、收敛曲线 | 1 周 |
| **Phase 5** | 多目标扩展 (MO) | 多目标 reward、Pareto 前沿优化 | 1-2 周 |
| **Phase 6** | v0.3 链式集成 | HeuristicNet 作为 v0.3 中间步 prior | 1 周 |

---

## 8. A* 相比现有搜索的优势

| 维度 | BFS (默认) | MCTS (可选) | A* + RL (本方案) |
|------|-----------|------------|-----------------|
| **搜索效率** | O(b^d) 全量扩展 | 依赖 rollout 次数 | 启发式引导，优先最有潜力路径 |
| **操作空间利用** | 全量生成后 top-50% 截断 | 全量生成后 PUCT 选择 | PolicyNet 直接筛选 top-K |
| **可解释性** | 无评分机制 | Q 值语义模糊 | h(s) = 剩余需要改善的量，语义清晰 |
| **收敛保证** | 不保证最优 | 不保证最优 | Admissible h 保证最优解 |
| **与 v0.3 兼容** | 不直接提供中间步引导 | 不直接提供 | HeuristicNet 可作为 step-level prior |
| **计算开销** | 低（无额外模型推理） | 高（大量 rollout） | 中等（每次扩展需推理 PolicyNet + HeuristicNet） |

---

## 9. 关键风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|---------|
| h(s) 不满足 admissible | A* 退化为贪心搜索，不保证最优 | 训练加入 conservative penalty；h(s) = max(0, target - V(s)) |
| 操作空间仍然太大 | PolicyNet 推理开销高 | 两级过滤：先选操作类型，再选具体参数；操作类型维度仅 17 |
| Reward 稀疏 | RL 训练困难 | v0 模型预测值作为 dense reward shaping；DFT 真实值作为最终验证 |
| 过拟合到搜索分布 | 泛化性差 | Dropout + 数据增强（随机扰动分子）；独立验证集评估 |
| 搜索结果数据偏差 | 离线训练数据分布与在线不同 | DAgger / importance sampling 纠偏；在线微调闭环 |
| 内存占用大 | A* open_set 存储所有候选节点 | 限制 open_set 大小；周期性清理已探索分支 |

---

## 10. 文件结构规划

```
mol_evo/
├── core/
│   ├── models/
│   │   └── astar_rl/
│   │       ├── __init__.py
│   │       ├── policy_network.py          # OperationPolicyNetwork
│   │       ├── heuristic_network.py        # HeuristicNetwork
│   │       └── reward.py                   # Reward 函数定义
│   └── search/
│       └── molecular_evolution_expansion.py  # 新增 generate_expansion_tree_astar()
├── train/
│   └── train_astar_rl.py                   # RL 训练脚本
├── scripts/
│   └── batch_optimizer_astar.py            # 带 A* 搜索的批量优化脚本
└── docs/
    └── ideas/
        └── astar-rl-molecular-expansion.md  # 本文档
```

---

## 11. 评估指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **搜索效率** | 找到目标解所需扩展的节点数 | 相比 BFS 减少 50%+ |
| **解质量** | 最终分子的属性变化量 | 不低于 BFS/MCTS 最优解 |
| **多样性** | Pareto 前沿上解的分布均匀性 | 高于 BFS 的 top-50% 截断 |
| **推理延迟** | 单次 A* 搜索的总耗时 | 可接受（< 原 BFS 的 80%） |
| **h(s) 准确性** | h(s) 与 h*(s) 的 MSE | 越低越好，同时满足 admissible |
| **π(a\|s) 覆盖率** | top-K 候选中包含最优操作的比例 | > 90% |
