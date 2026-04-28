# MOL-EVO MCTS 架构分析报告

## 1. 系统概述

该项目实现了一个分子进化树优化系统，包含三种搜索模式：
- **BFS** (广度优先搜索)：基础遍历模式
- **MCTS** (蒙特卡洛树搜索)：主要研究模式，支持 PUCT 算法
- **A* RL Demo**：带策略网络的高级搜索模式

本文档重点分析 **MCTS 实现**及其参数流向。

---

## 2. 批量优化器入口 → 参数流动

### 2.1 文件位置
```
scripts/optimization/batch_optimizer.py (第 1-679 行)
```

### 2.2 关键参数流向

#### 命令行参数定义 (第 139-167 行)
```python
parser.add_argument('--num-simulations', type=int, default=200,
                    help='MCTS 模拟轮数 (仅 mcts 模式)')
parser.add_argument('--exploration-weight', type=float, default=1.4,
                    help='MCTS PUCT 探索系数 (仅 mcts 模式)')
parser.add_argument('--mcts-prior-mode', type=str, choices=['softmax', 'uniform'], 
                    default='softmax', help='MCTS prior 构造方式')
parser.add_argument('--mcts-value-mode', type=str, choices=['accumulated', 'zero', 'step'], 
                    default='accumulated', help='MCTS 叶节点价值')
parser.add_argument('--mcts-expansion-mode', type=str, choices=['topk', 'random_topk', 'full'], 
                    default='topk', help='MCTS 扩展策略')
parser.add_argument('--mcts-random-seed', type=int, default=None,
                    help='MCTS 随机种子')
```

**关键参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_simulations` | 200 | MCTS 总模拟轮数（**步数预算**） |
| `exploration_weight` | 1.4 | PUCT 探索系数 c（控制 exploitation vs exploration） |
| `mcts_prior_mode` | softmax | Prior 构造方式（softmax 使用 OFO 打分） |
| `mcts_value_mode` | accumulated | 叶节点价值计算方式 |
| `mcts_expansion_mode` | topk | 候选选择策略 |
| `mcts_random_seed` | None | 随机种子（random_topk 模式时使用） |

#### 参数传递链路 (第 260-280 行)
```python
# batch_optimizer.py: run_evolution_optimizer() → optimizer.optimize_evolution_tree()
optimized_tree = optimizer.optimize_evolution_tree(
    smiles,
    args.max_depth,                    # 最大深度
    args.max_branching,                # 每层最多展开数
    args.direction,                    # 优化方向 (increase/decrease)
    pruning_patience=args.pruning_patience,
    logp_range=(args.logp_min, args.logp_max),
    logp_patience=args.logp_patience,
    search_mode=args.search_mode,      # 'bfs' | 'mcts' | 'astar_demo'
    num_simulations=args.num_simulations,           # ← MCTS 参数
    exploration_weight=args.exploration_weight,    # ← MCTS 参数
    mcts_prior_mode=args.mcts_prior_mode,         # ← MCTS 参数
    mcts_value_mode=args.mcts_value_mode,         # ← MCTS 参数
    mcts_expansion_mode=args.mcts_expansion_mode, # ← MCTS 参数
    mcts_random_seed=args.mcts_random_seed,       # ← MCTS 参数
)
```

---

## 3. MCTS 核心实现

### 3.1 文件位置
```
core/molecular_evolution_expansion.py
├── generate_expansion_tree_mcts()           [第 1126-1532 行]
├── _MCTSNode 类定义                         [第 1184-1223 行]
├── _expand_node() 函数                      [第 1229-1368 行]
├── _select_child() 函数                     [第 1369-1380 行]
├── _evaluate_leaf() 函数                    [第 1382-1396 行]
└── _backpropagate() 函数                    [第 1398-1404 行]
```

### 3.2 MCTS 算法流程

```
MCTS 主循环 (第 1422-1456 行):
  for sim_idx in range(num_simulations):      # ← 主模拟循环
    
    1. Selection (第 1429-1431):
       node = root
       while node.is_expanded and node.children and not node.is_terminal:
           node = _select_child(node)  # 使用 PUCT 选择最佳子节点
    
    2. Expansion (第 1434-1435):
       if not node.is_terminal and not node.is_expanded:
           _expand_node(node)  # 首次展开节点
    
    3. Evaluation & Backpropagation (第 1438-1449):
       if node.children:
           eval_node = _select_child(node)
           value = _evaluate_leaf(eval_node)
           _backpropagate(eval_node, value)
       else:
           value = _evaluate_leaf(node)
           _backpropagate(node, value)
```

**关键数据结构**：`_MCTSNode` 类（第 1184-1223）
```python
class _MCTSNode:
    __slots__ = (
        'smiles', 'depth', 'parent', 'children',
        'visit_count',          # 访问计数 (关键: 统计步数)
        'total_value',          # 累计价值
        'prior',                # Prior 概率
        'property_value', 'accumulated_change', 'logp',
        'logp_in_range', 'operation', 'operation_params',
        'is_expanded',          # 是否已展开过
        'is_terminal',          # 是否终止节点
    )
    
    def visit_count(self):      # 访问计数 (步数统计)
    def total_value(self):      # 累计价值
    def q_value(self):          # 平均价值 = total_value / visit_count
    def ucb_score(self, c):     # PUCT 得分 = q_value + c * prior * sqrt(parent.visit) / (1 + visit)
```

**步数计数方式**：
- 每次模拟中，从选择→评估→回传，每个被访问的节点的 `visit_count` 增加 1
- 总模拟轮数 = `num_simulations`
- 实际模拟完成数 = `root.visit_count`

---

## 4. MCTS 节点展开与预测

### 4.1 节点展开 (_expand_node 函数，第 1229-1368)

```python
def _expand_node(node: _MCTSNode):
    """首次展开一个节点：生成候选、批量预测、创建子节点"""
    
    if node.is_expanded or node.is_terminal:
        return
    
    node.is_expanded = True
    
    # 1. 终止条件检查
    if node.depth >= max_depth:
        node.is_terminal = True
        return
    
    # 2. Pruning patience 检查 (第 1240-1253)
    if pruning_patience > 0 and node.parent is not None:
        # 检查从根到当前节点的路径上连续无改善的代数
        stagnation = 0
        cur = node
        while cur is not None and cur.parent is not None:
            change = cur.accumulated_change - cur.parent.accumulated_change
            improved = (change > 0) if optimization_direction == 'increase' else (change < 0)
            if improved:
                break
            stagnation += 1
            cur = cur.parent
        if stagnation >= pruning_patience:
            node.is_terminal = True
            return
    
    # 3. LogP patience 检查 (第 1255-1266)
    if logp_patience > 0 and node.parent is not None:
        logp_out = 0
        cur = node
        while cur is not None:
            if cur.logp_in_range:
                break
            logp_out += 1
            cur = cur.parent
        if logp_out >= logp_patience:
            node.is_terminal = True
            return
    
    # 4. 生成候选操作 (第 1268-1304)
    # - 使用缓存避免重复预测
    # - 调用 predictor.predict_batch() 进行批量预测
    candidates = []  # List[(operation, new_smiles, property_change)]
    
    # 5. 候选选择 (第 1310-1325)
    # 根据 expansion_mode 选择候选：
    if expansion_mode == 'topk':
        selected_cands = sorted_cands[:max_branching]
    elif expansion_mode == 'random_topk':
        selected_cands = rng.sample(candidates, max_branching)
    else:  # full
        selected_cands = sorted_cands
    
    # 6. 计算 Prior (第 1326-1337)
    if prior_mode == 'uniform':
        priors = [1.0 / len(selected_cands)] * len(selected_cands)
    else:  # softmax (使用 OFO 预测分数)
        raw_scores = [c[2] for c in selected_cands]
        if optimization_direction == 'decrease':
            raw_scores = [-s for s in raw_scores]
        priors = softmax(raw_scores)
    
    # 7. 创建子节点 (第 1339-1364)
    for (op, new_smi, prop_change), prior in zip(selected_cands, priors):
        new_acc = node.accumulated_change + prop_change
        new_val = node.property_value + prop_change
        child = _MCTSNode(
            smiles=new_smi,
            depth=node.depth + 1,
            parent=node,
            prior=prior,
            property_value=new_val,
            accumulated_change=new_acc,
            operation=op["type"],
            operation_params=op.get("params", {}),
        )
        node.children.append(child)
```

**关键特性**：
1. **缓存机制**（第 1269-1304）：相同分子不重复预测
2. **批量预测**：调用 `predictor.predict_batch()`
3. **动态分支**：创建的子节点数 ≤ `max_branching`
4. **Prior 计算**：
   - `softmax`: `P(a) = exp(score_a) / Σ exp(scores)` （推荐）
   - `uniform`: `P(a) = 1 / num_actions`

---

## 5. PUCT 选择策略

### 5.1 UCB Score 计算 (第 1218-1223)

```python
def ucb_score(self, c: float):
    """PUCT (Predictor + UCT) 得分"""
    if self.parent is None:
        return 0.0
    
    # PUCT 核心公式：
    exploitation = self.q_value()  # 利用：平均价值
    exploration = c * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
    
    return exploitation + exploration
```

其中：
- `q_value()` = `total_value / visit_count`
- `c` = `exploration_weight` (默认 1.4)
- `prior` = 由 MCTS prior_mode 计算的先验概率
- `visit_count` = 当前节点被访问的次数

### 5.2 选择函数 (第 1369-1380)

```python
def _select_child(node: _MCTSNode) -> Optional[_MCTSNode]:
    """PUCT 选择最佳子节点，跳过异常分数并在必要时回退"""
    best, best_score = None, -float('inf')
    fallback = node.children[0] if node.children else None
    
    for ch in node.children:
        sc = ch.ucb_score(exploration_weight)
        if not math.isfinite(sc):
            continue
        if sc > best_score:
            best_score = sc
            best = ch
    
    return best if best is not None else fallback
```

---

## 6. 叶节点价值评估

### 6.1 评估函数 (第 1382-1396)

```python
def _evaluate_leaf(node: _MCTSNode) -> float:
    """叶节点估值：支持累计值 / 零值 / 单步值三种模式"""
    
    if value_mode == 'zero':
        # 恒为 0（目标是最大化访问频率）
        val = 0.0
    elif value_mode == 'step':
        # 仅当前步的增益
        if node.parent is None:
            val = 0.0
        else:
            val = node.accumulated_change - node.parent.accumulated_change
    else:  # accumulated (默认)
        # 从根到当前节点的累计增益
        val = node.accumulated_change
    
    # 根据优化方向调整符号
    if optimization_direction == 'decrease':
        val = -val  # 最小化问题转为最大化
    
    return val
```

**三种模式对比**：
| 模式 | 价值计算 | 适用场景 |
|------|---------|---------|
| `accumulated` | 从根到当前节点的累计增益 | **默认推荐**，评估全路径质量 |
| `zero` | 恒为 0 | 纯探索模式，平衡访问 |
| `step` | 仅当前边的增益 | 近视优化，短期收益 |

---

## 7. 反向传播与访问计数

### 7.1 回传函数 (第 1398-1404)

```python
def _backpropagate(node: _MCTSNode, value: float):
    """从叶节点回传价值到根节点"""
    cur = node
    while cur is not None:
        cur.visit_count += 1          # ← 关键：每次回传增加访问计数
        cur.total_value += value      # 累加价值
        cur = cur.parent
```

**步数统计方式**：
- 每个模拟周期，从选择→评估→回传，路径上所有节点的 `visit_count` 都增加 1
- 根节点 `visit_count` = 完成的模拟总轮数
- 实际模拟数 = `min(num_simulations, root.visit_count)`（中断信号可能导致提前结束）

---

## 8. 主循环与步数控制

### 8.1 主循环 (第 1418-1456)

```python
print(f"[MCTS] 开始搜索: simulations={num_simulations}, c={exploration_weight}, "
      f"max_depth={max_depth}, max_branching={max_branching}, ...")

for sim_idx in range(num_simulations):  # ← 外层循环：模拟总轮数
    
    # 检查中断信号
    if getattr(self, 'interrupted', False):
        print(f"[MCTS] 收到中断信号，在第 {sim_idx+1} 轮停止")
        break
    
    # 1. Selection + 2. Expansion
    node = root
    while node.is_expanded and node.children and not node.is_terminal:
        node = _select_child(node)
    
    if not node.is_terminal and not node.is_expanded:
        _expand_node(node)
    
    # 3. Evaluation + 4. Backpropagation
    if node.children:
        eval_node = _select_child(node)
        if eval_node is not None:
            value = _evaluate_leaf(eval_node)
            _backpropagate(eval_node, value)
        else:
            value = _evaluate_leaf(node)
            _backpropagate(node, value)
    else:
        value = _evaluate_leaf(node)
        _backpropagate(node, value)
    
    # 定期日志 (每 20% 进度)
    if (sim_idx + 1) % max(1, num_simulations // 5) == 0:
        print(f"[MCTS] simulation {sim_idx+1}/{num_simulations}, "
              f"root visits={root.visit_count}, "
              f"unique states cached={len(expansion_cache)}")

# 输出最终统计
actual_simulations = root.visit_count
print(f"[MCTS] 搜索完成: 树节点={total_nodes}, 边={total_edges}, "
      f"唯一状态={len(expansion_cache)}")
```

**模拟轮数流向**：
```
num_simulations (命令行参数)
    ↓
optimize_evolution_tree() 接收
    ↓
generate_expansion_tree_mcts() 接收
    ↓
主循环 for sim_idx in range(num_simulations):
    ↓
每个模拟执行：Select → Expand → Evaluate → Backpropagate
    ↓
root.visit_count 累加
    ↓
actual_simulations = root.visit_count (记录到 mcts_stats)
```

---

## 9. 树转换与输出

### 9.1 树转换 (第 1457-1532)

```python
# 将内部 MCTS 树转换为兼容的 nodes/edges JSON 结构
expansion_tree = {
    "initial_smiles": self.initial_smiles,
    "max_depth": max_depth,
    "max_branching": max_branching,
    "search_mode": "mcts",
    "mcts_stats": {
        "num_simulations": num_simulations,           # 期望模拟数
        "actual_simulations": actual_simulations,     # 实际完成数
        "exploration_weight": exploration_weight,
        "prior_mode": prior_mode,
        "value_mode": value_mode,
        "expansion_mode": expansion_mode,
        "random_seed": random_seed,
        "unique_states_expanded": len(expansion_cache),  # 展开过的唯一分子数
        "root_visits": root.visit_count,
    },
    "nodes": {},  # 被访问过的节点 (visit_count > 0)
    "edges": [],
}

# 只输出被访问过的子节点
def _traverse(mcts_node: _MCTSNode, parent_tree_id: Optional[str]):
    # ... 节点转换 ...
    visited_children = [ch for ch in mcts_node.children if ch.visit_count > 0]
    visited_children.sort(key=lambda c: c.visit_count, reverse=True)
    for child in visited_children:
        _traverse(child, nid)
```

**输出 JSON 中的 MCTS 字段**：
```json
{
  "nodes": {
    "0": {
      "id": "0",
      "smiles": "CC",
      "depth": 0,
      "mcts_visits": 200,          // ← visit_count
      "mcts_prior": 1.0,           // ← prior
      "mcts_q_value": 0.123456,    // ← q_value()
      "property_value": -5.2,
      "accumulated_change": 0.0
    },
    "1": {
      "id": "1",
      "smiles": "CCC",
      "depth": 1,
      "mcts_visits": 45,           // ← 该节点被访问 45 次
      "mcts_prior": 0.385,
      "mcts_q_value": 0.089,
      "accumulated_change": -0.3
    }
  },
  "mcts_stats": {
    "num_simulations": 200,
    "actual_simulations": 200,
    "unique_states_expanded": 127,   // ← 展开过 127 个不同分子
    "root_visits": 200
  }
}
```

---

## 10. 参数配置最佳实践

### 10.1 MCTS 参数含义与调优

| 参数 | 范围 | 默认 | 说明 | 调优建议 |
|------|------|------|------|---------|
| `num_simulations` | 10-10000 | 200 | **步数预算**，每个模拟一步 | 越大越精确，但计算时间 O(n) |
| `exploration_weight` | 0.1-3.0 | 1.4 | PUCT 探索系数 c | 越大越探索；1.4 为 AlphaGo 推荐值 |
| `max_depth` | 1-10 | 2 | 最大搜索深度 | 与分子复杂度匹配 |
| `max_branching` | 1-20 | 8 | 每层最多展开子节点 | 小值：贪心；大值：多样化 |
| `prior_mode` | softmax\|uniform | softmax | Prior 构造方式 | softmax 利用 OFO 打分信息 |
| `value_mode` | accumulated\|zero\|step | accumulated | 叶节点价值评估 | accumulated 推荐用于全路径优化 |
| `expansion_mode` | topk\|random_topk\|full | topk | 候选选择策略 | topk 贪心；random_topk 探索性强 |

### 10.2 预期的树规模

```
设置：
  num_simulations = 200
  max_depth = 2
  max_branching = 8

预期树规模 (在无剪枝情况下)：
  - 根节点：1
  - 第一层：最多 8 个
  - 第二层：最多 8 × 8 = 64 个
  - 总理论最多：1 + 8 + 64 = 73 个节点
  
实际情况 (因剪枝、终止、重复等)：
  - 通常远小于理论上限
  - 输出中 unique_states_expanded ≈ 展开过的唯一分子数
  - 输出中节点数 ≈ 被访问过的节点数 (visit_count > 0)
```

---

## 11. 数据流全景图

```
batch_optimizer.py
  ├─ 命令行参数解析
  │   ├─ --num-simulations (200)
  │   ├─ --exploration-weight (1.4)
  │   ├─ --mcts-prior-mode (softmax)
  │   ├─ --mcts-value-mode (accumulated)
  │   ├─ --mcts-expansion-mode (topk)
  │   └─ --mcts-random-seed (None)
  │
  └─ batch_process()
      └─ run_evolution_optimizer()
          └─ optimizer.optimize_evolution_tree()
              │
              ├─ [evolution_optimizer.py]
              │   └─ 设置优化器参数
              │
              └─ evolver.generate_expansion_tree_mcts()
                  │
                  ├─ [molecular_evolution_expansion.py]
                  │   └─ 创建根节点 (root)
                  │
                  ├─ 主循环: for sim_idx in range(num_simulations)
                  │   │
                  │   ├─ Selection
                  │   │   └─ _select_child() 使用 PUCT 选择
                  │   │
                  │   ├─ Expansion (首次展开)
                  │   │   └─ _expand_node()
                  │   │       ├─ 检查终止条件
                  │   │       ├─ 生成候选操作
                  │   │       ├─ predictor.predict_batch() 批量预测
                  │   │       └─ 创建子节点 (prior, visit_count=0)
                  │   │
                  │   ├─ Evaluation
                  │   │   └─ _evaluate_leaf() 计算叶节点价值
                  │   │       ├─ value_mode=accumulated: 累计增益
                  │   │       ├─ value_mode=zero: 0
                  │   │       └─ value_mode=step: 单步增益
                  │   │
                  │   └─ Backpropagation
                  │       └─ _backpropagate()
                  │           └─ 沿路径向上：visit_count++, total_value+=value
                  │
                  ├─ 树转换
                  │   └─ _traverse() 递归转换为 JSON nodes/edges
                  │       └─ 只输出 visit_count > 0 的节点
                  │
                  └─ 返回 expansion_tree JSON
                      ├─ nodes: {node_id → node_data}
                      ├─ edges: [parent_id → child_id]
                      └─ mcts_stats: {
                           num_simulations,
                           actual_simulations,
                           unique_states_expanded,
                           root_visits,
                           ...
                         }

save_optimized_tree()
  └─ 保存 JSON 到磁盘
```

---

## 12. 关键概念总结

### 12.1 步数与模拟的对应关系
- **1 个模拟周期** = 1 次 Select → Expand → Evaluate → Backpropagate
- **`num_simulations`** = 配置的总模拟轮数
- **`actual_simulations`** = 实际完成的模拟数（可能因中断而小于 `num_simulations`）
- **`root.visit_count`** = 完成的模拟总数 = `actual_simulations`
- **树中节点的 `visit_count`** = 该节点被访问的次数（通常 ≤ `root.visit_count`）

### 12.2 PUCT 公式的核心参数
```
UCB_score = q_value + c × prior × √(N_parent) / (1 + N_node)

其中：
  - q_value      = 平均价值（利用项）
  - c            = exploration_weight (默认 1.4)
  - prior        = 先验概率（由 prior_mode 计算）
  - N_parent     = 父节点访问次数
  - N_node       = 当前节点访问次数
```

### 12.3 树生长的三个制约因素
1. **深度限制**：`node.depth >= max_depth` → 终止
2. **剪枝耐心**：连续 `pruning_patience` 代无改善 → 终止
3. **LogP 范围**：连续 `logp_patience` 代超出范围 → 终止

### 12.4 Prior 的作用
- **softmax 模式**（推荐）：
  - 利用 OFO 模型的打分结果
  - 更新概率：`P(a) = exp(score_a) / Σ exp(scores)`
  - 作用：好分数的操作获得高 prior，更容易被选中

- **uniform 模式**：
  - 所有操作等概率：`P(a) = 1 / num_actions`
  - 作用：平衡探索，无偏见

### 12.5 输出结果的解释

查看 JSON 输出 `mcts_stats` 字段：
```json
"mcts_stats": {
  "num_simulations": 200,              // ← 请求的模拟数
  "actual_simulations": 198,           // ← 实际完成数 (可能因中断而 < 200)
  "unique_states_expanded": 127,       // ← 展开过多少个不同分子
  "root_visits": 198,                  // ← 同 actual_simulations
  "exploration_weight": 1.4,
  "prior_mode": "softmax",
  "value_mode": "accumulated",
  "expansion_mode": "topk"
}
```

---

## 13. 故障排查

### 问题1：树太小或太大
**症状**：`unique_states_expanded` 或总节点数远小于/大于预期

**原因分析**：
- **太小**：max_branching 太小 / 剪枝条件过严 / 分子无效操作多
- **太大**：max_depth 太大 / 剪枝条件过松 / expansion_mode=full

**解决**：
```bash
# 增大树规模
python batch_optimizer.py \
  --search-mode mcts \
  --num-simulations 500 \
  --max-branching 15 \
  --mcts-expansion-mode full

# 减小树规模
python batch_optimizer.py \
  --search-mode mcts \
  --num-simulations 100 \
  --max-branching 4 \
  --pruning-patience 2
```

### 问题2：优化质量差
**症状**：topK 结果的属性值改善不显著

**原因分析**：
- exploration_weight 太小 → 过度利用已探索区域
- prior_mode=uniform → 丢弃 OFO 打分信息
- value_mode=step → 优化器短视

**解决**：
```bash
# 增加探索
python batch_optimizer.py \
  --search-mode mcts \
  --exploration-weight 2.0 \
  --mcts-prior-mode softmax

# 改用累计价值评估
python batch_optimizer.py \
  --search-mode mcts \
  --mcts-value-mode accumulated
```

### 问题3：中断后无法续传
**症状**：MCTS 运行中止，但 actual_simulations < num_simulations

**原因分析**：
- 收到 SIGINT 信号（Ctrl+C）
- batch_optimizer 的中断处理逻辑捕获信号

**解决**：
- 中断时已完成的分子结果自动保存
- 下次运行相同输出目录会自动跳过已完成的分子
- 查看日志确认断点续传是否启动

---

## 14. 配置示例

### 14.1 快速探索（小规模）
```bash
python scripts/optimization/batch_optimizer.py \
  --input-csv dataset/eval-data/qm9_test.csv \
  --output-dir output/quick_explore \
  --search-mode mcts \
  --num-simulations 50 \
  --max-depth 1 \
  --max-branching 4 \
  --mcts-expansion-mode topk
```

### 14.2 标准配置（平衡）
```bash
python scripts/optimization/batch_optimizer.py \
  --input-csv dataset/eval-data/qm9_test.csv \
  --output-dir output/standard \
  --search-mode mcts \
  --num-simulations 200 \
  --max-depth 2 \
  --max-branching 8 \
  --exploration-weight 1.4 \
  --mcts-prior-mode softmax \
  --mcts-value-mode accumulated
```

### 14.3 深度探索（大规模）
```bash
python scripts/optimization/batch_optimizer.py \
  --input-csv dataset/eval-data/qm9_test.csv \
  --output-dir output/deep_explore \
  --search-mode mcts \
  --num-simulations 500 \
  --max-depth 3 \
  --max-branching 12 \
  --exploration-weight 1.8 \
  --mcts-expansion-mode random_topk \
  --mcts-random-seed 42
```

---

## 15. 文件索引与行号

| 功能 | 文件 | 行号 |
|------|------|------|
| 命令行参数定义 | batch_optimizer.py | 139-167 |
| 参数传递链路 | batch_optimizer.py | 260-280 |
| MCTS 主函数 | molecular_evolution_expansion.py | 1126-1532 |
| _MCTSNode 类 | molecular_evolution_expansion.py | 1184-1223 |
| PUCT 选择 | molecular_evolution_expansion.py | 1369-1380 |
| 节点展开 | molecular_evolution_expansion.py | 1229-1368 |
| 价值评估 | molecular_evolution_expansion.py | 1382-1396 |
| 回传函数 | molecular_evolution_expansion.py | 1398-1404 |
| 主循环 | molecular_evolution_expansion.py | 1422-1456 |
| 树转换 | molecular_evolution_expansion.py | 1457-1532 |

---

## 16. 相关文件快速导航

```
mol-evo/
├── scripts/optimization/
│   ├── batch_optimizer.py              ← 入口脚本（推荐阅读）
│   ├── batch_optimizer_visnet.py       ← ViSNet 版本
│   └── batch_optimizer_ic50.py         ← IC50 版本
│
├── core/
│   ├── evolution_optimizer.py          ← 优化器主类
│   ├── molecular_evolution_expansion.py ← MCTS 实现 ⭐
│   │   ├── generate_expansion_tree()
│   │   ├── generate_expansion_tree_mcts()   ← MCTS 核心 ⭐
│   │   └── generate_expansion_tree_astar_demo()
│   │
│   ├── data/
│   │   ├── processing.py               ← 操作配置加载
│   │   └── rl_demo_processing.py       ← RL 数据编码
│   │
│   └── utils/
│       └── molecule.py                 ← 分子缓存工具
│
├── tests/
│   └── evo-algorithm/tests/
│       └── test_mcts_search_mode.py    ← MCTS 单元测试
│
└── docs/
    └── paper/
        ├── tables/mcts-hparam-sensitivity.md
        └── plans/mol-mcts-params.md    ← 历史超参数分析
```

