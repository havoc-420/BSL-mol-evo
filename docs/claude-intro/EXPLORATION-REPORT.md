# 代码库探索报告

**时间**：2026-04-28  
**项目**：`mol-evo` - 分子进化优化系统

---

## 1. 文件内容总结

### 1.1 batch_optimizer.py 概览

**文件路径**：`scripts/optimization/batch_optimizer.py`  
**行数**：679 行

**主要功能**：
- 批量读取分子 CSV 数据
- 逐个分子调用 `EvolutionTreeOptimizer`
- 执行 BFS/MCTS/A* 搜索生成优化树
- 保存结果、提取 TopK、生成日志

**核心函数**：
1. `main()` - 入口，管理整个批处理流程
2. `parse_args()` - 解析 80+ 个 CLI 参数
3. `batch_process()` - 批量循环处理每个分子
4. `run_evolution_optimizer()` - 单分子优化执行
5. `save_results()` - 保存汇总结果 JSON

**关键参数**（与 MCTS 相关）：
```python
--search-mode              # 'bfs' / 'mcts' / 'astar_demo'
--num-simulations          # MCTS 模拟轮数，默认 200
--exploration-weight       # MCTS PUCT 探索系数，默认 1.4
--mcts-prior-mode          # 先验：'softmax' 或 'uniform'
--mcts-value-mode          # 价值：'accumulated' / 'zero' / 'step'
--mcts-expansion-mode      # 扩展：'topk' / 'random_topk' / 'full'
--mcts-random-seed         # 随机种子，用于可复现
--max-depth                # 搜索深度，默认 2
--max-branching            # 每层候选数，默认 8
```

---

## 2. MCTS 实现文件

**文件路径**：`core/molecular_evolution_expansion.py`  
**MCTS 方法**：`generate_expansion_tree_mcts()`（从第 1126 行开始）  
**关键长度**：约 400+ 行

### 2.1 MCTS 关键概念

#### 数据结构：`_MCTSNode`

```python
class _MCTSNode:
    smiles: str                      # SMILES 字符串
    depth: int                       # 深度
    parent: _MCTSNode                # 父节点
    children: List[_MCTSNode]        # 子节点
    visit_count: int                 # ★ 被访问次数
    total_value: float               # ★ 累计价值
    prior: float                     # 先验概率 P(a)
    property_value: float            # 预测属性值
    accumulated_change: float        # 累计改变量
    is_expanded: bool                # 是否已展开
    is_terminal: bool                # 是否终止
    
    def ucb_score(c):
        # PUCT = Q(a) + c * P(a) * sqrt(N_parent) / (1 + N(a))
        return q_value() + c * prior * sqrt(parent.visit_count) / (1 + visit_count)
```

#### 主循环（第 1422-1455 行）

```python
for sim_idx in range(num_simulations):      # ★ 关键：num_simulations 决定循环次数
    # 1. Selection：从根沿 PUCT 向下
    node = root
    while node.is_expanded and node.children and not node.is_terminal:
        node = _select_child(node)
    
    # 2. Expansion：首次展开生成子节点
    if not node.is_terminal and not node.is_expanded:
        _expand_node(node)
    
    # 3. Evaluation + 4. Backpropagation
    if node.children:
        eval_node = _select_child(node)
        value = _evaluate_leaf(eval_node)
        _backpropagate(eval_node, value)
    else:
        value = _evaluate_leaf(node)
        _backpropagate(node, value)
    
    # 定期输出日志
    if (sim_idx + 1) % max(1, num_simulations // 5) == 0:
        print(f"[MCTS] simulation {sim_idx+1}/{num_simulations}, ...")
```

### 2.2 步数统计关键点

| 关键变量 | 含义 | 更新时机 |
|---------|------|---------|
| `sim_idx` | 主循环计数（0 ~ num_simulations-1） | 每次循环迭代 |
| `root.visit_count` | 根节点访问次数 | 每次 backprop 时递增 |
| `node.visit_count` | 该节点访问次数 | 每次 backprop 时递增 |
| `len(expansion_cache)` | 展开的唯一分子数 | 首次展开节点时 |

**关键结论**：
- 最终 `root.visit_count == num_simulations`（约等于）
- 实际执行的模拟轮数被记录在 `mcts_stats.actual_simulations = root.visit_count`

### 2.3 参数流向

```
CLI args
  ├─ --num-simulations (200)
  │   └─ batch_optimizer.parse_args()
  │   └─ batch_process(args)
  │   └─ run_evolution_optimizer(args)
  │   └─ EvolutionTreeOptimizer.optimize_evolution_tree(num_simulations=...)
  │   └─ MolecularEvolutionExpansion.generate_expansion_tree_mcts(num_simulations=...)
  │   └─ for sim_idx in range(num_simulations):
  │
  ├─ --exploration-weight (1.4)
  │   └─ PUCT 主循环中的 _select_child(exploration_weight)
  │   └─ ch.ucb_score(exploration_weight)
  │
  ├─ --mcts-prior-mode ('softmax')
  │   └─ _expand_node() 中的先验计算
  │
  ├─ --mcts-value-mode ('accumulated')
  │   └─ _evaluate_leaf() 中的价值计算
  │
  ├─ --mcts-expansion-mode ('topk')
  │   └─ _expand_node() 中的候选截断
  │
  └─ --mcts-random-seed (None)
      └─ random.Random(random_seed) 初始化
      └─ 用于 random_topk 采样可复现
```

---

## 3. 核心搜索流程

### 3.1 EvolutionTreeOptimizer.optimize_evolution_tree()

**位置**：`core/evolution_optimizer.py` 第 728 行

**职责**：
- 加载 OFO 预测模型
- 创建 `MolecularEvolutionExpansion` 实例
- 根据 `search_mode` 调用对应的搜索方法
- 返回标准格式的 `evolution_tree` 字典

**调用链**：
```python
def optimize_evolution_tree(self, initial_smiles, ..., search_mode='bfs', ...):
    evolver = MolecularEvolutionExpansion(initial_smiles, ...)
    
    if search_mode == 'mcts':
        evolution_tree = evolver.generate_expansion_tree_mcts(
            num_simulations=num_simulations,
            exploration_weight=exploration_weight,
            prior_mode=mcts_prior_mode,
            value_mode=mcts_value_mode,
            expansion_mode=mcts_expansion_mode,
            random_seed=mcts_random_seed,
        )
    
    return evolution_tree
```

### 3.2 MolecularEvolutionExpansion 的角色

**位置**：`core/molecular_evolution_expansion.py`

**主要方法**：
- `generate_expansion_tree()` - BFS 模式
- `generate_expansion_tree_mcts()` - MCTS 模式
- `generate_expansion_tree_astar_demo()` - A* 模式

**每个方法都共享同一输出格式**：
```python
expansion_tree = {
    "initial_smiles": str,
    "search_mode": str,
    "nodes": { ... },
    "edges": [ ... ],
    "mcts_stats": { ... }  # 仅 MCTS 模式
}
```

---

## 4. 现有架构文档

### 4.1 docs/core/ 中的文档

已存在的文档：
- `batch-optimizer-framework.md` - 批量优化框架概述（详细！）
- `ofo-model-framework.md` - OFO 模型架构
- `mo-direction-recommendation.md` - 多目标优化方向

### 4.2 新建的文档

位置：`docs/claude-intro/`

创建的新文档：
- `optimization-pipeline-architecture.md` - **本次新增**（详细的参数流向、MCTS 实现解析）

---

## 5. 关键发现

### 5.1 步数统计方式

**MCTS 的"步"有三种理解**：

1. **模拟轮数** = `num_simulations`
   - 由 CLI 参数 `--num-simulations` 控制
   - 对应主循环的迭代次数

2. **实际执行轮数** = `root.visit_count`
   - 在输出 JSON 的 `mcts_stats.actual_simulations` 记录
   - 通常等于 `num_simulations`（除非提前中断）

3. **展开的唯一分子** = `len(expansion_cache)`
   - 实际展开过的不同分子数量
   - 可能远小于 `num_simulations`

### 5.2 参数流向完整链路

从 CLI 参数到 MCTS 主循环的完整路径已追踪：

```
--num-simulations (200)
  └→ args.num_simulations
  └→ batch_process(args, ...)
  └→ run_evolution_optimizer(..., args, ...)
  └→ optimizer.optimize_evolution_tree(..., num_simulations=args.num_simulations, ...)
  └→ evolver.generate_expansion_tree_mcts(..., num_simulations=num_simulations, ...)
  └→ for sim_idx in range(num_simulations):  ← ★ 实际使用点
```

### 5.3 MCTS 内部数据结构

**_MCTSNode** 是核心：
- 记录树拓扑（parent/children）
- 记录 MCTS 统计（visit_count/total_value/prior）
- 记录属性信息（accumulated_change/property_value）
- 记录终止条件（is_expanded/is_terminal）

**backpropagation** 原理：
```python
def _backpropagate(node, value):
    cur = node
    while cur is not None:
        cur.visit_count += 1       # ★ 递增访问次数
        cur.total_value += value   # ★ 累加价值
        cur = cur.parent           # 一直向上到根
```

### 5.4 剪枝机制

MCTS 中有两层剪枝：

**1. 属性改善停滞剪枝**
```python
if pruning_patience > 0 and node.parent is not None:
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
        node.is_terminal = True  # ← 标记为终止
```

**2. logP 范围剪枝**
```python
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
```

---

## 6. 建议的改造方向

### 6.1 添加展开预算机制

**目标**：限制展开的唯一分子数

```python
expansion_budget = 100  # 参数化
expansion_count = [0]

def _expand_node(node):
    if expansion_count[0] >= expansion_budget:
        node.is_terminal = True
        return
    # ... 原逻辑 ...
    expansion_count[0] += 1
```

### 6.2 添加早停条件

**目标**：当 TopK 稳定时自动停止

```python
best_value = -float('inf')
no_improve_rounds = 0

for sim_idx in range(num_simulations):
    # ... MCTS 步骤 ...
    
    if (sim_idx + 1) % 10 == 0:
        current_best = max([node.accumulated_change for node in get_topk(root, 5)])
        if current_best > best_value:
            best_value = current_best
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
        
        if no_improve_rounds >= 5:
            break  # 早停
```

### 6.3 在 batch_optimizer.py 中暴露新参数

```python
parser.add_argument('--mcts-expansion-budget', type=int, default=None,
                    help='MCTS 展开预算（唯一分子数上限）')
parser.add_argument('--mcts-early-stop-patience', type=int, default=None,
                    help='MCTS 早停耐心值（连续无改进轮数）')
```

---

## 7. 文件位置速查

### 7.1 搜索入口

- **batch_optimizer.py**：`scripts/optimization/batch_optimizer.py`
- **EvolutionTreeOptimizer**：`core/evolution_optimizer.py` (第 75 行)
- **optimize_evolution_tree**：`core/evolution_optimizer.py` (第 728 行)

### 7.2 搜索实现

- **MolecularEvolutionExpansion**：`core/molecular_evolution_expansion.py` (第 1 行)
- **generate_expansion_tree_mcts**：`core/molecular_evolution_expansion.py` (第 1126 行)

### 7.3 MCTS 主循环

- **主循环**：`core/molecular_evolution_expansion.py` (第 1422 行)
- **_expand_node**：`core/molecular_evolution_expansion.py` (第 1229 行)
- **_select_child**：`core/molecular_evolution_expansion.py` (第 1369 行)
- **_evaluate_leaf**：`core/molecular_evolution_expansion.py` (第 1382 行)
- **_backpropagate**：`core/molecular_evolution_expansion.py` (第 1398 行)

### 7.4 输出转换

- **_traverse（树输出）**：`core/molecular_evolution_expansion.py` (第 1482 行)
- **save_optimized_tree**：`core/evolution_optimizer.py`

---

## 8. 快速查询表

### 关键参数含义

| 参数 | 范围 | 含义 |
|------|------|------|
| `num_simulations` | 50-1000+ | MCTS 主循环的总迭代次数 |
| `exploration_weight` | 0.5-2.0 | PUCT 公式中的探索系数 c |
| `max_depth` | 1-10 | 搜索树最大深度 |
| `max_branching` | 1-20 | 每个节点最多保留的子节点数 |
| `pruning_patience` | 0-10 | 连续无改善后剪枝的阈值 |
| `logp_patience` | 0-10 | logP 连续超出范围后剪枝的阈值 |

### 输出字段

| 字段 | 含义 |
|------|------|
| `mcts_stats.actual_simulations` | 实际执行的模拟轮数 |
| `mcts_stats.unique_states_expanded` | 展开的唯一分子数 |
| `node.mcts_visits` | 该节点的访问次数 |
| `node.mcts_prior` | 该节点的先验概率 |
| `node.mcts_q_value` | 该节点的平均价值 |

---

## 9. 总结

### 已完成的理解

✅ batch_optimizer.py 的完整执行流程  
✅ MCTS 实现的关键代码位置（~400 行）  
✅ 参数从 CLI 到 MCTS 主循环的完整流向  
✅ 步数统计的三种理解方式  
✅ 内部数据结构（_MCTSNode、expansion_tree）  
✅ 现有架构文档的发现与整理  

### 已生成的文档

📄 `/docs/claude-intro/optimization-pipeline-architecture.md`  
- 详细的 MCTS 实现解析
- 参数流向追踪
- 步数统计机制
- 改造建议

### 后续可扩展方向

🔄 添加展开预算限制  
🔄 添加早停机制  
🔄 添加时间预算  
🔄 添加动态深度调整  

---

**文档由 Claude 自动生成**  
**生成时间**：2026-04-28

