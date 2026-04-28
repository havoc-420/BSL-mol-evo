# 分子优化管道架构解析

> **文档目的**：深入理解 `batch_optimizer.py` 的执行流程、MCTS 实现细节、参数流向与步数统计机制。
> 本文档针对需要修改或优化搜索策略的开发者。

---

## 目录

1. [整体管道架构](#1-整体管道架构)
2. [batch_optimizer.py 完整内容分析](#2-batch_optimizerpy-完整内容分析)
3. [MCTS 实现详解](#3-mcts-实现详解)
4. [参数流向追踪](#4-参数流向追踪)
5. [步数统计与预算机制](#5-步数统计与预算机制)
6. [数据结构速查](#6-数据结构速查)
7. [改造建议](#7-改造建议)

---

## 1. 整体管道架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      batch_optimizer.py                         │
│  • 读入 CSV (smiles + property_value)                           │
│  • 循环处理每个分子                                              │
│  • 管理输出与日志                                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              EvolutionTreeOptimizer (core)                       │
│  • 加载 OFO 预测模型                                             │
│  • 提供 predict_batch() 接口                                     │
│  • 触发搜索（BFS / MCTS / A*）                                  │
│  • 保存结果 & 提取 TopK                                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    ┌─────┐     ┌──────┐     ┌────────┐
    │ BFS │     │ MCTS │     │ A* RL  │
    └──┬──┘     └──┬───┘     └────┬───┘
       │           │              │
       └───────────┼──────────────┘
                   ▼
     ┌──────────────────────────────┐
     │ MolecularEvolutionExpansion  │
     │  • 生成可执行操作            │
     │  • 应用分子编辑操作          │
     │  • 构建进化树 (nodes/edges)  │
     └──────────────────────────────┘
```

**核心流程**：
1. CSV 读入 → 初始分子列表
2. 对每个分子设置初始属性值
3. 调用搜索策略生成候选树
4. 使用 OFO 模型对每一步操作打分
5. 按评分与约束剪枝，产出优化树
6. 从树中提取 TopK 结果

---

## 2. batch_optimizer.py 完整内容分析

**文件位置**：`scripts/optimization/batch_optimizer.py`

### 2.1 核心函数调用链

```python
main()
  ├─ parse_args()                    # 解析CLI参数
  ├─ create_output_dir()             # 创建输出目录
  ├─ scan_completed_smiles()         # 断点续传检测
  ├─ read_csv_data()                 # CSV读入
  └─ batch_process()                 # 批量处理核心
      ├─ EvolutionTreeOptimizer()    # 初始化优化器（仅一次）
      ├─ for each molecule:
      │   └─ run_evolution_optimizer()
      │       └─ optimizer.optimize_evolution_tree()
      │           ├─ MolecularEvolutionExpansion
      │           ├─ generate_expansion_tree_mcts()    # 若search_mode=='mcts'
      │           └─ optimizer.save_optimized_tree()
      └─ save_results()               # 保存汇总结果
```

### 2.2 关键参数列表

#### 搜索相关
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--search-mode` | `'bfs'` | 搜索策略：`'bfs'` / `'mcts'` / `'astar_demo'` |
| `--num-simulations` | 200 | **MCTS 总模拟轮数**（仅 mcts 模式） |
| `--exploration-weight` | 1.4 | MCTS PUCT 探索系数 c |
| `--max-depth` | 2 | 搜索树最大深度 |
| `--max-branching` | 8 | 每层/每节点最多保留候选数 |

#### MCTS 专属参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mcts-prior-mode` | `'softmax'` | 先验构造：`'softmax'` / `'uniform'` |
| `--mcts-value-mode` | `'accumulated'` | 叶节点价值：`'accumulated'` / `'zero'` / `'step'` |
| `--mcts-expansion-mode` | `'topk'` | 扩展策略：`'topk'` / `'random_topk'` / `'full'` |
| `--mcts-random-seed` | None | 随机种子（用于 `random_topk` 的可复现） |

#### 优化目标与约束
| 参数 | 说明 |
|------|------|
| `--target-property` | 优化目标属性（如 `'lumo'`） |
| `--direction` | 优化方向：`'increase'` / `'decrease'` |
| `--pruning-patience` | 连续无改善剪枝阈值 |
| `--logp-min`, `--logp-max` | logP 允许范围 |
| `--logp-patience` | logP 连续越界剪枝阈值 |

#### 数据与I/O
| 参数 | 说明 |
|------|------|
| `--input-csv` | 输入分子CSV路径 |
| `--output-dir` | 输出目录（默认创建时间戳目录） |
| `--start-index`, `--end-index` | CSV 切片范围 |

### 2.3 主函数 `main()` 的工作流

```python
def main():
    args = parse_args()
    output_dir = create_output_dir(args.output_dir)
    
    # 断点续传：检查已完成分子
    completed_smiles = scan_completed_smiles(output_dir)
    existing_results = load_existing_results(output_dir)
    
    # 读入 CSV，跳过已完成分子
    data_list = read_csv_data(
        args.input_csv,
        start_idx=args.start_index,
        end_idx=args.end_index,
        target_property=args.target_property,
        completed_smiles=completed_smiles
    )
    
    # 批量处理
    results_dict = batch_process(data_list, args, output_dir, existing_results)
    
    # 保存汇总结果
    save_results(results_dict, args.output_json, output_dir)
```

### 2.4 批量处理 `batch_process()`

```python
def batch_process(data_list, args, output_dir, existing_results=None):
    results_dict = {}
    
    # 仅初始化一次优化器（模型共用）
    optimizer = EvolutionTreeOptimizer(
        args.model_path,
        args.model_dir,
        args.config_file,
        None,
        args.target_property,
        None,  # 每个分子单独设置
        args.optimization_mode
    )
    
    for i, data in enumerate(data_list):
        smiles = data['smiles']
        property_value = data['property_value']
        
        # 注入当前分子的初始属性值
        optimizer.initial_property_value = property_value
        
        # 运行优化
        result = run_evolution_optimizer(
            optimizer,
            smiles,
            property_value,
            args,
            output_dir
        )
        
        # 存储结果
        results_dict[smiles] = {
            'original_data': data['original_row'],
            'optimization_result': result
        }
        
        # 每5个分子或结束时保存一次
        if (i+1) % 5 == 0 or (i+1) == len(data_list):
            save_results(results_dict, args.output_json, output_dir)
    
    return results_dict
```

### 2.5 单分子执行 `run_evolution_optimizer()`

```python
def run_evolution_optimizer(optimizer, smiles, property_value, args, output_dir):
    try:
        # 调用核心搜索
        optimized_tree = optimizer.optimize_evolution_tree(
            smiles,
            max_depth=args.max_depth,
            max_branching=args.max_branching,
            direction=args.direction,
            search_mode=args.search_mode,
            num_simulations=args.num_simulations,
            exploration_weight=args.exploration_weight,
            mcts_prior_mode=args.mcts_prior_mode,
            mcts_value_mode=args.mcts_value_mode,
            mcts_expansion_mode=args.mcts_expansion_mode,
            mcts_random_seed=args.mcts_random_seed,
        )
        
        # 保存树 JSON
        output_json = os.path.join(output_dir, f"{smiles[:20]}_{run_id}.json")
        optimizer.save_optimized_tree(optimized_tree, output_json, output_dir)
        
        # 提取 TopK 并保存 CSV
        topK_results = optimizer.get_topK_results(optimized_tree, args.topK)
        topk_csv = output_json.replace('.json', '_topK.csv')
        optimizer.save_topK_results_to_csv(topK_results, topk_csv, output_dir)
        
        return {
            'status': 'success',
            'smiles': smiles,
            'optimized_result': optimized_tree,
            'topk_results': topK_results,
            'runtime': end_time - start_time
        }
    except Exception as e:
        return {
            'status': 'error',
            'smiles': smiles,
            'error': str(e)
        }
```

---

## 3. MCTS 实现详解

**文件位置**：`core/molecular_evolution_expansion.py`

### 3.1 MCTS 主函数签名

```python
def generate_expansion_tree_mcts(
    self,
    max_depth: int = 5,
    max_branching: int = 8,
    predictor=None,
    optimization_direction: str = 'increase',
    pruning_patience: int = 3,
    initial_property_value=None,
    optimization_mode=None,
    logp_range=(0, 5),
    logp_patience: int = 3,
    # --- MCTS 专属参数 ---
    num_simulations: int = 200,           # ★ 关键：总模拟轮数
    exploration_weight: float = 1.4,      # PUCT 探索系数
    prior_mode: str = 'softmax',          # prior 构造方式
    value_mode: str = 'accumulated',      # 叶节点价值模式
    expansion_mode: str = 'topk',         # 扩展策略
    random_seed: Optional[int] = None,    # 随机种子
) -> Dict:
    """使用 MCTS + PUCT 生成分子进化树"""
```

### 3.2 内部数据结构：`_MCTSNode`

```python
class _MCTSNode:
    __slots__ = (
        'smiles',                 # SMILES 字符串
        'depth',                  # 当前深度（根 = 0）
        'parent',                 # 父节点引用
        'children',               # 子节点列表
        'visit_count',            # ★ 被访问次数
        'total_value',            # ★ 累计价值
        'prior',                  # 先验概率（从 OFO 预测打分计算）
        'property_value',         # 当前节点预测属性值
        'accumulated_change',     # 从根到当前累计改变量
        'logp',                   # logP 值
        'logp_in_range',          # logP 是否在允许范围内
        'operation',              # 从父到子的操作类型
        'operation_params',       # 操作参数
        'is_expanded',            # 是否已展开（生成过子节点）
        'is_terminal',            # 是否终止（不再展开）
    )
    
    def q_value(self):
        """平均价值 = total_value / visit_count"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb_score(self, c: float):
        """PUCT 分数 = Q(a) + c * P(a) * sqrt(N(parent)) / (1 + N(a))"""
        if self.parent is None:
            return 0.0
        exploration = c * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value() + exploration
```

### 3.3 MCTS 主循环逻辑

```python
# ---------- MCTS 主循环 ----------
for sim_idx in range(num_simulations):  # ★ num_simulations 决定总轮数
    if getattr(self, 'interrupted', False):
        break
    
    # 1️⃣ Selection：从根沿 PUCT 向下
    node = root
    while node.is_expanded and node.children and not node.is_terminal:
        node = _select_child(node)  # 选择 ucb_score 最高的子节点
    
    # 2️⃣ Expansion：首次展开节点
    if not node.is_terminal and not node.is_expanded:
        _expand_node(node)  # 生成候选、打分、创建子节点
    
    # 3️⃣ Evaluation：评估叶节点
    if node.children:
        eval_node = _select_child(node)
        value = _evaluate_leaf(eval_node)
    else:
        value = _evaluate_leaf(node)
    
    # 4️⃣ Backpropagation：回传价值
    _backpropagate(eval_node, value)
    
    # 定期日志
    if (sim_idx + 1) % max(1, num_simulations // 5) == 0:
        print(f"[MCTS] simulation {sim_idx+1}/{num_simulations}, "
              f"root visits={root.visit_count}, "
              f"unique states cached={len(expansion_cache)}")
```

### 3.4 关键子函数

#### `_expand_node(node)`：首次展开一个节点

```python
def _expand_node(node: _MCTSNode):
    """生成候选、批量预测、创建子节点"""
    
    if node.is_expanded or node.is_terminal:
        return
    node.is_expanded = True
    
    # 检查深度限制
    if node.depth >= max_depth:
        node.is_terminal = True
        return
    
    # 检查 pruning patience（连续无改善）
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
            node.is_terminal = True
            return
    
    # 检查 logP patience（连续超出范围）
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
    
    # 获取可执行操作
    mol = Chem.MolFromSmiles(node.smiles)
    if mol is None:
        node.is_terminal = True
        return
    
    possible_ops = self._get_possible_operations(mol)
    
    # 批量预测 property_change
    batch_from, batch_to, batch_ops, valid_pairs = [], [], [], []
    for op in possible_ops:
        new_mol = self._apply_operation(mol, op["type"], **op.get("params", {}))
        if new_mol and self.validate_molecule(new_mol):
            new_smiles = Chem.MolToSmiles(new_mol)
            batch_from.append(node.smiles)
            batch_to.append(new_smiles)
            batch_ops.append(op)
            valid_pairs.append((op, new_smiles))
    
    candidates = []
    if batch_from:
        try:
            predictions = predictor.predict_batch(batch_from, batch_to, batch_ops)
            for idx, (op, new_smi) in enumerate(valid_pairs):
                if idx < len(predictions) and predictions[idx] is not None:
                    candidates.append((op, new_smi, predictions[idx]))
        except Exception as e:
            print(f"[MCTS] 批量预测出错: {e}")
    
    if not candidates:
        node.is_terminal = True
        return
    
    # 候选选择（按 expansion_mode）
    if optimization_direction == 'increase':
        sorted_cands = sorted(candidates, key=lambda x: x[2], reverse=True)
    else:
        sorted_cands = sorted(candidates, key=lambda x: x[2])
    
    if expansion_mode == 'topk':
        selected_cands = sorted_cands[:max_branching]
    elif expansion_mode == 'random_topk':
        selected_cands = rng.sample(candidates, min(max_branching, len(candidates)))
    else:  # full
        selected_cands = sorted_cands
    
    # 计算 prior（softmax 或 uniform）
    if prior_mode == 'uniform':
        priors = [1.0 / len(selected_cands)] * len(selected_cands)
    else:  # softmax
        raw_scores = [c[2] for c in selected_cands]
        if optimization_direction == 'decrease':
            raw_scores = [-s for s in raw_scores]
        max_score = max(raw_scores) if raw_scores else 0
        exp_scores = [math.exp(s - max_score) for s in raw_scores]
        sum_exp = sum(exp_scores) or 1.0
        priors = [e / sum_exp for e in exp_scores]
    
    # 创建子节点
    for (op, new_smi, prop_change), prior in zip(selected_cands, priors):
        child = _MCTSNode(
            smiles=new_smi,
            depth=node.depth + 1,
            parent=node,
            prior=prior,
            property_value=node.property_value + prop_change,
            accumulated_change=node.accumulated_change + prop_change,
            operation=op["type"],
            operation_params=op.get("params", {}),
        )
        child.logp = self.calculate_logP(new_smi)
        child.logp_in_range = logp_range[0] <= child.logp <= logp_range[1]
        node.children.append(child)
```

#### `_select_child(node)`：PUCT 选择

```python
def _select_child(node: _MCTSNode) -> Optional[_MCTSNode]:
    """PUCT 选择最佳子节点"""
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

#### `_evaluate_leaf(node)`：叶节点估值

```python
def _evaluate_leaf(node: _MCTSNode) -> float:
    """支持三种叶节点价值模式"""
    
    if value_mode == 'zero':
        val = 0.0
    elif value_mode == 'step':
        # 仅当前步的改变
        if node.parent is None:
            val = 0.0
        else:
            val = node.accumulated_change - node.parent.accumulated_change
    else:  # 'accumulated'
        # 从根到当前的累计改变
        val = node.accumulated_change
    
    # 对于 'decrease' 方向需要取反
    if optimization_direction == 'decrease':
        val = -val
    
    return val
```

#### `_backpropagate(node, value)`：回传

```python
def _backpropagate(node: _MCTSNode, value: float):
    """回传价值到根"""
    cur = node
    while cur is not None:
        cur.visit_count += 1
        cur.total_value += value
        cur = cur.parent
```

### 3.5 MCTS 输出结构

```python
expansion_tree = {
    "initial_smiles": self.initial_smiles,
    "max_depth": max_depth,
    "max_branching": max_branching,
    "search_mode": "mcts",
    "mcts_stats": {
        "num_simulations": num_simulations,           # 请求的模拟轮数
        "actual_simulations": root.visit_count,       # ★ 实际执行轮数
        "exploration_weight": exploration_weight,
        "prior_mode": prior_mode,
        "value_mode": value_mode,
        "expansion_mode": expansion_mode,
        "random_seed": random_seed,
        "unique_states_expanded": len(expansion_cache),  # 展开过的唯一分子
        "root_visits": root.visit_count,
    },
    "nodes": {
        "0": {
            "id": "0",
            "smiles": "...",
            "depth": 0,
            "parent_id": None,
            "mcts_visits": root.visit_count,          # 该节点访问次数
            "mcts_prior": 0.0,                        # 先验概率
            "mcts_q_value": 0.5,                      # Q 值
            "property_value": ...,
            "accumulated_change": 0.0,
            ...
        },
        ...
    },
    "edges": [...]
}
```

---

## 4. 参数流向追踪

### 4.1 从 CLI 到 MCTS

```
CLI 参数
  ├─ batch_optimizer.py::parse_args()
  │   └─ args.num_simulations = 200
  │   └─ args.exploration_weight = 1.4
  │   └─ args.mcts_prior_mode = 'softmax'
  │   └─ args.mcts_value_mode = 'accumulated'
  │   └─ args.mcts_expansion_mode = 'topk'
  │   └─ args.mcts_random_seed = None
  │
  ├─ batch_process(args, ...)
  │
  ├─ run_evolution_optimizer(args, ...)
  │   └─ optimizer.optimize_evolution_tree(
  │       num_simulations=args.num_simulations,        # 200
  │       exploration_weight=args.exploration_weight,  # 1.4
  │       mcts_prior_mode=args.mcts_prior_mode,       # 'softmax'
  │       mcts_value_mode=args.mcts_value_mode,       # 'accumulated'
  │       mcts_expansion_mode=args.mcts_expansion_mode,  # 'topk'
  │       mcts_random_seed=args.mcts_random_seed,     # None
  │   )
  │
  └─ EvolutionTreeOptimizer::optimize_evolution_tree()
      └─ evolver.generate_expansion_tree_mcts(
          num_simulations=num_simulations,
          exploration_weight=exploration_weight,
          prior_mode=mcts_prior_mode,
          value_mode=mcts_value_mode,
          expansion_mode=mcts_expansion_mode,
          random_seed=mcts_random_seed,
      )
```

### 4.2 参数含义详解

| 参数 | 类型 | 流向 | 用途 |
|------|------|------|------|
| `num_simulations` | int | CLI → MCTS 主循环 | 控制 `for sim_idx in range(num_simulations)` 的循环次数 |
| `exploration_weight` | float | CLI → PUCT | 控制 `c` 系数，影响探索 vs 利用的平衡 |
| `prior_mode` | str | CLI → _expand_node | 控制先验 P(a) 的计算方式（softmax / uniform） |
| `value_mode` | str | CLI → _evaluate_leaf | 控制叶节点价值计算（accumulated / zero / step） |
| `expansion_mode` | str | CLI → _expand_node | 控制候选截断策略（topk / random_topk / full） |
| `max_depth` | int | CLI → _expand_node | 控制深度检查条件 |
| `max_branching` | int | CLI → _expand_node | 控制子节点数上限 |

---

## 5. 步数统计与预算机制

### 5.1 模拟轮数计数

**关键变量**：`sim_idx` 与 `root.visit_count`

```python
for sim_idx in range(num_simulations):  # ★ 实际 loop counter
    # 每次循环一轮完整的 MCTS (Selection → Expansion → Eval → Backprop)
    ...
    _backpropagate(node, value)  # ★ 在 backprop 时会对路径上所有节点的 visit_count++
```

**因此**：
- `sim_idx` 从 0 到 `num_simulations - 1`，共 `num_simulations` 次迭代
- `root.visit_count` 最终 = `num_simulations`（因为每次 backprop 都会访问根）
- 实际执行的模拟数 = `root.visit_count`（被记录在 `mcts_stats.actual_simulations`）

### 5.2 展开预算与步数统计

与 MCTS 相关的计数有：

| 计数器 | 含义 | 更新时机 |
|--------|------|---------|
| `sim_idx` | 当前循环迭代号（0-indexed） | 每次 for 循环开始 |
| `root.visit_count` | 根节点被访问次数 | 每次 backprop 中 |
| `node.visit_count` | 该节点被访问次数 | 每次 backprop 中 |
| `len(expansion_cache)` | 展开过的唯一分子数 | 首次展开节点时 |
| `node_counter` | 输出树中的节点总数 | 后续的 _traverse 中 |

### 5.3 "步"的概念

在 MCTS 语境下，常见的"步"定义有两种：

**定义1：模拟轮数**
- 每一次 `for sim_idx in range(num_simulations)` 循环称为一个"模拟"
- 一个模拟包括 1 次 Selection + 1 次 Expansion（可能） + 1 次 Eval + 1 次 Backprop
- 总模拟数 = `num_simulations`

**定义2：树展开步数**
- 每一次 `_expand_node()` 调用算一步展开
- 总展开数 ≤ `num_simulations`（因为不是每个模拟都会展开新节点）
- 实际展开数 = `len(expansion_cache)`

**定义3：单条路径深度**
- 从根到叶的路径长度
- 受 `max_depth` 限制

### 5.4 添加预算限制的扩展点

如果要添加"展开预算"限制，可以在以下位置：

```python
# 方案1：在 _expand_node 中添加预算检查
expansion_count = [0]  # 全局展开计数

def _expand_node(node: _MCTSNode):
    if expansion_count[0] >= expansion_budget:
        node.is_terminal = True  # 不再展开
        return
    
    # ... 原逻辑 ...
    expansion_count[0] += 1

# 方案2：在主循环中添加"实际模拟次数"检查
# 已有：
if (sim_idx + 1) % max(1, num_simulations // 5) == 0:
    print(f"[MCTS] simulation {sim_idx+1}/{num_simulations}, ...")

# 可添加：
if some_early_stopping_condition:
    print(f"[MCTS] 达到早停条件，在第 {sim_idx+1} 轮停止")
    break
```

---

## 6. 数据结构速查

### 6.1 _MCTSNode 结构

```python
_MCTSNode {
    # 树拓扑
    smiles: str                      # SMILES 字符串
    depth: int                       # 深度（0 = 根）
    parent: Optional[_MCTSNode]      # 父节点指针
    children: List[_MCTSNode]        # 子节点列表
    
    # MCTS 统计
    visit_count: int                 # 被访问次数
    total_value: float               # 累计价值
    prior: float                     # 先验概率
    
    # 属性预测
    property_value: Optional[float]  # 当前预测属性值
    accumulated_change: float        # 从根到当前的累计改变
    logp: Optional[float]            # logP 值
    logp_in_range: bool              # logP 是否有效
    
    # 操作信息
    operation: Optional[str]         # 操作类型
    operation_params: dict           # 操作参数
    
    # 状态标记
    is_expanded: bool                # 是否已展开过
    is_terminal: bool                # 是否终止
}
```

### 6.2 expansion_tree 结构

```python
expansion_tree: Dict = {
    "initial_smiles": str,
    "max_depth": int,
    "max_branching": int,
    "search_mode": "mcts",
    
    "mcts_stats": {
        "num_simulations": int,              # 请求轮数
        "actual_simulations": int,           # 实际轮数 = root.visit_count
        "exploration_weight": float,
        "prior_mode": str,
        "value_mode": str,
        "expansion_mode": str,
        "random_seed": Optional[int],
        "unique_states_expanded": int,       # 展开的唯一分子数
        "root_visits": int,
    },
    
    "nodes": {
        "0": {
            "id": str,
            "smiles": str,
            "depth": int,
            "parent_id": Optional[str],
            "operation": Optional[str],
            "details": dict,
            "logP": Optional[float],
            "logP_in_range": bool,
            
            # MCTS 特有字段
            "mcts_visits": int,               # visit_count
            "mcts_prior": float,              # prior
            "mcts_q_value": float,            # q_value()
            
            # 属性预测字段
            "property_value": Optional[float],
            "accumulated_change": float,
            "property_change": float,
        },
        ...
    },
    
    "edges": [
        {
            "from": str,                      # 父节点 ID
            "to": str,                        # 子节点 ID
            "operation": str,
            "details": dict,
        },
        ...
    ]
}
```

---

## 7. 改造建议

### 7.1 添加单个分子的模拟轮数上限

**目标**：某个分子的 MCTS 搜索超过某个轮数时自动停止。

**实现位置**：`MolecularEvolutionExpansion.generate_expansion_tree_mcts()`

```python
# 添加参数
def generate_expansion_tree_mcts(
    self,
    ...,
    max_simulations_per_molecule: Optional[int] = None,  # 新增
):
    ...
    for sim_idx in range(num_simulations):
        if max_simulations_per_molecule and sim_idx >= max_simulations_per_molecule:
            print(f"[MCTS] 达到单分子模拟上限 {max_simulations_per_molecule}，停止")
            break
        ...
```

### 7.2 添加展开预算限制

**目标**：限制展开过的唯一分子数量。

```python
expansion_budget = 100  # 最多展开100个不同的分子
expansion_count = [0]

def _expand_node(node: _MCTSNode):
    if expansion_count[0] >= expansion_budget:
        node.is_terminal = True
        return
    
    # ... 原逻辑 ...
    expansion_count[0] += 1
```

### 7.3 添加早停条件

**目标**：当 TopK 结果稳定（连续若干轮没有改进）时停止。

```python
best_topk_value = -float('inf')
no_improve_count = 0
max_no_improve = 10  # 连续10轮无改进就停止

for sim_idx in range(num_simulations):
    ...
    
    # 每 N 轮检查一次
    if (sim_idx + 1) % 10 == 0:
        current_topk = _get_current_topk(root, k=5)
        current_best = max([node.accumulated_change for node in current_topk])
        
        if current_best > best_topk_value:
            best_topk_value = current_best
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            print(f"[MCTS] 连续 {max_no_improve} 次无改进，停止")
            break
```

### 7.4 添加时间预算

**目标**：限制单个分子的搜索时间。

```python
import time

start_time = time.time()
timeout_seconds = 60  # 单分子最多搜索 60 秒

for sim_idx in range(num_simulations):
    if time.time() - start_time > timeout_seconds:
        print(f"[MCTS] 超时（{timeout_seconds}s），停止")
        break
    ...
```

### 7.5 在 batch_optimizer.py 中暴露新参数

```python
parser.add_argument('--mcts-max-simulations-per-molecule', type=int, default=None,
                    help='单分子MCTS最大模拟轮数')
parser.add_argument('--mcts-expansion-budget', type=int, default=None,
                    help='MCTS展开预算（唯一分子数上限）')
parser.add_argument('--mcts-timeout-seconds', type=float, default=None,
                    help='单分子MCTS搜索时间上限（秒）')

# 在 batch_process 中传递
optimized_tree = optimizer.optimize_evolution_tree(
    ...,
    max_simulations_per_molecule=args.mcts_max_simulations_per_molecule,
    expansion_budget=args.mcts_expansion_budget,
    timeout_seconds=args.mcts_timeout_seconds,
)
```

---

## 总结

| 关键概念 | 说明 |
|---------|------|
| **num_simulations** | MCTS 主循环的迭代次数，决定了总的蒙特卡洛模拟轮数 |
| **sim_idx** | 主循环计数器（0 ~ num_simulations-1） |
| **root.visit_count** | 根节点访问次数 = 最终实际模拟轮数 |
| **exploration_weight (c)** | PUCT 公式中的探索强度参数 |
| **prior (P(a))** | softmax 从预测打分计算的先验概率 |
| **value_mode** | 叶节点价值函数的语义（accumulated / zero / step） |
| **expansion_cache** | 缓存已展开的分子，避免重复计算 |
| **_MCTSNode** | 树节点结构，记录 SMILES、visit_count、total_value 等 |
| **mcts_stats** | 输出 JSON 中的统计字段，包括 actual_simulations、unique_states_expanded |

---

**文档版本**：2026-04-28  
**关键文件**：
- `scripts/optimization/batch_optimizer.py` (679 行)
- `core/evolution_optimizer.py` (optimize_evolution_tree 方法)
- `core/molecular_evolution_expansion.py` (generate_expansion_tree_mcts 方法, ~400 行)

