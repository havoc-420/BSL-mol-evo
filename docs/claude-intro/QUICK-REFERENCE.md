# 快速参考指南

## 一句话核心概念

| 概念 | 一句话说明 |
|------|-----------|
| batch_optimizer.py | 循环读 CSV 里的分子，逐个调 MCTS/BFS/A* 搜索生成进化树 |
| MCTS | 用 PUCT 公式选择路径，重复 200 次模拟，最后取访问次数最多的节点 |
| num_simulations | 主循环迭代次数，通常等于最终的 root.visit_count |
| exploration_weight | PUCT 公式里的 c 参数，控制探索 vs 利用的平衡 |
| _MCTSNode | 树节点，记录 SMILES、visit_count、total_value、prior 等 |
| expansion_cache | 缓存已展开的分子，避免重复批量预测 |
| mcts_stats | 输出 JSON 里记录的 MCTS 统计，包括 actual_simulations |

---

## 关键代码位置

```
CLI 入口
  ↓
scripts/optimization/batch_optimizer.py::main()
  ├─ parse_args()           (第 70-167 行)
  ├─ batch_process()        (第 353-535 行)
  │  └─ for each molecule:
  │     └─ run_evolution_optimizer()   (第 246-323 行)
  │        └─ EvolutionTreeOptimizer.optimize_evolution_tree()
  │           (core/evolution_optimizer.py 第 728 行)
  │           └─ evolver.generate_expansion_tree_mcts()
  │              (core/molecular_evolution_expansion.py 第 1126 行)
  │              ├─ 主循环: for sim_idx in range(num_simulations)  (第 1422 行)
  │              │  ├─ Selection: node = _select_child(node)      (第 1369 行)
  │              │  ├─ Expansion: _expand_node(node)              (第 1229 行)
  │              │  ├─ Evaluation: _evaluate_leaf(node)           (第 1382 行)
  │              │  └─ Backprop: _backpropagate(node, value)      (第 1398 行)
  │              └─ 输出: _traverse(root, None)                    (第 1482 行)
  └─ save_results()
```

---

## MCTS 执行流程图

```
START
  │
  ├─ 初始化根节点 root (depth=0)
  │
  ├─ for sim_idx = 0 to (num_simulations - 1):
  │  │
  │  ├─ [1] Selection
  │  │   node = root
  │  │   while node.is_expanded and node.children:
  │  │       node = _select_child(node)  # PUCT 选择
  │  │
  │  ├─ [2] Expansion
  │  │   if not node.is_expanded:
  │  │       _expand_node(node)  # 生成子节点
  │  │
  │  ├─ [3] Evaluation
  │  │   if node.children:
  │  │       eval_node = _select_child(node)
  │  │       value = _evaluate_leaf(eval_node)
  │  │   else:
  │  │       value = _evaluate_leaf(node)
  │  │
  │  └─ [4] Backpropagation
  │      _backpropagate(eval_node, value)
  │      # 更新路径上所有节点的 visit_count 和 total_value
  │
  ├─ _traverse(root)  # 将 MCTS 树转为 nodes/edges 格式
  │
  └─ return expansion_tree

END
```

---

## 参数流向图

```
--num-simulations=200
         ↓
   args.num_simulations
         ↓
   batch_process(args)
         ↓
   run_evolution_optimizer(args)
         ↓
   optimizer.optimize_evolution_tree(num_simulations=args.num_simulations)
         ↓
   evolver.generate_expansion_tree_mcts(num_simulations=200)
         ↓
   for sim_idx in range(num_simulations):  ← 循环 200 次
         ↓
   root.visit_count = 200  (最终)
```

---

## 常见修改点

### 1. 改变模拟轮数

**当前**：`--num-simulations 200`（默认）

**修改方式**：
```bash
python scripts/optimization/batch_optimizer.py \
  --search-mode mcts \
  --num-simulations 500
```

### 2. 改变探索系数

**当前**：`--exploration-weight 1.4`（默认）

**代码位置**：`core/molecular_evolution_expansion.py` 第 1374 行
```python
sc = ch.ucb_score(exploration_weight)  # ← c 在这里使用
```

### 3. 改变先验计算方式

**当前**：`--mcts-prior-mode softmax`（默认）

**可选项**：`'softmax'` 或 `'uniform'`

**代码位置**：`core/molecular_evolution_expansion.py` 第 1327-1337 行

### 4. 改变价值函数

**当前**：`--mcts-value-mode accumulated`（默认）

**可选项**：
- `'accumulated'` - 从根到当前的累计改变
- `'zero'` - 恒为 0
- `'step'` - 仅当前步的改变

**代码位置**：`core/molecular_evolution_expansion.py` 第 1382-1396 行

### 5. 改变扩展策略

**当前**：`--mcts-expansion-mode topk`（默认）

**可选项**：
- `'topk'` - 取排序后的前 max_branching 个
- `'random_topk'` - 随机采样 max_branching 个
- `'full'` - 全部保留（排序后）

**代码位置**：`core/molecular_evolution_expansion.py` 第 1316-1324 行

---

## 输出格式

### 单分子 JSON 结构

```json
{
  "initial_smiles": "C1=CC=CC=C1",
  "max_depth": 2,
  "max_branching": 8,
  "search_mode": "mcts",
  "mcts_stats": {
    "num_simulations": 200,
    "actual_simulations": 200,
    "exploration_weight": 1.4,
    "prior_mode": "softmax",
    "value_mode": "accumulated",
    "expansion_mode": "topk",
    "random_seed": null,
    "unique_states_expanded": 45,
    "root_visits": 200
  },
  "nodes": {
    "0": {
      "id": "0",
      "smiles": "C1=CC=CC=C1",
      "depth": 0,
      "parent_id": null,
      "mcts_visits": 200,
      "mcts_prior": 0.0,
      "mcts_q_value": 0.5,
      ...
    },
    ...
  },
  "edges": [...]
}
```

### 关键统计字段

| 字段 | 含义 | 类型 |
|------|------|------|
| `mcts_stats.num_simulations` | 请求的模拟轮数 | int |
| `mcts_stats.actual_simulations` | 实际执行的轮数 | int |
| `mcts_stats.unique_states_expanded` | 展开的唯一分子数 | int |
| `mcts_stats.root_visits` | 根节点访问次数 | int |
| `node.mcts_visits` | 该节点访问次数 | int |
| `node.mcts_prior` | 该节点先验概率 | float |
| `node.mcts_q_value` | 该节点平均价值 | float |

---

## 调试技巧

### 1. 查看 MCTS 进度

运行时会每 num_simulations/5 打印一次：
```
[MCTS] simulation 40/200, root visits=40, unique states cached=12
[MCTS] simulation 80/200, root visits=80, unique states cached=24
...
```

### 2. 验证参数是否传递

查看输出 JSON 中的 `mcts_stats` 字段，确认参数是否符合预期。

### 3. 计算树的展开效率

```
展开效率 = unique_states_expanded / num_simulations
```

例如：
- 200 次模拟，45 个唯一分子 → 22.5% 效率
- 说明每个模拟平均重复利用 4.4 个已展开节点

### 4. 查看单个分子的搜索耗时

在 `batch_optimization_total.log` 中查看：
```
=== 开始处理分子 1/50: CC(C)O ===
...
状态: success
耗时: 12.34 秒
```

---

## 常见问题

### Q: num_simulations 和 root.visit_count 什么关系？

**A**: 通常相等。每次 backpropagation 都会递增根的 visit_count，而主循环迭代 num_simulations 次，所以最终 root.visit_count ≈ num_simulations。

### Q: exploration_weight 应该设多少？

**A**: 
- 0.5-1.0：较保守，多利用已有信息
- 1.4（默认）：均衡
- 2.0+：更激进，多探索新区域

### Q: 为什么展开的分子数 < num_simulations？

**A**: 因为 MCTS 会重复访问已展开的节点。展开是"首次生成子节点"，而访问是"经过该节点"。

### Q: 能不能限制搜索时间？

**A**: 当前代码不支持，但可以在 `_expand_node()` 中添加 `time.time()` 检查来实现。

---

## 文件大小速查

| 文件 | 行数 | 关键函数 |
|------|------|---------|
| `scripts/optimization/batch_optimizer.py` | 679 | main, batch_process, run_evolution_optimizer |
| `core/evolution_optimizer.py` | ~1600 | optimize_evolution_tree |
| `core/molecular_evolution_expansion.py` | ~2000 | generate_expansion_tree_mcts (400+ 行) |

---

## 运行示例

### 最小化示例（BFS，快速验证）

```bash
python scripts/optimization/batch_optimizer.py \
  --input-csv dataset/eval-data/qm9_test_molecules.csv \
  --search-mode bfs \
  --max-depth 1 \
  --max-branching 3 \
  --start-index 0 \
  --end-index 5
```

### 标准 MCTS 运行

```bash
python scripts/optimization/batch_optimizer.py \
  --input-csv dataset/eval-data/qm9_test_molecules.csv \
  --search-mode mcts \
  --num-simulations 200 \
  --exploration-weight 1.4 \
  --max-depth 3 \
  --max-branching 8 \
  --mcts-prior-mode softmax \
  --mcts-value-mode accumulated \
  --mcts-expansion-mode topk
```

### 长期实验（高精度）

```bash
python scripts/optimization/batch_optimizer.py \
  --input-csv dataset/eval-data/qm9_test_molecules.csv \
  --search-mode mcts \
  --num-simulations 500 \
  --exploration-weight 1.2 \
  --max-depth 4 \
  --max-branching 12 \
  --mcts-expansion-mode full \
  --topK 50
```

---

## 性能参考

| 参数组合 | 单分子耗时 | 模拟轮数 | 展开分子数 |
|---------|----------|--------|----------|
| max_depth=1, max_branching=3, num_simulations=50 | ~0.5s | 50 | 5 |
| max_depth=2, max_branching=8, num_simulations=200 | ~3-5s | 200 | 30-50 |
| max_depth=3, max_branching=10, num_simulations=500 | ~15-20s | 500 | 80-150 |

**注**：耗时取决于模型加载、GPU 可用性、分子大小等因素。

---

**最后更新**：2026-04-28

