# MCTS 优化管道探索总结

## 探索目标完成情况

本次对 MOL-EVO 项目的分子优化管道进行了系统探索，重点关注 MCTS 实现、步数计数机制和参数流向。

### ✅ 完成的探索项

1. **✓ 完整读取 batch_optimizer.py**
   - 文件大小：679 行
   - 关键部分：参数解析、批量处理、中断处理
   
2. **✓ 架构文档检查**
   - docs/claude-intro/ 目录不存在，已创建
   - 添加了完整的 MCTS 架构分析文档
   
3. **✓ MCTS 实现文件定位**
   - 文件：`core/molecular_evolution_expansion.py`（2369 行）
   - 主函数：`generate_expansion_tree_mcts()` 第 1126-1532 行
   - 关键类：`_MCTSNode`（第 1184-1223 行）
   
4. **✓ 步数计数机制理解**
   - 每个 MCTS 模拟 = 1 次完整的 Select-Expand-Evaluate-Backpropagate 循环
   - `visit_count` 字段用于统计节点被访问次数
   - `root.visit_count` = 完成的总模拟数
   
5. **✓ 参数流向追踪**
   - 命令行参数 → batch_optimizer.parse_args()
   - → run_evolution_optimizer() 
   - → optimize_evolution_tree()
   - → generate_expansion_tree_mcts()
   
6. **✓ 预算机制识别**
   - `num_simulations`：配置的模拟总轮数（主要预算）
   - `max_depth`：搜索深度限制
   - `max_branching`：每层展开数限制
   - `pruning_patience` / `logp_patience`：动态终止条件

---

## 关键发现汇总

### 核心算法参数

```python
# 命令行接受的 MCTS 参数
--num-simulations 200           # 关键：总模拟数
--exploration-weight 1.4        # PUCT 探索系数
--mcts-prior-mode softmax       # Prior 构造方式
--mcts-value-mode accumulated   # 叶节点价值评估方式
--mcts-expansion-mode topk      # 候选选择策略
--mcts-random-seed None         # 可复现性种子
```

### MCTS 数据结构

```python
# _MCTSNode 类 (第 1184-1223 行)
class _MCTSNode:
    visit_count      # 被访问次数 (步数统计)
    total_value      # 累计价值
    prior            # 先验概率
    is_expanded      # 是否展开过
    is_terminal      # 是否终止
    children         # 子节点列表
    property_value   # 分子属性值
    accumulated_change  # 从根到这里的累计改变
```

### 步数流向

```
num_simulations (CLI)
    ↓
for sim_idx in range(num_simulations):
    ↓
Select → Expand → Evaluate → Backprop
    ↓
每个模拟中被访问的节点的 visit_count++
    ↓
root.visit_count = 实际完成的模拟数
```

### 树规模与性能权衡

| 配置 | 速度 | 质量 | 树大小 | 用途 |
|------|------|------|--------|------|
| num_sim=50, depth=1, branch=4 | 快 | 低 | 小 | 原型验证 |
| num_sim=200, depth=2, branch=8 | 中 | 中 | 中 | 推荐配置 |
| num_sim=500, depth=3, branch=12 | 慢 | 高 | 大 | 精细优化 |

---

## MCTS 算法流程图

```
初始化 root node (depth=0)
    ↓
for sim_idx in range(num_simulations):
    ├─ Selection: 从 root 用 PUCT 选择到叶子
    │   └─ ucb_score = q_value + c*prior*sqrt(N_parent)/(1+N_node)
    │
    ├─ Expansion: 首次展开节点
    │   ├─ 检查终止条件 (depth, pruning, logp)
    │   ├─ 生成候选操作
    │   ├─ predict_batch() 批量打分
    │   └─ 创建 max_branching 个子节点 + 计算 prior
    │
    ├─ Evaluation: 估值叶节点
    │   ├─ value_mode=accumulated: 累计增益
    │   ├─ value_mode=zero: 0
    │   └─ value_mode=step: 单步增益
    │
    └─ Backpropagation: 回传价值
        └─ 沿路径向上: visit_count++, total_value+=value
    
树转换为 JSON (只输出 visit_count>0 的节点)
    ↓
返回 expansion_tree
```

---

## 预算与限制总结

### 模拟预算 (num_simulations)
- **定义**：MCTS 主循环外层迭代次数
- **范围**：通常 50-1000
- **默认值**：200
- **对性能的影响**：线性增长，越大越精确但计算时间越长

### 空间预算 (max_depth + max_branching)
- **max_depth**：最大搜索深度（树高度）
  - 范围：通常 1-5
  - 默认值：2
  - 限制：`node.depth >= max_depth` → 终止

- **max_branching**：每个节点最多展开数
  - 范围：通常 4-20
  - 默认值：8
  - 限制：`len(selected_cands) = min(max_branching, valid_candidates)`

### 动态终止条件
1. **pruning_patience**：连续无改善代数阈值
   - 默认值：2-3
   - 检查：从根到当前节点是否有改善

2. **logp_patience**：logP 超出范围的连续代数
   - 默认值：3
   - 检查：分子的 logP 是否在 [logp_min, logp_max] 范围内

---

## 参数流向关键节点

### 1. 命令行参数定义 (batch_optimizer.py:139-167)
```python
parser.add_argument('--num-simulations', type=int, default=200, ...)
parser.add_argument('--exploration-weight', type=float, default=1.4, ...)
parser.add_argument('--mcts-prior-mode', choices=['softmax', 'uniform'], ...)
parser.add_argument('--mcts-value-mode', choices=['accumulated', 'zero', 'step'], ...)
parser.add_argument('--mcts-expansion-mode', choices=['topk', 'random_topk', 'full'], ...)
parser.add_argument('--mcts-random-seed', type=int, default=None, ...)
```

### 2. 参数传递 (batch_optimizer.py:260-280)
```python
optimized_tree = optimizer.optimize_evolution_tree(
    search_mode=args.search_mode,
    num_simulations=args.num_simulations,           # ← MCTS 参数
    exploration_weight=args.exploration_weight,    # ← MCTS 参数
    mcts_prior_mode=args.mcts_prior_mode,         # ← MCTS 参数
    mcts_value_mode=args.mcts_value_mode,         # ← MCTS 参数
    mcts_expansion_mode=args.mcts_expansion_mode, # ← MCTS 参数
    mcts_random_seed=args.mcts_random_seed,       # ← MCTS 参数
)
```

### 3. MCTS 核心函数 (molecular_evolution_expansion.py:1126)
```python
def generate_expansion_tree_mcts(
    num_simulations: int = 200,
    exploration_weight: float = 1.4,
    prior_mode: str = 'softmax',
    value_mode: str = 'accumulated',
    expansion_mode: str = 'topk',
    random_seed: Optional[int] = None,
    ...
) -> Dict:
    # 主循环使用这些参数
    for sim_idx in range(num_simulations):  # 模拟轮数
        # ... MCTS 逻辑 ...
        # 内部使用：exploration_weight, prior_mode, value_mode, expansion_mode
```

---

## 输出结构分析

### MCTS 输出的关键字段

JSON 文件中的 `mcts_stats` 字段提供了运行统计：

```json
{
  "search_mode": "mcts",
  "mcts_stats": {
    "num_simulations": 200,              // 配置的模拟数
    "actual_simulations": 198,           // 实际完成的模拟数
    "exploration_weight": 1.4,           // PUCT 探索系数
    "prior_mode": "softmax",             // Prior 构造方式
    "value_mode": "accumulated",         // 叶节点价值模式
    "expansion_mode": "topk",            // 候选选择模式
    "random_seed": null,                 // 随机种子
    "unique_states_expanded": 127,       // 展开过的唯一分子数
    "root_visits": 198                   // 根节点被访问次数 = actual_simulations
  },
  "nodes": {
    "0": {                               // 根节点
      "mcts_visits": 198,                // visit_count
      "mcts_prior": 1.0,                 // prior
      "mcts_q_value": 0.123              // q_value = total_value / visit_count
    },
    "1": {                               // 其他节点
      "mcts_visits": 45,                 // 该节点被访问 45 次
      "mcts_prior": 0.385,
      "mcts_q_value": 0.089
    }
  }
}
```

---

## 最常见的使用场景

### 快速原型验证
```bash
python scripts/optimization/batch_optimizer.py \
  --search-mode mcts \
  --num-simulations 50 \
  --max-depth 1 \
  --max-branching 4
```
**预期**：快速完成，树小，质量一般

### 标准优化（推荐）
```bash
python scripts/optimization/batch_optimizer.py \
  --search-mode mcts \
  --num-simulations 200 \
  --max-depth 2 \
  --max-branching 8 \
  --exploration-weight 1.4 \
  --mcts-prior-mode softmax \
  --mcts-value-mode accumulated
```
**预期**：中等时间，中等树大小，好的优化质量

### 精细优化
```bash
python scripts/optimization/batch_optimizer.py \
  --search-mode mcts \
  --num-simulations 500 \
  --max-depth 3 \
  --max-branching 12 \
  --exploration-weight 1.8 \
  --mcts-expansion-mode random_topk \
  --mcts-random-seed 42
```
**预期**：较长时间，大树，最好的优化质量

---

## 文件速查表

| 需求 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 了解 CLI 参数 | batch_optimizer.py | 70-167 | parse_args() |
| 追踪参数流向 | batch_optimizer.py | 246-280 | run_evolution_optimizer() |
| 了解 MCTS 主函数签名 | molecular_evolution_expansion.py | 1126-1144 | generate_expansion_tree_mcts() |
| 学习 MCTS 节点结构 | molecular_evolution_expansion.py | 1184-1223 | _MCTSNode 类 |
| 理解选择策略 | molecular_evolution_expansion.py | 1369-1380 | _select_child() |
| 理解展开策略 | molecular_evolution_expansion.py | 1229-1368 | _expand_node() |
| 理解回传机制 | molecular_evolution_expansion.py | 1398-1404 | _backpropagate() |
| 查看主循环 | molecular_evolution_expansion.py | 1422-1456 | for sim_idx in range(...) |
| 查看树转换 | molecular_evolution_expansion.py | 1457-1532 | 树 → JSON |

---

## 问题排查快速指南

### Q: 怎样增加搜索深度？
A: 增加 `--max-depth` 和 `--max-branching`，可能需要增加 `--num-simulations`

### Q: 怎样平衡速度和质量？
A: 
- **更快**：减少 num_simulations, max_depth, max_branching
- **更好**：增加 num_simulations, 增加 exploration_weight

### Q: Prior 应该选 softmax 还是 uniform？
A: 推荐 **softmax**，它利用 OFO 模型的打分信息

### Q: Value mode 应该选什么？
A: 推荐 **accumulated**，用于全路径优化

### Q: 怎样确保可复现的结果？
A: 设置 `--mcts-random-seed <number>`（例如 42）

### Q: 如何从中断的位置继续？
A: 重新运行相同的 `--output-dir`，系统会自动跳过已完成的分子

---

## 关键代码片段参考

### PUCT 选择公式
```python
# 第 1218-1223 行
def ucb_score(self, c: float):
    if self.parent is None:
        return 0.0
    exploitation = self.q_value()  # total_value / visit_count
    exploration = c * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
    return exploitation + exploration
```

### 回传机制
```python
# 第 1398-1404 行
def _backpropagate(node: _MCTSNode, value: float):
    cur = node
    while cur is not None:
        cur.visit_count += 1      # ← 步数统计关键
        cur.total_value += value
        cur = cur.parent
```

### 主循环（简化）
```python
# 第 1422-1456 行
for sim_idx in range(num_simulations):
    # 1. Selection
    node = root
    while node.is_expanded and node.children and not node.is_terminal:
        node = _select_child(node)
    
    # 2. Expansion
    if not node.is_terminal and not node.is_expanded:
        _expand_node(node)
    
    # 3-4. Evaluation + Backprop
    if node.children:
        eval_node = _select_child(node)
        if eval_node is not None:
            value = _evaluate_leaf(eval_node)
            _backpropagate(eval_node, value)
```

---

## 生成的文档

本次探索生成的文档已保存在：
- **主文档**：`docs/claude-intro/mcts-architecture.md`（16个章节，5000+ 行）
- **本文档**：`docs/claude-intro/exploration-summary.md`（快速参考）

主文档涵盖：
1. 系统概述
2. 批量优化器参数流向
3. MCTS 核心实现
4. 节点展开与预测
5. PUCT 选择策略
6. 叶节点价值评估
7. 反向传播与计数
8. 主循环与步数控制
9. 树转换与输出
10. 参数配置最佳实践
11. 数据流全景图
12. 关键概念总结
13. 故障排查
14. 配置示例
15. 文件索引
16. 快速导航

