# Claude 代码库探索文档

**项目**：`mol-evo` 分子进化优化系统  
**探索时间**：2026-04-28  
**主要目标**：理解 `batch_optimizer.py` 的优化管道、MCTS 实现细节、参数流向和步数统计机制

---

## 📚 文档导航

本目录包含三份核心文档，针对不同的需求场景：

### 1. 📋 **[QUICK-REFERENCE.md](./QUICK-REFERENCE.md)** ⭐ 推荐首先阅读
**适用场景**：快速上手、修改参数、调试问题

**包含内容**：
- 一句话核心概念速查表
- 关键代码位置速查（行号精确到位）
- MCTS 执行流程简图
- 参数流向图
- 常见修改点（5 个常见改造位置）
- 运行示例（3 种场景：快速验证、标准运行、长期实验）
- 常见问题 FAQ

**阅读时间**：10 分钟

---

### 2. 📖 **[optimization-pipeline-architecture.md](./optimization-pipeline-architecture.md)** ⭐ 推荐作为主要参考
**适用场景**：理解系统架构、改造搜索策略、性能优化

**包含内容**：
- 整体管道架构图（3 层架构）
- batch_optimizer.py 完整分析（679 行代码详解）
- MCTS 实现详解（400+ 行代码，逐函数分析）
- 参数流向完整追踪（CLI → MCTS 主循环）
- **步数统计与预算机制**（关键！三种"步"的定义）
- 数据结构速查（_MCTSNode、expansion_tree）
- 改造建议（添加预算、早停、时间限制）

**阅读时间**：30-45 分钟

---

### 3. 📝 **[EXPLORATION-REPORT.md](./EXPLORATION-REPORT.md)** ⭐ 推荐作为附录查阅
**适用场景**：深度理解代码、排除故障、编写文档

**包含内容**：
- batch_optimizer.py 代码总结（679 行概览）
- MCTS 实现文件路径与关键概念
- 核心搜索流程详解
- 现有架构文档发现
- 关键发现总结（步数统计、参数流向、数据结构、剪枝机制）
- 建议的改造方向（展开预算、早停、时间预算）
- 文件位置速查表
- 快速查询表（参数含义、输出字段）

**阅读时间**：20-30 分钟

---

## 🎯 快速开始

### 场景1：我只想修改 num_simulations（MCTS 轮数）

1. 打开 [QUICK-REFERENCE.md](./QUICK-REFERENCE.md)
2. 查看"常见修改点 → 1. 改变模拟轮数"
3. 运行示例命令

**预计时间**：2 分钟

---

### 场景2：我想理解 MCTS 是如何工作的

1. 阅读 [QUICK-REFERENCE.md](./QUICK-REFERENCE.md) 的"一句话核心概念"表
2. 查看"MCTS 执行流程图"
3. 打开 [optimization-pipeline-architecture.md](./optimization-pipeline-architecture.md)
4. 阅读"第 3 节 MCTS 实现详解"

**预计时间**：30 分钟

---

### 场景3：我想添加一个新的搜索预算限制机制

1. 阅读 [optimization-pipeline-architecture.md](./optimization-pipeline-architecture.md) 第 5-7 节
2. 查看"改造建议"中的代码示例
3. 参考"数据结构速查"理解 _MCTSNode 和 expansion_tree

**预计时间**：45 分钟

---

### 场景4：我需要追踪参数从 CLI 到执行的完整流程

1. 打开 [QUICK-REFERENCE.md](./QUICK-REFERENCE.md) 的"参数流向图"
2. 参考"关键代码位置"获取精确行号
3. 查看 [optimization-pipeline-architecture.md](./optimization-pipeline-architecture.md) 第 4 节"参数流向追踪"

**预计时间**：20 分钟

---

## 🔑 核心概念速记

### 三种"步"的定义

| 概念 | 定义 | 计数方式 |
|------|------|---------|
| **模拟轮数** | MCTS 主循环的迭代次数 | `sim_idx` in `range(num_simulations)` |
| **实际执行轮数** | 根节点被访问的次数 | `root.visit_count` |
| **展开分子数** | 被展开过的唯一分子 | `len(expansion_cache)` |

**关键公式**：
```
最终 root.visit_count ≈ num_simulations
```

---

### 参数流向核心路径

```
CLI: --num-simulations 200
  ↓
batch_optimizer.py::parse_args()
  ↓
batch_process(args)
  ↓
run_evolution_optimizer(args)
  ↓
EvolutionTreeOptimizer::optimize_evolution_tree(num_simulations=...)
  ↓
generate_expansion_tree_mcts(num_simulations=...)
  ↓
for sim_idx in range(num_simulations):  ← 关键！循环 200 次
  ↓
root.visit_count = 200
```

---

### MCTS 的四步循环

```
Selection  ← 从根沿 PUCT 向下选择子节点
    ↓
Expansion  ← 首次展开节点，生成子节点
    ↓
Evaluation ← 评估叶节点的价值
    ↓
Backprop   ← 回传价值到根
```

---

## 📊 文件结构速查

```
mol-evo/
├── scripts/optimization/
│   └── batch_optimizer.py           ← 批量任务入口 (679 行)
│
├── core/
│   ├── evolution_optimizer.py        ← 单分子优化器 (~1600 行)
│   │   └── optimize_evolution_tree() (line 728)
│   │
│   └── molecular_evolution_expansion.py  ← 搜索实现 (~2000 行)
│       ├── generate_expansion_tree_mcts()     (line 1126)
│       ├── 主循环                            (line 1422)
│       ├── _expand_node()                    (line 1229)
│       ├── _select_child()                   (line 1369)
│       ├── _evaluate_leaf()                  (line 1382)
│       ├── _backpropagate()                  (line 1398)
│       └── _traverse()                       (line 1482)
│
└── docs/claude-intro/
    ├── README.md                   ← 本文件，导航与总结
    ├── QUICK-REFERENCE.md          ← 快速参考（推荐首先阅读）
    ├── optimization-pipeline-architecture.md  ← 详细架构（推荐作为主要参考）
    └── EXPLORATION-REPORT.md       ← 探索报告（附录参考）
```

---

## 🚀 关键发现总结

### 已完成的理解

✅ **batch_optimizer.py 的完整执行流程**
- 从 CSV 读入 → 逐个分子循环 → 调用搜索 → 保存结果

✅ **MCTS 实现的关键代码位置**
- 约 400 行核心代码
- 从第 1126 行开始的 `generate_expansion_tree_mcts()` 方法

✅ **参数从 CLI 到 MCTS 的完整流向**
- 6 层调用链，每一层都清晰可追踪
- 精确到行号

✅ **步数统计的三种理解方式**
- 模拟轮数、实际执行轮数、展开分子数
- 关键：`root.visit_count ≈ num_simulations`

✅ **内部数据结构**
- `_MCTSNode`：树节点，记录 SMILES、visit_count、total_value、prior
- `expansion_cache`：缓存已展开的分子
- `expansion_tree`：输出 JSON 结构

✅ **MCTS 的四步循环**
- Selection → Expansion → Evaluation → Backpropagation

✅ **剪枝机制**
- 属性改善停滞剪枝：连续无改善则剪枝
- logP 范围剪枝：连续超出范围则剪枝

---

## 💡 改造建议

### 1. 添加展开预算限制

**目标**：限制展开的唯一分子数量（不是模拟轮数）

**实现复杂度**：⭐ 低

```python
# 在 generate_expansion_tree_mcts 中添加
expansion_budget = 100  # 新参数
expansion_count = [0]

def _expand_node(node):
    if expansion_count[0] >= expansion_budget:
        node.is_terminal = True
        return
    # ... 原逻辑 ...
    expansion_count[0] += 1
```

---

### 2. 添加早停机制

**目标**：当 TopK 结果稳定时自动停止搜索

**实现复杂度**：⭐⭐ 中

```python
# 在主循环中添加早停检查
best_value = -float('inf')
no_improve_rounds = 0

for sim_idx in range(num_simulations):
    # ... MCTS 步骤 ...
    
    if (sim_idx + 1) % 10 == 0:
        current_best = max([n.accumulated_change for n in get_topk(root, 5)])
        if current_best > best_value:
            best_value = current_best
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
        
        if no_improve_rounds >= patience:
            break
```

---

### 3. 添加时间预算

**目标**：限制单个分子的搜索时间

**实现复杂度**：⭐ 低

```python
import time

start_time = time.time()
timeout_seconds = 60

for sim_idx in range(num_simulations):
    if time.time() - start_time > timeout_seconds:
        print(f"[MCTS] Timeout after {timeout_seconds}s")
        break
    # ... MCTS 步骤 ...
```

---

## 📈 性能参考

单分子搜索的耗时估计（GPU 环境）：

| 参数组合 | 单分子耗时 | 模拟轮数 | 展开分子数 |
|---------|----------|--------|----------|
| depth=1, branching=3, sims=50 | ~0.5s | 50 | 5-10 |
| depth=2, branching=8, sims=200 | ~3-5s | 200 | 30-50 |
| depth=3, branching=10, sims=500 | ~15-20s | 500 | 80-150 |

**影响因素**：
- GPU 可用性（使用 CUDA 加速模型预测）
- 分子大小（影响图构建时间）
- 模型复杂度（影响预测时间）

---

## 🔍 FAQ

### Q1: num_simulations 和 root.visit_count 的确切关系是什么？

**A**: 通常 `root.visit_count ≈ num_simulations`。每次 backpropagation 都会对根节点的 visit_count 递增 1，而主循环执行 num_simulations 次，因此最终基本相等。除非：
- 搜索被中断（检查了 `interrupted` 标志）
- 人工添加了提前退出条件

---

### Q2: 为什么展开的分子数（unique_states_expanded）通常远小于 num_simulations？

**A**: 因为 MCTS 会重复访问已展开的节点。
- 展开 = "首次生成子节点"（调用 `_expand_node()`）
- 访问 = "经过该节点"（调用 `_select_child()` 选择）

一个节点可能被访问多次，但只展开一次。因此：
```
展开效率 = unique_states_expanded / num_simulations
```

例如 200 次模拟中只展开了 50 个分子 → 25% 效率 → 每个模拟平均重复利用 4 个已展开节点。

---

### Q3: 能否在不修改代码的情况下限制搜索时间？

**A**: 当前代码不支持。但可以通过两种方式实现：
1. **系统级**：使用 `timeout` 命令（Linux）
2. **代码级**：修改 `_expand_node()` 添加 `time.time()` 检查

---

### Q4: MCTS 中的"先验"是怎么计算的？

**A**: 取决于 `--mcts-prior-mode` 参数：

- **softmax 模式**（默认）：
  ```python
  raw_scores = [c[2] for c in selected_cands]  # OFO 预测值
  exp_scores = [exp(s - max_s) for s in raw_scores]
  priors = [e / sum_exp for e in exp_scores]  # softmax
  ```

- **uniform 模式**：
  ```python
  priors = [1.0 / len(selected_cands)] * len(selected_cands)  # 均匀
  ```

---

### Q5: 能否改变 MCTS 中的价值函数？

**A**: 可以。使用 `--mcts-value-mode` 参数：

- `'accumulated'`（默认）：从根到当前的累计改变
- `'zero'`：恒为 0（相当于只用访问计数排序）
- `'step'`：仅当前步的改变

---

## 📚 推荐阅读顺序

### 第一次接触这个项目

1. [QUICK-REFERENCE.md](./QUICK-REFERENCE.md)
   - 第"一句话核心概念"表
   - 第"关键代码位置"表
   - 第"MCTS 执行流程图"

2. [optimization-pipeline-architecture.md](./optimization-pipeline-architecture.md)
   - 第 1 节整体架构
   - 第 3 节 MCTS 实现详解（快速版）

### 需要修改某个参数

1. [QUICK-REFERENCE.md](./QUICK-REFERENCE.md)
   - 第"常见修改点"
   - 第"运行示例"

### 需要深入理解系统

1. [optimization-pipeline-architecture.md](./optimization-pipeline-architecture.md)
   - 完整阅读

2. [EXPLORATION-REPORT.md](./EXPLORATION-REPORT.md)
   - 第 5-6 节补充细节

### 需要进行性能优化或新功能开发

1. [optimization-pipeline-architecture.md](./optimization-pipeline-architecture.md)
   - 第 5 节"步数统计与预算机制"
   - 第 7 节"改造建议"

2. [EXPLORATION-REPORT.md](./EXPLORATION-REPORT.md)
   - 第 6 节"建议的改造方向"

---

## 🏆 本次探索成果

### 生成的文档

| 文件 | 大小 | 内容 |
|------|------|------|
| [QUICK-REFERENCE.md](./QUICK-REFERENCE.md) | 8.8 KB | 快速参考、运行示例、常见问题 |
| [optimization-pipeline-architecture.md](./optimization-pipeline-architecture.md) | 30 KB | 详细架构、参数流向、改造建议 |
| [EXPLORATION-REPORT.md](./EXPLORATION-REPORT.md) | 13 KB | 探索报告、发现总结、文件速查 |
| [README.md](./README.md) | 本文件 | 文档导航与总结 |

### 代码理解深度

- ✅ batch_optimizer.py：679 行，全部理解
- ✅ EvolutionTreeOptimizer：核心方法理解
- ✅ MolecularEvolutionExpansion：MCTS 实现理解（400+ 行）
- ✅ MCTS 算法：Selection → Expansion → Evaluation → Backprop 全流程理解

### 追踪精度

- ✅ 参数流向：CLI → 主循环，6 层调用链精确到行号
- ✅ 数据流向：输入 CSV → 输出 JSON，完整链路清晰
- ✅ 步数统计：三种"步"的定义明确，计数方式清楚

---

## 📞 联系与更新

**文档生成**：自动化系统  
**生成时间**：2026-04-28  
**最后更新**：2026-04-28

---

**Happy exploring! 🚀**

