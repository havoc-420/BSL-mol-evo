# Part 1 — BFS vs MCTS-full 方法级对比图

## 1. 目标

直接支撑论文核心论点：**OFO-MCTS 整体优于 OFO-BFS**。使用主实验（50 起始分子、400/800 simulations）同口径数据，确保对比公平性。

## 2. 数据来源

全部来自主实验表 5-2（`docs/paper/data/full.txt`），口径一致：
- 起始分子数：50
- 评估方式：真值评估
- MCTS 预算：400 / 800 simulations

## 3. 图形设计

### 单图双面板（slope chart / grouped bar）

- **Panel A — HOMO(D)**
  - X 轴：方法（OFO-BFS / OFO-MCTS/400 / OFO-MCTS/800）
  - Y 轴左：Average Improvement（柱状）
  - Y 轴右：Success Rate（折线 / 标注）
  - 或：双指标 grouped bar（improvement + success rate 并排）

- **Panel B — LUMO(U)**
  - 同 Panel A 结构

### 设计要点
- BFS 用浅色/虚线，MCTS 用深色/实线，突出方法差异
- 柱顶标注具体数值（improvement + success rate）
- 不混合不同口径数据

## 4. 关键数据

| Task | Method | Avg. Improvement | Success Rate | 改善倍率 vs BFS |
| --- | --- | ---: | ---: | ---: |
| HOMO(D) | OFO-BFS-Dec | 0.9125 | 72.06% | 1.00x |
| HOMO(D) | OFO-MCTS-Dec | 1.6361 | 88.74% | **1.79x** |
| LUMO(U) | OFO (BFS) | 0.4726 | 64.40% | 1.00x |
| LUMO(U) | OFO-MCTS/400 | 1.2847 | 83.30% | **2.72x** |
| LUMO(U) | OFO-MCTS/800 | 1.3292 | 83.13% | **2.81x** |

## 5. 输出物

- `part1_bfs_vs_mcts.png`
- `part1_bfs_vs_mcts.pdf`
- `part1_data.csv`
- `make_part1_bfs_vs_mcts.py`

## 6. 图注建议

> **Figure X.** Comparison of OFO-BFS and OFO-MCTS on molecular optimization tasks (main experiment, 50 starting molecules). Both average improvement and success rate demonstrate that MCTS substantially outperforms BFS, with the most pronounced gap on LUMO(U) where MCTS achieves a 2.7–2.8× higher improvement.
