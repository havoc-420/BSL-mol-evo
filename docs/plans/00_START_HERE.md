# 🎯 mol-evo 项目入门指南

**最后更新**: 2026/04/28  
**项目位置**: `/Users/havocrao/Documents/Projects/whu/mol-ofo/mol-evo`

---

## 📖 这是什么？

本目录包含 mol-evo 项目的**快速探索文档**（生成于 2026/04/28）。

这些文档帮助你快速理解：
- ✓ 项目的目录结构和核心脚本
- ✓ 如何运行实验
- ✓ 输出文件的格式
- ✓ 参数的含义和最佳实践

---

## 🚀 5秒快速开始

```bash
# 进入项目目录
cd /Users/havocrao/Documents/Projects/whu/mol-ofo/mol-evo

# 运行快速测试（~5分钟）
PROFILE=smoke bash scripts/runners/run_mcts_ablations.sh

# 查看结果
ls -la output/paper/ablations/*/*/seed42/search/
```

---

## 📚 文档导航

### 🔰 我是新手，想快速上手
**→ 阅读**: `20260428_quick_reference.md`
- 🚀 快速开始 (3 个命令)
- 📁 关键路径 (查找文件)
- 🛠️ 调试技巧 (解决问题)

**预计阅读时间**: 5 分钟

---

### 🔍 我想深入理解项目架构
**→ 阅读**: `20260428_exploration.md`
- Section 1: 项目结构 (scripts/ 目录树)
- Section 2: batch_optimizer.py (核心脚本参数)
- Section 3: 输出格式 (JSON/CSV 详解)
- Section 4: 实验脚本 (Ablation/Sweep 分析)

**预计阅读时间**: 15-20 分钟

---

### 📋 我需要快速查找信息
**→ 查看**: `INDEX.md`
- 📚 文档清单 (3个文档简介)
- 🎯 场景导航 (6种使用场景)
- 📊 核心要点速记 (关键信息精缩)

**预计阅读时间**: 2-3 分钟

---

## ❓ 常见问题快答

### Q: 项目是做什么的？
**A**: 分子优化项目。使用进化树搜索算法（支持BFS/MCTS/A*）来改进分子的化学属性。

### Q: 主要脚本在哪里？
**A**: 
- 核心优化: `scripts/optimization/batch_optimizer.py`
- 实验管理: `scripts/runners/run_mcts_ablations.sh`
- 超参数扫描: `scripts/runners/run_mcts_hparam_sweeps.sh`

### Q: 输入和输出在哪里？
**A**:
- 输入: `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`
- 输出: `output/paper/ablations/` 或 `output/evo-mo/`

### Q: 如何快速测试？
**A**: 
```bash
PROFILE=smoke bash scripts/runners/run_mcts_ablations.sh
```
这会只处理 5 个分子，大概需要 5-10 分钟。

### Q: 模型在哪里？
**A**: 
- LUMO: `mol_evo/output/v0/.../train-20251123_192921-lumo_change-120000-200/`
- HOMO: `mol_evo/output/v0/.../train-20251127_121257-homo_change-120000-200/`

### Q: 已完成的实验如何续传？
**A**: 自动的！只要在同一个 `--output-dir` 重新运行，脚本会检测已完成的 `_topK.csv` 文件，跳过它们。

### Q: 如何调试结果？
**A**: 参考 `20260428_quick_reference.md` 的 🛠️ 调试技巧 section。

---

## 🎓 学习路径

### 快速学习者 (15分钟)
1. 阅读本文件 (00_START_HERE.md) - 2分钟
2. 浏览 quick_reference.md 的 🚀 和 ⚙️ section - 5分钟
3. 运行 `PROFILE=smoke bash scripts/runners/run_mcts_ablations.sh` - 5分钟
4. 查看生成的 `output/paper/ablations/*/` - 3分钟

### 深度学习者 (40分钟)
1. 读完 exploration.md (20分钟)
2. 研究 batch_optimizer.py 源码 (15分钟)
3. 查看真实输出案例 (5分钟)

### 实验设计者 (60分钟)
1. 完成深度学习者的路径
2. 阅读 exploration.md Section 7 "后续实验建议"
3. 参考 quick_reference.md 的 📝 检查清单
4. 设计你的实验脚本

---

## 🔗 文档大纲

| 文件 | 行数 | 大小 | 难度 | 用途 |
|------|------|------|------|------|
| 00_START_HERE.md | ~80 | 3 KB | ⭐ | 入门指南 |
| INDEX.md | ~150 | 6 KB | ⭐⭐ | 索引导航 |
| 20260428_quick_reference.md | 194 | 4.6 KB | ⭐⭐ | 快速查阅 |
| 20260428_exploration.md | 390 | 11 KB | ⭐⭐⭐ | 深度理解 |

**总计**: ~814 行, ~25 KB

---

## 📝 关键术语速查

| 术语 | 解释 |
|------|------|
| **SMILES** | 分子表示法 (字符串格式) |
| **LUMO/HOMO** | 分子的轨道能级 (优化目标) |
| **MCTS** | 蒙特卡洛树搜索 (搜索算法) |
| **topK** | 最佳K个结果 (默认K=20) |
| **Ablation** | 消融研究 (逐个去掉算法组件测试) |
| **Sweep** | 参数网格搜索 |
| **Checkpoint** | 断点续传 (恢复已完成的工作) |

---

## 🚦 快速决策树

```
┌─ 我想运行实验
│  └─ 是快速测试吗? YES → PROFILE=smoke bash ...
│     └─ 是完整实验? YES → PROFILE=official bash ...
│
├─ 我想理解输出格式
│  └─ 阅读 exploration.md Section 3
│
├─ 我想添加新参数
│  └─ 查看 exploration.md Section 7
│
├─ 我想调试问题
│  └─ 查看 quick_reference.md 🛠️ section
│
└─ 我想快速查参数
   └─ 查看 quick_reference.md ⚙️ section
```

---

## ✅ 检查清单

- [ ] 我已经阅读了这个文件 (00_START_HERE.md)
- [ ] 我知道了快速启动命令
- [ ] 我找到了输入和输出的位置
- [ ] 我知道了主要脚本在哪里
- [ ] 我已经书签了快速参考卡 (quick_reference.md)

---

## 🎯 接下来做什么？

### 选项 A: 立即运行实验
```bash
cd /Users/havocrao/Documents/Projects/whu/mol-ofo/mol-evo
PROFILE=smoke bash scripts/runners/run_mcts_ablations.sh
```

### 选项 B: 先理解项目
1. 打开 `20260428_quick_reference.md`
2. 读 🚀 和 📁 section
3. 再读 ⚙️ section

### 选项 C: 深度学习
1. 打开 `20260428_exploration.md`
2. 依次读 Section 1-4
3. 查看源码 (`scripts/optimization/batch_optimizer.py`)

---

## 💬 问题反馈

如果文档有不清楚的地方，建议：
1. 查看 exploration.md 的更详细说明
2. 查看项目源码中的注释
3. 参考已有的输出样本 (`output/paper/ablations/20260413_113451/`)

---

## 📚 相关文件

**项目源码**:
- `mol_evo/core/evolution_optimizer.py` - 核心优化器
- `scripts/optimization/batch_optimizer.py` - 批量脚本
- `scripts/runners/run_mcts_ablations.sh` - Ablation实验管理

**项目数据**:
- `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`
- `mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml`

**项目输出**:
- `output/paper/ablations/` - 论文实验
- `output/evo-mo/` - 其他优化

---

## 🎓 学习资源

- **Python 参数解析**: 阅读 batch_optimizer.py L70-169 (parse_args 函数)
- **输出格式**: 阅读 exploration.md Section 3 或查看真实文件
- **脚本设计**: 参考 run_mcts_ablations.sh (379行，注释完善)

---

**准备好开始了吗？** 👇

```bash
# 方案1: 快速测试（推荐新手）
PROFILE=smoke bash scripts/runners/run_mcts_ablations.sh

# 方案2: 完整运行
PROFILE=official bash scripts/runners/run_mcts_ablations.sh

# 方案3: 超参数扫描
SWEEP_PRESET=paper_minimal bash scripts/runners/run_mcts_hparam_sweeps.sh
```

祝你研究顺利！🚀

