# mol-evo 项目探索文档索引

**探索日期**: 2026/04/28  
**探索级别**: Quick (5分钟)  
**文档存储位置**: `docs/plans/20260428_*/`

---

## 📚 文档清单

### 1. **20260428_exploration.md** (详细探索报告)
**用途**: 深入理解项目架构和实验框架

**章节**:
- Section 1: 项目结构总览 (scripts/ 目录树)
- Section 2: batch_optimizer.py 详解
- Section 3: 输出格式规范 (JSON/CSV 结构)
- Section 4: 实验脚本风格分析 (Bash脚本)
- Section 5: topK 字段汇总
- Section 6: 关键发现
- Section 7: 后续实验建议

**关键信息**:
- 项目无 Makefile，采用 Bash Shell 管理实验
- 输出格式严格规范，便于后续汇总对比
- 支持断点续传 (检测 _topK.csv 文件)
- batch_optimizer.py 是核心，支持 bfs/mcts/astar_demo 三种搜索模式

**行数**: ~390 行

---

### 2. **20260428_quick_reference.md** (快速参考卡)
**用途**: 日常快速查阅

**章节**:
- 🚀 快速开始 (3行命令示例)
- 📁 关键路径速查表
- ⚙️ 关键参数速查
- 📊 输出文件格式
- 🔄 断点续传说明
- 🔍 环境变量列表
- 📈 实验对标 (基准参数)
- 🛠️ 调试技巧
- ⚡ 性能优化
- 📝 新增实验检查清单

**快速访问**:
- 找 batch_optimizer 参数? → 见 ⚙️ section
- 找输出格式? → 见 📊 section
- 想跑实验? → 见 🚀 section + 📁 section

**行数**: ~194 行

---

## 🎯 按使用场景查询

### 场景 1: 我想快速运行一个实验
→ 打开 `20260428_quick_reference.md`  
→ 找 🚀 快速开始 section  
→ 复制命令并修改 PROFILE 参数

### 场景 2: 我想理解输出文件结构
→ 打开 `20260428_exploration.md`  
→ 找 Section 3 "输出目录结构和数据格式"  
→ 查看 batch_results.json / topK.csv 的详细格式

### 场景 3: 我需要添加新参数
→ 打开 `20260428_exploration.md`  
→ 找 Section 2 "batch_optimizer.py 分析"  
→ 参考 "后续实验设计建议" → "扩展batch_optimizer.py"

### 场景 4: 我想理解实验脚本的设计模式
→ 打开 `20260428_exploration.md`  
→ 找 Section 4 "现有实验脚本风格分析"  
→ 查看 run_mcts_ablations.sh 和 run_mcts_hparam_sweeps.sh 的详解

### 场景 5: 我想查单个参数的含义
→ 打开 `20260428_quick_reference.md`  
→ 找 ⚙️ 关键参数 section  
→ 快速查表

### 场景 6: 我想调试某个实验结果
→ 打开 `20260428_quick_reference.md`  
→ 找 🛠️ 调试技巧 section  
→ 复制相应的命令

---

## 📋 核心要点速记

### 项目三层架构
```
Layer 1: Shell Scripts (runners/)
  ↓ (调用)
Layer 2: Python Optimizer (batch_optimizer.py)
  ↓ (生成)
Layer 3: Results (JSON + CSV + Logs)
```

### 最常用的三个命令
```bash
# 1. 快速测试 (5个分子)
PROFILE=smoke bash scripts/runners/run_mcts_ablations.sh

# 2. 完整运行 (50个分子)
PROFILE=official bash scripts/runners/run_mcts_ablations.sh

# 3. 超参数扫描
SWEEP_PRESET=paper_minimal bash scripts/runners/run_mcts_hparam_sweeps.sh
```

### 最常改的参数
- `--search-mode`: bfs / mcts / astar_demo
- `--num-simulations`: MCTS模拟轮数
- `--max-depth`: 搜索树深度
- `--direction`: increase / decrease

### 最常查的文件
- 汇总结果: `output_dir/batch_results.json`
- 单分子详情: `output_dir/{smiles}_{id}.json`
- topK排名: `output_dir/{smiles}_{id}_topK.csv`
- 配置日志: `output_dir/batch_optimization_main.log`

---

## 🔗 外部参考

### 项目主要代码文件
- **核心优化器**: `mol_evo/core/evolution_optimizer.py`
- **批量脚本**: `scripts/optimization/batch_optimizer.py`
- **Ablation脚本**: `scripts/runners/run_mcts_ablations.sh`
- **超参数脚本**: `scripts/runners/run_mcts_hparam_sweeps.sh`
- **汇总脚本**: `scripts/evaluation/summarize_mcts_ablation_runs.py`

### 项目主要数据文件
- **输入数据**: `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`
- **配置文件**: `mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml`
- **模型 (LUMO)**: `mol_evo/output/v0/.../train-20251123_192921-lumo_change-120000-200/`
- **模型 (HOMO)**: `mol_evo/output/v0/.../train-20251127_121257-homo_change-120000-200/`

### 项目主要输出目录
- **论文实验**: `output/paper/ablations/`
- **其他优化**: `output/evo-mo/`
- **模型存储**: `output/v0/`

---

## 📊 文档统计

| 文档 | 行数 | 大小 | 用途 |
|------|------|------|------|
| exploration.md | 390 | 11 KB | 深度理解 |
| quick_reference.md | 194 | 4.6 KB | 快速查阅 |
| **合计** | **584** | **15.6 KB** | - |

---

## ✅ 文档覆盖范围

- ✓ scripts/ 目录完整映射
- ✓ batch_optimizer.py 全参数说明
- ✓ 输出格式 (JSON/CSV) 完整规范
- ✓ 实验脚本 (Ablation/Sweep) 完整分析
- ✓ 环境变量及默认值汇总
- ✓ topK字段标准化定义
- ✓ 使用示例 (10+ 个命令)
- ✓ 调试技巧及性能优化
- ✓ 新实验设计建议

---

## 📝 更新记录

### 2026-04-28 初版
- 完成快速探索 (5分钟扫描)
- 生成详细探索报告 (390行)
- 生成快速参考卡 (194行)
- 覆盖所有关键信息

---

**最后更新**: 2026/04/28 12:48:00 UTC+8  
**下次更新**: 当项目结构发生重大变化时

