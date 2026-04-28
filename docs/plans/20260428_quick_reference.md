# mol-evo 快速参考卡

## 🚀 快速开始

### 最小化运行
```bash
cd /Users/havocrao/Documents/Projects/whu/mol-ofo/mol-evo

# 快速测试（5个分子）
PROFILE=smoke bash scripts/runners/run_mcts_ablations.sh

# 完整运行（50个分子）
PROFILE=official bash scripts/runners/run_mcts_ablations.sh

# 超参数扫描
SWEEP_PRESET=paper_minimal bash scripts/runners/run_mcts_hparam_sweeps.sh
```

---

## 📁 关键路径速查

| 功能 | 路径 |
|------|------|
| 批量优化脚本 | `scripts/optimization/batch_optimizer.py` |
| Ablation实验 | `scripts/runners/run_mcts_ablations.sh` |
| 超参数扫描 | `scripts/runners/run_mcts_hparam_sweeps.sh` |
| 输入数据 | `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv` |
| 输出目录 | `output/paper/ablations/` 或 `output/evo-mo/` |
| 模型 LUMO | `mol_evo/output/v0/.../train-20251123_192921-lumo_change-120000-200/` |
| 模型 HOMO | `mol_evo/output/v0/.../train-20251127_121257-homo_change-120000-200/` |

---

## ⚙️ 关键参数

### batch_optimizer.py 参数

**最常改**:
- `--search-mode` (bfs/mcts/astar_demo)
- `--num-simulations` (MCTS模拟轮数，默认200)
- `--max-depth` (默认2)
- `--max-branching` (默认8)
- `--direction` (increase/decrease)
- `--target-property` (lumo/homo/...)
- `--topK` (默认20)

**MCTS特定**:
- `--mcts-prior-mode` (softmax/uniform)
- `--mcts-value-mode` (accumulated/zero/step)
- `--mcts-expansion-mode` (topk/random_topk/full)
- `--mcts-random-seed` (重现性)

**约束**:
- `--logp-min` / `--logp-max` (分子复杂度约束)
- `--logp-patience` (连续超出范围的容忍轮数)

---

## 📊 输出文件格式

### batch_results.json 结构
```
{
  "SMILES_str": {
    "original_data": { ... },
    "optimization_result": {
      "status": "success",
      "topk_results": [ { rank, smiles, property, improvement, path_length, path } ],
      "optimized_result": { nodes, mcts_stats, ... }
    }
  }
}
```

### topK.csv 结构
```
mol_start, value_start, mol_1, value_1, mol_2, value_2, ...
```

### 日志文件
- `batch_optimization_main.log`: 总日志（配置+统计）
- `{smiles}_{id}.log`: 单分子日志

---

## 🔄 断点续传

已完成的分子会通过 `_topK.csv` 自动检测，无需手动操作：

```bash
# 只有新分子会被处理，已完成的会被跳过
python -m mol_evo.scripts.optimization.batch_optimizer \
  --output-dir ./same/output/dir \
  ...
```

---

## 🔍 环境变量列表

### run_mcts_ablations.sh
```
PROFILE=smoke/pilot/official          # 预设规模
TASKS=lumo_up,homo_down               # 任务列表
VARIANTS=full,wo_prior,wo_leaf_value  # 消融变体
SEEDS=42,43,44                        # 随机种子
CUDA_VISIBLE_DEVICES=0                # GPU选择
CONDA_ENV=mol-ofo                     # conda环境
```

### run_mcts_hparam_sweeps.sh
```
SWEEP_PRESET=paper_minimal/full       # 扫描规模
TASKS=lumo_up,homo_down               # 任务列表
NUM_SIMULATION_VALUES_CSV=200,400,800 # 自定义扫描范围
EXPLORATION_WEIGHT_VALUES_CSV=1.0,2.0
```

---

## 📈 实验对标

| 指标 | 基准配置 |
|------|---------|
| num_simulations | 800 |
| max_depth | 10 |
| max_branching | 20 |
| exploration_weight | 2.0 |
| prior_mode | softmax |
| value_mode | accumulated |
| expansion_mode | topk |

---

## 🛠️ 调试技巧

### 查看单个分子结果
```bash
# JSON格式
jq '.["CC(O)(C=O)C=O"].optimization_result' batch_results.json

# CSV格式
head -5 "CC(O)(C=O)C=O_*.csv"
```

### 提取topK改进量
```bash
python -c "
import json
with open('batch_results.json') as f:
    data = json.load(f)
    for smiles, result in list(data.items())[:5]:
        topk = result['optimization_result']['topk_results']
        if topk:
            print(f'{smiles}: {topk[0][\"improvement\"]:.3f}')
"
```

### 查看搜索统计
```bash
jq '.[] | .optimization_result.optimized_result.mcts_stats' batch_results.json
```

---

## ⚡ 性能优化

### GPU内存优化
```bash
# 自动清理（每10个分子）
# 无需额外配置，batch_optimizer.py内置
```

### 并行化
```bash
# 多个GPU可分别运行不同TASKS
CUDA_VISIBLE_DEVICES=0 TASKS=lumo_up bash run_mcts_ablations.sh &
CUDA_VISIBLE_DEVICES=1 TASKS=homo_down bash run_mcts_ablations.sh &
```

---

## 📝 新增实验检查清单

- [ ] 是否在 `scripts/runners/` 下创建 `run_*.sh`
- [ ] 是否使用环境变量处理参数
- [ ] 是否生成 manifest.json / summary.tsv
- [ ] 输出目录是否为 `output/{category}/{timestamp}`
- [ ] 是否支持 `--start-index` / `--end-index` 进行数据切片
- [ ] 是否记录所有配置参数到日志
- [ ] 是否调用汇总脚本 (summarize_*.py)
- [ ] 是否支持多随机种子/变体

