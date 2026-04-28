# mol-evo 项目快速探索报告

**探索时间**: 2026/04/28  
**探索级别**: Quick (5分钟快速扫描)  

---

## 1. 项目结构总览

### scripts/ 目录结构
```
scripts/
├── __init__.py
├── README.md
├── convert_qm9_evo_to_paths.py
├── optimization/
│   ├── __init__.py
│   ├── batch_optimizer.py                    (核心批量优化脚本)
│   ├── batch_optimizer_visnet.py             (VisNet 变体)
│   ├── batch_optimizer_visnet_native.py      (VisNet Native 变体)
│   └── batch_optimizer_ic50.py               (IC50 变体)
├── data_prep/
│   ├── __init__.py
│   ├── convert_qm9_evo_to_paths.py
│   └── prepare_plan_a_holdout.py
├── evaluation/
│   ├── __init__.py
│   ├── eval_astar_rl_holdout.py
│   ├── stat_generation_time.py
│   ├── summarize_mcts_ablation_runs.py       (关键汇总脚本)
│   └── summarize_mcts_true_eval_runs.py
├── runners/                                  (实验管理脚本集)
│   ├── __init__.py
│   ├── run_mcts_ablations.sh                 (主实验脚本 - Ablation study)
│   ├── run_mcts_hparam_sweeps.sh            (超参数扫描脚本)
│   └── run_eval_mcts_ablations.sh            (评估脚本)
└── visualization/
    ├── __init__.py
    ├── plot_scatter_grid.py
    ├── plot_test_cases_2d.py
    ├── reassemble_selected_cases.py
    └── test_scatter_plot.py
```

### 项目根目录关键文件
- ✓ **无 Makefile** - 实验管理完全通过 bash shell script
- ✗ **无 run_*.sh 在根目录** - 所有runner脚本在 `scripts/runners/` 下
- ✓ **output/ 目录** - 存储所有实验输出
  - `output/paper/ablations/` - 主要文章Ablation实验
  - `output/evo-mo/` - 其他优化实验 (batch_optimization_*, hparam_sweeps_*)
  - `output/v0/` - 模型文件存储

---

## 2. batch_optimizer.py 分析

### 文件位置
`scripts/optimization/batch_optimizer.py`

### 典型命令行用法

**最小可运行命令**（使用默认参数）:
```bash
python -m mol_evo.scripts.optimization.batch_optimizer
```

**完整命令示例** (来自 run_mcts_ablations.sh):
```bash
python -m mol_evo.scripts.optimization.batch_optimizer \
  --input-csv data.csv \
  --output-dir ./results \
  --output-json ./results/batch_results.json \
  --model-path model.pth \
  --model-dir model_dir/ \
  --config-file config.yaml \
  --target-property lumo \
  --optimization-mode sub \
  --search-mode mcts \
  --num-simulations 800 \
  --max-depth 10 \
  --max-branching 20 \
  --direction decrease \
  --topK 20 \
  --start-index 0 \
  --end-index 50
```

### 关键参数说明

#### 必须参数
- `--input-csv` ⚠️ 有默认值，但需要检查路径有效性
  - 默认: `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`
  - CSV格式: 至少需要 `smiles`, `lumo`, `homo`, `gap` 列

#### 常用参数

**搜索控制**:
- `--search-mode` [bfs|mcts|astar_demo] - 默认 `bfs`
- `--max-depth` (int) - 默认 2
- `--max-branching` (int) - 默认 8

**MCTS特定参数**:
- `--num-simulations` (int) - 默认 200，仅MCTS模式
- `--step-budget` (int) - 节点展开预算，仅MCTS模式
- `--exploration-weight` (float) - PUCT系数，默认 1.4
- `--mcts-prior-mode` [softmax|uniform] - 默认 `softmax`
- `--mcts-value-mode` [accumulated|zero|step] - 默认 `accumulated`
- `--mcts-expansion-mode` [topk|random_topk|full] - 默认 `topk`
- `--mcts-random-seed` (int) - 可复现性种子

**优化控制**:
- `--target-property` - 目标属性名 (默认 `lumo`)
- `--optimization-mode` [sub|pct] - 默认 `sub`
- `--direction` [increase|decrease] - 默认 `decrease`
- `--topK` (int) - 保留最佳K个结果，默认 20

**约束条件**:
- `--logp-min` (float) - 默认 0.0
- `--logp-max` (float) - 默认 5.0
- `--logp-patience` (int) - 默认 3

**断点续传相关**:
- `--start-index`, `--end-index` - 数据范围

---

## 3. 输出目录结构和数据格式

### 输出目录布局
```
output_dir/
├── batch_results.json              # 所有分子的汇总结果
├── batch_optimization_main.log     # 主日志（配置 + 统计）
├── {smiles_prefix}_{run_id}.json   # 单个分子的详细优化树
├── {smiles_prefix}_{run_id}_topK.csv  # 单个分子的Top-K结果
└── {smiles_prefix}_{run_id}.log    # 单个分子处理日志
```

### batch_results.json 格式

```json
{
  "CC(O)(C=O)C=O": {
    "original_data": {
      "smiles": "CC(O)(C=O)C=O",
      "homo": -7.1184983253479,
      "lumo": -1.8748643398284912,
      "gap": 5.243634223937988
    },
    "optimization_result": {
      "status": "success",
      "smiles": "CC(O)(C=O)C=O",
      "initial_property": -7.1184983253479,
      "runtime": 12.34,
      "optimized_result": { ... },  # 详细见下
      "topk_results": { ... }        # 详细见下
    }
  }
}
```

### 优化树详细格式 (optimized_result)

```json
{
  "initial_smiles": "CC(O)(C=O)C=O",
  "max_depth": 6,
  "max_branching": 12,
  "search_mode": "mcts",
  "mcts_stats": {
    "num_simulations": 60,
    "actual_simulations": 60,
    "exploration_weight": 2.0,
    "prior_mode": "softmax",
    "value_mode": "accumulated",
    "expansion_mode": "topk",
    "random_seed": 42,
    "unique_states_expanded": 5,
    "root_visits": 60
  },
  "nodes": {
    "0": {
      "id": "0",
      "smiles": "CC(O)(C=O)C=O",
      "depth": 0,
      "parent_id": null,
      "operation": null,
      "logP": -0.8648,
      "property_value": -7.1184983253479,
      "property_change": 0.0,
      "accumulated_change": 0.0,
      "mcts_visits": 60,
      "mcts_prior": 0.0,
      "mcts_q_value": 5.095519
    },
    "1": {
      "id": "1",
      "smiles": "CC(O)(C=O)N=O",
      "depth": 1,
      "parent_id": "0",
      "operation": "replace_atom",
      "details": { "atom_idx": 3, "atom_symbol": "N" },
      "property_value": -8.675149321556091,
      "property_change": -1.556650996208191,
      "accumulated_change": -1.556650996208191,
      ...
    }
    ...
  }
}
```

### topK_results 格式

```json
{
  "initial_smiles": "CC(O)(C=O)C=O",
  "initial_property": -7.1184983253479,
  "topK_results": [
    {
      "rank": 1,
      "smiles": "CC(F)(N=O)N=O",
      "property": -11.2926,
      "improvement": -4.1741,
      "path_length": 3,
      "path": ["CC(O)(C=O)C=O", "CC(O)(C=O)N=O", "CC(O)(N=O)N=O", "CC(F)(N=O)N=O"]
    },
    ...
  ]
}
```

### topK.csv 格式

**列结构**: 
```
mol_start, value_start, mol_1, value_1, mol_2, value_2, mol_3, value_3, ...
```

**示例行**:
```
COCCOC=O, -7.355237, O=COCCOF, -8.129995, O=COCOOF, -9.201155, ...
```

---

## 4. 现有实验脚本风格分析

### 脚本风格: **纯 Bash Shell Scripts** ✓

#### run_mcts_ablations.sh (主要Ablation study)
- **目的**: 对MCTS算法做 ablation study（去掉不同组件看效果）
- **参数风格**: 环境变量 + 默认值
- **关键特性**:
  - Profile-based参数集合 (smoke/pilot/official)
  - Variant支持: full, wo_prior, wo_leaf_value, random_topb, wo_pruning, wo_logp
  - Task支持: lumo_up, lumo_down, homo_up, homo_down
  - Seed支持: 多随机种子可复现
  - 自动结果汇总（调用 `summarize_mcts_ablation_runs.py`）

#### run_mcts_hparam_sweeps.sh (超参数扫描)
- **目的**: 超参数网格搜索（搜索最优的搜索参数）
- **参数风格**: 环境变量 + 预设集合
- **关键特性**:
  - Sweep preset: paper_minimal / full
  - 自动生成sweep命令组合
  - 支持覆盖任何sweep参数范围
  - 输出: run_summary.tsv + run_commands.tsv

#### run_eval_mcts_ablations.sh (评估脚本)
- **目的**: 对ablation study结果进行评估
- **依赖**: 调用外部Python脚本 (`evaluate_batch_mo.py`, `evaluate_csv_results.py`)

### 使用示例

**快速测试**:
```bash
bash scripts/runners/run_mcts_ablations.sh  # 默认 smoke profile
```

**Pilot运行**:
```bash
PROFILE=pilot bash scripts/runners/run_mcts_ablations.sh
```

**官方运行** (完整):
```bash
PROFILE=official SEEDS=42,43,44 \
  bash scripts/runners/run_mcts_ablations.sh
```

**自定义超参数**:
```bash
CONDA_ENV=mol-ofo CUDA_VISIBLE_DEVICES=0 \
  TASKS=lumo_up,homo_down \
  SEEDS=42 \
  bash scripts/runners/run_mcts_ablations.sh
```

---

## 5. topK 结果关键字段汇总

### JSON格式中的topK字段 (用于后续汇总对比)

| 字段 | 来源 | 用途 |
|------|------|------|
| `rank` | topK_results[].rank | 排名位置 |
| `smiles` | topK_results[].smiles | 优化后分子 SMILES |
| `property` | topK_results[].property | 优化后的属性值 |
| `improvement` | topK_results[].improvement | 属性改进量 |
| `path_length` | topK_results[].path_length | 进化路径长度 |
| `path` | topK_results[].path | 完整进化路径 |
| `initial_property` | topK_results.initial_property | 初始属性值 |

### CSV格式中的字段 (topK.csv)

- **mol_start**: 起始分子
- **value_start**: 起始属性值
- **mol_1 ~ mol_K**: 优化后的K个分子SMILES
- **value_1 ~ value_K**: 对应的K个属性值

---

## 6. 关键发现

### 实验管理特点
1. ✓ **脚本驱动**: 完全通过shell脚本管理实验流程，无Makefile
2. ✓ **环境变量配置**: 高度可定制，支持环境变量覆盖所有参数
3. ✓ **配置文件**: 主要配置在shell脚本中，数据配置在YAML中
4. ✓ **结果自动汇总**: 实验完成后自动生成TSV汇总和manifest
5. ✓ **断点续传**: batch_optimizer.py支持通过scan_completed_smiles检测已完成的分子

### 数据流程
```
CSV输入 
  ↓
batch_optimizer.py (多搜索模式支持)
  ↓
{smiles}_{run_id}.json (优化树) + {smiles}_{run_id}_topK.csv (结果)
  ↓
batch_results.json (汇总)
  ↓
run_mcts_ablations.sh (evaluation) 
  ↓
run_summary.tsv / run_commands.tsv
```

### 模型和数据路径
- **模型**: `mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-*`
- **数据**: `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`
- **配置**: `mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml`

---

## 7. 后续的实验设计建议

根据现有脚本结构，新的实验应该：

1. **遵循命名规范**:
   - 脚本名: `run_*.sh`
   - 输出目录: `output/{category}/{timestamp}/`
   - 结果文件: `{smiles_prefix}_{run_id}_{suffix}.{ext}`

2. **输出结构**:
   - 每个实验单位（task/variant/seed）一个独立目录
   - 包含 json、csv、log三类文件
   - 生成 manifest.json 记录元数据
   - 生成 summary.tsv 用于快速查看

3. **参数传递**:
   - 优先使用环境变量而非命令行参数
   - 在脚本开头定义所有可配置项
   - 使用 `:=` 或 `:-` 提供默认值

4. **扩展batch_optimizer.py**:
   - 新参数应保留向后兼容性
   - 添加到parse_args()中
   - 在main_log_file中记录所有参数
   - 在output json中保存搜索配置

---

## 参考文件位置

- **主脚本**: `scripts/runners/run_mcts_ablations.sh` (L1-379)
- **批量优化**: `scripts/optimization/batch_optimizer.py` (L1-685)
- **汇总脚本**: `scripts/evaluation/summarize_mcts_ablation_runs.py`
- **示例输出**: `output/paper/ablations/20260413_113451/`

