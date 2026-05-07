# Step Budget 消融实验进度

> 最后更新：2026-04-28 15:50
> 实验脚本：`mol_evo/scripts/runners/run_step_budget_sweep.sh`
> Conda 环境：**`mol-edit`**（注意不是 `mol-ofo`）
> 运行机：`bsl-11-rhj`

## 1. 任务背景

在 `evo-mo` 分子属性优化任务中，做 **step_budget（MCTS 展开步数预算）** 消融：
固定 `num_simulations=1600`，扫描不同 `step_budget` 值，观察 `avg|Δproperty|` 随预算的变化曲线，验证"小预算即可逼近全量搜索效果"的假设。

### 实验变量

| 维度 | 取值 |
|---|---|
| 基准组 | `num_simulations=1600, step_budget=None`（不限） |
| 扫描组 | `num_simulations=1600, step_budget ∈ {...}` |
| 任务 | `lumo_up`（主）、`homo_down`、`gap_down`、`mu_up`（后续） |
| Preset | `official`：`max_depth=10, max_branching=20`，50 个分子 |
| 数据集 | `dataset/eval-data/20251205_131636/qm9_test_molecules.csv` |

## 2. 环境问题修复清单（已完成）

| # | 问题 | 定位 | 处理 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'mol_evo.modules.FragNet.fragnet'` | `mol_evo/modules/FragNet/` 为空目录，子包 `fragnet/` 丢失 | `cp -a mol_evo-tmp/modules/FragNet/. mol_evo/modules/FragNet/` |
| 2 | `FileNotFoundError: dataset/eval-data/20251205_131636/qm9_test_molecules.csv` | `mol_evo/dataset/eval-data/` 目录不存在 | `ln -s mol_evo-tmp/dataset/eval-data → mol_evo/dataset/eval-data` |
| 3 | `FileNotFoundError: output/v0/.../last.pth` | `mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/` 下只有 4 个残缺 `train-*`，权重 `last.pth` 全部缺失 | 备份原目录为 `.bak_20260428_141439`，改软链到 `mol_evo-tmp/output/v0/MoleculeEvolutionVisnetLinearPredictor`（22 个完整 `train-*`）；同时补软链 `MoleculeEvolutionVisnetLinearPredictorV01` |

## 3. 脚本改动清单（已完成）

文件：`mol_evo/scripts/runners/run_step_budget_sweep.sh`

| 改动 | 说明 |
|---|---|
| `STEP_BUDGET_VALUES_CSV=none` 语义 | 传 `none`/`NONE` → 清空扫描数组，仅跑 baseline 摸底（不再依赖空格 hack） |
| CSV 过滤空白元素 | 避免 `--step-budget  ` 之类非法参数 |
| 空数组安全 | `"${STEP_BUDGET_VALUES[@]:-}"`、循环前 `${#arr[@]} -eq 0` 判空跳过，兼容 `set -u` |
| 汇总解析修复 | 原版 `r.get("score", r.get("property_change"))` 在新版 JSON 下全部为 `None` → `avg_top1=nan`。修复为从 `topk_results.topK_results[*].property_value` 减去 `initial_property` 计算 `change`，并新增列 `avg\|Δprop\|`（方向无关，绝对值）、`avgΔprop`（带符号） |

## 4. Baseline 摸底结果（2026-04-28）

命令（5 分子，official 搜索空间）：

```bash
CONDA_ENV=mol-edit CUDA_VISIBLE_DEVICES=0 \
SWEEP_PRESET=official TASKS=lumo_up \
END_INDEX=5 \
STEP_BUDGET_VALUES_CSV=none \
bash mol_evo/scripts/runners/run_step_budget_sweep.sh
```

输出目录：`mol_evo/output/evo-mo/step_budget_sweep_20260428_141740/`

### actual_steps 分布（lumo_up, 1600 sims, 5 mol）

| 统计量 | 值 |
|---|---|
| n | 5 |
| min | 14 |
| median | 96 |
| mean | 85.6 |
| p90 | 149 |
| max | 149 |
| 明细 | `[14, 27, 96, 142, 149]` |

### 属性改善（topK 最佳）

| 指标 | 值 |
|---|---|
| avg\|Δlumo\| | 4.6638 |
| avgΔlumo | +4.6638（全部正向抬升，与 `lumo_up` 目标一致） |
| 单分子耗时 | ~307 s/mol（5 mol 总耗时 1538 s） |

### 关键观察

- `1600 sims` 下典型 `actual_steps` 只有 ~100，说明 **MCTS 很快就触达深度上限或收敛**，step_budget ≥ 200 基本等价于 baseline
- 5 分子太少，max=149 是巧合；50 分子正式跑时 max 可能冲到 200–400，设置上限要留余量

## 5. 正式 sweep 计划（待执行）

### 推荐命令

```bash
cd /home/rhj/projects/mol_opt/mol-ofo

# 方案 A（6 个点，约 30h）
CONDA_ENV=mol-edit CUDA_VISIBLE_DEVICES=0 \
SWEEP_PRESET=official TASKS=lumo_up \
STEP_BUDGET_VALUES_CSV=10,25,50,100,200,400 \
bash mol_evo/scripts/runners/run_step_budget_sweep.sh

# 方案 B（5 个点，约 25h，推荐）
CONDA_ENV=mol-edit CUDA_VISIBLE_DEVICES=0 \
SWEEP_PRESET=official TASKS=lumo_up \
STEP_BUDGET_VALUES_CSV=10,25,50,100,200 \
bash mol_evo/scripts/runners/run_step_budget_sweep.sh
```

建议在 `tmux` 中运行，防 ssh 断线：

```bash
tmux new -s sweep_lumo
# ... 粘贴上面命令 ...
# Ctrl-B D 脱离
tmux a -t sweep_lumo   # 恢复
```

### 预算点设计理由

| budget | 用途 |
|---|---|
| 10 | 严重截断（<< median），观察低预算下性能下降程度 |
| 25 | 中度截断 |
| 50 | 接近 median/2 |
| 100 | 接近 median，理论上轻微截断 |
| 200 | 略高于当前 max，近似无预算 |
| 400 | 明确无预算上限对照（方案 A 才包含）|

### 预期曲线

`avg|Δprop|` 应随 budget 单调上升，在 ~100–200 之间进入平台期。若平台开始点 < 100，说明该任务搜索空间更小，后续任务可把预算下移。

## 6. 后续任务（待启动）

1. [ ] `lumo_up` 正式 sweep（5–6 个预算点，~25 h）
2. [ ] 其他任务摸底：`homo_down`, `gap_down`, `mu_up`（5 mol each, `STEP_BUDGET_VALUES_CSV=none`）
3. [ ] 根据各任务 `actual_steps` 分布分别确定预算区间
4. [ ] 多任务正式 sweep
5. [ ] 画 `budget → avg|Δprop|` 曲线图，找"拐点预算"

## 7. 重要文件索引

| 类型 | 路径 |
|---|---|
| sweep 脚本 | `mol_evo/scripts/runners/run_step_budget_sweep.sh` |
| 批优化入口 | `mol_evo/scripts/optimization/batch_optimizer.py` |
| 评估数据集 | `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`（软链→`mol_evo-tmp/...`）|
| 模型权重根目录 | `mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/`（软链→`mol_evo-tmp/...`，含 22 个 `train-*`）|
| baseline 摸底结果 | `mol_evo/output/evo-mo/step_budget_sweep_20260428_141740/` |
| 原残缺权重目录备份 | `mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor.bak_20260428_141439/` |

## 8. 已知遗留 / 注意事项

- 全部资源通过软链接指向 `mol_evo-tmp/`，**不要删除 `mol_evo-tmp/` 目录**，否则 sweep 将全部失败
- `CONDA_ENV` 必须是 `mol-edit`（过去文档里的 `mol-ofo` 是错的）
- sweep 日志末尾自动打印的"推荐 STEP_BUDGET_VALUES_CSV" 下限偏激进（=min），实际使用按本文 §5 的设计为准
