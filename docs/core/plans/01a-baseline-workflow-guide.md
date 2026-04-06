# 01A：Baseline 操作手册（正式版）

> 本文档用于把历史 `workflow-cli.md` 中可运行、可复用的命令，整理成当前项目的正式 baseline 操作手册。
>
> 它的目标不是记录所有历史实验，而是回答：**当前 v0 / MO baseline 到底该怎么跑、怎么验、怎么保存结果。**

---

## 1. 文档定位

本手册覆盖三条与 baseline 最相关的链路：

1. **v0 数据准备**：pair 数据、属性变化、操作配置
2. **v0 单步模型**：训练与单步/批量预测
3. **MO baseline**：`batch_optimizer.py` 的 BFS / MCTS 运行方式

以下内容暂不作为本手册主线：

- 片段级 `fragment_op` 新路线
- `OFO-frag`
- 新 planner / value / policy
- IC50 专用分支

这些内容应进入 `plans/02+` 的后续阶段文档中处理。

---

## 2. 运行约定

### 2.1 工作目录

建议始终从项目根目录运行：

```bash
cd /Users/havoc420/Documents/Projects/whu/mol-ofo
```

### 2.2 环境

当前工作区已约定使用：

```bash
conda activate mol-opt-evo
```

### 2.3 文档与历史命令关系

`docs/models/model-v0/workflow-cli.md` 仍建议保留，但应视为：

- **历史命令记录**
- **实验痕迹备忘**
- **路径与机器环境的原始素材**

本文件则是针对当前仓库整理后的**正式 baseline 跑法**。

---

## 3. baseline 总流程

### 3.1 v0 单步建模 baseline

顺序为：

1. 生成 evo-pair 数据
2. 计算属性变化
3. 提取操作配置 YAML
4. 训练 v0 模型
5. 做单步预测验证

### 3.2 MO 搜索 baseline

顺序为：

1. 准备评测 CSV
2. 指定模型权重、模型目录、配置文件
3. 运行 `batch_optimizer.py`
4. 产出优化树 JSON、Top-k CSV、批量汇总 JSON、日志
5. 用评估脚本做结果汇总

---

## 4. 数据准备：evo-pair 数据

### 4.1 生成 evo-pair 数据

历史工作流中的入口是：

```bash
python mol_evo/dataset/extract_evolution_pairs.py --csv <your_input_csv>
```

调试时可用：

```bash
python mol_evo/dataset/extract_evolution_pairs.py --csv <your_debug_csv> --mode preview_with_file
```

### 4.2 计算属性变化

在项目根目录执行即可：

```bash
python mol_evo/dataset/calculate_property_changes.py -i mol_evo/dataset/data/<pairs_file>.json --compact
```

产物通常是：

- 带属性变化的 JSON 数据文件

### 4.3 提取操作配置

```bash
python mol_evo/dataset/extract_operation_config.py -i mol_evo/dataset/data/<pairs_with_properties>.json
```

产物通常是：

- 同名 `-config.yaml`

这个 YAML 很关键，因为训练、预测、MO baseline 都依赖它来恢复：

- 操作类型集合
- 原子类型集合

---

## 5. v0 模型训练 baseline

`train_v0.py` 的真实 CLI 入口至少包括：

- `--data-file`
- `--max-pairs`
- `--epochs`
- `--batch-size`
- `--target-property`
- `--seed`
- `--learning-rate`
- `--model-type`
- `--config-file`

一个推荐的正式写法如下：

```bash
python mol_evo/train_v0.py \
  --data-file mol_evo/dataset/data/<pairs_with_properties>.json \
  --max-pairs 120000 \
  --epochs 200 \
  --batch-size 512 \
  --target-property gap_change_pct \
  --learning-rate 0.0001 \
  --seed 42 \
  --model-type visnet_linear_linear
```

### 5.1 训练产物

训练输出目录位于：

- `mol_evo/output/v0/<model_type>/train-<timestamp>-<target>-<max_pairs>-<epochs>/`

建议至少关注：

- `last.pth`
- 配置快照
- 训练日志

### 5.2 注意事项

- 建议显式提供 `--model-type`，避免交互式选择影响复现
- `--config-file` 可直接指定模型配置 YAML
- 若数据文件变化，应同步确认对应 `-config.yaml` 是否匹配

---

## 6. v0 单步预测 baseline

### 6.1 单条预测

`predict_v0.py` 的真实 CLI 支持：

- `--model-path`
- `--model-dir`
- `--smiles-from`
- `--smiles-to`
- `--atom-symbol`
- `--operation-type`

示例：

```bash
python mol_evo/predict_v0.py \
  --model-path mol_evo/output/v0/<model_type>/<train_dir>/last.pth \
  --model-dir mol_evo/output/v0/<model_type>/<train_dir> \
  --smiles-from 'COC' \
  --smiles-to 'OCCO' \
  --atom-symbol 'O' \
  --operation-type 'add_atom'
```

### 6.2 批量预测

`predict_v0.py` 也支持：

- `--json-file`
- `--indices-file`
- `--use-test-indices`
- `--num-samples`
- `--sample-method`
- `--config-file`

示例：

```bash
python mol_evo/predict_v0.py \
  --model-path mol_evo/output/v0/<model_type>/<train_dir>/last.pth \
  --model-dir mol_evo/output/v0/<model_type>/<train_dir> \
  --json-file mol_evo/dataset/data/<pairs_with_properties>.json \
  --indices-file mol_evo/dataset/data/dataset_indices/<indices_file>.json \
  --use-test-indices \
  --num-samples 40 \
  --config-file mol_evo/dataset/data/<pairs_with_properties>-config.yaml \
  --sample-method sequential
```

### 6.3 用途

这一阶段主要用于确认：

- 模型 checkpoint 是否可用
- 单步属性变化预测是否方向合理
- 配置文件和模型目录是否配套

---

## 7. MO baseline：`batch_optimizer.py`

这是当前搜索式分子优化 baseline 的正式入口。

### 7.1 输入要求

至少需要一个 CSV，包含：

- `smiles`
- 与 `--target-property` 同名的属性列，例如 `lumo`、`homo`

当前 Phase 1 已约定的默认评测子集为：

- `mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv`
- 默认通过 `--start-index 0 --end-index 50` 选择前 50 个 case

### 7.2 BFS baseline 推荐命令

```bash
python mol_evo/scripts/batch_optimizer.py \
  --input-csv mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv \
  --target-property lumo \
  --direction decrease \
  --optimization-mode sub \
  --max-depth 4 \
  --max-branching 8 \
  --pruning-patience 2 \
  --logp-min 0.0 \
  --logp-max 5.0 \
  --logp-patience 3 \
  --topK 20 \
  --batch-size 10 \
  --start-index 0 \
  --end-index 50 \
  --search-mode bfs \
  --model-path mol_evo/output/v0/<model_type>/<train_dir>/last.pth \
  --model-dir mol_evo/output/v0/<model_type>/<train_dir> \
  --config-file mol_evo/dataset/data/<pairs_with_properties>-config.yaml
```

### 7.3 MCTS baseline 推荐命令

```bash
python mol_evo/scripts/batch_optimizer.py \
  --input-csv mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv \
  --target-property lumo \
  --direction decrease \
  --optimization-mode sub \
  --max-depth 10 \
  --max-branching 20 \
  --pruning-patience 3 \
  --logp-min -0.5 \
  --logp-max 6.0 \
  --logp-patience 5 \
  --topK 20 \
  --batch-size 10 \
  --start-index 0 \
  --end-index 50 \
  --search-mode mcts \
  --num-simulations 800 \
  --exploration-weight 2.0 \
  --model-path mol_evo/output/v0/<model_type>/<train_dir>/last.pth \
  --model-dir mol_evo/output/v0/<model_type>/<train_dir> \
  --config-file mol_evo/dataset/data/<pairs_with_properties>-config.yaml
```

### 7.4 使用建议

- **BFS**：适合作为默认 baseline
- **MCTS**：适合作为对照实验，不建议在 Phase 1 大量调参
- 建议保持同一评测集、同一模型、同一 target property，再对比 BFS / MCTS

---

## 8. baseline 结果分析

当前更推荐的 baseline 评估链路其实在 `utils/` 下，而不是 `mol_evo/evaluate_batch_mo.py`。

### 8.1 第一步：从 `*_topK.csv` 生成逐分子评估结果

```bash
python utils/evaluate_batch_mo.py \
  --result-dir mol_evo/output/evo-mo/<batch_run_dir> \
  --target-prop lumo \
  --direction decrease \
  --item-size 10
```

这一步会读取 `batch_optimizer.py` 输出目录中的 `*_topK.csv`，并生成：

- `.evaluation_results_<target>_<direction>/<eval_run_dir>/evaluation_config.json`
- `.evaluation_results_<target>_<direction>/<eval_run_dir>/batch_evaluation_results.csv`

### 8.2 第二步：对 `batch_evaluation_results.csv` 做汇总统计

```bash
python utils/evaluate_csv_results.py \
  --csv-file mol_evo/output/evo-mo/<batch_run_dir>/.evaluation_results_lumo_decrease/<eval_run_dir>/batch_evaluation_results.csv \
  --target-prop lumo \
  --direction decrease
```

这一步会继续输出：

- `statistics_report_<timestamp>.txt`
- `statistics_summary_<timestamp>.json`
- `best_results_<timestamp>.csv`
- `evaluation_plots_<target>_<timestamp>.png`
- `evaluation_results_<target>_<timestamp>.json`

### 8.3 注意事项

需要明确几点：

- `utils/evaluate_batch_mo.py` 是**当前更完整、也更贴近真实工作流**的评估入口
- `mol_evo/evaluate_batch_mo.py` 更适合视为**旧版、简化版脚本**，不建议继续作为 Phase 1 的首选入口
- 真实性质重评估依赖 `utils` 侧额外依赖（如 `pyscf`、`ase`、`pytdc`），仓库中已有环境文件 `utils/envs/mol-opt-tdc-environment.yml`
- 因此，Phase 1 目前并不是“没有评估模块”，而是**评估模块已存在，但还没有完全统一到正式 baseline 文档和环境约定里**

---

## 9. 当前推荐的 baseline 最小闭环

如果只保留最关键的最小闭环，建议顺序是：

1. 生成或确认一份 pair 数据 + `-config.yaml`
2. 训练一个 v0 模型 checkpoint
3. 用 `predict_v0.py` 做单步 sanity check
4. 准备一个固定评测 CSV
5. 用 `batch_optimizer.py` 跑 BFS baseline
6. 用 `utils/evaluate_batch_mo.py` 生成 `batch_evaluation_results.csv`
7. 用 `utils/evaluate_csv_results.py` 生成统计摘要与图表

---

## 10. 不建议继续放在本阶段主线里的内容

以下内容可以保留为历史记录，但不建议再混入 Phase 1 正式 baseline 手册：

- IC50 专用链路
- 各种机器绝对路径
- 某次实验的硬编码 checkpoint
- 没有对应仓库文件的旧分析脚本命令
- 新路线相关的 `fragment_op` / planner / value 试验命令

---

## 11. 和历史 `workflow-cli.md` 的关系

建议的维护方式：

- `workflow-cli.md`：保留原始实验命令与历史记录
- `01A`：保留当前正式 baseline 运行方式
- 后续新路线文档：放入 `plans/02+`

这样做的好处是：

- 不丢历史经验
- 不让 baseline 文档继续发散
- 让后续对比实验有统一入口
