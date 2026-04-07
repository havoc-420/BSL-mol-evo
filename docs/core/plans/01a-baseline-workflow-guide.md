# 01A：Baseline 操作手册（重构版）

> 本文档只回答一个问题：**当前仓库里的 baseline 到底该怎么跑，怎么验，怎么保存结果。**

---

## 1. 文档定位

本文只覆盖三条链路：

1. `v0` 单步数据与模型链路
2. `MO baseline` 的 `batch_optimizer.py` 运行链路
3. baseline 评估链路

以下内容不属于本文主线：

- `fragment_op`
- `OFO-frag`
- 新 planner
- path/value/RL

这些内容统一放在 `plans/02+` 中讨论。

---

## 2. 运行约定

### 2.1 工作目录

建议始终从项目根目录运行：

```bash
cd /Users/havoc420/Documents/Projects/whu/mol-ofo
```

### 2.2 环境

默认使用：

```bash
conda activate mol-opt-evo
```

### 2.3 运行原则

- 尽量显式指定模型权重和配置文件
- 重要运行必须保存 CLI 命令和输出目录
- baseline 运行不要夹带新路线实验参数

---

## 3. baseline 最小闭环

推荐的最小闭环顺序：

1. 准备 pair 数据与 `-config.yaml`
2. 训练一个 `v0` checkpoint
3. 用 `predict_v0.py` 做单步 sanity check
4. 准备固定评测 CSV
5. 用 `batch_optimizer.py` 跑 BFS 或 MCTS baseline
6. 用 `utils/evaluate_batch_mo.py` 和 `utils/evaluate_csv_results.py` 做汇总

---

## 4. `v0` 数据准备

### 4.1 生成 pair 数据

```bash
python mol_evo/dataset/extract_evolution_pairs.py --csv <your_input_csv>
```

调试预览：

```bash
python mol_evo/dataset/extract_evolution_pairs.py --csv <your_debug_csv> --mode preview_with_file
```

### 4.2 计算属性变化

```bash
python mol_evo/dataset/calculate_property_changes.py -i mol_evo/dataset/data/<pairs_file>.json --compact
```

### 4.3 提取操作配置

```bash
python mol_evo/dataset/extract_operation_config.py -i mol_evo/dataset/data/<pairs_with_properties>.json
```

这一步会生成同名 `-config.yaml`，后续训练、预测、搜索都依赖它。

---

## 5. `v0` 模型训练

推荐模板：

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

训练后至少确认：

- `last.pth`
- 训练日志
- 配置快照
- 数据文件与 `-config.yaml` 是否匹配

---

## 6. 单步预测 sanity check

### 6.1 单条预测

```bash
python mol_evo/predict_v0.py \
  --model-path mol_evo/output/v0/<model_type>/<train_dir>/last.pth \
  --model-dir mol_evo/output/v0/<model_type>/<train_dir> \
  --smiles-from 'COC' \
  --smiles-to 'OCCO' \
  --atom-symbol 'O' \
  --operation-type 'add_atom'
```

### 6.2 批量抽样预测

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

这一阶段只做 sanity check，不拿它代替 MO 对比。

---

## 7. MO baseline 正式入口

### 7.1 默认评测入口

继续使用固定评测 CSV 与固定索引范围作为 baseline 对照子集。

### 7.2 BFS 推荐模板

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

### 7.3 MCTS 推荐模板

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

---

## 8. baseline 评估链

### 8.1 逐分子评估

```bash
python utils/evaluate_batch_mo.py \
  --result-dir mol_evo/output/evo-mo/<batch_run_dir> \
  --target-prop lumo \
  --direction decrease \
  --item-size 10
```

### 8.2 统计汇总

```bash
python utils/evaluate_csv_results.py \
  --csv-file mol_evo/output/evo-mo/<batch_run_dir>/.evaluation_results_lumo_decrease/<eval_run_dir>/batch_evaluation_results.csv \
  --target-prop lumo \
  --direction decrease
```

推荐把 `utils/` 下的评估链视为正式 baseline 主入口，旧版简化脚本只保留为历史兼容。

---

## 9. 正式运行时至少要保存什么

每一次正式 baseline 运行，至少保留：

- 输入 CSV 或其版本信息
- 模型 checkpoint 路径
- 配置 YAML 路径
- 完整 CLI 命令
- 输出目录
- `batch_results.json`
- 日志文件
- 评估结果 CSV 与统计摘要

---

## 10. 本手册的边界

后续如果出现：

- `fragment_op`
- `semantic_step`
- 新 planner
- path/value

一律不要继续往本文追加，统一写回 `plans/02+`。
