# Data

## 生成 evo-pair data: task2

```bash
## 1.
python mol-ofo/mol_evo/dataset/extract_evolution_pairs.py --csv /home/xxx/projects/mol_opt/data/gdcsv2/cell/cell_687787.csv

## 2.

```

### 🐛 debug

```bash
python mol-ofo/mol_evo/dataset/extract_evolution_pairs.py --csv /home/xxx/projects/mol_opt/data/gdcsv2/cell/cell_687787_debug_v1.csv --mode preview_with_file
```

## calculate_property_changes: 补充属性变化

```bash
cd mol_evo/dataset
python calculate_property_changes.py -i data/qm9-evo-pairs-step-1-pairs-127730.json --compact
```

得到 `data/qm9-evo-pairs-step-1-with-properties-pct.json`（默认）。

```bash
(mol-edit) xxx@wb-ubuntu-110-31-underground:~/project/mol_editor/mol_evo/dataset$ python calculate_property_changes.py -i data/qm9-evo-pairs-step-1-pairs-127730.json --compact
加载配对文件...
总共 127730 对分子
加载重原子文件...
加载重原子文件: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 87.94it/s]
计算属性变化...
计算属性变化: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 127730/127730 [04:48<00:00, 442.80it/s]
保存结果到 /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json...
完成!

属性变化统计信息:
A: 有效值 127730, 平均变化 -4.957485, 标准差 5519.193873
B: 有效值 127730, 平均变化 -0.037508, 标准差 1.900756
C: 有效值 127730, 平均变化 -0.029945, 标准差 1.104485
mu: 有效值 127730, 平均变化 0.004469, 标准差 1.238474
alpha: 有效值 127730, 平均变化 0.456858, 标准差 5.265870
homo: 有效值 127730, 平均变化 0.002794, 标准差 0.475921
lumo: 有效值 127730, 平均变化 -0.007836, 标准差 0.978238
gap: 有效值 127730, 平均变化 -0.010636, 标准差 1.031918
r2: 有效值 127730, 平均变化 17.471499, 标准差 299.856731
zpve: 有效值 127730, 平均变化 0.026383, 标准差 0.716107
U0: 有效值 127730, 平均变化 -72.522067, 标准差 619.428492
U: 有效值 127730, 平均变化 -72.520268, 标准差 619.428748
H: 有效值 127730, 平均变化 -72.520268, 标准差 619.428749
G: 有效值 127730, 平均变化 -72.525011, 标准差 619.431363
```

## 总结 evo-pairs config

```bash
cd mol_evo/dataset
python extract_operation_config.py -i data/qm9-evo-pairs-step-1-with-properties-pct.json
```

得到 `data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml`（默认）。

```bash
(mol-edit) xxx@wb-ubuntu-110-31-underground:~/project/mol_editor/mol_evo/dataset$ python extract_operation_config.py -i data/qm9-evo-pairs-step-1-with-properties-pct.json
配置已保存到: data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml
操作类型数量: 12
原子类型数量: 4

操作类型统计:
  replace_atom: 59090
  remove_form_triple_bond: 9366
  form_triple_bond: 9366
  remove_form_double_bond: 7491
  form_double_bond: 7491
  add_stereo: 7415
  remove_add_stereo: 7415
  add_atom: 6572
  form_ring: 6497
  remove_form_ring: 6497

配置提取完成!
```

# Train

// TODO 后面将 `mol_evo` 作为一个单独的项目，这里的 cli 相对路径当时有一些不合适。

```bash
cd mol_evo/..
CUDA_VISIBLE_DEVICES=0 python mol_evo/train_v0.py \
  --epochs 200 \
  --learning-rate 0.0001 \
  --data-file '/home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json' \
  --max-pairs 120000 \
  --batch-size 512 \
  --target-property homo_change
```

> 随后选择目标模型，或者使用 `--model-type` 指定模型类型。
> 可选模型类型：
> // TODO

# Predict - model test

## single test

```bash
python mol_evo/predict_v0.py \
  --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth \
  --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200 \
  --smiles-from 'COC' \
  --smiles-to 'OCCO' \
  --atom-symbol 'O' \
  --operation-type 'add_atom'
```

## batch test

```bash
conda activate mol-edit && python mol_evo/predict_v0.py --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200 --json-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json --indices-file mol_evo/dataset/data/dataset_indices/indices_20251127_122155_seed42.json --use-test-indices --num-samples 40 --config-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml --sample-method sequential
```

## batch predict

### eval data genarate

```bash
python mol_evo/dataset/build_ab_pairs.py
```

### core predict

#### 1. Task-1: lumo/homo optimize

1. bfs
base single cli example in [mo-cli.md](../../mo/mo-cli.md).

```bash
## 1. lumo (decrease)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=0 python mol_evo/scripts/batch_optimizer.py --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv  --target-property lumo --start-index 0 --end-index 50 --max-depth 4 --direction decrease

## 2. lumo (increase)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=1 python mol_evo/scripts/batch_optimizer.py --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv  --target-property lumo --start-index 0 --end-index 50 --max-depth 4 --direction increase

## 3. homo (decrease)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=0 python mol_evo/scripts/batch_optimizer.py --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv  --target-property homo --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200/last.pth --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200 --start-index 0 --end-index 50 --max-depth 4 --direction decrease

## 4. homo (increase)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=1 python mol_evo/scripts/batch_optimizer.py --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv  --target-property homo --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200/last.pth --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200 --start-index 0 --end-index 50 --max-depth 4 --direction increase

## 5. gap (decrease)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=0 python mol_evo/scripts/batch_optimizer.py --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv  --target-property gap --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251124_044027-gap_change-120000-200/last.pth --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251124_044027-gap_change-120000-200 --start-index 0 --end-index 50 --max-depth 4 --direction decrease

## 6. gap (increase)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=1 python mol_evo/scripts/batch_optimizer.py --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv  --target-property gap --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251124_044027-gap_change-120000-200/last.pth --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251124_044027-gap_change-120000-200 --start-index 0 --end-index 50 --max-depth 4 --direction increase
```

2. mcts

```bash
## 1. homo (increase)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=0 python mol_evo/scripts/batch_optimizer.py   --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv      --target-property homo   --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200/last.pth   --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200   --start-index 0 --end-index 50  --max-depth 6 --max-branching 20 --direction increase   --search-mode mcts   --num-simulations 1600   --exploration-weight 2 --pruning-patience 3 --logp-min -0.5 --logp-max 6 --logp-patience 5

## 2. homo (decrease)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=1 python mol_evo/scripts/batch_optimizer.py   --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv      --target-property homo   --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200/last.pth   --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251127_121257-homo_change-120000-200   --start-index 0 --end-index 50  --max-depth 6 --max-branching 20 --direction decrease   --search-mode mcts   --num-simulations 1600   --exploration-weight 2 --pruning-patience 3 --logp-min -0.5 --logp-max 6 --logp-patience 5

## 3. lumo (decrease)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=0 python mol_evo/scripts/batch_optimizer.py   --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv      --target-property lumo   --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth   --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200   --start-index 0 --end-index 50  --max-depth 6 --max-branching 20 --direction decrease   --search-mode mcts   --num-simulations 1600   --exploration-weight 2 --pruning-patience 3 --logp-min -0.5 --logp-max 6 --logp-patience 5

## 4. lumo (increase)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=1 python mol_evo/scripts/batch_optimizer.py   --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv      --target-property lumo   --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth   --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200   --start-index 0 --end-index 50  --max-depth 6 --max-branching 20 --direction increase   --search-mode mcts   --num-simulations 1600   --exploration-weight 2 --pruning-patience 3 --logp-min -0.5 --logp-max 6 --logp-patience 5

## 5. gap (decrease)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=0 python mol_evo/scripts/batch_optimizer.py   --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv      --target-property gap   --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251124_044027-gap_change-120000-200/last.pth   --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251124_044027-gap_change-120000-200   --start-index 0 --end-index 50  --max-depth 6 --max-branching 20 --direction decrease   --search-mode mcts   --num-simulations 1600   --exploration-weight 2 --pruning-patience 3 --logp-min -0.5 --logp-max 6 --logp-patience 5

## 6. gap (increase)
conda activate mol-edit && CUDA_VISIBLE_DEVICES=1 python mol_evo/scripts/batch_optimizer.py   --input-csv /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/eval-data/20251205_131636/qm9_test_molecules.csv      --target-property gap   --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251124_044027-gap_change-120000-200/last.pth   --model-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251124_044027-gap_change-120000-200   --start-index 0 --end-index 50  --max-depth 6 --max-branching 20 --direction increase   --search-mode mcts   --num-simulations 1600   --exploration-weight 2 --pruning-patience 3 --logp-min -0.5 --logp-max 6 --logp-patience 5
```

#### 2. Task-2: ic50 optimize

```bash
# 1. single test
python mol_evo/core/evolution_optimizer_ic50.py

# 2. batch

## 2.1 cell_name: 909729
conda activate mol-edit && CUDA_VISIBLE_DEVICES=0 python mol_evo/scripts/batch_optimizer_ic50.py --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/SchNet_909729_checkpoints/0.9_N4_20260106_191804/best_model.pth --target-property ic50 --max-depth 3 --direction decrease --cell-name 909729

conda activate mol-edit && CUDA_VISIBLE_DEVICES=0 python mol_evo/scripts/batch_optimizer_ic50.py --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/SchNet_687787_checkpoints/0.9_N4_20260106_134954/best_model.pth --target-property ic50 --max-depth 3 --direction decrease --cell-name 687787

conda activate mol-edit && CUDA_VISIBLE_DEVICES=1 python mol_evo/scripts/batch_optimizer_ic50.py --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/SchNet_687800_checkpoints/0.9_N4_20260107_085835/best_model.pth --target-property ic50 --max-depth 6 --direction decrease --cell-name 687800

# PART-2: Cell: 687800 Cell: 906792 Cell: 684055 Cell: 908149

conda activate mol-edit && CUDA_VISIBLE_DEVICES=1 python mol_evo/scripts/batch_optimizer_ic50.py --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/SchNet_906792_checkpoints/0.9_N4_20260109_212701/best_model.pth --target-property ic50 --max-depth 6 --direction decrease --cell-name 906792

conda activate mol-edit && CUDA_VISIBLE_DEVICES=1 python mol_evo/scripts/batch_optimizer_ic50.py --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/SchNet_684055_checkpoints/0.9_N4_20260110_000235/best_model.pth --target-property ic50 --max-depth 6 --direction decrease --cell-name 684055

conda activate mol-edit && CUDA_VISIBLE_DEVICES=0 python mol_evo/scripts/batch_optimizer_ic50.py --model-path /home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/drp_ic50_p/SchNet_908149_checkpoints/0.9_N4_20260110_093217/best_model.pth --target-property ic50 --max-depth 4 --direction decrease --cell-name 908149

```

# Analyze MO Results

```bash
# 1. cal lumo/homo ... true value
conda activate mol-edit && python utils/evaluate_batch_mo.py --target-prop lumo --direction decrease --item-size 10 --result-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260308_112752 \
 --max-files 2 --item-size 2

## homo
conda activate mol-edit && python utils/evaluate_batch_mo.py --target-prop homo --direction increase --item-size 10 --result-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260408_042311

conda activate mol-edit && python utils/evaluate_batch_mo.py --target-prop homo --direction decrease --item-size 10 --result-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260408_042254

## gap
conda activate mol-edit && python utils/evaluate_batch_mo.py --target-prop gap --direction increase --item-size 10 --result-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260411_164651

conda activate mol-edit && python utils/evaluate_batch_mo.py --target-prop gap --direction decrease --item-size 10 --result-dir /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260411_164646


# 2. summary result csv
conda activate mol-opt-tdc &&  python utils/evaluate_csv_results.py --target-prop lumo --direction decrease --csv-file /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260308_010253/.evaluation_results_lumo_decrease/20260308_013106_size10_maxall/batch_evaluation_results.csv

conda activate mol-opt-tdc &&  python utils/evaluate_csv_results.py --target-prop lumo --direction increase --csv-file /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260330_014546/.evaluation_results_homo_increase/20260330_134457_size10_maxall/batch_evaluation_results.csv

conda activate mol-opt-tdc &&  python utils/evaluate_csv_results.py --target-prop homo --direction increase --csv-file /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260408_042311/.evaluation_results_homo_increase/20260410_022831_size10_maxall/batch_evaluation_results.csv

conda activate mol-opt-tdc &&  python utils/evaluate_csv_results.py --target-prop homo --direction decrease --csv-file /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260408_042254/.evaluation_results_homo_decrease/20260409_035514_size10_maxall/batch_evaluation_results.csv


## gap

```bash
conda activate mol-opt-tdc && python utils/evaluate_csv_results.py --target-prop gap --direction increase --csv-file /home/xxx/projects/mol_opt/mol-ofo/mol_evo/output/evo-mo/batch_optimization_20260411_164651/.evaluation_results_gap_increase/20260411_180006_size10_maxall/batch_evaluation_results.csv

conda activate mol-opt-tdc && python utils/evaluate_csv_results.py --target-prop gap --direction increase --csv-file /home/xxx/projects/mol_opt/CMOMO-master/results/20260413_035212-gap-up/.evaluation_results_gap_decrease/20260413_220403_size10_maxall/batch_evaluation_results.csv
```
