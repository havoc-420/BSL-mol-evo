# Data

## 生成 evo-pair data

// TODO

## calculate_property_changes: 补充属性变化

```bash
cd mol_evo/dataset
python calculate_property_changes.py -i data/qm9-evo-pairs-step-1-pairs-127730.json --compact
```

得到 `data/qm9-evo-pairs-step-1-with-properties-pct.json`（默认）。

```bash
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor/mol_evo/dataset$ python calculate_property_changes.py -i data/qm9-evo-pairs-step-1-pairs-127730.json --compact
加载配对文件...
总共 127730 对分子
加载重原子文件...
加载重原子文件: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 87.94it/s]
计算属性变化...
计算属性变化: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 127730/127730 [04:48<00:00, 442.80it/s]
保存结果到 /home/data2/rhj/project/mol_editor/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json...
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
(mol-edit) rhj@wb-ubuntu-110-31-underground:~/project/mol_editor/mol_evo/dataset$ python extract_operation_config.py -i data/qm9-evo-pairs-step-1-with-properties-pct.json
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
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py \
  --epochs 200 \
  --learning-rate 0.0001 \
  --data-file '/home/data2/rhj/project/mol_editor/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json' \
  --max-pairs 120000 \
  --batch-size 512 \
  --target-property gap_change_pct
```

> 随后选择目标模型，或者使用 `--model-type` 指定模型类型。
> 可选模型类型：
> // TODO

# Predict - model test

## single test

```bash
python mol_evo/predict_v0.py \
  --model-path /home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth \
  --model-dir /home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200 \
  --smiles-from 'CC' \
  --smiles-to 'CO' \
  --atom-symbol '0' \
  --operation-type 'replace_atom'
```

## batch test

```bash
conda activate mol-edit && python mol_evo/predict_v0.py --model-path /home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200/last.pth --model-dir /home/data2/rhj/project/mol_editor/mol_evo/output/v0/MoleculeEvolutionVisnetLinearPredictor/train-20251123_192921-lumo_change-120000-200 --json-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.json --indices-file mol_evo/dataset/data/dataset_indices/indices_20251127_122155_seed42.json --use-test-indices --num-samples 10 --config-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct-config.yaml
```
