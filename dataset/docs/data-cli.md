# MOL_EVO_DEBUG

## step 1

```bash
clipython extract_evolution_pairs.py --debug-from "CNC=O" --debug-to "COC=O" --log DEBUG
```

## step 2

```bash
clipython extract_evolution_pairs.py --debug-from "C" --debug-to "CO" --log DEBUG
python extract_evolution_pairs.py --debug-from "N" --debug-to "CC" --log DEBUG
python extract_evolution_pairs.py --debug-from "C#C" --debug-to "C#CC" --log DEBUG
python extract_evolution_pairs.py --debug-from "CC#N" --debug-to "CC(C)C" --log DEBUG

#
python extract_evolution_pairs.py --debug-from "COCCC#N" --debug-to "C[NH+](C)CC#N" --log DEBUG
python extract_evolution_pairs.py --debug-from "COCCC#N" --debug-to "CO[C@@H](C)C#N" --log DEBUG
```

# Extract Evolution Pairs

## 基本用法

```bash
python extract_evolution_pairs.py --step 1 --max-pairs 100
```

## 断点续传

```bash
# 首次运行，保存检查点
python extract_evolution_pairs.py --step 2 --max-pairs 2005

# 如果任务中断，使用以下命令从检查点恢复
python extract_evolution_pairs.py --step 2 --resume
```

# Calculate Prop Changes

```bash
python calculate_property_changes.py -i data/qm9-evo-pairs-step-1-v2-100.json
python calculate_property_changes.py -i data/qm9-evo-pairs-step-1-v2-100.json -o data/output-step-1-v2.json --compact
```

# Filter Option Types

```python
python3 -c "import json; data = json.load(open('/home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-2-pairs-v0-17472.json')); ops = set(); [ops.add(op['operation']) for item in data for op in item['operations']]; print('\\n'.join(sorted(ops)))"
```
