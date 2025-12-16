# Train

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 50 --epochs 20
CUDA_VISIBLE_DEVICES=1 python mol_evo/train_v0.py --max-pairs 200 --epochs 20 -lr 0.0001

CUDA_VISIBLE_DEVICES=4 python mol_evo/train_v0.py --max-pairs 2000 --epochs 50

CUDA_VISIBLE_DEVICES=4 python mol_evo/train_v0.py --max-pairs 30000 --batch-size 1024 --epochs 200
```

## model type

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 50 --epochs 20 --model-type gcn_linear_linear

# transformer & visnet 的 lr 需要调低；否则「梯度爆炸」
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 50 --epochs 20 --model-type gcn_transformer_transformer -lr 0.001
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --epochs 100 --model-type gcn_transformer_transformer -lr 0.001

CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --epochs 200 -lr 0.001
```

# Predict

1. 预测单个分子对

```bash
python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv --row-index 0
```

2. 批量预测

```bash
python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv --num-samples 200

python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv --num-samples 30000
```

3. same test

```bash
cd /home/rhj/projects/mol_opt/mol-ofo && python mol_evo/predict_v0_testset.py --model-path /home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/gcn/train-20251012_230612-mu_change-30000-500/molecule_evolution_gcn_v0_mu_predictor.pth --model-dir /home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/gcn/train-20251012_230612-mu_change-30000-500 --data-file /home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv -mpairs 30000 --seed 12420

cd /home/rhj/projects/mol_opt/mol-ofo && python mol_evo/predict_v0_testset.py --model-path /home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/gcn/train-20251012_230616-gap_change_pct-30000-500/molecule_evolution_gcn_v0_mu_predictor.pth --model-dir /home/rhj/projects/mol_opt/mol-ofo/mol_evo/output/v0/gcn/train-20251012_230616-gap_change_pct-30000-500 --data-file /home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv -mpairs 30000 --seed 42
```

# Train

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --epochs 200 -lr 0.0001 --data-file /home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-v0.json
```

# Train - homo/lomo/gap

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --target-property homo_change_pct -lr 0.0001 --epochs 300 --batch-size 256
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --target-property homo_change -lr 0.0001 --epochs 300 --batch-size 256
CUDA_VISIBLE_DEVICES=4 python mol_evo/train_v0.py --max-pairs 30000 --target-property lumo_change_pct -lr 0.0001 --epochs 200 --batch-size 512
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --target-property gap_change_pct -lr 0.0001 --epochs 200 --batch-size 512
```

# Train - new step-1 12w

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --epochs 200 -lr 0.0001 --data-file /home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/tmp/qm9-evo-pairs-step-1-pairs-v0-81005-with-properties-pct.json --max-pairs 80000 --batch-size 64 --target-property lumo_change_pct
```

## Test

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --epochs 50 -lr 0.001 --data-file /home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/tmp/qm9-evo-pairs-step-1-pairs-v0-81005-with-properties-pct.json --max-pairs 2000 --batch-size 32
```
