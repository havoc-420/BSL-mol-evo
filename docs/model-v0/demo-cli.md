# Train

```bash
CUDA_VISIBLE_DEVICES=0 python mol_evo/train_v0.py --max-pairs 50 --epochs 20
CUDA_VISIBLE_DEVICES=0 python mol_evo/train_v0.py --max-pairs 2000 --epochs 50

CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --epochs 500 --batch-size 1024
```

## model type

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 50 --epochs 20 --model-type gcn_linear_linear

# transformer & visnet 的 lr 需要调低；否则「梯度爆炸」
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 50 --epochs 20 --model-type gcn_transformer_transformer -lr 0.001
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --epochs 100 --model-type gcn_transformer_transformer -lr 0.001

CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --epochs 100 -lr 0.001

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
cd /home/data2/rhj/project/mol_editor && python mol_evo/predict_v0_testset.py --model-path /home/data2/rhj/project/mol_editor/mol_evo/output/v0/gcn/train-20251012_230612-mu_change-30000-500/molecule_evolution_gcn_v0_mu_predictor.pth --model-dir /home/data2/rhj/project/mol_editor/mol_evo/output/v0/gcn/train-20251012_230612-mu_change-30000-500 --data-file /home/data2/rhj/project/mol_editor/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv -mpairs 30000 --seed 12420

cd /home/data2/rhj/project/mol_editor && python mol_evo/predict_v0_testset.py --model-path /home/data2/rhj/project/mol_editor/mol_evo/output/v0/gcn/train-20251012_230616-gap_change_pct-30000-500/molecule_evolution_gcn_v0_mu_predictor.pth --model-dir /home/data2/rhj/project/mol_editor/mol_evo/output/v0/gcn/train-20251012_230616-gap_change_pct-30000-500 --data-file /home/data2/rhj/project/mol_editor/mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv -mpairs 30000 --seed 42
```


## Train

```bash
CUDA_VISIBLE_DEVICES=1 python mol_evo/train_v0.py --data-file mol_evo/dataset/data/MMF/MMF-GNN_RTI_neg_Covered_by_Model.csv --target-property Pred_RTI_Negative_ESI --epochs 20
```
