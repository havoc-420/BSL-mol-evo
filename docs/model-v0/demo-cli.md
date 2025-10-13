# Train

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 50 --epochs 20
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 2000 --epochs 500

CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --epochs 500 -lr 0.001 --batch-size 1024

```


# Predict

1. 预测单个分子对
```bash
python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv --row-index 0
```

2. 批量预测
```bash
python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv --num-samples 200 --prediction-mode standardized

python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties-pct.csv --num-samples 30000 --prediction-mode standardized
```