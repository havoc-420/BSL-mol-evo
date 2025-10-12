# Train

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 50 --epochs 20
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 2000 --epochs 500

CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0.py --max-pairs 30000 --epochs 200 -lr 0.0001

```


# Predict

```bash
python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 0
```