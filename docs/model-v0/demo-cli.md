# Train

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_v0_model.py --max-pairs 50 --epochs 20
```


# Predict

```bash
python mol_evo/predict_v0.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 0
```