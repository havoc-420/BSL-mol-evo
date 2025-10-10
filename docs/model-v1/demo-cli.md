# Train

```bash
CUDA_VISIBLE_DEVICES=2 python mol_evo/train_nnconv.py --max-pairs 30000 --epochs 500 --prop

CUDA_VISIBLE_DEVICES=2 python mol_evo/train_model.py --max-pairs 20000 --epochs 500 --model-type rgcn
```

# Evaluate

```bash
python mol_evo/predict_nnconv.py --csv-file mol_evo/dataset/data/qm9-evo-pairs-step-1-with-properties.csv --row-index 1
```