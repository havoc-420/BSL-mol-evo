# Batch 在分子特征提取中的使用

## 概述

在 PyTorch Geometric (PyG)中，`batch`属性是一个重要的概念，特别是在处理多个图（分子）的批量数据时。在我们的分子编辑器项目中，[MoleculeFeatureExtractor](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn.py#L15-L53)模型支持使用`batch`属性来处理批量数据。

## Batch 的作用

1. **批量处理**: 当需要同时处理多个分子时，PyG 的 DataLoader 会自动创建`batch`属性，指示每个节点属于哪个图（分子）。
2. **高效计算**: 使用`batch`可以更好地利用 GPU 并行计算能力，提高训练和推理效率。
3. **正确的池化操作**: 通过`global_mean_pool(x, batch)`，可以正确地对每个分子的节点特征进行池化，得到每个分子的图级表示。

## 当前实现

在[MoleculeFeatureExtractor](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn.py#L15-L53)中，我们有以下处理逻辑：

```python
batch = getattr(data, 'batch', None)

# 全局池化获取图表示
if batch is not None:
    x = global_mean_pool(x, batch)
else:
    x = torch.mean(x, dim=0, keepdim=True)
```

这段代码会检查输入数据是否包含`batch`属性：

- 如果有`batch`属性，则使用`global_mean_pool`按照 batch 进行池化
- 如果没有`batch`属性，则对所有节点特征取平均

## 测试脚本中的问题

当前的测试脚本[test_v0_graph_feature_extractor.py](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/test_v0_graph_feature_extractor.py)只测试了单个分子的情况，没有使用 batch。这虽然可以验证基本功能，但没有完全覆盖模型的批量处理能力。

## 改进建议

为了更全面地测试模型，建议在测试中添加对 batch 处理的支持：

1. 创建多个分子的测试数据
2. 使用 PyG 的 DataLoader 创建批量数据
3. 验证模型在批量数据上的处理能力

这样可以确保模型在实际使用场景中的正确性和效率。

# 使用 mini‑batch 方式加速 GCN 训练

下面给出 **使用 mini‑batch 方式加速 GCN 训练** 的完整思路与代码改动要点。  
核心思路是把 **每个分子对 (from‑graph, to‑graph, edge‑feature, target)** 组织成一个 `Dataset`，再交给 **PyG 的 `DataLoader`** 进行批处理。这样：

- **一次前向传播可以并行计算多个图**（利用 `Batch.from_data_list` 自动拼接节点、边、`batch` 索引）[[1]]
- **边特征（每个图的操作向量）** 直接在 `collate_fn` 中堆叠成 `(B, edge_feature_dim)`，模型的 `forward` 已经支持批维度。
- 训练、验证、测试循环只需遍历 `DataLoader`，不再在 `for i in range(N)` 中逐个调用模型，显著提升 GPU 利用率。

---

## 1. 新建 `MoleculePairDataset`

```python
from torch.utils.data import Dataset

class MoleculePairDataset(Dataset):
    """返回 (from_graph, to_graph, edge_feature, target) 的数据集"""
    def __init__(self, from_list, to_list, edge_attrs, targets):
        assert len(from_list) == len(to_list) == len(edge_attrs) == len(targets)
        self.from_list = from_list
        self.to_list   = to_list
        self.edge_attrs = edge_attrs      # Tensor (N, edge_feature_dim)
        self.targets    = targets         # Tensor (N, 1)

    def __len__(self):
        return len(self.from_list)

    def __getitem__(self, idx):
        return (self.from_list[idx],
                self.to_list[idx],
                self.edge_attrs[idx],
                self.targets[idx])
```

---

## 2. 自定义 `collate_fn`

```python
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

def pair_collate(batch):
    """把若干 (from, to, edge, target) 合并成 batch"""
    from_list, to_list, edge_list, target_list = zip(*batch)

    # PyG 自动把多个 Data 拼成一个 Batch
    from_batch = Batch.from_data_list(list(from_list))
    to_batch   = Batch.from_data_list(list(to_list))

    # edge 特征和目标直接 stack
    edge_batch = torch.stack(edge_list, dim=0)          # (B, edge_dim)
    target_batch = torch.stack(target_list, dim=0)      # (B, 1)

    return from_batch, to_batch, edge_batch, target_batch
```

> `Batch.from_data_list` 能把不同大小的图拼接成统一的 `batch`，并在内部生成 `batch` 索引供全局池化使用[[2]]。

---

## 3. 构造 DataLoader（可调 `batch_size`、`num_workers`）

```python
train_dataset = MoleculePairDataset(
        from_data_list, to_data_list, edge_attrs, target_features)

train_loader = DataLoader(
        train_dataset,
        batch_size=64,               # 根据显存自行调节
        shuffle=True,
        num_workers=4,
        collate_fn=pair_collate,
        pin_memory=True)
```

同理为 **验证 / 测试** 再各建一个 `DataLoader`（`shuffle=False`）。

---

## 4. 训练循环（核心改动）

```python
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for from_batch, to_batch, edge_batch, target_batch in train_loader:
        optimizer.zero_grad()

        # 移动到设备
        from_batch = from_batch.to(device)
        to_batch   = to_batch.to(device)
        edge_batch = edge_batch.to(device)
        target_batch = target_batch.to(device)

        # 前向
        preds = model(from_batch, to_batch, edge_batch)   # (B, output_dim)

        loss = criterion(preds, target_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * target_batch.size(0)

    epoch_loss /= len(train_dataset)
    train_losses.append(epoch_loss)

    # ---------- 验证 ----------
    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for from_b, to_b, edge_b, tgt_b in val_loader:
                from_b = from_b.to(device)
                to_b   = to_b.to(device)
                edge_b = edge_b.to(device)
                tgt_b  = tgt_b.to(device)

                val_pred = model(from_b, to_b, edge_b)
                vloss = criterion(val_pred, tgt_b)
                val_loss += vloss.item() * tgt_b.size(0)

            val_loss /= len(val_dataset)
            val_losses.append(val_loss)

            # 计算 R²、MAE 等（每 10 epoch 可选）
            # ……（保持原有代码逻辑，只是把循环改为 batch 形式）

        model.train()
```

> 训练、验证、测试的 **梯度累加、学习率调度、早停** 逻辑保持不变，只是把原来的 `for i in range(N)` 替换为 `for batch in loader`。

---

## 5. 测试循环（同理）

```python
model.eval()
with torch.no_grad():
    all_preds = []
    all_targets = []
    for from_b, to_b, edge_b, tgt_b in test_loader:
        from_b = from_b.to(device)
        to_b   = to_b.to(device)
        edge_b = edge_b.to(device)

        pred = model(from_b, to_b, edge_b)
        all_preds.append(pred.cpu())
        all_targets.append(tgt_b.cpu())

    test_pred = torch.cat(all_preds, dim=0)
    test_tgt = torch.cat(all_targets, dim=0)

    test_loss = criterion(test_pred, test_tgt)
    # 计算 RMSE、MAE、R²、阈值准确率等
```

---

## 6. 其它加速技巧（可选）

| 技巧                                                              | 说明                                                                                                    |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **混合精度** (`torch.cuda.amp`)                                   | 在 `with torch.cuda.amp.autocast():` 包裹前向、损失计算，可提升显存利用率并加速。                       |
| **梯度累加**                                                      | 当显存仍不足以容纳大 batch 时，可设 `batch_size=32`，每 `accum_steps=4` 次累加一次 `optimizer.step()`。 |
| **多 GPU** (`torch.nn.DataParallel` 或 `DistributedDataParallel`) | 若机器有多块 GPU，可在模型外层包裹 `DataParallel`，`DataLoader` 的 `batch_size` 乘以 GPU 数。           |
| **预取 & pin_memory**                                             | `DataLoader(..., pin_memory=True, prefetch_factor=2)` 能进一步提升 CPU→GPU 数据传输效率。               |

---

## 7. 完整改动概览（伪代码）

```python
# 1️⃣ 数据构建保持不变 → 得到 from_data_list, to_data_list, edge_attrs, target_features
# 2️⃣ 划分索引 → train_idx, val_idx, test_idx
# 3️⃣ 根据索引切片生成三个 Dataset
train_dataset = MoleculePairDataset(
        [from_data_list[i] for i in train_idx],
        [to_data_list[i]   for i in train_idx],
        edge_attrs[train_idx],
        target_features[train_idx])

val_dataset   = MoleculePairDataset(... same for val_idx ...)
test_dataset  = MoleculePairDataset(... same for test_idx ...)

# 4️⃣ DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                         num_workers=4, collate_fn=pair_collate, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False,
                         num_workers=2, collate_fn=pair_collate)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False,
                         num_workers=2, collate_fn=pair_collate)

# 5️⃣ 训练 / 验证 / 测试循环 → 参考上文代码块
```

---

### 小结

1. **把每个分子对包装成 `Dataset`**，并使用 **PyG 的 `DataLoader`** 进行 **mini‑batch**。
2. **`Batch.from_data_list`** 自动处理不同大小的图并生成 `batch` 索引，模型的全局池化 (`global_mean_pool / global_max_pool`) 已经支持批量输入[[3]][[4]]。
3. 只需把原来的 **单样本循环** 替换为 **批循环**，其余优化（学习率调度、早停、日志）保持不变。
4. 通过 **增大 batch_size、混合精度、梯度累加** 等手段，可进一步提升训练速度与显存利用率。

这样改写后，训练过程会在 GPU 上一次处理数十甚至上百个分子对，显著缩短每个 epoch 的耗时，同时保持原有模型结构和评估指标不变。祝你实验顺利！
