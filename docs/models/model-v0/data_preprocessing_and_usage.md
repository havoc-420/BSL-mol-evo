# MoleculeEvolutionGCNLinearPredictor 模型数据预处理与使用指南

## 概述

本文档详细介绍了如何为 [MoleculeEvolutionGCNLinearPredictor](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn.py#L79-L136) 模型预处理数据以及如何正确使用该模型进行训练和推理。该模型用于预测分子演化过程中属性的变化，需要成对的分子数据和操作信息作为输入。

## 模型输入要求

模型需要以下三种输入数据：

1. **起始分子图数据** (`from_data`)
2. **目标分子图数据** (`to_data`)
3. **操作信息特征** (`edge_attr`)

这三者构成一个完整的训练样本，表示一个分子演化过程。

## 数据预处理步骤

### 1. 分子图数据构建

可以直接使用项目中已有的工具函数 `mol_evo.core.utils.molecule.smile_to_graph_xyz` 将 SMILES 字符串转换为图结构数据：

```python
from mol_evo.core.utils.molecule import smile_to_graph_xyz
from torch_geometric.data import Data
import torch

def smiles_to_graph_data(smiles):
    """
    将SMILES字符串转换为图数据（使用项目中的实际函数）

    Args:
        smiles (str): SMILES字符串

    Returns:
        Data: PyTorch Geometric Data对象
    """
    # 定义原子类型映射
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

    # 使用项目中的函数将SMILES转换为图结构
    x, z, pos, edge_index, edge_attr = smile_to_graph_xyz(smiles, types)

    # 检查转换是否成功
    if x is None:
        print(f"无法处理SMILES: {smiles}")
        return None

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
```

该函数会生成包含节点特征、3D 坐标、边索引和边属性的完整图结构表示。

### 2. 操作信息特征构建

操作信息特征描述了从起始分子到目标分子的转换操作，直接使用项目中的工具函数：

```python
from mol_evo.core.data.processing import prepare_edge_features
import pandas as pd
import torch

def create_sample_edge_features(to_atom_symbol='C', operation_type='replace'):
    """
    创建示例边特征向量（直接使用processing.py中的prepare_edge_features函数）

    Args:
        to_atom_symbol (str): 目标原子符号
        operation_type (str): 操作类型

    Returns:
        torch.Tensor: 边特征张量 [1, feature_dim]
    """
    # 创建一个模拟的pandas Series，模仿CSV中的一行数据
    row_data = pd.Series({
        'to_atom_symbol': to_atom_symbol,
        'operation_type': operation_type
    })

    # 直接调用processing.py中的函数
    edge_features_list = prepare_edge_features(row_data)

    # 转换为张量并增加批次维度
    edge_attr = torch.tensor([edge_features_list], dtype=torch.float)

    return edge_attr
```

### 3. 构建完整训练样本

将上述组件组合成完整的训练样本：

```python
def create_training_sample(from_smiles, to_smiles, operation_type, atom_type):
    """
    创建完整的训练样本

    Args:
        from_smiles (str): 起始分子SMILES
        to_smiles (str): 目标分子SMILES
        operation_type (str): 操作类型
        atom_type (str): 原子类型

    Returns:
        tuple: (from_data, to_data, edge_attr)
    """
    # 创建起始分子图数据
    from_data = smiles_to_graph_data(from_smiles)

    # 创建目标分子图数据
    to_data = smiles_to_graph_data(to_smiles)

    # 创建操作信息特征
    edge_attr = create_operation_features(operation_type, atom_type)

    return from_data, to_data, edge_attr
```

## 模型使用示例

### 1. 模型初始化

```python
from mol_evo.core.models.v0.gcn import MoleculeEvolutionGCNLinearPredictor

# 模型参数
node_feature_dim = 37  # 节点特征维度
edge_feature_dim = 15  # 边特征维度
hidden_dim = 64        # 隐藏层维度
output_dim = 10        # 输出维度 (属性变化数量)

# 创建模型
model = MoleculeEvolutionGCNLinearPredictor(
    node_feature_dim=node_feature_dim,
    edge_feature_dim=edge_feature_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
)
```

### 2. 前向传播

```python
import torch

# 创建示例数据
from_data, to_data, edge_attr = create_training_sample(
    "CCO",           # 乙醇
    "CC=O",          # 乙醛
    "modify",        # 修改操作
    "C"              # 碳原子
)

# 设置模型为评估模式
model.eval()

# 前向传播
with torch.no_grad():
    output = model(from_data, to_data, edge_attr)

print(f"预测的属性变化: {output}")
```

## 批量处理

当前版本的模型每次处理一个分子对（batch_size=1）。如果需要处理多个样本，可以：

1. 分别处理每个样本
2. 修改模型以支持批量处理

```python
# 分别处理多个样本
samples = [
    ("CCO", "CC=O", "replace", "C"),
    ("CC", "CCC", "add", "C"),
    ("CCC", "CCO", "replace", "O")
]

predictions = []
for from_smiles, to_smiles, op_type, atom_type in samples:
    from_data, to_data, edge_attr = create_training_sample(
        from_smiles, to_smiles, op_type, atom_type)

    with torch.no_grad():
        pred = model(from_data, to_data, edge_attr)
        predictions.append(pred)
```

## 边特征详细说明

边特征通过 `mol_evo.core.data.processing.prepare_edge_features` 函数构建，包含以下组成部分：

1. **原子类型特征** (5 维)：独热编码表示目标原子类型

   - 支持的原子类型: C, N, O, F, P

2. **操作类型特征** (6 维)：独热编码表示操作类型

   - 支持的操作类型: add, replace, del, add_multi, del_multi, complex

3. **总计 11 维特征向量**

## 注意事项

1. **数据一致性**: 确保[from_data](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn.py#L139-L139)、[to_data](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn.py#L142-L142)和[edge_attr](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn.py#L145-L145)三者在语义上一致，共同描述一个分子演化过程。

2. **特征维度**: 确保输入数据的特征维度与模型初始化时指定的维度一致。

3. **批量大小**: 当前模型的[edge_attr](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn.py#L145-L145)应具有形状`[1, edge_feature_dim]`，表示处理单个分子对。

4. **图数据格式**: 确保图数据符合 PyTorch Geometric 的要求，特别是[edge_index](file:///home/xxx/projects/mol_opt/mol-ofo/tests/gnn-train/cora-demo-1.py#L98-L98)的格式。

## 运行示例

使用示例数据运行模型的输出示例：

```
模型创建成功:
  - 节点特征维度: 11
  - 边特征维度: 11
  - 隐藏层维度: 64
  - 输出维度: 10
  - 总参数数量: 27786
起始分子 (CCO): 9 个原子, 节点特征维度 11
目标分子 (CC=O): 7 个原子, 节点特征维度 11
操作信息 (脱氢反应): 批量大小 1, 特征维度 11

预测输出形状: torch.Size([1, 10])
预测值 (前5维): tensor([-0.2320,  0.0479,  0.0141, -0.0257,  0.0178])
```

## 总结

[MoleculeEvolutionGCNLinearPredictor](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn.py#L79-L136)模型通过处理成对的分子图数据和操作信息来预测分子属性的变化。正确预处理数据并确保三个输入组件的一致性是成功使用该模型的关键。在实际应用中，应优先复用项目中已有的工具函数，如`smile_to_graph_xyz`和`prepare_edge_features`，以确保数据处理的一致性和准确性。
