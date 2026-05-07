# Molecular Evolution Prediction Model v1

## 概述

v1 版本是对 v0 模型的重大升级，引入了**序列迭代式架构**，支持多步分子演化预测。与 v0 版本只能预测单步分子演化不同，v1 版本可以处理完整的分子演化序列，适用于复杂的分子优化路径预测任务。

## 核心创新

### 序列迭代式预测

v1 模型支持序列式的分子演化预测，可以处理多步演化过程：

```
s1 -> s2 使用 edge1 -> prediction1
s2 -> s3 使用 edge2 -> prediction2
s3 -> s4 使用 edge3 -> prediction3
...
```

每个步骤独立预测，输出完整的属性变化序列。

### 主要改进

相比 v0 版本，v1 版本的关键改进：

1. **输入格式升级**

   - v0: `from_data`, `to_data`, `edge_attr` (单个分子对)
   - v1: `from_data_list`, `to_data_list`, `edge_attrs` (分子序列)

2. **预测能力增强**

   - v0: 单步预测，输出单个属性变化值
   - v1: 多步预测，输出完整的属性变化序列

3. **应用场景扩展**
   - v0: 单步分子演化预测
   - v1: 多步分子演化路径预测

## 模型架构

### 整体架构

v1 模型保留了 v0 版本的核心组件架构，但在 forward 方法上进行了重大改进：

```
输入层:
├─ from_data_list: [s1, s2, s3, ...]
├─ to_data_list: [s2, s3, s4, ...]
└─ edge_attrs: [edge1, edge2, edge3, ...]

迭代处理层 (for each step):
├─ Molecule Extractor (from)
├─ Molecule Extractor (to)
├─ Edge Encoder
└─ Fusion Predictor

输出层:
└─ predictions: [pred1, pred2, pred3, ...]
```

### 组件说明

#### 1. Molecule Feature Extractor

使用 VisNet 提取分子特征表示：

- **输入**: 单个分子图数据 (Data 对象)
- **输出**: 分子特征向量 (维度: hidden_dims[-1] \* 2)
- **架构**: VisNet 网络，包含 mean 和 max 池化操作的拼接

#### 2. Edge Feature Encoder

线性边特征编码器：

- **输入**: 边特征张量 (维度: [1, edge_feature_dim])
- **输出**: 边特征向量 (维度: hidden_dims[-1])
- **架构**: 多层线性网络

#### 3. Fusion Predictor

特征融合预测器：

- **输入**: from_features, to_features, edge_features
- **输出**: 属性变化预测值 (维度: output_dim)
- **架构**: MLP 网络，隐藏层维度 [512, 256, 128]

### 迭代处理流程

```python
for step in range(num_steps):
    # 1. 获取当前步骤的数据
    current_from_data = from_data_list[step]
    current_to_data = to_data_list[step]
    current_edge_attr = edge_attrs[step:step+1]

    # 2. 提取分子特征
    from_features = molecule_extractor_from(current_from_data)
    to_features = molecule_extractor_to(current_to_data)

    # 3. 编码边特征
    edge_features = edge_encoder(current_edge_attr)

    # 4. 融合特征并预测
    property_change = fusion_predictor(from_features, to_features, edge_features)

    # 5. 收集预测结果
    predictions.append(property_change)
```

## 数据格式要求

### 输入格式

#### from_data_list

- **类型**: `list[Data]`
- **说明**: 起始分子图数据列表
- **示例**: `[s1, s2, s3]`
- **每个 Data 对象包含**:
  - `x`: 节点特征矩阵 [num_nodes, node_feature_dim]
  - `edge_index`: 边索引 [2, num_edges]
  - `pos`: 原子位置坐标 [num_nodes, 3]
  - 其他分子图信息

#### to_data_list

- **类型**: `list[Data]`
- **说明**: 目标分子图数据列表
- **示例**: `[s2, s3, s4]`
- **长度**: 必须与 from_data_list 相同

#### edge_attrs

- **类型**: `torch.Tensor`
- **形状**: `[num_steps, edge_feature_dim]`
- **说明**: 操作信息序列
- **示例**: `torch.tensor([[...], [...], [...]])`
- **长度**: 必须与 from_data_list 长度相同

### 输出格式

- **类型**: `torch.Tensor`
- **形状**: `[num_steps, output_dim]`
- **说明**: 每个步骤的属性变化预测值序列

### 长度一致性验证

模型会自动验证输入长度的一致性：

```python
assert len(to_data_list) == len(from_data_list)
assert edge_attrs.size(0) == len(from_data_list)
```

## 使用示例

### 基本使用

```python
import torch
from mol_evo.core.models import MoleculeEvolutionVisnetLinearIterativePredictor

# 初始化模型
model = MoleculeEvolutionVisnetLinearIterativePredictor(
    node_feature_dim=11,
    edge_feature_dim=15,
    hidden_dims=[128, 256, 256],
    output_dim=1
)

# 准备数据
from_data_list = [s1, s2, s3]  # 起始分子序列
to_data_list = [s2, s3, s4]    # 目标分子序列
edge_attrs = torch.tensor([[...], [...], [...]])  # 3个操作的特征

# 前向传播
predictions = model(from_data_list, to_data_list, edge_attrs)
# predictions.shape: [3, 1]
```

### 训练示例

```python
import torch
from torch.utils.data import DataLoader

# 假设有一个自定义的Dataset
dataset = SequenceEvolutionDataset(data_file="path/to/data.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MoleculeEvolutionVisnetLinearIterativePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        from_data_list, to_data_list, edge_attrs, targets = batch

        # 前向传播
        predictions = model(from_data_list, to_data_list, edge_attrs)

        # 计算损失
        loss = criterion(predictions, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 推理示例

```python
# 加载训练好的模型
model = MoleculeEvolutionVisnetLinearIterativePredictor()
model.load_state_dict(torch.load("path/to/model.pth"))
model.eval()

# 准备推理数据
from_data_list = [s1, s2, s3]
to_data_list = [s2, s3, s4]
edge_attrs = torch.tensor([[...], [...], [...]])

# 推理
with torch.no_grad():
    predictions = model(from_data_list, to_data_list, edge_attrs)

print("属性变化预测序列:", predictions)
```

## 与 v0 版本对比

| 特性         | v0 版本                       | v1 版本                                  |
| ------------ | ----------------------------- | ---------------------------------------- |
| **输入格式** | from_data, to_data, edge_attr | from_data_list, to_data_list, edge_attrs |
| **预测类型** | 单步预测                      | 多步序列预测                             |
| **输出形状** | [output_dim]                  | [num_steps, output_dim]                  |
| **应用场景** | 单步分子演化                  | 多步分子演化路径                         |
| **模型架构** | molecule + edge + fusion      | molecule + edge + fusion (迭代式)        |
| **数据处理** | 单次 forward                  | 循环多次 forward                         |
| **训练方式** | 端到端训练                    | 端到端训练                               |

## 模型参数

### 初始化参数

```python
MoleculeEvolutionVisnetLinearIterativePredictor(
    node_feature_dim: int = 11,      # 节点特征维度
    edge_feature_dim: int = 15,      # 边特征维度
    hidden_dims: list = [128, 256, 256],  # VisNet隐藏层维度
    output_dim: int = 1              # 输出维度
)
```

### 参数说明

- **node_feature_dim**: 分子图中节点的特征维度，默认为 11（原子类型、杂化状态等）
- **edge_feature_dim**: 边特征维度，默认为 15（操作信息）
- **hidden_dims**: VisNet 网络的隐藏层维度列表，影响分子特征提取能力
- **output_dim**: 输出维度，通常为 1（预测单个属性变化）

## 应用场景

### 1. 分子优化路径预测

预测从起始分子到目标分子的多步优化路径：

```
起始分子 -> 中间态1 -> 中间态2 -> 目标分子
   ↓           ↓           ↓           ↓
  edge1      edge2      edge3      predictions
```

### 2. 分子演化轨迹分析

分析分子在演化过程中的属性变化趋势：

- 每个步骤的属性变化值
- 整体演化趋势
- 关键步骤识别

### 3. 多步合成路径规划

规划多步合成路径，预测每一步的属性变化：

- 选择最优的合成路径
- 预测中间产物的性质
- 优化合成策略

## 技术优势

### 1. 序列建模能力

- 支持任意长度的演化序列
- 保留序列中的时序信息
- 适用于复杂的演化过程

### 2. 灵活的架构设计

- 保留 v0 的核心组件
- 模块化设计，易于扩展
- 可以与其他模型组件组合

### 3. 端到端训练

- 整个序列可以端到端训练
- 自动学习序列中的依赖关系
- 优化整体预测性能

### 4. 高效的推理

- 每个步骤独立计算
- 可以并行处理多个序列
- 支持批量推理

## 限制与注意事项

### 1. 计算复杂度

- 序列越长，计算开销越大
- 需要平衡序列长度和计算效率
- 建议根据实际需求选择合适的序列长度

### 2. 内存占用

- 需要存储整个序列的分子图数据
- 大规模序列可能占用较多内存
- 可以考虑使用数据流式处理

### 3. 数据质量要求

- 序列中的每个步骤都需要高质量的数据
- 缺失数据会影响整体预测性能
- 需要仔细的数据预处理

## 未来改进方向

### 1. 注意力机制

引入注意力机制来捕捉序列中的关键步骤：

- 自注意力机制
- 跨步注意力
- 动态权重分配

### 2. 序列到序列模型

考虑使用 Transformer 等序列模型：

- 更好的序列建模能力
- 全局上下文感知
- 并行计算能力

### 3. 增量学习

支持增量式学习和预测：

- 在线学习新步骤
- 动态调整模型
- 适应新的演化模式

### 4. 不确定性量化

为每个预测添加不确定性估计：

- 预测置信度
- 风险评估
- 决策支持

## 模型注册信息

- **模型名称**: `MoleculeEvolutionVisnetLinearIterativePredictor`
- **显示名称**: "VisNet Linear-Linear 序列迭代式模型"
- **保存目录**: `visnet_linear_linear_iterative`
- **注册方式**: 使用 `@register_model` 装饰器

## 相关文件

- **模型实现**: [mol_evo/core/models/v1/visnet_linear_linear.py](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v1/visnet_linear_linear.py)
- **模块初始化**: [mol_evo/core/models/v1/**init**.py](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/core/models/v1/__init__.py)
- **v0 版本文档**: [mol_evo/docs/model-v0/README.md](file:///home/xxx/projects/mol_opt/mol-ofo/mol_evo/docs/model-v0/README.md)

## 参考文献

1. VisNet: Vision-based Graph Neural Network for Molecular Property Prediction
2. PyTorch Geometric: Geometric Deep Learning Extension Library
3. RDKit: Open-Source Cheminformatics Software

## 联系与反馈

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发起 Pull Request
- 联系项目维护者
