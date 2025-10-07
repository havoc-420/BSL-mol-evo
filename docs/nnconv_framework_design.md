# NNConv框架下的分子进化预测网络设计

## 1. 概述

NNConv框架是一种基于边网络的图神经网络架构，特别适用于分子进化预测任务。在该框架中，节点代表分子，边代表分子间的进化关系，通过边网络动态生成权重矩阵来实现消息传递。

## 2. 数据结构设计

### 2.1 节点表示
- 每个节点代表一个完整的SMILES分子
- 节点特征维度：2048维（Morgan指纹）或15维（QM9量子化学属性）
- 特征类型根据任务需求选择

### 2.2 边表示
- 每条边代表分子间的进化关系
- 边特征维度：30维，包含以下信息：
  - 原子特征（5维）：添加原子类型的独热编码（C, N, O, F, P）
  - 操作特征（6维）：操作类型的独热编码（add, replace, del等）
  - 位置特征（3维）：起始分子重原子数、目标分子重原子数、原子数变化量
  - 相似性特征（1维）：Morgan指纹Tanimoto相似度
  - 属性变化特征（15维）：15个量子化学属性的变化值

## 3. 网络架构设计

### 3.1 整体架构
```
输入节点特征[N, F] → 节点嵌入层 → [N, H]
输入边特征[E, 26] → 边网络 → 权重矩阵参数[E, H*H]
NNConv层1 → [N, H] → 激活函数 → Dropout
NNConv层2 → [N, H] → 激活函数 → Dropout
NNConv层3 → [N, H] 
源/目标节点特征拼接 + 边特征编码 → [E, 3H]
全连接预测头 → 输出属性变化[E, 15]
```

### 3.2 关键组件

#### 3.2.1 节点嵌入层
```python
self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
```
将原始节点特征映射到隐藏空间表示。

#### 3.2.2 边网络
```python
edge_net = nn.Sequential(
    nn.Linear(edge_feature_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim, hidden_dim * hidden_dim)
)
```
将边特征映射为节点特征变换的权重矩阵参数。

#### 3.2.3 NNConv层
```python
NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean')
```
使用边网络生成的权重进行消息传递，聚合邻居节点信息。

#### 3.2.4 边特征编码器
```python
self.edge_feature_encoder = nn.Sequential(
    nn.Linear(edge_feature_dim, hidden_dim),
    nn.ReLU()
)
```
独立编码原始边特征，保留操作相关信息。

#### 3.2.5 预测头
```python
self.property_predictor = nn.Sequential(
    nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)
```
融合源节点、目标节点和边特征，预测属性变化。

## 4. 正则化设计

为防止过拟合，在网络中应用了多层正则化：
- 边网络中添加Dropout(0.1)
- GNN层间添加Dropout(0.2)
- 预测头中添加Dropout(0.2)

## 5. 前向传播流程

1. 节点特征通过嵌入层映射到隐藏空间
2. 多层NNConv进行消息传递，更新节点表示
3. 提取每条边的源节点和目标节点特征
4. 独立编码原始边特征
5. 拼接三种特征作为预测输入
6. 通过预测头输出属性变化预测值

## 6. 优化要点

1. **特征融合**：充分融合节点特征和边特征，而非仅依赖节点特征
2. **正则化**：在多个位置应用Dropout防止过拟合
3. **维度匹配**：确保边网络输入输出维度与实际特征匹配
4. **任务适配**：专注于边级别预测任务，移除不必要的全局池化模块

这种设计充分利用了NNConv的特性，能够有效处理分子进化预测任务，通过动态权重生成机制捕捉节点间复杂的相互作用。