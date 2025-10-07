# 基于分子进化的图神经网络设计

## 概述

本文档描述了如何构建一个基于分子进化思想的图神经网络(GNN)，其中：
- 节点代表完整的SMILES分子
- 边表示分子间的进化关系
- 边特征包含 `{pos, atom, op操作, 属性变化}` 四维信息

## 核心概念

### 节点表示
每个节点代表一个完整的SMILES分子，节点特征包括：
- 分子的全局属性（如偶极矩、极化率、HOMO/LUMO能级等）
- 分子指纹或图表示

### 边表示
边表示分子间的进化关系，边特征包含四个维度：
1. **pos（位置）**: 分子在进化路径中的位置
2. **atom（原子）**: 涉及转化的原子类型
3. **op操作（操作）**: 分子间转化的操作类型
4. **属性变化**: 分子属性的差异值

### 进化操作类型
分子进化操作包括：
- 添加原子
- 添加片段/附件
- 形成双键/三键
- 成环
- 指定手性

## NNConv网络设计

### 数据准备
从QM9数据集中筛选出一步进化关系对，数据格式如下：
- `smiles_from`, `smiles_to`: 起始和目标分子的SMILES表示
- `operation_type`: 操作类型（add, replace等）
- `to_atom_symbol`: 添加的原子类型
- `from_heavy_atoms`, `to_heavy_atoms`: 起始和目标分子的重原子数
- `A_change` 到 `Cv_change`: 15个量子化学属性的变化值

### 图结构构建
- 每个唯一的SMILES分子作为一个节点
- 每个进化关系对作为一条有向边（从起始分子指向目标分子）
- 边特征按 `{pos, atom, op操作, 属性变化}` 格式组织

### NNConv模型架构
NNConv模型利用边网络将边特征映射为权重矩阵：
- 节点嵌入表示分子特征
- 边特征通过神经网络转换为权重矩阵
- 消息传递机制聚合邻居信息

### 边特征工程
基于CSV数据提取以下特征：

#### pos（位置）
- 起始分子的重原子数 (`from_heavy_atoms`)
- 目标分子的重原子数 (`to_heavy_atoms`)

#### atom（原子变化）
- 添加的原子类型 (`to_atom_symbol`)，转换为独热编码
- 原子变化类型（添加、替换等）

#### op操作（操作类型）
- 操作类型分类 (`operation_type`): add, replace等
- 操作类型的独热编码表示

#### 属性变化
- 15个量子化学属性的变化值：
  - A, B, C (旋转常数)
  - mu (偶极矩)
  - alpha (各向同性极化率)
  - homo (最高占据分子轨道能量)
  - lumo (最低未占据分子轨道能量)
  - gap (HOMO-LUMO能隙)
  - r2 (电子空间范围)
  - zpve (零点振动能)
  - U0 (0K时的内能)
  - U (298.15K时的内能)
  - H (298.15K时的焓)
  - G (298.15K时的自由能)
  - Cv (298.15K时的热容)

## 模型实现细节

### 节点特征表示

我们提供了两种节点特征表示方法：

#### 方法1：QM9原始量子化学属性
使用QM9数据集中的15个原始量子化学属性作为节点特征：
- A, B, C (旋转常数)
- mu (偶极矩)
- alpha (各向同性极化率)
- homo (最高占据分子轨道能量)
- lumo (最低未占据分子轨道能量)
- gap (HOMO-LUMO能隙)
- r2 (电子空间范围)
- zpve (零点振动能)
- U0 (0K时的内能)
- U (298.15K时的内能)
- H (298.15K时的焓)
- G (298.15K时的自由能)
- Cv (298.15K时的热容)

**优点**：
- 直接对应分子的物理化学性质
- 与我们的预测目标（属性变化）在语义上更一致
- 可以更好地建模属性之间的相关性

**缺点**：
- 可能缺少足够的结构信息
- 需要考虑属性间的尺度差异（需要标准化）

#### 方法2：Morgan指纹
使用2048维Morgan指纹表示分子结构特征：
- 半径为2的Morgan指纹
- 2048位二进制向量

**优点**：
- 结构信息丰富，能捕获分子的拓扑结构特征
- 是一种成熟且广泛使用的分子表示方法
- 对分子结构的小变化敏感

**缺点**：
- 丢失了具体的量子化学数值信息
- 无法直接反映分子的电子性质、能量等物理化学特性

### 边特征表示
每条边使用以下特征表示：
1. 位置特征：起始和目标分子的原子数（from_heavy_atoms, to_heavy_atoms）
2. 原子特征：添加原子的类型（C, N, O等）的独热编码
3. 操作特征：操作类型（添加、替换等）的独热编码
4. 变化特征：15个量子化学属性的数值变化（X_change字段）

### 网络架构设计

#### 节点嵌入层
```python
# 将SMILES转换为分子特征
node_features = molecule_to_features(smiles)  # QM9属性或指纹
node_embedding = Linear(node_features_dim, embedding_dim)
```

#### 边特征处理层
```python
# 边特征处理网络
edge_network = Sequential(
    Linear(edge_features_dim, hidden_dim),
    ReLU(),
    Linear(hidden_dim, embedding_dim * embedding_dim)
)
```

#### NNConv层
```python
# 核心NNConv层
nn_conv = NNConv(embedding_dim, embedding_dim, edge_network, aggr='mean')
```

#### 多层架构
```python
# 多层NNConv堆叠
nn_conv1 = NNConv(embedding_dim, embedding_dim, edge_network1)
nn_conv2 = NNConv(embedding_dim, embedding_dim, edge_network2)
nn_conv3 = NNConv(embedding_dim, output_dim, edge_network3)
```

### 分子进化转换器模型

除了传统的图神经网络方法，我们还实现了一种基于转换器的模型，可以直接根据起始分子特征、涉及的原子和操作类型预测目标分子特征。

该模型包含以下组件：
1. **起始分子编码器**：将起始分子的Morgan指纹编码为隐藏表示
2. **边特征编码器**：将边特征（原子类型、操作类型等）编码为隐藏表示
3. **转换网络**：将编码后的起始分子和边特征合并，预测目标分子特征

这种设计更直接地实现了您的目标：根据起始分子（from-smile）、涉及的原子（atom）和操作（op）预测目标分子（to-smile）的特征。

### 模型训练目标
1. **属性变化预测**：给定起始分子和边特征，预测量子化学属性变化
2. **目标分子重建**：给定起始分子和边特征，预测目标分子特征
3. **进化路径预测**：预测可能的分子进化方向

## 代码实现

### 核心模块
代码实现在 `mol_evo/core/gnn.py` 文件中，包含以下类：

1. `MoleculeGNN`: 基于QM9属性的GNN模型
2. `MoleculeGNNWithFingerprint`: 基于指纹的GNN模型
3. `MoleculeEvolutionPredictor`: 基于QM9属性的属性变化预测器
4. `MoleculeEvolutionPredictorWithFingerprint`: 基于指纹的属性变化预测器
5. `MoleculeEvolutionTransformer`: 分子进化转换器模型

### 主要功能
- `smiles_to_fingerprint`: 将SMILES转换为Morgan指纹
- `atom_type_to_onehot`: 原子类型独热编码
- `operation_type_to_onehot`: 操作类型独热编码
- `prepare_edge_features`: 准备边特征向量
- `build_molecule_graph_with_properties`: 使用QM9属性构建图
- `build_molecule_graph_with_fingerprints`: 使用指纹构建图
- `prepare_evolution_data`: 准备转换器模型训练数据

### 使用示例
```python
# 导入模块
from core.gnn import (MoleculeEvolutionPredictor, MoleculeEvolutionPredictorWithFingerprint,
                      MoleculeEvolutionTransformer,
                      build_molecule_graph_with_properties, build_molecule_graph_with_fingerprints,
                      prepare_evolution_data)

# 构建图数据
data_prop, smiles_to_idx = build_molecule_graph_with_properties('qm9-evo-pairs-step-1-with-properties.csv')
data_fp, _ = build_molecule_graph_with_fingerprints('qm9-evo-pairs-step-1-with-properties.csv')

# 准备转换器数据
source_features, edge_features, target_features, stats = prepare_evolution_data('qm9-evo-pairs-step-1-with-properties.csv')

# 创建模型
model_prop = MoleculeEvolutionPredictor()  # 基于QM9属性
model_fp = MoleculeEvolutionPredictorWithFingerprint()  # 基于指纹
model_transformer = MoleculeEvolutionTransformer()  # 转换器模型

# 训练模型
losses_prop = train_model(model_prop, data_prop, target_changes)
losses_fp = train_model(model_fp, data_fp, target_changes)
losses_transformer = train_transformer_model(model_transformer, source_features, edge_features, target_features)
```

### 运行训练脚本
训练脚本位于 `mol_evo/train.py`，可以使用以下命令运行：

```bash
# 训练模型（使用前100个分子对，训练20轮）
python mol_evo/train.py --max-pairs 100 --epochs 20

# 训练模型（使用完整数据集，训练100轮）
python mol_evo/train.py --epochs 100

# 查看所有可用参数
python mol_evo/train.py --help
```

训练过程中会输出训练进度和损失值，训练完成后会将模型保存为 `molecule_evolution_transformer.pth` 文件。

## 应用场景

1. **分子性质预测**：利用邻居分子的信息改进预测
2. **分子生成**：学习分子进化规律指导新分子设计
3. **药物发现**：识别具有相似进化路径的活性化合物
4. **化学反应预测**：基于进化路径相似性预测反应可能性
5. **量子化学属性预测**：预测分子属性变化值

## 后续工作

1. 实现完整的图构建流程
2. 开发支持边特征的NNConv模型
3. 训练和评估模型性能
4. 优化边特征表示方法
5. 增加注意力机制以更好地处理边特征
6. 进行超参数调优和模型验证
7. 比较两种节点特征表示方法的效果