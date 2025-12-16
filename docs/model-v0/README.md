# Molecular Evolution Prediction Model v0

## 任务描述

本项目旨在构建一个基于分子进化的预测模型，根据两个 SMILES 字符串和操作信息（如原子类型、操作类型）来预测 QM9 数据集中分子属性的变化。

## 模型框架设计

### 核心思路

使用两层 GCN（Graph Convolutional Network）搭建的模型来提取单个 SMILES 的分子特征，同时将操作信息（op、atom 等边特征）通过几层线性网络进行特征提取，然后将这些特征进行拼接，再通过一些线性层得到属性变化的预测结果。

### 模型架构

1. **分子特征提取模块**

   - 使用两层 GCN 网络分别处理起始分子（from SMILES）和目标分子（to SMILES）
   - 每个 GCN 模块将分子图结构转换为固定维度的特征向量

2. **边特征处理模块**

   - 将操作信息（包括原子类型`atom`和操作类型`op`）作为边特征
   - 通过多层线性网络对边特征进行编码和提取

3. **特征融合与预测模块**
   - 将起始分子特征、目标分子特征和边特征进行拼接（concat）
   - 通过一系列线性层处理融合后的特征
   - 输出目标属性变化的预测值

### 数据预处理

在将 SMILES 输入到 GCN 网络之前，需要使用 RDKit 库对 SMILES 字符串进行解析，获取分子的图结构信息：

1. **节点特征提取**

   - 使用 RDKit 将 SMILES 转换为分子对象
   - 提取每个原子的属性作为节点特征，包括：
     - 原子类型（原子序数）
     - 原子的杂化状态
     - 原子的度数
     - 是否为芳香性原子
     - 显性价电子数等

2. **边特征提取**

   - 提取分子中原子间的连接关系
   - 获取键的类型信息（单键、双键、三键、芳香键等）
   - 构建邻接矩阵表示原子间的连接关系

3. **图结构构建**
   - 将节点特征组织为节点特征矩阵
   - 将边信息组织为边索引格式（COO 格式），便于图神经网络处理

### SMILES 到图结构的 RDKit 编码实现

在实际实现中，我们使用 RDKit 将 SMILES 字符串转换为图结构数据，以便输入到 GCN 网络中。具体的实现方式可以参考 `mol_evo/tests/smile-to-graph/demo-2.py` 中的 `smile_to_graph_xyz` 函数。

该函数的主要处理步骤包括：

1. **分子对象创建**

   - 使用 `Chem.MolFromSmiles(smile)` 将 SMILES 字符串转换为 RDKit 分子对象
   - 添加氢原子：`Chem.AddHs(mol)` 以确保考虑氢原子的影响

2. **3D 坐标生成**

   - 使用 `AllChem.EmbedMolecule(mol, ETKDG_PARAMS)` 生成分子的 3D 坐标
   - 如果生成失败，会尝试去除立体构型后再次生成

3. **原子特征提取**

   - 遍历分子中的每个原子，提取多种特征：
     - 原子类型索引
     - 原子序数
     - 芳香性
     - 杂化状态（sp, sp2, sp3）
   - 构建原子特征矩阵 `x`

4. **键信息提取**

   - 遍历分子中的每个键，提取键类型信息
   - 构建边索引 `edge_index` 和边属性 `edge_attr`

5. **特征向量构建**
   - 将原子特征和键特征组织为 PyTorch 张量格式
   - 返回可用于 GCN 网络的图结构数据

这种方法能够有效地将 SMILES 字符串转换为适合图神经网络处理的结构化数据，同时保留了分子的重要化学信息。

### 数据集构建：属性变化计算方式

对于属性变化的计算，我们采用以下策略：

1. **百分比变化**：

   - 对于大多数属性，使用公式：`delta = (B - A) / A` 计算百分比变化
   - 这种方式可以更好地比较不同数量级属性的变化

2. **绝对值变化**：
   - 对于可能为 0 或接近 0 的属性（如 mu - 偶极矩），直接使用差值：`delta = B - A`
   - 避免由于分母为 0 或接近 0 导致的数值不稳定问题

### 输入输出

**输入:**

- 起始分子 SMILES 字符串
- 目标分子 SMILES 字符串
- 操作信息（原子类型、操作类型等边特征）

**输出:**

- QM9 数据集属性变化的预测值（如偶极矩、HOMO/LUMO 能级等 15 个量子化学属性的变化值）

### 技术细节

- **图神经网络**: 使用 GCN 作为基础的图神经网络层
- **特征拼接**: 将分子特征和操作特征拼接后进行联合预测
- **端到端训练**: 整个模型可以进行端到端的训练优化
- **RDKit 预处理**: 使用 RDKit 库解析 SMILES 获取分子图结构信息

## 与 NNConv 框架的区别

相比于基于 NNConv 的框架，本框架采用更直接的特征提取和融合方式：

- 不使用边网络动态生成权重矩阵
- 采用静态的特征拼接方式
- 更加简洁明了，易于理解和实现

# v0 版本模型训练说明

## 基本用法

```bash
cd /home/rhj/projects/mol_opt/mol-ofo
python mol_evo/train_v0.py
```

## 命令行参数训练

通过命令行参数指定训练参数：

```bash
python mol_evo/train_v0.py \
  --data-file path/to/data.csv \
  --max-pairs 10000 \
  --epochs 200 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --seed 12345 \
  --target-property homo_change \
  --model-type gcn_linear
```

## YAML 配置文件训练（推荐）

可以通过指定 YAML 配置文件来配置训练参数，这种方式更加灵活且易于管理：

```bash
python mol_evo/train_v0.py --config-file mol_evo/configs/train_config_template.yaml
```

### 配置文件格式

YAML 配置文件采用了分层结构，明确区分了训练参数和模型参数：

```yaml
# ==================== 训练相关参数 ====================
# 这些参数用于控制训练过程，不会传递给模型构造函数
train:
  data_file: "path/to/data.csv"
  max_pairs: 30000
  epochs: 100
  batch_size: 512
  learning_rate: 0.01
  seed: 42
  target_property: "mu_change"

  # 学习率调度器参数
  min_lr: 1e-8

  # 其他训练参数
  # patience_limit: 50
  # weight_decay: 1e-5

# ==================== 模型相关参数 ====================
# 这些参数会传递给模型构造函数
model:
  # 模型类型 (如果未指定，将使用CLI交互式选择)
  # model_type: "gcn_linear"

  node_feature_dim: 11
  edge_feature_dim: 15
  output_dim: 1
  hidden_dims: [128, 256, 256]
  num_heads: 8
  num_layers: 2
```

配置文件中的参数分为两类：

1. **训练相关参数**：这些参数控制训练过程，位于[train](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/modules/equiformer/ocpmodels/common/relaxation/optimizers.py#L0-L0)部分下，如[epochs](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn_transformer_transformer.py#L41-L41)、[batch_size](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/modules/equiformer/deps/fairchem/ocpmodels/common/data_parallel.py#L0-L0)、[learning_rate](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/modules/equiformer/deps/fairchem/configs/s2ef/200k/cgcnn/cgcnn.yml#L11-L11)等，不会传递给模型构造函数
2. **模型相关参数**：这些参数会传递给模型构造函数，位于[model](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/modules/equiformer/ocpmodels/common/relaxation/optimizers.py#L0-L0)部分下，用于初始化模型，如[node_feature_dim](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn_transformer_transformer.py#L31-L31)、[hidden_dims](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn_transformer_transformer.py#L33-L33)等

当使用配置文件时，如果未指定[model_type](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/core/models/v0/gcn_transformer_transformer.py#L32-L32)，系统会尝试根据配置文件名推断模型类型。例如，使用[gcn-tf-tf.yaml](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/configs/gcn-tf-tf.yaml)配置文件时，模型类型会被设置为`gcn-tf-tf`。

如果根据配置文件名无法推断出有效的模型类型，系统将使用 CLI 交互式选择方式让用户选择模型类型。

也可以同时指定配置文件和模型类型：

```bash
python mol_evo/train_v0.py --config-file mol_evo/configs/train_config_template.yaml --model-type gcn_linear
```

这种方式下，配置文件中的模型参数会覆盖默认参数，而命令行指定的模型类型优先级最高。
