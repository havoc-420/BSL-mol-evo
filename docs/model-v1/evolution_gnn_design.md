# 基于分子进化的图神经网络设计

## 概述

本文档描述了如何构建一个基于分子进化思想的图神经网络(GNN)，其中：
- 节点代表完整的SMILES分子
- 边表示分子间的进化关系
- 边特征包含 `{atom, op操作, 属性变化}` 三维信息

## 核心概念

### 节点表示
每个节点代表一个完整的SMILES分子，节点特征包括：
- 分子的全局属性（如偶极矩、极化率、HOMO/LUMO能级等）
- 分子指纹或图表示

### 边表示
边表示分子间的进化关系，边特征包含三个维度：
1. **atom（原子）**: 涉及转化的原子类型
2. **op操作（操作）**: 分子间转化的操作类型
3. **属性变化**: 分子属性的差异值

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
- 边特征按 `{atom, op操作, 属性变化}` 格式组织

### NNConv模型架构
NNConv模型利用边网络将边特征映射为权重矩阵：
- 节点嵌入表示分子特征
- 边特征通过神经网络转换为权重矩阵
- 消息传递机制聚合邻居信息

### 边特征工程
基于CSV数据提取以下特征：

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

## 当前NN框架详细设计

### 数据形态

#### 输入数据
1. **节点特征**：
   - 形态：N × F 的张量，其中 N 是节点数（唯一SMILES分子数），F 是特征维度
   - 两种表示方法：
     - QM9属性：N × 15，包含15个量子化学属性
     - Morgan指纹：N × 2048，包含2048位二进制指纹向量

2. **边特征**：
   - 形态：E × 33 的张量，其中 E 是边数（分子对数），33 是特征维度
   - 包含四个部分：
     - 位置特征（2维）：起始和目标分子的重原子数
     - 原子特征（5维）：添加原子类型的独热编码（C, N, O, F, P）
     - 操作特征（6维）：操作类型的独热编码（add, replace, del, add_multi, del_multi, complex）
     - 变化特征（15维）：15个量子化学属性的具体变化值
     - 变化统计特征（5维）：15个属性变化的统计特征（均值、标准差、最大值、最小值、范围）

3. **边索引**：
   - 形态：2 × E 的张量，表示有向图的连接关系
   - 第一行是源节点索引，第二行是目标节点索引

4. **目标值**：
   - 形态：E × 15 的张量，表示15个量子化学属性的变化值

#### 输出数据
1. **属性变化预测**：
   - 形态：E × 15 的张量，预测15个量子化学属性的变化值
   - 与目标值具有相同维度，用于计算损失函数

2. **目标分子特征预测**：
   - 形态：E × 2048 的张量，预测目标分子的Morgan指纹
   - 用于重构目标分子的结构特征

### 网络框架

#### GNN模型架构
1. **节点嵌入层**：
   - 输入：节点特征（N × F）
   - 操作：线性变换将特征映射到隐藏空间
   - 输出：节点嵌入（N × H），其中 H 是隐藏维度

2. **边网络**：
   - 输入：边特征（E × 26）
   - 结构：两层全连接网络 + ReLU激活
   - 输出：权重矩阵参数（E × H × H）

3. **NNConv层**：
   - 输入：节点嵌入（N × H）和边特征（E × 26）
   - 操作：使用边网络生成的权重进行消息传递
   - 输出：更新后的节点表示（N × H）

4. **多层堆叠**：
   - 包含3个NNConv层
   - 每层使用相同的边网络结构
   - 中间层使用ReLU激活，最后一层不使用激活函数

5. **全局池化**：
   - 输入：所有节点的表示
   - 操作：使用global_mean_pool进行全局平均池化
   - 输出：图的全局表示（1 × H 或 B × H）

6. **预测头**：
   - 输入：图的全局表示
   - 结构：两层全连接网络 + ReLU激活
   - 输出：属性变化预测（1 × 15）

#### 转换器模型架构
1. **起始分子编码器**：
   - 输入：起始分子特征（E × 2048）
   - 操作：线性变换编码为隐藏表示
   - 输出：编码后的起始分子表示（E × H）

2. **边特征编码器**：
   - 输入：边特征（E × 26）
   - 结构：两层全连接网络 + ReLU激活
   - 输出：编码后的边表示（E × H）

3. **特征融合**：
   - 输入：起始分子表示（E × H）和边表示（E × H）
   - 操作：拼接两个表示
   - 输出：融合特征（E × 2H）

4. **转换网络**：
   - 输入：融合特征（E × 2H）
   - 结构：三层全连接网络 + ReLU激活
   - 输出：目标分子特征预测（E × 2048）

### 点&边特征

#### 节点特征
1. **QM9量子化学属性**：
   - A, B, C：分子旋转常数（单位：GHz）
   - mu：分子偶极矩（单位：Debye）
   - alpha：分子各向同性极化率（单位：Bohr^3）
   - homo：最高占据分子轨道能量（单位：Hartree）
   - lumo：最低未占据分子轨道能量（单位：Hartree）
   - gap：HOMO-LUMO能隙（单位：Hartree）
   - r2：零点振动时分子电子空间范围（单位：Bohr^2）
   - zpve：零点振动能（单位：Hartree）
   - U0：0K时的内能（单位：Hartree）
   - U：298.15K时的内能（单位：Hartree）
   - H：298.15K时的焓（单位：Hartree）
   - G：298.15K时的自由能（单位：Hartree）
   - Cv：298.15K时的热容（单位：cal/(mol·K)）

2. **Morgan指纹**：
   - 半径为2的指纹
   - 2048位二进制向量
   - 使用RDKit的GetMorganFingerprintAsBitVect生成

#### 边特征
1. **原子特征（5维）**：
   - to_atom_symbol的独热编码：
     - C（碳原子）
     - N（氮原子）
     - O（氧原子）
     - F（氟原子）
     - P（磷原子）

3. **操作特征（6维）**：
   - operation_type的独热编码：
     - add：添加原子
     - replace：替换原子
     - del：删除原子
     - add_multi：添加多个原子
     - del_multi：删除多个原子
     - complex：复杂操作

4. **变化特征（15维）**：
   - 15个量子化学属性的具体变化值：
     - A_change：旋转常数A的变化值
     - B_change：旋转常数B的变化值
     - C_change：旋转常数C的变化值
     - mu_change：偶极矩的变化值
     - alpha_change：极化率的变化值
     - homo_change：HOMO能量的变化值
     - lumo_change：LUMO能量的变化值
     - gap_change：能隙的变化值
     - r2_change：电子空间范围的变化值
     - zpve_change：零点振动能的变化值
     - U0_change：0K内能的变化值
     - U_change：298.15K内能的变化值
     - H_change：298.15K焓的变化值
     - G_change：298.15K自由能的变化值
     - Cv_change：298.15K热容的变化值

5. **变化统计特征（5维）**：
   - 15个属性变化的统计特征：
     - 均值：15个属性变化的平均值
     - 标准差：15个属性变化的标准差
     - 最大值：15个属性变化的最大值
     - 最小值：15个属性变化的最小值
     - 范围：最大值与最小值的差

### 预测结果

#### 属性变化预测
1. **预测目标**：
   - 预测15个量子化学属性的变化值
   - 与真实变化值进行比较

2. **输出格式**：
   - 张量维度：E × 15
   - 每一行对应一个分子对的属性变化预测

3. **应用场景**：
   - 分子性质预测
   - 化学反应预测
   - 分子设计指导

#### 目标分子特征预测
1. **预测目标**：
   - 预测目标分子的Morgan指纹
   - 用于重构目标分子的结构特征

2. **输出格式**：
   - 张量维度：E × 2048
   - 每一行对应一个目标分子的指纹预测

3. **应用场景**：
   - 分子生成
   - 虚拟筛选
   - 药物设计

### 评估标准

#### 损失函数
1. **均方误差（MSE）**：
   - 用于属性变化预测
   - 公式：MSE = (1/n) * Σ(y_i - ŷ_i)²
   - 其中 y_i 是真实值，ŷ_i 是预测值

2. **二元交叉熵（BCE）**：
   - 用于指纹预测（可选）
   - 公式：BCE = -Σ[y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
   - 适用于二进制指纹预测

#### 性能指标
1. **训练指标**：
   - 训练损失：监控模型在训练集上的拟合程度
   - 验证损失：监控模型在验证集上的泛化能力
   - 测试损失：评估模型在测试集上的最终性能

2. **回归指标**：
   - 均方根误差（RMSE）：√MSE
   - 平均绝对误差（MAE）：(1/n) * Σ|y_i - ŷ_i|
   - 决定系数（R²）：1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)

3. **分类指标**（如适用）：
   - 准确率：正确预测的比例
   - 精确率：预测为正例中实际为正例的比例
   - 召回率：实际为正例中预测为正例的比例
   - F1分数：精确率和召回率的调和平均

#### 模型选择标准
1. **验证集性能**：
   - 选择验证集损失最小的模型
   - 避免过拟合，监控训练和验证损失的差距

2. **早停机制**：
   - 当验证集损失连续多个epoch不改善时停止训练
   - 防止过拟合，节省计算资源

3. **超参数调优**：
   - 网格搜索或随机搜索优化超参数
   - 包括学习率、隐藏维度、层数等

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

## 当前框架不足与改进思路

### 主要不足分析

#### 1. 节点特征表示不完整
- **问题**: 在`build_molecule_graph_with_properties`函数中，QM9原始属性被初始化为0.0，丢失了实际的分子属性信息
- **影响**: 模型无法学习到真实的分子性质特征

#### 2. 边特征设计过于复杂
- **问题**: 边特征包含重复信息，缺乏位置敏感特征
- **影响**: 增加了模型复杂度，可能导致过拟合

#### 3. 模型架构不够灵活
- **问题**: NNConv层使用相同的边网络结构，限制了表达能力
- **影响**: 无法有效捕捉不同层次的边特征重要性

#### 4. 缺乏分子结构信息
- **问题**: 仅使用全局属性或指纹，缺少原子级结构信息
- **影响**: 无法建模局部化学环境对性质变化的影响

#### 5. 训练策略不完善
- **问题**: 数据分割方式不当，缺乏正则化机制
- **影响**: 模型泛化能力受限

### 改进方案

#### 1. 改进节点特征表示
- **目标**: 使用更丰富的分子结构信息
- **方案**:
  - 使用图神经网络编码分子结构（MPNN、GIN）
  - 结合多尺度特征：原子级、键级、分子级
  - 预训练分子表示模型
  - 修复QM9属性初始化问题

#### 2. 优化边特征设计
- **目标**: 简化特征维度，增强表达能力
- **方案**:
  - 简化边特征维度，移除冗余统计特征
  - 增加位置敏感特征（如反应位点信息）
  - 使用注意力机制动态加权边特征
  - 添加分子结构相似性特征

#### 3. 增强模型架构
- **目标**: 提高模型表达能力和灵活性
- **方案**:
  - 引入图注意力机制（GAT）处理边特征重要性
  - 使用分层图神经网络捕捉不同尺度信息
  - 添加残差连接和归一化层
  - 实现多任务学习框架

#### 4. 改进训练策略
- **目标**: 提升模型泛化能力和训练效率
- **方案**:
  - 实现正确的数据分割（按分子而非边）
  - 添加正则化和早停机制
  - 使用更合适的损失函数（Huber损失）
  - 实现学习率调度和梯度裁剪

#### 5. 增加可解释性
- **目标**: 提供模型决策的透明性
- **方案**:
  - 可视化边特征重要性
  - 分析关键进化路径
  - 提供分子性质变化归因分析
  - 实现注意力权重可视化

## 已实现的改进

### 1. 节点特征表示改进 ✅
- **修复QM9属性初始化问题**: 实现了`load_qm9_properties()`函数，正确加载QM9原始量子化学属性
- **增强特征完整性**: 使用真实的15个量子化学属性作为节点特征，而非初始化为0
- **错误处理**: 添加了缺失分子属性的警告机制

### 2. 边特征设计优化 ✅
- **简化特征维度**: 移除了冗余的统计特征，保持30维边特征
- **位置敏感特征**: 添加了起始/目标分子重原子数及原子数变化
- **分子结构相似性**: 添加了Tanimoto相似度特征
- **特征组合**: 30维 = 5个原子类型 + 6个操作类型 + 15个属性变化 + 3个位置特征 + 1个相似性特征

### 3. 模型架构增强 ✅
- **增强GNN模型**: `EnhancedMoleculeGNN`类结合NNConv和GAT
- **注意力机制**: 使用图注意力网络处理节点间重要性
- **残差连接**: 添加层间残差连接，缓解梯度消失
- **归一化层**: 使用BatchNorm提高训练稳定性
- **Dropout正则化**: 防止过拟合
- **增强预测器**: `EnhancedMoleculeEvolutionPredictor`使用增强GNN架构

### 4. 训练策略改进 ✅
- **数据分割**: `split_data_by_molecules()`按分子分割，避免数据泄露
- **增强训练函数**: `train_model_enhanced()`提供完整训练流程
- **优化器改进**: 使用AdamW优化器，带权重衰减
- **学习率调度**: ReduceLROnPlateau自动调整学习率
- **早停机制**: 监控验证损失，防止过拟合
- **梯度裁剪**: 防止梯度爆炸
- **鲁棒损失函数**: 使用HuberLoss对异常值更鲁棒

### 5. 可解释性增强 ✅
- **分子相似性计算**: 提供分子间结构相似性分析
- **训练历史记录**: 返回详细的训练过程信息
- **模型状态保存**: 自动保存最佳模型

## 后续工作

1. 实现完整的图构建流程
2. 开发支持边特征的NNConv模型
3. 训练和评估模型性能
4. 优化边特征表示方法
5. 增加注意力机制以更好地处理边特征
6. 进行超参数调优和模型验证
7. 比较两种节点特征表示方法的效果
8. 实现模型可解释性分析
9. 添加模型部署和推理接口