# FragNet 核心思想

FragNet (Fragment Network) 是一种专门用于分子属性预测的图神经网络架构，具有四层可解释性。它通过同时处理原子图和片段图来提取分子特征，能够提供对分子结构如何影响预测结果的深入洞察。

## 核心架构

### 1. 双图表示 (Dual Graph Representation)

FragNet 的核心创新在于其双图表示方法：

- **原子图 (Atom Graph)**：传统的分子图表示，节点代表原子，边代表化学键
- **片段图 (Fragment Graph)**：将分子分解为片段，节点代表片段，边代表片段间的连接关系

这种双图结构允许模型在不同粒度级别上捕获分子信息。

### 2. 层次化信息传递

FragNet 通过四个层次的信息传递实现深度特征提取：

1. **原子级别**：在原子图中传递信息
2. **键级别**：在键图中传递信息（原子间的连接）
3. **片段级别**：在片段图中传递信息
4. **片段连接级别**：在片段连接图中传递信息

### 3. 注意力机制

FragNet 在不同层级间使用注意力机制来加权信息传递：

- 原子图注意力 (Atom Graph Attention)
- 键图注意力 (Bond Graph Attention)
- 片段图注意力 (Fragment Graph Attention)
- 片段连接注意力 (Fragment Bond Attention)

这些注意力权重提供了模型决策过程的可解释性。

## 数据表示

### 输入特征

FragNet 需要以下输入特征：

- 原子特征 (atom_features)：描述原子的属性
- 片段特征 (frag_features)：描述片段的属性
- 边特征 (edge_features)：描述原子间连接的属性
- 片段连接特征 (fedge_in)：描述片段间连接的属性

### 特征聚合

FragNet 通过以下方式聚合特征：

1. 初始片段特征由组成该片段的原子特征求和得到
2. 使用 scatter_add 操作在不同层级间聚合信息
3. 通过注意力机制加权不同来源的信息

## 可解释性

FragNet 提供了四个层次的可解释性：

1. **原子注意力权重**：显示哪些原子对预测更重要
2. **键注意力权重**：显示哪些化学键对预测更重要
3. **片段注意力权重**：显示哪些片段对预测更重要
4. **片段连接贡献值**：显示片段间的连接如何影响预测

这种多层次的可解释性使研究人员能够理解模型如何基于分子结构做出预测。

## 特征提取策略

在实际应用中，我们可以从 FragNet 的不同层级提取特征用于下游任务：

### 1. 图级别特征（推荐）

这是最常用的特征提取方式，适用于大多数分子属性预测任务：

- 从 FragNet 主干网络获取原子和片段特征
- 使用全局池化操作（如 mean 和 max）获得固定维度的图级别表示
- 合并不同层级和不同池化方式的特征以获得更丰富的表示

```python
# 从 FragNet 获取原子和片段特征
x_atoms, x_frags, _, _ = self.fragnet(batch_dict)

# 对原子特征进行池化
x_atoms_mean = global_mean_pool(x_atoms, batch)
x_atoms_max = global_max_pool(x_atoms, batch)

# 对片段特征进行池化
x_frags_mean = global_mean_pool(x_frags, frag_batch)
x_frags_max = global_max_pool(x_frags, frag_batch)

# 合并所有特征
x_mean = torch.cat([x_atoms_mean, x_frags_mean], dim=1)
x_max = torch.cat([x_atoms_max, x_frags_max], dim=1)
x = torch.cat([x_mean, x_max], dim=1)  # 最终的特征向量
```

### 2. 片段层级特征

当关注分子的片段结构信息时，可以直接使用片段特征：

```python
x_frags, _, _, _ = self.fragnet(batch_dict)
# 然后对 x_frags 进行池化操作
```

### 3. 原子层级特征

当关注原子级别的细节时，可以使用原子特征：

```python
x_atoms, _, _, _ = self.fragnet(batch_dict)
# 然后对 x_atoms 进行池化操作
```

## 应用场景

FragNet 特别适用于需要理解分子结构与属性关系的任务：

- 分子属性预测
- 药物发现
- 分子设计
- 化学信息学研究

通过其独特的架构和可解释性，FragNet 为分子属性预测提供了一个强大而透明的工具。