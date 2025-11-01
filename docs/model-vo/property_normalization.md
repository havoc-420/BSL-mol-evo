# 属性标准化处理说明

在分子进化预测项目中，对属性变化值进行标准化处理是为了确保所有属性在相似的数值范围内，从而提高模型训练效果。

## 标准化的属性

处理的属性变化值包括以下15个量子化学属性：

- `A_change`, `B_change`, `C_change` - 转动常数变化
- `mu_change` - 偶极矩变化
- `alpha_change` - 各向极化率变化
- `homo_change`, `lumo_change` - HOMO/LUMO能级变化
- `gap_change` - 能隙变化
- `r2_change` - 电荷半径变化
- `zpve_change` - 振动零点能变化
- `U0_change`, `U_change` - 内能变化
- `H_change` - 焓变
- `G_change` - 自由能变化
- `Cv_change` - 恒容热容变化

## 标准化方法

### 1. 计算统计信息

在数据预处理阶段，首先计算每个属性的均值和标准差：

```python
property_stats = {}
for prop in property_names:
    if prop in df.columns:
        mean = df[prop].mean()
        std = df[prop].std()
        property_stats[prop] = (mean, std)
```

### 2. 标准化公式

使用Z-score标准化方法：

```
normalized_value = (value - mean) / std
```

其中：
- `value` 是原始属性变化值
- `mean` 是该属性在训练集中的平均值
- `std` 是该属性在训练集中的标准差

### 3. 实现代码

在 [prepare_property_change_targets](file:///Users/havoc420/Documents/Projects/python/mol-evo/mol_evo/core/data/processing.py#L354-L392) 函数中实现：

```python
for prop in property_names:
    if prop in row and not pd.isna(row[prop]):
        value = row[prop]
        # 标准化属性变化值
        if prop in property_stats:
            mean, std = property_stats[prop]
            if std > 0:
                value = (value - mean) / std
        properties.append(value)
    else:
        properties.append(0.0)
```

## 反标准化

在模型评估阶段，需要将标准化的预测结果转换回原始尺度：

### 反标准化公式

```
original_value = normalized_value * std + mean
```

### 实现代码

在 [inverse_standardize](file:///Users/havoc420/Documents/Projects/python/mol-evo/mol_evo/train_nnconv.py#L73-L97) 函数中实现：

```python
for i, prop in enumerate(property_names):
    if prop in property_stats:
        mean, std = property_stats[prop]
        if std > 0:
            inv_predictions[:, i] = predictions[:, i] * std + mean
```

## 标准化的好处

1. **统一数值范围**：不同属性的数值范围可能差异很大，标准化后都在相似范围内
2. **加速收敛**：神经网络在相似范围的输入上收敛更快
3. **提高精度**：避免某些大数值属性主导损失函数计算
4. **数值稳定性**：减少梯度消失或爆炸的风险