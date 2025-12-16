# 分子演化数据集处理工具

该目录包含了用于处理分子演化数据集的一系列工具脚本，主要用于从 QM9 数据集中提取分子、寻找演化对、计算属性变化等。

## 工具列表

### [extract_smiles.py](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/extract_smiles.py)

从 QM9 数据集中提取指定原子数的分子 SMILES 表示。

- 支持提取特定重原子数的分子
- 可以验证 SMILES 的有效性
- 输出为 CSV 格式文件

### [extract_evolution_pairs.py](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/extract_evolution_pairs.py)

寻找适合一步演化的分子对。

- 从不同重原子数的分子中寻找可能的演化对
- 计算分子间的演化相似性
- 输出为 CSV 格式的配对文件

### [calculate_property_changes.py](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/calculate_property_changes.py)

计算分子演化过程中属性变化。

- 根据配对文件计算属性变化百分比
- 用于分析分子演化过程中的性质变化
- 输出包含属性变化的增强版配对文件

### [update_evolution_types.py](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/update_evolution_types.py)

更新演化操作类型分类。

- 根据操作详情更新操作类型
- 特别处理涉及手性信息的操作类型
- 将基本操作类型细分为更精确的子类型

## 数据处理流程

1. 使用[extract_smiles.py](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/extract_smiles.py)从原始 QM9 数据集中提取不同重原子数的分子
2. 使用[extract_evolution_pairs.py](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/extract_evolution_pairs.py)寻找可能的演化对
3. 使用[calculate_property_changes.py](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/calculate_property_changes.py)计算演化过程中的属性变化
4. 使用[update_evolution_types.py](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/update_evolution_types.py)细化操作类型分类

## 输出文件

处理后的数据存储在[data](file:///home/rhj/projects/mol_opt/mol-ofo/mol_evo/dataset/data/)目录中，主要包括：

- 不同重原子数的分子集合
- 分子演化对数据
- 带属性变化信息的演化对数据
