# NNConv 模型预测机制详解

## 概述

[predict_nnconv.py](file:///Users/havoc420/Documents/Projects/python/mol-evo/mol_evo/predict_nnconv.py) 是一个用于使用训练好的 NNConv 模型进行分子进化属性变化预测的脚本。它能够执行单次预测和批量预测两种模式，支持自动查找和选择模型，并提供详细的预测结果和误差统计。

## 核心功能

### 1. 日志系统设置

```python
def setup_logger(log_level=logging.INFO):
```

该函数设置了日志记录器，用于记录预测过程中的关键信息。它创建了一个名为 'prediction' 的 logger 实例，避免重复添加处理器，并配置了控制台输出。

### 2. 模型自动查找与选择

```python
def find_model_files():
def select_model_interactively(model_files):
```

- `find_model_files()` 函数会自动在项目目录中查找所有可用的模型文件，路径模式为 `mol_evo/model-data/nnconv/training_*/molecule_evolution_nnconv_predictor.pth`
- `select_model_interactively()` 提供交互式界面让用户选择模型，如果只有一个模型则自动选择，如果有多个模型则让用户选择或默认选择最新的模型

### 3. 属性统计信息加载

```python
def load_property_stats(model_dir):
```

从模型目录中的 [training_data.json](file:///Users/havoc420/Documents/Projects/python/mol-evo/model-data/nnconv/training_20251008_043355/training_data.json) 文件加载属性统计信息（均值和标准差），这些信息用于后续的反标准化操作。

### 4. 数据准备

```python
def prepare_single_prediction_data(smiles_from, smiles_to, to_atom_symbol, operation_type, model_dir):
```

为单个预测准备数据：
- 使用 Morgan 指纹将 SMILES 转换为节点特征
- 构建图结构数据，包括节点特征、边索引和边属性
- 节点特征是两个分子的指纹向量（2048维）
- 边特征包括原子类型和操作类型信息（11维）

### 5. 单次预测

```python
def predict_property_changes(model_path, model_dir, smiles_from, smiles_to, to_atom_symbol, operation_type):
```

执行单个分子对的属性变化预测：
1. 准备预测数据
2. 创建并加载训练好的模型权重
3. 执行预测得到标准化的属性变化值
4. 使用加载的统计信息进行反标准化，得到原始尺度的预测值

### 6. 批量预测与误差统计

```python
def batch_predict(model_path, model_dir, csv_file, num_samples=10, random_seed=42, logger=None):
```

批量预测功能：
1. 从 CSV 文件中读取指定数量的样本
2. 对每个样本执行单次预测
3. 计算并统计各种误差指标：
   - MSE（均方误差）
   - RMSE（均方根误差）
   - MAE（平均绝对误差）
   - R²（决定系数）
   - 相对误差百分比
   - 预测值和真实值的均值与标准差

```python
def print_error_statistics(error_stats, logger=None):
```

以表格形式展示批量预测的误差统计结果。

### 7. 主函数与命令行接口

```python
def main():
```

主函数处理命令行参数并执行预测：
- 支持单次预测和批量预测两种模式
- 可以手动指定模型路径或自动查找选择
- 提供详细的命令行参数配置选项

## 预测流程详解

### 单次预测流程

1. 用户提供起始分子 SMILES、目标分子 SMILES、变化涉及的原子类型和操作类型
2. 系统自动查找并选择模型（如果未指定）
3. 准备预测数据：
   - 将 SMILES 转换为 Morgan 指纹作为节点特征
   - 构建图结构数据
4. 加载模型并执行预测
5. 如果有属性统计信息，则进行反标准化得到原始尺度的预测值
6. 输出标准化和原始尺度的预测结果

### 批量预测流程

1. 从 CSV 文件中读取指定数量的样本
2. 对每个样本执行单次预测流程
3. 收集所有预测结果和真实值
4. 计算各种误差指标：
   - 按属性维度分别计算 MSE、RMSE、MAE、R² 和相对误差
   - 计算预测值和真实值的统计信息（均值、标准差）
5. 以表格形式展示误差统计结果
6. 可选择将结果保存到 CSV 文件

## 关键技术点

### 1. 数据表示

- 使用 Morgan 指纹（2048维）表示分子特征
- 图结构包含两个节点（起始分子和目标分子）和一条边
- 边特征包含原子类型和操作类型信息（11维）

### 2. 模型处理

- 使用 [MoleculeEvolutionNNConvPredictor](file:///Users/havoc420/Documents/Projects/python/mol-evo/mol_evo/core/models/nnconv_predictor.py#L11-L101) 模型进行预测
- 模型输出 15 个属性的变化值：
  - A_change, B_change, C_change（旋转常数）
  - mu_change, alpha_change（偶极矩、极化率）
  - homo_change, lumo_change, gap_change（轨道能级）
  - r2_change, zpve_change（零点振动能量）
  - U0_change, U_change, H_change, G_change, Cv_change（热力学性质）

### 3. 标准化与反标准化

- 训练时对属性值进行了标准化处理
- 预测时输出标准化的值
- 使用训练时保存的统计信息进行反标准化，得到原始尺度的预测值

### 4. 误差评估

使用多种指标全面评估预测性能：
- MSE、RMSE、MAE：衡量预测值与真实值的差异
- R²：衡量模型解释方差的能力
- 相对误差：以百分比形式表示预测误差的相对大小
- 统计信息：比较预测值和真实值的分布情况

## 使用示例

### 单次预测

```bash
python predict_nnconv.py \
  --smiles-from "CC" \
  --smiles-to "CCC" \
  --atom-symbol "C" \
  --operation-type "add"
```

### 批量预测

```bash
python predict_nnconv.py \
  --csv-file data/qm9-evo-pairs-step-1-with-properties.csv \
  --num-samples 100
```

## 总结

[predict_nnconv.py](file:///Users/havoc420/Documents/Projects/python/mol-evo/mol_evo/predict_nnconv.py) 是一个功能完整的预测工具，具有以下特点：

1. **自动化程度高**：支持自动查找和选择模型
2. **双模式预测**：支持单次预测和批量预测
3. **详细的结果展示**：不仅显示预测值，还提供误差统计
4. **灵活的配置**：丰富的命令行参数选项
5. **健壮的错误处理**：具备完善的日志和异常处理机制
6. **标准化处理**：正确处理属性值的标准化和反标准化