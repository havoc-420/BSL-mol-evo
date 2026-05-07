# 分子进化树测试脚本

该目录包含用于测试和验证分子进化树生成算法的脚本。

## 脚本说明

### 1. `test_evolution_tree.py`

主测试脚本，用于测试分子进化路径的生成。

#### 功能

- 分析单个分子的构建路径
- 生成多条分子进化路径
- 生成分子进化树（支持全量扩展）
- 反向分析目标分子的构建路径

#### 使用方法

```bash
# 分析单个分子的构建路径
python test_evolution_tree.py c1ccccc1 --test-type analysis

# 生成多条进化路径
python test_evolution_tree.py CC --test-type evolution --num-paths 5 --steps-per-path 3

# 生成进化树（全量扩展）
python test_evolution_tree.py CC --test-type tree --max-depth 3

# 生成进化树（限制分支数）
python test_evolution_tree.py CC --test-type tree --max-depth 3 --max-branching 2

# 反向分析目标分子
python test_evolution_tree.py --test-type reverse --target-smiles CCO

# 使用配置文件生成进化路径
python test_evolution_tree.py CC --test-type evolution --config-file /path/to/config.yaml

# 使用配置文件运行进化树测试
python test_evolution_tree.py CC --test-type tree --config-file /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/data/tmp/qm9-evo-pairs-step-1-pairs-v0-81005-with-properties-pct.yaml

# 使用配置文件运行进化路径生成测试
python test_evolution_tree.py CC --test-type evolution --config-file /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/data/tmp/qm9-evo-pairs-step-1-pairs-v0-81005-with-properties-pct.yaml --num-paths 5 --steps-per-path 3
```

#### 配置文件支持

脚本支持通过 YAML 配置文件动态加载原子类型和操作类型。配置文件格式如下：

```yaml
atom_types:
  - C
  - N
  - O
  - F

operation_types:
  - add_atom
  - replace_atom
  - form_double_bond
  - form_ring
  - add_stereo
```

使用配置文件可以确保生成的进化路径与训练数据集保持一致。

### 2. `validate_dataset_consistency.py`

验证生成的分子进化路径与训练数据集格式一致性的脚本。

#### 功能

- 加载并分析数据集样本
- 分析生成路径的格式
- 比较数据集和生成路径的格式差异
- 验证操作格式的一致性

#### 使用方法

```bash
python validate_dataset_consistency.py
```

### 3. `generate_training_data.py`

生成与训练数据格式一致的分子演化数据。

#### 功能

- 生成分子演化对
- 转换为训练数据格式
- 保存训练数据

#### 使用方法

```bash
python generate_training_data.py
```

## 操作类型说明

生成的路径包含以下操作类型，与训练数据集保持一致（**已移除 add_fragment 操作**）：

- `add_atom`: 添加原子
- `replace_atom`: 替换原子
- `form_double_bond`: 形成双键
- `form_triple_bond`: 形成三键
- `form_ring`: 成环
- `form_double_ring`: 形成双键环
- `form_triple_ring`: 形成三键环
- `form_aromatic_ring`: 形成芳香环
- `add_stereo`: 添加立体化学信息
- `remove_atom`: 移除原子
- `remove_form_double_bond`: 移除并形成双键
- `remove_form_triple_bond`: 移除并形成三键
- `remove_form_ring`: 移除并成环
- `remove_form_double_ring`: 移除并形成双键环
- `remove_form_triple_ring`: 移除并形成三键环
- `remove_form_aromatic_ring`: 移除并形成芳香环
- `remove_add_stereo`: 移除并添加立体化学信息

## 位置格式说明

- 原子位置: 使用原子索引表示，如 `"0"`, `"1"`
- 键位置: 使用两个原子索引表示，如 `"0-1"`, `"1-2"`

## 数据格式

生成的数据与训练数据集格式保持一致：

```json
{
  "smiles_from": "起始分子SMILES",
  "smiles_to": "目标分子SMILES",
  "operations": [
    {
      "position": "操作位置",
      "atom": "原子或片段",
      "operation": "操作类型"
    }
  ]
}
```

## 验证要点

1. 操作类型与数据集一致（已移除 `add_fragment`）
2. 位置格式与数据集一致
3. 每个操作包含必需的字段: position, atom, operation
4. 生成路径可以正确地从起始分子演化到目标分子

## 注意事项

根据项目要求，我们做了以下修改：

1. **已从系统中完全移除 `add_fragment` 操作**。现在系统只使用 `add_atom` 操作来添加新的原子，包括原本通过 `add_fragment` 添加的复杂片段。这样做的好处是：

   - 确保与训练数据集的操作类型保持完全一致
   - 简化了操作类型系统
   - 保持了系统的统一性，所有添加操作都通过原子级别进行

2. **移除了操作权重配置，改为全量扩展树生成**。对于复杂片段（如甲基、乙基等），现在会分解为多个 `add_atom` 操作来实现，确保与训练数据集格式的一致性。全量扩展意味着：

   - 不再基于权重进行随机选择
   - 生成所有可能的操作变体
   - 提供更全面的分子演化可能性
   - 适用于需要完整搜索空间的场景
   - 移除了所有与权重相关的参数，包括 `operation_weights` 和 `weight` 字段

3. **支持从外部 YAML 配置文件动态加载原子类型和操作类型**。通过配置文件可以确保系统与特定数据集的操作类型和原子类型保持一致，提高系统的灵活性和适应性。

这种修改简化了操作类型系统，同时保持了功能的完整性，并且能够生成更全面的分子演化路径。

## 测试

```bash
cd /home/xxx/projects/mol_opt/mol-ofo && python mol_evo/tests/mo-tree-gen/standalone_test_evolution.py --config-file /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/data/tmp/qm9-evo-pairs-step-1-pairs-v0-81005-with-properties-pct.yaml

cd /home/xxx/projects/mol_opt/mol-ofo && python -m mol_evo.core.evolver expand CC --num-paths 2 --max-steps 3 --format json --config-file /home/xxx/projects/mol_opt/mol-ofo/mol_evo/dataset/data/tmp/qm9-evo-pairs-step-1-pairs-v0-81005-with-properties-pct.yaml
```
