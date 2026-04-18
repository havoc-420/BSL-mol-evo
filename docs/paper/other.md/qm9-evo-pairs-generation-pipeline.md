# QM9 Evo-Pairs 训练数据集生成链路

> 以 `qm9-evo-pairs-step-1-checkpoint.json` 为例，梳理从 QM9 原始数据到训练数据集的完整生成流程。

## 概览

整个数据生成分 **3 个阶段**：

```
QM9 原始数据 → 按重原子数分桶 → 笛卡尔积配对 + 路径差异筛选 → 属性计算 & 格式转换
   (阶段1)          (阶段1)              (阶段2)                    (阶段3)
```

---

## 阶段 1：QM9 原始数据提取与分桶

**脚本**：`extract_smiles.py`

### 流程

1. 从 PyTorch Geometric 的 QM9 数据集加载约 130K 分子
2. 按重原子数（heavy atoms = 1~9）分桶，分别保存为 CSV 文件
3. 每个 CSV 包含 `smiles` 列 + 15 个量子化学属性

### 输出文件

```
data/qm9_smiles_heavy_{1-9}_atoms.csv
```

### 各桶分子数量（来自 checkpoint 的 `dataset_info`）

| 重原子数 | 分子数 |
|---------|--------|
| 1       | 3      |
| 2       | 5      |
| 3       | 9      |
| 4       | 31     |
| 5       | 126    |
| 6       | 603    |
| 7       | 3,108  |
| 8       | 17,665 |
| 9       | 107,462 |

### 量子化学属性（15 个）

`mu`, `alpha`, `homo`, `lumo`, `gap`, `r2`, `zpve`, `U0`, `U`, `H`, `G`, `Cv`, `A`, `B`, `C`

---

## 阶段 2：进化对（evo-pairs）提取

**脚本**：`extract_evolution_pairs_v1.py` — 主函数 `main(step=1, ...)`

### 核心逻辑

#### 2.1 加载分桶数据

读取所有 `qm9_smiles_heavy_{i}_atoms.csv`，构建 `datasets` 字典（`{重原子数: DataFrame}`）。

#### 2.2 遍历重原子数组合

对于 `step=N`，遍历所有 `(from_heavy, to_heavy)` 组合，满足：

```
to_heavy - from_heavy ≤ step
```

例如 `step=1` 时，合法组合包括：
- `1→1`, `1→2`
- `2→2`, `2→3`
- ...
- `8→8`, `8→9`
- `9→9`

共 `total_combinations = 17` 个组合。

#### 2.3 笛卡尔积配对与路径差异筛选

对每个 `(from_heavy, to_heavy)` 组合，将源分子集 × 目标分子集进行笛卡尔积遍历：

```
for smiles_n in heavy_n_df:
    for smiles_m in heavy_m_df:
        path_n = MoleculeEvolverAnalysis(smiles_n).get_full_path_dict()
        path_m = MoleculeEvolverAnalysis(smiles_m).get_full_path_dict()
        operation_result = analyze_evolution_operation_dict(path_n, path_m)
        if len(operation_result) == step:   # 关键筛选条件
            保存该配对
```

**关键筛选条件**：两个分子的构建路径差异操作数恰好等于 `step`。

#### 2.4 保存结果

匹配的配对以如下格式写入 `qm9-evo-pairs-step-{step}-pairs.json`：

```json
{
  "smiles_from": "CCO",
  "smiles_to": "CC=O",
  "operations": [
    {"operation": "form_double_bond", "position": "1-2", "atom": ""}
  ]
}
```

#### 2.5 断点续传

通过 `checkpoint.json` 记录进度，支持中断恢复：

| 字段 | 含义 |
|------|------|
| `processed_index` | 已处理的笛卡尔积索引 |
| `from_heavy_atoms` / `to_heavy_atoms` | 当前处理的原子数组合 |
| `pairs_count` | 已找到的配对总数 |
| `completed_combinations` | 已完成的原子数组合数 |
| `current_combination` | 当前正在处理的组合（如 `8->8`） |
| `elapsed_time` | 已耗时（秒） |

---

## 阶段 3（可选）：属性变化计算 & 路径格式转换

### 3.1 属性变化计算

**脚本**：`calculate_property_changes.py`

从 `qm9_smiles_all_atoms.csv` 查询 `smiles_from` 和 `smiles_to` 的属性值，计算 15 个属性的：

- **绝对变化**：`{property}_change`（如 `homo_change = homo_to - homo_from`）
- **百分比变化**：`{property}_change_pct`

输出文件如 `qm9-evo-pairs-step-1-with-properties-pct.json`。

### 3.2 转换为路径格式（v0.3）

**脚本**：`convert_qm9_evo_to_paths.py`

将 pairs 格式转为 paths 格式，供 GNN 模型训练使用。支持：

- 单文件转换：`-i` 指定输入 pairs 文件
- 多 step 合并：`--merge` 合并 step-1 + step-2 等多个文件
- 样本平衡：`--balance` 平衡不同步骤数的样本

---

## 核心算法：MoleculeEvolverAnalysis

**位置**：`core/evolver.py`

### 功能

将一个 SMILES 分子解析为从单原子逐步构建该分子的操作序列（进化路径）。

### 算法步骤

1. **标准化分子**：RDKit 解析 SMILES → 规范化 → 移除氢 → 非芳香环 Kekulize
2. **找规范起始原子** `_find_canonical_start_atom()`：
   - 优先选择杂原子（按原子序数降序）
   - 同类原子选连接数最大的
   - 最终按 RDKit 规范排名决胜
3. **构建规范骨架** `_build_canonical_backbone(start_idx)`：
   - 从起始原子 DFS 遍历，生成唯一的生成树（骨架）
   - 建立 `backbone_map`：原始原子索引 → 骨架位置索引
4. **生成操作序列** `get_full_path_dict()`：
   - 骨架路径：逐个添加原子
   - 非骨架原子：作为附件添加
   - 额外键操作：双键、三键、芳香键等
   - 环操作：成环、芳香环等
   - 立体化学操作

### 操作类型

| 操作 | 含义 |
|------|------|
| `add_atom` | 添加原子 |
| `replace_atom` | 替换原子 |
| `form_double_bond` | 形成双键 |
| `form_triple_bond` | 形成三键 |
| `form_ring` | 成环 |
| `form_double_ring` | 形成双键环 |
| `form_triple_ring` | 形成三键环 |
| `form_aromatic_ring` | 形成芳香环 |
| `add_stereo` | 添加立体化学 |
| `remove_*` | 对应的逆操作 |

---

## 路径差异分析：analyze_evolution_operation_dict

**位置**：`extract_evolution_pairs.py` / `extract_evolution_pairs_v1.py`

### 功能

比较两个分子的构建路径，提取差异操作序列。

### 算法

1. 将两条路径转换为简单操作字符串集合（操作类型 + 原子，不含 position）
2. 计算差集：
   - `removed` = path1 独有的操作（需要删除的）
   - `added` = path2 独有的操作（需要添加的）
3. 对共有操作进行智能匹配（按操作类型和位置长度分组）
4. 输出差异操作列表

**筛选逻辑**：差异操作数 = `step` 的配对才是有效的训练数据。

---

## 一句话总结

> **QM9 分子 → 按重原子数分桶 → 笛卡尔积配对 → 比较两个分子的"构建路径差异" → 筛选差异步数 = step 的配对 → 计算属性变化 → 训练数据**
