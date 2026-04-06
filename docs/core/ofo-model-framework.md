# OFO (Optimization-Focused Operator) 模型框架总览

> 本文档为 `mol-ofo` 项目核心模型（`core/models/v0/`）的框架级概览，旨在帮助新接入的 Agent 快速理解整体架构、模块职责与数据流向。

---

## 1. 定位与目标

OFO（**O**ptimization-**F**ocused **O**perator）模型是一个**分子进化属性变化预测器**。其核心任务是：

- **输入**：起始分子（From SMILES）、目标分子（To SMILES）、操作信息（Edge Features）
- **输出**：分子演化过程中量子化学属性的变化值（QM9 数据集 15 种属性，如 HOMO/LUMO/GAP/mu 等）

该预测器是上游 **分子优化搜索算法**（BFS / MCTS / A* 等）的核心评估组件——搜索算法通过 OFO 模型快速估算「执行某操作后属性会如何变化」，从而指导搜索方向。

---

## 2. 整体架构：三组件模块化设计

```mermaid
flowchart TD
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,color:#000
    classDef modelStyle fill:#e8f5e8,stroke:#1b5e20,color:#000
    classDef outputStyle fill:#fff3e0,stroke:#e65100,color:#000
    classDef componentStyle fill:#fce4ec,stroke:#880e4f,color:#000

    input1[(起始分子 From SMILES)]:::inputStyle
    input2[(目标分子 To SMILES)]:::inputStyle
    input3[(操作特征 Edge Features)]:::inputStyle

    subgraph A[组件A: 分子特征提取 MoleculeFeatureExtractor]:::modelStyle
        A_ex[[GCN / VisNet / FragNet / EquiformerV1 / TensorNet]]:::componentStyle
    end

    subgraph B[组件B: 边特征提取 EdgeFeatureExtractor]:::modelStyle
        B_ex[[Linear / Transformer / Transformer+LAP]]:::componentStyle
    end

    subgraph C[组件C: 特征融合预测 FusionPredictor]:::modelStyle
        C_ex[[MLP / Transformer]]:::componentStyle
    end

    output[(属性变化预测 Property Delta)]:::outputStyle

    input1 --> A
    input2 --> A
    input3 --> B

    A --> C
    B --> C
    C --> output
```

### 统一 Forward 接口

所有 v0 模型共享统一签名：

```python
def forward(self, from_data: Data, to_data: Data, edge_attr: torch.Tensor) -> torch.Tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `from_data` | `torch_geometric.data.Data` | 起始分子图数据（含 `x`, `edge_index`, `z`, `pos`, `batch`） |
| `to_data` | `torch_geometric.data.Data` | 目标分子图数据（同上结构） |
| `edge_attr` | `torch.Tensor` shape `(B, edge_feature_dim)` | 操作信息编码 |
| **返回值** | `torch.Tensor` shape `(B, output_dim)` | 属性变化预测值 |

---

## 3. 三大组件详解

### 3.1 组件 A — 分子特征提取器（MoleculeFeatureExtractor）

**位置**：`core/models/v0/molecule_feature_extractors/`

**功能**：将分子图数据转换为固定维度的特征向量。每个完整模型包含 **两个独立实例**，分别处理 from-molecule 和 to-molecule。

| 提取器 | 类名 | 输出维度 | 核心技术 |
|--------|------|----------|----------|
| **GCN** | `GCNMoleculeFeatureExtractor` | 512 (mean+max cat) | 3层 GCNConv + 双池化拼接 |
| **VisNet** | `VisNetMoleculeFeatureExtractor` | 512 | 等变向量-标量交互消息传递（ViSNetBlock × 6） |
| **FragNet** | `FragNetMoleculeFeatureExtractor` | 256 (mean+max) | Fragment 分割 + GAT |
| **EquiformerV1** | `EquiformerV1MoleculeFeatureExtractor` | 512 (mean+max) | 等变 Transformer（SO(3) 等变性） |
| **TensorNet** | `TensorNetMoleculeFeatureExtractor` | 512 | 张量网络等变模型 |

**GCN 结构示意**（最基础的提取器）：
```
(x, edge_index)
 → GCNConv(node_dim→128) → ReLU
 → GCNConv(128→256) → ReLU
 → GCNConv(256→256)
 → global_mean_pool ─┐
 → global_max_pool  ─┘ → cat → 512d
```

---

### 3.2 组件 B — 边特征提取器（EdgeFeatureExtractor）

**位置**：`core/models/v0/edge_feature_extractors/`

**功能**：将操作信息（原子类型、操作类型等）编码为固定维度向量。

| 提取器 | 类名 | 默认维度变换 |
|--------|------|-------------|
| **Linear** | `LinearEdgeFeatureExtractor` | 11 → 64 → 128 → hidden_dim |
| **Transformer** | `TransformerEdgeFeatureExtractor` | 11 → 64 → 128 → d_model → TransformerEncoder(2L,8H) → squeeze |
| **Transformer+LAP** | `TransformerEdgeFeatureExtractorLap` | 同上 + 可选 Laplace 位置编码 |

**边特征构成**（默认 11 维）：
- 原子类型独热（5 维）：C, N, O, F, P
- 操作类型独热（6 维）：add, replace, del, add_multi, del_multi, complex

---

### 3.3 组件 C — 特征融合预测器（FusionPredictor）

**位置**：`core/models/v0/fusion_predictors/`

**功能**：拼接三者特征后，通过多层网络输出最终预测。

| 预测器 | 类名 | 结构 |
|--------|------|------|
| **MLP** | `MLPFusionPredictor` | cat → Linear(512) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → Linear(output) |
| **Transformer** | `TransformerFusionPredictor` | cat → 投影 → TransformerEncoder(3L,8H,d_model=256) → CLS 回归 |

---

## 4. 已注册模型清单

通过 `@register_model` 装饰器自动注册，支持工厂模式创建：

| 注册名 | 显示名 | 分子提取器 | 边提取器 | 融合预测器 |
|--------|--------|-----------|---------|-----------|
| `gcn_linear_linear` | GCN Linear-Linear | GCN | Linear | MLP |
| `gcn_transformer_linear` | GCN-Transformer-Linear | GCN | Transformer | MLP |
| `gcn_transformer_transformer` | GCN-TF-TF | GCN | Transformer | Transformer |
| `visnet_linear_linear` | VisNet L-L | VisNet | Linear | MLP |
| `visnet_transformer_linear` | ViSNet-TF-L | VisNet | Transformer | MLP |
| `frag_linear_linear` | FragNet L-L | FragNet | Linear | MLP |
| `equiformer_linear_linear` | Equiformer L-L | EquiformerV1 | Linear | MLP |
| `tensornet_linear_linear` | TensorNet L-L | TensorNet | Linear | MLP |

**使用方式**：
```python
from mol_evo.core.models.v0 import ModelFactory
model = ModelFactory.create("visnet_linear_linear", node_feature_dim=11, ...)
ModelFactory.list_models()   # 查看所有可用模型
```

---

## 5. 关键维度速查表

| 维度 | GCN | VisNet | FragNet | EquiformerV1 | TensorNet |
|------|-----|--------|---------|-------------|-----------|
| 分子提取器输出 | **512** | 512 | **256** | 512 | 512 |
| 默认边特征输入 | 11 | 15 | 16 | 11 | 15 |
| 边编码器输出 | 128~256 | 128~256 | 128 | 128~256 | 256~512 |
| 融合器输入 (2×mol+edge) | ~768~1280 | ~768~1280 | ~640 | ~768~1280 | ~1280~1536 |
| 融合器隐藏层 | 512→256→128 | 同左 | 同左 | 同左 | 同左 |

---

## 6. 代码目录结构

```
core/models/v0/
├── __init__.py                          # 注册系统 + ModelFactory 工厂 + BaseMoleculeEvolutionPredictor 基类
│
├── molecule_feature_extractors/         # 组件A: 分子特征提取器
│   ├── gcn.py                           #   GCN (3层 GCNConv)
│   ├── visnet.py                        #   VisNet (含 visnet_core/ 子包，完整 ViSNet 实现)
│   ├── fragnet.py                       #   FragNet (含 fragnet_core/)
│   ├── equiformer_v1.py                 #   EquiformerV1 (含 equiformer_v1_core/)
│   └── tensornet.py                     #   TensorNet (含 torchmdnet_t_core/)
│
├── edge_feature_extractors/             # 组件B: 边特征提取器
│   ├── linear.py                        #   多层线性映射
│   ├── transformer.py                   #   Transformer 编码器
│   └── transformer_lap.py              #   + Laplace 位置编码
│
├── fusion_predictors/                   # 组件C: 特征融合预测
│   ├── mlp.py                           #   多层感知机
│   └── transformer.py                   #   Transformer + CLS 回归
│
└── [各组合模型].py                       # 具体模型类（如 gcn_linear_linear.py）
    ├── gcn_linear_linear.py             #   = GCN + Linear + MLP
    ├── gcn_transformer_linear.py        #   = GCN + TF + MLP
    ├── gcn_transformer_transformer.py   #   = GCN + TF + TF
    ├── visnet_linear_linear.py          #   = VisNet + Linear + MLP
    ├── visnet_transformer_linear.py     #   = VisNet + TF + MLP
    ├── frag_linear_linear.py            #   = FragNet + Linear + MLP
    ├── equiformer_linear_linear.py      #   = EquiformerV1 + Linear + MLP
    └── tensornet_linear_linear.py       #   = TensorNet + Linear + MLP
```

---

## 7. 数据流全景

```
SMILES 字符串 (from, to)
  │
  ├─ RDKit 解析 → Mol 对象 → 3D 坐标生成 (ETKDG)
  │   ├─ 节点特征 x: 原子序数、杂化状态、度数、芳香性、形式电荷等 (~11~37维)
  │   ├─ 边索引 edge_index: COO 格式邻接关系
  │   ├─ 原子序数 z: 用于等变模型的类型嵌入
  │   └─ 坐标 pos: 3D 空间坐标（等变模型必需）
  │
  ▼
PyG Data 对象 (from_data, to_data)  +  edge_attr 张量
  │
  ├─► MoleculeFeatureExtractor(from_data) → from_feat  (512d 或 256d)
  ├─► MoleculeFeatureExtractor(to_data)   → to_feat    (512d 或 256d)
  └─► EdgeFeatureExtractor(edge_attr)     → edge_feat  (128~512d)
  │
  ▼
FusionPredictor(cat([from_feat, to_feat, edge_feat])) → property_delta
```

---

## 8. 训练与推理 CLI

### 训练入口
```bash
conda activate mol-opt-evo
python mol_evo/train_v0.py --config-file <config.yaml> [--model-type gcn_linear_linear]
# 或直接命令行参数:
python mol_evo/train_v0.py --max-pairs 30000 --epochs 200 --lr 0.001 --target-property gap_change_pct
```

配置文件格式（YAML）分为 `train`（训练参数）和 `model`（模型超参数）两个顶层 section。

### 推理入口
```bash
# 单样本预测
python mol_evo/predict_v0.py --model-path <path> --smiles-from 'CCO' --smiles-to 'CC=O' ...

# 批量测试集评估
python mol_evo/predict_v0_testset.py --model-path <path> --data-file <data.json> -mpairs 30000 ...
```

### 上游搜索集成
训练好的模型会被 **分子优化搜索引擎**（`scripts/batch_optimizer.py`）调用，支持两种搜索模式：
- **BFS**: 宽度优先搜索，适用于 lumo/homo 等属性优化
- **MCTS**: 蒙特卡洛树搜索，支持更深的探索深度

同时也支持 **IC50（药物活性）优化任务**（`scripts/batch_optimizer_ic50.py`），使用 SchNet 预测器。

---

## 9. 版本演进脉络

| 版本 | 核心变更 | 文档位置 |
|------|----------|----------|
| **v0** (当前) | 基础三组件架构，8 种模型组合 | `docs/models/model-v0/model-v0.md` |
| **v0.1** | 新增特征差值组件（to_feat − from_feat → 补充边特征输入） | `docs/models/model-v0/model-v0.1/` |
| **v0.2** | （见子目录文档） | `docs/models/model-v0/model-v0.2/` |
| **v0.3** | 链式自回归扩展——滑动窗口迭代实现路径级累积预测 | `docs/models/model-v0/model-v0.3/` |

---

## 10. 关键依赖

| 依赖库 | 用途 |
|--------|------|
| **PyTorch Geometric (PyG)** | 图神经网络基础 (`Data`, `Batch`, `GCNConv`, pool) |
| **RDKit** | SMILES → 分子图解析、3D 坐标生成、描述符计算 |
| **torch-scatter / torch-cluster / torch-sparse** | PyG 的稀疏图运算后端 |
| **equiformer_v1_core** (内嵌) | EquiformerV1 等变模型实现 |
| **visnet_core** (内嵌) | ViSNet 等变消息传递实现 |
| **torchmd-net** (内嵌) | TensorNet 实现 |

---

## 11. 注意事项

1. **模型选择**：若需快速验证思路，优先用 `gcn_linear_linear`；若追求精度，考虑 `visnet_linear_linear` 或 `tensornet_linear_linear`
2. **边特征维度**：不同分子提取器对应的默认 `edge_feature_dim` 不同（GCN/Equiformer 为 11，VisNet/TensorNet 为 15，FragNet 为 16），构造模型时需注意匹配
3. **位置编码需求**：部分模型（如 EquiformerV1）需要 3D 坐标作为输入；注册信息中 `requires_position_encoding` 字段可查
4. **批量推理**：所有模型已支持 PyG Batch 批量输入（通过 `global_mean_pool(x, batch)` 处理），可直接用于 DataLoader 加速
5. **FragNet 特殊模式**：`frag_linear_linear` 额外支持列表式逐样本输入（one-by-one 场景）
6. **Transformer 类模型**：学习率需调低（建议 ≤ 0.001），否则易出现梯度爆炸
7. **训练配置文件**：推荐 YAML 方式管理参数，支持从配置文件名自动推断模型类型
