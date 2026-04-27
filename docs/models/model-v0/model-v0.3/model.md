# v0.3 链式自回归路径模型

## 概述

v0.3 采用**链式自回归（滑动窗口迭代）架构**，在共享 v0 基础模型上，逐步预测属性变化 Δ，实现路径级别的累积预测。

### 数据条件

**已知信息**：
- `start_smiles`, `end_smiles`：起始和终止分子
- `operations`：操作序列
- `path_target`：路径总属性变化

**未知信息**：
- 中间节点 SMILES（需从 operations 推导，可能生成非法 SMILES）
- 每步属性变化（黑箱，无监督信号）

### 训练策略

由于只有**路径总监督信号**，采用**端到端链式预测**：
1. 从 `start_smiles` 逐步应用 `operations` 推导中间节点
2. 链式迭代预测每步变化 Δ̂i
3. 累积得到路径总预测 ΣΔ̂i
4. 损失只在路径级别计算，梯度通过链式结构反向传播

### 与 v0 的关键区别

| 维度 | v0 (单步) | v0.3 (链式自回归) |
|------|-----------|-------------------|
| 样本格式 | `(from, to, edge)` pair | `(start, operations, end)` path |
| 模型架构 | 独立预测单步变化 | 共享 v0，链式迭代预测 |
| 训练策略 | 单步监督 | 端到端路径监督 |
| 监督信号 | 单步属性变化 | 仅路径总变化 |
| 梯度传播 | 单步反向传播 | 路径损失 → 链式回传 |
| 中间节点 | 直接输入 | 从 operations 推导 |

---

## 核心设计思想

### 架构流程

```
输入: start_smiles, operations = [op1, op2, op3], path_target

┌─────────────────────────────────────────────────────────────────┐
│                      中间节点推导层                               │
├─────────────────────────────────────────────────────────────────┤
│  a1 (start)                                                      │
│      │                                                           │
│      ├── op1 ──→ a2 (推导，可能非法)                              │
│      │                                                           │
│      ├── op2 ──→ a3 (推导，可能非法)                              │
│      │                                                           │
│      └── op3 ──→ a4 (end)                                        │
│                                                                 │
│  注：中间节点通过分子操作工具推导，无真实属性变化标签               │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     链式预测层（共享 v0）                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: (a1, a2, e1) → [v0] → Δ̂1                              │
│                              ↓                                  │
│  Step 2: (a2, a3, e2) → [v0] → Δ̂2                              │
│                              ↓                                  │
│  Step 3: (a3, a4, e3) → [v0] → Δ̂3                              │
│                              ↓                                  │
│                      路径预测: ΣΔ̂i                              │
│                                                                 │
│  注：所有步骤共享同一个 v0 模型参数                               │
│      中间节点非法时，该步骤跳过或使用占位特征                       │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         损失计算层                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  监督信号: path_target (仅路径总变化)                             │
│                                                                 │
│  L = L1Loss(ΣΔ̂i, path_target)                                  │
│                                                                 │
│  梯度传播: L → ΣΔ̂i → [Δ̂1, Δ̂2, Δ̂3] → 共享 v0                    │
│                                                                 │
│  关键: 虽然没有每步监督，但链式结构强制模型学习逐步推理             │
└─────────────────────────────────────────────────────────────────┘
```

### 与 Transformer 并行方案对比

```
┌─────────────────────────────────────────────────────────────────┐
│              方案 A：Transformer 并行编码（原 v0.3 设计）          │
├─────────────────────────────────────────────────────────────────┤
│  [a1] [a2] [a3] [a4]                                            │
│    │    │    │    │                                             │
│    ▼    ▼    ▼    ▼                                             │
│  ┌─────────────────────┐                                        │
│  │  Transformer Encoder │  ← 所有步骤并行编码                    │
│  └─────────────────────┘                                        │
│           │                                                     │
│           ▼                                                     │
│       Path Head → path_pred                                     │
│                                                                 │
│  问题: 并行编码无法显式建模步骤间的因果关系                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              方案 B：链式自回归（当前设计）                         │
├─────────────────────────────────────────────────────────────────┤
│  (a1, a2, e1) → [v0] → Δ̂1                                      │
│                     ↓                                           │
│  (a2, a3, e2) → [v0] → Δ̂2                                      │
│                     ↓                                           │
│  (a3, a4, e3) → [v0] → Δ̂3                                      │
│                     ↓                                           │
│                ΣΔ̂i → path_pred                                  │
│                                                                 │
│  优势: 显式建模步骤间的链式依赖，梯度精细化回传                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 路径样本格式 (JSON)

```json
{
    "path_id": "path_001",
    "start_smiles": "CCO",
    "end_smiles": "CCNCC",
    "operations": [
        {"atom": "N", "operation": "replace_atom", "position": "2"},
        {"atom": "C", "operation": "add_atom", "position": "2"},
        {"atom": "C", "operation": "add_atom", "position": "3"}
    ],
    "target_property": "lumo_change",
    "path_target": -0.35
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `start_smiles` | ✅ | 起始分子 SMILES |
| `end_smiles` | ✅ | 终止分子 SMILES |
| `operations` | ✅ | 操作序列，长度 = 路径步数 T |
| `path_target` | ✅ | 路径总属性变化（唯一监督信号） |
| `target_property` | ❌ | 目标属性名称 |

### 数据预处理

中间节点在训练时动态推导：

```python
def derive_intermediate_nodes(start_smiles: str, operations: List[dict]) -> List[str]:
    """
    从 start_smiles 逐步应用 operations，推导中间节点
    
    Returns:
        node_smiles_list: [start, mid1, mid2, ..., end]
        其中部分中间节点可能是非法 SMILES（用 None 或占位符表示）
    """
    nodes = [start_smiles]
    current = start_smiles
    
    for op in operations:
        current = apply_operation(current, op)
        nodes.append(current)
    
    return nodes  # 长度 = T + 1
```

---

## 模型架构

### 核心类设计

```python
class VisNetChainPathV03(nn.Module):
    """链式自回归路径模型 - 端到端路径监督"""
    
    def __init__(self, v0_config: dict):
        super().__init__()
        # 共享 v0 基础模型
        self.v0 = MoleculeEvolutionVisnetLinearPredictor(**v0_config)
        # 非法节点占位特征
        self.invalid_node_embedding = nn.Parameter(torch.randn(512))
        
    def forward(
        self,
        node_pairs: List[Tuple[Data, Data]],  # [(a1, a2), (a2, a3), (a3, a4)]
        edge_features: torch.Tensor,           # (B, T, 15)
        valid_mask: torch.Tensor,              # (B, T) 每步是否有效
        path_target: torch.Tensor = None       # (B, 1)
    ) -> Dict:
        """
        端到端链式预测
        
        Args:
            node_pairs: 从 operations 推导的节点对
            edge_features: 操作特征
            valid_mask: 步骤有效性掩码（中间节点非法时为 0）
            path_target: 路径总属性变化（唯一监督信号）
        """
        B, T = edge_features.shape[:2]
        step_preds = []
        
        # 链式迭代预测
        for t in range(T):
            from_data, to_data = node_pairs[t]
            edge_t = edge_features[:, t, :]
            
            # 跳过无效步骤
            if not valid_mask[:, t].all():
                step_preds.append(torch.zeros(B, 1, device=edge_t.device))
                continue
            
            # v0 预测单步变化
            delta_t = self.v0(from_data, to_data, edge_t)
            step_preds.append(delta_t)
        
        # 聚合
        step_preds = torch.cat(step_preds, dim=1)  # (B, T)
        path_pred = step_preds.sum(dim=1, keepdim=True)  # (B, 1)
        
        output = {
            "step_preds": step_preds,
            "path_pred": path_pred
        }
        
        # 路径级监督损失
        if path_target is not None:
            output["loss"] = F.l1_loss(path_pred, path_target)
        
        return output
```

### 架构图

```
输入: start_smiles, operations = [op1, op2, op3], path_target
      (中间节点动态推导)

┌─────────────────────────────────────────────────────────────────┐
│                      中间节点推导层                               │
├─────────────────────────────────────────────────────────────────┤
│  a1 (start) ── op1 ──→ a2 (推导)                                │
│  a2 ────────── op2 ──→ a3 (推导)                                │
│  a3 ────────── op3 ──→ a4 (end)                                 │
│                                                                 │
│  输出: node_pairs = [(a1, a2), (a2, a3), (a3, a4)]             │
│        valid_mask = [1, 0, 1]  # a2 非法时                      │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     链式预测层（共享 v0）                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │ Step 1: (a1, a2, e1) → [v0] → Δ̂1        │                   │
│  └──────────────────────────────────────────┘                   │
│                      ↓                                          │
│  ┌──────────────────────────────────────────┐                   │
│  │ Step 2: (a2, a3, e2) → 跳过（非法节点）    │                   │
│  └──────────────────────────────────────────┘                   │
│                      ↓                                          │
│  ┌──────────────────────────────────────────┐                   │
│  │ Step 3: (a3, a4, e3) → [v0] → Δ̂3        │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│  注：所有步骤共享同一个 v0 模型参数                               │
│      非法步骤跳过，不参与预测和损失计算                            │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         损失计算层                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  路径预测: path_pred = Σ Δ̂i (仅有效步骤)                         │
│                                                                 │
│  监督信号: path_target                                          │
│                                                                 │
│  损失: L = L1Loss(path_pred, path_target)                       │
│                                                                 │
│  梯度传播: L → path_pred → [Δ̂1, Δ̂3] → 共享 v0                  │
│                                                                 │
│  关键: 无每步监督，但链式结构强制学习逐步推理                      │
└─────────────────────────────────────────────────────────────────┘
```

### 参数量估算

| 组件 | 参数 |
|------|------|
| 共享 v0 模型 | ~2.5M |
| **总计** | **~2.5M** |

相比 Transformer 并行方案（~2.6M），参数量相近，但模型结构更简洁。

---

## 损失函数

```python
# 仅路径级监督
L = L1Loss(path_pred, path_target)

# 其中 path_pred = Σ Δ̂i (仅有效步骤)
```

### 损失设计说明

| 特性 | 说明 |
|------|------|
| **单一监督信号** | 只有路径总变化作为监督 |
| **链式梯度传播** | 损失通过链式结构反向传播到每一步 |
| **隐式步骤学习** | 模型被迫学习每步合理的 Δ 分布 |
| **非法步骤处理** | 无效步骤不参与累积，直接跳过 |

---

## 非法节点处理

### 规则

1. **首尾节点非法** → 直接丢弃整条样本
2. **中间节点非法** → 该步骤跳过，不参与损失计算

### 实现示例

```python
# 节点合法性: [✓, ✓, ✗, ✓]
# 有效步骤: [step_0, step_2]
# 跳过: step_1（涉及非法节点 a3）

valid_mask = torch.tensor([1, 0, 1])  # 步骤有效性掩码
step_loss = (step_loss * valid_mask).sum() / valid_mask.sum()
```

---

## 训练配置

```bash
python mol_evo/train_v0_3_chain.py \
    -d mol_evo/dataset/data/paths.json \
    -p lumo_change \
    -e 200 \
    -b 32 \
    --path-loss-weight 0.5
```

### 关键超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `path_loss_weight` | 0.5 | 路径总损失权重（λ） |
| `max_path_length` | 20 | 最大路径步数 |
| `learning_rate` | 1e-4 | 学习率 |

---

## 推理模式

### 给定路径评估（v0.3 主任务）

```python
def evaluate_path(
    self,
    start_smiles: str,
    operations: List[dict]
) -> float:
    """
    给定路径，评估总属性变化
    
    Args:
        start_smiles: 起始分子
        operations: 操作序列
    
    Returns:
        total_delta: 预测的路径总属性变化
    """
    # 推导中间节点
    nodes = derive_intermediate_nodes(start_smiles, operations)
    
    # 链式预测
    total_delta = 0
    for i in range(len(operations)):
        from_data = encode_molecule(nodes[i])
        to_data = encode_molecule(nodes[i + 1])
        edge = encode_operation(operations[i])
        
        delta = self.v0(from_data, to_data, edge)
        total_delta += delta
    
    return total_delta
```

### 批量路径比较

```python
def rank_paths(
    self,
    start_smiles: str,
    candidate_paths: List[List[dict]]
) -> List[Tuple[float, List[dict]]]:
    """
    对多条候选路径排序
    
    Args:
        start_smiles: 起始分子
        candidate_paths: 多条操作序列
    
    Returns:
        ranked: [(predicted_delta, operations), ...]
    """
    results = []
    for ops in candidate_paths:
        delta = self.evaluate_path(start_smiles, ops)
        results.append((delta, ops))
    
    return sorted(results, key=lambda x: x[0])
```

---

## 输出目录结构

```
mol_evo/output/v0_3/visnet_chain_v0_3/train-{timestamp}-{property}/
├── last.pth                 # 模型权重
├── model_config.json        # 模型与训练配置
├── training_process.json    # 训练过程记录
└── training.log             # 训练日志
```

---

## 优势总结

| 特性 | 说明 |
|------|------|
| **链式显式建模** | 每步预测显式可解释，非黑箱并行 |
| **参数共享** | 所有步骤共享 v0，参数效率高 |
| **梯度精细化** | 路径损失通过链式结构精细化回传到每步 |
| **非法节点容错** | 动态推导时自动处理非法中间节点 |
| **路径评估友好** | 直接输出每步预测，便于分析和调试 |

---

## 后续扩展方向

1. **路径搜索**: 结合分子生成器做 beam search
2. **多属性联合训练**: 扩展为多任务预测
3. **课程学习**: 从短路径逐步扩展到长路径
4. **强化学习**: 引入中间奖励信号

---

## 潜在问题与解决

### 问题1：无每步监督，如何学习合理的逐步分布？

**问题本质**

模型只接收路径总属性变化 \( y = \sum_t \Delta_t \) 作为监督，但需要输出每步预测 \( \hat{\Delta}_t \)。这是一个 **ill-posed decomposition** 问题——满足 \( \sum_t \hat{\Delta}_t = y \) 的分解方式有无穷多种。如果不加任何约束，模型可能将所有属性变化集中在某一步，其余步置零，这虽然数值上拟合总量，但化学上不合理且泛化性差。

**解决策略（按优先级排列）**

1. **v0 单步预训练初始化**（首选）
   - 先在 v0 的单步 `(from, to, edge) → Δ` 数据上预训练共享 backbone（节点编码器 + 边编码器 + 步骤编码器）
   - 预训练后，模型已具备"局部结构变化 → 属性变化"的基本映射能力
   - 路径级微调时从此出发，而非从随机参数猜测每步贡献
   - 实现方式：加载 v0 模型权重到共享编码器，冻结若干轮后解冻

2. **课程学习（short-to-long）**
   - 训练初期仅使用 1~2 步短路径，逐步引入 3~5 步、再到更长路径
   - 短路径时路径总监督对单步分配约束更强（自由度更小），有利于模型学到合理的局部分解
   - 实现方式：训练脚本中按 epoch 动态调整 `max_path_length` 上限

3. **弱幅度正则**
   - 防止模型将全部变化硬塞到某一步，使用 L2 幅度惩罚：
     ```
     L = L_path + λ_mag · Σ_t (Δ̂_t)²
     ```
   - 或使用 L1 正则（稀疏性更强）：`λ_mag · Σ_t |Δ̂_t|`
   - 注意：**不建议**使用强"平均分配"假设（如相邻步差异平滑约束），因为化学反应中各步属性变化幅度可能差异很大

4. **可选：路径间排序损失（Ranking Loss）**
   - 如果同一 `start_smiles` 下有多条候选路径且路径总变化大小关系已知，可额外加入 margin ranking loss
   - 这能帮助模型区分路径质量，而不仅仅拟合绝对值

**默认推荐组合**：`v0 预训练初始化` + `short-to-long 课程学习` + `弱 L2 幅度正则`

### 问题2：非法中间节点如何处理？

**问题本质**

中间节点从 operations 逐步推导，可能生成非法 SMILES（化学键价不合法、环闭合失败、解析器失败等）。当前方案是"跳过非法步骤"，但这会导致：
- 路径总标签 `path_target` 仍然包含该步的真实属性影响
- 跳过等价于默认该步贡献为 0，造成监督与输入语义不一致
- 训练存在系统性偏差

**解决策略**

1. **首尾节点非法 → 丢弃整条样本**
   - 首尾节点是路径的锚点，非法则整条路径语义不可信
   - 当前实现已支持此策略

2. **中间节点非法 → 显式建模，而非简单跳过**
   - 保留 `invalid_node_embedding` 作为非法节点的可学习表示（当前已有）
   - 非法步骤仍参与 step encoder 和 path transformer 编码，但通过以下方式区分：
     - 输入端：标记 `invalid_transition_flag`，附加错误类型信息（valence error / ring closure / parser fail）
     - 模型端：step token 正常参与 Transformer 上下文建模（不跳过）
     - 输出端：step_valid_mask 控制该步预测是否参与路径加总

3. **辅助合法性预测头（validity head）**
   - 这是一个"白送"的监督信号——每个中间节点是否合法在数据构建时就已知
   - 增加辅助二分类头，预测每步的合法概率：
     ```
     L = L_path + λ_valid · L_validity
     ```
   - 作用：帮助模型学到"结构可靠性"特征，改善整体表示质量

4. **推理时输出路径置信度**
   - 根据路径中非法步骤比例和 validity head 预测，输出路径有效率
   - 非法比例高的路径标记低置信度，排序时降权

**默认推荐组合**：`首尾非法丢弃` + `中间非法显式编码（保留 invalid_node_embedding + invalid flag）` + `辅助合法性预测头` + `推理输出置信度`

### 问题3：长路径监督变粗与训练不稳定

**问题本质**

当前模型的路径预测形式为 \( \hat{y}_{path} = \sum_t \hat{\Delta}_t \)，从 path loss 到每个 \( \hat{\Delta}_t \) 的梯度路径是直接的（不是 RNN 那种层层乘法导致的梯度消失）。真正的问题是：
- **监督变粗**：路径越长，每一步从路径总损失中获得的 credit assignment 信号越弱
- **误差累积**：多步的预测误差在求和时累积，长路径整体噪声更大
- **训练统计不均匀**：长路径和短路径的损失量级、梯度分布不同，混合训练时优化不稳定
- **padding 开销**：batch 内路径长度差异大时，短路径大量 padding 浪费计算

**解决策略（按优先级排列）**

1. **长度课程学习（与问题1联动）**
   - 先训短路径，逐步扩展到长路径
   - 短路径阶段同时解决了"分解歧义"和"训练稳定性"两个问题

2. **按路径长度分桶（Length Bucketing）**
   - 类似 NLP 中按序列长度分桶
   - 同一 batch 内路径长度尽量接近，减少 padding 浪费
   - 长短路径的梯度统计更均匀，优化更稳定

3. **分段聚合（Hierarchical Aggregation）**（长路径增强）
   - 当路径长度 T > 8 或 T > 10 时，先将步骤按 3~5 步分段（chunk）
   - 段内聚合得到 chunk score，再跨段聚合得到 path score
   - 相比纯全局求和更稳定，也更利于长程依赖建模

4. **可选：Path-level Correction Head**
   - 在主干 `Σ Δ̂_t` 基础上增加轻量修正项：
     ```
     ŷ_path = Σ_t Δ̂_t + c_path
     ```
   - `c_path` 由整条路径的全局表示经小型 MLP 得到
   - 保留可解释性的同时，补偿步骤间非线性交互作用

5. **梯度裁剪（工程稳定性增强）**
   - 保留 `clip_grad_norm_(max_norm=1.0)` 防止梯度爆炸
   - 这是训练稳定性的兜底措施，不是核心方案

**默认推荐组合**：`长度课程学习` + `按长度分桶` + `梯度裁剪`，超长路径场景可追加 `分段聚合`
