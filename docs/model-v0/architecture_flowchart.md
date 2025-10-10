# Molecular Evolution Prediction Model v0 - Flowchart

```mermaid
flowchart TD
    %% 定义样式
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,color:#000
    classDef processStyle fill:#f3e5f5,stroke:#4a148c,color:#000
    classDef modelStyle fill:#e8f5e8,stroke:#1b5e20,color:#000
    classDef outputStyle fill:#fff3e0,stroke:#e65100,color:#000
    
    %% 输入部分
    input1[(From SMILES<br/>起始分子)]:::inputStyle
    input2[(To SMILES<br/>目标分子)]:::inputStyle
    input3[(Edge Features<br/>操作信息)]:::inputStyle
    
    %% 分子特征提取模块
    subgraph feature_extractor[分子特征提取模块]
        subgraph from_mol_extractor[起始分子特征提取器]
            from_gcn1[GCN Layer 1]:::processStyle
            from_gcn2[GCN Layer 2]:::processStyle
        end
        
        subgraph to_mol_extractor[目标分子特征提取器]
            to_gcn1[GCN Layer 1]:::processStyle
            to_gcn2[GCN Layer 2]:::processStyle
        end
    end
    
    %% 边特征处理模块
    subgraph edge_processing[边特征处理模块]
        edge_encoder[Edge Encoder<br/>Linear Layers]:::processStyle
    end
    
    %% 特征融合与预测模块
    subgraph fusion_prediction[特征融合与预测模块]
        concat[特征拼接<br/>Concatenation]:::processStyle
        predictor[Predictor<br/>Linear Layers]:::processStyle
    end
    
    %% 输出
    output[(Property Changes<br/>属性变化预测值)]:::outputStyle
    
    %% 连接关系
    input1 --> from_mol_extractor
    input2 --> to_mol_extractor
    input3 --> edge_encoder
    
    from_gcn1 --> from_gcn2
    to_gcn1 --> to_gcn2
    
    from_gcn2 --> concat
    to_gcn2 --> concat
    edge_encoder --> concat
    
    concat --> predictor
    predictor --> output
    
    %% 修正的类定义 - 每行一个类应用
    class feature_extractor modelStyle;
    class edge_processing modelStyle;
    class fusion_prediction modelStyle;
```

## 流程图说明

该流程图展示了基于GCN的分子进化预测模型v0版本的整体架构和数据流向：

1. **输入层**：
   - 起始分子（From SMILES）
   - 目标分子（To SMILES）
   - 操作信息（Edge Features）

2. **分子特征提取模块**：
   - 使用两个独立的GCN网络分别处理起始分子和目标分子
   - 每个GCN网络包含两层GCN层

3. **边特征处理模块**：
   - 使用线性网络对操作信息进行编码

4. **特征融合与预测模块**：
   - 将起始分子特征、目标分子特征和边特征进行拼接
   - 通过线性层预测属性变化

5. **输出层**：
   - 输出预测的属性变化值