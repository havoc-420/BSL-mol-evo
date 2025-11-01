根据您的需求，我修改了流程图，在 edge feature 处理中加入 to-smile 特征减去 from-smile 特征的操作：

```mermaid
flowchart TD
    %% 定义样式
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,color:#000
    classDef processStyle fill:#f3e5f5,stroke:#4a148c,color:#000
    classDef modelStyle fill:#e8f5e8,stroke:#1b5e20,color:#000
    classDef outputStyle fill:#fff3e0,stroke:#e65100,color:#000
    classDef componentStyle fill:#fce4ec,stroke:#880e4f,color:#000
    classDef operationStyle fill:#fff9c4,stroke:#f57f17,color:#000
    
    %% 输入部分
    input1[(起始分子<br/>From SMILES)]:::inputStyle
    input2[(目标分子<br/>To SMILES)]:::inputStyle
    input3[(操作特征<br/>Edge Features)]:::inputStyle
    
    %% 主要处理模块
    subgraph feature_extractor[分子特征提取]
        A[MoleculeFeatureExtractor]
        molecule_component[["
        • GCN
        • VisNet
        "]]:::componentStyle
    end
    
    subgraph edge_processing[边特征处理]
        B[Edge Feature Extractor]
        edge_component[["
         • Linear
         • Transformer
        "]]:::componentStyle
        
        %% 新增特征差值计算
        D[特征差值计算<br/>To - From]:::operationStyle
    end
    
    subgraph fusion_prediction[特征融合预测]
        C[Fusion Predictor]
        fusion_component[["
         • MLPFusion
         • TransformerFusion
         "]]:::componentStyle
    end
    
    %% 输出
    output[(属性变化预测<br/>Property Changes)]:::outputStyle
    
    %% 连接关系 - 修改后的数据流
    input1 --> A
    input2 --> A
    input3 --> B
    
    A --> molecule_component
    molecule_component --> C
    
    %% 新增连接：分子特征提取器输出到特征差值计算
    A --> D
    D --> B
    
    B --> edge_component
    edge_component --> C
    
    C --> fusion_component
    fusion_component --> output
    
    %% 样式应用
    class feature_extractor modelStyle;
    class edge_processing modelStyle;
    class fusion_prediction modelStyle;
```

## 修改说明

主要修改点：

1. **在边特征处理模块中新增了"特征差值计算"组件**：
   - 接收来自分子特征提取器的 to-smile 和 from-smile 特征
   - 计算 `to-smile特征 - from-smile特征` 的差值

2. **调整了数据流向**：
   - 分子特征提取器现在同时输出到融合预测器和特征差值计算器
   - 特征差值计算结果作为边特征处理模块的额外输入

3. **边特征处理模块的输入现在包括**：
   - 原始的 edge 特征
   - to-smile 特征减去 from-smile 特征的差值

这样的设计能够更好地捕捉分子转化过程中的结构变化信息，为边特征提供更丰富的上下文信息。