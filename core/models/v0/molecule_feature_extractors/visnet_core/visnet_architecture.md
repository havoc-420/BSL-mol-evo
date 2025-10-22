您说得对！我来重新绘制一个更适合显示的版本，避免使用深色背景：

```mermaid
graph TD
    %% 输入层 - 使用浅蓝色
    A[原子类型 z] --> F[ViSNet 模型]
    B[原子坐标 pos] --> F
    C[批次索引 batch] --> F
    
    %% 预处理模块 - 使用浅绿色
    F --> G{Distance 模块}
    G --> H[边索引 edge_index]
    G --> I[边权重 edge_weight]
    G --> J[边向量 edge_vec]
    
    I --> K{ExpNormalSmearing}
    J --> L{Sphere}
    
    K --> M[边特征 edge_attr]
    L --> N[球谐特征]
    
    %% 嵌入模块 - 使用浅黄色
    A --> O{Embedding}
    O --> P[原子嵌入 x]
    
    P --> Q{NeighborEmbedding}
    M --> Q
    H --> Q
    I --> Q
    Q --> R[邻居增强特征]
    
    R --> S{EdgeEmbedding}
    M --> S
    H --> S
    S --> T[边嵌入特征]
    
    %% 初始化向量特征
    R --> U[初始化向量特征 vec]
    
    %% ViSNetBlock 循环层 - 使用浅橙色
    subgraph ViSNetBlock - 多层消息传递
        V[输入标量特征 x] --> W[ViS_MP / ViS_MP_Vertex]
        U[输入向量特征 vec] --> W
        T[边特征 edge_attr] --> W
        N[球谐特征] --> W
        I[边权重] --> W
        H[边索引] --> W
        
        W --> X[更新标量特征 dx]
        W --> Y[更新向量特征 dvec]
        W --> Z[更新边特征 dedge_attr]
        
        X --> AA[残差连接 x = x + dx]
        Y --> BB[残差连接 vec = vec + dvec]
        Z --> CC[残差连接 edge_attr = edge_attr + dedge_attr]
        
        AA --> V
        BB --> U
        CC --> T
    end
    
    %% 输出归一化 - 使用浅紫色
    AA --> DD{LayerNorm}
    BB --> EE{VecLayerNorm}
    
    DD --> FF[归一化标量特征]
    EE --> GG[归一化向量特征]
    
    %% 输出模块 - 使用浅红色
    FF --> HH{EquivariantScalar}
    GG --> HH
    C --> HH
    
    HH --> II[原子级输出]
    HH --> JJ[图级表示 graph_rep]
    
    II --> KK{Scatter 聚合}
    C --> KK
    KK --> LL[分子性质预测 y]
    
    %% 先验模型
    A --> MM{Atomref}
    LL --> MM
    MM --> NN[最终输出]
    
    %% 导数计算分支 - 使用浅粉色
    B --> OO[位置梯度计算]
    NN --> OO
    OO --> PP[力预测 -dy]
    
    %% 使用更浅的颜色，确保文字可读性
    classDef input fill:#e3f2fd,stroke:#1976d2,color:#000000
    classDef preprocess fill:#e8f5e8,stroke:#388e3c,color:#000000
    classDef embedding fill:#fff3e0,stroke:#f57c00,color:#000000
    classDef messagepassing fill:#fce4ec,stroke:#c2185b,color:#000000
    classDef output fill:#fff8e1,stroke:#ff8f00,color:#000000
    classDef derivative fill:#f3e5f5,stroke:#7b1fa2,color:#000000
    classDef normalization fill:#e0f2f1,stroke:#00796b,color:#000000
    
    class A,B,C input
    class G,K,L preprocess
    class O,Q,S embedding
    class W,V,U,T messagepassing
    class HH,KK,MM,NN output
    class OO,PP derivative
    class DD,EE normalization
```

## 🎨 颜色方案说明

现在使用了更浅的颜色方案，确保黑色文字清晰可读：

- **输入层**: 浅蓝色 `#e3f2fd`
- **预处理**: 浅绿色 `#e8f5e8` 
- **嵌入层**: 浅橙色 `#fff3e0`
- **消息传递**: 浅粉色 `#fce4ec`
- **归一化**: 浅青绿色 `#e0f2f1`
- **输出层**: 浅黄色 `#fff8e1`
- **导数计算**: 浅紫色 `#f3e5f5`

## 📊 数据流简化版本

如果您需要更简洁的版本，这里还有一个层次化的结构图：

```mermaid
graph TD
    A[输入数据] --> B[几何特征提取]
    B --> C[初始嵌入]
    C --> D[ViSNet Block]
    
    subgraph D[ViSNet Block - 6层]
        D1[第1层] --> D2[第2层]
        D2 --> D3[...]
        D3 --> D4[第6层]
    end
    
    D --> E[输出处理]
    E --> F[分子性质 y]
    E --> G[图表示 graph_rep]
    E --> H[原子力 -dy]
    
    %% 样式 - 更简洁的颜色
    classDef stage1 fill:#e3f2fd,stroke:#1976d2,color:#000000
    classDef stage2 fill:#e8f5e8,stroke:#388e3c,color:#000000  
    classDef stage3 fill:#fff3e0,stroke:#f57c00,color:#000000
    classDef stage4 fill:#fce4ec,stroke:#c2185b,color:#000000
    classDef output fill:#fff8e1,stroke:#ff8f00,color:#000000
    
    class A,B stage1
    class C stage2
    class D stage3
    class E stage4
    class F,G,H output
```

这样的颜色方案既保持了模块的区分度，又确保了所有文字都清晰可读。您可以根据需要选择详细版本或简化版本。