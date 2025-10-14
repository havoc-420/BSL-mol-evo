import torch
import torch.nn as nn
import torch.optim as optim
import math

class TransformerTextClassifier(nn.Module):
    """
    基于Transformer的文本分类器
    用于情感分析（正面/负面）
    """
    
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2, 
                 num_classes=2, max_seq_length=50, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 1. 词嵌入层 - 将单词ID转换为向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 位置编码 - 让模型知道单词的位置信息
        self.position_encoding = self._create_position_encoding(max_seq_length, d_model)
        
        # 3. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True  # 输入格式: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def _create_position_encoding(self, max_len, d_model):
        """创建位置编码"""
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pos_encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        
        return nn.Parameter(pos_encoding, requires_grad=False)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: 输入token IDs (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
        Returns:
            logits: 分类logits (batch_size, num_classes)
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. 词嵌入
        # input_ids: (batch_size, seq_len) → (batch_size, seq_len, d_model)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # 2. 添加位置编码
        # 只取前seq_len个位置编码
        x = x + self.position_encoding[:seq_len, :]
        
        # 3. Transformer编码
        # 如果有注意力掩码，需要调整格式
        if attention_mask is not None:
            # Transformer需要 (seq_len, seq_len) 的掩码或 (batch_size, seq_len) 的key_padding_mask
            # 我们使用key_padding_mask
            key_padding_mask = attention_mask == 0  # 为True的位置会被忽略
        else:
            key_padding_mask = None
            
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        # x形状: (batch_size, seq_len, d_model)
        
        # 4. 池化 - 取第一个token ([CLS]) 或平均池化
        # 这里使用平均池化
        if attention_mask is not None:
            # 考虑padding，只对非padding位置求平均
            x = x * attention_mask.unsqueeze(-1)  # 将padding位置置为0
            seq_lengths = attention_mask.sum(dim=1, keepdim=True)  # 每个序列的实际长度
            pooled_output = x.sum(dim=1) / seq_lengths  # 按实际长度平均
        else:
            pooled_output = x.mean(dim=1)  # 全局平均池化
        
        # 5. 分类
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)
        
        return logits

# 示例：模拟数据训练过程
def demonstrate_transformer():
    # 模拟参数
    vocab_size = 5000  # 词汇表大小
    batch_size = 4
    seq_length = 20
    
    # 创建模型
    model = TransformerTextClassifier(
        vocab_size=vocab_size,
        d_model=128,
        nhead=8,
        num_layers=2,
        num_classes=2,  # 正面/负面
        max_seq_length=50
    )
    
    print("模型结构:")
    print(model)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟输入数据
    # 假设我们有4个句子，每个句子20个token（不够的用0填充）
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)  # 1表示真实token
    
    # 模拟一个短句子，后面是padding
    attention_mask[1, 10:] = 0  # 第二个句子只有前10个token是真实的
    attention_mask[3, 15:] = 0  # 第四个句子只有前15个token是真实的
    
    print(f"\n输入数据:")
    print(f"input_ids形状: {input_ids.shape}")      # (4, 20)
    print(f"attention_mask形状: {attention_mask.shape}")  # (4, 20)
    print(f"attention_mask:\n{attention_mask}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        print(f"\n输出形状: {outputs.shape}")  # (4, 2)
        print(f"输出logits:\n{outputs}")
        
        # 获取预测结果
        predictions = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(outputs, dim=1)
        print(f"预测概率:\n{predictions}")
        print(f"预测类别: {predicted_classes}")
    
    # 模拟训练步骤
    print(f"\n=== 模拟训练步骤 ===")
    
    # 模拟标签
    labels = torch.tensor([1, 0, 1, 0])  # 真实标签
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模式
    model.train()
    
    # 前向传播
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, labels)
    
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播（模拟）
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("训练步骤完成!")

# 可视化注意力权重（可选）
def visualize_attention():
    """可视化注意力权重"""
    print(f"\n=== 注意力机制理解 ===")
    
    # 模拟一个简单的自注意力计算
    seq_len = 3
    d_model = 4
    
    # 模拟3个token的嵌入向量
    tokens = ["我", "爱", "学习"]
    embeddings = torch.randn(seq_len, d_model)
    
    print(f"Token序列: {tokens}")
    print(f"嵌入向量形状: {embeddings.shape}")  # (3, 4)
    
    # 简化的自注意力计算
    W_q = torch.randn(d_model, d_model)  # 查询权重
    W_k = torch.randn(d_model, d_model)  # 键权重
    W_v = torch.randn(d_model, d_model)  # 值权重
    
    # 计算Q, K, V
    Q = embeddings @ W_q  # (3, 4)
    K = embeddings @ W_k  # (3, 4) 
    V = embeddings @ W_v  # (3, 4)
    
    # 计算注意力分数
    attention_scores = Q @ K.T  # (3, 3)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    
    print(f"\n注意力权重矩阵 (3x3):")
    print(attention_weights)
    print(f"\n解释:")
    print(f"- 第0行: '我' 关注 ['我', '爱', '学习'] 的程度")
    print(f"- 第1行: '爱' 关注 ['我', '爱', '学习'] 的程度")  
    print(f"- 第2行: '学习' 关注 ['我', '爱', '学习'] 的程度")

if __name__ == "__main__":
    demonstrate_transformer()
    visualize_attention()