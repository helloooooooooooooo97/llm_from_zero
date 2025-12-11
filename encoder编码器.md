# Transformer Encoder（编码器）架构与原理详解

Transformer的编码器（Encoder）是Transformer架构的核心组件之一，负责将输入序列转换为富含语义信息的表示。本文将深入讲解编码器的架构、原理以及每个组件的详细作用。

## 一、Encoder整体架构

### 1.1 架构概览

编码器由 $N$ 个**相同的编码器层**（Encoder Layer）堆叠而成，每层包含两个主要子层：

```
输入序列 [B, S]
    ↓
[词嵌入 + 位置编码] → [B, S, d_model]
    ↓
┌─────────────────────────────────────┐
│  Encoder Layer 1                     │
│  ┌───────────────────────────────┐  │
│  │ 1. 多头自注意力 (Self-Attn)   │  │
│  │    + 残差连接 + 层归一化      │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ 2. 前馈网络 (FFN)             │  │
│  │    + 残差连接 + 层归一化      │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Encoder Layer 2                     │
│  ... (与Layer 1结构相同)            │
└─────────────────────────────────────┘
    ↓
    ...
    ↓
┌─────────────────────────────────────┐
│  Encoder Layer N                     │
│  ... (与Layer 1结构相同)            │
└─────────────────────────────────────┘
    ↓
编码输出 [B, S, d_model]
```

### 1.2 数学表示

对于输入序列 $X \in \mathbb{R}^{B \times S \times d_{model}}$，编码器的计算过程：

1. **输入嵌入和位置编码**：
   $$X' = \text{Embedding}(X) + \text{PositionalEncoding}(X)$$

2. **逐层处理**（对于 $i = 1, 2, ..., N$）：
   $$X' = \text{LayerNorm}(X' + \text{MultiHeadSelfAttention}(X'))$$
   $$X' = \text{LayerNorm}(X' + \text{FFN}(X'))$$

3. **输出**：$X' \in \mathbb{R}^{B \times S \times d_{model}}$

## 二、编码器层的详细结构

### 2.1 单层编码器的组成

每个编码器层包含以下组件：

```
输入 x [B, S, d_model]
    ↓
┌─────────────────────────────────────┐
│ 子层1: 多头自注意力                  │
│  1.1 多头自注意力计算                │
│  1.2 Dropout                        │
│  1.3 残差连接: x + Dropout(attn)    │
│  1.4 层归一化                        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 子层2: 前馈神经网络                  │
│  2.1 线性层1: 升维 (d_model → 4d)   │
│  2.2 激活函数 (ReLU/GELU)           │
│  2.3 Dropout                        │
│  2.4 线性层2: 降维 (4d → d_model)  │
│  2.5 残差连接: x + Dropout(ffn)     │
│  2.6 层归一化                        │
└─────────────────────────────────────┘
    ↓
输出 x [B, S, d_model]
```

### 2.2 关键设计原则

1. **残差连接**：每个子层都有残差连接，帮助梯度传播
2. **层归一化**：每个子层后都进行归一化，稳定训练
3. **Dropout**：防止过拟合，提高泛化能力
4. **维度一致**：输入输出维度保持一致，便于堆叠

## 三、组件详细解析

### 3.1 多头自注意力机制（Multi-Head Self-Attention）

#### 作用

**自注意力机制**让序列中的每个位置都能关注到序列中**所有其他位置**的信息，从而捕捉长距离依赖关系。

#### 工作原理

```python
# 输入: x [B, S, d_model]
# 输出: attn_output [B, S, d_model]

# 1. 生成Q, K, V（所有都来自同一个输入x）
Q = x @ W_Q  # [B, S, d_model]
K = x @ W_K  # [B, S, d_model]
V = x @ W_V  # [B, S, d_model]

# 2. 计算注意力分数
scores = Q @ K.T / sqrt(d_k)  # [B, S, S]

# 3. 应用mask（如果有padding）
if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))

# 4. Softmax归一化
attn_weights = softmax(scores)  # [B, S, S]

# 5. 加权求和
attn_output = attn_weights @ V  # [B, S, d_model]
```

#### 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q, K, V$ 都来自输入 $X$
- $d_k$ 是每个头的维度（通常 $d_k = d_{model}/h$，$h$ 是头数）

#### 关键特性

1. **全局视野**：每个位置都能看到整个序列
2. **并行计算**：所有位置同时计算，训练速度快
3. **位置无关**：注意力机制本身不包含位置信息（需要位置编码）
4. **可解释性**：注意力权重可以可视化，显示模型关注的重点

#### 为什么需要"多头"？

- **单一注意力模式**：只能学习一种类型的注意力关系
- **多头注意力**：并行学习多种注意力模式（语法、语义、长距离等）

详见：[多头注意力机制](./多头注意力机制.md)

### 3.2 残差连接（Residual Connection）

#### 作用

残差连接将子层的输出**直接加回**输入，形成恒等映射路径。

#### 数学表示

$$\text{Output} = x + \text{Sublayer}(x)$$

#### 为什么重要？

1. **梯度传播**：
   - 提供直接的梯度路径：$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \text{Output}} \cdot (1 + \frac{\partial \text{Sublayer}}{\partial x})$
   - 即使子层的梯度很小，残差路径仍能传播梯度
   - 有助于训练深层网络

2. **信息保留**：
   - 保留原始输入信息
   - 子层只需要学习**增量变化**（residual），而不是完全变换

3. **训练稳定性**：
   - 初始阶段，子层输出接近0，残差连接保证输出接近输入
   - 训练过程更平滑

#### 代码实现

```python
# 残差连接
residual = x  # 保存原始输入
output = sublayer(x)  # 子层输出
output = residual + dropout(output)  # 残差连接 + Dropout
```

### 3.3 层归一化（Layer Normalization）

#### 作用

对每个样本的**特征维度**进行归一化，使得特征的均值为0、方差为1。

#### 数学公式

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$：特征维度的均值
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$：特征维度的方差
- $\gamma, \beta$：可学习的缩放和平移参数
- $\epsilon$：防止除零的小常数（通常 $10^{-5}$）

#### 为什么需要？

1. **稳定训练**：减少内部协变量偏移，稳定每层的输入分布
2. **加速收敛**：允许使用更大的学习率
3. **正则化效果**：具有一定的正则化作用，提高泛化能力

#### 在Encoder中的位置

编码器使用**Post-LN**（原始Transformer设计）：
- 归一化在残差连接**之后**
- `LayerNorm(x + Sublayer(x))`

详见：[层归一化](./层归一化.md)、[Pre-LN和Post-LN对比](./Pre-LN和Post-LN对比.md)

### 3.4 前馈神经网络（Feed-Forward Network, FFN）

#### 作用

对每个位置**独立**应用一个两层全连接网络，进行非线性变换，增强模型的表达能力。

#### 结构

```
输入 x [B, S, d_model]
    ↓
线性层1: 升维
    ↓
激活函数 (ReLU/GELU)
    ↓
Dropout
    ↓
线性层2: 降维
    ↓
输出 [B, S, d_model]
```

#### 数学公式

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

或者使用GELU激活函数：
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

其中：
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$，通常 $d_{ff} = 4 \times d_{model}$
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$

#### 维度变化

```python
# 输入: [B, S, d_model]
x = input  # [B, S, d_model]

# 第一层：升维
x = linear1(x)  # [B, S, d_model] → [B, S, 4*d_model]

# 激活
x = activation(x)  # [B, S, 4*d_model]

# Dropout
x = dropout(x)  # [B, S, 4*d_model]

# 第二层：降维
x = linear2(x)  # [B, S, 4*d_model] → [B, S, d_model]

# 输出: [B, S, d_model]
```

#### 为什么需要FFN？

1. **非线性变换**：注意力机制主要是线性变换，FFN提供非线性
2. **特征增强**：通过升维-激活-降维的过程，增强特征表示
3. **位置独立处理**：每个位置独立处理，不依赖其他位置

#### 为什么先升维再降维？

- **表达能力**：更大的中间维度提供更强的表达能力
- **参数量平衡**：虽然中间维度大，但参数量与直接映射相当
- **经验验证**：实验表明这种设计效果好

### 3.5 Dropout

#### 作用

在训练过程中随机将部分神经元输出设为0，防止过拟合。

#### 在Encoder中的使用位置

1. **注意力Dropout**：在注意力权重上应用
2. **残差连接Dropout**：在残差连接的子层输出上应用
3. **FFN内部Dropout**：在前馈网络的激活函数后应用

#### 代码示例

```python
# 注意力Dropout（在MultiheadAttention内部）
attn_output, _ = self.self_attn(..., dropout=dropout)

# 残差连接Dropout
attn_output = self.dropout1(attn_output)
src = src + attn_output

# FFN内部Dropout
ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
```

#### 为什么需要？

- **正则化**：减少过拟合
- **提高泛化**：让模型不过度依赖某些特征
- **训练稳定性**：增加训练的随机性，提高鲁棒性

## 四、完整的前向传播流程

### 4.1 单层编码器的前向传播

```python
def forward(self, src, src_mask=None):
    """
    src: [B, S, d_model] - 输入序列
    src_mask: [B, S] or None - padding mask
    return: [B, S, d_model] - 输出序列
    """
    
    # ========== 子层1: 多头自注意力 ==========
    # Step 1.1: 计算自注意力
    attn_output, _ = self.self_attn(
        src, src, src,  # Q, K, V都来自src
        key_padding_mask=src_mask  # 屏蔽padding位置
    )  # [B, S, d_model]
    
    # Step 1.2: Dropout
    attn_output = self.dropout1(attn_output)  # [B, S, d_model]
    
    # Step 1.3: 残差连接
    src = src + attn_output  # [B, S, d_model]
    
    # Step 1.4: 层归一化
    src = self.norm1(src)  # [B, S, d_model]
    
    # ========== 子层2: 前馈网络 ==========
    # Step 2.1: 第一层线性变换（升维）
    ff_output = self.linear1(src)  # [B, S, d_model] → [B, S, 4*d_model]
    
    # Step 2.2: 激活函数
    ff_output = self.activation(ff_output)  # [B, S, 4*d_model]
    
    # Step 2.3: Dropout
    ff_output = self.dropout(ff_output)  # [B, S, 4*d_model]
    
    # Step 2.4: 第二层线性变换（降维）
    ff_output = self.linear2(ff_output)  # [B, S, 4*d_model] → [B, S, d_model]
    
    # Step 2.5: Dropout
    ff_output = self.dropout2(ff_output)  # [B, S, d_model]
    
    # Step 2.6: 残差连接
    src = src + ff_output  # [B, S, d_model]
    
    # Step 2.7: 层归一化
    src = self.norm2(src)  # [B, S, d_model]
    
    return src  # [B, S, d_model]
```

### 4.2 多层编码器的堆叠

```python
# 输入嵌入和位置编码
src_emb = embedding(src) + positional_encoding(src)  # [B, S, d_model]

# 逐层处理
enc_out = src_emb
for layer in encoder_layers:
    enc_out = layer(enc_out, src_mask)  # [B, S, d_model]

# 输出
return enc_out  # [B, S, d_model]
```

## 五、各组件的协同作用

### 5.1 注意力机制 + 残差连接

- **注意力机制**：捕捉序列中的依赖关系
- **残差连接**：保留原始信息，让注意力学习增量变化
- **协同效果**：既能学习新特征，又不会丢失原始信息

### 5.2 残差连接 + 层归一化

- **残差连接**：提供梯度传播路径
- **层归一化**：稳定输入分布
- **协同效果**：稳定训练，加速收敛

### 5.3 注意力机制 + FFN

- **注意力机制**：捕捉位置间的关系（交互）
- **FFN**：增强每个位置的表示（独立处理）
- **协同效果**：既有关联性，又有独立性

### 5.4 多层堆叠的效果

- **浅层**：捕捉局部特征和简单模式
- **深层**：捕捉复杂特征和抽象模式
- **逐层抽象**：从词级 → 短语级 → 句子级 → 语义级

## 六、完整代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    包含两个子层：
    1. 多头自注意力机制 + 残差连接 + 层归一化
    2. 前馈神经网络 + 残差连接 + 层归一化
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # ========== 子层1: 多头自注意力 ==========
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # ========== 子层2: 前馈网络 ==========
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        前向传播
        
        Args:
            src: [B, S, d_model] - 输入序列
            src_mask: [B, S] or None - padding mask（1表示有效位置，0表示padding）
        
        Returns:
            [B, S, d_model] - 输出序列
        """
        # ========== 子层1: 多头自注意力 ==========
        # 计算自注意力（Q, K, V都来自src）
        attn_output, _ = self.self_attn(
            src, src, src,  # Query, Key, Value
            key_padding_mask=src_mask  # 屏蔽padding位置
        )
        
        # 残差连接 + Dropout
        src = src + self.dropout1(attn_output)
        
        # 层归一化
        src = self.norm1(src)

        # ========== 子层2: 前馈网络 ==========
        # 前馈网络：升维 → 激活 → Dropout → 降维
        ff_output = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(src)
                )
            )
        )
        
        # 残差连接 + Dropout
        src = src + self.dropout2(ff_output)
        
        # 层归一化
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """
    完整的Transformer编码器
    
    由N个相同的编码器层堆叠而成
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ) -> None:
        super().__init__()
        
        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        前向传播
        
        Args:
            src: [B, S] - 输入序列（token索引）
            src_mask: [B, S] or None - padding mask
        
        Returns:
            [B, S, d_model] - 编码后的序列表示
        """
        batch_size, seq_len = src.size()
        device = src.device
        
        # 位置编码
        pos = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        
        # 词嵌入 + 位置编码
        src_emb = self.embedding(src) + self.pos_embedding(pos)  # [B, S, d_model]
        
        # 逐层处理
        enc_out = src_emb
        for layer in self.layers:
            enc_out = layer(enc_out, src_mask)  # [B, S, d_model]
        
        return enc_out
```

## 七、关键设计要点总结

### 7.1 架构设计

| 设计要素 | 作用 | 重要性 |
|---------|------|--------|
| **多头自注意力** | 捕捉序列依赖关系 | ⭐⭐⭐⭐⭐ |
| **前馈网络** | 非线性变换，增强表示 | ⭐⭐⭐⭐ |
| **残差连接** | 梯度传播，信息保留 | ⭐⭐⭐⭐⭐ |
| **层归一化** | 稳定训练，加速收敛 | ⭐⭐⭐⭐⭐ |
| **Dropout** | 防止过拟合 | ⭐⭐⭐ |

### 7.2 数据流

```
输入 [B, S] (token索引)
    ↓
嵌入 + 位置编码
    ↓
[B, S, d_model]
    ↓
[编码器层1] → [编码器层2] → ... → [编码器层N]
    ↓
输出 [B, S, d_model] (富含语义的表示)
```

### 7.3 维度一致性

- **输入输出维度**：始终保持 `[B, S, d_model]`
- **便于堆叠**：多层可以无缝堆叠
- **残差连接**：要求维度一致才能相加

## 八、实际应用

### 8.1 BERT（仅编码器）

BERT使用Transformer编码器进行预训练：
- **Masked Language Model (MLM)**：预测被mask的词
- **Next Sentence Prediction (NSP)**：判断两个句子是否相邻

### 8.2 机器翻译（编码器-解码器）

编码器将源语言编码为语义表示，解码器基于此生成目标语言。

### 8.3 文本分类

使用编码器的输出（通常是`[CLS]` token或平均池化）进行分类。

## 总结

Transformer编码器通过以下设计实现了强大的序列建模能力：

1. **多头自注意力**：捕捉序列中任意位置间的依赖关系
2. **前馈网络**：增强每个位置的表示能力
3. **残差连接**：保证梯度传播和信息保留
4. **层归一化**：稳定训练过程
5. **多层堆叠**：逐层抽象，从局部到全局

这些组件的协同作用使得编码器能够将输入序列转换为富含语义信息的表示，为下游任务提供强大的特征。
