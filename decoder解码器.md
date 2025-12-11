# Transformer Decoder（解码器）架构与原理详解

Transformer的解码器（Decoder）是Transformer架构的另一个核心组件，负责基于编码器的输出生成目标序列。解码器与编码器的主要区别在于它包含**三个子层**，其中最关键的是**掩码自注意力**和**编码器-解码器交叉注意力**。本文将深入讲解解码器的架构、原理以及每个组件的详细作用。

## 一、Decoder整体架构

### 1.1 架构概览

解码器由 $N$ 个**相同的解码器层**（Decoder Layer）堆叠而成，每层包含**三个主要子层**：

```
目标序列 [B, T]
    ↓
[词嵌入 + 位置编码] → [B, T, d_model]
    ↓
┌─────────────────────────────────────┐
│  Decoder Layer 1                     │
│  ┌───────────────────────────────┐  │
│  │ 1. 掩码多头自注意力            │  │
│  │    (Masked Self-Attn)         │  │
│  │    + 残差连接 + 层归一化      │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ 2. 编码器-解码器交叉注意力     │  │
│  │    (Cross-Attention)          │  │
│  │    + 残差连接 + 层归一化      │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ 3. 前馈网络 (FFN)             │  │
│  │    + 残差连接 + 层归一化      │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Decoder Layer 2                     │
│  ... (与Layer 1结构相同)            │
└─────────────────────────────────────┘
    ↓
    ...
    ↓
┌─────────────────────────────────────┐
│  Decoder Layer N                     │
│  ... (与Layer 1结构相同)            │
└─────────────────────────────────────┘
    ↓
解码输出 [B, T, d_model]
    ↓
[线性层] → [B, T, vocab_size]
    ↓
[Softmax] → 预测概率分布
```

### 1.2 数学表示

对于目标序列 $Y \in \mathbb{R}^{B \times T \times d_{model}}$ 和编码器输出 $M \in \mathbb{R}^{B \times S \times d_{model}}$，解码器的计算过程：

1. **输入嵌入和位置编码**：
   $$Y' = \text{Embedding}(Y) + \text{PositionalEncoding}(Y)$$

2. **逐层处理**（对于 $i = 1, 2, ..., N$）：
   $$Y' = \text{LayerNorm}(Y' + \text{MaskedMultiHeadSelfAttention}(Y'))$$
   $$Y' = \text{LayerNorm}(Y' + \text{CrossAttention}(Y', M, M))$$
   $$Y' = \text{LayerNorm}(Y' + \text{FFN}(Y'))$$

3. **输出投影**：
   $$\text{Logits} = Y'W_O$$
   $$\text{Probabilities} = \text{softmax}(\text{Logits})$$

### 1.3 Decoder vs Encoder的关键区别

| 特性 | Encoder | Decoder |
|------|---------|---------|
| **子层数量** | 2个 | 3个 |
| **自注意力** | 可以看到所有位置 | 只能看到当前位置及之前（掩码） |
| **交叉注意力** | 无 | 有（融合编码器输出） |
| **用途** | 理解输入序列 | 生成输出序列 |
| **训练方式** | 并行处理所有位置 | 训练时并行，推理时自回归 |

## 二、解码器层的详细结构

### 2.1 单层解码器的组成

每个解码器层包含以下组件：

```
输入 tgt [B, T, d_model]
编码器输出 memory [B, S, d_model]
    ↓
┌─────────────────────────────────────┐
│ 子层1: 掩码多头自注意力              │
│  1.1 掩码多头自注意力计算            │
│  1.2 Dropout                        │
│  1.3 残差连接: tgt + Dropout(attn)  │
│  1.4 层归一化                        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 子层2: 编码器-解码器交叉注意力       │
│  2.1 Q来自解码器，K和V来自编码器    │
│  2.2 交叉注意力计算                 │
│  2.3 Dropout                        │
│  2.4 残差连接: tgt + Dropout(attn)  │
│  2.5 层归一化                        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 子层3: 前馈神经网络                  │
│  3.1 线性层1: 升维 (d_model → 4d)  │
│  3.2 激活函数 (ReLU/GELU)           │
│  3.3 Dropout                        │
│  3.4 线性层2: 降维 (4d → d_model)  │
│  3.5 残差连接: tgt + Dropout(ffn)   │
│  3.6 层归一化                        │
└─────────────────────────────────────┘
    ↓
输出 tgt [B, T, d_model]
```

### 2.2 关键设计原则

1. **因果掩码**：自注意力使用掩码，防止看到未来信息
2. **交叉注意力**：融合编码器的输出，建立源序列和目标序列的关联
3. **残差连接**：每个子层都有残差连接
4. **层归一化**：每个子层后都进行归一化
5. **维度一致**：输入输出维度保持一致

## 三、组件详细解析

### 3.1 掩码多头自注意力（Masked Multi-Head Self-Attention）

#### 作用

**掩码自注意力**让解码器中的每个位置只能关注到**当前位置及之前的位置**，确保在生成时不会"看到未来"的信息，符合自回归生成的要求。

#### 为什么需要掩码？

在训练时，解码器会并行处理整个目标序列。如果不使用掩码，位置 $i$ 的token会看到位置 $i+1, i+2, ...$ 的信息，这会导致：

1. **训练-推理不一致**：训练时能看到未来，推理时看不到
2. **信息泄露**：模型会"作弊"，直接复制未来的token
3. **无法自回归生成**：推理时无法逐步生成序列

#### 掩码的实现

```python
def generate_square_subsequent_mask(sz):
    """
    生成因果掩码（Causal Mask）
    形状: (sz, sz)
    上三角为 -inf，下三角（包括对角线）为 0
    """
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask

# 示例：序列长度为4
# mask = [
#     [0,    -inf, -inf, -inf],  # 位置0只能看到自己
#     [0,    0,    -inf, -inf],  # 位置1能看到0和1
#     [0,    0,    0,    -inf],  # 位置2能看到0,1,2
#     [0,    0,    0,    0   ]   # 位置3能看到所有
# ]
```

#### 工作原理

```python
# 输入: tgt [B, T, d_model]
# 输出: attn_output [B, T, d_model]

# 1. 生成Q, K, V（都来自tgt）
Q = tgt @ W_Q  # [B, T, d_model]
K = tgt @ W_K  # [B, T, d_model]
V = tgt @ W_V  # [B, T, d_model]

# 2. 计算注意力分数
scores = Q @ K.T / sqrt(d_k)  # [B, T, T]

# 3. 应用因果掩码（关键步骤！）
scores = scores + causal_mask  # 上三角变为 -inf

# 4. Softmax归一化（-inf经过softmax变为0）
attn_weights = softmax(scores)  # [B, T, T]

# 5. 加权求和
attn_output = attn_weights @ V  # [B, T, d_model]
```

#### 数学公式

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M$ 是掩码矩阵：
$$M_{ij} = \begin{cases}
0 & \text{if } i \geq j \text{ (可以看)} \\
-\infty & \text{if } i < j \text{ (不能看)}
\end{cases}$$

#### 可视化示例

假设序列长度为4，注意力权重矩阵：

```
无掩码（错误）：
     位置0  位置1  位置2  位置3
位置0  0.3   0.2   0.3   0.2   ← 能看到未来！
位置1  0.2   0.3   0.2   0.3   ← 能看到未来！
位置2  0.3   0.2   0.3   0.2   ← 能看到未来！
位置3  0.2   0.3   0.2   0.3

有掩码（正确）：
     位置0  位置1  位置2  位置3
位置0  1.0   0.0   0.0   0.0   ← 只能看到自己
位置1  0.4   0.6   0.0   0.0   ← 只能看到0和1
位置2  0.2   0.3   0.5   0.0   ← 只能看到0,1,2
位置3  0.1   0.2   0.3   0.4   ← 能看到所有（包括自己）
```

### 3.2 编码器-解码器交叉注意力（Cross-Attention）

#### 作用

**交叉注意力**是解码器独有的组件，它让解码器的每个位置能够关注到**编码器输出的所有位置**，从而建立源序列和目标序列之间的关联。

#### 工作原理

```python
# Query来自解码器，Key和Value来自编码器
Q = tgt @ W_Q  # [B, T, d_model] - 来自解码器
K = memory @ W_K  # [B, S, d_model] - 来自编码器
V = memory @ W_V  # [B, S, d_model] - 来自编码器

# 计算注意力分数
scores = Q @ K.T / sqrt(d_k)  # [B, T, S]

# 应用源序列的padding mask（如果有）
if memory_mask is not None:
    scores = scores.masked_fill(memory_mask == 0, float('-inf'))

# Softmax归一化
attn_weights = softmax(scores)  # [B, T, S]

# 加权求和
attn_output = attn_weights @ V  # [B, T, d_model]
```

#### 数学公式

$$\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$ 来自解码器：$Q = Y'W_Q$
- $K, V$ 来自编码器：$K = MW_K, V = MW_V$

#### 关键特性

1. **信息融合**：将编码器的语义信息融合到解码器
2. **对齐机制**：注意力权重可以显示源序列和目标序列的对齐关系
3. **全局视野**：解码器的每个位置都能看到源序列的所有位置
4. **双向关联**：建立源序列和目标序列之间的双向关联

#### 为什么Q来自解码器，K和V来自编码器？

- **Query（查询）**：解码器位置在"查询"源序列中哪些位置与自己相关
- **Key和Value（键值）**：编码器提供源序列的信息，供解码器查询和使用

这种设计符合机器翻译等任务的需求：目标语言的每个词需要关注源语言中的哪些词。

#### 可视化示例

假设源序列（编码器输出）长度为3，目标序列（解码器）长度为4：

```
交叉注意力权重矩阵 [T=4, S=3]：
     源位置0  源位置1  源位置2
目标0   0.1     0.8     0.1   ← "Hello"关注源序列的中间位置
目标1   0.3     0.5     0.2   ← "world"关注源序列的前两个位置
目标2   0.2     0.2     0.6   ← "!"关注源序列的最后一个位置
目标3   0.4     0.3     0.3   ← "<EOS>"关注源序列的整体
```

### 3.3 前馈神经网络（Feed-Forward Network, FFN）

#### 作用

与编码器中的FFN相同，对每个位置独立应用两层全连接网络，进行非线性变换。

#### 结构

```
输入 tgt [B, T, d_model]
    ↓
线性层1: 升维
    ↓
激活函数 (ReLU/GELU)
    ↓
Dropout
    ↓
线性层2: 降维
    ↓
输出 [B, T, d_model]
```

#### 数学公式

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

维度变化：$d_{model} \to 4d_{model} \to d_{model}$

详见：[Encoder中的FFN详解](./encoder编码器.md#34-前馈神经网络feed-forward-network-ffn)

### 3.4 残差连接和层归一化

#### 残差连接

每个子层都有残差连接：
- 子层1：$tgt = tgt + \text{MaskedSelfAttention}(tgt)$
- 子层2：$tgt = tgt + \text{CrossAttention}(tgt, memory)$
- 子层3：$tgt = tgt + \text{FFN}(tgt)$

#### 层归一化

每个子层后都进行层归一化：
- `norm1`：掩码自注意力后
- `norm2`：交叉注意力后
- `norm3`：前馈网络后

详见：[Encoder中的残差连接和层归一化](./encoder编码器.md#33-残差连接residual-connection)

### 3.5 Dropout

解码器中的Dropout使用位置：
1. **注意力Dropout**：在注意力权重上（MultiheadAttention内部）
2. **残差连接Dropout**：三个子层的输出上（`dropout1`, `dropout2`, `dropout3`）
3. **FFN内部Dropout**：前馈网络的激活函数后

## 四、完整的前向传播流程

### 4.1 单层解码器的前向传播

```python
def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
    """
    tgt: [B, T, d_model] - 目标序列（解码器输入）
    memory: [B, S, d_model] - 编码器输出
    tgt_mask: [T, T] or [B, T, T] or None - 因果掩码
    memory_mask: [B, S] or None - 源序列padding mask
    return: [B, T, d_model] - 输出序列
    """
    
    # ========== 子层1: 掩码多头自注意力 ==========
    # Step 1.1: 计算掩码自注意力
    attn_output, _ = self.self_attn(
        tgt, tgt, tgt,  # Q, K, V都来自tgt
        attn_mask=tgt_mask  # 因果掩码，防止看到未来
    )  # [B, T, d_model]
    
    # Step 1.2: Dropout
    attn_output = self.dropout1(attn_output)  # [B, T, d_model]
    
    # Step 1.3: 残差连接
    tgt = tgt + attn_output  # [B, T, d_model]
    
    # Step 1.4: 层归一化
    tgt = self.norm1(tgt)  # [B, T, d_model]
    
    # ========== 子层2: 编码器-解码器交叉注意力 ==========
    # Step 2.1: 计算交叉注意力
    # Q来自解码器，K和V来自编码器
    attn_output, _ = self.cross_attn(
        tgt,  # Query: 来自解码器
        memory, memory,  # Key, Value: 来自编码器
        key_padding_mask=memory_mask  # 屏蔽源序列的padding
    )  # [B, T, d_model]
    
    # Step 2.2: Dropout
    attn_output = self.dropout2(attn_output)  # [B, T, d_model]
    
    # Step 2.3: 残差连接
    tgt = tgt + attn_output  # [B, T, d_model]
    
    # Step 2.4: 层归一化
    tgt = self.norm2(tgt)  # [B, T, d_model]
    
    # ========== 子层3: 前馈网络 ==========
    # Step 3.1: 第一层线性变换（升维）
    ff_output = self.linear1(tgt)  # [B, T, d_model] → [B, T, 4*d_model]
    
    # Step 3.2: 激活函数
    ff_output = self.activation(ff_output)  # [B, T, 4*d_model]
    
    # Step 3.3: Dropout
    ff_output = self.dropout(ff_output)  # [B, T, 4*d_model]
    
    # Step 3.4: 第二层线性变换（降维）
    ff_output = self.linear2(ff_output)  # [B, T, 4*d_model] → [B, T, d_model]
    
    # Step 3.5: Dropout
    ff_output = self.dropout3(ff_output)  # [B, T, d_model]
    
    # Step 3.6: 残差连接
    tgt = tgt + ff_output  # [B, T, d_model]
    
    # Step 3.7: 层归一化
    tgt = self.norm3(tgt)  # [B, T, d_model]
    
    return tgt  # [B, T, d_model]
```

### 4.2 多层解码器的堆叠

```python
# 编码器处理源序列
enc_out = encoder(src, src_mask)  # [B, S, d_model]

# 目标序列嵌入和位置编码
tgt_emb = embedding(tgt) + positional_encoding(tgt)  # [B, T, d_model]

# 逐层处理
dec_out = tgt_emb
for layer in decoder_layers:
    dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)  # [B, T, d_model]

# 输出投影
logits = output_projection(dec_out)  # [B, T, vocab_size]
probabilities = softmax(logits)  # [B, T, vocab_size]
```

## 五、各组件的协同作用

### 5.1 掩码自注意力 + 交叉注意力

- **掩码自注意力**：捕捉目标序列内部的依赖关系（因果性）
- **交叉注意力**：建立源序列和目标序列的关联
- **协同效果**：既能理解目标序列的结构，又能利用源序列的信息

### 5.2 三个子层的顺序

为什么是这个顺序？

1. **掩码自注意力**：先理解目标序列自身的结构
2. **交叉注意力**：然后融合源序列的信息
3. **前馈网络**：最后增强每个位置的表示

这个顺序符合信息处理的逻辑：先理解自己，再融合外部信息，最后增强表示。

### 5.3 多层堆叠的效果

- **浅层**：捕捉局部对齐和简单模式
- **深层**：捕捉复杂对齐和抽象语义关系
- **逐层抽象**：从词级对齐 → 短语级对齐 → 语义级对齐

## 六、训练与推理的区别

### 6.1 训练时（Teacher Forcing）

```python
# 训练时：并行处理整个目标序列
tgt = [<BOS>, "Hello", "world", "!"]  # 完整序列
tgt_mask = generate_causal_mask(len(tgt))  # 因果掩码

# 并行计算所有位置的输出
outputs = decoder(tgt, enc_out, tgt_mask)  # [B, T, d_model]

# 计算损失（与真实标签对比）
loss = criterion(outputs, target_labels)
```

**特点**：
- 并行处理，训练速度快
- 使用Teacher Forcing（使用真实标签作为输入）
- 需要因果掩码防止信息泄露

### 6.2 推理时（Autoregressive）

```python
# 推理时：自回归生成，逐步生成每个token
generated = [<BOS>]

for step in range(max_len):
    # 当前已生成的序列
    current_seq = torch.tensor([generated])  # [1, current_len]
    
    # 生成因果掩码
    tgt_mask = generate_causal_mask(len(current_seq))
    
    # 解码
    output = decoder(current_seq, enc_out, tgt_mask)
    
    # 预测下一个token
    next_token_logits = output[0, -1, :]  # 最后一个位置的logits
    next_token = argmax(softmax(next_token_logits))
    
    # 添加到生成序列
    generated.append(next_token)
    
    # 如果生成了结束符，停止
    if next_token == <EOS>:
        break
```

**特点**：
- 逐步生成，每次生成一个token
- 使用之前生成的token作为输入
- 自然满足因果性，无需额外掩码（但代码中仍需要）

## 七、完整代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    包含三个子层：
    1. 掩码多头自注意力机制 + 残差连接 + 层归一化
    2. 编码器-解码器交叉注意力 + 残差连接 + 层归一化
    3. 前馈神经网络 + 残差连接 + 层归一化
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # ========== 子层1: 掩码多头自注意力 ==========
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # ========== 子层2: 编码器-解码器交叉注意力 ==========
        self.cross_attn = nn.MultiheadAttention(
            d_model, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # ========== 子层3: 前馈网络 ==========
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.activation = F.relu
    
    def forward(
        self, 
        tgt: Tensor, 
        memory: Tensor, 
        tgt_mask: Tensor = None, 
        memory_mask: Tensor = None
    ) -> Tensor:
        """
        前向传播
        
        Args:
            tgt: [B, T, d_model] - 目标序列（解码器输入）
            memory: [B, S, d_model] - 编码器输出
            tgt_mask: [T, T] or [B, T, T] or None - 因果掩码
            memory_mask: [B, S] or None - 源序列padding mask
        
        Returns:
            [B, T, d_model] - 输出序列
        """
        # ========== 子层1: 掩码多头自注意力 ==========
        attn_output, _ = self.self_attn(
            tgt, tgt, tgt,  # Q, K, V都来自tgt
            attn_mask=tgt_mask  # 因果掩码
        )
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)
        
        # ========== 子层2: 编码器-解码器交叉注意力 ==========
        attn_output, _ = self.cross_attn(
            tgt,  # Query: 来自解码器
            memory, memory,  # Key, Value: 来自编码器
            key_padding_mask=memory_mask  # 源序列padding mask
        )
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)
        
        # ========== 子层3: 前馈网络 ==========
        ff_output = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(tgt)
                )
            )
        )
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerDecoder(nn.Module):
    """
    完整的Transformer解码器
    
    由N个相同的解码器层堆叠而成
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
        
        # 解码器层堆叠
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.out_proj = nn.Linear(d_model, vocab_size)
    
    def forward(
        self, 
        tgt: Tensor, 
        memory: Tensor, 
        tgt_mask: Tensor = None, 
        memory_mask: Tensor = None
    ) -> Tensor:
        """
        前向传播
        
        Args:
            tgt: [B, T] - 目标序列（token索引）
            memory: [B, S, d_model] - 编码器输出
            tgt_mask: [T, T] or [B, T, T] or None - 因果掩码
            memory_mask: [B, S] or None - 源序列padding mask
        
        Returns:
            [B, T, vocab_size] - 输出logits
        """
        batch_size, tgt_len = tgt.size()
        device = tgt.device
        
        # 位置编码
        pos = torch.arange(0, tgt_len, device=device).unsqueeze(0).expand(batch_size, tgt_len)
        
        # 词嵌入 + 位置编码
        tgt_emb = self.embedding(tgt) + self.pos_embedding(pos)  # [B, T, d_model]
        
        # 逐层处理
        dec_out = tgt_emb
        for layer in self.layers:
            dec_out = layer(dec_out, memory, tgt_mask, memory_mask)  # [B, T, d_model]
        
        # 输出投影
        logits = self.out_proj(dec_out)  # [B, T, vocab_size]
        
        return logits


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    生成因果掩码（Causal Mask）
    
    用于防止解码器看到未来的信息
    
    Args:
        sz: 序列长度
    
    Returns:
        [sz, sz] - 上三角为 -inf，下三角（包括对角线）为 0
    """
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask
```

## 八、关键设计要点总结

### 8.1 架构设计

| 设计要素 | 作用 | 重要性 |
|---------|------|--------|
| **掩码自注意力** | 捕捉目标序列依赖，防止信息泄露 | ⭐⭐⭐⭐⭐ |
| **交叉注意力** | 融合编码器信息，建立源-目标关联 | ⭐⭐⭐⭐⭐ |
| **前馈网络** | 非线性变换，增强表示 | ⭐⭐⭐⭐ |
| **残差连接** | 梯度传播，信息保留 | ⭐⭐⭐⭐⭐ |
| **层归一化** | 稳定训练，加速收敛 | ⭐⭐⭐⭐⭐ |
| **因果掩码** | 确保自回归生成 | ⭐⭐⭐⭐⭐ |

### 8.2 数据流

```
源序列 [B, S] → 编码器 → [B, S, d_model] (memory)
    ↓
目标序列 [B, T] → 嵌入+位置编码 → [B, T, d_model]
    ↓
[解码器层1] → [解码器层2] → ... → [解码器层N]
    ↓
输出 [B, T, d_model] → 线性层 → [B, T, vocab_size]
    ↓
Softmax → 概率分布
```

### 8.3 维度一致性

- **解码器输入输出**：`[B, T, d_model]`
- **编码器输出**：`[B, S, d_model]`（作为memory）
- **最终输出**：`[B, T, vocab_size]`

### 8.4 掩码的使用

| 掩码类型 | 形状 | 作用 | 使用位置 |
|---------|------|------|---------|
| **因果掩码** | `[T, T]` | 防止看到未来 | 掩码自注意力 |
| **Padding掩码** | `[B, S]` | 屏蔽源序列padding | 交叉注意力 |
| **Padding掩码** | `[B, T]` | 屏蔽目标序列padding | 掩码自注意力 |

## 九、实际应用

### 9.1 机器翻译

- **编码器**：理解源语言句子
- **解码器**：基于编码器输出生成目标语言句子
- **交叉注意力**：建立源语言和目标语言的词对齐关系

### 9.2 文本摘要

- **编码器**：理解原文
- **解码器**：生成摘要
- **交叉注意力**：确定原文中哪些部分对摘要重要

### 9.3 对话系统

- **编码器**：理解用户输入
- **解码器**：生成回复
- **交叉注意力**：关注用户输入的关键信息

## 十、Decoder vs Encoder总结

### 10.1 结构对比

| 特性 | Encoder | Decoder |
|------|---------|---------|
| **子层数** | 2个 | 3个 |
| **自注意力** | 双向（看所有位置） | 单向（只看过去） |
| **交叉注意力** | 无 | 有 |
| **掩码** | Padding mask | Causal mask + Padding mask |
| **用途** | 理解输入 | 生成输出 |

### 10.2 功能对比

| 功能 | Encoder | Decoder |
|------|---------|---------|
| **理解能力** | ✅ 理解输入序列的语义 | ✅ 理解目标序列的结构 |
| **生成能力** | ❌ 不生成 | ✅ 生成输出序列 |
| **对齐能力** | ❌ 无 | ✅ 建立源-目标对齐 |
| **因果性** | ❌ 不需要 | ✅ 必须保证 |

## 总结

Transformer解码器通过以下设计实现了强大的序列生成能力：

1. **掩码自注意力**：保证因果性，防止信息泄露
2. **交叉注意力**：融合编码器信息，建立源-目标关联
3. **前馈网络**：增强每个位置的表示能力
4. **残差连接**：保证梯度传播和信息保留
5. **层归一化**：稳定训练过程
6. **多层堆叠**：逐层抽象，从局部到全局

这些组件的协同作用使得解码器能够基于编码器的输出，自回归地生成高质量的目标序列，广泛应用于机器翻译、文本摘要、对话系统等序列到序列任务。
