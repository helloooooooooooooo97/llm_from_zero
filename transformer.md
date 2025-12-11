# 什么是 Transformer？

Transformer 是一种基于**完全注意力机制**的神经网络架构，由 Vaswani 等人在 2017 年的论文 "Attention is All You Need" 中提出。它彻底改变了自然语言处理领域，成为现代大语言模型（如 GPT、BERT、LLaMA 等）的基础架构。

## Transformer 的核心特点

1. **完全基于注意力机制**：不使用循环或卷积结构
2. **并行计算**：所有位置可以同时处理，训练速度快
3. **长距离依赖**：能够直接捕捉序列中任意距离的关系
4. **可扩展性强**：易于扩展到大规模模型

## Transformer 的架构原理

Transformer 采用**编码器-解码器（Encoder-Decoder）**架构，主要由以下组件构成：

### 整体架构

```
输入序列 → [编码器] → [解码器] → 输出序列
```

### 编码器（Encoder）

编码器由 $N$ 个相同的层堆叠而成，每层包含两个子层：

1. **多头自注意力机制（Multi-Head Self-Attention）**
2. **前馈神经网络（Feed-Forward Network）**

每个子层都采用**残差连接（Residual Connection）**和**层归一化（Layer Normalization）**：

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

#### 编码器详细结构

```
输入嵌入 + 位置编码
    ↓
[多头自注意力]
    ↓ (残差连接 + 层归一化)
[前馈神经网络]
    ↓ (残差连接 + 层归一化)
输出
```

### 解码器（Decoder）

解码器也由 $N$ 个相同的层堆叠而成，每层包含三个子层：

1. **掩码多头自注意力机制（Masked Multi-Head Self-Attention）**
2. **编码器-解码器注意力机制（Encoder-Decoder Attention）**
3. **前馈神经网络（Feed-Forward Network）**

#### 解码器详细结构

```
输出嵌入 + 位置编码
    ↓
[掩码多头自注意力]  (只能看到当前位置之前的信息)
    ↓ (残差连接 + 层归一化)
[编码器-解码器注意力]  (Q来自解码器，K和V来自编码器)
    ↓ (残差连接 + 层归一化)
[前馈神经网络]
    ↓ (残差连接 + 层归一化)
输出
```

### 关键组件详解

#### 1. 输入嵌入（Input Embedding）

将输入序列中的每个词转换为 $d_{model}$ 维的向量表示。

#### 2. 位置编码（Positional Encoding）

由于 Transformer 没有循环或卷积结构，需要显式地添加位置信息。通常使用正弦位置编码：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

其中 $pos$ 是位置，$i$ 是维度索引。

#### 3. 多头自注意力机制

详见 [多头注意力机制](./多头注意力机制.md)。

#### 4. 掩码多头自注意力

在解码器中，为了防止模型看到未来的信息，使用掩码将未来位置的注意力分数设为 $-\infty$（经过 softmax 后变为 0）。

#### 5. 编码器-解码器注意力

- **Query（Q）**：来自解码器
- **Key（K）和 Value（V）**：来自编码器

这使得解码器能够关注输入序列的所有位置。

#### 6. 前馈神经网络（FFN）

每个位置独立应用一个两层全连接网络：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

通常使用 ReLU 激活函数，第一层将维度扩展到 $4 \times d_{model}$，第二层再映射回 $d_{model}$。

#### 7. 残差连接和层归一化

- **残差连接**：$x + \text{Sublayer}(x)$，有助于梯度传播和训练深层网络
- **层归一化**：对每个样本的特征维度进行归一化，稳定训练过程

### 完整的前向传播流程

#### 编码器流程

1. 输入序列 $X$ → 嵌入 + 位置编码 → $X'$
2. 对于每一层：
   - $X' = \text{LayerNorm}(X' + \text{MultiHead}(X', X', X'))$
   - $X' = \text{LayerNorm}(X' + \text{FFN}(X'))$
3. 输出编码表示

#### 解码器流程

1. 输出序列 $Y$ → 嵌入 + 位置编码 → $Y'$
2. 对于每一层：
   - $Y' = \text{LayerNorm}(Y' + \text{MaskedMultiHead}(Y', Y', Y'))$
   - $Y' = \text{LayerNorm}(Y' + \text{MultiHead}(Y', X', X'))$（$X'$ 来自编码器）
   - $Y' = \text{LayerNorm}(Y' + \text{FFN}(Y'))$
3. 输出 → 线性层 → softmax → 预测下一个词

## Transformer 的优势

| 优势 | 说明 |
|------|------|
| **并行计算** | 所有位置同时处理，训练速度快 |
| **长距离依赖** | 直接连接任意距离的位置 |
| **可解释性** | 注意力权重可视化 |
| **可扩展性** | 易于堆叠更多层，扩展到更大模型 |
| **通用性** | 适用于各种序列到序列任务 |

## Transformer 的应用

Transformer 架构被广泛应用于：

1. **机器翻译**：编码器-解码器架构的原始应用
2. **语言模型**：GPT 系列（仅解码器）、BERT（仅编码器）
3. **文本生成**：ChatGPT、Claude 等
4. **图像处理**：Vision Transformer (ViT)
5. **语音识别**：Whisper 等

## 总结

Transformer 通过**完全基于注意力机制的架构**，实现了：
- 高效的并行计算
- 强大的长距离依赖建模能力
- 优秀的可扩展性

这使得它成为现代深度学习，特别是大语言模型的基础架构。理解 Transformer 的原理，是理解 GPT、BERT 等现代模型的关键。

