import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)  # 输入输出: (B, S, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)    # (B, S, d_model) -> (B, S, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)    # (B, S, dim_feedforward) -> (B, S, d_model)
        self.norm1 = nn.LayerNorm(d_model)                    # (B, S, d_model)
        self.norm2 = nn.LayerNorm(d_model)                    # (B, S, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None):
        """
        src: (B, S, d_model)
        src_mask: (B, S) or None
        return: (B, S, d_model)
        """
        # Self-attention + Add & Norm
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=src_mask)  # attn_output: (B, S, d_model)
        src = src + self.dropout1(attn_output)                                     # (B, S, d_model)
        src = self.norm1(src)                                                      # (B, S, d_model)
        # Feed Forward + Add & Norm
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))  # (B, S, d_model)
        src = src + self.dropout2(ff_output)                                       # (B, S, d_model)
        src = self.norm2(src)                                                      # (B, S, d_model)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        解码器层初始化

        Args:
            d_model: 隐向量维度（模型维度）
            num_heads: 多头注意力中的head数量
            dim_feedforward: 前馈网络的隐层维度
            dropout: dropout概率
        """
        super().__init__()
        # 1. 掩码自注意力（Masked Self-Attention），batch_first=True 输入输出均为 (B, T, d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        # 2. 编码器-解码器交叉注意力（Cross-Attention）
        #    Q来自解码器,TGT，K/V来自编码器输出MEMORY
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        # 3. 前馈全连接网络（FFN），两层线性：d_model -> dim_feedforward -> d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)   # (B, T, d_model) -> (B, T, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)   # (B, T, dim_feedforward) -> (B, T, d_model)

        # 4. 三个层归一化（LayerNorm）：分别于三大子层后（见forward）
        self.norm1 = nn.LayerNorm(d_model)   # 掩码自注意力后
        self.norm2 = nn.LayerNorm(d_model)   # 交叉注意力后
        self.norm3 = nn.LayerNorm(d_model)   # 前馈网络后

        # 5. 三个Dropout，分别用在三个残差连接之后
        self.dropout1 = nn.Dropout(dropout)  # 掩码自注意力残差后
        self.dropout2 = nn.Dropout(dropout)  # 交叉注意力残差后
        self.dropout3 = nn.Dropout(dropout)  # 前馈网络残差后

        # 6. 激活函数（默认ReLU）
        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        解码器单层的前向传播

        Args:
            tgt: 解码器输入 (B, T, d_model)
            memory: 编码器输出 (B, S, d_model)
            tgt_mask: 目标序列的掩码（通常为因果掩码），shape可为 (T, T) or (B, T, T) 或 None
            memory_mask: 源序列的padding掩码，shape为 (B, S) 或 None

        Returns:
            更新后的解码器输出 (B, T, d_model)
        """

        # 1. 子层1：掩码多头自注意力（Mask掉未来位置信息，仅关注当前位置及过去）
        #    attn_mask专门用于掩盖未来，保证自回归机制，推理时一个token只能访问自己和左边
        attn_output, _ = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask # attn_mask: (B, T, T)代表哪些位置不能看到哪些位置
        ) # attn_output: (B, T, d_model)
        tgt = tgt + self.dropout1(attn_output)    # 残差连接
        tgt = self.norm1(tgt)                          # 层归一化 post-ln

        # 2. 子层2：编码器-解码器交叉多头注意力
        #    Q为当前目标序列，K/V为编码器输出，key_padding_mask用于mask掉src padding部分
        attn_output, _ = self.cross_attn(
            tgt, memory, memory, key_padding_mask=memory_mask # key_padding_mask: (B, S)代表哪些位置是padding
        )  # attn_output: (B, T, d_model)
        tgt = tgt + self.dropout2(attn_output)    # 残差连接 代表不阻塞主干道
        tgt = self.norm2(tgt)                     # 层归一化 post-ln

        # 3. 子层3：前馈全连接网络（FFN）：两层线性，中间激活
        ff_output = self.linear2(
            self.dropout(self.activation(self.linear1(tgt)))
        )                                         # (B, T, d_model)
        tgt = tgt + self.dropout3(ff_output)      # 残差连接 
        tgt = self.norm3(tgt)                     # 层归一化 post-ln
        return tgt

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)         # (B, L) -> (B, L, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)    # (B, L) -> (B, L, d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)             # (B, T, d_model) -> (B, T, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (B, S)  # 输入源序列，整数索引
        tgt: (B, T)  # 目标序列，整数索引
        src_mask: (B, S) or None
        tgt_mask: (T, T) or (B, T, T) or None
        return: logits (B, T, vocab_size)
        """
        batch_size, src_len = src.size()        # src: (B, S)
        _, tgt_len = tgt.size()                 # tgt: (B, T)
        device = src.device

        src_pos = torch.arange(0, src_len, device=device).unsqueeze(0).expand(batch_size, src_len)  # (B, S)
        tgt_pos = torch.arange(0, tgt_len, device=device).unsqueeze(0).expand(batch_size, tgt_len)  # (B, T)

        src_emb = self.embedding(src) + self.pos_embedding(src_pos)     # (B, S, d_model)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(tgt_pos)     # (B, T, d_model)

        enc_out = src_emb                                              # (B, S, d_model)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)                         # (B, S, d_model)

        dec_out = tgt_emb                                              # (B, T, d_model)
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)      # (B, T, d_model)

        logits = self.out_proj(dec_out)  # (B, T, vocab_size)
        return logits                   # (B, T, vocab_size)

def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. Mask out subsequent positions (for decoder).
    返回形状: (sz, sz)，上三角为 -inf，其余为0。用于防止解码器看到后续令牌。
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)