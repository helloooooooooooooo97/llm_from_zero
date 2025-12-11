import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, Union
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model 
        self.num_heads = num_heads # 头数
        self.head_dim = d_model // num_heads # // 代表整数除法，得到整数结果

        self.q_proj = nn.Linear(d_model, d_model) # (d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Union[Tensor, None] = None) -> Tensor:
        # query, key, value: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = query.size()

        # 1. 线性投影获得Q, K, V
        q = self.q_proj(query)  # q: [B, L, d_model]
        k = self.k_proj(key)    # k: [B, L, d_model]
        v = self.v_proj(value)  # v: [B, L, d_model]

        # 2. 拆分为多头并转置为 [B, num_heads, L, head_dim]
        def reshape(x):
            # x: [B, L, d_model] → [B, L, num_heads, head_dim] → [B, num_heads, L, head_dim]
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous() # 张量连续性
        
        q = reshape(q) # [B, num_heads, L, head_dim]
        k = reshape(k) # [B, num_heads, L, head_dim]
        v = reshape(v) # [B, num_heads, L, head_dim]

        # 3. 注意力分数
        # q: [B, num_heads, L, head_dim]
        # k.transpose(-2, -1): [B, num_heads, head_dim, L]
        # attn_scores: [B, num_heads, L, L]
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) 
        if mask is not None:
            # mask: [B, 1, L, L] 或 [B, L, L]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, num_heads, L, L]
       
        # 4. 加权求和
        # attn_weights: [B, num_heads, L, L]
        # v: [B, num_heads, L, head_dim]
        # attn_output: [B, num_heads, L, head_dim]
        attn_output = attn_weights @ v  

        # 5. 合并多头
        # attn_output: [B, num_heads, L, head_dim] → [B, L, num_heads, head_dim] → [B, L, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 6. 输出投影
        # attn_output: [B, L, d_model]
        output = self.out_proj(attn_output) # [B, L, d_model]
        return output