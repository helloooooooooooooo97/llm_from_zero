import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

class SelfAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(SelfAttention, self).__init__()
        self.d_model: int = d_model
        # q_proj, k_proj, v_proj, out_proj: (d_model, d_model) 映射
        self.q_proj: nn.Linear = nn.Linear(d_model, d_model) 
        self.k_proj: nn.Linear = nn.Linear(d_model, d_model)
        self.v_proj: nn.Linear = nn.Linear(d_model, d_model)
        self.out_proj: nn.Linear = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, seq_len, d_model)
        q: Tensor = self.q_proj(x)    # (batch_size, seq_len, d_model)
        k: Tensor = self.k_proj(x)    # (batch_size, seq_len, d_model)
        v: Tensor = self.v_proj(x)    # (batch_size, seq_len, d_model)
        # q @ k^T 结果为 (batch_size, seq_len, seq_len)
        attn_scores: Tensor = q @ k.transpose(-2, -1) / math.sqrt(self.d_model) # (batch_size, seq_len, seq_len)
        attn_weights: Tensor = F.softmax(attn_scores, dim=-1)                   # (batch_size, seq_len, seq_len)
        # attn_weights @ v 结果为 (batch_size, seq_len, d_model)
        output: Tensor = attn_weights @ v                                       # (batch_size, seq_len, d_model)
        return self.out_proj(output)                                            # (batch_size, seq_len, d_model)

# 1. 为什么需要缩放因子呢？