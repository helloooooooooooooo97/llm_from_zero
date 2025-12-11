"""
手动实现层归一化（Layer Normalization）

层归一化是Transformer中的关键组件，用于稳定训练过程。
本实现从零开始，不使用PyTorch的nn.LayerNorm。
"""
import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）的手动实现
    
    对每个样本的特征维度进行归一化，使得特征的均值为0、方差为1。
    
    公式：
        LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
    
    其中：
        - μ: 特征维度的均值
        - σ²: 特征维度的方差
        - γ: 可学习的缩放参数（初始化为1）
        - β: 可学习的平移参数（初始化为0）
        - ε: 防止除零的小常数（通常为1e-5）
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        """
        初始化层归一化
        
        Args:
            normalized_shape: 要归一化的特征维度大小
            eps: 防止除零的小常数，默认1e-5
        """
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # 可学习的参数：缩放（gamma）和平移（beta）
        # gamma初始化为1，beta初始化为0
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [..., normalized_shape]
               例如：[B, L, d] 或 [B, d] 等
        
        Returns:
            归一化后的张量，形状与输入相同
        """
        # 获取需要归一化的维度
        # 例如：如果x的形状是[B, L, d]，normalized_shape=d
        # 那么需要归一化的维度是最后一个维度
        
        # 计算均值：在最后一个维度上求均值
        # keepdim=True保持维度，便于后续广播
        mean = x.mean(dim=-1, keepdim=True)
        
        # 计算方差：在最后一个维度上求方差
        # 使用无偏估计：var = mean((x - mean)^2)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化：(x - mean) / sqrt(var + eps)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用可学习的缩放和平移：gamma * x_normalized + beta
        # gamma和beta会自动广播到正确的形状
        output = self.gamma * x_normalized + self.beta
        
        return output