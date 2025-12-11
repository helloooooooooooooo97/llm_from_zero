"""
Pre-LN和Post-LN的对比实现

展示两种层归一化位置的不同实现方式，并对比它们的训练特性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerEncoderLayerPostLN(nn.Module):
    """
    Post-LN Transformer编码器层（原始Transformer设计）
    
    归一化在残差连接之后：
    Output = LayerNorm(x + Sublayer(x))
    """
    
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
    
    def forward(self, src: Tensor) -> Tensor:
        # Post-LN: 先计算，再残差，最后归一化
        # 自注意力
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)  # Post-LN: 归一化在残差连接之后
        
        # 前馈网络
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)  # Post-LN: 归一化在残差连接之后
        return src


class TransformerEncoderLayerPreLN(nn.Module):
    """
    Pre-LN Transformer编码器层（现代变体）
    
    归一化在子层操作之前：
    Output = x + Sublayer(LayerNorm(x))
    """
    
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
    
    def forward(self, src: Tensor) -> Tensor:
        # Pre-LN: 先归一化，再计算，最后残差
        # 自注意力
        src_norm = self.norm1(src)  # Pre-LN: 归一化在子层之前
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm)
        src = src + self.dropout1(attn_output)
        
        # 前馈网络
        src_norm = self.norm2(src)  # Pre-LN: 归一化在子层之前
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_output)
        return src


def compare_gradient_flow():
    """对比两种架构的梯度流"""
    print("=" * 60)
    print("梯度流对比：Pre-LN vs Post-LN")
    print("=" * 60)
    
    d_model = 64
    batch_size = 2
    seq_len = 10
    
    # Post-LN
    x_post = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    post_ln_layer = TransformerEncoderLayerPostLN(d_model, num_heads=4)
    post_output = post_ln_layer(x_post)
    post_loss = post_output.sum()
    post_loss.backward()
    
    post_grad_norm = x_post.grad.norm().item() if x_post.grad is not None else 0.0
    print(f"\nPost-LN梯度范数: {post_grad_norm:.6f}")
    
    # Pre-LN
    x_pre = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    pre_ln_layer = TransformerEncoderLayerPreLN(d_model, num_heads=4)
    pre_output = pre_ln_layer(x_pre)
    pre_loss = pre_output.sum()
    pre_loss.backward()
    
    pre_grad_norm = x_pre.grad.norm().item() if x_pre.grad is not None else 0.0
    print(f"Pre-LN梯度范数: {pre_grad_norm:.6f}")
    
    if post_grad_norm > 0:
        print(f"\n梯度比 (Pre-LN / Post-LN): {pre_grad_norm / post_grad_norm:.4f}")
        print("(通常Pre-LN的梯度更大，训练更稳定)")
    else:
        print("\n注意: Post-LN的梯度可能很小，这体现了梯度流的问题")


def compare_output_variance():
    """对比两种架构的输出方差"""
    print("\n" + "=" * 60)
    print("输出方差对比：Pre-LN vs Post-LN")
    print("=" * 60)
    
    d_model = 64
    batch_size = 2
    seq_len = 10
    num_layers = 6
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Post-LN多层堆叠
    post_layers = nn.ModuleList([
        TransformerEncoderLayerPostLN(d_model, num_heads=4)
        for _ in range(num_layers)
    ])
    
    post_output = x.clone()
    for layer in post_layers:
        post_output = layer(post_output)
    
    post_variance = post_output.var().item()
    print(f"\nPost-LN输出方差: {post_variance:.6f}")
    
    # Pre-LN多层堆叠
    pre_layers = nn.ModuleList([
        TransformerEncoderLayerPreLN(d_model, num_heads=4)
        for _ in range(num_layers)
    ])
    
    pre_output = x.clone()
    for layer in pre_layers:
        pre_output = layer(pre_output)
    
    pre_variance = pre_output.var().item()
    print(f"Pre-LN输出方差: {pre_variance:.6f}")
    
    print(f"\n方差比 (Pre-LN / Post-LN): {pre_variance / post_variance:.4f}")
    print("(Post-LN的输出方差通常更稳定，因为每层都归一化)")


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("前向传播测试")
    print("=" * 60)
    
    d_model = 64
    batch_size = 2
    seq_len = 10
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Post-LN
    post_ln_layer = TransformerEncoderLayerPostLN(d_model, num_heads=4)
    post_output = post_ln_layer(x)
    print(f"\nPost-LN:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {post_output.shape}")
    print(f"  输出均值: {post_output.mean().item():.6f}")
    print(f"  输出方差: {post_output.var().item():.6f}")
    
    # Pre-LN
    pre_ln_layer = TransformerEncoderLayerPreLN(d_model, num_heads=4)
    pre_output = pre_ln_layer(x)
    print(f"\nPre-LN:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {pre_output.shape}")
    print(f"  输出均值: {pre_output.mean().item():.6f}")
    print(f"  输出方差: {pre_output.var().item():.6f}")


def demonstrate_difference():
    """演示两种架构的关键差异"""
    print("\n" + "=" * 60)
    print("架构差异演示")
    print("=" * 60)
    
    print("\n【Post-LN架构流程】")
    print("输入 x")
    print("  ↓")
    print("Sublayer(x)  # 子层操作（注意力/FFN）")
    print("  ↓")
    print("x + Sublayer(x)  # 残差连接")
    print("  ↓")
    print("LayerNorm(x + Sublayer(x))  # 归一化（在残差之后）")
    print("  ↓")
    print("输出")
    
    print("\n【Pre-LN架构流程】")
    print("输入 x")
    print("  ↓")
    print("LayerNorm(x)  # 归一化（在子层之前）")
    print("  ↓")
    print("Sublayer(LayerNorm(x))  # 子层操作（注意力/FFN）")
    print("  ↓")
    print("x + Sublayer(LayerNorm(x))  # 残差连接")
    print("  ↓")
    print("输出")
    
    print("\n【关键差异】")
    print("1. Post-LN: 归一化在残差连接之后")
    print("   - 优点: 更强的正则化，可能性能更好")
    print("   - 缺点: 训练不稳定，需要预热，梯度流较差")
    print("\n2. Pre-LN: 归一化在子层操作之前")
    print("   - 优点: 训练稳定，梯度流好，无需预热")
    print("   - 缺点: 正则化较弱，某些任务性能可能略差")


if __name__ == "__main__":
    # 运行所有对比测试
    test_forward_pass()
    compare_gradient_flow()
    compare_output_variance()
    demonstrate_difference()
    
    print("\n" + "=" * 60)
    print("对比完成！")
    print("=" * 60)
    print("\n建议:")
    print("- 大多数情况使用 Pre-LN（训练更稳定）")
    print("- 追求极致性能可以尝试 Post-LN（配合学习率预热）")
    print("- 深层网络（>24层）必须使用 Pre-LN")

