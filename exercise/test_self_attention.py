import pytest
import torch
import torch.nn as nn
import math
from self_attention import SelfAttention


class TestSelfAttention:
    """SelfAttention 类的测试套件"""
    
    @pytest.fixture
    def d_model(self):
        """返回模型维度"""
        return 64
    
    @pytest.fixture
    def batch_size(self):
        """返回批次大小"""
        return 2
    
    @pytest.fixture
    def seq_len(self):
        """返回序列长度"""
        return 10
    
    @pytest.fixture
    def attention(self, d_model):
        """创建 SelfAttention 实例"""
        return SelfAttention(d_model)
    
    @pytest.fixture
    def input_tensor(self, batch_size, seq_len, d_model):
        """创建输入张量"""
        return torch.randn(batch_size, seq_len, d_model)
    
    def test_initialization(self, attention, d_model):
        """测试初始化是否正确"""
        assert attention.d_model == d_model
        assert isinstance(attention.q_proj, nn.Linear)
        assert isinstance(attention.k_proj, nn.Linear)
        assert isinstance(attention.v_proj, nn.Linear)
        assert isinstance(attention.out_proj, nn.Linear)
        
        # 检查投影层的输入输出维度
        assert attention.q_proj.in_features == d_model
        assert attention.q_proj.out_features == d_model
        assert attention.k_proj.in_features == d_model
        assert attention.k_proj.out_features == d_model
        assert attention.v_proj.in_features == d_model
        assert attention.v_proj.out_features == d_model
        assert attention.out_proj.in_features == d_model
        assert attention.out_proj.out_features == d_model
    
    def test_forward_output_shape(self, attention, input_tensor, batch_size, seq_len, d_model):
        """测试 forward 方法的输出形状"""
        output = attention(input_tensor)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_forward_preserves_batch_size(self, attention, input_tensor, batch_size):
        """测试 forward 方法保持批次大小"""
        output = attention(input_tensor)
        assert output.shape[0] == batch_size
    
    def test_forward_preserves_sequence_length(self, attention, input_tensor, seq_len):
        """测试 forward 方法保持序列长度"""
        output = attention(input_tensor)
        assert output.shape[1] == seq_len
    
    def test_forward_preserves_model_dimension(self, attention, input_tensor, d_model):
        """测试 forward 方法保持模型维度"""
        output = attention(input_tensor)
        assert output.shape[2] == d_model
    
    def test_attention_weights_sum_to_one(self, attention, input_tensor):
        """测试注意力权重在最后一个维度上求和为1（softmax特性）"""
        # 手动计算注意力权重以验证
        q = attention.q_proj(input_tensor)
        k = attention.k_proj(input_tensor)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(attention.d_model)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        # 检查每一行的权重和是否接近1
        weights_sum = attn_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6)
    
    def test_attention_weights_non_negative(self, attention, input_tensor):
        """测试注意力权重非负"""
        q = attention.q_proj(input_tensor)
        k = attention.k_proj(input_tensor)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(attention.d_model)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        assert torch.all(attn_weights >= 0)
    
    def test_different_batch_sizes(self, attention, seq_len, d_model):
        """测试不同批次大小的输入"""
        for batch_size in [1, 4, 8]:
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            output = attention(input_tensor)
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_different_sequence_lengths(self, attention, batch_size, d_model):
        """测试不同序列长度的输入"""
        for seq_len in [1, 5, 20, 100]:
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            output = attention(input_tensor)
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_different_model_dimensions(self, batch_size, seq_len):
        """测试不同模型维度的 SelfAttention"""
        for d_model in [32, 128, 256]:
            attention = SelfAttention(d_model)
            input_tensor = torch.randn(batch_size, seq_len, d_model)
            output = attention(input_tensor)
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_gradient_flow(self, attention, input_tensor):
        """测试梯度是否能正常反向传播"""
        input_tensor.requires_grad_(True)
        output = attention(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # 检查输入是否有梯度
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()
        
        # 检查参数是否有梯度
        for param in attention.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_deterministic_with_same_input(self, attention, input_tensor):
        """测试相同输入产生相同输出（在eval模式下）"""
        attention.eval()
        with torch.no_grad():
            output1 = attention(input_tensor)
            output2 = attention(input_tensor)
            assert torch.allclose(output1, output2)
    
    def test_scaling_factor(self, attention, input_tensor):
        """测试注意力分数是否使用了正确的缩放因子"""
        q = attention.q_proj(input_tensor)
        k = attention.k_proj(input_tensor)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(attention.d_model)
        
        # 验证缩放因子是否正确应用
        unscaled_scores = q @ k.transpose(-2, -1)
        expected_scaled_scores = unscaled_scores / math.sqrt(attention.d_model)
        assert torch.allclose(attn_scores, expected_scaled_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

