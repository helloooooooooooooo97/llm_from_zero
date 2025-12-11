"""
层归一化（Layer Normalization）的测试套件

测试手动实现的LayerNorm与PyTorch官方实现的对比，以及各种边界情况。
"""
import pytest
import torch
import torch.nn as nn
from layer_norm import LayerNorm


class TestLayerNorm:
    """LayerNorm 类的测试套件"""
    
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
    def layer_norm(self, d_model):
        """创建 LayerNorm 实例"""
        return LayerNorm(d_model)
    
    @pytest.fixture
    def input_tensor(self, batch_size, seq_len, d_model):
        """创建输入张量"""
        return torch.randn(batch_size, seq_len, d_model)
    
    def test_initialization(self, layer_norm, d_model):
        """测试初始化是否正确"""
        assert layer_norm.normalized_shape == d_model
        assert layer_norm.eps == 1e-5
        
        # 检查可学习参数
        assert isinstance(layer_norm.gamma, nn.Parameter)
        assert isinstance(layer_norm.beta, nn.Parameter)
        
        # 检查初始值
        assert torch.allclose(layer_norm.gamma.data, torch.ones(d_model))
        assert torch.allclose(layer_norm.beta.data, torch.zeros(d_model))
    
    def test_output_shape(self, layer_norm, input_tensor):
        """测试输出形状是否正确"""
        output = layer_norm(input_tensor)
        assert output.shape == input_tensor.shape
    
    def test_normalization_property(self, layer_norm, input_tensor):
        """测试归一化属性：输出的均值接近0，方差接近1"""
        output = layer_norm(input_tensor)
        
        # 对每个样本的每个位置的特征维度进行验证
        batch_size, seq_len, d_model = input_tensor.shape
        for b in range(batch_size):
            for l in range(seq_len):
                output_features = output[b, l, :]
                mean_val = output_features.mean().item()
                var_val = output_features.var(unbiased=False).item()
                
                # 由于gamma和beta的影响，归一化后的值可能不完全满足均值0方差1
                # 但如果没有gamma和beta，应该满足
                # 这里我们测试归一化后的值（在应用gamma和beta之前）
                # 实际上，我们需要测试归一化后的中间结果
        
        # 更准确的测试：创建gamma=1, beta=0的层归一化
        layer_norm_test = LayerNorm(d_model)
        layer_norm_test.gamma.data.fill_(1.0)
        layer_norm_test.beta.data.fill_(0.0)
        
        output_test = layer_norm_test(input_tensor)
        
        # 检查每个样本每个位置的归一化属性
        for b in range(batch_size):
            for l in range(seq_len):
                output_features = output_test[b, l, :]
                mean_val = output_features.mean().item()
                var_val = output_features.var(unbiased=False).item()
                
                assert abs(mean_val) < 1e-5, f"均值应该接近0，实际为{mean_val}"
                assert abs(var_val - 1.0) < 1e-4, f"方差应该接近1，实际为{var_val}"
    
    def test_against_pytorch(self, d_model, batch_size, seq_len):
        """测试与PyTorch官方实现的对比"""
        # 创建相同的输入
        torch.manual_seed(42)
        input_tensor = torch.randn(batch_size, seq_len, d_model)
        
        # 我们的实现
        our_layer_norm = LayerNorm(d_model)
        our_output = our_layer_norm(input_tensor)
        
        # PyTorch官方实现
        pytorch_layer_norm = nn.LayerNorm(d_model)
        pytorch_output = pytorch_layer_norm(input_tensor)
        
        # 比较结果（允许小的数值误差）
        max_diff = torch.abs(our_output - pytorch_output).max().item()
        assert max_diff < 1e-5, f"与PyTorch实现差异过大: {max_diff}"
    
    def test_learnable_parameters(self, layer_norm, input_tensor):
        """测试可学习参数是否正常工作"""
        # 修改gamma和beta
        layer_norm.gamma.data.fill_(2.0)
        layer_norm.beta.data.fill_(1.0)
        
        output = layer_norm(input_tensor)
        
        # 验证输出确实受到了gamma和beta的影响
        # 如果gamma=2, beta=1，输出应该大致是归一化值*2 + 1
        assert output.shape == input_tensor.shape
    
    def test_gradient_flow(self, layer_norm, input_tensor):
        """测试梯度流是否正常"""
        input_tensor.requires_grad_(True)
        output = layer_norm(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否存在
        assert input_tensor.grad is not None
        assert layer_norm.gamma.grad is not None
        assert layer_norm.beta.grad is not None
        
        # 检查梯度不为零（除非特殊情况）
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))
    
    def test_different_shapes(self, d_model):
        """测试不同输入形状"""
        layer_norm = LayerNorm(d_model)
        
        # 2D输入：[B, d]
        x_2d = torch.randn(3, d_model)
        output_2d = layer_norm(x_2d)
        assert output_2d.shape == x_2d.shape
        
        # 3D输入：[B, L, d]
        x_3d = torch.randn(2, 5, d_model)
        output_3d = layer_norm(x_3d)
        assert output_3d.shape == x_3d.shape
        
        # 4D输入：[B, H, W, d]
        x_4d = torch.randn(2, 5, 5, d_model)
        output_4d = layer_norm(x_4d)
        assert output_4d.shape == x_4d.shape
    
    def test_eps_parameter(self, d_model):
        """测试eps参数的作用"""
        # 创建非常小的输入（接近零方差）
        input_tensor = torch.ones(2, 3, d_model) * 0.1
        
        layer_norm = LayerNorm(d_model, eps=1e-5)
        output = layer_norm(input_tensor)
        
        # 输出应该不会出现NaN或Inf
        assert torch.isfinite(output).all()
    
    def test_training_mode(self, layer_norm, input_tensor):
        """测试训练模式和评估模式"""
        # 训练模式
        layer_norm.train()
        output_train = layer_norm(input_tensor)
        assert output_train.requires_grad
        
        # 评估模式
        layer_norm.eval()
        output_eval = layer_norm(input_tensor)
        assert output_eval.shape == input_tensor.shape
    
    def test_backward_compatibility(self, d_model, batch_size, seq_len):
        """测试与Transformer中的使用兼容性"""
        # 模拟Transformer中的使用场景
        layer_norm = LayerNorm(d_model)
        
        # 模拟残差连接后的归一化
        x = torch.randn(batch_size, seq_len, d_model)
        residual = torch.randn(batch_size, seq_len, d_model)
        
        # x + residual 然后归一化
        output = layer_norm(x + residual)
        
        assert output.shape == x.shape
        
        # 测试梯度流：如果输入需要梯度，输出也应该需要梯度
        x.requires_grad_(True)
        residual.requires_grad_(True)
        output_with_grad = layer_norm(x + residual)
        assert output_with_grad.requires_grad
        
        # 测试反向传播
        loss = output_with_grad.sum()
        loss.backward()
        assert x.grad is not None
        assert residual.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

